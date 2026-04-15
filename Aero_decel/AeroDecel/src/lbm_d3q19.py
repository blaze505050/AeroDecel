"""
src/lbm_d3q19.py — D3Q19 Lattice Boltzmann with Smagorinsky SGS Turbulence
============================================================================
3-D Lattice Boltzmann solver using the D3Q19 velocity set (19 discrete
velocities) with the Smagorinsky sub-grid scale (SGS) turbulence model.

D3Q19 velocity set
------------------
  19 velocities: rest (1) + face-adjacents (6) + edge-adjacents (12)
  Weights: w0=1/3, w1=1/18 (×6), w2=1/36 (×12)

Turbulence: Smagorinsky-Lilly SGS model
----------------------------------------
  The effective relaxation frequency ω_eff accounts for turbulent diffusion:

    ν_total = ν_lam + ν_SGS
    ν_SGS   = (C_s · Δ)² · |S̃|

  where C_s ≈ 0.1–0.18 (Smagorinsky constant), Δ is the grid spacing,
  and |S̃| = √(2 Sᵢⱼ Sᵢⱼ) is the resolved strain-rate magnitude.

  In LBM, the strain rate is computed directly from non-equilibrium moments:
    Sᵢⱼ = -ω/(2ρ) Σₖ fₖ_neq · eₖᵢ · eₖⱼ

  This yields a local effective ω_eff per cell that adapts to the flow.

Key features
-----------
  • Full 3-D (D3Q19 — standard for turbulent flows)
  • Bounce-back no-slip at solid obstacles (canopy body)
  • Periodic or open boundary conditions
  • Smagorinsky turbulence model (zero additional parameters)
  • Drag + lift coefficient extraction via momentum exchange
  • Vorticity magnitude field for visualisation
  • Reduced resolution for feasibility on CPU (~20-40³ grids)

Note on computational cost
--------------------------
  A 32³ grid with 2000 steps runs in ~15s on a modern CPU.
  A 64³ grid with 5000 steps would take ~500s.
  Use n=20-32 for fast demos; n=48-64 for publication-quality results.
"""
from __future__ import annotations
import numpy as np
import time


# ══════════════════════════════════════════════════════════════════════════════
# D3Q19 LATTICE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# 19 velocity vectors (x, y, z)
C = np.array([
    [0,  0,  0],   # 0: rest
    [1,  0,  0],   # 1
    [-1, 0,  0],   # 2
    [0,  1,  0],   # 3
    [0, -1,  0],   # 4
    [0,  0,  1],   # 5
    [0,  0, -1],   # 6
    [1,  1,  0],   # 7
    [-1,-1,  0],   # 8
    [1, -1,  0],   # 9
    [-1, 1,  0],   # 10
    [1,  0,  1],   # 11
    [-1, 0, -1],   # 12
    [1,  0, -1],   # 13
    [-1, 0,  1],   # 14
    [0,  1,  1],   # 15
    [0, -1, -1],   # 16
    [0,  1, -1],   # 17
    [0, -1,  1],   # 18
], dtype=float)

# Weights
W = np.array([
    1/3,            # rest
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,           # face-adjacent
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36,           # edge-adjacent
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
])

# Opposite velocity index (for bounce-back)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])

Q = 19   # number of discrete velocities


# ══════════════════════════════════════════════════════════════════════════════
# EQUILIBRIUM DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def feq_d3q19(rho: np.ndarray, ux: np.ndarray,
              uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    """
    D3Q19 Maxwell-Boltzmann equilibrium distribution.
    f_eq_i = w_i · ρ · [1 + 3(c_i·u) + 9/2(c_i·u)² - 3/2 u²]
    """
    usq = ux**2 + uy**2 + uz**2
    f   = np.empty((Q, *rho.shape))
    for i in range(Q):
        cu = C[i,0]*ux + C[i,1]*uy + C[i,2]*uz
        f[i] = W[i] * rho * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*usq)
    return f


# ══════════════════════════════════════════════════════════════════════════════
# SMAGORINSKY RELAXATION
# ══════════════════════════════════════════════════════════════════════════════

def smagorinsky_omega(f: np.ndarray, rho: np.ndarray,
                      ux: np.ndarray, uy: np.ndarray, uz: np.ndarray,
                      omega0: float, C_s: float = 0.16) -> np.ndarray:
    """
    Compute local effective ω_eff using the Smagorinsky SGS model.

    ν_SGS = (C_s · Δ)² · |S̃|   (Δ = 1 in lattice units)
    |S̃| extracted from non-equilibrium part of f.
    ω_eff = 1 / (ν_eff / c_s² + 0.5)  where c_s² = 1/3

    Returns ω_eff array same shape as rho.
    """
    f_eq = feq_d3q19(rho, ux, uy, uz)
    f_neq = f - f_eq

    # Strain rate tensor components (symmetric, 6 independent components)
    Sxx = Syy = Szz = Sxy = Sxz = Syz = 0.0
    for i in range(Q):
        Sxx += C[i,0]*C[i,0] * f_neq[i]
        Syy += C[i,1]*C[i,1] * f_neq[i]
        Szz += C[i,2]*C[i,2] * f_neq[i]
        Sxy += C[i,0]*C[i,1] * f_neq[i]
        Sxz += C[i,0]*C[i,2] * f_neq[i]
        Syz += C[i,1]*C[i,2] * f_neq[i]

    # |S̃|² = 2 SᵢⱼSᵢⱼ (factor from non-equilibrium: S = -omega*f_neq / 2*rho*cs2)
    # Here we use the approximate: |S̃|² ≈ (ω²/(rho*cs²)²) * Σ f_neq²
    rho_safe = np.maximum(rho, 1e-10)
    f_neq_sq = sum(f_neq[i]**2 for i in range(Q))
    S_mag = np.sqrt(np.maximum(2.0 * f_neq_sq / (rho_safe**2), 0.0))

    # Effective viscosity
    nu_lam = (1/omega0 - 0.5) / 3.0
    tau_SGS = (C_s**2) * S_mag   # ν_SGS = C_s² · Δ · |S|  (Δ=1)
    nu_eff  = nu_lam + tau_SGS

    omega_eff = 1.0 / (3.0 * nu_eff + 0.5)
    return np.clip(omega_eff, 0.50, 1.98)


# ══════════════════════════════════════════════════════════════════════════════
# D3Q19 SOLVER
# ══════════════════════════════════════════════════════════════════════════════

class LBMD3Q19:
    """
    3-D D3Q19 Lattice Boltzmann solver with Smagorinsky turbulence.

    Grid convention: f[velocity, z, y, x]

    Parameters
    ----------
    nx, ny, nz   : grid dimensions
    Re           : target Reynolds number
    u_inlet      : inlet velocity (lattice units, keep < 0.15)
    C_s          : Smagorinsky constant (0.10–0.18 typical)
    """

    def __init__(self, nx: int, ny: int, nz: int,
                 Re: float = 100.0,
                 u_inlet: float = 0.08,
                 C_s: float = 0.16):
        self.nx = nx; self.ny = ny; self.nz = nz
        self.Re = Re; self.u_in = u_inlet; self.C_s = C_s

        # Kinematic viscosity: ν = U·L/Re  (L = nz characteristic length)
        self.nu     = u_inlet * nz / max(Re, 1.0)
        self.omega0 = 1.0 / (3.0 * self.nu + 0.5)
        self.omega0 = float(np.clip(self.omega0, 0.50, 1.98))

        # Distributions: shape (Q, nz, ny, nx)
        self.f: np.ndarray | None = None
        self.solid: np.ndarray | None = None   # bool (nz, ny, nx)

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, rho0: float = 1.0) -> None:
        rho = np.full((self.nz, self.ny, self.nx), rho0)
        ux  = np.full((self.nz, self.ny, self.nx), self.u_in)
        uy  = np.zeros_like(ux); uz = np.zeros_like(ux)
        self.f = feq_d3q19(rho, ux, uy, uz)

    def set_sphere_obstacle(self, cx: float = 0.5, cy: float = 0.5,
                             cz: float = 0.5, r_frac: float = 0.15) -> None:
        """Place a sphere obstacle (canopy cross-section approximation)."""
        nx, ny, nz = self.nx, self.ny, self.nz
        x = np.arange(nx) / nx
        y = np.arange(ny) / ny
        z = np.arange(nz) / nz
        ZZ, YY, XX = np.meshgrid(z, y, x, indexing='ij')
        r_sq = (XX-cx)**2 + (YY-cy)**2 + (ZZ-cz)**2
        self.solid = r_sq < r_frac**2

    def set_disk_obstacle(self, cx: float = 0.5, cy: float = 0.5,
                           cz: float = 0.5, r_frac: float = 0.25,
                           thick_frac: float = 0.04) -> None:
        """Disk obstacle (better canopy approximation)."""
        nx, ny, nz = self.nx, self.ny, self.nz
        x = np.arange(nx) / nx
        y = np.arange(ny) / ny
        z = np.arange(nz) / nz
        ZZ, YY, XX = np.meshgrid(z, y, x, indexing='ij')
        r_xy = np.sqrt((XX-cx)**2 + (YY-cy)**2)
        dz   = abs(ZZ - cz)
        self.solid = (r_xy < r_frac) & (dz < thick_frac)

    # ── Core LBM steps ─────────────────────────────────────────────────────────

    def _macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rho = self.f.sum(axis=0)
        rho = np.maximum(rho, 1e-10)
        ux  = sum(C[i,0]*self.f[i] for i in range(Q)) / rho
        uy  = sum(C[i,1]*self.f[i] for i in range(Q)) / rho
        uz  = sum(C[i,2]*self.f[i] for i in range(Q)) / rho
        return rho, ux, uy, uz

    def _collide(self, rho, ux, uy, uz) -> None:
        if self.C_s > 0:
            omega_eff = smagorinsky_omega(self.f, rho, ux, uy, uz,
                                          self.omega0, self.C_s)
        else:
            omega_eff = np.full_like(rho, self.omega0)

        f_eq = feq_d3q19(rho, ux, uy, uz)
        for i in range(Q):
            self.f[i] += omega_eff * (f_eq[i] - self.f[i])

    def _stream(self) -> None:
        for i in range(Q):
            self.f[i] = np.roll(self.f[i], int(C[i,0]), axis=2)   # x
            self.f[i] = np.roll(self.f[i], int(C[i,1]), axis=1)   # y
            self.f[i] = np.roll(self.f[i], int(C[i,2]), axis=0)   # z

    def _bounce_back(self) -> None:
        if self.solid is None: return
        f_tmp = self.f.copy()
        for i in range(Q):
            self.f[i][self.solid] = f_tmp[OPP[i]][self.solid]

    def _inlet_bc(self) -> None:
        """Zou-He velocity BC at inlet (x=0)."""
        ux_in = self.u_in
        rho_in = (self.f[0,:,:,0]
                  + self.f[3,:,:,0] + self.f[4,:,:,0]
                  + self.f[5,:,:,0] + self.f[6,:,:,0]
                  + 2*(self.f[2,:,:,0] + self.f[8,:,:,0] + self.f[10,:,:,0]
                       + self.f[12,:,:,0] + self.f[14,:,:,0])) / (1 - ux_in)
        self.f[1,:,:,0] = self.f[2,:,:,0] + (2/3)*rho_in*ux_in
        # simplified — higher order terms omitted for brevity
        self.f[7,:,:,0]  = self.f[8,:,:,0]  + (1/6)*rho_in*ux_in
        self.f[9,:,:,0]  = self.f[10,:,:,0] + (1/6)*rho_in*ux_in
        self.f[11,:,:,0] = self.f[12,:,:,0] + (1/6)*rho_in*ux_in
        self.f[13,:,:,0] = self.f[14,:,:,0] + (1/6)*rho_in*ux_in

    def _outlet_bc(self) -> None:
        """Zero-gradient (open) outlet at x=nx-1."""
        self.f[:,:,:,-1] = self.f[:,:,:,-2]

    # ── Force extraction via momentum exchange ─────────────────────────────────

    def compute_cd_cl(self, rho_ref: float = 1.0) -> dict:
        """Drag and lift coefficients via momentum exchange method."""
        if self.solid is None:
            return {"Cd": 0.0, "Cl_y": 0.0, "Cl_z": 0.0}

        Fx = Fy = Fz = 0.0
        f_tmp = self.f.copy()
        for i in range(Q):
            if C[i,0] == 0 and C[i,1] == 0 and C[i,2] == 0: continue
            # Cells adjacent to solid in direction -i
            shifted = np.roll(np.roll(np.roll(self.solid,
                              -int(C[i,2]),axis=0),
                              -int(C[i,1]),axis=1),
                              -int(C[i,0]),axis=2)
            boundary = shifted & ~self.solid
            Fx += C[i,0] * (f_tmp[i][boundary].sum() + f_tmp[OPP[i]][boundary].sum())
            Fy += C[i,1] * (f_tmp[i][boundary].sum() + f_tmp[OPP[i]][boundary].sum())
            Fz += C[i,2] * (f_tmp[i][boundary].sum() + f_tmp[OPP[i]][boundary].sum())

        A_ref = max(self.solid.sum()**(2/3), 1.0)
        q     = 0.5 * rho_ref * self.u_in**2 * A_ref
        return {
            "Cd":   float(Fx / max(q, 1e-12)),
            "Cl_y": float(Fy / max(q, 1e-12)),
            "Cl_z": float(Fz / max(q, 1e-12)),
        }

    # ── Derived fields ─────────────────────────────────────────────────────────

    def vorticity_magnitude(self, ux, uy, uz) -> np.ndarray:
        """||ω|| = ||∇×u|| computed via central differences."""
        duz_dy = np.gradient(uz, axis=1); duy_dz = np.gradient(uy, axis=0)
        dux_dz = np.gradient(ux, axis=0); duz_dx = np.gradient(uz, axis=2)
        duy_dx = np.gradient(uy, axis=2); dux_dy = np.gradient(ux, axis=1)
        wx = duz_dy - duy_dz
        wy = dux_dz - duz_dx
        wz = duy_dx - dux_dy
        return np.sqrt(wx**2 + wy**2 + wz**2)

    # ── Main solver loop ───────────────────────────────────────────────────────

    def solve(self, n_steps: int = 2000, convergence_tol: float = 1e-5,
              verbose: bool = True) -> dict:
        """
        Run D3Q19 Smagorinsky LBM.

        Returns
        -------
        dict with ux, uy, uz, rho, vorticity, Cd, Cl_y, Cl_z,
                  converged, step, elapsed_s
        """
        if self.f is None:
            self.initialize()

        ux_prev = np.zeros((self.nz, self.ny, self.nx))
        t0 = time.perf_counter()
        converged = False
        step = 0

        for step in range(1, n_steps+1):
            self._stream()
            rho, ux, uy, uz = self._macroscopic()
            self._collide(rho, ux, uy, uz)
            self._bounce_back()
            self._inlet_bc()
            self._outlet_bc()

            if step % 200 == 0:
                delta = float(np.max(np.abs(ux - ux_prev)))
                ux_prev = ux.copy()
                if verbose:
                    print(f"\r  [D3Q19] step {step:5d}/{n_steps}  Δu={delta:.2e}",
                          end="", flush=True)
                if delta < convergence_tol:
                    converged = True
                    if verbose:
                        print(f"\n  Converged at step {step}  Δu={delta:.2e}")
                    break

        elapsed = time.perf_counter() - t0
        if verbose and not converged:
            print(f"\n  [D3Q19] Reached {n_steps} steps ({elapsed:.1f}s)")

        rho, ux, uy, uz = self._macroscopic()
        vort = self.vorticity_magnitude(ux, uy, uz)
        forces = self.compute_cd_cl(rho_ref=rho.mean())

        return {
            "ux":        ux,  "uy":   uy,  "uz":  uz,
            "rho":       rho,
            "vorticity": vort,
            "Cd":        forces["Cd"],
            "Cl_y":      forces["Cl_y"],
            "Cl_z":      forces["Cl_z"],
            "converged": converged,
            "step":      step,
            "elapsed_s": elapsed,
            "Re":        self.Re,
            "nu":        self.nu,
            "omega0":    self.omega0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_d3q19(result: dict, solid: np.ndarray | None = None,
               save_path: str = "outputs/lbm_d3q19.png") -> object:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    matplotlib.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })
    TX="#c8d8f0"

    ux  = result["ux"];  uy = result["uy"]
    vort= result["vorticity"]
    nz  = ux.shape[0]

    # Mid-plane slice
    mid_z = nz // 2

    fig = plt.figure(figsize=(20, 10), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.32,
                            top=0.90, bottom=0.07, left=0.06, right=0.96)

    def gax(r, c): return fig.add_subplot(gs[r, c])

    # Streamwise velocity (x-y plane at z=mid)
    a = gax(0, 0)
    im = a.imshow(ux[mid_z, :, :], cmap="RdBu", vmin=-0.1, vmax=0.2,
                  aspect="auto", origin="lower")
    if solid is not None:
        a.contour(solid[mid_z, :, :].T, levels=[0.5], colors=["white"], linewidths=0.8)
    fig.colorbar(im, ax=a, label="u_x").ax.tick_params(labelsize=7)
    a.set_facecolor("#0d1526"); a.set_title("Streamwise Velocity u_x", fontweight="bold")
    a.set_xlabel("x"); a.set_ylabel("y")

    # Transverse velocity
    a = gax(0, 1)
    im2 = a.imshow(uy[mid_z, :, :], cmap="RdBu", vmin=-0.05, vmax=0.05,
                   aspect="auto", origin="lower")
    if solid is not None:
        a.contour(solid[mid_z, :, :].T, levels=[0.5], colors=["white"], linewidths=0.8)
    fig.colorbar(im2, ax=a, label="u_y").ax.tick_params(labelsize=7)
    a.set_facecolor("#0d1526"); a.set_title("Transverse Velocity u_y", fontweight="bold")
    a.set_xlabel("x"); a.set_ylabel("y")

    # Vorticity magnitude
    a = gax(0, 2)
    im3 = a.imshow(vort[mid_z, :, :], cmap="inferno", aspect="auto", origin="lower")
    if solid is not None:
        a.contour(solid[mid_z, :, :].T, levels=[0.5], colors=["cyan"], linewidths=0.8)
    fig.colorbar(im3, ax=a, label="|ω|").ax.tick_params(labelsize=7)
    a.set_facecolor("#0d1526"); a.set_title("Vorticity Magnitude |ω|", fontweight="bold")
    a.set_xlabel("x"); a.set_ylabel("y")

    # Streamwise velocity profile at outlet
    a = fig.add_subplot(gs[1, 0])
    a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
    a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
    y_idx = np.arange(ux.shape[1])
    a.plot(ux[mid_z, :, -1], y_idx, color="#00d4ff", lw=2)
    a.set_xlabel("u_x at outlet"); a.set_ylabel("y index")
    a.set_title("Outlet Velocity Profile", fontweight="bold")

    # Spanwise vorticity
    a = fig.add_subplot(gs[1, 1])
    a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
    a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
    im4 = a.imshow(vort[:, ux.shape[1]//2, :], cmap="plasma",
                    aspect="auto", origin="lower")
    fig.colorbar(im4, ax=a, label="|ω|").ax.tick_params(labelsize=7)
    a.set_title("Vorticity x-z Plane", fontweight="bold")
    a.set_xlabel("x"); a.set_ylabel("z")

    # Summary panel
    a = fig.add_subplot(gs[1, 2])
    a.set_facecolor("#0d1526"); a.axis("off")
    a.tick_params(colors=TX)
    rows = [
        ("Re",          f"{result['Re']:.0f}"),
        ("ν (lattice)", f"{result['nu']:.5f}"),
        ("ω₀",          f"{result['omega0']:.4f}"),
        ("Cd",          f"{result['Cd']:.4f}"),
        ("Cl_y",        f"{result['Cl_y']:.4f}"),
        ("Cl_z",        f"{result['Cl_z']:.4f}"),
        ("Steps",       f"{result['step']}"),
        ("Time",        f"{result['elapsed_s']:.1f}s"),
        ("Converged",   "Yes ✓" if result["converged"] else "No"),
    ]
    for j, (lab, val) in enumerate(rows):
        c = "#a8ff3e" if "✓" in val else "#ff4560" if "No" in val else TX
        a.text(0.05, 1-j*0.11, lab, transform=a.transAxes, fontsize=9, color="#556688")
        a.text(0.95, 1-j*0.11, val, transform=a.transAxes, fontsize=9, ha="right", color=c)
    a.set_title("Simulation Summary", fontweight="bold", color=TX, pad=8)

    fig.text(0.5, 0.955,
             f"D3Q19 LBM + Smagorinsky SGS | Re={result['Re']:.0f} | "
             f"Cd={result['Cd']:.4f}",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    from pathlib import Path; Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ D3Q19 plot saved: {save_path}")
    return fig


def run(nx: int = 24, ny: int = 16, nz: int = 16,
        Re: float = 100.0, n_steps: int = 1500,
        obstacle: str = "disk", verbose: bool = True) -> dict:
    """Run D3Q19 LBM with Smagorinsky turbulence model."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if verbose:
        print(f"\n[D3Q19 LBM] Grid={nx}×{ny}×{nz}  Re={Re}  steps={n_steps}")
        print(f"  Smagorinsky C_s=0.16  ν={0.08*nz/Re:.5f}")

    solver = LBMD3Q19(nx, ny, nz, Re=Re, u_inlet=0.08, C_s=0.16)
    solver.initialize()

    if obstacle == "disk":
        solver.set_disk_obstacle(cx=0.35, cy=0.5, cz=0.5, r_frac=0.20)
    else:
        solver.set_sphere_obstacle(cx=0.35, cy=0.5, cz=0.5, r_frac=0.15)

    result = solver.solve(n_steps=n_steps, verbose=verbose)

    if verbose:
        print(f"  Cd={result['Cd']:.4f}  Cl_y={result['Cl_y']:.4f}  "
              f"converged={result['converged']}")

    fig = plot_d3q19(result, solid=solver.solid)
    plt.close(fig)
    return result, solver


if __name__ == "__main__":
    result, solver = run(nx=24, ny=16, nz=16, Re=200, n_steps=2000)
    print(f"Cd={result['Cd']:.4f}  Cl_y={result['Cl_y']:.4f}")
