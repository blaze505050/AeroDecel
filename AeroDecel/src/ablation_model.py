"""
src/ablation_model.py — Amar Ablation Model with Boundary Layer Blowing
=========================================================================
Couples TPS mass loss to aerodynamic heating via the blowing parameter B'.

Physics
-------
Ablative materials (PICA, AVCOAT, SRP) decompose under high heat flux:

  1. Pyrolysis: polymer matrix decomposes → gas injection into boundary layer
     ṁ_pyrolysis = A_p · exp(-E_p / (R·T_s))  [Arrhenius]

  2. Surface ablation: char surface oxidises/sublimes
     ṁ_char = A_c · exp(-E_c / (R·T_s))

  3. Total mass loss rate:
     ṁ_total = ṁ_pyrolysis + ṁ_char   [kg/(m²·s)]

  4. Recession rate:
     ṡ = ṁ_total / ρ_material           [m/s]

  5. Blowing parameter B' (Lees 1956):
     B' = ṁ_total / (ρ_e · u_e · C_H)

  6. Wall heat flux with blowing correction (Mickley-Davis):
     q_wall = q_wall_0 · (1 - 0.68·B') / (1 + B')^0.5
     (The injected gas forms a film that partially blocks heating)

  7. Energy balance at ablating surface:
     q_net = q_aerodynamic - q_reradiation - q_conduction - ṁ·h_ablation
     where h_ablation = h_v (heat of vaporisation/pyrolysis)

  8. In-depth 1-D conduction with moving boundary (Stefan condition):
     The ablation front recedes at ṡ; nodes move accordingly.

Ablative material properties (Amar 2006 / Chen & Milos 1999)
-------------------------------------------------------------
  PICA   : porous carbon/phenolic  (MSL, Stardust, OSIRIS-REx)
  AVCOAT : foam-filled silica phenolic  (Apollo, Orion)
  SRP    : Silicone impregnated reusable surface insulation (Hayabusa)
  HEATS  : Honeycomb entry ablative TPS

Outputs
-------
  • q_wall(t)  — modified heat flux accounting for blowing
  • ṁ(t)       — mass loss rate [kg/(m²·s)]
  • ṡ(t)       — recession rate [m/s]
  • Δs(t)      — total recession depth [mm]
  • T_surf(t)  — surface temperature
  • B_prime(t) — blowing parameter
  • efficiency  — overall ablation efficiency (q_blocked / q_incident)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


# ══════════════════════════════════════════════════════════════════════════════
# ABLATIVE MATERIAL DATABASE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblativeMaterial:
    name:           str
    density_kgm3:   float    # virgin material density
    rho_char:       float    # char density (post-pyrolysis)
    cp_J_kgK:       float    # specific heat
    k_W_mK:         float    # thermal conductivity (virgin)
    k_char_W_mK:    float    # thermal conductivity (char)
    T_decomp_K:     float    # onset of pyrolysis
    T_ablate_K:     float    # onset of surface ablation
    h_pyrolysis:    float    # heat of pyrolysis [J/kg]
    h_ablation:     float    # heat of surface ablation [J/kg]
    emissivity:     float    # surface emissivity
    # Arrhenius pyrolysis
    A_pyro:         float    # pre-exponential [1/s]
    E_pyro:         float    # activation energy / R [K]
    # Arrhenius surface ablation
    A_surf:         float    # pre-exponential [kg/(m²·s)]
    E_surf:         float    # activation energy / R [K]
    # Blowing correction exponent
    eta_blowing:    float = 0.68  # Mickley-Davis coefficient


ABLATIVE_DB: dict[str, AblativeMaterial] = {
    "pica": AblativeMaterial(
        name="PICA",
        density_kgm3=220.0, rho_char=130.0,
        cp_J_kgK=1300.0, k_W_mK=0.14, k_char_W_mK=0.07,
        T_decomp_K=600.0, T_ablate_K=2800.0,
        h_pyrolysis=1_800_000.0,    # 1.8 MJ/kg
        h_ablation=24_000_000.0,    # 24 MJ/kg (carbon sublimation)
        emissivity=0.82,
        A_pyro=3.5e9, E_pyro=12_000.0,   # K (= Ea/R)
        A_surf=1.2e4, E_surf=36_000.0,
    ),
    "avcoat": AblativeMaterial(
        name="AVCOAT",
        density_kgm3=520.0, rho_char=280.0,
        cp_J_kgK=1240.0, k_W_mK=0.40, k_char_W_mK=0.18,
        T_decomp_K=550.0, T_ablate_K=3000.0,
        h_pyrolysis=2_200_000.0,
        h_ablation=30_000_000.0,
        emissivity=0.84,
        A_pyro=4.8e9, E_pyro=11_500.0,
        A_surf=0.8e4, E_surf=40_000.0,
    ),
    "srp": AblativeMaterial(
        name="SRP",
        density_kgm3=224.0, rho_char=150.0,
        cp_J_kgK=1000.0, k_W_mK=0.16, k_char_W_mK=0.10,
        T_decomp_K=500.0, T_ablate_K=2500.0,
        h_pyrolysis=1_500_000.0,
        h_ablation=20_000_000.0,
        emissivity=0.80,
        A_pyro=2.0e9, E_pyro=10_000.0,
        A_surf=0.5e4, E_surf=32_000.0,
    ),
}

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
R_GAS = 8.314  # J/(mol·K)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION SOLVER
# ══════════════════════════════════════════════════════════════════════════════

class AblationSolver:
    """
    Coupled ablation + heat conduction solver.

    The solver tracks:
      • Surface temperature T_s
      • Mass loss rate ṁ (pyrolysis + surface ablation)
      • Recession depth Δs
      • Modified heat flux q_wall (with blowing correction)
      • In-depth temperature profile T(x, t)

    The energy equation at the ablating surface is:

      q_wall - ε·σ·T_s⁴ - ṁ·(h_pyro + h_surf_abl) = k·dT/dx|_surface

    Blowing correction (Mickley & Davis 1954):
      q_wall = q_0 · [(1 - η·B') / max(1, (1 + B')^0.5)]
      B' = ṁ / (ρ_e · u_e · C_H)   ≈ ṁ / q_0 * (h_e - h_w)
    """

    def __init__(self, material: str, thickness_m: float = 0.05,
                 n_nodes: int = 30):
        self.mat = ABLATIVE_DB.get(material.lower())
        if self.mat is None:
            raise ValueError(f"Unknown ablative material '{material}'. "
                             f"Available: {list(ABLATIVE_DB.keys())}")
        self.L     = thickness_m
        self.n     = n_nodes
        self.dx    = thickness_m / n_nodes
        self.T_profile: np.ndarray | None = None
        self.recession_history: np.ndarray | None = None

    # ── Arrhenius mass loss rates ─────────────────────────────────────────────

    def _pyrolysis_rate(self, T_K: float) -> float:
        """Pyrolysis mass loss rate [kg/(m²·s)] via Arrhenius."""
        T = max(T_K, 300.0)
        if T < self.mat.T_decomp_K:
            return 0.0
        return float(self.mat.A_pyro * np.exp(-self.mat.E_pyro / T)
                     * self.mat.density_kgm3 * self.dx)

    def _surface_ablation_rate(self, T_K: float) -> float:
        """Surface oxidation/sublimation rate [kg/(m²·s)]."""
        T = max(T_K, 300.0)
        if T < self.mat.T_ablate_K:
            return 0.0
        return float(self.mat.A_surf * np.exp(-self.mat.E_surf / T))

    def _mdot(self, T_s: float) -> float:
        return self._pyrolysis_rate(T_s) + self._surface_ablation_rate(T_s)

    # ── Blowing correction ────────────────────────────────────────────────────

    def _blowing_parameter(self, mdot: float, q0: float, T_s: float,
                            h_aw: float = 5e6) -> float:
        """
        B' ≈ ṁ · h_aw / q_0   (simplified Lees formula)
        where h_aw is the adiabatic wall enthalpy [J/kg].
        """
        if q0 < 1.0:
            return 0.0
        return float(mdot * h_aw / max(q0, 1.0))

    def _corrected_heat_flux(self, q0: float, B_prime: float) -> float:
        """
        Modified heat flux with boundary-layer blowing (Mickley-Davis):
        q_wall = q_0 · (1 - η·B') / (1 + B')^0.5   clipped at 0
        """
        eta = self.mat.eta_blowing
        correction = (1.0 - eta * B_prime) / max(np.sqrt(1.0 + B_prime), 1e-6)
        return float(max(q0 * correction, 0.0))

    # ── Energy balance at surface ─────────────────────────────────────────────

    def _surface_temperature(self, q_wall: float, T_s_prev: float,
                              dt: float) -> float:
        """
        Newton-iteration to find T_s satisfying energy balance:
        q_wall - ε·σ·T_s⁴ - ṁ·h_total = k·(T_s - T_{s+1})/dx

        (quasi-static — T_s relaxes faster than thermal wave)
        """
        k  = self.mat.k_char_W_mK   # use char conductivity at surface
        T2 = T_s_prev + 50.0        # sub-surface temperature (estimated)
        eps_sig = self.mat.emissivity * STEFAN_BOLTZMANN
        h_abl = self.mat.h_pyrolysis + self.mat.h_ablation

        def residual(T_s):
            T_s = max(T_s, 300.0)
            mdot   = self._mdot(T_s)
            rerad  = eps_sig * T_s**4
            cond   = k * (T_s - T2) / max(self.dx, 1e-6)
            return q_wall - rerad - mdot * h_abl - cond

        # Bracket
        T_lo, T_hi = 300.0, 30_000.0
        f_lo, f_hi = residual(T_lo), residual(T_hi)
        if f_lo * f_hi > 0:
            # Can't bracket — return previous + gradient estimate
            dT = dt * (q_wall - eps_sig * T_s_prev**4) / (self.mat.cp_J_kgK * self.mat.density_kgm3 * self.dx)
            return float(np.clip(T_s_prev + dT, 300, 30000))

        from scipy.optimize import brentq
        try:
            T_s = brentq(residual, T_lo, T_hi, xtol=1.0, maxiter=50)
        except Exception:
            T_s = T_s_prev

        return float(np.clip(T_s, 300, 30_000))

    # ── Main coupled solver ───────────────────────────────────────────────────

    def solve(self, q_incident: np.ndarray, time_steps: np.ndarray,
              T_initial_K: float = 300.0,
              h_aw_Jkg: float = 5e6,
              verbose: bool = False) -> dict:
        """
        Run coupled ablation + heat conduction simulation.

        Parameters
        ----------
        q_incident : incident (frozen-flow) heat flux [W/m²] vs time
        time_steps : time array [s]
        T_initial_K: initial temperature
        h_aw_Jkg   : adiabatic wall enthalpy [J/kg]

        Returns
        -------
        dict with: q_wall, mdot, B_prime, recession_mm, T_surface,
                   T_profile, blowing_efficiency, blocking_pct
        """
        n_t = len(time_steps)
        if len(q_incident) < n_t:
            q_incident = np.interp(np.linspace(0,1,n_t),
                                   np.linspace(0,1,len(q_incident)), q_incident)

        # State arrays
        q_wall    = np.zeros(n_t)
        mdot_arr  = np.zeros(n_t)
        B_prime_a = np.zeros(n_t)
        T_surf    = np.zeros(n_t)
        recession = np.zeros(n_t)   # cumulative [m]
        T_profile = np.full((n_t, self.n), T_initial_K)

        T_s = T_initial_K
        delta_s = 0.0   # total recession depth [m]

        # FD arrays
        T_fd = np.full(self.n, T_initial_K)

        for i in range(n_t - 1):
            dt = time_steps[i+1] - time_steps[i]
            if dt <= 0:
                continue

            q0 = float(q_incident[i])

            # Surface temperature (iterative energy balance)
            T_s_new = self._surface_temperature(q0, T_s, dt)
            T_s     = T_s_new

            # Mass loss
            mdot = self._mdot(T_s)

            # Blowing parameter & corrected heat flux
            B    = self._blowing_parameter(mdot, q0, T_s, h_aw_Jkg)
            q_w  = self._corrected_heat_flux(q0, B)

            # Recession
            rho_eff = 0.5*(self.mat.density_kgm3 + self.mat.rho_char)
            s_dot   = mdot / max(rho_eff, 1.0)
            delta_s += s_dot * dt

            # In-depth 1-D conduction (explicit FD)
            k  = self.mat.k_W_mK
            rho = self.mat.density_kgm3
            cp  = self.mat.cp_J_kgK
            α  = k / (rho * cp)
            dx = self.dx
            Fo = α * dt / dx**2

            T_fd_new = T_fd.copy()
            if Fo < 0.49:  # explicit stability
                T_fd_new[1:-1] = T_fd[1:-1] + Fo*(T_fd[2:] - 2*T_fd[1:-1] + T_fd[:-2])
            else:  # implicit (tridiagonal)
                a_diag = np.full(self.n,  1+2*Fo)
                b_diag = np.full(self.n-1, -Fo)
                rhs = T_fd.copy()
                # solve tridiagonal (Thomas algorithm)
                a = a_diag.copy(); b = b_diag.copy()
                r = rhs.copy()
                for j in range(1, self.n):
                    m_ = b[j-1] / a[j-1]
                    a[j] -= m_ * b[j-1]
                    r[j] -= m_ * r[j-1]
                T_fd_new[-1] = r[-1] / a[-1]
                for j in range(self.n-2, -1, -1):
                    T_fd_new[j] = (r[j] - b[j]*T_fd_new[j+1]) / a[j]

            # BCs
            T_fd_new[0]  = T_s                  # surface temp from ablation model
            T_fd_new[-1] = T_initial_K           # cold back-face

            T_fd = np.clip(T_fd_new, T_initial_K, 30_000)

            # Store
            q_wall[i]    = q_w
            mdot_arr[i]  = mdot
            B_prime_a[i] = B
            T_surf[i]    = T_s
            recession[i] = delta_s
            T_profile[i] = T_fd

        # Final step
        q_wall[-1]    = q_wall[-2]
        mdot_arr[-1]  = mdot_arr[-2]
        B_prime_a[-1] = B_prime_a[-2]
        T_surf[-1]    = T_surf[-2]
        recession[-1] = delta_s
        T_profile[-1] = T_fd

        # Summary statistics
        q0_total = float(np.trapezoid(q_incident, time_steps)) if len(time_steps)>1 else 0.0
        qw_total = float(np.trapezoid(q_wall,    time_steps)) if len(time_steps)>1 else 0.0
        blocking_pct = float(np.clip(100 * (1 - qw_total/max(q0_total, 1e-6)), 0, 99.9))

        self.T_profile = T_profile
        self.recession_history = recession

        return {
            "q_incident_Wm2":  q_incident,
            "q_wall_Wm2":      q_wall,
            "mdot_kgm2s":      mdot_arr,
            "B_prime":         B_prime_a,
            "T_surface_K":     T_surf,
            "recession_mm":    recession * 1000.0,
            "T_profile_K":     T_profile,
            "peak_T_K":        float(T_surf.max()),
            "total_recession_mm": float(delta_s * 1000),
            "blocking_pct":    blocking_pct,
            "total_mdot_kgm2": float(np.trapezoid(mdot_arr, time_steps)) if len(time_steps)>1 else 0.0,
            "material":        self.mat.name,
        }

    def summary(self, result: dict) -> dict:
        """Return scalar summary of ablation results."""
        mat = self.mat
        return {
            "material":           mat.name,
            "initial_thickness_mm": self.L * 1000,
            "final_recession_mm": result["total_recession_mm"],
            "remaining_mm":       self.L*1000 - result["total_recession_mm"],
            "survived":           result["total_recession_mm"] < self.L * 1000,
            "peak_T_K":           result["peak_T_K"],
            "T_limit_K":          mat.T_ablate_K,
            "blocking_pct":       result["blocking_pct"],
            "peak_mdot_kgm2s":    result["mdot_kgm2s"].max(),
            "peak_B_prime":       result["B_prime"].max(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation(result: dict, time_steps: np.ndarray,
                  save_path: str = "outputs/ablation_model.png") -> object:
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
    TX="#c8d8f0"; C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"; CR="#ff4560"

    t = time_steps
    fig = plt.figure(figsize=(20, 11), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.46, wspace=0.36,
                            top=0.90, bottom=0.07, left=0.06, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    # P0: Heat flux — incident vs wall
    a = gax(0, 0)
    a.fill_between(t, result["q_incident_Wm2"]/1e6, alpha=0.2, color=CR)
    a.plot(t, result["q_incident_Wm2"]/1e6, color=CR, lw=1.8, label="Incident (frozen)")
    a.fill_between(t, result["q_wall_Wm2"]/1e6, alpha=0.3, color=C1)
    a.plot(t, result["q_wall_Wm2"]/1e6, color=C1, lw=2.0, label="Wall (with blowing)")
    a.legend(fontsize=7.5)
    a.set_title("Heat Flux — Blowing Correction", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("q [MW/m²]")

    # P1: Blowing parameter B'
    a = gax(0, 1)
    a.fill_between(t, result["B_prime"], alpha=0.2, color=C4)
    a.plot(t, result["B_prime"], color=C4, lw=1.8)
    a.axhline(0.1, color=C3, lw=0.8, ls="--", alpha=0.7, label="B'=0.1")
    a.legend(fontsize=7.5)
    a.set_title("Blowing Parameter B'", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("B' [-]")

    # P2: Mass loss rate
    a = gax(0, 2)
    a.fill_between(t, result["mdot_kgm2s"]*1000, alpha=0.2, color=C2)
    a.plot(t, result["mdot_kgm2s"]*1000, color=C2, lw=1.8)
    a.set_title("Mass Loss Rate ṁ", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("ṁ [g/(m²·s)]")

    # P3: Recession depth
    a = gax(0, 3)
    a.fill_between(t, result["recession_mm"], alpha=0.2, color=C3)
    a.plot(t, result["recession_mm"], color=C3, lw=1.8)
    a.set_title("Recession Depth", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("Δs [mm]")

    # P4: Surface temperature
    a = gax(1, 0)
    a.fill_between(t, result["T_surface_K"], alpha=0.15, color=CR)
    a.plot(t, result["T_surface_K"], color=CR, lw=1.8)
    a.set_title("Surface Temperature", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("T_s [K]")

    # P5: In-depth T profile at peak time
    a = gax(1, 1)
    T_prof = result["T_profile_K"]
    peak_t_idx = np.argmax(result["T_surface_K"])
    times_to_plot = np.linspace(0, len(t)-1, 5).astype(int)
    cmap_ = plt.cm.inferno
    for k, idx in enumerate(times_to_plot):
        x = np.linspace(0, 1, T_prof.shape[1])  # normalized depth
        a.plot(x, T_prof[idx], color=cmap_(k/4),
               lw=1.5, label=f"t={t[idx]:.0f}s")
    a.legend(fontsize=7); a.set_title("In-depth T Profile", fontweight="bold")
    a.set_xlabel("Depth (normalised)"); a.set_ylabel("T [K]")

    # P6: Blocking percentage
    a = gax(1, 2)
    q_block = 100*(1 - result["q_wall_Wm2"]/np.maximum(result["q_incident_Wm2"],1))
    a.fill_between(t, q_block, alpha=0.2, color="#9d60ff")
    a.plot(t, q_block, color="#9d60ff", lw=1.8)
    a.set_title("Heat Flux Blocking %", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("Blocked [%]")

    # P7: Summary table
    a = gax(1, 3); a.axis("off")
    rows = [
        ("Material", result["material"], TX),
        ("", "", TX),
        ("Peak T_surface", f"{result['peak_T_K']:.0f} K", CR),
        ("Total recession", f"{result['total_recession_mm']:.2f} mm", C3),
        ("Heat blocked", f"{result['blocking_pct']:.1f}%", C4),
        ("Peak ṁ", f"{result['mdot_kgm2s'].max()*1000:.3f} g/(m²·s)", C2),
        ("Peak B'", f"{result['B_prime'].max():.4f}", C4),
        ("Total mass loss", f"{result['total_mdot_kgm2']*1000:.2f} g/m²", TX),
    ]
    for j, (lab, val, col) in enumerate(rows):
        a.text(0.02, 1-j*0.13, lab, transform=a.transAxes, fontsize=8.5,
               color="#556688" if lab else TX)
        a.text(0.98, 1-j*0.13, val, transform=a.transAxes, fontsize=8.5,
               ha="right", color=col)
    a.set_title("Ablation Summary", fontweight="bold")

    fig.text(0.5, 0.955,
             f"Ablation Coupling (Amar Model) — {result['material']} | "
             f"Recession={result['total_recession_mm']:.2f}mm | "
             f"Blocking={result['blocking_pct']:.1f}%",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    from pathlib import Path; Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Ablation plot saved: {save_path}")
    return fig


def run(material: str = "pica", thickness_m: float = 0.05,
        q_peak_MW: float = 20.0, t_entry_s: float = 200.0,
        verbose: bool = True) -> dict:
    """Run ablation demo with triangular heat flux profile."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, t_entry_s, 300)
    # Triangular heat flux: peak at 30% of flight time
    t_peak = t_entry_s * 0.30
    q0 = q_peak_MW * 1e6 * np.where(
        t <= t_peak,
        t / t_peak,
        (t_entry_s - t) / (t_entry_s - t_peak)
    )
    q0 = np.clip(q0, 0, None)

    solver = AblationSolver(material, thickness_m)
    result = solver.solve(q0, t, verbose=verbose)
    summ   = solver.summary(result)

    if verbose:
        print(f"\n  Ablation summary ({material.upper()}):")
        for k, v in summ.items():
            print(f"    {k:30s}: {v}")

    fig = plot_ablation(result, t, f"outputs/ablation_{material}.png")
    plt.close(fig)
    return result, summ


if __name__ == "__main__":
    for mat in ["pica", "avcoat"]:
        result, summ = run(mat, thickness_m=0.05, q_peak_MW=15.0, t_entry_s=200.0)
        print(f"\n  {mat}: survived={summ['survived']} "
              f"recession={summ['final_recession_mm']:.2f}mm "
              f"blocking={summ['blocking_pct']:.1f}%")
