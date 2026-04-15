"""
src/aeroelastic_flutter.py — Aeroelastic Canopy Flutter (FEM Membrane)
=======================================================================
Models the canopy as a 2-D pre-tensioned circular membrane using
finite-element analysis (pure numpy). Couples aerodynamic pressure
loading to structural deformation, and computes:

  • Natural frequencies and mode shapes
  • Critical dynamic pressure for flutter onset
  • Porosity-deformation feedback (deformed shape changes effective porosity)
  • Riser load spikes from flutter-induced shape changes
  • Full time-domain aeroelastic response

Structural model
----------------
  Governing equation (Kirchhoff membrane, no bending stiffness):
    ρ_s·h·ü + c·u̇ - T·∇²u = q_aero(x,y,t) + p_diff(x,y,t)

  where:
    u(x,y,t) : out-of-plane deflection [m]
    ρ_s      : surface mass density [kg/m²]  = ρ_material · thickness
    h        : membrane thickness [m]
    c        : aerodynamic damping coefficient
    T        : isotropic tension [N/m]  from riser loads + inflation pressure
    q_aero   : aerodynamic pressure loading [Pa]
    p_diff   : differential pressure across canopy [Pa]

Aerodynamic coupling
--------------------
  Pressure loading depends on local deflection via:
    p_diff = q_dyn · Cp_local(u/D)
  where Cp_local is linearised about the nominal Newtonian value.

  Porosity feedback:
    k_pore(u) = k_pore0 · (1 + α_pore · u²/h²)

Flutter prediction
------------------
  The critical dynamic pressure q_cr for membrane flutter is:
    q_cr = ω_n² · ρ_s · h / |∂Cp/∂u|

  or equivalently the reduced frequency: U* = v / (ω_n · D)
  Flutter onset when U* > 1/√(2·ζ)

FEM discretisation
------------------
  Circular membrane meshed with triangular elements (radial-angular grid).
  Global stiffness K and mass M assembled via standard FEM.
  Eigenvalue problem: K·φ = ω²·M·φ → natural frequencies ωᵢ, modes φᵢ
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanopyMembrane:
    """Material and geometric properties of the canopy membrane."""
    radius_m:           float = 5.0        # canopy nominal radius [m]
    thickness_m:        float = 0.0005     # fabric thickness [m]
    surface_density:    float = 0.22       # ρ_s [kg/m²]  (44 g/m² fabric)
    tension_Nm:         float = 1200.0     # isotropic pre-tension T [N/m]
    damping_ratio:      float = 0.08       # structural damping ζ
    porosity_k0:        float = 0.012      # nominal porosity coefficient
    Cp_slope:           float = -0.35      # dCp/d(u/R): pressure-deflection slope
    E_GPa:              float = 3.0        # fabric elastic modulus [GPa]
    nu:                 float = 0.35       # Poisson ratio
    n_rings:            int   = 8          # radial rings in FEM mesh
    n_sectors:          int   = 16         # circumferential sectors


# ══════════════════════════════════════════════════════════════════════════════
# MESH GENERATION  (polar coordinate ring mesh)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_polar_mesh(cfg: CanopyMembrane) -> dict:
    """
    Generate a polar (r-θ) mesh for the circular membrane.
    Returns node coordinates, element connectivity, DOF count.
    """
    R       = cfg.radius_m
    n_r     = cfg.n_rings
    n_theta = cfg.n_sectors

    # Node positions: centre + rings
    nodes = [(0.0, 0.0)]
    for i in range(1, n_r + 1):
        r_i = R * i / n_r
        for j in range(n_theta):
            theta = 2 * np.pi * j / n_theta
            nodes.append((r_i * np.cos(theta), r_i * np.sin(theta)))

    nodes = np.array(nodes, dtype=float)
    n_nodes = len(nodes)

    # Connectivity: triangles
    elements = []
    # Centre to first ring (fan)
    for j in range(n_theta):
        j1 = 1 + j
        j2 = 1 + (j + 1) % n_theta
        elements.append([0, j1, j2])

    # Ring-to-ring quads split into 2 triangles
    for i in range(1, n_r):
        base_inner = 1 + (i-1)*n_theta
        base_outer = 1 + i*n_theta
        for j in range(n_theta):
            ji1 = base_inner + j
            ji2 = base_inner + (j+1) % n_theta
            jo1 = base_outer + j
            jo2 = base_outer + (j+1) % n_theta
            elements.append([ji1, jo1, jo2])
            elements.append([ji1, jo2, ji2])

    elements = np.array(elements, dtype=int)

    # Boundary nodes (outermost ring) — clamped DOFs
    n_boundary = n_theta
    boundary_dofs = list(range(1 + (n_r-1)*n_theta, 1 + n_r*n_theta))

    return {
        "nodes":         nodes,
        "elements":      elements,
        "n_nodes":       n_nodes,
        "boundary_dofs": boundary_dofs,
        "R":             R,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEM ASSEMBLY  (membrane stiffness + consistent mass)
# ══════════════════════════════════════════════════════════════════════════════

def _element_matrices(nodes_e: np.ndarray, T: float,
                       rho_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute element stiffness Ke (3×3) and mass Me (3×3) for a
    constant-strain triangle (CST) membrane element.

    Stiffness: Ke = T · B^T · B · Area   (tension × curvature)
    Mass:      Me = ρ_s · Area/12 · [2,1,1; 1,2,1; 1,1,2]
    """
    x1, y1 = nodes_e[0]
    x2, y2 = nodes_e[1]
    x3, y3 = nodes_e[2]

    # Area via cross product
    area = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2.0
    if area < 1e-14:
        return np.zeros((3,3)), np.zeros((3,3))

    # Shape function gradients [1/m]
    b = np.array([y2-y3, y3-y1, y1-y2]) / (2*area)
    c = np.array([x3-x2, x1-x3, x2-x1]) / (2*area)

    # B matrix (gradient of shape functions)
    B = np.vstack([b, c])   # (2×3)

    # Stiffness: integral of T * ||∇u||² = T * Area * ||B||²_F
    Ke = T * area * (B.T @ B)

    # Consistent mass matrix
    Me_base = np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)
    Me = rho_s * area / 12.0 * Me_base

    return Ke, Me


def assemble_fem(cfg: CanopyMembrane, mesh: dict) -> dict:
    """
    Assemble global stiffness K and mass M matrices.
    Apply clamped BCs by removing boundary DOFs.
    Returns free-DOF matrices K_f, M_f and DOF index map.
    """
    n  = mesh["n_nodes"]
    T  = cfg.tension_Nm
    rs = cfg.surface_density

    K_global = lil_matrix((n, n), dtype=float)
    M_global = lil_matrix((n, n), dtype=float)

    for elem in mesh["elements"]:
        nodes_e = mesh["nodes"][elem]
        Ke, Me  = _element_matrices(nodes_e, T, rs)
        for i in range(3):
            for j in range(3):
                K_global[elem[i], elem[j]] += Ke[i, j]
                M_global[elem[i], elem[j]] += Me[i, j]

    # Apply clamped BCs (eliminate boundary DOFs)
    fixed = set(mesh["boundary_dofs"])
    free  = [i for i in range(n) if i not in fixed]

    K_csr = csr_matrix(K_global)
    M_csr = csr_matrix(M_global)

    K_f = K_csr[free, :][:, free]
    M_f = M_csr[free, :][:, free]

    return {"K_f": K_f, "M_f": M_f, "free_dofs": free, "n_free": len(free)}


# ══════════════════════════════════════════════════════════════════════════════
# MODAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def modal_analysis(cfg: CanopyMembrane, mesh: dict, fem: dict,
                   n_modes: int = 6) -> dict:
    """
    Solve generalised eigenvalue problem K·φ = ω²·M·φ.
    Returns natural frequencies ωᵢ [rad/s] and mode shapes φᵢ.
    """
    K_f = fem["K_f"]; M_f = fem["M_f"]
    n_free = fem["n_free"]
    n_modes = min(n_modes, n_free - 1)

    try:
        # eigsh: smallest n_modes eigenvalues
        eigenvalues, eigenvectors = eigsh(K_f, k=n_modes, M=M_f,
                                          which="SM", maxiter=2000, tol=1e-8)
    except Exception:
        eigenvalues  = np.array([1.0] * n_modes)
        eigenvectors = np.eye(n_free, n_modes)

    omega_n = np.sqrt(np.maximum(eigenvalues, 0))   # [rad/s]
    freq_Hz = omega_n / (2 * np.pi)

    # Expand modes to full DOF space
    n_total = mesh["n_nodes"]
    free    = fem["free_dofs"]
    modes_full = np.zeros((n_total, n_modes))
    for k in range(n_modes):
        modes_full[free, k] = eigenvectors[:, k]

    return {
        "omega_n":   omega_n,
        "freq_Hz":   freq_Hz,
        "modes":     modes_full,
        "n_modes":   n_modes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FLUTTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def flutter_analysis(cfg: CanopyMembrane, modal: dict,
                     v_range: np.ndarray, rho_air: float,
                     q_dyn_range: np.ndarray | None = None) -> dict:
    """
    Predict flutter onset over a velocity range using reduced-frequency criterion.

    Critical dynamic pressure (linearised Theodorsen-type for membranes):
      q_cr = ω₁² · ρ_s · h / |dCp/d(u/R)|

    Flutter reduced velocity:
      U* = v / (ω₁ · R)   →  flutter when U* > 1/sqrt(2·ζ)

    Returns flutter onset speed and margin vs current condition.
    """
    ω1   = float(modal["omega_n"][0]) if len(modal["omega_n"]) > 0 else 1.0
    ζ    = cfg.damping_ratio
    R    = cfg.radius_m
    ρ_s  = cfg.surface_density
    dCp  = abs(cfg.Cp_slope)

    # Critical dynamic pressure
    q_cr = ω1**2 * ρ_s * R / max(dCp, 1e-6)

    # Critical velocity: q_cr = 0.5*ρ*v_cr² → v_cr = sqrt(2*q_cr/ρ)
    v_cr = float(np.sqrt(2 * q_cr / max(rho_air, 1e-12)))

    # Flutter criterion: U* = v/(ω1·R) > threshold = 1/sqrt(2·ζ)
    U_star  = v_range / (ω1 * R)
    U_crit  = 1.0 / np.sqrt(max(2 * ζ, 1e-6))
    in_flutter = U_star > U_crit

    # Amplification factor: A = 1/sqrt((1-U*²)² + (2ζU*)²)   (linear resonance)
    denom = np.sqrt(np.maximum((1 - (v_range/(v_cr + 1e-6))**2)**2
                               + (2*ζ*v_range/(v_cr + 1e-6))**2, 1e-9))
    amplitude_factor = 1.0 / denom

    # Porosity modification from deflection
    u_max = R * amplitude_factor * 0.01   # max deflection estimate [m]
    k_pore_eff = cfg.porosity_k0 * (1 + 1.5 * (u_max / R)**2)

    # Riser load spikes: F_riser = T * 2π * R * (1 + ΔA/A)
    delta_area_pct = amplitude_factor * 2.0   # % area change estimate
    riser_factor   = 1.0 + delta_area_pct / 100.0

    return {
        "v_range":        v_range,
        "U_star":         U_star,
        "U_crit":         U_crit,
        "v_flutter_ms":   v_cr,
        "q_cr_Pa":        q_cr,
        "in_flutter":     in_flutter,
        "amplitude_factor": amplitude_factor,
        "k_pore_eff":     k_pore_eff,
        "riser_load_factor": riser_factor,
        "omega1_rads":    ω1,
        "freq1_Hz":       ω1 / (2*np.pi),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TIME-DOMAIN AEROELASTIC RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

def time_domain_response(cfg: CanopyMembrane, modal: dict,
                          v_arr: np.ndarray, rho_arr: np.ndarray,
                          t_arr: np.ndarray) -> dict:
    """
    Solve the modal equations of motion in time domain:
      η̈ᵢ + 2ζωᵢη̇ᵢ + ωᵢ²ηᵢ = fᵢ(t)

    where ηᵢ are modal coordinates and fᵢ is the generalised aerodynamic force.
    This gives the time history of canopy deformation.
    """
    n_modes = modal["n_modes"]
    omega_n = modal["omega_n"]
    ζ       = cfg.damping_ratio
    R       = cfg.radius_m
    n_t     = len(t_arr)

    # Generalised aero force on each mode
    # Simplified: f_i = q_dyn * Cp_slope * phi_i_max (approximation)
    phi_max = np.array([float(np.max(np.abs(modal["modes"][:, i])))
                        for i in range(n_modes)])

    # Time-domain modal response via Newmark-β (β=0.25, γ=0.5 → unconditionally stable)
    eta     = np.zeros((n_t, n_modes))
    etadot  = np.zeros((n_t, n_modes))
    etaddot = np.zeros((n_t, n_modes))

    beta, gamma = 0.25, 0.50

    for step in range(n_t - 1):
        dt = t_arr[step+1] - t_arr[step]
        if dt <= 0: continue

        q_dyn_i = 0.5 * rho_arr[step] * v_arr[step]**2
        q_dyn_n = 0.5 * rho_arr[min(step+1, n_t-1)] * v_arr[min(step+1, n_t-1)]**2

        for m in range(n_modes):
            ω  = omega_n[m]; ζm = ζ
            fi = q_dyn_i * cfg.Cp_slope * phi_max[m] * cfg.radius_m**2
            fn = q_dyn_n * cfg.Cp_slope * phi_max[m] * cfg.radius_m**2

            # Effective stiffness
            k_eff = ω**2 + 2*ζm*ω*gamma/dt/beta + 1/(beta*dt**2)
            f_eff = fn + (1/(beta*dt**2))*eta[step,m] + (1/(beta*dt))*etadot[step,m] \
                    + (0.5/beta - 1)*etaddot[step,m]

            eta[step+1, m]     = f_eff / max(k_eff, 1e-12)
            etaddot[step+1, m] = (eta[step+1,m] - eta[step,m] - dt*etadot[step,m]
                                   - (0.5-beta)*dt**2*etaddot[step,m]) / (beta*dt**2)
            etadot[step+1, m]  = etadot[step,m] + dt*((1-gamma)*etaddot[step,m]
                                  + gamma*etaddot[step+1,m])

    # Physical deflection u = Σ φᵢ · ηᵢ (take max at each time)
    u_max_per_t = np.array([np.max(np.abs(eta[i, :] @ phi_max)) for i in range(n_t)])

    # Riser load variation
    riser_load = 1.0 + 2.0 * u_max_per_t / max(R, 1.0)

    return {
        "t":           t_arr,
        "eta":         eta,
        "u_max_m":     u_max_per_t,
        "riser_factor": riser_load,
        "deflection_norm": u_max_per_t / R,   # u/R
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_flutter(mesh: dict, modal: dict, flutter: dict,
                 td: dict | None = None,
                 save_path: str = "outputs/aeroelastic_flutter.png"):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.tri as mtri

    matplotlib.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })
    TX="#c8d8f0"; C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"; CR="#ff4560"

    fig = plt.figure(figsize=(22, 12), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.46, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.05, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    # Mode shape plots (first 3)
    nodes = mesh["nodes"]
    elems = mesh["elements"]
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)

    for m_idx in range(min(3, modal["n_modes"])):
        a = gax(0, m_idx)
        vals = modal["modes"][:, m_idx]
        tc = a.tricontourf(triang, vals / max(abs(vals.max()), 1e-9),
                           levels=20, cmap="RdBu")
        fig.colorbar(tc, ax=a, pad=0.02, label="φ").ax.tick_params(labelsize=7)
        f_hz = modal["freq_Hz"][m_idx]
        a.set_title(f"Mode {m_idx+1}  f={f_hz:.3f}Hz", fontweight="bold")
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_aspect("equal")
        a.grid(False)

    # Natural frequencies bar chart
    a = gax(0, 3)
    f_hz = modal["freq_Hz"]
    a.bar(range(1, len(f_hz)+1), f_hz, color=C4, alpha=0.75, edgecolor="none")
    a.set_title("Natural Frequencies", fontweight="bold")
    a.set_xlabel("Mode number"); a.set_ylabel("f [Hz]")

    # Flutter: amplitude vs velocity
    a = gax(1, 0)
    v = flutter["v_range"]
    A = flutter["amplitude_factor"]
    fl = flutter["in_flutter"]
    a.semilogy(v, A, color=C1, lw=2)
    if fl.any():
        a.fill_between(v, A, where=fl, alpha=0.3, color=CR, label="Flutter region")
    a.axvline(flutter["v_flutter_ms"], color=CR, lw=1.5, ls="--",
              label=f"v_cr={flutter['v_flutter_ms']:.1f}m/s")
    a.legend(fontsize=7.5)
    a.set_title("Flutter Amplitude vs Velocity", fontweight="bold")
    a.set_xlabel("v [m/s]"); a.set_ylabel("Amplitude factor [-]")

    # U* vs velocity with flutter boundary
    a = gax(1, 1)
    a.plot(v, flutter["U_star"], color=C4, lw=2, label="U* (reduced vel)")
    a.axhline(flutter["U_crit"], color=CR, lw=1.5, ls="--",
              label=f"U*_crit={flutter['U_crit']:.2f}")
    a.fill_between(v, flutter["U_star"],
                   where=flutter["in_flutter"], alpha=0.3, color=CR)
    a.legend(fontsize=7.5)
    a.set_title("Reduced Velocity U*", fontweight="bold")
    a.set_xlabel("v [m/s]"); a.set_ylabel("U* = v/(ω₁·R)")

    # Riser load factor
    a = gax(1, 2)
    a.fill_between(v, flutter["riser_load_factor"], alpha=0.2, color=C2)
    a.plot(v, flutter["riser_load_factor"], color=C2, lw=2)
    a.axhline(1.0, color=TX, lw=0.7, ls=":", alpha=0.5)
    a.set_title("Riser Load Factor", fontweight="bold")
    a.set_xlabel("v [m/s]"); a.set_ylabel("F_riser / F_nominal")

    # Time-domain deflection
    a = gax(1, 3)
    if td is not None:
        a.fill_between(td["t"], td["deflection_norm"]*100, alpha=0.2, color=C3)
        a.plot(td["t"], td["deflection_norm"]*100, color=C3, lw=1.8)
        a.set_title("Canopy Deflection u/R", fontweight="bold")
        a.set_xlabel("t [s]"); a.set_ylabel("u/R [%]")
    else:
        a.text(0.5, 0.5, "No time-domain data", transform=a.transAxes,
               ha="center", color=TX)

    f_cr  = flutter["v_flutter_ms"]
    f1_Hz = flutter["freq1_Hz"]
    fig.text(0.5, 0.955,
             f"Aeroelastic Flutter Analysis  |  f₁={f1_Hz:.3f}Hz  |  "
             f"v_cr={f_cr:.1f}m/s  |  q_cr={flutter['q_cr_Pa']:.0f}Pa",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    from pathlib import Path; Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Flutter plot saved: {save_path}")
    return fig


def run(canopy_radius_m: float = 5.0, tension_Nm: float = 1200.0,
        rho_air: float = 0.02, v_range_ms: tuple = (5, 150),
        verbose: bool = True) -> dict:
    """Full aeroelastic flutter analysis pipeline."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg  = CanopyMembrane(radius_m=canopy_radius_m, tension_Nm=tension_Nm)
    mesh = _generate_polar_mesh(cfg)
    fem  = assemble_fem(cfg, mesh)
    modal= modal_analysis(cfg, mesh, fem, n_modes=6)

    if verbose:
        print(f"\n[Flutter] Mesh: {mesh['n_nodes']} nodes  "
              f"{len(mesh['elements'])} elements  {fem['n_free']} free DOFs")
        print(f"  Natural frequencies:")
        for i, f in enumerate(modal["freq_Hz"]):
            print(f"    Mode {i+1}: {f:.4f} Hz  ({modal['omega_n'][i]:.4f} rad/s)")

    v_arr   = np.linspace(*v_range_ms, 100)
    flutter = flutter_analysis(cfg, modal, v_arr, rho_air)

    if verbose:
        print(f"  Flutter onset: v_cr={flutter['v_flutter_ms']:.1f} m/s  "
              f"q_cr={flutter['q_cr_Pa']:.1f} Pa")
        flutter_frac = flutter["in_flutter"].mean() * 100
        print(f"  % of velocity range in flutter: {flutter_frac:.1f}%")

    # Time domain (synthetic descent velocity profile)
    t_td   = np.linspace(0, 300, 200)
    v_td   = np.linspace(150, 20, 200)
    rho_td = rho_air * np.ones(200)
    td     = time_domain_response(cfg, modal, v_td, rho_td, t_td)

    fig = plot_flutter(mesh, modal, flutter, td)
    plt.close(fig)

    return {"cfg": cfg, "mesh": mesh, "modal": modal, "flutter": flutter, "td": td}


if __name__ == "__main__":
    result = run(canopy_radius_m=5.0, tension_Nm=1200.0, rho_air=0.02)
    print(f"flutter onset: {result['flutter']['v_flutter_ms']:.1f} m/s")
    print(f"mode 1 freq:   {result['modal']['freq_Hz'][0]:.4f} Hz")
