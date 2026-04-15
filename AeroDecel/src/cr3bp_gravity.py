"""
src/cr3bp_gravity.py — Circular Restricted 3-Body Problem Gravity
==================================================================
Replaces the standard inverse-square gravity in SixDOFDynamics with
the full CR3BP formulation for accurate entry interface dynamics in
3-body systems (Earth-Moon, Mars-Phobos, Jupiter-Europa, etc.).

CR3BP Physics
-------------
In the rotating frame of the two primaries (mass M1 and M2), the
equations of motion for a test particle include:

  ẍ - 2ω·ẏ = ∂Ω/∂x
  ÿ + 2ω·ẋ = ∂Ω/∂y
  z̈         = ∂Ω/∂z

where the effective potential Ω (pseudo-potential) is:

  Ω = (1/2)ω²(x² + y²) + (1-μ)/r₁ + μ/r₂

  μ   = M2/(M1+M2)       mass parameter
  r₁  = ||r - r₁_pos||   distance to primary (heavier body)
  r₂  = ||r - r₂_pos||   distance to secondary (lighter body)
  ω   = √(G(M1+M2)/L³)  rotation rate of the synodic frame

The Coriolis acceleration (-2ω × v_rot) and centrifugal acceleration
(ω × (ω × r)) are both included. These are the terms that create
the Lagrange points and chaotic trajectories missing from simple
inverse-square gravity.

Dimensional Conversion
----------------------
The CR3BP is normalised: L = semi-major axis of M2 orbit, T = 1/ω.
We convert to/from SI using the system's characteristic L and T.

Supported Systems
-----------------
  earth_moon  : μ=0.01215,  L=384,400 km,  T=2,360,584 s
  mars_phobos : μ=1.66e-8,  L=9,376 km,    T=27,553 s
  mars_deimos : μ=2.74e-10, L=23,463 km,   T=109,075 s
  jupiter_europa : μ=2.53e-5, L=671,100 km
  sun_earth   : μ=3.00e-6,  L=149,597,870 km

Usage
-----
  # Drop-in replacement for SixDOFDynamics._gravity_inertial()
  cr3bp = CR3BPGravity("mars_phobos")
  g_vec = cr3bp.gravity_and_pseudo(r_I, v_I)

  # Integrate full CR3BP trajectory
  traj = cr3bp.integrate(r0, v0, t_span)

  # Jacobi constant (conserved quantity — for validation)
  C = cr3bp.jacobi_constant(r_norm, v_norm)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CR3BPSystem:
    name:       str
    mu:         float    # mass parameter μ = M2/(M1+M2)
    L_m:        float    # characteristic length [m]  (semi-major axis of secondary)
    T_s:        float    # characteristic time [s]    (1/ω)
    M1_name:    str      # primary body name
    M2_name:    str      # secondary body name
    # Derived
    omega:      float = 1.0   # angular velocity in normalised units = 1

    @property
    def G_ms(self) -> float:
        """Gravitational parameter G*(M1+M2) in SI [m³/s²]."""
        return self.L_m**3 / self.T_s**2

    def to_normalised(self, r_SI: np.ndarray, v_SI: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Convert SI position/velocity to CR3BP normalised units."""
        return r_SI / self.L_m, v_SI * self.T_s / self.L_m

    def to_SI(self, r_norm: np.ndarray, v_norm: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        """Convert normalised CR3BP units back to SI."""
        return r_norm * self.L_m, v_norm * self.L_m / self.T_s

    def accel_to_SI(self, a_norm: np.ndarray) -> np.ndarray:
        """Convert normalised acceleration to SI [m/s²]."""
        return a_norm * self.L_m / self.T_s**2


# ── Known systems ─────────────────────────────────────────────────────────────
SYSTEMS: dict[str, CR3BPSystem] = {
    "earth_moon": CR3BPSystem(
        name="Earth-Moon", mu=1.21506683e-2,
        L_m=3.84400e8, T_s=2.360584e6,
        M1_name="Earth", M2_name="Moon",
    ),
    "mars_phobos": CR3BPSystem(
        name="Mars-Phobos", mu=1.66e-8,
        L_m=9.376e6, T_s=27553.0,
        M1_name="Mars", M2_name="Phobos",
    ),
    "mars_deimos": CR3BPSystem(
        name="Mars-Deimos", mu=2.74e-10,
        L_m=2.3463e7, T_s=109075.0,
        M1_name="Mars", M2_name="Deimos",
    ),
    "jupiter_europa": CR3BPSystem(
        name="Jupiter-Europa", mu=2.528e-5,
        L_m=6.711e8, T_s=3.067e5,
        M1_name="Jupiter", M2_name="Europa",
    ),
    "sun_earth": CR3BPSystem(
        name="Sun-Earth", mu=3.0034e-6,
        L_m=1.495978707e11, T_s=5.022642e6,
        M1_name="Sun", M2_name="Earth",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# CR3BP GRAVITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CR3BPGravity:
    """
    CR3BP gravity model — drop-in replacement for SixDOFDynamics._gravity_inertial().

    The synodic (rotating) frame has:
      • Primary (M1) at (-μ, 0, 0)
      • Secondary (M2) at (1-μ, 0, 0)
      • Origin at system barycentre
      • Frame rotates at ω = 1 (normalised) about z-axis

    To use in SixDOFDynamics, initialise CR3BPGravity and call
    gravity_and_pseudo(r_I_SI, v_I_SI) → returns SI acceleration.
    """

    def __init__(self, system: str | CR3BPSystem = "mars_phobos"):
        if isinstance(system, str):
            if system not in SYSTEMS:
                raise ValueError(f"Unknown system '{system}'. "
                                 f"Available: {list(SYSTEMS.keys())}")
            self.sys = SYSTEMS[system]
        else:
            self.sys = system

        self.mu = self.sys.mu

    # ── Core CR3BP equations (normalised units) ───────────────────────────────

    def _effective_potential_gradient(self, r_norm: np.ndarray) -> np.ndarray:
        """
        Gradient of the CR3BP pseudo-potential Ω in normalised units.
        ∇Ω = (∂Ω/∂x, ∂Ω/∂y, ∂Ω/∂z)

        Ω = ½(x² + y²) + (1-μ)/r₁ + μ/r₂
        """
        x, y, z = r_norm
        mu = self.mu

        # Position of primaries in synodic frame
        r1 = np.sqrt((x + mu)**2         + y**2 + z**2)   # distance to M1 at (-μ, 0, 0)
        r2 = np.sqrt((x - (1-mu))**2     + y**2 + z**2)   # distance to M2 at (1-μ, 0, 0)

        # Safety floor
        r1 = max(r1, 1e-6); r2 = max(r2, 1e-6)

        # ∂Ω/∂x = x - (1-μ)(x+μ)/r₁³ - μ(x-1+μ)/r₂³
        dOdx = x - (1-mu)*(x + mu)/r1**3 - mu*(x - (1-mu))/r2**3
        # ∂Ω/∂y = y - (1-μ)y/r₁³ - μy/r₂³
        dOdy = y - (1-mu)*y/r1**3       - mu*y/r2**3
        # ∂Ω/∂z = -(1-μ)z/r₁³ - μz/r₂³    (no centrifugal term in z)
        dOdz = -(1-mu)*z/r1**3          - mu*z/r2**3

        return np.array([dOdx, dOdy, dOdz])

    def _coriolis(self, v_norm: np.ndarray) -> np.ndarray:
        """
        Coriolis acceleration: -2ω × v  where ω = [0, 0, 1] (normalised).
        = [-2ω·vy, +2ω·vx, 0]  (ω=1 in normalised)
        """
        return np.array([-2*v_norm[1], 2*v_norm[0], 0.0])

    def total_acceleration_normalised(self, r_norm: np.ndarray,
                                       v_norm: np.ndarray) -> np.ndarray:
        """
        Full CR3BP acceleration in the synodic frame (normalised units):
        a = ∇Ω - 2ω×v  (Coriolis is already included in ∇Ω formulation)

        Note: the CR3BP equations ẍ-2ẏ = ∂Ω/∂x etc. combine both the
        pseudo-potential gradient AND Coriolis into a single expression.
        """
        grad_Omega = self._effective_potential_gradient(r_norm)
        coriolis   = self._coriolis(v_norm)
        return grad_Omega + coriolis

    def gravity_and_pseudo(self, r_SI: np.ndarray, v_SI: np.ndarray) -> np.ndarray:
        """
        Full CR3BP acceleration in SI [m/s²], ready for SixDOFDynamics.

        Parameters
        ----------
        r_SI : position in inertial (non-rotating) frame centred at barycentre [m]
               NOTE: r_SI is assumed to be expressed in synodic frame coordinates.
               For a pure EDL sim, the planet IS at position (-μ, 0, 0)*L,
               so the entry capsule position is naturally in this frame.
        v_SI : velocity in SI [m/s] in the synodic frame

        Returns
        -------
        acceleration in SI [m/s²]
        """
        r_n, v_n = self.sys.to_normalised(r_SI, v_SI)
        a_n      = self.total_acceleration_normalised(r_n, v_n)
        return self.sys.accel_to_SI(a_n)

    # ── Conserved quantity ────────────────────────────────────────────────────

    def jacobi_constant(self, r_norm: np.ndarray, v_norm: np.ndarray) -> float:
        """
        Jacobi constant C (conserved in CR3BP):
          C = 2Ω - v²  =  x²+y² + 2(1-μ)/r₁ + 2μ/r₂ - v²

        Useful for validation: C should remain constant along any trajectory.
        """
        x, y, z = r_norm; mu = self.mu
        r1 = max(np.sqrt((x + mu)**2        + y**2 + z**2), 1e-12)
        r2 = max(np.sqrt((x - (1-mu))**2    + y**2 + z**2), 1e-12)
        Omega2 = x**2 + y**2 + 2*(1-mu)/r1 + 2*mu/r2
        v2     = float(np.dot(v_norm, v_norm))
        return float(Omega2 - v2)

    # ── Lagrange points ───────────────────────────────────────────────────────

    def lagrange_points(self) -> dict:
        """
        Compute L1-L5 positions in normalised synodic frame.
        L4, L5 are trivially at (0.5-μ, ±√3/2, 0).
        L1, L2, L3 require Newton iteration on the collinear solutions.
        """
        from scipy.optimize import brentq
        mu = self.mu

        # Collinear points: solve dΩ/dx = 0 along y=z=0
        def dOdx_col(x):
            r1 = abs(x + mu); r2 = abs(x - (1-mu))
            return x - (1-mu)*(x+mu)/r1**3 - mu*(x-(1-mu))/r2**3

        L1_x = brentq(dOdx_col, (1-mu)-0.99*(1-mu), (1-mu)+0.99*(1-mu-0.01))
        L2_x = brentq(dOdx_col, (1-mu)+0.001, 2.0)
        L3_x = brentq(dOdx_col, -2.0, -mu-0.001)

        return {
            "L1": np.array([L1_x,           0, 0]),
            "L2": np.array([L2_x,           0, 0]),
            "L3": np.array([L3_x,           0, 0]),
            "L4": np.array([0.5 - mu,       np.sqrt(3)/2, 0]),
            "L5": np.array([0.5 - mu,      -np.sqrt(3)/2, 0]),
            "M1": np.array([-mu,            0, 0]),
            "M2": np.array([1-mu,           0, 0]),
        }

    # ── Full trajectory integrator ────────────────────────────────────────────

    def integrate(self, r0_SI: np.ndarray, v0_SI: np.ndarray,
                   t_span_s: tuple[float, float],
                   n_points: int = 2000,
                   events: list | None = None) -> dict:
        """
        Integrate the CR3BP equations of motion in SI units.

        Returns dict with: t, r (Nx3), v (Nx3), jacobi, jacobi_drift
        """
        L  = self.sys.L_m; T = self.sys.T_s
        r0n, v0n = self.sys.to_normalised(r0_SI, v0_SI)

        def rhs_norm(t, y):
            r = y[:3]; v = y[3:]
            a = self.total_acceleration_normalised(r, v)
            return np.concatenate([v, a])

        y0     = np.concatenate([r0n, v0n])
        t_eval = np.linspace(t_span_s[0]/T, t_span_s[1]/T, n_points)

        sol = solve_ivp(rhs_norm, (t_eval[0], t_eval[-1]), y0,
                        t_eval=t_eval, method="DOP853",
                        rtol=1e-10, atol=1e-12, dense_output=False)

        r_norm = sol.y[:3, :].T
        v_norm = sol.y[3:, :].T

        # Jacobi constant along trajectory (should be constant — drift = error)
        C_arr = np.array([self.jacobi_constant(r_norm[i], v_norm[i])
                           for i in range(len(sol.t))])

        # Back to SI
        r_SI_arr = r_norm * L
        v_SI_arr = v_norm * L / T

        return {
            "t_s":         sol.t * T,
            "t_norm":      sol.t,
            "r_SI":        r_SI_arr,    # (N, 3) [m]
            "v_SI":        v_SI_arr,    # (N, 3) [m/s]
            "r_norm":      r_norm,
            "v_norm":      v_norm,
            "jacobi":      C_arr,
            "jacobi_drift":float(C_arr.std()),   # → 0 for perfect integration
            "system":      self.sys.name,
            "converged":   sol.success,
        }

    # ── Patch into SixDOFDynamics ─────────────────────────────────────────────

    def patch_sixdof(self, dynamics_instance) -> None:
        """
        Monkey-patch a SixDOFDynamics instance to use CR3BP gravity.

        Usage
        -----
            cr3bp = CR3BPGravity("mars_phobos")
            dyn   = SixDOFDynamics(vehicle, planet_atm)
            cr3bp.patch_sixdof(dyn)
            # Now dyn.rhs() uses CR3BP gravity
        """
        cr3bp_ref = self

        def _gravity_cr3bp(self_dyn, r_I: np.ndarray) -> np.ndarray:
            """CR3BP gravity — replaces standard inverse-square."""
            # We need velocity from the state. Store it from last rhs call.
            v_I = getattr(self_dyn, "_last_v_I", np.zeros(3))
            return cr3bp_ref.gravity_and_pseudo(r_I, v_I)

        # Store the original method for comparison
        dynamics_instance._gravity_inertial_original = dynamics_instance._gravity_inertial
        dynamics_instance._gravity_inertial = lambda r: _gravity_cr3bp(dynamics_instance, r)

        print(f"  ✓ CR3BP gravity patched: system={self.sys.name}  μ={self.mu}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_cr3bp(traj: dict, system_name: str = "mars_phobos",
               save_path: str = "outputs/cr3bp_trajectory.png"):
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

    sys_data = SYSTEMS.get(system_name, SYSTEMS["mars_phobos"])
    cr3bp    = CR3BPGravity(sys_data)
    lpts     = cr3bp.lagrange_points()

    fig = plt.figure(figsize=(22, 11), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.05, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    # Trajectory in synodic frame
    rn = traj["r_norm"]
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor("#0d1526"); ax.grid(True, alpha=0.25)
    ax.tick_params(colors=TX); ax.spines[:].set_color("#2a3d6e")
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor("#0d1526"); ax.grid(True, alpha=0.25)
    ax.tick_params(colors=TX); ax.spines[:].set_color("#2a3d6e")
    sc = ax.scatter(rn[:, 0], rn[:, 1], c=traj["t_norm"], cmap="plasma", s=3, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="t [norm]").ax.tick_params(labelsize=7)
    # Primary bodies
    ax.scatter([lpts["M1"][0]], [lpts["M1"][1]], s=200, color="#ffd700", zorder=8, marker="o", label=sys_data.M1_name)
    ax.scatter([lpts["M2"][0]], [lpts["M2"][1]], s=80,  color="#aaaaff", zorder=8, marker="o", label=sys_data.M2_name)
    # Lagrange points
    for key in ["L1","L2","L3","L4","L5"]:
        L = lpts[key]
        ax.scatter([L[0]], [L[1]], s=25, color=C3, marker="x", zorder=7)
        ax.text(L[0]+0.02, L[1]+0.02, key, fontsize=7, color=C3)
    ax.set_xlabel("x [norm]"); ax.set_ylabel("y [norm]")
    ax.set_title(f"CR3BP Trajectory — {sys_data.name}", fontweight="bold")
    ax.legend(fontsize=8); ax.set_aspect("equal")

    # Jacobi constant (conservation check)
    a = gax(0, 2)
    C_arr = traj["jacobi"]
    a.plot(traj["t_norm"], C_arr, color=C1, lw=1.5)
    a.axhline(C_arr[0], color=C3, lw=0.8, ls="--", label=f"C₀={C_arr[0]:.6f}")
    a.set_title("Jacobi Constant (conserved)", fontweight="bold")
    a.set_xlabel("t [norm]"); a.set_ylabel("C")
    a.legend(fontsize=8)
    # Drift magnitude
    drift_pct = abs((C_arr - C_arr[0]) / max(abs(C_arr[0]), 1e-12)) * 100
    a_twin = a.twinx()
    a_twin.plot(traj["t_norm"], drift_pct, color=CR, lw=0.8, alpha=0.6)
    a_twin.set_ylabel("Drift [%]", color=CR); a_twin.tick_params(axis="y", colors=CR)
    a_twin.spines[:].set_color("#2a3d6e"); a_twin.set_facecolor("#0d1526")

    # Velocities
    a = gax(0, 3)
    v_mag = np.linalg.norm(traj["v_SI"], axis=1) / 1e3
    a.plot(traj["t_s"], v_mag, color=C4, lw=1.8)
    a.set_title("Speed [km/s]", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("v [km/s]")

    # Distances to both bodies
    a = gax(1, 0)
    mu_ = cr3bp.mu
    r1_dist = np.sqrt((rn[:,0]+mu_)**2         + rn[:,1]**2 + rn[:,2]**2) * sys_data.L_m / 1e3
    r2_dist = np.sqrt((rn[:,0]-(1-mu_))**2     + rn[:,1]**2 + rn[:,2]**2) * sys_data.L_m / 1e3
    a.fill_between(traj["t_s"], r1_dist, alpha=0.15, color="#ffd700")
    a.plot(traj["t_s"], r1_dist, color="#ffd700", lw=1.5, label=sys_data.M1_name)
    a.fill_between(traj["t_s"], r2_dist, alpha=0.15, color="#9d60ff")
    a.plot(traj["t_s"], r2_dist, color="#9d60ff", lw=1.5, label=sys_data.M2_name)
    a.legend(fontsize=8); a.set_xlabel("t [s]"); a.set_ylabel("Distance [km]")
    a.set_title("Distances to Bodies", fontweight="bold")

    # Lagrange point positions map
    a = gax(1, 1)
    for key, L in lpts.items():
        color_ = {"M1":"#ffd700","M2":"#aaaaff","L1":CR,"L2":C2,"L3":C3,"L4":C1,"L5":C4}.get(key,"#888")
        marker_ = "o" if "M" in key else "x"
        a.scatter([L[0]], [L[1]], s=60 if "M" in key else 40, color=color_, marker=marker_)
        a.text(L[0]+0.02, L[1]+0.02, key, fontsize=8, color=color_)
    a.scatter(rn[0,0], rn[0,1], s=100, color=C1, marker="D", zorder=5, label="Entry")
    a.scatter(rn[-1,0], rn[-1,1], s=100, color=C3, marker="*", zorder=5, label="Final")
    a.legend(fontsize=8); a.set_xlabel("x [norm]"); a.set_ylabel("y [norm]")
    a.set_title("Lagrange Points + Trajectory", fontweight="bold"); a.set_aspect("equal")

    # 3-D trajectory
    try:
        from mpl_toolkits.mplot3d import Axes3D
        a3 = fig.add_subplot(gs[1, 2:])
        a3.set_facecolor("#0d1526")
        a3.scatter(rn[:,0], rn[:,1], rn[:,2], c=traj["t_norm"], cmap="plasma", s=2, alpha=0.7)
        a3.scatter([lpts["M1"][0]],[lpts["M1"][1]],[0], s=200, color="#ffd700", zorder=8)
        a3.scatter([lpts["M2"][0]],[lpts["M2"][1]],[0], s=80,  color="#aaaaff", zorder=8)
        a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z")
        a3.set_title("3-D Trajectory", fontweight="bold")
        a3.tick_params(colors=TX, labelsize=7)
    except Exception:
        pass

    fig.text(0.5, 0.955,
             f"CR3BP Gravity — {sys_data.name}  |  μ={cr3bp.mu:.2e}  |  "
             f"L={sys_data.L_m/1e3:.0f}km  |  Jacobi drift={traj['jacobi_drift']:.2e}",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    from pathlib import Path; Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ CR3BP plot saved: {save_path}")
    plt.close(fig)


def run(system: str = "earth_moon", n_orbits: float = 2.0,
        verbose: bool = True) -> dict:
    """Demo: halo orbit-like trajectory in chosen 3-body system."""
    import matplotlib; matplotlib.use("Agg")

    cr3bp = CR3BPGravity(system)
    sys_  = cr3bp.sys

    # Near-L2 initial conditions for a halo-like orbit (normalised)
    lpts = cr3bp.lagrange_points()
    L2   = lpts["L2"]

    # Small perturbation from L2 to start an orbit (Richardson approximation)
    Az    = 0.1   # out-of-plane amplitude (normalised)
    Ax    = 0.15  # in-plane amplitude
    r0n   = L2 + np.array([Ax, 0.0, Az])
    # Velocity from CR3BP linear stability at L2
    v0n   = np.array([0.0, -0.12, 0.0])

    # Convert to SI
    r0_SI = r0n * sys_.L_m
    v0_SI = v0n * sys_.L_m / sys_.T_s

    # Period estimate: ~1 normalised time unit per revolution
    T_period_norm = 3.0   # approximate halo period in norm units
    t_span_s = (0.0, T_period_norm * n_orbits * sys_.T_s)

    if verbose:
        print(f"\n[CR3BP] System={sys_.name}  μ={cr3bp.mu:.2e}  "
              f"L={sys_.L_m/1e3:.0f}km  T={sys_.T_s:.0f}s")
        print(f"  Integrating {T_period_norm*n_orbits:.1f} normalised orbits")

    traj = cr3bp.integrate(r0_SI, v0_SI, t_span_s, n_points=3000)

    if verbose:
        print(f"  Jacobi constant C₀={traj['jacobi'][0]:.8f}  "
              f"drift={traj['jacobi_drift']:.2e}")

    plot_cr3bp(traj, system)
    return {"traj": traj, "cr3bp": cr3bp, "lagrange": lpts}


if __name__ == "__main__":
    import sys
    system = sys.argv[1] if len(sys.argv) > 1 else "earth_moon"
    result = run(system, n_orbits=2.0)
    print(f"Jacobi drift: {result['traj']['jacobi_drift']:.2e}  "
          f"(should be < 1e-8 for good integration)")
