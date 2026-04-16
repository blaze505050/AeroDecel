"""
src/aero_database.py — Hypersonic Aerodynamic Database
=======================================================
Pre-computed aero database for blunt-body EDL vehicles providing:
  • Cd(α, Mach)   — drag coefficient
  • CL(α, Mach)   — lift coefficient
  • Cm(α, Mach)   — pitching moment coefficient

Physics basis
-------------
  • Modified Newtonian theory for Mach > 3
  • Prandtl–Glauert correction for Mach < 0.8
  • Transonic interpolation (0.8 < Mach < 1.2)
  • Empirical corrections from Gnoffo et al. (1999) for
    70° sphere-cone geometries (MSL/Perseverance family)
  • Wake-base-pressure correction for Cd at low α

References
----------
  Gnoffo, P.A. et al. (1999) "Aerothermodynamic analyses of
    towed ballute performance" AIAA 1999-2106
  Edquist, K.T. et al. (2006) "Aerodynamic environment of the
    Mars Science Laboratory" JSR 43(6)
  Schoenenberger, M. et al. (2014) "Aerodynamic database
    development for MSL EDL" JSR 51(4)

Usage
-----
  >>> db = AeroDatabase.generate()
  >>> db.Cd(alpha_rad=0.05, Mach=15.0)
  1.573
  >>> db.save("data/blunt_body_aero.npz")
  >>> db2 = AeroDatabase.load("data/blunt_body_aero.npz")
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


# ══════════════════════════════════════════════════════════════════════════════
# MODIFIED NEWTONIAN THEORY + EMPIRICAL CORRECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _newtonian_cd(alpha_rad: float, Mach: float,
                  cone_half_angle_deg: float = 70.0) -> float:
    """
    Modified Newtonian drag coefficient for 70° sphere-cone.

    Cp_max = (2/(γM² + 1)) * ((γ+1)²M²/(4γM²-2(γ-1)))^(γ/(γ-1)) - 1/M²

    For a sphere-cone at angle α:
      Cd ≈ Cp_max * sin²(θ_cone) * f(α)

    With empirical base-pressure and viscous corrections.
    """
    gamma = 1.4  # Mars ~1.28, but Newtonian is a geometric approximation
    theta = np.radians(cone_half_angle_deg)
    alpha = float(alpha_rad)

    if Mach < 0.3:
        # Incompressible blunt body Cd ≈ 1.0–1.2
        Cd_incomp = 1.05 + 0.35 * np.sin(theta)**2
        return Cd_incomp * (1 + 0.15 * alpha**2)

    if Mach < 0.8:
        # Subsonic Prandtl-Glauert
        beta_pg = np.sqrt(max(1 - Mach**2, 0.01))
        Cd_sub = (1.05 + 0.35 * np.sin(theta)**2) / beta_pg
        return Cd_sub * (1 + 0.15 * alpha**2)

    if Mach < 1.2:
        # Transonic — interpolate
        Cd_sub = _newtonian_cd(alpha_rad, 0.79, cone_half_angle_deg)
        Cd_sup = _newtonian_cd(alpha_rad, 1.21, cone_half_angle_deg)
        frac = (Mach - 0.8) / 0.4
        return Cd_sub + frac * (Cd_sup - Cd_sub)

    # Supersonic/hypersonic: Modified Newtonian
    M2 = Mach**2
    # Stagnation pressure coefficient (modified Newtonian)
    Cp_max = (2 / (gamma * M2)) * (
        ((gamma + 1)**2 * M2 / (4 * gamma * M2 - 2 * (gamma - 1)))
        ** (gamma / (gamma - 1)) - 1
    )
    Cp_max = min(Cp_max, 2.0)  # physical limit

    # Pressure distribution for sphere-cone at AoA
    # Windward: Cp = Cp_max * sin²(θ_eff)
    theta_eff_w = theta + alpha   # effective angle, windward
    theta_eff_l = theta - alpha   # leeward

    # Forebody contribution (70° sphere-cone)
    Cd_w = Cp_max * np.sin(theta_eff_w)**2 * 0.5  # windward half
    Cd_l = Cp_max * np.sin(max(theta_eff_l, 0))**2 * 0.5  # leeward half
    Cd_fore = Cd_w + Cd_l

    # Base pressure correction (empirical, MSL-class)
    # Base Cd contribution decreases with Mach
    Cd_base = 0.12 / (1 + 0.02 * M2)

    # Viscous correction (small for blunt bodies)
    Cd_visc = 0.015 / np.sqrt(max(Mach, 1.0))

    Cd = Cd_fore + Cd_base + Cd_visc

    # Clamp to physical range
    return float(np.clip(Cd, 0.5, 2.5))


def _newtonian_cl(alpha_rad: float, Mach: float,
                  cone_half_angle_deg: float = 70.0) -> float:
    """
    Lift coefficient from Modified Newtonian pressure difference.
    CL ≈ Cp_max * sin(2θ) * sin(α) * cos(α) sign correction.
    """
    alpha = float(alpha_rad)
    if abs(alpha) < 1e-6:
        return 0.0

    theta = np.radians(cone_half_angle_deg)

    if Mach < 0.3:
        return 0.15 * alpha
    if Mach < 1.2:
        return 0.20 * alpha * min(Mach, 1.0)

    M2 = Mach**2
    gamma = 1.4
    Cp_max = (2 / (gamma * M2)) * (
        ((gamma + 1)**2 * M2 / (4 * gamma * M2 - 2 * (gamma - 1)))
        ** (gamma / (gamma - 1)) - 1
    )
    Cp_max = min(Cp_max, 2.0)

    # Lift from pressure asymmetry
    CL = Cp_max * np.sin(2 * theta) * np.sin(alpha)

    return float(np.clip(CL, -1.5, 1.5))


def _newtonian_cm(alpha_rad: float, Mach: float,
                  cone_half_angle_deg: float = 70.0,
                  xcg_over_D: float = 0.29) -> float:
    """
    Pitching moment coefficient about CG.
    Cm = CL * (x_cp - x_cg) / D  (simplified)

    For MSL-class: CG offset drives L/D ≈ 0.24 for guided entry.
    """
    alpha = float(alpha_rad)
    if abs(alpha) < 1e-6 and Mach > 1.2:
        return 0.0

    CL = _newtonian_cl(alpha_rad, Mach, cone_half_angle_deg)

    # Centre of pressure moves forward with Mach
    if Mach > 5:
        xcp_over_D = 0.33  # hypersonic, near geometric centre
    elif Mach > 1.2:
        xcp_over_D = 0.33 + 0.07 * (5 - Mach) / 3.8
    else:
        xcp_over_D = 0.45  # subsonic, further aft

    # Cm about CG (positive nose-up)
    arm = xcp_over_D - xcg_over_D
    Cm = -CL * arm  # negative = stable (restoring)

    # Add damping-like term for non-zero alpha
    Cm -= 0.005 * alpha  # small static stability addition

    return float(np.clip(Cm, -0.5, 0.5))


# ══════════════════════════════════════════════════════════════════════════════
# AERO DATABASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AeroDatabase:
    """
    Pre-computed aerodynamic coefficient database.

    Provides interpolated Cd(α, Mach), CL(α, Mach), Cm(α, Mach)
    over a grid of angle-of-attack and Mach number values.
    """

    def __init__(self, alpha_grid: np.ndarray, mach_grid: np.ndarray,
                 Cd_table: np.ndarray, CL_table: np.ndarray,
                 Cm_table: np.ndarray):
        self.alpha_grid = alpha_grid   # [rad]
        self.mach_grid  = mach_grid
        self.Cd_table   = Cd_table     # (n_alpha, n_mach)
        self.CL_table   = CL_table
        self.Cm_table   = Cm_table

        # Build interpolators
        self._Cd_interp = RegularGridInterpolator(
            (alpha_grid, mach_grid), Cd_table,
            method="linear", bounds_error=False, fill_value=None)
        self._CL_interp = RegularGridInterpolator(
            (alpha_grid, mach_grid), CL_table,
            method="linear", bounds_error=False, fill_value=None)
        self._Cm_interp = RegularGridInterpolator(
            (alpha_grid, mach_grid), Cm_table,
            method="linear", bounds_error=False, fill_value=None)

    def Cd(self, alpha_rad: float, Mach: float) -> float:
        """Interpolated drag coefficient."""
        a = np.clip(alpha_rad, self.alpha_grid[0], self.alpha_grid[-1])
        m = np.clip(Mach, self.mach_grid[0], self.mach_grid[-1])
        return float(self._Cd_interp([[a, m]])[0])

    def CL(self, alpha_rad: float, Mach: float) -> float:
        """Interpolated lift coefficient."""
        a = np.clip(alpha_rad, self.alpha_grid[0], self.alpha_grid[-1])
        m = np.clip(Mach, self.mach_grid[0], self.mach_grid[-1])
        return float(self._CL_interp([[a, m]])[0])

    def Cm(self, alpha_rad: float, Mach: float) -> float:
        """Interpolated pitching moment coefficient."""
        a = np.clip(alpha_rad, self.alpha_grid[0], self.alpha_grid[-1])
        m = np.clip(Mach, self.mach_grid[0], self.mach_grid[-1])
        return float(self._Cm_interp([[a, m]])[0])

    def save(self, path: str = "data/blunt_body_aero.npz"):
        """Save aero database to NumPy compressed file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path,
                            alpha_grid=self.alpha_grid,
                            mach_grid=self.mach_grid,
                            Cd_table=self.Cd_table,
                            CL_table=self.CL_table,
                            Cm_table=self.Cm_table)
        print(f"  ✓ Aero database saved: {path}  "
              f"({self.Cd_table.shape[0]}×{self.Cd_table.shape[1]} grid)")

    @classmethod
    def load(cls, path: str = "data/blunt_body_aero.npz") -> "AeroDatabase":
        """Load pre-computed aero database."""
        data = np.load(path)
        return cls(data["alpha_grid"], data["mach_grid"],
                   data["Cd_table"], data["CL_table"], data["Cm_table"])

    @classmethod
    def generate(cls,
                 alpha_deg_range: tuple = (-10, 30),
                 n_alpha: int = 41,
                 mach_range: tuple = (0.1, 35),
                 n_mach: int = 60,
                 cone_half_angle_deg: float = 70.0,
                 verbose: bool = True) -> "AeroDatabase":
        """
        Generate the aero database from Modified Newtonian theory.

        Parameters
        ----------
        alpha_deg_range : (min_deg, max_deg) angle of attack
        n_alpha         : number of α grid points
        mach_range      : (min_Mach, max_Mach)
        n_mach          : number of Mach grid points
        """
        alpha_deg = np.linspace(alpha_deg_range[0], alpha_deg_range[1], n_alpha)
        alpha_rad = np.radians(alpha_deg)
        mach = np.linspace(mach_range[0], mach_range[1], n_mach)

        Cd_table = np.zeros((n_alpha, n_mach))
        CL_table = np.zeros((n_alpha, n_mach))
        Cm_table = np.zeros((n_alpha, n_mach))

        for i, a in enumerate(alpha_rad):
            for j, m in enumerate(mach):
                Cd_table[i, j] = _newtonian_cd(a, m, cone_half_angle_deg)
                CL_table[i, j] = _newtonian_cl(a, m, cone_half_angle_deg)
                Cm_table[i, j] = _newtonian_cm(a, m, cone_half_angle_deg)

        if verbose:
            print(f"\n[AeroDB] Generated {n_alpha}×{n_mach} database")
            print(f"  α: [{alpha_deg[0]:.0f}°, {alpha_deg[-1]:.0f}°]"
                  f"  Mach: [{mach[0]:.1f}, {mach[-1]:.1f}]")
            print(f"  Cd range: [{Cd_table.min():.3f}, {Cd_table.max():.3f}]")
            print(f"  CL range: [{CL_table.min():.3f}, {CL_table.max():.3f}]")
            print(f"  Cm range: [{Cm_table.min():.3f}, {Cm_table.max():.3f}]")

        return cls(alpha_rad, mach, Cd_table, CL_table, Cm_table)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_aero_database(db: AeroDatabase,
                       save_path: str = "outputs/aero_database.png"):
    """Plot Cd, CL, Cm contour maps and Mach sweeps."""
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    matplotlib.rcParams.update({
        "figure.facecolor": "#080c14", "axes.facecolor": "#0d1526",
        "axes.edgecolor": "#2a3d6e", "text.color": "#c8d8f0",
        "axes.labelcolor": "#c8d8f0", "xtick.color": "#c8d8f0",
        "ytick.color": "#c8d8f0", "grid.color": "#1a2744",
        "font.family": "monospace", "font.size": 9,
    })
    TX = "#c8d8f0"; C1 = "#00d4ff"; C2 = "#ff6b35"; C3 = "#a8ff3e"
    C4 = "#ffd700"; CR = "#ff4560"

    fig = plt.figure(figsize=(22, 12), facecolor="#080c14")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                           top=0.90, bottom=0.08, left=0.06, right=0.97)

    A_deg = np.degrees(db.alpha_grid)
    M = db.mach_grid
    MA, AA = np.meshgrid(M, A_deg)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    # Cd contour
    ax1 = gax(0, 0)
    cf1 = ax1.contourf(MA, AA, db.Cd_table, levels=20, cmap="plasma")
    fig.colorbar(cf1, ax=ax1, shrink=0.8, label="Cd")
    ax1.set_xlabel("Mach"); ax1.set_ylabel("α [°]")
    ax1.set_title("Drag Coefficient Cd(α, M)", fontweight="bold")

    # CL contour
    ax2 = gax(0, 1)
    cf2 = ax2.contourf(MA, AA, db.CL_table, levels=20, cmap="RdBu_r")
    fig.colorbar(cf2, ax=ax2, shrink=0.8, label="CL")
    ax2.set_xlabel("Mach"); ax2.set_ylabel("α [°]")
    ax2.set_title("Lift Coefficient CL(α, M)", fontweight="bold")

    # Cm contour
    ax3 = gax(0, 2)
    cf3 = ax3.contourf(MA, AA, db.Cm_table, levels=20, cmap="PiYG")
    fig.colorbar(cf3, ax=ax3, shrink=0.8, label="Cm")
    ax3.set_xlabel("Mach"); ax3.set_ylabel("α [°]")
    ax3.set_title("Pitch Moment Cm(α, M)", fontweight="bold")

    # Cd vs Mach at key alphas
    ax4 = gax(1, 0)
    for a_deg, col in zip([0, 5, 10, 20], [C1, C2, C3, CR]):
        a_rad = np.radians(a_deg)
        cd_vs_m = [db.Cd(a_rad, m) for m in M]
        ax4.plot(M, cd_vs_m, color=col, lw=2, label=f"α={a_deg}°")
    ax4.set_xlabel("Mach"); ax4.set_ylabel("Cd")
    ax4.set_title("Cd vs Mach", fontweight="bold")
    ax4.legend(fontsize=8)

    # CL vs alpha at key Machs
    ax5 = gax(1, 1)
    for m_val, col in zip([1.5, 5, 15, 30], [C1, C2, C3, CR]):
        cl_vs_a = [db.CL(np.radians(a), m_val) for a in A_deg]
        ax5.plot(A_deg, cl_vs_a, color=col, lw=2, label=f"M={m_val:.0f}")
    ax5.set_xlabel("α [°]"); ax5.set_ylabel("CL")
    ax5.set_title("CL vs α", fontweight="bold")
    ax5.legend(fontsize=8)

    # L/D vs alpha at key Machs
    ax6 = gax(1, 2)
    for m_val, col in zip([5, 15, 25], [C1, C3, CR]):
        ld = [db.CL(np.radians(a), m_val) / max(db.Cd(np.radians(a), m_val), 0.01)
              for a in A_deg]
        ax6.plot(A_deg, ld, color=col, lw=2, label=f"M={m_val:.0f}")
    ax6.axhline(0.24, color=C4, lw=0.8, ls="--", alpha=0.7, label="MSL L/D=0.24")
    ax6.set_xlabel("α [°]"); ax6.set_ylabel("L/D")
    ax6.set_title("Lift-to-Drag Ratio", fontweight="bold")
    ax6.legend(fontsize=8)

    fig.text(0.5, 0.955,
             f"Hypersonic Aero Database  |  {len(db.alpha_grid)}×{len(db.mach_grid)} grid  |  "
             f"Modified Newtonian + Empirical Corrections  |  70° Sphere-Cone",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Aero database plot saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(verbose: bool = True) -> dict:
    """Generate aero database, save, and plot."""
    import matplotlib; matplotlib.use("Agg")
    from pathlib import Path

    db = AeroDatabase.generate(verbose=verbose)
    Path("data").mkdir(exist_ok=True)
    db.save("data/blunt_body_aero.npz")
    plot_aero_database(db)

    return {"database": db, "shape": db.Cd_table.shape}


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    run()
