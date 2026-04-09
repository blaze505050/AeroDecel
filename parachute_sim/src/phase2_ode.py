"""
phase2_ode.py — AeroDecel Physics ODE Simulator v5.0
=====================================================
Solves the parachute inflation dynamics ODE with full aerodynamic corrections:

    m_eff * dv/dt = m*g - 0.5 * ρ(h) * v² * Cd_eff(v,h) * A(t) - F_buoy
    dh/dt         = -v   (descending → positive v means downward)

where:
  - m_eff = m + C_a·ρ·V_canopy  (added/virtual mass for accelerating body)
  - Cd_eff incorporates Reynolds, Mach, and porosity corrections
  - F_buoy = ρ·g·V_canopy (Archimedes buoyancy, significant at altitude)

Integrates using scipy's RK45 (or configurable method) with:
  - ISA standard atmosphere for ρ(h) and a(h)
  - Interpolated A(t) from Phase 1 CSV
  - Full aerodynamic coefficient pipeline
  - Jerk, drag force, and dynamic pressure as output channels

Reference:
  - Knacke, T.W., "Parachute Recovery Systems Design Manual", 1992
  - MIL-HDBK-1791, "Designing for Internal Aerial Delivery"
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import (
    density, speed_of_sound, dynamic_viscosity,
    mach_number as isa_mach, vectorized_density, vectorized_speed_of_sound
)


# ─── Inflation Models ────────────────────────────────────────────────────────
class InflationModel:
    """
    Wraps A(t) as a smooth callable.
    Supports: CSV interpolation, generalized logistic fit, polynomial fit.
    """

    def __init__(self, df: pd.DataFrame, mode: str = "csv_interpolated"):
        self.mode = mode
        self.t_data = df["time_s"].values.astype(float)
        self.A_data = df["area_m2"].values.astype(float)
        self.A_max  = self.A_data.max()
        self.t_max  = self.t_data.max()

        if mode == "csv_interpolated":
            self._fn = interp1d(
                self.t_data, self.A_data,
                kind="cubic", bounds_error=False,
                fill_value=(self.A_data[0], self.A_max)
            )
        elif mode == "generalized_logistic":
            self._fn = self._fit_generalized_logistic()
        elif mode == "polynomial":
            self._fn = self._fit_polynomial()
        else:
            raise ValueError(f"Unknown inflation model: {mode}")

    def _fit_generalized_logistic(self):
        from scipy.optimize import curve_fit

        def logistic(t, k, t0, n, A_inf):
            return A_inf / (1 + np.exp(-k * (t - t0))) ** (1 / n)

        p0 = [2.0, self.t_max * 0.4, 1.5, self.A_max]
        bounds = ([0.1, 0, 0.1, 0.1 * self.A_max],
                  [20, self.t_max, 10, 1.5 * self.A_max])
        try:
            popt, _ = curve_fit(logistic, self.t_data, self.A_data,
                                 p0=p0, bounds=bounds, maxfev=5000)
            print(f"  Logistic fit: k={popt[0]:.3f}, t0={popt[1]:.3f}s, "
                  f"n={popt[2]:.3f}, A_inf={popt[3]:.2f}m²")
            return lambda t: np.clip(logistic(t, *popt), 0, 1.5 * self.A_max)
        except Exception as e:
            print(f"  Logistic fit failed ({e}), falling back to spline.")
            return interp1d(self.t_data, self.A_data, kind="cubic",
                            bounds_error=False, fill_value=(0, self.A_max))

    def _fit_polynomial(self):
        deg = min(7, len(self.t_data) // 10)
        coeffs = np.polyfit(self.t_data, self.A_data, deg)
        poly = np.poly1d(coeffs)
        return lambda t: np.clip(poly(t), 0, 1.5 * self.A_max)

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.clip(self._fn(t), 0, None)

    def derivative(self, t: float, dt: float = 1e-4) -> float:
        """Numerical derivative dA/dt via central difference."""
        return (self(t + dt) - self(t - dt)) / (2 * dt)


# ─── Aerodynamic Correction Pipeline (AeroDecel v5.0) ────────────────────────
class AerodynamicCorrections:
    """
    Computes Cd_eff by applying sequential multiplicative corrections:
      1. Reynolds-number drag correction (Knacke 1992)
      2. Mach-number compressibility (Prandtl-Glauert)
      3. Fabric porosity reduction (Pflanz 1952)
    """

    def __init__(
        self,
        Cd_base: float,
        canopy_diameter: float = None,
        porosity_k: float = None,
        apply_re: bool = None,
        apply_mach: bool = None,
        apply_porosity: bool = None,
    ):
        self.Cd_base = Cd_base
        self.D = canopy_diameter or cfg.CANOPY_DIAMETER_M
        self.k_p = porosity_k or cfg.POROSITY_COEFF
        self.apply_re = apply_re if apply_re is not None else cfg.RE_CORRECTION_ENABLED
        self.apply_mach = apply_mach if apply_mach is not None else cfg.MACH_CORRECTION_ENABLED
        self.apply_porosity = apply_porosity if apply_porosity is not None else cfg.POROSITY_ENABLED

    def corrected_Cd(self, v: float, h: float) -> tuple[float, dict]:
        """Return (Cd_corrected, diagnostics_dict)."""
        Cd = self.Cd_base
        rho = density(h)
        mu = dynamic_viscosity(h)
        a = speed_of_sound(h)

        M = v / max(a, 1.0)
        Re = rho * v * self.D / max(mu, 1e-10)

        corr_mach = 1.0
        corr_re = 1.0
        corr_por = 1.0

        # 1. Mach correction (Prandtl-Glauert, valid M < 0.8)
        if self.apply_mach and M > 0.05:
            M_clamp = min(M, 0.79)
            corr_mach = 1.0 / max(np.sqrt(1.0 - M_clamp**2), 0.01)
            Cd *= corr_mach

        # 2. Reynolds correction (Knacke 1992: drag crisis ~Re=3×10⁵)
        if self.apply_re and Re > 0:
            log_Re = np.log10(max(Re, 1e3))
            if log_Re < 4.5:
                corr_re = 1.05
            elif log_Re < 5.0:
                corr_re = 1.0 + 0.05 * (5.0 - log_Re) / 0.5
            elif log_Re < 5.5:
                corr_re = 1.0 - 0.08 * (log_Re - 5.0) / 0.5
            elif log_Re < 6.0:
                corr_re = 0.92 + 0.06 * (log_Re - 5.5) / 0.5
            else:
                corr_re = 0.98
            Cd *= corr_re

        # 3. Porosity correction (Pflanz 1952)
        if self.apply_porosity and self.k_p > 0:
            corr_por = max(0.05, 1.0 - self.k_p * max(v, 0.0))
            Cd *= corr_por

        diag = {
            "Cd_eff": Cd, "Cd_base": self.Cd_base,
            "Mach": M, "Re": Re,
            "corr_mach": corr_mach, "corr_re": corr_re, "corr_porosity": corr_por,
        }
        return Cd, diag


# ─── ODE System ──────────────────────────────────────────────────────────────
class ParachuteODE:
    """
    Encapsulates the 2-state parachute descent ODE with full physics:
        state = [v, h]   (velocity [m/s], altitude [m])

    AeroDecel v5.0 enhancements:
      - Added mass (virtual mass for body accelerating in fluid)
      - Reynolds/Mach/porosity Cd corrections
      - Buoyancy correction
      - ISA speed of sound for true Mach number
    """

    def __init__(
        self,
        At_model: InflationModel,
        mass: float = None,
        Cd: float = None,
        use_advanced_physics: bool = True,
    ):
        self.A    = At_model
        self.mass = mass or cfg.PARACHUTE_MASS
        self.Cd   = Cd   or cfg.CD_INITIAL
        self.use_advanced = use_advanced_physics

        # Initialize aerodynamic correction pipeline
        if self.use_advanced:
            self.aero = AerodynamicCorrections(self.Cd)
        else:
            self.aero = None

        # Added mass parameters
        self.added_mass_enabled = self.use_advanced and cfg.ADDED_MASS_ENABLED
        self.C_a = cfg.ADDED_MASS_COEFF
        self.V_canopy = cfg.CANOPY_VOLUME_M3

        # Buoyancy
        self.buoyancy_enabled = self.use_advanced and cfg.BUOYANCY_ENABLED

        # Tracking diagnostics for post-analysis
        self._diag = []

    def _effective_mass(self, rho: float) -> float:
        """Compute effective mass including added (virtual) mass.

        m_eff = m + C_a · ρ · V_canopy

        The added mass accounts for the fluid that must be accelerated
        along with the body. For a hemisphere, C_a ≈ 0.5.
        """
        if self.added_mass_enabled:
            return self.mass + self.C_a * rho * self.V_canopy
        return self.mass

    def _buoyancy_force(self, rho: float) -> float:
        """Archimedes buoyancy force [N] on inflated canopy."""
        if self.buoyancy_enabled:
            return rho * cfg.GRAVITY * self.V_canopy
        return 0.0

    def rhs(self, t: float, state: np.ndarray) -> list:
        """Right-hand side of the ODE system."""
        v, h = state
        v = max(v, 0.0)     # velocity cannot be negative (no bounce model)

        rho  = density(h)
        A_t  = self.A(t)

        # Compute Cd with corrections
        if self.aero is not None:
            Cd_eff, diag = self.aero.corrected_Cd(v, h)
        else:
            Cd_eff = self.Cd
            diag = {"Cd_eff": Cd_eff, "Mach": v / 342.0, "Re": 0,
                    "corr_mach": 1.0, "corr_re": 1.0, "corr_porosity": 1.0}

        # Forces
        drag = 0.5 * rho * v**2 * Cd_eff * A_t
        F_buoy = self._buoyancy_force(rho)
        m_eff = self._effective_mass(rho)

        # Net acceleration: gravity - (drag - buoyancy) / m_eff
        dv_dt = cfg.GRAVITY - (drag - F_buoy) / m_eff
        dh_dt = -v          # altitude decreases as vehicle descends

        return [dv_dt, dh_dt]

    def solve(
        self,
        t_span: tuple = None,
        t_eval: np.ndarray = None,
        method: str = None,
    ) -> dict:
        """
        Integrate the ODE over t_span.
        Returns a rich dict of physical quantities.
        """
        t0 = 0.0
        # Estimate descent time using terminal velocity: v_t = sqrt(2mg/(ρ·Cd·A))
        # Then t_descent ≈ h / v_t. Add 50% margin.
        rho_sl = density(0.0)
        A_ref = max(self.A.A_max, 1.0)
        v_term_est = np.sqrt(2 * self.mass * cfg.GRAVITY / (rho_sl * self.Cd * A_ref + 1e-6))
        t_descent_est = 1.5 * cfg.INITIAL_ALT / max(v_term_est, 1.0)
        t_from_data = self.A.t_max + 5.0
        t_end = t_span[1] if t_span else max(t_from_data, t_descent_est, 60.0)
        if t_eval is None:
            t_eval = np.linspace(t0, t_end, max(2000, int(t_end * 100)))

        y0 = [cfg.INITIAL_VEL, cfg.INITIAL_ALT]

        # Terminal condition: ground impact
        def ground_hit(t, y): return y[1]
        ground_hit.terminal  = True
        ground_hit.direction = -1

        sol = solve_ivp(
            self.rhs,
            t_span=(t0, t_end),
            y0=y0,
            method=method or cfg.ODE_METHOD,
            t_eval=t_eval,
            events=ground_hit,
            rtol=cfg.ODE_RTOL,
            atol=cfg.ODE_ATOL,
            dense_output=True,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t  = sol.t
        v  = sol.y[0]
        h  = sol.y[1]
        At = self.A(t)

        rho_arr  = np.array([density(hi) for hi in h])
        a_arr_sound = np.array([speed_of_sound(hi) for hi in h])

        # Compute corrected Cd at each timestep for output
        if self.aero is not None:
            cd_arr = np.array([self.aero.corrected_Cd(vi, hi)[0]
                               for vi, hi in zip(v, h)])
        else:
            cd_arr = np.full_like(v, self.Cd)

        drag_arr = 0.5 * rho_arr * v**2 * cd_arr * At
        accel_arr = np.gradient(v, t)                     # acceleration m/s²
        q_arr    = 0.5 * rho_arr * v**2                   # dynamic pressure Pa
        Ma_arr   = v / a_arr_sound                        # True Mach number from ISA

        # Effective mass at each point
        m_eff_arr = np.array([self._effective_mass(rho) for rho in rho_arr])

        # Energy bookkeeping
        KE  = 0.5 * self.mass * v**2
        PE  = self.mass * cfg.GRAVITY * h

        # Snatch load estimation (peak drag force during inflation)
        dA_dt = np.gradient(At, t)
        snatch_force = 0.5 * rho_arr * v**2 * cd_arr * dA_dt

        # Jerk (rate of acceleration change) [m/s³]
        jerk_arr = np.gradient(accel_arr, t)

        return {
            "time_s"        : t,
            "velocity_ms"   : v,
            "altitude_m"    : h,
            "area_m2"       : At,
            "acceleration"  : accel_arr,
            "jerk_ms3"      : jerk_arr,
            "drag_force_N"  : drag_arr,
            "dynamic_press" : q_arr,
            "density_kgm3"  : rho_arr,
            "mach"          : Ma_arr,
            "Cd_effective"  : cd_arr,
            "m_effective_kg": m_eff_arr,
            "KE_J"          : KE,
            "PE_J"          : PE,
            "snatch_force_N": snatch_force,
        }

    def to_dataframe(self, results: dict) -> pd.DataFrame:
        return pd.DataFrame({k: v for k, v in results.items()
                              if isinstance(v, np.ndarray)})


# ─── Entry Point ─────────────────────────────────────────────────────────────
def run(at_df: pd.DataFrame = None, use_advanced: bool = True) -> pd.DataFrame:
    if at_df is None:
        if not cfg.AT_CSV.exists():
            raise FileNotFoundError(
                "A(t) CSV not found. Run Phase 1 first, or pass at_df directly."
            )
        at_df = pd.read_csv(cfg.AT_CSV)

    # Determine if any advanced physics is enabled
    adv = use_advanced and any([
        cfg.ADDED_MASS_ENABLED,
        cfg.RE_CORRECTION_ENABLED,
        cfg.MACH_CORRECTION_ENABLED,
        cfg.POROSITY_ENABLED,
        cfg.BUOYANCY_ENABLED,
    ])

    physics_label = "advanced (Re+Mach+porosity+added-mass+buoyancy)" if adv else "classical"
    print(f"\n[Phase 2] AeroDecel ODE Simulator — {physics_label}")
    print(f"  Mass: {cfg.PARACHUTE_MASS} kg | Cd₀: {cfg.CD_INITIAL} | "
          f"Alt₀: {cfg.INITIAL_ALT} m | V₀: {cfg.INITIAL_VEL} m/s")
    print(f"  Inflation model: {cfg.INFLATION_MODEL}")
    if adv:
        flags = []
        if cfg.ADDED_MASS_ENABLED:      flags.append(f"added-mass(C_a={cfg.ADDED_MASS_COEFF})")
        if cfg.RE_CORRECTION_ENABLED:   flags.append(f"Re-correction(D={cfg.CANOPY_DIAMETER_M}m)")
        if cfg.MACH_CORRECTION_ENABLED: flags.append("Mach(P-G)")
        if cfg.POROSITY_ENABLED:        flags.append(f"porosity(k={cfg.POROSITY_COEFF})")
        if cfg.BUOYANCY_ENABLED:        flags.append(f"buoyancy(V={cfg.CANOPY_VOLUME_M3}m³)")
        print(f"  Advanced: {' · '.join(flags)}")

    At_model = InflationModel(at_df, mode=cfg.INFLATION_MODEL)
    ode      = ParachuteODE(At_model, use_advanced_physics=adv)

    print(f"  Integrating [{cfg.ODE_METHOD}]...")
    results = ode.solve()

    t   = results["time_s"]
    v   = results["velocity_ms"]
    h   = results["altitude_m"]
    drag = results["drag_force_N"]
    cd   = results["Cd_effective"]

    terminal_v = v[-1] if h[-1] < 1.0 else None
    print(f"  ✓ Solved {len(t)} points over {t[-1]:.2f}s")
    if terminal_v is not None:
        print(f"  Terminal velocity at ground: {terminal_v:.2f} m/s ({terminal_v*3.6:.1f} km/h)")
    print(f"  Peak drag force: {drag.max():.1f} N | Peak deceleration: {abs(results['acceleration'].min()):.2f} m/s²")
    print(f"  Cd range: [{cd.min():.4f}, {cd.max():.4f}] (mean={cd.mean():.4f})")
    print(f"  Mach range: [{results['mach'].min():.5f}, {results['mach'].max():.5f}]")

    df = ode.to_dataframe(results)
    df.to_csv(cfg.ODE_CSV, index=False)
    print(f"  ✓ Saved: {cfg.ODE_CSV}")

    return df, At_model


if __name__ == "__main__":
    df, _ = run()
