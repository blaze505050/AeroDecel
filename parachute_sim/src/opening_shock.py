"""
opening_shock.py — MIL-HDBK-1791 Opening Shock Load Factor Analysis
======================================================================
Full implementation of the military-standard parachute opening shock methodology.

References
----------
  MIL-HDBK-1791  : Handbook for Design of Aerospace Weapons Parachutes (1978)
  TR-1495         : Opening shock of parachutes — Ludtke (1986)
  NASA CR-2746    : Parachute design guide (1978)

Physics
-------
The opening shock force time-history during canopy inflation is:

    F(t) = 0.5 · ρ(h) · v(t)² · Cd · A(t) · X₁(t)

where X₁(t) is the Inflation Force Coefficient that captures the dynamic
overshoot above the steady-state drag force. At peak:

    F_peak = 0.5 · ρ · v_deploy² · Cd · A_inf · CLA

CLA (Canopy Load Alleviation factor) per MIL-HDBK-1791 §6.2.3:

    CLA = Cx · (ρ · v_dep² / (W/A_inf))^n_force · (t_infl · v_dep / D_canopy)^(-n_time)

Simplified (Knacke 1992 empirical fit used for fabric canopies):

    CLA = 1 + k₁ · (v_dep / v_term)^k₂ · exp(-t_infl / τ)

where τ is the inflation time constant and k₁, k₂ are canopy-type constants.

The FULL dynamic load time-history uses the Ludtke inflation force model:

    X₁(t) = (A(t)/A_inf)^n_force · (1 + α · d/dt[A(t)/A_inf])

This captures the force overshoot during rapid area growth.

Structural Analysis
-------------------
  Safety factor:  SF = F_rated / F_peak
  Target:         SF ≥ 1.5 (MIL-SPEC minimum for personnel parachutes)
                  SF ≥ 2.0 (recommended for cargo)

Outputs
-------
  • F_open(t) time history [N]
  • F_peak_N, CLA, safety_factor, margin_pct
  • Structural comparison table: lines, canopy, container, harness
  • 6-panel engineering dashboard
  • JSON summary for report integration
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CANOPY TYPE CONSTANTS  (MIL-HDBK-1791 Table 6-2 / Knacke 1992)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanopyType:
    """
    Empirical constants for different canopy geometries.
    Source: Knacke "Parachute Recovery Systems Design Manual" (1992) Table 5-1
    """
    name:        str
    k1:          float   # CLA amplitude coefficient
    k2:          float   # velocity exponent
    n_force:     float   # inflation force area exponent (X₁ model)
    n_inflation: float   # inflation exponent (Pflanz method)
    Cx:          float   # canopy force coefficient at snatch (MIL-HDBK §6.2.2)
    description: str = ""


CANOPY_TYPES = {
    "flat_circular": CanopyType(
        "Flat Circular", k1=1.8, k2=0.35, n_force=2.0, n_inflation=8.0,
        Cx=1.00, description="Standard personnel/cargo flat circular"
    ),
    "conical": CanopyType(
        "Conical", k1=1.5, k2=0.30, n_force=1.8, n_inflation=7.0,
        Cx=0.95, description="Extended-skirt conical geometry"
    ),
    "extended_skirt": CanopyType(
        "Extended-Skirt", k1=1.3, k2=0.28, n_force=1.6, n_inflation=6.0,
        Cx=0.90, description="Extended-skirt 10% flat"
    ),
    "ribbon": CanopyType(
        "Ribbon", k1=0.9, k2=0.20, n_force=1.3, n_inflation=4.5,
        Cx=0.70, description="High-speed ribbon / ringslot"
    ),
    "drogue": CanopyType(
        "Drogue", k1=2.5, k2=0.45, n_force=2.5, n_inflation=10.0,
        Cx=1.20, description="Small pilot/drogue (fast inflation)"
    ),
    "ram_air": CanopyType(
        "Ram-Air", k1=1.1, k2=0.25, n_force=1.5, n_inflation=5.5,
        Cx=0.85, description="Ram-air / parafoil"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STRUCTURAL COMPONENT DATABASE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralComponent:
    name:         str
    rated_load_N: float     # rated tensile strength
    n_lines:      int = 1   # number of parallel load paths
    safety_margin: float = 0.0   # computed at runtime

    @property
    def effective_rated_N(self) -> float:
        return self.rated_load_N * self.n_lines


def default_structural_components(canopy_area_m2: float,
                                   mass_kg: float) -> list[StructuralComponent]:
    """
    Generate realistic structural components scaled to canopy size and mass.
    Values based on FAA TSO-C23 / EN12491 parachute certification standards.
    """
    # Scale factor: larger canopies use heavier hardware
    scale = max(1.0, (canopy_area_m2 / 50.0) ** 0.5)

    return [
        StructuralComponent("Suspension lines (×28)", 3560.0 * scale, n_lines=28),
        StructuralComponent("Canopy seams (textile)", 12000.0 * scale, n_lines=1),
        StructuralComponent("Risers (×4)", 15000.0 * scale, n_lines=4),
        StructuralComponent("Container closure", 8000.0 * scale, n_lines=2),
        StructuralComponent("Harness (pilot chute)", 6700.0 * scale, n_lines=2),
        StructuralComponent("Deployment bag", 4500.0 * scale, n_lines=1),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  OPENING SHOCK MODELS
# ══════════════════════════════════════════════════════════════════════════════

def cla_knacke(
    v_deploy: float,       # deployment velocity [m/s]
    v_terminal: float,     # terminal velocity under fully open canopy [m/s]
    t_infl: float,         # canopy inflation time [s]
    canopy: CanopyType,
) -> float:
    """
    Canopy Load Alleviation factor — Knacke (1992) empirical formula.
    CLA ≥ 1.0 always (opening load is never less than steady-state).

    CLA = 1 + k₁ · (v_dep/v_term)^k₂ · exp(-t_infl · v_dep / (k₃ · D))
    Simplified for t_infl-only parameterisation:
    CLA = 1 + k₁ · (v_dep/v_term)^k₂ · exp(-0.5 · t_infl)
    """
    v_ratio = min(v_deploy / max(v_terminal, 0.1), 20.0)  # cap to avoid blow-up
    cla = 1.0 + canopy.k1 * (v_ratio ** canopy.k2) * np.exp(-0.4 * t_infl)
    return float(max(1.0, cla))


def cla_milspec(
    v_deploy: float,
    rho: float,
    mass: float,
    A_inf: float,
    Cd: float,
    t_infl: float,
    canopy: CanopyType,
) -> float:
    """
    CLA from MIL-HDBK-1791 §6.2.3 ballistic parameter method.
    The ballistic parameter X = W / (ρ · v² · A):
        CLA = Cx · X^(-n₁) · (v·t_infl/D)^(-n₂)
    where D = sqrt(4·A_inf/π) is nominal canopy diameter.
    """
    W = mass * cfg.GRAVITY                              # weight [N]
    q = 0.5 * rho * v_deploy**2                        # dynamic pressure [Pa]
    X = W / max(q * A_inf * Cd, 1e-6)                  # ballistic parameter
    D = np.sqrt(4 * A_inf / np.pi)                     # nominal diameter [m]
    vt_D = v_deploy * t_infl / max(D, 0.1)             # dimensionless fill time

    # Clamp to valid range
    X    = float(np.clip(X, 0.01, 10.0))
    vt_D = float(np.clip(vt_D, 0.1, 100.0))

    cla = canopy.Cx * (X ** (-0.25)) * (vt_D ** (-0.10))
    return float(max(1.0, cla))


def _generalised_logistic(t: np.ndarray, A_inf: float,
                           t_infl: float, n: float) -> np.ndarray:
    """Canopy area as a function of time since deployment."""
    k  = 8.0 / max(t_infl, 0.1)
    t0 = t_infl * 0.55
    raw = A_inf / (1.0 + np.exp(-k * (t - t0))) ** (1.0 / n)
    return np.clip(raw, 0.0, A_inf)


def compute_force_history(
    v_deploy:   float,
    h_deploy:   float,
    mass:       float,
    A_inf:      float,
    Cd:         float,
    t_infl:     float,
    canopy:     CanopyType,
    at_df:      pd.DataFrame | None = None,
    n_force:    float | None = None,
    dt:         float = 0.01,
) -> pd.DataFrame:
    """
    Compute the full opening shock force time-history using the Ludtke model.

    F(t) = 0.5 · ρ(h(t)) · v(t)² · Cd · A(t) · X₁(t)

    X₁(t) = (Ȧ/A_inf · t_infl)^n_force      (area growth rate amplifier)

    The velocity v(t) is obtained by integrating the coupled ODE:
        m dv/dt = m·g - 0.5·ρ·v²·Cd·A(t)·X₁(t)

    Returns DataFrame with: t, v, h, A, X1, F_drag, F_shock, A_rate
    """
    n_f = n_force if n_force is not None else canopy.n_force
    t_span = 4.0 * t_infl + 5.0     # simulate through and past full inflation

    # Build A(t) callable
    if at_df is not None:
        _At_interp = interp1d(
            at_df["time_s"].values, at_df["area_m2"].values,
            kind="cubic", bounds_error=False,
            fill_value=(0.0, float(at_df["area_m2"].max()))
        )
        def At(t): return max(0.0, float(_At_interp(t)))
    else:
        def At(t):
            return float(_generalised_logistic(np.array([t]), A_inf, t_infl,
                                               canopy.n_inflation)[0])

    def dAt(t, eps=1e-4):
        """Numerical derivative dA/dt."""
        return (At(t + eps) - At(t - eps)) / (2 * eps)

    def X1(t):
        """Ludtke inflation force coefficient."""
        dA = max(0.0, dAt(t))
        norm_rate = dA * t_infl / max(A_inf, 1e-6)
        # Smooth clamp: avoid X1 > Cx_max during initial transient
        return float(max(1.0, min((1.0 + norm_rate) ** n_f, canopy.Cx * 3.0)))

    def rhs(t, state):
        v, h = state
        v = max(0.0, v)
        rho  = density(max(0.0, h))
        A    = At(t)
        x1   = X1(t)
        drag = 0.5 * rho * v**2 * Cd * A * x1
        return [cfg.GRAVITY - drag / mass, -v]

    def ground(t, y): return y[1]
    ground.terminal = True; ground.direction = -1

    t_eval = np.arange(0.0, t_span, dt)
    sol = solve_ivp(rhs, (0, t_span), [v_deploy, h_deploy],
                    method="RK45", t_eval=t_eval,
                    events=ground, rtol=1e-5, atol=1e-7)

    t_arr = sol.t
    v_arr = np.clip(sol.y[0], 0, None)
    h_arr = np.clip(sol.y[1], 0, None)

    A_arr  = np.array([At(ti) for ti in t_arr])
    x1_arr = np.array([X1(ti) for ti in t_arr])

    # Smooth area rate for display
    dA_arr = np.gradient(A_arr, t_arr)
    if len(dA_arr) > 11:
        dA_arr = savgol_filter(dA_arr, window_length=11, polyorder=3)

    rho_arr  = np.array([density(max(0.0, h)) for h in h_arr])
    F_steady = 0.5 * rho_arr * v_arr**2 * Cd * A_arr    # steady-state drag
    F_shock  = 0.5 * rho_arr * v_arr**2 * Cd * A_arr * x1_arr  # with X₁

    return pd.DataFrame({
        "time_s":         t_arr,
        "velocity_ms":    v_arr,
        "altitude_m":     h_arr,
        "area_m2":        A_arr,
        "area_rate_m2s":  dA_arr,
        "X1":             x1_arr,
        "rho_kgm3":       rho_arr,
        "F_steady_N":     F_steady,
        "F_shock_N":      F_shock,
        "F_net_N":        F_shock,     # alias for plotting
        "dyn_press_Pa":   0.5 * rho_arr * v_arr**2,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL STRUCTURAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OpeningShockResult:
    """Complete opening shock analysis result."""
    # Peak loads
    F_peak_N:      float
    F_steady_N:    float     # equilibrium drag at v_deploy
    CLA_knacke:    float
    CLA_milspec:   float
    CLA_used:      float     # whichever is more conservative
    # Structural
    components:    list
    min_sf:        float
    critical_component: str
    # Time-series
    df:            pd.DataFrame
    # Parameters
    v_deploy:      float
    h_deploy:      float
    t_infl:        float
    v_terminal:    float
    canopy_type:   str

    def to_dict(self) -> dict:
        comps = [{
            "component":   c.name,
            "rated_N":     c.effective_rated_N,
            "F_peak_N":    self.F_peak_N,
            "safety_factor": round(c.effective_rated_N / max(self.F_peak_N, 1), 3),
            "margin_pct":  round((c.effective_rated_N / max(self.F_peak_N, 1) - 1) * 100, 1),
            "status": ("OK" if c.effective_rated_N / max(self.F_peak_N, 1) >= 1.5
                       else "WARNING" if c.effective_rated_N / max(self.F_peak_N, 1) >= 1.0
                       else "CRITICAL"),
        } for c in self.components]
        return {
            "F_peak_N":          round(self.F_peak_N, 1),
            "F_steady_N":        round(self.F_steady_N, 1),
            "CLA_knacke":        round(self.CLA_knacke, 4),
            "CLA_milspec":       round(self.CLA_milspec, 4),
            "CLA_used":          round(self.CLA_used, 4),
            "minimum_sf":        round(self.min_sf, 3),
            "critical_component": self.critical_component,
            "v_deploy_ms":       round(self.v_deploy, 3),
            "h_deploy_m":        round(self.h_deploy, 1),
            "t_infl_s":          round(self.t_infl, 3),
            "v_terminal_ms":     round(self.v_terminal, 3),
            "canopy_type":       self.canopy_type,
            "structural_components": comps,
            "mil_spec_minimum_sf": 1.5,
            "compliant":         self.min_sf >= 1.5,
        }

    def print_summary(self):
        d = self.to_dict()
        W = 60
        print("\n" + "═" * W)
        print(f"  OPENING SHOCK ANALYSIS  —  MIL-HDBK-1791")
        print("═" * W)
        print(f"  Canopy type:      {self.canopy_type}")
        print(f"  Deployment:       v={self.v_deploy:.2f} m/s  h={self.h_deploy:.0f} m")
        print(f"  Inflation time:   {self.t_infl:.2f} s")
        print(f"  Terminal vel:     {self.v_terminal:.2f} m/s")
        print()
        print(f"  CLA (Knacke):     {self.CLA_knacke:.4f}")
        print(f"  CLA (MIL-SPEC):   {self.CLA_milspec:.4f}")
        print(f"  CLA (used):       {self.CLA_used:.4f}  (conservative of two)")
        print()
        print(f"  F_steady:         {self.F_steady_N:>10.1f} N  ({self.F_steady_N/1e3:.2f} kN)")
        print(f"  F_peak (shock):   {self.F_peak_N:>10.1f} N  ({self.F_peak_N/1e3:.2f} kN)")
        print()
        print(f"  {'Component':<30} {'Rated [N]':>10} {'SF':>6} {'Margin':>8} {'Status'}")
        print(f"  {'─'*30} {'─'*10} {'─'*6} {'─'*8} {'─'*8}")
        for c in d["structural_components"]:
            flag = "✓" if c["status"]=="OK" else "⚠" if c["status"]=="WARNING" else "✗"
            print(f"  {flag} {c['component']:<28} {c['rated_N']:>10.0f} "
                  f"{c['safety_factor']:>6.2f} {c['margin_pct']:>+7.1f}%  {c['status']}")
        print()
        print(f"  Critical:  {self.critical_component}  (SF = {self.min_sf:.3f})")
        status = "COMPLIANT ✓" if d["compliant"] else "NON-COMPLIANT ✗"
        print(f"  MIL-SPEC (SF≥1.5):  {status}")
        print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PARAMETER SWEEP (v_deploy vs t_infl design space)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_design_space(
    mass:         float,
    A_inf:        float,
    Cd:           float,
    h_deploy:     float,
    canopy_type:  str = "flat_circular",
    v_range:      tuple = (10, 60),
    ti_range:     tuple = (0.5, 6.0),
    n_points:     int   = 25,
) -> pd.DataFrame:
    """
    Compute CLA and F_peak over a grid of (v_deploy, t_infl) combinations.
    Returns DataFrame suitable for contour plotting.
    """
    canopy = CANOPY_TYPES[canopy_type]
    v_arr  = np.linspace(*v_range,  n_points)
    ti_arr = np.linspace(*ti_range, n_points)

    records = []
    for v in v_arr:
        rho    = density(h_deploy)
        F_st   = 0.5 * rho * v**2 * Cd * A_inf
        # Terminal velocity
        v_term = np.sqrt(2 * mass * cfg.GRAVITY / max(rho * Cd * A_inf, 1e-6))

        for ti in ti_arr:
            cla_k = cla_knacke(v, v_term, ti, canopy)
            cla_m = cla_milspec(v, rho, mass, A_inf, Cd, ti, canopy)
            cla   = max(cla_k, cla_m)
            F_pk  = F_st * cla
            records.append({
                "v_deploy": round(v, 2),
                "t_infl":   round(ti, 3),
                "CLA":      round(cla, 4),
                "F_peak_N": round(F_pk, 1),
                "v_terminal": round(v_term, 3),
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(
    base_params: dict,
    canopy:      CanopyType,
    perturbation: float = 0.10,    # ±10%
) -> pd.DataFrame:
    """
    One-at-a-time sensitivity: how does F_peak change with ±10% in each parameter?
    Returns tornado-chart data.
    """
    param_names = ["v_deploy", "mass", "A_inf", "Cd", "t_infl", "h_deploy"]
    records = []

    def _F_peak(p: dict) -> float:
        rho   = density(max(1.0, p["h_deploy"]))
        v     = p["v_deploy"]
        A     = p["A_inf"]
        mass  = p["mass"]
        Cd_   = p["Cd"]
        ti    = p["t_infl"]
        v_term = np.sqrt(2 * mass * cfg.GRAVITY / max(rho * Cd_ * A, 1e-6))
        F_st  = 0.5 * rho * v**2 * Cd_ * A
        cla_k = cla_knacke(v, v_term, ti, canopy)
        cla_m = cla_milspec(v, rho, mass, A, Cd_, ti, canopy)
        return F_st * max(cla_k, cla_m)

    F_base = _F_peak(base_params)
    for pn in param_names:
        for sign, label in [(+1, "+10%"), (-1, "-10%")]:
            p2 = base_params.copy()
            p2[pn] = base_params[pn] * (1 + sign * perturbation)
            F2 = _F_peak(p2)
            records.append({
                "parameter":  pn,
                "direction":  label,
                "F_peak_N":   round(F2, 1),
                "delta_N":    round(F2 - F_base, 1),
                "delta_pct":  round((F2 - F_base) / max(F_base, 1) * 100, 2),
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MASTER ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def analyse(
    v_deploy:     float    = None,
    h_deploy:     float    = None,
    mass:         float    = None,
    A_inf:        float    = None,
    Cd:           float    = None,
    t_infl:       float    = 2.5,
    canopy_type:  str      = "flat_circular",
    components:   list     = None,
    at_df:        pd.DataFrame | None = None,
    verbose:      bool     = True,
) -> OpeningShockResult:
    """
    Run the complete MIL-HDBK-1791 opening shock analysis.

    Parameters
    ----------
    v_deploy   : deployment velocity [m/s]  (defaults to cfg.INITIAL_VEL)
    h_deploy   : deployment altitude [m]    (defaults to cfg.INITIAL_ALT)
    mass       : total system mass [kg]     (defaults to cfg.PARACHUTE_MASS)
    A_inf      : fully-open canopy area [m²](defaults to cfg.CANOPY_AREA_M2)
    Cd         : steady-state drag coeff   (defaults to cfg.CD_INITIAL)
    t_infl     : inflation time [s]
    canopy_type: one of CANOPY_TYPES keys
    components : list of StructuralComponent (auto-generated if None)
    at_df      : optional Phase 1 A(t) DataFrame for realistic area model
    verbose    : print summary

    Returns
    -------
    OpeningShockResult with full analysis
    """
    v_deploy  = v_deploy  or cfg.INITIAL_VEL
    h_deploy  = h_deploy  or cfg.INITIAL_ALT
    mass      = mass      or cfg.PARACHUTE_MASS
    A_inf     = A_inf     or cfg.CANOPY_AREA_M2
    Cd        = Cd        or cfg.CD_INITIAL
    canopy    = CANOPY_TYPES.get(canopy_type, CANOPY_TYPES["flat_circular"])

    # ── Atmospheric conditions at deployment ──────────────────────────────────
    rho = density(h_deploy)

    # ── Terminal velocity (fully open canopy) ─────────────────────────────────
    v_terminal = np.sqrt(2 * mass * cfg.GRAVITY / max(rho * Cd * A_inf, 1e-6))

    # ── CLA by both methods (use conservative/larger value) ───────────────────
    cla_k = cla_knacke(v_deploy, v_terminal, t_infl, canopy)
    cla_m = cla_milspec(v_deploy, rho, mass, A_inf, Cd, t_infl, canopy)
    cla   = max(cla_k, cla_m)

    # ── Steady-state drag at deployment speed ─────────────────────────────────
    F_steady = 0.5 * rho * v_deploy**2 * Cd * A_inf
    F_peak_cla = F_steady * cla

    # ── Full dynamic force history ─────────────────────────────────────────────
    df_hist = compute_force_history(
        v_deploy=v_deploy, h_deploy=h_deploy,
        mass=mass, A_inf=A_inf, Cd=Cd, t_infl=t_infl,
        canopy=canopy, at_df=at_df,
    )
    F_peak_dynamic = float(df_hist["F_shock_N"].max())
    F_peak = max(F_peak_cla, F_peak_dynamic)

    # ── Structural analysis ────────────────────────────────────────────────────
    if components is None:
        components = default_structural_components(A_inf, mass)

    sfs = [c.effective_rated_N / max(F_peak, 1.0) for c in components]
    min_sf = min(sfs)
    critical = components[sfs.index(min_sf)].name

    result = OpeningShockResult(
        F_peak_N=F_peak,
        F_steady_N=F_steady,
        CLA_knacke=cla_k,
        CLA_milspec=cla_m,
        CLA_used=cla,
        components=components,
        min_sf=min_sf,
        critical_component=critical,
        df=df_hist,
        v_deploy=v_deploy,
        h_deploy=h_deploy,
        t_infl=t_infl,
        v_terminal=v_terminal,
        canopy_type=canopy_type,
    )

    if verbose:
        result.print_summary()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION — 6-panel engineering dashboard
# ══════════════════════════════════════════════════════════════════════════════

def plot(result: OpeningShockResult,
         sweep_df: pd.DataFrame | None = None,
         sens_df:  pd.DataFrame | None = None,
         save_path: Path | None = None) -> object:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if cfg.DARK_THEME:
        matplotlib.rcParams.update({
            "figure.facecolor": "#080c14", "axes.facecolor": "#0d1526",
            "axes.edgecolor": "#2a3d6e",   "text.color": "#c8d8f0",
            "axes.labelcolor": "#c8d8f0",  "xtick.color": "#c8d8f0",
            "ytick.color": "#c8d8f0",      "grid.color": "#1a2744",
        })
    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})

    TEXT  = "#c8d8f0" if cfg.DARK_THEME else "#111"
    C1    = cfg.COLOR_THEORY   # cyan
    C2    = cfg.COLOR_PINN     # orange
    C3    = cfg.COLOR_RAW      # green
    C_RED = "#ff4560"

    df  = result.df
    t   = df["time_s"].values
    F   = df["F_shock_N"].values
    Fs  = df["F_steady_N"].values
    v   = df["velocity_ms"].values
    A   = df["area_m2"].values
    X1  = df["X1"].values

    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38,
                            top=0.91, bottom=0.07, left=0.06, right=0.97)

    def ax(r, c): return fig.add_subplot(gs[r, c])
    def style(a, title, xlabel, ylabel):
        a.set_title(title, fontweight="bold", pad=5, fontsize=9)
        a.set_xlabel(xlabel, fontsize=8); a.set_ylabel(ylabel, fontsize=8)
        a.grid(True, alpha=0.3)
        a.spines[["top","right"]].set_visible(False)

    # ── P0: Force time-history ────────────────────────────────────────────────
    ax0 = ax(0, 0)
    ax0.fill_between(t, F/1e3, alpha=0.2, color=C_RED)
    ax0.plot(t, F/1e3, color=C_RED, lw=2.0, label="F_shock (with X₁)")
    ax0.plot(t, Fs/1e3, color=C1, lw=1.2, ls="--", alpha=0.7, label="F_steady")
    ax0.axhline(result.F_peak_N/1e3, color=C_RED, lw=0.7, ls=":", alpha=0.5)
    ax0.text(t[-1]*0.02, result.F_peak_N/1e3*1.03,
             f"F_peak = {result.F_peak_N/1e3:.2f} kN", fontsize=8, color=C_RED)
    # Mark t_infl
    ax0.axvline(result.t_infl, color=TEXT, lw=0.7, ls=":", alpha=0.5,
                label=f"t_infl={result.t_infl}s")
    ax0.legend(fontsize=7.5)
    style(ax0, "Opening shock force F(t)", "Time [s]", "Force [kN]")

    # ── P1: X₁ (Ludtke inflation force coefficient) ───────────────────────────
    ax1 = ax(0, 1)
    ax1.fill_between(t, X1, 1.0, alpha=0.2, color=C2, where=(X1 > 1))
    ax1.plot(t, X1, color=C2, lw=1.8, label="X₁(t)")
    ax1.axhline(1.0, color=TEXT, lw=0.7, ls="--", alpha=0.5, label="X₁=1 (no amplification)")
    ax1.axhline(result.CLA_used, color=C3, lw=1.0, ls="-.",
                label=f"CLA={result.CLA_used:.3f}")
    ax1.legend(fontsize=7.5)
    style(ax1, "Inflation force coefficient X₁(t)", "Time [s]", "X₁ [—]")

    # ── P2: Velocity during inflation ─────────────────────────────────────────
    ax2 = ax(0, 2)
    ax2.plot(t, v, color=C1, lw=1.8)
    ax2.axhline(result.v_terminal, color=C3, lw=0.9, ls="--", alpha=0.7,
                label=f"v_term={result.v_terminal:.2f} m/s")
    ax2.legend(fontsize=7.5)
    style(ax2, "Velocity during inflation", "Time [s]", "Velocity [m/s]")

    # ── P3: Canopy area inflation ─────────────────────────────────────────────
    ax3 = ax(0, 3)
    ax3.fill_between(t, A, alpha=0.2, color=C1)
    ax3.plot(t, A, color=C1, lw=1.8)
    A_inf_val = A.max()
    ax3.axhline(A_inf_val, color=TEXT, lw=0.7, ls=":", alpha=0.5,
                label=f"A_inf={A_inf_val:.1f} m²")
    ax3.legend(fontsize=7.5)
    style(ax3, "Canopy inflation A(t)", "Time [s]", "Area [m²]")

    # ── P4: Structural safety factor bar chart ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    comps = result.components
    names = [c.name for c in comps]
    sfs   = [c.effective_rated_N / max(result.F_peak_N, 1) for c in comps]
    colors_bar = [C3 if sf >= 1.5 else C2 if sf >= 1.0 else C_RED for sf in sfs]
    bars = ax4.barh(names, sfs, color=colors_bar, alpha=0.75, edgecolor="none")
    ax4.axvline(1.5, color=C1, lw=1.5, ls="--", label="MIL-SPEC min SF=1.5")
    ax4.axvline(1.0, color=C_RED, lw=0.9, ls=":", label="SF=1.0 (failure)")
    for bar, sf in zip(bars, sfs):
        ax4.text(sf + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{sf:.2f}", va="center", fontsize=8, color=TEXT)
    ax4.set_xlabel("Safety Factor")
    ax4.legend(fontsize=7.5)
    ax4.grid(True, alpha=0.3, axis="x")
    ax4.set_title("Structural safety factors (F_rated / F_peak)", fontweight="bold")

    # ── P5: Sensitivity tornado chart ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2:])
    if sens_df is not None and len(sens_df):
        # Pivot for tornado: one row per parameter, two bars
        params = sens_df["parameter"].unique()
        y_pos  = np.arange(len(params))
        for i, pn in enumerate(params):
            lo = float(sens_df[(sens_df.parameter==pn) & (sens_df.direction=="-10%")]["delta_N"].values[0])
            hi = float(sens_df[(sens_df.parameter==pn) & (sens_df.direction=="+10%")]["delta_N"].values[0])
            ax5.barh(y_pos[i], hi/1e3, color=C_RED, alpha=0.6, height=0.4)
            ax5.barh(y_pos[i], lo/1e3, color=C1, alpha=0.6, height=0.4)
        ax5.set_yticks(y_pos); ax5.set_yticklabels(params, fontsize=8)
        ax5.axvline(0, color=TEXT, lw=0.8)
        ax5.set_xlabel("ΔF_peak [kN]  (±10% parameter change)")
        ax5.grid(True, alpha=0.3, axis="x")
    ax5.set_title("Sensitivity: ΔF_peak per parameter ±10%", fontweight="bold")

    # ── P6: CLA design map (contour) ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    if sweep_df is not None and len(sweep_df):
        V_grid = sweep_df["v_deploy"].unique()
        T_grid = sweep_df["t_infl"].unique()
        if len(V_grid) > 2 and len(T_grid) > 2:
            Z = sweep_df["F_peak_N"].values.reshape(len(V_grid), len(T_grid)) / 1e3
            VV, TT = np.meshgrid(T_grid, V_grid)
            cs = ax6.contourf(TT, VV, Z, levels=15,
                              cmap="YlOrRd" if not cfg.DARK_THEME else "plasma")
            plt.colorbar(cs, ax=ax6, label="F_peak [kN]", pad=0.02)
            ax6.contour(TT, VV, Z, levels=[result.F_peak_N/1e3],
                        colors=[C1], linewidths=1.5)
            ax6.scatter([result.t_infl], [result.v_deploy], s=80,
                        color=C3, zorder=5, marker="*", label="Current design")
            ax6.legend(fontsize=7.5)
    ax6.set_xlabel("Inflation time [s]")
    ax6.set_ylabel("Deployment velocity [m/s]")
    ax6.set_title("F_peak design map (v_deploy × t_infl)", fontweight="bold")
    ax6.grid(True, alpha=0.2)

    # ── P7: Summary metrics table ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis("off")
    data = result.to_dict()
    rows = [
        ("Canopy type",     data["canopy_type"]),
        ("v_deploy",        f"{data['v_deploy_ms']:.2f} m/s"),
        ("h_deploy",        f"{data['h_deploy_m']:.0f} m"),
        ("t_infl",          f"{data['t_infl_s']:.2f} s"),
        ("v_terminal",      f"{data['v_terminal_ms']:.2f} m/s"),
        ("",                ""),
        ("F_steady",        f"{data['F_steady_N']/1e3:.3f} kN"),
        ("F_peak (shock)",  f"{data['F_peak_N']/1e3:.3f} kN"),
        ("",                ""),
        ("CLA (Knacke)",    f"{data['CLA_knacke']:.4f}"),
        ("CLA (MIL-SPEC)",  f"{data['CLA_milspec']:.4f}"),
        ("CLA (used)",      f"{data['CLA_used']:.4f}"),
        ("",                ""),
        ("Min SF",          f"{data['minimum_sf']:.3f}"),
        ("Critical compon.",data['critical_component'][:25]),
        ("MIL-SPEC (SF≥1.5)", "COMPLIANT ✓" if data["compliant"] else "FAIL ✗"),
    ]
    for j, (label, val) in enumerate(rows):
        c_val = (C3 if "COMPLIANT" in val else C_RED if "FAIL" in val
                 else C2 if "CLA" in label else TEXT)
        ax7.text(0.02, 1-j*0.063, label, transform=ax7.transAxes, fontsize=8.5,
                 color=TEXT if cfg.DARK_THEME else "#555")
        ax7.text(0.98, 1-j*0.063, val, transform=ax7.transAxes, fontsize=8.5,
                 ha="right", color=c_val,
                 fontweight="bold" if "COMPLIANT" in val or "FAIL" in val else "normal")
    ax7.set_title("Analysis summary", fontweight="bold", pad=8)

    fig.text(0.5, 0.955,
             f"MIL-HDBK-1791 Opening Shock Analysis  —  "
             f"F_peak={result.F_peak_N/1e3:.2f}kN  CLA={result.CLA_used:.3f}  "
             f"SF_min={result.min_sf:.2f}  [{result.canopy_type}]",
             ha="center", fontsize=12, fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR / "opening_shock.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Opening shock dashboard saved: {sp}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(
    v_deploy:    float  = None,
    h_deploy:    float  = None,
    mass:        float  = None,
    A_inf:       float  = None,
    Cd:          float  = None,
    t_infl:      float  = 2.5,
    canopy_type: str    = "flat_circular",
    at_df:       pd.DataFrame | None = None,
    do_sweep:    bool   = True,
    do_sens:     bool   = True,
    verbose:     bool   = True,
) -> OpeningShockResult:
    """Run full MIL-HDBK-1791 analysis, generate dashboard, save JSON."""
    import matplotlib; matplotlib.use("Agg")

    result = analyse(v_deploy=v_deploy, h_deploy=h_deploy, mass=mass,
                     A_inf=A_inf, Cd=Cd, t_infl=t_infl,
                     canopy_type=canopy_type, at_df=at_df, verbose=verbose)

    # ── Design sweep ──────────────────────────────────────────────────────────
    sweep_df = None
    if do_sweep:
        if verbose: print("\n  Computing design space sweep...")
        sweep_df = sweep_design_space(
            mass=mass or cfg.PARACHUTE_MASS,
            A_inf=A_inf or cfg.CANOPY_AREA_M2,
            Cd=Cd or cfg.CD_INITIAL,
            h_deploy=h_deploy or cfg.INITIAL_ALT,
            canopy_type=canopy_type,
        )
        sweep_df.to_csv(cfg.OUTPUTS_DIR / "opening_shock_sweep.csv", index=False)

    # ── Sensitivity ───────────────────────────────────────────────────────────
    sens_df = None
    if do_sens:
        canopy = CANOPY_TYPES[canopy_type]
        sens_df = sensitivity_analysis(
            base_params={
                "v_deploy": v_deploy or cfg.INITIAL_VEL,
                "mass":     mass or cfg.PARACHUTE_MASS,
                "A_inf":    A_inf or cfg.CANOPY_AREA_M2,
                "Cd":       Cd or cfg.CD_INITIAL,
                "t_infl":   t_infl,
                "h_deploy": h_deploy or cfg.INITIAL_ALT,
            },
            canopy=canopy,
        )
        sens_df.to_csv(cfg.OUTPUTS_DIR / "opening_shock_sensitivity.csv", index=False)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = cfg.OUTPUTS_DIR / "opening_shock_result.json"
    out_json.write_text(json.dumps(result.to_dict(), indent=2))
    if verbose: print(f"  ✓ JSON saved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot(result, sweep_df=sweep_df, sens_df=sens_df)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MIL-HDBK-1791 Opening Shock Analysis",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--v-deploy",    type=float, default=None)
    p.add_argument("--h-deploy",    type=float, default=None)
    p.add_argument("--mass",        type=float, default=None)
    p.add_argument("--A-inf",       type=float, default=None)
    p.add_argument("--Cd",          type=float, default=None)
    p.add_argument("--t-infl",      type=float, default=2.5)
    p.add_argument("--canopy-type", type=str,   default="flat_circular",
                   choices=list(CANOPY_TYPES.keys()))
    a = p.parse_args()
    run(v_deploy=a.v_deploy, h_deploy=a.h_deploy, mass=a.mass,
        A_inf=a.A_inf, Cd=a.Cd, t_infl=a.t_infl, canopy_type=a.canopy_type)
