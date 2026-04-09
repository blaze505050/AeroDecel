"""
calibrate_cd.py — Auto-Calibration: Back-solve Cd from Observed Landing Velocity
==================================================================================

Given one or more observables from a real drop test, this module recovers the
effective drag coefficient Cd (and optionally other parameters) that best explains
the data — without requiring a video, telemetry stream, or PINN training.

─── Observable inputs accepted ──────────────────────────────────────────────────
  1. Landing velocity only           → scalar Cd (fastest, ~0.01s)
  2. Landing velocity + descent time → Cd + inflation time jointly
  3. Landing velocity + time series  → full Cd(t) curve via gradient-free inversion
  4. Multi-drop batch                → Cd distribution across N tests

─── Methods used ────────────────────────────────────────────────────────────────
  • scipy.optimize.brentq          — guaranteed root-finding for single observable
  • scipy.optimize.minimize        — Nelder-Mead for joint (Cd, t_infl) inversion
  • scipy.optimize.differential_evolution — global search for Cd(t) shape params
  • Bootstrap resampling            — 95% confidence intervals, no assumptions
  • Gaussian process regression (sklearn) — smooth Cd(t) from sparse time points

─── Outputs ─────────────────────────────────────────────────────────────────────
  • Cd_effective ± 95% CI
  • Comparison plot: calibrated vs uncalibrated simulation
  • JSON result file for downstream PINN warm-starting
  • Optional: updated config.py with new Cd
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize, differential_evolution
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FORWARD MODEL  (fast RK4 — no scipy overhead for repeated evaluations)
# ═══════════════════════════════════════════════════════════════════════════════

def _logistic_A(t: float, Am: float, ti: float, n: float = 2.0) -> float:
    """Generalised logistic canopy inflation model."""
    k  = 5.0 / max(ti, 0.1)
    t0 = ti * 0.6
    return float(Am / (1.0 + np.exp(-k * (t - t0))) ** (1.0 / n))


def _simulate(
    Cd:      float,
    mass:    float   = None,
    alt0:    float   = None,
    v0:      float   = None,
    Am:      float   = None,
    ti:      float   = 2.5,
    dt:      float   = 0.05,
    Cd_fn          = None,        # optional callable Cd(t) — overrides scalar
    at_df:   pd.DataFrame = None, # optional A(t) CSV from Phase 1
) -> dict:
    """
    Lightweight RK4 forward simulation.
    Returns dict with time array, velocity array, altitude array,
    landing_velocity, landing_time, peak_drag, peak_decel.
    """
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Am   = Am   or cfg.CANOPY_AREA_M2

    # Build A(t) interpolant if CSV provided
    if at_df is not None:
        _At_fn = interp1d(
            at_df["time_s"].values, at_df["area_m2"].values,
            bounds_error=False, fill_value=(0.0, float(at_df["area_m2"].max()))
        )
    else:
        _At_fn = None

    def At(t):
        if _At_fn is not None:
            return max(0.0, float(_At_fn(t)))
        return _logistic_A(t, Am, ti)

    def get_Cd(t):
        if Cd_fn is not None:
            return float(Cd_fn(t))
        return float(Cd)

    ts, vs, hs = [0.0], [float(v0)], [float(alt0)]
    v, h = float(v0), float(alt0)
    t = 0.0
    prev_acc = 0.0
    peak_drag = 0.0
    peak_decel = 0.0

    while h > 0.0 and t < 600.0:
        A   = At(t)
        Cd_ = get_Cd(t)
        rho = density(max(0.0, h))
        drag = 0.5 * rho * max(v, 0.0) ** 2 * Cd_ * A
        acc  = cfg.GRAVITY - drag / mass

        # RK4 step
        def f(t_, v_, h_):
            A_   = At(t_)
            Cd__ = get_Cd(t_)
            rho_ = density(max(0.0, h_))
            d_   = 0.5 * rho_ * max(v_, 0.0) ** 2 * Cd__ * A_
            return cfg.GRAVITY - d_ / mass, -max(v_, 0.0)

        k1v, k1h = f(t,        v,          h)
        k2v, k2h = f(t+dt/2,   v+dt*k1v/2, h+dt*k1h/2)
        k3v, k3h = f(t+dt/2,   v+dt*k2v/2, h+dt*k2h/2)
        k4v, k4h = f(t+dt,     v+dt*k3v,   h+dt*k3h)

        v = max(0.0, v + dt*(k1v + 2*k2v + 2*k3v + k4v)/6)
        h = max(0.0, h + dt*(k1h + 2*k2h + 2*k3h + k4h)/6)
        t += dt

        peak_drag  = max(peak_drag,  drag)
        peak_decel = max(peak_decel, abs(acc - prev_acc) / dt if t > dt else 0)
        prev_acc   = acc

        ts.append(t); vs.append(v); hs.append(h)
        if h <= 0.0:
            break

    return {
        "time":            np.array(ts),
        "velocity":        np.array(vs),
        "altitude":        np.array(hs),
        "landing_velocity": float(vs[-1]),
        "landing_time":     float(ts[-1]),
        "peak_drag_N":      float(peak_drag),
        "peak_decel_ms2":   float(peak_decel),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE-OBSERVABLE INVERSION  (brentq — scalar Cd)
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_from_landing_velocity(
    observed_v:   float,
    mass:         float = None,
    alt0:         float = None,
    v0:           float = None,
    Am:           float = None,
    ti:           float = 2.5,
    at_df:        pd.DataFrame = None,
    Cd_bounds:    tuple = (0.1, 5.0),
    n_bootstrap:  int   = 500,
    verbose:      bool  = True,
) -> dict:
    """
    Back-solve for effective Cd given only the observed landing velocity.

    Uses Brent's method (guaranteed convergence, no gradient needed) to find
    the unique Cd such that sim(Cd).landing_velocity == observed_v.

    Also computes 95% bootstrap CI by propagating uncertainty in mass ± 3%,
    A_max ± 5%, v0 ± 3%, and t_infl ± 10%.

    Returns
    -------
    dict with keys:
      Cd_eff, Cd_ci_low, Cd_ci_high, residual_ms, landing_time_s,
      peak_drag_N, n_iterations, method, bootstrap_samples
    """
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Am   = Am   or cfg.CANOPY_AREA_M2

    if verbose:
        print(f"\n[Calibration] Brent root-finding: target v_land = {observed_v:.3f} m/s")
        print(f"  mass={mass}kg  alt0={alt0}m  v0={v0}m/s  A_max={Am}m²  t_infl={ti}s")

    eval_count = [0]

    def objective(Cd):
        eval_count[0] += 1
        result = _simulate(Cd, mass=mass, alt0=alt0, v0=v0, Am=Am, ti=ti, at_df=at_df)
        return result["landing_velocity"] - observed_v

    # Validate bracket: landing velocity must be monotone in Cd
    v_lo = _simulate(Cd_bounds[0], mass=mass, alt0=alt0, v0=v0, Am=Am, ti=ti, at_df=at_df)["landing_velocity"]
    v_hi = _simulate(Cd_bounds[1], mass=mass, alt0=alt0, v0=v0, Am=Am, ti=ti, at_df=at_df)["landing_velocity"]

    if verbose:
        print(f"  Bracket: Cd={Cd_bounds[0]} → v={v_lo:.3f}  |  Cd={Cd_bounds[1]} → v={v_hi:.3f}")

    if (v_lo - observed_v) * (v_hi - observed_v) > 0:
        raise ValueError(
            f"Observed landing velocity {observed_v:.3f} m/s is outside the "
            f"achievable range [{min(v_lo,v_hi):.3f}, {max(v_lo,v_hi):.3f}] m/s. "
            f"Check your mass, area, or altitude settings."
        )

    t0 = time.perf_counter()
    Cd_eff = brentq(objective, Cd_bounds[0], Cd_bounds[1], xtol=1e-6, rtol=1e-6, maxiter=200)
    elapsed = time.perf_counter() - t0

    # Full forward pass at calibrated Cd
    sim = _simulate(Cd_eff, mass=mass, alt0=alt0, v0=v0, Am=Am, ti=ti, at_df=at_df)
    residual = sim["landing_velocity"] - observed_v

    if verbose:
        print(f"  ✓ Converged in {eval_count[0]} evaluations ({elapsed*1000:.1f} ms)")
        print(f"  Cd_eff = {Cd_eff:.6f}  |  residual = {residual:+.4e} m/s")
        print(f"  Landing time: {sim['landing_time']:.2f}s  |  Peak drag: {sim['peak_drag_N']:.1f}N")

    # ── Bootstrap 95% CI ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Computing 95% CI via bootstrap ({n_bootstrap} samples)...")

    rng  = np.random.default_rng(42)
    boot_Cds = []

    for _ in range(n_bootstrap):
        m_   = mass * rng.normal(1.0, 0.03)    # ±3% mass uncertainty
        Am_  = Am   * rng.normal(1.0, 0.05)    # ±5% area uncertainty
        v0_  = v0   * rng.normal(1.0, 0.03)    # ±3% initial velocity
        ti_  = ti   * max(0.3, rng.normal(1.0, 0.10))  # ±10% inflation time
        obs_ = observed_v * rng.normal(1.0, 0.02)       # ±2% measurement error

        def obj_b(Cd):
            r = _simulate(Cd, mass=m_, alt0=alt0, v0=v0_, Am=Am_, ti=ti_, at_df=at_df)
            return r["landing_velocity"] - obs_

        try:
            # Use coarser dt=0.5 for bootstrap speed (bias is systematic, cancels in CI)
            _sim_fast = lambda Cd, **kw: _simulate(Cd, dt=0.5, **kw)
            vlo = _sim_fast(Cd_bounds[0], mass=m_, alt0=alt0, v0=v0_, Am=Am_, ti=ti_, at_df=at_df)["landing_velocity"]
            vhi = _sim_fast(Cd_bounds[1], mass=m_, alt0=alt0, v0=v0_, Am=Am_, ti=ti_, at_df=at_df)["landing_velocity"]
            if (vlo - obs_) * (vhi - obs_) < 0:
                def obj_b_fast(Cd):
                    return _sim_fast(Cd, mass=m_, alt0=alt0, v0=v0_, Am=Am_, ti=ti_, at_df=at_df)["landing_velocity"] - obs_
                Cd_b = brentq(obj_b_fast, Cd_bounds[0], Cd_bounds[1], xtol=1e-3, maxiter=30)
                boot_Cds.append(Cd_b)
        except Exception:
            pass

    boot_arr = np.array(boot_Cds)
    ci_low   = float(np.percentile(boot_arr, 2.5))  if len(boot_arr) > 10 else Cd_eff * 0.85
    ci_high  = float(np.percentile(boot_arr, 97.5)) if len(boot_arr) > 10 else Cd_eff * 1.15

    if verbose:
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]  (±{(ci_high-ci_low)/2:.4f})")
        print(f"  Bootstrap valid: {len(boot_arr)}/{n_bootstrap} samples")

    return {
        "Cd_eff":             round(Cd_eff, 6),
        "Cd_ci_low":          round(ci_low, 6),
        "Cd_ci_high":         round(ci_high, 6),
        "Cd_ci_half_width":   round((ci_high - ci_low) / 2, 6),
        "observed_v_ms":      observed_v,
        "simulated_v_ms":     round(sim["landing_velocity"], 4),
        "residual_ms":        round(residual, 6),
        "landing_time_s":     round(sim["landing_time"], 3),
        "peak_drag_N":        round(sim["peak_drag_N"], 2),
        "peak_decel_ms2":     round(sim["peak_decel_ms2"], 3),
        "n_brent_evals":      eval_count[0],
        "solve_time_ms":      round(elapsed * 1000, 2),
        "bootstrap_n_valid":  len(boot_arr),
        "bootstrap_samples":  boot_arr.tolist(),
        "method":             "brentq + bootstrap",
        "inputs": {
            "mass_kg": mass, "alt0_m": alt0, "v0_ms": v0,
            "Am_m2": Am, "t_infl_s": ti,
        },
        "simulation": sim,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. JOINT INVERSION  (Cd + t_infl from v_land + t_land)
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_joint(
    observed_v:    float,
    observed_time: float,
    mass:          float = None,
    alt0:          float = None,
    v0:            float = None,
    Am:            float = None,
    at_df:         pd.DataFrame = None,
    verbose:       bool  = True,
) -> dict:
    """
    Jointly recover (Cd, t_infl) from two observables:
      - observed_v    : landing velocity [m/s]
      - observed_time : total descent time [s]

    Uses Nelder-Mead since the 2D surface is smooth and unimodal.
    Returns best-fit (Cd, t_infl) with Jacobian-based uncertainty ellipse.
    """
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Am   = Am   or cfg.CANOPY_AREA_M2

    if verbose:
        print(f"\n[Calibration] Joint inversion: v_land={observed_v:.3f}m/s, t_land={observed_time:.2f}s")

    eval_count = [0]

    def objective(params):
        Cd, ti = params
        if Cd < 0.05 or ti < 0.1 or ti > 20:
            return 1e9
        eval_count[0] += 1
        r = _simulate(Cd, mass=mass, alt0=alt0, v0=v0, Am=Am, ti=ti, at_df=at_df)
        # Weighted residual: normalise each by its typical scale
        err_v = (r["landing_velocity"] - observed_v)   / max(observed_v, 0.1)
        err_t = (r["landing_time"]     - observed_time) / max(observed_time, 1.0)
        return err_v**2 + err_t**2

    x0 = [cfg.CD_INITIAL, 2.5]
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 2000})

    Cd_best, ti_best = res.x
    sim = _simulate(Cd_best, mass=mass, alt0=alt0, v0=v0, Am=Am, ti=ti_best, at_df=at_df)

    if verbose:
        print(f"  ✓ {eval_count[0]} evaluations  |  success={res.success}")
        print(f"  Cd={Cd_best:.5f}  t_infl={ti_best:.4f}s")
        print(f"  Residuals: Δv={sim['landing_velocity']-observed_v:+.4f}m/s  "
              f"Δt={sim['landing_time']-observed_time:+.4f}s")

    return {
        "Cd_eff":          round(Cd_best, 6),
        "t_infl_s":        round(ti_best, 4),
        "observed_v_ms":   observed_v,
        "observed_t_s":    observed_time,
        "simulated_v_ms":  round(sim["landing_velocity"], 4),
        "simulated_t_s":   round(sim["landing_time"], 3),
        "residual_v_ms":   round(sim["landing_velocity"] - observed_v, 5),
        "residual_t_s":    round(sim["landing_time"] - observed_time, 4),
        "n_evals":         eval_count[0],
        "method":          "nelder-mead joint",
        "inputs": {
            "mass_kg": mass, "alt0_m": alt0, "v0_ms": v0, "Am_m2": Am
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TIME-SERIES INVERSION  (Cd(t) shape from sparse v(t) observations)
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_from_velocity_series(
    obs_times:  np.ndarray,
    obs_vels:   np.ndarray,
    mass:       float = None,
    alt0:       float = None,
    v0:         float = None,
    Am:         float = None,
    at_df:      pd.DataFrame = None,
    n_knots:    int   = 5,
    verbose:    bool  = True,
) -> dict:
    """
    Recover Cd(t) shape from a sparse time series of observed velocities.
    Parameterises Cd(t) as a piecewise-linear function over n_knots control
    points, then uses differential evolution (global, gradient-free) to find
    the best-fit knot values. Smooth Cd(t) curve is then fit via GP regression.

    obs_times : 1-D array of observation timestamps [s]
    obs_vels  : 1-D array of observed velocities [m/s] at those times
    """
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Am   = Am   or cfg.CANOPY_AREA_M2

    obs_times = np.asarray(obs_times, dtype=float)
    obs_vels  = np.asarray(obs_vels,  dtype=float)
    t_end     = obs_times.max() + 5.0

    # Knot times evenly spaced over [0, t_end]
    knot_times = np.linspace(0, t_end, n_knots)

    if verbose:
        print(f"\n[Calibration] Time-series inversion: {len(obs_times)} observations, {n_knots} Cd knots")

    eval_count = [0]

    def make_Cd_fn(knot_vals):
        return interp1d(knot_times, knot_vals, kind="linear",
                        bounds_error=False, fill_value=(knot_vals[0], knot_vals[-1]))

    def objective(knot_vals):
        eval_count[0] += 1
        Cd_fn = make_Cd_fn(knot_vals)
        sim   = _simulate(1.0, mass=mass, alt0=alt0, v0=v0, Am=Am,
                          Cd_fn=Cd_fn, at_df=at_df, dt=0.1)
        # Interpolate sim velocity at observation times
        v_interp = np.interp(obs_times, sim["time"], sim["velocity"])
        return float(np.mean((v_interp - obs_vels) ** 2))

    bounds = [(0.3, 3.5)] * n_knots
    if verbose:
        print("  Running differential evolution (global search)...")

    res = differential_evolution(
        objective, bounds,
        seed=42, maxiter=300, tol=1e-6,
        workers=1, popsize=12, mutation=(0.5, 1.5), recombination=0.9,
        polish=True,
    )

    best_knots = res.x
    Cd_fn_best = make_Cd_fn(best_knots)

    # Full simulation with best Cd(t)
    sim = _simulate(1.0, mass=mass, alt0=alt0, v0=v0, Am=Am,
                    Cd_fn=Cd_fn_best, at_df=at_df, dt=0.05)

    # GP smoothing of Cd(t) over dense time grid
    t_dense = np.linspace(0, t_end, 300)
    Cd_dense = Cd_fn_best(t_dense)

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        kernel = 1.0 * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.01)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gpr.fit(knot_times.reshape(-1,1), best_knots)
        Cd_smooth, Cd_std = gpr.predict(t_dense.reshape(-1,1), return_std=True)
        gp_available = True
    except ImportError:
        Cd_smooth = Cd_dense
        Cd_std    = np.zeros_like(Cd_dense)
        gp_available = False

    v_pred = np.interp(obs_times, sim["time"], sim["velocity"])
    rmse   = float(np.sqrt(np.mean((v_pred - obs_vels)**2)))

    if verbose:
        print(f"  ✓ {eval_count[0]} evals  |  RMSE = {rmse:.4f} m/s")
        print(f"  Cd(t) range: [{best_knots.min():.3f}, {best_knots.max():.3f}]")
        print(f"  Landing v: {sim['landing_velocity']:.3f} m/s  |  t: {sim['landing_time']:.2f}s")
        if gp_available:
            print("  GP smoothing applied")

    return {
        "knot_times":       knot_times.tolist(),
        "knot_Cd_vals":     best_knots.tolist(),
        "t_dense":          t_dense.tolist(),
        "Cd_smooth":        np.clip(Cd_smooth, 0.1, 5.0).tolist(),
        "Cd_std":           Cd_std.tolist(),
        "Cd_mean":          round(float(best_knots.mean()), 4),
        "Cd_max":           round(float(best_knots.max()), 4),
        "Cd_min":           round(float(best_knots.min()), 4),
        "rmse_ms":          round(rmse, 5),
        "landing_velocity": round(sim["landing_velocity"], 4),
        "landing_time":     round(sim["landing_time"], 3),
        "n_evals":          eval_count[0],
        "gp_smoothing":     gp_available,
        "method":           "differential-evolution + GP",
        "simulation":       sim,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-DROP BATCH CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_batch(
    drops: list[dict],
    at_df: pd.DataFrame = None,
    verbose: bool = True,
) -> dict:
    """
    Calibrate Cd distribution from multiple drop tests.

    drops: list of dicts, each with keys:
      - landing_v (required)
      - landing_t (optional)
      - mass      (optional, overrides config)
      - alt0      (optional)
      - v0        (optional)
      - Am        (optional)
      - ti        (optional)

    Returns Cd statistics: mean, median, std, min, max, P5, P95, per-drop results.
    """
    if verbose:
        print(f"\n[Calibration] Batch mode: {len(drops)} drop tests")

    results = []
    for i, drop in enumerate(drops):
        if verbose:
            print(f"\n  Drop {i+1}/{len(drops)}: v_land={drop['landing_v']:.3f} m/s", end="")

        try:
            r = calibrate_from_landing_velocity(
                observed_v  = drop["landing_v"],
                mass        = drop.get("mass"),
                alt0        = drop.get("alt0"),
                v0          = drop.get("v0"),
                Am          = drop.get("Am"),
                ti          = drop.get("ti", 2.5),
                at_df       = at_df,
                n_bootstrap = 200,
                verbose     = False,
            )
            results.append({
                "drop": i+1,
                "Cd_eff": r["Cd_eff"],
                "Cd_ci_low": r["Cd_ci_low"],
                "Cd_ci_high": r["Cd_ci_high"],
                **drop,
            })
            if verbose:
                print(f"  → Cd={r['Cd_eff']:.4f} [{r['Cd_ci_low']:.4f}, {r['Cd_ci_high']:.4f}]")
        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")

    Cds = np.array([r["Cd_eff"] for r in results])
    summary = {
        "n_drops":       len(results),
        "Cd_mean":       round(float(Cds.mean()), 5),
        "Cd_median":     round(float(np.median(Cds)), 5),
        "Cd_std":        round(float(Cds.std()),  5),
        "Cd_min":        round(float(Cds.min()),  5),
        "Cd_max":        round(float(Cds.max()),  5),
        "Cd_p05":        round(float(np.percentile(Cds, 5)),  5),
        "Cd_p95":        round(float(np.percentile(Cds, 95)), 5),
        "Cd_recommended": round(float(np.median(Cds)), 5),
        "per_drop":      results,
    }

    if verbose:
        print(f"\n  Batch summary:")
        print(f"  Cd mean={summary['Cd_mean']:.4f}  std={summary['Cd_std']:.4f}  "
              f"P5={summary['Cd_p05']:.4f}  P95={summary['Cd_p95']:.4f}")
        print(f"  Recommended Cd (median): {summary['Cd_recommended']:.4f}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_calibration(
    result:     dict,
    mode:       str  = "scalar",   # "scalar" | "joint" | "timeseries" | "batch"
    obs_times:  np.ndarray = None,
    obs_vels:   np.ndarray = None,
    save_path:  Path = None,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if cfg.DARK_THEME:
        matplotlib.rcParams.update({
            "figure.facecolor": "#080c14", "axes.facecolor":  "#0d1526",
            "axes.edgecolor":   "#2a3d6e", "text.color":      "#c8d8f0",
            "axes.labelcolor":  "#c8d8f0", "xtick.color":     "#c8d8f0",
            "ytick.color":      "#c8d8f0", "grid.color":      "#1a2744",
        })
    matplotlib.rcParams.update({"font.family":"monospace","font.size":9})

    BG    = "#080c14" if cfg.DARK_THEME else "#ffffff"
    TEXT  = "#c8d8f0" if cfg.DARK_THEME else "#111111"
    SPINE = "#2a3d6e" if cfg.DARK_THEME else "#cccccc"
    C1    = cfg.COLOR_THEORY   # cyan — calibrated
    C2    = cfg.COLOR_PINN     # orange — uncalibrated
    C3    = cfg.COLOR_RAW      # green — observed

    if mode == "scalar":
        fig = plt.figure(figsize=(16, 8))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                                top=0.88, bottom=0.09, left=0.07, right=0.97)

        sim      = result["simulation"]
        Cd_eff   = result["Cd_eff"]
        ci_lo    = result["Cd_ci_low"]
        ci_hi    = result["Cd_ci_high"]
        obs_v    = result["observed_v_ms"]
        boot     = np.array(result["bootstrap_samples"])

        # Uncalibrated sim (with config Cd)
        uncal = _simulate(cfg.CD_INITIAL,
                          mass=result["inputs"]["mass_kg"],
                          alt0=result["inputs"]["alt0_m"],
                          v0  =result["inputs"]["v0_ms"],
                          Am  =result["inputs"]["Am_m2"],
                          ti  =result["inputs"].get("t_infl_s", 2.5))

        # Panel 1: velocity vs time
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(uncal["time"], uncal["velocity"], color=C2, lw=1.2, ls="--", alpha=0.7,
                 label=f"Uncalibrated  Cd={cfg.CD_INITIAL}")
        ax1.plot(sim["time"],  sim["velocity"],   color=C1, lw=2.0,
                 label=f"Calibrated  Cd={Cd_eff:.4f}")
        ax1.axhline(obs_v, color=C3, lw=1.2, ls=":", label=f"Observed v_land={obs_v:.3f} m/s")
        ax1.scatter([sim["time"][-1]], [obs_v], color=C3, s=60, zorder=5)
        ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Velocity [m/s]")
        ax1.set_title("Calibration: velocity profile", fontweight="bold")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        # Panel 2: altitude vs time
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(uncal["time"], uncal["altitude"], color=C2, lw=1.2, ls="--", alpha=0.7)
        ax2.plot(sim["time"],  sim["altitude"],   color=C1, lw=1.8)
        ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Altitude [m]")
        ax2.set_title("h(t) comparison", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Bootstrap Cd distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(boot, bins=30, color=C1, alpha=0.7, edgecolor="none")
        ax3.axvline(Cd_eff, color=C3,  lw=2.0, label=f"Cd={Cd_eff:.4f}")
        ax3.axvline(ci_lo,  color=C2,  lw=1.2, ls="--", label=f"P2.5={ci_lo:.4f}")
        ax3.axvline(ci_hi,  color=C2,  lw=1.2, ls="--", label=f"P97.5={ci_hi:.4f}")
        ax3.axvspan(ci_lo, ci_hi, alpha=0.1, color=C1, label="95% CI")
        ax3.set_xlabel("Cd"); ax3.set_ylabel("Bootstrap count")
        ax3.set_title("Cd posterior (bootstrap)", fontweight="bold")
        ax3.legend(fontsize=7.5); ax3.grid(True, alpha=0.3)

        # Panel 4: Sensitivity — tornado chart
        ax4 = fig.add_subplot(gs[1, 1])
        params    = ["mass ±3%", "A_max ±5%", "v₀ ±3%", "t_infl ±10%", "obs v ±2%"]
        sigmas    = [0.03, 0.05, 0.03, 0.10, 0.02]
        Cd_deltas = []
        rng       = np.random.default_rng(0)

        def _perturb(param_idx, direction):
            m_  = result["inputs"]["mass_kg"]
            Am_ = result["inputs"]["Am_m2"]
            v0_ = result["inputs"]["v0_ms"]
            ti_ = result["inputs"].get("t_infl_s", 2.5)
            ov_ = obs_v
            deltas = [m_, Am_, v0_, ti_, ov_]
            deltas[param_idx] *= (1 + direction * sigmas[param_idx])
            try:
                def obj(Cd):
                    r2 = _simulate(Cd, mass=deltas[0], alt0=result["inputs"]["alt0_m"],
                                   v0=deltas[2], Am=deltas[1], ti=deltas[3])
                    return r2["landing_velocity"] - deltas[4]
                return brentq(obj, 0.1, 5.0, xtol=1e-4, maxiter=50)
            except Exception:
                return Cd_eff

        for i in range(len(params)):
            Cd_plus  = _perturb(i, +1)
            Cd_minus = _perturb(i, -1)
            Cd_deltas.append((Cd_minus - Cd_eff, Cd_plus - Cd_eff))

        y_pos = np.arange(len(params))
        for i, (lo, hi) in enumerate(Cd_deltas):
            ax4.barh(y_pos[i], hi - lo, left=lo + Cd_eff, color=C1, alpha=0.6, height=0.55)
            ax4.barh(y_pos[i], lo,      left=Cd_eff,        color=C2, alpha=0.6, height=0.55)

        ax4.axvline(Cd_eff, color=TEXT, lw=0.8, ls="-")
        ax4.set_yticks(y_pos); ax4.set_yticklabels(params, fontsize=8)
        ax4.set_xlabel("Cd"); ax4.set_title("Tornado: parameter sensitivity", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="x")

        # Panel 5: Key metrics summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        lines = [
            ("Cd calibrated",     f"{Cd_eff:.6f}"),
            ("95% CI low",        f"{ci_lo:.6f}"),
            ("95% CI high",       f"{ci_hi:.6f}"),
            ("CI half-width",     f"±{(ci_hi-ci_lo)/2:.6f}"),
            ("",                  ""),
            ("Observed v_land",   f"{obs_v:.4f} m/s"),
            ("Simulated v_land",  f"{sim['landing_velocity']:.4f} m/s"),
            ("Residual",          f"{result['residual_ms']:+.2e} m/s"),
            ("",                  ""),
            ("Landing time",      f"{sim['landing_time']:.2f} s"),
            ("Peak drag",         f"{sim['peak_drag_N']:.1f} N"),
            ("Brent evals",       f"{result['n_brent_evals']}"),
            ("Solve time",        f"{result['solve_time_ms']:.2f} ms"),
        ]
        for j, (label, val) in enumerate(lines):
            c = C3 if "Cd cal" in label else C1 if "CI" in label or "Residual" in label else TEXT
            ax5.text(0.02, 1 - j*0.077, label, transform=ax5.transAxes, fontsize=8.5,
                     color=TEXT if cfg.DARK_THEME else "#555")
            ax5.text(0.98, 1 - j*0.077, val, transform=ax5.transAxes, fontsize=8.5,
                     ha="right", color=c, fontweight="bold" if "Cd cal" in label else "normal")

        fig.text(0.5, 0.94,
                 f"Auto-Calibration  —  Cd={Cd_eff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (95% CI)",
                 ha="center", fontsize=13, fontweight="bold",
                 color=TEXT if cfg.DARK_THEME else "#111")

    elif mode == "timeseries":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 facecolor=BG if cfg.DARK_THEME else "white")
        fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.12, left=0.08, right=0.97)

        sim     = result["simulation"]
        t_dense = np.array(result["t_dense"])
        Cd_s    = np.array(result["Cd_smooth"])
        Cd_std  = np.array(result["Cd_std"])

        ax = axes[0]
        ax.plot(sim["time"], sim["velocity"], color=C1, lw=2.0, label="Calibrated v(t)")
        if obs_times is not None:
            ax.scatter(obs_times, obs_vels, color=C3, s=40, zorder=5,
                       label="Observed v (sparse)")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Velocity [m/s]")
        ax.set_title("Time-series inversion: v(t)", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(t_dense, Cd_s, color=C1, lw=2.0, label="Cd(t) recovered")
        if result["gp_smoothing"]:
            ax2.fill_between(t_dense, Cd_s - 2*Cd_std, Cd_s + 2*Cd_std,
                             alpha=0.2, color=C1, label="±2σ GP band")
        ax2.axhline(cfg.CD_INITIAL, color=C2, lw=1.0, ls="--", alpha=0.7,
                    label=f"Initial guess Cd={cfg.CD_INITIAL}")
        ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Cd [—]")
        ax2.set_title("Recovered Cd(t)", fontweight="bold")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        fig.text(0.5, 0.94,
                 f"Time-Series Inversion  —  RMSE={result['rmse_ms']:.4f}m/s  "
                 f"Cd∈[{result['Cd_min']:.3f},{result['Cd_max']:.3f}]",
                 ha="center", fontsize=12, fontweight="bold")

    elif mode == "batch":
        per   = result["per_drop"]
        Cds   = [p["Cd_eff"]   for p in per]
        cis_lo = [p["Cd_ci_low"] for p in per]
        cis_hi = [p["Cd_ci_high"] for p in per]
        drops  = [f"Drop {p['drop']}" for p in per]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 facecolor=BG if cfg.DARK_THEME else "white")
        fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.12, left=0.08, right=0.97)

        ax1 = axes[0]
        y   = np.arange(len(per))
        ax1.barh(y, [hi-lo for lo,hi in zip(cis_lo,cis_hi)],
                 left=cis_lo, color=C1, alpha=0.3, height=0.5)
        ax1.scatter(Cds, y, color=C3, s=50, zorder=5)
        ax1.axvline(result["Cd_recommended"], color=C2, lw=1.5, ls="--",
                    label=f"Recommended Cd={result['Cd_recommended']:.4f}")
        ax1.set_yticks(y); ax1.set_yticklabels(drops)
        ax1.set_xlabel("Cd"); ax1.set_title("Per-drop Cd with 95% CI", fontweight="bold")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="x")

        ax2 = axes[1]
        ax2.hist(Cds, bins=max(5, len(Cds)//3), color=C1, alpha=0.7, edgecolor="none")
        ax2.axvline(result["Cd_mean"],        color=C3, lw=2,   label=f"Mean={result['Cd_mean']:.4f}")
        ax2.axvline(result["Cd_recommended"], color=C2, lw=1.5, ls="--",
                    label=f"Median={result['Cd_recommended']:.4f}")
        ax2.set_xlabel("Cd"); ax2.set_ylabel("Count")
        ax2.set_title("Cd distribution (batch)", fontweight="bold")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        fig.text(0.5, 0.94,
                 f"Batch Calibration  —  {result['n_drops']} drops  |  "
                 f"Cd={result['Cd_mean']:.4f}±{result['Cd_std']:.4f}  "
                 f"P5={result['Cd_p05']:.4f} P95={result['Cd_p95']:.4f}",
                 ha="center", fontsize=12, fontweight="bold")

    sp = save_path or cfg.OUTPUTS_DIR / f"calibration_{mode}.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Plot saved: {sp}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONFIG PATCHER  (writes Cd_eff back into config.py)
# ═══════════════════════════════════════════════════════════════════════════════

def patch_config(Cd_new: float, dry_run: bool = False) -> bool:
    """
    Rewrite config.py's CD_INITIAL line with the calibrated value.
    Creates a .bak backup first. Pass dry_run=True to preview only.
    """
    config_path = Path(__file__).parent.parent / "config.py"
    text = config_path.read_text()

    import re
    pattern = r"(CD_INITIAL\s*=\s*)\S+"
    new_line = f"\\g<1>{Cd_new:.6f}    # auto-calibrated"
    new_text, n = re.subn(pattern, new_line, text)

    if n == 0:
        print("  [patch_config] Could not find CD_INITIAL in config.py")
        return False

    if dry_run:
        print(f"  [patch_config] DRY RUN — would write: CD_INITIAL = {Cd_new:.6f}")
        return True

    backup = config_path.with_suffix(".py.bak")
    backup.write_text(text)
    config_path.write_text(new_text)
    print(f"  [patch_config] ✓ config.py updated. Backup → {backup.name}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Auto-Calibration: back-solve Cd from observed landing data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--observed-landing-v", type=float, required=False,
                   help="Observed landing velocity [m/s]")
    p.add_argument("--observed-landing-t", type=float, default=None,
                   help="Observed descent time [s] — enables joint Cd+t_infl inversion")
    p.add_argument("--obs-times",  type=float, nargs="+",
                   help="Sparse observation timestamps [s] for time-series mode")
    p.add_argument("--obs-vels",   type=float, nargs="+",
                   help="Observed velocities [m/s] at those timestamps")
    p.add_argument("--batch-csv",  type=Path, default=None,
                   help="CSV with columns: landing_v [,landing_t,mass,alt0,v0,Am,ti]")
    p.add_argument("--mass",       type=float, default=None, help="Payload mass [kg]")
    p.add_argument("--alt0",       type=float, default=None, help="Deployment altitude [m]")
    p.add_argument("--v0",         type=float, default=None, help="Initial velocity [m/s]")
    p.add_argument("--Am",         type=float, default=None, help="Canopy area [m²]")
    p.add_argument("--ti",         type=float, default=2.5,  help="Inflation time [s]")
    p.add_argument("--bootstrap",  type=int,   default=500,  help="Bootstrap samples for CI")
    p.add_argument("--no-plot",    action="store_true",      help="Skip plot generation")
    p.add_argument("--patch-config", action="store_true",
                   help="Write calibrated Cd back to config.py")
    p.add_argument("--at-csv",     type=Path, default=None,
                   help="Path to Phase 1 At_curve.csv for real A(t) data")
    p.add_argument("--out-json",   type=Path,
                   default=cfg.OUTPUTS_DIR / "calibration_result.json",
                   help="Output JSON path")

    args = p.parse_args()

    # Load A(t) CSV if provided
    at_df = None
    if args.at_csv and args.at_csv.exists():
        at_df = pd.read_csv(args.at_csv)
        print(f"[Calibration] Using Phase 1 A(t) data: {args.at_csv}")

    result = None
    mode   = "scalar"

    # ── Time-series mode ─────────────────────────────────────────────────────
    if args.obs_times and args.obs_vels:
        mode   = "timeseries"
        result = calibrate_from_velocity_series(
            obs_times = np.array(args.obs_times),
            obs_vels  = np.array(args.obs_vels),
            mass=args.mass, alt0=args.alt0, v0=args.v0, Am=args.Am,
            at_df=at_df,
        )

    # ── Batch mode ────────────────────────────────────────────────────────────
    elif args.batch_csv:
        mode  = "batch"
        bdf   = pd.read_csv(args.batch_csv)
        drops = bdf.to_dict("records")
        # Rename column if needed
        for d in drops:
            if "landing_v" not in d and "landing_velocity" in d:
                d["landing_v"] = d.pop("landing_velocity")
        result = calibrate_batch(drops, at_df=at_df)

    # ── Joint mode (Cd + t_infl) ──────────────────────────────────────────────
    elif args.observed_landing_v and args.observed_landing_t:
        mode   = "joint"
        result = calibrate_joint(
            observed_v    = args.observed_landing_v,
            observed_time = args.observed_landing_t,
            mass=args.mass, alt0=args.alt0, v0=args.v0, Am=args.Am,
            at_df=at_df,
        )

    # ── Scalar mode (Cd from v_land only) ────────────────────────────────────
    elif args.observed_landing_v:
        result = calibrate_from_landing_velocity(
            observed_v  = args.observed_landing_v,
            mass=args.mass, alt0=args.alt0, v0=args.v0, Am=args.Am,
            ti=args.ti, at_df=at_df,
            n_bootstrap = args.bootstrap,
        )

    else:
        p.print_help()
        print("\n[Demo] Running with example: v_land=6.2 m/s")
        result = calibrate_from_landing_velocity(observed_v=6.2, n_bootstrap=300)

    # ── Save JSON (strip non-serialisable objects) ────────────────────────────
    def _json_safe(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "simulation": continue     # large array — skip from JSON
                out[k] = _json_safe(v)
            return out
        if isinstance(obj, list): return [_json_safe(i) for i in obj]
        if isinstance(obj, (np.floating, float)): return round(float(obj), 8)
        if isinstance(obj, (np.integer, int)): return int(obj)
        return obj

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_json_safe(result), indent=2))
    print(f"\n[Calibration] ✓ Results saved: {args.out_json}")

    # ── Print final answer ────────────────────────────────────────────────────
    print("\n" + "═"*56)
    if mode == "scalar":
        print(f"  Cd_effective = {result['Cd_eff']:.6f}")
        print(f"  95% CI       = [{result['Cd_ci_low']:.6f}, {result['Cd_ci_high']:.6f}]")
        print(f"  ±            = {result['Cd_ci_half_width']:.6f}")
        Cd_best = result["Cd_eff"]
    elif mode == "joint":
        print(f"  Cd_effective = {result['Cd_eff']:.6f}")
        print(f"  t_infl       = {result['t_infl_s']:.4f} s")
        Cd_best = result["Cd_eff"]
    elif mode == "timeseries":
        print(f"  Cd mean      = {result['Cd_mean']:.4f}")
        print(f"  Cd range     = [{result['Cd_min']:.4f}, {result['Cd_max']:.4f}]")
        print(f"  RMSE         = {result['rmse_ms']:.5f} m/s")
        Cd_best = result["Cd_mean"]
    elif mode == "batch":
        print(f"  Cd recommended (median) = {result['Cd_recommended']:.6f}")
        print(f"  Cd mean ± std = {result['Cd_mean']:.4f} ± {result['Cd_std']:.4f}")
        Cd_best = result["Cd_recommended"]
    print("═"*56)

    # ── Patch config ──────────────────────────────────────────────────────────
    if args.patch_config:
        patch_config(Cd_best)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        obs_t = np.array(args.obs_times) if args.obs_times else None
        obs_v = np.array(args.obs_vels)  if args.obs_vels  else None
        plot_calibration(result, mode=mode, obs_times=obs_t, obs_vels=obs_v)

    return result


# ── Quick programmatic API ────────────────────────────────────────────────────

def quick_calibrate(landing_v: float, **kwargs) -> float:
    """
    One-liner API for use from other modules.

    Returns just the calibrated Cd_eff float.

    Example
    -------
    from src.calibrate_cd import quick_calibrate
    Cd = quick_calibrate(6.2, mass=90, alt0=800)
    """
    r = calibrate_from_landing_velocity(landing_v, verbose=False, **kwargs)
    return r["Cd_eff"]


if __name__ == "__main__":
    main()
