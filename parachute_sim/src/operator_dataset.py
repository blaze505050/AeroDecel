"""
operator_dataset.py — Parametric Sweep Dataset Generator (AeroDecel v6.0)
==========================================================================
Generates thousands of ODE solutions across a Latin Hypercube-sampled
parameter space to train neural operators (FNO / DeepONet).

Each sample is a complete parachute descent trajectory:
  Input:  (mass, Cd, canopy_area, initial_alt, initial_vel, air_density)
  Output: v(t), h(t), Cd(t) as time series on a uniform grid

This lets the neural operator learn the *solution map* of the parachute ODE
system, not just a single solution.

Dataset format:
  inputs:  (N_samples, n_params)       — parameter vectors
  outputs: (N_samples, n_timesteps, 3) — [v, h, Cd] time series
  t_grid:  (n_timesteps,)              — common time grid
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path
import sys, time

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LATIN HYPERCUBE SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def latin_hypercube(n_samples: int, n_params: int,
                    bounds: list[tuple], seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube samples in the parameter space.

    This provides better coverage of the parameter space than random sampling,
    critical for training neural operators on high-dimensional spaces.

    Parameters
    ----------
    n_samples : number of sample points
    n_params  : number of parameters
    bounds    : list of (low, high) tuples for each parameter

    Returns
    -------
    samples : (n_samples, n_params) array
    """
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_params))

    for j in range(n_params):
        # Create evenly-spaced strata
        low, high = bounds[j]
        cut = np.linspace(low, high, n_samples + 1)
        # Sample uniformly within each stratum
        for i in range(n_samples):
            result[i, j] = rng.uniform(cut[i], cut[i + 1])
        # Shuffle to remove correlation between parameters
        rng.shuffle(result[:, j])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARACHUTE ODE SOLVER (lightweight, fast)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_parachute_ode(
    mass:        float,
    Cd:          float,
    area:        float,
    alt0:        float,
    vel0:        float,
    rho0:        float,
    gravity:     float = 9.80665,
    t_max:       float = 60.0,
    n_points:    int   = 200,
    inflation_t: float = 2.5,
) -> tuple:
    """
    Solve the parachute descent ODE for one parameter set.

    State: [v, h]
      dv/dt = g - (ρ·v²·Cd(t)·A(t)) / (2m)
      dh/dt = -v

    Returns (t_grid, v_array, h_array, Cd_array) on a uniform time grid.
    """
    from src.atmosphere import density

    # Area inflation model
    A_max = area
    ti = inflation_t
    k = 5.0 / ti
    t0_infl = ti * 0.6

    def A_t(t):
        return A_max / (1 + np.exp(-k * (t - t0_infl))) ** 0.5

    def Cd_t(t):
        # Cd with opening transient
        return Cd + Cd * 0.38 * np.exp(-0.5 * ((t - ti) / (ti / 3))**2)

    def rhs(t, state):
        v, h = state
        v = max(v, 0.0)
        h = max(h, 0.0)

        rho = density(h) if h > 0 else rho0
        A = A_t(t)
        cd = Cd_t(t)

        drag = 0.5 * rho * v**2 * cd * A / mass
        dv = gravity - drag
        dh = -v

        return [dv, dh]

    def ground(t, y):
        return y[1]
    ground.terminal = True
    ground.direction = -1

    t_grid_req = np.linspace(0, t_max, n_points)
    sol = solve_ivp(rhs, (0, t_max), [vel0, alt0],
                    method="RK45", t_eval=t_grid_req,
                    events=ground, rtol=1e-6, atol=1e-8)

    # Pad to uniform length if ground was hit early
    t_out = np.linspace(0, t_max, n_points)
    v_out = np.interp(t_out, sol.t, sol.y[0], right=sol.y[0][-1])
    h_out = np.interp(t_out, sol.t, sol.y[1], right=0.0)
    cd_out = np.array([Cd_t(t) for t in t_out])

    v_out = np.clip(v_out, 0, None)
    h_out = np.clip(h_out, 0, None)

    return t_out, v_out, h_out, cd_out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Parameter bounds for Latin Hypercube sampling
DEFAULT_BOUNDS = [
    (1.0,    200.0),    # mass [kg]
    (0.3,      2.5),    # Cd [-]
    (1.0,    200.0),    # canopy area [m²]
    (100.0, 5000.0),    # initial altitude [m]
    (5.0,    150.0),    # initial velocity [m/s]
    (0.4,      1.3),    # surface air density [kg/m³]
]

PARAM_NAMES = ["mass", "Cd", "area", "alt0", "vel0", "rho0"]


def generate_dataset(
    n_samples:    int   = 2000,
    n_timesteps:  int   = 200,
    t_max:        float = 60.0,
    bounds:       list  = None,
    seed:         int   = 42,
    verbose:      bool  = True,
    save_path:    Path  = None,
) -> dict:
    """
    Generate a full training dataset for neural operator training.

    Returns dict with:
      'inputs'   : (n_samples, 6) parameter vectors
      'outputs'  : (n_samples, n_timesteps, 3) — [v, h, Cd] time series
      't_grid'   : (n_timesteps,) common time grid
      'params'   : parameter names
      'bounds'   : parameter bounds used
    """
    bounds = bounds or DEFAULT_BOUNDS
    n_params = len(bounds)

    if verbose:
        print(f"\n[Dataset] Generating {n_samples} parametric ODE solutions...")
        print(f"  Timesteps: {n_timesteps}  |  t_max: {t_max}s  |  Params: {n_params}")

    # Latin Hypercube Sampling
    inputs = latin_hypercube(n_samples, n_params, bounds, seed=seed)

    # Common time grid
    t_grid = np.linspace(0, t_max, n_timesteps)

    # Solve all
    outputs = np.zeros((n_samples, n_timesteps, 3))  # [v, h, Cd]
    t0 = time.perf_counter()
    n_failed = 0

    for i in range(n_samples):
        mass, Cd, area, alt0, vel0, rho0 = inputs[i]
        try:
            t_out, v, h, cd = solve_parachute_ode(
                mass=mass, Cd=Cd, area=area,
                alt0=alt0, vel0=vel0, rho0=rho0,
                t_max=t_max, n_points=n_timesteps,
            )
            outputs[i, :, 0] = v
            outputs[i, :, 1] = h
            outputs[i, :, 2] = cd
        except Exception:
            n_failed += 1
            # Fill with simple exponential decay as fallback
            v_term = np.sqrt(2 * mass * 9.81 / (rho0 * Cd * area))
            outputs[i, :, 0] = v_term * np.tanh(9.81 * t_grid / max(v_term, 0.1))
            outputs[i, :, 1] = np.clip(alt0 - vel0 * t_grid, 0, None)
            outputs[i, :, 2] = Cd

        if verbose and (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / max(rate, 0.01)
            print(f"\r  [{i+1:>5}/{n_samples}] {rate:.0f} samples/s  ETA: {eta:.0f}s", end="", flush=True)

    elapsed = time.perf_counter() - t0

    # Normalize inputs to [0, 1]
    bounds_arr = np.array(bounds)
    inputs_norm = (inputs - bounds_arr[:, 0]) / (bounds_arr[:, 1] - bounds_arr[:, 0])

    # Normalize outputs
    v_max = outputs[:, :, 0].max() + 1e-6
    h_max = outputs[:, :, 1].max() + 1e-6
    cd_max = outputs[:, :, 2].max() + 1e-6

    outputs_norm = outputs.copy()
    outputs_norm[:, :, 0] /= v_max
    outputs_norm[:, :, 1] /= h_max
    outputs_norm[:, :, 2] /= cd_max

    dataset = {
        "inputs":       inputs,
        "inputs_norm":  inputs_norm,
        "outputs":      outputs,
        "outputs_norm": outputs_norm,
        "t_grid":       t_grid,
        "params":       PARAM_NAMES,
        "bounds":       bounds,
        "norm_scales":  {"v": v_max, "h": h_max, "cd": cd_max},
        "n_samples":    n_samples,
        "n_failed":     n_failed,
    }

    if verbose:
        print(f"\r  Generated {n_samples} samples in {elapsed:.1f}s "
              f"({n_samples/elapsed:.0f} samples/s)  "
              f"Failed: {n_failed}     ")

    # Optionally save
    if save_path is not None:
        np.savez_compressed(
            save_path,
            inputs=inputs, inputs_norm=inputs_norm,
            outputs=outputs, outputs_norm=outputs_norm,
            t_grid=t_grid,
            v_max=v_max, h_max=h_max, cd_max=cd_max,
        )
        if verbose:
            size_mb = save_path.stat().st_size / 1e6
            print(f"  ✓ Dataset saved: {save_path} ({size_mb:.1f} MB)")

    return dataset


def load_dataset(path: Path) -> dict:
    """Load a previously generated dataset."""
    data = np.load(path)
    return {
        "inputs":       data["inputs"],
        "inputs_norm":  data["inputs_norm"],
        "outputs":      data["outputs"],
        "outputs_norm": data["outputs_norm"],
        "t_grid":       data["t_grid"],
        "norm_scales":  {"v": float(data["v_max"]),
                         "h": float(data["h_max"]),
                         "cd": float(data["cd_max"])},
        "params":       PARAM_NAMES,
        "bounds":       DEFAULT_BOUNDS,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    save_dir = cfg.OUTPUTS_DIR
    save_dir.mkdir(exist_ok=True)
    dataset = generate_dataset(
        n_samples=2000,
        save_path=save_dir / "operator_dataset.npz",
    )
