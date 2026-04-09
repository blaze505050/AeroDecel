"""
bayes_cd.py — Bayesian MCMC Drag Coefficient Identification
=============================================================
Computes the full posterior distribution P(Cd | v_obs, θ) over the drag
coefficient given observed velocity telemetry, using Bayes' theorem:

    P(Cd | data) ∝ P(data | Cd) · P(Cd)

Likelihood model
----------------
    v_sim(t; Cd) from forward ODE
    v_obs(t)     from telemetry / video CV

    log L(Cd) = -½ Σ [(v_sim(tᵢ; Cd) - v_obs(tᵢ))² / σᵢ²] - Σ log σᵢ

where σᵢ is the per-point measurement uncertainty (defaults to 0.5 m/s for
radar altimeter / video-CV derived velocity).

Prior
-----
    Cd ~ LogNormal(μ=log(1.5), σ=0.5)    — weakly informative, correct support Cd>0
    (or Uniform[0.1, 5.0] for non-informative)

Inference backends
------------------
  PRIMARY:  emcee v3  (Affine-Invariant MCMC, Foreman-Mackey 2013)
            pip install emcee   (MIT licence, free)

  FALLBACK: scipy.optimize.minimize (Laplace approximation around MAP)
            Always available, no additional install required.
            Provides MAP estimate + Hessian-based CI instead of full posterior.

Joint parameter inference
--------------------------
The module supports joint Bayesian inference over:
    θ = (Cd, t_infl)              2-parameter model
    θ = (Cd, t_infl, A_inf_frac)  3-parameter model

Outputs
-------
  • Full posterior samples (emcee chains)
  • Marginal posteriors: mean, median, std, 95% HDI
  • Corner plot (with emcee) or 2D posterior contours (scipy fallback)
  • Posterior predictive check: v_sim bands vs v_obs
  • Gelman-Rubin R̂ convergence diagnostic
  • JSON summary for downstream use
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FAST FORWARD MODEL  (reused from calibrate_cd.py — RK4, no scipy)
# ══════════════════════════════════════════════════════════════════════════════

def _logistic_A(t: float, Am: float, ti: float, n: float = 2.0) -> float:
    k  = 5.0 / max(ti, 0.1)
    t0 = ti * 0.6
    return float(Am / (1.0 + np.exp(-k * (t - t0))) ** (1.0 / n))


def _simulate_rk4(
    Cd: float, ti: float = 2.5, Am: float = None,
    mass: float = None, alt0: float = None, v0: float = None,
    t_query: np.ndarray = None,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Fast RK4 simulation. Returns v_sim at t_query timestamps.
    Designed for repeated evaluation inside MCMC (minimal overhead).
    """
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Am   = Am   or cfg.CANOPY_AREA_M2

    t_end = (t_query.max() + 5.0) if t_query is not None else (alt0 / max(v0, 1) + 50)

    ts_out = []; vs_out = []
    v, h, t = float(v0), float(alt0), 0.0

    while h > 0.0 and t <= t_end:
        A    = _logistic_A(t, Am, ti)
        rho  = density(max(0.0, h))
        drag = 0.5 * rho * v**2 * Cd * A

        k1v = cfg.GRAVITY - drag / mass;    k1h = -v
        v2 = v + dt/2*k1v; h2 = h + dt/2*k1h
        A2 = _logistic_A(t+dt/2, Am, ti)
        rho2 = density(max(0.0, h2))
        d2 = 0.5*rho2*max(v2,0)**2*Cd*A2
        k2v = cfg.GRAVITY - d2/mass;        k2h = -max(v2,0)

        v3 = v + dt/2*k2v; h3 = h + dt/2*k2h
        A3 = _logistic_A(t+dt/2, Am, ti)
        rho3 = density(max(0.0, h3))
        d3 = 0.5*rho3*max(v3,0)**2*Cd*A3
        k3v = cfg.GRAVITY - d3/mass;        k3h = -max(v3,0)

        v4 = v + dt*k3v; h4 = h + dt*k3h
        A4 = _logistic_A(t+dt, Am, ti)
        rho4 = density(max(0.0, h4))
        d4 = 0.5*rho4*max(v4,0)**2*Cd*A4
        k4v = cfg.GRAVITY - d4/mass;        k4h = -max(v4,0)

        v = max(0.0, v + dt*(k1v+2*k2v+2*k3v+k4v)/6)
        h = max(0.0, h + dt*(k1h+2*k2h+2*k3h+k4h)/6)
        t += dt

        ts_out.append(t); vs_out.append(v)
        if h <= 0.0: break

    ts_out = np.array(ts_out)
    vs_out = np.array(vs_out)

    if t_query is None:
        return vs_out

    # Interpolate to query times
    if len(ts_out) > 1:
        fn = interp1d(ts_out, vs_out, bounds_error=False,
                      fill_value=(vs_out[0], vs_out[-1]))
        return fn(t_query)
    return np.full(len(t_query), v0)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROBABILISTIC MODEL
# ══════════════════════════════════════════════════════════════════════════════

class BayesianCdModel:
    """
    Probabilistic model for Bayesian Cd identification.
    Supports 1-D (Cd only), 2-D (Cd, t_infl), or 3-D (Cd, t_infl, A_frac) inference.
    """

    PARAM_NAMES_1D = ["Cd"]
    PARAM_NAMES_2D = ["Cd", "t_infl"]
    PARAM_NAMES_3D = ["Cd", "t_infl", "A_frac"]

    def __init__(
        self,
        t_obs:      np.ndarray,    # observation times [s]
        v_obs:      np.ndarray,    # observed velocities [m/s]
        sigma_obs:  float | np.ndarray = 0.5,   # measurement std [m/s]
        mass:       float = None,
        alt0:       float = None,
        v0:         float = None,
        Am_nominal: float = None,   # nominal max canopy area [m²]
        n_params:   int   = 1,      # 1=Cd only, 2=Cd+t_infl, 3=Cd+t_infl+A_frac
        prior:      str   = "lognormal",  # "lognormal" | "uniform"
        Cd_bounds:  tuple = (0.1, 5.0),
        ti_bounds:  tuple = (0.3, 10.0),
        Afrac_bounds: tuple = (0.5, 1.5),
    ):
        self.t_obs    = np.asarray(t_obs, dtype=float)
        self.v_obs    = np.asarray(v_obs, dtype=float)
        self.sigma    = (np.full(len(t_obs), sigma_obs) if np.isscalar(sigma_obs)
                         else np.asarray(sigma_obs))
        self.mass     = mass or cfg.PARACHUTE_MASS
        self.alt0     = alt0 or cfg.INITIAL_ALT
        self.v0       = v0   or cfg.INITIAL_VEL
        self.Am       = Am_nominal or cfg.CANOPY_AREA_M2
        self.n_params = n_params
        self.prior    = prior
        self.bounds   = {
            "Cd":     Cd_bounds,
            "t_infl": ti_bounds,
            "A_frac": Afrac_bounds,
        }

        if n_params == 1: self.param_names = self.PARAM_NAMES_1D
        elif n_params == 2: self.param_names = self.PARAM_NAMES_2D
        else: self.param_names = self.PARAM_NAMES_3D

    def _unpack(self, theta: np.ndarray) -> tuple:
        """Unpack parameter vector into (Cd, ti, Am)."""
        Cd = float(theta[0])
        ti = float(theta[1]) if self.n_params >= 2 else 2.5
        Am = self.Am * float(theta[2]) if self.n_params >= 3 else self.Am
        return Cd, ti, Am

    def log_prior(self, theta: np.ndarray) -> float:
        """Log prior probability."""
        Cd, ti, Am_frac_val = self._unpack(theta) if self.n_params < 3 else (
            float(theta[0]), float(theta[1]), float(theta[2]))

        # Check hard bounds
        if not (self.bounds["Cd"][0] < Cd < self.bounds["Cd"][1]):
            return -np.inf
        if self.n_params >= 2:
            if not (self.bounds["t_infl"][0] < ti < self.bounds["t_infl"][1]):
                return -np.inf
        if self.n_params >= 3:
            af = float(theta[2])
            if not (self.bounds["A_frac"][0] < af < self.bounds["A_frac"][1]):
                return -np.inf

        if self.prior == "lognormal":
            # Cd ~ LogNormal(log(1.5), 0.5) — weakly informative
            lp = -0.5 * ((np.log(Cd) - np.log(1.5)) / 0.5) ** 2 - np.log(Cd * 0.5)
            if self.n_params >= 2:
                lp += -0.5 * ((np.log(ti) - np.log(2.5)) / 0.6) ** 2 - np.log(ti * 0.6)
        else:  # uniform
            lp = 0.0

        return float(lp)

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Log likelihood: Gaussian residuals between v_sim and v_obs."""
        Cd, ti, Am = self._unpack(theta)

        # Guard: Cd must be positive
        if Cd <= 0 or ti <= 0 or Am <= 0:
            return -np.inf

        v_sim = _simulate_rk4(
            Cd=Cd, ti=ti, Am=Am,
            mass=self.mass, alt0=self.alt0, v0=self.v0,
            t_query=self.t_obs, dt=0.15,   # coarser for MCMC speed
        )
        residuals = v_sim - self.v_obs
        log_L = -0.5 * np.sum((residuals / self.sigma) ** 2 + np.log(2 * np.pi * self.sigma**2))
        return float(log_L) if np.isfinite(log_L) else -np.inf

    def log_posterior(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + self.log_likelihood(theta)

    def negative_log_posterior(self, theta: np.ndarray) -> float:
        """For scipy MAP estimation."""
        return -self.log_posterior(theta)

    def map_estimate(self, x0: np.ndarray | None = None) -> dict:
        """
        Maximum a posteriori (MAP) estimate via scipy.optimize.minimize.
        Always fast; gives a starting point for MCMC.
        """
        if x0 is None:
            x0 = np.array([cfg.CD_INITIAL, 2.5, 1.0][:self.n_params])

        bounds_list = [self.bounds["Cd"], self.bounds["t_infl"], self.bounds["A_frac"]]
        bounds_list = bounds_list[:self.n_params]

        result = minimize(
            self.negative_log_posterior, x0,
            method="L-BFGS-B", bounds=bounds_list,
            options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8},
        )

        # Hessian-based CI (Laplace approximation)
        from scipy.optimize import approx_fprime
        H = np.zeros((self.n_params, self.n_params))
        eps = 1e-4 * np.abs(result.x) + 1e-6
        for i in range(self.n_params):
            def f_i(theta, i=i):
                ep = np.zeros(self.n_params); ep[i] = eps[i]
                g1 = approx_fprime(theta + ep, self.negative_log_posterior, eps)
                g0 = approx_fprime(theta - ep, self.negative_log_posterior, eps)
                return (g1 - g0) / (2 * eps[i])
            H[i, :] = f_i(result.x)

        try:
            cov = np.linalg.inv(H)
            std = np.sqrt(np.maximum(np.diag(cov), 0))
        except np.linalg.LinAlgError:
            std = np.abs(result.x) * 0.1   # fallback

        ci95 = 1.96 * std

        return {
            "map":    dict(zip(self.param_names, result.x.tolist())),
            "std":    dict(zip(self.param_names, std.tolist())),
            "ci95":   dict(zip(self.param_names, ci95.tolist())),
            "log_posterior": float(-result.fun),
            "success": result.success,
            "message": result.message,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MCMC SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

class MCMCSampler:
    """
    Affine-invariant MCMC using emcee (Foreman-Mackey et al. 2013).
    Falls back to Laplace approximation if emcee is not installed.
    """

    def __init__(self, model: BayesianCdModel):
        self.model  = model
        self.chains = None   # shape (n_steps, n_walkers, n_params)
        self.flat_samples = None
        self._backend = "laplace"

    def _check_emcee(self) -> bool:
        try:
            import emcee  # noqa
            return True
        except ImportError:
            return False

    def run(
        self,
        n_walkers:    int   = 32,
        n_steps:      int   = 3000,
        n_burnin:     int   = 1000,
        init_jitter:  float = 0.05,
        progress:     bool  = True,
        verbose:      bool  = True,
    ) -> dict:
        """
        Run MCMC sampling. Returns posterior statistics dict.
        Uses emcee if available, Laplace approximation otherwise.
        """
        # ── MAP estimate as starting point ────────────────────────────────────
        if verbose: print("  Computing MAP estimate...")
        t0 = time.perf_counter()
        map_result = self.model.map_estimate()
        if verbose:
            for k, v in map_result["map"].items():
                print(f"    MAP {k} = {v:.5f} ± {map_result['std'].get(k,0):.5f}")

        map_theta = np.array(list(map_result["map"].values()))

        # ── emcee MCMC ────────────────────────────────────────────────────────
        if self._check_emcee():
            import emcee
            self._backend = "emcee"

            if verbose:
                print(f"\n  Running emcee MCMC: {n_walkers} walkers × {n_steps} steps")
                print(f"  Burn-in: {n_burnin} steps  |  Production: {n_steps-n_burnin} steps")

            # Initialise walkers in a ball around MAP
            ndim     = self.model.n_params
            p0_scale = np.maximum(np.abs(map_theta) * init_jitter, 0.02)
            p0       = map_theta[None, :] + p0_scale[None, :] * np.random.randn(n_walkers, ndim)
            # Ensure within bounds
            for i, pn in enumerate(self.model.param_names):
                lo, hi = self.model.bounds[pn]
                p0[:, i] = np.clip(p0[:, i], lo + 1e-4, hi - 1e-4)

            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, self.model.log_posterior,
                vectorize=False,
            )

            # Burn-in
            state = sampler.run_mcmc(p0, n_burnin, progress=progress and verbose,
                                      skip_initial_state_check=True)
            sampler.reset()
            # Production
            sampler.run_mcmc(state, n_steps - n_burnin, progress=progress and verbose,
                              skip_initial_state_check=True)

            self.chains = sampler.get_chain()                        # (n_prod, n_walkers, ndim)
            self.flat_samples = sampler.get_chain(flat=True, discard=0)   # (n_prod*n_walkers, ndim)

            # Gelman-Rubin R̂ convergence diagnostic
            try:
                tau = sampler.get_autocorr_time(quiet=True)
                r_hat = self._gelman_rubin(self.chains)
                converged = bool(np.all(r_hat < 1.05))
                if verbose:
                    print(f"\n  Autocorr time: {np.round(tau,1)}")
                    print(f"  Gelman-Rubin R̂: {np.round(r_hat,4)}  "
                          f"{'converged ✓' if converged else 'not converged — increase n_steps'}")
            except Exception:
                r_hat = np.array([np.nan])
                converged = None

        else:
            # ── Laplace fallback ──────────────────────────────────────────────
            if verbose:
                print("\n  [emcee not installed] Using Laplace approximation.")
                print("  Install: pip install emcee corner")
            self._backend = "laplace"
            # Simulate posterior samples from Laplace covariance
            n_samples = n_walkers * (n_steps - n_burnin)
            std_arr = np.array([map_result["std"].get(pn, 0.01)
                                 for pn in self.model.param_names])
            std_arr = np.maximum(std_arr, 1e-4)
            # Draw from multivariate normal, clip to bounds
            raw = np.random.randn(n_samples, self.model.n_params)
            self.flat_samples = map_theta[None, :] + raw * std_arr[None, :]
            for i, pn in enumerate(self.model.param_names):
                lo, hi = self.model.bounds[pn]
                self.flat_samples[:, i] = np.clip(self.flat_samples[:, i], lo, hi)
            self.chains = self.flat_samples.reshape(-1, 4, self.model.n_params)
            r_hat = np.array([np.nan])
            converged = None

        elapsed = time.perf_counter() - t0

        # ── Posterior statistics ───────────────────────────────────────────────
        stats = self._compute_stats(self.flat_samples, r_hat, converged, elapsed)
        stats["map"] = map_result
        stats["backend"] = self._backend

        if verbose:
            self._print_stats(stats)

        return stats

    def _gelman_rubin(self, chains: np.ndarray) -> np.ndarray:
        """
        Gelman-Rubin R̂ statistic.
        chains: (n_steps, n_walkers, n_params)
        """
        n, m, d = chains.shape
        chain_means = chains.mean(axis=0)              # (m, d)
        grand_mean  = chain_means.mean(axis=0)          # (d,)
        B = n / (m - 1) * np.sum((chain_means - grand_mean)**2, axis=0)
        W = np.mean(np.var(chains, axis=0, ddof=1), axis=0)
        V_hat = (n - 1) / n * W + B / n
        return np.sqrt(V_hat / np.maximum(W, 1e-12))

    def _compute_stats(self, samples: np.ndarray, r_hat, converged, elapsed) -> dict:
        stats = {
            "n_samples":   len(samples),
            "elapsed_s":   round(elapsed, 2),
            "r_hat":       r_hat.tolist() if hasattr(r_hat, "tolist") else [r_hat],
            "converged":   converged,
            "parameters":  {},
        }
        for i, pn in enumerate(self.model.param_names):
            s = samples[:, i]
            lo95, hi95 = np.percentile(s, [2.5, 97.5])
            lo68, hi68 = np.percentile(s, [16, 84])
            stats["parameters"][pn] = {
                "mean":    round(float(s.mean()), 6),
                "median":  round(float(np.median(s)), 6),
                "std":     round(float(s.std()), 6),
                "ci95_lo": round(float(lo95), 6),
                "ci95_hi": round(float(hi95), 6),
                "ci68_lo": round(float(lo68), 6),
                "ci68_hi": round(float(hi68), 6),
                "mode_kde": round(float(self._kde_mode(s)), 6),
            }
        return stats

    def _kde_mode(self, samples: np.ndarray) -> float:
        """Kernel density mode estimate."""
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(samples)
            x   = np.linspace(samples.min(), samples.max(), 500)
            return float(x[np.argmax(kde(x))])
        except Exception:
            return float(np.median(samples))

    def _print_stats(self, stats: dict):
        print(f"\n  {'─'*56}")
        print(f"  BAYESIAN POSTERIOR  ({stats['backend'].upper()}, "
              f"n={stats['n_samples']}, t={stats['elapsed_s']:.1f}s)")
        print(f"  {'─'*56}")
        for pn, ps in stats["parameters"].items():
            print(f"  {pn:12s}: {ps['mean']:.5f} ± {ps['std']:.5f}  "
                  f"95% HDI [{ps['ci95_lo']:.4f}, {ps['ci95_hi']:.4f}]  "
                  f"mode={ps['mode_kde']:.5f}")
        r_hat = stats.get("r_hat", [float("nan")])
        if not np.isnan(r_hat[0]):
            print(f"  R̂: {[round(r,4) for r in r_hat]}  "
                  f"{'✓' if stats.get('converged') else '?'}")
        print(f"  {'─'*56}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  POSTERIOR PREDICTIVE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def posterior_predictive(
    sampler:    MCMCSampler,
    t_eval:     np.ndarray,
    n_draws:    int = 200,
) -> dict:
    """
    Draw n_draws samples from the posterior, simulate v(t) for each, and
    compute prediction bands: mean, 68%, 95% credible intervals.
    """
    model   = sampler.model
    samples = sampler.flat_samples
    idx     = np.random.choice(len(samples), size=min(n_draws, len(samples)),
                                replace=False)
    draws   = samples[idx]

    v_curves = []
    for theta in draws:
        Cd, ti, Am = model._unpack(theta)
        v = _simulate_rk4(Cd=Cd, ti=ti, Am=Am, mass=model.mass,
                           alt0=model.alt0, v0=model.v0, t_query=t_eval)
        v_curves.append(v)

    v_mat = np.array(v_curves)   # (n_draws, n_eval)
    return {
        "t":        t_eval.tolist(),
        "v_mean":   v_mat.mean(axis=0).tolist(),
        "v_median": np.median(v_mat, axis=0).tolist(),
        "v_p05":    np.percentile(v_mat, 5,  axis=0).tolist(),
        "v_p95":    np.percentile(v_mat, 95, axis=0).tolist(),
        "v_p16":    np.percentile(v_mat, 16, axis=0).tolist(),
        "v_p84":    np.percentile(v_mat, 84, axis=0).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_posterior(
    sampler:   MCMCSampler,
    stats:     dict,
    ppc:       dict,
    t_obs:     np.ndarray = None,
    v_obs:     np.ndarray = None,
    save_path: Path | None = None,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import gaussian_kde

    if cfg.DARK_THEME:
        matplotlib.rcParams.update({
            "figure.facecolor": "#080c14", "axes.facecolor": "#0d1526",
            "axes.edgecolor": "#2a3d6e",   "text.color": "#c8d8f0",
            "axes.labelcolor": "#c8d8f0",  "xtick.color": "#c8d8f0",
            "ytick.color": "#c8d8f0",      "grid.color": "#1a2744",
        })
    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})

    TEXT = "#c8d8f0" if cfg.DARK_THEME else "#111"
    C1   = cfg.COLOR_THEORY
    C2   = cfg.COLOR_PINN
    C3   = cfg.COLOR_RAW

    samples  = sampler.flat_samples
    n_params = sampler.model.n_params
    pnames   = sampler.model.param_names

    # Determine layout
    n_rows = 2 if n_params <= 2 else 3
    n_cols = max(3, n_params + 1)
    fig = plt.figure(figsize=(max(16, 5*n_cols), 5*n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.48, wspace=0.38,
                            top=0.91, bottom=0.07, left=0.06, right=0.97)

    def ax(r, c, **kw): return fig.add_subplot(gs[r, c], **kw)

    # ── Marginal posteriors ───────────────────────────────────────────────────
    for i, pn in enumerate(pnames):
        a = ax(0, i)
        s = samples[:, i]
        a.hist(s, bins=60, color=C1, alpha=0.7, edgecolor="none", density=True)
        x = np.linspace(s.min(), s.max(), 300)
        try:
            kde = gaussian_kde(s)
            a.plot(x, kde(x), color=C3, lw=1.8)
        except Exception:
            pass
        ps = stats["parameters"][pn]
        a.axvline(ps["mean"],    color=C1, lw=1.5, label=f"mean={ps['mean']:.4f}")
        a.axvline(ps["median"],  color=C2, lw=1.0, ls="--", label=f"median={ps['median']:.4f}")
        a.axvline(ps["ci95_lo"], color=TEXT, lw=0.7, ls=":", alpha=0.6)
        a.axvline(ps["ci95_hi"], color=TEXT, lw=0.7, ls=":", alpha=0.6)
        a.fill_betweenx([0, a.get_ylim()[1] if a.get_ylim()[1]>0 else 1],
                        ps["ci95_lo"], ps["ci95_hi"], alpha=0.12, color=C1)
        a.set_xlabel(pn); a.set_ylabel("Density")
        a.set_title(f"P({pn} | data)", fontweight="bold")
        a.legend(fontsize=7.5); a.grid(True, alpha=0.3)
        a.spines[["top","right"]].set_visible(False)

    # ── 2D joint posterior (if ≥2 params) ────────────────────────────────────
    if n_params >= 2:
        a2d = ax(0, n_params)
        try:
            from scipy.stats import gaussian_kde as gkde
            x2, y2 = samples[:, 0], samples[:, 1]
            xy = np.vstack([x2, y2])
            z  = gkde(xy)(xy)
            idx = z.argsort()
            a2d.scatter(x2[idx], y2[idx], c=z[idx], cmap="plasma", s=1.5, alpha=0.6)
            a2d.set_xlabel(pnames[0]); a2d.set_ylabel(pnames[1])
        except Exception:
            a2d.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.4, color=C1)
        a2d.set_title("Joint posterior", fontweight="bold")
        a2d.grid(True, alpha=0.3)

    # ── MCMC chains ──────────────────────────────────────────────────────────
    for i, pn in enumerate(pnames):
        a_ch = ax(1, i)
        if sampler.chains is not None and sampler._backend == "emcee":
            chain_data = sampler.chains[:, :, i]
            n_show = min(8, chain_data.shape[1])
            for w in range(n_show):
                a_ch.plot(chain_data[:, w], lw=0.4, alpha=0.6, color=C1)
            a_ch.plot(chain_data.mean(axis=1), color=C3, lw=1.5)
        else:
            a_ch.plot(samples[:, i], color=C1, lw=0.4, alpha=0.5)
        a_ch.set_xlabel("Step"); a_ch.set_ylabel(pn)
        a_ch.set_title(f"MCMC chain: {pn}", fontweight="bold")
        a_ch.grid(True, alpha=0.3)

    # ── Posterior predictive check ────────────────────────────────────────────
    a_ppc = ax(1, n_params)
    t_ppc = np.array(ppc["t"])
    a_ppc.fill_between(t_ppc, ppc["v_p05"], ppc["v_p95"], alpha=0.15, color=C1, label="95% CI")
    a_ppc.fill_between(t_ppc, ppc["v_p16"], ppc["v_p84"], alpha=0.25, color=C1, label="68% CI")
    a_ppc.plot(t_ppc, ppc["v_mean"], color=C1, lw=1.8, label="Posterior mean")
    if t_obs is not None and v_obs is not None:
        a_ppc.scatter(t_obs, v_obs, color=C3, s=20, zorder=5, label="Observed v")
    a_ppc.set_xlabel("Time [s]"); a_ppc.set_ylabel("Velocity [m/s]")
    a_ppc.set_title("Posterior predictive check", fontweight="bold")
    a_ppc.legend(fontsize=7.5); a_ppc.grid(True, alpha=0.3)

    # ── Summary stats table ───────────────────────────────────────────────────
    if n_rows >= 2 and n_cols > n_params + 1:
        a_tbl = ax(1, n_params + 1) if n_cols > n_params + 1 else None
    else:
        a_tbl = None

    backend_label = stats.get("backend", "?").upper()
    fig.text(0.5, 0.955,
             f"Bayesian Cd Identification  —  Backend: {backend_label}  "
             f"n={stats['n_samples']}  "
             f"Cd={stats['parameters']['Cd']['mean']:.4f}"
             f"±{stats['parameters']['Cd']['std']:.4f}  "
             f"95%HDI[{stats['parameters']['Cd']['ci95_lo']:.4f},"
             f"{stats['parameters']['Cd']['ci95_hi']:.4f}]",
             ha="center", fontsize=11, fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR / "bayes_cd_posterior.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Posterior plot saved: {sp}")

    # ── Corner plot (emcee only, if n_params ≥ 2) ────────────────────────────
    if sampler._backend == "emcee" and n_params >= 2:
        try:
            import corner
            cf = corner.corner(
                samples, labels=pnames,
                quantiles=[0.025, 0.5, 0.975],
                show_titles=True, title_kwargs={"fontsize": 9},
                color=C1 if not cfg.DARK_THEME else "#85B7EB",
            )
            csp = save_path.parent / "bayes_cd_corner.png" if save_path else \
                  cfg.OUTPUTS_DIR / "bayes_cd_corner.png"
            cf.savefig(csp, dpi=cfg.DPI)
            print(f"  ✓ Corner plot saved: {csp}")
        except ImportError:
            print("  [info] Install 'corner' for triangle plots: pip install corner")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SYNTHETIC DATA GENERATOR (for testing without telemetry)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_telemetry(
    true_Cd:    float = 1.32,
    true_ti:    float = 2.5,
    sigma_obs:  float = 0.4,
    n_obs:      int   = 40,
    t_end:      float = 60.0,
    seed:       int   = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic v(t) observations by simulating with known Cd, then
    adding Gaussian noise. Use to validate recovery of true parameters.
    """
    rng = np.random.default_rng(seed)
    t_dense = np.linspace(0, t_end, 600)
    v_true  = _simulate_rk4(Cd=true_Cd, ti=true_ti, t_query=t_dense)
    # Sparse observations with noise
    idx    = np.sort(rng.choice(len(t_dense), n_obs, replace=False))
    t_obs  = t_dense[idx]
    v_obs  = v_true[idx] + rng.normal(0, sigma_obs, n_obs)
    v_obs  = np.clip(v_obs, 0, None)
    return t_obs, v_obs


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(
    t_obs:      np.ndarray | None = None,
    v_obs:      np.ndarray | None = None,
    sigma_obs:  float = 0.5,
    true_Cd:    float = 1.32,      # used only for synthetic test
    n_params:   int   = 1,
    n_walkers:  int   = 32,
    n_steps:    int   = 2000,
    n_burnin:   int   = 500,
    prior:      str   = "lognormal",
    verbose:    bool  = True,
) -> dict:
    """Run full Bayesian inference pipeline."""
    import matplotlib; matplotlib.use("Agg")

    # ── Synthetic data if none provided ───────────────────────────────────────
    if t_obs is None or v_obs is None:
        if verbose:
            print(f"[Bayes] No telemetry provided — using synthetic data (true Cd={true_Cd})")
        t_obs, v_obs = generate_synthetic_telemetry(true_Cd=true_Cd)

    if verbose:
        print(f"[Bayes] {len(t_obs)} observations  t=[{t_obs.min():.1f},{t_obs.max():.1f}]s  "
              f"v=[{v_obs.min():.2f},{v_obs.max():.2f}]m/s")

    # ── Model + MCMC ──────────────────────────────────────────────────────────
    model   = BayesianCdModel(t_obs=t_obs, v_obs=v_obs, sigma_obs=sigma_obs,
                               n_params=n_params, prior=prior)
    sampler = MCMCSampler(model)
    stats   = sampler.run(n_walkers=n_walkers, n_steps=n_steps,
                           n_burnin=n_burnin, verbose=verbose)

    # ── Posterior predictive ──────────────────────────────────────────────────
    t_eval = np.linspace(0, t_obs.max() + 5, 200)
    ppc    = posterior_predictive(sampler, t_eval, n_draws=200)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_posterior(sampler, stats, ppc, t_obs, v_obs)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    def _safe(obj):
        if isinstance(obj, (float, np.floating)): return round(float(obj), 8)
        if isinstance(obj, (int, np.integer)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, list): return [_safe(i) for i in obj]
        if isinstance(obj, dict): return {k: _safe(v) for k, v in obj.items()}
        return obj

    out = cfg.OUTPUTS_DIR / "bayes_cd_stats.json"
    out.write_text(json.dumps(_safe(stats), indent=2))
    if verbose: print(f"  ✓ Stats saved: {out}")

    # ── Print recommended Cd ──────────────────────────────────────────────────
    Cd_post = stats["parameters"]["Cd"]
    print(f"\n  RECOMMENDED: Cd = {Cd_post['mean']:.5f}  "
          f"95% HDI [{Cd_post['ci95_lo']:.5f}, {Cd_post['ci95_hi']:.5f}]")
    print(f"  (vs MAP = {stats['map']['map']['Cd']:.5f})")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bayesian MCMC Cd Identification",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--telemetry-csv", type=Path, default=None,
                   help="CSV with columns: time_s, velocity_ms [,sigma]")
    p.add_argument("--true-Cd",    type=float, default=1.32, help="True Cd (synthetic test)")
    p.add_argument("--sigma-obs",  type=float, default=0.5)
    p.add_argument("--n-params",   type=int,   default=1,  choices=[1,2,3])
    p.add_argument("--n-walkers",  type=int,   default=32)
    p.add_argument("--n-steps",    type=int,   default=2000)
    p.add_argument("--n-burnin",   type=int,   default=500)
    p.add_argument("--prior",      type=str,   default="lognormal",
                   choices=["lognormal","uniform"])
    a = p.parse_args()

    t_obs = v_obs = None
    if a.telemetry_csv and a.telemetry_csv.exists():
        df = pd.read_csv(a.telemetry_csv)
        t_obs = df["time_s"].values
        v_obs = df["velocity_ms"].values

    run(t_obs=t_obs, v_obs=v_obs, true_Cd=a.true_Cd, sigma_obs=a.sigma_obs,
        n_params=a.n_params, n_walkers=a.n_walkers, n_steps=a.n_steps,
        n_burnin=a.n_burnin, prior=a.prior)
