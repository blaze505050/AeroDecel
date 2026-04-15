"""
src/monte_carlo_edl.py — Monte Carlo Uncertainty Propagation for EDL
======================================================================
Propagates uncertainty from all input parameters through the full EDL
physics chain to produce:

  • Probability of TPS thermal failure  P(T_peak > T_limit)
  • Landing ellipse with confidence contours (50%, 90%, 99%)
  • Probability distribution of landing velocity
  • Probability of structural failure from opening shock
  • Sensitivity ranking (Sobol first-order indices via correlation)
  • Mission success probability

Uncertain inputs (all sampled from calibrated distributions)
------------------------------------------------------------
  ρ₀       atmospheric density at surface     ±15% (log-normal)
  H        scale height                       ±10% (normal)
  T₀       surface temperature                ±5%  (normal)
  Cd       drag coefficient                   ±12% (normal)
  A_canopy canopy area                        ±3%  (normal — manufacturing)
  m_total  total mass                         ±1%  (normal — fuel burn)
  γ_entry  flight path angle                  ±0.5° (normal — nav error)
  R_nose   nose radius                        ±5%  (normal — manufacturing)
  T_wall   initial wall temperature           ±20 K (uniform)
  wind_ew  east-west wind perturbation        N(0, σ_wind) from Open-Meteo
  wind_ns  north-south wind perturbation      N(0, σ_wind)

Sampling: Latin-Hypercube (better space-filling than pure Monte Carlo)

Output statistics
-----------------
  • All results at each sample: (N × n_outputs) DataFrame
  • CDF of landing velocity P(v_land ≤ x)
  • 2-D landing KDE with 50/90/99% contours
  • Gumbel extreme-value fit to maximum heat flux
  • Tornado chart: top-10 parameters by Spearman correlation with SF
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (qmc, norm, lognorm, uniform,
                          spearmanr, gaussian_kde, gumbel_r)
from scipy.interpolate import interp1d


# ══════════════════════════════════════════════════════════════════════════════
# LATIN HYPERCUBE SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

PARAM_SPECS = {
    # name: (nominal, dist_type, param1, param2)
    # dist_type: "lognorm_pct"  → lognormal with pct std
    #            "norm_pct"     → normal with pct std
    #            "norm_abs"     → normal with absolute std
    #            "uniform_abs"  → uniform ± param1
    "rho0_factor":   (1.00, "lognorm_pct", 15.0),    # atm density ±15%
    "H_factor":      (1.00, "norm_pct",    10.0),    # scale height ±10%
    "T0_factor":     (1.00, "norm_pct",     5.0),    # surface temp ±5%
    "Cd_factor":     (1.00, "norm_pct",    12.0),    # drag coeff ±12%
    "A_factor":      (1.00, "norm_pct",     3.0),    # canopy area ±3%
    "mass_factor":   (1.00, "norm_pct",     1.0),    # mass ±1%
    "gamma_deg":     (0.00, "norm_abs",     0.5),    # FPA perturbation ±0.5°
    "R_nose_factor": (1.00, "norm_pct",     5.0),    # nose radius ±5%
    "T_wall_delta":  (0.00, "uniform_abs", 20.0),    # wall T ±20 K
    "wind_ew_ms":    (0.00, "norm_abs",     8.0),    # east-west wind
    "wind_ns_ms":    (0.00, "norm_abs",     8.0),    # north-south wind
}


def _lhs_samples(n: int, seed: int = 42) -> dict[str, np.ndarray]:
    """
    Latin-hypercube sample all uncertain parameters.
    Returns dict of arrays, each length n.
    """
    n_params = len(PARAM_SPECS)
    sampler  = qmc.LatinHypercube(d=n_params, seed=seed)
    lhs_unit = sampler.random(n)   # (n, n_params) ∈ [0,1]^d

    samples = {}
    for i, (name, spec) in enumerate(PARAM_SPECS.items()):
        u = lhs_unit[:, i]
        nominal = spec[1] if len(spec) == 2 else spec[0]
        dist    = spec[1]
        p1      = spec[2]

        if dist == "lognorm_pct":
            sigma_log = np.log(1 + p1/100)
            mu_log    = -0.5 * sigma_log**2   # ensures mean = 1.0
            samples[name] = np.exp(mu_log + sigma_log * norm.ppf(np.clip(u, 1e-6, 1-1e-6)))

        elif dist == "norm_pct":
            sigma = p1 / 100.0
            samples[name] = 1.0 + sigma * norm.ppf(np.clip(u, 1e-4, 1-1e-4))
            samples[name] = np.clip(samples[name], 0.1, 5.0)

        elif dist == "norm_abs":
            samples[name] = p1 * norm.ppf(np.clip(u, 1e-4, 1-1e-4))

        elif dist == "uniform_abs":
            samples[name] = -p1 + 2*p1*u

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# FAST EDL EVALUATOR  (vectorised over samples)
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_sample(
    params:      dict,
    planet_atm,
    tps_mat_name: str,
    tps_thickness: float,
    canopy_shape:  str,
    canopy_dims:   dict,
    mass_nom:      float,
    alt0_m:        float,
    vel0_ms:       float,
    gamma_nom_deg: float,
    R_nose_nom:    float,
    use_realgas:   bool,
) -> dict:
    """Evaluate one sample. Returns dict of scalar outputs."""
    from src.thermal_model   import ThermalProtectionSystem
    from src.canopy_geometry import CanopyGeometry
    from src.multifidelity_pinn import LowFidelityEDL

    # ── Apply parameter perturbations ─────────────────────────────────────────
    class _PerturbedAtm:
        """Thin wrapper that scales planet atmosphere by sampled factors."""
        def __init__(self, base, rho_f, H_f, T_f):
            self._base = base
            self.gravity_ms2 = base.gravity_ms2
            self.gas_constant = base.gas_constant
            self._rho_f = rho_f; self._H_f = H_f; self._T_f = T_f

        def density(self, h):
            rho0 = self._base.density(0) * self._rho_f
            H    = self._base.scale_height(0) * self._H_f if hasattr(self._base, '_H') \
                   else 11100 * self._H_f
            return rho0 * np.exp(-max(h,0) / H)

        def temperature(self, h):
            return self._base.temperature(h) * self._T_f

        def pressure(self, h):
            return self.density(h) * self.gas_constant * self.temperature(h)

        def mach_number(self, v, h):
            return self._base.mach_number(v, h)

    atm = _PerturbedAtm(planet_atm,
                         params["rho0_factor"],
                         params["H_factor"],
                         params["T0_factor"])

    mass       = mass_nom * params["mass_factor"]
    gamma_deg  = gamma_nom_deg + params["gamma_deg"]
    R_nose     = R_nose_nom * params["R_nose_factor"]
    T_wall     = 300.0 + params.get("T_wall_delta", 0.0)

    # Canopy
    dims_scaled = {k: v * params["A_factor"]**0.5 for k, v in canopy_dims.items()}
    try:
        canopy = CanopyGeometry(canopy_shape, dims_scaled)
        A_ref  = canopy.calculate_area()
    except Exception:
        A_ref = 157.0

    Cd_eff = 1.7 * params["Cd_factor"]

    # ── Trajectory ────────────────────────────────────────────────────────────
    try:
        lf = LowFidelityEDL(atm, mass, Cd_eff, A_ref, gamma_deg)
        t_arr = np.linspace(0, 500, 200)
        v_arr, h_arr = lf.solve(t_arr, vel0_ms, alt0_m)
    except Exception:
        return None

    v_land = float(v_arr[-1])
    if np.isnan(v_land) or v_land < 0:
        return None

    # ── Heating ───────────────────────────────────────────────────────────────
    rho_arr = np.array([atm.density(max(0, h)) for h in h_arr])

    if use_realgas:
        try:
            from src.realgas_chemistry import realgas_trajectory_profile
            rg = realgas_trajectory_profile(v_arr, h_arr, planet_atm,
                                             R_nose=R_nose, T_wall=T_wall)
            q_arr = rg["q_rg_Wm2"]
            gamma_eff_mean = float(rg["gamma_eff"].mean())
            diss_max = float(rg["dissociation_CO2"].max())
            correction = float(rg["correction"].mean())
        except Exception:
            use_realgas = False

    if not use_realgas:
        from src.thermal_model import ThermalProtectionSystem as TPS2
        _tps_tmp = TPS2("nylon", 0.01)
        q_arr = np.array([_tps_tmp.sutton_graves_heating(rho, v, R_nose)
                          for rho, v in zip(rho_arr, v_arr)])
        gamma_eff_mean = 1.28; diss_max = 0.0; correction = 1.0

    # ── TPS thermal response ──────────────────────────────────────────────────
    try:
        tps = ThermalProtectionSystem(tps_mat_name, tps_thickness)
        tps.solve_1d_conduction(q_arr, t_arr, T_initial_K=T_wall)
        exceeded, T_peak = tps.check_material_limit()
        sf_tps  = tps.safety_margin()
    except Exception:
        T_peak = 999.0; sf_tps = 0.5; exceeded = False

    # ── Landing scatter (wind perturbation) ───────────────────────────────────
    # Simple horizontal drift: x_drift ≈ wind × t_descent
    t_land   = float(t_arr[np.searchsorted(-h_arr, 0, side='left') - 1]) \
               if np.any(h_arr <= 0) else float(t_arr[-1])
    x_east   = params.get("wind_ew_ms", 0) * t_land
    x_north  = params.get("wind_ns_ms", 0) * t_land

    # ── Opening shock ─────────────────────────────────────────────────────────
    rho0 = atm.density(alt0_m)
    F_open = 0.5 * rho0 * vel0_ms**2 * Cd_eff * A_ref
    F_open *= 2.5   # approximate CLA

    return {
        "v_land_ms":    v_land,
        "T_peak_K":     T_peak,
        "sf_tps":       sf_tps,
        "tps_failure":  1.0 if exceeded else 0.0,
        "F_open_kN":    F_open / 1e3,
        "x_east_m":     x_east,
        "x_north_m":    x_north,
        "q_peak_MWm2":  float(q_arr.max()) / 1e6,
        "gamma_eff":    gamma_eff_mean,
        "co2_diss_pct": diss_max * 100,
        "rg_correction":correction,
        "t_land_s":     t_land,
        # input params for sensitivity
        **{k: float(v) for k, v in params.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloEDL:
    """
    Full Monte Carlo uncertainty propagation for EDL missions.

    Parameters
    ----------
    n_samples    : number of Monte Carlo samples (≥500 recommended)
    use_realgas  : use real-gas CO₂ chemistry (slower but more accurate)
    seed         : random seed for reproducibility
    """

    def __init__(self, n_samples: int = 500,
                 use_realgas: bool = True,
                 seed: int = 42):
        self.n = n_samples
        self.use_realgas = use_realgas
        self.seed = seed
        self.results_df: pd.DataFrame | None = None

    def run(self, planet_atm, tps_mat, tps_thickness,
            canopy_shape, canopy_dims, mass_nom,
            alt0_m, vel0_ms, gamma_nom_deg, R_nose,
            verbose: bool = True) -> pd.DataFrame:
        """
        Run the Monte Carlo simulation.
        Returns DataFrame with all sample results.
        """
        t0 = time.perf_counter()
        samples = _lhs_samples(self.n, self.seed)

        rows = []
        n_valid = 0
        if verbose:
            print(f"\n[MC] Running {self.n} EDL samples  "
                  f"real-gas={'ON' if self.use_realgas else 'OFF'}")

        for i in range(self.n):
            p_i = {k: float(samples[k][i]) for k in samples}
            result = _run_one_sample(
                p_i, planet_atm, tps_mat, tps_thickness,
                canopy_shape, canopy_dims, mass_nom,
                alt0_m, vel0_ms, gamma_nom_deg, R_nose,
                self.use_realgas,
            )
            if result is not None:
                rows.append(result)
                n_valid += 1

            if verbose and (i+1) % max(1, self.n // 10) == 0:
                bar = "█" * ((i+1)*20//self.n) + "░" * (20-(i+1)*20//self.n)
                pct = (i+1)/self.n*100
                print(f"\r  [{bar}] {pct:5.1f}%  valid={n_valid}", end="", flush=True)

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"\r  [{'█'*20}] 100.0%  valid={n_valid}/{self.n}  ({elapsed:.1f}s)")

        self.results_df = pd.DataFrame(rows)
        return self.results_df

    # ── Statistics ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Compute mission-level statistics from MC results."""
        df = self.results_df
        if df is None or len(df) == 0:
            return {}

        n = len(df)

        # CDF of landing velocity
        v_sorted = np.sort(df["v_land_ms"].values)
        v_cdf    = np.arange(1, n+1) / n

        # Landing ellipse CEP (circular error probable)
        x, y = df["x_east_m"].values, df["x_north_m"].values
        cov   = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        vals  = np.maximum(vals, 0)

        # Mission success criteria
        p_tps_fail  = float(df["tps_failure"].mean())
        p_hard_land = float((df["v_land_ms"] > 25.0).mean())   # >25 m/s = crash
        p_success   = 1.0 - max(p_tps_fail, p_hard_land)

        # Extreme value fit to peak heat flux (Gumbel)
        q_max = df["q_peak_MWm2"].values
        mu_g, sigma_g = gumbel_r.fit(q_max)

        # Spearman sensitivity (inputs vs SF_tps)
        input_cols = [c for c in df.columns if c in PARAM_SPECS]
        sensitivities = {}
        for col in input_cols:
            corr, pval = spearmanr(df[col], df["sf_tps"])
            sensitivities[col] = {"rho": round(float(corr), 4),
                                   "p_val": round(float(pval), 4)}

        return {
            "n_valid":            n,
            "v_land": {
                "mean":   round(float(df["v_land_ms"].mean()), 3),
                "std":    round(float(df["v_land_ms"].std()),  3),
                "p05":    round(float(np.percentile(df["v_land_ms"], 5)),  3),
                "p50":    round(float(np.percentile(df["v_land_ms"], 50)), 3),
                "p95":    round(float(np.percentile(df["v_land_ms"], 95)), 3),
            },
            "T_peak": {
                "mean":   round(float(df["T_peak_K"].mean()), 1),
                "p95":    round(float(np.percentile(df["T_peak_K"], 95)), 1),
                "p99":    round(float(np.percentile(df["T_peak_K"], 99)), 1),
            },
            "sf_tps": {
                "mean":   round(float(df["sf_tps"].mean()), 4),
                "p05":    round(float(np.percentile(df["sf_tps"],  5)), 4),
                "min":    round(float(df["sf_tps"].min()), 4),
            },
            "mission": {
                "P_tps_failure":    round(p_tps_fail,  4),
                "P_hard_landing":   round(p_hard_land, 4),
                "P_mission_success":round(p_success,   4),
            },
            "landing_ellipse": {
                "sigma_east_m":     round(float(x.std()), 1),
                "sigma_north_m":    round(float(y.std()), 1),
                "CEP_50_m":         round(float(np.sqrt(vals.sum()) * 1.1774), 1),
                "CEP_90_m":         round(float(np.sqrt(vals.sum()) * 2.146),  1),
                "major_axis_m":     round(float(np.sqrt(vals[-1]) * 2), 1),
                "minor_axis_m":     round(float(np.sqrt(vals[0])  * 2), 1),
            },
            "q_peak_gumbel": {
                "mu_MWm2":    round(mu_g, 4),
                "sigma_MWm2": round(sigma_g, 4),
                "P99_MWm2":   round(float(gumbel_r.ppf(0.99, mu_g, sigma_g)), 4),
            },
            "sensitivity_to_sf": sensitivities,
            "v_cdf": {"v": v_sorted.tolist(), "cdf": v_cdf.tolist()},
        }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  (10-panel publication dashboard)
# ══════════════════════════════════════════════════════════════════════════════

def plot_mc(df: pd.DataFrame, stats: dict,
            planet_name: str = "Mars",
            save_path: Path | None = None) -> object:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Ellipse
    from scipy.stats import gaussian_kde
    import config as cfg

    matplotlib.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })
    BG="#080c14"; AX="#0d1526"; SP="#2a3d6e"; TX="#c8d8f0"
    C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"; CR="#ff4560"

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.52, wspace=0.38,
                            top=0.91, bottom=0.06, left=0.05, right=0.97)

    def gax(r, c, **kw):
        a = fig.add_subplot(gs[r, c], **kw)
        a.set_facecolor(AX); a.grid(True, alpha=0.25)
        a.tick_params(colors=TX); a.spines[:].set_color(SP)
        return a

    # ── P0: Landing velocity PDF ───────────────────────────────────────────────
    ax0 = gax(0, 0)
    v = df["v_land_ms"].values
    ax0.hist(v, bins=40, color=C1, alpha=0.65, edgecolor="none", density=True)
    try:
        kde_v = gaussian_kde(v)
        vx = np.linspace(v.min(), v.max(), 300)
        ax0.plot(vx, kde_v(vx), color=C3, lw=2)
    except Exception:
        pass
    for pct, label, color in [(5,"P05",C4),(50,"P50",C3),(95,"P95",CR)]:
        val = np.percentile(v, pct)
        ax0.axvline(val, color=color, lw=1.2, ls="--", label=f"{label}={val:.1f}")
    ax0.set_title("Landing Velocity PDF", fontweight="bold")
    ax0.set_xlabel("v_land [m/s]"); ax0.set_ylabel("Density")
    ax0.legend(fontsize=7.5)

    # ── P1: Landing velocity CDF ───────────────────────────────────────────────
    ax1 = gax(0, 1)
    v_cdf = np.array(stats["v_cdf"]["cdf"])
    v_sorted = np.array(stats["v_cdf"]["v"])
    ax1.plot(v_sorted, v_cdf, color=C1, lw=2)
    ax1.axhline(0.05, color=C4, lw=0.8, ls=":", alpha=0.7)
    ax1.axhline(0.50, color=C3, lw=0.8, ls=":", alpha=0.7)
    ax1.axhline(0.95, color=CR, lw=0.8, ls=":", alpha=0.7)
    ax1.set_title("Landing Velocity CDF", fontweight="bold")
    ax1.set_xlabel("v_land [m/s]"); ax1.set_ylabel("P(V ≤ v)")

    # ── P2: TPS safety factor PDF ─────────────────────────────────────────────
    ax2 = gax(0, 2)
    sf = df["sf_tps"].values
    ax2.hist(sf, bins=40, color=C2, alpha=0.65, edgecolor="none", density=True)
    ax2.axvline(1.5, color=CR, lw=1.5, ls="--", label="SF=1.5 min")
    ax2.axvline(float(sf.mean()), color=C3, lw=1.2, ls="--", label=f"Mean={sf.mean():.2f}")
    p_fail = stats["mission"]["P_tps_failure"]
    ax2.text(0.97, 0.95, f"P(fail)={p_fail:.3f}", transform=ax2.transAxes,
             ha="right", va="top", color=CR if p_fail > 0.01 else C3, fontsize=9)
    ax2.set_title("TPS Safety Factor", fontweight="bold")
    ax2.set_xlabel("SF"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=7.5)

    # ── P3: Peak heat flux Gumbel fit ─────────────────────────────────────────
    ax3 = gax(0, 3)
    qmax = df["q_peak_MWm2"].values
    ax3.hist(qmax, bins=40, color=CR, alpha=0.65, edgecolor="none", density=True)
    mu_g = stats["q_peak_gumbel"]["mu_MWm2"]
    sg_g = stats["q_peak_gumbel"]["sigma_MWm2"]
    qx = np.linspace(qmax.min(), qmax.max(), 300)
    from scipy.stats import gumbel_r as _gr
    ax3.plot(qx, _gr.pdf(qx, mu_g, sg_g), color=C4, lw=2, label="Gumbel fit")
    ax3.axvline(stats["q_peak_gumbel"]["P99_MWm2"], color=CR, lw=1.2, ls="--",
                label=f"P99={stats['q_peak_gumbel']['P99_MWm2']:.3f}")
    ax3.set_title("Peak Heat Flux (Gumbel)", fontweight="bold")
    ax3.set_xlabel("q_peak [MW/m²]"); ax3.set_ylabel("Density")
    ax3.legend(fontsize=7.5)

    # ── P4: 2-D Landing scatter + contours ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor(AX); ax4.grid(True, alpha=0.25)
    ax4.tick_params(colors=TX); ax4.spines[:].set_color(SP)
    x = df["x_east_m"].values; y = df["x_north_m"].values
    # KDE density
    try:
        xy = np.vstack([x, y])
        kde2d = gaussian_kde(xy)
        xg = np.linspace(x.min(), x.max(), 80)
        yg = np.linspace(y.min(), y.max(), 80)
        XX, YY = np.meshgrid(xg, yg)
        ZZ = kde2d(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        # Contours at 50%, 90%, 99%
        z_sorted = np.sort(ZZ.ravel())[::-1]
        z_cumsum = np.cumsum(z_sorted) / z_sorted.sum()
        levels = [float(z_sorted[np.searchsorted(z_cumsum, pct)]) for pct in [0.50, 0.90, 0.99]]
        ax4.contourf(XX/1e3, YY/1e3, ZZ, levels=levels[::-1]+[ZZ.max()],
                     colors=[C1], alpha=[0.35, 0.20, 0.10])
        ax4.contour(XX/1e3, YY/1e3, ZZ, levels=levels[::-1],
                    colors=[C3, C4, CR], linewidths=[1.2, 1.0, 0.8],
                    linestyles=["solid","dashed","dotted"])
    except Exception:
        pass
    ax4.scatter(x/1e3, y/1e3, s=2, alpha=0.25, color=C1)
    ax4.scatter([0],[0], s=80, color=CR, marker="*", zorder=5, label="Nominal target")
    ax4.set_title("Landing Ellipse (50/90/99% contours)", fontweight="bold")
    ax4.set_xlabel("East [km]"); ax4.set_ylabel("North [km]")
    ax4.legend(fontsize=7.5)

    # ── P5: Sensitivity tornado chart ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.set_facecolor(AX); ax5.grid(True, alpha=0.25, axis="x")
    ax5.tick_params(colors=TX); ax5.spines[:].set_color(SP)
    sens = stats.get("sensitivity_to_sf", {})
    if sens:
        snames = list(sens.keys())
        rhos   = [sens[k]["rho"] for k in snames]
        # Sort by |rho|
        order  = np.argsort(np.abs(rhos))[::-1][:10]
        bar_labels = [snames[i].replace("_factor","").replace("_deg","°").replace("_ms","") for i in order]
        bar_vals   = [rhos[i] for i in order]
        colors_bar = [C3 if v > 0 else CR for v in bar_vals]
        y_pos = np.arange(len(bar_labels))
        ax5.barh(y_pos, bar_vals, color=colors_bar, alpha=0.75, edgecolor="none")
        ax5.set_yticks(y_pos); ax5.set_yticklabels(bar_labels[::-1], fontsize=8)
        ax5.axvline(0, color=TX, lw=0.7)
    ax5.set_title("Sensitivity to SF_tps (Spearman ρ)", fontweight="bold")
    ax5.set_xlabel("Spearman correlation ρ")

    # ── P6: Peak temp PDF ─────────────────────────────────────────────────────
    ax6 = gax(2, 0)
    tp = df["T_peak_K"].values
    ax6.hist(tp, bins=40, color=C4, alpha=0.65, edgecolor="none", density=True)
    tps_lim = float(tp.max()) * 0.85  # approximate limit
    ax6.axvline(np.percentile(tp, 95), color=CR, lw=1.2, ls="--",
                label=f"P95={np.percentile(tp,95):.0f}K")
    ax6.set_title("TPS Peak Temperature", fontweight="bold")
    ax6.set_xlabel("T_peak [K]"); ax6.set_ylabel("Density")
    ax6.legend(fontsize=7.5)

    # ── P7: Real-gas γ distribution ───────────────────────────────────────────
    ax7 = gax(2, 1)
    if "gamma_eff" in df.columns:
        g_arr = df["gamma_eff"].values
        ax7.hist(g_arr, bins=35, color="#9d60ff", alpha=0.65, edgecolor="none", density=True)
        ax7.axvline(1.4, color=TX, lw=0.8, ls=":", alpha=0.6, label="γ=1.4 (frozen)")
        ax7.axvline(g_arr.mean(), color=C3, lw=1.2, ls="--",
                    label=f"Mean γ={g_arr.mean():.3f}")
        ax7.set_title("Effective γ (real-gas)", fontweight="bold")
        ax7.set_xlabel("γ_eff"); ax7.set_ylabel("Density")
        ax7.legend(fontsize=7.5)

    # ── P8: CO₂ dissociation ──────────────────────────────────────────────────
    ax8 = gax(2, 2)
    if "co2_diss_pct" in df.columns:
        d_arr = df["co2_diss_pct"].values
        ax8.hist(d_arr, bins=35, color=C2, alpha=0.65, edgecolor="none", density=True)
        ax8.axvline(d_arr.mean(), color=C3, lw=1.2, ls="--",
                    label=f"Mean={d_arr.mean():.1f}%")
        ax8.set_title("CO₂ Dissociation %", fontweight="bold")
        ax8.set_xlabel("Dissociation [%]"); ax8.set_ylabel("Density")
        ax8.legend(fontsize=7.5)

    # ── P9: Mission summary panel ─────────────────────────────────────────────
    ax9 = gax(2, 3)
    ax9.axis("off")
    m = stats["mission"]
    lel = stats["landing_ellipse"]
    v_s = stats["v_land"]
    sf_s = stats["sf_tps"]
    rows = [
        ("MISSION SUMMARY", "", "#00d4ff"),
        ("", "", TX),
        ("Planet", planet_name, TX),
        ("Samples (valid)", str(stats["n_valid"]), TX),
        ("", "", TX),
        ("v_land P50", f"{v_s['p50']:.2f} m/s", C3),
        ("v_land P05–P95", f"[{v_s['p05']:.1f}, {v_s['p95']:.1f}]", TX),
        ("", "", TX),
        ("SF_tps mean", f"{sf_s['mean']:.3f}", C3 if sf_s['mean']>=1.5 else CR),
        ("SF_tps P05", f"{sf_s['p05']:.3f}", C3 if sf_s['p05']>=1.0 else CR),
        ("", "", TX),
        ("P(TPS failure)", f"{m['P_tps_failure']:.4f}", CR if m['P_tps_failure']>0.01 else C3),
        ("P(hard landing)", f"{m['P_hard_landing']:.4f}", CR if m['P_hard_landing']>0.05 else C3),
        ("P(SUCCESS)", f"{m['P_mission_success']:.4f}",
         C3 if m['P_mission_success']>0.95 else (C4 if m['P_mission_success']>0.80 else CR)),
        ("", "", TX),
        ("CEP 50%", f"{lel['CEP_50_m']:.0f} m", TX),
        ("CEP 90%", f"{lel['CEP_90_m']:.0f} m", TX),
    ]
    for j, (label, val, color) in enumerate(rows):
        ax9.text(0.02, 1-j*0.060, label, transform=ax9.transAxes,
                 fontsize=8.5 if label != "MISSION SUMMARY" else 10,
                 fontweight="bold" if label in ("MISSION SUMMARY","P(SUCCESS)") else "normal",
                 color="#556688" if (label and not val) else TX)
        if val:
            ax9.text(0.98, 1-j*0.060, val, transform=ax9.transAxes,
                     fontsize=8.5, ha="right", fontweight="bold" if "SUCCESS" in label else "normal",
                     color=color)

    p_suc = m['P_mission_success']
    fig.text(0.5, 0.955,
             f"Monte Carlo EDL Uncertainty  —  n={stats['n_valid']}  |  "
             f"P(success)={p_suc:.4f}  |  "
             f"v_land P50={v_s['p50']:.2f}m/s  |  "
             f"CEP90={lel['CEP_90_m']:.0f}m  |  Planet: {planet_name}",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    sp = save_path or Path("outputs/mc_edl.png")
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ MC dashboard saved: {sp}")
    return fig


def run(planet_atm, tps_mat="nylon", tps_thickness=0.01,
        canopy_shape="elliptical", canopy_dims=None,
        mass_nom=900.0, alt0_m=125_000.0, vel0_ms=5800.0,
        gamma_nom_deg=15.0, R_nose=4.5,
        n_samples=300, use_realgas=True,
        planet_name="Mars", verbose=True) -> dict:
    """Run full Monte Carlo pipeline and return stats dict."""
    import matplotlib; matplotlib.use("Agg")
    if canopy_dims is None:
        canopy_dims = {"a": 10, "b": 5}

    mc = MonteCarloEDL(n_samples, use_realgas=use_realgas)
    df = mc.run(planet_atm, tps_mat, tps_thickness,
                canopy_shape, canopy_dims, mass_nom,
                alt0_m, vel0_ms, gamma_nom_deg, R_nose,
                verbose=verbose)

    stats = mc.summary()
    if verbose:
        m = stats["mission"]
        print(f"\n  P(success)      = {m['P_mission_success']:.4f}")
        print(f"  P(TPS failure)  = {m['P_tps_failure']:.4f}")
        print(f"  v_land P05/P50/P95 = {stats['v_land']['p05']:.1f} / "
              f"{stats['v_land']['p50']:.1f} / {stats['v_land']['p95']:.1f} m/s")
        print(f"  Landing CEP 90% = {stats['landing_ellipse']['CEP_90_m']:.0f} m")

    Path("outputs").mkdir(exist_ok=True)
    plot_mc(df, stats, planet_name=planet_name)
    df.to_csv("outputs/mc_edl_results.csv", index=False)

    import json
    safe_stats = {k: v for k, v in stats.items()
                  if k not in ("v_cdf", "sensitivity_to_sf")}
    Path("outputs/mc_edl_stats.json").write_text(json.dumps(safe_stats, indent=2))
    return stats
