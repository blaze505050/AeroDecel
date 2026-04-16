"""
src/edl_optimiser.py — EDL Sequence Optimiser (Differential Evolution)
=======================================================================
Given landing site constraints, optimises the full EDL sequence:
  • Parachute deployment trigger altitude
  • TPS thickness
  • Entry flight-path angle
  • Deployment velocity limit

Uses scipy.optimize.differential_evolution (free, global optimiser).

Certifies the result with a margin analysis:
  - Nominal performance
  - 3-sigma worst-case (from Monte Carlo sensitivities)
  - Compliance table against mission requirements
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import differential_evolution
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EDLRequirements:
    """Mission requirements for EDL certification."""
    target_landing_v_ms:   float = 12.0    # max landing velocity [m/s]
    max_tps_mass_kgm2:     float = 15.0    # max TPS areal mass
    min_safety_factor:     float = 1.5     # min structural SF
    max_peak_heatflux_MW:  float = 25.0    # max stagnation heat flux [MW/m²]
    target_alt_m:          float = 0.0     # landing site elevation [m] (MOLA AGL)
    max_g_load:            float = 20.0    # max deceleration in g
    chute_deploy_mach_max: float = 1.8     # maximum Mach at chute deploy
    chute_deploy_q_max_Pa: float = 850.0   # max dynamic pressure at deploy [Pa]


@dataclass
class EDLDesignPoint:
    """Design variables for the EDL sequence."""
    entry_fpa_deg:         float = -15.0   # flight-path angle [°]
    chute_deploy_alt_m:    float = 10_000.0  # parachute deployment altitude [m]
    tps_thickness_m:       float = 0.05    # TPS thickness [m]
    tps_material:          str   = "pica"  # TPS material
    entry_velocity_ms:     float = 5800.0  # entry interface velocity [m/s]


def _sim_edl(planet_atm, x: np.ndarray, reqs: EDLRequirements,
              mass_kg: float = 900.0,
              area_m2: float = 78.5,
              Cd: float = 1.7) -> dict:
    """
    Fast EDL simulation for a design point x = [fpa, deploy_alt, tps_thick, v0].
    Returns performance metrics vs requirements.
    """
    from src.multifidelity_pinn import LowFidelityEDL
    from src.ablation_model     import AblationSolver, ABLATIVE_DB
    from src.realgas_chemistry   import fay_riddell_heating

    fpa        = float(np.clip(x[0], -30, -3))
    deploy_alt = float(np.clip(x[1], 500, 50_000))
    tps_thick  = float(np.clip(x[2], 0.005, 0.15))
    v0         = float(np.clip(x[3], 4000, 13_000))

    # 1. Trajectory
    try:
        lf = LowFidelityEDL(planet_atm, mass_kg, Cd, area_m2, gamma_deg=abs(fpa))
        t  = np.linspace(0, 600, 200)
        v, h = lf.solve(t, v0, 125_000.0)
    except Exception:
        return {"valid": False, "penalty": 1e9}

    # 2. Chute deployment conditions
    deploy_idx = np.searchsorted(-h, -deploy_alt)
    if deploy_idx >= len(h):
        deploy_idx = len(h) - 1
    v_deploy = float(v[deploy_idx])
    h_deploy = float(h[deploy_idx])
    rho_dep  = planet_atm.density(max(h_deploy, 0))
    a_dep    = planet_atm.speed_of_sound(max(h_deploy, 0))
    M_deploy = v_deploy / max(a_dep, 1)
    q_deploy = 0.5 * rho_dep * v_deploy**2

    # 3. Landing velocity (below deployment altitude)
    land_mask = h <= reqs.target_alt_m + 10
    v_land = float(v[land_mask][0]) if land_mask.any() else float(v[-1])

    # 4. Peak heat flux (Sutton-Graves along trajectory)
    rho_arr = np.array([planet_atm.density(max(0, float(hh))) for hh in h])
    q_sg    = 1.74e-4 * np.sqrt(np.maximum(rho_arr, 0) / 4.5) * np.maximum(v, 0)**3
    q_peak  = float(q_sg.max())

    # 5. TPS ablation check
    try:
        t_entry = float(t[deploy_idx]) if deploy_idx > 0 else 200.0
        solver  = AblationSolver("pica", tps_thick, n_nodes=6)
        t_a     = np.linspace(0, max(t_entry, 10), 40)
        t_pk    = t_entry * 0.30
        q0_a    = q_peak * np.where(t_a<=t_pk, t_a/t_pk, (t_entry-t_a)/(t_entry-t_pk+1e-6))
        q0_a    = np.clip(q0_a, 0, None)
        abl_res = solver.solve(q0_a, t_a, verbose=False)
        recession_m = abl_res["total_recession_mm"] / 1000.0
        sf_tps  = (tps_thick * 1000) / max(abl_res["total_recession_mm"], 0.01)
        tps_mass = 220.0 * tps_thick   # PICA density
    except Exception:
        recession_m = tps_thick; sf_tps = 0.5; tps_mass = 220*tps_thick

    # 6. G-load peak (from velocity gradient)
    if len(v) > 2:
        dvdt = np.abs(np.gradient(v, t))
        g_peak = float(dvdt.max()) / planet_atm.gravity_ms2
    else:
        g_peak = 0.0

    # 7. Constraint violations (positive = violated)
    constraints = {
        "landing_v":    v_land - reqs.target_landing_v_ms,
        "tps_mass":     tps_mass - reqs.max_tps_mass_kgm2,
        "sf":           reqs.min_safety_factor - sf_tps,
        "heatflux":     q_peak/1e6 - reqs.max_peak_heatflux_MW,
        "g_load":       g_peak - reqs.max_g_load,
        "mach_deploy":  M_deploy - reqs.chute_deploy_mach_max,
        "q_deploy":     q_deploy - reqs.chute_deploy_q_max_Pa,
    }

    # 8. Objective: weighted sum of violations + landing velocity
    penalty = sum(max(v, 0)**2 * 1e3 for v in constraints.values())
    objective = v_land + penalty + tps_mass * 0.1

    return {
        "valid":        True,
        "objective":    objective,
        "penalty":      penalty,
        "v_land_ms":    v_land,
        "v_deploy_ms":  v_deploy,
        "h_deploy_m":   h_deploy,
        "M_deploy":     M_deploy,
        "q_deploy_Pa":  q_deploy,
        "q_peak_MWm2":  q_peak / 1e6,
        "g_peak":       g_peak,
        "tps_mass_kgm2":tps_mass,
        "sf_tps":       sf_tps,
        "recession_mm": recession_m * 1000,
        "tps_thick_mm": tps_thick * 1000,
        "entry_fpa_deg":fpa,
        "constraints":  constraints,
        "feasible":     all(v <= 0 for v in constraints.values()),
    }


class EDLOptimiser:
    """
    Differential Evolution optimiser for EDL sequence design.

    Optimises:  [entry_fpa, deploy_alt, tps_thick, entry_velocity]
    Subject to: all mission requirements
    """

    def __init__(self, planet_atm, requirements: EDLRequirements = None,
                 mass_kg: float = 900.0, area_m2: float = 78.5):
        self.planet = planet_atm
        self.reqs   = requirements or EDLRequirements()
        self.mass   = mass_kg
        self.A      = area_m2
        self._best: dict | None = None
        self._history: list = []

    def optimise(self, n_pop: int = 20, n_gen: int = 50,
                 verbose: bool = True) -> dict:
        """
        Run differential evolution. Returns optimal design point + certification.
        """
        bounds = [
            (-28, -5),          # entry_fpa [°]
            (2_000, 40_000),    # deploy_alt [m]
            (0.01, 0.12),       # tps_thickness [m]
            (4500, 11_000),     # entry_velocity [m/s]
        ]

        call_count = [0]

        def obj(x):
            res = _sim_edl(self.planet, x, self.reqs, self.mass, self.A)
            call_count[0] += 1
            if res["valid"]:
                self._history.append(res)
                return res["objective"]
            return 1e12

        if verbose:
            print(f"\n[EDL Optimiser] Differential Evolution  "
                  f"n_pop={n_pop}  n_gen={n_gen}")
            print(f"  Requirements: v_land≤{self.reqs.target_landing_v_ms}m/s  "
                  f"SF≥{self.reqs.min_safety_factor}  "
                  f"q≤{self.reqs.max_peak_heatflux_MW}MW/m²")

        result = differential_evolution(
            obj, bounds,
            popsize=max(n_pop//4, 4),
            maxiter=n_gen,
            tol=1e-6,
            mutation=(0.5, 1.2),
            recombination=0.8,
            seed=42,
            disp=False,
            updating="deferred",
            workers=1,
        )

        x_opt  = result.x
        res_opt = _sim_edl(self.planet, x_opt, self.reqs, self.mass, self.A)
        self._best = res_opt

        if verbose:
            print(f"\n  Optimal design:")
            print(f"    Entry FPA:       {res_opt['entry_fpa_deg']:.2f}°")
            print(f"    Deploy altitude: {res_opt['h_deploy_m']/1e3:.2f} km")
            print(f"    TPS thickness:   {res_opt['tps_thick_mm']:.1f} mm")
            print(f"    Deploy Mach:     {res_opt['M_deploy']:.3f}")
            print(f"    Landing v:       {res_opt['v_land_ms']:.2f} m/s")
            print(f"    Peak heat flux:  {res_opt['q_peak_MWm2']:.3f} MW/m²")
            print(f"    SF_TPS:          {res_opt['sf_tps']:.3f}")
            print(f"    Feasible:        {'✓' if res_opt['feasible'] else '✗'}")

        # Margin analysis (worst-case 3-sigma)
        margin = self._margin_analysis(x_opt)
        res_opt["margin_analysis"] = margin
        res_opt["n_evaluations"]   = call_count[0]
        res_opt["de_result"]       = result

        return res_opt

    def _margin_analysis(self, x_opt: np.ndarray) -> dict:
        """3-sigma margin analysis around optimal point."""
        sigmas = np.array([0.5, 500, 0.003, 100.0])  # 1-sigma uncertainties
        perturb_results = []
        rng = np.random.default_rng(0)
        for _ in range(50):
            x_p = x_opt + rng.normal(0, sigmas)
            res = _sim_edl(self.planet, x_p, self.reqs, self.mass, self.A)
            if res["valid"]:
                perturb_results.append(res)

        if not perturb_results:
            return {}

        keys = ["v_land_ms", "q_peak_MWm2", "sf_tps", "g_peak"]
        margin = {}
        for k in keys:
            vals = [r[k] for r in perturb_results]
            margin[k] = {
                "nominal": float(self._best.get(k, 0)),
                "mean":    float(np.mean(vals)),
                "std":     float(np.std(vals)),
                "p99":     float(np.percentile(vals, 99)),
                "p01":     float(np.percentile(vals, 1)),
            }
        return margin

    def plot_certification(self, result: dict, save_path: str = "outputs/edl_optimiser.png"):
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

        fig = plt.figure(figsize=(20, 11), facecolor="#080c14")
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38,
                                top=0.90, bottom=0.07, left=0.05, right=0.97)

        def gax(r, c):
            a = fig.add_subplot(gs[r, c])
            a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
            a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
            return a

        # Compliance table
        ax = gax(0, 0); ax.axis("off")
        reqs_items = [
            ("Landing v", f"{result['v_land_ms']:.2f}", f"≤{self.reqs.target_landing_v_ms}", "m/s"),
            ("Peak q",    f"{result['q_peak_MWm2']:.3f}", f"≤{self.reqs.max_peak_heatflux_MW}", "MW/m²"),
            ("SF_TPS",    f"{result['sf_tps']:.3f}", f"≥{self.reqs.min_safety_factor}", ""),
            ("G-load",    f"{result['g_peak']:.1f}", f"≤{self.reqs.max_g_load}", "g"),
            ("M_deploy",  f"{result['M_deploy']:.3f}", f"≤{self.reqs.chute_deploy_mach_max}", ""),
            ("q_deploy",  f"{result['q_deploy_Pa']:.0f}", f"≤{self.reqs.chute_deploy_q_max_Pa}", "Pa"),
        ]
        ax.text(0.5, 0.98, "COMPLIANCE TABLE", ha="center", va="top",
                transform=ax.transAxes, fontsize=10, fontweight="bold", color=C1)
        for j, (param, val, req, unit) in enumerate(reqs_items):
            y = 0.86 - j*0.14
            # Check compliance
            try:
                v_num = float(val.replace("≤","").replace("≥",""))
                r_num = float(req.replace("≤","").replace("≥",""))
                ok = (v_num <= r_num) if "≤" in req else (v_num >= r_num)
            except: ok = True
            col = C3 if ok else CR
            ax.text(0.02, y, param, transform=ax.transAxes, fontsize=9, color="#556688")
            ax.text(0.50, y, f"{val} {unit}", transform=ax.transAxes, fontsize=9, color=col, fontweight="bold")
            ax.text(0.80, y, req, transform=ax.transAxes, fontsize=9, color="#556688")
        feasible_col = C3 if result["feasible"] else CR
        ax.text(0.5, 0.02, "CERTIFIED ✓" if result["feasible"] else "NOT CERTIFIED ✗",
                ha="center", transform=ax.transAxes, fontsize=11, fontweight="bold", color=feasible_col)

        # Margin analysis
        if result.get("margin_analysis"):
            ax2 = gax(0, 1); ax2.axis("off")
            ax2.text(0.5, 0.98, "3-σ MARGIN ANALYSIS", ha="center", va="top",
                     transform=ax2.transAxes, fontsize=10, fontweight="bold", color=C4)
            ma = result["margin_analysis"]
            for j, (k, vals) in enumerate(ma.items()):
                y = 0.86 - j*0.18
                ax2.text(0.02, y, k.replace("_", " "), transform=ax2.transAxes, fontsize=8, color="#556688")
                ax2.text(0.50, y, f"nom={vals['nominal']:.3g}", transform=ax2.transAxes, fontsize=8, color=TX)
                ax2.text(0.50, y-0.07, f"P99={vals['p99']:.3g}  σ={vals['std']:.3g}",
                         transform=ax2.transAxes, fontsize=7.5, color=C2)

        # History: objective convergence
        if self._history:
            ax3 = gax(0, 2)
            obj_hist = [r["objective"] for r in self._history]
            ax3.semilogy(range(len(obj_hist)), obj_hist, color=C1, lw=1.0, alpha=0.5)
            # Running minimum
            running_min = np.minimum.accumulate(obj_hist)
            ax3.semilogy(range(len(running_min)), running_min, color=C3, lw=2)
            ax3.set_xlabel("Evaluation"); ax3.set_ylabel("Objective (log)")
            ax3.set_title("DE Convergence", fontweight="bold")

        # Feasibility in history
        ax4 = gax(0, 3)
        if self._history:
            feas = [r["feasible"] for r in self._history]
            cumfeas = np.cumsum(feas)
            ax4.plot(cumfeas, color=C3, lw=2)
            ax4.set_xlabel("Evaluation"); ax4.set_ylabel("Cumulative feasible")
            ax4.set_title("Feasible Solution Count", fontweight="bold")

        # Constraint margin bar chart
        ax5 = fig.add_subplot(gs[1, :2])
        ax5.set_facecolor("#0d1526"); ax5.grid(True, alpha=0.28, axis="x")
        ax5.tick_params(colors=TX); ax5.spines[:].set_color("#2a3d6e")
        constr = result.get("constraints", {})
        names  = list(constr.keys())
        vals_c = [constr[k] for k in names]
        colors_bar = [C3 if v <= 0 else CR for v in vals_c]
        bars = ax5.barh(names, vals_c, color=colors_bar, alpha=0.75)
        ax5.axvline(0, color=TX, lw=1.0)
        ax5.set_title("Constraint violations (≤0 = satisfied)", fontweight="bold")
        ax5.set_xlabel("Violation magnitude")

        # FPA vs landing v sweep
        ax6 = gax(1, 2)
        fpa_arr = np.linspace(-25, -5, 15)
        vland_arr = []
        for fpa_v in fpa_arr:
            x_v = np.array([fpa_v, result["h_deploy_m"], result["tps_thick_mm"]/1000, result.get("entry_velocity_ms", 5800)])
            res_v = _sim_edl(self.planet, x_v, self.reqs, self.mass, self.A)
            vland_arr.append(res_v["v_land_ms"] if res_v["valid"] else np.nan)
        ax6.plot(fpa_arr, vland_arr, color=C1, lw=2)
        ax6.axvline(result["entry_fpa_deg"], color=C4, lw=1.5, ls="--",
                    label=f"Optimal FPA={result['entry_fpa_deg']:.1f}°")
        ax6.axhline(self.reqs.target_landing_v_ms, color=CR, lw=1, ls=":", label="Req")
        ax6.set_xlabel("Entry FPA [°]"); ax6.set_ylabel("Landing v [m/s]")
        ax6.set_title("Sensitivity: FPA → landing v", fontweight="bold")
        ax6.legend(fontsize=7.5)

        # Alt sensitivity
        ax7 = gax(1, 3)
        alt_arr = np.linspace(2000, 40000, 15)
        vland_alt = []
        for alt_v in alt_arr:
            x_v = np.array([result["entry_fpa_deg"], alt_v, result["tps_thick_mm"]/1000, result.get("entry_velocity_ms",5800)])
            res_v = _sim_edl(self.planet, x_v, self.reqs, self.mass, self.A)
            vland_alt.append(res_v["v_land_ms"] if res_v["valid"] else np.nan)
        ax7.plot(alt_arr/1e3, vland_alt, color=C2, lw=2)
        ax7.axvline(result["h_deploy_m"]/1e3, color=C4, lw=1.5, ls="--", label=f"Optimal")
        ax7.axhline(self.reqs.target_landing_v_ms, color=CR, lw=1, ls=":", label="Req")
        ax7.set_xlabel("Deploy alt [km]"); ax7.set_ylabel("Landing v [m/s]")
        ax7.set_title("Sensitivity: deploy alt → landing v", fontweight="bold")
        ax7.legend(fontsize=7.5)

        fig.text(0.5, 0.955,
                 f"EDL Sequence Optimisation (Differential Evolution)  |  "
                 f"{'CERTIFIED ✓' if result['feasible'] else 'NOT CERTIFIED ✗'}  |  "
                 f"n_evals={result['n_evaluations']}",
                 ha="center", fontsize=11, fontweight="bold", color=TX)

        Path(save_path).parent.mkdir(exist_ok=True)
        fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
        print(f"  ✓ EDL optimiser plot saved: {save_path}")
        plt.close(fig)


def run_edl_optimiser(planet_atm=None, n_pop: int = 20, n_gen: int = 30,
                       verbose: bool = True) -> dict:
    import matplotlib; matplotlib.use("Agg")
    if planet_atm is None:
        from src.planetary_atm import MarsAtmosphere
        planet_atm = MarsAtmosphere()

    reqs = EDLRequirements(
        target_landing_v_ms=15.0, max_tps_mass_kgm2=20.0,
        min_safety_factor=1.5, max_peak_heatflux_MW=30.0,
        chute_deploy_mach_max=2.0, chute_deploy_q_max_Pa=1000.0,
    )
    opt = EDLOptimiser(planet_atm, reqs, mass_kg=900, area_m2=78.5)
    result = opt.optimise(n_pop=n_pop, n_gen=n_gen, verbose=verbose)
    opt.plot_certification(result)
    return result
