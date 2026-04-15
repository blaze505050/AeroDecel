"""
main.py — AeroDecel v6.0 Project Icarus — Master CLI
=====================================================
Usage
-----
  python main.py --demo mars_edl
  python main.py --masterpiece              # real-gas + MC + 3D viz
  python main.py --all-demos                # all 4 EDL scenarios
  python main.py --sixdof                   # T1: 6-DOF attitude dynamics
  python main.py --ablation                 # T1: Amar ablation coupling
  python main.py --flutter                  # T1: aeroelastic flutter FEM
  python main.py --d3q19                    # T1: D3Q19 LBM turbulence
  python main.py --flow                     # T2: normalising flows
  python main.py --multiplanet              # T2: multi-planet FNO
  python main.py --gnn                      # T2: canopy GNN
  python main.py --gp-opt                   # T2: GP Bayesian optimisation
  python main.py --online-kalman            # T2: online PINN-Kalman
  python main.py --generate-dataset
"""
import argparse, time, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from src.planetary_atm   import get_planet_atmosphere
from src.thermal_model   import ThermalProtectionSystem
from src.canopy_geometry import CanopyGeometry

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║  AeroDecel v6.0  —  Project Icarus                                      ║
║  Planetary EDL · 6-DOF · Real-Gas · Ablation · Flutter · D3Q19         ║
║  Norm-Flows · Multi-Planet FNO · Canopy GNN · GP-BayesOpt · EKF-PINN  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

def _sec(t): print(f"\n{'─'*70}\n  {t}\n{'─'*70}")
def _el(t0): return f"{time.time()-t0:.1f}s"


def run_pipeline(args):
    print(BANNER)
    t_total = time.time()
    planet = get_planet_atmosphere(args.planet)
    tps    = ThermalProtectionSystem(args.material, args.thickness)
    canopy = CanopyGeometry(args.canopy_shape, cfg.CANOPY_DIMENSIONS)
    Path("outputs").mkdir(exist_ok=True)

    print(f"  Planet   : {planet.name}  g={planet.gravity_ms2:.3f} m/s²")
    print(f"  TPS      : {tps.mat.name}  {tps.thickness*100:.1f}cm  T_lim={tps.mat.max_temperature_K:.0f}K")
    print(f"  Canopy   : {canopy.shape}  A={canopy.calculate_area():.1f}m²")

    # ── Core demos ─────────────────────────────────────────────────────────
    if args.all_demos:
        for name, fn in DEMO_MAP.items():
            fn(planet, tps, canopy, verbose=args.verbose)

    elif args.masterpiece:
        demo_masterpiece(planet, tps, canopy, verbose=args.verbose)

    elif args.demo:
        DEMO_MAP[args.demo](planet, tps, canopy, verbose=args.verbose)

    # ── Tier 1 Physics ─────────────────────────────────────────────────────
    if args.sixdof:
        _sec("Tier 1: Full 6-DOF Trajectory"); t0=time.time()
        from src.sixdof_trajectory import run as r6, VehicleConfig
        veh = VehicleConfig()
        df6, stab = r6(planet, veh,
                       entry_speed_ms=5800, entry_alt_m=125_000,
                       entry_fpa_deg=-15, verbose=args.verbose)
        print(f"  6-DOF: {len(df6)} steps  max-g={df6['g_load'].max():.2f}  "
              f"max-α={df6['alpha_deg'].abs().max():.1f}°  ⏱ {_el(t0)}")

    if args.ablation:
        _sec("Tier 1: Amar Ablation Coupling"); t0=time.time()
        from src.ablation_model import run as rabl
        from src.multifidelity_pinn import LowFidelityEDL
        lf = LowFidelityEDL(planet, 900, 1.7, canopy.calculate_area())
        t_arr = np.linspace(0, 400, 100)
        v_arr, h_arr = lf.solve(t_arr, 5800, 125_000)
        rho_arr = np.array([planet.density(max(0,float(h))) for h in h_arr])
        q_sg = 1.74e-4 * np.sqrt(np.maximum(rho_arr,0)/4.5) * np.maximum(v_arr,0)**3
        q_pk = max(float(q_sg.max())/1e6, 0.1)
        _, summ = rabl(args.material if args.material in ['pica','avcoat','srp'] else 'pica',
                       0.05, q_peak_MW=q_pk, t_entry_s=float(t_arr[-1]), verbose=args.verbose)
        print(f"  Ablation: recession={summ['final_recession_mm']:.2f}mm  "
              f"blocking={summ['blocking_pct']:.1f}%  survived={summ['survived']}  ⏱ {_el(t0)}")

    if args.flutter:
        _sec("Tier 1: Aeroelastic Canopy Flutter"); t0=time.time()
        from src.aeroelastic_flutter import run as rfl
        rho_30 = planet.density(30_000)
        fr = rfl(canopy_radius_m=np.sqrt(canopy.calculate_area()/np.pi),
                  tension_Nm=1200, rho_air=rho_30,
                  v_range_ms=(5, 300), verbose=args.verbose)
        print(f"  Flutter: f₁={fr['modal']['freq_Hz'][0]:.4f}Hz  "
              f"v_cr={fr['flutter']['v_flutter_ms']:.1f}m/s  ⏱ {_el(t0)}")

    if args.d3q19:
        _sec("Tier 1: D3Q19 LBM + Smagorinsky Turbulence"); t0=time.time()
        from src.lbm_d3q19 import run as rlbm
        lr, _ = rlbm(nx=24, ny=16, nz=16, Re=200, n_steps=1000, verbose=args.verbose)
        print(f"  D3Q19: Cd={lr['Cd']:.4f}  Cl={lr['Cl_y']:.4f}  "
              f"steps={lr['step']}  ⏱ {_el(t0)}")

    # ── Tier 2 ML ──────────────────────────────────────────────────────────
    if args.flow:
        _sec("Tier 2: Physics-Constrained Normalising Flows"); t0=time.time()
        from src.normalizing_flows import run as rnf
        nf = rnf(n_training_trajs=150, n_epochs=300, verbose=args.verbose)
        p  = nf["posterior"]
        print(f"  Flow: v_mean={p['v_mean'][-1]:.2f}m/s  "
              f"σ={p['v_std'][-1]:.2f}  backend={p['backend']}  ⏱ {_el(t0)}")

    if args.multiplanet:
        _sec("Tier 2: Multi-Planet FNO + Zero-Shot Triton"); t0=time.time()
        from src.multiplanet_operator import run_multiplanet
        mp = run_multiplanet(n_traj=80, n_epochs=200, verbose=args.verbose)
        print(f"  MultiPlanet: Triton pred shape={mp['triton_pred'].shape}  "
              f"backend={mp['operator']._backend}  ⏱ {_el(t0)}")

    if args.gnn:
        _sec("Tier 2: Canopy Graph Neural Network"); t0=time.time()
        from src.canopy_gnn import run_gnn
        gr = run_gnn(n_gores=12, n_radial=6, q_dyn=150.0, verbose=args.verbose)
        print(f"  GNN: N={gr['graph'].N} nodes  "
              f"max_stress={gr['result']['max_stress_Pa']:.1f}Pa  "
              f"min_SF={gr['sf'].min():.3f}  ⏱ {_el(t0)}")

    if args.gp_opt:
        _sec("Tier 2: GP Emulator + Bayesian TPS Optimisation"); t0=time.time()
        from src.gp_emulator import run_gp_emulator
        mat = args.material if args.material in ['pica','avcoat','srp'] else 'pica'
        gpr = run_gp_emulator(mat, q_peak_MW=15.0, n_iter=40, verbose=args.verbose)
        opt = gpr["optimal"]
        print(f"  GP-BayesOpt: optimal={opt['thickness_m']*1000:.2f}mm  "
              f"mass={opt['mass_kgm2']:.3f}kg/m²  "
              f"survived={opt['survived']}  evals={gpr['n_evals']}  ⏱ {_el(t0)}")

    if args.online_kalman:
        _sec("Tier 2: Online PINN-Kalman Real-Time Estimation"); t0=time.time()
        from src.online_pinn_kalman import run as rk
        kr = rk(true_Cd=1.55, true_ti=2.8, n_obs=80, verbose=args.verbose)
        f  = kr["final"]
        print(f"  Kalman: Cd={f['Cd_hat']:.5f}±{f['Cd_std']:.5f}  "
              f"t_infl={f['ti_hat']:.3f}±{f['ti_std']:.3f}s  ⏱ {_el(t0)}")

    if args.generate_dataset:
        _sec("Dataset Generation"); t0=time.time()
        from src.operator_dataset import OperatorDataset
        ds   = OperatorDataset(output_resolution=100, planet_name=args.planet)
        data = ds.generate(n_samples=args.n_samples, verbose=args.verbose)
        ds.save(data, "outputs/operator_dataset.npz")
        print(f"  Dataset: {data['n_valid']} samples  ⏱ {_el(t0)}")

    # ── Summary ────────────────────────────────────────────────────────────
    out   = Path("outputs").resolve()
    files = sorted(out.glob("*.*"))
    print(f"\n{'═'*70}")
    print(f"  ✅  COMPLETE  |  {time.time()-t_total:.1f}s  |  📁  {out}")
    print(f"  Output files ({len(files)}):")
    for f in files:
        sz = f.stat().st_size
        print(f"    {f.name:<45} {sz//1024:>5}KB")
    print(f"{'═'*70}")


# ── EDL Demo functions ─────────────────────────────────────────────────────

def _edl_sim(planet, mass, alt0, v0, gamma, Cd, A, t_max=500):
    from src.multifidelity_pinn import LowFidelityEDL
    lf = LowFidelityEDL(planet, mass, Cd, A, gamma_deg=gamma)
    t  = np.linspace(0, t_max, 300)
    v, h = lf.solve(t, v0, alt0)
    return t, v, h

def _edl_plot(planet, tps, canopy, mass, alt0, v0, gamma, Cd, title, save):
    t, v, h = _edl_sim(planet, mass, alt0, v0, gamma, Cd, canopy.calculate_area())
    from src.thermal_model import ThermalProtectionSystem
    rho = np.array([planet.density(max(0,float(h_))) for h_ in h])
    q   = tps.sutton_graves_heating(rho.mean(), float(v.max()), 1.0)
    t_a = np.linspace(0, float(t[-1]), 100)
    tps2 = ThermalProtectionSystem(tps.mat.name.lower(), tps.thickness)
    tps2.solve_1d_conduction(q, t_a)
    exc, T_pk = tps2.check_material_limit()
    sf = tps2.safety_margin()
    print(f"  ✓ {title}: v_f={v[-1]:.1f}m/s  q={q/1e6:.3f}MW  T_pk={T_pk:.0f}K  SF={sf:.3f}")

    import matplotlib.gridspec as gridspec
    plt.rcParams.update({"figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0","axes.labelcolor":"#c8d8f0",
        "xtick.color":"#c8d8f0","ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#080c14")
    for ax in axes:
        ax.set_facecolor("#0d1526"); ax.grid(True,alpha=0.3)
        ax.tick_params(colors="#c8d8f0"); ax.spines[:].set_color("#2a3d6e")
    axes[0].fill_between(t,v/1e3,alpha=0.2,color="#00d4ff"); axes[0].plot(t,v/1e3,color="#00d4ff",lw=2)
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("v [km/s]"); axes[0].set_title("Velocity",fontweight="bold")
    axes[1].fill_between(t,h/1e3,alpha=0.2,color="#9d60ff"); axes[1].plot(t,h/1e3,color="#9d60ff",lw=2)
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("h [km]"); axes[1].set_title("Altitude",fontweight="bold")
    axes[2].plot(v/1e3, h/1e3, color="#a8ff3e", lw=2)
    axes[2].set_xlabel("v [km/s]"); axes[2].set_ylabel("h [km]"); axes[2].set_title("Phase Portrait",fontweight="bold")
    fig.suptitle(title, fontweight="bold", fontsize=11, color="#c8d8f0")
    fig.savefig(save, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save}")

def demo_mars_edl(planet, tps, canopy, verbose=True):
    _sec("Mars EDL — Perseverance-style")
    _edl_plot(planet, tps, canopy, 900, 125_000, 5800, 15, 1.7,
              "Mars EDL — Perseverance Style", "outputs/mars_edl.png")

def demo_drone_recovery(planet, tps, canopy, verbose=True):
    _sec("Drone Recovery")
    _edl_plot(planet, tps, canopy, 12, 800, 40, 80, 1.35,
              "Drone Recovery — Low Altitude", "outputs/drone_recovery.png")

def demo_reentry_capsule(planet, tps, canopy, verbose=True):
    _sec("Re-entry Capsule")
    _edl_plot(planet, tps, canopy, 5400, 200_000, 11_000, 6.5, 1.7,
              "Re-entry Capsule — Orion Style", "outputs/reentry_capsule.png")

def demo_military_airdrop(planet, tps, canopy, verbose=True):
    _sec("Military Airdrop")
    _edl_plot(planet, tps, canopy, 4500, 7000, 90, 85, 1.35,
              "Military Airdrop — Heavy LAPES", "outputs/military_airdrop.png")

def demo_masterpiece(planet, tps, canopy, verbose=True):
    _sec("MASTERPIECE — Real-Gas + Monte Carlo + 3-D Visualisation")
    from src.multifidelity_pinn import LowFidelityEDL
    from src.realgas_chemistry  import realgas_trajectory_profile
    from src.monte_carlo_edl    import MonteCarloEDL, plot_mc
    from src.visualization_3d   import visualize

    lf   = LowFidelityEDL(planet, 900, 1.7, canopy.calculate_area(), gamma_deg=15)
    t_a  = np.linspace(0, 500, 300)
    v_a, h_a = lf.solve(t_a, 5800, 125_000)
    pname = planet.name.lower()

    print(f"  [1/3] Real-gas CO₂ chemistry…")
    rg  = realgas_trajectory_profile(v_a, h_a, planet, R_nose=4.5, planet_name=pname)
    print(f"    q_rg={rg['q_rg_Wm2'].max()/1e6:.3f}MW  γ=[{rg['gamma_eff'].min():.3f},{rg['gamma_eff'].max():.3f}]  CO₂diss={rg['dissociation_CO2'].max()*100:.1f}%")

    print(f"  [2/3] Monte Carlo n=200…")
    mc  = MonteCarloEDL(200, use_realgas=False, seed=42)
    df  = mc.run(planet, tps.mat.name.lower() if tps.mat.name.lower() in ['pica','avcoat','srp','nylon','kevlar','nomex','vectran','zylon'] else 'nylon',
                 tps.thickness, canopy.shape, cfg.CANOPY_DIMENSIONS,
                 900, 125_000, 5800, 15, 4.5, verbose=verbose)
    ms  = mc.summary()
    fig = plot_mc(df, ms, planet_name=planet.name, save_path=Path("outputs/mc_edl_dashboard.png"))
    plt.close(fig)
    print(f"    P(success)={ms['mission']['P_mission_success']:.4f}  CEP90={ms['landing_ellipse']['CEP_90_m']:.0f}m")

    print(f"  [3/3] 3-D visualisation…")
    out = visualize(t_a, v_a, h_a, rg["q_rg_Wm2"]/1e6, planet_name=pname, mc_df=df,
                    gamma_deg=15, title=f"{planet.name} EDL Masterpiece\nReal-Gas + MC + 3-D",
                    save_dir=Path("outputs"))
    print(f"    {Path(out['matplotlib_path']).stat().st_size//1024}KB")
    print(f"  ✅  Masterpiece complete")

DEMO_MAP = {
    "mars_edl":         demo_mars_edl,
    "drone_recovery":   demo_drone_recovery,
    "reentry_capsule":  demo_reentry_capsule,
    "military_airdrop": demo_military_airdrop,
}


def build_parser():
    p = argparse.ArgumentParser(description="AeroDecel v6.0 — Project Icarus",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Planet & materials
    p.add_argument("--planet",      default=cfg.PLANET, choices=["mars","venus","titan"])
    p.add_argument("--material",    default=cfg.TPS_MATERIAL,
                   choices=["nylon","kevlar","nomex","vectran","zylon","pica","avcoat","srp"])
    p.add_argument("--thickness",   type=float, default=cfg.TPS_THICKNESS)
    p.add_argument("--canopy-shape",default=cfg.CANOPY_SHAPE,
                   choices=["elliptical","circular","rectangular","disk_gap_band","tricone"])
    # Pipeline
    p.add_argument("--demo",        default=None, choices=list(DEMO_MAP.keys()))
    p.add_argument("--all-demos",   action="store_true")
    p.add_argument("--masterpiece", action="store_true", help="Real-gas+MC+3D flagship")
    p.add_argument("--verbose",     action="store_true", default=True)
    p.add_argument("--generate-dataset", action="store_true")
    p.add_argument("--n-samples",   type=int, default=500)
    # Tier 1
    p.add_argument("--sixdof",      action="store_true", help="Full 6-DOF trajectory")
    p.add_argument("--ablation",    action="store_true", help="Amar ablation coupling")
    p.add_argument("--flutter",     action="store_true", help="Aeroelastic flutter FEM")
    p.add_argument("--d3q19",       action="store_true", help="D3Q19 LBM + Smagorinsky")
    # Tier 2
    p.add_argument("--flow",         action="store_true", help="Normalising flows")
    p.add_argument("--multiplanet",  action="store_true", help="Multi-planet FNO")
    p.add_argument("--gnn",          action="store_true", help="Canopy GNN")
    p.add_argument("--gp-opt",       action="store_true", help="GP Bayesian optimisation")
    p.add_argument("--online-kalman",action="store_true", help="Online PINN-Kalman EKF")
    return p


def main():
    args = build_parser().parse_args()
    # Default: show help if no action specified
    if not any([args.demo, args.all_demos, args.masterpiece,
                args.sixdof, args.ablation, args.flutter, args.d3q19,
                args.flow, args.multiplanet, args.gnn, args.gp_opt,
                args.online_kalman, args.generate_dataset]):
        args.demo = "mars_edl"
    run_pipeline(args)

if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3-5 FLAGS (append to run_pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def run_tier3_tier5(args, planet, tps, canopy):
    """Run Tier 3/4/5 features."""
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if getattr(args, "pareto", False):
        _sec("Tier 3: Multi-Objective TPS Optimisation (NSGA-II)"); t0=time.time()
        from src.tps_multiobjective import run_pareto_optimisation
        mat = args.material if args.material in ['pica','avcoat','srp','kevlar','zylon'] else 'pica'
        par = run_pareto_optimisation(q_peak_MW=15.0, t_entry_s=200.0,
                                       n_pop=args.nsga_pop, n_gen=args.nsga_gen,
                                       materials=[mat,'pica','avcoat','srp'],
                                       verbose=args.verbose)
        print(f"  Pareto: {par['n_feasible']} feasible  {len(par['pareto_df'])} Pareto solutions  ⏱ {_el(t0)}")

    if getattr(args, "edl_opt", False):
        _sec("Tier 3: EDL Sequence Optimiser (Differential Evolution)"); t0=time.time()
        from src.edl_optimiser import run_edl_optimiser
        res = run_edl_optimiser(planet_atm=planet, n_pop=20, n_gen=30, verbose=args.verbose)
        print(f"  EDL opt: v_land={res['v_land_ms']:.2f}m/s  certified={res['feasible']}  ⏱ {_el(t0)}")

    if getattr(args, "fta", False):
        _sec("Tier 3: Fault Tree Analysis (FTA)"); t0=time.time()
        from src.fault_tree import run as fta_run
        rep = fta_run(sf_tps=2.0, sf_structure=2.5, n_mc=5000, verbose=args.verbose)
        print(f"  FTA: P(success)={rep['P_mission_success']:.6f}  P(failure)={rep['P_mission_failure']:.2e}  ⏱ {_el(t0)}")

    if getattr(args, "gantt", False):
        _sec("Tier 4: Mission Timeline Gantt + Animated Entry"); t0=time.time()
        from src.mission_gantt import run as gantt_run
        gr = gantt_run(planet_atm=planet, verbose=args.verbose)
        print(f"  Gantt: {len(gr['phases'])} phases  animation={Path(gr['animation_path']).stat().st_size//1024}KB  ⏱ {_el(t0)}")

    if getattr(args, "track", False):
        _sec("Tier 5: Experiment Tracker Summary")
        from src.experiment_tracker import get_tracker
        tracker = get_tracker()
        tracker.summary()
        tracker.plot_history()
        print(f"  Database: {Path('outputs/experiments.db').stat().st_size//1024}KB")
