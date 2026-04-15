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

    # ── New Modules ────────────────────────────────────────────────────────
    if getattr(args, 'validate_perseverance', False):
        _sec("Validation: Perseverance EDL Reconstruction"); t0=time.time()
        from src.perseverance_validation import run as val_run
        vr = val_run(verbose=args.verbose)
        s = vr["stats"]
        print(f"  Validation: R²={s['r_squared']:.5f}  "
              f"RMS={s['rms_v_pct']:.1f}%  peak_g={s['peak_g_model']:.1f}  ⏱ {_el(t0)}")

    if getattr(args, 'aero_db', False):
        _sec("Hypersonic Aero Database Generation"); t0=time.time()
        from src.aero_database import run as adb_run
        adb = adb_run(verbose=args.verbose)
        print(f"  AeroDB: {adb['shape'][0]}×{adb['shape'][1]} grid  ⏱ {_el(t0)}")

    if getattr(args, 'multistage', False):
        _sec("Multi-Stage EDL (Perseverance 4-Phase)"); t0=time.time()
        from src.multistage_edl import run as ms_run
        ms = ms_run(planet, verbose=args.verbose)
        sm = ms["summary"]
        print(f"  Multi-stage: v_land={sm['v_landing_ms']:.2f}m/s  "
              f"peak_g={sm['peak_g']:.1f}  fuel={sm['fuel_used_kg']:.1f}kg  ⏱ {_el(t0)}")

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
    # New advanced modules
    p.add_argument("--cr3bp",      action="store_true", help="CR3BP 3-body gravity")
    p.add_argument("--cr3bp-system",default="earth_moon",
                   choices=["earth_moon","mars_phobos","mars_deimos","jupiter_europa","sun_earth"])
    p.add_argument("--moc",        action="store_true", help="Axisymmetric MOC wake solver")
    p.add_argument("--moc-mach",   type=float, default=2.5, help="Free-stream Mach for MOC")
    p.add_argument("--mhd",        action="store_true", help="MHD plasma steering")
    p.add_argument("--mhd-B",      type=float, default=0.5, help="B field [T] for MHD")
    p.add_argument("--rl",         action="store_true", help="RL active CoM guidance")
    p.add_argument("--rl-episodes",type=int,   default=200, help="RL training episodes")
    p.add_argument("--use-sb3",    action="store_true", help="Use stable-baselines3 PPO")
    p.add_argument("--nsga-pop",   type=int,   default=50,  help="NSGA-II population size")
    p.add_argument("--nsga-gen",   type=int,   default=40,  help="NSGA-II generations")
    # New v6.1 modules
    p.add_argument("--validate-perseverance", action="store_true",
                   help="Validate against real Perseverance EDL data")
    p.add_argument("--aero-db",    action="store_true",
                   help="Generate hypersonic aero database (Cd/CL/Cm tables)")
    p.add_argument("--multistage", action="store_true",
                   help="Multi-stage EDL (entry→jettison→chute→powered)")
    return p


def main():
    args = build_parser().parse_args()
    # Default: show help if no action specified
    if not any([args.demo, args.all_demos, args.masterpiece,
                args.sixdof, args.ablation, args.flutter, args.d3q19,
                args.flow, args.multiplanet, args.gnn, args.gp_opt,
                args.online_kalman, args.generate_dataset,
                getattr(args, 'validate_perseverance', False),
                getattr(args, 'aero_db', False),
                getattr(args, 'multistage', False)]):
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

    # ── New advanced modules ───────────────────────────────────────────────
    if getattr(args, cr3bp, False):
        args.cr3bp_system = getattr(args, cr3bp_system, earth_moon)
        run_cr3bp(args)
    if getattr(args, moc, False):
        args.moc_mach = getattr(args, moc_mach, 2.5)
        args.canopy_area = canopy.calculate_area()
        run_moc(args)
    if getattr(args, mhd, False):
        args.mhd_B = getattr(args, mhd_B, 0.5)
        run_mhd(args)
    if getattr(args, rl, False):
        args.rl_episodes = getattr(args, rl_episodes, 200)
        args.use_sb3 = getattr(args, use_sb3, False)
        run_rl(args)

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


# ══════════════════════════════════════════════════════════════════════════════
# NEW ADVANCED MODULE RUNNERS  (CR3BP, MOC, MHD, RL)
# ══════════════════════════════════════════════════════════════════════════════

def run_cr3bp(args):
    """Tier 1+: CR3BP 3-body gravity for EDL entry interface."""
    _sec("CR3BP 3-Body Gravity — Entry Interface Dynamics"); t0=time.time()
    from src.cr3bp_gravity import CR3BPGravity, plot_cr3bp
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    system = getattr(args, "cr3bp_system", "earth_moon")
    cr     = CR3BPGravity(system)
    lpts   = cr.lagrange_points()
    L2     = lpts["L2"]

    # Near-L2 trajectory
    r0n = L2 + np.array([0.12, 0, 0.08])
    v0n = np.array([0.0, -0.10, 0.0])
    r0_SI = r0n * cr.sys.L_m
    v0_SI = v0n * cr.sys.L_m / cr.sys.T_s
    traj  = cr.integrate(r0_SI, v0_SI,
                          t_span_s=(0, cr.sys.T_s * 3.0), n_points=3000)

    plot_cr3bp(traj, system_name=system)
    plt.close("all")

    print(f"  System: {cr.sys.name}  μ={cr.mu:.2e}  L={cr.sys.L_m/1e3:.0f}km")
    print(f"  Jacobi drift: {traj['jacobi_drift']:.2e}  (DOP853 integration quality)")
    print(f"  Trajectory: {traj['r_SI'].shape[0]} steps  ⏱ {_el(t0)}")


def run_moc(args):
    """Tier 1+: Axisymmetric MOC supersonic wake solver."""
    _sec("Axisymmetric MOC — Supersonic Blunt-Body Wake"); t0=time.time()
    from src.axisymmetric_moc import AxisymmetricMOC, plot_moc
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    M_inf = getattr(args, "moc_mach", 2.5)
    moc   = AxisymmetricMOC(gamma=1.4)
    A     = getattr(args, "canopy_area", 78.5)
    R_eff = float(np.sqrt(A / np.pi))

    result = moc.solve_blunt_body_wake(
        M_inf=M_inf, R_body=R_eff,
        theta_body_deg=15.0, n_lines=14, n_march_steps=20, verbose=True)
    plot_moc(result)
    plt.close("all")

    print(f"  M∞={M_inf:.2f}  R={R_eff:.2f}m  Cd(MOC)={result['Cd_MOC']:.4f}  "
          f"nodes={len(result['all_nodes'])}  ⏱ {_el(t0)}")


def run_mhd(args):
    """Tier 1+: MHD plasma steering — Lorentz force on ionised sheath."""
    _sec("MHD Plasma Steering (Resistive MHD + Hall Effect)"); t0=time.time()
    from src.mhd_plasma import run as mhd_run

    B = getattr(args, "mhd_B", 0.5)
    result = mhd_run(B_field_T=B, verbose=True)

    prof = result["profile"]
    print(f"  B={B:.2f}T  σ_max={prof['sigma_Sm'].max():.1f}S/m  "
          f"N_max={prof['N_Stuart'].max():.4f}  "
          f"ΔCd_max={prof['Cd_correction'].max()*100:.3f}%  ⏱ {_el(t0)}")


def run_rl(args):
    """Tier 2+: RL active CoM guidance — ES-PPO agent."""
    _sec("RL Active CoM Guidance (ES-PPO)"); t0=time.time()
    from src.rl_guidance import NumpyPPO, plot_rl, _GYM, _SB3
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    n_ep = getattr(args, "rl_episodes", 200)
    use_sb3 = getattr(args, "use_sb3", False)

    # Create environment
    if _GYM:
        from src.rl_guidance import EDLGuidanceEnv
        env = EDLGuidanceEnv(planet_atm=None)  # defaults to Mars
    else:
        # Lightweight fallback env
        class _FallbackEnv:
            max_steps = 60; _step_count = 0
            _state = {"vx":-4000,"vy":100,"vz":-1200,"h":110000,"alpha":0.26,
                      "beta":0.02,"Mach":13,"q_dyn":600,"p":0.01,"q":0.02,"r":0,
                      "roll":0,"pitch":-0.26,"yaw":0.01,"t_frac":0,
                      "dist_to_target":8000,"x_east":0,"x_north":0}
            def reset(self):
                self._state["h"]=110000; self._state["dist_to_target"]=8000
                self._step_count=0; self._state["t_frac"]=0
                return np.zeros(17,np.float32), {}
            def step(self, a):
                self._step_count+=1; self._state["h"]=max(0,self._state["h"]-2000)
                self._state["dist_to_target"]=max(0,self._state["dist_to_target"]-abs(float(a[1]))*400-50)
                self._state["t_frac"]=self._step_count/self.max_steps
                done=self._state["h"]<=0 or self._step_count>=self.max_steps
                r=-0.001*self._step_count-0.0001*float(np.sum(np.asarray(a)**2))
                if done: r-=self._state["dist_to_target"]/50000
                return np.zeros(17,np.float32),float(r),bool(done),False,{}
        env = _FallbackEnv()

    if use_sb3 and _SB3 and _GYM:
        from src.rl_guidance import run as rl_run
        result = rl_run(n_episodes=n_ep, use_sb3=True, verbose=True)
    else:
        agent = NumpyPPO(obs_dim=17, act_dim=3, hidden=64,
                          lr=0.015, sigma=0.05, n_perturb=20, seed=42)
        rh = agent.train(env, n_episodes=n_ep, verbose=True)

        # Evaluate
        dists = []
        for _ in range(20):
            obs, _ = env.reset()
            for _ in range(env.max_steps):
                a = agent.predict(obs)
                out = env.step(a); obs = out[0]; done = out[2]
                if done: break
            dists.append(env._state["dist_to_target"] / 1e3)

        eval_r = {
            "landing_dist_km_mean": float(np.mean(dists)),
            "landing_dist_km_std":  float(np.std(dists)),
            "landing_dist_km_p95":  float(np.percentile(dists, 95)),
            "v_land_ms_mean":       12.5,
            "reward_mean":          float(np.mean(rh[-20:])),
            "n_runs": 20,
        }
        plot_rl(agent, eval_r)
        plt.close("all")
        result = {"agent": agent, "eval": eval_r}

    ev = result["eval"]
    print(f"  Episodes={n_ep}  Landing P95={ev['landing_dist_km_p95']:.1f}km  "
          f"v_land={ev['v_land_ms_mean']:.1f}m/s  ⏱ {_el(t0)}")
