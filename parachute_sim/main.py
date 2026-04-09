"""main.py — AeroDecel: AI-Driven Aerodynamic Deceleration Analysis v5.0"""
import argparse, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BANNER = """
+===========================================================================+
|   AERODECEL v{version}                                                    |
|   AI-Driven Aerodynamic Deceleration Analysis Framework                   |
|                                                                           |
|   Ph1:CV - Ph2:ODE - Ph3:PINN - Ph4:Dash - Ph5:MC - Ph6:Traj            |
|   Ph7:Multi-Stage - Ph8:Pendulum - Shock - Bayes - Design - Ingest       |
+===========================================================================+
  {eq}  |  ISA 7-layer  |  PINN {layers}
  Advanced: Re-correction - Mach(P-G) - Porosity - Added-mass - Buoyancy
"""

def _banner(): print(BANNER.format(
    version=cfg.AERODECEL_VERSION,
    eq=cfg.AERODECEL_EQ,
    layers=cfg.PINN_HIDDEN_LAYERS
))
def _sec(t): print(f"\n{'-'*72}\n  {t}\n{'-'*72}")
def _el(t0): return f"{time.time()-t0:.1f}s"

def run_pipeline(args):
    _banner()
    t_total = time.time()
    results = {}

    # Telemetry ingest (pre-step)
    tel_df = None
    if args.ingest:
        _sec("Telemetry Ingest")
        t0=time.time()
        from src.ingest_telemetry import run as r
        syn = (args.ingest == "synthetic")
        fp  = None if syn else args.ingest
        tel_df = r(file_path=fp, synthetic=syn, verbose=True)
        results["telemetry"] = tel_df
        print(f"  ⏱  {_el(t0)}")

    at_df=ode_df=pinn_df=at_model=loss_history=mc_agg=traj_df=None

    if args.phase_start<=1<=args.phase_end:
        _sec("Phase 1 — AeroDecel CV Extraction")
        t0=time.time()
        from src.phase1_cv import run as r1
        at_df=r1(video_path=args.video, synthetic=(args.synthetic or args.video is None))
        results["at_df"]=at_df; print(f"  ⏱  {_el(t0)}")

    if args.phase_start<=2<=args.phase_end:
        _sec("Phase 2 — AeroDecel ODE Simulation (Advanced Physics)")
        t0=time.time()
        from src.phase2_ode import run as r2
        ode_df,at_model=r2(at_df=at_df, use_advanced=not args.no_advanced)
        results["ode_df"]=ode_df; print(f"  ⏱  {_el(t0)}")

    if args.run_pinn and args.phase_start<=3<=args.phase_end:
        _sec("Phase 3 — AeroDecel PINN Cd(t) (Research-Grade)")
        t0=time.time()
        from src.phase3_pinn import run as r3
        pinn_df,loss_history=r3(ode_df=ode_df, at_df=at_df)
        results["pinn_df"]=pinn_df; print(f"  ⏱  {_el(t0)}")

    if args.phase_start<=4<=args.phase_end:
        _sec("Phase 4 — Dashboard")
        t0=time.time()
        from src.phase4_viz import run as r4
        r4(at_df=at_df, ode_df=ode_df, pinn_df=pinn_df, loss_history=loss_history)
        print(f"  ⏱  {_el(t0)}")

    if args.run_mc and args.phase_start<=5<=args.phase_end:
        _sec("Phase 5 — Monte Carlo UQ")
        t0=time.time()
        from src.phase5_montecarlo import run as r5, N_SAMPLES
        mc_agg,_=r5(at_df=at_df, n=args.mc_n or N_SAMPLES)
        results["mc_agg"]=mc_agg; print(f"  ⏱  {_el(t0)}")

    if args.run_traj and args.phase_start<=6<=args.phase_end:
        lbl="Open-Meteo" if (args.lat and args.lon) else "power-law"
        _sec(f"Phase 6 — 3D Trajectory [{lbl}]")
        t0=time.time()
        from src.phase6_trajectory import run as r6
        traj_df=r6(at_model=at_model, wind_speed=args.wind, wind_dir=args.wind_dir,
                   lat=args.lat, lon=args.lon, wind_csv=args.wind_csv,
                   mc_n=min(args.mc_n or 60, 60))
        results["traj_df"]=traj_df; print(f"  ⏱  {_el(t0)}")

    if args.run_multistage and args.phase_start<=7<=args.phase_end:
        _sec("Phase 7 — Multi-Stage Drogue → Main")
        t0=time.time()
        from src.phase7_multistage import run as r7
        ms_df,_=r7(drogue_area=args.drogue_area, drogue_Cd=args.drogue_cd,
                   drogue_infl=args.drogue_infl,
                   main_area=args.main_area or cfg.CANOPY_AREA_M2,
                   main_Cd=args.main_cd or cfg.CD_INITIAL,
                   main_infl=args.main_infl, main_alt=args.main_alt)
        results["ms_df"]=ms_df; print(f"  ⏱  {_el(t0)}")

    if args.run_pendulum and args.phase_start<=8<=args.phase_end:
        _sec("Phase 8 — Pendulum Oscillation")
        t0=time.time()
        from src.phase8_pendulum import run as r8
        r8(riser_length=args.riser_length, theta0_deg=args.theta0,
           zeta=args.zeta, wind_speed=args.wind, wind_dir=args.wind_dir,
           mc_n=min(args.mc_n or 60, 60))
        print(f"  ⏱  {_el(t0)}")

    # Analysis modules
    if args.shock:
        _sec("Opening Shock — MIL-HDBK-1791")
        t0=time.time()
        import matplotlib; matplotlib.use("Agg")
        from src.opening_shock import run as rsh
        rsh(v_deploy=args.v_deploy or cfg.INITIAL_VEL,
            h_deploy=args.h_deploy or cfg.INITIAL_ALT,
            mass=args.mass or cfg.PARACHUTE_MASS,
            A_inf=args.a_inf or cfg.CANOPY_AREA_M2,
            Cd=args.cd or cfg.CD_INITIAL,
            t_infl=args.t_infl, canopy_type=args.canopy_type, at_df=at_df)
        print(f"  ⏱  {_el(t0)}")

    if args.bayes:
        _sec("Bayesian Cd Estimation")
        t0=time.time()
        import matplotlib; matplotlib.use("Agg")
        import pandas as pd
        t_obs=v_obs=None
        if tel_df is not None:
            t_obs=tel_df["time_s"].values; v_obs=tel_df["velocity_ms"].values
        elif args.telemetry and Path(args.telemetry).exists():
            df=pd.read_csv(args.telemetry)
            t_obs=df.iloc[:,0].values; v_obs=df.iloc[:,1].values
        from src.bayes_cd import run as rbay
        rbay(t_obs=t_obs, v_obs=v_obs, n_params=args.n_params,
             n_walkers=args.n_walkers, n_steps=args.n_steps,
             n_burnin=args.n_burnin, prior=args.prior)
        print(f"  ⏱  {_el(t0)}")

    if args.ensemble:
        _sec(f"PINN Ensemble ({args.n_members} members, {args.strategy})")
        t0=time.time()
        import matplotlib; matplotlib.use("Agg")
        from src.pinn_ensemble import run as rens
        rens(ode_df=ode_df, at_df=at_df, n_members=args.n_members,
             strategy=args.strategy, n_epochs=args.epochs)
        print(f"  ⏱  {_el(t0)}")

    if args.design:
        _sec("Design Calculator")
        t0=time.time()
        import matplotlib; matplotlib.use("Agg")
        from src.design_calc import run as rdc
        rdc(target_v=args.target_v, mass=args.mass or cfg.PARACHUTE_MASS,
            alt0=args.alt0 or cfg.INITIAL_ALT, v0=args.v0 or cfg.INITIAL_VEL,
            Cd=args.cd or cfg.CD_INITIAL, t_infl=args.t_infl,
            canopy_type=args.canopy_type,
            do_sweep=not args.no_sweep, do_shock=not args.no_shock)
        print(f"  ⏱  {_el(t0)}")

    # HTML report
    if at_df is not None or ode_df is not None:
        _sec("Engineering Report")
        t0=time.time()
        from src.report_generator import generate
        generate(at_df=at_df, ode_df=ode_df, pinn_df=pinn_df, mc_agg=mc_agg, traj_df=traj_df)
        print(f"  ⏱  {_el(t0)}")

    out=cfg.OUTPUTS_DIR.resolve()
    print(f"\n{'═'*72}")
    print(f"  ✅ AeroDecel v{cfg.AERODECEL_VERSION} COMPLETE | {time.time()-t_total:.1f}s | 📁 {out}")
    outputs=[
        ("📊","dashboard.png"),("🪂","multistage_dashboard.png"),
        ("🔮","pendulum_dashboard.png"),("🌐","mc_dashboard.png"),
        ("🗺 ","trajectory_3d.png"),("⚡","opening_shock.png"),
        ("∫ ","bayes_cd_posterior.png"),("✨","pinn_ensemble.png"),
        ("📐","design_calc.png"),("📡","telemetry_ingest.png"),
        ("📄","engineering_report.html"),("📋","design_datasheet.html"),
        ("🗂 ","trajectory.kml"),
    ]
    for icon,fn in outputs:
        if (out/fn).exists(): print(f"  {icon} {fn}")
    print(f"{'═'*72}")
    return results


def main():
    p=argparse.ArgumentParser(description=f"AeroDecel v{cfg.AERODECEL_VERSION} — AI-Driven Aerodynamic Deceleration Analysis",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Pipeline
    p.add_argument("--synthetic",     action="store_true")
    p.add_argument("--full",          action="store_true", help="ALL phases + ALL modules")
    p.add_argument("--video",         type=Path,  default=None)
    p.add_argument("--phase",         type=int,   default=None, choices=range(1,9))
    p.add_argument("--from-phase",    type=int,   default=1)
    p.add_argument("--to-phase",      type=int,   default=8)
    p.add_argument("--no-pinn",       action="store_true")
    p.add_argument("--no-mc",         action="store_true")
    p.add_argument("--no-traj",       action="store_true")
    p.add_argument("--no-multistage", action="store_true")
    p.add_argument("--no-pendulum",   action="store_true")
    p.add_argument("--no-advanced",   action="store_true", help="Disable advanced physics (Re/Mach/porosity/added-mass)")
    p.add_argument("--mc-n",          type=int,   default=None)
    # CV model
    p.add_argument("--cv-model",      type=str,   default="auto",
                   choices=["auto", "hsv", "yolo", "sam"],
                   help="Computer vision segmentation model")
    # Modules
    p.add_argument("--shock",    action="store_true")
    p.add_argument("--bayes",    action="store_true")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--design",   action="store_true")
    p.add_argument("--ingest",   type=str, default=None, metavar="FILE_OR_SYNTHETIC")
    # Physics
    p.add_argument("--mass",    type=float, default=None)
    p.add_argument("--alt0",    type=float, default=None)
    p.add_argument("--v0",      type=float, default=None)
    p.add_argument("--cd",      type=float, default=None)
    p.add_argument("--a-inf",   type=float, default=None)
    p.add_argument("--t-infl",  type=float, default=2.5)
    # Wind
    p.add_argument("--wind",     type=float, default=8.0)
    p.add_argument("--wind-dir", type=float, default=270.0)
    p.add_argument("--lat",      type=float, default=None)
    p.add_argument("--lon",      type=float, default=None)
    p.add_argument("--wind-csv", type=Path,  default=None)
    # Multi-stage
    p.add_argument("--drogue-area", type=float, default=2.5)
    p.add_argument("--drogue-cd",   type=float, default=0.97)
    p.add_argument("--drogue-infl", type=float, default=0.6)
    p.add_argument("--main-area",   type=float, default=None)
    p.add_argument("--main-cd",     type=float, default=None)
    p.add_argument("--main-infl",   type=float, default=2.5)
    p.add_argument("--main-alt",    type=float, default=300.0)
    # Pendulum
    p.add_argument("--riser-length", type=float, default=8.0)
    p.add_argument("--theta0",       type=float, default=14.0)
    p.add_argument("--zeta",         type=float, default=0.12)
    # Opening shock
    p.add_argument("--v-deploy",    type=float, default=None)
    p.add_argument("--h-deploy",    type=float, default=None)
    p.add_argument("--canopy-type", type=str,   default="flat_circular",
                   choices=["flat_circular","conical","extended_skirt","ribbon","drogue","ram_air"])
    # Bayesian
    p.add_argument("--telemetry",  type=str, default=None)
    p.add_argument("--n-params",   type=int, default=1, choices=[1,2,3])
    p.add_argument("--n-walkers",  type=int, default=32)
    p.add_argument("--n-steps",    type=int, default=2000)
    p.add_argument("--n-burnin",   type=int, default=500)
    p.add_argument("--prior",      type=str, default="lognormal", choices=["lognormal","uniform"])
    # PINN ensemble
    p.add_argument("--n-members",  type=int, default=5)
    p.add_argument("--strategy",   type=str, default="full",
                   choices=["seeds","width","depth","act","full"])
    p.add_argument("--epochs",     type=int, default=None)
    # Design
    p.add_argument("--target-v",   type=float, default=5.0)
    p.add_argument("--no-sweep",   action="store_true")
    p.add_argument("--no-shock",   action="store_true")
    # PINN training
    p.add_argument("--pinn-epochs", type=int, default=None)

    a=p.parse_args()

    # Apply config overrides
    if a.pinn_epochs: cfg.PINN_EPOCHS    = a.pinn_epochs
    if a.mass:        cfg.PARACHUTE_MASS = a.mass
    if a.alt0:        cfg.INITIAL_ALT    = a.alt0
    if a.v0:          cfg.INITIAL_VEL    = a.v0
    if a.cd:          cfg.CD_INITIAL     = a.cd
    if a.a_inf:       cfg.CANOPY_AREA_M2 = a.a_inf
    if a.cv_model:    cfg.CV_MODEL       = a.cv_model

    if a.phase: a.phase_start=a.phase; a.phase_end=a.phase
    else:       a.phase_start=a.from_phase; a.phase_end=a.to_phase

    if a.full:
        a.synthetic=True; a.shock=True; a.bayes=True
        a.ensemble=True;  a.design=True
        if a.ingest is None: a.ingest="synthetic"

    a.run_pinn       = not a.no_pinn
    a.run_mc         = not a.no_mc
    a.run_traj       = not a.no_traj
    a.run_multistage = not a.no_multistage
    a.run_pendulum   = not a.no_pendulum
    if a.video is None and not a.synthetic: a.synthetic=True

    run_pipeline(a)

if __name__=="__main__":
    main()
