"""
report_generator.py — AeroDecel v5.0 Automated HTML Engineering Report
======================================================================
Generates a self-contained, single-file HTML report with all results,
charts (base64-embedded), tables, and physics equations.
No external dependencies at render time.
"""

import base64
import json
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _df_table(df: pd.DataFrame, cols: list = None, n_rows: int = 10) -> str:
    if cols:
        df = df[cols]
    df = df.head(n_rows).round(4)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def generate(
    at_df: pd.DataFrame = None,
    ode_df: pd.DataFrame = None,
    pinn_df: pd.DataFrame = None,
    mc_agg: dict = None,
    traj_df: pd.DataFrame = None,
    save_path: Path = None,
):
    save_path = save_path or cfg.OUTPUTS_DIR / "engineering_report.html"

    # Load available CSVs
    if at_df is None and cfg.AT_CSV.exists():
        at_df = pd.read_csv(cfg.AT_CSV)
    if ode_df is None and cfg.ODE_CSV.exists():
        ode_df = pd.read_csv(cfg.ODE_CSV)
    if pinn_df is None and cfg.PINN_CSV.exists():
        pinn_df = pd.read_csv(cfg.PINN_CSV)

    # Embed images
    img_dash  = _img_b64(cfg.OUTPUTS_DIR / "dashboard.png")
    img_mc    = _img_b64(cfg.OUTPUTS_DIR / "mc_dashboard.png")
    img_traj  = _img_b64(cfg.OUTPUTS_DIR / "trajectory_3d.png")

    # Key metrics
    if ode_df is not None:
        land_v  = float(ode_df["velocity_ms"].iloc[-1])
        land_t  = float(ode_df["time_s"].iloc[-1])
        peak_D  = float(ode_df["drag_force_N"].max())
        peak_acc= float(abs(ode_df["acceleration"].min())) if "acceleration" in ode_df.columns else 0
        max_A   = float(at_df["area_m2"].max()) if at_df is not None else cfg.CANOPY_AREA_M2
        mean_Cd = float(pinn_df["Cd"].mean()) if pinn_df is not None else cfg.CD_INITIAL
    else:
        land_v=land_t=peak_D=peak_acc=max_A=mean_Cd=0

    mc_v50 = mc_agg["land_v_p50"] if mc_agg else "N/A"
    mc_v95 = mc_agg["land_v_p95"] if mc_agg else "N/A"
    mc_sn95= mc_agg["snatch_p95"]  if mc_agg else "N/A"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AeroDecel — Engineering Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Courier New',monospace;background:#080c14;color:#c8d8f0;padding:32px 24px;max-width:1100px;margin:0 auto;line-height:1.65}}
  h1{{font-size:22px;font-weight:600;margin-bottom:6px;color:#fff}}
  h2{{font-size:16px;font-weight:500;margin:32px 0 10px;color:#7ab8f5;border-bottom:1px solid #1a2744;padding-bottom:5px}}
  h3{{font-size:13px;font-weight:500;margin:18px 0 7px;color:#a8d0f0}}
  p{{font-size:12px;color:#8899bb;margin-bottom:10px;line-height:1.7}}
  .eq{{background:#0d1526;border-left:3px solid #00d4ff;padding:10px 16px;font-size:13px;color:#00d4ff;margin:14px 0;font-family:'Courier New',monospace;border-radius:0 6px 6px 0}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}}
  .card{{background:#0d1526;border:1px solid #1a2744;border-radius:8px;padding:12px 14px}}
  .card .lbl{{font-size:10px;color:#556688;margin-bottom:3px}}
  .card .val{{font-size:20px;font-weight:600;color:#00d4ff}}
  .card .unt{{font-size:10px;color:#445566}}
  .card.warn .val{{color:#ff6b35}}
  img{{width:100%;border-radius:8px;border:1px solid #1a2744;margin:12px 0}}
  table{{width:100%;border-collapse:collapse;font-size:11px;margin:12px 0}}
  th{{background:#1a2744;color:#7ab8f5;padding:5px 8px;text-align:left;font-weight:500}}
  td{{padding:4px 8px;border-bottom:1px solid #0d1526;color:#9ab0cc}}
  tr:nth-child(even) td{{background:#0a1020}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:10px;margin-right:5px}}
  .bi{{background:#0e2040;color:#7ab8f5;border:1px solid #1a4080}}
  .bg{{background:#0e2e1a;color:#5dca85;border:1px solid #1a5030}}
  .bw{{background:#2e1e00;color:#f0a827;border:1px solid #604010}}
  .br{{background:#2e0e0e;color:#f05050;border:1px solid #601010}}
  footer{{margin-top:40px;padding-top:16px;border-top:1px solid #1a2744;font-size:10px;color:#334455;text-align:center}}
  .sens-bar{{display:flex;align-items:center;gap:8px;margin:5px 0}}
  .sens-lbl{{font-size:11px;color:#8899bb;width:80px}}
  .sens-fg{{background:#00d4ff;height:10px;border-radius:3px}}
  .sens-val{{font-size:10px;color:#556688}}
</style>
</head>
<body>

<h1>AeroDecel — AI-Driven Aerodynamic Deceleration Analysis v{cfg.AERODECEL_VERSION}</h1>
<p style="color:#556688;font-size:11px">Engineering Report &nbsp;·&nbsp; Generated {ts}</p>

<div style="margin:8px 0 20px">
  <span class="badge bi">ISA Atmosphere</span>
  <span class="badge bi">RK45 ODE</span>
  <span class="badge bg">PINN Cd(t)</span>
  <span class="badge bw">Monte Carlo UQ</span>
  <span class="badge bi">3D Trajectory</span>
</div>

<h2>1. Physics Framework</h2>
<div class="eq">m · dv/dt = mg − ½ · ρ(h) · v² · Cd(t) · A(t)</div>
<div class="eq">dh/dt = −v</div>
<div class="eq">ρ(h) = P(h) / (R_air · T(h))   [ISA 7-layer model]</div>
<div class="eq">A(t) = A∞ / [1 + exp(−k(t−t₀))]^(1/n)   [Generalized Logistic]</div>
<div class="eq">Cd(t) = PINN(t ; θ*)   [Physics-Informed Neural Network]</div>

<h2>2. Configuration</h2>
<div class="cards">
  <div class="card"><div class="lbl">mass</div><div class="val">{cfg.PARACHUTE_MASS}</div><div class="unt">kg</div></div>
  <div class="card"><div class="lbl">alt₀</div><div class="val">{cfg.INITIAL_ALT:.0f}</div><div class="unt">m AGL</div></div>
  <div class="card"><div class="lbl">v₀</div><div class="val">{cfg.INITIAL_VEL}</div><div class="unt">m/s</div></div>
  <div class="card"><div class="lbl">A_ref</div><div class="val">{cfg.CANOPY_AREA_M2}</div><div class="unt">m²</div></div>
  <div class="card"><div class="lbl">Cd initial</div><div class="val">{cfg.CD_INITIAL}</div><div class="unt">—</div></div>
  <div class="card"><div class="lbl">ODE solver</div><div class="val" style="font-size:14px">{cfg.ODE_METHOD}</div><div class="unt">—</div></div>
  <div class="card"><div class="lbl">PINN layers</div><div class="val" style="font-size:11px">{cfg.PINN_HIDDEN_LAYERS}</div><div class="unt">—</div></div>
  <div class="card"><div class="lbl">PINN epochs</div><div class="val">{cfg.PINN_EPOCHS}</div><div class="unt">—</div></div>
</div>

<h2>3. Key Results — Nominal Case</h2>
<div class="cards">
  <div class="card"><div class="lbl">terminal velocity</div><div class="val">{land_v:.2f}</div><div class="unt">m/s  ({land_v*3.6:.1f} km/h)</div></div>
  <div class="card"><div class="lbl">descent time</div><div class="val">{land_t:.1f}</div><div class="unt">seconds</div></div>
  <div class="card warn"><div class="lbl">peak drag force</div><div class="val">{peak_D:.0f}</div><div class="unt">N</div></div>
  <div class="card"><div class="lbl">peak decel.</div><div class="val">{peak_acc:.2f}</div><div class="unt">m/s²</div></div>
  <div class="card"><div class="lbl">max canopy area</div><div class="val">{max_A:.1f}</div><div class="unt">m²</div></div>
  <div class="card"><div class="lbl">mean Cd (PINN)</div><div class="val">{mean_Cd:.3f}</div><div class="unt">—</div></div>
</div>

<h2>4. Main Simulation Dashboard</h2>
{"<img src='"+img_dash+"' alt='Dashboard'>" if img_dash else "<p style='color:#ff4560'>Dashboard image not found — run Phase 4 first.</p>"}

<h2>5. Tabulated ODE Results (first 10 rows)</h2>
{_df_table(ode_df, ["time_s","velocity_ms","altitude_m","drag_force_N","area_m2","dynamic_press"]) if ode_df is not None else "<p>ODE results not available.</p>"}

<h2>6. PINN Drag Coefficient Cd(t)</h2>
{_df_table(pinn_df) if pinn_df is not None else "<p>PINN results not available.</p>"}
<p>The Physics-Informed Neural Network learns Cd(t) as a smooth function that satisfies both the measured velocity profile and the governing ODE residual simultaneously. The Cd peak during inflation corresponds to the high-drag transient when the canopy is rapidly pressurizing.</p>

<h2>7. Monte Carlo Uncertainty Quantification</h2>
{"<img src='"+img_mc+"' alt='MC Dashboard'>" if img_mc else "<p style='color:#ff6b35'>MC dashboard not available — run Phase 5.</p>"}
<div class="cards">
  <div class="card"><div class="lbl">landing v P50</div><div class="val">{f"{mc_v50:.2f}" if isinstance(mc_v50,float) else mc_v50}</div><div class="unt">m/s</div></div>
  <div class="card warn"><div class="lbl">landing v P95</div><div class="val">{f"{mc_v95:.2f}" if isinstance(mc_v95,float) else mc_v95}</div><div class="unt">m/s</div></div>
  <div class="card warn"><div class="lbl">snatch load P95</div><div class="val">{f"{mc_sn95:.0f}" if isinstance(mc_sn95,float) else mc_sn95}</div><div class="unt">N</div></div>
</div>
{_sensitivity_html(mc_agg) if mc_agg and "sensitivity" in mc_agg else ""}

<h2>8. 3D Wind-Drift Trajectory & Landing Zone</h2>
{"<img src='"+img_traj+"' alt='3D Trajectory'>" if img_traj else "<p style='color:#ff6b35'>3D trajectory not available — run Phase 6.</p>"}

<h2>9. ISA Atmosphere Validation</h2>
{_isa_table()}

<footer>
  AeroDecel v{cfg.AERODECEL_VERSION} — AI-Driven Aerodynamic Deceleration Analysis &nbsp;·&nbsp;
  Governing eq: m·dv/dt = mg − ½ρv²CdA(t) &nbsp;·&nbsp;
  {ts}
</footer>
</body>
</html>"""

    save_path.write_text(html, encoding="utf-8")
    print(f"[Report] ✓ HTML report saved: {save_path}")
    return save_path


def _sensitivity_html(mc_agg: dict) -> str:
    if not mc_agg or "sensitivity" not in mc_agg:
        return ""
    sens = mc_agg["sensitivity"]
    rows = ""
    for k, v in sens.items():
        w = int(v * 180)
        rows += (f'<div class="sens-bar"><span class="sens-lbl">{k}</span>'
                 f'<div class="sens-fg" style="width:{w}px"></div>'
                 f'<span class="sens-val">{v:.4f}</span></div>')
    return f"<h3>Sensitivity Ranking (|Pearson r| with landing velocity)</h3>{rows}"


def _isa_table() -> str:
    from src.atmosphere import temperature, pressure, density as rho, speed_of_sound
    rows = ""
    for alt in [0, 500, 1000, 2000, 5000, 8000, 11000, 15000]:
        T = temperature(alt); P = pressure(alt); r = rho(alt); a = speed_of_sound(alt)
        rows += f"<tr><td>{alt}</td><td>{T:.2f}</td><td>{P:.1f}</td><td>{r:.5f}</td><td>{a:.2f}</td></tr>"
    return (f"<table><thead><tr><th>Alt [m]</th><th>T [K]</th><th>P [Pa]</th>"
            f"<th>ρ [kg/m³]</th><th>a [m/s]</th></tr></thead><tbody>{rows}</tbody></table>")


if __name__ == "__main__":
    generate()
