"""
compare.py — Multi-Scenario Comparison Dashboard
=================================================
Run any number of parameter configurations side-by-side and produce
a single overlay dashboard + CSV table of key metrics.

Usage
-----
    python src/compare.py                          # built-in presets
    python src/compare.py --scenarios my.json      # custom JSON
    python src/compare.py --mass 60 80 100 120     # sweep over mass

JSON format
-----------
    [
      {"label": "Light 60kg",  "mass": 60,  "Cd": 1.2, "A_inf": 40},
      {"label": "Heavy 120kg", "mass": 120, "Cd": 1.5, "A_inf": 60}
    ]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.calibrate_cd import _simulate, _logistic_A


# ── Palette ───────────────────────────────────────────────────────────────────
_COLORS = ["#00d4ff","#ff6b35","#a8ff3e","#ffd700","#ff4560",
           "#9d60ff","#00e5b4","#f06292","#80cbc4","#ffab40"]


def _scenario_defaults(s: dict) -> dict:
    return {
        "label":   s.get("label", "Scenario"),
        "mass":    s.get("mass",   cfg.PARACHUTE_MASS),
        "alt0":    s.get("alt0",   cfg.INITIAL_ALT),
        "v0":      s.get("v0",     cfg.INITIAL_VEL),
        "Cd":      s.get("Cd",     cfg.CD_INITIAL),
        "A_inf":   s.get("A_inf",  cfg.CANOPY_AREA_M2),
        "t_infl":  s.get("t_infl", 2.5),
    }


def run_scenario(s: dict) -> dict:
    r = _simulate(Cd=s["Cd"], mass=s["mass"], alt0=s["alt0"],
                  v0=s["v0"], Am=s["A_inf"], ti=s["t_infl"], dt=0.05)
    t  = r["time"];  v = r["velocity"];  h = r["altitude"]
    A  = np.array([_logistic_A(ti, s["A_inf"], s["t_infl"]) for ti in t])
    drag = np.array([0.5 * 1.225 * vi**2 * s["Cd"] * Ai for vi, Ai in zip(v, A)])
    return {
        "label": s["label"], "t": t, "v": v, "h": h, "A": A, "drag": drag,
        "v_term":  round(float(v[-1]), 3),
        "t_land":  round(float(t[-1]), 2),
        "peak_drag": round(float(drag.max()), 1),
        "params": s,
    }


def build_dashboard(scenarios: list[dict], save_path: Path = None) -> plt.Figure:
    if cfg.DARK_THEME:
        plt.rcParams.update({"figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
                             "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
                             "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
                             "ytick.color":"#c8d8f0","grid.color":"#1a2744"})
    plt.rcParams.update({"font.family":"monospace","font.size":9})
    TEXT = "#c8d8f0" if cfg.DARK_THEME else "#111"

    results = [run_scenario(_scenario_defaults(s)) for s in scenarios]

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.06, right=0.97)

    def ax(r,c): return fig.add_subplot(gs[r,c])
    def style(a,t,xl,yl):
        a.set_title(t,fontweight="bold",pad=5,fontsize=9)
        a.set_xlabel(xl,fontsize=8); a.set_ylabel(yl,fontsize=8)
        a.grid(True,alpha=0.3); a.spines[["top","right"]].set_visible(False)

    ax0=ax(0,0); ax1=ax(0,1); ax2=ax(0,2)
    ax3=ax(1,0); ax4=ax(1,1); ax5=ax(1,2)

    for i, r in enumerate(results):
        c = _COLORS[i % len(_COLORS)]
        lw = 2.0 if i == 0 else 1.3
        ax0.plot(r["t"], r["v"],    color=c, lw=lw, label=r["label"])
        ax1.plot(r["t"], r["h"],    color=c, lw=lw, label=r["label"])
        ax2.plot(r["t"], r["A"],    color=c, lw=lw, label=r["label"])
        ax3.plot(r["t"], r["drag"]/1e3, color=c, lw=lw, label=r["label"])
        ax4.plot(r["h"], r["v"],    color=c, lw=lw, label=r["label"])  # phase portrait

    for a,t,xl,yl in [
        (ax0,"Velocity v(t)","Time [s]","Velocity [m/s]"),
        (ax1,"Altitude h(t)","Time [s]","Altitude [m]"),
        (ax2,"Canopy area A(t)","Time [s]","Area [m²]"),
        (ax3,"Drag force F(t)","Time [s]","Drag [kN]"),
        (ax4,"Phase portrait","Altitude [m]","Velocity [m/s]"),
    ]:
        style(a,t,xl,yl); a.legend(fontsize=7,ncol=2 if len(results)>4 else 1)

    # Metrics bar chart
    ax5.set_title("Key metrics comparison",fontweight="bold",fontsize=9)
    metrics = ["v_term", "t_land"]
    x = np.arange(len(results))
    labels = [r["label"] for r in results]
    colors = [_COLORS[i%len(_COLORS)] for i in range(len(results))]

    vt = [r["v_term"] for r in results]
    tl = [r["t_land"] for r in results]
    ax5.bar(x-0.2, vt, 0.35, color=colors, alpha=0.9, label="v_term [m/s]")
    ax5b = ax5.twinx()
    ax5b.bar(x+0.2, tl, 0.35, color=colors, alpha=0.5, hatch="//", label="t_land [s]")
    ax5.set_xticks(x); ax5.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax5.set_ylabel("Landing velocity [m/s]",fontsize=8)
    ax5b.set_ylabel("Descent time [s]",fontsize=8)
    ax5.grid(True,alpha=0.3)

    # Summary table (row 2 full width)
    ax_tbl = fig.add_subplot(gs[2,:])
    ax_tbl.axis("off")
    col_names = ["Scenario","Mass [kg]","Alt [m]","Cd","A_inf [m²]",
                 "t_infl [s]","v_term [m/s]","t_land [s]","Peak drag [N]"]
    rows = []
    for r in results:
        p = r["params"]
        rows.append([r["label"], p["mass"], p["alt0"], p["Cd"], p["A_inf"],
                     p["t_infl"], r["v_term"], r["t_land"], r["peak_drag"]])

    tbl = ax_tbl.table(cellText=rows, colLabels=col_names,
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.8)
    for (row,col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2a3d6e" if cfg.DARK_THEME else "#ccc")
        cell.set_facecolor("#1a2744" if row==0 else
                           ("#0a1020" if row%2==0 else "#0d1526"))
        cell.set_text_props(color=TEXT)

    fig.text(0.5,0.955,
             f"Multi-Scenario Comparison  —  {len(results)} configurations",
             ha="center",fontsize=13,fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR/"comparison_dashboard.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Comparison dashboard saved: {sp}")

    # Save metrics CSV
    df = pd.DataFrame([{
        "label": r["label"], "v_term_ms": r["v_term"],
        "t_land_s": r["t_land"], "peak_drag_N": r["peak_drag"],
        **{k:v for k,v in r["params"].items() if k!="label"}
    } for r in results])
    df.to_csv(cfg.OUTPUTS_DIR/"comparison_metrics.csv", index=False)
    print(f"  ✓ Metrics CSV saved: {cfg.OUTPUTS_DIR/'comparison_metrics.csv'}")
    return fig


# ── Prebuilt scenario sets ────────────────────────────────────────────────────

PRESETS = {
    "mass_sweep": [
        {"label":f"{m}kg", "mass":m, "Cd":1.35, "A_inf":50} for m in [50,70,90,110,130]
    ],
    "canopy_area": [
        {"label":f"A={A}m²","mass":80,"Cd":1.35,"A_inf":A} for A in [25,35,50,65,80]
    ],
    "cd_range": [
        {"label":f"Cd={Cd}","mass":80,"Cd":Cd,"A_inf":50} for Cd in [0.8,1.0,1.2,1.5,1.8,2.2]
    ],
    "altitude": [
        {"label":f"h={h}m","mass":80,"Cd":1.35,"A_inf":50,"alt0":h} for h in [400,600,800,1000,1500]
    ],
    "inflation_time": [
        {"label":f"ti={ti}s","mass":80,"Cd":1.35,"A_inf":50,"t_infl":ti} for ti in [1.0,1.5,2.5,4.0,6.0]
    ],
}


def run(scenarios=None, preset="mass_sweep", save_path=None):
    if scenarios is None:
        scenarios = PRESETS[preset]
    fig = build_dashboard(scenarios, save_path=save_path)
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=Path, default=None)
    p.add_argument("--preset",    type=str,  default="mass_sweep",
                   choices=list(PRESETS.keys()))
    p.add_argument("--mass",  type=float, nargs="+", default=None)
    p.add_argument("--cd",    type=float, nargs="+", default=None)
    p.add_argument("--a-inf", type=float, nargs="+", default=None)
    a = p.parse_args()

    if a.scenarios and a.scenarios.exists():
        scen = json.loads(a.scenarios.read_text())
    elif a.mass:
        scen = [{"label":f"{m}kg","mass":m} for m in a.mass]
    elif a.cd:
        scen = [{"label":f"Cd={c}","Cd":c} for c in a.cd]
    elif a.a_inf:
        scen = [{"label":f"A={A}m²","A_inf":A} for A in a.a_inf]
    else:
        scen = PRESETS[a.preset]

    run(scen)
