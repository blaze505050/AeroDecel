"""
design_calc.py — Parachute Design Calculator
=============================================
Solves the inverse problem: given performance requirements, compute the
minimum canopy specifications that meet them.

Problem statement
-----------------
GIVEN:    target landing velocity v_target [m/s]
          deployment altitude h₀ [m]
          payload mass m [kg]
          deployment velocity v₀ [m/s]
          inflation time t_infl [s]
          canopy type

FIND:     minimum canopy area A_inf [m²]
          required Cd (if area is fixed)
          suspension line rated load [N] from MIL-HDBK-1791
          diameter D_nominal [m]
          risers, pack volume, weight estimates

Inversion method
----------------
The terminal velocity under a fully-open canopy defines the inverse map:

    v_term = sqrt(2 · m · g / (ρ · Cd · A_inf))

→  A_inf = 2 · m · g / (ρ · v_term² · Cd)    [direct]

For a more accurate answer, brentq solves the full ODE to find A_inf such
that the simulated landing velocity matches v_target exactly, accounting for
the altitude-varying density profile and the transient inflation dynamics.

Extended capabilities
---------------------
  • Multi-target sweep: v_land vs A_inf relationship across mass/alt grid
  • Safety margin analysis: structural loads from opening shock
  • Trade-off dashboard: area vs mass vs terminal velocity
  • Pack volume and weight estimates (Fruehauf empirical formulae)
  • Export: CSV + JSON + engineering datasheet (HTML)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density
from src.calibrate_cd import _simulate, _logistic_A
from src.opening_shock import (analyse as shock_analyse, cla_knacke, cla_milspec,
                                CANOPY_TYPES, CanopyType)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CANOPY SIZING FORMULAE
# ══════════════════════════════════════════════════════════════════════════════

def nominal_diameter(area_m2: float) -> float:
    """Nominal (flat) canopy diameter [m] from area."""
    return float(np.sqrt(4 * area_m2 / np.pi))


def constructed_diameter(area_m2: float, gore_factor: float = 1.08) -> float:
    """
    Constructed diameter (fabric cut diameter) accounting for gore geometry.
    gore_factor ≈ 1.05–1.12 depending on canopy type.
    """
    return nominal_diameter(area_m2) * gore_factor


def pack_volume_m3(area_m2: float, Cd: float = 1.35,
                   pack_density: float = 1100.0) -> float:
    """
    Pack volume estimate [m³] using Fruehauf empirical formula:
        V_pack ≈ C_pv · (A_inf)^1.5 / pack_density

    where pack_density is the bulk density of the packed canopy [kg/m³].
    Reference: Fruehauf (1978) parachute packing data.
    """
    C_pv = 0.009   # empirical constant for round canopies
    return float(C_pv * (area_m2 ** 1.5) / pack_density)


def pack_weight_kg(area_m2: float, fabric_gsm: float = 44.0) -> float:
    """
    Canopy fabric weight [kg] from area and fabric GSM.
    Includes 15% overhead for lines, risers, connectors, deployment bag.
    """
    fabric_kg = (area_m2 * fabric_gsm / 1000.0) * 1.15
    return float(fabric_kg)


def suspension_line_rated(mass_kg: float, Cd: float, A_inf: float,
                           t_infl: float, v_deploy: float,
                           h_deploy: float, canopy_type: str,
                           sf_target: float = 2.5) -> float:
    """
    Required suspension line rated tensile load [N] per line for sf_target margin.
    Assumes 28 lines (standard round canopy).
    """
    canopy = CANOPY_TYPES[canopy_type]
    rho    = density(h_deploy)
    v_term = np.sqrt(2 * mass_kg * cfg.GRAVITY / max(rho * Cd * A_inf, 1e-6))
    F_st   = 0.5 * rho * v_deploy**2 * Cd * A_inf
    cla_k  = cla_knacke(v_deploy, v_term, t_infl, canopy)
    cla_m  = cla_milspec(v_deploy, rho, mass_kg, A_inf, Cd, t_infl, canopy)
    F_pk   = F_st * max(cla_k, cla_m)
    N_lines = 28
    return float(F_pk * sf_target / N_lines)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CANOPY AREA SOLVER  (brentq ODE inversion)
# ══════════════════════════════════════════════════════════════════════════════

def solve_area(
    target_v:   float,
    mass:       float   = None,
    alt0:       float   = None,
    v0:         float   = None,
    Cd:         float   = None,
    t_infl:     float   = 2.5,
    A_bounds:   tuple   = (1.0, 500.0),
    dt:         float   = 0.1,
    verbose:    bool    = True,
) -> dict:
    """
    Solve for minimum A_inf such that landing velocity ≤ target_v.

    Uses brentq on the ODE forward simulation.

    Parameters
    ----------
    target_v   : target landing velocity [m/s]
    mass       : payload mass [kg]
    alt0       : deployment altitude [m AGL]
    v0         : deployment velocity [m/s]
    Cd         : drag coefficient
    t_infl     : inflation time [s]
    A_bounds   : search bracket for area [m²]
    dt         : ODE integration step [s]

    Returns
    -------
    dict with A_inf, D_nominal, v_land_actual, iterations, CI bounds
    """
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Cd   = Cd   or cfg.CD_INITIAL

    if verbose:
        print(f"\n[DesignCalc] Solving for A_inf → v_land ≤ {target_v:.2f} m/s")
        print(f"  mass={mass}kg  alt0={alt0}m  v0={v0}m/s  Cd={Cd}  t_infl={t_infl}s")

    n_evals = [0]

    def objective(A_inf: float) -> float:
        n_evals[0] += 1
        r = _simulate(Cd=Cd, mass=mass, alt0=alt0, v0=v0, Am=A_inf, ti=t_infl, dt=dt)
        return r["landing_velocity"] - target_v

    # Validate bracket
    vlo = objective(A_bounds[0]) + target_v
    vhi = objective(A_bounds[1]) + target_v

    if verbose:
        print(f"  Bracket: A={A_bounds[0]}m² → v={vlo:.2f}  |  A={A_bounds[1]}m² → v={vhi:.2f}")

    if vlo <= target_v:
        # Tiny canopy already meets target — something is wrong
        A_sol = float(A_bounds[0])
        note  = "WARNING: lower bound already meets target"
    elif vhi >= target_v:
        A_sol = float(A_bounds[1])
        note  = "WARNING: upper bound not sufficient — increase A_bounds"
    else:
        A_sol = brentq(objective, A_bounds[0], A_bounds[1], xtol=0.01, maxiter=200)
        note  = "OK"

    # Verify
    r_verify = _simulate(Cd=Cd, mass=mass, alt0=alt0, v0=v0, Am=A_sol, ti=t_infl, dt=0.05)
    v_actual = r_verify["landing_velocity"]
    residual = v_actual - target_v

    # ── Sensitivity analysis: ±10% Cd and mass ────────────────────────────────
    ci_bounds = {}
    for pname, delta in [("Cd+10%", (Cd*1.1, mass)), ("Cd-10%", (Cd*0.9, mass)),
                           ("m+10%", (Cd, mass*1.1)),  ("m-10%", (Cd, mass*0.9))]:
        try:
            r2 = _simulate(Cd=delta[0], mass=delta[1], alt0=alt0, v0=v0,
                            Am=A_sol, ti=t_infl, dt=0.1)
            ci_bounds[pname] = round(r2["landing_velocity"], 3)
        except Exception:
            ci_bounds[pname] = None

    if verbose:
        print(f"\n  ✓ A_inf = {A_sol:.3f} m²  D_nominal = {nominal_diameter(A_sol):.2f} m")
        print(f"  Actual v_land = {v_actual:.4f} m/s  (residual = {residual:+.4f} m/s)")
        print(f"  Brent evaluations: {n_evals[0]}")
        print(f"  Sensitivity (v_land with A_inf fixed):")
        for k, v in ci_bounds.items():
            print(f"    {k}: {v} m/s")

    return {
        "A_inf_m2":      round(A_sol, 4),
        "D_nominal_m":   round(nominal_diameter(A_sol), 3),
        "D_constructed_m": round(constructed_diameter(A_sol), 3),
        "v_land_target":  round(target_v, 3),
        "v_land_actual":  round(v_actual, 4),
        "residual_ms":    round(residual, 5),
        "n_evals":        n_evals[0],
        "pack_volume_m3": round(pack_volume_m3(A_sol), 5),
        "pack_weight_kg": round(pack_weight_kg(A_sol), 3),
        "sensitivity":    ci_bounds,
        "note":           note,
        "inputs": {
            "mass_kg": mass, "alt0_m": alt0, "v0_ms": v0,
            "Cd": Cd, "t_infl_s": t_infl, "target_v_ms": target_v,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Cd SOLVER  (find Cd given fixed area and target landing velocity)
# ══════════════════════════════════════════════════════════════════════════════

def solve_cd(
    target_v:   float,
    A_inf:      float,
    mass:       float = None,
    alt0:       float = None,
    v0:         float = None,
    t_infl:     float = 2.5,
    verbose:    bool  = True,
) -> dict:
    """Solve for required Cd given fixed canopy area and target v_land."""
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL

    def obj(Cd):
        r = _simulate(Cd=Cd, mass=mass, alt0=alt0, v0=v0, Am=A_inf, ti=t_infl, dt=0.1)
        return r["landing_velocity"] - target_v

    vlo = _simulate(0.05, mass=mass, alt0=alt0, v0=v0, Am=A_inf, ti=t_infl)["landing_velocity"]
    vhi = _simulate(5.0,  mass=mass, alt0=alt0, v0=v0, Am=A_inf, ti=t_infl)["landing_velocity"]

    if (vlo - target_v) * (vhi - target_v) > 0:
        return {"error": f"No solution in Cd ∈ [0.05, 5.0] for v_land={target_v}",
                "v_lo": vlo, "v_hi": vhi}

    Cd_sol = brentq(obj, 0.05, 5.0, xtol=1e-4, maxiter=200)
    r_v    = _simulate(Cd=Cd_sol, mass=mass, alt0=alt0, v0=v0, Am=A_inf, ti=t_infl, dt=0.05)

    if verbose:
        print(f"\n[DesignCalc] Cd solution: Cd = {Cd_sol:.5f}  "
              f"v_land = {r_v['landing_velocity']:.4f} m/s")

    return {
        "Cd_required":   round(Cd_sol, 5),
        "v_land_actual": round(r_v["landing_velocity"], 4),
        "A_inf_m2":      A_inf,
        "inputs":        {"mass_kg": mass, "alt0_m": alt0, "v0_ms": v0,
                          "t_infl_s": t_infl, "target_v_ms": target_v},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PARAMETER SWEEP DASHBOARD DATA
# ══════════════════════════════════════════════════════════════════════════════

def sweep_performance(
    mass_values:   list[float],
    alt_values:    list[float],
    Cd:            float  = None,
    t_infl:        float  = 2.5,
    target_v:      float  = 5.0,
    A_bounds:      tuple  = (1.0, 400.0),
) -> pd.DataFrame:
    """
    Compute required A_inf for a grid of (mass, alt) combinations.
    Returns DataFrame for surface/contour plotting.
    """
    Cd = Cd or cfg.CD_INITIAL
    records = []
    for mass in mass_values:
        for alt in alt_values:
            try:
                r = solve_area(target_v=target_v, mass=mass, alt0=alt,
                               Cd=Cd, t_infl=t_infl, A_bounds=A_bounds,
                               verbose=False)
                rho  = density(alt)
                v_t  = np.sqrt(2*mass*cfg.GRAVITY / max(rho*Cd*r["A_inf_m2"],1e-6))
                records.append({
                    "mass_kg":       mass,
                    "alt_m":         alt,
                    "A_inf_m2":      r["A_inf_m2"],
                    "D_nominal_m":   r["D_nominal_m"],
                    "v_land_ms":     r["v_land_actual"],
                    "v_terminal_ms": round(v_t, 3),
                    "pack_vol_m3":   r["pack_volume_m3"],
                    "pack_wt_kg":    r["pack_weight_kg"],
                })
            except Exception as e:
                records.append({"mass_kg": mass, "alt_m": alt,
                                 "error": str(e), "A_inf_m2": np.nan})

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HTML DATASHEET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_datasheet(
    area_result:   dict,
    shock_result   = None,
    sweep_df:      pd.DataFrame | None = None,
    save_path:     Path | None = None,
) -> Path:
    """
    Generate a self-contained HTML engineering datasheet summarising
    all design calculator outputs.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    inp = area_result.get("inputs", {})

    def row(label, val, unit="", highlight=False):
        style = "background:#0a1830;color:#00d4ff;" if highlight else ""
        return f"<tr><td>{label}</td><td style='{style}'>{val}</td><td>{unit}</td></tr>"

    # Structural table
    struct_rows = ""
    if shock_result is not None:
        for c in shock_result.to_dict().get("structural_components", []):
            color = ("#a8ff3e" if c["status"]=="OK" else
                     "#ffd700" if c["status"]=="WARNING" else "#ff4560")
            struct_rows += (f"<tr><td>{c['component']}</td>"
                            f"<td>{c['rated_N']:,.0f}</td>"
                            f"<td>{c['F_peak_N']:,.0f}</td>"
                            f"<td style='color:{color}'>{c['safety_factor']:.2f}</td>"
                            f"<td style='color:{color}'>{c['status']}</td></tr>")

    # Sweep table
    sweep_tbl = ""
    if sweep_df is not None and len(sweep_df):
        hdrs = "<tr>" + "".join(f"<th>{c}</th>" for c in sweep_df.columns) + "</tr>"
        body = ""
        for _, row_ in sweep_df.iterrows():
            body += "<tr>" + "".join(f"<td>{v:.3f}</td>" for v in row_.values) + "</tr>"
        sweep_tbl = f"<table><thead>{hdrs}</thead><tbody>{body}</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Parachute Design Datasheet</title>
<style>
  body{{font-family:'Courier New',monospace;background:#080c14;color:#c8d8f0;
       padding:32px 24px;max-width:960px;margin:0 auto;line-height:1.6}}
  h1{{font-size:20px;font-weight:600;margin-bottom:4px;color:#fff}}
  h2{{font-size:14px;font-weight:500;color:#7ab8f5;margin:24px 0 8px;
      border-bottom:1px solid #1a2744;padding-bottom:4px}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0}}
  th{{background:#1a2744;color:#7ab8f5;padding:5px 8px;text-align:left}}
  td{{padding:4px 8px;border-bottom:1px solid #0d1526;color:#9ab0cc}}
  tr:nth-child(even) td{{background:#0a1020}}
  .hi{{color:#00d4ff;font-weight:bold}}
  footer{{margin-top:32px;font-size:10px;color:#334455;border-top:1px solid #1a2744;padding-top:12px}}
</style>
</head>
<body>
<h1>Parachute Design Datasheet</h1>
<p style="color:#556688;font-size:11px">{ts}</p>

<h2>Design Requirements</h2>
<table><tbody>
  {row("Target landing velocity", f"{inp.get('target_v_ms','?'):.2f}", "m/s", True)}
  {row("Payload mass", f"{inp.get('mass_kg','?'):.1f}", "kg")}
  {row("Deployment altitude", f"{inp.get('alt0_m','?'):.0f}", "m AGL")}
  {row("Deployment velocity", f"{inp.get('v0_ms','?'):.1f}", "m/s")}
  {row("Drag coefficient Cd", f"{inp.get('Cd','?')}", "")}
  {row("Inflation time", f"{inp.get('t_infl_s','?'):.2f}", "s")}
</tbody></table>

<h2>Canopy Sizing</h2>
<table><tbody>
  {row("Minimum area A_inf", f"{area_result.get('A_inf_m2','?'):.3f}", "m²", True)}
  {row("Nominal diameter D₀", f"{area_result.get('D_nominal_m','?'):.3f}", "m")}
  {row("Constructed diameter", f"{area_result.get('D_constructed_m','?'):.3f}", "m")}
  {row("Actual v_land (ODE)", f"{area_result.get('v_land_actual','?'):.4f}", "m/s")}
  {row("Residual", f"{area_result.get('residual_ms','?'):+.5f}", "m/s")}
</tbody></table>

<h2>Weight & Volume Estimates</h2>
<table><tbody>
  {row("Pack volume", f"{area_result.get('pack_volume_m3','?'):.5f}", "m³")}
  {row("Pack weight (canopy)", f"{area_result.get('pack_weight_kg','?'):.3f}", "kg")}
</tbody></table>

{"<h2>Opening Shock (MIL-HDBK-1791)</h2><table><tbody>" +
 row("F_peak", f"{shock_result.F_peak_N/1e3:.3f}" if shock_result else "N/A", "kN", True) +
 row("CLA used", f"{shock_result.CLA_used:.4f}" if shock_result else "N/A", "") +
 row("Min safety factor", f"{shock_result.min_sf:.3f}" if shock_result else "N/A", "") +
 row("Compliant (SF≥1.5)", ("YES ✓" if shock_result and shock_result.min_sf>=1.5 else "NO ✗") if shock_result else "N/A", "") +
 "</tbody></table>" if shock_result else ""}

{"<h2>Structural Components</h2><table><thead><tr><th>Component</th><th>Rated [N]</th><th>F_peak [N]</th><th>SF</th><th>Status</th></tr></thead><tbody>" + struct_rows + "</tbody></table>" if struct_rows else ""}

{"<h2>Performance Sweep</h2>" + sweep_tbl if sweep_tbl else ""}

<footer>
  AI-Driven Parachute Design Calculator &nbsp;·&nbsp;
  m·dv/dt = mg − ½ρv²CdA(t) &nbsp;·&nbsp; {ts}
</footer>
</body>
</html>"""

    sp = save_path or cfg.OUTPUTS_DIR / "design_datasheet.html"
    sp.write_text(html, encoding="utf-8")
    print(f"  ✓ HTML datasheet saved: {sp}")
    return sp


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_design(
    area_result: dict,
    sweep_df:    pd.DataFrame | None = None,
    shock_result = None,
    save_path:   Path | None = None,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

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

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.06, right=0.97)

    def ax(r, c): return fig.add_subplot(gs[r, c])
    def style(a, t, xl, yl):
        a.set_title(t, fontweight="bold", pad=5, fontsize=9)
        a.set_xlabel(xl, fontsize=8); a.set_ylabel(yl, fontsize=8)
        a.grid(True, alpha=0.3); a.spines[["top","right"]].set_visible(False)

    inp    = area_result.get("inputs", {})
    A_sol  = area_result["A_inf_m2"]
    v_sol  = area_result["v_land_actual"]
    target = area_result["v_land_target"]
    mass   = inp.get("mass_kg", cfg.PARACHUTE_MASS)
    alt0   = inp.get("alt0_m",  cfg.INITIAL_ALT)
    Cd     = inp.get("Cd",      cfg.CD_INITIAL)
    ti     = inp.get("t_infl_s", 2.5)

    # ── P0: v_land vs A_inf curve ─────────────────────────────────────────────
    ax0 = ax(0, 0)
    A_arr = np.linspace(max(1.0, A_sol*0.2), A_sol*2.5, 60)
    v_arr = []
    for A_ in A_arr:
        try:
            r_ = _simulate(Cd=Cd, mass=mass, alt0=alt0, Am=A_, ti=ti, dt=0.15)
            v_arr.append(r_["landing_velocity"])
        except Exception:
            v_arr.append(np.nan)
    v_arr = np.array(v_arr)

    ax0.plot(A_arr, v_arr, color=C1, lw=2.0)
    ax0.axhline(target, color=C3, lw=1.2, ls="--", label=f"Target v={target} m/s")
    ax0.axvline(A_sol, color=C2, lw=1.2, ls="--", label=f"A_inf={A_sol:.1f} m²")
    ax0.scatter([A_sol], [v_sol], color=C2, s=80, zorder=5, marker="*")
    ax0.legend(fontsize=7.5)
    style(ax0, "v_land vs canopy area", "A_inf [m²]", "Landing velocity [m/s]")

    # ── P1: v_land vs Cd (with fixed area) ───────────────────────────────────
    ax1 = ax(0, 1)
    Cd_arr = np.linspace(0.3, 2.5, 60)
    v_cd   = []
    for Cd_ in Cd_arr:
        try:
            r_ = _simulate(Cd=Cd_, mass=mass, alt0=alt0, Am=A_sol, ti=ti, dt=0.15)
            v_cd.append(r_["landing_velocity"])
        except Exception:
            v_cd.append(np.nan)
    ax1.plot(Cd_arr, np.array(v_cd), color=C1, lw=2.0)
    ax1.axhline(target, color=C3, lw=1.2, ls="--", label=f"Target v={target} m/s")
    ax1.axvline(Cd, color=C2, lw=1.2, ls="--", label=f"Cd={Cd}")
    ax1.legend(fontsize=7.5)
    style(ax1, "v_land vs Cd (A fixed)", "Cd [—]", "Landing velocity [m/s]")

    # ── P2: Pack volume vs area ───────────────────────────────────────────────
    ax2 = ax(0, 2)
    vol_arr = [pack_volume_m3(A_)*1000 for A_ in A_arr]  # litres
    wt_arr  = [pack_weight_kg(A_) for A_ in A_arr]
    ax2.plot(A_arr, vol_arr, color=C1, lw=1.8, label="Pack vol [L]")
    ax2b = ax2.twinx()
    ax2b.plot(A_arr, wt_arr, color=C2, lw=1.5, ls="--", label="Pack wt [kg]")
    ax2b.tick_params(axis='y', labelcolor=C2 if not cfg.DARK_THEME else C2)
    ax2.axvline(A_sol, color=C3, lw=0.9, ls=":", alpha=0.7)
    ax2.set_xlabel("A_inf [m²]"); ax2.set_ylabel("Pack volume [L]")
    ax2b.set_ylabel("Pack weight [kg]")
    ax2.set_title("Pack volume & weight vs area", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # ── P3: Performance sweep contour ─────────────────────────────────────────
    ax3 = ax(0, 3)
    if sweep_df is not None and "A_inf_m2" in sweep_df.columns and not sweep_df["A_inf_m2"].isna().all():
        masses = np.sort(sweep_df["mass_kg"].unique())
        alts   = np.sort(sweep_df["alt_m"].unique())
        if len(masses) > 1 and len(alts) > 1:
            Z = (sweep_df.pivot_table(values="A_inf_m2",
                                       index="mass_kg", columns="alt_m")
                 .fillna(np.nan).values)
            cs = ax3.contourf(alts, masses, Z, levels=12,
                              cmap="YlOrRd" if not cfg.DARK_THEME else "plasma")
            plt.colorbar(cs, ax=ax3, label="Required A_inf [m²]", pad=0.02)
            ax3.scatter([alt0], [mass], s=80, color=C3, zorder=5,
                        marker="*", label="Current design")
            ax3.legend(fontsize=7.5)
    ax3.set_xlabel("Altitude [m]"); ax3.set_ylabel("Mass [kg]")
    style(ax3, f"Required area for v_land≤{target} m/s", "Altitude [m]", "Mass [kg]")

    # ── P4-5: Shock force history ──────────────────────────────────────────────
    if shock_result is not None:
        df_sh = shock_result.df
        ax4 = ax(1, 0)
        ax4.fill_between(df_sh["time_s"], df_sh["F_shock_N"]/1e3, alpha=0.2, color="#ff4560")
        ax4.plot(df_sh["time_s"], df_sh["F_shock_N"]/1e3, color="#ff4560", lw=1.8)
        ax4.plot(df_sh["time_s"], df_sh["F_steady_N"]/1e3, color=C1, lw=1.2,
                 ls="--", alpha=0.7, label="Steady")
        ax4.legend(fontsize=7.5)
        style(ax4, "Opening shock F(t)", "Time [s]", "Force [kN]")

        # SF bar chart
        ax5 = ax(1, 1)
        comps = shock_result.components
        sfs   = [c.effective_rated_N / max(shock_result.F_peak_N, 1) for c in comps]
        colors_bar = [C3 if sf>=1.5 else C2 if sf>=1.0 else "#ff4560" for sf in sfs]
        ax5.barh([c.name for c in comps], sfs, color=colors_bar, alpha=0.75)
        ax5.axvline(1.5, color=C1, lw=1.5, ls="--", label="SF=1.5 min")
        ax5.legend(fontsize=7.5); ax5.set_xlabel("Safety Factor")
        ax5.grid(True, alpha=0.3, axis="x")
        ax5.set_title("Structural safety factors", fontweight="bold")

    # ── P6: Summary table ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis("off")
    rows = [
        ("Target v_land",    f"{target:.2f} m/s"),
        ("Solved A_inf",     f"{A_sol:.4f} m²"),
        ("D nominal",        f"{area_result['D_nominal_m']:.3f} m"),
        ("D constructed",    f"{area_result['D_constructed_m']:.3f} m"),
        ("Actual v_land",    f"{v_sol:.4f} m/s"),
        ("",                 ""),
        ("Pack volume",      f"{area_result['pack_volume_m3']*1000:.2f} L"),
        ("Pack weight",      f"{area_result['pack_weight_kg']:.2f} kg"),
        ("",                 ""),
    ]
    if shock_result:
        rows += [
            ("Peak shock force",  f"{shock_result.F_peak_N/1e3:.3f} kN"),
            ("CLA",               f"{shock_result.CLA_used:.4f}"),
            ("Min SF",            f"{shock_result.min_sf:.3f}"),
            ("Compliant",         "YES ✓" if shock_result.min_sf>=1.5 else "NO ✗"),
        ]
    for j, (label, val) in enumerate(rows):
        hi = ("✓" in val or "A_inf" in label)
        c  = C3 if "✓" in val else "#ff4560" if "✗" in val else (C1 if hi else TEXT)
        ax6.text(0.02, 1-j*0.067, label, transform=ax6.transAxes, fontsize=8.5,
                 color=TEXT if cfg.DARK_THEME else "#555")
        ax6.text(0.98, 1-j*0.067, val, transform=ax6.transAxes, fontsize=8.5,
                 ha="right", color=c)
    ax6.set_title("Design summary", fontweight="bold")

    fig.text(0.5, 0.955,
             f"Parachute Design Calculator  —  "
             f"A_inf={A_sol:.2f}m²  D={area_result['D_nominal_m']:.2f}m  "
             f"v_land={v_sol:.3f}m/s  [target={target}m/s]",
             ha="center", fontsize=12, fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR / "design_calc.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Design plot saved: {sp}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(
    target_v:     float  = 5.0,
    mass:         float  = None,
    alt0:         float  = None,
    v0:           float  = None,
    Cd:           float  = None,
    t_infl:       float  = 2.5,
    canopy_type:  str    = "flat_circular",
    do_sweep:     bool   = True,
    do_shock:     bool   = True,
    verbose:      bool   = True,
) -> dict:
    """Run full design calculation pipeline."""
    import matplotlib; matplotlib.use("Agg")

    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Cd   = Cd   or cfg.CD_INITIAL

    if verbose:
        print(f"\n[DesignCalc] Target v_land ≤ {target_v} m/s")

    # ── Area solver ───────────────────────────────────────────────────────────
    area_result = solve_area(target_v=target_v, mass=mass, alt0=alt0,
                             v0=v0, Cd=Cd, t_infl=t_infl, verbose=verbose)
    A_sol = area_result["A_inf_m2"]

    # ── Opening shock ─────────────────────────────────────────────────────────
    shock_result = None
    if do_shock:
        if verbose: print(f"\n  Running opening shock analysis...")
        shock_result = shock_analyse(
            v_deploy=v0, h_deploy=alt0, mass=mass,
            A_inf=A_sol, Cd=Cd, t_infl=t_infl,
            canopy_type=canopy_type, verbose=verbose,
        )
        # Required suspension line strength
        req_line = suspension_line_rated(mass, Cd, A_sol, t_infl, v0, alt0,
                                          canopy_type, sf_target=2.5)
        area_result["req_line_rated_N"] = round(req_line, 1)
        if verbose:
            print(f"\n  Required line strength (SF≥2.5): {req_line:.0f} N per line (×28 lines)")

    # ── Performance sweep ─────────────────────────────────────────────────────
    sweep_df = None
    if do_sweep:
        if verbose: print(f"\n  Running performance sweep...")
        mass_vals = np.linspace(max(5, mass*0.5), mass*2.0, 8).tolist()
        alt_vals  = np.linspace(max(100, alt0*0.3), alt0*1.5, 8).tolist()
        sweep_df  = sweep_performance(mass_vals, alt_vals, Cd=Cd, t_infl=t_infl,
                                       target_v=target_v)
        sweep_df.to_csv(cfg.OUTPUTS_DIR / "design_sweep.csv", index=False)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    def _safe(v):
        if isinstance(v, float): return round(v, 6)
        return v

    out = {k: _safe(v) for k, v in area_result.items() if not isinstance(v, dict)}
    out["inputs"] = area_result.get("inputs", {})
    (cfg.OUTPUTS_DIR / "design_result.json").write_text(json.dumps(out, indent=2))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_design(area_result, sweep_df=sweep_df, shock_result=shock_result)

    # ── Datasheet ─────────────────────────────────────────────────────────────
    generate_datasheet(area_result, shock_result=shock_result, sweep_df=sweep_df)

    return area_result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parachute Design Calculator",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target-v",    type=float, default=5.0, help="Target landing v [m/s]")
    p.add_argument("--mass",        type=float, default=None)
    p.add_argument("--alt",         type=float, default=None, help="Deployment altitude [m]")
    p.add_argument("--v0",          type=float, default=None, help="Deployment velocity [m/s]")
    p.add_argument("--Cd",          type=float, default=None)
    p.add_argument("--t-infl",      type=float, default=2.5)
    p.add_argument("--canopy-type", type=str,   default="flat_circular",
                   choices=list(CANOPY_TYPES.keys()))
    p.add_argument("--no-sweep",    action="store_true")
    p.add_argument("--no-shock",    action="store_true")
    a = p.parse_args()
    run(target_v=a.target_v, mass=a.mass, alt0=a.alt, v0=a.v0,
        Cd=a.Cd, t_infl=a.t_infl, canopy_type=a.canopy_type,
        do_sweep=not a.no_sweep, do_shock=not a.no_shock)
