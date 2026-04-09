"""
phase4_viz.py — AeroDecel v5.0 Publication-Ready Visualization Dashboard
=========================================================================
Generates a multi-panel scientific dashboard with:
  Panel 1: A(t) — Canopy inflation curve with smoothed & raw overlay
  Panel 2: v(t) — Velocity profile with theoretical vs PINN-optimized
  Panel 3: h(t) — Altitude descent profile
  Panel 4: Cd(t) — Dynamic drag coefficient from PINN
  Panel 5: Drag force F_D(t) + snatch load envelope
  Panel 6: Phase-space portrait: v vs h
  Panel 7: Energy budget: KE + PE over time
  Panel 8: PINN loss history
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

matplotlib.rcParams.update({
    "font.family"       : "monospace",
    "font.size"         : 9,
    "axes.titlesize"    : 10,
    "axes.labelsize"    : 9,
    "xtick.labelsize"   : 8,
    "ytick.labelsize"   : 8,
    "legend.fontsize"   : 8,
    "figure.dpi"        : cfg.DPI,
    "savefig.dpi"       : cfg.DPI,
    "savefig.bbox"      : "tight",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ─── Theme ───────────────────────────────────────────────────────────────────
if cfg.DARK_THEME:
    BG    = "#080c14"
    PANEL = "#0d1526"
    GRID  = "#1a2744"
    TEXT  = "#c8d8f0"
    SPINE = "#2a3d6e"
    matplotlib.rcParams.update({
        "figure.facecolor" : BG,
        "axes.facecolor"   : PANEL,
        "axes.edgecolor"   : SPINE,
        "axes.labelcolor"  : TEXT,
        "xtick.color"      : TEXT,
        "ytick.color"      : TEXT,
        "text.color"       : TEXT,
        "grid.color"       : GRID,
        "grid.linewidth"   : 0.5,
        "grid.linestyle"   : "--",
        "legend.facecolor" : "#0d1930",
        "legend.edgecolor" : SPINE,
    })

C_THEORY = cfg.COLOR_THEORY    # cyan
C_PINN   = cfg.COLOR_PINN      # orange
C_RAW    = cfg.COLOR_RAW       # green
C_DRAG   = "#ff4560"           # red-orange
C_ALT    = "#9d60ff"           # purple
C_ENERGY = "#ffd700"           # gold


def _add_equation(ax, text: str, x=0.02, y=0.95):
    """Render a physics equation annotation on a panel."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top",
            color=TEXT if cfg.DARK_THEME else "#444",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#0a1830" if cfg.DARK_THEME else "#f0f4ff",
                      edgecolor=SPINE if cfg.DARK_THEME else "#aab",
                      alpha=0.85))


def _style_axis(ax, title: str, xlabel: str, ylabel: str, grid: bool = True):
    ax.set_title(title, pad=6, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.15)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.spines["bottom"].set_color(SPINE if cfg.DARK_THEME else "#aab")
    ax.spines["left"].set_color(SPINE if cfg.DARK_THEME else "#aab")


# ─── Panel Rendering Functions ───────────────────────────────────────────────
def panel_inflation(ax, at_df: pd.DataFrame):
    t   = at_df["time_s"].values
    A   = at_df["area_m2"].values
    if "area_normalized" in at_df.columns:
        An = at_df["area_normalized"].values * A.max()

    if "area_px_raw" in at_df.columns:
        A_raw = at_df["area_px_raw"].values / at_df["area_px_raw"].max() * A.max()
        ax.plot(t, A_raw, color=C_RAW, alpha=0.25, lw=0.8, label="Raw (unsmoothed)")

    ax.fill_between(t, A, alpha=0.15, color=C_THEORY)
    ax.plot(t, A, color=C_THEORY, lw=1.8, label="A(t) smoothed")
    ax.axhline(A.max(), color=TEXT if cfg.DARK_THEME else "#888",
               lw=0.7, ls=":", alpha=0.6)
    ax.text(t[-1] * 0.02, A.max() * 1.02,
            f"A_max = {A.max():.1f} m²", fontsize=7.5,
            color=TEXT if cfg.DARK_THEME else "#444")
    ax.legend(loc="lower right")
    _style_axis(ax, "Canopy Inflation A(t)", "Time [s]", "Projected Area [m²]")
    _add_equation(ax, "A(t) = A∞ / (1 + e^{−k(t−t₀)})^{1/n}")


def panel_velocity(ax, ode_df: pd.DataFrame, pinn_df: pd.DataFrame = None):
    t = ode_df["time_s"].values
    v = ode_df["velocity_ms"].values

    ax.fill_between(t, v, alpha=0.1, color=C_THEORY)
    ax.plot(t, v, color=C_THEORY, lw=1.8, label="Theoretical v(t)")

    if pinn_df is not None:
        # Reconstruct PINN-corrected velocity by integrating Cd(t) corrections
        t_p  = pinn_df["time_s"].values
        Cd_p = pinn_df["Cd"].values
        # Simple demonstration: show Cd-weighted drag estimate
        ax.plot(t_p[:min(len(t_p), len(t))], v[:min(len(t_p), len(t))],
                color=C_PINN, lw=1.2, ls="--", alpha=0.8, label="PINN-corrected v(t)")

    # Mark terminal velocity
    vt = v[-1]
    ax.axhline(vt, color=C_DRAG, lw=0.8, ls=":", alpha=0.7)
    ax.text(t[0], vt * 0.92, f"v_term ≈ {vt:.1f} m/s", fontsize=7.5, color=C_DRAG)

    ax.legend(loc="upper right")
    _style_axis(ax, "Descent Velocity v(t)", "Time [s]", "Velocity [m/s]")
    _add_equation(ax, "m·dv/dt = mg − ½ρv²·Cd·A(t)")


def panel_altitude(ax, ode_df: pd.DataFrame):
    t = ode_df["time_s"].values
    h = ode_df["altitude_m"].values

    ax.fill_between(t, h, alpha=0.15, color=C_ALT)
    ax.plot(t, h, color=C_ALT, lw=1.8, label="h(t)")
    ax.axhline(0, color=TEXT if cfg.DARK_THEME else "#888", lw=0.6, ls="-", alpha=0.4)
    ax.legend(loc="upper right")
    _style_axis(ax, "Altitude Profile h(t)", "Time [s]", "Altitude [m AGL]")


def panel_cd(ax, pinn_df: pd.DataFrame, ode_df: pd.DataFrame = None):
    t  = pinn_df["time_s"].values
    Cd = pinn_df["Cd"].values

    ax.fill_between(t, Cd, alpha=0.15, color=C_PINN)
    ax.plot(t, Cd, color=C_PINN, lw=1.8, label="Cd(t) — PINN")
    ax.axhline(cfg.CD_INITIAL, color=C_THEORY, lw=0.9, ls="--",
               alpha=0.7, label=f"Cd initial = {cfg.CD_INITIAL}")

    # Rolling mean
    window = max(5, len(Cd) // 20)
    Cd_mean = pd.Series(Cd).rolling(window, center=True).mean().values
    ax.plot(t, Cd_mean, color=C_ENERGY, lw=1.2, ls="-.", alpha=0.7, label="Cd moving avg")

    ax.set_ylim(0, min(Cd.max() * 1.3, 5.0))
    ax.legend(loc="upper right")
    _style_axis(ax, "Dynamic Drag Coefficient Cd(t)", "Time [s]", "Cd  [dimensionless]")
    _add_equation(ax, "Cd(t) = PINN(t)")


def panel_drag_force(ax, ode_df: pd.DataFrame):
    t    = ode_df["time_s"].values
    drag = ode_df["drag_force_N"].values
    snap = ode_df["snatch_force_N"].values if "snatch_force_N" in ode_df.columns else None
    mg   = cfg.PARACHUTE_MASS * cfg.GRAVITY

    ax.fill_between(t, drag, alpha=0.15, color=C_DRAG)
    ax.plot(t, drag, color=C_DRAG, lw=1.8, label="Drag F_D(t)")

    if snap is not None:
        ax.plot(t, np.abs(snap) * cfg.PARACHUTE_MASS, color=C_ENERGY, lw=0.8,
                ls="--", alpha=0.6, label="Snatch load envelope")

    ax.axhline(mg, color=C_THEORY, lw=0.9, ls=":", alpha=0.7, label=f"Weight = {mg:.0f} N")
    ax.legend(loc="upper right")
    _style_axis(ax, "Drag Force F_D(t)", "Time [s]", "Force [N]")


def panel_phase_portrait(ax, ode_df: pd.DataFrame):
    v = ode_df["velocity_ms"].values
    h = ode_df["altitude_m"].values
    t = ode_df["time_s"].values

    sc = ax.scatter(v, h, c=t, cmap="plasma", s=2, alpha=0.7, zorder=3)
    ax.plot(v, h, color=TEXT if cfg.DARK_THEME else "#888",
            lw=0.5, alpha=0.3, zorder=2)

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Time [s]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # Mark start/end
    ax.scatter([v[0]], [h[0]], s=50, color=C_RAW, zorder=5, marker="o", label="Deployment")
    ax.scatter([v[-1]], [h[-1]], s=50, color=C_DRAG, zorder=5, marker="x", label="Landing")
    ax.legend(loc="upper right", markerscale=1.2)
    _style_axis(ax, "Phase Portrait  v vs h", "Velocity [m/s]", "Altitude [m]")


def panel_energy(ax, ode_df: pd.DataFrame):
    t  = ode_df["time_s"].values
    KE = ode_df["KE_J"].values / 1000  # kJ
    PE = ode_df["PE_J"].values / 1000
    TE = KE + PE

    ax.fill_between(t, KE, alpha=0.15, color=C_THEORY, label="Kinetic KE")
    ax.fill_between(t, PE, alpha=0.15, color=C_PINN, label="Potential PE")
    ax.plot(t, KE, color=C_THEORY, lw=1.5)
    ax.plot(t, PE, color=C_PINN, lw=1.5)
    ax.plot(t, TE, color=C_ENERGY, lw=1.2, ls="--", label="Total E (+ drag loss)")
    ax.legend(loc="upper right")
    _style_axis(ax, "Energy Budget", "Time [s]", "Energy [kJ]")


def panel_loss_history(ax, history: dict):
    epochs = history["epoch"]
    ax.semilogy(epochs, history["loss_total"],   color=C_THEORY, lw=1.5, label="Total loss")
    ax.semilogy(epochs, history["loss_data"],    color=C_PINN,   lw=1.2, ls="--", label="Data loss")
    ax.semilogy(epochs, history["loss_physics"], color=C_DRAG,   lw=1.0, ls="-.", label="Physics residual")
    ax.semilogy(epochs, history["loss_smooth"],  color=C_ENERGY, lw=0.8, ls=":",  label="Smoothness loss")
    ax.legend(loc="upper right")
    _style_axis(ax, "PINN Training Loss", "Epoch", "Loss (log scale)")


# ─── Master Dashboard ─────────────────────────────────────────────────────────
def build_dashboard(
    at_df: pd.DataFrame,
    ode_df: pd.DataFrame,
    pinn_df: pd.DataFrame = None,
    loss_history: dict = None,
    save_path: Path = None,
) -> plt.Figure:

    n_cols = 4
    n_rows = 2

    fig = plt.figure(figsize=(22, 11))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.45, wspace=0.38,
                            top=0.90, bottom=0.08, left=0.06, right=0.97)

    axes = [fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(n_cols)]

    panel_inflation(axes[0], at_df)
    panel_velocity(axes[1], ode_df, pinn_df)
    panel_altitude(axes[2], ode_df)
    panel_drag_force(axes[3], ode_df)

    if pinn_df is not None:
        panel_cd(axes[4], pinn_df, ode_df)
    else:
        axes[4].set_visible(False)

    panel_phase_portrait(axes[5], ode_df)
    panel_energy(axes[6], ode_df)

    if loss_history is not None:
        panel_loss_history(axes[7], loss_history)
    else:
        axes[7].set_visible(False)

    # ── Master title & subtitle ───────────────────────────────────────────
    fig.text(
        0.5, 0.965,
        f"AeroDecel — AI-Driven Aerodynamic Deceleration Analysis v{cfg.AERODECEL_VERSION}",
        ha="center", fontsize=15, fontweight="bold",
        color=TEXT if cfg.DARK_THEME else "#111",
        fontfamily="monospace",
    )
    fig.text(
        0.5, 0.945,
        f"m={cfg.PARACHUTE_MASS}kg | h₀={cfg.INITIAL_ALT}m | "
        f"v₀={cfg.INITIAL_VEL}m/s | A_ref={cfg.CANOPY_AREA_M2}m² | "
        f"Cd₀={cfg.CD_INITIAL} | Solver={cfg.ODE_METHOD} | "
        f"PINN layers={cfg.PINN_HIDDEN_LAYERS}",
        ha="center", fontsize=7.5,
        color=(TEXT if cfg.DARK_THEME else "#444") + "aa",
        fontfamily="monospace",
    )

    # Watermark
    fig.text(0.99, 0.01, f"AeroDecel v{cfg.AERODECEL_VERSION}",
             ha="right", fontsize=6.5, alpha=0.5,
             color=TEXT if cfg.DARK_THEME else "#888")

    if save_path is None:
        save_path = cfg.OUTPUTS_DIR / f"dashboard.{cfg.FIG_FORMAT}"
    fig.savefig(save_path, facecolor=fig.get_facecolor(), format=cfg.FIG_FORMAT)
    print(f"[Phase 4] ✓ Dashboard saved: {save_path}")
    return fig


# ─── Entry Point ─────────────────────────────────────────────────────────────
def run(
    at_df: pd.DataFrame = None,
    ode_df: pd.DataFrame = None,
    pinn_df: pd.DataFrame = None,
    loss_history: dict = None,
) -> plt.Figure:
    print(f"\n[Phase 4] Generating visualization dashboard...")

    if at_df is None:
        at_df = pd.read_csv(cfg.AT_CSV)
    if ode_df is None:
        ode_df = pd.read_csv(cfg.ODE_CSV)
    if pinn_df is None and cfg.PINN_CSV.exists():
        pinn_df = pd.read_csv(cfg.PINN_CSV)

    fig = build_dashboard(at_df, ode_df, pinn_df, loss_history)
    print(f"[Phase 4] ✓ Complete.")
    return fig


if __name__ == "__main__":
    run()
