"""
mach_cd.py — Mach-Number and Reynolds-Number Corrected Drag Coefficient
========================================================================
Improves physics accuracy by accounting for:

1. Mach correction (compressibility):
   Prandtl-Glauert  Cd(M) = Cd0 / sqrt(1 - M²)       [subsonic, M < 0.8]
   Karman-Tsien     Cd(M) = more accurate transonic correction
   Supersonic       Cd rises sharply past M=1 (shock drag)

2. Reynolds number correction (fabric canopy):
   At very low Re (<1e4), drag coefficient rises due to laminar separation.
   Cd_Re = Cd0 * (1 + k_Re / Re^0.2)

3. Porosity correction:
   Fabric canopies are not impermeable. Effective Cd:
   Cd_eff = Cd * (1 - porosity_factor * sqrt(v))

4. Angle-of-attack effect:
   During pendulum swing, the canopy presents a different cross-section.
   A_eff = A_inf * cos²(theta)   [already in phase8 — unified here]

All corrections are multiplicative and independent, applied in sequence.
"""
from __future__ import annotations
import numpy as np
from src.atmosphere import density, speed_of_sound, dynamic_viscosity


def mach_number(velocity_ms: float, altitude_m: float) -> float:
    a = speed_of_sound(altitude_m)
    return velocity_ms / max(a, 1.0)


def prandtl_glauert_factor(M: float) -> float:
    """Subsonic compressibility correction. Valid M < 0.8."""
    M = min(M, 0.79)  # clamp before singularity
    return 1.0 / max(np.sqrt(1.0 - M**2), 0.01)


def karman_tsien_factor(M: float, Cp0: float = -0.5) -> float:
    """
    Karman-Tsien transonic correction (more accurate than P-G near M=0.7).
    Cp_compressible = Cp0 / (sqrt(1-M²) + M²/(1+sqrt(1-M²)) * Cp0/2)
    Returns multiplicative factor on Cd.
    """
    M = min(M, 0.79)
    beta = max(np.sqrt(1.0 - M**2), 0.01)
    denom = beta + (M**2 / (1 + beta)) * Cp0 / 2
    return abs(Cp0 / max(abs(denom), 0.01) / abs(Cp0)) if Cp0 != 0 else 1.0


def reynolds_correction(Cd0: float, Re: float, k_Re: float = 2.0) -> float:
    """
    Low-Reynolds correction for fabric/bluff-body canopies.
    At Re < 1e5, separation is earlier → effective Cd rises.
    k_Re ≈ 1–3 for round canopies (calibrated empirically).
    """
    if Re < 1e3:
        return Cd0 * (1.0 + k_Re)
    return Cd0 * (1.0 + k_Re / max(Re, 1.0) ** 0.2)


def porosity_correction(Cd: float, velocity_ms: float,
                        porosity_k: float = 0.015) -> float:
    """
    Effective Cd accounting for fabric air permeability.
    Cd_eff = Cd * (1 - k_p * v)
    k_p ≈ 0.010–0.020 for standard nylon ripstop at typical descent speeds.
    Set k_p=0 for impermeable (plastic) canopies.
    """
    return max(0.05, Cd * max(0.0, 1.0 - porosity_k * abs(velocity_ms)))


def corrected_Cd(
    Cd_nominal:   float,
    velocity_ms:  float,
    altitude_m:   float,
    char_length_m: float = 8.0,   # canopy nominal diameter
    porosity_k:   float = 0.012,
    apply_mach:   bool  = True,
    apply_re:     bool  = True,
    apply_porosity: bool = True,
) -> dict:
    """
    Compute fully corrected drag coefficient with all physical effects.

    Parameters
    ----------
    Cd_nominal   : baseline drag coefficient at low speed
    velocity_ms  : current descent velocity
    altitude_m   : current altitude AGL
    char_length_m: characteristic length for Re (canopy diameter)
    porosity_k   : fabric porosity coefficient
    apply_mach   : apply Mach number correction
    apply_re     : apply Reynolds number correction
    apply_porosity: apply porosity correction

    Returns
    -------
    dict with: Cd_corrected, Mach, Re, correction_mach, correction_re,
               correction_porosity
    """
    v   = abs(velocity_ms)
    h   = max(0.0, altitude_m)
    rho = density(h)
    mu  = dynamic_viscosity(h)
    a   = speed_of_sound(h)

    M  = v / max(a, 1.0)
    Re = rho * v * char_length_m / max(mu, 1e-9)

    Cd = Cd_nominal
    corr_M  = 1.0
    corr_Re = 1.0
    corr_po = 1.0

    # Mach correction (only significant above M ~ 0.1 for canopies)
    if apply_mach and M > 0.05:
        corr_M = prandtl_glauert_factor(M)
        Cd = Cd * corr_M

    # Reynolds correction (mainly at very low Re or very low altitude)
    if apply_re and Re < 5e5:
        Cd_re = reynolds_correction(Cd, Re)
        corr_Re = Cd_re / max(Cd, 1e-6)
        Cd = Cd_re

    # Porosity correction
    if apply_porosity and porosity_k > 0:
        Cd_po = porosity_correction(Cd, v, porosity_k)
        corr_po = Cd_po / max(Cd, 1e-6)
        Cd = Cd_po

    return {
        "Cd_corrected":       round(float(Cd), 6),
        "Cd_nominal":         round(float(Cd_nominal), 6),
        "Mach":               round(float(M), 5),
        "Reynolds":           round(float(Re), 0),
        "correction_mach":    round(float(corr_M), 5),
        "correction_reynolds":round(float(corr_Re), 5),
        "correction_porosity":round(float(corr_po), 5),
        "total_correction":   round(float(Cd / max(Cd_nominal, 1e-6)), 5),
    }


def correction_profile(
    Cd_nominal: float,
    v_range:    tuple = (2, 80),
    altitude_m: float = 1000.0,
    char_length: float = 8.0,
    porosity_k: float = 0.012,
    n_points:   int   = 200,
) -> "pd.DataFrame":
    """
    Compute Cd correction profile over a velocity range.
    Returns DataFrame for plotting.
    """
    import pandas as pd
    vs = np.linspace(v_range[0], v_range[1], n_points)
    records = []
    for v in vs:
        r = corrected_Cd(Cd_nominal, v, altitude_m,
                         char_length_m=char_length, porosity_k=porosity_k)
        r["velocity_ms"] = round(float(v), 3)
        records.append(r)
    return pd.DataFrame(records)


def plot_corrections(Cd_nominal: float = 1.35, altitude_m: float = 1000.0,
                     save_path=None):
    """Plot Cd correction breakdown across velocity range."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import config as cfg

    if cfg.DARK_THEME:
        plt.rcParams.update({
            "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
            "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
            "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
            "ytick.color":"#c8d8f0","grid.color":"#1a2744",
            "font.family":"monospace","font.size":9,
        })

    df = correction_profile(Cd_nominal, altitude_m=altitude_m)
    vs = df["velocity_ms"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#080c14")
    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.12, left=0.07, right=0.97)
    TEXT = "#c8d8f0" if cfg.DARK_THEME else "#111"

    ax0 = axes[0]
    ax0.plot(vs, df["Cd_corrected"],  color="#00d4ff", lw=2.0, label="Cd corrected")
    ax0.axhline(Cd_nominal, color="#888", lw=0.9, ls="--", label=f"Cd nominal={Cd_nominal}")
    ax0.set_title("Corrected Cd(v)", fontweight="bold")
    ax0.set_xlabel("Velocity [m/s]"); ax0.set_ylabel("Cd"); ax0.legend(fontsize=8); ax0.grid(True,alpha=0.3)

    ax1 = axes[1]
    ax1.plot(vs, df["correction_mach"],     color="#ff6b35", lw=1.8, label="Mach (P-G)")
    ax1.plot(vs, df["correction_reynolds"],  color="#a8ff3e", lw=1.8, label="Reynolds")
    ax1.plot(vs, df["correction_porosity"],  color="#9d60ff", lw=1.8, label="Porosity")
    ax1.plot(vs, df["total_correction"],     color="#ffd700", lw=2.2, ls="--", label="Total")
    ax1.axhline(1.0, color="#888", lw=0.7, ls=":", alpha=0.6)
    ax1.set_title("Individual correction factors", fontweight="bold")
    ax1.set_xlabel("Velocity [m/s]"); ax1.set_ylabel("Factor"); ax1.legend(fontsize=8); ax1.grid(True,alpha=0.3)

    ax2 = axes[2]
    alts = np.linspace(0, 5000, 100)
    Cd_vs_alt = [corrected_Cd(Cd_nominal, 30.0, h)["Cd_corrected"] for h in alts]
    ax2.plot(Cd_vs_alt, alts, color="#00d4ff", lw=2.0)
    ax2.axvline(Cd_nominal, color="#888", lw=0.9, ls="--", label=f"Nominal={Cd_nominal}")
    ax2.set_title("Cd vs altitude (v=30 m/s)", fontweight="bold")
    ax2.set_xlabel("Cd"); ax2.set_ylabel("Altitude [m]"); ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)

    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)

    fig.text(0.5, 0.95,
             f"Mach + Reynolds + Porosity Drag Corrections  |  Cd₀={Cd_nominal}  h={altitude_m}m",
             ha="center", fontsize=11, fontweight="bold", color=TEXT)

    sp = save_path or (cfg.OUTPUTS_DIR / "mach_cd_corrections.png")
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Mach/Re/porosity plot saved: {sp}")
    return fig


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    print("Cd correction at typical deployment:")
    for v in [5, 15, 25, 40, 60]:
        r = corrected_Cd(1.35, v, 1000.0)
        print(f"  v={v:3d}m/s  M={r['Mach']:.4f}  Re={r['Reynolds']:.0f}  "
              f"Cd={r['Cd_corrected']:.5f}  total_corr={r['total_correction']:.5f}")
    plot_corrections()
