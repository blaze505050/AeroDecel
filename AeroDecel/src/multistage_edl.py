"""
src/multistage_edl.py — Multi-Stage EDL Simulation (Perseverance-style)
=========================================================================
Simulates the complete 4-stage Mars Entry, Descent, and Landing sequence:

  Stage 1: Guided Entry (125 km → ~11 km)
    • Ballistic entry with capsule aerodynamics
    • Heat shield TPS protection
    • Peak heating ~80 s, peak deceleration ~90 s

  Stage 2: Heat Shield Jettison (~11 km, Mach ~1.7)
    • Mass step-change (shed ~400 kg heat shield)
    • Cd changes from capsule to backshell config
    • Brief free-fall before chute deploy

  Stage 3: Parachute Descent (11 km → ~2.1 km)
    • DGB (Disk-Gap-Band) supersonic parachute
    • Cd_chute ≈ 0.65, D_chute = 21.5 m
    • Continuous deceleration to ~80 m/s

  Stage 4: Powered Descent (2.1 km → 0 m)
    • Backshell + chute jettison
    • Mars Lander Engines (MLE): 8 × throttleable hydrazine
    • Gravity turn guidance to ~0.75 m/s at touchdown

References
----------
  Sell, S.W. et al. (2022) "Mars 2020 EDL System Overview"
  Way, D.W. (2013) "Powered descent guidance for MSL"
  Chen, A. et al. (2022) "Mars 2020 EDL Performance"
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.integrate import solve_ivp


# ══════════════════════════════════════════════════════════════════════════════
# STAGE CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageConfig:
    """Configuration for one EDL stage."""
    name:          str
    mass_kg:       float    # vehicle mass at start of stage
    Cd:            float    # effective drag coefficient
    area_m2:       float    # reference area
    color:         str      # plot colour
    has_thrust:    bool = False
    thrust_N:      float = 0.0     # max thrust per engine
    n_engines:     int   = 0
    Isp_s:         float = 0.0     # specific impulse [s]
    throttle_min:  float = 0.0     # min throttle fraction
    throttle_max:  float = 1.0


# Perseverance-class EDL stages
STAGES = {
    "entry": StageConfig(
        name="Guided Entry",
        mass_kg=1025.0,        # entry mass
        Cd=1.62,               # 70° sphere-cone
        area_m2=15.9,          # π×(4.5/2)²
        color="#ff4560",
    ),
    "jettison": StageConfig(
        name="Heat Shield Jettison",
        mass_kg=625.0,         # after shedding 400 kg heatshield
        Cd=1.0,                # backshell Cd
        area_m2=15.9,
        color="#ffd700",
    ),
    "parachute": StageConfig(
        name="Parachute Descent",
        mass_kg=625.0,
        Cd=0.65,               # DGB chute Cd
        area_m2=363.1,         # π×(21.5/2)² — 21.5 m diameter DGB
        color="#a8ff3e",
    ),
    "powered": StageConfig(
        name="Powered Descent",
        mass_kg=400.0,          # after dropping backshell+chute (~225 kg)
        Cd=0.5,                 # descent stage, small reference area
        area_m2=8.0,            # descent stage cross-section
        color="#00d4ff",
        has_thrust=True,
        thrust_N=3070.0,        # MLE thrust per engine (peak)
        n_engines=8,
        Isp_s=226.0,            # hydrazine Isp
        throttle_min=0.25,
        throttle_max=1.0,
    ),
}

# Stage transition triggers
TRANSITIONS = {
    "entry_to_jettison": {
        "trigger": "mach",           # Mach < threshold
        "mach_threshold": 2.0,       # deploy chute at Mach ~2
        "altitude_min_m": 8_000,     # safety: don't deploy above 15 km
        "altitude_max_m": 15_000,
    },
    "jettison_to_parachute": {
        "trigger": "delay",          # fixed delay after jettison
        "delay_s": 2.0,              # 2s free-fall before chute inflation
    },
    "parachute_to_powered": {
        "trigger": "altitude",
        "altitude_m": 2_100,         # backshell sep + powered descent
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 3-DOF EDL DYNAMICS (with thrust)
# ══════════════════════════════════════════════════════════════════════════════

def _edl_rhs(t, y, planet_atm, stage: StageConfig, thrust_profile=None):
    """
    Right-hand side for 3-DOF EDL:
      y = [v, gamma, h, m]
      where v = speed, gamma = FPA (positive = up), h = altitude, m = mass

    For powered stage, includes thrust and mass flow.
    """
    v, gamma, h, m = y
    v = max(v, 0.1)
    h = max(h, 0.0)
    m = max(m, 100.0)

    g = planet_atm.gravity_ms2
    rho = planet_atm.density(h)
    R = planet_atm.radius_m

    # Aerodynamic drag
    q_dyn = 0.5 * rho * v**2
    D = q_dyn * stage.Cd * stage.area_m2

    # Thrust
    T = 0.0
    m_dot = 0.0
    if stage.has_thrust and thrust_profile is not None:
        throttle = thrust_profile(t, h, v)
        T = throttle * stage.thrust_N * stage.n_engines
        g0 = 9.80665
        m_dot = -T / (stage.Isp_s * g0)

    # 3-DOF equations of motion
    # Note: thrust for retro-fire opposes velocity, so +T/m in dv/dt
    T_sign = -1.0 if stage.has_thrust else 0.0  # retro-fire: opposes motion
    dv_dt     = -D / m - g * np.sin(gamma) + T / m
    dgamma_dt = -(g / v - v / (R + h)) * np.cos(gamma) if v > 1.0 else 0.0
    dh_dt     = v * np.sin(gamma)
    dm_dt     = m_dot

    return [dv_dt, dgamma_dt, dh_dt, dm_dt]


# ══════════════════════════════════════════════════════════════════════════════
# GRAVITY TURN GUIDANCE (powered descent)
# ══════════════════════════════════════════════════════════════════════════════

def _gravity_turn_throttle(t, h, v):
    """
    Gravity turn throttle schedule for powered descent.

    Uses a constant-deceleration profile to reach ~1 m/s at ground level.
    T_required = m×(a_target + g) where a_target is chosen to brake from
    current velocity to ~1 m/s over remaining altitude.
    """
    h = max(h, 0.1)
    v = max(v, 0.1)

    # Target velocity profile: v² = v_land² + 2·a_brake·h
    # Solve for a_brake: a_brake = (v² - v_land²) / (2·h)
    v_land = 1.0  # target landing velocity [m/s]
    a_brake = max((v**2 - v_land**2) / (2.0 * h), 0.0)

    # Mars gravity
    g_mars = 3.721

    # Required total deceleration (a_brake is along velocity, gravity adds)
    a_total = a_brake + g_mars  # need to overcome gravity + decelerate

    # Thrust needed: T = m × a_total, but we don't know mass here
    # Instead control via throttle as fraction of available specific thrust
    # Available: 8 × 3070 N / 400 kg ≈ 61.4 m/s²
    a_available = 8 * 3070 / 400.0  # ≈ 61.4 m/s²

    throttle = a_total / a_available

    # Clamp to throttleable range
    return float(np.clip(throttle, 0.20, 1.0))


def _powered_descent_1d(h0, v_down0, mass0, stage, g_mars=3.721):
    """
    Simple 1-D vertical powered descent (no gamma dynamics).

    State: [h, v_down, m] where v_down > 0 means falling.
    Equation:
      dv_down/dt = g - T/m - D/m     (gravity accel down, thrust/drag decel)
      dh/dt      = -v_down            (positive v_down = h decreasing)
      dm/dt      = -T/(Isp·g0)
    """
    g0 = 9.80665
    T_max = stage.thrust_N * stage.n_engines  # total max thrust

    def rhs(t, y):
        h, v_down, m = y
        h = max(h, 0.0)
        m = max(m, 100.0)

        # Desired deceleration profile: constant-decel to v_land at h=0
        # v² = v_land² + 2·a·h  →  a = (v_down² - v_land²)/(2h)
        v_land = 0.75
        if h > 0.1 and v_down > v_land:
            a_brake = (v_down**2 - v_land**2) / (2.0 * h)
        elif v_down > v_land:
            a_brake = 10.0  # emergency final braking
        else:
            a_brake = 0.0  # at or below target speed

        # Required upward acceleration = a_brake + gravity compensation
        a_required = a_brake + g_mars

        # Throttle
        throttle = a_required * m / T_max
        throttle = float(np.clip(throttle, 0.0, 1.0))

        T = throttle * T_max
        m_dot = -T / (stage.Isp_s * g0) if T > 0 else 0.0

        # Drag
        rho = 0.015  # near-surface Mars density
        D = 0.5 * rho * abs(v_down) * v_down * stage.Cd * stage.area_m2

        # Equations of motion (v_down positive = downward)
        dv_dt = g_mars - T / m - D / m
        dh_dt = -v_down

        return [dh_dt, dv_dt, m_dot]

    def landed(t, y): return y[0] - 0.3  # land at h=0.3m
    landed.terminal = True; landed.direction = -1

    sol = solve_ivp(rhs, (0, 120), [h0, v_down0, mass0],
                    method="RK45", max_step=0.2, events=[landed],
                    rtol=1e-6, atol=1e-8)

    return sol.t, sol.y[0], sol.y[1], sol.y[2]  # t, h, v_down, mass


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-STAGE INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_multistage(planet_atm=None, verbose: bool = True) -> dict:
    """
    Run the full 4-stage EDL sequence.

    Returns
    -------
    dict with:
      stages: list of stage result dicts
      timeline: combined timeline DataFrame
      summary: key metrics
    """
    if planet_atm is None:
        from src.planetary_atm import get_planet_atmosphere
        planet_atm = get_planet_atmosphere("mars")

    g0 = 9.80665   # for g-load calculation
    results = []

    # ── Stage 1: Guided Entry ────────────────────────────────────────────────
    stage = STAGES["entry"]
    v0, gamma0, h0, m0 = 5586.0, np.radians(-15.47), 125_000.0, stage.mass_kg

    if verbose:
        print(f"\n[Multi-Stage EDL] Starting 4-stage sequence")
        print(f"  Stage 1: {stage.name}  v={v0:.0f}m/s  h={h0/1e3:.0f}km")

    def ground_event(t, y): return y[2]  # h = 0
    ground_event.terminal = True; ground_event.direction = -1

    def mach_event(t, y):
        v = max(y[0], 0.1); h = max(y[2], 0)
        a = planet_atm.speed_of_sound(h)
        return v / a - TRANSITIONS["entry_to_jettison"]["mach_threshold"]
    mach_event.terminal = True; mach_event.direction = -1

    sol1 = solve_ivp(
        lambda t, y: _edl_rhs(t, y, planet_atm, stage),
        (0, 300), [v0, gamma0, h0, m0],
        method="RK45", max_step=0.5, events=[ground_event, mach_event],
        dense_output=True, rtol=1e-6
    )
    t1 = sol1.t; v1 = sol1.y[0]; gam1 = sol1.y[1]; h1 = sol1.y[2]; m1 = sol1.y[3]
    dv1 = np.gradient(v1, t1) if len(t1) > 1 else np.zeros_like(t1)
    g_load1 = np.abs(dv1) / g0

    results.append({
        "stage": "entry", "name": stage.name, "color": stage.color,
        "t": t1, "v": v1, "gamma": gam1, "h": h1, "m": m1,
        "g_load": g_load1,
    })
    if verbose:
        print(f"    → Done: t={t1[-1]:.1f}s  v={v1[-1]:.0f}m/s  "
              f"h={h1[-1]/1e3:.1f}km  peak_g={g_load1.max():.1f}")

    # ── Stage 2: Heat Shield Jettison (brief free-fall) ──────────────────────
    stage = STAGES["jettison"]
    t_offset = t1[-1]
    v_start = v1[-1]; gam_start = gam1[-1]; h_start = h1[-1]

    if verbose:
        print(f"  Stage 2: {stage.name}  v={v_start:.0f}m/s  h={h_start/1e3:.1f}km")

    delay = TRANSITIONS["jettison_to_parachute"]["delay_s"]
    sol2 = solve_ivp(
        lambda t, y: _edl_rhs(t, y, planet_atm, stage),
        (0, delay), [v_start, gam_start, h_start, stage.mass_kg],
        method="RK45", max_step=0.1, rtol=1e-6
    )
    t2 = sol2.t + t_offset; v2 = sol2.y[0]; gam2 = sol2.y[1]; h2 = sol2.y[2]; m2 = sol2.y[3]
    dv2 = np.gradient(v2, sol2.t) if len(sol2.t) > 1 else np.zeros_like(sol2.t)
    g_load2 = np.abs(dv2) / g0

    results.append({
        "stage": "jettison", "name": stage.name, "color": stage.color,
        "t": t2, "v": v2, "gamma": gam2, "h": h2, "m": m2,
        "g_load": g_load2,
    })
    if verbose:
        print(f"    → Done: t={t2[-1]:.1f}s  v={v2[-1]:.0f}m/s  h={h2[-1]/1e3:.1f}km")

    # ── Stage 3: Parachute Descent ───────────────────────────────────────────
    stage = STAGES["parachute"]
    t_offset = t2[-1]
    v_start = v2[-1]; gam_start = gam2[-1]; h_start = h2[-1]

    if verbose:
        print(f"  Stage 3: {stage.name}  v={v_start:.0f}m/s  h={h_start/1e3:.1f}km")

    alt_trigger = TRANSITIONS["parachute_to_powered"]["altitude_m"]

    def chute_alt_event(t, y): return y[2] - alt_trigger
    chute_alt_event.terminal = True; chute_alt_event.direction = -1

    sol3 = solve_ivp(
        lambda t, y: _edl_rhs(t, y, planet_atm, stage),
        (0, 300), [v_start, gam_start, h_start, stage.mass_kg],
        method="RK45", max_step=0.5,
        events=[ground_event, chute_alt_event], rtol=1e-6
    )
    t3 = sol3.t + t_offset; v3 = sol3.y[0]; gam3 = sol3.y[1]; h3 = sol3.y[2]; m3 = sol3.y[3]
    dv3 = np.gradient(v3, sol3.t) if len(sol3.t) > 1 else np.zeros_like(sol3.t)
    g_load3 = np.abs(dv3) / g0

    results.append({
        "stage": "parachute", "name": stage.name, "color": stage.color,
        "t": t3, "v": v3, "gamma": gam3, "h": h3, "m": m3,
        "g_load": g_load3,
    })
    if verbose:
        print(f"    → Done: t={t3[-1]:.1f}s  v={v3[-1]:.0f}m/s  h={h3[-1]/1e3:.1f}km")

    # ── Stage 4: Powered Descent (1-D vertical) ─────────────────────────────
    stage = STAGES["powered"]
    t_offset = t3[-1]
    h_start = h3[-1]
    # Approximate vertical descent speed (from gamma and speed)
    v_down_start = abs(v3[-1] * np.sin(gam3[-1]))

    if verbose:
        print(f"  Stage 4: {stage.name}  v_down={v_down_start:.0f}m/s  h={h_start/1e3:.1f}km")

    t4_local, h4, v_down4, m4 = _powered_descent_1d(
        h_start, v_down_start, stage.mass_kg, stage
    )
    t4 = t4_local + t_offset
    # Convert v_down (positive=falling) to v (speed magnitude) for plotting
    v4 = np.abs(v_down4)
    gam4 = np.full_like(v4, -np.pi/2)  # vertical
    g_load4 = np.abs(np.gradient(v_down4, t4_local)) / g0 if len(t4_local) > 1 else np.zeros_like(t4_local)

    results.append({
        "stage": "powered", "name": stage.name, "color": stage.color,
        "t": t4, "v": v4, "gamma": gam4, "h": h4, "m": m4,
        "g_load": g_load4,
    })
    if verbose:
        print(f"    → Done: t={t4[-1]:.1f}s  v={v4[-1]:.1f}m/s  "
              f"h={h4[-1]:.1f}m  m_final={m4[-1]:.1f}kg")

    # ── Summary ──────────────────────────────────────────────────────────────
    t_total = float(t4[-1])
    v_land  = float(v4[-1])
    m_final = float(m4[-1])
    fuel_used = float(STAGES["powered"].mass_kg - m_final)
    peak_g = float(max(g_load1.max(), g_load2.max(), g_load3.max(), g_load4.max()))

    summary = {
        "t_total_s":     round(t_total, 1),
        "v_landing_ms":  round(v_land, 2),
        "m_final_kg":    round(m_final, 1),
        "fuel_used_kg":  round(fuel_used, 1),
        "peak_g":        round(peak_g, 1),
        "n_stages":      4,
        "survived":      v_land < 5.0,
    }

    if verbose:
        print(f"\n  ✅ Multi-stage EDL complete")
        print(f"     Total time:   {summary['t_total_s']:.1f}s")
        print(f"     Landing v:    {summary['v_landing_ms']:.2f}m/s")
        print(f"     Fuel used:    {summary['fuel_used_kg']:.1f}kg")
        print(f"     Peak g:       {summary['peak_g']:.1f}g")

    return {
        "stages": results,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_multistage(result: dict,
                    save_path: str = "outputs/multistage_edl.png"):
    """
    Publication-quality multi-stage EDL timeline.
    6 panels: v(t), h(t), g_load(t), v(h) phase, mass(t), stage timeline.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    matplotlib.rcParams.update({
        "figure.facecolor": "#080c14", "axes.facecolor": "#0d1526",
        "axes.edgecolor": "#2a3d6e", "text.color": "#c8d8f0",
        "axes.labelcolor": "#c8d8f0", "xtick.color": "#c8d8f0",
        "ytick.color": "#c8d8f0", "grid.color": "#1a2744",
        "font.family": "monospace", "font.size": 9,
    })
    TX = "#c8d8f0"; BG = "#080c14"

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
                           top=0.90, bottom=0.06, left=0.06, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    stages = result["stages"]
    summary = result["summary"]

    # ── Panel 1: Velocity vs Time ────────────────────────────────────────────
    ax1 = gax(0, 0)
    for s in stages:
        ax1.plot(s["t"], s["v"] / 1e3, color=s["color"], lw=2.5, label=s["name"])
        ax1.fill_between(s["t"], s["v"] / 1e3, alpha=0.12, color=s["color"])
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Velocity [km/s]")
    ax1.set_title("Velocity Profile", fontweight="bold")
    ax1.legend(fontsize=7.5, loc="upper right")

    # ── Panel 2: Altitude vs Time ────────────────────────────────────────────
    ax2 = gax(0, 1)
    for s in stages:
        ax2.plot(s["t"], s["h"] / 1e3, color=s["color"], lw=2.5, label=s["name"])
        ax2.fill_between(s["t"], s["h"] / 1e3, alpha=0.12, color=s["color"])
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Altitude [km]")
    ax2.set_title("Altitude Profile", fontweight="bold")
    ax2.legend(fontsize=7.5)

    # ── Panel 3: g-load vs Time ──────────────────────────────────────────────
    ax3 = gax(1, 0)
    for s in stages:
        ax3.plot(s["t"], s["g_load"], color=s["color"], lw=2, label=s["name"])
    ax3.axhline(12, color="#ffd700", lw=0.8, ls="--", alpha=0.6, label="12g limit")
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("g-load [g$_{Earth}$]")
    ax3.set_title("Deceleration", fontweight="bold")
    ax3.legend(fontsize=7.5)

    # ── Panel 4: Phase Portrait v(h) ────────────────────────────────────────
    ax4 = gax(1, 1)
    for s in stages:
        ax4.plot(s["v"] / 1e3, s["h"] / 1e3, color=s["color"], lw=2.5,
                 label=s["name"])
    ax4.set_xlabel("Velocity [km/s]"); ax4.set_ylabel("Altitude [km]")
    ax4.set_title("Phase Portrait v(h)", fontweight="bold")
    ax4.legend(fontsize=7.5)

    # ── Panel 5: Mass vs Time ────────────────────────────────────────────────
    ax5 = gax(2, 0)
    for s in stages:
        ax5.plot(s["t"], s["m"], color=s["color"], lw=2.5, label=s["name"])
    ax5.set_xlabel("Time [s]"); ax5.set_ylabel("Mass [kg]")
    ax5.set_title("Vehicle Mass (fuel burn)", fontweight="bold")
    ax5.legend(fontsize=7.5)

    # ── Panel 6: Stage timeline bar ──────────────────────────────────────────
    ax6 = gax(2, 1)
    ax6.axis("off")

    y_bar = 0.7
    t_total = summary["t_total_s"]
    for i, s in enumerate(stages):
        t_start = s["t"][0]; t_end = s["t"][-1]
        width = (t_end - t_start) / t_total
        left = t_start / t_total
        ax6.barh(y_bar, width, left=left, height=0.15,
                 color=s["color"], alpha=0.8, edgecolor="white", linewidth=0.5)
        ax6.text(left + width / 2, y_bar, s["name"],
                 ha="center", va="center", fontsize=7, fontweight="bold", color="black")

    ax6.text(0.5, 0.92, "EDL MISSION TIMELINE", transform=ax6.transAxes,
             ha="center", fontsize=11, fontweight="bold", color="#00d4ff")

    # Summary stats
    rows = [
        ("Total time", f"{summary['t_total_s']:.0f} s"),
        ("Landing velocity", f"{summary['v_landing_ms']:.2f} m/s"),
        ("Peak g-load", f"{summary['peak_g']:.1f} g"),
        ("Fuel used", f"{summary['fuel_used_kg']:.1f} kg"),
        ("Final mass", f"{summary['m_final_kg']:.1f} kg"),
        ("LANDED",
         "SAFELY ✅" if summary["survived"] else "CRASH ❌"),
    ]
    for j, (label, val) in enumerate(rows):
        y = 0.42 - j * 0.065
        ax6.text(0.05, y, label, transform=ax6.transAxes, fontsize=9, color=TX)
        col = "#a8ff3e" if "✅" in val else ("#ff4560" if "❌" in val else TX)
        ax6.text(0.95, y, val, transform=ax6.transAxes,
                 fontsize=9, ha="right", fontweight="bold", color=col)

    fig.text(0.5, 0.955,
             f"Multi-Stage EDL — Perseverance Style  |  "
             f"4 stages  |  v_land={summary['v_landing_ms']:.2f}m/s  |  "
             f"peak_g={summary['peak_g']:.1f}  |  "
             f"fuel={summary['fuel_used_kg']:.1f}kg  |  "
             f"{'SURVIVED ✅' if summary['survived'] else 'FAILED ❌'}",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Multi-stage EDL plot saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(planet_atm=None, verbose: bool = True) -> dict:
    """Run full multi-stage EDL and generate publication plot."""
    import matplotlib; matplotlib.use("Agg")

    result = run_multistage(planet_atm, verbose=verbose)
    plot_multistage(result)

    return result


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    run()
