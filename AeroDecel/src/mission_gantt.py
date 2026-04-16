"""
src/mission_gantt.py — Mission Timeline Gantt + Animated Entry Export
======================================================================
Renders the complete EDL sequence as an interactive publication-quality
Gantt chart and exports an animated GIF/MP4 of the descent.

Gantt phases
------------
  Phase 1: Atmospheric Entry        (entry interface → peak heating)
  Phase 2: Peak Heating             (max q point)
  Phase 3: Peak Deceleration        (max g-load point)
  Phase 4: Chute Deployment         (Mach trigger → full inflation)
  Phase 5: Supersonic Descent       (chute stable → subsonic)
  Phase 6: Terminal Descent         (subsonic → 200 m AGL)
  Phase 7: Landing                  (200 m → touchdown)

Each phase has computed duration from the physics trajectory.

Animation
---------
  Pure matplotlib FuncAnimation:
    - Planet sphere (coloured for each body)
    - Vehicle glowing as heat flux rises (inferno colourmap)
    - Chute deploying at trigger altitude
    - Landing ellipse projected on surface
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# PHASE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_phases(t: np.ndarray, v: np.ndarray, h: np.ndarray,
                  q: np.ndarray, planet_atm,
                  deploy_alt_m: float = 10_000.0) -> list[dict]:
    """
    Detect EDL phases from trajectory data.

    Returns list of phase dicts with: name, t_start, t_end, color, icon
    """
    # Find key events
    q_peak_idx   = int(np.argmax(q))
    t_peak_q     = float(t[q_peak_idx])

    if len(v) > 2:
        dv = np.abs(np.gradient(v, t))
        g_peak_idx = int(np.argmax(dv))
        t_peak_g   = float(t[g_peak_idx])
    else:
        g_peak_idx = 0; t_peak_g = float(t[0])

    deploy_idx = np.searchsorted(-h, -deploy_alt_m)
    deploy_idx = min(deploy_idx, len(h)-1)
    t_deploy   = float(t[deploy_idx])

    subsonic_idx = np.searchsorted(-v, -340)  # below 340 m/s
    subsonic_idx = min(max(subsonic_idx, deploy_idx), len(v)-1)
    t_subsonic   = float(t[subsonic_idx])

    low_alt_idx = np.searchsorted(-h, -200)
    low_alt_idx = min(max(low_alt_idx, subsonic_idx), len(h)-1)
    t_low_alt   = float(t[low_alt_idx])

    t_land = float(t[-1])

    phases = [
        {"name": "🔥 Atmospheric Entry",     "t_start": 0,          "t_end": t_peak_q,
         "color": "#ff4560", "description": f"Entry interface → peak heating at t={t_peak_q:.1f}s"},
        {"name": "⚡ Peak Heating",           "t_start": t_peak_q,   "t_end": t_peak_g,
         "color": "#ff6b35", "description": f"Peak q={q.max()/1e6:.2f}MW/m² → peak deceleration"},
        {"name": "🔴 Peak Deceleration",      "t_start": t_peak_g,   "t_end": t_deploy,
         "color": "#ffd700", "description": f"Peak g-load → chute deployment at h={deploy_alt_m/1e3:.0f}km"},
        {"name": "🪂 Chute Deployment",       "t_start": t_deploy,   "t_end": t_subsonic,
         "color": "#a8ff3e", "description": f"Mortar fire + inflation ({t_subsonic-t_deploy:.1f}s)"},
        {"name": "💨 Supersonic Descent",     "t_start": t_subsonic, "t_end": t_low_alt,
         "color": "#00d4ff", "description": f"Chute stable descent to 200m AGL"},
        {"name": "🎯 Terminal Descent",       "t_start": t_low_alt,  "t_end": t_land,
         "color": "#9d60ff", "description": "Final approach to surface"},
        {"name": "🏁 Landing",               "t_start": t_land,     "t_end": t_land + 1,
         "color": "#c8d8f0", "description": f"Touchdown at v={v[-1]:.1f}m/s"},
    ]

    # Remove zero-duration phases
    phases = [p for p in phases if p["t_end"] > p["t_start"]]
    return phases


# ══════════════════════════════════════════════════════════════════════════════
# GANTT CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_gantt(phases: list[dict], trajectory: dict,
               planet_name: str = "Mars",
               save_path: str = "outputs/mission_gantt.png"):
    """
    Render full EDL mission timeline as publication Gantt + trajectory panels.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrow

    matplotlib.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })
    TX="#c8d8f0"; BG="#080c14"; AX="#0d1526"; SP="#2a3d6e"

    fig = plt.figure(figsize=(22, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.50, wspace=0.38,
                            top=0.91, bottom=0.06, left=0.15, right=0.97)

    # ── Gantt chart (main panel, left column) ─────────────────────────────────
    ax_gantt = fig.add_subplot(gs[:, 0])
    ax_gantt.set_facecolor(AX)
    ax_gantt.tick_params(colors=TX); ax_gantt.spines[:].set_color(SP)
    ax_gantt.grid(True, alpha=0.2, axis="x")

    n_phases = len(phases)
    y_pos    = np.arange(n_phases)[::-1]   # top = phase 1

    for i, (phase, y) in enumerate(zip(phases, y_pos)):
        dur    = phase["t_end"] - phase["t_start"]
        ax_gantt.barh(y, dur, left=phase["t_start"], height=0.65,
                      color=phase["color"], alpha=0.80, edgecolor="none")
        ax_gantt.text(phase["t_start"] + dur/2, y,
                      f"{dur:.1f}s", ha="center", va="center",
                      fontsize=7.5, color="#000", fontweight="bold")

    ax_gantt.set_yticks(y_pos)
    ax_gantt.set_yticklabels([p["name"] for p in phases], fontsize=8.5)
    ax_gantt.set_xlabel("Mission elapsed time [s]", color=TX)
    ax_gantt.set_title(f"{planet_name} EDL Mission Timeline", fontweight="bold", fontsize=10)

    # Phase annotations
    for i, phase in enumerate(phases):
        ax_gantt.text(phase["t_end"] + 1, n_phases-1-i,
                      phase["description"], va="center",
                      fontsize=6.5, color="#778899")

    # ── Trajectory panels ──────────────────────────────────────────────────────
    t = trajectory["t"]; v = trajectory["v"]
    h = trajectory["h"]; q = trajectory.get("q", np.zeros_like(t))

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor(AX); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color(SP)
        return a

    def add_phase_spans(ax, y_min, y_max):
        for phase in phases:
            ax.axvspan(phase["t_start"], phase["t_end"],
                       alpha=0.08, color=phase["color"])

    ax = gax(0, 1)
    ax.fill_between(t, v/1e3, alpha=0.15, color="#00d4ff")
    ax.plot(t, v/1e3, color="#00d4ff", lw=2)
    add_phase_spans(ax, 0, v.max()/1e3)
    ax.set_title("Velocity", fontweight="bold"); ax.set_xlabel("t [s]"); ax.set_ylabel("v [km/s]")

    ax = gax(0, 2)
    ax.fill_between(t, h/1e3, alpha=0.15, color="#9d60ff")
    ax.plot(t, h/1e3, color="#9d60ff", lw=2)
    add_phase_spans(ax, 0, h.max()/1e3)
    ax.set_title("Altitude", fontweight="bold"); ax.set_xlabel("t [s]"); ax.set_ylabel("h [km]")

    ax = gax(0, 3)
    ax.fill_between(t[:len(q)], q/1e6, alpha=0.2, color="#ff4560")
    ax.plot(t[:len(q)], q/1e6, color="#ff4560", lw=2)
    add_phase_spans(ax, 0, q.max()/1e6)
    ax.set_title("Heat Flux", fontweight="bold"); ax.set_xlabel("t [s]"); ax.set_ylabel("q [MW/m²]")

    ax = gax(1, 1)
    if len(v) > 2:
        g_load = np.abs(np.gradient(v, t)) / 9.81
        ax.fill_between(t, g_load, alpha=0.2, color="#ff6b35")
        ax.plot(t, g_load, color="#ff6b35", lw=1.8)
    add_phase_spans(ax, 0, 30)
    ax.set_title("G-Load", fontweight="bold"); ax.set_xlabel("t [s]"); ax.set_ylabel("g")

    ax = gax(1, 2)
    from src.planetary_atm import get_planet_atmosphere
    try:
        atm   = get_planet_atmosphere(planet_name)
        mach  = np.array([atm.mach_number(float(vi), max(float(hi),0)) for vi,hi in zip(v,h)])
        ax.fill_between(t, mach, alpha=0.2, color="#ffd700")
        ax.plot(t, mach, color="#ffd700", lw=1.8)
        ax.axhline(1, color=TX, lw=0.7, ls="--", alpha=0.5)
    except: pass
    add_phase_spans(ax, 0, 25)
    ax.set_title("Mach Number", fontweight="bold"); ax.set_xlabel("t [s]"); ax.set_ylabel("M")

    ax = gax(1, 3)
    try:
        rho_arr = np.array([atm.density(max(0,float(hi))) for hi in h])
        q_dyn = 0.5 * rho_arr * v**2 / 1e3
        ax.fill_between(t, q_dyn, alpha=0.2, color="#9d60ff")
        ax.plot(t, q_dyn, color="#9d60ff", lw=1.8)
    except: pass
    add_phase_spans(ax, 0, 100)
    ax.set_title("Dynamic Pressure", fontweight="bold"); ax.set_xlabel("t [s]"); ax.set_ylabel("q [kPa]")

    # Phase duration bar
    ax = gax(2, 1)
    pnames  = [p["name"].split(" ", 1)[1][:20] for p in phases]
    durs    = [p["t_end"]-p["t_start"] for p in phases]
    colors_ = [p["color"] for p in phases]
    ax.barh(pnames, durs, color=colors_, alpha=0.8)
    ax.set_xlabel("Duration [s]"); ax.set_title("Phase Durations", fontweight="bold")

    # Mission metrics summary
    ax = gax(2, 2); ax.axis("off")
    total_time = float(t[-1])
    metrics = [
        ("Total duration",  f"{total_time:.1f} s"),
        ("Entry speed",     f"{v[0]/1e3:.2f} km/s"),
        ("Landing speed",   f"{v[-1]:.2f} m/s"),
        ("Peak q",          f"{q.max()/1e6:.3f} MW/m²"),
        ("Phases detected", str(len(phases))),
        ("Planet",          planet_name),
    ]
    for j, (lab, val) in enumerate(metrics):
        ax.text(0.05, 1-j*0.15, lab, transform=ax.transAxes, fontsize=9, color="#556688")
        ax.text(0.95, 1-j*0.15, val, transform=ax.transAxes, fontsize=9,
                ha="right", color=TX, fontweight="bold")
    ax.set_title("Mission Metrics", fontweight="bold", color=TX)

    # Phase timeline with events
    ax = gax(2, 3)
    ax.set_xlim(0, float(t[-1])*1.05); ax.set_ylim(-0.5, len(phases)+0.5)
    for i, phase in enumerate(phases):
        ax.barh(i, phase["t_end"]-phase["t_start"], left=phase["t_start"],
                height=0.5, color=phase["color"], alpha=0.7)
        ax.text(-1, i, phase["name"].split()[0], ha="right", va="center", fontsize=9)
    ax.set_xlabel("t [s]"); ax.set_yticks([]); ax.set_title("Timeline Overview", fontweight="bold")

    fig.text(0.5, 0.955,
             f"{planet_name} EDL Mission Gantt  |  "
             f"Duration={total_time:.1f}s  |  {len(phases)} phases",
             ha="center", fontsize=12, fontweight="bold", color=TX)

    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Gantt chart saved: {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED ENTRY VIDEO EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_entry_animation(t: np.ndarray, v: np.ndarray, h: np.ndarray,
                            q: np.ndarray, planet_name: str = "mars",
                            fps: int = 15, speed: float = 20.0,
                            fmt: str = "gif",
                            save_path: str | None = None) -> Path:
    """
    Export animated descent as GIF or MP4.

    Features
    --------
    • Planet sphere (coloured)
    • Vehicle with heat glow (inferno intensity ∝ heat flux)
    • Chute deploy animation at trigger altitude
    • Real-time telemetry readouts
    • Phase indicator
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib import cm

    matplotlib.use("Agg")

    PLANET_PALETTE = {
        "mars":  {"surf": (0.75, 0.38, 0.22), "sky": (0.85, 0.50, 0.25, 0.12)},
        "venus": {"surf": (0.82, 0.72, 0.38), "sky": (0.90, 0.80, 0.45, 0.22)},
        "titan": {"surf": (0.48, 0.35, 0.22), "sky": (0.62, 0.48, 0.28, 0.28)},
    }
    pal = PLANET_PALETTE.get(planet_name.lower(), PLANET_PALETTE["mars"])

    # Subsample for fps
    n_sim = len(t)
    sim_dur = float(t[-1])
    n_frames = max(30, int(sim_dur / speed * fps))
    frame_idx = np.round(np.linspace(0, n_sim-1, n_frames)).astype(int)

    # Normalise heat flux for glow
    q_norm = Normalize(vmin=0, vmax=max(q.max(), 1e-3))

    plt.rcParams.update({
        "figure.facecolor":"#04060d","axes.facecolor":"#060c1a",
        "axes.edgecolor":"#1a2540","text.color":"#b8c8e0",
        "axes.labelcolor":"#b8c8e0","xtick.color":"#b8c8e0",
        "ytick.color":"#b8c8e0","grid.color":"#111f38",
        "font.family":"monospace","font.size":8,
    })

    fig = plt.figure(figsize=(16, 8), facecolor="#04060d")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.40, wspace=0.30,
                            top=0.89, bottom=0.07, left=0.05, right=0.97)

    # Scene canvas
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_scene.set_facecolor("#020508")
    ax_scene.set_xlim(-1.1, 1.1); ax_scene.set_ylim(-0.08, 1.10)
    ax_scene.set_xticks([]); ax_scene.set_yticks([])

    # Planet surface
    theta_surf = np.linspace(0, np.pi, 100)
    x_surf = np.cos(np.linspace(-np.pi, np.pi, 100))
    ax_scene.fill_between(x_surf, -0.08, 0.02, color=pal["surf"], alpha=0.9)
    # Atmosphere
    ax_scene.fill_between(np.cos(np.linspace(-np.pi, np.pi, 100)),
                           0.02, 0.10, color=pal["sky"], alpha=0.6)

    # Stars
    rng_s = np.random.default_rng(42)
    star_x = rng_s.uniform(-1, 1, 150)
    star_y = rng_s.uniform(0.12, 1.08, 150)
    ax_scene.scatter(star_x, star_y, s=0.5, color="white", alpha=0.5, zorder=1)

    # Animated elements
    vehicle_dot   = ax_scene.scatter([], [], s=80, color="#00d4ff", zorder=8)
    glow_scatter  = ax_scene.scatter([], [], s=300, alpha=0.0, zorder=7)
    chute_lines   = [ax_scene.plot([], [], color="#aaaaff", lw=1.2, alpha=0.0)[0] for _ in range(8)]
    chute_dome    = ax_scene.fill([], [], color="#5588ff", alpha=0.0)[0]
    trail_line,   = ax_scene.plot([], [], color="#ff6b35", lw=1.0, alpha=0.6)
    info_text     = ax_scene.text(0.02, 0.97, "", transform=ax_scene.transAxes,
                                   fontsize=8, color="#b8c8e0", va="top", fontfamily="monospace")
    phase_text    = ax_scene.text(0.5, 1.03, "", transform=ax_scene.transAxes,
                                   ha="center", fontsize=9, fontweight="bold", color="#ffd700")
    ax_scene.set_title(f"{planet_name.title()} Entry, Descent & Landing", fontweight="bold",
                        fontsize=9, pad=5)

    # Live charts
    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#060c1a"); a.grid(True, alpha=0.25)
        a.tick_params(colors="#b8c8e0", labelsize=7); a.spines[:].set_color("#1a2540")
        return a

    ax_v = gax(0, 1); ax_v.set_xlim(0, sim_dur); ax_v.set_ylim(0, v.max()/1e3*1.05)
    ax_v.set_xlabel("t [s]"); ax_v.set_ylabel("v [km/s]"); ax_v.set_title("Velocity", fontweight="bold")
    line_v, = ax_v.plot([], [], color="#00d4ff", lw=1.8); dot_v = ax_v.scatter([], [], s=40, color="#a8ff3e", zorder=5)

    ax_h = gax(0, 2); ax_h.set_xlim(0, sim_dur); ax_h.set_ylim(0, h.max()/1e3*1.05)
    ax_h.set_xlabel("t [s]"); ax_h.set_ylabel("h [km]"); ax_h.set_title("Altitude", fontweight="bold")
    line_h, = ax_h.plot([], [], color="#9d60ff", lw=1.8); dot_h = ax_h.scatter([], [], s=40, color="#a8ff3e", zorder=5)

    ax_q = gax(1, 1); ax_q.set_xlim(0, sim_dur); ax_q.set_ylim(0, q.max()/1e6*1.1)
    ax_q.set_xlabel("t [s]"); ax_q.set_ylabel("q [MW/m²]"); ax_q.set_title("Heat Flux", fontweight="bold")
    line_q, = ax_q.plot([], [], color="#ff4560", lw=1.8); dot_q = ax_q.scatter([], [], s=40, color="#ffd700", zorder=5)

    ax_M = gax(1, 2); ax_M.set_xlim(0, sim_dur); ax_M.set_ylim(0, 22)
    ax_M.set_xlabel("t [s]"); ax_M.set_ylabel("Mach"); ax_M.set_title("Mach Number", fontweight="bold")
    ax_M.axhline(1, color="#888", lw=0.7, ls="--", alpha=0.5)
    from src.planetary_atm import get_planet_atmosphere
    try:
        atm_ = get_planet_atmosphere(planet_name)
        mach_arr = np.array([atm_.mach_number(float(vi), max(float(hi),0)) for vi,hi in zip(v,h)])
    except: mach_arr = v / 340
    line_M, = ax_M.plot([], [], color="#ffd700", lw=1.8); dot_M = ax_M.scatter([], [], s=40, color="#a8ff3e", zorder=5)

    trail_x, trail_y = [], []

    # Determine chute deploy index
    chute_alt = h.max() * 0.08
    deploy_idx_anim = max(0, np.searchsorted(-h, -chute_alt))

    def update(frame):
        i = frame_idx[min(frame, len(frame_idx)-1)]
        t_i = float(t[i]); v_i = float(v[i]); h_i = float(h[i]); q_i = float(q[min(i,len(q)-1)])
        M_i = float(mach_arr[i])

        # Scene position
        h_norm = float(h_i) / float(h.max()) if h.max() > 0 else 0
        x_pos  = 0.1 * np.sin(t_i * 0.05)   # slight horizontal drift
        y_pos  = 0.05 + h_norm * 0.95

        # Glow intensity from heat flux
        glow_intensity = float(q_norm(q_i))
        glow_color = cm.inferno(glow_intensity)
        vehicle_dot.set_offsets([[x_pos, y_pos]])
        glow_scatter.set_offsets([[x_pos, y_pos]])
        glow_scatter.set_color([glow_color]); glow_scatter.set_alpha(glow_intensity * 0.7)

        # Trail
        trail_x.append(x_pos); trail_y.append(y_pos)
        if len(trail_x) > 30:
            trail_x.pop(0); trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        # Chute (visible below deploy altitude)
        chute_visible = (i >= deploy_idx_anim) and (h_norm < 0.15)
        if chute_visible:
            chute_frac = min(1.0, (deploy_idx_anim - i + 30) / -30)
            chute_w = 0.20 * min(1, (i - deploy_idx_anim)/max(10,1))
            n_gor = 8
            for k, cl in enumerate(chute_lines):
                ang = 2*np.pi*k/n_gor
                cl.set_data([x_pos, x_pos + chute_w*np.sin(ang)],
                             [y_pos + 0.02, y_pos + 0.08 + chute_w*0.6*abs(np.cos(ang))])
                cl.set_alpha(0.6 * min(1, (i-deploy_idx_anim)/10))
            # Dome
            dome_x = x_pos + chute_w * np.cos(np.linspace(0, np.pi, 20))
            dome_y = y_pos + 0.08 + chute_w * 0.5 * np.abs(np.sin(np.linspace(0, np.pi, 20)))
            chute_dome.set_xy(np.column_stack([dome_x, dome_y]))
            chute_dome.set_alpha(0.35 * min(1, (i-deploy_idx_anim)/10))
        else:
            for cl in chute_lines: cl.set_alpha(0)
            chute_dome.set_alpha(0)

        # Telemetry
        info_text.set_text(f"t = {t_i:7.1f} s\n"
                           f"v = {v_i:7.1f} m/s\n"
                           f"h = {h_i/1e3:7.2f} km\n"
                           f"M = {M_i:7.3f}\n"
                           f"q = {q_i/1e6:7.3f} MW/m²")

        # Phase indicator
        phase_names = ["Entry", "Peak Heating", "Peak Decel", "Chute Deploy",
                       "Descent", "Terminal", "Landing"]
        phase_thresholds = [0, 0.1, 0.2, 0.35, 0.50, 0.80, 0.95]
        t_frac = t_i / max(sim_dur, 1)
        phase_idx = max(0, np.searchsorted(phase_thresholds, t_frac) - 1)
        phase_text.set_text(f"◆ {phase_names[min(phase_idx, len(phase_names)-1)]}")

        # Charts
        t_cur = t[:i+1]; v_cur = v[:i+1]; h_cur = h[:i+1]
        q_cur = q[:min(i+1, len(q))]; M_cur = mach_arr[:i+1]
        line_v.set_data(t_cur, v_cur/1e3); dot_v.set_offsets([[t_i, v_i/1e3]])
        line_h.set_data(t_cur, h_cur/1e3); dot_h.set_offsets([[t_i, h_i/1e3]])
        line_q.set_data(t[:len(q_cur)], q_cur/1e6); dot_q.set_offsets([[t_i, q_i/1e6]])
        line_M.set_data(t_cur, M_cur); dot_M.set_offsets([[t_i, M_i]])

        return (vehicle_dot, glow_scatter, trail_line, info_text, phase_text,
                line_v, dot_v, line_h, dot_h, line_q, dot_q, line_M, dot_M,
                chute_dome, *chute_lines)

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000//fps, blit=False)

    sp = Path(save_path) if save_path else Path(f"outputs/entry_animation_{planet_name}.{fmt}")
    sp.parent.mkdir(exist_ok=True)

    if fmt == "mp4":
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1500,
                                            extra_args=["-vcodec","libx264"])
            ani.save(str(sp), writer=writer)
        except:
            sp = sp.with_suffix(".gif")
            ani.save(str(sp), writer="pillow", fps=fps)
    else:
        ani.save(str(sp), writer="pillow", fps=fps)

    print(f"  ✓ Animation saved: {sp}  ({n_frames} frames @ {fps}fps)")
    plt.close(fig)
    return sp


def run(planet_atm=None, verbose: bool = True) -> dict:
    """Run Gantt + animation pipeline."""
    import matplotlib; matplotlib.use("Agg")
    from src.multifidelity_pinn import LowFidelityEDL

    if planet_atm is None:
        from src.planetary_atm import MarsAtmosphere
        planet_atm = MarsAtmosphere()

    lf = LowFidelityEDL(planet_atm, 900, 1.7, 78.5, gamma_deg=15)
    t  = np.linspace(0, 400, 200)
    v, h = lf.solve(t, 5800, 125_000)
    rho  = np.array([planet_atm.density(max(0, float(hi))) for hi in h])
    q    = 1.74e-4 * np.sqrt(np.maximum(rho, 0)/4.5) * np.maximum(v, 0)**3

    phases = detect_phases(t, v, h, q, planet_atm)
    traj   = {"t": t, "v": v, "h": h, "q": q}

    if verbose:
        print(f"\n[Gantt] {planet_atm.name} EDL — {len(phases)} phases detected")
        for p in phases:
            print(f"  {p['name'][:30]:30s}  {p['t_start']:6.1f}s → {p['t_end']:6.1f}s  ({p['t_end']-p['t_start']:.1f}s)")

    plot_gantt(phases, traj, planet_name=planet_atm.name)

    if verbose:
        print("  [Animation] Exporting entry GIF…")
    sp = export_entry_animation(t, v, h, q, planet_name=planet_atm.name.lower(),
                                 fps=12, speed=15.0, fmt="gif")

    return {"phases": phases, "animation_path": str(sp), "traj": traj}
