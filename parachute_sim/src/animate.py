"""
animate.py — Animated Simulation Export (MP4 / GIF)
====================================================
Renders the parachute descent as a smooth animation with:
  - Side-view canopy + payload + riser visualization
  - Live-updating v(t), h(t), A(t) strip charts
  - Stage indicator, phase badge, key metrics HUD
  - Exports to MP4 (ffmpeg) or GIF (Pillow) — both free

Usage
-----
    python src/animate.py                    # synthetic, saves animation.gif
    python src/animate.py --format mp4       # MP4 via ffmpeg
    python src/animate.py --fps 30 --speed 8
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Ellipse
import matplotlib.animation as animation
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.calibrate_cd import _simulate, _logistic_A


def _sim_full(mass=None, alt0=None, v0=None, Cd=None, Am=None, ti=2.5, dt=0.05):
    mass=mass or cfg.PARACHUTE_MASS; alt0=alt0 or cfg.INITIAL_ALT
    v0=v0 or cfg.INITIAL_VEL; Cd=Cd or cfg.CD_INITIAL; Am=Am or cfg.CANOPY_AREA_M2
    r = _simulate(Cd=Cd, mass=mass, alt0=alt0, v0=v0, Am=Am, ti=ti, dt=dt)
    t = r["time"]; v = r["velocity"]; h = r["altitude"]
    A = np.array([_logistic_A(ti_, Am, ti) for ti_ in t])
    drag = np.array([0.5*1.225*max(vi,0)**2*Cd*Ai for vi,Ai in zip(v,A)])
    return t, v, h, A, drag


def make_animation(
    t_arr, v_arr, h_arr, A_arr, drag_arr,
    fps: int = 20,
    speed: int = 8,
    fmt: str = "gif",
    save_path: Path = None,
    title: str = "Parachute Descent Simulation",
):
    if cfg.DARK_THEME:
        plt.rcParams.update({"figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
                             "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
                             "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
                             "ytick.color":"#c8d8f0","grid.color":"#1a2744"})
    plt.rcParams["font.family"] = "monospace"

    TEXT  = "#c8d8f0" if cfg.DARK_THEME else "#111"
    BG    = "#080c14" if cfg.DARK_THEME else "#f0f4f8"
    PANEL = "#0d1526" if cfg.DARK_THEME else "#ffffff"
    SKY   = "#050d1a" if cfg.DARK_THEME else "#cce0f5"
    GND   = "#0a1f0a" if cfg.DARK_THEME else "#3a7a20"
    CAN   = "#2255aa" if cfg.DARK_THEME else "#1869c9"
    PAY   = "#8a8a80" if cfg.DARK_THEME else "#5F5E5A"
    TRAIL = "rgba(85,170,140,0.4)"

    total_frames = len(t_arr)
    stride = max(1, speed)
    anim_frames = list(range(0, total_frames, stride))

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.07, left=0.05, right=0.97)

    # Main scene axes
    ax_scene = fig.add_subplot(gs[:, :2])
    ax_scene.set_facecolor(SKY)
    ax_scene.set_xlim(-1, 1)
    ax_scene.set_ylim(-0.05, 1.05)
    ax_scene.axis("off")

    # Strip chart axes
    ax_v  = fig.add_subplot(gs[0, 2:])
    ax_h  = fig.add_subplot(gs[1, 2:])
    ax_A  = fig.add_subplot(gs[2, 2:])

    for ax, ylabel, color in [
        (ax_v, "Velocity [m/s]", "#00d4ff"),
        (ax_h, "Altitude [m]",   "#9d60ff"),
        (ax_A, "Area [m²]",      "#a8ff3e"),
    ]:
        ax.set_facecolor(PANEL)
        ax.set_ylabel(ylabel, fontsize=8, color=TEXT)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.grid(True, alpha=0.3, color="#1a2744")
        ax.spines[:].set_color("#2a3d6e")

    # Background: ground
    ax_scene.axhspan(-0.05, 0.02, color=GND, zorder=0)

    # Pre-draw trajectory ghost
    h_norm = h_arr / max(h_arr.max(), 1)
    ax_scene.plot([0]*len(h_norm), h_norm, color="#1a3a2a", lw=1, alpha=0.3, zorder=1)

    # Animated elements
    trail_line, = ax_scene.plot([], [], color="#2ec48a", lw=1.5, alpha=0.6, zorder=2)
    canopy_patch = Ellipse((0, 0.8), 0.3, 0.08, color=CAN, zorder=4)
    payload_rect = plt.Rectangle((-0.04, 0.72), 0.08, 0.06, color=PAY, zorder=5)
    ax_scene.add_patch(canopy_patch)
    ax_scene.add_patch(payload_rect)

    riser_lines = [ax_scene.plot([], [], color="#445566", lw=0.7, alpha=0.7, zorder=3)[0]
                   for _ in range(8)]

    hud_text = ax_scene.text(0.02, 0.98, "", transform=ax_scene.transAxes,
                              va="top", fontsize=8.5, color=TEXT,
                              fontfamily="monospace", zorder=10)

    phase_text = ax_scene.text(0.98, 0.98, "", transform=ax_scene.transAxes,
                                va="top", ha="right", fontsize=9,
                                fontweight="bold", color="#ffd700", zorder=10)

    # Strip chart lines
    line_v, = ax_v.plot([], [], color="#00d4ff", lw=1.5)
    line_h, = ax_h.plot([], [], color="#9d60ff", lw=1.5)
    line_A, = ax_A.plot([], [], color="#a8ff3e", lw=1.5)

    # Vertical cursor on strip charts
    cursors = [ax.axvline(0, color="#ffd700", lw=1, alpha=0.7, zorder=5)
               for ax in [ax_v, ax_h, ax_A]]

    t_max = t_arr.max()
    for ax, arr in [(ax_v, v_arr),(ax_h, h_arr),(ax_A, A_arr)]:
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, arr.max()*1.1)

    fig.text(0.5, 0.965, title, ha="center", fontsize=12,
             fontweight="bold", color=TEXT, fontfamily="monospace")

    def _stage(h, A):
        Am = A_arr.max()
        if A < 0.05*Am: return "Free fall"
        if A < 0.93*Am: return "Inflation"
        if h > 50:      return "Stable descent"
        return "Landing"

    def init():
        trail_line.set_data([], [])
        for rl in riser_lines: rl.set_data([], [])
        line_v.set_data([], [])
        line_h.set_data([], [])
        line_A.set_data([], [])
        return [trail_line, canopy_patch, payload_rect,
                line_v, line_h, line_A, hud_text, phase_text] + riser_lines + cursors

    def animate_frame(fi):
        i = anim_frames[fi] if fi < len(anim_frames) else anim_frames[-1]
        i = min(i, len(t_arr)-1)

        h_norm_i = h_arr[i] / max(h_arr.max(), 1)
        A_i      = A_arr[i]
        A_max    = A_arr.max()
        v_i      = v_arr[i]
        t_i      = t_arr[i]
        frac_A   = np.clip(A_i / max(A_max, 1), 0.05, 1.0)

        # Canopy size scales with inflation
        cw = 0.08 + 0.25*frac_A
        ch = 0.035 + 0.05*frac_A
        cy = h_norm_i + 0.04 + ch
        canopy_patch.set_center((0, cy))
        canopy_patch.width  = cw*2
        canopy_patch.height = ch*2
        canopy_patch.set_color(CAN)

        # Payload
        payload_rect.set_xy((-0.04, h_norm_i))
        payload_rect.set_color(PAY)

        # Risers
        for k, rl in enumerate(riser_lines):
            ang = (k / 8) * np.pi
            rx  = -cw * np.cos(ang) * 0.9
            ry  = cy - ch * np.sin(ang) * 0.5
            rl.set_data([rx, 0], [ry, h_norm_i+0.06])

        # Trail
        i0 = max(0, i-100)
        trail_line.set_data([0]*( i-i0+1), h_norm_i - np.linspace(0, 0.001*(i-i0), i-i0+1))
        # simpler: just vertical line
        trail_line.set_data([0]*(i-i0+1),
                             (h_arr[i0:i+1] / max(h_arr.max(), 1)))

        # Strip charts
        line_v.set_data(t_arr[:i+1], v_arr[:i+1])
        line_h.set_data(t_arr[:i+1], h_arr[:i+1])
        line_A.set_data(t_arr[:i+1], A_arr[:i+1])
        for cur in cursors:
            cur.set_xdata([t_i, t_i])

        # HUD
        hud_text.set_text(
            f"t = {t_i:6.1f} s\n"
            f"v = {v_i:6.2f} m/s\n"
            f"h = {h_arr[i]:7.1f} m\n"
            f"A = {A_i:6.1f} m²\n"
            f"D = {drag_arr[i]/1e3:5.2f} kN"
        )
        phase_text.set_text(_stage(h_arr[i], A_i))

        return [trail_line, canopy_patch, payload_rect,
                line_v, line_h, line_A, hud_text, phase_text] + riser_lines + cursors

    anim = animation.FuncAnimation(
        fig, animate_frame, frames=len(anim_frames),
        init_func=init, interval=1000//fps, blit=False,
    )

    sp = save_path or (cfg.OUTPUTS_DIR / f"animation.{fmt}")
    if fmt == "mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                         extra_args=["-vcodec","libx264","-pix_fmt","yuv420p"])
        anim.save(str(sp), writer=writer)
    else:
        anim.save(str(sp), writer="pillow", fps=fps)

    print(f"  ✓ Animation saved: {sp}  ({len(anim_frames)} frames @ {fps}fps)")
    plt.close(fig)
    return sp


def run(mass=None, alt0=None, v0=None, Cd=None, Am=None, ti=2.5,
        fps=20, speed=8, fmt="gif", save_path=None, title=None):
    print(f"[Animate] Generating {fmt.upper()} animation...")
    t, v, h, A, drag = _sim_full(mass=mass, alt0=alt0, v0=v0,
                                   Cd=Cd, Am=Am, ti=ti, dt=0.05)
    lbl = title or (f"Parachute Descent  |  m={mass or cfg.PARACHUTE_MASS}kg  "
                    f"h₀={alt0 or cfg.INITIAL_ALT}m  "
                    f"Cd={Cd or cfg.CD_INITIAL}  A={Am or cfg.CANOPY_AREA_M2}m²")
    return make_animation(t, v, h, A, drag, fps=fps, speed=speed,
                           fmt=fmt, save_path=save_path, title=lbl)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mass",  type=float, default=None)
    p.add_argument("--alt0",  type=float, default=None)
    p.add_argument("--cd",    type=float, default=None)
    p.add_argument("--am",    type=float, default=None)
    p.add_argument("--fps",   type=int,   default=20)
    p.add_argument("--speed", type=int,   default=8,  help="Playback speed multiplier")
    p.add_argument("--format",type=str,   default="gif", choices=["gif","mp4"])
    a = p.parse_args()
    run(mass=a.mass, alt0=a.alt0, Cd=a.cd, Am=a.am,
        fps=a.fps, speed=a.speed, fmt=a.format)
