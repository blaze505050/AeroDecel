"""
export_animation.py — Animated MP4 / GIF Export
=================================================
Renders the mission control scene as a publication-quality animation.
Requires: matplotlib  (always available)
Optional: ffmpeg for MP4 (pip install ffmpeg-python or install system ffmpeg)
Fallback: always saves animated GIF without ffmpeg.

Usage
-----
python src/export_animation.py --fps 30 --duration auto
python src/export_animation.py --format gif --fps 15
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density
from src.calibrate_cd import _simulate, _logistic_A


def _run_sim(mass, alt0, v0, Cd, Am, ti, dt=0.05):
    ts, vs, hs, As, Ds = [], [], [], [], []
    v_, h_, t_ = float(v0), float(alt0), 0.0
    while h_ > 0 and t_ < 600:
        A = _logistic_A(t_, Am, ti); rho = density(max(0, h_))
        drag = 0.5 * rho * v_**2 * Cd * A
        v_ = max(0, v_ + dt * (cfg.GRAVITY - drag/mass)); h_ = max(0, h_ - dt*v_); t_ += dt
        ts.append(t_); vs.append(v_); hs.append(h_); As.append(A); Ds.append(drag)
        if h_ <= 0: break
    return np.array(ts), np.array(vs), np.array(hs), np.array(As), np.array(Ds)


def create_animation(
    mass:   float = None,
    alt0:   float = None,
    v0:     float = None,
    Cd:     float = None,
    Am:     float = None,
    ti:     float = 2.5,
    fps:    int   = 20,
    fmt:    str   = "gif",     # "gif" | "mp4"
    speed:  float = 8.0,       # simulation seconds per real second
    save_path: Path | None = None,
) -> Path:
    """Render animated descent to GIF or MP4."""
    mass = mass or cfg.PARACHUTE_MASS
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    Cd   = Cd   or cfg.CD_INITIAL
    Am   = Am   or cfg.CANOPY_AREA_M2

    ts, vs, hs, As, Ds = _run_sim(mass, alt0, v0, Cd, Am, ti)

    # Subsample to target fps × duration
    sim_duration   = float(ts[-1])
    real_duration  = sim_duration / speed
    n_frames       = max(30, int(real_duration * fps))
    frame_indices  = np.round(np.linspace(0, len(ts)-1, n_frames)).astype(int)

    plt.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })

    fig = plt.figure(figsize=(14, 7), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.45, wspace=0.40,
                            top=0.90, bottom=0.08, left=0.06, right=0.97)

    # Scene canvas
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_scene.set_facecolor("#080d1a")
    ax_scene.set_xlim(-1, 1); ax_scene.set_ylim(-0.05, 1.05)
    ax_scene.set_xticks([]); ax_scene.set_yticks([])
    ax_scene.set_title("Descent", fontweight="bold", fontsize=10)

    # Animated lines
    ax_v  = fig.add_subplot(gs[0, 1]); ax_v.set_facecolor("#0d1526")
    ax_h  = fig.add_subplot(gs[0, 2]); ax_h.set_facecolor("#0d1526")
    ax_A  = fig.add_subplot(gs[0, 3]); ax_A.set_facecolor("#0d1526")
    ax_D  = fig.add_subplot(gs[1, 1]); ax_D.set_facecolor("#0d1526")
    ax_ph = fig.add_subplot(gs[1, 2:4]); ax_ph.set_facecolor("#0d1526")

    for ax in [ax_v, ax_h, ax_A, ax_D, ax_ph]:
        ax.grid(True, alpha=0.3); ax.tick_params(colors="#c8d8f0")
        ax.spines[:].set_color("#2a3d6e")

    ax_v.set_xlim(0, ts[-1]); ax_v.set_ylim(0, vs.max()*1.1)
    ax_v.set_xlabel("t [s]"); ax_v.set_ylabel("v [m/s]"); ax_v.set_title("Velocity",fontweight="bold",fontsize=9)
    ax_h.set_xlim(0, ts[-1]); ax_h.set_ylim(0, alt0*1.05)
    ax_h.set_xlabel("t [s]"); ax_h.set_ylabel("h [m]"); ax_h.set_title("Altitude",fontweight="bold",fontsize=9)
    ax_A.set_xlim(0, ts[-1]); ax_A.set_ylim(0, Am*1.1)
    ax_A.set_xlabel("t [s]"); ax_A.set_ylabel("A [m²]"); ax_A.set_title("Canopy Area",fontweight="bold",fontsize=9)
    ax_D.set_xlim(0, ts[-1]); ax_D.set_ylim(0, Ds.max()*1.15)
    ax_D.set_xlabel("t [s]"); ax_D.set_ylabel("F [N]"); ax_D.set_title("Drag",fontweight="bold",fontsize=9)
    ax_ph.set_xlim(0, vs.max()*1.1); ax_ph.set_ylim(0, alt0*1.05)
    ax_ph.set_xlabel("v [m/s]"); ax_ph.set_ylabel("h [m]"); ax_ph.set_title("Phase Portrait",fontweight="bold",fontsize=9)

    # Ground
    ax_scene.fill_between([-1,1], [-0.05,-0.05], [0,0], color="#0a1f0a")

    # Animated artists
    line_v,  = ax_v.plot([],[],  color="#00d4ff", lw=1.8)
    line_h,  = ax_h.plot([],[],  color="#9d60ff", lw=1.8)
    line_A,  = ax_A.plot([],[],  color="#ffd700", lw=1.8)
    line_D,  = ax_D.plot([],[],  color="#ff4560", lw=1.8)
    scat_ph  = ax_ph.scatter([],[],c=[],cmap="plasma",s=4,alpha=0.8,vmin=0,vmax=ts[-1])
    dot_ph   = ax_ph.scatter([],[],s=60,color="#a8ff3e",zorder=5)

    # Scene actors
    canopy_line, = ax_scene.plot([], [], color="#2255aa", lw=2)
    payload_rect = plt.Rectangle((0,0), 0.05, 0.03, color="#9d9c94")
    ax_scene.add_patch(payload_rect)
    vel_arrow = ax_scene.annotate("", xy=(0,0), xytext=(0,0),
                                   arrowprops=dict(arrowstyle="->", color="#5dca85", lw=2))
    info_text = ax_scene.text(0.02, 0.97, "", transform=ax_scene.transAxes,
                               fontsize=8, color="#c8d8f0", va="top", fontfamily="monospace")

    fig.text(0.5, 0.95,
             f"Parachute Dynamics  |  m={mass}kg  Cd={Cd}  A_max={Am}m²  alt₀={alt0}m",
             ha="center", fontsize=10, fontweight="bold", color="#c8d8f0")

    def update(frame_num):
        i = frame_indices[frame_num]
        t_cur = ts[:i+1]; v_cur = vs[:i+1]; h_cur = hs[:i+1]
        A_cur = As[:i+1]; D_cur = Ds[:i+1]

        line_v.set_data(t_cur, v_cur)
        line_h.set_data(t_cur, h_cur)
        line_A.set_data(t_cur, A_cur)
        line_D.set_data(t_cur, D_cur)

        if len(t_cur) > 1:
            scat_ph.set_offsets(np.c_[v_cur, h_cur])
            scat_ph.set_array(t_cur)
        dot_ph.set_offsets([[v_cur[-1], h_cur[-1]]])

        # Scene
        h_norm = float(hs[i]) / alt0
        v_norm = float(vs[i]) / max(vs[0], 1)
        A_norm = float(As[i]) / max(Am, 1)
        cw = 0.08 + 0.35 * np.sqrt(max(0, A_norm))

        cx_vals = np.linspace(0.5-cw, 0.5+cw, 20)
        cy_base = h_norm + 0.04
        cy_vals = cy_base + cw*0.4 * np.sin(np.linspace(0, np.pi, 20))
        canopy_line.set_data(cx_vals, cy_vals)

        payload_rect.set_xy((0.5-0.025, h_norm))
        vel_len = 0.12 * v_norm
        vel_arrow.set_position((0.5, h_norm))
        vel_arrow.xy = (0.5, h_norm - vel_len)

        info_text.set_text(f"t = {ts[i]:.1f}s\nv = {vs[i]:.1f} m/s\nh = {hs[i]:.0f} m")
        return (line_v, line_h, line_A, line_D, scat_ph, dot_ph,
                canopy_line, payload_rect, info_text)

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000//fps, blit=False)

    # ── Save ──────────────────────────────────────────────────────────────────
    sp = save_path or (cfg.OUTPUTS_DIR / f"simulation_animation.{fmt}")

    if fmt == "mp4":
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                            extra_args=["-vcodec","libx264","-pix_fmt","yuv420p"])
            ani.save(str(sp), writer=writer)
            print(f"  ✓ MP4 animation saved: {sp}  ({n_frames} frames @ {fps}fps)")
        except Exception as e:
            print(f"  FFmpeg unavailable ({e}) — saving GIF instead")
            sp = sp.with_suffix(".gif")
            ani.save(str(sp), writer="pillow", fps=fps)
            print(f"  ✓ GIF animation saved: {sp}")
    else:
        ani.save(str(sp), writer="pillow", fps=fps)
        print(f"  ✓ GIF animation saved: {sp}  ({n_frames} frames @ {fps}fps)")

    plt.close(fig)
    return sp


def run(fps=20, fmt="gif", speed=8.0, save_path=None):
    print(f"[Animation] fps={fps}  format={fmt}  speed={speed}x")
    return create_animation(fps=fps, fmt=fmt, speed=speed, save_path=save_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export simulation animation")
    p.add_argument("--fps",      type=int,   default=20)
    p.add_argument("--format",   type=str,   default="gif", choices=["gif","mp4"])
    p.add_argument("--speed",    type=float, default=8.0, help="Simulation speed-up factor")
    p.add_argument("--mass",     type=float, default=None)
    p.add_argument("--alt0",     type=float, default=None)
    a = p.parse_args()
    if a.mass: cfg.PARACHUTE_MASS = a.mass
    if a.alt0: cfg.INITIAL_ALT    = a.alt0
    run(fps=a.fps, fmt=a.format, speed=a.speed)
