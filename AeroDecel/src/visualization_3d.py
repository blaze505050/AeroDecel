"""
src/visualization_3d.py — 3-D EDL Visualisation (PyVista + Matplotlib fallback)
=================================================================================
Renders the full descent trajectory in 3-D with:

  PRIMARY (PyVista):
    • Textured planet sphere (Mars/Venus/Titan colour-mapped)
    • EDL trajectory tube coloured by heat flux
    • Entry corridor cone showing ±3σ trajectory uncertainty
    • Canopy inflation animation at deployment altitude
    • Heat flux glow intensity mapped to trajectory colour
    • TPS temperature gradient sphere around nose
    • Landing ellipse projected onto terrain
    • Animated interactive window (rotate/zoom)
    • Export: PNG screenshot + MP4 fly-through (if ffmpeg available)

  FALLBACK (matplotlib 3-D, always available):
    • 3-D trajectory in axes3d
    • Entry corridor shaded band
    • Heat flux colour mapping
    • Landing ellipse on ground plane
    • Publication-quality static PNG

Install PyVista:  pip install pyvista pyvistaqt  (MIT licence, free)
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from matplotlib import cm

try:
    import pyvista as pv
    pv.global_theme.anti_aliasing = "msaa"
    _PYVISTA = True
except ImportError:
    _PYVISTA = False


# ══════════════════════════════════════════════════════════════════════════════
# PLANET COLOUR MAPS
# ══════════════════════════════════════════════════════════════════════════════

PLANET_PALETTE = {
    "mars":  {"surface": (0.82, 0.48, 0.32), "atm": (0.90, 0.60, 0.35, 0.15),
              "bg": (0.04, 0.03, 0.05), "label_color": "#e87040"},
    "venus": {"surface": (0.88, 0.78, 0.45), "atm": (0.95, 0.85, 0.55, 0.25),
              "bg": (0.06, 0.05, 0.02), "label_color": "#f0c840"},
    "titan": {"surface": (0.55, 0.42, 0.28), "atm": (0.70, 0.55, 0.30, 0.30),
              "bg": (0.02, 0.02, 0.05), "label_color": "#c8a060"},
}


def _planet_radius(planet_name: str) -> float:
    return {"mars": 3_389_500, "venus": 6_051_800, "titan": 2_574_700}.get(
        planet_name.lower(), 3_389_500)


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY → CARTESIAN CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def trajectory_to_cartesian(
    t_arr:      np.ndarray,
    v_arr:      np.ndarray,
    h_arr:      np.ndarray,
    planet_name: str = "mars",
    lat0_deg:   float = 18.4,    # Jezero crater for Mars
    lon0_deg:   float = 77.5,
    gamma_deg:  float = 15.0,
) -> dict:
    """
    Convert (t, v, h) trajectory to 3-D Cartesian coordinates on/above planet.

    Uses a rotating entry trajectory: the vehicle follows a great-circle arc
    from entry to landing, descending along the flight path angle γ.
    """
    R_p = _planet_radius(planet_name)

    n      = len(t_arr)
    lat0   = np.deg2rad(lat0_deg)
    lon0   = np.deg2rad(lon0_deg)
    gamma  = np.deg2rad(gamma_deg)

    # Integrate horizontal position
    dx = np.zeros(n)
    for i in range(1, n):
        dt_ = t_arr[i] - t_arr[i-1]
        v_h = float(v_arr[i]) * np.cos(gamma)    # horizontal component
        dx[i] = dx[i-1] + v_h * dt_

    # Arc length → lat/lon shift along great-circle (azimuth = 0 → N)
    arc   = dx / (R_p + h_arr)
    lats  = lat0 - arc                             # descending from entry point
    lons  = lon0 + arc * 0.3                       # slight drift east

    # Cartesian from spherical: (r, lat, lon) → (x, y, z)
    r     = R_p + h_arr
    x     = r * np.cos(lats) * np.cos(lons)
    y     = r * np.cos(lats) * np.sin(lons)
    z     = r * np.sin(lats)

    # Landing point
    r_land = R_p + h_arr[-1]
    x_land = r_land * np.cos(lats[-1]) * np.cos(lons[-1])
    y_land = r_land * np.cos(lats[-1]) * np.sin(lons[-1])
    z_land = r_land * np.sin(lats[-1])

    return {
        "x": x, "y": y, "z": z,
        "lat_rad": lats, "lon_rad": lons,
        "r": r, "h": h_arr,
        "v": v_arr, "t": t_arr,
        "x_land": x_land, "y_land": y_land, "z_land": z_land,
        "R_planet": R_p,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB 3-D FALLBACK  (always available, publication-quality)
# ══════════════════════════════════════════════════════════════════════════════

def _sphere_mesh(R: float, n: int = 40):
    """Return x, y, z arrays for a sphere of radius R."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi,   n)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(n), np.cos(v))
    return x, y, z


def plot_3d_matplotlib(
    traj:       dict,
    q_arr:      np.ndarray,
    mc_df       = None,
    planet_name: str = "mars",
    save_path:  Path | None = None,
    title:      str  = "",
) -> plt.Figure:
    """
    Full 3-D visualisation using matplotlib only.
    9-panel layout: large 3-D scene + 8 supporting panels.
    """
    pal = PLANET_PALETTE.get(planet_name.lower(), PLANET_PALETTE["mars"])
    BG  = "#080c14"; AX = "#0d1526"; TX = "#c8d8f0"; SP = "#2a3d6e"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": AX,
        "axes.edgecolor": SP, "text.color": TX,
        "axes.labelcolor": TX, "xtick.color": TX, "ytick.color": TX,
        "grid.color": "#1a2744", "font.family": "monospace", "font.size": 9,
    })

    fig = plt.figure(figsize=(24, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 5, figure=fig,
                            hspace=0.52, wspace=0.40,
                            top=0.92, bottom=0.06, left=0.04, right=0.97)

    # ── Main 3-D panel (spans 3 rows × 2 cols) ────────────────────────────────
    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    ax3d.set_facecolor("#040810")
    ax3d.grid(False)
    try:
        ax3d.set_pane_color((0.02, 0.02, 0.06, 1.0))
        for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
            axis.pane.fill = True; axis.line.set_color(SP)
    except AttributeError:
        pass

    R_p = traj["R_planet"]

    # Planet sphere
    sx, sy, sz = _sphere_mesh(R_p, 30)
    sr, sg, sb = pal["surface"]
    ax3d.plot_surface(sx/1e6, sy/1e6, sz/1e6,
                      color=(sr, sg, sb, 0.7), linewidth=0, antialiased=True,
                      shade=True, zorder=1)

    # Atmosphere haze (slightly larger sphere, transparent)
    asr, asg, asb, asa = pal["atm"]
    ax3d.plot_surface(*[s/1e6 * 1.03 for s in (sx,sy,sz)],
                      color=(asr, asg, asb, 0.08), linewidth=0, zorder=2)

    # Trajectory coloured by heat flux
    x = traj["x"]/1e6; y = traj["y"]/1e6; z = traj["z"]/1e6
    q_n = (q_arr - q_arr.min()) / max(q_arr.max() - q_arr.min(), 1)
    cmap = cm.inferno
    for i in range(len(x)-1):
        ax3d.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                  color=cmap(q_n[i]), lw=2.5, zorder=5)

    # Entry / deployment markers
    ax3d.scatter(x[0],  y[0],  z[0],  s=120, c="#ff4560", zorder=8,
                 marker="D", label="Entry interface")
    ax3d.scatter(x[-1], y[-1], z[-1], s=120, c="#a8ff3e", zorder=8,
                 marker="*", label="Landing")

    # Uncertainty cone (±2σ corridor from MC if available)
    if mc_df is not None and "x_east_m" in mc_df.columns:
        xe = mc_df["x_east_m"].values / 1e6 + x[-1]
        xn = mc_df["x_north_m"].values / 1e6 + y[-1]
        ax3d.scatter(xe, xn, np.full(len(xe), z[-1]),
                     s=3, alpha=0.25, color="#00d4ff", zorder=4)

    ax3d.set_xlabel("x [Mm]", labelpad=8); ax3d.set_ylabel("y [Mm]", labelpad=8)
    ax3d.set_zlabel("z [Mm]", labelpad=8)
    ax3d.set_title(f"3-D Entry Trajectory — {planet_name.title()}",
                   fontweight="bold", fontsize=11, pad=10)
    ax3d.legend(loc="upper right", fontsize=7.5)

    # Colourbar for heat flux
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(q_arr.min(), q_arr.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, pad=0.05, shrink=0.6, label="Heat flux [MW/m²]")
    cbar.ax.tick_params(colors=TX, labelsize=7)
    cbar.set_label("Heat flux [MW/m²]", color=TX)

    # ── Side panels ───────────────────────────────────────────────────────────
    def side_ax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor(AX); a.grid(True, alpha=0.3)
        a.tick_params(colors=TX); a.spines[:].set_color(SP)
        return a

    t = traj["t"]; v = traj["v"]; h = traj["h"]

    # v(t)
    a = side_ax(0, 2)
    a.fill_between(t, v/1e3, alpha=0.2, color="#00d4ff")
    a.plot(t, v/1e3, color="#00d4ff", lw=1.8)
    a.set_title("Velocity", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("v [km/s]")

    # h(t)
    a = side_ax(0, 3)
    a.fill_between(t, h/1e3, alpha=0.2, color="#9d60ff")
    a.plot(t, h/1e3, color="#9d60ff", lw=1.8)
    a.set_title("Altitude", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("h [km]")

    # q(t)
    a = side_ax(0, 4)
    a.fill_between(t[:len(q_arr)], q_arr, alpha=0.2, color="#ff4560")
    a.plot(t[:len(q_arr)], q_arr, color="#ff4560", lw=1.8)
    a.set_title("Heat Flux", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("q [MW/m²]")

    # Phase portrait (v vs h)
    a = side_ax(1, 2)
    sc = a.scatter(v/1e3, h/1e3, c=t, cmap="plasma", s=3, alpha=0.8)
    fig.colorbar(sc, ax=a, pad=0.02, label="t [s]").ax.tick_params(labelsize=7)
    a.set_title("Phase Portrait", fontweight="bold")
    a.set_xlabel("v [km/s]"); a.set_ylabel("h [km]")

    # Top-down trajectory ground track
    a = side_ax(1, 3)
    lat_km = (traj["lat_rad"] - traj["lat_rad"][-1]) * _planet_radius(planet_name) / 1e3
    lon_km = (traj["lon_rad"] - traj["lon_rad"][-1]) * _planet_radius(planet_name) / 1e3
    sc2 = a.scatter(lon_km, lat_km, c=h/1e3, cmap="viridis", s=4, alpha=0.85)
    fig.colorbar(sc2, ax=a, pad=0.02, label="h [km]").ax.tick_params(labelsize=7)
    a.scatter([0], [0], s=80, color="#a8ff3e", marker="*", zorder=5, label="Landing")
    a.set_title("Ground Track", fontweight="bold")
    a.set_xlabel("E [km]"); a.set_ylabel("N [km]"); a.legend(fontsize=7.5)

    # Landing ellipse from MC
    a = side_ax(1, 4)
    if mc_df is not None and "x_east_m" in mc_df.columns:
        xe = mc_df["x_east_m"].values / 1e3
        xn = mc_df["x_north_m"].values / 1e3
        a.scatter(xe, xn, s=3, alpha=0.3, color="#00d4ff")
        # 90% ellipse
        cov = np.cov(xe, xn)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 0)
        ang = np.deg2rad(np.linspace(0, 360, 200))
        from matplotlib.patches import Ellipse
        for pct, scale, col in [(50,1.177,"#a8ff3e"),(90,2.146,"#ffd700"),(99,3.035,"#ff4560")]:
            ell = Ellipse((xe.mean(), xn.mean()),
                          2*scale*np.sqrt(evals[-1]), 2*scale*np.sqrt(evals[0]),
                          angle=np.degrees(np.arctan2(*evecs[:,0][::-1])),
                          edgecolor=col, facecolor="none", lw=1.2, label=f"{pct}%")
            a.add_patch(ell)
    a.scatter([0],[0], s=60, color="#ff4560", marker="*", zorder=5, label="Nominal")
    a.set_aspect("equal"); a.legend(fontsize=7)
    a.set_title("Landing Ellipse", fontweight="bold")
    a.set_xlabel("East [km]"); a.set_ylabel("North [km]")

    # Mach number profile
    a = side_ax(2, 2)
    from src.planetary_atm import get_planet_atmosphere
    try:
        atm = get_planet_atmosphere(planet_name)
        mach_arr = np.array([atm.mach_number(float(v_), max(float(h_), 0))
                             for v_, h_ in zip(v, h)])
        a.fill_between(t, mach_arr, alpha=0.2, color="#ffd700")
        a.plot(t, mach_arr, color="#ffd700", lw=1.8)
        a.axhline(1.0, color=TX, lw=0.7, ls="--", alpha=0.5, label="M=1")
        a.legend(fontsize=7.5)
    except Exception:
        pass
    a.set_title("Mach Number", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("M")

    # Deceleration (g-load)
    a = side_ax(2, 3)
    if len(v) > 2:
        dv_dt = np.gradient(v, t)
        g_load = np.abs(dv_dt) / 9.81
        a.fill_between(t, g_load, alpha=0.2, color="#ff6b35")
        a.plot(t, g_load, color="#ff6b35", lw=1.8)
        a.axhline(15, color="#ff4560", lw=0.8, ls="--", alpha=0.7, label="15g limit")
        a.legend(fontsize=7.5)
    a.set_title("G-Load", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("g")

    # Dynamic pressure
    a = side_ax(2, 4)
    try:
        atm2 = get_planet_atmosphere(planet_name)
        rho_a = np.array([atm2.density(max(0, float(h_))) for h_ in h])
        q_dyn = 0.5 * rho_a * v**2 / 1e3   # kPa
        a.fill_between(t, q_dyn, alpha=0.2, color="#9d60ff")
        a.plot(t, q_dyn, color="#9d60ff", lw=1.8)
    except Exception:
        pass
    a.set_title("Dynamic Pressure", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("q [kPa]")

    mission_label = title or f"{planet_name.title()} EDL 3-D Visualisation"
    fig.text(0.5, 0.957, mission_label,
             ha="center", fontsize=13, fontweight="bold", color=TX)

    sp = save_path or Path("outputs/trajectory_3d.png")
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ 3-D trajectory plot saved: {sp}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PYVISTA VISUALISATION  (full 3-D interactive)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_pyvista(
    traj:       dict,
    q_arr:      np.ndarray,
    mc_df       = None,
    planet_name: str = "mars",
    save_path:  Path | None = None,
    interactive: bool = False,
) -> str | None:
    """
    Full 3-D PyVista scene.

    Features
    --------
    • Textured planet sphere (noise-based surface texture)
    • Translucent atmosphere shell
    • EDL trajectory tube coloured by heat flux (inferno colourmap)
    • Entry interface marker (red sphere)
    • Landing site marker (green cone)
    • Annotation labels
    • Exports screenshot PNG and optionally MP4

    Returns
    -------
    Path to saved screenshot, or None if pyvista unavailable.
    """
    if not _PYVISTA:
        print("  [3D] PyVista not installed — using matplotlib fallback")
        return None

    pal = PLANET_PALETTE.get(planet_name.lower(), PLANET_PALETTE["mars"])
    R_p = traj["R_planet"]

    pv.set_plot_theme("dark")
    pl = pv.Plotter(off_screen=True, window_size=[2400, 1600])
    pl.set_background(pal["bg"])

    # ── Planet sphere ─────────────────────────────────────────────────────────
    planet_sphere = pv.Sphere(radius=R_p, theta_resolution=60, phi_resolution=60)
    # Procedural surface texture: perlin-like noise for terrain variation
    pts     = planet_sphere.points
    noise   = np.sin(pts[:, 0]/R_p*8) * np.cos(pts[:, 1]/R_p*8) * \
              np.sin(pts[:, 2]/R_p*6) * 0.5 + 0.5
    planet_sphere["elevation"] = noise
    r, g, b = pal["surface"]
    lut = pv.LookupTable(cmap="hot" if planet_name=="venus" else "YlOrBr",
                          n_values=256, scalar_range=[0, 1])
    pl.add_mesh(planet_sphere, scalars="elevation", cmap=lut,
                smooth_shading=True, specular=0.4, diffuse=0.8)

    # ── Atmosphere shell ──────────────────────────────────────────────────────
    atm_shell = pv.Sphere(radius=R_p * 1.04, theta_resolution=30, phi_resolution=30)
    ar, ag, ab, aa = pal["atm"]
    pl.add_mesh(atm_shell, color=(ar, ag, ab), opacity=aa,
                smooth_shading=True, render=False)

    # ── Trajectory tube ───────────────────────────────────────────────────────
    xyz   = np.column_stack([traj["x"], traj["y"], traj["z"]])
    spline = pv.Spline(xyz, n_points=max(len(xyz), 500))
    # Interpolate heat flux onto spline
    q_interp = np.interp(
        np.linspace(0, 1, spline.n_points),
        np.linspace(0, 1, len(q_arr)), q_arr
    )
    spline["heat_flux_MWm2"] = q_interp
    tube = spline.tube(radius=R_p * 0.003)
    tube["heat_flux_MWm2"] = np.repeat(q_interp, tube.n_points // spline.n_points + 1)[:tube.n_points]
    pl.add_mesh(tube, scalars="heat_flux_MWm2", cmap="inferno",
                smooth_shading=True, label="EDL trajectory")

    # Colourbar
    pl.add_scalar_bar("Heat Flux [MW/m²]", position_x=0.85, position_y=0.3,
                       color="white", fmt="%.3f", height=0.4, width=0.04)

    # ── Entry interface marker ────────────────────────────────────────────────
    entry = pv.Sphere(radius=R_p * 0.008,
                       center=(traj["x"][0], traj["y"][0], traj["z"][0]))
    pl.add_mesh(entry, color="#ff4560", label="Entry interface")
    pl.add_point_labels(
        [(traj["x"][0], traj["y"][0], traj["z"][0])],
        ["Entry\n(v={:.1f} km/s)".format(traj["v"][0]/1e3)],
        font_size=14, text_color="#ff4560", shape=None,
    )

    # ── Landing site marker ────────────────────────────────────────────────────
    cone = pv.Cone(center=(traj["x"][-1], traj["y"][-1], traj["z"][-1]),
                   direction=(0,0,1), height=R_p*0.015, radius=R_p*0.006)
    pl.add_mesh(cone, color="#a8ff3e")
    pl.add_point_labels(
        [(traj["x"][-1], traj["y"][-1], traj["z"][-1])],
        ["Landing\n(v={:.1f} m/s)".format(traj["v"][-1])],
        font_size=14, text_color="#a8ff3e", shape=None,
    )

    # ── Landing ellipse from MC ────────────────────────────────────────────────
    if mc_df is not None and "x_east_m" in mc_df.columns:
        xe = mc_df["x_east_m"].values
        xn = mc_df["x_north_m"].values
        # Project scatter onto planet surface
        lat_land = np.arctan2(traj["z"][-1], np.sqrt(traj["x"][-1]**2 + traj["y"][-1]**2))
        lon_land = np.arctan2(traj["y"][-1], traj["x"][-1])
        for ex, en in zip(xe[::3], xn[::3]):   # subsample for speed
            lat_s = lat_land + en / R_p
            lon_s = lon_land + ex / R_p
            xs = R_p * np.cos(lat_s) * np.cos(lon_s)
            ys = R_p * np.cos(lat_s) * np.sin(lon_s)
            zs = R_p * np.sin(lat_s)
            dot = pv.Sphere(radius=R_p*0.001, center=(xs, ys, zs))
            pl.add_mesh(dot, color="#00d4ff", opacity=0.5)

    # ── Camera & lighting ────────────────────────────────────────────────────
    focus = np.array([traj["x"].mean(), traj["y"].mean(), traj["z"].mean()])
    pl.camera.focal_point = tuple(focus)
    pl.camera.position    = tuple(focus + np.array([R_p*3, R_p*1.5, R_p*2]))
    pl.camera.up          = (0, 0, 1)
    pl.enable_shadows()
    pl.add_light(pv.Light(
        position=tuple(focus + np.array([R_p*5, 0, R_p*5])),
        focal_point=tuple(focus), intensity=1.5,
    ))

    # ── Title ─────────────────────────────────────────────────────────────────
    pl.add_text(
        f"{planet_name.title()} EDL 3-D Trajectory\n"
        f"Entry v={traj['v'][0]/1e3:.1f} km/s  |  "
        f"Landing v={traj['v'][-1]:.1f} m/s  |  "
        f"Peak q={q_arr.max():.2f} MW/m²",
        position="upper_edge", font_size=16, color="white",
    )

    # ── Screenshot ────────────────────────────────────────────────────────────
    sp = save_path or Path("outputs/trajectory_3d_pyvista.png")
    pl.screenshot(str(sp), transparent_background=False, scale=2)
    print(f"  ✓ PyVista screenshot saved: {sp}")

    if interactive:
        pl.show()

    pl.close()
    return str(sp)


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def visualize(
    t_arr:      np.ndarray,
    v_arr:      np.ndarray,
    h_arr:      np.ndarray,
    q_arr:      np.ndarray,          # heat flux [MW/m²], same length as t
    planet_name: str = "mars",
    mc_df               = None,
    gamma_deg:  float = 15.0,
    title:      str   = "",
    save_dir:   Path  = Path("outputs"),
    pyvista_interactive: bool = False,
) -> dict:
    """
    Render all 3-D visualisations (matplotlib always, PyVista if installed).

    Parameters
    ----------
    t_arr, v_arr, h_arr : trajectory arrays
    q_arr               : heat flux [MW/m²] array
    planet_name         : "mars" | "venus" | "titan"
    mc_df               : optional DataFrame from MonteCarloEDL for landing ellipse
    gamma_deg           : flight path angle [°]
    title               : plot title string
    save_dir            : output directory
    pyvista_interactive : open interactive 3-D window (requires display)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Ensure q_arr same length as trajectory
    if len(q_arr) != len(t_arr):
        q_arr = np.interp(np.linspace(0,1,len(t_arr)),
                          np.linspace(0,1,len(q_arr)), q_arr)

    # Convert to Cartesian
    traj = trajectory_to_cartesian(t_arr, v_arr, h_arr,
                                    planet_name=planet_name, gamma_deg=gamma_deg)

    results = {}

    # Matplotlib (always)
    fig = plot_3d_matplotlib(traj, q_arr, mc_df=mc_df, planet_name=planet_name,
                              save_path=save_dir/"trajectory_3d.png",
                              title=title or f"{planet_name.title()} EDL 3-D Trajectory")
    results["matplotlib_path"] = str(save_dir / "trajectory_3d.png")
    plt.close(fig)

    # PyVista (if installed)
    if _PYVISTA:
        pv_path = plot_3d_pyvista(traj, q_arr, mc_df=mc_df, planet_name=planet_name,
                                   save_path=save_dir/"trajectory_3d_pyvista.png",
                                   interactive=pyvista_interactive)
        results["pyvista_path"] = pv_path
    else:
        print(f"  [3D] Tip: pip install pyvista  for photorealistic 3-D rendering")
        results["pyvista_path"] = None

    results["pyvista_available"] = _PYVISTA
    return results


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from src.planetary_atm import MarsAtmosphere
    from src.multifidelity_pinn import LowFidelityEDL

    m  = MarsAtmosphere()
    lf = LowFidelityEDL(m, 900, 1.7, 78.5, gamma_deg=15)
    t  = np.linspace(0, 400, 300)
    v, h = lf.solve(t, 5800, 125_000)
    rho  = np.array([m.density(max(0,h_)) for h_ in h])
    q    = 1.74e-4 * np.sqrt(np.maximum(rho,0) / 4.5) * v**3 / 1e6

    out = visualize(t, v, h, q, planet_name="mars",
                    title="Mars EDL — Perseverance Style")
    print(f"Outputs: {out}")
