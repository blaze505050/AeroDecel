"""
phase6_trajectory.py — 3D Wind-Drift Trajectory & Landing Zone Predictor
=========================================================================
Extends the 1D vertical ODE to 3D by adding horizontal wind drift.

State vector: [vx, vy, vz, x, y, z]
  - z    : altitude (positive up)
  - x, y : horizontal position (East, North) in metres from deployment point
  - vz   : vertical velocity (positive down to match Phase 2 convention)
  - vx   : eastward velocity component
  - vy   : northward velocity component

Wind model options:
  1. Constant vector wind
  2. Exponential wind profile with altitude (power law: U(z) = U10 * (z/10)^α)
  3. Tabular wind profile (altitude → speed/direction table)

Additional outputs:
  - 3D flight path CSV and KML
  - Landing ellipse (CEP, 50%, 90% ellipses) from MC scatter
  - Max drift distance
  - Drift vs wind speed sensitivity
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density


# ─── Wind Profile Models ─────────────────────────────────────────────────────
class WindProfile:
    """Base class for wind speed profiles."""

    def __call__(self, altitude_m: float) -> tuple:
        """Returns (u_east, u_north) in m/s at given altitude."""
        raise NotImplementedError


class ConstantWind(WindProfile):
    def __init__(self, speed_ms: float = 5.0, direction_deg: float = 270.0):
        """
        direction_deg: meteorological convention (0=N, 90=E, 180=S, 270=W → FROM that direction)
        """
        d = np.deg2rad(direction_deg)
        # Wind blows FROM direction → payload drifts TO opposite
        self.u = speed_ms * np.sin(d)   # eastward component
        self.v = speed_ms * np.cos(d)   # northward component

    def __call__(self, altitude_m: float) -> tuple:
        return self.u, self.v


class PowerLawWind(WindProfile):
    """Wind speed follows power law with altitude: U(z) = U_ref * (z/z_ref)^alpha."""
    def __init__(self, speed_ref: float = 8.0, z_ref: float = 10.0,
                 alpha: float = 0.143, direction_deg: float = 270.0):
        self.speed_ref = speed_ref
        self.z_ref     = z_ref
        self.alpha     = alpha
        d = np.deg2rad(direction_deg)
        self.dir_u = np.sin(d)
        self.dir_v = np.cos(d)

    def __call__(self, altitude_m: float) -> tuple:
        z   = max(1.0, altitude_m)
        spd = self.speed_ref * (z / self.z_ref) ** self.alpha
        return spd * self.dir_u, spd * self.dir_v


class TabularWind(WindProfile):
    """Wind from a table of (altitude_m, u_east, u_north)."""
    def __init__(self, table: pd.DataFrame):
        # table columns: alt_m, u_east, u_north
        alts = table["alt_m"].values
        self._fu = interp1d(alts, table["u_east"].values,
                            bounds_error=False, fill_value="extrapolate")
        self._fv = interp1d(alts, table["u_north"].values,
                            bounds_error=False, fill_value="extrapolate")
        self.table    = table.copy()
        self.source   = table["source"].iloc[0] if "source" in table.columns else "tabular"
        self.n_levels = len(table)

    def __call__(self, altitude_m: float) -> tuple:
        return float(self._fu(altitude_m)), float(self._fv(altitude_m))

    def summary(self) -> str:
        spds = self.table["speed_ms"].values if "speed_ms" in self.table.columns else []
        return (f"TabularWind [{self.source}] — {self.n_levels} levels  "
                f"speed: {min(spds):.1f}–{max(spds):.1f} m/s"
                if len(spds) else f"TabularWind [{self.source}]")


class OpenMeteoWind(WindProfile):
    """
    Live wind profile from the Open-Meteo free API (no key required).
    Automatically fetches pressure-level wind U/V components and builds
    a smooth tabular interpolant across all deployment altitudes.
    Falls back to a synthetic power-law profile if offline.
    """
    def __init__(self, lat: float, lon: float, target_dt=None, verbose: bool = True):
        from src.fetch_wind import build_wind_profile
        self._inner = build_wind_profile(lat=lat, lon=lon,
                                         target_dt=target_dt, verbose=verbose)
        self.lat = lat; self.lon = lon

    def __call__(self, altitude_m: float) -> tuple:
        return self._inner(altitude_m)


# ─── 3D ODE System ───────────────────────────────────────────────────────────
class Trajectory3D:
    """
    6-DOF (translational only) parachute descent with wind coupling.
    Drag force acts along resultant velocity relative to air (body-relative wind).
    """

    def __init__(self, wind: WindProfile, at_model=None,
                 mass: float = None, Cd: float = None):
        self.wind    = wind
        self.mass    = mass or cfg.PARACHUTE_MASS
        self.Cd      = Cd   or cfg.CD_INITIAL
        self.at_model = at_model   # callable A(t) from Phase 2

    def _At(self, t: float) -> float:
        if self.at_model is not None:
            return float(self.at_model(t))
        # Fallback: generalized logistic
        ti  = 2.5
        Am  = cfg.CANOPY_AREA_M2
        k   = 5.0 / ti
        t0  = ti * 0.6
        return Am / (1 + np.exp(-k * (t - t0))) ** 0.5

    def _pinn_Cd(self, t: float) -> float:
        ti = 2.5
        return self.Cd + self.Cd * 0.38 * np.exp(-0.5 * ((t - ti) / (ti / 3)) ** 2)

    def rhs(self, t: float, state: np.ndarray) -> list:
        vx, vy, vz, x, y, z = state
        z  = max(0.0, z)
        vz = max(0.0, vz)

        uw, vw = self.wind(z)
        rho    = density(z)
        A_t    = self._At(t)
        Cd_t   = self._pinn_Cd(t)

        # Relative velocity of payload w.r.t. air
        vrel_x = vx - uw
        vrel_y = vy - vw
        vrel_z = vz          # vertical: no vertical wind component
        v_rel  = np.sqrt(vrel_x**2 + vrel_y**2 + vrel_z**2)

        # Drag magnitude
        D = 0.5 * rho * v_rel**2 * Cd_t * A_t

        # Drag direction (opposite to relative velocity)
        if v_rel > 1e-6:
            ax_drag = -D * vrel_x / (self.mass * v_rel)
            ay_drag = -D * vrel_y / (self.mass * v_rel)
            az_drag = -D * vrel_z / (self.mass * v_rel)
        else:
            ax_drag = ay_drag = az_drag = 0.0

        # Equations of motion (z is altitude, positive up → vz is descent rate → sign flip)
        dvx = ax_drag
        dvy = ay_drag
        dvz = cfg.GRAVITY + az_drag    # gravity pulls down, drag decelerates
        dx  = vx
        dy  = vy
        dz  = -vz   # altitude decreases at rate vz

        return [dvx, dvy, dvz, dx, dy, dz]

    def solve(self, t_span: tuple = None, t_eval: np.ndarray = None) -> pd.DataFrame:
        t_end = t_span[1] if t_span else cfg.INITIAL_ALT / max(1, cfg.INITIAL_VEL) + 60
        t_eval = np.linspace(0, t_end, 3000)
        y0 = [0.0, 0.0, cfg.INITIAL_VEL, 0.0, 0.0, cfg.INITIAL_ALT]

        def ground(t, y): return y[5]
        ground.terminal  = True
        ground.direction = -1

        sol = solve_ivp(
            self.rhs, (0, t_end), y0,
            method="RK45", t_eval=t_eval, events=ground,
            rtol=1e-5, atol=1e-7, dense_output=False,
        )

        t  = sol.t
        vx, vy, vz = sol.y[0], sol.y[1], sol.y[2]
        x,  y_,  z  = sol.y[3], sol.y[4], np.clip(sol.y[5], 0, None)
        v_total = np.sqrt(vx**2 + vy**2 + vz**2)
        drift   = np.sqrt(x**2 + y_**2)

        uw_arr = np.array([self.wind(zi)[0] for zi in z])
        vw_arr = np.array([self.wind(zi)[1] for zi in z])
        spd_w  = np.sqrt(uw_arr**2 + vw_arr**2)

        df = pd.DataFrame({
            "time_s"    : t,
            "vx"        : vx, "vy": vy, "vz": vz,
            "speed_ms"  : v_total,
            "x_east_m"  : x, "y_north_m": y_,
            "altitude_m": z,
            "drift_m"   : drift,
            "wind_speed" : spd_w,
        })

        land_x = float(x[-1]); land_y = float(y_[-1])
        land_drift = float(drift[-1])
        print(f"  Landing point: East={land_x:+.1f}m, North={land_y:+.1f}m, "
              f"Drift={land_drift:.1f}m from deployment")
        print(f"  Landing speed: {v_total[-1]:.2f} m/s  |  Time: {t[-1]:.1f} s")

        return df


# ─── Landing Ellipse (from MC scatter) ───────────────────────────────────────
def landing_ellipse_from_mc(n_mc: int = 200,
                             wind_speed_sigma: float = 4.0,
                             wind_dir_sigma: float = 45.0,
                             rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Propagate wind uncertainty through 3D trajectory to get landing scatter.
    Returns DataFrame of (x_east, y_north) landing positions.
    """
    rng   = rng or np.random.default_rng(42)
    lands = []
    print(f"\n  Computing landing ellipse from {n_mc} wind realizations...")

    for i in range(n_mc):
        spd = max(0, rng.normal(8.0, wind_speed_sigma))
        dir_  = rng.normal(270.0, wind_dir_sigma) % 360

        wind  = PowerLawWind(speed_ref=spd, direction_deg=dir_)
        traj  = Trajectory3D(wind=wind)
        t_end = cfg.INITIAL_ALT / max(1, cfg.INITIAL_VEL) + 80
        t_ev  = np.linspace(0, t_end, 1500)

        def ground(t, y): return y[5]
        ground.terminal  = True
        ground.direction = -1

        try:
            sol = solve_ivp(traj.rhs, (0, t_end),
                            [0,0,cfg.INITIAL_VEL,0,0,cfg.INITIAL_ALT],
                            method="RK45", t_eval=t_ev, events=ground,
                            rtol=1e-4, atol=1e-6)
            lands.append({"x_east": sol.y[3,-1], "y_north": sol.y[4,-1]})
        except Exception:
            pass

        if (i+1) % 50 == 0:
            print(f"\r    {i+1}/{n_mc} runs", end="", flush=True)

    print(f"\r    {len(lands)}/{n_mc} valid landing positions computed  ")
    return pd.DataFrame(lands)


# ─── KML Export ──────────────────────────────────────────────────────────────
def to_kml(df: pd.DataFrame, origin_lat: float = 51.5074,
           origin_lon: float = -0.1278, save_path: Path = None) -> str:
    """
    Convert trajectory x,y,z to KML for Google Earth / Maps visualization.
    Uses equirectangular projection from origin point.
    """
    R_EARTH = 6371000.0  # m
    lats = origin_lat + np.degrees(df["y_north_m"].values / R_EARTH)
    lons = origin_lon + np.degrees(df["x_east_m"].values /
                                    (R_EARTH * np.cos(np.deg2rad(origin_lat))))
    alts = df["altitude_m"].values

    coords = "\n".join(f"          {lon:.6f},{lat:.6f},{alt:.1f}"
                       for lon, lat, alt in zip(lons, lats, alts))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Parachute Trajectory</name>
  <Placemark>
    <name>Deployment Point</name>
    <Point><coordinates>{origin_lon},{origin_lat},{alts[0]}</coordinates></Point>
  </Placemark>
  <Placemark>
    <name>Landing Point</name>
    <Point><coordinates>{lons[-1]:.6f},{lats[-1]:.6f},0</coordinates></Point>
  </Placemark>
  <Placemark>
    <name>Flight Path</name>
    <LineString>
      <altitudeMode>absolute</altitudeMode>
      <tessellate>1</tessellate>
      <coordinates>
{coords}
      </coordinates>
    </LineString>
  </Placemark>
</Document>
</kml>"""
    path = save_path or cfg.OUTPUTS_DIR / "trajectory.kml"
    path.write_text(kml)
    print(f"  ✓ KML saved: {path}")
    return kml


# ─── Visualization ────────────────────────────────────────────────────────────
def plot_3d(traj_df: pd.DataFrame, lands_df: pd.DataFrame = None,
            save_path: Path = None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Ellipse
    import config as cfg

    if cfg.DARK_THEME:
        plt.rcParams.update({"figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
                             "text.color":"#c8d8f0","axes.labelcolor":"#c8d8f0",
                             "xtick.color":"#c8d8f0","ytick.color":"#c8d8f0",
                             "grid.color":"#1a2744"})

    fig = plt.figure(figsize=(16,9))
    gs  = plt.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.38,
                       top=0.90,bottom=0.08,left=0.06,right=0.97)

    # 3D trajectory
    ax3 = fig.add_subplot(gs[:,0], projection="3d")
    x   = traj_df["x_east_m"].values
    y   = traj_df["y_north_m"].values
    z   = traj_df["altitude_m"].values
    t   = traj_df["time_s"].values

    sc = ax3.scatter(x, y, z, c=t, cmap="plasma", s=1.5, alpha=0.8)
    ax3.set_xlabel("East [m]",fontsize=7); ax3.set_ylabel("North [m]",fontsize=7)
    ax3.set_zlabel("Alt [m]",fontsize=7)
    ax3.set_title("3D Trajectory",fontweight="bold")
    plt.colorbar(sc,ax=ax3,pad=0.1,label="t [s]",shrink=0.6)
    # Ground projection
    ax3.plot(x, y, np.zeros_like(z), color="#555", lw=0.5, alpha=0.4)

    # Drift vs time
    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(t, traj_df["drift_m"].values, color=cfg.COLOR_THEORY, lw=1.8)
    ax2.set_title("Horizontal drift vs time",fontweight="bold",fontsize=9)
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Drift [m]")
    ax2.grid(True,alpha=0.3)

    # Speed vs altitude
    ax_sv = fig.add_subplot(gs[1,1])
    ax_sv.plot(traj_df["speed_ms"].values, z, color=cfg.COLOR_PINN, lw=1.8)
    ax_sv.set_title("Speed vs altitude",fontweight="bold",fontsize=9)
    ax_sv.set_xlabel("Speed [m/s]"); ax_sv.set_ylabel("Altitude [m]")
    ax_sv.grid(True,alpha=0.3)

    # Landing ellipse
    ax_le = fig.add_subplot(gs[:,2])
    ax_le.set_aspect("equal")
    ax_le.axhline(0,color="#555",lw=0.5); ax_le.axvline(0,color="#555",lw=0.5)
    ax_le.scatter([0],[0],s=80,color=cfg.COLOR_RAW,zorder=5,marker="^",label="Deployment")
    ax_le.scatter([x[-1]],[y[-1]],s=80,color=cfg.COLOR_PINN,zorder=5,marker="x",
                  label=f"Nominal landing\n({x[-1]:+.0f},{y[-1]:+.0f})m")

    if lands_df is not None and len(lands_df):
        lx = lands_df["x_east"].values; ly = lands_df["y_north"].values
        ax_le.scatter(lx, ly, s=4, alpha=0.35, color=cfg.COLOR_THEORY, label=f"MC scatter (n={len(lands_df)})")
        for pct,alpha in [(50,0.5),(90,0.25)]:
            from matplotlib.patches import Ellipse as MPEllipse
            from scipy.stats import chi2
            cov  = np.cov(lx, ly)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]; vals=vals[order]; vecs=vecs[:,order]
            angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            scale = np.sqrt(chi2.ppf(pct/100, 2))
            ell   = MPEllipse(xy=(lx.mean(),ly.mean()),
                              width=2*scale*np.sqrt(vals[0]),
                              height=2*scale*np.sqrt(vals[1]),
                              angle=angle,
                              edgecolor=cfg.COLOR_PINN if pct==90 else cfg.COLOR_RAW,
                              facecolor="none", lw=1.2, ls="--",
                              label=f"{pct}% ellipse")
            ax_le.add_patch(ell)

    ax_le.set_title("Landing zone prediction",fontweight="bold",fontsize=9)
    ax_le.set_xlabel("East [m]"); ax_le.set_ylabel("North [m]")
    ax_le.legend(fontsize=7); ax_le.grid(True,alpha=0.3)

    fig.text(0.5,0.96,"3D Wind-Drift Trajectory & Landing Zone",
             ha="center",fontsize=12,fontweight="bold")

    sp = save_path or cfg.OUTPUTS_DIR/"trajectory_3d.png"
    fig.savefig(sp,facecolor=fig.get_facecolor(),dpi=cfg.DPI)
    print(f"  ✓ 3D trajectory plot saved: {sp}")
    return fig


# ─── Entry Point ─────────────────────────────────────────────────────────────
def run(at_model=None,
        wind_speed:  float = 8.0,
        wind_dir:    float = 270.0,
        wind_model:  str   = "powerlaw",
        lat:         float | None = None,
        lon:         float | None = None,
        target_dt          = None,
        wind_csv:    Path  | None = None,
        mc_n:        int   = 150,
        ) -> pd.DataFrame:
    """
    Run Phase 6 3D trajectory.

    Wind source priority (first available wins):
      1. lat/lon  → Open-Meteo live fetch (free, no key)
      2. wind_csv → pre-saved tabular profile CSV
      3. powerlaw / constant → parametric model
    """
    # ── Wind profile selection ────────────────────────────────────────────────
    if lat is not None and lon is not None:
        print(f"\n[Phase 6] 3D Trajectory — Open-Meteo wind  lat={lat}  lon={lon}")
        wind = OpenMeteoWind(lat=lat, lon=lon, target_dt=target_dt)
    elif wind_csv is not None and Path(wind_csv).exists():
        print(f"\n[Phase 6] 3D Trajectory — tabular wind from CSV: {wind_csv}")
        from src.fetch_wind import build_wind_profile
        wind = build_wind_profile(csv_path=wind_csv)
    elif wind_model == "constant":
        print(f"\n[Phase 6] 3D Trajectory — constant wind {wind_speed:.1f} m/s @ {wind_dir}°")
        wind = ConstantWind(wind_speed, wind_dir)
    else:
        print(f"\n[Phase 6] 3D Trajectory — power-law wind {wind_speed:.1f} m/s @ {wind_dir}°")
        wind = PowerLawWind(speed_ref=wind_speed, direction_deg=wind_dir)

    # ── Solve ─────────────────────────────────────────────────────────────────
    traj  = Trajectory3D(wind=wind, at_model=at_model)
    t_end = cfg.INITIAL_ALT / max(1, cfg.INITIAL_VEL) + 100
    df    = traj.solve(t_eval=np.linspace(0, t_end, 3000))

    df.to_csv(cfg.OUTPUTS_DIR / "trajectory_3d.csv", index=False)
    print(f"  ✓ Trajectory CSV saved")

    to_kml(df)
    lands_df = landing_ellipse_from_mc(n_mc=mc_n)
    lands_df.to_csv(cfg.OUTPUTS_DIR / "landing_ellipse.csv", index=False)

    plot_3d(df, lands_df)
    return df


if __name__ == "__main__":
    run()
