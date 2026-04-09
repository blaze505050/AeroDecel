"""
fetch_wind.py — Open-Meteo Real Wind Profile (Zero Cost, No API Key)
=====================================================================
Fetches actual measured + forecast wind speed and direction at multiple
pressure levels from the Open-Meteo API and returns a ready-to-use
TabularWind profile for Phase 6 / Phase 7.

API:   https://open-meteo.com  (free, no signup, no key, CC-BY 4.0)
Data:  ERA5 reanalysis + ECMWF IFS forecast, updated hourly

Pressure levels supported:
  1000, 925, 850, 700, 500, 300 hPa
  ≈ 110, 800, 1500, 3000, 5500, 9000 m ISA

Also fetches surface wind at 10 m.

Usage (standalone)
------------------
python src/fetch_wind.py --lat 51.5 --lon -0.12
python src/fetch_wind.py --lat 28.6 --lon 77.2 --datetime 2025-06-15T06:00

Programmatic
------------
from src.fetch_wind import fetch_wind_profile, build_wind_profile
wind = build_wind_profile(lat=51.5, lon=-0.12)   # returns TabularWind
"""

from __future__ import annotations
import argparse
import json
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import pressure as isa_pressure


# ─── Pressure → Altitude conversion using ISA inversion ──────────────────────
# Pre-computed table: pressure [Pa] → altitude [m]
_P_ALT_TABLE = np.array([
    (101325.0,    0.0),
    ( 97700.0,  300.0),
    ( 89880.0,  800.0),      # ≈ 925 hPa (hPa × 100 = Pa)
    ( 84560.0, 1500.0),
    ( 70120.0, 3000.0),      # ≈ 700 hPa
    ( 54050.0, 5000.0),
    ( 50000.0, 5574.0),      # ≈ 500 hPa
    ( 30000.0, 9164.0),      # ≈ 300 hPa
    ( 20000.0,11784.0),
    ( 10000.0,16180.0),
])

def pressure_hpa_to_altitude_m(hpa: float) -> float:
    """
    Convert pressure level [hPa] to geometric altitude [m] via ISA inversion.
    Uses precomputed table + linear interpolation.
    """
    pa = hpa * 100.0
    alts = _P_ALT_TABLE[:, 1]
    pres = _P_ALT_TABLE[:, 0]
    return float(np.interp(pa, pres[::-1], alts[::-1]))


# ─── Open-Meteo API Parameters ───────────────────────────────────────────────
_BASE_URL  = "https://api.open-meteo.com/v1/forecast"
_LEVELS    = [1000, 925, 850, 700, 500, 300]   # hPa
_LEVEL_ALT = {lvl: pressure_hpa_to_altitude_m(lvl) for lvl in _LEVELS}

def _build_url(lat: float, lon: float, target_dt: datetime | None) -> str:
    """Construct Open-Meteo API URL."""
    speed_vars = [f"wind_speed_{lvl}hPa"     for lvl in _LEVELS]
    dir_vars   = [f"wind_direction_{lvl}hPa" for lvl in _LEVELS]
    all_vars   = speed_vars + dir_vars + ["wind_speed_10m", "wind_direction_10m"]

    params = {
        "latitude":        f"{lat:.6f}",
        "longitude":       f"{lon:.6f}",
        "hourly":          ",".join(all_vars),
        "wind_speed_unit": "ms",
        "timezone":        "UTC",
        "forecast_days":   "1",
    }
    return f"{_BASE_URL}?{urllib.parse.urlencode(params)}"


def _fetch_json(url: str, timeout: int = 10) -> dict:
    """Fetch JSON from URL using stdlib urllib (no requests needed)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ParachuteDynamicsSystem/2.0 (educational)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"Open-Meteo fetch failed: {e}") from e


def _nearest_hour_index(data: dict, target_dt: datetime | None) -> int:
    """Find the index of the hourly slot closest to target_dt (or now)."""
    times = data["hourly"]["time"]
    if target_dt is None:
        target_dt = datetime.now(timezone.utc)
    target_str = target_dt.strftime("%Y-%m-%dT%H:00")
    try:
        return times.index(target_str)
    except ValueError:
        # Fall back to first available slot
        return 0


# ─── Main Fetch Function ──────────────────────────────────────────────────────
def fetch_wind_profile(
    lat:       float,
    lon:       float,
    target_dt: datetime | None = None,
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Fetch wind profile from Open-Meteo for a given location and time.

    Returns a DataFrame with columns:
        alt_m       : geometric altitude [m]
        pressure_hPa: pressure level
        speed_ms    : wind speed [m/s]
        direction_deg: meteorological direction (FROM, degrees)
        u_east      : eastward wind component [m/s]
        u_north     : northward wind component [m/s]
        source      : 'open-meteo' or 'synthetic-fallback'

    Sorted by altitude ascending.
    """
    if verbose:
        print(f"\n[Wind] Fetching Open-Meteo profile  lat={lat:.4f}  lon={lon:.4f}")

    url  = _build_url(lat, lon, target_dt)
    data = _fetch_json(url)
    idx  = _nearest_hour_index(data, target_dt)
    h    = data["hourly"]

    rows = []

    # Surface 10 m
    spd10 = h["wind_speed_10m"][idx]
    dir10 = h["wind_direction_10m"][idx]
    if spd10 is not None and dir10 is not None:
        d_rad   = np.deg2rad(dir10)
        u_east  = -spd10 * np.sin(d_rad)   # wind blows FROM direction
        u_north = -spd10 * np.cos(d_rad)
        rows.append(dict(alt_m=10.0, pressure_hPa=None,
                         speed_ms=spd10, direction_deg=dir10,
                         u_east=u_east, u_north=u_north, source="open-meteo"))

    # Pressure levels
    for lvl in _LEVELS:
        spd = h[f"wind_speed_{lvl}hPa"][idx]
        drn = h[f"wind_direction_{lvl}hPa"][idx]
        if spd is None or drn is None:
            continue
        alt    = _LEVEL_ALT[lvl]
        d_rad  = np.deg2rad(drn)
        u_east  = -spd * np.sin(d_rad)
        u_north = -spd * np.cos(d_rad)
        rows.append(dict(alt_m=alt, pressure_hPa=lvl,
                         speed_ms=spd, direction_deg=drn,
                         u_east=u_east, u_north=u_north, source="open-meteo"))

    df = pd.DataFrame(rows).sort_values("alt_m").reset_index(drop=True)

    if verbose:
        print(f"  ✓ {len(df)} levels fetched  (slot: {h['time'][idx]})")
        print(f"  {'Alt (m)':>8} {'hPa':>6} {'Speed':>7} {'Dir':>6} {'U_E':>7} {'U_N':>7}")
        for _, row in df.iterrows():
            lvl_s = f"{row.pressure_hPa:.0f}" if row.pressure_hPa else " sfc"
            print(f"  {row.alt_m:>8.0f} {lvl_s:>6} "
                  f"{row.speed_ms:>6.2f}m/s {row.direction_deg:>5.1f}° "
                  f"{row.u_east:>+7.2f} {row.u_north:>+7.2f}")

    return df


# ─── Synthetic Fallback (when offline) ───────────────────────────────────────
def synthetic_wind_profile(
    surface_speed: float = 5.0,
    surface_dir:   float = 270.0,
    shear_alpha:   float = 0.25,
    backing_deg_per_km: float = 15.0,
    max_alt:       float = 10000.0,
    verbose:       bool  = True,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic wind profile when offline.
    Applies power-law speed shear and directional backing with altitude.
    """
    if verbose:
        print(f"[Wind] Generating synthetic profile (offline fallback)")

    alts = np.array([10, 110, 500, 800, 1500, 3000, 5500, 9000, max_alt])
    rows = []
    for alt in alts:
        spd = surface_speed * max(0.5, (alt / 10) ** shear_alpha) if alt > 10 else surface_speed
        drn = (surface_dir - backing_deg_per_km * (alt / 1000)) % 360
        d_rad   = np.deg2rad(drn)
        u_east  = -spd * np.sin(d_rad)
        u_north = -spd * np.cos(d_rad)
        rows.append(dict(alt_m=float(alt), pressure_hPa=None,
                         speed_ms=round(float(spd), 3), direction_deg=round(float(drn), 1),
                         u_east=round(float(u_east), 4), u_north=round(float(u_north), 4),
                         source="synthetic-fallback"))

    return pd.DataFrame(rows)


# ─── Build TabularWind Profile ────────────────────────────────────────────────
def build_wind_profile(
    lat:       float | None = None,
    lon:       float | None = None,
    target_dt: datetime | None = None,
    csv_path:  Path | None = None,
    fallback_speed: float  = 5.0,
    fallback_dir:   float  = 270.0,
    verbose:   bool  = True,
) -> "TabularWind":
    """
    Build a TabularWind object from the best available data source:
      1. CSV file (pre-saved profile)
      2. Open-Meteo API (if lat/lon provided and network available)
      3. Synthetic fallback

    Returns a TabularWind instance ready for Phase 6/7.
    """
    from src.phase6_trajectory import TabularWind

    df = None

    # Option 1: load from CSV
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"[Wind] Loaded profile from CSV: {csv_path}  ({len(df)} levels)")

    # Option 2: Open-Meteo
    elif lat is not None and lon is not None:
        try:
            df = fetch_wind_profile(lat, lon, target_dt, verbose=verbose)
            out = cfg.OUTPUTS_DIR / "wind_profile.csv"
            df.to_csv(out, index=False)
            if verbose:
                print(f"  Saved: {out}")
        except Exception as e:
            if verbose:
                print(f"  [Wind] Open-Meteo unavailable ({e}) — using synthetic fallback")
            df = synthetic_wind_profile(fallback_speed, fallback_dir, verbose=verbose)

    # Option 3: synthetic fallback
    else:
        df = synthetic_wind_profile(fallback_speed, fallback_dir, verbose=verbose)

    return TabularWind(df)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Fetch Open-Meteo wind profile for a location",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lat",      type=float, required=True,  help="Latitude")
    p.add_argument("--lon",      type=float, required=True,  help="Longitude")
    p.add_argument("--datetime", type=str,   default=None,
                   help="Target UTC datetime (ISO 8601: 2025-06-15T06:00)")
    p.add_argument("--out",      type=Path,
                   default=cfg.OUTPUTS_DIR / "wind_profile.csv",
                   help="Output CSV path")
    p.add_argument("--synthetic-fallback", action="store_true",
                   help="Use synthetic profile even if network is available")
    a = p.parse_args()

    dt = datetime.fromisoformat(a.datetime) if a.datetime else None

    if a.synthetic_fallback:
        df = synthetic_wind_profile()
    else:
        try:
            df = fetch_wind_profile(a.lat, a.lon, dt)
        except Exception as e:
            print(f"[Wind] Fetch failed: {e}  — using synthetic fallback")
            df = synthetic_wind_profile()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n[Wind] ✓ Profile saved: {a.out}")


if __name__ == "__main__":
    main()
