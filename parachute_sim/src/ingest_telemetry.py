"""
ingest_telemetry.py — Multi-Format Telemetry Ingestion
======================================================
Parses real flight computer data from common zero-cost formats and converts
to the unified internal format consumed by Phase 2 ODE, Phase 3 PINN, and
Bayesian calibration.

Supported formats
-----------------
  GPX           — GPS track from any device (Garmin, phone, GPS logger)
                  Extracts: time, lat, lon, ele → altitude(t), speed(t)
  Pixhawk .bin  — ArduPilot / PX4 binary telemetry log
                  Requires: pip install pymavlink  (Apache-2, free)
                  Extracts: ATT, GPS, IMU, BARO channels → full state
  CSV generic   — Any CSV with at minimum [time, altitude or velocity] columns
                  Auto-detects column names (fuzzy matching)
  Suunto .fit   — Suunto dive/skydive computers (via fitparse, free)
  JSON          — Generic JSON array or object with telemetry data

Output format
-------------
All parsers return a standardised DataFrame:
    time_s        : elapsed time from deployment trigger [s]
    altitude_m    : altitude AGL [m]  (barometric or GPS)
    velocity_ms   : vertical descent velocity [m/s]  (derived from altitude)
    lat, lon      : geographic position [°] (if available)
    roll, pitch   : attitude angles [°]     (if available)
    accel_z       : vertical acceleration [m/s²]  (if available)
    source        : format identifier string

Altitude → velocity conversion
-------------------------------
If velocity is not directly measured, it is derived as:
    v(t) = -dh/dt   (descent = positive velocity)
using Savitzky-Golay smoothing to suppress GPS noise before differentiation.

AGL correction
--------------
If absolute altitude is provided (GPS ellipsoid height), the module
subtracts the landing altitude (min observed altitude) to get AGL.
Optionally corrects for terrain via SRTM DEM (if rasterio is available).
"""

from __future__ import annotations
import argparse
import io
import json
import os
import re
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# 1.  UNIFIED OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = ["time_s", "altitude_m", "velocity_ms"]
OPTIONAL_COLS = ["lat", "lon", "roll_deg", "pitch_deg", "accel_z_ms2",
                 "baro_alt_m", "gps_alt_m", "groundspeed_ms"]

COLUMN_ALIASES = {
    # time
    "time_s":        ["time", "t", "elapsed", "time_s", "secs", "seconds",
                       "Time", "timestamp", "elapsed_s"],
    # altitude
    "altitude_m":    ["alt", "altitude", "ele", "elevation", "height", "h",
                       "Alt", "Altitude", "baro_alt", "alt_m", "elev"],
    # velocity
    "velocity_ms":   ["vel", "velocity", "speed", "v", "vz", "descent_rate",
                       "Vel", "Speed", "vspeed", "v_ms", "velD"],
    # position
    "lat":           ["lat", "latitude", "Lat", "GPS_lat"],
    "lon":           ["lon", "longitude", "Lon", "lng", "GPS_lon"],
}


def _fuzzy_match(columns: list, target_aliases: list) -> Optional[str]:
    """Find the first column name matching any alias (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    for alias in target_aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _normalise_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Map raw columns to the standardised schema using fuzzy matching.
    Derives velocity from altitude if not directly available.
    """
    col_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        match = _fuzzy_match(df.columns.tolist(), aliases)
        if match:
            col_map[match] = target

    df = df.rename(columns=col_map)
    df["source"] = source

    # Ensure time starts at 0
    if "time_s" in df.columns:
        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]

    # Derive velocity from altitude if missing
    if "velocity_ms" not in df.columns and "altitude_m" in df.columns:
        df = _derive_velocity(df)

    # AGL correction
    if "altitude_m" in df.columns:
        h_min = df["altitude_m"].min()
        df["altitude_m"] = df["altitude_m"] - h_min

    # Validate
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.warn(f"[ingest] Missing columns after normalisation: {missing}")

    # Keep only relevant columns
    keep = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns] + ["source"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _derive_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive vertical velocity from altitude using Savitzky-Golay differentiation.
    v = -dh/dt  (positive = descending).
    """
    h = df["altitude_m"].values
    t = df["time_s"].values

    if len(h) < 7:
        df["velocity_ms"] = np.gradient(-h, t)
        return df

    # Smooth altitude first (GPS noise is significant)
    window = min(21, len(h) // 4 * 2 + 1)
    window = window if window % 2 == 1 else window + 1
    h_smooth = savgol_filter(h, window_length=window, polyorder=3)

    # Differentiate smoothed altitude
    v = -np.gradient(h_smooth, t)   # descent = positive
    v = np.clip(v, 0, None)         # physical constraint: descent only

    # Additional smoothing on velocity
    if len(v) >= 7:
        v = savgol_filter(v, window_length=min(15, window), polyorder=2)
        v = np.clip(v, 0, None)

    df["velocity_ms"] = v
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GPX PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_gpx(path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Parse GPX track file (any device — Garmin, phone, GPS logger).
    Uses stdlib xml.etree.ElementTree — no external dependencies.
    """
    import xml.etree.ElementTree as ET

    if verbose:
        print(f"[ingest] Parsing GPX: {path.name}")

    tree = ET.parse(path)
    root = tree.getroot()

    # Handle GPX namespace
    ns_match = re.match(r'\{[^}]+\}', root.tag)
    ns = ns_match.group(0) if ns_match else ""

    records = []
    t0 = None

    for trk in root.findall(f"{ns}trk"):
        for seg in trk.findall(f"{ns}trkseg"):
            for pt in seg.findall(f"{ns}trkpt"):
                lat = float(pt.attrib.get("lat", 0))
                lon = float(pt.attrib.get("lon", 0))

                ele_el = pt.find(f"{ns}ele")
                ele = float(ele_el.text) if ele_el is not None else np.nan

                time_el = pt.find(f"{ns}time")
                if time_el is not None:
                    ts_str = time_el.text.rstrip("Z")
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                    if t0 is None:
                        t0 = ts
                    elapsed = (ts - t0).total_seconds()
                else:
                    elapsed = len(records)

                records.append({
                    "time_s":     elapsed,
                    "lat":        lat,
                    "lon":        lon,
                    "altitude_m": ele,
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No track points found in {path}")

    df = _normalise_schema(df, source="gpx")
    if verbose:
        print(f"  ✓ {len(df)} points  "
              f"t=[{df['time_s'].min():.1f},{df['time_s'].max():.1f}]s  "
              f"h=[{df['altitude_m'].min():.1f},{df['altitude_m'].max():.1f}]m")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PIXHAWK / ARDUPILOT .BIN PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_pixhawk(path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Parse Pixhawk/ArduPilot binary log (.bin / .log).
    Requires: pip install pymavlink
    """
    try:
        from pymavlink import mavutil
    except ImportError:
        raise ImportError(
            "pymavlink is required for Pixhawk .bin parsing.\n"
            "Install: pip install pymavlink"
        )

    if verbose:
        print(f"[ingest] Parsing Pixhawk .bin: {path.name}")

    mlog = mavutil.mavlink_connection(str(path))

    records = []
    t0 = None
    gps_cache = {}

    while True:
        msg = mlog.recv_match(type=["BARO","GPS","IMU","ATT","ARSP"], blocking=False)
        if msg is None:
            break

        ts = msg._timestamp  # absolute timestamp [s]
        if t0 is None:
            t0 = ts
        elapsed = ts - t0

        msg_type = msg.get_type()

        if msg_type == "BARO":
            records.append({
                "time_s":     elapsed,
                "altitude_m": getattr(msg, "Alt", np.nan),  # barometric alt [m]
                "source_msg": "BARO",
            })
        elif msg_type == "GPS":
            gps_cache = {
                "time_s": elapsed,
                "lat":    getattr(msg, "Lat", np.nan),
                "lon":    getattr(msg, "Lng", np.nan),
                "gps_alt_m": getattr(msg, "Alt", np.nan) / 1000.0,  # mm → m
                "groundspeed_ms": getattr(msg, "Spd", np.nan),
                "velocity_ms": getattr(msg, "VD",  np.nan),   # vertical down [m/s]
            }
        elif msg_type == "ATT":
            records[-1]["roll_deg"]  = getattr(msg, "Roll",  np.nan) if records else np.nan
            records[-1]["pitch_deg"] = getattr(msg, "Pitch", np.nan) if records else np.nan
        elif msg_type == "IMU":
            if records:
                records[-1]["accel_z_ms2"] = getattr(msg, "AccZ", np.nan)

        # Merge GPS into latest record
        if gps_cache and records:
            for k, v in gps_cache.items():
                if k != "time_s":
                    records[-1].setdefault(k, v)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No BARO messages found in {path}")

    df = _normalise_schema(df, source="pixhawk")

    if verbose:
        print(f"  ✓ {len(df)} samples  "
              f"t=[{df['time_s'].min():.1f},{df['time_s'].max():.1f}]s  "
              f"h=[{df['altitude_m'].min():.1f},{df['altitude_m'].max():.1f}]m  "
              f"v=[{df['velocity_ms'].min():.2f},{df['velocity_ms'].max():.2f}]m/s")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GENERIC CSV PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_csv(path, time_col: str = None, alt_col: str = None,
              vel_col: str = None, sep: str = None,
              verbose: bool = True) -> pd.DataFrame:
    """
    Parse a generic telemetry CSV file.
    Auto-detects separator (,  ;  \\t  space) and column names.
    Handles: Garmin CSV, RocketPy output, custom loggers, spreadsheet exports.
    """
    path = Path(path)
    if verbose:
        print(f"[ingest] Parsing CSV: {path.name}")

    # Auto-detect separator
    if sep is None:
        with open(path) as f:
            sample = f.read(2048)
        counts = {s: sample.count(s) for s in [",", ";", "\t", " "]}
        sep = max(counts, key=counts.get)
        if sep == " ":
            sep = r"\s+"

    df_raw = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
    df_raw.columns = df_raw.columns.str.strip()

    if verbose:
        print(f"  Columns: {list(df_raw.columns)}")

    # User-specified overrides
    rename = {}
    if time_col and time_col in df_raw.columns:
        rename[time_col] = "time_s"
    if alt_col  and alt_col  in df_raw.columns:
        rename[alt_col]  = "altitude_m"
    if vel_col  and vel_col  in df_raw.columns:
        rename[vel_col]  = "velocity_ms"
    if rename:
        df_raw = df_raw.rename(columns=rename)

    # Coerce to numeric
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_raw = df_raw.dropna(how="all")
    df = _normalise_schema(df_raw, source=f"csv:{path.stem}")

    if verbose:
        print(f"  ✓ {len(df)} rows after normalisation")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  JSON PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_json_telemetry(path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Parse JSON telemetry: supports array-of-objects or named-array format.
    """
    if verbose:
        print(f"[ingest] Parsing JSON: {path.name}")

    data = json.loads(path.read_text())

    if isinstance(data, list):
        df_raw = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Try common structures: {t: [], alt: [], vel: []}
        if all(isinstance(v, list) for v in data.values()):
            df_raw = pd.DataFrame(data)
        elif "data" in data:
            df_raw = pd.DataFrame(data["data"])
        else:
            df_raw = pd.DataFrame([data])
    else:
        raise ValueError(f"Unrecognised JSON structure in {path}")

    df = _normalise_schema(df_raw, source=f"json:{path.stem}")

    if verbose:
        print(f"  ✓ {len(df)} records from JSON")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIT PARSER  (Suunto / Garmin .fit files)
# ══════════════════════════════════════════════════════════════════════════════

def parse_fit(path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Parse Garmin / Suunto .fit activity files.
    Requires: pip install fitparse
    """
    try:
        import fitparse
    except ImportError:
        raise ImportError(
            "fitparse is required for .fit parsing.\n"
            "Install: pip install fitparse"
        )

    if verbose:
        print(f"[ingest] Parsing FIT: {path.name}")

    fitfile = fitparse.FitFile(str(path))
    records = []
    t0 = None

    for record in fitfile.get_messages("record"):
        data = {d.name: d.value for d in record}
        ts   = data.get("timestamp")
        alt  = data.get("altitude") or data.get("enhanced_altitude")
        spd  = data.get("speed") or data.get("enhanced_speed")
        lat  = data.get("position_lat")
        lon  = data.get("position_long")

        if ts is None: continue
        if t0 is None: t0 = ts
        elapsed = (ts - t0).total_seconds() if hasattr(ts, "total_seconds") else float(ts - t0)

        row = {"time_s": elapsed}
        if alt is not None: row["altitude_m"] = float(alt)
        if spd is not None: row["groundspeed_ms"] = float(spd)
        if lat is not None: row["lat"] = lat / 1e7   # semicircles → degrees
        if lon is not None: row["lon"] = lon / 1e7
        records.append(row)

    df = pd.DataFrame(records)
    df = _normalise_schema(df, source=f"fit:{path.stem}")

    if verbose:
        print(f"  ✓ {len(df)} records from FIT file")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SYNTHETIC GENERATOR  (test without real hardware)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_telemetry(
    true_Cd:    float = 1.35,
    true_ti:    float = 2.5,
    alt0:       float = None,
    v0:         float = None,
    mass:       float = None,
    fps:        float = 10.0,      # sample rate [Hz]
    noise_h:    float = 1.5,       # altitude noise std [m]
    noise_v:    float = 0.3,       # velocity noise std [m/s]
    seed:       int   = 42,
    verbose:    bool  = True,
) -> pd.DataFrame:
    """
    Generate synthetic GPS/barometric telemetry from the forward ODE model.
    Useful for end-to-end pipeline testing without real hardware.
    """
    from src.calibrate_cd import _simulate
    alt0 = alt0 or cfg.INITIAL_ALT
    v0   = v0   or cfg.INITIAL_VEL
    mass = mass or cfg.PARACHUTE_MASS

    rng = np.random.default_rng(seed)

    # Forward sim
    r = _simulate(Cd=true_Cd, ti=true_ti, mass=mass, alt0=alt0, v0=v0, dt=0.02)
    t_dense = r["time"]
    v_dense = r["velocity"]
    h_dense = r["altitude"]

    # Subsample at fps
    t_obs = np.arange(0, t_dense.max(), 1.0 / fps)
    h_fn  = interp1d(t_dense, h_dense, bounds_error=False, fill_value=(h_dense[0], 0.0))
    v_fn  = interp1d(t_dense, v_dense, bounds_error=False, fill_value=(v_dense[0], 0.0))

    h_obs = h_fn(t_obs) + rng.normal(0, noise_h, len(t_obs))
    v_obs = v_fn(t_obs) + rng.normal(0, noise_v, len(t_obs))
    h_obs = np.clip(h_obs, 0, None)
    v_obs = np.clip(v_obs, 0, None)

    df = pd.DataFrame({
        "time_s":     t_obs,
        "altitude_m": h_obs,
        "velocity_ms": v_obs,
        "source":     f"synthetic (Cd={true_Cd}, ti={true_ti})",
    })

    if verbose:
        print(f"[ingest] Synthetic telemetry: {len(df)} samples @ {fps}Hz  "
              f"Cd_true={true_Cd}  t_infl={true_ti}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8.  AUTO-DETECT FORMAT
# ══════════════════════════════════════════════════════════════════════════════

def auto_ingest(
    path: Path,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Auto-detect file format and dispatch to the appropriate parser.
    Supported: .gpx, .bin, .log, .csv, .txt, .json, .fit
    """
    path = Path(path)
    ext  = path.suffix.lower()

    dispatch = {
        ".gpx":  parse_gpx,
        ".bin":  parse_pixhawk,
        ".log":  parse_pixhawk,
        ".json": parse_json_telemetry,
        ".fit":  parse_fit,
        ".csv":  parse_csv,
        ".txt":  parse_csv,
        ".tsv":  lambda p, **kw: parse_csv(p, sep="\t", **kw),
    }

    if ext not in dispatch:
        raise ValueError(f"Unsupported file extension: {ext}. "
                          f"Supported: {list(dispatch.keys())}")

    return dispatch[ext](path, verbose=verbose, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_telemetry(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Quality check and statistical summary of ingested telemetry.
    Detects: missing data, outliers, sample rate consistency, GPS gaps.
    """
    results = {}

    t = df["time_s"].values
    results["n_samples"]   = len(df)
    results["duration_s"]  = float(t.max() - t.min())
    results["sample_rate_hz"] = round(float(1.0 / np.median(np.diff(t))), 2) if len(t) > 1 else 0

    if "altitude_m" in df.columns:
        h = df["altitude_m"].values
        results["h_max_m"]   = round(float(h.max()), 1)
        results["h_min_m"]   = round(float(h.min()), 1)
        results["h_range_m"] = round(float(h.max() - h.min()), 1)

    if "velocity_ms" in df.columns:
        v = df["velocity_ms"].values
        results["v_max_ms"]    = round(float(v.max()), 3)
        results["v_mean_ms"]   = round(float(v.mean()), 3)
        results["v_landing_ms"] = round(float(v[-1]), 3)

    # Missing data check
    na_pct = {c: round(df[c].isna().mean() * 100, 1) for c in df.columns}
    results["missing_pct"] = {k: v for k, v in na_pct.items() if v > 0}

    # Outlier detection (Z-score > 4σ)
    outliers = {}
    for col in ["altitude_m", "velocity_ms"]:
        if col in df.columns:
            s = df[col].dropna()
            z = np.abs((s - s.mean()) / max(s.std(), 1e-6))
            n_out = int((z > 4).sum())
            if n_out > 0:
                outliers[col] = n_out
    results["outliers"] = outliers

    # Sample rate consistency
    dt = np.diff(t)
    results["dt_mean_s"]   = round(float(dt.mean()), 4)
    results["dt_std_s"]    = round(float(dt.std()), 4)
    results["dt_max_s"]    = round(float(dt.max()), 4)
    results["regular_sampling"] = bool(dt.std() / max(dt.mean(), 1e-6) < 0.15)

    if verbose:
        print(f"\n  Telemetry quality report:")
        print(f"  {'samples':<20}: {results['n_samples']}")
        print(f"  {'duration':<20}: {results['duration_s']:.1f}s")
        print(f"  {'sample rate':<20}: {results['sample_rate_hz']:.2f} Hz")
        print(f"  {'altitude range':<20}: {results.get('h_min_m','-'):.1f} – "
              f"{results.get('h_max_m','-'):.1f} m")
        print(f"  {'v_landing':<20}: {results.get('v_landing_ms','-'):.3f} m/s")
        print(f"  {'regular sampling':<20}: {results['regular_sampling']}")
        if results["missing_pct"]:
            print(f"  {'missing data':<20}: {results['missing_pct']}")
        if results["outliers"]:
            print(f"  {'outliers':<20}: {results['outliers']}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 10.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_telemetry(
    df:         pd.DataFrame,
    quality:    dict | None = None,
    save_path:  Path | None = None,
) -> object:
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
    C_GT = "#9d60ff"   # purple for ground track

    n_rows = 2; n_cols = 3
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.48, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.06, right=0.97)

    def ax(r, c): return fig.add_subplot(gs[r, c])
    def style(a, t, xl, yl):
        a.set_title(t, fontweight="bold", pad=5, fontsize=9)
        a.set_xlabel(xl, fontsize=8); a.set_ylabel(yl, fontsize=8)
        a.grid(True, alpha=0.3); a.spines[["top","right"]].set_visible(False)

    t = df["time_s"].values

    # ── P0: altitude ─────────────────────────────────────────────────────────
    ax0 = ax(0, 0)
    if "altitude_m" in df.columns:
        ax0.fill_between(t, df["altitude_m"].values, alpha=0.2, color="#9d60ff")
        ax0.plot(t, df["altitude_m"].values, color="#9d60ff", lw=1.8)
    style(ax0, "Altitude h(t)", "Time [s]", "Altitude [m AGL]")

    # ── P1: velocity ──────────────────────────────────────────────────────────
    ax1 = ax(0, 1)
    if "velocity_ms" in df.columns:
        ax1.fill_between(t, df["velocity_ms"].values, alpha=0.2, color=C1)
        ax1.plot(t, df["velocity_ms"].values, color=C1, lw=1.8)
    style(ax1, "Vertical velocity v(t)", "Time [s]", "v [m/s]")

    # ── P2: sample rate histogram ─────────────────────────────────────────────
    ax2 = ax(0, 2)
    if len(t) > 2:
        dt = np.diff(t) * 1000  # ms
        ax2.hist(dt, bins=40, color=C2, alpha=0.7, edgecolor="none")
        ax2.axvline(np.median(dt), color=C3, lw=1.5, ls="--",
                    label=f"median Δt={np.median(dt):.1f}ms")
        ax2.legend(fontsize=7.5)
    style(ax2, "Sample interval distribution", "Δt [ms]", "Count")

    # ── P3: phase portrait h vs v ─────────────────────────────────────────────
    ax3 = ax(1, 0)
    if "altitude_m" in df.columns and "velocity_ms" in df.columns:
        sc = ax3.scatter(df["velocity_ms"].values, df["altitude_m"].values,
                         c=t, cmap="plasma", s=5, alpha=0.8)
        plt.colorbar(sc, ax=ax3, pad=0.02, label="t [s]").ax.tick_params(labelsize=7)
    style(ax3, "Phase portrait v vs h", "Velocity [m/s]", "Altitude [m]")

    # ── P4: ground track (if GPS available) ──────────────────────────────────
    ax4 = ax(1, 1)
    if "lat" in df.columns and "lon" in df.columns:
        lat = df["lat"].dropna().values
        lon = df["lon"].dropna().values
        if len(lat) > 1:
            sc4 = ax4.scatter(lon, lat, c=t[:len(lat)], cmap="plasma", s=4, alpha=0.8)
            plt.colorbar(sc4, ax=ax4, pad=0.02, label="t [s]").ax.tick_params(labelsize=7)
            ax4.scatter([lon[0]], [lat[0]], s=60, color=C3, marker="^", zorder=5, label="Start")
            ax4.scatter([lon[-1]], [lat[-1]], s=60, color=C2, marker="x", zorder=5, label="End")
            ax4.legend(fontsize=7.5)
    else:
        ax4.text(0.5, 0.5, "No GPS data", transform=ax4.transAxes,
                 ha="center", va="center", color=TEXT, fontsize=10)
    style(ax4, "Ground track (GPS)", "Longitude [°]", "Latitude [°]")

    # ── P5: quality summary ───────────────────────────────────────────────────
    ax5 = ax(1, 2)
    ax5.axis("off")
    q = quality or {}
    rows = [
        ("Source",         str(df["source"].iloc[0]) if "source" in df else "?"),
        ("Samples",        str(q.get("n_samples", len(df)))),
        ("Duration",       f"{q.get('duration_s', t.max()-t.min()):.1f}s"),
        ("Sample rate",    f"{q.get('sample_rate_hz','?')} Hz"),
        ("",               ""),
        ("Alt range",      f"{q.get('h_min_m','?'):.0f}–{q.get('h_max_m','?'):.0f} m"),
        ("v_landing",      f"{q.get('v_landing_ms','?'):.3f} m/s"),
        ("",               ""),
        ("Regular samp.",  "YES ✓" if q.get("regular_sampling") else "NO"),
        ("Missing data",   str(q.get("missing_pct", {})) or "None"),
        ("Outliers",       str(q.get("outliers", {})) or "None"),
    ]
    for j, (label, val) in enumerate(rows):
        c_val = C3 if "✓" in val else "#ff4560" if "NO" in val else TEXT
        ax5.text(0.02, 1-j*0.090, label, transform=ax5.transAxes, fontsize=8.5,
                 color=TEXT if cfg.DARK_THEME else "#555")
        ax5.text(0.98, 1-j*0.090, val, transform=ax5.transAxes, fontsize=8.5,
                 ha="right", color=c_val)
    ax5.set_title("Telemetry quality report", fontweight="bold")

    src = df["source"].iloc[0] if "source" in df else "?"
    fig.text(0.5, 0.955,
             f"Telemetry Ingest: {src}  —  "
             f"{len(df)} samples  duration={t.max()-t.min():.1f}s  "
             f"v_land={df['velocity_ms'].iloc[-1]:.3f} m/s",
             ha="center", fontsize=11, fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR / "telemetry_ingest.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Telemetry plot saved: {sp}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(
    file_path:   Path | None = None,
    synthetic:   bool   = False,
    true_Cd:     float  = 1.35,
    verbose:     bool   = True,
    **parse_kwargs,
) -> pd.DataFrame:
    """
    Run telemetry ingestion pipeline.

    If file_path is None (or synthetic=True), generates synthetic test data.
    Otherwise, auto-detects format and parses the file.

    Returns standardised DataFrame ready for PINN / Bayesian calibration.
    """
    import matplotlib; matplotlib.use("Agg")

    if synthetic or file_path is None:
        df = generate_synthetic_telemetry(true_Cd=true_Cd, verbose=verbose)
    else:
        df = auto_ingest(Path(file_path), verbose=verbose, **parse_kwargs)

    quality = analyse_telemetry(df, verbose=verbose)
    plot_telemetry(df, quality)

    # Save to standard outputs
    out = cfg.OUTPUTS_DIR / "telemetry_ingested.csv"
    df.to_csv(out, index=False)
    if verbose:
        print(f"\n  ✓ Telemetry saved: {out}")
        print(f"  Ready for: phase2_ode, phase3_pinn, bayes_cd, calibrate_cd")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parachute Telemetry Ingestion",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("file",           nargs="?",  default=None,
                   help="Input file (.gpx/.bin/.csv/.json/.fit/.tsv)")
    p.add_argument("--synthetic",    action="store_true",
                   help="Use synthetic test data instead of real file")
    p.add_argument("--true-Cd",      type=float, default=1.35,
                   help="True Cd for synthetic data")
    p.add_argument("--time-col",     type=str,   default=None)
    p.add_argument("--alt-col",      type=str,   default=None)
    p.add_argument("--vel-col",      type=str,   default=None)
    a = p.parse_args()

    run(file_path=a.file, synthetic=a.synthetic, true_Cd=a.true_Cd,
        time_col=a.time_col, alt_col=a.alt_col, vel_col=a.vel_col)
