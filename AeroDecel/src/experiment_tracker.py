"""
src/experiment_tracker.py — Versioned Experiment Tracking (SQLite)
===================================================================
Logs every simulation run to a local SQLite database.
Query interface for finding runs by performance criteria.

Usage
-----
    tracker = ExperimentTracker()
    run_id  = tracker.log_run(params, results, tags)
    df      = tracker.query("sf_tps > 2 AND q_peak_MW < 0.5 AND planet = 'mars'")
    tracker.summary()
    tracker.plot_history()
"""
from __future__ import annotations
import sqlite3
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime


DB_PATH = Path("outputs/experiments.db")


class ExperimentTracker:
    """
    SQLite-backed experiment tracker for AeroDecel simulations.

    Schema
    ------
    runs(id, run_id, timestamp, planet, material, tags, params_json, results_json,
         duration_s, status)

    Quick metrics columns (pre-extracted for fast querying):
    v_land_ms, q_peak_MW, sf_tps, tps_mass_kgm2, p_success,
    n_mc_samples, entry_fpa_deg, tps_thickness_mm
    """

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE,
                    timestamp TEXT,
                    planet TEXT,
                    material TEXT,
                    tags TEXT,
                    params_json TEXT,
                    results_json TEXT,
                    duration_s REAL,
                    status TEXT,
                    -- Quick-query columns
                    v_land_ms REAL,
                    q_peak_MW REAL,
                    sf_tps REAL,
                    tps_mass_kgm2 REAL,
                    p_success REAL,
                    entry_fpa_deg REAL,
                    tps_thickness_mm REAL,
                    n_mc_samples INTEGER,
                    notes TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_planet ON runs(planet)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp)
            """)
            conn.commit()

    def _make_run_id(self, params: dict) -> str:
        """Deterministic run ID from parameter hash + timestamp."""
        h = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{h}"

    def log_run(self, params: dict, results: dict,
                tags: list[str] | None = None,
                notes: str = "",
                duration_s: float = 0.0) -> str:
        """
        Log a simulation run to the database.

        Parameters
        ----------
        params   : dict of input parameters
        results  : dict of output metrics
        tags     : list of string tags (e.g. ["mars_edl", "pica", "tier1"])
        notes    : free-text notes

        Returns
        -------
        run_id   : unique run identifier
        """
        run_id    = self._make_run_id(params)
        timestamp = datetime.now().isoformat()
        tags_str  = json.dumps(tags or [])

        # Extract quick-query metrics
        v_land     = float(results.get("v_land_ms",          results.get("v_final", 0)))
        q_peak     = float(results.get("q_peak_MWm2",        results.get("q_peak_MW", 0)))
        sf_tps     = float(results.get("sf_tps",             results.get("safety_factor", 0)))
        tps_mass   = float(results.get("tps_mass_kgm2",      results.get("mass_kgm2", 0)))
        p_success  = float(results.get("p_success",          results.get("P_mission_success", -1)))
        entry_fpa  = float(params.get("entry_fpa_deg",       params.get("gamma_deg", 0)))
        tps_thick  = float(params.get("tps_thickness_m",     params.get("thickness_m", 0))) * 1000
        n_mc       = int(results.get("n_mc_samples",         results.get("n_valid", 0)))
        planet     = str(params.get("planet", results.get("planet", "unknown")))
        material   = str(params.get("material", results.get("material", "unknown")))

        # Serialise (handle numpy arrays)
        def _safe(d):
            out = {}
            for k, v in d.items():
                try:
                    json.dumps(v)
                    out[k] = v
                except (TypeError, ValueError):
                    out[k] = str(v)
            return out

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs
                (run_id, timestamp, planet, material, tags, params_json, results_json,
                 duration_s, status, v_land_ms, q_peak_MW, sf_tps, tps_mass_kgm2,
                 p_success, entry_fpa_deg, tps_thickness_mm, n_mc_samples, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, timestamp, planet, material, tags_str,
                json.dumps(_safe(params)), json.dumps(_safe(results)),
                duration_s, "complete",
                v_land, q_peak, sf_tps, tps_mass, p_success,
                entry_fpa, tps_thick, n_mc, notes,
            ))
            conn.commit()

        return run_id

    def query(self, where_clause: str = "1=1",
              limit: int = 100) -> "pd.DataFrame":
        """
        Query runs using SQL WHERE clause.

        Examples
        --------
        tracker.query("sf_tps > 2 AND q_peak_MW < 0.5 AND planet = 'mars'")
        tracker.query("p_success > 0.99 AND material = 'pica'")
        tracker.query("1=1 ORDER BY v_land_ms ASC LIMIT 10")
        """
        import pandas as pd
        sql = f"""
            SELECT run_id, timestamp, planet, material, tags,
                   v_land_ms, q_peak_MW, sf_tps, tps_mass_kgm2,
                   p_success, entry_fpa_deg, tps_thickness_mm, n_mc_samples,
                   duration_s, notes
            FROM runs
            WHERE {where_clause}
            LIMIT {limit}
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn)

    def get_run(self, run_id: str) -> dict | None:
        """Get full run record including params and results JSON."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT params_json, results_json, timestamp, planet FROM runs WHERE run_id=?",
                (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "params":    json.loads(row[0]),
            "results":   json.loads(row[1]),
            "timestamp": row[2],
            "planet":    row[3],
        }

    def summary(self) -> dict:
        """Summary statistics of all logged runs."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            planets = conn.execute(
                "SELECT planet, COUNT(*) FROM runs GROUP BY planet").fetchall()
            best_sf = conn.execute(
                "SELECT run_id, sf_tps, planet, material FROM runs "
                "WHERE sf_tps > 0 ORDER BY sf_tps DESC LIMIT 1").fetchone()
            best_ps = conn.execute(
                "SELECT run_id, p_success, planet FROM runs "
                "WHERE p_success > 0 ORDER BY p_success DESC LIMIT 1").fetchone()

        print(f"\n[Tracker] {total} runs logged in {self.db_path}")
        print(f"  By planet: {dict(planets)}")
        if best_sf:
            print(f"  Best SF:   {best_sf[1]:.3f} ({best_sf[2]}/{best_sf[3]}, id={best_sf[0]})")
        if best_ps:
            print(f"  Best P(success): {best_ps[1]:.6f} ({best_ps[2]}, id={best_ps[0]})")

        return {"total": total, "by_planet": dict(planets)}

    def plot_history(self, save_path: str = "outputs/experiment_history.png"):
        """Plot run history dashboard."""
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        df = self.query("v_land_ms >= 0", limit=500)
        if len(df) == 0:
            print("  [Tracker] No runs to plot")
            return

        matplotlib.rcParams.update({
            "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
            "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
            "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
            "ytick.color":"#c8d8f0","grid.color":"#1a2744",
            "font.family":"monospace","font.size":9,
        })
        TX="#c8d8f0"; C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"

        fig = plt.figure(figsize=(18, 10), facecolor="#080c14")
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35,
                                top=0.90, bottom=0.07, left=0.07, right=0.97)

        def gax(r, c):
            a = fig.add_subplot(gs[r, c])
            a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
            a.tick_params(colors=TX, labelsize=8); a.spines[:].set_color("#2a3d6e")
            return a

        planets = df["planet"].unique()
        pal_ = {"mars":"#ff4560","venus":"#ffd700","titan":"#00d4ff","unknown":"#888"}

        # v_land distribution
        a = gax(0, 0)
        for p in planets:
            sub = df[df["planet"]==p]["v_land_ms"].dropna()
            if len(sub): a.hist(sub, bins=20, alpha=0.6, label=p, color=pal_.get(p,"#888"))
        a.set_xlabel("v_land [m/s]"); a.set_ylabel("Count"); a.legend(fontsize=7.5)
        a.set_title("Landing Velocity Distribution", fontweight="bold")

        # SF distribution
        a = gax(0, 1)
        sf_vals = df["sf_tps"].dropna(); sf_vals = sf_vals[sf_vals > 0]
        if len(sf_vals):
            a.hist(sf_vals, bins=25, color=C3, alpha=0.65)
            a.axvline(1.5, color="#ff4560", lw=1.5, ls="--", label="SF=1.5 min")
            a.legend(fontsize=7.5)
        a.set_xlabel("SF_TPS"); a.set_ylabel("Count"); a.set_title("Safety Factor Dist.", fontweight="bold")

        # P(success) over time
        a = gax(0, 2)
        df_ps = df[df["p_success"] > 0].copy()
        if len(df_ps):
            a.scatter(range(len(df_ps)), df_ps["p_success"], s=6, color=C4, alpha=0.7)
            a.axhline(0.99, color=C3, lw=1, ls="--", label="P=0.99")
            a.legend(fontsize=7.5)
        a.set_xlabel("Run index"); a.set_ylabel("P(success)"); a.set_title("Mission Success History", fontweight="bold")

        # q_peak vs SF
        a = gax(1, 0)
        df_valid = df[(df["q_peak_MW"] > 0) & (df["sf_tps"] > 0)]
        if len(df_valid):
            sc = a.scatter(df_valid["q_peak_MW"], df_valid["sf_tps"],
                           c=df_valid["v_land_ms"], cmap="plasma", s=15, alpha=0.7)
            fig.colorbar(sc, ax=a, label="v_land [m/s]", pad=0.02).ax.tick_params(labelsize=7)
        a.set_xlabel("q_peak [MW/m²]"); a.set_ylabel("SF_TPS")
        a.set_title("Heat Flux vs Safety Factor", fontweight="bold")

        # Material histogram
        a = gax(1, 1)
        mat_counts = df["material"].value_counts()
        mat_col = {"pica":"#ff4560","avcoat":"#ff6b35","srp":"#ffd700",
                   "kevlar":"#00d4ff","nylon":"#9d60ff","unknown":"#888"}
        bars = a.barh(mat_counts.index, mat_counts.values,
                      color=[mat_col.get(m,"#888") for m in mat_counts.index], alpha=0.75)
        a.set_xlabel("Run count"); a.set_title("Runs by Material", fontweight="bold")

        # TPS mass vs v_land
        a = gax(1, 2)
        df_m = df[(df["tps_mass_kgm2"] > 0) & (df["v_land_ms"] > 0)]
        if len(df_m):
            a.scatter(df_m["tps_mass_kgm2"], df_m["v_land_ms"],
                      s=10, alpha=0.6, color=C2)
        a.set_xlabel("TPS mass [kg/m²]"); a.set_ylabel("v_land [m/s]")
        a.set_title("TPS Mass vs Landing Velocity", fontweight="bold")

        fig.text(0.5, 0.955,
                 f"Experiment History  |  {len(df)} runs  |  {df['planet'].nunique()} planets",
                 ha="center", fontsize=11, fontweight="bold", color=TX)

        Path(save_path).parent.mkdir(exist_ok=True)
        fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
        print(f"  ✓ Experiment history plot saved: {save_path}")
        plt.close(fig)


# Module-level singleton
_tracker: ExperimentTracker | None = None

def get_tracker() -> ExperimentTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker
