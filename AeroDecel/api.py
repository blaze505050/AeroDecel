"""
api.py — AeroDecel v6.0 REST API (FastAPI)
==========================================
Wraps the core EDL simulation engine as a REST API.

Usage
-----
  pip install fastapi uvicorn
  uvicorn api:app --reload --port 8000

  # POST /simulate
  curl -X POST http://localhost:8000/simulate \
    -H "Content-Type: application/json" \
    -d '{"planet":"mars","entry_velocity_ms":5800,"entry_fpa_deg":-15}'

Endpoints
---------
  GET  /                 — API info
  GET  /health           — health check
  GET  /planets          — list available planets with properties
  GET  /materials        — list TPS materials
  POST /simulate         — run EDL simulation → trajectory + metrics
  POST /ablation         — run ablation-only analysis
  POST /monte_carlo      — run MC uncertainty propagation
  POST /fault_tree       — run FTA analysis
  GET  /experiments      — query logged experiments (SQLite)

Deploy free on Render.com or Railway.app
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

import numpy as np
import json

# ── Simple fallback if FastAPI not installed ──────────────────────────────────
if not _FASTAPI:
    print("FastAPI not installed. Install: pip install fastapi uvicorn")
    print("Running CLI demo instead...")

    import sys
    sys.path.insert(0, ".")
    from src.planetary_atm import MarsAtmosphere
    from src.multifidelity_pinn import LowFidelityEDL

    planet = MarsAtmosphere()
    lf = LowFidelityEDL(planet, 900, 1.7, 78.5, gamma_deg=15)
    t  = np.linspace(0, 400, 100)
    v, h = lf.solve(t, 5800, 125_000)
    print(f"Demo EDL: v_final={v[-1]:.2f}m/s  h_final={h[-1]:.0f}m")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# APP + MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title        = "AeroDecel v6.0 — Project Icarus",
    description  = ("Planetary EDL simulation API. "
                    "Full physics: real-gas CO₂, 6-DOF, ablation, flutter, LBM, PINN."),
    version      = "6.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class SimulateRequest(BaseModel):
    planet:            str   = Field("mars",   description="mars | venus | titan")
    entry_velocity_ms: float = Field(5800.0,   description="Entry velocity [m/s]")
    entry_fpa_deg:     float = Field(-15.0,    description="Flight path angle [°]")
    entry_alt_m:       float = Field(125_000., description="Entry altitude [m]")
    mass_kg:           float = Field(900.0,    description="Vehicle mass [kg]")
    canopy_area_m2:    float = Field(78.5,     description="Canopy area [m²]")
    drag_coeff:        float = Field(1.7,      description="Drag coefficient")
    tps_material:      str   = Field("pica",   description="TPS material")
    tps_thickness_m:   float = Field(0.05,     description="TPS thickness [m]")
    nose_radius_m:     float = Field(4.5,      description="Nose radius [m]")
    use_realgas:       bool  = Field(False,    description="Use real-gas CO₂ chemistry")
    n_output_points:   int   = Field(100,      description="Number of output time points")


class AblationRequest(BaseModel):
    material:      str   = Field("pica",  description="Ablative material")
    thickness_m:   float = Field(0.05,   description="TPS thickness [m]")
    q_peak_MW:     float = Field(15.0,   description="Peak heat flux [MW/m²]")
    t_entry_s:     float = Field(200.0,  description="Entry duration [s]")


class MCRequest(BaseModel):
    planet:        str   = Field("mars")
    n_samples:     int   = Field(100,    ge=10, le=1000)
    use_realgas:   bool  = Field(False)
    mass_kg:       float = Field(900.0)


class FTARequest(BaseModel):
    sf_tps:        float = Field(1.5)
    sf_structure:  float = Field(2.0)
    n_mc:          int   = Field(2000, ge=100, le=20000)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name":        "AeroDecel v6.0 — Project Icarus",
        "version":     "6.0.0",
        "description": "Planetary EDL simulation REST API",
        "endpoints":   ["/simulate", "/ablation", "/monte_carlo", "/fault_tree",
                         "/planets", "/materials", "/experiments", "/docs"],
        "github":      "https://github.com/blaze505050/AeroDecel",
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/planets")
def get_planets():
    """List available planets with key atmospheric properties."""
    from src.planetary_atm import get_planet_atmosphere
    result = {}
    for name in ["mars", "venus", "titan"]:
        try:
            atm = get_planet_atmosphere(name)
            result[name] = {
                "gravity_ms2":    atm.gravity_ms2,
                "surface_rho":    atm.density(0),
                "surface_T_K":    atm.temperature(0),
                "surface_P_Pa":   atm.pressure(0),
                "sound_speed_ms": atm.speed_of_sound(0),
                "composition":    atm.composition,
            }
        except Exception as e:
            result[name] = {"error": str(e)}
    return result


@app.get("/materials")
def get_materials():
    """List TPS materials with properties."""
    from src.ablation_model  import ABLATIVE_DB
    from src.thermal_model   import MATERIAL_DB
    result = {}
    for name, mat in ABLATIVE_DB.items():
        result[name] = {
            "type":             "ablative",
            "density_kgm3":     mat.density_kgm3,
            "T_limit_K":        mat.T_ablate_K,
            "h_ablation_MJkg":  mat.h_ablation / 1e6,
            "emissivity":       mat.emissivity,
        }
    for name, mat in MATERIAL_DB.items():
        if name not in result:
            result[name] = {
                "type":             "passive",
                "density_kgm3":     mat.density_kgm3,
                "T_limit_K":        mat.max_temperature_K,
                "conductivity_WmK": mat.conductivity_WmK,
                "emissivity":       mat.emissivity,
            }
    return result


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """
    Run a full EDL trajectory simulation.
    Returns velocity/altitude/heat-flux time histories and key metrics.
    """
    t0 = time.perf_counter()

    try:
        from src.planetary_atm       import get_planet_atmosphere
        from src.multifidelity_pinn  import LowFidelityEDL
        from src.ablation_model      import AblationSolver
        from src.experiment_tracker  import get_tracker

        planet = get_planet_atmosphere(req.planet)
        lf     = LowFidelityEDL(planet, req.mass_kg, req.drag_coeff,
                                  req.canopy_area_m2,
                                  gamma_deg=abs(req.entry_fpa_deg))
        t_arr  = np.linspace(0, 800, 300)
        v_arr, h_arr = lf.solve(t_arr, req.entry_velocity_ms, req.entry_alt_m)

        # Heat flux
        rho_arr = np.array([planet.density(max(0, float(h))) for h in h_arr])
        q_arr_sg = (1.74e-4 * np.sqrt(np.maximum(rho_arr, 0) / req.nose_radius_m)
                    * np.maximum(v_arr, 0)**3)

        # Real-gas correction if requested
        if req.use_realgas:
            try:
                from src.realgas_chemistry import realgas_trajectory_profile
                rg = realgas_trajectory_profile(v_arr, h_arr, planet,
                                                 R_nose=req.nose_radius_m,
                                                 planet_name=req.planet)
                q_final = rg["q_rg_Wm2"]
                gamma_eff = rg["gamma_eff"].tolist()
            except Exception:
                q_final = q_arr_sg; gamma_eff = None
        else:
            q_final = q_arr_sg; gamma_eff = None

        # Ablation check
        solver = AblationSolver(req.tps_material, req.tps_thickness_m, n_nodes=6)
        t_entry = float(t_arr[np.argmax(h_arr <= 0)]) if (h_arr <= 0).any() else float(t_arr[-1])
        t_abl   = np.linspace(0, t_entry, 40)
        q_pk    = float(q_final.max())
        q_abl   = q_pk * np.where(t_abl <= t_entry*0.3,
                                   t_abl/(t_entry*0.3+1e-9),
                                   (t_entry-t_abl)/(t_entry*0.7+1e-9))
        q_abl   = np.clip(q_abl, 0, None)
        abl_res = solver.solve(q_abl, t_abl, verbose=False)

        # Subsample output
        n = req.n_output_points
        idx_out = np.round(np.linspace(0, len(t_arr)-1, n)).astype(int)

        elapsed = time.perf_counter() - t0

        result = {
            "status":        "success",
            "elapsed_s":     round(elapsed, 3),
            "planet":        req.planet,
            "trajectory": {
                "t_s":      t_arr[idx_out].tolist(),
                "v_ms":     v_arr[idx_out].tolist(),
                "h_m":      h_arr[idx_out].tolist(),
                "q_Wm2":    q_final[idx_out].tolist(),
                "rho_kgm3": rho_arr[idx_out].tolist(),
            },
            "metrics": {
                "v_land_ms":        round(float(v_arr[-1]), 4),
                "h_final_m":        round(float(h_arr[-1]), 1),
                "q_peak_MWm2":      round(float(q_final.max())/1e6, 6),
                "tps_recession_mm": round(abl_res["total_recession_mm"], 4),
                "tps_sf":           round((req.tps_thickness_m*1000)/max(abl_res["total_recession_mm"],0.01), 3),
                "tps_survived":     bool(abl_res["total_recession_mm"] < req.tps_thickness_m*1000),
                "heat_blocked_pct": round(abl_res["blocking_pct"], 2),
                "peak_T_K":         round(float(abl_res["peak_T_K"]), 1),
            },
        }

        if gamma_eff:
            result["realgas"] = {
                "gamma_eff_mean": round(float(np.mean(gamma_eff)), 5),
                "gamma_eff_min":  round(float(np.min(gamma_eff)), 5),
                "co2_diss_max":   round(float(np.max([0])), 3),
            }

        # Log to experiment tracker
        try:
            tracker = get_tracker()
            tracker.log_run(
                params  = req.model_dump(),
                results = result["metrics"],
                tags    = [req.planet, req.tps_material, "api"],
                duration_s = elapsed,
            )
        except Exception:
            pass

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ablation")
def ablation(req: AblationRequest):
    """Run TPS ablation-only analysis."""
    try:
        from src.ablation_model import AblationSolver

        solver = AblationSolver(req.material, req.thickness_m, n_nodes=10)
        t      = np.linspace(0, req.t_entry_s, 80)
        t_pk   = req.t_entry_s * 0.30
        q0     = req.q_peak_MW * 1e6 * np.where(
            t <= t_pk, t/t_pk, (req.t_entry_s-t)/(req.t_entry_s-t_pk+1e-9))
        q0     = np.clip(q0, 0, None)
        res    = solver.solve(q0, t, verbose=False)
        summ   = solver.summary(res)

        return {
            "status":   "success",
            "summary":  {k: (float(v) if hasattr(v,'item') else v) for k,v in summ.items()},
            "timeseries": {
                "t_s":          t.tolist(),
                "q_incident":   res["q_incident_Wm2"].tolist(),
                "q_wall":       res["q_wall_Wm2"].tolist(),
                "T_surface_K":  res["T_surface_K"].tolist(),
                "recession_mm": res["recession_mm"].tolist(),
                "B_prime":      res["B_prime"].tolist(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monte_carlo")
def monte_carlo(req: MCRequest):
    """Run Monte Carlo uncertainty propagation."""
    try:
        from src.planetary_atm    import get_planet_atmosphere
        from src.monte_carlo_edl  import MonteCarloEDL

        planet = get_planet_atmosphere(req.planet)
        mc     = MonteCarloEDL(req.n_samples, use_realgas=req.use_realgas, seed=0)
        df     = mc.run(planet, "nylon", 0.015, "elliptical", {"a":10,"b":5},
                        req.mass_kg, 125_000, 5800, 15.0, 4.5, verbose=False)
        stats  = mc.summary()

        return {
            "status": "success",
            "n_valid": len(df),
            "statistics": {k: (v if not isinstance(v, dict) or k != "sensitivity_to_sf" else {})
                           for k, v in stats.items() if k != "v_cdf"},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fault_tree")
def fault_tree(req: FTARequest):
    """Run Fault Tree Analysis."""
    try:
        from src.fault_tree import build_edl_fault_tree, FaultTreeAnalysis
        top, events = build_edl_fault_tree(sf_tps=req.sf_tps, sf_structure=req.sf_structure)
        fta    = FaultTreeAnalysis(top, events)
        report = fta.full_report(n_mc=req.n_mc)

        # Remove numpy arrays for JSON serialisation
        report.pop("mc_result", None)
        return {"status": "success", **{k: v for k, v in report.items()
                                         if not isinstance(v, np.ndarray)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments")
def get_experiments(where: str = "1=1", limit: int = 50):
    """Query logged experiment runs."""
    try:
        from src.experiment_tracker import get_tracker
        df = get_tracker().query(where, limit)
        return {"status": "success", "n_runs": len(df), "runs": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
        print("AeroDecel v6.0 REST API → http://localhost:8000")
        print("Docs: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except ImportError:
        print("Install: pip install uvicorn fastapi")
