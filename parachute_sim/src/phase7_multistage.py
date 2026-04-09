"""
phase7_multistage.py — Multi-Stage Drogue → Main Canopy Deployment Model
=========================================================================
Models the complete two-canopy deployment sequence with a full state machine.

Deployment sequence
-------------------
  FREEFALL       → payload separates, accelerating under gravity only
  DROGUE_INFLATE → small pilot/drogue chute opens (fast: ~0.5–1.5 s)
  DROGUE_STABLE  → drogue fully open, controlled descent at ~30–60 m/s
  MAIN_INFLATE   → main canopy triggered (altitude or time), slow inflation
  MAIN_STABLE    → main fully open, terminal descent ~5–8 m/s
  LANDED         → h <= 0

State vector: [v, h]  (same as Phase 2, extended with stage tracking)

Governing equations per stage
------------------------------
  m * dv/dt = m*g - D_drogue(t) - D_main(t)

where each drag term is:
  D = 0.5 * ρ(h) * v² * Cd * A(t)

and A(t) per canopy follows the generalised logistic inflation model,
gated by the state machine (zero outside its active window).

Opening shock / snatch load
---------------------------
  F_snatch = 0.5 * ρ * v_deploy² * Cd * A_inf * CLA

  CLA (Canopy Load Alleviation) modelled as:
    CLA = 1 + k_dyn * (t_infl / v_deploy)^{-0.5}   [simplified MIL-HDBK-1791]

Outputs
-------
  - Full time-series CSV: v, h, stage, A_drogue, A_main, drag, rho
  - Opening shock events (timestamp, velocity, peak force, safety factor)
  - Comparison plot: drogue-only vs. 2-stage vs. no-chute (ballistic)
  - Phase portrait coloured by stage
"""

from __future__ import annotations
import sys
import time
from enum import IntEnum
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEPLOYMENT STAGE ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class Stage(IntEnum):
    FREEFALL       = 0
    DROGUE_INFLATE = 1
    DROGUE_STABLE  = 2
    MAIN_INFLATE   = 3
    MAIN_STABLE    = 4
    LANDED         = 5

STAGE_LABELS = {
    Stage.FREEFALL       : "Free fall",
    Stage.DROGUE_INFLATE : "Drogue inflating",
    Stage.DROGUE_STABLE  : "Drogue stable",
    Stage.MAIN_INFLATE   : "Main inflating",
    Stage.MAIN_STABLE    : "Main stable",
    Stage.LANDED         : "Landed",
}

STAGE_COLORS = {
    Stage.FREEFALL       : "#ff4560",
    Stage.DROGUE_INFLATE : "#ffb01a",
    Stage.DROGUE_STABLE  : "#ffd700",
    Stage.MAIN_INFLATE   : "#00d4ff",
    Stage.MAIN_STABLE    : "#a8ff3e",
    Stage.LANDED         : "#888888",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CANOPY CONFIGURATION DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanopyConfig:
    """Physical specification for one canopy stage."""
    name:          str   = "unnamed"
    area_m2:       float = 1.0        # fully open reference area [m²]
    Cd:            float = 1.5        # steady-state drag coefficient
    infl_time_s:   float = 0.8        # time from trigger to 95% open [s]
    infl_n:        float = 2.0        # logistic shape exponent (higher → sharper)
    rated_load_N:  float = 50000.0    # suspension line rated tensile load [N]
    porosity_k:    float = 0.0        # fabric porosity coefficient (0 = impermeable)

    # Trigger conditions (first satisfied wins)
    trigger_alt_m:  float | None = None   # deploy when h <= this value [m AGL]
    trigger_time_s: float | None = None   # deploy after this elapsed time [s]
    trigger_vel_ms: float | None = None   # deploy when v <= this value [m/s]


@dataclass
class MultiStageConfig:
    """Complete two-stage system configuration."""
    mass_kg:     float = None   # total payload + both canopies

    # Drogue canopy
    drogue: CanopyConfig = field(default_factory=lambda: CanopyConfig(
        name        = "drogue",
        area_m2     = 2.5,
        Cd          = 0.97,
        infl_time_s = 0.6,
        infl_n      = 3.0,
        rated_load_N= 15000.0,
        trigger_alt_m = None,     # triggered at t=0 (ejection)
        trigger_time_s= 0.0,
    ))

    # Main canopy
    main: CanopyConfig = field(default_factory=lambda: CanopyConfig(
        name        = "main",
        area_m2     = 50.0,
        Cd          = 1.35,
        infl_time_s = 2.5,
        infl_n      = 2.0,
        rated_load_N= 80000.0,
        trigger_alt_m = 300.0,    # open main at 300 m AGL
    ))

    # System
    initial_alt_m:  float = None
    initial_vel_ms: float = None
    freefall_Cd:    float = 0.05   # payload body Cd during free fall (low)
    freefall_area_m2: float = 0.3  # payload reference area [m²]

    def __post_init__(self):
        self.mass_kg         = self.mass_kg         or cfg.PARACHUTE_MASS
        self.initial_alt_m   = self.initial_alt_m   or cfg.INITIAL_ALT
        self.initial_vel_ms  = self.initial_vel_ms  or cfg.INITIAL_VEL


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INFLATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _generalised_logistic(t_since_trigger: float, A_inf: float,
                           t_infl: float, n: float) -> float:
    """
    Generalised logistic (Richards) canopy inflation model.
    A(Δt) ramps from 0 → A_inf over ~t_infl seconds.
    """
    if t_since_trigger < 0:
        return 0.0
    k  = 8.0 / max(t_infl, 0.05)
    t0 = t_infl * 0.55
    raw = A_inf / (1.0 + np.exp(-k * (t_since_trigger - t0))) ** (1.0 / n)
    # Clamp to [0, A_inf]
    return float(np.clip(raw, 0.0, A_inf))


def _effective_Cd(Cd_base: float, v: float, porosity_k: float) -> float:
    """Effective Cd accounting for fabric porosity: Cd_eff = Cd * (1 - k_p*v)."""
    return max(0.1, Cd_base * max(0.0, 1.0 - porosity_k * v))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OPENING SHOCK CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SnatchEvent:
    """Record of one opening shock event."""
    stage:         str
    time_s:        float
    altitude_m:    float
    velocity_ms:   float
    A_inf_m2:      float
    Cd:            float
    rho:           float
    t_infl_s:      float
    rated_load_N:  float

    def peak_force_N(self) -> float:
        """
        Simplified MIL-HDBK-1791 opening load:
          F_open = 0.5 * ρ * v² * Cd * A_inf * CLA
        CLA (Canopy Load Alleviation factor):
          CLA = 1 + 0.38 * (v / sqrt(A_inf))^0.5 / t_infl^0.5
        """
        CLA = 1.0 + 0.38 * (self.velocity_ms / max(np.sqrt(self.A_inf_m2), 0.1)) ** 0.5 \
              / max(self.t_infl_s ** 0.5, 0.1)
        return 0.5 * self.rho * self.velocity_ms**2 * self.Cd * self.A_inf_m2 * CLA

    def safety_factor(self) -> float:
        """SF = rated_load / peak_force. Target SF > 1.5 for parachute systems."""
        F = self.peak_force_N()
        return self.rated_load_N / max(F, 1.0)

    def to_dict(self) -> dict:
        F  = self.peak_force_N()
        SF = self.safety_factor()
        return {
            "stage":         self.stage,
            "time_s":        round(self.time_s, 3),
            "altitude_m":    round(self.altitude_m, 1),
            "velocity_ms":   round(self.velocity_ms, 3),
            "peak_force_N":  round(F, 1),
            "rated_load_N":  self.rated_load_N,
            "safety_factor": round(SF, 3),
            "CLA":           round(F / max(0.5 * self.rho * self.velocity_ms**2
                                           * self.Cd * self.A_inf_m2, 1e-3), 4),
            "status":        "OK" if SF >= 1.5 else "WARNING" if SF >= 1.0 else "CRITICAL",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-STAGE ODE SYSTEM (event-driven)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiStageODE:
    """
    Solves the two-stage descent ODE using sequential solve_ivp calls,
    one per stage. Transitions are exact (not approximate) because each
    stage boundary is an event root.

    The simulation runs through stages in order. For each stage,
    solve_ivp integrates until either the stage-exit event fires or
    the ground is reached.
    """

    def __init__(self, sys_cfg: MultiStageConfig):
        self.cfg          = sys_cfg
        self.records: list[dict]       = []
        self.snatch_events: list[SnatchEvent] = []
        self._t_offset    = 0.0
        self._t_drogue    = None
        self._t_main      = None

    # ── Drag force for a given canopy at current time-in-stage ────────────────
    def _drag(self, canopy: CanopyConfig, t_since_trigger: float,
              v: float, h: float) -> tuple[float, float]:
        """Returns (drag_force_N, area_m2)."""
        A   = _generalised_logistic(t_since_trigger, canopy.area_m2,
                                    canopy.infl_time_s, canopy.infl_n)
        Cd  = _effective_Cd(canopy.Cd, v, canopy.porosity_k)
        rho = density(max(0.0, h))
        D   = 0.5 * rho * max(v, 0.0)**2 * Cd * A
        return D, A

    # ── ODE RHS builders ──────────────────────────────────────────────────────
    def _rhs_freefall(self, t: float, state: list) -> list:
        v, h = state
        v = max(v, 0.0)
        rho  = density(max(0.0, h))
        D    = 0.5 * rho * v**2 * self.cfg.freefall_Cd * self.cfg.freefall_area_m2
        return [cfg.GRAVITY - D / self.cfg.mass_kg, -v]

    def _rhs_drogue(self, t_abs: float, state: list) -> list:
        v, h = state
        v = max(v, 0.0)
        t_rel = t_abs - self._t_drogue
        D, _  = self._drag(self.cfg.drogue, t_rel, v, h)
        return [cfg.GRAVITY - D / self.cfg.mass_kg, -v]

    def _rhs_main(self, t_abs: float, state: list) -> list:
        v, h = state
        v = max(v, 0.0)
        t_rel_d = t_abs - self._t_drogue
        t_rel_m = t_abs - self._t_main
        D_d, _  = self._drag(self.cfg.drogue, t_rel_d, v, h)   # drogue still open
        D_m, _  = self._drag(self.cfg.main,   t_rel_m, v, h)
        return [cfg.GRAVITY - (D_d + D_m) / self.cfg.mass_kg, -v]

    # ── Record callback ───────────────────────────────────────────────────────
    def _record_solution(self, sol, stage: Stage):
        t_arr = sol.t
        v_arr = np.clip(sol.y[0], 0, None)
        h_arr = np.clip(sol.y[1], 0, None)

        for i, t in enumerate(t_arr):
            v = v_arr[i]; h = h_arr[i]
            rho = density(max(0.0, h))

            A_d = 0.0; D_d = 0.0
            A_m = 0.0; D_m = 0.0

            if self._t_drogue is not None and t >= self._t_drogue:
                rel_d = t - self._t_drogue
                D_d, A_d = self._drag(self.cfg.drogue, rel_d, v, h)
            if self._t_main is not None and t >= self._t_main:
                rel_m = t - self._t_main
                D_m, A_m = self._drag(self.cfg.main, rel_m, v, h)

            # Free-fall body drag
            D_body = (0.5 * rho * v**2 * self.cfg.freefall_Cd
                      * self.cfg.freefall_area_m2)

            self.records.append({
                "time_s"       : round(t, 4),
                "velocity_ms"  : round(v, 4),
                "altitude_m"   : round(h, 2),
                "stage"        : int(stage),
                "stage_label"  : STAGE_LABELS[stage],
                "area_drogue_m2": round(A_d, 4),
                "area_main_m2" : round(A_m, 4),
                "drag_drogue_N": round(D_d, 2),
                "drag_main_N"  : round(D_m, 2),
                "drag_body_N"  : round(D_body, 2),
                "drag_total_N" : round(D_d + D_m + D_body, 2),
                "rho_kgm3"     : round(rho, 5),
                "dyn_press_Pa" : round(0.5 * rho * v**2, 2),
            })

    # ── Run the full multi-stage simulation ───────────────────────────────────
    def run(self, verbose: bool = True) -> pd.DataFrame:
        sc = self.cfg
        t  = 0.0
        y0 = [sc.initial_vel_ms, sc.initial_alt_m]

        def ground_event(t, y): return y[1]
        ground_event.terminal  = True
        ground_event.direction = -1

        t_max = sc.initial_alt_m / max(sc.initial_vel_ms, 1.0) * 6 + 300

        if verbose:
            print(f"\n[Phase 7] Multi-Stage Descent Simulation")
            print(f"  Drogue: A={sc.drogue.area_m2}m²  Cd={sc.drogue.Cd}  "
                  f"t_infl={sc.drogue.infl_time_s}s  "
                  f"trigger={'t=0' if sc.drogue.trigger_time_s==0 else f'alt={sc.drogue.trigger_alt_m}m'}")
            print(f"  Main:   A={sc.main.area_m2}m²  Cd={sc.main.Cd}  "
                  f"t_infl={sc.main.infl_time_s}s  "
                  f"trigger=alt<{sc.main.trigger_alt_m}m")
            print(f"  Mass: {sc.mass_kg}kg  Alt0: {sc.initial_alt_m}m  V0: {sc.initial_vel_ms}m/s")

        # ── Stage 0: Free fall (if drogue doesn't trigger immediately) ─────────
        drogue_trigger_t = sc.drogue.trigger_time_s
        if drogue_trigger_t is not None and drogue_trigger_t > 0:
            def drogue_time_event(t, y): return t - drogue_trigger_t
            drogue_time_event.terminal  = True
            drogue_time_event.direction = 1

            sol = solve_ivp(self._rhs_freefall, (t, t + drogue_trigger_t + 0.01), y0,
                            method="RK45", dense_output=True, max_step=0.1,
                            events=[ground_event, drogue_time_event],
                            rtol=1e-6, atol=1e-8)
            self._record_solution(sol, Stage.FREEFALL)
            t  = sol.t[-1]
            y0 = [sol.y[0, -1], sol.y[1, -1]]
            if y0[1] <= 0:
                if verbose: print("  Landed during free fall")
                return self._finalise(verbose)

        # ── Stage 1: Drogue inflation ──────────────────────────────────────────
        self._t_drogue = t
        rho_d   = density(max(0.0, float(y0[1])))
        snatch_d = SnatchEvent("drogue", t, float(y0[1]), float(y0[0]),
                               sc.drogue.area_m2, sc.drogue.Cd, rho_d,
                               sc.drogue.infl_time_s, sc.drogue.rated_load_N)
        self.snatch_events.append(snatch_d)

        if verbose:
            sd = snatch_d.to_dict()
            print(f"\n  Drogue fires @ t={t:.2f}s  h={y0[1]:.0f}m  v={y0[0]:.2f}m/s")
            print(f"  Opening shock: {sd['peak_force_N']:.0f}N  SF={sd['safety_factor']:.2f}  [{sd['status']}]")

        t_infl_end = t + sc.drogue.infl_time_s * 1.8

        def drogue_open(t_, y):
            rel = t_ - self._t_drogue
            A = _generalised_logistic(rel, sc.drogue.area_m2, sc.drogue.infl_time_s, sc.drogue.infl_n)
            return A - 0.97 * sc.drogue.area_m2

        drogue_open.terminal  = True
        drogue_open.direction = 1

        sol = solve_ivp(self._rhs_drogue, (t, t_infl_end + 5), y0,
                        method="RK45", dense_output=True, max_step=0.05,
                        events=[ground_event, drogue_open],
                        rtol=1e-6, atol=1e-8)
        self._record_solution(sol, Stage.DROGUE_INFLATE)
        t  = sol.t[-1]; y0 = [sol.y[0, -1], sol.y[1, -1]]
        if y0[1] <= 0: return self._finalise(verbose)

        # ── Stage 2: Drogue stable — wait for main trigger altitude ───────────
        main_alt = sc.main.trigger_alt_m or 300.0
        main_t   = sc.main.trigger_time_s

        def main_alt_event(t_, y):   return y[1] - main_alt
        def main_time_event(t_, y):  return t_ - (self._t_drogue + (main_t or 9e9))
        main_alt_event.terminal  = True;  main_alt_event.direction  = -1
        main_time_event.terminal = True;  main_time_event.direction = 1

        events = [ground_event, main_alt_event]
        if main_t is not None:
            events.append(main_time_event)

        sol = solve_ivp(self._rhs_drogue, (t, t_max), y0,
                        method="RK45", dense_output=True, max_step=0.1,
                        events=events, rtol=1e-6, atol=1e-8)
        self._record_solution(sol, Stage.DROGUE_STABLE)
        t  = sol.t[-1]; y0 = [sol.y[0, -1], sol.y[1, -1]]
        if y0[1] <= 0: return self._finalise(verbose)

        # ── Stage 3: Main inflation ────────────────────────────────────────────
        self._t_main = t
        rho_m  = density(max(0.0, float(y0[1])))
        snatch_m = SnatchEvent("main", t, float(y0[1]), float(y0[0]),
                               sc.main.area_m2, sc.main.Cd, rho_m,
                               sc.main.infl_time_s, sc.main.rated_load_N)
        self.snatch_events.append(snatch_m)

        if verbose:
            sm = snatch_m.to_dict()
            print(f"\n  Main fires  @ t={t:.2f}s  h={y0[1]:.0f}m  v={y0[0]:.2f}m/s")
            print(f"  Opening shock: {sm['peak_force_N']:.0f}N  SF={sm['safety_factor']:.2f}  [{sm['status']}]")

        def main_open(t_, y):
            rel = t_ - self._t_main
            A = _generalised_logistic(rel, sc.main.area_m2, sc.main.infl_time_s, sc.main.infl_n)
            return A - 0.97 * sc.main.area_m2

        main_open.terminal  = True
        main_open.direction = 1

        sol = solve_ivp(self._rhs_main, (t, t + sc.main.infl_time_s * 3), y0,
                        method="RK45", dense_output=True, max_step=0.05,
                        events=[ground_event, main_open],
                        rtol=1e-6, atol=1e-8)
        self._record_solution(sol, Stage.MAIN_INFLATE)
        t  = sol.t[-1]; y0 = [sol.y[0, -1], sol.y[1, -1]]
        if y0[1] <= 0: return self._finalise(verbose)

        # ── Stage 4: Main stable ───────────────────────────────────────────────
        sol = solve_ivp(self._rhs_main, (t, t_max), y0,
                        method="RK45", dense_output=True, max_step=0.2,
                        events=[ground_event], rtol=1e-6, atol=1e-8)
        self._record_solution(sol, Stage.MAIN_STABLE)

        return self._finalise(verbose)

    def _finalise(self, verbose: bool) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        if not df.empty:
            df = df.drop_duplicates("time_s").sort_values("time_s").reset_index(drop=True)

        if verbose:
            land_v = float(df["velocity_ms"].iloc[-1])
            land_t = float(df["time_s"].iloc[-1])
            land_h = float(df["altitude_m"].iloc[-1])
            peak_D = float(df["drag_total_N"].max())
            print(f"\n  Simulation complete:")
            print(f"  Landing  v={land_v:.3f} m/s  t={land_t:.1f}s  h_final={land_h:.1f}m")
            print(f"  Peak drag={peak_D:.0f} N")
            print(f"\n  Opening shock summary:")
            for ev in self.snatch_events:
                d = ev.to_dict()
                flag = "✓" if d["status"]=="OK" else "⚠" if d["status"]=="WARNING" else "✗"
                print(f"    [{flag}] {d['stage']:8s}  F={d['peak_force_N']:7.0f}N  "
                      f"SF={d['safety_factor']:.2f}  CLA={d['CLA']:.3f}  [{d['status']}]")

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 3D MULTI-STAGE TRAJECTORY (drogue + main + wind)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiStage3D:
    """
    Extends MultiStageODE to 3D by adding horizontal wind coupling.
    State: [vx, vy, vz, x, y, z]
    """

    def __init__(self, sys_cfg: MultiStageConfig, wind_profile):
        self.cfg  = sys_cfg
        self.wind = wind_profile
        self._t_drogue = None
        self._t_main   = None
        self.snatch_events = []
        self.records = []

    def _drag_3d(self, canopy: CanopyConfig, t_since: float, vz: float, h: float):
        A   = _generalised_logistic(t_since, canopy.area_m2, canopy.infl_time_s, canopy.infl_n)
        Cd  = _effective_Cd(canopy.Cd, vz, canopy.porosity_k)
        return A, Cd

    def _rhs(self, t: float, state: list) -> list:
        vx, vy, vz, x, y, z = state
        z  = max(0.0, z)
        vz = max(0.0, vz)

        uw, vw = self.wind(z)
        rho    = density(z)

        # Accumulated drag area
        CdA_total = self.cfg.freefall_Cd * self.cfg.freefall_area_m2

        if self._t_drogue is not None and t >= self._t_drogue:
            A_d, Cd_d = self._drag_3d(self.cfg.drogue, t - self._t_drogue, vz, z)
            CdA_total += Cd_d * A_d

        if self._t_main is not None and t >= self._t_main:
            A_m, Cd_m = self._drag_3d(self.cfg.main, t - self._t_main, vz, z)
            CdA_total += Cd_m * A_m

        vrel_x = vx - uw; vrel_y = vy - vw; vrel_z = vz
        v_rel  = max(1e-9, np.sqrt(vrel_x**2 + vrel_y**2 + vrel_z**2))
        D      = 0.5 * rho * v_rel**2 * CdA_total

        ax = -D * vrel_x / (self.cfg.mass_kg * v_rel)
        ay = -D * vrel_y / (self.cfg.mass_kg * v_rel)
        az =  cfg.GRAVITY - D * vrel_z / (self.cfg.mass_kg * v_rel)

        return [ax, ay, az, vx, vy, -vz]

    def run(self, verbose: bool = True) -> pd.DataFrame:
        sc    = self.cfg
        t     = 0.0
        y0    = [0.0, 0.0, sc.initial_vel_ms, 0.0, 0.0, sc.initial_alt_m]
        t_end = sc.initial_alt_m / max(sc.initial_vel_ms, 1.0) * 8 + 400

        def ground(t_, y): return y[5]
        ground.terminal  = True
        ground.direction = -1

        # Drogue fires at t=0 for simplicity (most common real scenario)
        self._t_drogue = 0.0
        rho0 = density(sc.initial_alt_m)
        self.snatch_events.append(SnatchEvent(
            "drogue", 0.0, sc.initial_alt_m, sc.initial_vel_ms,
            sc.drogue.area_m2, sc.drogue.Cd, rho0,
            sc.drogue.infl_time_s, sc.drogue.rated_load_N,
        ))

        # --- Drogue phase ---
        def main_trigger(t_, y): return y[5] - (sc.main.trigger_alt_m or 300.0)
        main_trigger.terminal  = True
        main_trigger.direction = -1

        sol1 = solve_ivp(self._rhs, (0, t_end), y0,
                         method="RK45", max_step=0.2, dense_output=True,
                         events=[ground, main_trigger], rtol=1e-5, atol=1e-7)
        self._append_records(sol1, Stage.DROGUE_INFLATE)
        t  = sol1.t[-1]; y0 = sol1.y[:, -1].tolist()

        if y0[5] > 0:
            # --- Main fires ---
            self._t_main = t
            rho_m = density(max(0.0, y0[5]))
            self.snatch_events.append(SnatchEvent(
                "main", t, y0[5], y0[2],
                sc.main.area_m2, sc.main.Cd, rho_m,
                sc.main.infl_time_s, sc.main.rated_load_N,
            ))

            sol2 = solve_ivp(self._rhs, (t, t_end), y0,
                             method="RK45", max_step=0.3, dense_output=True,
                             events=[ground], rtol=1e-5, atol=1e-7)
            stage2 = Stage.MAIN_INFLATE
            self._append_records(sol2, stage2)

        return pd.DataFrame(self.records)

    def _append_records(self, sol, stage: Stage):
        for i, t in enumerate(sol.t):
            vx, vy, vz, x, y, z = sol.y[:, i]
            self.records.append({
                "time_s": round(t, 3),
                "vx": round(vx, 3), "vy": round(vy, 3), "vz": round(max(0, vz), 3),
                "x_east_m": round(x, 2), "y_north_m": round(y, 2),
                "altitude_m": round(max(0, z), 2),
                "speed_ms": round(np.sqrt(vx**2+vy**2+vz**2), 3),
                "drift_m": round(np.sqrt(x**2+y**2), 2),
                "stage": int(stage), "stage_label": STAGE_LABELS[stage],
            })


# ═══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multistage(df: pd.DataFrame, ode: MultiStageODE,
                    sys_cfg: MultiStageConfig, save_path: Path = None):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    if cfg.DARK_THEME:
        matplotlib.rcParams.update({
            "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
            "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
            "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
            "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        })
    matplotlib.rcParams.update({"font.family":"monospace","font.size":9})

    TEXT  = "#c8d8f0" if cfg.DARK_THEME else "#111"
    SPINE = "#2a3d6e" if cfg.DARK_THEME else "#ccc"

    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.06, right=0.97)

    t  = df["time_s"].values
    v  = df["velocity_ms"].values
    h  = df["altitude_m"].values
    st = df["stage"].values
    Ad = df["area_drogue_m2"].values
    Am = df["area_main_m2"].values
    D  = df["drag_total_N"].values
    q  = df["dyn_press_Pa"].values

    stage_vals = sorted(df["stage"].unique())

    def colorise(ax, x, y, alpha=1.0, lw=1.8):
        for sv in stage_vals:
            mask = st == sv
            if mask.any():
                c = STAGE_COLORS.get(Stage(sv), "#888")
                ax.plot(x[mask], y[mask], color=c, lw=lw, alpha=alpha)

    # Panel 0: velocity
    ax0 = fig.add_subplot(gs[0, :2])
    colorise(ax0, t, v)
    ax0.set_title("Velocity v(t) — coloured by stage", fontweight="bold")
    ax0.set_xlabel("Time [s]"); ax0.set_ylabel("Velocity [m/s]")
    ax0.grid(True, alpha=0.3)
    # Mark snatch events
    for ev in ode.snatch_events:
        ax0.axvline(ev.time_s, color="#ffd700", lw=0.9, ls="--", alpha=0.7)
        d = ev.to_dict()
        ax0.text(ev.time_s+0.3, ev.velocity_ms*1.04,
                 f"{ev.stage} fire\nF={d['peak_force_N']:.0f}N\nSF={d['safety_factor']:.2f}",
                 fontsize=7, color="#ffd700")

    # Panel 1: altitude
    ax1 = fig.add_subplot(gs[0, 2:])
    colorise(ax1, t, h)
    ax1.set_title("Altitude h(t)", fontweight="bold")
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Altitude [m AGL]")
    # Mark main trigger altitude
    if sys_cfg.main.trigger_alt_m:
        ax1.axhline(sys_cfg.main.trigger_alt_m, color="#00d4ff", lw=0.9, ls=":",
                    label=f"Main trigger h={sys_cfg.main.trigger_alt_m}m")
        ax1.legend(fontsize=7.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: drogue area
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(t, Ad, alpha=0.2, color="#ffd700")
    ax2.plot(t, Ad, color="#ffd700", lw=1.8, label="A_drogue")
    ax2.axhline(sys_cfg.drogue.area_m2, color="#ffd700", lw=0.7, ls=":", alpha=0.6)
    ax2.set_title("Drogue canopy area", fontweight="bold")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Area [m²]")
    ax2.grid(True, alpha=0.3)

    # Panel 3: main area
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(t, Am, alpha=0.2, color="#00d4ff")
    ax3.plot(t, Am, color="#00d4ff", lw=1.8, label="A_main")
    ax3.axhline(sys_cfg.main.area_m2, color="#00d4ff", lw=0.7, ls=":", alpha=0.6)
    ax3.set_title("Main canopy area", fontweight="bold")
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Area [m²]")
    ax3.grid(True, alpha=0.3)

    # Panel 4: total drag
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.fill_between(t, D, alpha=0.2, color="#ff4560")
    ax4.plot(t, D, color="#ff4560", lw=1.8)
    mg = sys_cfg.mass_kg * cfg.GRAVITY
    ax4.axhline(mg, color=TEXT, lw=0.8, ls=":", alpha=0.5,
                label=f"Weight = {mg:.0f} N")
    ax4.set_title("Total drag F_D(t)", fontweight="bold")
    ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Force [N]")
    ax4.legend(fontsize=7.5); ax4.grid(True, alpha=0.3)

    # Panel 5: dynamic pressure
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.fill_between(t, q, alpha=0.2, color="#a8ff3e")
    ax5.plot(t, q, color="#a8ff3e", lw=1.8)
    ax5.set_title("Dynamic pressure q(t)", fontweight="bold")
    ax5.set_xlabel("Time [s]"); ax5.set_ylabel("q [Pa]")
    ax5.grid(True, alpha=0.3)

    # Panel 6: phase portrait v vs h
    ax6 = fig.add_subplot(gs[2, :2])
    colorise(ax6, h, v, lw=1.5)
    ax6.set_xlabel("Altitude [m]"); ax6.set_ylabel("Velocity [m/s]")
    ax6.set_title("Phase portrait  v vs h", fontweight="bold")
    ax6.grid(True, alpha=0.3)
    for ev in ode.snatch_events:
        d = ev.to_dict()
        ax6.scatter([ev.altitude_m], [ev.velocity_ms], s=60,
                    color="#ffd700", zorder=5, marker="^")

    # Panel 7: snatch load summary table
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis("off")
    headers = ["Stage", "Time [s]", "Alt [m]", "v [m/s]",
               "Peak F [N]", "Rated [N]", "SF", "Status"]
    rows_data = []
    for ev in ode.snatch_events:
        d = ev.to_dict()
        rows_data.append([d["stage"], f"{d['time_s']:.2f}", f"{d['altitude_m']:.0f}",
                          f"{d['velocity_ms']:.2f}", f"{d['peak_force_N']:.0f}",
                          f"{d['rated_load_N']:.0f}", f"{d['safety_factor']:.2f}",
                          d["status"]])

    col_colors = [[SPINE]*len(headers)]
    cell_colors = []
    for row in rows_data:
        status = row[-1]
        rc = "#0a2010" if status=="OK" else "#2e1e00" if status=="WARNING" else "#2e0e0e"
        cell_colors.append([rc]*len(headers))

    if rows_data:
        tbl = ax7.table(
            cellText=rows_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            cellColours=cell_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(SPINE)
            cell.set_text_props(color=TEXT)

    ax7.set_title("Opening shock / snatch load analysis", fontweight="bold", pad=14)

    # Legend
    patches = [mpatches.Patch(color=STAGE_COLORS[Stage(sv)],
               label=STAGE_LABELS[Stage(sv)]) for sv in stage_vals]
    fig.legend(handles=patches, loc="upper right", fontsize=8,
               bbox_to_anchor=(0.98, 0.98), framealpha=0.3)

    fig.text(0.5, 0.955,
             f"Multi-Stage Drogue → Main  |  "
             f"Drogue A={sys_cfg.drogue.area_m2}m² Cd={sys_cfg.drogue.Cd}  |  "
             f"Main A={sys_cfg.main.area_m2}m² Cd={sys_cfg.main.Cd}  trigger={sys_cfg.main.trigger_alt_m}m",
             ha="center", fontsize=11, fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR / "multistage_dashboard.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Multi-stage plot saved: {sp}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    drogue_area:    float = None,
    drogue_Cd:      float = None,
    drogue_infl:    float = None,
    main_area:      float = None,
    main_Cd:        float = None,
    main_infl:      float = None,
    main_alt:       float = None,
    mass:           float = None,
    alt0:           float = None,
    v0:             float = None,
    wind_profile           = None,
    verbose:        bool  = True,
) -> tuple[pd.DataFrame, MultiStageODE]:

    sys_cfg = MultiStageConfig(
        mass_kg       = mass or cfg.PARACHUTE_MASS,
        initial_alt_m = alt0 or cfg.INITIAL_ALT,
        initial_vel_ms= v0   or cfg.INITIAL_VEL,
        drogue = CanopyConfig(
            name        = "drogue",
            area_m2     = drogue_area or 2.5,
            Cd          = drogue_Cd   or 0.97,
            infl_time_s = drogue_infl or 0.6,
            infl_n      = 3.0,
            rated_load_N= 15000.0,
            trigger_time_s = 0.0,
        ),
        main = CanopyConfig(
            name        = "main",
            area_m2     = main_area or cfg.CANOPY_AREA_M2,
            Cd          = main_Cd   or cfg.CD_INITIAL,
            infl_time_s = main_infl or 2.5,
            infl_n      = 2.0,
            rated_load_N= 80000.0,
            trigger_alt_m = main_alt or 300.0,
        ),
    )

    ode = MultiStageODE(sys_cfg)
    df  = ode.run(verbose=verbose)

    out = cfg.OUTPUTS_DIR / "multistage_results.csv"
    df.to_csv(out, index=False)
    if verbose:
        print(f"\n  ✓ Results saved: {out}")

    import matplotlib; matplotlib.use("Agg")
    plot_multistage(df, ode, sys_cfg)

    # Snatch events JSON
    snatch_out = cfg.OUTPUTS_DIR / "snatch_events.json"
    import json
    snatch_out.write_text(json.dumps([e.to_dict() for e in ode.snatch_events], indent=2))
    if verbose:
        print(f"  ✓ Snatch events: {snatch_out}")

    return df, ode


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multi-Stage Drogue → Main Simulation")
    p.add_argument("--drogue-area", type=float, default=2.5)
    p.add_argument("--drogue-Cd",   type=float, default=0.97)
    p.add_argument("--drogue-infl", type=float, default=0.6)
    p.add_argument("--main-area",   type=float, default=50.0)
    p.add_argument("--main-Cd",     type=float, default=1.35)
    p.add_argument("--main-infl",   type=float, default=2.5)
    p.add_argument("--main-alt",    type=float, default=300.0)
    p.add_argument("--mass",        type=float, default=None)
    p.add_argument("--alt0",        type=float, default=None)
    p.add_argument("--v0",          type=float, default=None)
    a = p.parse_args()
    run(drogue_area=a.drogue_area, drogue_Cd=a.drogue_Cd, drogue_infl=a.drogue_infl,
        main_area=a.main_area, main_Cd=a.main_Cd, main_infl=a.main_infl,
        main_alt=a.main_alt, mass=a.mass, alt0=a.alt0, v0=a.v0)
