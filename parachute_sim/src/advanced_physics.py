"""
advanced_physics.py — Advanced Physics Upgrades
================================================
1. Dryden MIL-SPEC turbulence model  (coloured-noise wind gusts)
2. Reynolds-number dependent Cd correction  (Cd = f(Re))
3. Fabric porosity drag model
4. Buoyancy correction at high altitude
"""
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density, dynamic_viscosity, reynolds_number


# ── 1. Dryden Turbulence (MIL-HDBK-1797B) ────────────────────────────────────

class DrydenTurbulence:
    """
    MIL-SPEC Dryden coloured-noise wind gust model.
    Generates correlated turbulence velocity components (u,v,w) at each timestep.

    Turbulence intensity σ and scale length L vary with altitude per MIL-HDBK-1797B.
    """

    def __init__(self, intensity: str = "light", seed: int = 42):
        """
        intensity: 'light' | 'moderate' | 'severe'
        """
        self._rng   = np.random.default_rng(seed)
        self._scale = {"light": 0.5, "moderate": 1.5, "severe": 4.0}[intensity]
        self._state = np.zeros(6)   # filter states [u1,u2,v1,v2,w1,w2]

    def _scale_length(self, h: float) -> tuple[float, float, float]:
        """Altitude-dependent scale lengths Lu, Lv, Lw [m]."""
        h = max(10.0, h)
        Lu = min(762.0, h / 0.177 ** 0.884 * (h / 1000) ** 0.1)
        Lv = Lu / 2.0
        Lw = min(762.0, h)
        return Lu, Lv, Lw

    def _sigma(self, h: float) -> tuple[float, float, float]:
        """Turbulence intensity σ [m/s] at altitude h."""
        h   = max(10.0, h)
        # MIL-HDBK-1797B Fig 3.7-1 parameterisation
        base = self._scale * max(0.1, 1.0 - h / 15000.0)
        return base, base * 0.75, base * 0.5

    def sample(self, dt: float, h: float, v: float) -> tuple[float, float, float]:
        """
        Generate one turbulence sample (ug, vg, wg) [m/s].
        Call once per ODE timestep.
        dt: timestep [s], h: altitude [m], v: airspeed [m/s]
        """
        Lu, Lv, Lw = self._scale_length(h)
        su, sv, sw  = self._sigma(h)
        V = max(v, 1.0)

        def _filter(L, sig, s1, s2, noise):
            """2nd-order Dryden shaping filter step (Euler)."""
            a1 = V / L;  a2 = (V / L)**2
            ds1 = s2
            ds2 = -2*a1*s2 - a2*s1 + sig * np.sqrt(2*a2/L) * noise
            return s1 + dt*ds1, s2 + dt*ds2

        n = self._rng.standard_normal(3)
        self._state[0], self._state[1] = _filter(Lu, su, self._state[0], self._state[1], n[0])
        self._state[2], self._state[3] = _filter(Lv, sv, self._state[2], self._state[3], n[1])
        self._state[4], self._state[5] = _filter(Lw, sw, self._state[4], self._state[5], n[2])

        return self._state[0], self._state[2], self._state[4]


# ── 2. Reynolds-dependent Cd correction ──────────────────────────────────────

def cd_reynolds_correction(Cd_ref: float, Re: float, canopy_type: str = "flat_circular") -> float:
    """
    Correct steady-state Cd for Reynolds number effects.

    Flat circular canopies show a characteristic Cd dip around Re~3×10⁵
    (similar to sphere drag crisis) followed by recovery.

    Based on: Knacke (1992) Fig 5-21, Lingard (1986) wind-tunnel data.

    Re typical range for parachutes:
      - Small drogue at 50 m/s:  Re ≈ 2×10⁵  (near drag crisis)
      - Main canopy at 5 m/s:    Re ≈ 1×10⁶  (fully turbulent, Cd stable)
    """
    if Re <= 0:
        return Cd_ref

    log_Re = np.log10(max(Re, 1e3))

    corrections = {
        "flat_circular": _cd_flat_circular,
        "ribbon":        _cd_ribbon,
        "ram_air":       _cd_ram_air,
        "drogue":        _cd_drogue,
    }
    fn = corrections.get(canopy_type, _cd_flat_circular)
    factor = fn(log_Re)
    return float(Cd_ref * factor)


def _cd_flat_circular(log_Re: float) -> float:
    # Knacke 1992: flat circular canopy drag ratio vs Re
    # Parameterised as piecewise cubic through key data points
    if log_Re < 4.5:   return 1.05
    elif log_Re < 5.0: return 1.0 + 0.05 * (5.0 - log_Re) / 0.5
    elif log_Re < 5.5: return 1.0 - 0.08 * (log_Re - 5.0) / 0.5   # drag dip
    elif log_Re < 6.0: return 0.92 + 0.06 * (log_Re - 5.5) / 0.5  # recovery
    else:              return 0.98


def _cd_ribbon(log_Re: float) -> float:
    # Ribbon canopies: more stable Cd (perforations prevent flow separation crisis)
    if log_Re < 5.5: return 1.02
    elif log_Re < 6.5: return 1.0 - 0.03*(log_Re - 5.5)
    else: return 0.97


def _cd_ram_air(log_Re: float) -> float:
    # Ram-air: aspect ratio and cell geometry dominate over Re
    return max(0.90, 1.0 - 0.04*(log_Re - 5.0))


def _cd_drogue(log_Re: float) -> float:
    # Small drogue: significant Re effect at high speeds
    if log_Re < 4.8: return 1.10
    elif log_Re < 5.3: return 1.10 - 0.20*(log_Re-4.8)/0.5
    else: return 0.90 + 0.08*(log_Re-5.3)/0.5


def compute_reynolds(v: float, h: float, D_canopy: float) -> float:
    """Compute canopy Reynolds number."""
    return reynolds_number(v, D_canopy, h)


# ── 3. Porosity drag correction ───────────────────────────────────────────────

def cd_porosity_correction(Cd_ref: float, v: float, k_p: float = 0.015) -> float:
    """
    Fabric porosity reduces effective Cd at higher speeds.
    Cd_eff = Cd_ref * max(0.1, 1 - k_p * v)
    k_p ≈ 0.010–0.020 for standard nylon ripstop (Pflanz 1952).
    """
    return float(Cd_ref * max(0.05, 1.0 - k_p * max(0.0, v)))


# ── 4. Buoyancy correction ────────────────────────────────────────────────────

def buoyancy_correction(mass_kg: float, altitude_m: float,
                        canopy_volume_m3: float = 0.5) -> float:
    """
    Archimedes buoyancy force [N] on inflated canopy at altitude.
    Negligible near sea level but non-trivial at >5 km.
    F_buoy = ρ_air * g * V_canopy (upward, opposing gravity)
    """
    rho = density(max(0.0, altitude_m))
    return float(rho * cfg.GRAVITY * canopy_volume_m3)


# ── 5. Full augmented ODE (Dryden + Re-Cd + porosity + buoyancy) ─────────────

class AugmentedParachuteODE:
    """
    Extended ODE incorporating:
      - Dryden coloured turbulence
      - Reynolds-number Cd correction
      - Fabric porosity
      - Buoyancy
    """

    def __init__(
        self,
        At_fn,
        Cd_base:       float = None,
        mass:          float = None,
        canopy_type:   str   = "flat_circular",
        D_canopy:      float = 8.0,
        k_porosity:    float = 0.012,
        canopy_volume: float = 0.4,
        turbulence:    str   = "light",   # 'none' | 'light' | 'moderate' | 'severe'
        seed:          int   = 42,
    ):
        self.At_fn        = At_fn
        self.Cd_base      = Cd_base or cfg.CD_INITIAL
        self.mass         = mass or cfg.PARACHUTE_MASS
        self.canopy_type  = canopy_type
        self.D_canopy     = D_canopy
        self.k_p          = k_porosity
        self.V_canopy     = canopy_volume

        self._use_turb = (turbulence != "none")
        self._turb = DrydenTurbulence(turbulence, seed=seed) if self._use_turb else None
        self._last_dt = 0.05
        self._diag: list[dict] = []

    def rhs(self, t: float, state: list) -> list:
        v, h = state
        v = max(0.0, v)
        h = max(0.0, h)

        A   = max(0.0, self.At_fn(t))
        rho = density(h)
        mu  = dynamic_viscosity(h)

        # Reynolds-number corrected Cd
        Re  = rho * v * self.D_canopy / max(mu, 1e-10)
        Cd  = cd_reynolds_correction(self.Cd_base, Re, self.canopy_type)

        # Porosity correction
        Cd  = cd_porosity_correction(Cd, v, self.k_p)

        # Turbulence: adds random gust to effective airspeed
        v_eff = v
        if self._use_turb and v > 0.5:
            ug, vg, wg = self._turb.sample(self._last_dt, h, v)
            v_eff = max(0.0, v + wg)   # vertical gust modifies descent speed

        drag = 0.5 * rho * v_eff**2 * Cd * A

        # Buoyancy (upward)
        F_buoy = buoyancy_correction(self.mass, h, self.V_canopy)

        # Net acceleration
        dv_dt = cfg.GRAVITY - (drag - F_buoy) / self.mass
        dh_dt = -v

        # Diagnostics
        self._diag.append({
            "t": t, "Re": Re, "Cd_eff": Cd, "drag": drag,
            "F_buoy": F_buoy, "v_eff": v_eff
        })

        return [dv_dt, dh_dt]

    def diagnostics(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(self._diag)
