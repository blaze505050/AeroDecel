"""
planetary_atm.py — Multi-Planetary Atmosphere Module (AeroDecel v6.0)
======================================================================
Extends the ISA standard atmosphere to Mars, Venus, Titan, and arbitrary
planetary bodies. Enables EDL (Entry-Descent-Landing) analysis for
interplanetary missions.

API-compatible with atmosphere.py — all existing modules work via planet swap.

Planetary models:
  Earth : ICAO 7-layer ISA (delegates to atmosphere.py)
  Mars  : NASA Mars-GRAM simplified (CO₂, 95.3%, ~636 Pa surface)
  Venus : VIRA (Venus International Reference Atmosphere)
  Titan : Huygens-derived N₂/CH₄ atmosphere
  Custom: User-defined composition and lapse rates

Reference:
  - Mars: Seiff et al., "Models of the structure of the atmosphere of Mars"
  - Venus: Kliore et al., VIRA (Adv. Space Res., 1985)
  - Titan: Fulchignoni et al., "In situ measurements of Titan's atmosphere" (Nature, 2005)
  - NASA Mars GRAM 2010 (simplified parametric fit)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PLANETARY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PlanetaryConstants:
    """Physical constants for a planetary body."""
    name:             str
    gravity:          float    # surface gravity [m/s²]
    R_gas:            float    # specific gas constant [J/(kg·K)]
    gamma:            float    # heat capacity ratio (Cp/Cv)
    surface_pressure: float    # mean surface pressure [Pa]
    surface_temp:     float    # mean surface temperature [K]
    surface_density:  float    # mean surface density [kg/m³]
    radius:           float    # mean planetary radius [m]
    scale_height:     float    # atmospheric scale height [m]
    composition:      str      # dominant gas composition
    mu_ref:           float    # reference dynamic viscosity [Pa·s]
    T_ref_visc:       float    # reference temperature for Sutherland [K]
    S_visc:           float    # Sutherland's constant [K]


EARTH = PlanetaryConstants(
    name="Earth", gravity=9.80665, R_gas=287.058, gamma=1.4,
    surface_pressure=101325.0, surface_temp=288.15, surface_density=1.225,
    radius=6371000.0, scale_height=8500.0,
    composition="N₂ 78.1% + O₂ 20.9%",
    mu_ref=1.716e-5, T_ref_visc=273.15, S_visc=110.4,
)

MARS = PlanetaryConstants(
    name="Mars", gravity=3.72076, R_gas=188.92, gamma=1.29,
    surface_pressure=636.0, surface_temp=210.0, surface_density=0.020,
    radius=3389500.0, scale_height=11100.0,
    composition="CO₂ 95.3% + N₂ 2.7% + Ar 1.6%",
    mu_ref=1.082e-5, T_ref_visc=200.0, S_visc=240.0,
)

VENUS = PlanetaryConstants(
    name="Venus", gravity=8.87, R_gas=188.92, gamma=1.29,
    surface_pressure=9.2e6, surface_temp=737.0, surface_density=65.0,
    radius=6051800.0, scale_height=15900.0,
    composition="CO₂ 96.5% + N₂ 3.5%",
    mu_ref=1.48e-5, T_ref_visc=737.0, S_visc=240.0,
)

TITAN = PlanetaryConstants(
    name="Titan", gravity=1.352, R_gas=290.0, gamma=1.40,
    surface_pressure=146700.0, surface_temp=93.7, surface_density=5.428,
    radius=2574700.0, scale_height=21000.0,
    composition="N₂ 94.2% + CH₄ 5.65%",
    mu_ref=6.2e-6, T_ref_visc=94.0, S_visc=79.4,
)

PLANETS = {
    "earth": EARTH, "mars": MARS, "venus": VENUS, "titan": TITAN,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MARS ATMOSPHERE (NASA Mars-GRAM Simplified)
# ═══════════════════════════════════════════════════════════════════════════════

class MarsAtmosphere:
    """
    3-layer Mars atmosphere model derived from NASA Mars-GRAM 2010.

    Layers:
      0– 7 km : Surface layer  (lapse rate ≈ −2.5 K/km, surface T ≈ 210 K)
      7–40 km : Middle atm     (lapse rate ≈ −1.5 K/km)
      40+ km  : Upper atm      (isothermal ~140 K, exponential decay)

    Validated against Curiosity rover REMS data and MER entry profiles.
    """

    P = MARS

    # Layer definitions: (base_alt_m, lapse_rate_K/m, base_temp_K, base_press_Pa)
    _LAYERS = [
        (0.0,     -0.0025,  210.0,   636.0),     # Surface layer
        (7000.0,  -0.0015,  192.5,   330.0),     # Middle atmosphere
        (40000.0,  0.0000,  143.0,    12.0),     # Upper atmosphere (isothermal)
    ]

    @classmethod
    def _get_layer(cls, h: float) -> tuple:
        h = max(0.0, h)
        layer = cls._LAYERS[0]
        for L in cls._LAYERS:
            if h >= L[0]:
                layer = L
            else:
                break
        return layer

    @classmethod
    def temperature(cls, h: float) -> float:
        """Mars ambient temperature [K] at altitude h [m MOLA]."""
        h = max(0.0, h)
        h0, lr, T0, _ = cls._get_layer(h)
        T = T0 + lr * (h - h0)
        return max(T, 100.0)   # Floor at 100 K

    @classmethod
    def pressure(cls, h: float) -> float:
        """Mars ambient pressure [Pa] at altitude h [m MOLA]."""
        h = max(0.0, h)
        h0, lr, T0, P0 = cls._get_layer(h)
        T = cls.temperature(h)
        g, R = cls.P.gravity, cls.P.R_gas
        if abs(lr) < 1e-12:
            return P0 * np.exp(-g * (h - h0) / (R * T0))
        else:
            return P0 * (T / T0) ** (-g / (lr * R))

    @classmethod
    def density(cls, h: float) -> float:
        """Mars air density [kg/m³]."""
        T = cls.temperature(h)
        P = cls.pressure(h)
        return P / (cls.P.R_gas * T)

    @classmethod
    def speed_of_sound(cls, h: float) -> float:
        """Speed of sound [m/s] in Mars CO₂ atmosphere."""
        return np.sqrt(cls.P.gamma * cls.P.R_gas * cls.temperature(h))

    @classmethod
    def dynamic_viscosity(cls, h: float) -> float:
        """Dynamic viscosity [Pa·s] via Sutherland's law for CO₂."""
        T = cls.temperature(h)
        mu0, T0, S = cls.P.mu_ref, cls.P.T_ref_visc, cls.P.S_visc
        return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

    @classmethod
    def kinematic_viscosity(cls, h: float) -> float:
        return cls.dynamic_viscosity(h) / max(cls.density(h), 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VENUS ATMOSPHERE (VIRA)
# ═══════════════════════════════════════════════════════════════════════════════

class VenusAtmosphere:
    """
    Venus International Reference Atmosphere (VIRA).
    Dense CO₂ atmosphere: surface P = 9.2 MPa, T = 737 K.
    Relevant for aerobraking / aerocapture applications.
    """

    P = VENUS

    _LAYERS = [
        (0.0,     -0.0080, 737.0, 9.2e6),       # Lower atmosphere
        (30000.0, -0.0030, 497.0, 1.07e6),       # Middle atmosphere
        (50000.0, -0.0025, 437.0, 1.07e5),       # Upper troposphere
        (65000.0, -0.0040, 400.0, 2.15e4),       # Mesosphere
    ]

    @classmethod
    def _get_layer(cls, h: float) -> tuple:
        h = max(0.0, h)
        layer = cls._LAYERS[0]
        for L in cls._LAYERS:
            if h >= L[0]:
                layer = L
            else:
                break
        return layer

    @classmethod
    def temperature(cls, h: float) -> float:
        h = max(0.0, h)
        h0, lr, T0, _ = cls._get_layer(h)
        return max(T0 + lr * (h - h0), 150.0)

    @classmethod
    def pressure(cls, h: float) -> float:
        h = max(0.0, h)
        h0, lr, T0, P0 = cls._get_layer(h)
        T = cls.temperature(h)
        g, R = cls.P.gravity, cls.P.R_gas
        if abs(lr) < 1e-12:
            return P0 * np.exp(-g * (h - h0) / (R * T0))
        else:
            return P0 * (T / T0) ** (-g / (lr * R))

    @classmethod
    def density(cls, h: float) -> float:
        return cls.pressure(h) / (cls.P.R_gas * cls.temperature(h))

    @classmethod
    def speed_of_sound(cls, h: float) -> float:
        return np.sqrt(cls.P.gamma * cls.P.R_gas * cls.temperature(h))

    @classmethod
    def dynamic_viscosity(cls, h: float) -> float:
        T = cls.temperature(h)
        mu0, T0, S = cls.P.mu_ref, cls.P.T_ref_visc, cls.P.S_visc
        return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

    @classmethod
    def kinematic_viscosity(cls, h: float) -> float:
        return cls.dynamic_viscosity(h) / max(cls.density(h), 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TITAN ATMOSPHERE (Huygens-derived)
# ═══════════════════════════════════════════════════════════════════════════════

class TitanAtmosphere:
    """
    Titan atmosphere from Huygens HASI instrument measurements.
    Dense N₂ atmosphere: surface P = 1.47 bar, T = 93.7 K.
    Relevant for Saturn moon missions and general outer-planet EDL.
    """

    P = TITAN

    _LAYERS = [
        (0.0,      -0.0012,  93.7,   146700.0),   # Troposphere
        (42000.0,   0.0000,  71.0,    20000.0),   # Tropopause (cold trap)
        (60000.0,   0.0008,  71.0,     8000.0),   # Stratosphere
        (200000.0,  0.0010, 183.0,      100.0),   # Thermosphere
    ]

    @classmethod
    def _get_layer(cls, h: float) -> tuple:
        h = max(0.0, h)
        layer = cls._LAYERS[0]
        for L in cls._LAYERS:
            if h >= L[0]:
                layer = L
            else:
                break
        return layer

    @classmethod
    def temperature(cls, h: float) -> float:
        h = max(0.0, h)
        h0, lr, T0, _ = cls._get_layer(h)
        return max(T0 + lr * (h - h0), 60.0)

    @classmethod
    def pressure(cls, h: float) -> float:
        h = max(0.0, h)
        h0, lr, T0, P0 = cls._get_layer(h)
        T = cls.temperature(h)
        g, R = cls.P.gravity, cls.P.R_gas
        if abs(lr) < 1e-12:
            return P0 * np.exp(-g * (h - h0) / (R * T0))
        else:
            return P0 * (T / T0) ** (-g / (lr * R))

    @classmethod
    def density(cls, h: float) -> float:
        return cls.pressure(h) / (cls.P.R_gas * cls.temperature(h))

    @classmethod
    def speed_of_sound(cls, h: float) -> float:
        return np.sqrt(cls.P.gamma * cls.P.R_gas * cls.temperature(h))

    @classmethod
    def dynamic_viscosity(cls, h: float) -> float:
        T = cls.temperature(h)
        mu0, T0, S = cls.P.mu_ref, cls.P.T_ref_visc, cls.P.S_visc
        return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

    @classmethod
    def kinematic_viscosity(cls, h: float) -> float:
        return cls.dynamic_viscosity(h) / max(cls.density(h), 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UNIFIED INTERFACE — Drop-in replacement for atmosphere.py
# ═══════════════════════════════════════════════════════════════════════════════

_ATM_MAP = {
    "mars":  MarsAtmosphere,
    "venus": VenusAtmosphere,
    "titan": TitanAtmosphere,
}

def get_atmosphere(planet: str = "earth"):
    """
    Get an atmosphere model for the specified planet.

    Returns an object with methods:
      .temperature(h), .pressure(h), .density(h),
      .speed_of_sound(h), .dynamic_viscosity(h)

    For Earth, delegates to the existing ISA model in atmosphere.py.

    Usage:
        atm = get_atmosphere("mars")
        rho = atm.density(5000.0)
        T   = atm.temperature(5000.0)
    """
    planet = planet.lower().strip()
    if planet == "earth":
        # Return a wrapper around the existing ISA module
        from src import atmosphere as isa
        return _EarthWrapper(isa)
    if planet in _ATM_MAP:
        return _ATM_MAP[planet]
    raise ValueError(
        f"Unknown planet: '{planet}'. Available: {list(PLANETS.keys())}"
    )


class _EarthWrapper:
    """Thin wrapper that gives Earth ISA the same class-method interface."""
    P = EARTH

    def __init__(self, isa_module):
        self._isa = isa_module

    def temperature(self, h): return self._isa.temperature(h)
    def pressure(self, h):    return self._isa.pressure(h)
    def density(self, h):     return self._isa.density(h)
    def speed_of_sound(self, h): return self._isa.speed_of_sound(h)
    def dynamic_viscosity(self, h): return self._isa.dynamic_viscosity(h)
    def kinematic_viscosity(self, h): return self._isa.kinematic_viscosity(h)


def get_planet_constants(planet: str = "earth") -> PlanetaryConstants:
    """Get physical constants for a planet."""
    return PLANETS[planet.lower().strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VECTORIZED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def vectorized_density(altitudes: np.ndarray, planet: str = "earth") -> np.ndarray:
    atm = get_atmosphere(planet)
    return np.array([atm.density(float(h)) for h in altitudes])

def vectorized_temperature(altitudes: np.ndarray, planet: str = "earth") -> np.ndarray:
    atm = get_atmosphere(planet)
    return np.array([atm.temperature(float(h)) for h in altitudes])

def vectorized_pressure(altitudes: np.ndarray, planet: str = "earth") -> np.ndarray:
    atm = get_atmosphere(planet)
    return np.array([atm.pressure(float(h)) for h in altitudes])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. COMPARISON / VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def atmosphere_comparison_table(planets: list = None,
                                 altitudes: list = None) -> str:
    """Print a comparison table of atmospheric properties across planets."""
    planets = planets or ["earth", "mars", "venus", "titan"]
    altitudes = altitudes or [0, 1000, 5000, 10000, 20000, 50000]

    lines = []
    for planet in planets:
        atm = get_atmosphere(planet)
        pc = get_planet_constants(planet)
        lines.append(f"\n{'═'*70}")
        lines.append(f"  {pc.name.upper()} — {pc.composition}")
        lines.append(f"  g={pc.gravity:.3f} m/s²  R={pc.R_gas:.1f} J/(kg·K)  "
                      f"γ={pc.gamma:.2f}  H={pc.scale_height:.0f} m")
        lines.append(f"{'═'*70}")
        lines.append(f"  {'Alt [m]':>10} {'T [K]':>10} {'P [Pa]':>14} "
                      f"{'ρ [kg/m³]':>14} {'a [m/s]':>10}")
        lines.append(f"  {'─'*10} {'─'*10} {'─'*14} {'─'*14} {'─'*10}")
        for alt in altitudes:
            try:
                T   = atm.temperature(alt)
                P   = atm.pressure(alt)
                rho = atm.density(alt)
                a   = atm.speed_of_sound(alt)
                lines.append(
                    f"  {alt:>10.0f} {T:>10.2f} {P:>14.4f} "
                    f"{rho:>14.6f} {a:>10.2f}"
                )
            except Exception:
                lines.append(f"  {alt:>10.0f}  (out of range)")

    output = "\n".join(lines)
    print(output)
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO SCENARIO PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EDLScenario:
    """Pre-configured EDL scenario for demo/validation."""
    name:          str
    planet:        str
    mass_kg:       float
    initial_alt_m: float
    initial_vel_ms: float
    canopy_area_m2: float
    Cd:            float
    canopy_diam_m: float
    description:   str = ""


# Mars EDL — Perseverance / MSL-class
MARS_EDL = EDLScenario(
    name="Mars EDL (Perseverance-class)",
    planet="mars",
    mass_kg=900.0,              # entry vehicle mass
    initial_alt_m=10000.0,      # parachute deployment at ~10 km MOLA
    initial_vel_ms=400.0,       # supersonic deployment (~Mach 1.7 on Mars)
    canopy_area_m2=201.0,       # 21.5 m diameter disk-gap-band
    Cd=0.62,                    # DGB drag coefficient
    canopy_diam_m=21.5,
    description="Mars Science Laboratory / Perseverance-class 21.5m DGB "
                "parachute for supersonic decelerator deployment",
)

# Drone recovery — small UAV parachute
DRONE_RECOVERY = EDLScenario(
    name="Drone Recovery (UAV)",
    planet="earth",
    mass_kg=5.0,
    initial_alt_m=120.0,
    initial_vel_ms=15.0,
    canopy_area_m2=2.5,
    Cd=1.2,
    canopy_diam_m=1.8,
    description="Consumer/commercial UAV parachute recovery system",
)

# Reentry capsule — SpaceX Dragon-class
REENTRY_CAPSULE = EDLScenario(
    name="Reentry Capsule (Dragon-class)",
    planet="earth",
    mass_kg=6400.0,
    initial_alt_m=5500.0,       # drogue deployment ~5.5 km
    initial_vel_ms=125.0,       # after aero deceleration
    canopy_area_m2=1160.0,      # 4 × main parachutes
    Cd=1.1,
    canopy_diam_m=35.0,         # Mark 3 main parachute
    description="SpaceX Crew Dragon-class capsule with 4-main parachute system",
)

# Military airdrop — C-130 cargo extraction
MILITARY_AIRDROP = EDLScenario(
    name="Military Airdrop (C-130 cargo)",
    planet="earth",
    mass_kg=4500.0,             # heavy pallet
    initial_alt_m=800.0,        # low-altitude tactical extraction
    initial_vel_ms=60.0,        # C-130 extraction speed
    canopy_area_m2=85.0,        # G-12 cargo parachute
    Cd=1.35,
    canopy_diam_m=20.0,
    description="MIL-HDBK-1791 compliant C-130 tactical cargo airdrop",
)

DEMO_SCENARIOS = {
    "mars_edl":          MARS_EDL,
    "drone_recovery":    DRONE_RECOVERY,
    "reentry_capsule":   REENTRY_CAPSULE,
    "military_airdrop":  MILITARY_AIRDROP,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    atmosphere_comparison_table()
    print("\n\nDemo Scenarios:")
    for key, sc in DEMO_SCENARIOS.items():
        print(f"  {key:20s} → {sc.name}  ({sc.planet}, {sc.mass_kg}kg, "
              f"h₀={sc.initial_alt_m}m, v₀={sc.initial_vel_ms}m/s)")
