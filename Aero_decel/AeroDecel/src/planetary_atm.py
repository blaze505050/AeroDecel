"""
src/planetary_atm.py — Planetary Atmosphere Models
====================================================
Multi-layer atmosphere models for Mars, Venus, Titan, and generic planets.
Each model provides density, pressure, temperature, and scale-height profiles
calibrated to published mission data (MCS, VIRA, Huygens).

References
----------
  Mars  : Mars Climate Database v5.3 / Seiff & Kirk (1977)
  Venus : VIRA (Seiff et al. 1985)
  Titan : Fulchignoni et al. 2005 (Huygens probe)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanetaryAtmosphere:
    name:                   str
    radius_m:               float    # m
    mass_kg:                float    # kg
    gravity_ms2:            float    # m/s²
    surface_pressure_pa:    float    # Pa
    gas_constant:           float    # J/(kg·K)  specific gas constant
    composition:            dict     # {species: mole_fraction}

    def density(self, altitude_m: float) -> float:
        raise NotImplementedError

    def temperature(self, altitude_m: float) -> float:
        raise NotImplementedError

    def pressure(self, altitude_m: float) -> float:
        """Hydrostatic pressure from density and temperature."""
        return self.density(altitude_m) * self.gas_constant * self.temperature(altitude_m)

    def scale_height(self, altitude_m: float) -> float:
        """Local pressure scale height H = RT/g."""
        return self.gas_constant * self.temperature(altitude_m) / self.gravity_ms2

    def speed_of_sound(self, altitude_m: float, gamma: float = 1.4) -> float:
        return float(np.sqrt(gamma * self.gas_constant * self.temperature(altitude_m)))

    def mach_number(self, velocity_ms: float, altitude_m: float) -> float:
        return velocity_ms / max(self.speed_of_sound(altitude_m), 1.0)

    def profile(self, alt_max_m: float = 100_000, n: int = 200) -> dict:
        """Return altitude, density, temperature, pressure arrays for plotting."""
        alts = np.linspace(0, alt_max_m, n)
        return {
            "altitude_m":    alts,
            "density":       np.array([self.density(h)     for h in alts]),
            "temperature_K": np.array([self.temperature(h) for h in alts]),
            "pressure_Pa":   np.array([self.pressure(h)    for h in alts]),
        }

    def __repr__(self):
        return (f"<{self.__class__.__name__} g={self.gravity_ms2:.3f}m/s² "
                f"R={self.gas_constant:.1f}J/kgK>")


# ══════════════════════════════════════════════════════════════════════════════
# MARS  — multi-layer exponential model (Seiff & Kirk layers)
# ══════════════════════════════════════════════════════════════════════════════

class MarsAtmosphere(PlanetaryAtmosphere):
    """
    Mars atmosphere using a two-layer exponential model.
    Lower  troposphere (0–30 km): richer, warmer
    Upper  mesosphere (>30 km):   rarefied, colder
    Surface conditions: ρ₀=0.020 kg/m³, T₀=210 K, P₀≈636 Pa
    """

    def __init__(self):
        super().__init__(
            name                = "Mars",
            radius_m            = 3_389_500.0,
            mass_kg             = 6.4171e23,
            gravity_ms2         = 3.72076,
            surface_pressure_pa = 636.0,
            gas_constant        = 188.9,    # J/(kg·K)  CO₂-dominated
            composition         = {"CO2": 0.953, "N2": 0.027, "Ar": 0.016, "O2": 0.001},
        )
        # Layer definitions: (alt_base, scale_height, base_density, base_temp, lapse_rate)
        self._layers = [
            (0,      11_100, 0.0200, 210.0, -0.0005),
            (30_000, 7_500,  0.0012, 180.0, -0.0003),
            (70_000, 5_000,  2e-5,   130.0,  0.0000),
        ]

    def _layer(self, h: float):
        """Return layer parameters for altitude h [m]."""
        layer = self._layers[0]
        for L in self._layers:
            if h >= L[0]:
                layer = L
        return layer

    def density(self, altitude_m: float) -> float:
        h = max(0.0, altitude_m)
        alt0, H, rho0, T0, lr = self._layer(h)
        T = T0 + lr * (h - alt0)
        rho = rho0 * np.exp(-(h - alt0) / H)
        return float(max(rho, 1e-12))

    def temperature(self, altitude_m: float) -> float:
        h = max(0.0, altitude_m)
        alt0, H, rho0, T0, lr = self._layer(h)
        return float(max(T0 + lr * (h - alt0), 80.0))


# ══════════════════════════════════════════════════════════════════════════════
# VENUS  — VIRA model
# ══════════════════════════════════════════════════════════════════════════════

class VenusAtmosphere(PlanetaryAtmosphere):
    """
    Venus atmosphere based on VIRA (Venus International Reference Atmosphere).
    Extremely dense CO₂ atmosphere; surface: 65 kg/m³, 737 K, 92 bar.
    """

    def __init__(self):
        super().__init__(
            name                = "Venus",
            radius_m            = 6_051_800.0,
            mass_kg             = 4.8675e24,
            gravity_ms2         = 8.870,
            surface_pressure_pa = 9_200_000.0,
            gas_constant        = 188.9,
            composition         = {"CO2": 0.965, "N2": 0.035},
        )
        self._layers = [
            (0,      15_900, 65.0,  737.0, -7.8e-3),
            (50_000,  9_000,  1.2,  255.0, -4.0e-3),
            (70_000, 10_000,  0.06, 180.0, -2.0e-3),
        ]

    def _layer(self, h):
        layer = self._layers[0]
        for L in self._layers:
            if h >= L[0]:
                layer = L
        return layer

    def density(self, altitude_m: float) -> float:
        h = max(0.0, altitude_m)
        alt0, H, rho0, T0, lr = self._layer(h)
        return float(max(rho0 * np.exp(-(h - alt0) / H), 1e-12))

    def temperature(self, altitude_m: float) -> float:
        h = max(0.0, altitude_m)
        alt0, H, rho0, T0, lr = self._layer(h)
        return float(max(T0 + lr * (h - alt0), 160.0))


# ══════════════════════════════════════════════════════════════════════════════
# TITAN  — Huygens probe model
# ══════════════════════════════════════════════════════════════════════════════

class TitanAtmosphere(PlanetaryAtmosphere):
    """
    Titan atmosphere calibrated to Huygens HASI measurements (2005).
    Dense N₂/CH₄ atmosphere; surface: 1.4 kg/m³, 94 K, 147 kPa.
    """

    def __init__(self):
        super().__init__(
            name                = "Titan",
            radius_m            = 2_574_700.0,
            mass_kg             = 1.3452e23,
            gravity_ms2         = 1.352,
            surface_pressure_pa = 147_000.0,
            gas_constant        = 290.0,    # N₂-dominated
            composition         = {"N2": 0.984, "CH4": 0.014, "H2": 0.002},
        )
        self._layers = [
            (0,       40_000, 1.40,  94.0, +0.00090),
            (40_000,  35_000, 0.08, 130.0, -0.00030),
            (100_000, 25_000, 1e-3, 175.0,  0.00000),
        ]

    def _layer(self, h):
        layer = self._layers[0]
        for L in self._layers:
            if h >= L[0]:
                layer = L
        return layer

    def density(self, altitude_m: float) -> float:
        h = max(0.0, altitude_m)
        alt0, H, rho0, T0, lr = self._layer(h)
        return float(max(rho0 * np.exp(-(h - alt0) / H), 1e-15))

    def temperature(self, altitude_m: float) -> float:
        h = max(0.0, altitude_m)
        alt0, H, rho0, T0, lr = self._layer(h)
        return float(max(T0 + lr * (h - alt0), 70.0))


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC PLANET
# ══════════════════════════════════════════════════════════════════════════════

class GenericPlanetAtmosphere(PlanetaryAtmosphere):
    """User-defined exponential atmosphere."""

    def __init__(self, name, radius_m, mass_kg, gravity_ms2, composition,
                 scale_height_m, base_density, base_temp_K,
                 lapse_rate=0.0, gas_constant=287.0, surface_pressure_pa=101325.0):
        super().__init__(name, radius_m, mass_kg, gravity_ms2,
                         surface_pressure_pa, gas_constant, composition)
        self._H   = scale_height_m
        self._rho0 = base_density
        self._T0   = base_temp_K
        self._lr   = lapse_rate

    def density(self, altitude_m: float) -> float:
        return float(max(self._rho0 * np.exp(-max(altitude_m, 0) / self._H), 1e-15))

    def temperature(self, altitude_m: float) -> float:
        return float(max(self._T0 + self._lr * max(altitude_m, 0), 50.0))


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

_REGISTRY: dict[str, type] = {
    "mars":  MarsAtmosphere,
    "venus": VenusAtmosphere,
    "titan": TitanAtmosphere,
}


def get_planet_atmosphere(planet_name: str) -> PlanetaryAtmosphere:
    """Return an atmosphere model by name (case-insensitive)."""
    key = planet_name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unsupported planet '{planet_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]()


def register_planet(name: str, cls: type):
    """Register a custom PlanetaryAtmosphere subclass."""
    _REGISTRY[name.lower()] = cls


if __name__ == "__main__":
    for pname in ["mars", "venus", "titan"]:
        atm = get_planet_atmosphere(pname)
        print(f"{atm.name:6s}  ρ_surf={atm.density(0):.4f} kg/m³  "
              f"T_surf={atm.temperature(0):.1f}K  "
              f"g={atm.gravity_ms2:.3f}m/s²  "
              f"a={atm.speed_of_sound(0):.1f}m/s")
