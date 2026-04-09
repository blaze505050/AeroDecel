"""
atmosphere.py — International Standard Atmosphere (ISA) model.

AeroDecel v5.0 — Enhanced with geopotential corrections, vectorized operations,
and convenience functions for Mach number and kinematic viscosity.

Provides density, pressure, temperature, speed of sound, viscosity, and
Reynolds/Mach numbers at any geometric altitude.

Reference: ICAO Doc 7488/3, US Standard Atmosphere 1976.
"""

import numpy as np


# ─── ISA Layer Definitions ────────────────────────────────────────────────────
# Each layer: (base_alt_m, lapse_rate_K/m, base_temp_K, base_press_Pa)
_LAYERS = [
    (0.0,     -0.0065,  288.15, 101325.0),   # Troposphere
    (11000.0,  0.0000,  216.65,  22632.1),   # Tropopause (isothermal)
    (20000.0,  0.0010,  216.65,   5474.89),  # Stratosphere 1
    (32000.0,  0.0028,  228.65,    868.019), # Stratosphere 2
    (47000.0,  0.0000,  270.65,    110.906), # Stratopause
    (51000.0, -0.0028,  270.65,     66.9389),# Mesosphere 1
    (71000.0, -0.0020,  214.65,      3.9564),# Mesosphere 2
]

# Constants
R_AIR   = 287.058   # J/(kg·K)  specific gas constant for dry air
GAMMA   = 1.4       # adiabatic index
G0      = 9.80665   # m/s²
M_AIR   = 0.0289644 # kg/mol
R_EARTH = 6356766.0 # m — effective Earth radius for geopotential correction


def geopotential_altitude(geometric_m: float) -> float:
    """Convert geometric altitude [m] to geopotential altitude [m].

    Geopotential altitude accounts for the decrease in gravitational
    acceleration with altitude: h_gp = R_e · h_geo / (R_e + h_geo)
    """
    h = max(0.0, float(geometric_m))
    return R_EARTH * h / (R_EARTH + h)


def geometric_altitude(geopotential_m: float) -> float:
    """Convert geopotential altitude [m] to geometric altitude [m]."""
    h = max(0.0, float(geopotential_m))
    return R_EARTH * h / (R_EARTH - h)


def _get_layer(altitude_m: float) -> tuple:
    """Return the ISA layer parameters for a given geometric altitude."""
    h = max(0.0, float(altitude_m))
    layer = _LAYERS[0]
    for L in _LAYERS:
        if h >= L[0]:
            layer = L
        else:
            break
    return layer


def temperature(altitude_m: float) -> float:
    """Ambient temperature [K] at geometric altitude [m]."""
    h0, lr, T0, _ = _get_layer(altitude_m)
    return T0 + lr * (max(0.0, altitude_m) - h0)


def pressure(altitude_m: float) -> float:
    """Ambient pressure [Pa] at geometric altitude [m]."""
    h = max(0.0, float(altitude_m))
    h0, lr, T0, P0 = _get_layer(h)
    T = T0 + lr * (h - h0)
    if abs(lr) < 1e-12:                          # isothermal layer
        return P0 * np.exp(-G0 * (h - h0) / (R_AIR * T0))
    else:
        return P0 * (T / T0) ** (-G0 / (lr * R_AIR))


def density(altitude_m: float) -> float:
    """Air density ρ [kg/m³] at geometric altitude [m]."""
    T = temperature(altitude_m)
    P = pressure(altitude_m)
    return P / (R_AIR * T)


def speed_of_sound(altitude_m: float) -> float:
    """Speed of sound [m/s] at geometric altitude [m]."""
    return np.sqrt(GAMMA * R_AIR * temperature(altitude_m))


def dynamic_viscosity(altitude_m: float) -> float:
    """Dynamic viscosity [Pa·s] via Sutherland's law."""
    T   = temperature(altitude_m)
    mu0 = 1.716e-5    # Pa·s at T0
    T0  = 273.15
    C   = 110.4       # Sutherland's constant [K]
    return mu0 * (T / T0) ** 1.5 * (T0 + C) / (T + C)


def kinematic_viscosity(altitude_m: float) -> float:
    """Kinematic viscosity ν [m²/s] at geometric altitude [m].

    ν = μ / ρ — used for Reynolds number calculations.
    """
    mu  = dynamic_viscosity(altitude_m)
    rho = density(altitude_m)
    return mu / max(rho, 1e-12)


def mach_number(velocity_ms: float, altitude_m: float) -> float:
    """Compute Mach number for given velocity and altitude.

    M = V / a(h) where a(h) is the ISA speed of sound.
    """
    a = speed_of_sound(altitude_m)
    return abs(velocity_ms) / max(a, 1.0)


def reynolds_number(velocity_m_s: float, length_m: float, altitude_m: float) -> float:
    """Reynolds number for given velocity, characteristic length, and altitude.

    Re = ρ·V·L / μ
    """
    rho = density(altitude_m)
    mu  = dynamic_viscosity(altitude_m)
    return rho * abs(velocity_m_s) * length_m / mu


def vectorized_density(altitudes: np.ndarray) -> np.ndarray:
    """Vectorized version of density for NumPy arrays."""
    return np.array([density(float(h)) for h in altitudes])


def vectorized_temperature(altitudes: np.ndarray) -> np.ndarray:
    """Vectorized version of temperature for NumPy arrays."""
    return np.array([temperature(float(h)) for h in altitudes])


def vectorized_speed_of_sound(altitudes: np.ndarray) -> np.ndarray:
    """Vectorized version of speed_of_sound for NumPy arrays."""
    return np.array([speed_of_sound(float(h)) for h in altitudes])


def vectorized_mach(velocities: np.ndarray, altitudes: np.ndarray) -> np.ndarray:
    """Vectorized Mach number for parallel velocity/altitude arrays."""
    return np.array([mach_number(float(v), float(h))
                     for v, h in zip(velocities, altitudes)])


if __name__ == "__main__":
    print("ISA Model Validation (AeroDecel v5.0):")
    print(f"{'Alt (m)':>10} {'T (K)':>10} {'P (Pa)':>12} {'ρ (kg/m³)':>12} {'a (m/s)':>10} {'ν (m²/s)':>12}")
    for alt in [0, 1000, 5000, 11000, 20000, 32000]:
        T = temperature(alt)
        P = pressure(alt)
        rho = density(alt)
        a = speed_of_sound(alt)
        nu = kinematic_viscosity(alt)
        print(f"{alt:>10.0f} {T:>10.2f} {P:>12.2f} {rho:>12.5f} {a:>10.2f} {nu:>12.2e}")
