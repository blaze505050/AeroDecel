"""
canopy_geometry.py — Parametric Canopy Cross-Section Generator (AeroDecel v6.0)
================================================================================
Generates 2D cross-section profiles for different canopy types, ready for
injection into the LBM solver as immersed boundary conditions.

Supported geometries:
  - Flat circular (standard round canopy)
  - Hemisphere (deployed hemispherical canopy — most common)
  - Extended skirt (with configurable skirt angle)
  - Ribbon / ringslot (perforated — modeled as porous boundary)
  - Ram-air / parafoil (airfoil cross-section)
  - Disk-Gap-Band (DGB — Mars EDL, Supersonic decelerator)

Each geometry returns:
  - (x, y) boundary points
  - Normal vectors at each boundary point
  - Porosity flag per segment
  - Reference area and diameter

Reference:
  - Knacke, "Parachute Recovery Systems Design Manual", 1992
  - Cruz et al., "Parachute Design Guide", NASA CR-2746
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class CanopyProfile:
    """2D canopy cross-section for LBM simulation."""
    name:        str
    x:           np.ndarray    # x coordinates [m] (relative to center)
    y:           np.ndarray    # y coordinates [m]
    nx:          np.ndarray    # outward normal x-component
    ny:          np.ndarray    # outward normal y-component
    porosity:    np.ndarray    # porosity per segment (0=solid, 1=fully open)
    diameter:    float         # nominal canopy diameter [m]
    area_ref:    float         # reference projected area [m²]
    n_points:    int           # number of boundary points


def flat_circular(diameter: float = 8.0, n_points: int = 200,
                  curvature: float = 0.05) -> CanopyProfile:
    """
    Flat circular canopy cross-section.
    Slightly curved disk with thickness proportional to curvature parameter.
    """
    R = diameter / 2.0
    theta = np.linspace(-np.pi/2, np.pi/2, n_points)

    # Upper surface (windward — slightly convex)
    x_upper = R * np.sin(theta)
    y_upper = curvature * R * np.cos(theta)

    # Lower surface (leeward — mirror)
    x_lower = x_upper[::-1]
    y_lower = -y_upper[::-1]

    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])

    # Normals (finite difference approximation)
    nx, ny = _compute_normals(x, y)

    return CanopyProfile(
        name="flat_circular", x=x, y=y, nx=nx, ny=ny,
        porosity=np.zeros(len(x)),
        diameter=diameter,
        area_ref=np.pi * R**2,
        n_points=len(x),
    )


def hemisphere(diameter: float = 8.0, n_points: int = 200,
               depth_ratio: float = 0.5) -> CanopyProfile:
    """
    Hemispherical canopy cross-section.
    depth_ratio: ratio of canopy depth to diameter (0.5 = perfect hemisphere)
    """
    R = diameter / 2.0
    depth = R * depth_ratio

    theta = np.linspace(-np.pi/2, np.pi/2, n_points)

    # Windward surface (arc)
    x = R * np.sin(theta)
    y = depth * np.cos(theta)

    # Close the base (straight line across the opening)
    x_base = np.linspace(R, -R, n_points // 4)
    y_base = np.full_like(x_base, 0.0)

    x = np.concatenate([x, x_base])
    y = np.concatenate([y, y_base])

    nx, ny = _compute_normals(x, y)

    return CanopyProfile(
        name="hemisphere", x=x, y=y, nx=nx, ny=ny,
        porosity=np.zeros(len(x)),
        diameter=diameter,
        area_ref=np.pi * R**2,
        n_points=len(x),
    )


def extended_skirt(diameter: float = 8.0, n_points: int = 250,
                   skirt_angle_deg: float = 25.0,
                   skirt_length_ratio: float = 0.15) -> CanopyProfile:
    """
    Extended-skirt conical canopy.
    Central hemisphere with angled skirt extensions.
    """
    R = diameter / 2.0
    skirt_len = R * skirt_length_ratio
    skirt_angle = np.deg2rad(skirt_angle_deg)

    # Main hemisphere
    n_hem = int(n_points * 0.7)
    n_skirt = (n_points - n_hem) // 2

    theta = np.linspace(-np.pi/2 + 0.2, np.pi/2 - 0.2, n_hem)
    x_hem = R * 0.85 * np.sin(theta)
    y_hem = R * 0.4 * np.cos(theta)

    # Left skirt
    x_skirt_l = np.linspace(x_hem[0], x_hem[0] - skirt_len * np.cos(skirt_angle), n_skirt)
    y_skirt_l = np.linspace(y_hem[0], y_hem[0] - skirt_len * np.sin(skirt_angle), n_skirt)

    # Right skirt
    x_skirt_r = np.linspace(x_hem[-1], x_hem[-1] + skirt_len * np.cos(skirt_angle), n_skirt)
    y_skirt_r = np.linspace(y_hem[-1], y_hem[-1] - skirt_len * np.sin(skirt_angle), n_skirt)

    x = np.concatenate([x_skirt_l[::-1], x_hem, x_skirt_r])
    y = np.concatenate([y_skirt_l[::-1], y_hem, y_skirt_r])

    nx, ny = _compute_normals(x, y)

    return CanopyProfile(
        name="extended_skirt", x=x, y=y, nx=nx, ny=ny,
        porosity=np.zeros(len(x)),
        diameter=diameter,
        area_ref=np.pi * R**2,
        n_points=len(x),
    )


def ribbon(diameter: float = 8.0, n_points: int = 200,
           slot_fraction: float = 0.15,
           n_ribbons: int = 12) -> CanopyProfile:
    """
    Ribbon / ringslot canopy — porous sections between solid ribbons.
    Used for high-speed applications (drogues).
    """
    R = diameter / 2.0
    theta = np.linspace(-np.pi/2, np.pi/2, n_points)
    x = R * np.sin(theta)
    y = R * 0.3 * np.cos(theta)

    nx, ny = _compute_normals(x, y)

    # Assign porosity: alternating solid/porous segments
    porosity = np.zeros(len(x))
    segment = len(x) // (2 * n_ribbons)
    for i in range(n_ribbons):
        # Slot (porous) zone
        start = (2 * i + 1) * segment
        end = min(start + segment, len(x))
        porosity[start:end] = slot_fraction / 0.15  # normalized

    return CanopyProfile(
        name="ribbon", x=x, y=y, nx=nx, ny=ny,
        porosity=np.clip(porosity, 0, 1),
        diameter=diameter,
        area_ref=np.pi * R**2,
        n_points=len(x),
    )


def disk_gap_band(diameter: float = 21.5, n_points: int = 300,
                  band_height_ratio: float = 0.12,
                  gap_ratio: float = 0.015) -> CanopyProfile:
    """
    Disk-Gap-Band (DGB) canopy — used for Mars EDL (MSL/Perseverance).
    Disk → small gap → cylindrical band.
    """
    R = diameter / 2.0
    band_h = diameter * band_height_ratio
    gap = diameter * gap_ratio

    n_disk = int(n_points * 0.4)
    n_gap = int(n_points * 0.1)
    n_band = n_points - n_disk - n_gap

    # Disk (slightly curved)
    theta = np.linspace(-np.pi/2, np.pi/2, n_disk)
    x_disk = R * 0.95 * np.sin(theta)
    y_disk = R * 0.08 * np.cos(theta)

    # Gap (porous opening between disk and band)
    x_gap_r = np.full(n_gap // 2, R * 0.95)
    y_gap_r = np.linspace(y_disk[-1], y_disk[-1] - gap, n_gap // 2)

    x_gap_l = np.full(n_gap // 2, -R * 0.95)
    y_gap_l = np.linspace(y_disk[0] - gap, y_disk[0], n_gap // 2)

    # Band (cylindrical)
    n_band_side = n_band // 3
    n_band_bottom = n_band - 2 * n_band_side

    x_band_r = np.full(n_band_side, R * 0.95)
    y_band_r = np.linspace(y_disk[-1] - gap, y_disk[-1] - gap - band_h, n_band_side)

    x_band_bot = np.linspace(R * 0.95, -R * 0.95, n_band_bottom)
    y_band_bot = np.full(n_band_bottom, y_disk[-1] - gap - band_h)

    x_band_l = np.full(n_band_side, -R * 0.95)
    y_band_l = np.linspace(y_disk[0] - gap - band_h, y_disk[0] - gap, n_band_side)

    x = np.concatenate([x_gap_l, x_disk, x_gap_r, x_band_r, x_band_bot, x_band_l])
    y = np.concatenate([y_gap_l, y_disk, y_gap_r, y_band_r, y_band_bot, y_band_l])

    nx, ny = _compute_normals(x, y)

    # Gap regions are porous
    porosity = np.zeros(len(x))
    porosity[:n_gap // 2] = 0.8   # left gap
    porosity[n_disk + n_gap // 2:n_disk + n_gap] = 0.8  # right gap

    return CanopyProfile(
        name="disk_gap_band", x=x, y=y, nx=nx, ny=ny,
        porosity=porosity,
        diameter=diameter,
        area_ref=np.pi * R**2,
        n_points=len(x),
    )


def ram_air(chord: float = 3.0, span: float = 10.0,
            n_points: int = 200) -> CanopyProfile:
    """
    Ram-air / parafoil cross-section (NACA 0015 -like airfoil).
    """
    # NACA symmetric airfoil (0015 thickness distribution)
    t_max = 0.15  # 15% thickness
    x_c = np.linspace(0, 1, n_points // 2)

    y_t = 5 * t_max * (
        0.2969 * np.sqrt(x_c)
        - 0.1260 * x_c
        - 0.3516 * x_c**2
        + 0.2843 * x_c**3
        - 0.1015 * x_c**4
    )

    x_upper = x_c * chord
    y_upper = y_t * chord

    x_lower = x_upper[::-1]
    y_lower = -y_upper[::-1]

    x = np.concatenate([x_upper, x_lower]) - chord / 2
    y = np.concatenate([y_upper, y_lower])

    nx, ny = _compute_normals(x, y)

    return CanopyProfile(
        name="ram_air", x=x, y=y, nx=nx, ny=ny,
        porosity=np.zeros(len(x)),
        diameter=span,
        area_ref=chord * span,
        n_points=len(x),
    )


# ─── Utility ──────────────────────────────────────────────────────────────────

def _compute_normals(x: np.ndarray, y: np.ndarray) -> tuple:
    """Compute outward unit normals via central differences."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    # Outward normal: perpendicular to tangent, pointing away from interior
    length = np.sqrt(dx**2 + dy**2) + 1e-12
    nx = dy / length   # perpendicular
    ny = -dx / length
    return nx, ny


CANOPY_GENERATORS = {
    "flat_circular":  flat_circular,
    "hemisphere":     hemisphere,
    "extended_skirt": extended_skirt,
    "ribbon":         ribbon,
    "disk_gap_band":  disk_gap_band,
    "ram_air":        ram_air,
}


def generate(canopy_type: str = "hemisphere", **kwargs) -> CanopyProfile:
    """Generate a canopy profile by name."""
    fn = CANOPY_GENERATORS.get(canopy_type)
    if fn is None:
        raise ValueError(f"Unknown canopy type: '{canopy_type}'. "
                         f"Available: {list(CANOPY_GENERATORS.keys())}")
    return fn(**kwargs)
