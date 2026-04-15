"""
src/canopy_geometry.py — Parametric Canopy Geometry
====================================================
Provides cross-section generation, area/perimeter calculations,
and drag-coefficient models for elliptical, circular, rectangular,
disk-gap-band, and tricone canopy shapes.

Drag correlations
-----------------
  Subsonic (M < 0.3)  : empirical shape factor
  Transonic (0.3–1.2) : Prandtl-Glauert + wave drag onset
  Supersonic (>1.2)   : supersonic drag rise
"""
from __future__ import annotations
import numpy as np


class CanopyGeometry:

    SUPPORTED = ("elliptical", "circular", "rectangular",
                 "disk_gap_band", "tricone")

    def __init__(self, shape: str, dimensions: dict):
        shape = shape.lower()
        if shape not in self.SUPPORTED:
            raise ValueError(f"Unknown canopy shape '{shape}'. "
                             f"Supported: {self.SUPPORTED}")
        self.shape      = shape
        self.dimensions = dimensions

    # ── Cross-section outline ─────────────────────────────────────────────────

    def generate_cross_section(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (x, y) arrays of the canopy outline in canopy-plane coordinates.
        All units in metres.
        """
        d = self.dimensions
        if self.shape == "circular":
            r = d.get("r", d.get("radius", 5.0))
            θ = np.linspace(0, 2*np.pi, n)
            return r * np.cos(θ), r * np.sin(θ)

        if self.shape == "elliptical":
            a, b = d["a"], d["b"]
            θ = np.linspace(0, 2*np.pi, n)
            return a * np.cos(θ), b * np.sin(θ)

        if self.shape == "rectangular":
            w, h = d["width"], d["height"]
            x = np.array([-w/2, w/2,  w/2, -w/2, -w/2])
            y = np.array([-h/2, -h/2, h/2,  h/2, -h/2])
            return x, y

        if self.shape == "disk_gap_band":
            # Disk + small gore gap + band — approximate as annular disk
            r_disk = d.get("r_disk", 5.0)
            gap    = d.get("gap", 0.15) * r_disk
            r_band = d.get("r_band", r_disk * 1.20)
            θ = np.linspace(0, 2*np.pi, n//2)
            # Outer band
            xo = r_band * np.cos(θ); yo = r_band * np.sin(θ)
            # Inner disk edge
            xi = (r_disk - gap) * np.cos(θ[::-1]); yi = (r_disk - gap) * np.sin(θ[::-1])
            return np.concatenate([xo, xi, [xo[0]]]), np.concatenate([yo, yi, [yo[0]]])

        if self.shape == "tricone":
            # Three coaxial cones joined — approximate cross-section
            r_base = d.get("r_base", 5.0)
            θ = np.linspace(0, 2*np.pi, n)
            # Sinusoidal ripple to hint at tricone shape
            r_θ = r_base * (1 + 0.15 * np.cos(3 * θ))
            return r_θ * np.cos(θ), r_θ * np.sin(θ)

        raise ValueError(self.shape)

    # ── Area and perimeter ────────────────────────────────────────────────────

    def calculate_area(self) -> float:
        """Reference cross-sectional area [m²]."""
        d = self.dimensions
        if self.shape == "circular":
            r = d.get("r", d.get("radius", 5.0))
            return np.pi * r**2
        if self.shape == "elliptical":
            return np.pi * d["a"] * d["b"]
        if self.shape == "rectangular":
            return d["width"] * d["height"]
        if self.shape == "disk_gap_band":
            r_band = d.get("r_band", d.get("r_disk", 5.0) * 1.20)
            return np.pi * r_band**2
        if self.shape == "tricone":
            return np.pi * d.get("r_base", 5.0)**2 * 0.95
        return 0.0

    def nominal_diameter(self) -> float:
        """Effective diameter D₀ = sqrt(4A/π) [m]."""
        return float(np.sqrt(4 * self.calculate_area() / np.pi))

    # ── Drag coefficient model ────────────────────────────────────────────────

    def _base_cd(self) -> float:
        """Shape-dependent base drag coefficient at M≈0."""
        return {
            "circular":      1.50,
            "elliptical":    1.35,
            "rectangular":   1.40,
            "disk_gap_band": 0.55,   # DGB canopies have lower Cd
            "tricone":       0.80,
        }[self.shape]

    def _shape_factor(self) -> float:
        d = self.dimensions
        if self.shape == "elliptical":
            ar = max(d["a"], d["b"]) / max(min(d["a"], d["b"]), 0.01)
            return max(0.70, 1.0 - 0.08 * (ar - 1))
        if self.shape == "rectangular":
            ar = max(d["width"], d["height"]) / max(min(d["width"], d["height"]), 0.01)
            return max(0.70, 0.95 - 0.05 * (ar - 1))
        return 1.0

    def calculate_drag_coefficient(self, mach_number: float) -> float:
        """
        Cd(M) using Prandtl-Glauert compressibility + wave drag model.
        """
        M  = max(0.0, mach_number)
        Cd0 = self._base_cd() * self._shape_factor()

        if M < 0.3:
            return float(Cd0)
        if M < 0.8:
            # Prandtl-Glauert
            pg = 1.0 / np.sqrt(max(1.0 - M**2, 0.01))
            return float(Cd0 * (1 + 0.35 * (pg - 1)))
        if M < 1.2:
            # Transonic peak (30 % rise at M=1)
            peak = Cd0 * 1.30
            t = (M - 0.8) / 0.4
            return float(Cd0 + (peak - Cd0) * np.sin(np.pi * t / 2))
        # Supersonic decay
        return float(Cd0 * 1.10 / (M ** 0.5))

    def drag_force(self, velocity_ms: float, density_kgm3: float,
                   mach_number: float | None = None) -> float:
        """
        F_drag = 0.5 · ρ · v² · Cd · A  [N]
        """
        if mach_number is None:
            mach_number = velocity_ms / 340.0
        Cd = self.calculate_drag_coefficient(mach_number)
        return 0.5 * density_kgm3 * velocity_ms**2 * Cd * self.calculate_area()

    def summary(self) -> dict:
        return {
            "shape":            self.shape,
            "area_m2":          round(self.calculate_area(), 4),
            "nominal_diam_m":   round(self.nominal_diameter(), 3),
            "Cd_subsonic":      round(self.calculate_drag_coefficient(0.1), 4),
            "Cd_mach1":         round(self.calculate_drag_coefficient(1.0), 4),
            "Cd_mach2":         round(self.calculate_drag_coefficient(2.0), 4),
        }


if __name__ == "__main__":
    for shape, dims in [
        ("circular",      {"r": 5}),
        ("elliptical",    {"a": 10, "b": 5}),
        ("rectangular",   {"width": 8, "height": 6}),
        ("disk_gap_band", {"r_disk": 5, "r_band": 6}),
        ("tricone",       {"r_base": 4}),
    ]:
        cg = CanopyGeometry(shape, dims)
        s  = cg.summary()
        print(f"{shape:16s}  A={s['area_m2']:7.2f}m²  "
              f"D={s['nominal_diam_m']:.2f}m  "
              f"Cd(M0.1)={s['Cd_subsonic']:.3f}  "
              f"Cd(M1)={s['Cd_mach1']:.3f}")
