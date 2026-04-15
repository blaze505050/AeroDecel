"""
src/thermal_model.py — Thermal Protection System (TPS) Model
=============================================================
Implements:
  • Sutton-Graves convective heating (stagnation point)
  • Fay-Riddell radiative heating (high-velocity re-entry)
  • 1-D transient finite-difference heat conduction through TPS layering
  • Material database with temperature / ablation limits
  • Safety margin reporting

Reference
---------
  Sutton & Graves (1971), "A General Stagnation-Point Convective Heating
  Equation for Arbitrary Gas Mixtures", NASA TR R-376.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# MATERIAL DATABASE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialProperties:
    name:                 str
    density_kgm3:         float   # kg/m³
    specific_heat_JkgK:   float   # J/(kg·K)
    conductivity_WmK:     float   # W/(m·K)
    max_temperature_K:    float   # structural / melting limit
    emissivity:           float   # 0–1
    ablation_enthalpy:    float   # J/kg  (0 = non-ablative)

    @property
    def diffusivity(self) -> float:
        """Thermal diffusivity α = k / (ρ·cp)  [m²/s]."""
        return self.conductivity_WmK / (self.density_kgm3 * self.specific_heat_JkgK)


MATERIAL_DB: dict[str, MaterialProperties] = {
    "nylon":   MaterialProperties("Nylon",   1150, 1700, 0.25, 450,  0.92, 0),
    "kevlar":  MaterialProperties("Kevlar",  1440, 1420, 0.04, 800,  0.90, 0),
    "nomex":   MaterialProperties("Nomex",   1100, 1200, 0.13, 700,  0.88, 0),
    "vectran": MaterialProperties("Vectran", 1400, 1000, 0.05, 600,  0.85, 0),
    "zylon":   MaterialProperties("Zylon",   1540, 1100, 0.06, 900,  0.87, 0),
    # Ablative heat-shields
    "pica":    MaterialProperties("PICA",     220, 1300, 0.14, 3000, 0.80, 56_000_000),
    "avcoat":  MaterialProperties("AVCOAT",   520, 1240, 0.40, 3500, 0.82, 38_000_000),
    "srp":     MaterialProperties("SRP",      224, 1000, 0.16, 2800, 0.80, 50_000_000),
}


# ══════════════════════════════════════════════════════════════════════════════
# TPS SOLVER
# ══════════════════════════════════════════════════════════════════════════════

class ThermalProtectionSystem:
    """
    One-dimensional transient heat conduction through a TPS slab.

    State
    -----
    temperature_profile : ndarray shape (n_time, n_x)
    """

    STEFAN_BOLTZMANN = 5.670374419e-8   # W/(m²·K⁴)

    def __init__(self, material: str, thickness_m: float, n_nodes: int = 20):
        key = material.lower()
        if key not in MATERIAL_DB:
            raise ValueError(f"Unknown material '{material}'. "
                             f"Available: {sorted(MATERIAL_DB.keys())}")
        self.mat              = MATERIAL_DB[key]
        self.thickness        = thickness_m
        self.n_nodes          = n_nodes
        self.dx               = thickness_m / n_nodes
        self.temperature_profile: np.ndarray | None = None

    # ── Heating models ────────────────────────────────────────────────────────

    @staticmethod
    def sutton_graves_heating(density_kgm3: float, velocity_ms: float,
                              nose_radius_m: float,
                              k_sg: float = 1.83e-4) -> float:
        """
        Stagnation-point convective heat flux [W/m²].

        q_conv = k_sg · sqrt(ρ / R_n) · v³

        Parameters
        ----------
        k_sg : Sutton-Graves constant (default 1.83e-4 for Earth/Mars CO₂ air)
               Use 1.74e-4 for CO₂ atmospheres (Mars/Venus) — slightly lower.
        """
        return float(k_sg * np.sqrt(max(density_kgm3, 1e-12) / max(nose_radius_m, 0.01))
                     * abs(velocity_ms) ** 3)

    @staticmethod
    def radiative_heating(density_kgm3: float, velocity_ms: float,
                          nose_radius_m: float) -> float:
        """
        Simplified stagnation-point radiative heat flux [W/m²].
        Tauber & Sutton (1991) correlation.
        q_rad ≈ C · ρ^a · v^b · R_n^c
        Only significant at v > 8 km/s.
        """
        if velocity_ms < 6000:
            return 0.0
        C = 4.736e4
        return float(C * (density_kgm3 ** 1.22) * ((velocity_ms / 1e4) ** 8.5)
                     * nose_radius_m ** 0.5)

    def surface_reradiation(self, surface_temp_K: float) -> float:
        """Net radiative re-emission from hot TPS surface [W/m²]."""
        return float(self.mat.emissivity * self.STEFAN_BOLTZMANN * surface_temp_K ** 4)

    # ── Heat conduction solver ────────────────────────────────────────────────

    def solve_1d_conduction(self, heat_flux_Wm2: float | np.ndarray,
                             time_steps: np.ndarray,
                             T_initial_K: float = 300.0,
                             T_ambient_K: float = 300.0) -> np.ndarray:
        """
        Explicit finite-difference 1-D transient heat conduction.

        Boundary conditions
        -------------------
        x = 0  (outer face) : Neumann  — applied heat flux (net of re-radiation)
        x = L  (inner face) : Dirichlet — ambient temperature (structure)

        Stability: dt is auto-limited to 0.4·dx²/α (Fourier criterion).

        Returns
        -------
        T : ndarray (n_time, n_nodes)
        """
        nt  = len(time_steps)
        nx  = self.n_nodes
        dx  = self.dx
        α   = self.mat.diffusivity
        k   = self.mat.conductivity_WmK

        # Ensure dt satisfies stability: Fo = α·dt/dx² ≤ 0.5
        if nt > 1:
            dt_input = time_steps[1] - time_steps[0]
        else:
            dt_input = 1.0
        dt = min(dt_input, 0.4 * dx ** 2 / max(α, 1e-12))
        Fo = α * dt / dx ** 2      # Fourier number (≤ 0.5)

        # Initialise
        T = np.full((nt, nx), T_initial_K)

        # Allow scalar or time-varying heat flux
        if np.isscalar(heat_flux_Wm2):
            q_arr = np.full(nt, float(heat_flux_Wm2))
        else:
            q_arr = np.asarray(heat_flux_Wm2, dtype=float)
            if len(q_arr) < nt:
                q_arr = np.interp(np.linspace(0, 1, nt),
                                  np.linspace(0, 1, len(q_arr)), q_arr)

        for n in range(nt - 1):
            # Interior nodes
            T[n+1, 1:-1] = T[n, 1:-1] + Fo * (T[n, 2:] - 2*T[n, 1:-1] + T[n, :-2])

            # Net heat flux at outer surface (minus re-radiation)
            q_net = q_arr[n] - self.surface_reradiation(T[n, 0])
            T[n+1, 0]  = T[n+1, 1] + dx * q_net / k   # Neumann BC

            # Inner face: fixed ambient (structure backing)
            T[n+1, -1] = T_ambient_K

            # Physical floor
            T[n+1] = np.maximum(T[n+1], T_ambient_K)

        self.temperature_profile = T
        return T

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def check_material_limit(self) -> tuple[bool, float]:
        """Return (limit_exceeded, peak_temperature_K)."""
        if self.temperature_profile is None:
            return False, 0.0
        peak = float(self.temperature_profile.max())
        return peak > self.mat.max_temperature_K, peak

    def safety_margin(self) -> float:
        """SF = T_max_allowed / T_peak.  SF < 1 → failure."""
        _, peak = self.check_material_limit()
        if peak <= 0:
            return float("inf")
        return self.mat.max_temperature_K / peak

    def time_to_limit(self, time_steps: np.ndarray) -> float | None:
        """Return elapsed time [s] when surface first exceeds T_max, or None."""
        if self.temperature_profile is None:
            return None
        surface_T = self.temperature_profile[:, 0]
        idx = np.argmax(surface_T > self.mat.max_temperature_K)
        if idx == 0 and surface_T[0] <= self.mat.max_temperature_K:
            return None
        return float(time_steps[idx])

    def summary(self) -> dict:
        exceeded, peak = self.check_material_limit()
        return {
            "material":         self.mat.name,
            "thickness_m":      self.thickness,
            "T_max_allowed_K":  self.mat.max_temperature_K,
            "T_peak_K":         round(peak, 2),
            "safety_factor":    round(self.safety_margin(), 3),
            "limit_exceeded":   exceeded,
        }


if __name__ == "__main__":
    tps = ThermalProtectionSystem("zylon", 0.015)
    # Mars EDL scenario
    q = tps.sutton_graves_heating(density_kgm3=0.005, velocity_ms=4000,
                                   nose_radius_m=1.0)
    print(f"Heat flux: {q/1e6:.3f} MW/m²")
    t = np.linspace(0, 120, 200)
    tps.solve_1d_conduction(q, t)
    s = tps.summary()
    for k, v in s.items():
        print(f"  {k:25s}: {v}")
