"""
thermal_model.py — Aerothermal Protection Analysis (AeroDecel v6.0)
====================================================================
Computes aerodynamic heating, material temperature response, and
time-to-failure for parachute canopy fabric during high-speed deployment.

Physics:
  1. Stagnation-point convective heat flux (Sutton-Graves):
     q̇_conv = k_sg · √(ρ / R_n) · V³

  2. Radiative heat flux (Tauber-Sutton for velocities > 3 km/s):
     q̇_rad = C_r · ρ^a · R_n^b · f(V)

  3. 1D transient heat conduction through canopy fabric (explicit FD):
     ρ_fab · c_p · ∂T/∂t = κ · ∂²T/∂x² + q̇_surface - q̇_re-radiation

  4. Material degradation model:
     Strength(T) = σ₀ · [1 - (T/T_decomp)^n]  for T > T_onset
     t_fail = ∫ dt / t_char(T)  — Arrhenius cumulative damage

Fabric material database:
  - Nylon 6,6     (T_max = 180°C, standard sport canopy)
  - Kevlar 49     (T_max = 427°C, high-performance)
  - Nomex III     (T_max = 370°C, fire-resistant aramid)
  - Vectran HT    (T_max = 330°C, liquid crystal polymer)
  - Zylon PBO     (T_max = 650°C, highest performance fiber)
  - UHMWPE/Dyneema (T_max = 130°C, ultra-light)

Reference:
  - Sutton & Graves, "A general stagnation-point convective-heating
    equation for arbitrary gas mixtures", NASA TR R-376, 1971
  - Tauber & Sutton, "Prediction of convective heating on a blunt body",
    J. Spacecraft Rockets, 1991
  - Knacke, "Parachute Recovery Systems Design Manual", 1992
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FABRIC MATERIAL DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FabricMaterial:
    """Thermo-mechanical properties of parachute canopy fabric."""
    name:          str
    density:       float    # fabric density [kg/m³]
    specific_heat: float    # c_p [J/(kg·K)]
    conductivity:  float    # thermal conductivity κ [W/(m·K)]
    emissivity:    float    # surface emissivity ε (for re-radiation)
    thickness:     float    # fabric thickness [m]
    T_max_use:     float    # max continuous use temperature [K]
    T_decompose:   float    # decomposition / melting temperature [K]
    T_onset:       float    # onset of strength degradation [K]
    tensile_kNm:   float    # tensile strength [kN/m] (per unit width)
    description:   str = ""


NYLON_66 = FabricMaterial(
    name="Nylon 6,6", density=1140.0, specific_heat=1670.0,
    conductivity=0.25, emissivity=0.88, thickness=4.0e-5,
    T_max_use=453.15, T_decompose=533.15, T_onset=373.15,
    tensile_kNm=0.44, description="Standard sport / military canopy material",
)

KEVLAR_49 = FabricMaterial(
    name="Kevlar 49", density=1440.0, specific_heat=1420.0,
    conductivity=0.04, emissivity=0.92, thickness=3.0e-5,
    T_max_use=700.15, T_decompose=773.15, T_onset=523.15,
    tensile_kNm=3.0, description="High-performance para-aramid fiber",
)

NOMEX_III = FabricMaterial(
    name="Nomex III", density=1380.0, specific_heat=1260.0,
    conductivity=0.13, emissivity=0.90, thickness=5.0e-5,
    T_max_use=643.15, T_decompose=693.15, T_onset=473.15,
    tensile_kNm=0.58, description="Fire-resistant meta-aramid fiber",
)

VECTRAN_HT = FabricMaterial(
    name="Vectran HT", density=1400.0, specific_heat=1180.0,
    conductivity=0.37, emissivity=0.85, thickness=2.5e-5,
    T_max_use=603.15, T_decompose=643.15, T_onset=473.15,
    tensile_kNm=3.2, description="Liquid crystal polymer — Mars EDL suspension lines",
)

ZYLON_PBO = FabricMaterial(
    name="Zylon PBO", density=1560.0, specific_heat=1400.0,
    conductivity=0.30, emissivity=0.91, thickness=2.0e-5,
    T_max_use=923.15, T_decompose=1023.15, T_onset=623.15,
    tensile_kNm=5.8, description="Highest performance fiber. UV-sensitive.",
)

DYNEEMA = FabricMaterial(
    name="Dyneema (UHMWPE)", density=970.0, specific_heat=1800.0,
    conductivity=0.40, emissivity=0.78, thickness=3.5e-5,
    T_max_use=403.15, T_decompose=423.15, T_onset=343.15,
    tensile_kNm=3.5, description="Ultra-light, ultra-high MW polyethylene. Low T_max.",
)

MATERIALS = {
    "nylon": NYLON_66, "kevlar": KEVLAR_49, "nomex": NOMEX_III,
    "vectran": VECTRAN_HT, "zylon": ZYLON_PBO, "dyneema": DYNEEMA,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AERODYNAMIC HEATING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

STEFAN_BOLTZMANN = 5.670374419e-8   # W/(m²·K⁴)

# Sutton-Graves constant for different gas compositions [kg^0.5 / m]
SG_CONSTANTS = {
    "earth":  1.7415e-4,   # Air (N₂/O₂)
    "mars":   1.8981e-4,   # CO₂-dominant
    "venus":  1.8981e-4,   # CO₂-dominant (same as Mars approx)
    "titan":  1.5600e-4,   # N₂-dominant (similar to Earth but lower T)
}


def stagnation_heat_flux(velocity: float, density: float,
                         nose_radius: float, planet: str = "earth") -> float:
    """
    Sutton-Graves stagnation-point convective heat flux [W/m²].

    q̇ = k_sg · √(ρ∞ / R_n) · V³

    Parameters
    ----------
    velocity    : freestream velocity [m/s]
    density     : freestream density [kg/m³]
    nose_radius : effective nose radius [m] (canopy curvature)
    planet      : "earth", "mars", "venus", "titan"

    Returns
    -------
    Heat flux [W/m²]
    """
    k_sg = SG_CONSTANTS.get(planet, SG_CONSTANTS["earth"])
    rho  = max(density, 1e-12)
    Rn   = max(nose_radius, 0.01)
    V    = max(abs(velocity), 0.0)
    return k_sg * np.sqrt(rho / Rn) * V ** 3


def radiative_heat_flux(velocity: float, density: float,
                        nose_radius: float) -> float:
    """
    Tauber-Sutton radiative heat flux [W/m²].
    Significant only at hypersonic velocities (> ~3 km/s).

    q̇_rad ≈ C_r · ρ^1.22 · R_n^0.49 · V^8.5 × 1e-6
    (simplified Tauber-Sutton for air)
    """
    V = max(abs(velocity), 0.0)
    if V < 3000.0:
        return 0.0    # Negligible below 3 km/s
    rho = max(density, 1e-12)
    Rn  = max(nose_radius, 0.01)
    return 4.736e4 * rho ** 1.22 * Rn ** 0.49 * (V / 1e4) ** 8.5


def equilibrium_wall_temperature(q_total: float, emissivity: float) -> float:
    """
    Radiative equilibrium wall temperature [K].
    T_wall = (q̇_total / (ε · σ))^(1/4)
    """
    if q_total <= 0:
        return 200.0   # ambient
    return (q_total / (emissivity * STEFAN_BOLTZMANN)) ** 0.25


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 1D TRANSIENT HEAT CONDUCTION (through fabric thickness)
# ═══════════════════════════════════════════════════════════════════════════════

def fabric_temperature_history(
    q_flux_fn:   Callable[[float], float],  # q̇(t) [W/m²]
    material:    FabricMaterial,
    t_total:     float = 30.0,              # simulation time [s]
    dt:          float = 0.001,             # timestep [s]
    n_nodes:     int   = 10,                # nodes through fabric thickness
    T_init:      float = 250.0,             # initial fabric temperature [K]
    T_back:      float = 250.0,             # backside boundary temperature [K]
) -> dict:
    """
    Solve 1D transient heat conduction through the fabric thickness.

    Boundary conditions:
      Front surface: q̇_net = q̇_aero - ε·σ·T⁴     (heating minus re-radiation)
      Back surface:  T = T_back  (convective cooling / insulated)

    Returns dict with time series of:
      - Front surface temperature T_surface(t)
      - Peak temperature through thickness
      - Remaining strength fraction
      - Cumulative damage parameter
    """
    fab = material
    dx = fab.thickness / max(n_nodes - 1, 1)
    alpha = fab.conductivity / (fab.density * fab.specific_heat)

    # CFL stability check
    dt_max = 0.5 * dx**2 / max(alpha, 1e-15)
    dt = min(dt, dt_max * 0.9)

    n_steps = int(t_total / dt) + 1
    T = np.full(n_nodes, T_init)

    # Output storage (sampled every 10 steps)
    sample_every = max(1, n_steps // 2000)
    times       = []
    T_surface   = []
    T_peak      = []
    strength    = []
    damage      = []
    q_in_hist   = []

    cumulative_damage = 0.0

    for step in range(n_steps):
        t = step * dt

        # Heat flux at current time
        q_aero = q_flux_fn(t)
        q_rerad = fab.emissivity * STEFAN_BOLTZMANN * T[0]**4
        q_net = max(0.0, q_aero - q_rerad)

        # Interior nodes (explicit finite difference)
        T_new = T.copy()
        for i in range(1, n_nodes - 1):
            T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])

        # Front surface: Neumann BC — q_net = -κ · dT/dx
        T_new[0] = T_new[1] + q_net * dx / fab.conductivity

        # Back surface: Dirichlet (approximate convective cooling)
        T_new[-1] = T_back + 0.3 * (T_new[-2] - T_back)

        T = T_new

        # Material degradation (Arrhenius-type cumulative damage)
        T_max_local = T.max()
        if T_max_local > fab.T_onset:
            # Damage rate increases exponentially with temperature
            damage_rate = np.exp(
                -10000.0 / max(T_max_local, 100.0)
                + 10000.0 / fab.T_decompose
            )
            cumulative_damage += damage_rate * dt

        # Strength fraction
        if T_max_local < fab.T_onset:
            str_frac = 1.0
        elif T_max_local > fab.T_decompose:
            str_frac = 0.0
        else:
            frac = (T_max_local - fab.T_onset) / (fab.T_decompose - fab.T_onset)
            str_frac = max(0.0, 1.0 - frac ** 2)

        # Sample output
        if step % sample_every == 0:
            times.append(t)
            T_surface.append(float(T[0]))
            T_peak.append(float(T_max_local))
            strength.append(str_frac)
            damage.append(cumulative_damage)
            q_in_hist.append(q_aero)

    return {
        "time_s":            np.array(times),
        "T_surface_K":       np.array(T_surface),
        "T_peak_K":          np.array(T_peak),
        "strength_fraction": np.array(strength),
        "cumulative_damage":  np.array(damage),
        "q_aero_Wm2":       np.array(q_in_hist),
        "material":          fab.name,
        "thickness_m":       fab.thickness,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FULL THERMAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThermalResult:
    """Complete thermal analysis result."""
    q_peak_Wm2:         float
    T_peak_K:           float
    T_peak_C:           float
    T_max_material_K:   float
    thermal_margin_K:   float
    thermal_margin_pct: float
    strength_at_peak:   float
    time_to_Tmax_s:     float
    safe:               bool
    material:           str
    history:            dict

    def print_summary(self):
        W = 60
        flag = "✓ SAFE" if self.safe else "✗ THERMAL FAILURE"
        print(f"\n{'═'*W}")
        print(f"  THERMAL PROTECTION ANALYSIS — {flag}")
        print(f"{'═'*W}")
        print(f"  Material:        {self.material}")
        print(f"  Peak heat flux:  {self.q_peak_Wm2:.1f} W/m²  "
              f"({self.q_peak_Wm2/1e4:.2f} W/cm²)")
        print(f"  Peak fabric T:   {self.T_peak_K:.1f} K  "
              f"({self.T_peak_C:.1f}°C)")
        print(f"  Material limit:  {self.T_max_material_K:.1f} K  "
              f"({self.T_max_material_K - 273.15:.1f}°C)")
        print(f"  Thermal margin:  {self.thermal_margin_K:+.1f} K  "
              f"({self.thermal_margin_pct:+.1f}%)")
        print(f"  Strength at pk:  {self.strength_at_peak*100:.1f}%")
        print(f"  Time to T_max:   {self.time_to_Tmax_s:.3f} s")
        print(f"{'═'*W}")


def analyse_thermal(
    velocity_fn:    Callable[[float], float],   # v(t) [m/s]
    density_fn:     Callable[[float], float],   # ρ(t) [kg/m³]
    nose_radius:    float = 4.0,                # effective canopy curvature [m]
    material:       str   = "nylon",
    planet:         str   = "earth",
    t_total:        float = 30.0,
    verbose:        bool  = True,
) -> ThermalResult:
    """
    Run complete aerothermal analysis for a descent trajectory.

    Parameters
    ----------
    velocity_fn : callable returning velocity [m/s] at time t
    density_fn  : callable returning density [kg/m³] at time t
    nose_radius : effective canopy nose radius [m]
    material    : fabric material name (see MATERIALS dict)
    planet      : planet name for Sutton-Graves constant
    t_total     : total simulation time [s]
    verbose     : print summary

    Returns
    -------
    ThermalResult with full analysis
    """
    fab = MATERIALS.get(material, NYLON_66)

    def q_flux(t):
        v   = velocity_fn(t)
        rho = density_fn(t)
        q_c = stagnation_heat_flux(v, rho, nose_radius, planet)
        q_r = radiative_heat_flux(v, rho, nose_radius)
        return q_c + q_r

    # Get initial temperature from planet
    T_init = {"earth": 250.0, "mars": 200.0, "venus": 300.0, "titan": 94.0}
    T0 = T_init.get(planet, 250.0)

    hist = fabric_temperature_history(
        q_flux_fn=q_flux,
        material=fab,
        t_total=t_total,
        T_init=T0,
        T_back=T0,
    )

    T_peak = float(hist["T_peak_K"].max())
    q_peak = float(hist["q_aero_Wm2"].max())
    t_Tmax = float(hist["time_s"][np.argmax(hist["T_peak_K"])])

    idx_peak = np.argmax(hist["T_peak_K"])
    str_at_peak = float(hist["strength_fraction"][idx_peak])

    margin_K = fab.T_max_use - T_peak
    margin_pct = (margin_K / fab.T_max_use) * 100

    result = ThermalResult(
        q_peak_Wm2=q_peak,
        T_peak_K=T_peak,
        T_peak_C=T_peak - 273.15,
        T_max_material_K=fab.T_max_use,
        thermal_margin_K=margin_K,
        thermal_margin_pct=margin_pct,
        strength_at_peak=str_at_peak,
        time_to_Tmax_s=t_Tmax,
        safe=T_peak < fab.T_max_use,
        material=fab.name,
        history=hist,
    )

    if verbose:
        result.print_summary()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MATERIAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_materials(
    velocity_fn:  Callable[[float], float],
    density_fn:   Callable[[float], float],
    nose_radius:  float = 4.0,
    planet:       str   = "earth",
    t_total:      float = 30.0,
    verbose:      bool  = True,
) -> dict:
    """Run thermal analysis for ALL materials and compare."""
    results = {}

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  MATERIAL THERMAL COMPARISON — {planet.upper()}")
        print(f"{'═'*70}")
        print(f"  {'Material':<20} {'T_peak[K]':>10} {'T_max[K]':>10} "
              f"{'Margin[K]':>10} {'Strength':>10} {'Status':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for key, fab in MATERIALS.items():
        r = analyse_thermal(
            velocity_fn=velocity_fn,
            density_fn=density_fn,
            nose_radius=nose_radius,
            material=key,
            planet=planet,
            t_total=t_total,
            verbose=False,
        )
        results[key] = r
        if verbose:
            flag = "✓" if r.safe else "✗"
            print(f"  {flag} {fab.name:<18} {r.T_peak_K:>10.1f} "
                  f"{r.T_max_material_K:>10.1f} {r.thermal_margin_K:>+10.1f} "
                  f"{r.strength_at_peak*100:>9.1f}% "
                  f"{'SAFE' if r.safe else 'FAIL':>10}")

    if verbose:
        print(f"{'═'*70}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_thermal(result: ThermalResult, save_path=None):
    """Generate 4-panel thermal analysis dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import config as cfg
        DARK = cfg.DARK_THEME
    except Exception:
        DARK = True

    if DARK:
        plt.rcParams.update({
            "figure.facecolor": "#080c14", "axes.facecolor": "#0d1526",
            "axes.edgecolor": "#2a3d6e", "text.color": "#c8d8f0",
            "axes.labelcolor": "#c8d8f0", "xtick.color": "#c8d8f0",
            "ytick.color": "#c8d8f0", "grid.color": "#1a2744",
        })
    plt.rcParams.update({"font.family": "monospace", "font.size": 9})

    TEXT = "#c8d8f0" if DARK else "#111"
    h = result.history
    t = h["time_s"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#080c14" if DARK else "white")

    # P0: Heat flux
    ax = axes[0, 0]
    ax.fill_between(t, h["q_aero_Wm2"] / 1e4, alpha=0.2, color="#ff4560")
    ax.plot(t, h["q_aero_Wm2"] / 1e4, color="#ff4560", lw=2)
    ax.set_title("Aerodynamic Heat Flux", fontweight="bold")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("q̇ [W/cm²]")
    ax.grid(True, alpha=0.3)

    # P1: Surface temperature
    ax = axes[0, 1]
    ax.plot(t, h["T_surface_K"] - 273.15, color="#ff6b35", lw=2, label="T_surface")
    ax.plot(t, h["T_peak_K"] - 273.15, color="#ffd700", lw=1.5, ls="--", label="T_peak")
    ax.axhline(result.T_max_material_K - 273.15, color="#ff4560", lw=1, ls=":",
               label=f"T_max ({result.material})")
    ax.set_title("Fabric Temperature", fontweight="bold")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [°C]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # P2: Strength fraction
    ax = axes[1, 0]
    ax.fill_between(t, h["strength_fraction"] * 100, alpha=0.2, color="#a8ff3e")
    ax.plot(t, h["strength_fraction"] * 100, color="#a8ff3e", lw=2)
    ax.axhline(50, color="#ffd700", lw=0.8, ls="--", label="50% strength")
    ax.set_title("Remaining Fabric Strength", fontweight="bold")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Strength [%]")
    ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # P3: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    flag_color = "#a8ff3e" if result.safe else "#ff4560"
    flag_text = "SAFE ✓" if result.safe else "THERMAL FAILURE ✗"
    rows = [
        ("STATUS", flag_text, flag_color),
        ("Material", result.material, TEXT),
        ("", "", TEXT),
        ("q̇_peak", f"{result.q_peak_Wm2:.1f} W/m²", "#ff4560"),
        ("T_peak", f"{result.T_peak_C:.1f}°C ({result.T_peak_K:.1f} K)", "#ff6b35"),
        ("T_max material", f"{result.T_max_material_K - 273.15:.1f}°C", TEXT),
        ("Thermal margin", f"{result.thermal_margin_K:+.1f} K ({result.thermal_margin_pct:+.1f}%)",
         "#a8ff3e" if result.safe else "#ff4560"),
        ("Strength at peak", f"{result.strength_at_peak * 100:.1f}%", "#a8ff3e"),
        ("Time to T_max", f"{result.time_to_Tmax_s:.3f} s", TEXT),
    ]
    for j, (label, val, color) in enumerate(rows):
        y = 0.95 - j * 0.095
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=10, color=TEXT)
        ax.text(0.95, y, val, transform=ax.transAxes, fontsize=10,
                ha="right", color=color, fontweight="bold" if j == 0 else "normal")
    ax.set_title("Thermal Summary", fontweight="bold")

    fig.suptitle(f"Aerothermal Analysis — {result.material}  |  "
                 f"T_peak={result.T_peak_C:.1f}°C  q̇_peak={result.q_peak_Wm2:.0f} W/m²",
                 fontsize=12, fontweight="bold", color=TEXT)

    plt.tight_layout()

    if save_path is None:
        try:
            import config as cfg
            save_path = cfg.OUTPUTS_DIR / "thermal_analysis.png"
        except Exception:
            save_path = Path("thermal_analysis.png")

    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Thermal dashboard saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick demo: Mars EDL thermal analysis
    from src.planetary_atm import MarsAtmosphere

    def v_mars(t):
        """Simulated Mars EDL velocity decay from 400 m/s."""
        return max(5.0, 400.0 * np.exp(-0.15 * t))

    def rho_mars(t):
        """Density at ~10 km MOLA descending."""
        h = max(0, 10000 - 50 * t)
        return MarsAtmosphere.density(h)

    print("Mars EDL Thermal Analysis:")
    result = analyse_thermal(
        velocity_fn=v_mars,
        density_fn=rho_mars,
        nose_radius=5.0,
        material="kevlar",
        planet="mars",
        t_total=60.0,
    )

    print("\nMaterial Comparison:")
    compare_materials(v_mars, rho_mars, nose_radius=5.0, planet="mars", t_total=60.0)
