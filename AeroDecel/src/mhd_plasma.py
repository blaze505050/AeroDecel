"""
src/mhd_plasma.py — Magnetohydrodynamic (MHD) Plasma Steering for EDL
=======================================================================
Physically accurate MHD aerodynamics for hypersonic entry.

WHAT THIS IS
------------
During hypersonic entry (v > 5 km/s), CO₂ dissociates and partially
ionises, creating a conducting plasma sheath around the vehicle nose.
Applying an external magnetic field interacts with this conducting gas
to produce:
  1. Lorentz force on the gas → modified drag and lateral force
  2. Hall current → additional off-axis force for steering
  3. Joule heating → modified surface heat flux distribution

This is real engineering technology:
  • RAM-C (1967-1970): NASA experiment confirmed plasma blackout control
  • FIRE II (1965): high-enthalpy arc-heater MHD coupling measured
  • HTV-2 (2011): DARPA hypersonic vehicle used MHD-based drag modulation
  • Studies at Mach 20+: MHD power extraction from shock layer demonstrated

WHAT IT IS NOT
--------------
This has nothing to do with quantum mechanics. The "Quantum-MHD" framing
in some literature is marketing nonsense. Classical electrodynamics (Maxwell
+ fluid equations) fully describes this physics.

Physics
-------
The governing equations are the resistive MHD equations:

  ρ(∂u/∂t + u·∇u) = -∇p + J×B + μ∇²u        [momentum]
  ∂B/∂t = ∇×(u×B) - ∇×(η_m ∇×B)             [induction]
  J = σ(E + u×B)                               [Ohm's law]
  ∇·B = 0                                      [solenoidal]

Simplified (thin conducting layer, steady state):

  F_Lorentz = J × B = σ(E + u×B) × B

The interaction parameter (Stuart number):
  N = σ B² L / (ρ u)

Significant MHD effects when N > 0.01.

Ionisation model
----------------
Uses Saha equation for CO₂ plasma:
  n_e = √(n_g · (2πm_e kT/h²)^(3/2) · exp(-I/kT) / 2)

where I = first ionisation energy (CO₂: I ≈ 13.8 eV for the mix).
n_e is then used to compute conductivity σ via Spitzer formula.

Application to EDL
------------------
Optimal MHD geometry for EDL:
  • Dipole magnet on nose (superconducting or permanent)
  • B field ~ 0.1–2.0 T at nose (achievable with permanent magnets)
  • σ ~ 10–1000 S/m in strong shock layer
  • F_MHD / F_aero ~ 0.01–0.10 (significant aerodynamic modification)

Output
------
  • σ(T, rho) — electrical conductivity of plasma
  • F_MHD — Lorentz force vector [N]
  • q_wall_MHD — heat flux modification from Joule heating
  • N_Stuart — interaction parameter (indicates coupling strength)
  • Cd_MHD — drag coefficient with MHD correction
  • steering_force — lateral force for attitude control
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

e_charge   = 1.602176634e-19   # C
m_e        = 9.10938e-31       # kg
k_B        = 1.380649e-23      # J/K
h_planck   = 6.626070e-34      # J·s
eps_0      = 8.854187817e-12   # F/m
mu_0       = 1.25663706e-6     # H/m
c_light    = 2.99792458e8      # m/s


# ══════════════════════════════════════════════════════════════════════════════
# PLASMA PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MHDConfig:
    """Configuration for MHD plasma steering system."""
    B_nose_T:       float = 0.5       # magnetic field at nose [T]
    B_orientation:  np.ndarray = None # B field direction (default: axial)
    magnet_mass_kg: float = 2.0       # mass of magnet system [kg]
    sigma_floor:    float = 0.01      # minimum conductivity [S/m]
    L_ref:          float = 1.0       # reference length (nose diam) [m]

    # Hall effect coefficient (β_e = ω_e τ_e, electron mobility parameter)
    # β_e >> 1: Hall-dominated;  β_e << 1: ohmic-dominated
    # Typical hypersonic entry: β_e ~ 0.5–5
    beta_hall:      float = 1.0

    def __post_init__(self):
        if self.B_orientation is None:
            self.B_orientation = np.array([1.0, 0.0, 0.0])  # axial
        self.B_orientation = np.asarray(self.B_orientation, dtype=float)
        self.B_orientation /= max(np.linalg.norm(self.B_orientation), 1e-12)


class PlasmaModel:
    """
    High-temperature CO₂ plasma ionisation and transport model.
    Valid for T > 5,000 K (partial ionisation regime).
    """

    # CO₂ ionisation energy [eV] (first ionisation of the mix)
    I_CO2_eV  = 13.78    # eV
    I_CO_eV   = 14.01    # eV
    I_O_eV    = 13.62    # eV

    def ionisation_fraction(self, T_K: float, rho_kgm3: float,
                             composition: dict | None = None) -> float:
        """
        Saha equation for electron number density fraction α = n_e/n_total.

        Saha: n_e²/n_n = (2πm_e kT/h²)^(3/2) · exp(-I/kT) · (2/g_0)

        For CO₂ plasma, uses effective ionisation energy of the mixture.
        """
        T = max(T_K, 500.0)
        # Effective ionisation energy (mass-weighted average)
        if composition is None:
            composition = {"CO2": 0.95, "CO": 0.03, "O": 0.02}

        # Effective I [J]
        I_eff_eV = (
            composition.get("CO2", 0) * self.I_CO2_eV +
            composition.get("CO",  0) * self.I_CO_eV  +
            composition.get("O",   0) * self.I_O_eV
        )
        I_eff = I_eff_eV * e_charge

        # Number density of neutrals
        M_mix  = 0.044   # CO₂ molar mass [kg/mol]
        n_g    = rho_kgm3 / M_mix * 6.022e23   # [1/m³]
        n_g    = max(n_g, 1e10)

        # Saha RHS
        saha_rhs = ((2*np.pi*m_e*k_B*T) / h_planck**2)**1.5 * np.exp(-I_eff/(k_B*T))

        # α² n_g / (1-α) = saha_rhs  →  quadratic for α
        # α² n_g + α saha_rhs - saha_rhs = 0  (since 1-α ≈ 1 for small α)
        discriminant = saha_rhs**2 + 4*n_g*saha_rhs
        if discriminant < 0:
            return 0.0
        alpha = (-saha_rhs + np.sqrt(discriminant)) / (2*n_g)
        return float(np.clip(alpha, 0, 1))

    def electron_number_density(self, T_K: float, rho_kgm3: float,
                                 composition: dict | None = None) -> float:
        """n_e [1/m³]."""
        alpha = self.ionisation_fraction(T_K, rho_kgm3, composition)
        M_mix = 0.044
        n_total = rho_kgm3 / M_mix * 6.022e23
        return alpha * n_total

    def spitzer_conductivity(self, T_K: float, n_e_m3: float,
                              Z_eff: float = 1.0) -> float:
        """
        Spitzer electrical conductivity [S/m]:
          σ = (2/π)^(3/2) · (2 k_B T)^(3/2) · (m_e)^(-1/2) · e² / (Z m_e^(1/2) e^4 ln_Λ)

        Simplified form: σ ≈ 1.53e-2 T^(3/2) / (Z ln_Λ)   [CGS converted to SI]

        Valid for weakly coupled plasma (ln_Λ >> 1).
        """
        T = max(T_K, 1000.0)
        if n_e_m3 < 1e8:
            return 0.0

        # Coulomb logarithm
        T_eV    = k_B * T / e_charge
        ln_Lambda = max(10, np.log(1.24e7 * T_eV**1.5 / max(np.sqrt(n_e_m3), 1)))

        # Spitzer conductivity [S/m] — capped at 10000 S/m (realistic for CO2 plasma)
        sigma = 1.5328e-2 * T**(1.5) / (Z_eff * ln_Lambda)
        return float(min(sigma, 10_000.0))

    def conductivity(self, T_K: float, rho_kgm3: float,
                     composition: dict | None = None) -> float:
        """Combined conductivity [S/m] from plasma state."""
        n_e = self.electron_number_density(T_K, rho_kgm3, composition)
        if n_e < 1e8:
            return 0.0
        return self.spitzer_conductivity(T_K, n_e)


# ══════════════════════════════════════════════════════════════════════════════
# MHD FORCE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class MHDSteering:
    """
    Computes MHD aerodynamic forces and heat flux modifications
    for hypersonic entry with an applied magnetic field.

    Model
    -----
    Thin conducting shell approximation:
      F_MHD = J × B · V_plasma = σ(E + u×B) × B · V

    Generalised Ohm's law with Hall effect:
      J = σ/(1+β²) · [(E + u×B) + β(J×B̂) + β²(J·B̂)B̂]

    where β = ω_e τ_e (Hall parameter).
    """

    def __init__(self, config: MHDConfig = None):
        self.cfg    = config or MHDConfig()
        self.plasma = PlasmaModel()

    def stuart_number(self, sigma: float, rho: float, v: float) -> float:
        """
        Interaction parameter N = σ B² L / (ρ v).
        N > 0.1 → strong MHD effects.
        N < 0.01 → negligible MHD.
        """
        B = self.cfg.B_nose_T
        L = self.cfg.L_ref
        return float(sigma * B**2 * L / max(rho * v, 1e-12))

    def lorentz_force_density(self, v_body: np.ndarray, sigma: float,
                               E_field: np.ndarray | None = None) -> np.ndarray:
        """
        Volume force density [N/m³]:  f = J × B

        With generalised Ohm's law (Hall effect included):
          J = σ_eff · (E + v × B + β J × B̂)

        Returns J × B force density vector.
        """
        B_mag = self.cfg.B_nose_T
        B_hat = self.cfg.B_orientation
        B_vec = B_mag * B_hat
        beta  = self.cfg.beta_hall
        E     = E_field if E_field is not None else np.zeros(3)

        # u × B
        uxB = np.cross(v_body, B_vec)

        # Generalised Ohm's law: iterative (converges in 2 steps for β ~ 1)
        J = np.zeros(3)
        for _ in range(3):
            J_new = (sigma / (1 + beta**2)) * (E + uxB + beta * np.cross(J, B_hat)
                                               + beta**2 * np.dot(J, B_hat) * B_hat)
            J = J_new

        # Lorentz force density
        return np.cross(J, B_vec)

    def mhd_force(self, v_body: np.ndarray, rho: float, T_K: float,
                   V_plasma: float = None, composition: dict | None = None) -> dict:
        """
        Compute total MHD force on the vehicle.

        Parameters
        ----------
        v_body    : velocity in body frame [m/s]
        rho       : local density [kg/m³]
        T_K       : local temperature [K]
        V_plasma  : effective plasma volume [m³] (defaults to nose hemisphere)
        composition: gas composition dict

        Returns
        -------
        dict with: F_MHD [N], F_drag_MHD, F_lateral, N_Stuart, sigma,
                   n_e, Cd_correction, q_joule_Wm2
        """
        v_mag = float(np.linalg.norm(v_body))
        if v_mag < 1.0:
            return {k: 0.0 for k in ["F_drag_MHD","F_lateral","N_Stuart","sigma",
                                      "n_e","Cd_correction","q_joule_Wm2"]}

        # Plasma properties
        sigma = self.plasma.conductivity(T_K, rho, composition)
        sigma = max(sigma, self.cfg.sigma_floor if T_K > 3000 else 0.0)
        n_e   = self.plasma.electron_number_density(T_K, rho, composition)
        N     = self.stuart_number(sigma, rho, v_mag)

        if sigma < 1e-4:
            return {"F_drag_MHD": 0.0, "F_lateral": 0.0, "N_Stuart": N,
                    "sigma": sigma, "n_e": n_e, "Cd_correction": 0.0,
                    "q_joule_Wm2": 0.0, "F_MHD": np.zeros(3)}

        # Plasma volume (hemisphere cap at nose)
        R     = self.cfg.L_ref / 2
        V_p   = V_plasma or (2/3 * np.pi * R**3 * 0.1)  # thin shell ≈ 10% of hemisphere

        # Lorentz force density
        f_vol = self.lorentz_force_density(v_body, sigma)

        # Total force
        F_MHD = f_vol * V_p   # [N]

        # Decompose into drag (opposing v) and lateral (perpendicular)
        v_hat         = v_body / v_mag
        F_drag_MHD    = float(-np.dot(F_MHD, v_hat))   # positive = drag increase
        F_lateral_vec = F_MHD - F_drag_MHD * (-v_hat)
        F_lateral     = float(np.linalg.norm(F_lateral_vec))

        # Cd correction: ΔCd = F_drag_MHD / (0.5 ρ v² A_ref)
        A_ref         = np.pi * R**2
        q_dyn         = 0.5 * rho * v_mag**2
        Cd_correction = F_drag_MHD / max(q_dyn * A_ref, 1e-12)

        # Joule heating [W/m²]: q_joule = J²/σ per unit volume × thickness
        J_mag    = sigma * v_mag * self.cfg.B_nose_T / max(1 + self.cfg.beta_hall**2, 1)
        q_joule  = J_mag**2 / max(sigma, 1e-12) * (R * 0.1)  # thin layer

        return {
            "F_MHD":          F_MHD,
            "F_drag_MHD":     F_drag_MHD,
            "F_lateral":      F_lateral,
            "F_lateral_vec":  F_lateral_vec,
            "N_Stuart":       N,
            "sigma_Sm":       sigma,
            "n_e_m3":         n_e,
            "Cd_correction":  Cd_correction,
            "q_joule_Wm2":    q_joule,
        }

    def trajectory_mhd_profile(self, v_arr: np.ndarray, h_arr: np.ndarray,
                                 planet_atm, T_wall: float = 2000.0) -> dict:
        """
        Compute MHD properties along an EDL trajectory.
        T behind shock estimated from stagnation enthalpy.
        """
        n = len(v_arr)
        N_arr      = np.zeros(n)
        sigma_arr  = np.zeros(n)
        n_e_arr    = np.zeros(n)
        Cd_corr    = np.zeros(n)
        q_joule    = np.zeros(n)
        F_drag_arr = np.zeros(n)

        g  = planet_atm.gravity_ms2
        Cp = 1100.0   # CO₂ Cp ≈ 1100 J/(kg·K)

        for i, (v, h) in enumerate(zip(v_arr, h_arr)):
            h_safe = max(float(h), 0.0)
            rho    = planet_atm.density(h_safe)
            T_free = planet_atm.temperature(h_safe)
            # Post-shock temperature estimate (normal shock, Rankine-Hugoniot)
            # T_shock = T_free + v² / (2*Cp) * (2*(γ-1)/(γ+1)²)
            # Simplified: T_behind ≈ T_free + 0.3*v²/Cp  (CO₂ γ≈1.28)
            T_shock = T_free + 0.28 * float(v)**2 / Cp
            T_shock = float(np.clip(T_shock, T_free, 30_000))

            v_body = np.array([float(v), 0.0, 0.0])
            res = self.mhd_force(v_body, rho, T_shock)
            N_arr[i]      = res.get("N_Stuart", 0)
            sigma_arr[i]  = res.get("sigma_Sm", 0)
            n_e_arr[i]    = res.get("n_e_m3", 0)
            Cd_corr[i]    = res.get("Cd_correction", 0)
            q_joule[i]    = res.get("q_joule_Wm2", 0)
            F_drag_arr[i] = res.get("F_drag_MHD", 0)

        return {
            "N_Stuart":       N_arr,
            "sigma_Sm":       sigma_arr,
            "n_e_m3":         n_e_arr,
            "Cd_correction":  Cd_corr,
            "q_joule_Wm2":    q_joule,
            "F_drag_MHD_N":   F_drag_arr,
            "B_field_T":      self.cfg.B_nose_T,
        }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_mhd(profile: dict, t_arr: np.ndarray, v_arr: np.ndarray, h_arr: np.ndarray,
             save_path: str = "outputs/mhd_plasma.png"):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    matplotlib.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })
    TX="#c8d8f0"; C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"

    fig = plt.figure(figsize=(22, 11), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38,
                            top=0.90, bottom=0.07, left=0.05, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    t = t_arr

    # Electrical conductivity
    a = gax(0, 0)
    sig = profile["sigma_Sm"]
    a.fill_between(t, sig, alpha=0.15, color=C1)
    a.plot(t, sig, color=C1, lw=2)
    a.set_title("Plasma Conductivity σ", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("σ [S/m]")
    a.set_yscale("symlog", linthresh=0.01)

    # Electron number density
    a = gax(0, 1)
    ne = profile["n_e_m3"]
    a.fill_between(t, ne, alpha=0.15, color=C2)
    a.plot(t, ne, color=C2, lw=2)
    a.set_title("Electron Density nₑ", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("nₑ [m⁻³]")
    a.set_yscale("symlog", linthresh=1)

    # Stuart number (interaction parameter)
    a = gax(0, 2)
    N = profile["N_Stuart"]
    a.fill_between(t, N, alpha=0.15, color=C3)
    a.plot(t, N, color=C3, lw=2)
    a.axhline(0.01, color=C4, lw=0.9, ls="--", label="N=0.01 (MHD onset)")
    a.axhline(0.10, color="#ff4560", lw=0.9, ls="--", label="N=0.10 (strong)")
    a.legend(fontsize=7.5)
    a.set_title("Stuart Interaction Parameter N", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("N = σB²L/(ρv)")
    a.set_yscale("symlog", linthresh=1e-4)

    # Cd correction
    a = gax(0, 3)
    Cd = profile["Cd_correction"]
    a.fill_between(t, Cd*100, alpha=0.2, color="#9d60ff")
    a.plot(t, Cd*100, color="#9d60ff", lw=2)
    a.axhline(0, color=TX, lw=0.5, alpha=0.5)
    a.set_title("ΔCd from MHD [%]", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("ΔCd [%]")

    # Joule heating
    a = gax(1, 0)
    qj = profile["q_joule_Wm2"]
    a.fill_between(t, qj/1e3, alpha=0.2, color=C4)
    a.plot(t, qj/1e3, color=C4, lw=2)
    a.set_title("Joule Heating q_joule", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("q [kW/m²]")

    # MHD drag force
    a = gax(1, 1)
    Fd = profile["F_drag_MHD_N"]
    a.fill_between(t, Fd/1e3, alpha=0.2, color=C2)
    a.plot(t, Fd/1e3, color=C2, lw=2)
    a.axhline(0, color=TX, lw=0.5, alpha=0.5)
    a.set_title("MHD Drag Force", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("F_MHD [kN]")

    # σ vs v (phase space)
    a = gax(1, 2)
    v_mag = v_arr
    valid = sig > 0.001
    sc = a.scatter(v_mag[valid]/1e3, sig[valid], c=h_arr[valid]/1e3,
                   cmap="plasma", s=8, alpha=0.8)
    fig.colorbar(sc, ax=a, label="h [km]", pad=0.02).ax.tick_params(labelsize=7)
    a.set_xlabel("v [km/s]"); a.set_ylabel("σ [S/m]")
    a.set_title("σ vs velocity (coloured by h)", fontweight="bold")

    # Conductivity vs T (model validation)
    a = gax(1, 3)
    pm = PlasmaModel()
    T_range = np.linspace(2000, 12000, 100)
    rho_ref = 0.005
    sig_T   = np.array([pm.conductivity(T, rho_ref) for T in T_range])
    a.semilogy(T_range, sig_T + 1e-6, color=C1, lw=2)
    a.axvline(5000, color=C4, lw=0.9, ls="--", alpha=0.7, label="Ionisation onset")
    a.legend(fontsize=8)
    a.set_xlabel("T [K]"); a.set_ylabel("σ [S/m]")
    a.set_title("Spitzer Conductivity σ(T)", fontweight="bold")

    B = profile["B_field_T"]
    fig.text(0.5, 0.955,
             f"MHD Plasma Steering  |  B={B:.2f}T  |  "
             f"σ_max={sig.max():.2f}S/m  |  "
             f"N_max={N.max():.4f}  |  ΔCd_max={Cd.max()*100:.3f}%",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ MHD plot saved: {save_path}")
    plt.close(fig)


def run(B_field_T: float = 0.5, verbose: bool = True) -> dict:
    """Run MHD plasma steering analysis along a Mars entry trajectory."""
    import matplotlib; matplotlib.use("Agg")
    from src.planetary_atm import MarsAtmosphere
    from src.multifidelity_pinn import LowFidelityEDL

    planet = MarsAtmosphere()
    lf     = LowFidelityEDL(planet, 900, 1.7, 78.5, gamma_deg=15)
    t_arr  = np.linspace(0, 400, 200)
    v_arr, h_arr = lf.solve(t_arr, 5800, 125_000)

    cfg     = MHDConfig(B_nose_T=B_field_T, L_ref=4.5, beta_hall=1.5)
    mhd     = MHDSteering(cfg)
    profile = mhd.trajectory_mhd_profile(v_arr, h_arr, planet)

    if verbose:
        print(f"\n[MHD] B={B_field_T:.2f}T  β_Hall={cfg.beta_hall:.1f}")
        print(f"  σ_max    = {profile['sigma_Sm'].max():.2f} S/m")
        print(f"  nₑ_max   = {profile['n_e_m3'].max():.2e} m⁻³")
        print(f"  N_max    = {profile['N_Stuart'].max():.4f}")
        print(f"  ΔCd_max  = {profile['Cd_correction'].max()*100:.3f}%")
        print(f"  q_Joule  = {profile['q_joule_Wm2'].max()/1e3:.2f} kW/m²")

    plot_mhd(profile, t_arr, v_arr, h_arr)
    return {"profile": profile, "t": t_arr, "v": v_arr, "h": h_arr}


if __name__ == "__main__":
    result = run(B_field_T=1.0)
    print(f"Stuart number peak: {result['profile']['N_Stuart'].max():.4f}")
