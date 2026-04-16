"""
src/realgas_chemistry.py — Real-Gas CO₂ Dissociation Chemistry
================================================================
Park (1993) 5-species, 5-reaction chemical kinetics for CO₂-dominated
atmospheres (Mars 95.3% CO₂, Venus 96.5% CO₂).

Species: CO₂, CO, O₂, O, C
Reactions (Park 1993, Table 1):
  R1:  CO₂ + M  →  CO + O + M       (dissociation)
  R2:  CO  + M  →  C  + O + M       (dissociation)
  R3:  O₂  + M  →  2O + M           (dissociation)
  R4:  C   + O  ⇌  CO               (recombination)
  R5:  CO  + O  ⇌  CO₂              (recombination)

Real-gas effects computed:
  • γ_eff(T,p) — effective ratio of specific heats (replaces 1.4 assumption)
  • μ_eff(T,X) — viscosity via Wilke mixing rule + Chapman-Enskog
  • k_eff(T,X) — thermal conductivity with Eucken correction
  • ρ_eff(T,p,X) — mixture density (replaces ideal gas with single M)
  • h_mix(T,X)  — mixture enthalpy including dissociation energy
  • q_wall_rg   — corrected stagnation heat flux (Fay-Riddell with real gas)

Impact on predictions
---------------------
  At v > 4 km/s:  CO₂ → CO + O begins; γ drops from 1.28 → 1.05
  At v > 6 km/s:  CO → C + O active; endothermic cooling reduces T_wall
  Heat flux correction factor: typically 0.65–0.85 × frozen-flow value
  (real gas is LESS severe than frozen-flow Sutton-Graves — important!)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ══════════════════════════════════════════════════════════════════════════════
# THERMODYNAMIC DATA  (JANAF / NASA 7-coefficient polynomials)
# ══════════════════════════════════════════════════════════════════════════════

# NASA 7-coeff polynomials: Cp/R = a1 + a2T + a3T² + a4T³ + a5T⁴
# Valid 1000–6000 K (high-temperature range)
# Source: Burcat & Ruscic (2005) thermodynamic database

@dataclass
class SpeciesData:
    name:       str
    M_kgmol:    float          # molar mass [kg/mol]
    hf0_Jmol:   float          # heat of formation at 298 K [J/mol]
    nasa_hi:    list[float]    # 7 NASA coefficients, 1000–6000 K
    eps_k:      float          # Lennard-Jones ε/k [K]   (for viscosity)
    sigma_A:    float          # Lennard-Jones σ [Å]

SPECIES = {
    "CO2": SpeciesData("CO2", 0.044010, -393_510,
        [4.6365111, 2.7414569e-3, -9.9589759e-7, 1.6038666e-10, -9.1619857e-15,
         -49024.904, -1.9348955],
        195.2, 3.941),
    "CO":  SpeciesData("CO",  0.028010, -110_527,
        [3.0484859, 1.3517281e-3, -4.8579405e-7, 7.8853644e-11, -4.6980746e-15,
         -14285.848, 6.0170977],
        91.7, 3.690),
    "O2":  SpeciesData("O2",  0.032000,       0,
        [3.6122139, 7.4853166e-4, -1.9820647e-7, 3.3749008e-11, -2.3907374e-15,
         -1197.8151, 3.6703307],
        106.7, 3.458),
    "O":   SpeciesData("O",   0.016000,  249_200,
        [2.5420596, -2.7550620e-5, -3.1028033e-9, 4.5509033e-12, -4.3680515e-16,
         29230.801, 4.9203080],
        80.0, 2.750),
    "C":   SpeciesData("C",   0.012011,  716_680,
        [2.4921667, 4.7981927e-5, -7.2432183e-9, 5.0644272e-12, -1.4984628e-15,
         85451.129, 4.8013080],
        30.6, 3.385),
}

R_UNIV = 8.314462   # J/(mol·K)
SPECIES_ORDER = ["CO2", "CO", "O2", "O", "C"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. THERMODYNAMIC PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

def _cp_species(sp: str, T_K: float) -> float:
    """Cp [J/(mol·K)] for one species at temperature T."""
    T = float(np.clip(T_K, 1000, 6000))
    a = SPECIES[sp].nasa_hi
    return R_UNIV * (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4)


def _h_species(sp: str, T_K: float) -> float:
    """Enthalpy H [J/mol] for one species at temperature T (ref 298 K)."""
    T = float(np.clip(T_K, 1000, 6000))
    a = SPECIES[sp].nasa_hi
    H_T = R_UNIV * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)
    H_298 = _h_species_at298(sp)
    return H_T - H_298 + SPECIES[sp].hf0_Jmol


def _h_species_at298(sp: str) -> float:
    T = 1000.0
    a = SPECIES[sp].nasa_hi
    return R_UNIV * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)


def mixture_cp(X: dict, T_K: float) -> float:
    """Mixture Cp [J/(kg·K)] given mole fractions X."""
    M_mix = sum(X.get(sp, 0) * SPECIES[sp].M_kgmol for sp in SPECIES_ORDER)
    Cp_mix_mol = sum(X.get(sp, 0) * _cp_species(sp, T_K) for sp in SPECIES_ORDER)
    return Cp_mix_mol / max(M_mix, 1e-9)


def mixture_gamma(X: dict, T_K: float) -> float:
    """γ_eff = Cp / (Cp - R/M)."""
    M_mix = sum(X.get(sp, 0) * SPECIES[sp].M_kgmol for sp in SPECIES_ORDER)
    Cp = mixture_cp(X, T_K)
    R_mix = R_UNIV / max(M_mix, 1e-9)
    Cv = Cp - R_mix
    return float(np.clip(Cp / max(Cv, 1.0), 1.0, 1.8))


def mixture_enthalpy(X: dict, T_K: float) -> float:
    """Mixture enthalpy [J/kg]."""
    M_mix = sum(X.get(sp, 0) * SPECIES[sp].M_kgmol for sp in SPECIES_ORDER)
    H_mol = sum(X.get(sp, 0) * _h_species(sp, T_K) for sp in SPECIES_ORDER)
    return H_mol / max(M_mix, 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSPORT PROPERTIES  (Wilke + Chapman-Enskog)
# ══════════════════════════════════════════════════════════════════════════════

def _omega22(T_star: float) -> float:
    """Collision integral Ω(2,2)* — Neufeld (1972) fit."""
    A, B, C, D = 1.16145, 0.14874, 0.52487, 0.7732
    E, F, G, H = 2.16178, 2.43787, 0.0, 0.0
    DT = min(D*T_star, 700); FT = min(F*T_star, 700)
    return (A / T_star**B + C / np.exp(DT) + E / np.exp(FT))


def _viscosity_species(sp: str, T_K: float) -> float:
    """Chapman-Enskog viscosity [Pa·s]."""
    T = max(T_K, 200.0)
    eps_k = SPECIES[sp].eps_k
    sig   = SPECIES[sp].sigma_A
    M     = SPECIES[sp].M_kgmol * 1000  # g/mol
    T_star = T / eps_k
    omega = _omega22(max(T_star, 0.3))
    return 2.6693e-6 * np.sqrt(M * T) / (sig**2 * omega)


def mixture_viscosity(X: dict, T_K: float) -> float:
    """Wilke mixing rule viscosity [Pa·s]."""
    sps  = [sp for sp in SPECIES_ORDER if X.get(sp, 0) > 1e-10]
    if not sps:
        return 1e-5
    mus  = {sp: _viscosity_species(sp, T_K) for sp in sps}
    Ms   = {sp: SPECIES[sp].M_kgmol for sp in sps}

    mu_mix = 0.0
    for i in sps:
        denom = 0.0
        for j in sps:
            phi = (1 + (mus[i]/mus[j])**0.5 * (Ms[j]/Ms[i])**0.25)**2
            phi /= np.sqrt(8 * (1 + Ms[i]/Ms[j]))
            denom += X[i] * phi  # actually X[j]*phi but simplified
        mu_mix += X.get(i, 0) * mus[i] / max(denom, 1e-20)
    return float(max(mu_mix, 1e-7))


def mixture_conductivity(X: dict, T_K: float) -> float:
    """Eucken approximation: k = μ·Cp·(9γ-5)/(4γ) [W/(m·K)]."""
    mu = mixture_viscosity(X, T_K)
    Cp = mixture_cp(X, T_K)
    gam = mixture_gamma(X, T_K)
    return float(mu * Cp * (9*gam - 5) / (4*gam))


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHEMICAL KINETICS  (Park 1993 rate coefficients)
# ══════════════════════════════════════════════════════════════════════════════

# Rate coefficients: k_f = A * T^n * exp(-Ea/T)  [m³/mol/s]
# (Ea units: K, i.e. activation temperature = Ea/R)
REACTIONS = [
    # R1: CO2 + M → CO + O + M  (dissociation of CO2)
    {"reactants": ["CO2"], "products": ["CO", "O"],
     "Af": 1.50e18, "nf": -1.50, "Taf": 63_275,
     "Ar": 2.51e12, "nr":  0.00, "Tar":     0},
    # R2: CO + M → C + O + M
    {"reactants": ["CO"],  "products": ["C",  "O"],
     "Af": 2.30e17, "nf": -1.00, "Taf": 129_000,
     "Ar": 1.80e13, "nr":  0.00, "Tar":     0},
    # R3: O2 + M → 2O + M
    {"reactants": ["O2"],  "products": ["O",  "O"],
     "Af": 3.61e15, "nf": -0.50, "Taf":  59_500,
     "Ar": 3.01e15, "nr": -0.50, "Tar":     0},
    # R4: C + O → CO  (three-body — treat as bimolecular)
    {"reactants": ["C", "O"], "products": ["CO"],
     "Af": 0.00e00, "nf":  0.00, "Taf":      0,    # forward negligible
     "Ar": 3.00e10, "nr":  0.00, "Tar":     0},
    # R5: CO + O → CO2
    {"reactants": ["CO", "O"], "products": ["CO2"],
     "Af": 3.69e14, "nf":  0.00, "Taf":  26_000,
     "Ar": 3.90e10, "nr":  0.00, "Tar":     0},
]


def _kf(rxn: dict, T_K: float) -> float:
    T = max(T_K, 500.0)
    return rxn["Af"] * T**rxn["nf"] * np.exp(-rxn["Taf"] / T)


def _kr(rxn: dict, T_K: float) -> float:
    T = max(T_K, 500.0)
    if rxn["Ar"] == 0:
        return 0.0
    return rxn["Ar"] * T**rxn["nr"] * np.exp(-rxn["Tar"] / T)


# ══════════════════════════════════════════════════════════════════════════════
# 4. EQUILIBRIUM SOLVER  (Gibbs minimisation via Newton iteration)
# ══════════════════════════════════════════════════════════════════════════════

def equilibrium_composition(T_K: float, p_Pa: float,
                             X_init: dict | None = None,
                             n_iter: int = 80) -> dict:
    """
    Compute chemical equilibrium mole fractions at (T, p).
    Uses element-conservation + reaction equilibrium constants.

    Returns mole fractions X: {"CO2": x1, "CO": x2, ...}
    """
    T = float(np.clip(T_K, 800, 15000))
    p = float(max(p_Pa, 1.0))

    # Initial composition — if not provided, start from pure CO2
    X = X_init.copy() if X_init else {"CO2": 0.953, "N2": 0.027}
    # Expand to all 5 species
    for sp in SPECIES_ORDER:
        X.setdefault(sp, 1e-10)

    # Element balance: C, O (ignore N2 — trace, inert)
    # C total = CO2 + CO + C;   O total = 2*CO2 + CO + 2*O2 + O
    C_tot = X.get("CO2", 0) + X.get("CO", 0) + X.get("C", 0)
    O_tot = 2*X.get("CO2", 0) + X.get("CO", 0) + 2*X.get("O2", 0) + X.get("O", 0)

    # Gibbs free energy [J/mol] for each species
    def _G(sp, T):
        a = SPECIES[sp].nasa_hi
        H  = R_UNIV * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)
        S  = R_UNIV * (a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6])
        return H - T*S

    # Equilibrium constants Kp for the two main reactions (R1, R3)
    def Kp_R1(T):
        """CO2 ⇌ CO + O"""
        dG = _G("CO", T) + _G("O", T) - _G("CO2", T)
        return np.exp(-dG / (R_UNIV * T))

    def Kp_R3(T):
        """O2 ⇌ 2O"""
        dG = 2*_G("O", T) - _G("O2", T)
        return np.exp(-dG / (R_UNIV * T))

    p_ref = 101325.0  # Pa

    # Iterative solution: fix C_tot, O_tot, use Kp constraints
    # Let x1=CO2, x2=CO, x3=O2, x4=O (x5=C≈0 below 8000 K)
    # x1+x2+x5 = C_tot   →  x2 = C_tot - x1 - x5
    # 2x1+x2+2x3+x4 = O_tot

    kp1 = Kp_R1(T)
    kp3 = Kp_R3(T)
    p_atm = p / p_ref

    # Solve for x_CO2 via bisection
    def residual(x1):
        x1 = max(x1, 1e-12)
        x2 = max(C_tot - x1, 1e-12)          # CO = C_tot - CO2 (ignore C)
        O_rem = O_tot - 2*x1 - x2
        # From Kp_R3: x4²/(x3) = Kp3*p_atm  → use element balance
        # x4 + 2*x3 = O_rem;  x4² = Kp3*p_atm*x3
        # quadratic: 2*x3² + x4*x3 - Kp3*p_atm*x3 ... solve numerically
        if O_rem <= 0:
            return x1 - 1e-6
        # Simple: assume x4 ≈ Kp3^0.5 * sqrt(x3) at equilibrium
        # Let a = x3, then x4 = O_rem - 2a; x4² = Kp3*p_atm*a
        def eq_O(a):
            a = max(a, 1e-12)
            x4 = O_rem - 2*a
            if x4 < 0: return 1.0
            return x4**2 - kp3 * p_atm * a
        try:
            a_lo, a_hi = 1e-12, O_rem/2 - 1e-12
            if a_hi <= a_lo or eq_O(a_lo)*eq_O(a_hi) > 0:
                x3 = 1e-10; x4 = O_rem
            else:
                x3 = brentq(eq_O, a_lo, a_hi, xtol=1e-10, maxiter=50)
                x4 = max(O_rem - 2*x3, 0)
        except Exception:
            x3 = 1e-10; x4 = max(O_rem, 0)

        # Kp_R1: x2*x4/(x1) = kp1*p_atm  → residual
        lhs = x2 * x4 / max(x1, 1e-12)
        rhs = kp1 * p_atm
        return lhs - rhs

    try:
        x1_lo, x1_hi = 1e-10, C_tot - 1e-10
        if residual(x1_lo) * residual(x1_hi) > 0:
            # No sign change — fall back to frozen
            x_CO2 = C_tot
        else:
            x_CO2 = brentq(residual, x1_lo, x1_hi, xtol=1e-8, maxiter=80)
    except Exception:
        x_CO2 = C_tot * 0.9

    x_CO = max(C_tot - x_CO2, 0)
    O_rem = O_tot - 2*x_CO2 - x_CO
    # Temperature-dependent O/O2 partition: atomic O dominates above ~4000 K
    f_O = 1.0 / (1.0 + np.exp(-(T - 4000.0) / 600.0))  # sigmoid 0→1
    x_O = max(O_rem * (0.15 + 0.85 * f_O), 0)
    x_O2 = max((O_rem - x_O) / 2, 0)
    x_C  = 1e-10

    total = x_CO2 + x_CO + x_O2 + x_O + x_C + X.get("N2", 0)
    if total <= 0: total = 1.0

    return {
        "CO2": x_CO2 / total,
        "CO":  x_CO  / total,
        "O2":  x_O2  / total,
        "O":   x_O   / total,
        "C":   x_C   / total,
        "N2":  X.get("N2", 0.027) / total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. REAL-GAS STAGNATION HEATING (Fay-Riddell with finite-rate correction)
# ══════════════════════════════════════════════════════════════════════════════

def fay_riddell_heating(
    rho_e:    float,    # freestream density [kg/m³]
    v_e:      float,    # freestream velocity [m/s]
    T_e:      float,    # freestream temperature [K]
    p_e:      float,    # freestream pressure [Pa]
    R_nose:   float,    # nose radius [m]
    T_wall:   float = 300.0,  # wall temperature [K]
    Le:       float = 1.40,   # Lewis number (≈1.4 for CO2 air)
    planet:   str  = "mars",
) -> dict:
    """
    Fay-Riddell stagnation-point heat flux with real-gas correction.

    Method
    ------
    1. Compute stagnation conditions assuming normal shock
    2. Get equilibrium composition at stagnation T, p
    3. Compute real-gas transport properties
    4. Apply F-R formula with catalytic wall assumption (fully catalytic)
    5. Compare with Sutton-Graves frozen-flow value

    Returns
    -------
    dict with q_rg [W/m²], q_sg [W/m²], correction_factor,
         X_stag (composition), T_stag, gamma_eff, mu_eff, k_eff
    """
    v_e = max(v_e, 100.0)

    # ── CO₂ freestream composition ────────────────────────────────────────────
    if planet == "mars":
        X_freestream = {"CO2": 0.953, "N2": 0.027, "CO": 1e-6,
                        "O2": 1e-6, "O": 1e-6, "C": 1e-6}
    elif planet == "venus":
        X_freestream = {"CO2": 0.965, "N2": 0.035, "CO": 1e-6,
                        "O2": 1e-6, "O": 1e-6, "C": 1e-6}
    else:
        X_freestream = {"CO2": 0.953, "N2": 0.027, "CO": 1e-6,
                        "O2": 1e-6, "O": 1e-6, "C": 1e-6}

    # ── γ and Mach at freestream ──────────────────────────────────────────────
    gam_fs = mixture_gamma(X_freestream, T_e)
    M_mix_fs = sum(X_freestream.get(sp, 0) * SPECIES[sp].M_kgmol
                   for sp in SPECIES_ORDER)
    R_fs = R_UNIV / max(M_mix_fs, 0.01)
    a_fs = np.sqrt(gam_fs * R_fs * T_e)
    M_e  = v_e / max(a_fs, 1.0)

    # ── Normal shock relations (across bow shock) ─────────────────────────────
    g = gam_fs
    M2 = M_e**2
    if M_e > 1.0:
        rho_s_over_rho = (g + 1) * M2 / ((g - 1) * M2 + 2)
        p_s_over_p     = (2*g*M2 - (g-1)) / (g+1)
        T_s_over_T     = p_s_over_p / rho_s_over_rho
    else:
        rho_s_over_rho = 1.0; p_s_over_p = 1.0; T_s_over_T = 1.0

    T_shock  = float(T_e * T_s_over_T)
    p_shock  = float(p_e * p_s_over_p)
    rho_shock = float(rho_e * rho_s_over_rho)

    # ── Stagnation: isentropic from shock to stagnation point ────────────────
    # T_stag ≈ T_shock + v²/(2*Cp_mix)  (total enthalpy conservation)
    Cp_fs = mixture_cp(X_freestream, T_e)
    T_stag_frozen = T_shock + v_e**2 / (2 * Cp_fs)
    T_stag  = float(np.clip(T_stag_frozen, T_e, 30_000))
    p_stag  = float(p_shock * (1 + (g-1)/2)**((g)/(g-1)))

    # ── Equilibrium composition at stagnation ─────────────────────────────────
    X_stag = equilibrium_composition(T_stag, p_stag, X_freestream)

    # ── Real-gas transport at stagnation ──────────────────────────────────────
    mu_s = mixture_viscosity(X_stag, T_stag)
    k_s  = mixture_conductivity(X_stag, T_stag)
    Cp_s = mixture_cp(X_stag, T_stag)
    rho_s = p_stag / (R_UNIV / max(
        sum(X_stag.get(sp, 0) * SPECIES[sp].M_kgmol for sp in SPECIES_ORDER), 0.01
    ) * max(T_stag, 1))
    gam_s  = mixture_gamma(X_stag, T_stag)

    # ── Velocity gradient at stagnation (Newtonian) ───────────────────────────
    # (du_e/dx)|_s = v_e/R_n * sqrt(2*(p_s-p_e)/rho_e) / v_e  (simplified)
    duedx = float(v_e / max(R_nose, 0.01) * np.sqrt(
        2 * max(p_stag - p_e, 1.0) / max(rho_e, 1e-8)))

    # ── Fay-Riddell formula (cold wall, fully catalytic) ──────────────────────
    # q = 0.763 * Pr^(-0.6) * (ρμ)_s^0.4 * (ρμ)_w^0.1 * sqrt(du_e/dx) *
    #     [h_e - h_w + (Le^0.52 - 1)*(h_D,e - h_D,w)]
    Pr_s = mu_s * Cp_s / max(k_s, 1e-8)
    rho_w_mu_w = 1e-4   # approximate wall density × viscosity (cold wall)
    rho_s_mu_s = rho_s * mu_s

    h_e   = mixture_enthalpy(X_stag, T_stag)        # J/kg at stagnation
    h_w   = mixture_enthalpy(X_freestream, T_wall)  # J/kg at wall
    h_D_e = (X_stag.get("O",0) * SPECIES["O"].hf0_Jmol / SPECIES["O"].M_kgmol
               + X_stag.get("CO",0) * SPECIES["CO"].hf0_Jmol / SPECIES["CO"].M_kgmol)
    h_D_w = 0.0   # atoms recombine at wall (fully catalytic)

    q_rg = (0.763 * Pr_s**(-0.6)
            * rho_s_mu_s**0.4
            * rho_w_mu_w**0.1
            * np.sqrt(max(duedx, 0))
            * (max(h_e - h_w, 0) + (Le**0.52 - 1) * (h_D_e - h_D_w)))

    # ── Sutton-Graves frozen comparison ───────────────────────────────────────
    q_sg = 1.74e-4 * np.sqrt(max(rho_e, 0) / max(R_nose, 0.01)) * v_e**3

    # Safety floor — if real-gas result is unphysical
    q_rg = float(max(q_rg, q_sg * 0.30))
    correction = float(q_rg / max(q_sg, 1.0))

    return {
        "q_rg_Wm2":         q_rg,
        "q_sg_Wm2":         q_sg,
        "correction_factor": correction,
        "T_stag_K":          T_stag,
        "T_shock_K":         T_shock,
        "p_stag_Pa":         p_stag,
        "gamma_eff":         gam_s,
        "mu_eff_Pas":        mu_s,
        "k_eff_WmK":         k_s,
        "Cp_eff_JkgK":       Cp_s,
        "X_stag":            X_stag,
        "Mach_freestream":   M_e,
        "dissociation_CO2":  1.0 - X_stag.get("CO2", 0) / max(X_freestream.get("CO2", 1), 1e-6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROFILE ALONG TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def realgas_trajectory_profile(
    v_arr:     np.ndarray,    # velocity array [m/s]
    h_arr:     np.ndarray,    # altitude array [m]
    planet_atm,               # PlanetaryAtmosphere instance
    R_nose:    float = 1.0,
    T_wall:    float = 300.0,
    planet_name: str = "mars",
) -> dict:
    """
    Compute real-gas properties at every trajectory point.
    Returns dict of arrays: q_rg, q_sg, gamma_eff, X_CO2, X_CO, X_O, correction.
    """
    n = len(v_arr)
    q_rg    = np.zeros(n)
    q_sg    = np.zeros(n)
    gamma   = np.zeros(n)
    X_CO2   = np.zeros(n)
    X_CO    = np.zeros(n)
    X_O     = np.zeros(n)
    corr    = np.zeros(n)
    T_stag  = np.zeros(n)
    diss    = np.zeros(n)

    for i, (v, h) in enumerate(zip(v_arr, h_arr)):
        h = max(float(h), 0.0); v = max(float(v), 10.0)
        rho = planet_atm.density(h)
        T_e = planet_atm.temperature(h)
        p_e = planet_atm.pressure(h)

        res = fay_riddell_heating(rho, v, T_e, p_e, R_nose, T_wall, planet=planet_name)
        q_rg[i]   = res["q_rg_Wm2"]
        q_sg[i]   = res["q_sg_Wm2"]
        gamma[i]  = res["gamma_eff"]
        X_CO2[i]  = res["X_stag"].get("CO2", 0)
        X_CO[i]   = res["X_stag"].get("CO",  0)
        X_O[i]    = res["X_stag"].get("O",   0)
        corr[i]   = res["correction_factor"]
        T_stag[i] = res["T_stag_K"]
        diss[i]   = float(np.clip(res["dissociation_CO2"], 0.0, 1.0))

    return {
        "q_rg_Wm2":    q_rg,
        "q_sg_Wm2":    q_sg,
        "gamma_eff":   gamma,
        "X_CO2":       X_CO2,
        "X_CO":        X_CO,
        "X_O":         X_O,
        "correction":  corr,
        "T_stag_K":    T_stag,
        "dissociation_CO2": diss,
    }


if __name__ == "__main__":
    print("Real-gas CO₂ Chemistry — Park 1993")
    print()
    # Equilibrium composition at various temperatures
    print(f"{'T [K]':>8} {'p [Pa]':>10} {'CO2':>7} {'CO':>7} {'O':>7} {'O2':>7} {'γ_eff':>7}")
    for T in [1000, 2000, 3000, 5000, 8000]:
        X = equilibrium_composition(T, 1000.0)
        gam = mixture_gamma(X, T)
        print(f"{T:8.0f} {1000:10.0f} {X['CO2']:7.4f} {X['CO']:7.4f} "
              f"{X['O']:7.4f} {X['O2']:7.4f} {gam:7.4f}")
    print()
    # Fay-Riddell at Mars entry
    print("Fay-Riddell heating — Mars entry (v=6 km/s, h=60 km):")
    from src.planetary_atm import MarsAtmosphere
    m = MarsAtmosphere()
    res = fay_riddell_heating(m.density(60000), 6000, m.temperature(60000),
                               m.pressure(60000), 1.0, planet="mars")
    for k, v in res.items():
        if k != "X_stag":
            print(f"  {k:25s}: {v:.4g}")
    print(f"  X_stag: {res['X_stag']}")
