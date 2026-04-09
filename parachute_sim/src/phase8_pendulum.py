"""
phase8_pendulum.py — Canopy Pendulum Oscillation Dynamics
==========================================================
Extends the Phase 6 3D trajectory with a fully coupled pendulum model for the
payload swinging below the canopy on its risers.

Physics derivation
------------------
The system is modelled as a SPHERICAL PENDULUM in the NON-INERTIAL reference
frame of the moving canopy pivot. The Lagrangian is:

    T = ½ m_p L² (θ̇² + sin²θ · φ̇²)
    V = -m_p · g_eff · L · n̂(θ,φ)

where:
    n̂(θ,φ) = (sinθ cosφ,  sinθ sinφ,  -cosθ)
        — unit vector from canopy pivot DOWN to payload
        — θ=0 ⟹ payload directly below canopy (stable equilibrium)

    g_eff = g_hat - a_canopy
          = (0,0,-g) - D⃗/(m_total)
        — effective gravity in non-inertial pivot frame

The Euler-Lagrange equations give:

    θ̈ = sinθ cosθ φ̇²
         + (1/L)[g_ex cosθ cosφ  +  g_ey cosθ sinφ  +  g_ez sinθ]
         - (c_θ / m_p L²) θ̇
         + F_vortex_θ

    φ̈ = -2 cotθ θ̇ φ̇
         + [1/(L sinθ)][-g_ex sinφ  +  g_ey cosφ]
         - (c_φ / m_p L²) φ̇
         + F_vortex_φ

Riser twist (torsional oscillator, coupled weakly through gyroscopic term):

    ψ̈ = -(k_ψ / I_z) ψ  -  (c_ψ / I_z) ψ̇  +  F_gyro(θ̇, φ̇)

State vector (12-D):
    q = [vx, vy, vz, x_c, y_c, z_c, θ, φ, θ̇, φ̇, ψ, ψ̇]

Key effects captured
--------------------
  • Pendulum restoring force from effective gravity (drag-modified)
  • Aerodynamic damping (calibrated to ζ ≈ 0.10–0.15 for real canopies)
  • Wind-shear excitation: lateral drag modifies g_eff, directly torquing pendulum
  • Vortex shedding: periodic forcing at Strouhal frequency (St · v / D_canopy)
  • Canopy area reduction: A_eff = A₀ · cos²(θ)  (tilt reduces projected area)
  • Riser twist ψ: torsional spring-damper, further modulates projected area
  • Payload landing offset: position and velocity at ground include swing contribution
  • Landing scatter: MC ensemble quantifies how oscillation spreads the landing zone

Outputs
-------
  • Full 12-state time series CSV
  • Landing position (canopy footprint + payload offset from pendulum)
  • Oscillation analysis: natural frequency, damping ratio, energy decay
  • 9-panel dashboard: swing angle, angular rate, trajectory, energy, polar plot,
      Poincaré section, phase portrait, landing scatter, riser twist
  • KML with both canopy track and payload footprint
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import density


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PENDULUM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PendulumConfig:
    """
    Physical parameters for the canopy–payload pendulum system.
    All values tunable; defaults represent a typical 50 m² ram-air sport canopy.
    """
    # ── Riser geometry ───────────────────────────────────────────────────────
    riser_length_m: float = 8.0       # L  — riser length from canopy to payload [m]

    # ── Mass breakdown ───────────────────────────────────────────────────────
    payload_mass_kg:   float = None   # m_p — payload alone (defaults to cfg.PARACHUTE_MASS)
    canopy_mass_kg:    float = 5.0    # m_c — canopy + lines (small vs payload)

    # ── Initial pendulum conditions ──────────────────────────────────────────
    theta0_rad:  float = 0.25         # θ₀  — initial swing angle from vertical [rad] ≈14°
    phi0_rad:    float = 0.0          # φ₀  — initial azimuth of swing plane [rad]
    thetadot0:   float = 0.0          # θ̇₀  — initial angular rate [rad/s]
    phidot0:     float = 0.0          # φ̇₀  — initial azimuthal rate [rad/s]

    # ── Riser twist (torsional DOF) ──────────────────────────────────────────
    psi0_rad:    float = 0.0          # ψ₀  — initial riser twist angle [rad]
    psidot0:     float = 0.0          # ψ̇₀  — initial twist rate [rad/s]
    I_twist:     float = 12.0         # I_z — payload axial moment of inertia [kg·m²]
    k_twist:     float = 150.0        # k_ψ — torsional stiffness [N·m/rad]
    c_twist:     float = 8.0          # c_ψ — torsional damping [N·m·s/rad]

    # ── Aerodynamic damping ──────────────────────────────────────────────────
    # Damping ratio ζ for the swing oscillation.
    # Real parachutes: ζ ≈ 0.08–0.20 (higher = more stable, damps faster)
    # c_θ is derived from ζ: c_θ = 2ζ · m_p · L² · ω_n
    zeta_swing:  float = 0.12         # damping ratio (dimensionless)

    # ── Canopy geometry ──────────────────────────────────────────────────────
    canopy_area_m2: float = None      # A₀ — defaults to cfg.CANOPY_AREA_M2
    canopy_Cd:      float = None      # Cd  — defaults to cfg.CD_INITIAL
    canopy_diam_m:  float = 8.0       # characteristic diameter for Strouhal calc

    # ── Vortex shedding excitation ───────────────────────────────────────────
    strouhal_number: float = 0.18     # St = f·D/v (typical for bluff bodies)
    vortex_amp_theta: float = 0.008   # amplitude of vortex forcing on θ [N·m / (kg·m²)]
    vortex_amp_phi:   float = 0.005   # amplitude of vortex forcing on φ

    # ── Area model ───────────────────────────────────────────────────────────
    # A_eff = A0 * cos²(θ) * (1 - sin²(ψ/ψ_max) * twist_area_factor)
    twist_area_factor: float = 0.3    # max area reduction from full twist
    psi_max_rad:       float = np.pi  # ψ at which full twist area reduction applies

    def __post_init__(self):
        if self.payload_mass_kg is None:
            self.payload_mass_kg = cfg.PARACHUTE_MASS - self.canopy_mass_kg
        if self.canopy_area_m2 is None:
            self.canopy_area_m2 = cfg.CANOPY_AREA_M2
        if self.canopy_Cd is None:
            self.canopy_Cd = cfg.CD_INITIAL

    @property
    def total_mass(self) -> float:
        return self.payload_mass_kg + self.canopy_mass_kg

    @property
    def omega_natural(self) -> float:
        """Natural frequency of pendulum [rad/s] at steady-state descent."""
        # At terminal velocity, g_eff ≈ g (drag acceleration ≈ gravity)
        # so the restoring force is 2g (effective g doubled)
        g_eff_mag = 2.0 * cfg.GRAVITY   # conservative estimate
        return np.sqrt(g_eff_mag / self.riser_length_m)

    @property
    def c_theta(self) -> float:
        """Aerodynamic damping coefficient for polar oscillation [N·m·s/rad]."""
        return (2.0 * self.zeta_swing
                * self.payload_mass_kg
                * self.riser_length_m**2
                * self.omega_natural)

    @property
    def period_s(self) -> float:
        """Natural oscillation period [s]."""
        return 2.0 * np.pi / self.omega_natural

    @property
    def decay_time_s(self) -> float:
        """1/e amplitude decay time [s]."""
        return 1.0 / (self.zeta_swing * self.omega_natural)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  12-STATE COUPLED ODE RHS
# ═══════════════════════════════════════════════════════════════════════════════

class PendulumODE:
    """
    Singularity-free 14-state ODE using CARTESIAN pendulum coordinates
    with Baumgarte constraint stabilisation.

    State ordering:
        0-2:   vx_c, vy_c, vz_c  — canopy velocity [m/s]  (vz_c < 0 = descending)
        3-5:   x_c, y_c, z_c     — canopy position [m]
        6-8:   nx, ny, nz         — riser vector (pivot→payload), |n|=L [m]
        9-11:  vnx, vny, vnz      — riser velocity [m/s]
        12:    ψ                  — riser twist [rad]
        13:    ψ̇                  — riser twist rate [rad/s]

    Physics:
        Translational EOM:  m_total * a_c = D + m_total*g_hat
        Pendulum EOM:       m_p * a_q = m_p*g_eff + F_tension + F_damp + F_vortex
        Constraint:         |n|² = L²   (Baumgarte stabilised)
        Twist EOM:          I_z * ψ̈ = -k_ψ*ψ - c_ψ*ψ̇ + τ_gyro
    """

    # Baumgarte constraint stabilisation parameters
    ALPHA = 2.0      # velocity constraint damping
    BETA  = 1.0      # position constraint damping

    def __init__(self, pend: PendulumConfig, wind, at_fn=None):
        self.p    = pend
        self.wind = wind
        self.at_fn = at_fn
        self._L2  = pend.riser_length_m ** 2

    def _At(self, t: float) -> float:
        if self.at_fn is not None:
            return max(0.0, float(self.at_fn(t)))
        Am = self.p.canopy_area_m2
        ti, k = 2.5, 5.0 / 2.5
        return Am / (1 + np.exp(-k * (t - ti * 0.6))) ** 0.5

    def _A_eff(self, t: float, nx: float, ny: float, nz: float, psi: float) -> float:
        """Effective area: reduced by tilt (nz/L gives cos θ) and riser twist."""
        A0    = self._At(t)
        L     = self.p.riser_length_m
        cos_t = -nz / max(L, 1e-6)                # nz = -L*cos(theta) → cos θ = -nz/L
        cos_t = float(np.clip(cos_t, 0.0, 1.0))
        tilt  = cos_t ** 2
        twist = 1.0 - self.p.twist_area_factor * float(
            np.sin(psi / max(self.p.psi_max_rad, 1e-6)) ** 2)
        return A0 * tilt * max(0.05, twist)

    def _strouhal_force(self, t: float, vz_c: float, nx: float, ny: float) -> tuple:
        """
        Vortex-shedding body force on pendulum bob [m/s²] in Cartesian coords.
        Amplitude is low — only slowly modulates oscillation, does NOT create stiff dynamics.
        """
        v_mag = max(1.0, abs(vz_c))
        f_s   = self.p.strouhal_number * v_mag / max(self.p.canopy_diam_m, 0.1)
        # Cap Strouhal frequency to avoid stiff forcing relative to pendulum period
        f_s   = min(f_s, 0.3)
        phase = 2.0 * np.pi * f_s * t
        # Perpendicular-to-riser direction in horizontal plane
        nh    = max(np.sqrt(nx**2 + ny**2), 1e-6)
        ex    = -ny / nh;  ey = nx / nh    # unit vector ⊥ horizontal projection of riser
        amp   = self.p.vortex_amp_theta * v_mag / max(v_mag, 5.0)  # scale with speed
        fx    = amp * np.sin(phase) * ex
        fy    = amp * np.sin(phase) * ey
        fz    = amp * np.cos(phase + np.pi/4) * 0.3
        return fx, fy, fz

    def __call__(self, t: float, state: np.ndarray) -> list:
        (vx_c, vy_c, vz_c,
         x_c,  y_c,  z_c,
         nx,   ny,   nz,
         vnx,  vny,  vnz,
         psi,  psidot) = state

        z_c = max(0.0, z_c)
        L   = self.p.riser_length_m

        # ── Wind & drag ───────────────────────────────────────────────────────
        u_wind, v_wind = self.wind(z_c)
        vrel_x = vx_c - u_wind
        vrel_y = vy_c - v_wind
        vrel_z = vz_c
        v_rel  = max(np.sqrt(vrel_x**2 + vrel_y**2 + vrel_z**2), 1e-6)

        rho   = density(z_c)
        A_eff = self._A_eff(t, nx, ny, nz, psi)
        D_mag = 0.5 * rho * v_rel**2 * self.p.canopy_Cd * A_eff
        D_x   = -D_mag * vrel_x / v_rel
        D_y   = -D_mag * vrel_y / v_rel
        D_z   = -D_mag * vrel_z / v_rel

        # ── Translational EOM ─────────────────────────────────────────────────
        m   = self.p.total_mass
        ax  = D_x / m
        ay  = D_y / m
        az  = -cfg.GRAVITY + D_z / m

        # ── Effective gravity in non-inertial pivot frame ─────────────────────
        # g_eff = g_hat - a_pivot = (0,0,-g) - (ax,ay,az)
        g_ex = -ax
        g_ey = -ay
        g_ez = -cfg.GRAVITY - az     # = -D_z/m  (always ≤ 0 when drag is upward)

        # ── Pendulum Cartesian EOM (constrained) ──────────────────────────────
        m_p = self.p.payload_mass_kg
        c_d = self.p.c_theta          # aerodynamic damping [N·m·s] ÷ L² → [N·s/m/m_p]

        # Constraint quantities
        n_dot_v = nx*vnx + ny*vny + nz*vnz        # n · ṅ  (should be 0 on constraint)
        n_sq    = nx**2  + ny**2  + nz**2          # |n|²  (should be L²)
        v_sq    = vnx**2 + vny**2 + vnz**2         # |ṅ|²

        # Riser tension  λ = m_p*(|ṅ|² + n·g_eff) / L
        n_dot_g = nx*g_ex + ny*g_ey + nz*g_ez
        lam     = m_p * (v_sq + n_dot_g) / max(L, 1e-6)

        # Baumgarte stabilisation: C1 = n·ṅ/L,  C2 = (|n|²-L²)/L²
        C1 = n_dot_v / max(L, 1e-6)
        C2 = (n_sq - self._L2) / max(self._L2, 1e-6)

        # Damping force: proportional to velocity component perpendicular to riser
        # v_perp = ṅ - (n·ṅ/|n|²)*n
        n_dot_v_over_nsq = n_dot_v / max(n_sq, 1e-6)
        f_damp_x = -(c_d / (m_p * max(self._L2, 1e-6))) * (vnx - n_dot_v_over_nsq*nx)
        f_damp_y = -(c_d / (m_p * max(self._L2, 1e-6))) * (vny - n_dot_v_over_nsq*ny)
        f_damp_z = -(c_d / (m_p * max(self._L2, 1e-6))) * (vnz - n_dot_v_over_nsq*nz)

        # Vortex shedding
        fv_x, fv_y, fv_z = self._strouhal_force(t, vz_c, nx, ny)

        # Tension direction: from payload toward canopy = -n/L
        f_ten_fac = lam / (m_p * max(L, 1e-6))

        # Full pendulum acceleration
        d_vnx = g_ex - f_ten_fac*nx + f_damp_x + fv_x                 - self.ALPHA*2*C1*nx/max(L,1e-6) - self.BETA**2*C2*nx
        d_vny = g_ey - f_ten_fac*ny + f_damp_y + fv_y                 - self.ALPHA*2*C1*ny/max(L,1e-6) - self.BETA**2*C2*ny
        d_vnz = g_ez - f_ten_fac*nz + f_damp_z + fv_z                 - self.ALPHA*2*C1*nz/max(L,1e-6) - self.BETA**2*C2*nz

        # ── Riser twist ───────────────────────────────────────────────────────
        # Gyroscopic coupling: angular momentum of swing drives twist
        omega_swing = np.sqrt(max(0.0, v_sq - n_dot_v_over_nsq**2*n_sq)) / max(L, 1e-6)
        tau_gyro    = 0.03 * self.p.I_twist * omega_swing * psidot
        d_psidot = (
            -(self.p.k_twist / self.p.I_twist) * psi
            -(self.p.c_twist / self.p.I_twist) * psidot
            + tau_gyro / self.p.I_twist
        )

        return [
            ax,  ay,  az,           # d(v_canopy)/dt
            vx_c, vy_c, vz_c,       # d(x_canopy)/dt
            vnx, vny, vnz,           # d(n)/dt
            d_vnx, d_vny, d_vnz,     # d(vn)/dt
            psidot, d_psidot,        # d(ψ)/dt, d(ψ̇)/dt
        ]
# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_pendulum(
    pend:    PendulumConfig,
    wind,
    at_fn         = None,
    v0:    float  = None,
    alt0:  float  = None,
    t_max: float  = None,
    dt_out: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Integrate the full 12-state pendulum ODE from deployment to ground.

    Returns a DataFrame with all state variables plus derived quantities:
    payload position & velocity, swing energy, riser tension, landing data.
    """
    v0   = v0   or cfg.INITIAL_VEL
    alt0 = alt0 or cfg.INITIAL_ALT
    p    = pend

    if t_max is None:
        # Generous upper bound: assume at least 5 m/s descent → budget extra
        t_max = alt0 / 3.0 + 300.0

    # Initial state
    y0 = [
        0.0, 0.0, -v0,                  # vx, vy, vz (vz negative = descending)
        0.0, 0.0, alt0,                 # x, y, z  (canopy position)
        p.theta0_rad, p.phi0_rad,       # θ, φ
        p.thetadot0,  p.phidot0,        # θ̇, φ̇
        p.psi0_rad,   p.psidot0,        # ψ, ψ̇
    ]

    t_eval = np.arange(0.0, t_max, dt_out)

    def ground_event(t, y): return y[5]
    ground_event.terminal  = True
    ground_event.direction = -1

    if verbose:
        print(f"\n[Phase 8] Pendulum ODE  —  L={p.riser_length_m}m  "
              f"θ₀={np.degrees(p.theta0_rad):.1f}°  ζ={p.zeta_swing}  "
              f"T_n={p.period_s:.2f}s")
        print(f"  Payload mass={p.payload_mass_kg:.1f}kg  "
              f"c_θ={p.c_theta:.1f} N·m·s/rad  decay τ={p.decay_time_s:.1f}s")

    rhs = PendulumODE(p, wind, at_fn)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_max),
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        events=ground_event,
        rtol=1e-6, atol=1e-8,
        dense_output=False,
    )

    if not sol.success and sol.t.size == 0:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t   = sol.t
    (vx_c, vy_c, vz_c,
     x_c,  y_c,  z_c,
     theta, phi, thetadot, phidot,
     psi, psidot) = sol.y

    z_c    = np.clip(z_c, 0.0, None)
    theta  = np.clip(theta, PendulumODE.THETA_MIN, PendulumODE.THETA_MAX)

    L = p.riser_length_m

    # ── Payload position (canopy + pendulum offset) ───────────────────────────
    x_payload = x_c + L * np.sin(theta) * np.cos(phi)
    y_payload = y_c + L * np.sin(theta) * np.sin(phi)
    z_payload = z_c - L * np.cos(theta)    # payload is BELOW canopy

    # ── Payload velocity ──────────────────────────────────────────────────────
    # v_payload = v_canopy + d/dt(L·n̂)
    # dn̂/dt = (θ̇ cosθ cosφ - φ̇ sinθ sinφ,  θ̇ cosθ sinφ + φ̇ sinθ cosφ,  θ̇ sinθ)
    vx_p = vx_c + L * (thetadot * np.cos(theta) * np.cos(phi)
                        - phidot  * np.sin(theta) * np.sin(phi))
    vy_p = vy_c + L * (thetadot * np.cos(theta) * np.sin(phi)
                        + phidot  * np.sin(theta) * np.cos(phi))
    vz_p = vz_c + L * thetadot * np.sin(theta)   # wait: n̂_z = -cosθ → dn̂_z/dt = sinθ·θ̇
    # Correction: n̂_z = -cosθ → d(-cosθ)/dt = sinθ·θ̇  (already correct above)

    v_payload_total = np.sqrt(vx_p**2 + vy_p**2 + vz_p**2)

    # ── Pendulum energy ───────────────────────────────────────────────────────
    m_p = p.payload_mass_kg
    KE_swing  = 0.5 * m_p * L**2 * (thetadot**2 + np.sin(theta)**2 * phidot**2)
    PE_swing  = m_p * cfg.GRAVITY * L * (1.0 - np.cos(theta))  # 0 at θ=0

    # ── Riser tension ─────────────────────────────────────────────────────────
    # T = m_p * (g_eff_z*cosθ + L*(θ̇² + sin²θ·φ̇²))  [centripetal + gravity component]
    # Approximate with effective g at each point
    rho_arr  = np.array([density(max(0.0, zi)) for zi in z_c])
    A_arr    = np.array([rhs._A_eff(ti, thi, pi)
                         for ti, thi, pi in zip(t, theta, psi)])
    v_rel_arr = np.sqrt(vx_c**2 + vy_c**2 + vz_c**2)
    D_arr    = 0.5 * rho_arr * v_rel_arr**2 * p.canopy_Cd * A_arr
    g_eff_z  = np.abs(cfg.GRAVITY - D_arr / p.total_mass) + cfg.GRAVITY  # effective |g| downward
    T_riser  = m_p * (g_eff_z * np.cos(theta)
                       + L * (thetadot**2 + np.sin(theta)**2 * phidot**2))

    # ── Swing amplitude (envelope) ────────────────────────────────────────────
    theta_deg = np.degrees(theta)

    # ── Drift from deployment ─────────────────────────────────────────────────
    drift_canopy  = np.sqrt(x_c**2 + y_c**2)
    drift_payload = np.sqrt(x_payload**2 + y_payload**2)

    df = pd.DataFrame({
        "time_s"         : t,
        # Translational — canopy
        "vx_c"           : vx_c,
        "vy_c"           : vy_c,
        "vz_c"           : vz_c,
        "x_canopy_m"     : x_c,
        "y_canopy_m"     : y_c,
        "z_canopy_m"     : z_c,
        # Pendulum angles
        "theta_rad"      : theta,
        "theta_deg"      : theta_deg,
        "phi_rad"        : phi,
        "phi_deg"        : np.degrees(phi),
        "thetadot"       : thetadot,
        "phidot"         : phidot,
        # Riser twist
        "psi_rad"        : psi,
        "psi_deg"        : np.degrees(psi),
        "psidot"         : psidot,
        # Payload (derived)
        "x_payload_m"    : x_payload,
        "y_payload_m"    : y_payload,
        "z_payload_m"    : np.clip(z_payload, 0.0, None),
        "vx_payload"     : vx_p,
        "vy_payload"     : vy_p,
        "vz_payload"     : vz_p,
        "v_payload_ms"   : v_payload_total,
        # Energy & loads
        "KE_swing_J"     : KE_swing,
        "PE_swing_J"     : PE_swing,
        "E_swing_J"      : KE_swing + PE_swing,
        "T_riser_N"      : np.clip(T_riser, 0.0, None),
        "drag_N"         : D_arr,
        "rho_kgm3"       : rho_arr,
        # Drift
        "drift_canopy_m" : drift_canopy,
        "drift_payload_m": drift_payload,
        "A_eff_m2"       : A_arr,
    })

    if verbose:
        land_v    = float(v_payload_total[-1])
        land_x    = float(x_payload[-1])
        land_y    = float(y_payload[-1])
        offset    = float(np.sqrt((x_payload[-1] - x_c[-1])**2
                                   + (y_payload[-1] - y_c[-1])**2))
        max_theta = float(theta_deg.max())
        t_land    = float(t[-1])
        print(f"\n  Simulation complete @ t={t_land:.1f}s")
        print(f"  Payload landing:  ({land_x:+.1f}, {land_y:+.1f}) m  "
              f"speed={land_v:.3f} m/s")
        print(f"  Canopy footprint: ({float(x_c[-1]):+.1f}, {float(y_c[-1]):+.1f}) m")
        print(f"  Pendulum offset at landing: {offset:.2f} m")
        print(f"  Max swing angle:  {max_theta:.2f}°  "
              f"(θ at landing: {float(theta_deg[-1]):.2f}°)")
        print(f"  Peak riser tension: {float(T_riser.max()):.1f} N")
        print(f"  Max riser twist: {float(np.degrees(np.abs(psi)).max()):.1f}°")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  OSCILLATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_oscillation(df: pd.DataFrame) -> dict:
    """
    Extract oscillation characteristics from the time-series.
    Computes: natural frequency, measured damping ratio, amplitude envelope,
    and half-life of the swing energy.
    """
    t     = df["time_s"].values
    theta = df["theta_deg"].values
    E     = df["E_swing_J"].values

    # ── Natural frequency from peak-to-peak timing ────────────────────────────
    peaks, props = find_peaks(theta, prominence=0.05, distance=5)
    if len(peaks) >= 3:
        dt_peaks = np.diff(t[peaks])
        T_meas   = float(np.median(dt_peaks))
        f_meas   = 1.0 / T_meas if T_meas > 0 else np.nan
        omega_meas = 2.0 * np.pi * f_meas
    else:
        T_meas = omega_meas = f_meas = np.nan

    # ── Damping ratio from logarithmic decrement ──────────────────────────────
    if len(peaks) >= 4:
        peak_vals = theta[peaks[:8]]     # use first 8 peaks for stability
        if peak_vals[-1] > 0.01:
            log_dec  = np.log(peak_vals[0] / peak_vals[-1]) / (len(peak_vals) - 1)
            zeta_meas = log_dec / np.sqrt((2 * np.pi)**2 + log_dec**2)
        else:
            zeta_meas = np.nan
    else:
        zeta_meas = np.nan

    # ── Swing energy half-life ────────────────────────────────────────────────
    E_stable = E[E > 0]
    if len(E_stable) > 20:
        E_half_idx = np.searchsorted(np.cumsum(E_stable > E_stable[0] / 2),
                                     len(E_stable) // 2)
        E_halflife = float(t[min(E_half_idx, len(t)-1)])
    else:
        E_halflife = np.nan

    # ── Poincaré section: (θ, θ̇) at each φ crossing ─────────────────────────
    phi   = df["phi_rad"].values
    tdot  = df["thetadot"].values
    # Find zero-crossings of φ (every half-oscillation in azimuth)
    phi_cross = np.where(np.diff(np.sign(np.sin(phi))))[0]
    poincare  = {
        "theta_deg": theta[phi_cross].tolist() if len(phi_cross) else [],
        "thetadot":  tdot[phi_cross].tolist()  if len(phi_cross) else [],
    }

    return {
        "T_natural_s":        T_meas,
        "f_natural_Hz":       f_meas,
        "omega_natural_rads": omega_meas,
        "zeta_measured":      zeta_meas,
        "E_swing_halflife_s": E_halflife,
        "n_oscillations":     len(peaks),
        "max_theta_deg":      float(theta.max()),
        "min_theta_deg":      float(theta.min()),
        "theta_final_deg":    float(theta[-1]),
        "poincare":           poincare,
        "peak_times_s":       t[peaks].tolist() if len(peaks) else [],
        "peak_vals_deg":      theta[peaks].tolist() if len(peaks) else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MONTE CARLO LANDING SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

def pendulum_mc_scatter(
    pend:  PendulumConfig,
    wind,
    n:     int   = 200,
    seed:  int   = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run N pendulum simulations with perturbed initial conditions to quantify
    how oscillation uncertainty spreads the landing zone.

    Perturbed parameters:
      - θ₀: ±50% of nominal initial swing angle
      - φ₀: uniform in [0, 2π) (random azimuth)
      - θ̇₀: ±0.1 rad/s jitter
      - L:  ±5% riser length tolerance
    """
    rng   = np.random.default_rng(seed)
    lands = []

    if verbose:
        print(f"\n[Phase 8] MC landing scatter: {n} pendulum runs...")

    for i in range(n):
        # Perturb initial conditions
        L_i = pend.riser_length_m * rng.uniform(0.95, 1.05)
        p_i = PendulumConfig(
            riser_length_m   = L_i,
            payload_mass_kg  = pend.payload_mass_kg * rng.uniform(0.97, 1.03),
            canopy_mass_kg   = pend.canopy_mass_kg,
            theta0_rad       = abs(rng.normal(pend.theta0_rad, pend.theta0_rad * 0.4)),
            phi0_rad         = rng.uniform(0, 2 * np.pi),
            thetadot0        = rng.normal(0.0, 0.08),
            phidot0          = rng.normal(0.0, 0.04),
            psi0_rad         = rng.normal(0.0, 0.3),
            psidot0          = rng.normal(0.0, 0.02),
            zeta_swing       = rng.uniform(0.08, 0.18),
            canopy_area_m2   = pend.canopy_area_m2,
            canopy_Cd        = pend.canopy_Cd,
        )

        try:
            df_i = solve_pendulum(p_i, wind, verbose=False, dt_out=0.2)
            if len(df_i) > 0:
                lands.append({
                    "x_payload": float(df_i["x_payload_m"].iloc[-1]),
                    "y_payload": float(df_i["y_payload_m"].iloc[-1]),
                    "x_canopy":  float(df_i["x_canopy_m"].iloc[-1]),
                    "y_canopy":  float(df_i["y_canopy_m"].iloc[-1]),
                    "land_v":    float(df_i["v_payload_ms"].iloc[-1]),
                    "max_theta": float(df_i["theta_deg"].max()),
                    "theta0":    np.degrees(p_i.theta0_rad),
                    "phi0":      np.degrees(p_i.phi0_rad),
                })
        except Exception:
            pass

        if verbose and (i+1) % 50 == 0:
            bar = "█" * ((i+1)*20//n) + "░" * (20 - (i+1)*20//n)
            print(f"\r  [{bar}] {i+1}/{n} runs  ({len(lands)} valid)", end="", flush=True)

    if verbose:
        print(f"\r  [{'█'*20}] {n}/{n} — {len(lands)} valid{' '*10}")

    return pd.DataFrame(lands)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATION — 9-panel dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pendulum(
    df:      pd.DataFrame,
    analysis: dict,
    mc_df:   pd.DataFrame = None,
    pend:    PendulumConfig = None,
    save_path: Path = None,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if cfg.DARK_THEME:
        matplotlib.rcParams.update({
            "figure.facecolor":"#080c14", "axes.facecolor":"#0d1526",
            "axes.edgecolor":"#2a3d6e",   "text.color":"#c8d8f0",
            "axes.labelcolor":"#c8d8f0",  "xtick.color":"#c8d8f0",
            "ytick.color":"#c8d8f0",      "grid.color":"#1a2744",
        })
    matplotlib.rcParams.update({"font.family":"monospace", "font.size":8.5})

    TEXT  = "#c8d8f0" if cfg.DARK_THEME else "#111"
    BG    = "#080c14" if cfg.DARK_THEME else "#ffffff"
    SPINE = "#2a3d6e" if cfg.DARK_THEME else "#cccccc"
    C_TH  = cfg.COLOR_THEORY      # cyan — translational
    C_PI  = cfg.COLOR_PINN        # orange — pendulum
    C_RAW = cfg.COLOR_RAW         # green — payload
    C_RED = "#ff4560"             # riser tension

    t     = df["time_s"].values
    theta = df["theta_deg"].values
    phi   = df["phi_deg"].values
    tdot  = df["thetadot"].values
    psi   = df["psi_deg"].values
    E     = df["E_swing_J"].values
    T_ris = df["T_riser_N"].values
    x_c   = df["x_canopy_m"].values
    y_c   = df["y_canopy_m"].values
    x_p   = df["x_payload_m"].values
    y_p   = df["y_payload_m"].values
    z_c   = df["z_canopy_m"].values
    vz_c  = df["vz_c"].values

    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.50, wspace=0.40,
                            top=0.91, bottom=0.06, left=0.05, right=0.97)

    def gax(r, c, **kw): return fig.add_subplot(gs[r, c], **kw)

    def style(ax, title, xlabel, ylabel):
        ax.set_title(title, fontweight="bold", pad=5, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    # ── Panel 0: swing angle θ(t) ─────────────────────────────────────────────
    ax0 = gax(0, 0)
    ax0.plot(t, theta, color=C_PI, lw=1.5, label="θ(t)")
    ax0.fill_between(t, theta, alpha=0.15, color=C_PI)
    ax0.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
    # Mark peaks
    peaks = analysis["peak_times_s"]
    pvals = analysis["peak_vals_deg"]
    if peaks:
        ax0.scatter(peaks, pvals, color=C_RAW, s=20, zorder=5, label="peaks")
    if not np.isnan(analysis["zeta_measured"]):
        decay = analysis.get("zeta_measured", 0)
        ax0.text(0.97, 0.95,
                 f"ζ={decay:.3f}\nT_n={analysis['T_natural_s']:.2f}s",
                 transform=ax0.transAxes, ha="right", va="top",
                 fontsize=8, color=C_PI)
    ax0.legend(fontsize=7.5)
    style(ax0, "Swing angle θ(t)", "Time [s]", "θ [deg]")

    # ── Panel 1: azimuth φ(t) ─────────────────────────────────────────────────
    ax1 = gax(0, 1)
    ax1.plot(t, phi, color=C_TH, lw=1.2)
    style(ax1, "Swing azimuth φ(t)", "Time [s]", "φ [deg]")

    # ── Panel 2: swing energy E(t) ────────────────────────────────────────────
    ax2 = gax(0, 2)
    ax2.fill_between(t, E, alpha=0.2, color=C_PI)
    ax2.plot(t, E, color=C_PI, lw=1.5, label="E_swing")
    # Exponential decay overlay
    if not np.isnan(analysis["E_swing_halflife_s"]) and len(t) > 2:
        tau  = analysis["E_swing_halflife_s"] / np.log(2)
        E0   = float(E[E > 0][0]) if E[E > 0].size else 1.0
        E_th = E0 * np.exp(-t / max(tau, 0.1))
        ax2.plot(t, E_th, color=C_TH, lw=1.0, ls="--", alpha=0.7, label="decay fit")
    ax2.legend(fontsize=7.5)
    style(ax2, "Pendulum swing energy", "Time [s]", "E [J]")

    # ── Panel 3: riser tension ────────────────────────────────────────────────
    ax3 = gax(0, 3)
    ax3.plot(t, T_ris, color=C_RED, lw=1.5)
    ax3.fill_between(t, T_ris, alpha=0.15, color=C_RED)
    m_total = (pend.total_mass if pend else cfg.PARACHUTE_MASS)
    static_T = m_total * cfg.GRAVITY
    ax3.axhline(static_T, color=TEXT, lw=0.8, ls=":", alpha=0.6,
                label=f"Static = {static_T:.0f} N")
    ax3.legend(fontsize=7.5)
    style(ax3, "Riser tension T(t)", "Time [s]", "T [N]")

    # ── Panel 4: riser twist ψ(t) ────────────────────────────────────────────
    ax4 = gax(1, 0)
    ax4.plot(t, psi, color="#ffd700", lw=1.5)
    ax4.fill_between(t, psi, alpha=0.15, color="#ffd700")
    ax4.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
    style(ax4, "Riser twist ψ(t)", "Time [s]", "ψ [deg]")

    # ── Panel 5: phase portrait (θ, θ̇) ──────────────────────────────────────
    ax5 = gax(1, 1)
    sc5 = ax5.scatter(theta, tdot, c=t, cmap="plasma", s=2, alpha=0.7)
    plt.colorbar(sc5, ax=ax5, pad=0.02, label="t [s]").ax.tick_params(labelsize=7)
    ax5.axhline(0, color=TEXT, lw=0.4, alpha=0.4)
    ax5.axvline(0, color=TEXT, lw=0.4, alpha=0.4)
    style(ax5, "Phase portrait (θ, θ̇)", "θ [deg]", "θ̇ [rad/s]")

    # ── Panel 6: polar plot (canopy swing trajectory) ─────────────────────────
    ax6 = gax(1, 2, projection="polar")
    sc6 = ax6.scatter(df["phi_rad"].values, theta,
                      c=t, cmap="viridis", s=1.5, alpha=0.7)
    ax6.set_title("Polar: θ vs φ swing trace", fontweight="bold",
                  pad=20, fontsize=9)
    ax6.tick_params(labelsize=7)

    # ── Panel 7: Poincaré section ─────────────────────────────────────────────
    ax7 = gax(1, 3)
    pc = analysis["poincare"]
    if pc["theta_deg"]:
        ax7.scatter(pc["theta_deg"], pc["thetadot"],
                    color=C_PI, s=15, alpha=0.7, zorder=4)
    ax7.axhline(0, color=TEXT, lw=0.4, alpha=0.4)
    ax7.axvline(0, color=TEXT, lw=0.4, alpha=0.4)
    style(ax7, "Poincaré section (at φ=0 crossings)", "θ [deg]", "θ̇ [rad/s]")

    # ── Panel 8: Top-down ground track (canopy vs payload) ────────────────────
    ax8 = gax(2, 0)
    ax8.plot(x_c, y_c, color=C_TH, lw=1.5, label="Canopy", zorder=3)
    ax8.plot(x_p, y_p, color=C_RAW, lw=1.0, ls="--", alpha=0.8,
             label="Payload", zorder=3)
    ax8.scatter([0], [0], color=C_RAW, s=60, marker="^", zorder=5, label="Deploy")
    ax8.scatter([x_c[-1]], [y_c[-1]], color=C_TH, s=40, marker="x", zorder=5)
    ax8.scatter([x_p[-1]], [y_p[-1]], color=C_RAW, s=40, marker="*", zorder=5)
    ax8.set_aspect("equal")
    ax8.legend(fontsize=7.5)
    style(ax8, "Ground track: canopy vs payload", "East [m]", "North [m]")

    # ── Panel 9: Descent velocity ─────────────────────────────────────────────
    ax9 = gax(2, 1)
    ax9.plot(t, -vz_c, color=C_TH, lw=1.5, label="Canopy v_z")
    ax9.plot(t, df["v_payload_ms"].values, color=C_RAW, lw=1.2,
             ls="--", alpha=0.85, label="Payload |v|")
    ax9.legend(fontsize=7.5)
    style(ax9, "Descent velocity", "Time [s]", "Speed [m/s]")

    # ── Panel 10: Effective area A_eff(t) ─────────────────────────────────────
    ax10 = gax(2, 2)
    ax10.fill_between(t, df["A_eff_m2"].values, alpha=0.2, color=C_TH)
    ax10.plot(t, df["A_eff_m2"].values, color=C_TH, lw=1.5, label="A_eff (tilt+twist)")
    A_nom = pend.canopy_area_m2 if pend else cfg.CANOPY_AREA_M2
    ax10.axhline(A_nom, color=TEXT, lw=0.8, ls=":", alpha=0.6,
                 label=f"A₀={A_nom:.0f} m²")
    ax10.legend(fontsize=7.5)
    style(ax10, "Effective drag area A_eff(t)", "Time [s]", "Area [m²]")

    # ── Panel 11: MC landing scatter ──────────────────────────────────────────
    ax11 = gax(2, 3)
    ax11.set_aspect("equal")
    ax11.axhline(0, color=SPINE, lw=0.5); ax11.axvline(0, color=SPINE, lw=0.5)
    ax11.scatter([0], [0], color=C_RAW, s=60, marker="^", zorder=5, label="Deploy")
    ax11.scatter([x_c[-1]], [y_c[-1]], s=40, color=C_TH, marker="x",
                 zorder=5, label=f"Canopy ({x_c[-1]:+.0f},{y_c[-1]:+.0f})")
    ax11.scatter([x_p[-1]], [y_p[-1]], s=40, color=C_PI, marker="*",
                 zorder=5, label=f"Payload ({x_p[-1]:+.0f},{y_p[-1]:+.0f})")

    if mc_df is not None and len(mc_df) > 0:
        ax11.scatter(mc_df["x_payload"], mc_df["y_payload"],
                     s=5, alpha=0.35, color=C_TH, label=f"MC n={len(mc_df)}")
        # 90% CEP ellipse
        try:
            from scipy.stats import chi2
            lx = mc_df["x_payload"].values; ly = mc_df["y_payload"].values
            cov = np.cov(lx, ly)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]; vals=vals[order]; vecs=vecs[:,order]
            angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            scale = np.sqrt(chi2.ppf(0.90, 2))
            from matplotlib.patches import Ellipse as MPE
            ell = MPE(xy=(lx.mean(), ly.mean()),
                      width=2*scale*np.sqrt(vals[0]),
                      height=2*scale*np.sqrt(vals[1]),
                      angle=angle, edgecolor=C_PI,
                      facecolor="none", lw=1.2, ls="--", label="90% CEP")
            ax11.add_patch(ell)
        except Exception:
            pass

    ax11.legend(fontsize=7)
    style(ax11, "Landing zone scatter (pendulum MC)", "East [m]", "North [m]")

    # ── Super-title ──────────────────────────────────────────────────────────
    L    = pend.riser_length_m if pend else "?"
    zeta = pend.zeta_swing     if pend else "?"
    th0  = np.degrees(pend.theta0_rad) if pend else "?"
    fig.text(0.5, 0.955,
             f"Canopy Pendulum Oscillation  —  L={L}m  ζ={zeta}  θ₀={th0:.1f}°  "
             f"T_n={analysis.get('T_natural_s', '?'):.2f}s  "
             f"n_osc={analysis.get('n_oscillations',0)}",
             ha="center", fontsize=12, fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR / "pendulum_dashboard.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Pendulum dashboard saved: {sp}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    riser_length:  float = 8.0,
    theta0_deg:    float = 14.0,
    phi0_deg:      float = 0.0,
    zeta:          float = 0.12,
    k_twist:       float = 150.0,
    wind_speed:    float = 6.0,
    wind_dir:      float = 270.0,
    at_fn               = None,
    mc_n:          int   = 150,
    verbose:       bool  = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run Phase 8 pendulum simulation.

    Parameters
    ----------
    riser_length  : riser length [m]
    theta0_deg    : initial swing angle from vertical [degrees]
    phi0_deg      : initial azimuth of swing [degrees]
    zeta          : aerodynamic damping ratio
    k_twist       : riser torsional stiffness [N·m/rad]
    wind_speed    : wind speed for trajectory [m/s]
    wind_dir      : wind direction [deg, meteorological]
    at_fn         : callable A(t) from Phase 2, or None for logistic fallback
    mc_n          : number of MC runs for landing scatter
    verbose       : print progress

    Returns
    -------
    (df, analysis)  where df is the full time-series DataFrame,
                    analysis is the oscillation characterisation dict.
    """
    import matplotlib; matplotlib.use("Agg")
    import json

    from src.phase6_trajectory import PowerLawWind
    wind = PowerLawWind(speed_ref=wind_speed, direction_deg=wind_dir)

    pend = PendulumConfig(
        riser_length_m  = riser_length,
        theta0_rad      = np.deg2rad(theta0_deg),
        phi0_rad        = np.deg2rad(phi0_deg),
        zeta_swing      = zeta,
        k_twist         = k_twist,
    )

    if verbose:
        print(f"\n[Phase 8] Canopy Pendulum Oscillation")
        print(f"  Natural period: {pend.period_s:.2f}s  "
              f"Decay τ: {pend.decay_time_s:.1f}s  "
              f"c_θ: {pend.c_theta:.1f} N·m·s/rad")

    # ── Nominal run ────────────────────────────────────────────────────────────
    df = solve_pendulum(pend, wind, at_fn=at_fn, verbose=verbose)

    out = cfg.OUTPUTS_DIR / "pendulum_results.csv"
    df.to_csv(out, index=False)
    if verbose:
        print(f"  ✓ Results saved: {out}")

    # ── Oscillation analysis ───────────────────────────────────────────────────
    analysis = analyse_oscillation(df)
    if verbose:
        print(f"\n  Oscillation analysis:")
        for k, v in analysis.items():
            if k not in ("poincare", "peak_times_s", "peak_vals_deg"):
                print(f"    {k:28s}: {v}")

    # ── MC landing scatter ─────────────────────────────────────────────────────
    mc_df = pendulum_mc_scatter(pend, wind, n=mc_n, verbose=verbose)
    mc_df.to_csv(cfg.OUTPUTS_DIR / "pendulum_mc_scatter.csv", index=False)
    if verbose:
        print(f"  ✓ MC scatter saved ({len(mc_df)} valid runs)")
        if len(mc_df):
            sx = mc_df["x_payload"].std()
            sy = mc_df["y_payload"].std()
            print(f"  Landing std: σ_E={sx:.2f}m  σ_N={sy:.2f}m  "
                  f"CEP≈{np.sqrt(sx**2+sy**2):.2f}m")

    # ── Save analysis JSON ────────────────────────────────────────────────────
    safe = {k: (v if not isinstance(v, float) or np.isfinite(v) else None)
            for k, v in analysis.items()
            if k not in ("poincare", "peak_times_s", "peak_vals_deg")}
    (cfg.OUTPUTS_DIR / "pendulum_analysis.json").write_text(
        json.dumps(safe, indent=2)
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_pendulum(df, analysis, mc_df=mc_df, pend=pend)

    return df, analysis


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 8 — Canopy Pendulum Oscillation")
    p.add_argument("--riser-length", type=float, default=8.0, help="Riser length [m]")
    p.add_argument("--theta0",       type=float, default=14.0, help="Initial swing angle [deg]")
    p.add_argument("--phi0",         type=float, default=0.0,  help="Initial azimuth [deg]")
    p.add_argument("--zeta",         type=float, default=0.12, help="Damping ratio")
    p.add_argument("--k-twist",      type=float, default=150.0,help="Torsional stiffness [N·m/rad]")
    p.add_argument("--wind",         type=float, default=6.0,  help="Wind speed [m/s]")
    p.add_argument("--wind-dir",     type=float, default=270.0,help="Wind direction [deg]")
    p.add_argument("--mc-n",         type=int,   default=150,  help="MC scatter runs")
    a = p.parse_args()
    run(riser_length=a.riser_length, theta0_deg=a.theta0, phi0_deg=a.phi0,
        zeta=a.zeta, k_twist=a.k_twist, wind_speed=a.wind,
        wind_dir=a.wind_dir, mc_n=a.mc_n)
