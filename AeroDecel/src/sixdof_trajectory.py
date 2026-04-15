"""
src/sixdof_trajectory.py — Full 6-DOF Rigid Body EDL Trajectory
================================================================
State vector (13-D):
    [0:3]  position      r_I       [m]        inertial frame
    [3:6]  velocity      v_I       [m/s]      inertial frame
    [6:10] quaternion    q_BI      [-]        body←inertial rotation
    [10:13] angular rate  ω_B      [rad/s]    body frame

Physics
-------
  Translation:  m·ṙ = F_aero_I + m·g_I
  Rotation:     I·ω̇ = M_aero_B - ω×(I·ω)   (Euler equations)
  Kinematics:   q̇ = 0.5·Ξ(q)·ω_B
  Baumgarte:    quaternion normalised every step (|q|=1 constraint)

Aerodynamic model
-----------------
  Force:   F = q_dyn·A·[Cd·(-v̂) + CL·(l̂×v̂)×v̂ + CS·l̂×v̂]
  Moments: M = q_dyn·A·D·[Cm·ê_pitch + Cn·ê_yaw + Cl·ê_roll]

  Cd, CL, CS  depend on α (angle of attack), β (sideslip), Mach
  Cm, Cn, Cl  depend on α, β, angular rate (p,q,r) — stability derivatives

Stability derivatives (Newtonian + empirical for blunt bodies)
--------------------------------------------------------------
  Cmα < 0  →  aerodynamically stable  (restoring pitch moment)
  Cmq < 0  →  pitch damping
  Cnβ > 0  →  directional stability
  Clp < 0  →  roll damping

Canopy rigid-body inertia
-------------------------
  Models the canopy as a flat disk + suspended payload.
  Inertia tensor I computed from geometry at each time step
  (changes as canopy inflates: A(t) grows).

Outputs
-------
  Full time-series: position, velocity, attitude (Euler angles),
  angular rates, angle of attack, sideslip, Mach, g-load, stability margins
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# QUATERNION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p⊗q.  Convention: q = [qw, qx, qy, qz]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix C_BI (body ← inertial) from quaternion."""
    qw, qx, qy, qz = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy+qw*qz), 2*(qx*qz-qw*qy)],
        [2*(qx*qy-qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz+qw*qx)],
        [2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), 1-2*(qx**2+qy**2)],
    ])


def dcm_to_euler(C: np.ndarray) -> np.ndarray:
    """ZYX Euler angles [roll, pitch, yaw] from DCM in radians."""
    pitch = np.arcsin(np.clip(-C[2, 0], -1, 1))
    roll  = np.arctan2(C[2, 1], C[2, 2])
    yaw   = np.arctan2(C[1, 0], C[0, 0])
    return np.array([roll, pitch, yaw])


def Xi_matrix(q: np.ndarray) -> np.ndarray:
    """4×3 kinematic matrix: q̇ = 0.5·Ξ(q)·ω."""
    qw, qx, qy, qz = q
    return 0.5 * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
    ])


# ══════════════════════════════════════════════════════════════════════════════
# VEHICLE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleConfig:
    """Physical properties of the EDL vehicle + canopy system."""

    # Mass properties
    mass_kg:          float = 900.0    # total mass [kg]
    payload_mass_kg:  float = 800.0    # payload alone
    canopy_mass_kg:   float = 100.0    # canopy mass

    # Reference geometry (at full inflation)
    canopy_area_m2:   float = 78.5     # A_ref [m²]
    nose_radius_m:    float = 4.5      # nose radius [m]
    D_ref_m:          float = 10.0     # reference diameter [m]
    riser_length_m:   float = 15.0     # canopy riser length [m]

    # Aerodynamic coefficients (body frame, angle-of-attack dependent)
    Cd0:   float = 1.70    # zero-α drag coefficient
    CL_alpha: float = 0.20 # lift curve slope [1/rad]
    CS_beta:  float = 0.15 # side force slope [1/rad]

    # Stability derivatives [1/rad] or [rad/rad]
    Cm_alpha: float = -0.25   # pitch stiffness (negative = stable)
    Cm_q:     float = -0.80   # pitch damping (Cmq + Cmalpha_dot)
    Cn_beta:  float =  0.12   # yaw stiffness
    Cn_r:     float = -0.60   # yaw damping
    Cl_p:     float = -0.40   # roll damping

    # Inertia tensor [kg·m²] at full inflation (symmetric about z)
    Ixx:  float = 2500.0    # roll inertia
    Iyy:  float = 2500.0    # pitch inertia
    Izz:  float = 1800.0    # yaw inertia
    Ixz:  float = 0.0       # cross product (axisymmetric → 0)

    # Inflation model
    t_inflation_s: float = 3.0   # time to full inflation

    @property
    def I_tensor(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    def A_inflated(self, t: float) -> float:
        """Canopy area vs time (logistic inflation)."""
        k  = 8.0 / self.t_inflation_s
        t0 = self.t_inflation_s * 0.55
        A  = self.canopy_area_m2 / (1 + np.exp(-k*(t-t0)))**0.5
        return float(np.clip(A, 0.01, self.canopy_area_m2))

    def I_tensor_at(self, t: float) -> np.ndarray:
        """Inertia tensor scaled with inflated canopy area."""
        f = self.A_inflated(t) / self.canopy_area_m2
        Ixx_t = self.payload_mass_kg*1.5 + self.canopy_mass_kg * self.D_ref_m**2 * f / 4
        Iyy_t = Ixx_t
        Izz_t = self.payload_mass_kg*1.2 + self.canopy_mass_kg * self.D_ref_m**2 * f / 8
        return np.diag([Ixx_t, Iyy_t, Izz_t])


# ══════════════════════════════════════════════════════════════════════════════
# AERODYNAMIC MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AeroModel:
    """
    6-DOF aerodynamic forces and moments for blunt-body EDL capsule.
    Newtonian aerodynamics + empirical stability derivatives.
    """

    def __init__(self, cfg: VehicleConfig, planet_atm):
        self.cfg = cfg
        self.atm = planet_atm

    def _mach(self, v_mag: float, altitude_m: float) -> float:
        return self.atm.mach_number(v_mag, altitude_m)

    def _cd(self, alpha: float, M: float, A: float) -> float:
        """Drag coefficient: Newtonian + compressibility + angle effect."""
        Cd_base = self.cfg.Cd0 * (1 + 0.15 * alpha**2)
        # Prandtl-Glauert (subsonic)
        if M < 0.8:
            pg = 1 / np.sqrt(max(1 - M**2, 0.01))
            Cd_base *= (1 + 0.25*(pg-1))
        # Area inflation factor
        A_frac = float(np.clip(A / max(self.cfg.canopy_area_m2, 1e-3), 0, 1))
        return Cd_base * A_frac

    def forces_moments_body(
        self, v_body: np.ndarray, omega: np.ndarray,
        altitude_m: float, t: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute aerodynamic force (body frame) and moment (body frame).
        Also returns alpha, beta.

        Parameters
        ----------
        v_body   : velocity in body frame [m/s]
        omega    : angular rate in body frame [rad/s]
        altitude_m: altitude
        t        : time (for inflation)

        Returns
        -------
        F_aero_B, M_aero_B, alpha, beta
        """
        v_mag = np.linalg.norm(v_body)
        if v_mag < 0.1:
            return np.zeros(3), np.zeros(3), 0.0, 0.0

        # Convention: body x-axis points in thrust direction (nose forward)
        # For a canopy vehicle, x points down (into flow)
        u, v_, w = v_body[0], v_body[1], v_body[2]

        # Angle of attack and sideslip
        alpha = float(np.arctan2(w, max(u, 1e-6)))   # pitch AoA
        beta  = float(np.arcsin(np.clip(v_ / v_mag, -1, 1)))  # sideslip

        rho = self.atm.density(max(altitude_m, 0))
        A   = self.cfg.A_inflated(t)
        M   = self._mach(v_mag, altitude_m)
        q_dyn = 0.5 * rho * v_mag**2

        # === FORCES (body frame) ===
        v_hat = v_body / v_mag

        # Drag: opposite to velocity
        Cd  = self._cd(alpha, M, A)
        F_drag_B = -q_dyn * A * Cd * v_hat

        # Lift: perpendicular to velocity in pitch plane
        CL = self.cfg.CL_alpha * alpha
        # Lift direction: perpendicular to v in the x-z plane, upward
        if abs(u) > 1e-6 or abs(w) > 1e-6:
            pitch_plane_normal = np.array([0, 1, 0])   # y-axis
            lift_dir = np.cross(pitch_plane_normal, v_hat)
            lift_dir /= max(np.linalg.norm(lift_dir), 1e-9)
        else:
            lift_dir = np.array([0, 0, 1])
        F_lift_B = q_dyn * A * CL * lift_dir

        # Side force
        CS = self.cfg.CS_beta * beta
        side_dir = np.array([0, 1, 0])
        F_side_B = q_dyn * A * CS * side_dir

        F_aero_B = F_drag_B + F_lift_B + F_side_B

        # === MOMENTS (body frame) ===
        D = self.cfg.D_ref_m
        p, q_ang, r = omega[0], omega[1], omega[2]
        omega_hat = 0.5 * v_mag / D   # normalisation for rate derivatives

        # Pitch moment (about body y)
        Cm = (self.cfg.Cm_alpha * alpha
              + self.cfg.Cm_q * q_ang / max(omega_hat, 1e-6))
        Mx_pitch = q_dyn * A * D * Cm

        # Yaw moment (about body z)
        Cn = (self.cfg.Cn_beta * beta
              + self.cfg.Cn_r * r / max(omega_hat, 1e-6))
        Mz_yaw = q_dyn * A * D * Cn

        # Roll moment (about body x)
        Cl = self.cfg.Cl_p * p / max(omega_hat, 1e-6)
        Mx_roll = q_dyn * A * D * Cl

        M_aero_B = np.array([Mx_roll, Mx_pitch, Mz_yaw])

        return F_aero_B, M_aero_B, alpha, beta


# ══════════════════════════════════════════════════════════════════════════════
# 6-DOF ODE RIGHT-HAND SIDE
# ══════════════════════════════════════════════════════════════════════════════

class SixDOFDynamics:
    """
    Full 6-DOF rigid body dynamics for EDL.

    State: x = [r_I(3), v_I(3), q_BI(4), ω_B(3)]  → 13 elements
    """

    BAUMGARTE_BETA = 2.0    # quaternion constraint stabilisation

    def __init__(self, vehicle: VehicleConfig, planet_atm):
        self.veh  = vehicle
        self.atm  = planet_atm
        self.aero = AeroModel(vehicle, planet_atm)
        self._R   = planet_atm.radius_m   # planet radius for gravity

    def _gravity_inertial(self, r_I: np.ndarray) -> np.ndarray:
        """Gravity in inertial frame: g = -g_surface * R²/r² * r̂."""
        r_mag = max(np.linalg.norm(r_I), self._R)
        g_mag = self.atm.gravity_ms2 * (self._R / r_mag)**2
        return -g_mag * r_I / r_mag

    def _altitude(self, r_I: np.ndarray) -> float:
        return float(np.linalg.norm(r_I) - self._R)

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        r_I  = x[0:3]
        v_I  = x[3:6]
        q    = x[6:10] / max(np.linalg.norm(x[6:10]), 1e-12)  # normalise
        omega= x[10:13]

        altitude = max(self._altitude(r_I), 0.0)
        v_mag    = np.linalg.norm(v_I)

        # ── Rotation matrix (body ← inertial) ────────────────────────────────
        C_BI = quat_to_dcm(q)

        # ── Velocity in body frame ─────────────────────────────────────────────
        v_B = C_BI @ v_I

        # ── Aero forces & moments (body frame) ───────────────────────────────
        F_aero_B, M_aero_B, alpha, beta = self.aero.forces_moments_body(
            v_B, omega, altitude, t)

        # ── Transform aero force to inertial ──────────────────────────────────
        C_IB = C_BI.T   # inertial ← body
        F_aero_I = C_IB @ F_aero_B

        # ── Gravity ───────────────────────────────────────────────────────────
        g_I = self._gravity_inertial(r_I)

        # ── Translational EOM (inertial) ──────────────────────────────────────
        r_dot = v_I.copy()
        v_dot = F_aero_I / self.veh.mass_kg + g_I

        # ── Attitude kinematics q̇ = 0.5·Ξ(q)·ω ──────────────────────────────
        Xi = Xi_matrix(q)
        q_dot = Xi @ omega
        # Baumgarte stabilisation: |q|=1
        q_dot -= self.BAUMGARTE_BETA * (np.dot(q, q) - 1.0) * q

        # ── Euler's rotation equations I·ω̇ = M - ω×(I·ω) ────────────────────
        I   = self.veh.I_tensor_at(t)
        Iw  = I @ omega
        M_gyro = np.cross(omega, Iw)
        try:
            omega_dot = np.linalg.solve(I, M_aero_B - M_gyro)
        except np.linalg.LinAlgError:
            omega_dot = np.zeros(3)

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot])


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_6dof(
    vehicle:    VehicleConfig,
    planet_atm,
    r0_I:    np.ndarray,      # initial position inertial [m]
    v0_I:    np.ndarray,      # initial velocity inertial [m/s]
    q0:      np.ndarray,      # initial quaternion [qw,qx,qy,qz]
    omega0:  np.ndarray,      # initial angular rate body [rad/s]
    t_max:   float = 600.0,
    dt_out:  float = 0.5,
    verbose: bool  = True,
) -> "pd.DataFrame":
    """
    Integrate the 6-DOF equations of motion.

    Returns
    -------
    DataFrame with columns: t, x, y, z, vx, vy, vz, qw, qx, qy, qz,
    p, q, r, altitude, v_total, alpha_deg, beta_deg,
    roll_deg, pitch_deg, yaw_deg, g_load, Mach
    """
    import pandas as pd

    x0   = np.concatenate([r0_I, v0_I, q0/np.linalg.norm(q0), omega0])
    dyn  = SixDOFDynamics(vehicle, planet_atm)
    R_p  = planet_atm.radius_m

    def ground_event(t, x):
        return np.linalg.norm(x[0:3]) - R_p - 10.0
    ground_event.terminal  = True
    ground_event.direction = -1

    t_eval = np.arange(0, t_max, dt_out)
    if len(t_eval)==0 or t_eval[-1]<t_max*0.5:
        t_eval = np.linspace(0, t_max*0.95, max(int(t_max/dt_out),10))

    if verbose:
        v_mag = np.linalg.norm(v0_I)
        alt0  = np.linalg.norm(r0_I) - R_p
        print(f"\n[6-DOF] Integration  v0={v_mag/1e3:.2f}km/s  h0={alt0/1e3:.1f}km  "
              f"t_max={t_max}s  planet={planet_atm.name}")

    sol = solve_ivp(
        dyn.rhs, (0, t_max), x0,
        method="DOP853",
        t_eval=t_eval,
        events=ground_event,
        rtol=1e-6, atol=1e-8,
        dense_output=False,
        max_step=dt_out * 2,
    )

    t = sol.t
    r_I   = sol.y[0:3, :].T
    v_I   = sol.y[3:6, :].T
    q_all = sol.y[6:10, :].T
    omega = sol.y[10:13, :].T

    # Normalise quaternions
    q_norms = np.linalg.norm(q_all, axis=1, keepdims=True)
    q_all  /= np.maximum(q_norms, 1e-12)

    alt    = np.linalg.norm(r_I, axis=1) - R_p
    v_mag  = np.linalg.norm(v_I, axis=1)

    # Euler angles and AoA/sideslip at each step
    rolls, pitches, yaws = [], [], []
    alphas, betas = [], []
    g_loads, machs = [], []

    for i in range(len(t)):
        C = quat_to_dcm(q_all[i])
        eu = dcm_to_euler(C)
        rolls.append(eu[0]); pitches.append(eu[1]); yaws.append(eu[2])
        v_b = C @ v_I[i]
        a_  = float(np.degrees(np.arctan2(v_b[2], max(v_b[0], 1e-6))))
        b_  = float(np.degrees(np.arcsin(np.clip(v_b[1]/max(v_mag[i],1), -1,1))))
        alphas.append(a_); betas.append(b_)
        M_  = planet_atm.mach_number(float(v_mag[i]), max(float(alt[i]), 0))
        machs.append(M_)
        # g-load = |aero + gravity| / g_surface
        F_  = dyn.aero.forces_moments_body(v_b, omega[i], max(float(alt[i]),0), t[i])
        g_loads.append(np.linalg.norm(F_[0]) / (vehicle.mass_kg * planet_atm.gravity_ms2))

    df = pd.DataFrame({
        "time_s":    t,
        "x_m":       r_I[:, 0],  "y_m":    r_I[:, 1],  "z_m":    r_I[:, 2],
        "vx_ms":     v_I[:, 0],  "vy_ms":  v_I[:, 1],  "vz_ms":  v_I[:, 2],
        "qw": q_all[:,0], "qx": q_all[:,1], "qy": q_all[:,2], "qz": q_all[:,3],
        "p_rads":    omega[:, 0], "q_rads": omega[:, 1], "r_rads": omega[:, 2],
        "altitude_m": np.clip(alt, 0, None),
        "v_ms":      v_mag,
        "alpha_deg": alphas,
        "beta_deg":  betas,
        "roll_deg":  np.degrees(rolls),
        "pitch_deg": np.degrees(pitches),
        "yaw_deg":   np.degrees(yaws),
        "g_load":    g_loads,
        "Mach":      machs,
        "A_canopy_m2": [vehicle.A_inflated(ti) for ti in t],
    })

    if verbose:
        v_f = float(v_mag[-1]); h_f = float(alt[-1])
        print(f"  Done: t_f={t[-1]:.1f}s  v_f={v_f:.2f}m/s  h_f={h_f:.0f}m")
        print(f"  Max alpha: {max(abs(a) for a in alphas):.2f}°  "
              f"Max g-load: {max(g_loads):.2f}g")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def stability_analysis(vehicle: VehicleConfig, planet_atm,
                        v_range: np.ndarray, h_range: np.ndarray) -> dict:
    """
    Compute aerodynamic stability margins over a (v, h) grid.

    Returns
    -------
    dict with:
      v, h grids
      static_margin: Cm_alpha / CL_alpha  (negative = stable)
      dynamic_margin: pitch damping ratio ζ
      dutch_roll_freq: [rad/s] dutch-roll natural frequency
      stable: bool array — True where dynamically stable
    """
    Vg, Hg = np.meshgrid(v_range, h_range)
    n_v, n_h = len(v_range), len(h_range)

    static_margin  = np.full((n_h, n_v), vehicle.Cm_alpha / max(vehicle.CL_alpha, 1e-9))
    dynamic_margin = np.zeros((n_h, n_v))
    omega_dr       = np.zeros((n_h, n_v))

    A_ref  = vehicle.canopy_area_m2
    D_ref  = vehicle.D_ref_m
    mass   = vehicle.mass_kg

    for i, h in enumerate(h_range):
        rho = planet_atm.density(max(h, 0))
        for j, v in enumerate(v_range):
            q_dyn  = 0.5 * rho * v**2
            M_aero = q_dyn * A_ref   # [N]

            # Pitch damping: ω_n = sqrt(-Cm_alpha * q*A*D / Iyy)
            if vehicle.Cm_alpha < 0:
                k_pitch  = -vehicle.Cm_alpha * M_aero * D_ref / vehicle.Iyy
                omega_n  = float(np.sqrt(max(k_pitch, 0)))
                c_damp   = -vehicle.Cm_q * M_aero * D_ref**2 / (2 * vehicle.Iyy)
                zeta     = c_damp / max(2 * vehicle.Iyy * omega_n, 1e-9)
            else:
                omega_n = 0.0; zeta = -1.0

            # Dutch roll natural frequency
            Cnb_val = vehicle.Cn_beta * M_aero * D_ref
            omega_DR = float(np.sqrt(max(Cnb_val * v / (mass * vehicle.Iyy), 0)))

            dynamic_margin[i, j] = zeta
            omega_dr[i, j]       = omega_DR

    stable = (static_margin < 0) & (dynamic_margin > 0)

    return {
        "v":             v_range,
        "h":             h_range,
        "static_margin": static_margin,
        "dynamic_margin":dynamic_margin,
        "dutch_roll_Hz": omega_dr / (2*np.pi),
        "stable":        stable,
        "Cm_alpha":      vehicle.Cm_alpha,
        "Cm_q":          vehicle.Cm_q,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_sixdof(df, stability=None, save_path=None):
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
    TX="#c8d8f0"; BG="#080c14"; AX="#0d1526"; SP="#2a3d6e"
    C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"; CR="#ff4560"

    t  = df["time_s"].values
    fig = plt.figure(figsize=(22, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.48, wspace=0.38,
                            top=0.91, bottom=0.06, left=0.05, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor(AX); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color(SP)
        return a

    # v(t)
    a = gax(0, 0)
    a.fill_between(t, df["v_ms"]/1e3, alpha=0.15, color=C1)
    a.plot(t, df["v_ms"]/1e3, color=C1, lw=2)
    a.set_title("Velocity", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("v [km/s]")

    # h(t)
    a = gax(0, 1)
    a.fill_between(t, df["altitude_m"]/1e3, alpha=0.15, color="#9d60ff")
    a.plot(t, df["altitude_m"]/1e3, color="#9d60ff", lw=2)
    a.set_title("Altitude", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("h [km]")

    # Euler angles
    a = gax(0, 2)
    a.plot(t, df["roll_deg"],  color=C1,  lw=1.5, label="Roll")
    a.plot(t, df["pitch_deg"], color=C2,  lw=1.5, label="Pitch")
    a.plot(t, df["yaw_deg"],   color=C3,  lw=1.5, label="Yaw")
    a.axhline(0, color=TX, lw=0.5, alpha=0.4)
    a.legend(fontsize=7.5)
    a.set_title("Euler Angles", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("Angle [°]")

    # α, β
    a = gax(0, 3)
    a.plot(t, df["alpha_deg"], color=C4, lw=1.8, label="α (AoA)")
    a.plot(t, df["beta_deg"],  color=CR, lw=1.5, label="β (sideslip)")
    a.axhline(0, color=TX, lw=0.5, alpha=0.4)
    a.legend(fontsize=7.5)
    a.set_title("Aerodynamic Angles", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("Angle [°]")

    # Angular rates
    a = gax(1, 0)
    a.plot(t, np.degrees(df["p_rads"]), color=C1, lw=1.5, label="p (roll)")
    a.plot(t, np.degrees(df["q_rads"]), color=C2, lw=1.5, label="q (pitch)")
    a.plot(t, np.degrees(df["r_rads"]), color=C3, lw=1.5, label="r (yaw)")
    a.axhline(0, color=TX, lw=0.5, alpha=0.4)
    a.legend(fontsize=7.5)
    a.set_title("Angular Rates", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("[°/s]")

    # g-load
    a = gax(1, 1)
    a.fill_between(t, df["g_load"], alpha=0.2, color=CR)
    a.plot(t, df["g_load"], color=CR, lw=1.8)
    a.axhline(15, color=C4, lw=0.8, ls="--", alpha=0.7, label="15g limit")
    a.legend(fontsize=7.5)
    a.set_title("G-Load", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("g")

    # Mach
    a = gax(1, 2)
    a.fill_between(t, df["Mach"], alpha=0.2, color=C4)
    a.plot(t, df["Mach"], color=C4, lw=1.8)
    a.axhline(1.0, color=TX, lw=0.7, ls="--", alpha=0.5, label="M=1")
    a.legend(fontsize=7.5)
    a.set_title("Mach Number", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("M")

    # Canopy inflation
    a = gax(1, 3)
    a.fill_between(t, df["A_canopy_m2"], alpha=0.2, color="#ffd700")
    a.plot(t, df["A_canopy_m2"], color="#ffd700", lw=1.8)
    a.set_title("Canopy Area A(t)", fontweight="bold"); a.set_xlabel("t [s]"); a.set_ylabel("A [m²]")

    # 3-D trajectory (x-z plane)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        a3 = fig.add_subplot(gs[2, :2], projection="3d")
        a3.set_facecolor(AX)
        sc = a3.scatter(df["x_m"]/1e6, df["y_m"]/1e6, df["altitude_m"]/1e3,
                        c=t, cmap="plasma", s=3, alpha=0.8)
        fig.colorbar(sc, ax=a3, pad=0.1, shrink=0.6, label="t [s]").ax.tick_params(labelsize=7)
        a3.set_xlabel("x [Mm]"); a3.set_ylabel("y [Mm]"); a3.set_zlabel("h [km]")
        a3.set_title("3-D Trajectory", fontweight="bold")
    except Exception:
        pass

    # Stability — dynamic margin vs v
    a_s = gax(2, 2)
    if stability:
        v = stability["v"]
        zeta_h0 = stability["dynamic_margin"][0, :]   # at lowest altitude
        a_s.plot(v/1e3, zeta_h0, color=C3, lw=2, label="ζ (pitch damping)")
        a_s.axhline(0, color=CR, lw=1.2, ls="--", label="Stability boundary")
        a_s.fill_between(v/1e3, np.minimum(zeta_h0, 0), alpha=0.3, color=CR)
        a_s.legend(fontsize=7.5)
    a_s.set_title("Pitch Damping Ratio ζ", fontweight="bold")
    a_s.set_xlabel("v [km/s]"); a_s.set_ylabel("ζ")

    # Phase portrait α vs p
    a = gax(2, 3)
    sc2 = a.scatter(df["alpha_deg"], np.degrees(df["p_rads"]),
                    c=t, cmap="viridis", s=3, alpha=0.8)
    fig.colorbar(sc2, ax=a, pad=0.02, label="t [s]").ax.tick_params(labelsize=7)
    a.axhline(0, color=TX, lw=0.4, alpha=0.4)
    a.axvline(0, color=TX, lw=0.4, alpha=0.4)
    a.set_title("Phase: α vs roll rate", fontweight="bold")
    a.set_xlabel("α [°]"); a.set_ylabel("p [°/s]")

    fig.text(0.5, 0.955,
             f"6-DOF EDL Trajectory  |  v_f={df['v_ms'].iloc[-1]:.1f}m/s  "
             f"Max-g={df['g_load'].max():.1f}  "
             f"Max-α={df['alpha_deg'].abs().max():.1f}°",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    sp = save_path or "outputs/sixdof_trajectory.png"
    from pathlib import Path; Path(sp).parent.mkdir(exist_ok=True)
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ 6-DOF plot saved: {sp}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(planet_atm, vehicle: VehicleConfig = None,
        entry_speed_ms: float = 5800.0,
        entry_alt_m:    float = 125_000.0,
        entry_fpa_deg:  float = -15.0,
        entry_heading_deg: float = 0.0,
        initial_rates_rads: np.ndarray = None,
        verbose: bool = True) -> "pd.DataFrame":
    """
    Run 6-DOF EDL simulation from entry interface to landing/timeout.

    Parameters
    ----------
    planet_atm    : PlanetaryAtmosphere instance
    vehicle       : VehicleConfig (default Mars EDL config)
    entry_speed_ms: entry velocity magnitude [m/s]
    entry_alt_m   : entry altitude AGL [m]
    entry_fpa_deg : flight path angle [°] (negative = descending)
    entry_heading_deg: heading [°] (0 = north)

    Returns
    -------
    DataFrame with full 6-DOF time series
    """
    import matplotlib; matplotlib.use("Agg")
    import pandas as pd

    if vehicle is None:
        vehicle = VehicleConfig()

    R_p  = planet_atm.radius_m
    fpa  = np.deg2rad(entry_fpa_deg)
    hdg  = np.deg2rad(entry_heading_deg)

    # Initial position: on planet's x-axis at entry altitude
    r0 = np.array([R_p + entry_alt_m, 0.0, 0.0])

    # Initial velocity (inertial): speed along entry direction
    # Entry in x-z plane: vx = v*cos(fpa)*cos(hdg), vz = v*sin(fpa)
    v0 = np.array([
        entry_speed_ms * np.cos(fpa) * np.cos(hdg),
        entry_speed_ms * np.cos(fpa) * np.sin(hdg),
        entry_speed_ms * np.sin(fpa),  # negative = downward
    ])

    # Initial attitude: point into velocity (nose into wind)
    v_hat = v0 / np.linalg.norm(v0)
    # Simple: align body x-axis with velocity → pitch quaternion
    angle  = float(np.arccos(np.clip(v_hat[0], -1, 1)))
    axis   = np.cross([1,0,0], v_hat)
    ax_norm = np.linalg.norm(axis)
    if ax_norm > 1e-9:
        axis /= ax_norm
        q0 = np.array([np.cos(angle/2),
                        *(np.sin(angle/2) * axis)])
    else:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])

    omega0 = initial_rates_rads if initial_rates_rads is not None else np.zeros(3)

    # Estimate t_max
    h0 = entry_alt_m; v_term_est = 80.0
    t_max = min(h0 / max(abs(entry_speed_ms * np.sin(abs(fpa))), v_term_est) + 300, 1200)

    df = solve_6dof(vehicle, planet_atm, r0, v0, q0, omega0,
                    t_max=t_max, dt_out=0.5, verbose=verbose)

    # Stability analysis
    v_arr = np.linspace(100, entry_speed_ms, 30)
    h_arr = np.linspace(0, entry_alt_m, 20)
    stab  = stability_analysis(vehicle, planet_atm, v_arr, h_arr)

    # Save outputs
    import matplotlib; matplotlib.use("Agg")
    fig = plot_sixdof(df, stability=stab)
    import matplotlib.pyplot as plt; plt.close(fig)

    df.to_csv("outputs/sixdof_trajectory.csv", index=False)
    if verbose:
        print(f"  ✓ 6-DOF results: outputs/sixdof_trajectory.csv  ({len(df)} steps)")

    return df, stab


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from src.planetary_atm import MarsAtmosphere
    planet = MarsAtmosphere()
    df, stab = run(planet, entry_speed_ms=5800, entry_alt_m=125_000,
                   entry_fpa_deg=-15, verbose=True)
    print(df[["time_s","altitude_m","v_ms","alpha_deg","g_load","Mach"]].tail(5))
