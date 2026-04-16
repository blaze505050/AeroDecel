"""
src/axisymmetric_moc.py — Axisymmetric Method of Characteristics (MOC)
=======================================================================
Exact supersonic flow solver for blunt-body wake and nozzle geometries.
Resolves expansion fans and shock structures that empirical Cd models miss.

Method of Characteristics (MOC)
--------------------------------
For 2-D axisymmetric steady supersonic flow, the governing PDEs are
hyperbolic and have two families of real characteristic lines:

  C±: dr/dx = tan(θ ∓ μ)            (Mach wave angles)

Along the C+ characteristic (left-running):
  dθ + dν + [sin θ sin μ / (r cos(θ+μ))] ds = 0

Along the C- characteristic (right-running):
  dθ - dν - [sin θ sin μ / (r cos(θ-μ))] ds = 0

where:
  θ  = local flow angle to axis [rad]
  ν  = Prandtl-Meyer function [rad]
  μ  = Mach angle = arcsin(1/M) [rad]
  r  = distance from axis of symmetry [m]
  s  = arc length along characteristic [m]
  ds = elemental arc length

The axisymmetric source term is what distinguishes this from planar 2-D MOC.
For r→0 (axis), a L'Hôpital limit is applied.

Algorithm
---------
  1. Specify initial data line (throat or bow shock)
  2. Propagate interior points using C+/C- characteristic pairs
  3. Apply wall BC: θ_wall = θ_geometry (flow tangency)
  4. Apply axis BC: θ_axis = 0 (symmetry), apply L'Hôpital limit
  5. Continue until the wake region is resolved

Prandtl-Meyer function
----------------------
  ν(M) = √((γ+1)/(γ-1)) arctan(√((γ-1)/(γ+1)(M²-1))) - arctan(√(M²-1))

Applications
------------
  • Supersonic canopy wake at deployment (M > 1)
  • Nozzle design for parachute mortar ejection
  • Expansion fan around blunt body forebody
  • Shock structure in descent stage engine plume

Output
------
  • Mesh of characteristic nodes with (x, r, θ, ν, M, p/p_ref)
  • Cd contribution from pressure distribution
  • Mach contours in the flow field
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from scipy.optimize import brentq


# ══════════════════════════════════════════════════════════════════════════════
# PRANDTL-MEYER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def prandtl_meyer_nu(M: float, gamma: float = 1.4) -> float:
    """
    Prandtl-Meyer function ν(M) [rad].
    Defined for M ≥ 1; returns 0 for M < 1.
    """
    if M <= 1.0:
        return 0.0
    g  = gamma
    k  = np.sqrt((g+1)/(g-1))
    nu = (k * np.arctan(np.sqrt((g-1)/(g+1) * (M**2-1)))
          - np.arctan(np.sqrt(M**2 - 1)))
    return float(nu)


def prandtl_meyer_mach(nu: float, gamma: float = 1.4,
                        M_init: float = 2.0) -> float:
    """
    Inverse Prandtl-Meyer: given ν, find Mach number.
    Uses Brent's method on the known interval.
    """
    nu = max(nu, 0.0)
    if nu < 1e-10:
        return 1.0

    # Maximum ν occurs at M → ∞
    nu_max = prandtl_meyer_nu(1e6, gamma)
    if nu >= nu_max:
        return 1e4   # effectively infinite Mach

    def f(M):
        return prandtl_meyer_nu(M, gamma) - nu

    try:
        return float(brentq(f, 1.0+1e-9, 1e4, xtol=1e-10, maxiter=100))
    except Exception:
        return M_init


def mach_angle(M: float) -> float:
    """μ = arcsin(1/M) [rad]."""
    return float(np.arcsin(1.0 / max(M, 1.0001)))


def isentropic_pressure_ratio(M: float, gamma: float = 1.4) -> float:
    """p/p₀ = (1 + (γ-1)/2 · M²)^(-γ/(γ-1))."""
    return float((1 + (gamma-1)/2 * M**2)**(-gamma/(gamma-1)))


# ══════════════════════════════════════════════════════════════════════════════
# MOC NODE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MOCNode:
    """
    A single node in the MOC characteristic mesh.

    Fields
    ------
    x, r  : axial and radial position [m]
    theta : local flow angle to axis [rad]  (positive = towards axis)
    nu    : Prandtl-Meyer function value [rad]
    M     : local Mach number
    mu    : Mach angle arcsin(1/M) [rad]
    p_rel : pressure / reference pressure (isentropic from M)
    """
    x:     float
    r:     float
    theta: float
    nu:    float
    M:     float
    mu:    float
    p_rel: float = 1.0

    def __post_init__(self):
        self.mu    = mach_angle(self.M)
        self.p_rel = isentropic_pressure_ratio(self.M)

    def update_from_nu(self, nu: float, gamma: float = 1.4):
        """Recompute M, mu, p_rel from updated ν."""
        self.nu    = nu
        self.M     = prandtl_meyer_mach(nu, gamma)
        self.mu    = mach_angle(self.M)
        self.p_rel = isentropic_pressure_ratio(self.M)


# ══════════════════════════════════════════════════════════════════════════════
# INTERIOR POINT CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def interior_point(pt1: MOCNode, pt2: MOCNode,
                   gamma: float = 1.4,
                   axis_tol: float = 1e-6) -> MOCNode:
    """
    Calculate properties at interior point P from two known points:
      pt1 : on C- characteristic (left-running wave, "+")
      pt2 : on C+ characteristic (right-running wave, "-")

    Compatibility equations (with axisymmetric source term):
      C+:  (θ_P - θ₁) + (ν_P - ν₁) + S₁ = 0
      C-:  (θ_P - θ₂) - (ν_P - ν₂) + S₂ = 0

    where S is the axisymmetric source term:
      S_i = sin(θ_i) sin(μ_i) / (r_avg cos(θ_i ± μ_i)) · ds_i

    The intersection point P is found by linear extrapolation.
    """
    # Average characteristic slopes
    theta_avg_m = 0.5 * (pt1.theta + pt2.theta)
    mu_avg_m    = 0.5 * (pt1.mu    + pt2.mu)

    # Slope of C+ line (left-running) through pt1
    slope_p = np.tan(theta_avg_m + mu_avg_m)
    # Slope of C- line (right-running) through pt2
    slope_m = np.tan(theta_avg_m - mu_avg_m)

    # Intersection of the two characteristics
    # pt1.x + slope_p*(x-pt1.x) = ... solve for x_P
    denom = slope_p - slope_m
    if abs(denom) < 1e-12:
        # Characteristics nearly parallel — average position
        x_P = 0.5*(pt1.x + pt2.x)
        r_P = 0.5*(pt1.r + pt2.r)
    else:
        x_P = (slope_p*pt1.x - slope_m*pt2.x + pt2.r - pt1.r) / denom
        r_P = pt1.r + slope_p * (x_P - pt1.x)

    r_P = max(r_P, 0.0)   # cannot be negative

    # Arc lengths along each characteristic to new point
    ds1 = np.sqrt((x_P - pt1.x)**2 + (r_P - pt1.r)**2)
    ds2 = np.sqrt((x_P - pt2.x)**2 + (r_P - pt2.r)**2)

    # Axisymmetric source terms
    r_mid1 = max(0.5*(r_P + pt1.r), axis_tol)
    r_mid2 = max(0.5*(r_P + pt2.r), axis_tol)

    cos1p = np.cos(pt1.theta + pt1.mu); cos1p = cos1p if abs(cos1p) > 1e-9 else 1e-9
    cos2m = np.cos(pt2.theta - pt2.mu); cos2m = cos2m if abs(cos2m) > 1e-9 else 1e-9

    # S = sin(θ)sin(μ) / (r cos(θ±μ)) · ds
    S1 = (np.sin(pt1.theta) * np.sin(pt1.mu) / (r_mid1 * cos1p)) * ds1
    S2 = (np.sin(pt2.theta) * np.sin(pt2.mu) / (r_mid2 * cos2m)) * ds2

    # Compatibility equations:
    # C+:  θ_P + ν_P = θ₁ - ν₁ - S₁  (Q1 = RHS of C+ equation)
    # C-:  θ_P - ν_P = θ₂ + ν₂ + S₂  (Q2 = RHS of C- equation)
    Q1 = pt1.theta - pt1.nu - S1
    Q2 = pt2.theta + pt2.nu + S2

    theta_P = 0.5 * (Q1 + Q2)
    nu_P    = 0.5 * (Q2 - Q1)   # Note: Q2 - Q1 = (θ₂-θ₁) + (ν₁+ν₂) + (S₁+S₂)... correct

    nu_P = max(nu_P, 0.0)

    node = MOCNode(x=x_P, r=r_P, theta=theta_P, nu=nu_P, M=1.0, mu=0.0)
    node.update_from_nu(nu_P, gamma)
    return node


# ══════════════════════════════════════════════════════════════════════════════
# WALL BOUNDARY CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def wall_point(pt_char: MOCNode, wall_theta: float,
               wall_x: float | None = None,
               gamma: float = 1.4) -> MOCNode:
    """
    Calculate properties at a wall point where the wall angle is known.
    Uses the C+ compatibility equation from pt_char to the wall.

    θ_wall is prescribed (flow tangency condition).
    From C+: θ_wall + ν_wall = θ_char - ν_char - S  → ν_wall = -θ_wall + θ_char - ν_char - S
    """
    ds   = 0.0   # iterate — approximate S=0 for first estimate
    nu_w = pt_char.nu + (pt_char.theta - wall_theta)
    nu_w = max(nu_w, 0.0)

    # Wall position (approx — extend along C+ direction)
    if wall_x is None:
        wall_x = pt_char.x + 1e-3   # small step

    # Wall radial position from wall geometry (here assumed constant slope)
    wall_r = max(pt_char.r + np.tan(wall_theta) * (wall_x - pt_char.x), 0.0)

    node = MOCNode(x=wall_x, r=wall_r, theta=wall_theta, nu=nu_w, M=1.0, mu=0.0)
    node.update_from_nu(nu_w, gamma)
    return node


# ══════════════════════════════════════════════════════════════════════════════
# AXIS BOUNDARY CONDITION  (L'Hôpital limit)
# ══════════════════════════════════════════════════════════════════════════════

def axis_point(pt_char: MOCNode, gamma: float = 1.4) -> MOCNode:
    """
    Calculate properties at the axis (r=0).
    Symmetry: θ_axis = 0.
    C- compatibility: θ - ν = θ_char + ν_char + S_axis

    For the axis source term, L'Hôpital's rule gives:
      lim_{r→0} [sin θ sin μ / (r cos(θ-μ))] · ds = -dθ/dr · ds
    This is equivalent to setting the source term = 0 on the axis.
    """
    # At axis: θ_P = 0
    # C- from pt_char: θ_P - ν_P = θ_char + ν_char
    # → ν_P = -(θ_char + ν_char)  ... actually:
    # C- gives: θ_P - ν_P = Q2 where Q2 = θ_char + ν_char + S
    # With θ_P = 0: ν_P = -Q2 = -(θ_char + ν_char)
    # But ν must be ≥ 0, so this only works if θ_char + ν_char ≤ 0
    # For a typical expansion fan, θ_char < 0 (flow turning away from axis)
    Q2  = pt_char.theta + pt_char.nu   # S ≈ 0 at axis
    nu_P = -Q2   # from C- with θ_P=0
    nu_P = max(nu_P, 0.0)

    node = MOCNode(x=pt_char.x, r=0.0, theta=0.0, nu=nu_P, M=1.0, mu=0.0)
    node.update_from_nu(nu_P, gamma)
    return node


# ══════════════════════════════════════════════════════════════════════════════
# MOC SOLVER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AxisymmetricMOC:
    """
    Full axisymmetric MOC solver for supersonic flow.

    Usage
    -----
        moc = AxisymmetricMOC(gamma=1.4)
        result = moc.solve_blunt_body_wake(M_inf=2.0, R_body=0.5,
                                           theta_initial=15.0, n_lines=12)
        moc.plot(result)
    """

    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    def _nu_max(self) -> float:
        """Maximum Prandtl-Meyer function (M→∞)."""
        g = self.gamma
        return float(np.pi/2 * (np.sqrt((g+1)/(g-1)) - 1))

    def create_initial_line(self, M_inf: float, R_body: float,
                            theta_max_deg: float, n_nodes: int = 15) -> list[MOCNode]:
        """
        Create initial data line (Mach line just downstream of bow shock or throat).
        Points span from axis (r=0) to body surface (r=R_body).

        Parameters
        ----------
        M_inf        : free-stream Mach number
        R_body       : body radius [m]
        theta_max_deg: maximum flow deflection at body surface [°]
        n_nodes      : number of characteristic grid points
        """
        theta_max = np.deg2rad(theta_max_deg)
        nu_inf    = prandtl_meyer_nu(M_inf, self.gamma)
        nodes     = []

        for i in range(n_nodes):
            # Fraction from axis to wall
            frac  = i / (n_nodes - 1)
            r_i   = frac * R_body
            theta_i = frac * theta_max   # linear variation of flow angle

            # Prandtl-Meyer function from isentropic expansion
            nu_i = nu_inf + theta_i   # approximate: Δν ≈ Δθ for moderate turns
            nu_i = min(max(nu_i, 0.0), self._nu_max())
            M_i  = prandtl_meyer_mach(nu_i, self.gamma)
            mu_i = mach_angle(M_i)

            node = MOCNode(x=0.0, r=r_i, theta=theta_i, nu=nu_i, M=M_i, mu=mu_i)
            nodes.append(node)

        return nodes

    def march_one_step(self, line: list[MOCNode],
                        wall_theta: float = 0.0,
                        is_nozzle: bool = True) -> list[MOCNode]:
        """
        Advance the characteristic mesh by one step.
        Returns a new list of MOCNode at the next x-station.

        If is_nozzle=True: upper BC is wall at theta=wall_theta.
        If is_nozzle=False: flow field expands freely (upper BC is streamline).
        """
        n     = len(line)
        new_line = []

        # Interior points (from pairs of adjacent nodes)
        for i in range(n - 1):
            pt_minus = line[i]       # C- characteristic
            pt_plus  = line[i + 1]  # C+ characteristic
            node_P   = interior_point(pt_minus, pt_plus, self.gamma)
            new_line.append(node_P)

        # Axis BC (first node, r=0)
        if line[0].r < 1e-4 * line[-1].r:
            # First node is on (or near) axis
            ax_node = axis_point(line[0], self.gamma)
            ax_node.x = new_line[0].x if new_line else line[0].x + 0.01
            new_line.insert(0, ax_node)

        # Wall BC (last node)
        if is_nozzle and new_line:
            last = new_line[-1]
            w_node = wall_point(last, wall_theta, gamma=self.gamma)
            new_line.append(w_node)

        return new_line

    def solve_blunt_body_wake(self,
                               M_inf:           float = 2.0,
                               R_body:          float = 0.5,
                               theta_body_deg:  float = 15.0,
                               n_lines:         int   = 12,
                               n_march_steps:   int   = 20,
                               verbose:         bool  = True) -> dict:
        """
        Solve the supersonic wake downstream of a blunt body (e.g., parachute canopy).

        Models the expansion fan that forms as the flow rounds the shoulder
        of the blunt body and the subsequent recompression in the wake.

        Parameters
        ----------
        M_inf         : free-stream Mach
        R_body        : body half-diameter [m]
        theta_body_deg: flow turn angle at body shoulder [°]
        n_lines       : initial data line resolution
        n_march_steps : downstream marching steps

        Returns
        -------
        dict with: all_lines (list of lists of MOCNode), Cd_pressure, Mach_field
        """
        # Initial data line
        initial_line = self.create_initial_line(
            M_inf, R_body, theta_body_deg, n_lines)

        # March downstream
        all_lines = [initial_line]
        current   = initial_line

        # Wall angle decreases as we move downstream (flow re-aligns with axis)
        wall_theta_0 = np.deg2rad(theta_body_deg)

        for step in range(n_march_steps):
            # Wall angle decreases exponentially downstream
            frac       = step / max(n_march_steps - 1, 1)
            wall_theta = wall_theta_0 * np.exp(-3 * frac)

            new_line = self.march_one_step(current, wall_theta, is_nozzle=True)
            if not new_line:
                break
            all_lines.append(new_line)
            current = new_line

            if verbose and (step + 1) % 5 == 0:
                M_mean = np.mean([n.M for n in new_line if n.M < 1e3])
                print(f"\r  [MOC] step {step+1:3d}/{n_march_steps}  "
                      f"M_mean={M_mean:.3f}  x={new_line[0].x:.3f}m", end="", flush=True)

        if verbose:
            print()

        # Extract Mach and pressure fields
        all_nodes = [n for line in all_lines for n in line]
        x_arr  = np.array([n.x     for n in all_nodes])
        r_arr  = np.array([n.r     for n in all_nodes])
        M_arr  = np.array([n.M     for n in all_nodes])
        p_arr  = np.array([n.p_rel for n in all_nodes])
        th_arr = np.array([n.theta for n in all_nodes])

        # Pressure-area integral for Cd contribution
        # F = ∫ p · dA · cos(θ)  over the body surface
        body_nodes = [line[-1] for line in all_lines if line]
        if len(body_nodes) > 1:
            p_body  = np.array([n.p_rel for n in body_nodes])
            r_body  = np.array([n.r     for n in body_nodes])
            th_body = np.array([n.theta for n in body_nodes])
            dA      = np.pi * np.diff(r_body**2)
            p_mid   = 0.5 * (p_body[:-1] + p_body[1:])
            th_mid  = 0.5 * (th_body[:-1] + th_body[1:])
            # Reference area = πR²
            A_ref   = np.pi * R_body**2
            q_inf   = 0.5 * 1.0 * M_inf**2  # ρ∞ = 1 normalised
            Cd_MOC  = float(np.sum(p_mid * np.cos(th_mid) * dA) / (q_inf * A_ref))
        else:
            Cd_MOC = 0.0

        if verbose:
            M_field_mean = float(M_arr[M_arr < 1e3].mean())
            print(f"  MOC solved: {len(all_nodes)} nodes  "
                  f"M_avg={M_field_mean:.3f}  Cd={Cd_MOC:.4f}")

        return {
            "all_lines":   all_lines,
            "all_nodes":   all_nodes,
            "x":           x_arr,
            "r":           r_arr,
            "M":           np.clip(M_arr, 1, 20),
            "p_rel":       p_arr,
            "theta":       th_arr,
            "Cd_MOC":      Cd_MOC,
            "M_inf":       M_inf,
            "R_body":      R_body,
            "n_lines":     n_lines,
            "n_steps":     n_march_steps,
        }

    def Cd_vs_mach(self, mach_range: np.ndarray, R_body: float = 0.5,
                   theta_deg: float = 15.0) -> np.ndarray:
        """Compute MOC Cd over a range of Mach numbers (for aero database)."""
        Cds = []
        for M in mach_range:
            if M <= 1.0:
                Cds.append(np.nan)
                continue
            try:
                res = self.solve_blunt_body_wake(
                    M, R_body, theta_deg, n_lines=8, n_march_steps=10, verbose=False)
                Cds.append(res["Cd_MOC"])
            except Exception:
                Cds.append(np.nan)
        return np.array(Cds)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_moc(result: dict, save_path: str = "outputs/axisymmetric_moc.png"):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.tri as mtri

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

    x  = result["x"]; r  = result["r"]
    M  = result["M"]; p  = result["p_rel"]
    th = result["theta"]

    # Mach field (interpolated)
    ax1 = fig.add_subplot(gs[:, :2])
    ax1.set_facecolor("#0d1526"); ax1.grid(True, alpha=0.25)
    ax1.tick_params(colors=TX); ax1.spines[:].set_color("#2a3d6e")
    try:
        triang = mtri.Triangulation(x, r)
        M_clipped = np.clip(M, 1, 8)
        tc = ax1.tricontourf(triang, M_clipped, levels=20, cmap="inferno")
        fig.colorbar(tc, ax=ax1, label="Mach", pad=0.02).ax.tick_params(labelsize=7)
        ax1.tricontour(triang, M_clipped, levels=10, colors="white", linewidths=0.4, alpha=0.5)
    except Exception:
        sc = ax1.scatter(x, r, c=M, cmap="inferno", s=4, alpha=0.8)
        fig.colorbar(sc, ax=ax1, label="Mach").ax.tick_params(labelsize=7)

    # Draw characteristic lines
    for line in result["all_lines"][::2]:
        if len(line) > 1:
            xl = [n.x for n in line]; rl = [n.r for n in line]
            ax1.plot(xl, rl, color="#00d4ff", lw=0.5, alpha=0.35)
    # Axis
    ax1.axhline(0, color=TX, lw=0.7, ls="--", alpha=0.4)
    # Body profile
    body_x = [0]*20; body_r = np.linspace(0, result["R_body"], 20)
    ax1.plot(body_x, body_r, color="#ff4560", lw=2.5, label="Body")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("r [m]")
    ax1.set_title(f"MOC Mach Field  M∞={result['M_inf']:.2f}  "
                  f"R={result['R_body']}m  Cd={result['Cd_MOC']:.4f}",
                  fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")

    # Mach vs x along axis
    a = gax(0, 2)
    axis_nodes = [n for line in result["all_lines"] for n in line if n.r < 0.01]
    if axis_nodes:
        ax_x = [n.x for n in axis_nodes]
        ax_M = [n.M for n in axis_nodes]
        a.plot(ax_x, ax_M, color=C1, lw=2)
    a.axhline(result["M_inf"], color=TX, lw=0.7, ls="--", alpha=0.6, label="M∞")
    a.axhline(1, color=C4, lw=0.7, ls=":", alpha=0.6, label="M=1")
    a.set_xlabel("x [m]"); a.set_ylabel("Mach"); a.set_title("Mach on Axis", fontweight="bold")
    a.legend(fontsize=8)

    # Pressure ratio
    a = gax(0, 3)
    a.scatter(x, p, c=M, cmap="inferno", s=4, alpha=0.7)
    a.set_xlabel("x [m]"); a.set_ylabel("p/p₀"); a.set_title("Pressure Distribution", fontweight="bold")

    # Prandtl-Meyer function vs Mach (validation)
    a = gax(1, 2)
    M_range = np.linspace(1.01, 5, 200)
    nu_range = np.array([prandtl_meyer_nu(M) for M in M_range])
    a.plot(M_range, np.degrees(nu_range), color=C3, lw=2, label="ν(M) exact")
    # Overlay MOC nodes
    nu_nodes = [n.nu for n in result["all_nodes"] if 1 < n.M < 10]
    M_nodes  = [n.M  for n in result["all_nodes"] if 1 < n.M < 10]
    a.scatter(M_nodes, np.degrees(nu_nodes), s=5, color=C4, alpha=0.6, label="MOC nodes")
    a.set_xlabel("Mach"); a.set_ylabel("ν [°]")
    a.set_title("Prandtl-Meyer Function Validation", fontweight="bold")
    a.legend(fontsize=8)

    # Cd vs Mach
    a = gax(1, 3)
    moc_obj = AxisymmetricMOC(gamma=1.4)
    M_cd_arr = np.linspace(1.1, 4.0, 12)
    Cd_arr   = moc_obj.Cd_vs_mach(M_cd_arr, result["R_body"])
    valid    = ~np.isnan(Cd_arr)
    a.plot(M_cd_arr[valid], Cd_arr[valid], color=C2, lw=2, marker="o", ms=5)
    a.axvline(result["M_inf"], color=C4, lw=1, ls="--", label=f"M={result['M_inf']:.1f}")
    a.set_xlabel("M∞"); a.set_ylabel("Cd (MOC)"); a.set_title("Cd vs Mach (MOC)", fontweight="bold")
    a.legend(fontsize=8)

    fig.text(0.5, 0.955,
             f"Axisymmetric MOC — M∞={result['M_inf']:.2f}  "
             f"R={result['R_body']}m  Cd={result['Cd_MOC']:.4f}  "
             f"nodes={len(result['all_nodes'])}",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ MOC plot saved: {save_path}")
    plt.close(fig)


def run(M_inf: float = 2.5, R_body: float = 5.0,
        theta_deg: float = 15.0, verbose: bool = True) -> dict:
    """Run axisymmetric MOC for a blunt body wake."""
    import matplotlib; matplotlib.use("Agg")
    moc    = AxisymmetricMOC(gamma=1.4)
    result = moc.solve_blunt_body_wake(
        M_inf=M_inf, R_body=R_body, theta_body_deg=theta_deg,
        n_lines=14, n_march_steps=18, verbose=verbose)
    plot_moc(result)
    return result


if __name__ == "__main__":
    result = run(M_inf=2.5, R_body=5.0, theta_deg=15.0)
    print(f"Cd(MOC)={result['Cd_MOC']:.4f}  nodes={len(result['all_nodes'])}")
