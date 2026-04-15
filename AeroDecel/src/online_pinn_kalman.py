"""
src/online_pinn_kalman.py — Online PINN with Kalman-PINN Fusion
================================================================
Feeds live sensor data into the PINN during descent — it updates
its Cd estimate in real time using Kalman-PINN fusion.

Architecture
------------
  The Kalman filter tracks state [Cd, t_infl] and propagates uncertainty.
  The PINN provides a physics-consistent velocity prediction given Cd.
  At each sensor update, the Kalman innovation updates the PINN warm-start.

  Measurement model:
    z_k = v_measured(t_k)
    h(x_k) = v_PINN(t_k; Cd_k, t_infl_k)   [nonlinear observation]

  EKF linearisation:
    H_k = ∂h/∂x  computed via finite-difference on PINN output

  State update:
    K_k  = P_k H_k^T (H_k P_k H_k^T + R_k)^{-1}
    x̂_k+ = x̂_k + K_k (z_k - h(x̂_k))
    P_k+ = (I - K_k H_k) P_k
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import odeint


class OnlinePINNKalman:
    """
    Extended Kalman Filter fused with a forward ODE model for real-time
    Cd estimation during EDL descent.

    State: x = [Cd, t_infl]   (2-D)
    Process noise: small drift in Cd (physical model uncertainty)
    Observation: v_measured(t)
    """

    def __init__(self, planet_atm, mass_kg: float = 900.0,
                 area_m2: float = 78.5, alt0_m: float = 125_000.0,
                 v0_ms: float = 5800.0,
                 Cd_init: float = 1.7, ti_init: float = 3.0,
                 P_init: np.ndarray | None = None,
                 Q_noise: np.ndarray | None = None,
                 R_noise: float = 4.0):
        """
        Parameters
        ----------
        planet_atm : PlanetaryAtmosphere
        mass_kg    : vehicle mass
        area_m2    : canopy reference area
        alt0_m     : initial altitude
        v0_ms      : initial velocity
        Cd_init    : initial Cd estimate
        ti_init    : initial inflation time estimate
        P_init     : initial state covariance (2×2)
        Q_noise    : process noise covariance (2×2)
        R_noise    : measurement noise variance [m/s]²
        """
        self.atm   = planet_atm
        self.mass  = mass_kg
        self.A     = area_m2
        self.alt0  = alt0_m
        self.v0    = v0_ms
        self.g     = planet_atm.gravity_ms2

        # State
        self.x_hat = np.array([Cd_init, ti_init])

        # Covariances
        self.P = P_init if P_init is not None else np.diag([0.04, 0.25])
        self.Q = Q_noise if Q_noise is not None else np.diag([1e-4, 1e-3])
        self.R = np.array([[R_noise]])

        # History
        self.history: list[dict] = []
        self._t_last = 0.0

    # ── Forward ODE model ─────────────────────────────────────────────────────

    def _logistic_A(self, t: float, Am: float, ti: float) -> float:
        k  = 8.0 / max(ti, 0.1)
        t0 = ti * 0.55
        return float(Am / (1 + np.exp(-k*(t-t0)))**0.5)

    def _forward_velocity(self, t_query: float, Cd: float, ti: float,
                          dt: float = 0.1) -> float:
        """Simulate velocity at t_query using current Cd, ti estimate."""
        v = float(self.v0); h = float(self.alt0); t = 0.0
        while h > 0 and t < t_query:
            A   = self._logistic_A(t, self.A, ti)
            rho = self.atm.density(max(h, 0))
            drag = 0.5 * rho * v**2 * Cd * A
            dv  = self.g - drag/self.mass
            v   = max(0.0, v + dt*dv)
            h   = max(0.0, h - dt*v)
            t  += dt
        return v

    # ── EKF prediction step ───────────────────────────────────────────────────

    def predict(self, dt: float = 1.0) -> None:
        """
        EKF prediction: propagate state and covariance.
        State model: x_{k+1} = x_k + w_k   (near-constant Cd assumption)
        """
        # F = I (identity state transition for near-constant Cd)
        self.P = self.P + self.Q * dt

    # ── EKF update step ───────────────────────────────────────────────────────

    def update(self, t_meas: float, v_meas: float) -> dict:
        """
        EKF update with a new velocity measurement at time t_meas.

        Parameters
        ----------
        t_meas : measurement time [s]
        v_meas : measured velocity [m/s]

        Returns
        -------
        dict with updated estimates and Kalman gain
        """
        Cd_hat, ti_hat = self.x_hat

        # ── Predict measurement from current state (nonlinear h) ─────────────
        h_hat = self._forward_velocity(t_meas, Cd_hat, ti_hat)
        innovation = v_meas - h_hat

        # ── Linearise h around x_hat via finite differences ──────────────────
        eps_Cd = max(Cd_hat * 0.005, 0.005)
        eps_ti = max(ti_hat * 0.005, 0.01)

        dh_dCd = (self._forward_velocity(t_meas, Cd_hat+eps_Cd, ti_hat)
                  - self._forward_velocity(t_meas, Cd_hat-eps_Cd, ti_hat)) / (2*eps_Cd)
        dh_dti = (self._forward_velocity(t_meas, Cd_hat, ti_hat+eps_ti)
                  - self._forward_velocity(t_meas, Cd_hat, ti_hat-eps_ti)) / (2*eps_ti)

        H = np.array([[dh_dCd, dh_dti]])   # (1, 2)

        # ── Kalman gain ───────────────────────────────────────────────────────
        S = H @ self.P @ H.T + self.R       # innovation covariance (1,1)
        K = self.P @ H.T / max(float(S[0,0]), 1e-9)   # (2,1)

        # ── State update ──────────────────────────────────────────────────────
        self.x_hat = self.x_hat + K[:, 0] * innovation

        # Physical bounds
        self.x_hat[0] = float(np.clip(self.x_hat[0], 0.3, 5.0))   # Cd
        self.x_hat[1] = float(np.clip(self.x_hat[1], 0.3, 15.0))  # t_infl

        # ── Covariance update (Joseph form — numerically stable) ─────────────
        I_KH  = np.eye(2) - np.outer(K[:, 0], H[0, :])
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K[:,0], K[:,0]) * float(self.R[0,0])

        # Store
        result = {
            "t_meas":      t_meas,
            "v_meas":      v_meas,
            "v_pred":      h_hat,
            "innovation":  innovation,
            "Cd_hat":      float(self.x_hat[0]),
            "ti_hat":      float(self.x_hat[1]),
            "Cd_std":      float(np.sqrt(max(self.P[0,0], 0))),
            "ti_std":      float(np.sqrt(max(self.P[1,1], 0))),
            "K_Cd":        float(K[0, 0]),
            "K_ti":        float(K[1, 0]),
        }
        self.history.append(result)
        return result

    # ── Streaming interface ───────────────────────────────────────────────────

    def process_stream(self, t_arr: np.ndarray, v_arr: np.ndarray,
                        verbose: bool = True) -> list[dict]:
        """
        Process a stream of (t, v) measurements sequentially.
        Simulates real-time online estimation during descent.
        """
        results = []
        if verbose:
            print(f"\n[Online PINN-Kalman] Processing {len(t_arr)} measurements")
            print(f"  Initial state: Cd={self.x_hat[0]:.4f}  t_infl={self.x_hat[1]:.4f}")

        for i, (t, v) in enumerate(zip(t_arr, v_arr)):
            dt = float(t - self._t_last) if i > 0 else 1.0
            self._t_last = t
            self.predict(dt)
            r = self.update(t, v)
            results.append(r)

            if verbose and (i+1) % max(1, len(t_arr)//5) == 0:
                print(f"  t={t:.1f}s  v_meas={v:.2f}  v_pred={r['v_pred']:.2f}  "
                      f"Cd={r['Cd_hat']:.5f}±{r['Cd_std']:.5f}  "
                      f"innov={r['innovation']:+.3f}")

        if verbose:
            final = results[-1]
            print(f"\n  Final: Cd={final['Cd_hat']:.5f}±{final['Cd_std']:.5f}  "
                  f"t_infl={final['ti_hat']:.4f}±{final['ti_std']:.4f}")

        return results

    def final_estimate(self) -> dict:
        """Return final state estimate with full uncertainty."""
        if not self.history:
            return {"Cd_hat": self.x_hat[0], "ti_hat": self.x_hat[1],
                    "Cd_std": np.sqrt(self.P[0,0]), "ti_std": np.sqrt(self.P[1,1])}
        return self.history[-1]


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def plot_online_pinn(results: list[dict], true_Cd: float | None = None,
                     save_path: str = "outputs/online_pinn_kalman.png"):
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

    t      = np.array([r["t_meas"] for r in results])
    Cd_hat = np.array([r["Cd_hat"] for r in results])
    Cd_std = np.array([r["Cd_std"] for r in results])
    ti_hat = np.array([r["ti_hat"] for r in results])
    v_meas = np.array([r["v_meas"] for r in results])
    v_pred = np.array([r["v_pred"] for r in results])
    innov  = np.array([r["innovation"] for r in results])
    K_Cd   = np.array([r["K_Cd"] for r in results])

    fig = plt.figure(figsize=(20, 11), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.36,
                            top=0.90, bottom=0.07, left=0.06, right=0.96)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    # Cd estimate convergence
    a = gax(0, 0)
    a.fill_between(t, Cd_hat-2*Cd_std, Cd_hat+2*Cd_std, alpha=0.25, color=C1, label="±2σ")
    a.fill_between(t, Cd_hat-Cd_std,   Cd_hat+Cd_std,   alpha=0.40, color=C1)
    a.plot(t, Cd_hat, color=C1, lw=2, label="Cd estimate")
    if true_Cd:
        a.axhline(true_Cd, color=C3, lw=1.5, ls="--", label=f"True Cd={true_Cd}")
    a.set_title("Online Cd Estimation", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("Cd"); a.legend(fontsize=7.5)

    # Uncertainty reduction
    a = gax(0, 1)
    a.fill_between(t, Cd_std, alpha=0.2, color=C2)
    a.plot(t, Cd_std, color=C2, lw=2, label="σ_Cd")
    a.set_title("Uncertainty σ_Cd(t)", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("σ_Cd"); a.legend(fontsize=7.5)

    # v_meas vs v_pred
    a = gax(0, 2)
    a.scatter(t, v_meas, s=8, color=C3, alpha=0.7, label="Measured v")
    a.plot(t, v_pred, color=C1, lw=1.8, label="Predicted v")
    a.set_title("Velocity: Measured vs Predicted", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("v [m/s]"); a.legend(fontsize=7.5)

    # Innovation
    a = gax(0, 3)
    a.fill_between(t, innov, alpha=0.2, color=C4)
    a.plot(t, innov, color=C4, lw=1.5)
    a.axhline(0, color=TX, lw=0.7, alpha=0.5)
    a.set_title("Innovation ν = z - h(x̂)", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("ν [m/s]")

    # Kalman gain
    a = gax(1, 0)
    a.fill_between(t, np.abs(K_Cd), alpha=0.2, color="#9d60ff")
    a.plot(t, np.abs(K_Cd), color="#9d60ff", lw=1.8)
    a.set_title("Kalman Gain K_Cd(t)", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("|K_Cd|")

    # t_infl estimate
    a = gax(1, 1)
    ti_std = np.array([r["ti_std"] for r in results])
    a.fill_between(t, ti_hat-2*ti_std, ti_hat+2*ti_std, alpha=0.2, color=C2)
    a.plot(t, ti_hat, color=C2, lw=2, label="t_infl estimate")
    a.set_title("t_infl Online Estimate", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("t_infl [s]"); a.legend(fontsize=7.5)

    # Innovation histogram
    a = gax(1, 2)
    a.hist(innov, bins=25, color=C4, alpha=0.65, edgecolor="none", density=True)
    from scipy.stats import norm as norm_dist
    x_h = np.linspace(innov.min(), innov.max(), 100)
    a.plot(x_h, norm_dist.pdf(x_h, innov.mean(), max(innov.std(), 0.01)),
           color=C1, lw=2, label="Gaussian fit")
    a.axvline(0, color=TX, lw=0.7, alpha=0.5)
    a.legend(fontsize=7.5); a.set_title("Innovation Distribution", fontweight="bold")
    a.set_xlabel("ν [m/s]"); a.set_ylabel("Density")

    # Cd vs v_meas scatter (convergence)
    a = gax(1, 3)
    sc = a.scatter(v_meas, Cd_hat, c=t, cmap="plasma", s=8, alpha=0.8)
    fig.colorbar(sc, ax=a, label="t [s]", pad=0.02).ax.tick_params(labelsize=7)
    a.set_title("Cd vs v_meas (coloured by t)", fontweight="bold")
    a.set_xlabel("v_meas [m/s]"); a.set_ylabel("Cd estimate")

    final = results[-1]
    fig.text(0.5, 0.955,
             f"Online PINN-Kalman Fusion  |  "
             f"Cd_final={final['Cd_hat']:.5f}±{final['Cd_std']:.5f}  |  "
             f"t_infl={final['ti_hat']:.3f}±{final['ti_std']:.3f}s",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    from pathlib import Path; Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Online PINN-Kalman plot saved: {save_path}")
    return fig


def run(true_Cd: float = 1.55, true_ti: float = 2.8,
        n_obs: int = 60, sigma_obs: float = 1.5,
        verbose: bool = True) -> dict:
    """Run online PINN-Kalman on synthetic streaming telemetry."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.planetary_atm import MarsAtmosphere
    from src.multifidelity_pinn import LowFidelityEDL

    planet = MarsAtmosphere()
    lf_true = LowFidelityEDL(planet, 900, true_Cd, 78.5, gamma_deg=15)
    t_arr   = np.linspace(5, 300, n_obs)
    v_true, _ = lf_true.solve(t_arr, 5800, 125_000)

    rng   = np.random.default_rng(42)
    v_obs = v_true + rng.normal(0, sigma_obs, n_obs)
    v_obs = np.clip(v_obs, 0, None)

    estimator = OnlinePINNKalman(
        planet, mass_kg=900, area_m2=78.5,
        alt0_m=125_000, v0_ms=5800,
        Cd_init=1.4, ti_init=3.5,   # intentionally wrong initial guess
        R_noise=sigma_obs**2,
    )

    results = estimator.process_stream(t_arr, v_obs, verbose=verbose)
    final   = estimator.final_estimate()

    if verbose:
        err = abs(final["Cd_hat"] - true_Cd)
        print(f"\n  True Cd={true_Cd}  |  Estimated={final['Cd_hat']:.5f}  |  Error={err:.5f}")

    fig = plot_online_pinn(results, true_Cd=true_Cd)
    plt.close(fig)

    return {
        "results":   results,
        "final":     final,
        "true_Cd":   true_Cd,
        "estimator": estimator,
    }


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    out = run(true_Cd=1.55, n_obs=60, verbose=True)
    print(f"Final Cd={out['final']['Cd_hat']:.5f}  (true={out['true_Cd']})")
