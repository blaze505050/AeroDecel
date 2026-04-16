"""
src/normalizing_flows.py — Physics-Constrained Normalising Flows for EDL
=========================================================================
Models the full posterior P(trajectory | entry conditions) using a
Real-NVP (Non-Volume Preserving) normalising flow.

Key idea
--------
  Instead of a point prediction (PINN) or Gaussian posterior (Kalman),
  a normalising flow learns a bijective mapping f_θ: z → x where:
    z ~ N(0, I)  (simple base distribution)
    x ~ P(trajectory | conditions)  (complex posterior)

  This enables:
    • Exact density evaluation: log p(x) = log p(z) - log|det J_f|
    • Efficient sampling: x = f_θ(z), z ~ N(0, I)
    • Full posterior uncertainty (not just mean + variance)

Physics constraint
------------------
  The physics residual of the drag ODE is added as a penalty:
    L_total = L_NLL + λ_phys · L_physics

  L_physics = ||m·dv/dt + D(v,h) - m·g||²  averaged over flow samples

Real-NVP architecture (pure numpy + optional torch)
----------------------------------------------------
  Each coupling layer:
    x₁ = x₁                        (identity)
    x₂ = x₂ · exp(s(x₁)) + t(x₁)  (affine coupling)

  where s, t are neural networks (the "scale and translate" functions).
  The Jacobian is triangular → log|det J| = Σ s(x₁) (sum of scale outputs).

  Alternating masks ensure all dimensions are transformed.

Without torch:
  → Use scipy.optimize + analytical Gaussian mixture as approximation
  → Still provides non-Gaussian posterior samples
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS (numpy)
# ══════════════════════════════════════════════════════════════════════════════

def _ode_residual(v_traj: np.ndarray, h_traj: np.ndarray,
                   t_arr: np.ndarray, params: dict) -> float:
    """
    Physics residual: m·dv/dt + D(v,h) - m·g = 0
    Returns mean squared residual.
    """
    mass = params.get("mass_kg", 900.0)
    Cd   = params.get("Cd", 1.7)
    A    = params.get("area_m2", 78.5)
    g    = params.get("gravity_ms2", 3.72)
    rho0 = params.get("rho0", 0.02)
    H    = params.get("H", 11100.0)

    rho_arr = rho0 * np.exp(-np.maximum(h_traj, 0) / H)
    D_arr   = 0.5 * rho_arr * v_traj**2 * Cd * A
    dvdt    = np.gradient(v_traj, t_arr)
    residuals = mass * dvdt + D_arr - mass * g
    return float(np.mean(residuals**2))


# ══════════════════════════════════════════════════════════════════════════════
# TORCH REAL-NVP
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:
    class _AffineNet(nn.Module):
        """Scale-translate network for one coupling layer."""
        def __init__(self, in_dim: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, in_dim * 2),   # output: s and t concatenated
            )

        def forward(self, x: torch.Tensor):
            out = self.net(x)
            s   = torch.tanh(out[..., :x.shape[-1]])   # log-scale, clamped
            t   = out[..., x.shape[-1]:]
            return s, t

    class RealNVP(nn.Module):
        """
        Real-NVP normalising flow with alternating masks.
        Transforms a Gaussian z → trajectory distribution x.
        """
        def __init__(self, dim: int, n_layers: int = 6, hidden: int = 64):
            super().__init__()
            self.dim = dim
            self.nets = nn.ModuleList([_AffineNet(dim//2, hidden) for _ in range(n_layers)])
            self.n_layers = n_layers
            # Alternating masks
            self.masks = []
            for i in range(n_layers):
                m = torch.zeros(dim, dtype=torch.bool)
                if i % 2 == 0:
                    m[:dim//2] = True
                else:
                    m[dim//2:] = True
                self.masks.append(m)

        def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """z → x (sampling direction). Returns (x, log_det_J)."""
            x = z.clone()
            log_det = torch.zeros(z.shape[0], device=z.device)
            for i, (net, mask) in enumerate(zip(self.nets, self.masks)):
                x_mask = x[:, mask]
                s, t   = net(x_mask)
                x[:, ~mask] = x[:, ~mask] * torch.exp(s) + t
                log_det += s.sum(dim=-1)
            return x, log_det

        def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """x → z (density evaluation). Returns (z, log_det_J)."""
            z = x.clone()
            log_det = torch.zeros(x.shape[0], device=x.device)
            for i, (net, mask) in enumerate(zip(reversed(self.nets),
                                                 self.masks[::-1])):
                z_mask = z[:, mask]
                s, t   = net(z_mask)
                z[:, ~mask] = (z[:, ~mask] - t) * torch.exp(-s)
                log_det -= s.sum(dim=-1)
            return z, log_det

        def log_prob(self, x: torch.Tensor) -> torch.Tensor:
            """Exact log-likelihood log p(x) = log p(z) + log|det J_f^{-1}|."""
            z, log_det = self.inverse(x)
            log_pz = -0.5 * (z**2 + np.log(2*np.pi)).sum(dim=-1)
            return log_pz + log_det

        def sample(self, n: int) -> torch.Tensor:
            """Draw n samples from the learned distribution."""
            self.eval()
            with torch.no_grad():
                z = torch.randn(n, self.dim)
                x, _ = self.forward(z)
            return x


# ══════════════════════════════════════════════════════════════════════════════
# SCIPY FALLBACK: Gaussian Mixture Posterior
# ══════════════════════════════════════════════════════════════════════════════

class _GaussianMixturePosterior:
    """
    Gaussian mixture model as a normalising flow fallback (no torch).
    Fitted via EM algorithm on trajectory samples.
    Provides: log_prob, sample, physics-constrained mean.
    """

    def __init__(self, n_components: int = 4):
        self.K   = n_components
        self.means: np.ndarray | None = None
        self.covs:  np.ndarray | None = None
        self.pis:   np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray, n_iter: int = 50, reg: float = 1e-4):
        """EM algorithm for GMM fitting."""
        n, d = X.shape
        K    = self.K
        rng  = np.random.default_rng(0)

        # Init: K-means-like initialisation
        idx   = rng.choice(n, K, replace=False)
        means = X[idx].copy()
        covs  = np.array([np.eye(d) + reg*np.eye(d) for _ in range(K)])
        pis   = np.full(K, 1.0/K)

        for it in range(n_iter):
            # E step: responsibilities
            log_r = np.zeros((n, K))
            for k in range(K):
                diff = X - means[k]
                try:
                    cov_inv = np.linalg.inv(covs[k] + reg*np.eye(d))
                    log_det = np.log(np.linalg.det(covs[k] + reg*np.eye(d)) + 1e-300)
                    log_r[:, k] = (np.log(max(pis[k], 1e-300))
                                   - 0.5*log_det
                                   - 0.5*np.sum(diff @ cov_inv * diff, axis=1))
                except Exception:
                    log_r[:, k] = -1e9

            # Numerically stable softmax
            log_r -= log_r.max(axis=1, keepdims=True)
            r = np.exp(log_r)
            r /= r.sum(axis=1, keepdims=True) + 1e-300

            # M step
            r_sum = r.sum(axis=0) + 1e-9
            pis   = r_sum / n
            means = (r.T @ X) / r_sum[:, None]
            for k in range(K):
                diff   = X - means[k]
                covs[k] = (r[:, k:k+1] * diff).T @ diff / r_sum[k] + reg*np.eye(d)

        self.means = means; self.covs = covs; self.pis = pis
        self._fitted = True

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """Log-likelihood for each sample in X."""
        n, d = X.shape
        log_pX = np.full(n, -np.inf)
        for k in range(self.K):
            diff    = X - self.means[k]
            try:
                cov_inv = np.linalg.inv(self.covs[k])
                log_det = np.log(max(np.linalg.det(self.covs[k]), 1e-300))
                log_pk  = (np.log(max(self.pis[k], 1e-300))
                           - 0.5*log_det
                           - 0.5*np.sum(diff @ cov_inv * diff, axis=1)
                           - 0.5*d*np.log(2*np.pi))
                log_pX = np.logaddexp(log_pX, log_pk)
            except Exception:
                pass
        return log_pX

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the mixture."""
        if not self._fitted:
            raise RuntimeError("Fit the model first")
        rng = np.random.default_rng()
        k_samples = rng.choice(self.K, size=n, p=self.pis)
        samples   = np.vstack([
            rng.multivariate_normal(self.means[k], self.covs[k])
            for k in k_samples
        ])
        return samples


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS-CONSTRAINED NORMALISING FLOW
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsConstrainedFlow:
    """
    Normalising flow with physics residual loss.

    Training objective:
      L = E_x[-log p_θ(x)] + λ · E_x[physics_residual(x)]

    The physics constraint penalises trajectories that violate the drag ODE.
    This strongly regularises the posterior toward physically plausible trajectories.
    """

    def __init__(self, traj_dim: int = 50, n_layers: int = 6,
                 hidden: int = 64, lam_phys: float = 0.5):
        self.dim       = traj_dim
        self.lam_phys  = lam_phys
        self._backend  = "torch" if _TORCH else "numpy"

        if _TORCH:
            self.flow  = RealNVP(traj_dim, n_layers, hidden)
            self.opt   = torch.optim.Adam(self.flow.parameters(), lr=5e-4)
            self.sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
                             self.opt, T_max=1000, eta_min=1e-6)
        else:
            self.flow  = _GaussianMixturePosterior(n_components=5)

        self._trained  = False
        self._t_arr    = None
        self._phys_params: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def _physics_loss_torch(self, x_samples: "torch.Tensor") -> "torch.Tensor":
        """Penalise ODE residual on flow samples."""
        import torch
        p = self._phys_params
        n, d = x_samples.shape
        t    = torch.linspace(0, 300, d)
        dt   = (t[1] - t[0]).item()

        v_samples = x_samples  # interpret last dim as velocity trajectory
        # Simple FD: dv/dt ≈ (v_{i+1} - v_{i-1}) / (2*dt)
        dvdt = torch.zeros_like(v_samples)
        dvdt[:, 1:-1] = (v_samples[:, 2:] - v_samples[:, :-2]) / (2*dt)
        dvdt[:, 0]    = dvdt[:, 1]; dvdt[:, -1] = dvdt[:, -2]

        # Rough altitude from v (integrate assuming constant FPA)
        h = torch.linspace(p.get("alt0_m", 125000), 0, d).unsqueeze(0).expand(n,-1)
        rho = p.get("rho0", 0.02) * torch.exp(-h / p.get("H", 11100.0))
        D   = 0.5 * rho * v_samples**2 * p.get("Cd", 1.7) * p.get("area_m2", 78.5)
        mass = p.get("mass_kg", 900.0)
        g    = p.get("gravity_ms2", 3.72)

        residual = mass * dvdt + D - mass * g
        return (residual**2).mean()

    def train(self, v_trajectories: np.ndarray, t_arr: np.ndarray,
              phys_params: dict | None = None,
              n_epochs: int = 1000, verbose: bool = True) -> list[float]:
        """
        Train the normalising flow on observed trajectory data.

        Parameters
        ----------
        v_trajectories : (N_traj, N_t) array of velocity time series
        t_arr          : time array [s], length N_t
        phys_params    : physical parameters for ODE residual
        n_epochs       : training epochs

        Returns
        -------
        Training loss history
        """
        self._t_arr = t_arr
        self._phys_params = phys_params or {}

        # Normalise trajectories to [0, 1] for training
        v_mean = v_trajectories.mean(axis=0, keepdims=True)
        v_std  = v_trajectories.std(axis=0, keepdims=True) + 1e-6
        v_norm = (v_trajectories - v_mean) / v_std
        self._v_mean = v_mean; self._v_std = v_std

        # Resize to model dim if needed
        n_traj = v_norm.shape[0]
        from scipy.interpolate import interp1d
        if v_norm.shape[1] != self.dim:
            v_resized = np.zeros((n_traj, self.dim))
            for i in range(n_traj):
                fn = interp1d(np.linspace(0,1,v_norm.shape[1]), v_norm[i])
                v_resized[i] = fn(np.linspace(0,1,self.dim))
            v_norm = v_resized

        losses = []

        if _TORCH:
            X = torch.tensor(v_norm, dtype=torch.float32)
            for ep in range(1, n_epochs+1):
                self.flow.train()
                self.opt.zero_grad()

                # NLL loss
                log_p = self.flow.log_prob(X)
                L_nll = -log_p.mean()

                # Physics constraint on generated samples
                n_phys = min(32, n_traj)
                z_phys = torch.randn(n_phys, self.dim)
                x_phys, _ = self.flow.forward(z_phys)
                # Denormalise for physics
                x_phys_phys = x_phys * torch.tensor(v_std.ravel()[:self.dim],
                                                      dtype=torch.float32) + \
                              torch.tensor(v_mean.ravel()[:self.dim], dtype=torch.float32)
                L_phys = self._physics_loss_torch(x_phys_phys)

                loss = L_nll + self.lam_phys * L_phys
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
                self.opt.step(); self.sch.step()

                lv = float(loss)
                losses.append(lv)
                if verbose and ep % max(1, n_epochs//5) == 0:
                    print(f"  [NFlow] ep {ep:5d}/{n_epochs}  "
                          f"loss={lv:.4e}  nll={float(L_nll):.3e}  "
                          f"phys={float(L_phys):.3e}")
        else:
            # GMM fallback
            self.flow.fit(v_norm, n_iter=100)
            losses = [0.0]

        self._trained = True
        return losses

    # ── Inference ─────────────────────────────────────────────────────────────

    def sample_trajectories(self, n: int = 200) -> np.ndarray:
        """
        Draw n trajectory samples from the learned posterior.
        Returns (n, n_t_original) array of velocity trajectories.
        """
        if not self._trained:
            raise RuntimeError("Train the flow first")

        if _TORCH:
            self.flow.eval()
            with torch.no_grad():
                z = torch.randn(n, self.dim)
                x, _ = self.flow.forward(z)
                x_np = x.numpy()
        else:
            x_np = self.flow.sample(n)

        # Denormalise
        v_std  = self._v_std.ravel()[:self.dim]
        v_mean = self._v_mean.ravel()[:self.dim]
        x_phys = x_np * v_std[None, :] + v_mean[None, :]
        return np.maximum(x_phys, 0)

    def posterior_statistics(self, n_samples: int = 500) -> dict:
        """
        Compute posterior mean, std, and credible intervals from flow samples.
        """
        samples = self.sample_trajectories(n_samples)
        return {
            "v_mean":   samples.mean(axis=0),
            "v_std":    samples.std(axis=0),
            "v_p05":    np.percentile(samples, 5,  axis=0),
            "v_p50":    np.percentile(samples, 50, axis=0),
            "v_p95":    np.percentile(samples, 95, axis=0),
            "v_p25":    np.percentile(samples, 25, axis=0),
            "v_p75":    np.percentile(samples, 75, axis=0),
            "samples":  samples,
            "n_samples":n_samples,
            "backend":  self._backend,
        }

    def log_prob(self, v_traj: np.ndarray) -> float:
        """Evaluate log-likelihood of a trajectory under the learned distribution."""
        from scipy.interpolate import interp1d
        v_norm = (v_traj - self._v_mean.ravel()[:len(v_traj)]) / \
                 (self._v_std.ravel()[:len(v_traj)] + 1e-6)
        if len(v_norm) != self.dim:
            fn = interp1d(np.linspace(0,1,len(v_norm)), v_norm, bounds_error=False,
                          fill_value="extrapolate")
            v_norm = fn(np.linspace(0,1,self.dim))

        if _TORCH:
            self.flow.eval()
            with torch.no_grad():
                x_t = torch.tensor(v_norm[None, :], dtype=torch.float32)
                return float(self.flow.log_prob(x_t).item())
        else:
            return float(self.flow.log_prob(v_norm[None, :]).item())


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_flow(t_arr, observed_trajs, posterior, save_path="outputs/normalizing_flow.png"):
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

    fig = plt.figure(figsize=(18, 10), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.36,
                            top=0.91, bottom=0.07, left=0.06, right=0.96)

    samples = posterior["samples"]
    t_plot  = np.linspace(t_arr[0], t_arr[-1], samples.shape[1])

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    # P0: Observed trajectories
    a = gax(0, 0)
    for i, traj in enumerate(observed_trajs[:20]):
        a.plot(t_arr, traj, color="#00d4ff", lw=0.6, alpha=0.4)
    a.set_title("Training Trajectories (observed)", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("v [m/s]")

    # P1: Posterior samples
    a = gax(0, 1)
    for i in range(min(100, len(samples))):
        a.plot(t_plot, samples[i], color="#ff6b35", lw=0.4, alpha=0.3)
    a.fill_between(t_plot, posterior["v_p05"], posterior["v_p95"],
                   alpha=0.4, color=C1, label="90% CI")
    a.plot(t_plot, posterior["v_mean"], color=C1, lw=2, label="Posterior mean")
    a.legend(fontsize=7.5)
    a.set_title("Posterior Predictive Samples", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("v [m/s]")

    # P2: Uncertainty band
    a = gax(0, 2)
    a.fill_between(t_plot, posterior["v_p05"], posterior["v_p95"],
                   alpha=0.35, color=C3, label="90% CI")
    a.fill_between(t_plot, posterior["v_p25"], posterior["v_p75"],
                   alpha=0.5, color=C4, label="50% CI")
    a.plot(t_plot, posterior["v_mean"], color=C1, lw=2, label="Mean")
    a.plot(t_plot, posterior["v_p50"],  color=C3, lw=1.2, ls="--", label="Median")
    a.legend(fontsize=7.5)
    a.set_title("Posterior Credible Intervals", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("v [m/s]")

    # P3: Uncertainty magnitude
    a = gax(1, 0)
    a.fill_between(t_plot, posterior["v_std"], alpha=0.3, color=C2)
    a.plot(t_plot, posterior["v_std"], color=C2, lw=1.8)
    a.set_title("Posterior Std σ(t)", fontweight="bold")
    a.set_xlabel("t [s]"); a.set_ylabel("σ [m/s]")

    # P4: PDF at specific time (end of trajectory)
    a = gax(1, 1)
    v_end = samples[:, -1]
    a.hist(v_end, bins=40, color=C1, alpha=0.65, edgecolor="none", density=True)
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(v_end)
        vx  = np.linspace(v_end.min(), v_end.max(), 200)
        a.plot(vx, kde(vx), color=C3, lw=2)
    except Exception: pass
    a.axvline(np.mean(v_end), color=C4, lw=1.5, ls="--",
              label=f"Mean={np.mean(v_end):.1f}")
    a.legend(fontsize=7.5)
    a.set_title("v_final PDF (from flow)", fontweight="bold")
    a.set_xlabel("v_final [m/s]"); a.set_ylabel("Density")

    # P5: Summary
    a = gax(1, 2); a.axis("off")
    rows = [
        ("Backend",     posterior["backend"].upper()),
        ("N samples",   str(posterior["n_samples"])),
        ("v_mean end",  f"{posterior['v_mean'][-1]:.2f} m/s"),
        ("σ_end",       f"{posterior['v_std'][-1]:.2f} m/s"),
        ("90% CI end",  f"[{posterior['v_p05'][-1]:.1f}, {posterior['v_p95'][-1]:.1f}]"),
        ("Max σ(t)",    f"{posterior['v_std'].max():.2f} m/s"),
    ]
    for j, (lab, val) in enumerate(rows):
        a.text(0.05, 1-j*0.15, lab, transform=a.transAxes, fontsize=9, color="#556688")
        a.text(0.95, 1-j*0.15, val, transform=a.transAxes, fontsize=9,
               ha="right", color=TX)
    a.set_title("Flow Summary", fontweight="bold", color=TX)

    fig.text(0.5, 0.955,
             f"Physics-Constrained Normalising Flow  |  "
             f"Backend={posterior['backend'].upper()}  |  N={posterior['n_samples']}",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    from pathlib import Path; Path(save_path).parent.mkdir(exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Flow plot saved: {save_path}")
    return fig


def run(n_training_trajs: int = 200, n_epochs: int = 500,
        verbose: bool = True) -> dict:
    """Train physics-constrained normalising flow on synthetic EDL trajectories."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.planetary_atm import MarsAtmosphere
    from src.multifidelity_pinn import LowFidelityEDL

    planet = MarsAtmosphere()
    rng    = np.random.default_rng(42)

    # Generate training trajectories with parameter scatter
    t_arr = np.linspace(0, 400, 100)
    trajs = []
    for _ in range(n_training_trajs):
        Cd   = rng.uniform(1.4, 2.0)
        mass = rng.uniform(800, 1000)
        lf   = LowFidelityEDL(planet, mass, Cd, 78.5, gamma_deg=15.0)
        v, _ = lf.solve(t_arr, 5800, 125_000)
        trajs.append(v)

    trajs = np.array(trajs)
    if verbose:
        print(f"[NFlow] Training on {n_training_trajs} trajectories  "
              f"backend={('torch' if _TORCH else 'numpy')}")

    # Build and train flow
    flow = PhysicsConstrainedFlow(traj_dim=50, n_layers=6, hidden=32, lam_phys=0.3)
    phys_params = {"mass_kg": 900, "Cd": 1.7, "area_m2": 78.5,
                   "gravity_ms2": planet.gravity_ms2,
                   "rho0": planet.density(0), "H": 11100,
                   "alt0_m": 125_000}
    losses = flow.train(trajs, t_arr, phys_params=phys_params,
                        n_epochs=n_epochs, verbose=verbose)

    # Posterior statistics
    posterior = flow.posterior_statistics(n_samples=500)

    if verbose:
        print(f"\n  Posterior v_mean(t_end) = {posterior['v_mean'][-1]:.2f} m/s")
        print(f"  Posterior σ(t_end)      = {posterior['v_std'][-1]:.2f} m/s")
        print(f"  90% CI at t_end         = [{posterior['v_p05'][-1]:.1f}, "
              f"{posterior['v_p95'][-1]:.1f}] m/s")

    fig = plot_flow(t_arr, trajs[:30], posterior)
    plt.close(fig)
    return {"flow": flow, "posterior": posterior, "losses": losses}


if __name__ == "__main__":
    result = run(n_training_trajs=100, n_epochs=200)
    print(f"Flow trained. Posterior mean v_final={result['posterior']['v_mean'][-1]:.2f}m/s")
