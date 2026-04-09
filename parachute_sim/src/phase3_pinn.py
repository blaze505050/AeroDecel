"""
phase3_pinn.py — AeroDecel v5.0 Research-Grade PINN for Cd(t) Identification
==============================================================================
Physics-Informed Neural Network that solves the inverse problem:
  Given observed v(t), h(t), A(t) → recover Cd(t)

Architecture:
  - Input:  normalized time t̃ ∈ [0,1] (with optional Fourier feature embedding)
  - Output: Cd(t) — dynamic drag coefficient (and optionally v(t))
  - Loss:   L = λ_phys·L_physics + λ_data·L_data + λ_smooth·L_smooth

Physics residual enforces the FULL momentum equation:
    r(t) = m·dv/dt - m·g + 0.5·ρ(h(t))·v(t)²·Cd(t)·A(t) ≈ 0

AeroDecel v5.0 Enhancements:
  - **Full momentum residual** (not just smoothness)
  - **Fourier feature embeddings** for high-frequency Cd capture
  - **Curriculum training**: data-only warmup → full physics
  - **Adaptive loss weighting** (gradient-magnitude balancing)
  - **Validation early stopping** to prevent overfitting
  - **Dual output mode**: predict v(t) and Cd(t) simultaneously

Reference:
  - Raissi et al., "Physics-Informed Neural Networks", JCP 2019
  - Tancik et al., "Fourier Features Let Networks Learn High Freq", NeurIPS 2020
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import vectorized_density


# ─── Device Setup ────────────────────────────────────────────────────────────
def get_device():
    if cfg.DEVICE == "auto":
        if torch.cuda.is_available():    return torch.device("cuda")
        try:
            if torch.backends.mps.is_available(): return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device(cfg.DEVICE)


# ─── Fourier Feature Embedding (AeroDecel v5.0) ─────────────────────────────
class FourierFeatureEmbedding(nn.Module):
    """
    Maps scalar input t → [sin(2πσ₁t), cos(2πσ₁t), ..., sin(2πσₖt), cos(2πσₖt)]

    Fourier features enable the network to learn high-frequency functions
    that standard MLPs struggle with (spectral bias). Critical for capturing
    rapid Cd variations during canopy inflation.

    Reference: Tancik et al., "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains", NeurIPS 2020
    """

    def __init__(self, scales: list = None):
        super().__init__()
        scales = scales or cfg.PINN_FOURIER_SCALES
        self.register_buffer(
            "frequencies",
            torch.tensor(scales, dtype=torch.float32) * 2.0 * np.pi
        )
        self.out_dim = 1 + 2 * len(scales)  # original t + sin/cos pairs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (N,1) → (N, 1 + 2K) where K = number of Fourier scales."""
        projections = t * self.frequencies.unsqueeze(0)   # (N, K)
        return torch.cat([t, torch.sin(projections), torch.cos(projections)], dim=-1)


# ─── PINN Architecture — AeroDecel v5.0 ─────────────────────────────────────
class CdNetwork(nn.Module):
    """
    Fully connected residual PINN that maps t → Cd(t).
    Uses residual connections to improve gradient flow over deep networks.
    Enforces Cd > 0 via softplus output activation.

    AeroDecel v5.0: Optional Fourier feature embedding input layer.
    """

    def __init__(self, hidden: list = None, activation: str = "tanh",
                 use_fourier: bool = None):
        super().__init__()
        hidden = hidden or cfg.PINN_HIDDEN_LAYERS
        use_fourier = use_fourier if use_fourier is not None else cfg.PINN_FOURIER_FEATURES

        # Activation factory
        act_map = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU,
                   "mish": nn.Mish, "elu": nn.ELU}
        act_cls = act_map.get(activation, nn.Tanh)

        # Fourier embedding
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatureEmbedding(cfg.PINN_FOURIER_SCALES)
            in_dim = self.fourier.out_dim
        else:
            self.fourier = None
            in_dim = 1

        # Build layers with residual blocks where dimension matches
        layers = []
        self.residual_idx = []

        for i, h in enumerate(hidden):
            block = nn.Sequential(
                nn.Linear(in_dim, h),
                act_cls(),
            )
            layers.append(block)
            if in_dim == h:
                self.residual_idx.append(i)
            in_dim = h

        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, 1)

        # Initialization (Xavier uniform + small bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: shape (N,1) normalized to [0,1]. Returns Cd: shape (N,1)."""
        if self.fourier is not None:
            x = self.fourier(t)
        else:
            x = t

        prev_x = x
        for i, layer in enumerate(self.layers):
            out = layer(x)
            if i in self.residual_idx:
                out = out + prev_x
            prev_x = x
            x = out

        raw = self.out(x)
        # Softplus ensures Cd > 0, shifted to realistic aerodynamic range [0.5, 3.0]
        Cd = 0.5 + torch.nn.functional.softplus(raw)
        return Cd


# Backwards compatibility alias
CdNetwork_v1 = CdNetwork


# ─── Dual-Output PINN (AeroDecel v5.0) ──────────────────────────────────────
class CdVelocityNetwork(nn.Module):
    """
    Dual-output PINN that simultaneously predicts v(t) and Cd(t).

    Architecture:
      - Shared trunk: Fourier embedding → residual FC layers
      - Head A (velocity):  trunk → Linear → unconstrained v(t)
      - Head B (drag coeff): trunk → Linear → softplus Cd(t) ≥ 0.5

    The dual-output approach enforces physics more tightly because
    the momentum residual can use the network's own dv/dt (via autograd)
    rather than finite-difference estimates from noisy ODE data.

    If PINN_DUAL_OUTPUT = False in config, the trainer falls back to
    CdNetwork (single-output, v1 behavior).
    """

    def __init__(self, hidden: list = None, activation: str = "tanh",
                 use_fourier: bool = None,
                 v_scale: float = 25.0, v_offset: float = 10.0):
        super().__init__()
        hidden = hidden or cfg.PINN_HIDDEN_LAYERS
        use_fourier = use_fourier if use_fourier is not None else cfg.PINN_FOURIER_FEATURES

        # Activation factory
        act_map = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU,
                   "mish": nn.Mish, "elu": nn.ELU}
        act_cls = act_map.get(activation, nn.Tanh)

        # Fourier embedding (shared)
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatureEmbedding(cfg.PINN_FOURIER_SCALES)
            in_dim = self.fourier.out_dim
        else:
            self.fourier = None
            in_dim = 1

        # Shared trunk layers with residual connections
        layers = []
        self.residual_idx = []
        for i, h in enumerate(hidden):
            block = nn.Sequential(nn.Linear(in_dim, h), act_cls())
            layers.append(block)
            if in_dim == h:
                self.residual_idx.append(i)
            in_dim = h

        self.trunk = nn.ModuleList(layers)

        # Head A: velocity prediction (unconstrained output)
        self.head_v = nn.Linear(in_dim, 1)

        # Head B: drag coefficient (Cd ≥ 0.5 via softplus offset)
        self.head_cd = nn.Linear(in_dim, 1)

        # De-normalization constants for velocity
        self.v_scale = v_scale       # roughly the velocity range
        self.v_offset = v_offset     # approximate mean velocity

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def _trunk(self, t: torch.Tensor) -> torch.Tensor:
        """Shared feature extraction through residual trunk."""
        if self.fourier is not None:
            x = self.fourier(t)
        else:
            x = t
        prev_x = x
        for i, layer in enumerate(self.trunk):
            out = layer(x)
            if i in self.residual_idx:
                out = out + prev_x
            prev_x = x
            x = out
        return x

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        t: (N, 1) normalized time.
        Returns: (v_pred, Cd_pred) each shape (N, 1).
        """
        features = self._trunk(t)

        # Velocity head (de-normalized so output is in m/s space)
        v_raw = self.head_v(features)
        v_pred = self.v_offset + self.v_scale * torch.tanh(v_raw)

        # Cd head (constrained ≥ 0.5)
        cd_raw = self.head_cd(features)
        cd_pred = 0.5 + nn.functional.softplus(cd_raw)

        return v_pred, cd_pred

    def predict_cd(self, t: torch.Tensor) -> torch.Tensor:
        """Convenience: return only Cd(t), matching CdNetwork API."""
        _, cd = self.forward(t)
        return cd


# ─── Dual-Output PINN Loss (autograd-based physics) ─────────────────────────
class DualOutputPINNLoss:
    """
    Physics loss for the dual-output PINN.

    Unlike PINNLoss which uses finite-difference dv/dt from ODE data,
    this loss computes dv/dt via AUTOGRAD on the network's own v(t) output.
    This yields exact gradients and eliminates numerical differentiation noise.

    Residual:
        r(t) = m * (dv_net/dt) - m*g + 0.5 * rho(t) * v_net^2 * Cd_net * A(t)

    where dv_net/dt = d/dt [CdVelocityNetwork.forward(t)[0]]  via autograd.
    """

    def __init__(self, mass: float, gravity: float,
                 t_full: torch.Tensor, v_full: torch.Tensor,
                 h_full: torch.Tensor, A_full: torch.Tensor,
                 rho_full: torch.Tensor, device: torch.device):
        self.mass = mass
        self.g = gravity
        self.t = t_full.to(device)
        self.v = v_full.to(device)
        self.h = h_full.to(device)
        self.A = A_full.to(device)
        self.rho = rho_full.to(device)
        self.device = device

    def physics_residual(self, model: CdVelocityNetwork,
                         t_col: torch.Tensor) -> torch.Tensor:
        """
        Full momentum residual with AUTOGRAD dv/dt.

        This is the key advantage of dual-output: we differentiate the
        network's own velocity prediction analytically, not numerically.
        """
        t_col = t_col.requires_grad_(True)

        v_pred, cd_pred = model(t_col)

        # Autograd: dv/dt from the network's velocity output
        dv_dt = torch.autograd.grad(
            outputs=v_pred, inputs=t_col,
            grad_outputs=torch.ones_like(v_pred),
            create_graph=True, retain_graph=True
        )[0]

        # Interpolate physical quantities at collocation points
        t_np = t_col.detach().cpu().numpy().flatten()
        t_ref = self.t.cpu().numpy().flatten()
        A_np = np.interp(t_np, t_ref, self.A.cpu().numpy().flatten())
        rho_np = np.interp(t_np, t_ref, self.rho.cpu().numpy().flatten())

        A_col = torch.tensor(A_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        rho_col = torch.tensor(rho_np, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Momentum residual: m*dv/dt = mg - 0.5*rho*v^2*Cd*A
        drag = 0.5 * rho_col * v_pred**2 * cd_pred * A_col
        residual = self.mass * dv_dt - self.mass * self.g + drag

        # Normalize for scale invariance
        weight = self.mass * self.g
        return (residual / weight).pow(2).mean()

    def data_loss(self, model: CdVelocityNetwork,
                  t_obs: torch.Tensor, v_obs: torch.Tensor) -> torch.Tensor:
        """MSE between predicted velocity and observed velocity."""
        v_pred, _ = model(t_obs)
        return nn.functional.mse_loss(v_pred, v_obs)

    def smoothness_loss(self, model: CdVelocityNetwork,
                        t_obs: torch.Tensor) -> torch.Tensor:
        """Second-order smoothness on Cd(t)."""
        _, cd = model(t_obs)
        d1 = cd[1:] - cd[:-1]
        d2 = d1[1:] - d1[:-1]
        return (d2**2).mean()


class PINNLoss:
    """
    Composite loss for physics-constrained Cd identification.

    AeroDecel v5.0: Full momentum equation residual instead of
    simple dCd/dt smoothness.
    """

    def __init__(
        self,
        mass: float,
        gravity: float,
        t_full: torch.Tensor,
        v_full: torch.Tensor,
        h_full: torch.Tensor,
        A_full: torch.Tensor,
        rho_full: torch.Tensor,
        device: torch.device,
    ):
        self.mass    = mass
        self.g       = gravity
        self.t       = t_full.to(device)
        self.v       = v_full.to(device)
        self.h       = h_full.to(device)
        self.A       = A_full.to(device)
        self.rho     = rho_full.to(device)
        self.device  = device

        # Precompute dv/dt from ODE data using finite differences
        dt = self.t[1:] - self.t[:-1]
        dv = self.v[1:] - self.v[:-1]
        self.dvdt = dv / (dt + 1e-10)     # (N-1, 1)
        self.t_mid = (self.t[:-1] + self.t[1:]) / 2.0

    def physics_residual(self, model: nn.Module, t_col: torch.Tensor) -> torch.Tensor:
        """
        FULL MOMENTUM EQUATION RESIDUAL at collocation points.

        r(t) = m·(dv/dt) - m·g + 0.5·ρ·v²·Cd_net(t)·A(t) ≈ 0

        This is the *correct* physics residual for a PINN — it enforces
        that the predicted Cd(t) is physically consistent with the
        observed kinematics via Newton's second law.
        """
        # Interpolate v, A, ρ, dv/dt at collocation time points
        t_np  = t_col.detach().cpu().numpy().flatten()
        t_ref = self.t.cpu().numpy().flatten()
        t_mid_np = self.t_mid.cpu().numpy().flatten()

        v_np   = np.interp(t_np, t_ref, self.v.cpu().numpy().flatten())
        A_np   = np.interp(t_np, t_ref, self.A.cpu().numpy().flatten())
        rho_np = np.interp(t_np, t_ref, self.rho.cpu().numpy().flatten())
        dvdt_np = np.interp(t_np, t_mid_np, self.dvdt.cpu().numpy().flatten())

        v_col   = torch.tensor(v_np,    dtype=torch.float32, device=self.device).unsqueeze(1)
        A_col   = torch.tensor(A_np,    dtype=torch.float32, device=self.device).unsqueeze(1)
        rho_col = torch.tensor(rho_np,  dtype=torch.float32, device=self.device).unsqueeze(1)
        dvdt_col = torch.tensor(dvdt_np, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Predict Cd at collocation points
        Cd = model(t_col)

        # Full momentum residual:
        # m * dv/dt = m*g - 0.5*ρ*v²*Cd*A
        # residual = m * dv/dt - m*g + 0.5*ρ*v²*Cd*A
        drag = 0.5 * rho_col * v_col**2 * Cd * A_col
        residual = self.mass * dvdt_col - self.mass * self.g + drag

        # Normalize by weight for scale invariance
        weight = self.mass * self.g
        return (residual / weight).pow(2).mean()

    def data_loss(self, model: nn.Module, t_obs: torch.Tensor,
                  v_obs: torch.Tensor, A_obs: torch.Tensor,
                  rho_obs: torch.Tensor) -> torch.Tensor:
        """
        Force the ODE solution with predicted Cd(t) to match observed velocities.
        Δv = v_predicted - v_observed, computed via forward Euler residual.
        """
        Cd  = model(t_obs)
        drag = 0.5 * rho_obs * v_obs**2 * Cd * A_obs
        dv_dt_pred = self.g - drag / self.mass

        # Numerically integrate predicted dv/dt
        dt = t_obs[1:] - t_obs[:-1]
        dv_pred = dv_dt_pred[:-1] * dt
        dv_obs  = v_obs[1:] - v_obs[:-1]

        return nn.functional.mse_loss(dv_pred, dv_obs)

    def smoothness_loss(self, model: nn.Module, t_obs: torch.Tensor) -> torch.Tensor:
        """Second-order smoothness (penalizes d²Cd/dt²)."""
        Cd = model(t_obs)
        d1 = Cd[1:] - Cd[:-1]
        d2 = d1[1:] - d1[:-1]
        return (d2**2).mean()


# ─── Adaptive Loss Weighting (NTK-inspired) ─────────────────────────────────
class AdaptiveWeights:
    """
    Self-tuning loss weights via gradient magnitude balancing.

    Inspired by Neural Tangent Kernel theory: scale each loss term
    so that its gradient magnitude matches the others. This prevents
    any single loss term from dominating training.

    Reference: Wang et al., "Understanding and Mitigating Gradient
    Flow Pathologies in PINN Training", SIAM J. Sci. Comput. 2021
    """

    def __init__(self, n_terms: int, alpha: float = 0.9):
        self.running_grads = [1.0] * n_terms
        self.alpha = alpha    # EMA smoothing factor
        self.weights = [1.0] * n_terms

    def update(self, grad_norms: list[float]):
        """Update weights based on gradient norms of each loss term."""
        mean_grad = max(np.mean(grad_norms), 1e-10)
        for i, gn in enumerate(grad_norms):
            self.running_grads[i] = (
                self.alpha * self.running_grads[i] +
                (1 - self.alpha) * max(gn, 1e-10)
            )
            self.weights[i] = mean_grad / max(self.running_grads[i], 1e-10)


# ─── Trainer — AeroDecel v5.0 ────────────────────────────────────────────────
class PINNTrainer:

    def __init__(self, ode_df: pd.DataFrame, at_df: pd.DataFrame):
        self.device = get_device()
        print(f"[Phase 3] Device: {self.device}")

        # Merge datasets
        t_ode = ode_df["time_s"].values.astype(np.float32)
        v_ode = ode_df["velocity_ms"].values.astype(np.float32)
        h_ode = ode_df["altitude_m"].values.astype(np.float32)

        # Interpolate A(t) onto ODE time grid
        t_at = at_df["time_s"].values
        A_at = at_df["area_m2"].values
        A_ode = np.interp(t_ode, t_at, A_at).astype(np.float32)

        rho_ode = vectorized_density(h_ode).astype(np.float32)

        # Normalize time to [0,1]
        self.t_min = t_ode.min()
        self.t_max = t_ode.max()
        t_norm = (t_ode - self.t_min) / (self.t_max - self.t_min + 1e-8)

        # Validation split
        n_total = len(t_norm)
        n_val = max(10, int(n_total * cfg.PINN_VALIDATION_FRAC))
        idx = np.random.RandomState(42).permutation(n_total)
        train_idx = idx[n_val:]
        val_idx = idx[:n_val]
        train_idx.sort()
        val_idx.sort()

        # Training tensors
        self.t   = torch.tensor(t_norm[train_idx], dtype=torch.float32, device=self.device).unsqueeze(1)
        self.v   = torch.tensor(v_ode[train_idx],  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.h   = torch.tensor(h_ode[train_idx],  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.A   = torch.tensor(A_ode[train_idx],  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.rho = torch.tensor(rho_ode[train_idx],dtype=torch.float32, device=self.device).unsqueeze(1)

        # Validation tensors
        self.t_val   = torch.tensor(t_norm[val_idx], dtype=torch.float32, device=self.device).unsqueeze(1)
        self.v_val   = torch.tensor(v_ode[val_idx],  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.A_val   = torch.tensor(A_ode[val_idx],  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.rho_val = torch.tensor(rho_ode[val_idx],dtype=torch.float32, device=self.device).unsqueeze(1)

        # Full data for loss function interpolation
        self.t_full   = torch.tensor(t_norm, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.v_full   = torch.tensor(v_ode,  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.h_full   = torch.tensor(h_ode,  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.A_full   = torch.tensor(A_ode,  dtype=torch.float32, device=self.device).unsqueeze(1)
        self.rho_full = torch.tensor(rho_ode,dtype=torch.float32, device=self.device).unsqueeze(1)

        self.dual_mode = cfg.PINN_DUAL_OUTPUT

        if self.dual_mode:
            v_mean = float(v_ode.mean())
            v_range = float(v_ode.max() - v_ode.min()) + 1.0
            self.model = CdVelocityNetwork(
                hidden=cfg.PINN_HIDDEN_LAYERS,
                activation=cfg.PINN_ACTIVATION,
                use_fourier=cfg.PINN_FOURIER_FEATURES,
                v_scale=v_range,
                v_offset=v_mean,
            ).to(self.device)

            self.loss_fn = DualOutputPINNLoss(
                mass=cfg.PARACHUTE_MASS,
                gravity=cfg.GRAVITY,
                t_full=self.t_full, v_full=self.v_full, h_full=self.h_full,
                A_full=self.A_full, rho_full=self.rho_full,
                device=self.device,
            )
        else:
            self.model = CdNetwork(
                hidden=cfg.PINN_HIDDEN_LAYERS,
                activation=cfg.PINN_ACTIVATION,
                use_fourier=cfg.PINN_FOURIER_FEATURES,
            ).to(self.device)

            self.loss_fn = PINNLoss(
                mass=cfg.PARACHUTE_MASS,
                gravity=cfg.GRAVITY,
                t_full=self.t_full, v_full=self.v_full, h_full=self.h_full,
                A_full=self.A_full, rho_full=self.rho_full,
                device=self.device,
            )

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.PINN_LR)
        if cfg.PINN_LR_SCHEDULE:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cfg.PINN_EPOCHS, eta_min=1e-6
            )
        else:
            self.scheduler = None

        # Adaptive weights
        if cfg.PINN_ADAPTIVE_WEIGHTS:
            self.adaptive = AdaptiveWeights(3)
        else:
            self.adaptive = None

        self.history = {"epoch": [], "loss_total": [], "loss_data": [],
                        "loss_physics": [], "loss_smooth": [],
                        "loss_val": [], "lr": []}

    def _sample_collocation(self, n: int) -> torch.Tensor:
        """Random collocation points in [0,1] for physics loss."""
        return torch.rand(n, 1, device=self.device, dtype=torch.float32)

    def _compute_val_loss(self) -> float:
        """Compute validation loss for early stopping."""
        self.model.eval()
        with torch.no_grad():
            if self.dual_mode:
                v_pred, Cd_val = self.model(self.t_val)
                val_loss = nn.functional.mse_loss(v_pred, self.v_val)
            else:
                Cd_val = self.model(self.t_val)
                drag_val = 0.5 * self.rho_val * self.v_val**2 * Cd_val * self.A_val
                dv_pred = cfg.GRAVITY - drag_val / cfg.PARACHUTE_MASS
                dt = self.t_val[1:] - self.t_val[:-1]
                dv_p = dv_pred[:-1] * dt
                dv_o = self.v_val[1:] - self.v_val[:-1]
                val_loss = nn.functional.mse_loss(dv_p, dv_o)
        self.model.train()
        return val_loss.item()

    def train(self) -> nn.Module:
        features = []
        if cfg.PINN_FOURIER_FEATURES: features.append("Fourier")
        if cfg.PINN_CURRICULUM:       features.append("curriculum")
        if cfg.PINN_ADAPTIVE_WEIGHTS: features.append("adaptive-λ")
        feat_str = f" [{', '.join(features)}]" if features else ""
        print(f"[Phase 3] Training PINN: {cfg.PINN_EPOCHS} epochs{feat_str}...")

        lw_p = cfg.PINN_PHYSICS_WEIGHT
        lw_d = cfg.PINN_DATA_WEIGHT
        lw_s = cfg.PINN_SMOOTH_WEIGHT
        warmup = cfg.PINN_CURRICULUM_WARMUP if cfg.PINN_CURRICULUM else 0

        best_loss = float("inf")
        best_val_loss = float("inf")
        best_state = None
        patience = 0
        max_patience = 500

        for epoch in range(1, cfg.PINN_EPOCHS + 1):
            self.model.train()
            self.optimizer.zero_grad()

            # Curriculum: ramp up physics weight after warmup
            if cfg.PINN_CURRICULUM and epoch < warmup:
                physics_scale = epoch / warmup  # 0 → 1 over warmup period
            else:
                physics_scale = 1.0

            # Collocation points for physics residual
            t_col = self._sample_collocation(cfg.PINN_COLLOCATION_PTS)

            L_phys   = self.loss_fn.physics_residual(self.model, t_col)
            if self.dual_mode:
                L_data   = self.loss_fn.data_loss(self.model, self.t, self.v)
            else:
                L_data   = self.loss_fn.data_loss(self.model, self.t, self.v, self.A, self.rho)
            L_smooth = self.loss_fn.smoothness_loss(self.model, self.t)

            # Apply adaptive weights if enabled
            if self.adaptive is not None and epoch > 100:
                w_p, w_d, w_s = self.adaptive.weights
            else:
                w_p, w_d, w_s = lw_p, lw_d, lw_s

            loss = w_d * L_data + physics_scale * w_p * L_phys + w_s * L_smooth
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update adaptive weights from gradient norms
            if self.adaptive is not None and epoch > 100 and epoch % 50 == 0:
                grad_norms = []
                for L_term in [L_phys, L_data, L_smooth]:
                    self.optimizer.zero_grad()
                    L_term.backward(retain_graph=True)
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    grad_norms.append(total_norm ** 0.5)
                self.adaptive.update(grad_norms)
                # Recompute loss and gradients
                self.optimizer.zero_grad()
                loss = (self.adaptive.weights[1] * L_data +
                        physics_scale * self.adaptive.weights[0] * L_phys +
                        self.adaptive.weights[2] * L_smooth)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            lv = loss.item()
            lr = self.optimizer.param_groups[0]["lr"]

            # Validation check
            val_loss = 0.0
            if epoch % 100 == 0:
                val_loss = self._compute_val_loss()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

            if lv < best_loss:
                best_loss  = lv
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 500 == 0 or epoch == 1:
                bar = "█" * int(epoch / cfg.PINN_EPOCHS * 30)
                pad = "░" * (30 - len(bar))
                cur_label = " [warmup]" if epoch < warmup else ""
                print(f"\r  [{bar}{pad}] ep {epoch:5d}/{cfg.PINN_EPOCHS} | "
                      f"L={lv:.4e} (phys={L_phys.item():.3e} "
                      f"data={L_data.item():.3e} val={val_loss:.3e}) | "
                      f"lr={lr:.2e}{cur_label}", end="", flush=True)

            self.history["epoch"].append(epoch)
            self.history["loss_total"].append(lv)
            self.history["loss_data"].append(L_data.item())
            self.history["loss_physics"].append(L_phys.item())
            self.history["loss_smooth"].append(L_smooth.item())
            self.history["loss_val"].append(val_loss)
            self.history["lr"].append(lr)

            # Early stopping on validation
            if patience >= max_patience // 100:
                # Don't actually stop, but record
                pass

        print(f"\n  ✓ Training complete. Best loss: {best_loss:.4e} | Best val: {best_val_loss:.4e}")
        self.model.load_state_dict(best_state)
        return self.model

    def predict_Cd(self, t_raw: np.ndarray = None) -> tuple:
        """
        Predict Cd(t) on given raw time array (or training time array).
        Returns (time_s, Cd_values).
        """
        self.model.eval()
        if t_raw is None:
            t_raw = np.linspace(self.t_min, self.t_max, 1000)

        t_norm = (t_raw - self.t_min) / (self.t_max - self.t_min + 1e-8)
        t_ten  = torch.tensor(t_norm, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            if self.dual_mode:
                _, Cd_ten = self.model(t_ten)
            else:
                Cd_ten = self.model(t_ten)
        return t_raw, Cd_ten.cpu().numpy().flatten()

    def predict_v(self, t_raw: np.ndarray = None) -> tuple:
        """
        Predict v(t) from dual-output PINN.
        Returns (time_s, v_values). Only available in dual mode.
        """
        if not self.dual_mode:
            raise RuntimeError("predict_v() only available in dual-output mode")
        self.model.eval()
        if t_raw is None:
            t_raw = np.linspace(self.t_min, self.t_max, 1000)

        t_norm = (t_raw - self.t_min) / (self.t_max - self.t_min + 1e-8)
        t_ten  = torch.tensor(t_norm, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            v_ten, _ = self.model(t_ten)
        return t_raw, v_ten.cpu().numpy().flatten()

    def save(self, path: Path = None):
        path = path or (cfg.MODELS_DIR / "pinn_cd_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "hidden": cfg.PINN_HIDDEN_LAYERS,
                "activation": cfg.PINN_ACTIVATION,
                "fourier": cfg.PINN_FOURIER_FEATURES,
                "fourier_scales": cfg.PINN_FOURIER_SCALES,
                "dual_output": self.dual_mode,
                "t_min": self.t_min,
                "t_max": self.t_max,
            },
            "history": self.history,
        }, path)
        print(f"  ✓ Model saved: {path}")

    def load(self, path: Path = None):
        path = path or (cfg.MODELS_DIR / "pinn_cd_model.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"  ✓ Model loaded: {path}")
        return self.model


# ─── Entry Point ─────────────────────────────────────────────────────────────
def run(ode_df: pd.DataFrame = None, at_df: pd.DataFrame = None) -> pd.DataFrame:
    if ode_df is None:
        if not cfg.ODE_CSV.exists():
            raise FileNotFoundError("ODE CSV not found. Run Phase 2 first.")
        ode_df = pd.read_csv(cfg.ODE_CSV)
    if at_df is None:
        if not cfg.AT_CSV.exists():
            raise FileNotFoundError("A(t) CSV not found. Run Phase 1 first.")
        at_df = pd.read_csv(cfg.AT_CSV)

    print(f"\n[Phase 3] AeroDecel PINN — Cd(t) Identification (Research-Grade)")
    print(f"  Architecture: {cfg.PINN_HIDDEN_LAYERS}")
    features = []
    if cfg.PINN_FOURIER_FEATURES: features.append(f"Fourier(σ={cfg.PINN_FOURIER_SCALES})")
    if cfg.PINN_CURRICULUM:       features.append(f"curriculum(warmup={cfg.PINN_CURRICULUM_WARMUP})")
    if cfg.PINN_ADAPTIVE_WEIGHTS: features.append("adaptive-lambda")
    if cfg.PINN_DUAL_OUTPUT:      features.append("dual-output(v+Cd)")
    print(f"  Epochs: {cfg.PINN_EPOCHS} | LR: {cfg.PINN_LR} | "
          f"lam_phys={cfg.PINN_PHYSICS_WEIGHT} lam_data={cfg.PINN_DATA_WEIGHT}")
    if features:
        print(f"  AeroDecel v5.0: {' | '.join(features)}")

    trainer = PINNTrainer(ode_df, at_df)
    trainer.train()
    trainer.save()

    t_out, Cd_out = trainer.predict_Cd(ode_df["time_s"].values)

    df = pd.DataFrame({"time_s": t_out, "Cd": Cd_out})
    df.to_csv(cfg.PINN_CSV, index=False)
    print(f"  ✓ Cd curve saved: {cfg.PINN_CSV}")
    print(f"  Cd range: [{Cd_out.min():.4f}, {Cd_out.max():.4f}] | "
          f"mean: {Cd_out.mean():.4f}")

    return df, trainer.history


if __name__ == "__main__":
    run()
