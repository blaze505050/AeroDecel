"""
pinn_ensemble.py — PINN Ensemble with Epistemic + Aleatoric Uncertainty
Requires: torch (pip install torch)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.atmosphere import vectorized_density

# Optional torch imports - graceful degradation
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.phase3_pinn import CdNetwork, PINNLoss, get_device
    _TORCH_AVAILABLE = True
except ImportError:
    torch = nn = optim = CosineAnnealingLR = CdNetwork = PINNLoss = get_device = None

def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required. Install: pip install torch")

def _diverse_configs(n: int, strategy: str) -> list[dict]:
    base_hidden = cfg.PINN_HIDDEN_LAYERS
    activations = ["tanh", "silu", "gelu", "mish", "elu"]
    seeds = list(range(n * 7, n * 7 + n))
    configs = []
    for i in range(n):
        seed = seeds[i]
        act  = activations[i % len(activations)]
        if strategy == "seeds":
            hidden = base_hidden; act = "tanh"
        elif strategy == "width":
            scale  = 0.7 + 0.6 * (i / max(n-1, 1))
            hidden = [max(16, int(h * scale)) for h in base_hidden]
        elif strategy == "depth":
            delta  = (i % 3) - 1
            hidden = base_hidden[1:] if delta < 0 else base_hidden + [base_hidden[-1]] if delta > 0 else base_hidden
        elif strategy == "act":
            hidden = base_hidden
        else:
            scale  = 0.8 + 0.4 * (i / max(n-1, 1))
            hidden = [max(16, int(h * scale)) for h in base_hidden]
            if i >= n // 2: hidden = hidden + [hidden[-1]]
        configs.append({"hidden": hidden, "activation": act, "seed": seed})
    return configs

class HeteroscedasticCdNetwork(nn.Module if nn is not None else object):
    """
    Extends CdNetwork to output both Cd(t) and log σ²(t) for aleatoric estimation.
    The second output head predicts the log of the per-point noise variance.

    Loss uses NLL under Gaussian likelihood:
        L_nll = 0.5 * [log σ² + (y - ŷ)² / σ²]

    This forces the model to be less confident where it fits poorly.
    """

    def __init__(self, hidden: list = None, activation: str = "tanh"):
        super().__init__()
        hidden = hidden or cfg.PINN_HIDDEN_LAYERS

        act_map = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU,
                   "mish": nn.Mish, "elu":  nn.ELU}
        act_cls = act_map.get(activation, nn.Tanh)

        layers = []; in_dim = 1
        self.residual_idx = []
        for i, h in enumerate(hidden):
            block = nn.Sequential(nn.Linear(in_dim, h), act_cls())
            layers.append(block)
            if in_dim == h: self.residual_idx.append(i)
            in_dim = h

        self.layers   = nn.ModuleList(layers)
        self.out_Cd   = nn.Linear(in_dim, 1)    # Cd head
        self.out_lsig = nn.Linear(in_dim, 1)    # log σ² head

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = t
        for i, layer in enumerate(self.layers):
            out = layer(x)
            if i in self.residual_idx and out.shape == x.shape:
                out = out + x
            x = out

        Cd   = 0.5 + torch.nn.functional.softplus(self.out_Cd(x))
        lsig = torch.clamp(self.out_lsig(x), min=-6.0, max=4.0)   # log σ² ∈ [-6, 4]
        return Cd, lsig

    def predict_Cd(self, t: torch.Tensor) -> torch.Tensor:
        """Convenience: return only Cd (for compatibility with PINNLoss)."""
        Cd, _ = self.forward(t)
        return Cd


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ARCHITECTURE DIVERSITY CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _diverse_configs(n: int, strategy: str) -> list[dict]:
    """
    Generate n diverse (hidden_layers, activation, seed) configurations.
    """
    base_hidden = cfg.PINN_HIDDEN_LAYERS
    activations = ["tanh", "silu", "gelu", "mish", "elu"]
    seeds       = list(range(n * 7, n * 7 + n))   # deterministic seeds

    configs = []
    for i in range(n):
        seed   = seeds[i]
        act    = activations[i % len(activations)]

        if strategy == "seeds":
            hidden = base_hidden
            act    = "tanh"

        elif strategy == "width":
            scale  = 0.7 + 0.6 * (i / max(n-1, 1))
            hidden = [max(16, int(h * scale)) for h in base_hidden]

        elif strategy == "depth":
            depth_delta = (i % 3) - 1   # -1, 0, +1
            if depth_delta < 0:
                hidden = base_hidden[1:]
            elif depth_delta > 0:
                hidden = base_hidden + [base_hidden[-1]]
            else:
                hidden = base_hidden

        elif strategy == "act":
            hidden = base_hidden

        else:  # "full" — combine all variations
            scale  = 0.8 + 0.4 * (i / max(n-1, 1))
            hidden = [max(16, int(h * scale)) for h in base_hidden]
            if i >= n // 2:
                hidden = hidden + [hidden[-1]]   # add extra layer for half

        configs.append({"hidden": hidden, "activation": act, "seed": seed})

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SINGLE-MEMBER TRAINER  (heteroscedastic)
# ══════════════════════════════════════════════════════════════════════════════

def _train_member(
    member_id:    int,
    config:       dict,
    ode_df:       pd.DataFrame,
    at_df:        pd.DataFrame,
    n_epochs:     int,
    device,
    verbose:      bool = True,
) -> tuple:
    """Train one ensemble member. Returns (model, loss_history)."""
    _import_torch()
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HeteroscedasticCdNetwork(
        hidden     = config["hidden"],
        activation = config["activation"],
    ).to(device)

    t_ode = ode_df["time_s"].values.astype(np.float32)
    v_ode = ode_df["velocity_ms"].values.astype(np.float32)
    h_ode = ode_df["altitude_m"].values.astype(np.float32)
    t_at  = at_df["time_s"].values
    A_at  = at_df["area_m2"].values
    A_ode = np.interp(t_ode, t_at, A_at).astype(np.float32)
    rho_ode = vectorized_density(h_ode).astype(np.float32)

    t_min, t_max = t_ode.min(), t_ode.max()
    t_norm = (t_ode - t_min) / (t_max - t_min + 1e-8)

    t_ten   = torch.tensor(t_norm,  dtype=torch.float32, device=device).unsqueeze(1)
    v_ten   = torch.tensor(v_ode,   dtype=torch.float32, device=device).unsqueeze(1)
    A_ten   = torch.tensor(A_ode,   dtype=torch.float32, device=device).unsqueeze(1)
    rho_ten = torch.tensor(rho_ode, dtype=torch.float32, device=device).unsqueeze(1)

    # Adapt PINNLoss to use heteroscedastic model's predict_Cd method
    class _HeteroModel(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, t): return self.m.predict_Cd(t)

    loss_fn = PINNLoss(
        mass     = cfg.PARACHUTE_MASS,
        gravity  = cfg.GRAVITY,
        t_full   = t_ten, v_full=v_ten, h_full=torch.tensor(h_ode).unsqueeze(1),
        A_full   = A_ten, rho_full=rho_ten, device=device,
    )

    opt = optim.Adam(model.parameters(), lr=cfg.PINN_LR * (0.8 + 0.4 * (member_id % 3) / 2))
    sch = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    lw_phys = cfg.PINN_PHYSICS_WEIGHT
    lw_data = cfg.PINN_DATA_WEIGHT
    lw_smth = 0.1
    lw_nll  = 1.0   # heteroscedastic NLL weight

    history = {"loss": [], "loss_nll": [], "loss_phys": []}
    best_loss  = float("inf")
    best_state = None

    _wrap = _HeteroModel(model)

    for epoch in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()

        t_col = torch.rand(cfg.PINN_COLLOCATION_PTS // 2, 1, device=device)

        # Physics + smoothness loss (using wrapped model)
        L_phys   = loss_fn.physics_residual(_wrap, t_col)
        L_data   = loss_fn.data_loss(_wrap, t_ten, v_ten, A_ten, rho_ten)
        L_smooth = loss_fn.smoothness_loss(_wrap, t_ten)

        # Heteroscedastic NLL loss
        Cd_pred, lsig_pred = model(t_ten)
        # NLL w.r.t. velocity residual from ODE
        dv_pred  = cfg.GRAVITY - (0.5 * rho_ten * v_ten**2 * Cd_pred * A_ten) / cfg.PARACHUTE_MASS
        dv_obs   = torch.gradient(v_ten.squeeze(), spacing=(t_ten.squeeze(),))[0].unsqueeze(1)
        residual = (dv_pred - dv_obs) ** 2
        L_nll    = 0.5 * (torch.exp(-lsig_pred) * residual + lsig_pred).mean()

        loss = (lw_phys * L_phys + lw_data * L_data +
                lw_smth * L_smooth + lw_nll * L_nll)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sch.step()

        lv = float(loss)
        if lv < best_loss:
            best_loss  = lv
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history["loss"].append(lv)
        history["loss_nll"].append(float(L_nll))
        history["loss_phys"].append(float(L_phys))

        if verbose and epoch % max(1, n_epochs // 5) == 0:
            pct = epoch / n_epochs * 100
            print(f"\r    [{member_id+1}] {pct:5.1f}%  loss={lv:.4e}  nll={float(L_nll):.3e}",
                  end="", flush=True)

    if verbose: print()
    if best_state: model.load_state_dict(best_state)
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

class PINNEnsemble:
    """
    Manages an ensemble of N heteroscedastic PINNs.
    Provides uncertainty-quantified Cd(t) prediction.
    """

    def __init__(
        self,
        n_members:  int   = 5,
        strategy:   str   = "full",   # seeds | width | depth | act | full
        n_epochs:   int   = None,
    ):
        self.n_members = n_members
        self.strategy  = strategy
        self.n_epochs  = n_epochs or max(500, cfg.PINN_EPOCHS // 4)
        self.device    = get_device()
        self.members:  list[HeteroscedasticCdNetwork] = []
        self.configs:  list[dict] = []
        self.histories: list[dict] = []
        self._t_min = 0.0
        self._t_max = 1.0

    def train(
        self,
        ode_df: pd.DataFrame,
        at_df:  pd.DataFrame,
        verbose: bool = True,
    ) -> "PINNEnsemble":
        _import_torch()
        self.configs = _diverse_configs(self.n_members, self.strategy)
        self._t_min  = float(ode_df["time_s"].min())
        self._t_max  = float(ode_df["time_s"].max())

        if verbose:
            print(f"\n[Ensemble] Training {self.n_members} members  "
                  f"strategy={self.strategy}  epochs={self.n_epochs}  device={self.device}")

        self.members   = []
        self.histories = []

        for i, cfg_i in enumerate(self.configs):
            if verbose:
                print(f"\n  Member {i+1}/{self.n_members}: "
                      f"hidden={cfg_i['hidden']}  act={cfg_i['activation']}  seed={cfg_i['seed']}")
            t0 = time.perf_counter()
            model, history = _train_member(
                member_id=i, config=cfg_i,
                ode_df=ode_df, at_df=at_df,
                n_epochs=self.n_epochs, device=self.device, verbose=verbose,
            )
            elapsed = time.perf_counter() - t0
            self.members.append(model)
            self.histories.append(history)
            if verbose:
                print(f"    ✓ Done in {elapsed:.1f}s  final loss={history['loss'][-1]:.4e}")

        return self

    def predict(
        self,
        t_raw:   np.ndarray,
        n_sigma: int = 2,
    ) -> dict:
        """
        Predict Cd(t) with full uncertainty decomposition.

        Returns
        -------
        dict with:
          t:           time array
          Cd_mean:     ensemble mean
          Cd_members:  per-member curves (shape n_members × n_t)
          sigma_epist: epistemic std (from member disagreement)
          sigma_aleat: aleatoric std (from per-model noise prediction)
          sigma_total: total predictive std
          ci_epist:    ±n_sigma epistemic band
          ci_total:    ±n_sigma total band
        """
        t_norm = (t_raw - self._t_min) / (self._t_max - self._t_min + 1e-8)
        t_ten  = torch.tensor(t_norm.astype(np.float32),
                               device=self.device, dtype=torch.float32).unsqueeze(1)

        Cd_preds   = []
        sigma_aleat_preds = []

        for model in self.members:
            model.eval()
            with torch.no_grad():
                Cd_k, lsig_k = model(t_ten)
            Cd_preds.append(Cd_k.cpu().numpy().flatten())
            sigma_aleat_preds.append(np.exp(0.5 * lsig_k.cpu().numpy().flatten()))

        Cd_mat       = np.array(Cd_preds)             # (n_members, n_t)
        sigma_a_mat  = np.array(sigma_aleat_preds)

        Cd_mean    = Cd_mat.mean(axis=0)
        sigma_epist = Cd_mat.std(axis=0)               # disagreement = epistemic
        sigma_aleat = sigma_a_mat.mean(axis=0)         # mean aleatoric
        sigma_total = np.sqrt(sigma_epist**2 + sigma_aleat**2)

        return {
            "t":             t_raw.tolist(),
            "Cd_mean":       Cd_mean.tolist(),
            "Cd_median":     np.median(Cd_mat, axis=0).tolist(),
            "Cd_members":    Cd_mat.tolist(),
            "sigma_epist":   sigma_epist.tolist(),
            "sigma_aleat":   sigma_aleat.tolist(),
            "sigma_total":   sigma_total.tolist(),
            # Credible intervals
            "Cd_epist_lo":   (Cd_mean - n_sigma * sigma_epist).tolist(),
            "Cd_epist_hi":   (Cd_mean + n_sigma * sigma_epist).tolist(),
            "Cd_total_lo":   (Cd_mean - n_sigma * sigma_total).tolist(),
            "Cd_total_hi":   (Cd_mean + n_sigma * sigma_total).tolist(),
            "Cd_p05":        np.percentile(Cd_mat, 5,  axis=0).tolist(),
            "Cd_p95":        np.percentile(Cd_mat, 95, axis=0).tolist(),
            # Scalar summaries
            "Cd_global_mean":   float(Cd_mean.mean()),
            "Cd_global_std":    float(sigma_epist.mean()),
            "epistemic_frac":   float(sigma_epist.mean() / max(sigma_total.mean(), 1e-9)),
            "aleatoric_frac":   float(sigma_aleat.mean() / max(sigma_total.mean(), 1e-9)),
        }

    def save(self, path: Path | None = None):
        """Save all ensemble members + metadata."""
        path = path or (cfg.MODELS_DIR / "pinn_ensemble")
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_members":  self.n_members,
            "strategy":   self.strategy,
            "n_epochs":   self.n_epochs,
            "t_min":      self._t_min,
            "t_max":      self._t_max,
            "configs":    self.configs,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        for i, (model, history) in enumerate(zip(self.members, self.histories)):
            torch.save({
                "state": model.state_dict(),
                "config": self.configs[i],
                "history": history,
            }, path / f"member_{i:02d}.pt")
        print(f"  ✓ Ensemble saved: {path}  ({self.n_members} members)")

    def load(self, path: Path | None = None) -> "PINNEnsemble":
        _import_torch()
        path = path or (cfg.MODELS_DIR / "pinn_ensemble")
        meta = json.loads((path / "meta.json").read_text())
        self.n_members = meta["n_members"]
        self.strategy  = meta["strategy"]
        self.n_epochs  = meta["n_epochs"]
        self._t_min    = meta["t_min"]
        self._t_max    = meta["t_max"]
        self.configs   = meta["configs"]
        self.members   = []; self.histories = []
        for i in range(self.n_members):
            ckpt = torch.load(path / f"member_{i:02d}.pt", map_location=self.device)
            model = HeteroscedasticCdNetwork(
                hidden=ckpt["config"]["hidden"],
                activation=ckpt["config"]["activation"],
            ).to(self.device)
            model.load_state_dict(ckpt["state"])
            self.members.append(model)
            self.histories.append(ckpt["history"])
        print(f"  ✓ Ensemble loaded: {path}  ({self.n_members} members)")
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_ensemble(
    pred:       dict,
    histories:  list[dict],
    ode_df:     pd.DataFrame | None = None,
    save_path:  Path | None = None,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if cfg.DARK_THEME:
        matplotlib.rcParams.update({
            "figure.facecolor": "#080c14", "axes.facecolor": "#0d1526",
            "axes.edgecolor": "#2a3d6e",   "text.color": "#c8d8f0",
            "axes.labelcolor": "#c8d8f0",  "xtick.color": "#c8d8f0",
            "ytick.color": "#c8d8f0",      "grid.color": "#1a2744",
        })
    matplotlib.rcParams.update({"font.family": "monospace", "font.size": 9})

    TEXT = "#c8d8f0" if cfg.DARK_THEME else "#111"
    C1   = cfg.COLOR_THEORY   # cyan
    C2   = cfg.COLOR_PINN     # orange
    C3   = cfg.COLOR_RAW      # green
    C_EP = "#9d60ff"          # purple — epistemic

    t  = np.array(pred["t"])
    Cm = np.array(pred["Cd_mean"])
    Ce_lo = np.array(pred["Cd_epist_lo"])
    Ce_hi = np.array(pred["Cd_epist_hi"])
    Ct_lo = np.array(pred["Cd_total_lo"])
    Ct_hi = np.array(pred["Cd_total_hi"])
    s_ep  = np.array(pred["sigma_epist"])
    s_al  = np.array(pred["sigma_aleat"])
    s_tot = np.array(pred["sigma_total"])
    mems  = np.array(pred["Cd_members"])   # (n, t)

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                            top=0.91, bottom=0.07, left=0.06, right=0.97)

    def ax(r, c): return fig.add_subplot(gs[r, c])
    def style(a, title, xlabel, ylabel):
        a.set_title(title, fontweight="bold", pad=5, fontsize=9)
        a.set_xlabel(xlabel, fontsize=8); a.set_ylabel(ylabel, fontsize=8)
        a.grid(True, alpha=0.3); a.spines[["top","right"]].set_visible(False)

    # ── P0: Ensemble Cd(t) + uncertainty bands ────────────────────────────────
    ax0 = ax(0, 0)
    # Individual members (light, behind)
    for i, cd_k in enumerate(mems):
        ax0.plot(t, cd_k, lw=0.6, alpha=0.35, color=C1)
    # Total band
    ax0.fill_between(t, Ct_lo, Ct_hi, alpha=0.15, color=C2, label="±2σ total")
    # Epistemic band
    ax0.fill_between(t, Ce_lo, Ce_hi, alpha=0.30, color=C_EP, label="±2σ epistemic")
    # Mean
    ax0.plot(t, Cm, color=C1, lw=2.0, label=f"Ensemble mean (Cd̄={pred['Cd_global_mean']:.4f})")
    ax0.axhline(cfg.CD_INITIAL, color=TEXT, lw=0.7, ls=":", alpha=0.5,
                label=f"Prior Cd={cfg.CD_INITIAL}")
    ax0.legend(fontsize=7.5)
    style(ax0, "Ensemble Cd(t) with uncertainty", "Time [s]", "Cd [—]")

    # ── P1: Uncertainty decomposition ─────────────────────────────────────────
    ax1 = ax(0, 1)
    ax1.fill_between(t, 0, s_ep, alpha=0.6, color=C_EP, label="Epistemic σ")
    ax1.fill_between(t, s_ep, s_ep + s_al, alpha=0.5, color=C2, label="Aleatoric σ")
    ax1.plot(t, s_tot, color=C1, lw=1.5, label="Total σ")
    ax1.legend(fontsize=7.5)
    style(ax1, "Uncertainty decomposition σ(t)", "Time [s]", "σ(Cd) [—]")

    # ── P2: Epistemic / Total fraction over time ───────────────────────────────
    ax2 = ax(0, 2)
    epist_frac = s_ep / np.maximum(s_tot, 1e-9)
    aleat_frac = s_al / np.maximum(s_tot, 1e-9)
    ax2.fill_between(t, 0, epist_frac, alpha=0.7, color=C_EP, label="Epistemic fraction")
    ax2.fill_between(t, epist_frac, 1.0, alpha=0.7, color=C2, label="Aleatoric fraction")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7.5)
    style(ax2, "Uncertainty type fractions", "Time [s]", "Fraction of σ²_total")

    # ── P3: Per-member Cd curves (coloured) ───────────────────────────────────
    ax3 = ax(1, 0)
    cmap = plt.cm.plasma
    for i, cd_k in enumerate(mems):
        c = cmap(i / max(len(mems)-1, 1))
        ax3.plot(t, cd_k, lw=1.5, color=c, label=f"M{i+1}")
    ax3.legend(fontsize=7, ncol=2)
    style(ax3, "Individual member Cd(t) curves", "Time [s]", "Cd [—]")

    # ── P4: Training loss curves ───────────────────────────────────────────────
    ax4 = ax(1, 1)
    for i, hist in enumerate(histories):
        c = cmap(i / max(len(histories)-1, 1))
        ax4.semilogy(hist["loss"], lw=1.0, alpha=0.8, color=c, label=f"M{i+1}")
    ax4.legend(fontsize=7, ncol=2)
    style(ax4, "Training loss (all members)", "Epoch", "Loss")

    # ── P5: Disagreement heat-map ─────────────────────────────────────────────
    ax5 = ax(1, 2)
    n_cd_bins = 50
    Cd_range  = (max(0.3, Cm.min() - 3*s_tot.max()),
                  min(4.0, Cm.max() + 3*s_tot.max()))
    Cd_bins   = np.linspace(*Cd_range, n_cd_bins)
    heat      = np.zeros((n_cd_bins, len(t)))
    for cd_k in mems:
        idx = np.digitize(cd_k, Cd_bins) - 1
        idx = np.clip(idx, 0, n_cd_bins-1)
        for j, ix in enumerate(idx):
            heat[ix, j] += 1
    im = ax5.imshow(heat, aspect="auto", origin="lower",
                    extent=[t.min(), t.max(), Cd_range[0], Cd_range[1]],
                    cmap="YlOrRd" if not cfg.DARK_THEME else "inferno",
                    interpolation="nearest")
    ax5.plot(t, Cm, color="white", lw=1.5, alpha=0.8, label="Mean")
    plt.colorbar(im, ax=ax5, label="Member count", pad=0.02)
    ax5.set_xlabel("Time [s]"); ax5.set_ylabel("Cd")
    ax5.set_title("Ensemble disagreement heat-map", fontweight="bold")

    # ── P6: Posterior predictive velocity check ───────────────────────────────
    ax6 = ax(2, 0)
    if ode_df is not None:
        from src.calibrate_cd import _simulate
        t_ode = ode_df["time_s"].values
        v_ode = ode_df["velocity_ms"].values

        v_curves = []
        for cd_k in mems:
            Cd_fn = lambda t_q, _cd=np.array(pred["t"]), _cv=np.array(cd_k): float(
                np.interp(t_q, _cd, np.clip(_cv, 0.3, 4.0)))
            r = _simulate(1.0, Cd_fn=Cd_fn, dt=0.15)
            v_curves.append(np.interp(t_ode, r["time"], r["velocity"]))

        v_mat = np.array(v_curves)
        ax6.fill_between(t_ode, v_mat.min(axis=0), v_mat.max(axis=0),
                         alpha=0.2, color=C1, label="Member range")
        ax6.fill_between(t_ode,
                         np.percentile(v_mat, 16, axis=0),
                         np.percentile(v_mat, 84, axis=0),
                         alpha=0.3, color=C1, label="68% band")
        ax6.plot(t_ode, v_mat.mean(axis=0), color=C1, lw=1.8, label="Ensemble mean v(t)")
        ax6.plot(t_ode, v_ode, color=C3, lw=1.2, ls="--", alpha=0.7, label="ODE reference v(t)")
    style(ax6, "Posterior predictive v(t)", "Time [s]", "Velocity [m/s]")
    ax6.legend(fontsize=7.5)

    # ── P7: Per-member final Cd at t_max ─────────────────────────────────────
    ax7 = ax(2, 1)
    final_Cds = [mems[i, -1] for i in range(len(mems))]
    ax7.bar(range(1, len(mems)+1), final_Cds,
            color=[cmap(i/max(len(mems)-1,1)) for i in range(len(mems))],
            alpha=0.8, edgecolor="none")
    ax7.axhline(pred["Cd_global_mean"], color=C1, lw=1.5, ls="--",
                label=f"Mean={pred['Cd_global_mean']:.4f}")
    ax7.set_xlabel("Member"); ax7.set_ylabel("Cd at t_max")
    ax7.set_xticks(range(1, len(mems)+1))
    ax7.legend(fontsize=7.5)
    style(ax7, "Final Cd per member", "Member", "Cd")

    # ── P8: Summary metrics table ──────────────────────────────────────────────
    ax8 = ax(2, 2)
    ax8.axis("off")
    rows = [
        ("Ensemble members",   str(len(mems))),
        ("Strategy",           pred.get("strategy", "?")),
        ("",                   ""),
        ("Cd mean",            f"{pred['Cd_global_mean']:.5f}"),
        ("Cd std (epistemic)", f"{pred['Cd_global_std']:.5f}"),
        ("Epistemic fraction", f"{pred['epistemic_frac']*100:.1f}%"),
        ("Aleatoric fraction", f"{pred['aleatoric_frac']*100:.1f}%"),
        ("",                   ""),
        ("Cd P05–P95",         f"[{np.array(pred['Cd_p05']).mean():.4f}, "
                                f"{np.array(pred['Cd_p95']).mean():.4f}]"),
        ("Max σ_total",        f"{max(pred['sigma_total']):.5f}"),
        ("Mean σ_epist",       f"{np.mean(pred['sigma_epist']):.5f}"),
        ("Mean σ_aleat",       f"{np.mean(pred['sigma_aleat']):.5f}"),
    ]
    for j, (label, val) in enumerate(rows):
        ax8.text(0.02, 1-j*0.083, label, transform=ax8.transAxes, fontsize=8.5,
                 color=TEXT if cfg.DARK_THEME else "#555")
        ax8.text(0.98, 1-j*0.083, val, transform=ax8.transAxes, fontsize=8.5,
                 ha="right", color=C1 if label else TEXT)
    ax8.set_title("Ensemble summary", fontweight="bold")

    n_m = len(mems)
    ep_pct = pred["epistemic_frac"] * 100
    fig.text(0.5, 0.955,
             f"PINN Ensemble ({n_m} members, {pred.get('strategy','?')} strategy)  —  "
             f"Cd̄={pred['Cd_global_mean']:.4f}  σ_epist={pred['Cd_global_std']:.4f}  "
             f"Epistemic {ep_pct:.0f}% / Aleatoric {100-ep_pct:.0f}%",
             ha="center", fontsize=11, fontweight="bold",
             color=TEXT if cfg.DARK_THEME else "#111")

    sp = save_path or cfg.OUTPUTS_DIR / "pinn_ensemble.png"
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=cfg.DPI, bbox_inches="tight")
    print(f"  ✓ Ensemble plot saved: {sp}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(
    ode_df:    pd.DataFrame | None = None,
    at_df:     pd.DataFrame  | None = None,
    n_members: int   = 5,
    strategy:  str   = "full",
    n_epochs:  int   = None,
    verbose:   bool  = True,
) -> dict:
    """Run the full PINN ensemble training and uncertainty analysis."""
    import matplotlib; matplotlib.use("Agg")

    # ── Load data if not provided ─────────────────────────────────────────────
    if ode_df is None:
        if not cfg.ODE_CSV.exists():
            raise FileNotFoundError("ODE CSV not found. Run Phase 2 first.")
        ode_df = pd.read_csv(cfg.ODE_CSV)
    if at_df is None:
        if not cfg.AT_CSV.exists():
            raise FileNotFoundError("A(t) CSV not found. Run Phase 1 first.")
        at_df = pd.read_csv(cfg.AT_CSV)

    if verbose:
        print(f"\n[PINN Ensemble] {n_members} members  strategy={strategy}")

    # ── Train ──────────────────────────────────────────────────────────────────
    ensemble = PINNEnsemble(n_members=n_members, strategy=strategy, n_epochs=n_epochs)
    ensemble.train(ode_df=ode_df, at_df=at_df, verbose=verbose)
    ensemble.save()

    # ── Predict ───────────────────────────────────────────────────────────────
    t_raw = ode_df["time_s"].values
    pred  = ensemble.predict(t_raw)
    pred["strategy"] = strategy

    # ── Save prediction CSV ───────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "time_s":       pred["t"],
        "Cd_mean":      pred["Cd_mean"],
        "Cd_median":    pred["Cd_median"],
        "sigma_epist":  pred["sigma_epist"],
        "sigma_aleat":  pred["sigma_aleat"],
        "sigma_total":  pred["sigma_total"],
        "Cd_epist_lo":  pred["Cd_epist_lo"],
        "Cd_epist_hi":  pred["Cd_epist_hi"],
        "Cd_total_lo":  pred["Cd_total_lo"],
        "Cd_total_hi":  pred["Cd_total_hi"],
    })
    pred_df.to_csv(cfg.OUTPUTS_DIR / "pinn_ensemble_cd.csv", index=False)

    # ── Save summary JSON ──────────────────────────────────────────────────────
    summary = {k: v for k, v in pred.items() if k not in ("Cd_members", "t",
               "Cd_mean", "Cd_median", "sigma_epist", "sigma_aleat", "sigma_total",
               "Cd_epist_lo","Cd_epist_hi","Cd_total_lo","Cd_total_hi",
               "Cd_p05","Cd_p95")}
    summary["n_members"] = n_members
    summary["strategy"]  = strategy
    (cfg.OUTPUTS_DIR / "pinn_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_ensemble(pred, ensemble.histories, ode_df=ode_df)

    if verbose:
        print(f"\n  Cd_mean = {pred['Cd_global_mean']:.5f}")
        print(f"  σ_epist = {pred['Cd_global_std']:.5f}  "
              f"({pred['epistemic_frac']*100:.1f}% of total uncertainty)")
        print(f"  σ_aleat = {np.mean(pred['sigma_aleat']):.5f}  "
              f"({pred['aleatoric_frac']*100:.1f}% of total uncertainty)")

    return pred


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PINN Ensemble Uncertainty Quantification")
    p.add_argument("--n-members", type=int, default=5)
    p.add_argument("--strategy",  type=str, default="full",
                   choices=["seeds","width","depth","act","full"])
    p.add_argument("--epochs",    type=int, default=None)
    a = p.parse_args()
    run(n_members=a.n_members, strategy=a.strategy, n_epochs=a.epochs)
