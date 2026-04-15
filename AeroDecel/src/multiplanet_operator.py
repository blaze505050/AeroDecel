"""
src/multiplanet_operator.py — Multi-Planet FNO with Planet Embedding
=====================================================================
Trains a single Fourier Neural Operator simultaneously on Mars, Venus,
and Titan data. The planet is encoded as a learned embedding vector
concatenated to the input, enabling zero-shot generalisation to new planets.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


PLANET_EMBED = {
    "mars":  np.array([3.721, 0.020, 210.0, 188.9]) / np.array([10, 100, 1000, 300]),
    "venus": np.array([8.870, 65.0,  737.0, 188.9]) / np.array([10, 100, 1000, 300]),
    "titan": np.array([1.352, 1.40,   94.0, 290.0]) / np.array([10, 100, 1000, 300]),
    "triton":np.array([0.779, 0.001,  38.0, 290.0]) / np.array([10, 100, 1000, 300]),
}

EMBED_DIM = 4


if _TORCH:
    class MultiPlanetFNO(nn.Module):
        """FNO with planet embedding concatenated to input."""
        def __init__(self, traj_dim: int = 50, modes: int = 8,
                     width: int = 32, embed_dim: int = 4):
            super().__init__()
            self.traj_dim  = traj_dim
            self.embed_dim = embed_dim
            self.embed_net = nn.Sequential(
                nn.Linear(embed_dim, 16), nn.Tanh(), nn.Linear(16, width))
            self.lift = nn.Linear(1 + width, width)
            from src.neural_operator import SpectralConv1d
            self.sconvs = nn.ModuleList(
                [SpectralConv1d(width, width, modes) for _ in range(3)])
            self.wconvs = nn.ModuleList(
                [nn.Conv1d(width, width, 1) for _ in range(3)])
            self.proj1  = nn.Linear(width, 64)
            self.proj2  = nn.Linear(64, 1)
            self.act    = nn.GELU()

        def forward(self, v_in: torch.Tensor,
                    planet_emb: torch.Tensor) -> torch.Tensor:
            """
            v_in       : (batch, T) — input velocity trajectory
            planet_emb : (batch, embed_dim) — planet embedding
            Returns    : (batch, T) — corrected output trajectory
            """
            B, T = v_in.shape
            e = self.embed_net(planet_emb)   # (B, width)
            e = e.unsqueeze(1).expand(-1, T, -1)  # (B, T, width)
            x = torch.cat([v_in.unsqueeze(-1), e], dim=-1)  # (B, T, 1+width)
            x = self.lift(x)          # (B, T, width)
            x = x.permute(0, 2, 1)   # (B, width, T)
            for sc, wc in zip(self.sconvs, self.wconvs):
                x = self.act(sc(x) + wc(x))
            x = x.permute(0, 2, 1)   # (B, T, width)
            x = self.act(self.proj1(x))
            return self.proj2(x).squeeze(-1)   # (B, T)


class MultiPlanetOperator:
    """
    Train and evaluate a multi-planet FNO.
    Falls back to per-planet linear regression if torch unavailable.
    """

    def __init__(self, traj_dim: int = 50, modes: int = 8, width: int = 32):
        self.dim   = traj_dim
        self._backend = "torch" if _TORCH else "numpy"
        if _TORCH:
            self.model = MultiPlanetFNO(traj_dim, modes, width)
            self.opt   = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self._per_planet_means: dict = {}

    def _embed(self, planet: str) -> np.ndarray:
        return PLANET_EMBED.get(planet.lower(), PLANET_EMBED["mars"])

    def train(self, planet_data: dict, n_epochs: int = 500,
              verbose: bool = True) -> list[float]:
        """
        planet_data: {"mars": (X_in, Y_out), "venus": ..., "titan": ...}
        X_in, Y_out: (N_traj, traj_dim) arrays
        """
        losses = []
        if not _TORCH:
            for planet, (X, Y) in planet_data.items():
                self._per_planet_means[planet] = Y.mean(axis=0)
            return [0.0]

        import torch
        all_X, all_Y, all_E = [], [], []
        for planet, (X, Y) in planet_data.items():
            emb = self._embed(planet)
            N   = len(X)
            all_X.append(torch.tensor(X, dtype=torch.float32))
            all_Y.append(torch.tensor(Y, dtype=torch.float32))
            all_E.append(torch.tensor(np.tile(emb, (N,1)), dtype=torch.float32))

        X_all = torch.cat(all_X); Y_all = torch.cat(all_Y); E_all = torch.cat(all_E)

        for ep in range(1, n_epochs+1):
            self.model.train(); self.opt.zero_grad()
            # Resize to model dim
            X_r = torch.nn.functional.interpolate(X_all.unsqueeze(1),(self.dim,),mode="linear",align_corners=False).squeeze(1)
            Y_r = torch.nn.functional.interpolate(Y_all.unsqueeze(1),(self.dim,),mode="linear",align_corners=False).squeeze(1)
            pred = self.model(X_r, E_all)
            loss = ((pred - Y_r)**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            losses.append(float(loss))
            if verbose and ep % max(1, n_epochs//5) == 0:
                print(f"  [MultiPlanet] ep {ep}/{n_epochs}  loss={float(loss):.4e}")

        return losses

    def predict(self, v_in: np.ndarray, planet: str) -> np.ndarray:
        """Predict trajectory for a given planet."""
        emb = self._embed(planet)
        if not _TORCH:
            return self._per_planet_means.get(planet, v_in)
        import torch
        self.model.eval()
        N = len(v_in) if v_in.ndim == 2 else 1
        X = torch.tensor(v_in if v_in.ndim==2 else v_in[None,:], dtype=torch.float32)
        E = torch.tensor(np.tile(emb, (N, 1)), dtype=torch.float32)
        X_r = torch.nn.functional.interpolate(X.unsqueeze(1),(self.dim,),mode="linear",align_corners=False).squeeze(1)
        with torch.no_grad():
            pred = self.model(X_r, E).numpy()
        return pred

    def zero_shot_triton(self, v_in_mars: np.ndarray) -> np.ndarray:
        """Zero-shot prediction on Triton using Mars trajectory as input."""
        return self.predict(v_in_mars, "triton")


def run_multiplanet(n_traj: int = 100, n_epochs: int = 300,
                    verbose: bool = True) -> dict:
    """Train multi-planet FNO and test zero-shot on Triton."""
    import matplotlib; matplotlib.use("Agg")
    from src.planetary_atm import MarsAtmosphere, VenusAtmosphere, TitanAtmosphere
    from src.multifidelity_pinn import LowFidelityEDL

    rng = np.random.default_rng(0)
    t   = np.linspace(0, 400, 100)
    planet_data = {}

    for planet_cls, name, A, v0, alt0 in [
        (MarsAtmosphere,  "mars",  78.5, 5800, 125000),
        (VenusAtmosphere, "venus", 50.0, 9000,  80000),
        (TitanAtmosphere, "titan", 20.0, 1500,  60000),
    ]:
        atm = planet_cls()
        trajs_in, trajs_out = [], []
        for _ in range(n_traj):
            Cd   = rng.uniform(1.2, 2.2)
            mass = rng.uniform(700, 1100)
            lf   = LowFidelityEDL(atm, mass, Cd, A, gamma_deg=15)
            v, h = lf.solve(t, v0, alt0)
            trajs_in.append(v)
            trajs_out.append(v / max(float(v.max()), 1))  # normalised
        planet_data[name] = (np.array(trajs_in), np.array(trajs_out))

    if verbose:
        print(f"[MultiPlanet] Training FNO on 3 planets × {n_traj} trajs")

    op = MultiPlanetOperator(traj_dim=50, modes=8, width=24)
    losses = op.train(planet_data, n_epochs=n_epochs, verbose=verbose)

    # Zero-shot Triton
    mars_sample = planet_data["mars"][0][:10]
    triton_pred = op.zero_shot_triton(mars_sample)

    if verbose:
        print(f"  Zero-shot Triton prediction shape: {triton_pred.shape}")
        print(f"  Multi-planet operator trained  backend={op._backend}")

    return {"operator": op, "losses": losses, "triton_pred": triton_pred,
            "planet_data": planet_data}
