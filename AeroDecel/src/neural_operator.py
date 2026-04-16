"""
src/neural_operator.py — FNO + DeepONet Neural Operators
=========================================================
Implements:
  • SpectralConv1d  — 1-D spectral convolution layer
  • FNO1d           — 1-D Fourier Neural Operator (Lift → Fourier → Project)
  • BranchNet       — DeepONet branch (encodes input function)
  • TrunkNet        — DeepONet trunk (encodes query location)
  • DeepONet        — operator learning via branch·trunk inner product
  • NeuralOperator  — unified API for both (falls back to numpy if no torch)

All classes degrade gracefully when torch is unavailable:
  FNO    → linear interpolation
  DeepONet → RBF interpolation
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
# TORCH-DEPENDENT CLASSES
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    class SpectralConv1d(nn.Module):
        """
        1-D spectral convolution: multiply in Fourier space then iFFT back.
        """
        def __init__(self, in_ch: int, out_ch: int, modes: int):
            super().__init__()
            self.in_ch  = in_ch
            self.out_ch = out_ch
            self.modes  = modes
            scale = 1 / (in_ch * out_ch)
            self.W = nn.Parameter(
                scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x : (batch, in_ch, n_pts)"""
            B, C, N = x.shape
            x_ft = torch.fft.rfft(x, norm="ortho")
            out_ft = torch.zeros(B, self.out_ch, N//2+1, dtype=torch.cfloat, device=x.device)
            modes = min(self.modes, N//2+1)
            out_ft[:, :, :modes] = torch.einsum(
                "bci,coi->boi", x_ft[:, :, :modes], self.W[:, :, :modes])
            return torch.fft.irfft(out_ft, n=N, norm="ortho")

    class FNO1d(nn.Module):
        """
        1-D Fourier Neural Operator.
        Input  : (batch, n_in,  N)  — N spatial/time points
        Output : (batch, n_out, N)
        """
        def __init__(self, n_in: int, n_out: int,
                     modes: int = 16, width: int = 64, n_layers: int = 4):
            super().__init__()
            self.lift    = nn.Linear(n_in, width)
            self.sconvs  = nn.ModuleList(
                [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
            self.wconvs  = nn.ModuleList(
                [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
            self.project1 = nn.Linear(width, 128)
            self.project2 = nn.Linear(128, n_out)
            self.act      = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x : (batch, N, n_in)"""
            x = self.lift(x)                           # (B, N, width)
            x = x.permute(0, 2, 1)                     # (B, width, N)
            for sc, wc in zip(self.sconvs, self.wconvs):
                x = self.act(sc(x) + wc(x))
            x = x.permute(0, 2, 1)                     # (B, N, width)
            x = self.act(self.project1(x))
            x = self.project2(x)
            return x

    class BranchNet(nn.Module):
        """Encodes the input function u(x) sampled at m sensor locations."""
        def __init__(self, m_sensors: int, hidden: int = 64, out_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(m_sensors, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),    nn.Tanh(),
                nn.Linear(hidden, out_dim),
            )
        def forward(self, u): return self.net(u)

    class TrunkNet(nn.Module):
        """Encodes query location y."""
        def __init__(self, in_dim: int = 1, hidden: int = 64, out_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, out_dim),
            )
        def forward(self, y): return self.net(y)

    class _DeepONetTorch(nn.Module):
        """G(u)(y) = Σ_k branch_k(u) · trunk_k(y)  +  bias."""
        def __init__(self, branch: BranchNet, trunk: TrunkNet, n_out: int = 1):
            super().__init__()
            self.branch = branch
            self.trunk  = trunk
            self.bias   = nn.Parameter(torch.zeros(n_out))

        def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b = self.branch(u)   # (B, p)
            t = self.trunk(y)    # (Q, p)
            return torch.einsum("bp,qp->bq", b, t) + self.bias


# ══════════════════════════════════════════════════════════════════════════════
# NUMPY FALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class _FNOFallback:
    """Linear interpolation fallback when torch is unavailable."""
    def __init__(self, *a, **kw): self._fitted = False
    def fit(self, x_train, y_train): self._x=x_train; self._y=y_train; self._fitted=True
    def predict(self, x_query):
        if not self._fitted: raise RuntimeError("Call fit() first")
        from scipy.interpolate import interp1d
        fn = interp1d(self._x.ravel(), self._y.ravel(),
                      bounds_error=False, fill_value="extrapolate")
        return fn(x_query.ravel()).reshape(-1, 1)


class _DeepONetFallback:
    """RBF-kernel operator approximation (numpy-only)."""
    def __init__(self, *a, **kw): self._alpha = None
    def fit(self, u_train, y_train, g_train=None, rbf_gamma=1.0):
        from scipy.spatial.distance import cdist
        if g_train is None:
            g_train = y_train
        self._u  = u_train; self._y = y_train; self._g = rbf_gamma
        K = np.exp(-rbf_gamma * cdist(u_train, u_train, metric="sqeuclidean"))
        self._alpha = np.linalg.lstsq(K, g_train, rcond=None)[0]
        self._u_train = u_train
    def predict(self, u_query, y_query=None):
        from scipy.spatial.distance import cdist
        if y_query is None:
            y_query = u_query
        K = np.exp(-self._g * cdist(u_query, self._u_train, metric="sqeuclidean"))
        return K @ self._alpha


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED API
# ══════════════════════════════════════════════════════════════════════════════

class NeuralOperator:
    """
    Unified interface for FNO and DeepONet.

    Usage (torch available)
    -----------------------
        op = NeuralOperator("fno", n_in=1, n_out=1, modes=16, width=32)
        op.train(x_train, y_train, epochs=500)
        y_pred = op.predict(x_test)

    Usage (no torch)
    ----------------
        op = NeuralOperator("fno", n_in=1, n_out=1)   # uses scipy interp
        op.train(x_train, y_train)
        y_pred = op.predict(x_test)
    """

    def __init__(self, operator_type: str, n_in: int = 1, n_out: int = 1,
                 modes: int = 16, width: int = 64, n_layers: int = 4,
                 m_sensors: int = 50, p_dim: int = 64):
        self.type    = operator_type.lower()
        self.n_in    = n_in
        self.n_out   = n_out
        self._backend = "torch" if _TORCH else "numpy"

        if _TORCH:
            if self.type == "fno":
                self.model = FNO1d(n_in, n_out, modes, width, n_layers)
            elif self.type == "deeponet":
                branch = BranchNet(m_sensors, width, p_dim)
                trunk  = TrunkNet(1, width, p_dim)
                self.model = _DeepONetTorch(branch, trunk, n_out)
            else:
                raise ValueError(f"Unknown operator type '{operator_type}'")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                 self.optimizer, patience=100, factor=0.5)
        else:
            if self.type == "fno":
                self.model = _FNOFallback()
            elif self.type == "deeponet":
                self.model = _DeepONetFallback()
            else:
                raise ValueError(f"Unknown operator type '{operator_type}'")

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 1000, verbose: bool = True) -> list[float]:
        """
        Train the neural operator.
        x_train : (N, n_in) or (N, L, n_in) for sequence inputs
        y_train : (N, n_out) or (N, L, n_out)
        """
        losses = []
        if not _TORCH:
            self.model.fit(x_train, y_train)
            return [0.0]

        x_t = torch.tensor(x_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        # Reshape for FNO: (B, L, C)
        if self.type == "fno" and x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)   # (B, 1, n_in) → treat as L=1
            y_t = y_t.unsqueeze(1)

        for ep in range(1, epochs + 1):
            self.optimizer.zero_grad()
            y_pred = self.model(x_t) if self.type == "fno" else self.model(x_t, x_t)
            loss   = ((y_pred - y_t)**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step(loss)
            lv = float(loss)
            losses.append(lv)
            if verbose and ep % max(1, epochs // 5) == 0:
                print(f"  [NeuralOp/{self.type}] ep {ep:5d}/{epochs}  loss={lv:.4e}")

        return losses

    def predict(self, x_query: np.ndarray) -> np.ndarray:
        if not _TORCH:
            return self.model.predict(x_query)

        x_t = torch.tensor(x_query, dtype=torch.float32)
        if self.type == "fno" and x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x_t) if self.type == "fno" else self.model(x_t, x_t)
        return out.numpy()

    def save(self, path: str):
        if _TORCH:
            torch.save(self.model.state_dict(), path)
            print(f"  ✓ Model saved: {path}")

    def load(self, path: str):
        if _TORCH:
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
            self.model.eval()
            print(f"  ✓ Model loaded: {path}")


if __name__ == "__main__":
    print(f"torch available: {_TORCH}")
    op = NeuralOperator("fno", n_in=1, n_out=1, modes=8, width=16)
    x = np.linspace(0, 1, 100)[:, None].astype(np.float32)
    y = np.sin(2 * np.pi * x)
    op.train(x, y, epochs=200, verbose=True)
    y_pred = op.predict(x)
    rmse = float(np.sqrt(((y_pred - y)**2).mean()))
    print(f"RMSE: {rmse:.4f}")
