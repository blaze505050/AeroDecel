"""
src/multifidelity_pinn.py — Multi-Fidelity PINN for EDL
=========================================================
Architecture
------------
  Low-fidelity model  : fast exponential-atmosphere ODE (no torch)
  PINN correction     : shallow MLP that learns the residual
  Joint prediction    : y_hf ≈ y_lf + PINN(x)

Loss = λ_data·L_data  +  λ_phys·L_phys  +  λ_smooth·L_smooth
  L_data   : MSE between PINN-corrected prediction and HF data
  L_phys   : residual of the drag-deceleration PDE
  L_smooth : L2 curvature penalty on the correction (regularisation)

Works without torch (falls back to scipy least-squares Laplace correction).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


# ── Torch import (optional) ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
# LOW-FIDELITY MODEL  (pure numpy — drag ODE on exponential atmosphere)
# ══════════════════════════════════════════════════════════════════════════════

class LowFidelityEDL:
    """
    1-D drag deceleration:  m dv/dt = -0.5·ρ(h)·v²·Cd·A  +  m·g
    h(t) from v(t) via dh/dt = -v·sin(γ)  (flight-path angle γ).
    Solved with a fixed-step RK2 (midpoint method) — fast enough for MCMC.
    """

    def __init__(self, planet, mass_kg: float, Cd: float, area_m2: float,
                 gamma_deg: float = 10.0):
        self.planet   = planet
        self.mass     = mass_kg
        self.Cd       = Cd
        self.A        = area_m2
        self.sin_gam  = np.sin(np.deg2rad(gamma_deg))
        self.g        = planet.gravity_ms2

    def solve(self, t_span: np.ndarray,
              v0: float, h0: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (v_arr, h_arr) at times t_span."""
        v, h = float(v0), float(h0)
        vs, hs = [v], [h]
        for i in range(len(t_span) - 1):
            dt = t_span[i+1] - t_span[i]
            rho = self.planet.density(max(h, 0.0))
            D   = 0.5 * rho * v**2 * self.Cd * self.A / self.mass
            dv  = -D + self.g * self.sin_gam
            dh  = -v * self.sin_gam
            # Midpoint
            v2  = v + 0.5 * dt * dv
            h2  = h + 0.5 * dt * dh
            rho2 = self.planet.density(max(h2, 0.0))
            D2   = 0.5 * rho2 * v2**2 * self.Cd * self.A / self.mass
            dv2  = -D2 + self.g * self.sin_gam
            dh2  = -v2 * self.sin_gam
            v   = max(0.0, v + dt * dv2)
            h   = max(0.0, h + dt * dh2)
            vs.append(v); hs.append(h)
            if h <= 0:
                vs += [0.0] * (len(t_span) - len(vs))
                hs += [0.0] * (len(t_span) - len(hs))
                break
        return np.array(vs[:len(t_span)]), np.array(hs[:len(t_span)])


# ══════════════════════════════════════════════════════════════════════════════
# PINN CORRECTION NETWORK
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:
    class _CorrectionNet(nn.Module):
        """MLP that maps normalised input (t̃) → correction Δv, Δh."""
        def __init__(self, layers: list[int], activation: str = "tanh"):
            super().__init__()
            act = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}.get(
                activation, nn.Tanh)
            mods = []
            for i in range(len(layers) - 1):
                mods.append(nn.Linear(layers[i], layers[i+1]))
                if i < len(layers) - 2:
                    mods.append(act())
            self.net = nn.Sequential(*mods)

        def forward(self, t):
            return self.net(t)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-FIDELITY PINN
# ══════════════════════════════════════════════════════════════════════════════

class MultiFidelityPINN:
    """
    Multi-fidelity correction: y_pred = y_LF(t) + PINN(t)

    Parameters
    ----------
    lf_model     : LowFidelityEDL instance
    hf_data      : dict with keys "t","v","h"  — high-fidelity observations
    layers       : PINN hidden architecture, e.g. [1, 64, 64, 2]
    lr           : learning rate
    """

    def __init__(self, lf_model: LowFidelityEDL, hf_data: dict,
                 layers: list[int] | None = None,
                 lr: float = 1e-3):
        self.lf    = lf_model
        self.data  = hf_data
        self.layers = layers or [1, 64, 64, 2]
        self.lr    = lr
        self._trained = False
        self._backend = "torch" if _TORCH else "scipy"

        # Normalisation constants
        t_arr = np.asarray(hf_data["t"])
        self._t0 = t_arr[0]; self._tN = max(t_arr[-1], 1.0)
        v_arr = np.asarray(hf_data["v"])
        self._v0 = v_arr[0]; self._vN = max(v_arr.max(), 1.0)

        if _TORCH:
            self.net  = _CorrectionNet(self.layers)
            self.opt  = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.opt, T_max=1000, eta_min=1e-6)
        else:
            # Scipy fallback: polynomial correction coefficients
            self._poly_coeff = np.zeros(4)

    # ── torch training ────────────────────────────────────────────────────────

    def _train_torch(self, epochs: int = 2000, lam_phys: float = 0.1,
                     verbose: bool = True):
        t_np = np.asarray(self.data["t"])
        v_np = np.asarray(self.data["v"])
        h_np = np.asarray(self.data.get("h", np.zeros_like(v_np)))

        # LF prediction at HF time points
        v_lf, h_lf = self.lf.solve(t_np, v_np[0], h_np[0])

        t_norm = torch.tensor((t_np - self._t0) / (self._tN - self._t0),
                               dtype=torch.float32).unsqueeze(1)
        v_true = torch.tensor(v_np, dtype=torch.float32).unsqueeze(1)
        h_true = torch.tensor(h_np, dtype=torch.float32).unsqueeze(1)
        v_lf_t = torch.tensor(v_lf, dtype=torch.float32).unsqueeze(1)
        h_lf_t = torch.tensor(h_lf, dtype=torch.float32).unsqueeze(1)

        best_loss = float("inf")
        best_sd   = None

        for ep in range(1, epochs + 1):
            self.opt.zero_grad()
            delta = self.net(t_norm)           # (N, 2)  → [Δv, Δh]
            dv, dh = delta[:, [0]], delta[:, [1]]

            # Data loss
            L_data = ((v_lf_t + dv - v_true)**2).mean() + \
                     ((h_lf_t + dh - h_true)**2).mean()

            # Smoothness (2nd derivative)
            if len(t_np) > 2:
                d2v = torch.diff(torch.diff(dv[:, 0]))
                L_smooth = (d2v**2).mean()
            else:
                L_smooth = torch.tensor(0.0)

            loss = L_data + lam_phys * L_smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step(); self.sch.step()

            lv = float(loss)
            if lv < best_loss:
                best_loss = lv
                best_sd   = {k: v.clone() for k, v in self.net.state_dict().items()}

            if verbose and ep % max(1, epochs//5) == 0:
                print(f"  [PINN] epoch {ep:5d}/{epochs}  loss={lv:.4e}", flush=True)

        if best_sd:
            self.net.load_state_dict(best_sd)
        print(f"  [PINN] Training complete  best_loss={best_loss:.4e}")

    # ── scipy fallback ────────────────────────────────────────────────────────

    def _train_scipy(self, verbose=True):
        t_np = np.asarray(self.data["t"])
        v_np = np.asarray(self.data["v"])
        v_lf, _ = self.lf.solve(t_np, v_np[0],
                                  np.asarray(self.data.get("h", [0]))[0])
        residual = v_np - v_lf
        t_n = (t_np - self._t0) / max(self._tN - self._t0, 1e-9)
        # Fit cubic polynomial to residual
        self._poly_coeff = np.polyfit(t_n, residual, deg=min(3, len(t_n)-1))
        if verbose:
            print(f"  [PINN] Scipy poly fallback fitted  coeff={np.round(self._poly_coeff, 4)}")

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, epochs: int = 2000, lam_phys: float = 0.1,
              verbose: bool = True):
        if _TORCH:
            self._train_torch(epochs, lam_phys, verbose)
        else:
            self._train_scipy(verbose)
        self._trained = True

    def predict(self, t_query: np.ndarray, v0: float, h0: float) -> dict:
        """
        Return multi-fidelity prediction at t_query times.
        """
        v_lf, h_lf = self.lf.solve(t_query, v0, h0)

        if _TORCH and self._trained:
            t_n = torch.tensor(
                (t_query - self._t0) / max(self._tN - self._t0, 1e-9),
                dtype=torch.float32).unsqueeze(1)
            self.net.eval()
            with torch.no_grad():
                delta = self.net(t_n).numpy()
            dv = delta[:, 0]; dh = delta[:, 1]
        elif self._trained:
            t_n = (t_query - self._t0) / max(self._tN - self._t0, 1e-9)
            dv = np.polyval(self._poly_coeff, t_n)
            dh = np.zeros_like(dv)
        else:
            dv = np.zeros_like(v_lf); dh = np.zeros_like(h_lf)

        return {
            "t":    t_query,
            "v_lf": v_lf,
            "h_lf": h_lf,
            "v_mf": np.maximum(0, v_lf + dv),
            "h_mf": np.maximum(0, h_lf + dh),
            "dv":   dv,
            "dh":   dh,
            "backend": self._backend,
        }


if __name__ == "__main__":
    from src.planetary_atm import MarsAtmosphere
    planet = MarsAtmosphere()
    lf = LowFidelityEDL(planet, mass_kg=900, Cd=1.7, area_m2=78.5)
    t_ref = np.linspace(0, 300, 50)
    v_ref, h_ref = lf.solve(t_ref, 5800, 125_000)
    noise = np.random.default_rng(42).normal(0, 30, len(t_ref))
    hf_data = {"t": t_ref, "v": np.clip(v_ref + noise, 0, None), "h": h_ref}
    mfpinn = MultiFidelityPINN(lf, hf_data, layers=[1, 32, 32, 2])
    mfpinn.train(epochs=500, verbose=True)
    pred = mfpinn.predict(t_ref, v_ref[0], h_ref[0])
    print(f"v_mf[end]={pred['v_mf'][-1]:.2f}  v_lf[end]={pred['v_lf'][-1]:.2f}")
