"""
turbulence.py — MIL-SPEC Dryden Atmospheric Turbulence Model
=============================================================
Generates coloured-noise wind gusts per MIL-HDBK-1797B / MIL-F-8785C.
Turbulence intensity σ and scale length L scale with altitude.

Turbulence power spectral densities (Dryden):
  Φ_w(ω) = σ_w² · (2L_w/πV) · [1 + (L_w·ω/V)²]^(-1)   (vertical)
  Φ_u(ω) = σ_u² · (L_u/πV) · [1]^(-1)                   (longitudinal)

Implemented as state-space shaping filters driven by white noise.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.signal import lfilter

@dataclass
class TurbulenceIntensity:
    """MIL-HDBK-1797B Table V turbulence intensity vs altitude."""
    level: str = "light"   # light | moderate | severe

    def sigma_w(self, altitude_m: float) -> float:
        """Vertical turbulence intensity σ_w [m/s]."""
        h = max(1.0, altitude_m)
        base = {"light": 1.5, "moderate": 4.0, "severe": 8.0}[self.level]
        # Decrease with altitude (boundary layer effect)
        factor = np.exp(-h / 600.0) * 0.6 + 0.4
        return base * factor

    def sigma_u(self, altitude_m: float) -> float:
        """Horizontal turbulence intensity σ_u [m/s]."""
        return self.sigma_w(altitude_m) * 1.3

    def L_w(self, altitude_m: float) -> float:
        """Vertical scale length [m]."""
        h = max(1.0, altitude_m)
        return min(h / 2.0, 533.0)   # MIL-HDBK-1797B §2.5.1

    def L_u(self, altitude_m: float) -> float:
        """Horizontal scale length [m]."""
        return 1750.0   # constant above boundary layer

class DrydenTurbulence:
    """
    Generates time-series turbulence velocities (u, v, w) [m/s].
    State-space implementation — no scipy.signal needed at runtime.

    Usage:
        turb = DrydenTurbulence(airspeed=20.0, dt=0.05)
        u, v, w = turb.sample(altitude_m=500.0)
    """
    def __init__(self, airspeed: float = 20.0, dt: float = 0.05,
                 level: str = "light", seed: int = 42):
        self.V    = max(airspeed, 1.0)
        self.dt   = dt
        self.inten = TurbulenceIntensity(level)
        self.rng  = np.random.default_rng(seed)
        # State vectors for shaping filters
        self._xu = self._xv = self._xw1 = self._xw2 = 0.0

    def sample(self, altitude_m: float) -> tuple[float, float, float]:
        """Return instantaneous (u, v, w) turbulence gust [m/s]."""
        h   = max(1.0, altitude_m)
        V   = self.V
        dt  = self.dt
        sig_u = self.inten.sigma_u(h)
        sig_w = self.inten.sigma_w(h)
        Lu    = self.inten.L_u(h)
        Lw    = self.inten.L_w(h)

        # Longitudinal (u) — first-order Dryden shaping filter
        a_u   = V / Lu
        b_u   = sig_u * np.sqrt(2 * V / (np.pi * Lu))
        wn_u  = self.rng.normal(0, 1) * np.sqrt(dt)
        self._xu = (1 - a_u*dt) * self._xu + b_u * wn_u
        u = self._xu

        # Lateral (v) — same as u (isotropic horizontal)
        self._xv = (1 - a_u*dt) * self._xv + b_u * self.rng.normal(0,1)*np.sqrt(dt)
        v = self._xv

        # Vertical (w) — second-order Dryden filter
        a_w   = V / Lw
        b_w1  = sig_w * np.sqrt(3 * V / (np.pi * Lw))
        b_w2  = sig_w * np.sqrt(V / (np.pi * Lw))
        wn_w  = self.rng.normal(0, 1) * np.sqrt(dt)
        xw1_n = (1 - a_w*dt) * self._xw1 + wn_w
        xw2_n = (1 - a_w*dt) * self._xw2 + a_w*dt * xw1_n
        w = b_w1 * xw1_n + b_w2 * xw2_n
        self._xw1, self._xw2 = xw1_n, xw2_n

        return float(np.clip(u, -3*sig_u, 3*sig_u)), \
               float(np.clip(v, -3*sig_u, 3*sig_u)), \
               float(np.clip(w, -3*sig_w, 3*sig_w))

    def batch(self, altitudes: np.ndarray) -> np.ndarray:
        """Generate (N,3) array of turbulence vectors for an altitude sequence."""
        out = np.zeros((len(altitudes), 3))
        for i, h in enumerate(altitudes):
            out[i] = self.sample(h)
        return out
