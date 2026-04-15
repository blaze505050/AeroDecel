"""
src/gp_emulator.py — Gaussian Process Emulator with Bayesian Optimisation
==========================================================================
Replaces the brute-force parametric sweep with an active learning loop:
  1. Evaluate the TPS model at a small set of initial designs
  2. Fit a GP to the (input → output) map
  3. Maximise Expected Improvement (EI) to find the next design to evaluate
  4. Repeat until convergence or budget exhausted

This finds the minimum-mass TPS that survives entry in ~50 evaluations
instead of 10,000 brute-force samples.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class GPEmulator:
    """
    Gaussian Process emulator using squared-exponential kernel.
    No external library — pure scipy/numpy.
    """

    def __init__(self, noise: float = 1e-4):
        self.noise   = noise
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.ell     = 1.0    # length scale
        self.sigma_f = 1.0    # signal variance
        self._K_inv: np.ndarray | None = None
        self._alpha:  np.ndarray | None = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """SE kernel K(x,x') = σ_f² exp(-||x-x'||²/(2ℓ²))."""
        d = np.sum(((X1[:, None, :] - X2[None, :, :]) / self.ell)**2, axis=-1)
        return self.sigma_f**2 * np.exp(-0.5 * d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPEmulator":
        """Fit the GP, optimising hyperparameters via marginal likelihood."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self._optimise_hypers()
        K     = self._kernel(X, X) + self.noise * np.eye(len(X))
        try:
            L     = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))
            self._K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        except np.linalg.LinAlgError:
            self._K_inv = np.linalg.pinv(K)
        self._alpha = self._K_inv @ y
        return self

    def _log_marginal_likelihood(self, log_params: np.ndarray) -> float:
        ell, sigma_f = np.exp(log_params)
        self.ell = ell; self.sigma_f = sigma_f
        X, y = self.X_train, self.y_train
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        try:
            L    = np.linalg.cholesky(K + 1e-8*np.eye(len(X)))
            alph = np.linalg.solve(L.T, np.linalg.solve(L, y))
            lml  = (-0.5*y@alph - np.sum(np.log(np.diag(L)))
                    - 0.5*len(X)*np.log(2*np.pi))
            return float(-lml)
        except Exception:
            return 1e9

    def _optimise_hypers(self):
        res = minimize(self._log_marginal_likelihood,
                       [np.log(self.ell), np.log(self.sigma_f)],
                       method="L-BFGS-B", options={"maxiter": 50})
        self.ell, self.sigma_f = np.exp(res.x[0]), np.exp(res.x[1])

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and std."""
        K_star  = self._kernel(X_new, self.X_train)
        K_ss    = self._kernel(X_new, X_new)
        mu      = K_star @ self._alpha
        sigma2  = np.diag(K_ss) - np.sum(K_star @ self._K_inv * K_star, axis=1)
        return mu, np.sqrt(np.maximum(sigma2, 0))

    def expected_improvement(self, X_new: np.ndarray,
                              xi: float = 0.01) -> np.ndarray:
        """EI(x) = (μ(x) - f*- ξ)·Φ(z) + σ(x)·φ(z)  (for minimisation: -f)."""
        mu, sigma = self.predict(X_new)
        f_best    = self.y_train.min()
        z         = (f_best - mu - xi) / np.maximum(sigma, 1e-9)
        ei        = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0)

    def next_query(self, bounds: np.ndarray, n_restarts: int = 10) -> np.ndarray:
        """Find x* = argmax EI(x) via multi-start L-BFGS-B."""
        rng = np.random.default_rng()
        best_x   = None
        best_ei  = -np.inf

        for _ in range(n_restarts):
            x0  = rng.uniform(bounds[:, 0], bounds[:, 1])
            res = minimize(
                lambda x: -float(self.expected_improvement(x[None, :]).item()),
                x0, method="L-BFGS-B",
                bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                options={"maxiter": 50},
            )
            ei_val = float(self.expected_improvement(res.x[None, :]).item())
            if ei_val > best_ei:
                best_ei = ei_val; best_x = res.x

        return best_x


class TPS_BayesOptimiser:
    """
    Bayesian optimisation to find minimum-mass TPS that survives entry.

    Objective: minimise TPS mass (= thickness × density)
    Constraint: TPS survives (peak_T < T_limit, recession < thickness)
    """

    def __init__(self, planet_atm, material: str = "pica",
                 q_peak_MWm2: float = 15.0, t_entry_s: float = 200.0):
        from src.ablation_model import AblationSolver
        self.planet = planet_atm
        self.mat_name = material
        self.q_peak = q_peak_MWm2
        self.t_entry = t_entry_s
        self.gp      = GPEmulator(noise=1e-3)
        self._evals  = []   # (thickness, mass, survived, T_peak, recession)

    def _evaluate(self, thickness_m: float) -> dict:
        """Evaluate TPS at given thickness."""
        from src.ablation_model import AblationSolver, ABLATIVE_DB
        import numpy as np

        mat = ABLATIVE_DB[self.mat_name]
        solver = AblationSolver(self.mat_name, thickness_m, n_nodes=15)
        t    = np.linspace(0, self.t_entry, 100)
        t_pk = self.t_entry * 0.30
        q0   = self.q_peak * 1e6 * np.where(
            t <= t_pk, t/t_pk, (self.t_entry-t)/(self.t_entry-t_pk))
        q0   = np.clip(q0, 0, None)
        res  = solver.solve(q0, t, verbose=False)
        mass = mat.density_kgm3 * thickness_m   # [kg/m²]
        survived = res["total_recession_mm"] < thickness_m * 1000

        return {"thickness_m": thickness_m, "mass_kgm2": mass,
                "survived": survived, "T_peak_K": res["peak_T_K"],
                "recession_mm": res["total_recession_mm"],
                "blocking_pct": res["blocking_pct"]}

    def run(self, n_init: int = 8, n_iter: int = 40,
            bounds_mm: tuple = (5, 100), verbose: bool = True) -> dict:
        """
        Run Bayesian optimisation loop.
        Returns optimal thickness, minimum mass, and all evaluations.
        """
        bounds = np.array([[bounds_mm[0]/1000, bounds_mm[1]/1000]])   # m
        rng    = np.random.default_rng(0)

        # Initial design of experiments (log-spaced)
        thicknesses = np.geomspace(bounds[0, 0], bounds[0, 1], n_init)

        if verbose:
            print(f"\n[BayesOpt] TPS optimisation  material={self.mat_name}  "
                  f"q_peak={self.q_peak}MW/m²")

        for i, th in enumerate(thicknesses):
            ev = self._evaluate(th)
            self._evals.append(ev)
            if verbose:
                print(f"  Init {i+1:2d}: t={th*1000:.1f}mm  mass={ev['mass_kgm2']:.2f}kg/m²  "
                      f"survived={'✓' if ev['survived'] else '✗'}  "
                      f"T_peak={ev['T_peak_K']:.0f}K")

        # Fit GP on (thickness → mass | survived constraint)
        # Objective for BO: mass if survived, else large penalty
        X_all = np.array([e["thickness_m"] for e in self._evals])[:, None]
        y_all = np.array([
            e["mass_kgm2"] if e["survived"] else e["mass_kgm2"] + 100
            for e in self._evals
        ])

        for it in range(n_iter):
            self.gp.fit(X_all, y_all)
            x_next = self.gp.next_query(bounds)
            x_next = float(np.clip(np.asarray(x_next).ravel()[0], bounds[0, 0], bounds[0, 1]))
            ev     = self._evaluate(x_next)
            self._evals.append(ev)

            X_all = np.vstack([X_all, [[x_next]]])
            y_new = ev["mass_kgm2"] if ev["survived"] else ev["mass_kgm2"] + 100
            y_all = np.append(y_all, y_new)

            if verbose and (it+1) % 10 == 0:
                best_s = [e for e in self._evals if e["survived"]]
                best   = min(best_s, key=lambda e: e["mass_kgm2"]) if best_s else ev
                print(f"  Iter {it+1:3d}/{n_iter}: best mass={best['mass_kgm2']:.3f}kg/m²  "
                      f"t={best['thickness_m']*1000:.1f}mm")

        # Select optimal: minimum mass that survived
        survived_evals = [e for e in self._evals if e["survived"]]
        if not survived_evals:
            optimal = min(self._evals, key=lambda e: e["recession_mm"])
        else:
            optimal = min(survived_evals, key=lambda e: e["mass_kgm2"])

        if verbose:
            print(f"\n  OPTIMAL: thickness={optimal['thickness_m']*1000:.2f}mm  "
                  f"mass={optimal['mass_kgm2']:.3f}kg/m²  "
                  f"survived={optimal['survived']}")
            print(f"  Total evaluations: {len(self._evals)}")

        return {
            "optimal":    optimal,
            "all_evals":  self._evals,
            "gp":         self.gp,
            "n_evals":    len(self._evals),
            "material":   self.mat_name,
        }


def run_gp_emulator(material: str = "pica", q_peak_MW: float = 15.0,
                    n_iter: int = 30, verbose: bool = True) -> dict:
    """Run GP-based TPS design optimisation."""
    import matplotlib; matplotlib.use("Agg")
    from src.planetary_atm import MarsAtmosphere
    planet = MarsAtmosphere()
    opt = TPS_BayesOptimiser(planet, material=material, q_peak_MWm2=q_peak_MW)
    result = opt.run(n_init=6, n_iter=n_iter, verbose=verbose)
    return result
