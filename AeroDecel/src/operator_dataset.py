"""
src/operator_dataset.py — Parametric Sweep Dataset Generator
=============================================================
Generates training data for the neural operator by sweeping over
EDL parameters and recording (input_function, output_field) pairs.

Each sample = one EDL trajectory parameterised by
  (Cd, mass, A, altitude_0, velocity_0, gamma_deg, planet_variant)

The "input function" for the operator is the altitude-time history h(t)
from the LF model; the "output function" is the drag-deceleration profile.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from src.planetary_atm import get_planet_atmosphere


class OperatorDataset:
    """
    Generates a parametric sweep dataset for operator learning.

    parameter_ranges : dict  {param_name: (min, max)}
    output_resolution: (n_time,)  — number of output time-points
    """

    DEFAULT_RANGES = {
        "Cd":          (0.5,   2.5),
        "mass_kg":     (100,   2000),
        "area_m2":     (10,    150),
        "alt0_km":     (80,    150),
        "vel0_kms":    (3.0,   7.0),
        "gamma_deg":   (5.0,   25.0),
    }

    def __init__(self, parameter_ranges: dict | None = None,
                 output_resolution: int = 100,
                 planet_name: str = "mars"):
        self.ranges   = parameter_ranges or self.DEFAULT_RANGES
        self.n_out    = output_resolution
        self.planet_name = planet_name

    # ── Low-fidelity simulator ────────────────────────────────────────────────

    def _simulate_one(self, params: dict) -> dict | None:
        """Run one LF EDL trajectory. Returns feature/label arrays or None."""
        from src.multifidelity_pinn import LowFidelityEDL
        try:
            planet = get_planet_atmosphere(self.planet_name)
            lf = LowFidelityEDL(
                planet,
                mass_kg   = params["mass_kg"],
                Cd        = params["Cd"],
                area_m2   = params["area_m2"],
                gamma_deg = params["gamma_deg"],
            )
            v0  = params["vel0_kms"] * 1000.0
            h0  = params["alt0_km"]  * 1000.0
            t_end = h0 / (v0 * max(np.sin(np.deg2rad(params["gamma_deg"])), 0.05))
            t_end = float(np.clip(t_end, 10, 1200))
            t_arr = np.linspace(0, t_end, self.n_out)
            v_arr, h_arr = lf.solve(t_arr, v0, h0)

            # Feature: normalised time + density along trajectory
            rho_arr = np.array([planet.density(max(0, h)) for h in h_arr])
            drag_arr = 0.5 * rho_arr * v_arr**2 * params["Cd"] * params["area_m2"]

            return {
                "params":   list(params.values()),
                "t":        t_arr.tolist(),
                "v":        v_arr.tolist(),
                "h":        h_arr.tolist(),
                "drag":     drag_arr.tolist(),
                "rho":      rho_arr.tolist(),
                "t_end":    t_end,
            }
        except Exception as e:
            return None

    # ── Dataset generation ────────────────────────────────────────────────────

    def generate(self, n_samples: int = 1000, seed: int = 42,
                 verbose: bool = True) -> dict:
        """
        Generate n_samples EDL trajectories by Latin-hypercube sampling
        of the parameter space.

        Returns
        -------
        dict with keys: params, inputs, outputs, param_names, meta
        """
        rng = np.random.default_rng(seed)

        # Latin-hypercube sampling
        n_params = len(self.ranges)
        lhs = rng.uniform(size=(n_samples, n_params))
        param_names = list(self.ranges.keys())

        all_params = []
        inputs     = []    # shape (N, n_out)  — velocity profile as operator input
        outputs    = []    # shape (N, n_out)  — drag force profile as target

        valid = 0
        for i, row in enumerate(lhs):
            params = {}
            for j, (pname, (lo, hi)) in enumerate(self.ranges.items()):
                params[pname] = lo + (hi - lo) * row[j]

            result = self._simulate_one(params)
            if result is None:
                continue

            # Normalise to [0, 1]
            v = np.array(result["v"])
            d = np.array(result["drag"])
            if v.max() > 0 and d.max() > 0:
                inputs.append(v / v.max())
                outputs.append(d / d.max())
                all_params.append(result["params"])
                valid += 1

            if verbose and (i + 1) % max(1, n_samples // 10) == 0:
                pct = (i + 1) / n_samples * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"\r  [{bar}] {i+1}/{n_samples} ({valid} valid)", end="", flush=True)

        if verbose:
            print(f"\r  [{'█'*20}] {n_samples}/{n_samples} → {valid} valid samples  ")

        return {
            "param_names": param_names,
            "params":      np.array(all_params),
            "inputs":      np.array(inputs),
            "outputs":     np.array(outputs),
            "n_valid":     valid,
            "n_out":       self.n_out,
            "planet":      self.planet_name,
            "meta": {
                "ranges":     self.ranges,
                "n_samples":  n_samples,
                "seed":       seed,
            },
        }

    # ── I/O ──────────────────────────────────────────────────────────────────

    def save(self, dataset: dict, path: str | Path, format: str = "npz"):
        """Save dataset to .npz (compact) or .json (readable)."""
        path = Path(path)
        if format == "npz":
            np.savez_compressed(
                str(path),
                params=dataset["params"],
                inputs=dataset["inputs"],
                outputs=dataset["outputs"],
            )
            meta_path = path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump({k: v for k, v in dataset.items()
                           if k not in ("params", "inputs", "outputs")}, f, indent=2)
            print(f"  ✓ Saved dataset: {path}  ({dataset['n_valid']} samples)")
        elif format == "json":
            out = {k: (v.tolist() if hasattr(v, "tolist") else v)
                   for k, v in dataset.items()}
            with open(path, "w") as f:
                json.dump(out, f)
            print(f"  ✓ Saved dataset JSON: {path}")
        else:
            raise ValueError(f"Unknown format '{format}'")

    @staticmethod
    def load(path: str | Path) -> dict:
        """Load a previously saved dataset."""
        path = Path(path)
        data = np.load(str(path), allow_pickle=False)
        meta_path = path.with_suffix("").with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        return {"params": data["params"], "inputs": data["inputs"],
                "outputs": data["outputs"], **meta}

    def train_test_split(self, dataset: dict,
                          test_frac: float = 0.2,
                          seed: int = 0) -> tuple[dict, dict]:
        """Split dataset into train / test dictionaries."""
        n = len(dataset["params"])
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_test = int(n * test_frac)
        tr_idx, te_idx = idx[n_test:], idx[:n_test]

        def _sub(d, ix):
            return {k: (v[ix] if isinstance(v, np.ndarray) else v)
                    for k, v in d.items()}
        return _sub(dataset, tr_idx), _sub(dataset, te_idx)


if __name__ == "__main__":
    ds = OperatorDataset(output_resolution=60, planet_name="mars")
    data = ds.generate(n_samples=100, verbose=True)
    print(f"inputs shape:  {data['inputs'].shape}")
    print(f"outputs shape: {data['outputs'].shape}")
    ds.save(data, "outputs/operator_dataset.npz")
