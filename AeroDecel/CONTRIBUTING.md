# Contributing to AeroDecel

Thank you for your interest in contributing to AeroDecel! This guide explains the
project's modular architecture and how to extend it.

---

## 🏗️ Architecture Overview

AeroDecel is organised into **tiers**, each self-contained:

```
AeroDecel/
├── src/              ← All physics, ML, and systems modules
├── tests/            ← Pytest suite
├── notebooks/        ← Jupyter tutorials (Colab-ready)
├── data/             ← Aero databases, validation data
├── main.py           ← CLI entry point (all flags)
├── app.py            ← Dash dashboard
└── api.py            ← FastAPI REST API
```

Every module in `src/` follows the same pattern:

```python
# src/my_module.py

def run(..., verbose=True) -> dict:
    """Entry point — called from main.py and tests."""
    ...
    result = do_physics(...)
    plot_results(result)
    return result

def plot_results(result, save_path="outputs/my_module.png"):
    """Self-contained visualisation."""
    ...

if __name__ == "__main__":
    run()
```

---

## 🪐 Adding a New Planet

1. **Create a subclass** of `PlanetaryAtmosphere` in `src/planetary_atm.py`:

```python
class EuropaAtmosphere(PlanetaryAtmosphere):
    def __init__(self):
        super().__init__(
            name="Europa",
            radius_m=1_560_800.0,
            mass_kg=4.7998e22,
            gravity_ms2=1.314,
            surface_pressure_pa=0.1,        # ~1 μPa (tenuous)
            gas_constant=260.0,              # O₂ dominated
            composition={"O2": 0.95, "H2": 0.05},
        )
        self._layers = [
            (0, 120_000, 1e-11, 102.0, 0.0),
        ]

    def density(self, altitude_m: float) -> float:
        h = max(0.0, altitude_m)
        alt0, H, rho0, T0, lr = self._layers[0]
        return float(max(rho0 * np.exp(-(h - alt0) / H), 1e-20))

    def temperature(self, altitude_m: float) -> float:
        return 102.0  # isothermal
```

2. **Register it** at the bottom of `planetary_atm.py`:

```python
register_planet("europa", EuropaAtmosphere)
```

3. **Add a test** in `tests/test_v6.py`:

```python
def test_europa_atmosphere():
    from src.planetary_atm import get_planet_atmosphere
    atm = get_planet_atmosphere("europa")
    assert atm.gravity_ms2 > 0
    assert atm.density(0) > 0
```

4. **Update the CLI** — add `"europa"` to the `--planet` choices in `main.py`.

---

## 🛡️ Adding a New TPS Material

1. **Add to `MATERIALS` dict** in `src/thermal_model.py`:

```python
"cork": TPSMaterial(
    name="Cork P50",
    density_kgm3=500.0,
    conductivity_WmK=0.052,
    specific_heat_JkgK=1600.0,
    max_temperature_K=570.0,
    ablation_rate_factor=0.3,
),
```

2. **Add a test**:

```python
def test_cork_tps():
    from src.thermal_model import ThermalProtectionSystem
    tps = ThermalProtectionSystem("cork", 0.02)
    assert tps.mat.max_temperature_K == 570.0
```

---

## 🤖 Adding a New ML Module

Follow these steps:

1. **Create `src/my_ml_module.py`** following the standard pattern:
   - Must work with **numpy/scipy fallback** when PyTorch is missing.
   - Use `try: import torch; _TORCH = True except: _TORCH = False` at the top.
   - Provide a `run(...)` entry point and a `plot_*(...)` function.

2. **Add CLI flag** in `main.py: build_parser()`:
   ```python
   p.add_argument("--my-ml", action="store_true", help="My new ML module")
   ```

3. **Add handler** in `main.py: run_pipeline()`:
   ```python
   if args.my_ml:
       _sec("My ML Module"); t0 = time.time()
       from src.my_ml_module import run as my_run
       result = my_run(verbose=args.verbose)
       print(f"  Result: ... ⏱ {_el(t0)}")
   ```

4. **Add test** in `tests/test_v6.py`.

5. **Update README.md** tier table.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_v6.py -v

# Run with coverage
pytest tests/test_v6.py -v --cov=src --cov-report=term-missing

# Smoke test (generates all outputs)
python main.py --masterpiece
```

All tests must pass before submitting a PR. The CI pipeline runs automatically
on push.

---

## 📋 Code Style

- **Python 3.10+** with type hints
- **Docstrings**: NumPy/SciPy style
- **Line length**: 100 chars max (soft limit)
- **Linting**: `ruff check src/ --ignore E501,F401`
- **Dark-theme plots**: use the project's colour palette:
  ```python
  BG = "#080c14"   # figure background
  AX = "#0d1526"   # axes background
  C1 = "#00d4ff"   # cyan
  C2 = "#ff6b35"   # orange
  C3 = "#a8ff3e"   # green
  C4 = "#ffd700"   # gold
  CR = "#ff4560"   # red (danger/failure)
  ```

---

## 🔄 Pull Request Checklist

- [ ] Module has a `run()` entry point
- [ ] Plot function uses project dark theme
- [ ] CLI flag added to `main.py`
- [ ] Test added to `tests/test_v6.py`
- [ ] README updated with new feature
- [ ] `pip install -r requirements-core.txt` still works (no new hard dependencies)
- [ ] All existing tests pass

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the
same terms as the rest of the project (MIT License).

## 💬 Questions?

Open an issue on GitHub or start a discussion. We're happy to help!
