# 🪂 AeroDecel — AI-Driven Aerodynamic Deceleration Analysis Framework

> **Open-source computational aerodynamics that eliminates the need for wind tunnel testing.**

```
 ▄▀█ █▀▀ █▀█ █▀█ █▀▄ █▀▀ █▀▀ █▀▀ █░░
 █▀█ ██▄ █▀▄ █▄█ █▄▀ ██▄ █▄▄ ██▄ █▄▄   v5.0
```

**Core equation:**
```
m_eff · dv/dt = mg − ½ρ(h) · v² · Cd_eff(v,h,t) · A(t) − F_buoy
```

where `Cd_eff` incorporates Reynolds, Mach, and porosity corrections; `m_eff` includes virtual/added mass; and `F_buoy` accounts for Archimedes buoyancy.

---

## Why AeroDecel?

| Current Methods | Cost | Compute | Fidelity |
|---|---|---|---|
| Wind tunnel testing | $1M+ | — | High |
| 3D FSI simulation | $50K+ | 100+ CPU-hours | High |
| **AeroDecel** | **$0** | **<60s on laptop** | **High** |

AeroDecel bridges the gap between hyper-expensive empirical testing and computationally prohibitive 3D fluid-structure interaction simulations. It ingests raw telemetry video, uses AI segmentation to quantify canopy expansion frame-by-frame, feeds empirical area data into a classical ODE physics engine, and deploys a research-grade Physics-Informed Neural Network (PINN) to predict the dynamic drag coefficient during canopy inflation.

---

## Quick Start

```bash
pip install numpy scipy pandas matplotlib scikit-learn opencv-python tqdm Pillow torch

python main.py --synthetic --shock --design --bayes --no-pinn --no-mc   # fast (~60s)
python main.py --synthetic --full                                        # everything
streamlit run app.py                                                     # web dashboard
```

---

## Physics Framework

### Governing Equations

> Eq. 1 — **Momentum (Newton's Second Law with aerodynamic corrections)**
>
> `m_eff · dv/dt = m·g − ½·ρ(h)·v²·Cd_eff·A(t) − F_buoy`

> Eq. 2 — **Effective mass (added/virtual mass for accelerating body in fluid)**
>
> `m_eff = m + C_a · ρ(h) · V_canopy`  where `C_a ≈ 0.5` for hemisphere

> Eq. 3 — **ISA Standard Atmosphere (ICAO 7-layer)**
>
> `ρ(h) = P(h) / (R_air · T(h))`

> Eq. 4 — **Generalized logistic canopy inflation (Richards model)**
>
> `A(t) = A_∞ / [1 + exp(−k(t−t₀))]^(1/n)`

> Eq. 5 — **Physics-Informed Neural Network Cd identification**
>
> `Cd(t) = PINN(t; θ*)` where `θ* = argmin[λ_d·L_data + λ_p·L_physics + λ_s·L_smooth]`

### AeroDecel v5.0 Aerodynamic Corrections

| Correction | Model | Reference |
|---|---|---|
| Mach compressibility | Prandtl-Glauert: `Cd/√(1−M²)` | Prandtl 1933 |
| Reynolds drag | Knacke drag crisis curve | Knacke 1992, Fig 5-21 |
| Fabric porosity | `Cd·(1 − k_p·v)` | Pflanz 1952 |
| Added mass | `m + C_a·ρ·V` | Lamb 1932 |
| Buoyancy | `ρ·g·V_canopy` | Archimedes |

---

## Architecture

```
AeroDecel/
├── app.py                    ← Streamlit web dashboard (7 interactive pages)
├── main.py                   ← Master CLI launcher
├── config.py                 ← All parameters + AeroDecel v5.0 toggles
├── requirements.txt
├── tests/test_all.py         ← Comprehensive test suite
└── src/
    ├── atmosphere.py         ← ICAO ISA 7-layer + geopotential + Mach
    ├── phase1_cv.py          ← AI-enhanced CV (YOLO → SAM → HSV fallback)
    ├── phase2_ode.py         ← RK45 ODE + ISA + Re/Mach/porosity/added-mass
    ├── phase3_pinn.py        ← Research-grade PINN (Fourier + curriculum + adaptive-λ)
    ├── phase4_viz.py         ← 8-panel publication dashboard
    ├── phase5_montecarlo.py  ← Monte Carlo UQ (P5/P50/P95)
    ├── phase6_trajectory.py  ← 3D wind-drift + Open-Meteo + KML
    ├── phase7_multistage.py  ← Drogue→Main state machine + snatch loads
    ├── phase8_pendulum.py    ← 12-state pendulum ODE
    ├── opening_shock.py      ← MIL-HDBK-1791 CLA + structural SF
    ├── bayes_cd.py           ← Bayesian MCMC Cd posterior
    ├── pinn_ensemble.py      ← Heteroscedastic ensemble UQ
    ├── design_calc.py        ← Area solver + HTML datasheet
    ├── ingest_telemetry.py   ← GPX / Pixhawk / CSV / JSON / FIT
    ├── calibrate_cd.py       ← brentq + bootstrap Cd back-solver
    ├── mach_cd.py            ← Mach + Reynolds + porosity corrections
    ├── advanced_physics.py   ← Dryden MIL-SPEC turbulence model
    ├── turbulence.py         ← State-space Dryden shaping filters
    ├── export_animation.py   ← Animated GIF / MP4 export
    └── fetch_wind.py         ← Open-Meteo live wind (free, no key)
```

---

## Commands

```bash
# Web UI
pip install streamlit && streamlit run app.py

# Pipeline
python main.py --synthetic                        # phases 1-8
python main.py --video drop_test.mp4              # real video
python main.py --synthetic --no-pinn --no-mc      # fast run
python main.py --phase 7                          # single phase

# AeroDecel v5.0 — Advanced Physics
python main.py --synthetic --no-advanced           # disable Re/Mach/porosity
python main.py --synthetic --cv-model yolo         # YOLO AI segmentation

# Analysis modules
python main.py --shock                            # MIL-HDBK-1791
python main.py --design --target-v 5.0           # design calculator
python main.py --bayes                            # Bayesian Cd
python main.py --ensemble --n-members 5           # PINN ensemble UQ
python main.py --ingest flight.gpx                # telemetry ingest

# Live wind (Open-Meteo, free, no key)
python main.py --phase 6 --lat 51.5 --lon -0.12

# Calibration
python src/calibrate_cd.py --observed-landing-v 6.2
python src/calibrate_cd.py --observed-landing-v 6.2 --patch-config

# Animation
python src/export_animation.py --fps 20 --format gif
python src/export_animation.py --fps 30 --format mp4   # needs ffmpeg

# Tests
pip install pytest && pytest tests/ -v
```

---

## Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PARACHUTE_MASS` | 80 kg | Payload mass |
| `INITIAL_ALT` | 1000 m | Deployment altitude AGL |
| `INITIAL_VEL` | 25 m/s | Deployment velocity |
| `CANOPY_AREA_M2` | 50 m² | Reference canopy area |
| `CD_INITIAL` | 1.5 | Drag coefficient |
| `ADDED_MASS_COEFF` | 0.5 | C_a for virtual mass |
| `CANOPY_DIAMETER_M` | 8.0 m | For Reynolds number |
| `POROSITY_COEFF` | 0.012 | Fabric porosity k_p |
| `PINN_EPOCHS` | 8000 | PINN training epochs |
| `PINN_FOURIER_FEATURES` | True | Fourier feature embeddings |
| `PINN_CURRICULUM` | True | Curriculum training schedule |
| `CV_MODEL` | "auto" | auto / hsv / yolo / sam |

---

## Outputs

| File | Description |
|------|-------------|
| `dashboard.png` | 8-panel simulation dashboard |
| `multistage_dashboard.png` | Drogue→Main + snatch load analysis |
| `pendulum_dashboard.png` | Pendulum oscillation + Poincaré section |
| `mc_dashboard.png` | Monte Carlo P5/P50/P95 bands |
| `trajectory_3d.png` | 3D wind-drift + landing ellipse |
| `opening_shock.png` | Force history + structural safety factors |
| `bayes_cd_posterior.png` | Bayesian posterior + MCMC chains |
| `pinn_ensemble.png` | Ensemble disagreement heat-map |
| `design_calc.png` | Area solver + performance sweep |
| `mach_cd_corrections.png` | Mach + Reynolds + porosity corrections |
| `simulation_animation.gif` | Animated descent (shareable) |
| `engineering_report.html` | Self-contained HTML report |
| `design_datasheet.html` | Engineering datasheet |
| `trajectory.kml` | Google Earth flight path |

---

## Optional Dependencies

```bash
pip install torch           # PINN + ensemble (Phase 3)
pip install streamlit       # Web dashboard
pip install emcee corner    # Full MCMC chains
pip install pymavlink       # Pixhawk .bin telemetry
pip install gpxpy           # GPX track files
pip install ultralytics     # YOLO AI segmentation (AeroDecel v5.0)
```

---

## Commercial Applications

- **Drone Recovery Systems**: UAV parachute sizing and certification
- **Hypersonic Drogue Deployment**: Mach-corrected drag prediction
- **Orbital Reentry Capsules**: Heavy-payload deceleration analysis
- **Cargo Airdrop**: MIL-HDBK-1791 compliance verification
- **Sport Parachutes**: Performance optimization and safety analysis

---

## Citation

```bibtex
@software{aerodecel2025,
  title     = {AeroDecel: AI-Driven Aerodynamic Deceleration Analysis Framework},
  version   = {5.0.0},
  year      = {2025},
  url       = {https://github.com/aerodecel/aerodecel},
  note      = {Open-source computational aerodynamics for parachute dynamics}
}
```

### Key References

1. Knacke, T.W., *"Parachute Recovery Systems Design Manual"*, Para Publishing, 1992.
2. Pflanz, E., *"Effect of Porosity on Drag of Parachute Canopies"*, WADC TR 52-38, 1952.
3. MIL-HDBK-1791, *"Designing for Internal Aerial Delivery in Fixed Wing Aircraft"*, 1997.
4. Raissi, M., Perdikaris, P., Karniadakis, G.E., *"Physics-Informed Neural Networks"*, JCP, 2019.
5. Tancik, M. et al., *"Fourier Features Let Networks Learn High Frequency Functions"*, NeurIPS, 2020.
6. Lamb, H., *"Hydrodynamics"*, 6th ed., Cambridge Univ. Press, 1932.

---

**Total cost: $0.00 · All open source · All local · Zero API keys required**
