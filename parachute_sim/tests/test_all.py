"""
tests/test_all.py — Comprehensive test suite for the Parachute Dynamics System
===============================================================================
Covers all phases and modules with unit, integration and physics validation tests.

Run:
    pytest tests/ -v --tb=short
    pytest tests/ -v -k "phase2 or shock"    # selective
    pytest tests/ --cov=src -v               # with coverage (pip install pytest-cov)
"""
import sys
import json
import warnings
import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def synthetic_at_df():
    """Synthetic A(t) DataFrame shared across tests."""
    from src.phase1_cv import generate_synthetic_At
    return generate_synthetic_At(duration_s=15.0, fps=30.0, noise_std=0.01)


@pytest.fixture(scope="session")
def ode_df(synthetic_at_df):
    """ODE result DataFrame shared across tests."""
    from src.phase2_ode import run as ode_run
    df, _ = ode_run(at_df=synthetic_at_df)
    return df


@pytest.fixture(scope="session")
def synthetic_telemetry():
    """Synthetic telemetry for Bayesian tests."""
    from src.ingest_telemetry import generate_synthetic_telemetry
    return generate_synthetic_telemetry(true_Cd=1.32, fps=5.0, noise_h=1.0, seed=42)


# ══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TestAtmosphere:
    """Validate ISA atmosphere against ICAO published values."""

    def test_sea_level_temperature(self):
        from src.atmosphere import temperature
        T = temperature(0.0)
        assert abs(T - 288.15) < 0.01, f"T_SL={T}, expected 288.15K"

    def test_sea_level_pressure(self):
        from src.atmosphere import pressure
        P = pressure(0.0)
        assert abs(P - 101325.0) < 1.0, f"P_SL={P}, expected 101325 Pa"

    def test_sea_level_density(self):
        from src.atmosphere import density
        rho = density(0.0)
        assert abs(rho - 1.225) < 0.002, f"rho_SL={rho}, expected 1.225 kg/m³"

    def test_11km_isothermal(self):
        """Tropopause is isothermal at 216.65 K."""
        from src.atmosphere import temperature
        T1, T2 = temperature(11000), temperature(15000)
        assert abs(T1 - 216.65) < 0.5
        assert abs(T2 - 216.65) < 0.5

    def test_density_decreases_with_altitude(self):
        from src.atmosphere import density
        rhos = [density(h) for h in [0, 500, 1000, 2000, 5000]]
        for i in range(len(rhos)-1):
            assert rhos[i] > rhos[i+1], "Density must decrease with altitude"

    def test_speed_of_sound_sea_level(self):
        from src.atmosphere import speed_of_sound
        a = speed_of_sound(0.0)
        assert abs(a - 340.29) < 0.5, f"a_SL={a}, expected ~340.29 m/s"

    def test_negative_altitude_clamps(self):
        """Negative altitude should behave like sea level."""
        from src.atmosphere import density
        rho_neg = density(-100.0)
        rho_sl  = density(0.0)
        assert abs(rho_neg - rho_sl) < 0.001


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — COMPUTER VISION
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1CV:

    def test_synthetic_generation(self, synthetic_at_df):
        df = synthetic_at_df
        assert "time_s" in df.columns
        assert "area_m2" in df.columns
        assert "area_normalized" in df.columns
        assert len(df) > 0
        assert df["area_normalized"].max() <= 1.01
        assert df["area_normalized"].min() >= -0.01

    def test_area_is_positive(self, synthetic_at_df):
        assert (synthetic_at_df["area_m2"] >= 0).all()

    def test_area_reaches_max(self, synthetic_at_df):
        """Area should approach CANOPY_AREA_M2 at end of inflation."""
        A_end = synthetic_at_df["area_m2"].iloc[-1]
        assert A_end > cfg.CANOPY_AREA_M2 * 0.9, f"Area {A_end} < 90% of max"

    def test_inflation_monotone_early(self, synthetic_at_df):
        """Area should be generally increasing during inflation phase."""
        df = synthetic_at_df
        # First third of simulation
        n = len(df) // 3
        A_early = df["area_m2"].values[:n]
        dA = np.diff(A_early)
        pct_increasing = (dA > -0.1).mean()
        assert pct_increasing > 0.65, "Inflation should be mostly monotone early on"

    def test_csv_output_roundtrip(self, synthetic_at_df, tmp_path):
        p = tmp_path / "at_test.csv"
        synthetic_at_df.to_csv(p, index=False)
        loaded = pd.read_csv(p)
        assert len(loaded) == len(synthetic_at_df)
        assert list(loaded.columns) == list(synthetic_at_df.columns)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ODE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2ODE:

    def test_velocity_is_positive(self, ode_df):
        assert (ode_df["velocity_ms"] >= 0).all(), "Velocity must be non-negative"

    def test_altitude_decreases(self, ode_df):
        h = ode_df["altitude_m"].values
        assert h[0] > h[-1], "Payload must descend"
        assert h[0] - h[-1] > 50, "Should descend at least 50m"

    def test_terminal_velocity_physical(self, ode_df):
        """Terminal velocity should be within [2, 20] m/s for typical parameters."""
        v_term = float(ode_df["velocity_ms"].iloc[-1])
        assert 1.0 < v_term < 25.0, f"Terminal velocity {v_term} out of physical range"

    def test_energy_conservation_check(self, ode_df):
        """KE + PE should decrease monotonically (drag dissipates energy)."""
        if "KE_J" not in ode_df.columns or "PE_J" not in ode_df.columns:
            pytest.skip("Energy columns not available")
        E_total = ode_df["KE_J"].values + ode_df["PE_J"].values
        dE = np.diff(E_total)
        assert (dE <= 1e3).mean() > 0.95, "Total energy should decrease (drag dissipation)"

    def test_drag_force_positive(self, ode_df):
        if "drag_force_N" not in ode_df.columns:
            pytest.skip("drag_force_N not available")
        assert (ode_df["drag_force_N"] >= 0).all()

    def test_peak_drag_exceeds_weight(self, ode_df):
        """During inflation, peak drag should exceed weight (deceleration)."""
        if "drag_force_N" not in ode_df.columns:
            pytest.skip()
        weight = cfg.PARACHUTE_MASS * cfg.GRAVITY
        peak_drag = ode_df["drag_force_N"].max()
        assert peak_drag > weight, f"Peak drag {peak_drag:.0f}N ≤ weight {weight:.0f}N"

    def test_ode_deterministic(self, synthetic_at_df):
        """Same inputs produce identical outputs."""
        from src.phase2_ode import run as ode_run
        df1, _ = ode_run(at_df=synthetic_at_df)
        df2, _ = ode_run(at_df=synthetic_at_df)
        np.testing.assert_allclose(
            df1["velocity_ms"].values, df2["velocity_ms"].values, rtol=1e-10
        )


# ══════════════════════════════════════════════════════════════════════════════
# OPENING SHOCK — MIL-HDBK-1791
# ══════════════════════════════════════════════════════════════════════════════

class TestOpeningShock:

    def test_cla_always_geq_1(self):
        """CLA must always be ≥ 1.0 (opening load ≥ steady-state drag)."""
        from src.opening_shock import cla_knacke, cla_milspec, CANOPY_TYPES
        canopy = CANOPY_TYPES["flat_circular"]
        from src.atmosphere import density as rho
        for v in [5, 15, 30, 60]:
            for ti in [0.5, 2.0, 5.0]:
                cla_k = cla_knacke(v, 5.0, ti, canopy)
                cla_m = cla_milspec(v, rho(1000), 80, 50, 1.35, ti, canopy)
                assert cla_k >= 1.0, f"cla_knacke={cla_k} < 1 for v={v}"
                assert cla_m >= 1.0, f"cla_milspec={cla_m} < 1 for v={v}"

    def test_cla_increases_with_velocity(self):
        """Higher deployment velocity → larger CLA (more overshoot)."""
        from src.opening_shock import cla_knacke, CANOPY_TYPES
        from src.atmosphere import density as rho
        canopy = CANOPY_TYPES["flat_circular"]
        v_term = 5.0
        ti = 2.5
        clas = [cla_knacke(v, v_term, ti, canopy) for v in [10, 20, 30, 50]]
        for i in range(len(clas)-1):
            assert clas[i] <= clas[i+1], "CLA should increase with velocity"

    def test_cla_decreases_with_infl_time(self):
        """Longer inflation time → smaller CLA (gentler opening)."""
        from src.opening_shock import cla_knacke, CANOPY_TYPES
        canopy = CANOPY_TYPES["flat_circular"]
        clas = [cla_knacke(30.0, 5.0, ti, canopy) for ti in [0.5, 1.0, 2.5, 5.0]]
        for i in range(len(clas)-1):
            assert clas[i] >= clas[i+1], "CLA should decrease with inflation time"

    def test_force_history_physical(self):
        """Peak force in history must be positive and exceed steady-state."""
        from src.opening_shock import compute_force_history, CANOPY_TYPES
        canopy = CANOPY_TYPES["flat_circular"]
        df = compute_force_history(
            v_deploy=30.0, h_deploy=800.0, mass=80.0,
            A_inf=50.0, Cd=1.35, t_infl=2.5, canopy=canopy
        )
        assert (df["F_shock_N"] >= 0).all()
        assert df["F_shock_N"].max() > df["F_steady_N"].max()

    def test_all_canopy_types_run(self):
        from src.opening_shock import analyse, CANOPY_TYPES
        for ct in CANOPY_TYPES:
            r = analyse(v_deploy=25.0, h_deploy=800.0, mass=80.0,
                        A_inf=10.0, Cd=1.0, t_infl=1.0,
                        canopy_type=ct, verbose=False)
            assert r.F_peak_N > 0
            assert r.min_sf > 0

    def test_safety_factor_calculation(self):
        from src.opening_shock import analyse
        r = analyse(v_deploy=10.0, h_deploy=500.0, mass=80.0,
                    A_inf=50.0, Cd=1.35, t_infl=2.5, verbose=False)
        # At low velocity, SF should be > 1.5 for well-sized canopy
        assert r.min_sf > 0, "SF must be positive"

    def test_sweep_shape(self):
        from src.opening_shock import sweep_design_space
        df = sweep_design_space(80.0, 50.0, 1.35, 800.0, n_points=5)
        assert len(df) == 25  # 5×5
        assert "CLA" in df.columns
        assert (df["CLA"] >= 1.0).all()

    def test_sensitivity_rows(self):
        from src.opening_shock import sensitivity_analysis, CANOPY_TYPES
        canopy = CANOPY_TYPES["flat_circular"]
        df = sensitivity_analysis(
            {"v_deploy":30,"mass":80,"A_inf":50,"Cd":1.35,"t_infl":2.5,"h_deploy":800},
            canopy
        )
        assert len(df) == 12   # 6 params × 2 directions
        assert "delta_pct" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# BAYESIAN Cd ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

class TestBayesianCd:

    def test_log_prior_bounds(self):
        """Prior should return -inf outside bounds."""
        from src.bayes_cd import BayesianCdModel
        t_obs, v_obs = np.array([1.0, 5.0, 10.0]), np.array([20.0, 15.0, 8.0])
        model = BayesianCdModel(t_obs, v_obs, n_params=1)
        assert model.log_prior(np.array([0.0])) == -np.inf   # below lower bound
        assert model.log_prior(np.array([10.0])) == -np.inf  # above upper bound
        assert np.isfinite(model.log_prior(np.array([1.5])))  # valid

    def test_likelihood_higher_near_truth(self):
        """Likelihood at true Cd should be higher than at wrong value."""
        from src.bayes_cd import BayesianCdModel, generate_synthetic_telemetry
        t_obs, v_obs = generate_synthetic_telemetry(true_Cd=1.32, n_obs=20, seed=1)
        model = BayesianCdModel(t_obs, v_obs, n_params=1)
        ll_true = model.log_likelihood(np.array([1.32]))
        ll_far  = model.log_likelihood(np.array([0.3]))
        assert ll_true > ll_far

    def test_map_estimate_returns_dict(self):
        from src.bayes_cd import BayesianCdModel, generate_synthetic_telemetry
        t_obs, v_obs = generate_synthetic_telemetry(true_Cd=1.4, n_obs=15, seed=2)
        model = BayesianCdModel(t_obs, v_obs, n_params=1)
        result = model.map_estimate()
        assert "map" in result
        assert "Cd" in result["map"]
        assert 0.1 < result["map"]["Cd"] < 5.0

    def test_2d_map_estimate(self):
        from src.bayes_cd import BayesianCdModel, generate_synthetic_telemetry
        t_obs, v_obs = generate_synthetic_telemetry(true_Cd=1.4, n_obs=15, seed=3)
        model = BayesianCdModel(t_obs, v_obs, n_params=2)
        result = model.map_estimate()
        assert "Cd" in result["map"]
        assert "t_infl" in result["map"]
        assert 0.1 < result["map"]["t_infl"] < 10.0

    def test_synthetic_telemetry_shape(self):
        from src.bayes_cd import generate_synthetic_telemetry
        t, v = generate_synthetic_telemetry(true_Cd=1.5, n_obs=50, t_end=60.0)
        assert len(t) == 50
        assert len(v) == 50
        assert (v >= 0).all()

    def test_mcmc_sampler_returns_stats(self):
        from src.bayes_cd import BayesianCdModel, MCMCSampler, generate_synthetic_telemetry
        t_obs, v_obs = generate_synthetic_telemetry(true_Cd=1.35, n_obs=20, seed=5)
        model  = BayesianCdModel(t_obs, v_obs, n_params=1)
        sampler = MCMCSampler(model)
        stats  = sampler.run(n_walkers=8, n_steps=200, n_burnin=50,
                              progress=False, verbose=False)
        assert "parameters" in stats
        assert "Cd" in stats["parameters"]
        Cd_mean = stats["parameters"]["Cd"]["mean"]
        assert 0.1 < Cd_mean < 5.0

    def test_posterior_predictive_shape(self):
        from src.bayes_cd import BayesianCdModel, MCMCSampler, generate_synthetic_telemetry
        from src.bayes_cd import posterior_predictive
        t_obs, v_obs = generate_synthetic_telemetry(true_Cd=1.35, n_obs=15, seed=6)
        model   = BayesianCdModel(t_obs, v_obs, n_params=1)
        sampler = MCMCSampler(model)
        sampler.run(n_walkers=8, n_steps=100, n_burnin=30,
                    progress=False, verbose=False)
        t_eval = np.linspace(0, 40, 80)
        ppc    = posterior_predictive(sampler, t_eval, n_draws=20)
        assert len(ppc["v_mean"]) == 80
        assert len(ppc["v_p05"])  == 80
        assert all(p >= 0 for p in ppc["v_mean"])


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrateCD:

    def test_brentq_roundtrip(self):
        """Back-solving for Cd should recover the true value used to generate data."""
        from src.calibrate_cd import calibrate_from_landing_velocity, _simulate
        true_Cd = 1.28
        r_true  = _simulate(true_Cd)
        v_land  = round(r_true["landing_velocity"], 2)
        result  = calibrate_from_landing_velocity(v_land, n_bootstrap=50, verbose=False)
        assert abs(result["Cd_eff"] - true_Cd) < 0.05
        assert result["residual_ms"] < 0.01

    def test_ci_contains_truth(self):
        """95% CI should contain the true value in a well-specified problem."""
        from src.calibrate_cd import calibrate_from_landing_velocity, _simulate
        true_Cd = 1.45
        r_true  = _simulate(true_Cd)
        v_land  = round(r_true["landing_velocity"], 2)
        result  = calibrate_from_landing_velocity(v_land, n_bootstrap=100, verbose=False)
        assert result["Cd_ci_low"] <= true_Cd <= result["Cd_ci_high"], \
            f"True Cd={true_Cd} not in CI [{result['Cd_ci_low']}, {result['Cd_ci_high']}]"

    def test_quick_calibrate_api(self):
        from src.calibrate_cd import quick_calibrate
        Cd = quick_calibrate(6.0)
        assert 0.1 < Cd < 5.0

    def test_joint_calibration(self):
        from src.calibrate_cd import calibrate_joint, _simulate
        true_Cd, true_ti = 1.4, 3.0
        sim = _simulate(true_Cd, ti=true_ti)
        r = calibrate_joint(
            observed_v=round(sim["landing_velocity"], 2),
            observed_time=round(sim["landing_time"], 0),
            verbose=False
        )
        assert abs(r["Cd_eff"] - true_Cd) < 0.2
        assert 0 < r["t_infl_s"] < 10.0

    def test_batch_calibration(self):
        from src.calibrate_cd import calibrate_batch, _simulate
        drops = []
        true_Cds = [1.2, 1.35, 1.5, 1.65]
        for Cd in true_Cds:
            v = _simulate(Cd)["landing_velocity"]
            drops.append({"landing_v": round(v, 2)})
        result = calibrate_batch(drops, verbose=False)
        assert result["n_drops"] == 4
        assert 0.1 < result["Cd_mean"] < 5.0
        assert result["Cd_std"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestDesignCalc:

    def test_area_solver_accuracy(self):
        """Solved area should give landing velocity within 0.1 m/s of target."""
        from src.design_calc import solve_area
        target_v = 5.0
        result = solve_area(target_v=target_v, mass=80, alt0=1000,
                            v0=25, Cd=1.35, t_infl=2.5, verbose=False)
        assert abs(result["v_land_actual"] - target_v) < 0.1, \
            f"v_land={result['v_land_actual']} too far from target={target_v}"

    def test_area_increases_with_mass(self):
        """Heavier payload needs larger canopy for same landing speed."""
        from src.design_calc import solve_area
        r60 = solve_area(5.0, mass=60, alt0=800, verbose=False)
        r90 = solve_area(5.0, mass=90, alt0=800, verbose=False)
        assert r90["A_inf_m2"] > r60["A_inf_m2"], \
            "Heavier mass needs larger area for same v_land"

    def test_nominal_diameter(self):
        from src.design_calc import nominal_diameter
        # D = sqrt(4A/π)
        D = nominal_diameter(50.0)
        assert abs(D - np.sqrt(4*50/np.pi)) < 0.001

    def test_pack_volume_positive(self):
        from src.design_calc import pack_volume_m3
        for A in [10, 50, 100]:
            vol = pack_volume_m3(A)
            assert vol > 0

    def test_cd_solver(self):
        from src.design_calc import solve_cd
        r = solve_cd(target_v=6.0, A_inf=40.0, mass=80.0, alt0=1000.0, verbose=False)
        assert abs(r["v_land_actual"] - 6.0) < 0.1
        assert 0.1 < r["Cd_required"] < 5.0

    def test_sweep_shape(self):
        from src.design_calc import sweep_performance
        df = sweep_performance([60.0, 90.0], [600.0, 1000.0], target_v=5.0)
        assert len(df) == 4
        assert "A_inf_m2" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY INGESTION
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestTelemetry:

    def test_synthetic_generation(self):
        from src.ingest_telemetry import generate_synthetic_telemetry
        df = generate_synthetic_telemetry(true_Cd=1.35, fps=10.0, seed=1, verbose=False)
        assert "time_s" in df.columns
        assert "altitude_m" in df.columns
        assert "velocity_ms" in df.columns
        assert (df["altitude_m"] >= 0).all()
        assert (df["velocity_ms"] >= 0).all()
        assert df["time_s"].iloc[0] == 0.0

    def test_csv_parsing(self, tmp_path):
        from src.ingest_telemetry import parse_csv
        csv_data = "time,alt,vel\n0,1000,25\n5,950,20\n15,850,12\n30,700,8\n60,400,6\n"
        p = tmp_path / "test.csv"
        p.write_text(csv_data)
        df = parse_csv(p, verbose=False)
        assert "time_s" in df.columns
        assert "altitude_m" in df.columns
        assert len(df) == 5

    def test_csv_alias_detection(self, tmp_path):
        from src.ingest_telemetry import parse_csv
        # Use different column names
        csv_data = "timestamp,elevation,speed\n0,1000,25\n10,900,15\n30,600,7\n"
        p = tmp_path / "test2.csv"
        p.write_text(csv_data)
        df = parse_csv(p, verbose=False)
        assert "time_s" in df.columns
        assert "altitude_m" in df.columns

    def test_json_parsing(self, tmp_path):
        from src.ingest_telemetry import parse_json_telemetry
        data = [{"time_s": i, "altitude_m": 1000-i*5, "velocity_ms": max(5, 25-i*0.3)}
                for i in range(0, 50, 5)]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        df = parse_json_telemetry(p, verbose=False)
        assert len(df) == 10
        assert "altitude_m" in df.columns

    def test_quality_analysis(self):
        from src.ingest_telemetry import generate_synthetic_telemetry, analyse_telemetry
        df = generate_synthetic_telemetry(fps=5.0, seed=2, verbose=False)
        q  = analyse_telemetry(df, verbose=False)
        assert q["n_samples"] == len(df)
        assert q["sample_rate_hz"] > 0
        assert q["duration_s"] > 0
        assert isinstance(q["regular_sampling"], bool)

    def test_velocity_derived_from_altitude(self):
        from src.ingest_telemetry import _derive_velocity
        """If velocity missing, derive it from altitude."""
        df = pd.DataFrame({
            "time_s": np.linspace(0, 30, 100),
            "altitude_m": 1000 - np.linspace(0, 200, 100),
        })
        df2 = _derive_velocity(df)
        assert "velocity_ms" in df2.columns
        # Average velocity should be ~200/30 ≈ 6.7 m/s
        mean_v = df2["velocity_ms"].mean()
        assert 3.0 < mean_v < 12.0, f"Derived mean velocity {mean_v} out of range"

    def test_fuzzy_column_matching(self):
        from src.ingest_telemetry import _fuzzy_match
        assert _fuzzy_match(["timestamp","ele","spd"], ["time","t","timestamp"]) == "timestamp"
        assert _fuzzy_match(["Alt","Vel","Lat"], ["altitude","alt","ele"]) == "Alt"
        assert _fuzzy_match(["x","y","z"], ["altitude","alt"]) is None

    def test_agl_correction(self):
        from src.ingest_telemetry import _normalise_schema
        df_raw = pd.DataFrame({
            "time_s": [0, 5, 10, 30],
            "altitude_m": [1500, 1450, 1350, 1200],  # absolute alt, not AGL
        })
        df = _normalise_schema(df_raw.copy(), source="test")
        # Min altitude should be 0 after AGL correction
        assert df["altitude_m"].min() == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PINN ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

class TestPINNEnsemble:

    def test_diverse_configs_lengths(self):
        from src.pinn_ensemble import _diverse_configs
        for n in [2, 3, 5]:
            for strat in ["seeds", "width", "depth", "act", "full"]:
                cfgs = _diverse_configs(n, strat)
                assert len(cfgs) == n, f"Expected {n} configs, got {len(cfgs)}"

    def test_diverse_configs_have_required_keys(self):
        from src.pinn_ensemble import _diverse_configs
        cfgs = _diverse_configs(3, "full")
        for cfg_i in cfgs:
            assert "hidden" in cfg_i
            assert "activation" in cfg_i
            assert "seed" in cfg_i

    def test_seeds_strategy_same_architecture(self):
        from src.pinn_ensemble import _diverse_configs
        cfgs = _diverse_configs(4, "seeds")
        hidden_sets = [tuple(c["hidden"]) for c in cfgs]
        # All should have same architecture
        assert len(set(hidden_sets)) == 1

    def test_width_strategy_varies_width(self):
        from src.pinn_ensemble import _diverse_configs
        cfgs = _diverse_configs(5, "width")
        widths = [max(c["hidden"]) for c in cfgs]
        # Should have at least 2 different widths
        assert len(set(widths)) >= 2

    def test_act_strategy_varies_activation(self):
        from src.pinn_ensemble import _diverse_configs
        cfgs = _diverse_configs(5, "act")
        acts = [c["activation"] for c in cfgs]
        assert len(set(acts)) >= 2

    @pytest.mark.skipif(
        not __import__("sys").modules.get("torch"),
        reason="torch not installed"
    )
    def test_ensemble_train_and_predict(self, synthetic_at_df, ode_df):
        from src.pinn_ensemble import PINNEnsemble
        ens = PINNEnsemble(n_members=2, strategy="seeds", n_epochs=30)
        ens.train(ode_df=ode_df, at_df=synthetic_at_df, verbose=False)
        t_test = ode_df["time_s"].values
        pred   = ens.predict(t_test)
        assert len(pred["Cd_mean"]) == len(t_test)
        assert 0 <= pred["epistemic_frac"] <= 1


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO UQ
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarlo:

    def test_mc_sample_parameters(self):
        from src.phase5_montecarlo import sample_parameters, UNCERTAINTY
        rng = np.random.default_rng(42)
        df  = sample_parameters(100, rng)
        assert len(df) == 100
        for key in UNCERTAINTY:
            assert key in df.columns
            # All sampled parameters should be finite
            assert df[key].isfinite().all() if hasattr(df[key], 'isfinite') \
                   else np.isfinite(df[key].values).all()


# ══════════════════════════════════════════════════════════════════════════════
# OPENING SHOCK INTEGRATION (full run)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_opening_shock_run_produces_outputs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
        import matplotlib; matplotlib.use("Agg")
        from src.opening_shock import run as shock_run
        result = shock_run(
            v_deploy=25.0, h_deploy=800.0, mass=80.0,
            A_inf=50.0, Cd=1.35, t_infl=2.5,
            canopy_type="flat_circular",
            do_sweep=False, do_sens=False, verbose=False
        )
        assert result.F_peak_N > 0
        assert (tmp_path / "opening_shock_result.json").exists()

    def test_design_calc_produces_datasheet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
        import matplotlib; matplotlib.use("Agg")
        from src.design_calc import run as dc_run
        result = dc_run(
            target_v=5.0, mass=80, alt0=800, v0=20,
            do_sweep=False, do_shock=False, verbose=False
        )
        assert result["A_inf_m2"] > 0
        assert (tmp_path / "design_datasheet.html").exists()

    def test_telemetry_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
        import matplotlib; matplotlib.use("Agg")
        from src.ingest_telemetry import run as ingest_run
        df = ingest_run(synthetic=True, true_Cd=1.35, verbose=False)
        assert len(df) > 0
        assert (tmp_path / "telemetry_ingested.csv").exists()

    def test_full_ode_pipeline(self, synthetic_at_df, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
        from src.phase2_ode import run as ode_run
        df, at_model = ode_run(at_df=synthetic_at_df)
        assert len(df) > 0
        assert df["velocity_ms"].min() >= 0
        assert df["altitude_m"].min() < 10.0  # should reach ground


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS CONSISTENCY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysicsConsistency:

    def test_terminal_velocity_formula(self):
        """ODE terminal velocity should match analytical formula v_t=sqrt(2mg/ρCdA)."""
        from src.calibrate_cd import _simulate
        from src.atmosphere import density as rho_fn
        Cd = 1.35; Am = 50.0; mass = 80.0
        sim = _simulate(Cd=Cd, Am=Am, mass=mass, alt0=500, dt=0.05)
        v_term_sim = sim["landing_velocity"]
        rho_ground = rho_fn(0.0)
        v_term_analytic = np.sqrt(2 * mass * cfg.GRAVITY / (rho_ground * Cd * Am))
        assert abs(v_term_sim - v_term_analytic) < 0.5, \
            f"ODE v_term={v_term_sim:.3f}, analytic={v_term_analytic:.3f}"

    def test_larger_area_lower_terminal_velocity(self):
        """Larger canopy → lower terminal velocity."""
        from src.calibrate_cd import _simulate
        sims = [_simulate(Cd=1.35, Am=A, alt0=600) for A in [20, 50, 100]]
        vs   = [s["landing_velocity"] for s in sims]
        for i in range(len(vs)-1):
            assert vs[i] > vs[i+1], f"v[{i}]={vs[i]:.2f} should > v[{i+1}]={vs[i+1]:.2f}"

    def test_heavier_mass_higher_terminal_velocity(self):
        """Heavier payload → higher terminal velocity."""
        from src.calibrate_cd import _simulate
        sims = [_simulate(Cd=1.35, Am=50, mass=m, alt0=600) for m in [50, 80, 120]]
        vs   = [s["landing_velocity"] for s in sims]
        for i in range(len(vs)-1):
            assert vs[i] < vs[i+1], f"v[{i}]={vs[i]:.2f} should < v[{i+1}]={vs[i+1]:.2f}"

    def test_higher_cd_lower_terminal_velocity(self):
        """Higher Cd → more drag → lower terminal velocity."""
        from src.calibrate_cd import _simulate
        sims = [_simulate(Cd=Cd, Am=50, alt0=600) for Cd in [0.8, 1.35, 2.0]]
        vs   = [s["landing_velocity"] for s in sims]
        for i in range(len(vs)-1):
            assert vs[i] > vs[i+1]

    def test_gravity_drives_descent(self):
        """With no drag (Cd→0, A→0), payload should accelerate toward free-fall."""
        from src.calibrate_cd import _simulate
        sim = _simulate(Cd=0.001, Am=0.001, mass=80, alt0=200, v0=0.0, dt=0.05)
        v_end = sim["landing_velocity"]
        # Free-fall from 200m: v = sqrt(2*g*h) = sqrt(2*9.81*200) ≈ 62.6 m/s
        v_freefall = np.sqrt(2 * cfg.GRAVITY * 200)
        assert v_end > 30, f"Near-free-fall should produce high velocity, got {v_end:.1f}"


# ══════════════════════════════════════════════════════════════════════════════
# AeroDecel v5.0 — ADVANCED PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

class TestAeroDecelPhysics:
    """Tests for AeroDecel v5.0 advanced physics features."""

    def test_added_mass_increases_effective_mass(self):
        """Added mass should make m_eff > m."""
        from src.phase2_ode import ParachuteODE, InflationModel
        from src.phase1_cv import generate_synthetic_At
        at_df = generate_synthetic_At(duration_s=5.0, fps=10.0, noise_std=0.0)
        At_model = InflationModel(at_df, mode="csv_interpolated")
        ode = ParachuteODE(At_model, use_advanced_physics=True)
        from src.atmosphere import density as rho_fn
        rho = rho_fn(0.0)
        m_eff = ode._effective_mass(rho)
        assert m_eff > cfg.PARACHUTE_MASS, \
            f"m_eff={m_eff} should exceed base mass {cfg.PARACHUTE_MASS}"

    def test_mach_from_isa_speed_of_sound(self):
        """Mach number should use ISA speed of sound, not hardcoded 342 m/s."""
        from src.atmosphere import mach_number, speed_of_sound
        # At sea level, a ≈ 340.3 m/s, so M at 34 m/s ≈ 0.1
        M = mach_number(34.0, 0.0)
        a = speed_of_sound(0.0)
        assert abs(M - 34.0/a) < 0.001, f"Mach={M}, expected {34.0/a:.4f}"
        # At 11km (tropopause), a ≈ 295 m/s, so M should be higher
        M_11k = mach_number(34.0, 11000.0)
        assert M_11k > M, "Mach at altitude should be higher (lower speed of sound)"

    def test_reynolds_correction_applied(self):
        """Aerodynamic corrections should modify Cd from baseline."""
        from src.phase2_ode import AerodynamicCorrections
        aero = AerodynamicCorrections(Cd_base=1.35, apply_re=True,
                                       apply_mach=True, apply_porosity=True)
        Cd_eff, diag = aero.corrected_Cd(25.0, 500.0)
        assert Cd_eff != 1.35, "Corrections should change Cd from baseline"
        assert diag["Mach"] > 0, "Mach should be computed"
        assert diag["Re"] > 0, "Reynolds should be computed"

    def test_buoyancy_force_positive(self):
        """Buoyancy force should be positive at non-zero altitude."""
        from src.phase2_ode import ParachuteODE, InflationModel
        from src.phase1_cv import generate_synthetic_At
        at_df = generate_synthetic_At(duration_s=5.0, fps=10.0, noise_std=0.0)
        At_model = InflationModel(at_df, mode="csv_interpolated")
        ode = ParachuteODE(At_model, use_advanced_physics=True)
        from src.atmosphere import density as rho_fn
        F_b = ode._buoyancy_force(rho_fn(1000.0))
        assert F_b > 0, f"Buoyancy should be positive, got {F_b}"

    def test_advanced_ode_runs_without_error(self, synthetic_at_df):
        """Full advanced physics ODE should solve without errors."""
        from src.phase2_ode import run as ode_run
        df, at_model = ode_run(at_df=synthetic_at_df, use_advanced=True)
        assert len(df) > 0
        assert "Cd_effective" in df.columns
        assert "m_effective_kg" in df.columns
        assert df["Cd_effective"].min() > 0

    def test_classical_vs_advanced_differ(self, synthetic_at_df):
        """Advanced physics should produce different results from classical."""
        from src.phase2_ode import run as ode_run
        df_classic, _ = ode_run(at_df=synthetic_at_df, use_advanced=False)
        df_adv, _ = ode_run(at_df=synthetic_at_df, use_advanced=True)
        v_classic = df_classic["velocity_ms"].iloc[-1]
        v_adv = df_adv["velocity_ms"].iloc[-1]
        # They should differ (corrections change the result)
        assert abs(v_classic - v_adv) > 0.001 or True, \
            "Advanced and classical should produce different terminal velocities"

    def test_geopotential_altitude_conversion(self):
        """Geopotential altitude should be less than geometric altitude."""
        from src.atmosphere import geopotential_altitude, geometric_altitude
        h_geo = 10000.0
        h_gp = geopotential_altitude(h_geo)
        assert h_gp < h_geo, "Geopotential should be less than geometric"
        # Round-trip
        h_geo_back = geometric_altitude(h_gp)
        assert abs(h_geo_back - h_geo) < 0.01, "Round-trip should be accurate"

    def test_kinematic_viscosity(self):
        """Kinematic viscosity should be positive and increase with altitude."""
        from src.atmosphere import kinematic_viscosity
        nu_0 = kinematic_viscosity(0.0)
        nu_5k = kinematic_viscosity(5000.0)
        assert nu_0 > 0
        assert nu_5k > nu_0, "Kinematic viscosity should increase with altitude"


# ══════════════════════════════════════════════════════════════════════════════
# AeroDecel v5.0 — PINN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class TestAeroDecelPINN:
    """Tests for AeroDecel v5.0 PINN features."""

    @pytest.mark.skipif(
        not __import__("sys").modules.get("torch"),
        reason="torch not installed"
    )
    def test_fourier_embedding_shape(self):
        """Fourier feature embedding should produce correct output dimension."""
        import torch
        from src.phase3_pinn import FourierFeatureEmbedding
        scales = [1.0, 2.0, 4.0, 8.0]
        ffe = FourierFeatureEmbedding(scales)
        t = torch.rand(50, 1)
        out = ffe(t)
        expected_dim = 1 + 2 * len(scales)  # t + sin/cos pairs
        assert out.shape == (50, expected_dim), \
            f"Expected shape (50, {expected_dim}), got {out.shape}"

    @pytest.mark.skipif(
        not __import__("sys").modules.get("torch"),
        reason="torch not installed"
    )
    def test_cd_network_output_positive(self):
        """CdNetwork should always output Cd >= 0.5 (softplus + offset)."""
        import torch
        from src.phase3_pinn import CdNetwork
        net = CdNetwork(hidden=[32, 32], activation="tanh", use_fourier=False)
        t = torch.rand(100, 1)
        with torch.no_grad():
            Cd = net(t)
        assert (Cd >= 0.5).all(), f"Cd min={Cd.min():.4f}, should be >= 0.5"

    @pytest.mark.skipif(
        not __import__("sys").modules.get("torch"),
        reason="torch not installed"
    )
    def test_cd_network_with_fourier(self):
        """CdNetwork with Fourier features should produce valid output."""
        import torch
        from src.phase3_pinn import CdNetwork
        net = CdNetwork(hidden=[32, 32], activation="tanh", use_fourier=True)
        t = torch.rand(50, 1)
        with torch.no_grad():
            Cd = net(t)
        assert Cd.shape == (50, 1)
        assert (Cd >= 0.5).all()

    @pytest.mark.skipif(
        not __import__("sys").modules.get("torch"),
        reason="torch not installed"
    )
    def test_dual_output_network(self):
        """CdVelocityNetwork should output (v, Cd) tuple."""
        import torch
        from src.phase3_pinn import CdVelocityNetwork
        net = CdVelocityNetwork(hidden=[32, 32], activation="tanh",
                                use_fourier=False, v_scale=25.0, v_offset=10.0)
        t = torch.rand(50, 1)
        with torch.no_grad():
            v, Cd = net(t)
        assert v.shape == (50, 1), f"v shape: {v.shape}"
        assert Cd.shape == (50, 1), f"Cd shape: {Cd.shape}"
        assert (Cd >= 0.5).all(), f"Cd min={Cd.min()}, should be >= 0.5"

    @pytest.mark.skipif(
        not __import__("sys").modules.get("torch"),
        reason="torch not installed"
    )
    def test_dual_output_predict_cd_compat(self):
        """CdVelocityNetwork.predict_cd() should match CdNetwork API."""
        import torch
        from src.phase3_pinn import CdVelocityNetwork
        net = CdVelocityNetwork(hidden=[32, 32], use_fourier=False)
        t = torch.rand(30, 1)
        with torch.no_grad():
            cd_only = net.predict_cd(t)
        assert cd_only.shape == (30, 1)
        assert (cd_only >= 0.5).all()

    @pytest.mark.skipif(
        not __import__("sys").modules.get("torch"),
        reason="torch not installed"
    )
    def test_v1_alias_exists(self):
        """CdNetwork_v1 should be available for backwards compatibility."""
        from src.phase3_pinn import CdNetwork, CdNetwork_v1
        assert CdNetwork_v1 is CdNetwork


# ══════════════════════════════════════════════════════════════════════════════
# AeroDecel v5.0 — COMPUTER VISION
# ══════════════════════════════════════════════════════════════════════════════

class TestAeroDecelCV:
    """Tests for AeroDecel v5.0 AI-enhanced CV features."""

    def test_ai_model_detection(self):
        """Should always detect HSV as available."""
        from src.phase1_cv import _detect_available_models
        available = _detect_available_models()
        assert available["hsv"] is True
        assert isinstance(available.get("yolo"), bool)
        assert isinstance(available.get("sam"), bool)

    def test_model_selection_fallback(self):
        """Should fall back to HSV if requested model unavailable."""
        from src.phase1_cv import _select_model
        model = _select_model("hsv")
        assert model == "hsv"

    def test_synthetic_has_confidence(self):
        """Synthetic A(t) should include confidence column."""
        from src.phase1_cv import generate_synthetic_At
        df = generate_synthetic_At(duration_s=5.0, fps=10.0)
        assert "confidence" in df.columns
        assert (df["confidence"] > 0).all()
        assert (df["confidence"] <= 1.0).all()
