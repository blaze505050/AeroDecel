"""
tests/test_v6.py — AeroDecel v6.0 Test Suite
=============================================
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.planetary_atm   import (MarsAtmosphere, VenusAtmosphere, TitanAtmosphere,
                                   get_planet_atmosphere, GenericPlanetAtmosphere)
from src.thermal_model   import ThermalProtectionSystem, MATERIAL_DB
from src.canopy_geometry import CanopyGeometry


# ══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE
# ══════════════════════════════════════════════════════════════════════════════

class TestPlanetaryAtmospheres:

    def test_mars_surface_density(self):
        m = MarsAtmosphere()
        assert abs(m.density(0) - 0.02) < 1e-6

    def test_mars_surface_temperature(self):
        m = MarsAtmosphere()
        assert abs(m.temperature(0) - 210.0) < 0.1

    def test_venus_surface_density(self):
        v = VenusAtmosphere()
        assert abs(v.density(0) - 65.0) < 0.01

    def test_titan_surface_density(self):
        t = TitanAtmosphere()
        assert abs(t.density(0) - 1.4) < 0.01

    def test_density_decreases_with_altitude(self):
        for PlanetCls in [MarsAtmosphere, VenusAtmosphere, TitanAtmosphere]:
            p = PlanetCls()
            assert p.density(0) > p.density(10_000) > p.density(50_000)

    def test_density_always_positive(self):
        for PlanetCls in [MarsAtmosphere, VenusAtmosphere, TitanAtmosphere]:
            p = PlanetCls()
            for h in [0, 1000, 50000, 200000]:
                assert p.density(h) > 0

    def test_temperature_always_positive(self):
        for PlanetCls in [MarsAtmosphere, VenusAtmosphere, TitanAtmosphere]:
            p = PlanetCls()
            for h in [0, 10000, 80000]:
                assert p.temperature(h) > 0

    def test_pressure_ideal_gas(self):
        m = MarsAtmosphere()
        rho = m.density(0); T = m.temperature(0); R = m.gas_constant
        P_calc = m.pressure(0)
        assert abs(P_calc - rho * R * T) < 1.0

    def test_speed_of_sound_positive(self):
        for PlanetCls in [MarsAtmosphere, VenusAtmosphere, TitanAtmosphere]:
            p = PlanetCls()
            assert p.speed_of_sound(0) > 0

    def test_mach_number(self):
        m = MarsAtmosphere()
        a = m.speed_of_sound(0)
        assert abs(m.mach_number(a, 0) - 1.0) < 0.001

    def test_factory_mars(self):
        p = get_planet_atmosphere("mars")
        assert p.name == "Mars"

    def test_factory_case_insensitive(self):
        p = get_planet_atmosphere("VENUS")
        assert p.name == "Venus"

    def test_factory_invalid(self):
        with pytest.raises(ValueError):
            get_planet_atmosphere("jupiter")

    def test_generic_planet(self):
        g = GenericPlanetAtmosphere(
            "TestWorld", 3e6, 1e24, 5.0, {"N2": 1.0},
            scale_height_m=10000, base_density=0.5, base_temp_K=200,
        )
        assert g.density(0) == 0.5
        assert abs(g.density(10000) - 0.5 * np.exp(-1)) < 0.01

    def test_profile_shape(self):
        m = MarsAtmosphere()
        prof = m.profile(alt_max_m=50000, n=100)
        assert len(prof["altitude_m"])    == 100
        assert len(prof["density"])       == 100
        assert len(prof["temperature_K"]) == 100
        assert len(prof["pressure_Pa"])   == 100


# ══════════════════════════════════════════════════════════════════════════════
# THERMAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TestThermalModel:

    def test_all_materials_importable(self):
        for mat in MATERIAL_DB:
            tps = ThermalProtectionSystem(mat, 0.01)
            assert tps.mat.name is not None

    def test_invalid_material(self):
        with pytest.raises(ValueError):
            ThermalProtectionSystem("unobtanium", 0.01)

    def test_sutton_graves_positive(self):
        tps = ThermalProtectionSystem("nylon", 0.01)
        q = tps.sutton_graves_heating(0.02, 5000, 1.0)
        assert q > 0

    def test_sutton_graves_scales_v_cubed(self):
        tps = ThermalProtectionSystem("nylon", 0.01)
        q1 = tps.sutton_graves_heating(0.02, 1000, 1.0)
        q2 = tps.sutton_graves_heating(0.02, 2000, 1.0)
        ratio = q2 / q1
        assert abs(ratio - 8.0) < 0.1    # v³ scaling

    def test_radiative_zero_below_threshold(self):
        tps = ThermalProtectionSystem("pica", 0.05)
        q = tps.radiative_heating(0.02, 4000, 1.0)
        assert q == 0.0

    def test_radiative_positive_high_v(self):
        tps = ThermalProtectionSystem("pica", 0.05)
        q = tps.radiative_heating(0.02, 9000, 1.0)
        assert q > 0

    def test_1d_conduction_shape(self):
        tps = ThermalProtectionSystem("kevlar", 0.02)
        t = np.linspace(0, 100, 50)
        T = tps.solve_1d_conduction(50000, t)
        assert T.shape == (50, tps.n_nodes)

    def test_1d_conduction_surface_hotter(self):
        tps = ThermalProtectionSystem("kevlar", 0.02, n_nodes=10)
        t = np.linspace(0, 100, 100)
        T = tps.solve_1d_conduction(100000, t)
        # Surface should be hotter than interior (after warm-up)
        assert T[-1, 0] >= T[-1, -1]

    def test_safety_margin_high_sf(self):
        tps = ThermalProtectionSystem("pica", 0.05)   # T_lim=3000K
        t = np.linspace(0, 10, 20)
        tps.solve_1d_conduction(100, t, T_initial_K=300)   # very low flux
        assert tps.safety_margin() > 5.0

    def test_safety_margin_low_sf(self):
        tps = ThermalProtectionSystem("nylon", 0.001, n_nodes=5)   # T_lim=450K
        t = np.linspace(0, 300, 100)
        tps.solve_1d_conduction(1_000_000, t, T_initial_K=400)    # extreme flux
        exceeded, _ = tps.check_material_limit()
        assert exceeded

    def test_diffusivity_positive(self):
        for mat in MATERIAL_DB:
            assert MATERIAL_DB[mat].diffusivity > 0

    def test_summary_keys(self):
        tps = ThermalProtectionSystem("nomex", 0.01)
        t = np.linspace(0, 30, 30)
        tps.solve_1d_conduction(20000, t)
        s = tps.summary()
        for key in ["material","thickness_m","T_max_allowed_K","T_peak_K","safety_factor","limit_exceeded"]:
            assert key in s

    def test_surface_reradiation_positive(self):
        tps = ThermalProtectionSystem("zylon", 0.01)
        q = tps.surface_reradiation(1000)
        assert q > 0


# ══════════════════════════════════════════════════════════════════════════════
# CANOPY GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

class TestCanopyGeometry:

    def test_circular_area(self):
        cg = CanopyGeometry("circular", {"r": 5})
        assert abs(cg.calculate_area() - np.pi * 25) < 1e-6

    def test_elliptical_area(self):
        cg = CanopyGeometry("elliptical", {"a": 10, "b": 5})
        assert abs(cg.calculate_area() - np.pi * 50) < 1e-4

    def test_rectangular_area(self):
        cg = CanopyGeometry("rectangular", {"width": 10, "height": 5})
        assert abs(cg.calculate_area() - 50) < 1e-6

    def test_disk_gap_band_area(self):
        cg = CanopyGeometry("disk_gap_band", {"r_disk": 5, "r_band": 6})
        assert cg.calculate_area() > 0

    def test_tricone_area(self):
        cg = CanopyGeometry("tricone", {"r_base": 4})
        assert cg.calculate_area() > 0

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            CanopyGeometry("balloon", {})

    def test_cross_section_circular(self):
        cg = CanopyGeometry("circular", {"r": 3})
        x, y = cg.generate_cross_section(100)
        r = np.sqrt(x**2 + y**2)
        assert abs(r.mean() - 3.0) < 0.05

    def test_cross_section_all_shapes(self):
        for shape, dims in [
            ("circular",      {"r": 5}),
            ("elliptical",    {"a": 10, "b": 5}),
            ("rectangular",   {"width": 8, "height": 4}),
            ("disk_gap_band", {"r_disk": 5}),
            ("tricone",       {"r_base": 4}),
        ]:
            cg = CanopyGeometry(shape, dims)
            x, y = cg.generate_cross_section()
            assert len(x) > 0 and len(y) > 0

    def test_cd_subsonic_positive(self):
        for shape, dims in [("circular",{"r":5}), ("elliptical",{"a":10,"b":5})]:
            cg = CanopyGeometry(shape, dims)
            assert cg.calculate_drag_coefficient(0.1) > 0

    def test_cd_transonic_higher(self):
        cg = CanopyGeometry("circular", {"r": 5})
        cd_sub  = cg.calculate_drag_coefficient(0.2)
        cd_tran = cg.calculate_drag_coefficient(1.0)
        # Transonic peak should be >= subsonic
        assert cd_tran >= cd_sub * 0.8   # allow some slack

    def test_nominal_diameter_circular(self):
        cg = CanopyGeometry("circular", {"r": 5})
        assert abs(cg.nominal_diameter() - 10.0) < 0.01

    def test_drag_force_positive(self):
        cg = CanopyGeometry("elliptical", {"a": 10, "b": 5})
        F = cg.drag_force(500, 0.02, mach_number=1.5)
        assert F > 0

    def test_summary_keys(self):
        cg = CanopyGeometry("elliptical", {"a": 8, "b": 4})
        s = cg.summary()
        for k in ["shape","area_m2","nominal_diam_m","Cd_subsonic","Cd_mach1","Cd_mach2"]:
            assert k in s


# ══════════════════════════════════════════════════════════════════════════════
# LBM (smoke tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestLBMSolver:

    def test_init_and_solve(self):
        from src.lbm_solver import LBMSolver
        solver = LBMSolver((16, 32), reynolds=50)
        solver.initialize()
        result = solver.solve(steps=100, verbose=False)
        assert "ux" in result and "uy" in result and "rho" in result

    def test_vorticity_shape(self):
        from src.lbm_solver import LBMSolver
        Ny, Nx = 16, 32
        solver = LBMSolver((Ny, Nx), reynolds=50)
        solver.initialize()
        result = solver.solve(steps=100, verbose=False)
        assert result["vorticity"].shape == (Ny, Nx)

    def test_geometry_mask(self):
        from src.lbm_solver import LBMSolver
        solver = LBMSolver((16, 32), reynolds=50)
        solver.initialize()
        mask = np.zeros((16, 32), dtype=bool)
        mask[6:10, 12:20] = True
        solver.set_geometry(mask)
        result = solver.solve(steps=100, verbose=False, flow_type="channel")
        assert result["Cd"] is not None

    def test_lid_driven(self):
        from src.lbm_solver import LBMSolver
        solver = LBMSolver((16, 16), reynolds=30)
        solver.initialize()
        result = solver.solve(steps=100, flow_type="lid_driven", verbose=False)
        assert "ux" in result


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-FIDELITY PINN (smoke tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiFidelityPINN:

    def _make_data(self):
        from src.multifidelity_pinn import LowFidelityEDL
        planet = MarsAtmosphere()
        lf = LowFidelityEDL(planet, 900, 1.7, 78.5)
        t = np.linspace(0, 200, 40)
        v, h = lf.solve(t, 5800, 125000)
        return t, v, h, lf

    def test_lf_model_runs(self):
        t, v, h, _ = self._make_data()
        assert len(v) == len(t)
        assert v[0] > v[-1]      # decelerates
        assert (h >= 0).all()

    def test_mfpinn_predict_without_training(self):
        from src.multifidelity_pinn import MultiFidelityPINN
        t, v, h, lf = self._make_data()
        hf = {"t": t, "v": v, "h": h}
        mf = MultiFidelityPINN(lf, hf, layers=[1, 16, 2])
        pred = mf.predict(t, v[0], h[0])
        assert "v_mf" in pred and "h_mf" in pred
        assert len(pred["v_mf"]) == len(t)

    def test_mfpinn_train_scipy(self):
        from src.multifidelity_pinn import MultiFidelityPINN
        t, v, h, lf = self._make_data()
        noise = np.random.default_rng(0).normal(0, 20, len(t))
        hf = {"t": t, "v": np.clip(v + noise, 0, None), "h": h}
        mf = MultiFidelityPINN(lf, hf, layers=[1, 16, 2])
        mf.train(epochs=50, verbose=False)
        assert mf._trained


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL OPERATOR (smoke tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestNeuralOperator:

    def test_fno_train_predict(self):
        from src.neural_operator import NeuralOperator
        op = NeuralOperator("fno", n_in=1, n_out=1, modes=4, width=8)
        x = np.random.rand(20, 1).astype(np.float32)
        y = np.sin(x * np.pi)
        op.train(x, y, epochs=20, verbose=False)
        pred = op.predict(x)
        assert pred is not None

    def test_deeponet_train_predict(self):
        from src.neural_operator import NeuralOperator
        op = NeuralOperator("deeponet", n_in=1, n_out=1, m_sensors=10, p_dim=8)
        x = np.random.rand(20, 1).astype(np.float32)
        y = np.cos(x * np.pi)
        op.train(x, y, epochs=20, verbose=False)
        pred = op.predict(x)
        assert pred is not None

    def test_invalid_type(self):
        from src.neural_operator import NeuralOperator
        with pytest.raises(ValueError):
            NeuralOperator("transformer")


# ══════════════════════════════════════════════════════════════════════════════
# OPERATOR DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TestOperatorDataset:

    def test_generate_small(self):
        from src.operator_dataset import OperatorDataset
        ds = OperatorDataset(output_resolution=20)
        data = ds.generate(n_samples=10, verbose=False)
        assert "inputs" in data and "outputs" in data
        assert data["inputs"].shape[1] == 20

    def test_train_test_split(self):
        from src.operator_dataset import OperatorDataset
        ds = OperatorDataset(output_resolution=20)
        data = ds.generate(n_samples=20, verbose=False)
        tr, te = ds.train_test_split(data, test_frac=0.2)
        n = len(tr["params"]) + len(te["params"])
        assert n == data["n_valid"]

    def test_save_load(self, tmp_path):
        from src.operator_dataset import OperatorDataset
        ds = OperatorDataset(output_resolution=20)
        data = ds.generate(n_samples=10, verbose=False)
        path = tmp_path / "test_ds.npz"
        ds.save(data, path)
        loaded = OperatorDataset.load(path)
        assert "inputs" in loaded


# ══════════════════════════════════════════════════════════════════════════════
# REAL-GAS CHEMISTRY  (Park 1993)
# ══════════════════════════════════════════════════════════════════════════════

class TestRealGasChemistry:

    def test_equilibrium_frozen_at_low_T(self):
        from src.realgas_chemistry import equilibrium_composition
        X = equilibrium_composition(1000.0, 1e5)
        assert X["CO2"] > 0.95, "At 1000K CO2 should be nearly undissociated"

    def test_equilibrium_dissociated_at_high_T(self):
        from src.realgas_chemistry import equilibrium_composition
        X = equilibrium_composition(5000.0, 1000.0)
        assert X["CO2"] < 0.20, "At 5000K CO2 should be heavily dissociated"

    def test_equilibrium_mole_fractions_sum_to_one(self):
        from src.realgas_chemistry import equilibrium_composition
        for T in [1000, 3000, 6000]:
            X = equilibrium_composition(T, 1e4)
            total = sum(X.values())
            assert abs(total - 1.0) < 0.02, f"Mole fractions sum={total:.4f} at T={T}"

    def test_mean_molecular_weight_decreases_with_dissociation(self):
        from src.realgas_chemistry import equilibrium_composition, SPECIES, SPECIES_ORDER
        X1 = equilibrium_composition(1000, 1e5)
        X2 = equilibrium_composition(5000, 1000)
        M1 = sum(X1.get(sp, 0) * SPECIES[sp].M_kgmol for sp in SPECIES_ORDER)
        M2 = sum(X2.get(sp, 0) * SPECIES[sp].M_kgmol for sp in SPECIES_ORDER)
        assert M2 < M1, f"M_mix should decrease with dissociation: M1={M1:.4f} M2={M2:.4f}"

    def test_gamma_always_above_one(self):
        from src.realgas_chemistry import equilibrium_composition, mixture_gamma
        for T in [1000, 2000, 4000, 8000]:
            X = equilibrium_composition(T, 1000)
            assert mixture_gamma(X, T) > 1.0

    def test_viscosity_positive(self):
        from src.realgas_chemistry import mixture_viscosity
        X = {"CO2":0.9,"CO":0.05,"O":0.05,"O2":0.0,"C":0.0}
        assert mixture_viscosity(X, 3000) > 0

    def test_conductivity_positive(self):
        from src.realgas_chemistry import mixture_conductivity
        X = {"CO2":0.9,"CO":0.05,"O":0.05,"O2":0.0,"C":0.0}
        assert mixture_conductivity(X, 3000) > 0

    def test_fay_riddell_q_positive(self):
        from src.realgas_chemistry import fay_riddell_heating
        from src.planetary_atm import MarsAtmosphere
        m = MarsAtmosphere()
        res = fay_riddell_heating(m.density(60000), 6000,
                                   m.temperature(60000), m.pressure(60000),
                                   1.0, planet="mars")
        assert res["q_rg_Wm2"] > 0

    def test_fay_riddell_stagnation_hotter(self):
        from src.realgas_chemistry import fay_riddell_heating
        from src.planetary_atm import MarsAtmosphere
        m = MarsAtmosphere()
        res = fay_riddell_heating(m.density(60000), 6000,
                                   m.temperature(60000), m.pressure(60000), 1.0)
        assert res["T_stag_K"] > m.temperature(60000)

    def test_fay_riddell_mach_supersonic(self):
        from src.realgas_chemistry import fay_riddell_heating
        from src.planetary_atm import MarsAtmosphere
        m = MarsAtmosphere()
        res = fay_riddell_heating(m.density(80000), 5000,
                                   m.temperature(80000), m.pressure(80000), 1.0)
        assert res["Mach_freestream"] > 1.0

    def test_profile_shapes(self):
        from src.realgas_chemistry import realgas_trajectory_profile
        from src.planetary_atm import MarsAtmosphere
        import numpy as np
        m = MarsAtmosphere()
        v = np.linspace(5800, 400, 20)
        h = np.linspace(125000, 5000, 20)
        prof = realgas_trajectory_profile(v, h, m, planet_name="mars")
        for key in ["q_rg_Wm2","q_sg_Wm2","gamma_eff","X_CO2","dissociation_CO2"]:
            assert len(prof[key]) == 20
        assert (prof["q_rg_Wm2"] >= 0).all()
        assert prof["gamma_eff"].min() > 1.0
        assert prof["dissociation_CO2"].min() >= 0.0
        assert prof["dissociation_CO2"].max() <= 1.001


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO EDL
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarlEDL:

    def test_lhs_samples_shape(self):
        from src.monte_carlo_edl import _lhs_samples
        samp = _lhs_samples(50, seed=0)
        assert len(samp) == 11
        for v in samp.values():
            assert len(v) == 50

    def test_lhs_rho_factor_positive(self):
        from src.monte_carlo_edl import _lhs_samples
        samp = _lhs_samples(100, seed=1)
        assert (samp["rho0_factor"] > 0).all()

    def test_lhs_wind_bidirectional(self):
        from src.monte_carlo_edl import _lhs_samples
        samp = _lhs_samples(100, seed=2)
        assert samp["wind_ew_ms"].min() < 0 and samp["wind_ew_ms"].max() > 0

    def test_mc_run_returns_df(self):
        from src.monte_carlo_edl import MonteCarloEDL
        from src.planetary_atm import MarsAtmosphere
        m  = MarsAtmosphere()
        mc = MonteCarloEDL(n_samples=20, use_realgas=False, seed=0)
        df = mc.run(m,"nylon",0.015,"elliptical",{"a":10,"b":5},
                    900,125000,5800,15.0,4.5,verbose=False)
        assert len(df) > 0
        assert "v_land_ms" in df.columns

    def test_mc_v_land_positive(self):
        from src.monte_carlo_edl import MonteCarloEDL
        from src.planetary_atm import MarsAtmosphere
        m  = MarsAtmosphere()
        mc = MonteCarloEDL(n_samples=20, use_realgas=False, seed=1)
        df = mc.run(m,"nylon",0.015,"elliptical",{"a":10,"b":5},
                    900,125000,5800,15.0,4.5,verbose=False)
        assert (df["v_land_ms"] >= 0).all()

    def test_mc_summary_keys(self):
        from src.monte_carlo_edl import MonteCarloEDL
        from src.planetary_atm import MarsAtmosphere
        m  = MarsAtmosphere()
        mc = MonteCarloEDL(n_samples=20, use_realgas=False, seed=2)
        mc.run(m,"nylon",0.015,"elliptical",{"a":10,"b":5},
               900,125000,5800,15.0,4.5,verbose=False)
        s = mc.summary()
        for k in ["mission","v_land","landing_ellipse","sf_tps","q_peak_gumbel"]:
            assert k in s

    def test_mc_p_success_range(self):
        from src.monte_carlo_edl import MonteCarloEDL
        from src.planetary_atm import MarsAtmosphere
        m  = MarsAtmosphere()
        mc = MonteCarloEDL(n_samples=20, use_realgas=False, seed=3)
        mc.run(m,"nylon",0.015,"elliptical",{"a":10,"b":5},
               900,125000,5800,15.0,4.5,verbose=False)
        s = mc.summary()
        p = s["mission"]["P_mission_success"]
        assert 0.0 <= p <= 1.0

    def test_mc_cep_positive(self):
        from src.monte_carlo_edl import MonteCarloEDL
        from src.planetary_atm import MarsAtmosphere
        m  = MarsAtmosphere()
        mc = MonteCarloEDL(n_samples=20, use_realgas=False, seed=4)
        mc.run(m,"nylon",0.015,"elliptical",{"a":10,"b":5},
               900,125000,5800,15.0,4.5,verbose=False)
        s = mc.summary()
        assert s["landing_ellipse"]["CEP_90_m"] > 0

    def test_mc_percentile_order(self):
        from src.monte_carlo_edl import MonteCarloEDL
        from src.planetary_atm import MarsAtmosphere
        m  = MarsAtmosphere()
        mc = MonteCarloEDL(n_samples=30, use_realgas=False, seed=5)
        mc.run(m,"nylon",0.015,"elliptical",{"a":10,"b":5},
               900,125000,5800,15.0,4.5,verbose=False)
        s = mc.summary()["v_land"]
        assert s["p05"] <= s["p50"] <= s["p95"]


# ══════════════════════════════════════════════════════════════════════════════
# 3-D VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

class TestVisualization3D:

    def _make_traj(self):
        from src.planetary_atm import MarsAtmosphere
        from src.multifidelity_pinn import LowFidelityEDL
        import numpy as np
        m  = MarsAtmosphere()
        lf = LowFidelityEDL(m, 900, 1.7, 78.5, gamma_deg=15)
        t  = np.linspace(0, 300, 80)
        v, h = lf.solve(t, 5800, 125_000)
        q = np.abs(1e-5 * np.maximum(v, 0)**2.5) + 0.001
        return t, v, h, q

    def test_trajectory_cartesian_shape(self):
        from src.visualization_3d import trajectory_to_cartesian
        import numpy as np
        t, v, h, q = self._make_traj()
        traj = trajectory_to_cartesian(t, v, h, "mars")
        assert len(traj["x"]) == len(t)
        assert len(traj["y"]) == len(t)
        assert len(traj["z"]) == len(t)

    def test_trajectory_r_above_planet(self):
        from src.visualization_3d import trajectory_to_cartesian
        import numpy as np
        t, v, h, q = self._make_traj()
        traj = trajectory_to_cartesian(t, v, h, "mars")
        assert (traj["r"] >= traj["R_planet"] * 0.999).all()

    def test_planet_radii(self):
        from src.visualization_3d import _planet_radius
        assert _planet_radius("mars")  > 3_000_000
        assert _planet_radius("venus") > 6_000_000
        assert _planet_radius("titan") > 2_000_000

    def test_visualize_creates_file(self, tmp_path):
        from src.visualization_3d import visualize
        import matplotlib; matplotlib.use("Agg")
        t, v, h, q = self._make_traj()
        out = visualize(t, v, h, q, planet_name="mars",
                        title="Test", save_dir=tmp_path)
        from pathlib import Path
        assert Path(out["matplotlib_path"]).exists()
        assert Path(out["matplotlib_path"]).stat().st_size > 50_000

    def test_visualize_with_mc(self, tmp_path):
        from src.visualization_3d import visualize
        from src.monte_carlo_edl import MonteCarloEDL
        from src.planetary_atm import MarsAtmosphere
        import matplotlib; matplotlib.use("Agg")
        m  = MarsAtmosphere()
        mc = MonteCarloEDL(n_samples=15, use_realgas=False, seed=0)
        df = mc.run(m,"nylon",0.015,"elliptical",{"a":10,"b":5},
                    900,125000,5800,15.0,4.5,verbose=False)
        t, v, h, q = self._make_traj()
        out = visualize(t, v, h, q, planet_name="mars", mc_df=df,
                        save_dir=tmp_path)
        from pathlib import Path
        assert Path(out["matplotlib_path"]).exists()

    def test_pyvista_flag(self):
        from src.visualization_3d import _PYVISTA
        assert isinstance(_PYVISTA, bool)
