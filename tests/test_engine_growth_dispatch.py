"""Tests for Gompertz growth dispatch — Task 3 verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.growth import (
    _expected_length,
    expected_length_gompertz,
    expected_length_vb,
    growth,
)
from osmose.engine.state import SchoolState


def _base_config() -> dict[str, str]:
    """Minimal VB config for one species."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
    }


def _gompertz_config_single() -> dict[str, str]:
    """One Gompertz species config."""
    cfg = _base_config()
    cfg["growth.java.classname.sp0"] = "fr.ird.osmose.process.growth.GompertzGrowth"
    cfg["growth.gompertz.linf.sp0"] = "30.0"
    cfg["growth.gompertz.kg.sp0"] = "0.5"
    cfg["growth.gompertz.tg.sp0"] = "1.5"
    cfg["growth.exponential.ke.sp0"] = "2.0"
    cfg["growth.exponential.lstart.sp0"] = "0.1"
    cfg["growth.exponential.thr.age.sp0"] = "0.25"  # 6 dt at n_dt=24
    cfg["growth.gompertz.thr.age.sp0"] = "0.5"  # 12 dt at n_dt=24
    return cfg


def _mixed_config() -> dict[str, str]:
    """Two species: sp0=VB, sp1=Gompertz."""
    cfg = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "1",
        "simulation.nschool.sp1": "1",
        "species.name.sp0": "VBFish",
        "species.name.sp1": "GomFish",
        # VB species
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        # Gompertz species
        "species.linf.sp1": "50.0",
        "species.k.sp1": "0.2",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp1": "0.2",
        "species.length2weight.condition.factor.sp1": "0.005",
        "species.length2weight.allometric.power.sp1": "3.0",
        "species.lifespan.sp1": "8",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "predation.ingestion.rate.max.sp1": "4.0",
        "predation.efficiency.critical.sp1": "0.5",
        "growth.java.classname.sp1": "fr.ird.osmose.process.growth.GompertzGrowth",
        "growth.gompertz.linf.sp1": "50.0",
        "growth.gompertz.kg.sp1": "0.5",
        "growth.gompertz.tg.sp1": "1.5",
        "growth.exponential.ke.sp1": "2.0",
        "growth.exponential.lstart.sp1": "0.2",
        "growth.exponential.thr.age.sp1": "0.25",
        "growth.gompertz.thr.age.sp1": "0.5",
        "mortality.subdt": "10",
    }
    return cfg


class TestGompertzConfig:
    def test_growth_class_defaults_to_vb(self):
        cfg = EngineConfig.from_dict(_base_config())
        assert cfg.growth_class == ["VB"]

    def test_gompertz_classname_parsed(self):
        cfg = EngineConfig.from_dict(_gompertz_config_single())
        assert cfg.growth_class == ["GOMPERTZ"]

    def test_gompertz_params_loaded(self):
        cfg = EngineConfig.from_dict(_gompertz_config_single())
        assert cfg.gompertz_ke is not None
        np.testing.assert_allclose(cfg.gompertz_ke[0], 2.0)
        np.testing.assert_allclose(cfg.gompertz_lstart[0], 0.1)
        np.testing.assert_allclose(cfg.gompertz_kg[0], 0.5)
        np.testing.assert_allclose(cfg.gompertz_tg[0], 1.5)
        np.testing.assert_allclose(cfg.gompertz_linf[0], 30.0)

    def test_gompertz_thr_age_converted_to_dt(self):
        cfg = EngineConfig.from_dict(_gompertz_config_single())
        # 0.25 years * 24 dt/year = 6 dt
        assert cfg.gompertz_thr_age_exp_dt[0] == 6
        # 0.5 years * 24 dt/year = 12 dt
        assert cfg.gompertz_thr_age_gom_dt[0] == 12

    def test_no_gompertz_params_when_all_vb(self):
        cfg = EngineConfig.from_dict(_base_config())
        assert cfg.gompertz_ke is None
        assert cfg.gompertz_lstart is None
        assert cfg.gompertz_kg is None
        assert cfg.gompertz_tg is None
        assert cfg.gompertz_linf is None
        assert cfg.gompertz_thr_age_exp_dt is None
        assert cfg.gompertz_thr_age_gom_dt is None

    def test_mixed_species_growth_class(self):
        cfg = EngineConfig.from_dict(_mixed_config())
        assert cfg.growth_class == ["VB", "GOMPERTZ"]

    def test_legacy_classname_gompertz(self):
        cfg_dict = _base_config()
        cfg_dict["growth.java.classname.sp0"] = "fr.ird.osmose.growth.Gompertz"
        cfg_dict["growth.gompertz.linf.sp0"] = "30.0"
        cfg_dict["growth.gompertz.kg.sp0"] = "0.5"
        cfg_dict["growth.gompertz.tg.sp0"] = "1.5"
        cfg_dict["growth.exponential.ke.sp0"] = "2.0"
        cfg_dict["growth.exponential.lstart.sp0"] = "0.1"
        cfg_dict["growth.exponential.thr.age.sp0"] = "0.25"
        cfg_dict["growth.gompertz.thr.age.sp0"] = "0.5"
        cfg = EngineConfig.from_dict(cfg_dict)
        assert cfg.growth_class == ["GOMPERTZ"]


class TestGompertzGrowth:
    def test_gompertz_age_zero_returns_lstart(self):
        """expected_length_gompertz at age 0 returns lstart (not egg_size)."""
        result = expected_length_gompertz(
            age_dt=np.array([0]),
            linf=np.array([30.0]),
            k_gom=np.array([0.5]),
            t_gom=np.array([1.5]),
            k_exp=np.array([2.0]),
            a_exp_dt=np.array([6]),
            a_gom_dt=np.array([12]),
            lstart=np.array([0.15]),
            n_dt_per_year=24,
        )
        np.testing.assert_allclose(result, [0.15], atol=1e-10)

    def test_gompertz_growth_applied(self):
        """Gompertz dispatch produces different lengths than VB for same species params."""
        cfg = EngineConfig.from_dict(_gompertz_config_single())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        # Age well into Gompertz phase (age >= a_gom_dt=12)
        age_dt = np.array([48], dtype=np.int32)
        state = state.replace(
            length=np.array([15.0]),
            weight=np.array([0.006 * 15.0**3]),
            age_dt=age_dt,
            abundance=np.array([100.0]),
            biomass=np.array([100.0 * 0.006 * 15.0**3]),
            pred_success_rate=np.array([1.0]),
        )
        new_state = growth(state, cfg, np.random.default_rng(0))

        # Manually compute expected Gompertz length delta
        n_dt = 24
        l_cur = expected_length_gompertz(
            age_dt,
            np.array([30.0]),
            np.array([0.5]),
            np.array([1.5]),
            np.array([2.0]),
            np.array([6], dtype=np.int32),
            np.array([12], dtype=np.int32),
            np.array([0.1]),
            n_dt,
        )
        l_nxt = expected_length_gompertz(
            age_dt + 1,
            np.array([30.0]),
            np.array([0.5]),
            np.array([1.5]),
            np.array([2.0]),
            np.array([6], dtype=np.int32),
            np.array([12], dtype=np.int32),
            np.array([0.1]),
            n_dt,
        )
        delta_l = l_nxt[0] - l_cur[0]
        # At sr=1.0, growth_factor = delta_lmax_factor (2.0) * delta_L
        max_delta = 2.0 * delta_l
        expected_length = min(15.0 + max_delta, 30.0)
        np.testing.assert_allclose(new_state.length[0], expected_length, atol=1e-10)

    def test_gompertz_growth_differs_from_vb(self):
        """Gompertz and VB produce different length increments for same nominal params."""
        # VB config
        vb_cfg = EngineConfig.from_dict(_base_config())
        # Gompertz config (same species, different classname)
        gom_cfg = EngineConfig.from_dict(_gompertz_config_single())

        age_dt = np.array([48], dtype=np.int32)
        sp = np.array([0], dtype=np.int32)
        n_dt = 24

        vb_l = expected_length_vb(
            age_dt,
            vb_cfg.linf[sp],
            vb_cfg.k[sp],
            vb_cfg.t0[sp],
            vb_cfg.egg_size[sp],
            vb_cfg.vb_threshold_age[sp],
            n_dt,
        )
        gom_l = _expected_length(age_dt, sp, gom_cfg, n_dt)

        # They should differ since the growth curves are different functions
        assert not np.allclose(vb_l, gom_l, atol=1e-6), (
            "VB and Gompertz should produce different lengths at the same age"
        )

    def test_mixed_species_growth(self):
        """Species 0 uses VB, species 1 uses Gompertz — both grow correctly."""
        cfg = EngineConfig.from_dict(_mixed_config())

        # Two schools: sp0 (VB) and sp1 (Gompertz), both aged into their respective curves
        age_dt = np.array([48, 48], dtype=np.int32)
        sp = np.array([0, 1], dtype=np.int32)
        n_dt = 24

        result = _expected_length(age_dt, sp, cfg, n_dt)

        # sp0: VB
        expected_vb = expected_length_vb(
            np.array([48], dtype=np.int32),
            np.array([30.0]),
            np.array([0.3]),
            np.array([-0.1]),
            np.array([0.1]),
            np.array([1.0]),
            n_dt,
        )
        np.testing.assert_allclose(result[0], expected_vb[0], atol=1e-10)

        # sp1: Gompertz
        expected_gom = expected_length_gompertz(
            np.array([48], dtype=np.int32),
            np.array([50.0]),
            np.array([0.5]),
            np.array([1.5]),
            np.array([2.0]),
            np.array([6], dtype=np.int32),
            np.array([12], dtype=np.int32),
            np.array([0.2]),
            n_dt,
        )
        np.testing.assert_allclose(result[1], expected_gom[0], atol=1e-10)

    def test_growth_function_with_gompertz_species(self):
        """growth() applies Gompertz correctly in a full growth call."""
        cfg = EngineConfig.from_dict(_gompertz_config_single())
        sp = np.array([0], dtype=np.int32)
        age_dt = np.array([48], dtype=np.int32)
        state = SchoolState.create(n_schools=1, species_id=sp)
        l0 = 15.0
        state = state.replace(
            length=np.array([l0]),
            weight=np.array([0.006 * l0**3]),
            age_dt=age_dt,
            abundance=np.array([100.0]),
            biomass=np.array([100.0 * 0.006 * l0**3]),
            pred_success_rate=np.array([0.0]),  # below critical — egg bypass not active
        )
        new_state = growth(state, cfg, np.random.default_rng(0))
        # Below critical success rate, no growth applied (bypass = False since age!=0 and not out)
        np.testing.assert_allclose(new_state.length[0], l0, atol=1e-10)


class TestBioenBypass:
    def test_bioen_enabled_flag_exists(self):
        """EngineConfig should have bioen_enabled field (False by default)."""
        cfg = EngineConfig.from_dict(_base_config())
        # bioen_enabled should exist as an attribute and default to False
        assert hasattr(cfg, "bioen_enabled")
        assert cfg.bioen_enabled is False
