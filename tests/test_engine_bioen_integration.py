"""Tests for bioenergetic config parsing and schema expansion."""
import numpy as np
import pytest

from osmose.engine.config import EngineConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config() -> dict[str, str]:
    """Minimal config that satisfies all required EngineConfig keys."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "5",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "20",
        "simulation.nschool.sp1": "15",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        "species.linf.sp0": "15.0",
        "species.linf.sp1": "25.0",
        "species.k.sp0": "0.4",
        "species.k.sp1": "0.3",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.15",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.008",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.1",
        "species.lifespan.sp0": "3",
        "species.lifespan.sp1": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.0",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
    }


@pytest.fixture
def bioen_config(minimal_config) -> dict[str, str]:
    """Minimal config with all bioenergetic keys."""
    cfg = minimal_config.copy()
    cfg.update({
        "simulation.bioen.enabled": "true",
        "simulation.bioen.phit.enabled": "true",
        "simulation.bioen.fo2.enabled": "false",
        # Per-species bioen params for sp0 and sp1
        "species.beta.sp0": "0.75",
        "species.beta.sp1": "0.80",
        "species.zlayer.sp0": "0",
        "species.zlayer.sp1": "1",
        "species.bioen.assimilation.sp0": "0.68",
        "species.bioen.assimilation.sp1": "0.72",
        "species.bioen.maint.energy.c_m.sp0": "0.00123",
        "species.bioen.maint.energy.c_m.sp1": "0.00098",
        "species.bioen.maturity.eta.sp0": "1.4",
        "species.bioen.maturity.eta.sp1": "1.6",
        "species.bioen.maturity.r.sp0": "0.45",
        "species.bioen.maturity.r.sp1": "0.50",
        "species.bioen.maturity.m0.sp0": "4.5",
        "species.bioen.maturity.m0.sp1": "6.0",
        "species.bioen.maturity.m1.sp0": "1.8",
        "species.bioen.maturity.m1.sp1": "2.1",
        "species.bioen.mobilized.e.mobi.sp0": "0.62",
        "species.bioen.mobilized.e.mobi.sp1": "0.58",
        "species.bioen.mobilized.e.D.sp0": "1.45",
        "species.bioen.mobilized.e.D.sp1": "1.55",
        "species.bioen.mobilized.Tp.sp0": "18.0",
        "species.bioen.mobilized.Tp.sp1": "22.0",
        "species.bioen.maint.e.maint.sp0": "0.63",
        "species.bioen.maint.e.maint.sp1": "0.67",
        "species.oxygen.c1.sp0": "0.95",
        "species.oxygen.c1.sp1": "0.90",
        "species.oxygen.c2.sp0": "2.5",
        "species.oxygen.c2.sp1": "3.0",
        "predation.ingestion.rate.max.bioen.sp0": "4.2",
        "predation.ingestion.rate.max.bioen.sp1": "3.8",
        "predation.coef.ingestion.rate.max.larvae.bioen.sp0": "1.1",
        "predation.coef.ingestion.rate.max.larvae.bioen.sp1": "1.05",
        "predation.c.bioen.sp0": "0.01",
        "predation.c.bioen.sp1": "0.008",
        "species.bioen.forage.k_for.sp0": "0.002",
        "species.bioen.forage.k_for.sp1": "0.0015",
    })
    return cfg


# ── Tests: bioen disabled ─────────────────────────────────────────────────────

class TestBioenDisabled:
    def test_bioen_disabled_by_default(self, minimal_config):
        """bioen_enabled is False when key is absent."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.bioen_enabled is False

    def test_bioen_params_none_when_disabled(self, minimal_config):
        """All bioen parameter arrays are None when bioen is disabled."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.bioen_beta is None
        assert cfg.bioen_assimilation is None
        assert cfg.bioen_c_m is None
        assert cfg.bioen_eta is None
        assert cfg.bioen_r is None
        assert cfg.bioen_m0 is None
        assert cfg.bioen_m1 is None
        assert cfg.bioen_e_mobi is None
        assert cfg.bioen_e_d is None
        assert cfg.bioen_tp is None
        assert cfg.bioen_e_maint is None
        assert cfg.bioen_o2_c1 is None
        assert cfg.bioen_o2_c2 is None
        assert cfg.bioen_i_max is None
        assert cfg.bioen_theta is None
        assert cfg.bioen_c_rate is None
        assert cfg.bioen_k_for is None

    def test_explicit_bioen_disabled(self, minimal_config):
        """Explicit simulation.bioen.enabled=false keeps all params None."""
        minimal_config["simulation.bioen.enabled"] = "false"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.bioen_enabled is False
        assert cfg.bioen_beta is None


# ── Tests: bioen enabled ──────────────────────────────────────────────────────

class TestBioenEnabled:
    def test_bioen_enabled_flag(self, bioen_config):
        """bioen_enabled=True when key is set."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_enabled is True

    def test_bioen_phit_fo2_flags(self, bioen_config):
        """phit_enabled and fo2_enabled parsed correctly."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_phit_enabled is True
        assert cfg.bioen_fo2_enabled is False  # explicitly set to false

    def test_bioen_beta_parsed(self, bioen_config):
        """Beta exponent parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_beta is not None
        assert cfg.bioen_beta[0] == pytest.approx(0.75)
        assert cfg.bioen_beta[1] == pytest.approx(0.80)

    def test_bioen_zlayer_parsed(self, bioen_config):
        """Depth layer index parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_zlayer is not None
        assert cfg.bioen_zlayer[0] == 0
        assert cfg.bioen_zlayer[1] == 1

    def test_bioen_assimilation_parsed(self, bioen_config):
        """Assimilation efficiency parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_assimilation is not None
        assert cfg.bioen_assimilation[0] == pytest.approx(0.68)
        assert cfg.bioen_assimilation[1] == pytest.approx(0.72)

    def test_bioen_maintenance_parsed(self, bioen_config):
        """Maintenance coefficient parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_c_m is not None
        assert cfg.bioen_c_m[0] == pytest.approx(0.00123)
        assert cfg.bioen_c_m[1] == pytest.approx(0.00098)

    def test_bioen_maturity_params_parsed(self, bioen_config):
        """LMRN and energy allocation params parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_eta[0] == pytest.approx(1.4)
        assert cfg.bioen_r[0] == pytest.approx(0.45)
        assert cfg.bioen_m0[0] == pytest.approx(4.5)
        assert cfg.bioen_m1[0] == pytest.approx(1.8)

    def test_bioen_thermal_params_parsed(self, bioen_config):
        """Johnson curve parameters parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_e_mobi[0] == pytest.approx(0.62)
        assert cfg.bioen_e_d[0] == pytest.approx(1.45)
        assert cfg.bioen_tp[0] == pytest.approx(18.0)
        assert cfg.bioen_e_maint[0] == pytest.approx(0.63)

    def test_bioen_oxygen_params_parsed(self, bioen_config):
        """Oxygen dose-response parameters parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_o2_c1[0] == pytest.approx(0.95)
        assert cfg.bioen_o2_c2[0] == pytest.approx(2.5)
        assert cfg.bioen_o2_c1[1] == pytest.approx(0.90)
        assert cfg.bioen_o2_c2[1] == pytest.approx(3.0)

    def test_bioen_predation_params_parsed(self, bioen_config):
        """Bioen-specific predation parameters parsed per species."""
        cfg = EngineConfig.from_dict(bioen_config)
        assert cfg.bioen_i_max[0] == pytest.approx(4.2)
        assert cfg.bioen_theta[0] == pytest.approx(1.1)
        assert cfg.bioen_c_rate[0] == pytest.approx(0.01)
        assert cfg.bioen_k_for[0] == pytest.approx(0.002)

    def test_bioen_arrays_have_correct_length(self, bioen_config):
        """All bioen arrays have length n_species."""
        cfg = EngineConfig.from_dict(bioen_config)
        n = cfg.n_species
        for attr in (
            "bioen_beta", "bioen_zlayer", "bioen_assimilation", "bioen_c_m",
            "bioen_eta", "bioen_r", "bioen_m0", "bioen_m1",
            "bioen_e_mobi", "bioen_e_d", "bioen_tp", "bioen_e_maint",
            "bioen_o2_c1", "bioen_o2_c2", "bioen_i_max",
            "bioen_theta", "bioen_c_rate", "bioen_k_for",
        ):
            arr = getattr(cfg, attr)
            assert arr is not None, f"{attr} should not be None when bioen enabled"
            assert len(arr) == n, f"{attr} length {len(arr)} != n_species {n}"

    def test_bioen_defaults_when_keys_absent(self, minimal_config):
        """Optional bioen keys fall back to defaults when absent."""
        minimal_config["simulation.bioen.enabled"] = "true"
        cfg = EngineConfig.from_dict(minimal_config)
        # Defaults are applied: beta=0.8, theta=1.0, c_rate=0.0, k_for=0.0
        assert cfg.bioen_beta[0] == pytest.approx(0.8)
        assert cfg.bioen_theta[0] == pytest.approx(1.0)
        assert cfg.bioen_c_rate[0] == pytest.approx(0.0)
        assert cfg.bioen_k_for[0] == pytest.approx(0.0)
