"""Tests for EngineConfig — typed parameter extraction from flat config dicts."""

import pytest

from osmose.engine.config import EngineConfig


@pytest.fixture
def minimal_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "10",
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


class TestEngineConfig:
    def test_from_dict_basic(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.n_species == 2
        assert cfg.n_dt_per_year == 24
        assert cfg.n_year == 10
        assert cfg.n_steps == 240

    def test_species_names(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.species_names == ["Anchovy", "Sardine"]

    def test_growth_params_arrays(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.linf[0] == pytest.approx(15.0)
        assert cfg.linf[1] == pytest.approx(25.0)
        assert cfg.k[0] == pytest.approx(0.4)
        assert cfg.t0[1] == pytest.approx(-0.2)

    def test_lifespan_in_dt(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.lifespan_dt[0] == 3 * 24
        assert cfg.lifespan_dt[1] == 5 * 24

    def test_mortality_subdt(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.mortality_subdt == 10

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            EngineConfig.from_dict({})
