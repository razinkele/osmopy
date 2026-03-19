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

    def test_delta_lmax_factor(self, minimal_config):
        minimal_config["species.delta.lmax.factor.sp0"] = "2.0"
        minimal_config["species.delta.lmax.factor.sp1"] = "1.8"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.delta_lmax_factor[0] == pytest.approx(2.0)
        assert cfg.delta_lmax_factor[1] == pytest.approx(1.8)

    def test_delta_lmax_factor_default(self, minimal_config):
        """delta_lmax_factor defaults to 2.0 when not specified."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.delta_lmax_factor[0] == pytest.approx(2.0)
        assert cfg.delta_lmax_factor[1] == pytest.approx(2.0)

    def test_additional_mortality_rate(self, minimal_config):
        minimal_config["mortality.additional.rate.sp0"] = "0.2"
        minimal_config["mortality.additional.rate.sp1"] = "0.15"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.additional_mortality_rate[0] == pytest.approx(0.2)
        assert cfg.additional_mortality_rate[1] == pytest.approx(0.15)

    def test_sex_ratio_default(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.sex_ratio[0] == pytest.approx(0.5)

    def test_relative_fecundity(self, minimal_config):
        minimal_config["species.relativefecundity.sp0"] = "800"
        minimal_config["species.relativefecundity.sp1"] = "200"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.relative_fecundity[0] == pytest.approx(800.0)
        assert cfg.relative_fecundity[1] == pytest.approx(200.0)

    def test_maturity_size(self, minimal_config):
        minimal_config["species.maturity.size.sp0"] = "12.0"
        minimal_config["species.maturity.size.sp1"] = "40.0"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.maturity_size[0] == pytest.approx(12.0)
        assert cfg.maturity_size[1] == pytest.approx(40.0)

    def test_movement_method(self, minimal_config):
        minimal_config["movement.distribution.method.sp0"] = "random"
        minimal_config["movement.distribution.method.sp1"] = "maps"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.movement_method == ["random", "maps"]

    def test_random_walk_range(self, minimal_config):
        minimal_config["movement.randomwalk.range.sp0"] = "1"
        minimal_config["movement.randomwalk.range.sp1"] = "2"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.random_walk_range[0] == 1
        assert cfg.random_walk_range[1] == 2

    def test_size_ratio_params(self, minimal_config):
        minimal_config["predation.predprey.sizeratio.min.sp0"] = "3.5"
        minimal_config["predation.predprey.sizeratio.min.sp1"] = "2.0"
        minimal_config["predation.predprey.sizeratio.max.sp0"] = "1.0"
        minimal_config["predation.predprey.sizeratio.max.sp1"] = "0.5"
        cfg = EngineConfig.from_dict(minimal_config)
        # Java convention: min > max gets swapped → min becomes the smaller value
        assert cfg.size_ratio_min[0, 0] == pytest.approx(1.0)
        assert cfg.size_ratio_min[1, 0] == pytest.approx(0.5)
        assert cfg.size_ratio_max[0, 0] == pytest.approx(3.5)
        assert cfg.size_ratio_max[1, 0] == pytest.approx(2.0)

    def test_size_ratio_defaults(self, minimal_config):
        """size_ratio_min defaults to 1.0, size_ratio_max defaults to 3.5."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.size_ratio_min[0, 0] == pytest.approx(1.0)
        assert cfg.size_ratio_max[0, 0] == pytest.approx(3.5)

    def test_out_mortality_rate(self, minimal_config):
        minimal_config["mortality.out.rate.sp0"] = "0.1"
        minimal_config["mortality.out.rate.sp1"] = "0.05"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.out_mortality_rate[0] == pytest.approx(0.1)

    def test_starvation_rate_max(self, minimal_config):
        minimal_config["mortality.starvation.rate.max.sp0"] = "3.0"
        minimal_config["mortality.starvation.rate.max.sp1"] = "2.0"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.starvation_rate_max[0] == pytest.approx(3.0)

    def test_fishing_enabled(self, minimal_config):
        minimal_config["simulation.fishing.mortality.enabled"] = "false"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.fishing_enabled is False
