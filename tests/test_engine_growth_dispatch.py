"""Tests for growth classname parsing in EngineConfig."""

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


class TestGrowthClassnameParsing:
    def test_vb_classname_parsed(self, minimal_config):
        """New-style VonBertalanffyGrowth classname -> 'VB'."""
        minimal_config["growth.java.classname.sp0"] = (
            "fr.ird.osmose.process.growth.VonBertalanffyGrowth"
        )
        minimal_config["growth.java.classname.sp1"] = (
            "fr.ird.osmose.process.growth.VonBertalanffyGrowth"
        )
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["VB", "VB"]

    def test_gompertz_classname_parsed(self, minimal_config):
        """New-style GompertzGrowth classname -> 'GOMPERTZ'."""
        minimal_config["growth.java.classname.sp0"] = (
            "fr.ird.osmose.process.growth.GompertzGrowth"
        )
        minimal_config["growth.java.classname.sp1"] = (
            "fr.ird.osmose.process.growth.GompertzGrowth"
        )
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["GOMPERTZ", "GOMPERTZ"]

    def test_mixed_classnames_parsed(self, minimal_config):
        """Mixed classnames per species are parsed independently."""
        minimal_config["growth.java.classname.sp0"] = (
            "fr.ird.osmose.process.growth.VonBertalanffyGrowth"
        )
        minimal_config["growth.java.classname.sp1"] = (
            "fr.ird.osmose.process.growth.GompertzGrowth"
        )
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["VB", "GOMPERTZ"]

    def test_legacy_vb_classname_defaults_to_vb(self, minimal_config):
        """Old-style 'fr.ird.osmose.growth.VonBertalanffy' -> 'VB'."""
        minimal_config["growth.java.classname.sp0"] = "fr.ird.osmose.growth.VonBertalanffy"
        minimal_config["growth.java.classname.sp1"] = "fr.ird.osmose.growth.VonBertalanffy"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["VB", "VB"]

    def test_legacy_gompertz_classname_parsed(self, minimal_config):
        """Old-style 'fr.ird.osmose.growth.Gompertz' -> 'GOMPERTZ'."""
        minimal_config["growth.java.classname.sp0"] = "fr.ird.osmose.growth.Gompertz"
        minimal_config["growth.java.classname.sp1"] = "fr.ird.osmose.growth.Gompertz"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["GOMPERTZ", "GOMPERTZ"]

    def test_legacy_linear_classname_maps_to_vb(self, minimal_config):
        """Old-style Linear classname (never real) maps to 'VB'."""
        minimal_config["growth.java.classname.sp0"] = "fr.ird.osmose.growth.Linear"
        minimal_config["growth.java.classname.sp1"] = "fr.ird.osmose.growth.Linear"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["VB", "VB"]

    def test_unknown_classname_defaults_to_vb(self, minimal_config):
        """Unrecognised classname falls back to 'VB'."""
        minimal_config["growth.java.classname.sp0"] = "com.example.SomeOtherGrowth"
        minimal_config["growth.java.classname.sp1"] = "com.example.SomeOtherGrowth"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["VB", "VB"]

    def test_missing_classname_defaults_to_vb(self, minimal_config):
        """Absent classname key defaults to 'VB'."""
        # minimal_config has no growth.java.classname keys
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.growth_class == ["VB", "VB"]

    def test_growth_class_length_matches_n_species(self, minimal_config):
        """growth_class list length always equals n_species."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert len(cfg.growth_class) == cfg.n_species
