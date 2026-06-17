"""Tests for Python-engine community outputs: DistribBySize + meanTL."""

from osmose.engine.config import EngineConfig


def _base_config(extra: dict | None = None) -> dict[str, str]:
    base = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "5",
        "simulation.nschool.sp1": "5",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Hake",
        "species.linf.sp0": "19.5",
        "species.linf.sp1": "110.0",
        "species.k.sp0": "0.364",
        "species.k.sp1": "0.106",
        "species.t0.sp0": "-0.70",
        "species.t0.sp1": "-0.17",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.005",
        "species.length2weight.allometric.power.sp0": "3.06",
        "species.length2weight.allometric.power.sp1": "3.14",
        "species.lifespan.sp0": "4",
        "species.lifespan.sp1": "12",
        "species.vonbertalanffy.threshold.age.sp0": "0",
        "species.vonbertalanffy.threshold.age.sp1": "0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
    }
    if extra:
        base.update(extra)
    return base


def test_output_meantl_flag_defaults_false():
    config = EngineConfig.from_dict(_base_config())
    assert config.output_meantl is False


def test_output_meantl_flag_enabled():
    # NOTE: the key is LOWERCASE. The real OsmoseConfigReader lowercases every config key
    # (reader.py:157), and config.py reads the flag with the lowercase literal. _base_config
    # builds a hand dict that bypasses the reader, so it MUST use the lowercase key here.
    config = EngineConfig.from_dict(_base_config({"output.meantl.enabled": "true"}))
    assert config.output_meantl is True
