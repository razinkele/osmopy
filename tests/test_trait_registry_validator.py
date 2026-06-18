from pathlib import Path

import pytest

from osmose.engine.config import EngineConfig


EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _base_dict() -> dict[str, str]:
    """Validator test base. Note: the validator runs INSIDE from_dict during
    config parsing, so we need a complete schema-passing config. Hand-rolled
    dicts (e.g., {nspecies, lifespan, name}) will fail upstream of the
    validator on missing linf/K/t0/length-weight keys before the validator
    is even reached."""
    from osmose.config import OsmoseConfigReader

    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    # Reader canonicalizes to NEW 4.4.0 keys; set the canonical genetics key so it
    # survives from_dict's skip-if-target-exists merge.
    raw["module.genetics.enabled"] = "true"
    return raw


def test_declared_trait_with_nonzero_variance_requires_mean() -> None:
    cfg = _base_dict()
    cfg.update(
        {
            "evolution.trait.imax.target": "bioen_i_max",
            "evolution.trait.imax.var.sp0": "0.1",  # nonzero variance — needs mean
            # NOTE: no evolution.trait.imax.mean.sp0
            "evolution.trait.imax.var.sp1": "0.0",  # zero variance ok without mean
        }
    )
    with pytest.raises(ValueError, match="evolution.trait.imax.mean.sp0"):
        EngineConfig.from_dict(cfg)


def test_declared_trait_with_zero_variance_does_not_require_mean() -> None:
    cfg = _base_dict()
    cfg.update(
        {
            "evolution.trait.imax.target": "bioen_i_max",
            "evolution.trait.imax.var.sp0": "0.0",
            "evolution.trait.imax.var.sp1": "0.0",
        }
    )
    # Should not raise
    EngineConfig.from_dict(cfg)


def test_complete_declaration_passes() -> None:
    cfg = _base_dict()
    cfg.update(
        {
            "evolution.trait.imax.target": "bioen_i_max",
            "evolution.trait.imax.mean.sp0": "3.5",
            "evolution.trait.imax.var.sp0": "0.1",
            "evolution.trait.imax.mean.sp1": "5.0",
            "evolution.trait.imax.var.sp1": "0.0",
        }
    )
    EngineConfig.from_dict(cfg)
