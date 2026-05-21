import numpy as np
import pytest

from osmose.engine.simulate import StepOutput, TraitStats


def test_trait_stats_dataclass_shape() -> None:
    ts = TraitStats(mean=1.5, variance=0.25, n_individuals=42)
    assert ts.mean == pytest.approx(1.5)
    assert ts.variance == pytest.approx(0.25)
    assert ts.n_individuals == 42


def test_step_output_accepts_trait_stats_none() -> None:
    out = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
    )
    assert out.trait_stats is None


def test_step_output_accepts_trait_stats_populated() -> None:
    trait_stats = {"imax": {0: TraitStats(mean=3.5, variance=0.1, n_individuals=100)}}
    out = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats=trait_stats,
    )
    assert out.trait_stats == trait_stats
