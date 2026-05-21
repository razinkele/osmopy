from pathlib import Path

import numpy as np
import pytest

from osmose.engine.simulate import (
    StepOutput,
    TraitStats,
    _average_step_outputs,
    _collect_trait_stats,
)
from osmose.engine.state import SchoolState


def _state_species_id(species_id: list[int]) -> SchoolState:
    """Build a SchoolState with the given species_id assignment.

    Tests target `_collect_trait_stats` directly (the new unit added by this
    plan) instead of the umbrella `_collect_outputs`. The umbrella drags in
    `_collect_biomass_abundance`, `_collect_distributions`, `_collect_bioen`,
    etc., which can fail noisily on a zero-filled state and obscure the
    intent of these unit tests. Integration coverage of the umbrella's
    `phenotypes` kwarg lives in Task 8.5 (spy on the real baltic_ev run).
    """
    n = len(species_id)
    sp = np.array(species_id, dtype=np.int32)  # species_id is int32 per state.py:39
    state = SchoolState.create(n_schools=n, species_id=sp)
    return state


def test_collect_trait_stats_empty_when_phenotypes_empty() -> None:
    state = _state_species_id([0, 0, 0, 0])
    out = _collect_trait_stats(state, phenotypes={})
    assert out == {}


def test_collect_trait_stats_populated_from_phenotypes() -> None:
    state = _state_species_id([0, 0, 0, 0])
    phenotypes = {"imax": np.array([3.0, 4.0, 5.0, 6.0])}

    out = _collect_trait_stats(state, phenotypes)

    assert "imax" in out
    assert 0 in out["imax"]
    ts = out["imax"][0]
    assert ts.mean == pytest.approx(4.5)
    assert ts.variance == pytest.approx(1.25)  # np.var of [3,4,5,6]
    assert ts.n_individuals == 4


def test_collect_trait_stats_groups_by_species() -> None:
    """Mixed-species state: two sp0 schools and two sp1 schools share one
    phenotype array; trait_stats must split per species."""
    state = _state_species_id([0, 0, 1, 1])
    phenotypes = {"imax": np.array([3.0, 5.0, 10.0, 12.0])}

    out = _collect_trait_stats(state, phenotypes)

    assert set(out["imax"].keys()) == {0, 1}
    assert out["imax"][0].mean == pytest.approx(4.0)
    assert out["imax"][1].mean == pytest.approx(11.0)
    assert out["imax"][0].n_individuals == 2
    assert out["imax"][1].n_individuals == 2


def test_average_step_outputs_single_element_propagates_trait_stats() -> None:
    """Short-circuit path at simulate.py:998-1018 (len(accumulated) == 1).
    Most runs use output.recordfrequency.ndt=1 so this is the hot path."""
    only = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats={"imax": {0: TraitStats(mean=4.0, variance=1.0, n_individuals=10)}},
    )
    merged = _average_step_outputs([only], freq=1, record_step=0)
    assert merged.trait_stats is not None
    assert merged.trait_stats["imax"][0].mean == pytest.approx(4.0)


def test_average_step_outputs_merges_trait_stats() -> None:
    """Multi-element merge path."""
    out1 = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats={"imax": {0: TraitStats(mean=4.0, variance=1.0, n_individuals=10)}},
    )
    out2 = StepOutput(
        step=1,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats={"imax": {0: TraitStats(mean=6.0, variance=2.0, n_individuals=20)}},
    )
    merged = _average_step_outputs([out1, out2], freq=2, record_step=1)
    # Mean over the two accumulated steps, equal-weight (matches existing _avg_bioen)
    assert merged.trait_stats["imax"][0].mean == pytest.approx(5.0)
    # `variance` is mean-of-step-variances, NOT pooled variance — see Step 2.4 note.
    assert merged.trait_stats["imax"][0].variance == pytest.approx(1.5)
    # n_individuals carries through as the latest value (snapshot semantic)
    assert merged.trait_stats["imax"][0].n_individuals == 20


def test_focal_outputs_thread_phenotypes_when_genetics_on(monkeypatch) -> None:
    """When ctx.genetic_state is non-None, the focal `_collect_outputs` call
    must receive the same phenotypes dict that `express_traits` produced.

    Patched on the module attribute (`sim_mod._collect_outputs`) because the
    step loop at `simulate.py:1402` resolves the name in the simulate module's
    namespace at call time. Top-of-file imports of `_collect_outputs` would
    create a separate binding that bypasses the patch — we intentionally do
    NOT import `_collect_outputs` at the top of this test file.
    """
    import osmose.engine.simulate as sim_mod

    captured: dict = {}
    real_collect = sim_mod._collect_outputs

    def spy_collect(*args, **kwargs):
        captured["phenotypes"] = kwargs.get("phenotypes")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(sim_mod, "_collect_outputs", spy_collect)

    pytest.skip("baltic_ev fixture not wired until Task 8; unskipped in Step 8.5.")
