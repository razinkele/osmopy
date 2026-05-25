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


def _state_species_id(
    species_id: list[int],
    abundance: list[float] | None = None,
) -> SchoolState:
    """Build a SchoolState with the given species_id and per-school abundance.

    Tests target `_collect_trait_stats` directly (the new unit added by this
    plan) instead of the umbrella `_collect_outputs`. The umbrella drags in
    `_collect_biomass_abundance`, `_collect_distributions`, `_collect_bioen`,
    etc., which can fail noisily on a zero-filled state and obscure the
    intent of these unit tests. Integration coverage of the umbrella's
    `phenotypes` kwarg lives in Task 8.5 (spy on the real baltic_ev run).

    `_collect_trait_stats` reports individual-level statistics, so it filters
    on `abundance > 0` (the engine's live-school convention, state.py:198) and
    weights by abundance. `SchoolState.create` zeroes abundance, which would
    make every school count as dead, so abundance defaults to one individual
    per school here (equal weights -> weighted stats reduce to unweighted).
    """
    n = len(species_id)
    sp = np.array(species_id, dtype=np.int32)  # species_id is int32 per state.py:39
    state = SchoolState.create(n_schools=n, species_id=sp)
    abund = np.ones(n, dtype=np.float64) if abundance is None else np.asarray(abundance, np.float64)
    return state.replace(abundance=abund)


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


def test_collect_trait_stats_excludes_zero_abundance_schools() -> None:
    """CodeRabbit major (simulate.py:1055): zero-abundance school slots are dead
    and must not participate. Outputs are collected before compaction, so empty
    slots are present in `state`."""
    state = _state_species_id([0, 0, 0], abundance=[0.0, 5.0, 0.0])
    phenotypes = {"imax": np.array([1.0, 9.0, 100.0])}

    out = _collect_trait_stats(state, phenotypes)

    # Only the live school (abundance 5) contributes; the 1.0 and 100.0 phenotypes
    # belong to dead slots and are excluded. Unfiltered mean would be ~36.67.
    assert out["imax"][0].mean == pytest.approx(9.0)
    assert out["imax"][0].variance == pytest.approx(0.0)
    assert out["imax"][0].n_individuals == 5


def test_collect_trait_stats_weights_by_abundance() -> None:
    """CodeRabbit major (simulate.py:1055): a school is a super-individual, so a
    big cohort must outweigh a tiny one in the reported mean/variance, and
    n_individuals must be the summed abundance, not the school count."""
    state = _state_species_id([0, 0], abundance=[10.0, 30.0])
    phenotypes = {"imax": np.array([4.0, 8.0])}

    out = _collect_trait_stats(state, phenotypes)

    # Weighted mean = (10*4 + 30*8) / 40 = 7.0 (unweighted would be 6.0).
    assert out["imax"][0].mean == pytest.approx(7.0)
    # Weighted population variance = (10*(4-7)^2 + 30*(8-7)^2) / 40 = 3.0.
    assert out["imax"][0].variance == pytest.approx(3.0)
    # n_individuals is the total head-count, not the 2 school slots.
    assert out["imax"][0].n_individuals == 40


def test_collect_trait_stats_drops_species_with_no_live_schools() -> None:
    """A species whose schools are all zero-abundance must be absent from the
    output entirely (feeds the merge-step extinction handling)."""
    state = _state_species_id([0, 1], abundance=[5.0, 0.0])
    phenotypes = {"imax": np.array([2.0, 9.0])}

    out = _collect_trait_stats(state, phenotypes)

    assert set(out["imax"].keys()) == {0}


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


def test_average_step_outputs_drops_species_absent_in_final_step() -> None:
    """CodeRabbit major (simulate.py:1169): a species present early in the window
    but gone by the final sampled step must NOT carry forward its last-seen stats,
    otherwise extinction within the window looks like persistence."""
    out1 = StepOutput(
        step=0,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        trait_stats={
            "imax": {
                0: TraitStats(mean=4.0, variance=1.0, n_individuals=10),
                1: TraitStats(mean=7.0, variance=2.0, n_individuals=20),
            }
        },
    )
    out2 = StepOutput(
        step=1,
        biomass=np.zeros(1),
        abundance=np.zeros(1),
        mortality_by_cause=np.zeros((1, 1)),
        # Species 1 has gone extinct by the final sampled step.
        trait_stats={"imax": {0: TraitStats(mean=6.0, variance=2.0, n_individuals=30)}},
    )
    merged = _average_step_outputs([out1, out2], freq=2, record_step=1)

    # Species 1 is absent at record_step -> dropped, not reported with stale 20/7.0.
    assert set(merged.trait_stats["imax"].keys()) == {0}
    # Species 0 (present in both) still averages mean/variance over the window
    # and reports the final-step head-count.
    assert merged.trait_stats["imax"][0].mean == pytest.approx(5.0)
    assert merged.trait_stats["imax"][0].variance == pytest.approx(1.5)
    assert merged.trait_stats["imax"][0].n_individuals == 30


def test_focal_outputs_thread_phenotypes_when_genetics_on(monkeypatch, tmp_path) -> None:
    """When ctx.genetic_state is non-None, the focal `_collect_outputs` call
    must receive the same phenotypes dict that `express_traits` produced.

    Patched on the module attribute (`sim_mod._collect_outputs`) because the
    step loop at `simulate.py:1402` resolves the name in the simulate module's
    namespace at call time. Top-of-file imports of `_collect_outputs` would
    create a separate binding that bypasses the patch — we intentionally do
    NOT import `_collect_outputs` at the top of this test file.
    """
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    import osmose.engine.simulate as sim_mod  # not exported from osmose.engine.__init__

    captured: dict = {}
    real_collect = sim_mod._collect_outputs

    def spy_collect(*args, **kwargs):
        if kwargs.get("phenotypes") is not None:
            captured["phenotypes"] = kwargs["phenotypes"]
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(sim_mod, "_collect_outputs", spy_collect)
    # Anchor the fixture on this file's location so the test passes regardless of
    # pytest's cwd (matches tests/test_genetic_trait_means_csv.py).
    fixture = Path(__file__).parent.parent / "data" / "baltic_ev" / "baltic_ev_all-parameters.csv"
    cfg = OsmoseConfigReader().read(fixture)
    cfg["simulation.time.nyear"] = "1"
    PythonEngine().run(cfg, tmp_path, seed=0)
    assert "phenotypes" in captured
    assert "imax" in captured["phenotypes"]
