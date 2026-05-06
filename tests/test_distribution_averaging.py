"""M1: distribution outputs are averaged over the recording window (Java parity).

Java OSMOSE `AbstractDistribOutput.write` (verified 2026-05-06 against
osmose-model/osmose master):

    array[iClass][cpt++] = values[iSpec][iClass] / getRecordFrequency();

i.e. distribution values (biomass-by-age, abundance-by-age, biomass-by-size,
abundance-by-size) are accumulated across the recording window then divided
by the window size.

Pre-M1, Python's `_average_step_outputs` used `accumulated[-1]` (last step)
for these dicts while everything else (biomass, abundance, mortality,
yield, bioen, diet, spatial) aggregated correctly. The mismatch was
silent: a sim with `output.recordfrequency.ndt = 1` (the default in most
fixtures) didn't trip it, and a sim with > 1 produced subtly wrong
`*_by_age` / `*_by_size` outputs.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.simulate import StepOutput, _average_step_outputs


def _make_step(
    *,
    biomass_by_age: dict[int, np.ndarray] | None = None,
    biomass_by_size: dict[int, np.ndarray] | None = None,
    step: int = 0,
) -> StepOutput:
    """Minimal StepOutput skeleton — only the fields M1 cares about."""
    return StepOutput(
        step=step,
        biomass=np.array([1.0]),
        abundance=np.array([1.0]),
        mortality_by_cause=np.array([[0.0]]),
        yield_by_species=np.array([0.0]),
        biomass_by_age=biomass_by_age,
        abundance_by_age=None,
        biomass_by_size=biomass_by_size,
        abundance_by_size=None,
    )


def test_biomass_by_age_is_averaged_across_window() -> None:
    """ndt=4 window: biomass-by-age must be the mean of all 4, not the last."""
    accumulated = [
        _make_step(biomass_by_age={0: np.array([10.0, 20.0, 30.0])}),
        _make_step(biomass_by_age={0: np.array([14.0, 24.0, 34.0])}),
        _make_step(biomass_by_age={0: np.array([18.0, 28.0, 38.0])}),
        _make_step(biomass_by_age={0: np.array([22.0, 32.0, 42.0])}),
    ]
    out = _average_step_outputs(accumulated, freq=4, record_step=3)
    expected = np.array([(10 + 14 + 18 + 22) / 4, (20 + 24 + 28 + 32) / 4, (30 + 34 + 38 + 42) / 4])
    assert out.biomass_by_age is not None
    np.testing.assert_allclose(out.biomass_by_age[0], expected)
    # Pre-fix: would have returned accumulated[-1].biomass_by_age = [22, 32, 42]
    assert not np.allclose(out.biomass_by_age[0], np.array([22.0, 32.0, 42.0])), (
        "regressed to last-step semantics"
    )


def test_biomass_by_size_is_averaged_across_window() -> None:
    accumulated = [
        _make_step(biomass_by_size={1: np.array([5.0, 10.0])}),
        _make_step(biomass_by_size={1: np.array([7.0, 14.0])}),
    ]
    out = _average_step_outputs(accumulated, freq=2, record_step=1)
    expected = np.array([6.0, 12.0])
    assert out.biomass_by_size is not None
    np.testing.assert_allclose(out.biomass_by_size[1], expected)


def test_distribution_averaging_handles_missing_step_dicts() -> None:
    """If a step's dict is None (e.g. a non-distribution-output step), the
    average is taken over the steps that DO have it — matching how the
    bioen / spatial averagers handle the same situation."""
    accumulated = [
        _make_step(biomass_by_age={0: np.array([10.0, 20.0])}),
        _make_step(biomass_by_age=None),
        _make_step(biomass_by_age={0: np.array([20.0, 30.0])}),
    ]
    out = _average_step_outputs(accumulated, freq=3, record_step=2)
    expected = np.array([15.0, 25.0])  # mean of the 2 non-None steps
    assert out.biomass_by_age is not None
    np.testing.assert_allclose(out.biomass_by_age[0], expected)


def test_distribution_averaging_returns_none_when_all_steps_lack_dict() -> None:
    """No step in the window had a distribution dict -> output dict is None."""
    accumulated = [_make_step() for _ in range(3)]  # all default biomass_by_age=None
    out = _average_step_outputs(accumulated, freq=3, record_step=2)
    assert out.biomass_by_age is None
    assert out.biomass_by_size is None


def test_distribution_averaging_per_species_handling() -> None:
    """Different species can appear in different steps; each is averaged
    only over the steps where it's present (matches `_avg_spatial`)."""
    accumulated = [
        _make_step(biomass_by_age={0: np.array([10.0]), 1: np.array([100.0])}),
        _make_step(biomass_by_age={0: np.array([20.0])}),  # sp 1 absent
    ]
    out = _average_step_outputs(accumulated, freq=2, record_step=1)
    assert out.biomass_by_age is not None
    np.testing.assert_allclose(out.biomass_by_age[0], np.array([15.0]))  # mean of 2
    np.testing.assert_allclose(out.biomass_by_age[1], np.array([100.0]))  # mean of 1


def test_freq_one_window_unchanged() -> None:
    """Sanity: freq=1 (window of 1) returns the single accumulated step."""
    s = _make_step(biomass_by_age={0: np.array([1.0, 2.0])})
    out = _average_step_outputs([s], freq=1, record_step=0)
    assert out.biomass_by_age is not None
    np.testing.assert_allclose(out.biomass_by_age[0], np.array([1.0, 2.0]))
