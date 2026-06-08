"""Property-based tests: diet_network_at per-timestep aggregation."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings

from osmose.trophic_network import _split_species, diet_network_at
from tests.strategies import diet_matrices

# The diet pipeline (melt + groupbys + read_csv) is ~20 ms/example; cap examples
# so the file stays snappy (plan-review measured ~6s at 75; 50 keeps it ~4s while
# the strategies still hit their interesting cases >70% of the time).
DIET = settings(max_examples=50)


def _write(df, td):
    (Path(td) / "x_dietMatrix.csv").write_text(df.to_csv(index=False))
    return Path(td)


@DIET
@given(df=diet_matrices())
def test_proportions_nonneg_and_clean_names(df):
    with tempfile.TemporaryDirectory() as td:
        net = diet_network_at(_write(df, td), time=1.0, threshold=0.0)
    assert (net["proportion"] >= 0).all()
    names = set(net["predator"]) | set(net["prey"])
    assert not any(" in [" in s for s in names)


@DIET
@given(df=diet_matrices())
def test_threshold_monotonic(df):
    with tempfile.TemporaryDirectory() as td:
        d = _write(df, td)
        lo = diet_network_at(d, time=1.0, threshold=10.0)
        hi = diet_network_at(d, time=1.0, threshold=40.0)
    lo_edges = {(r.predator, r.prey) for r in lo.itertuples()}
    hi_edges = {(r.predator, r.prey) for r in hi.itertuples()}
    assert hi_edges <= lo_edges


@DIET
@given(df=diet_matrices())
def test_prey_sum_exactness_stage_level(df):
    with tempfile.TemporaryDirectory() as td:
        net = diet_network_at(_write(df, td), time=1.0, threshold=0.0, predator_level="stage")
    # ~28% of matrices are all-dead/all-NaN -> empty net; don't let this (the
    # reorder-catching property) pass vacuously on those.
    assume(len(net) > 0)
    prey_species = df["Prey"].map(_split_species)
    for r in net.itertuples():
        # stage label is r.predator; prey species is r.prey
        expected = df.loc[prey_species == r.prey, r.predator].sum(skipna=True)
        assert r.proportion == pytest.approx(expected, rel=1e-9, abs=1e-12)


@DIET
@given(df=diet_matrices())
def test_dead_stage_never_surfaces(df):
    pred_cols = [c for c in df.columns if c not in ("Time", "Prey")]
    dead = [c for c in pred_cols if df[c].fillna(0.0).sum() == 0]
    with tempfile.TemporaryDirectory() as td:
        net = diet_network_at(_write(df, td), time=1.0, threshold=0.0, predator_level="stage")
    preds = set(net["predator"])
    for dcol in dead:
        assert dcol not in preds
