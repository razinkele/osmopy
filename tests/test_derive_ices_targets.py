"""Derivation of ICES catch targets from the in-repo snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.derive_ices_targets import WINDOW, derive_catch_targets

SNAP = Path(__file__).resolve().parents[1] / "data" / "baltic" / "reference" / "ices_snapshots"


def test_window_is_2018_2022():
    assert WINDOW == (2018, 2022)


def test_catch_targets_for_all_assessed_species():
    rows = {r["species"]: r for r in derive_catch_targets(SNAP)}
    assert set(rows) == {"cod_west", "cod_east", "herring", "sprat", "flounder"}
    for r in rows.values():
        assert r["reference_point_type"] == "catch"
        assert float(r["weight"]) == 0.5
        lo, tgt, hi = float(r["lower_tonnes"]), float(r["target_tonnes"]), float(r["upper_tonnes"])
        assert 0 < lo <= tgt <= hi  # positive, ordered band


def test_sprat_catch_matches_snapshot_mean(tmp_path):
    # Sprat is a single-stock species (spr.27.22-32); its catch target mean must equal the
    # mean of that stock's catches (falling back to landings where catches are empty) over
    # 2018-2022 — the same catches-preferred field the derivation code reads.
    import json

    import numpy as np

    with open(SNAP / "spr.27.22-32.assessment.json") as _f:
        recs = json.load(_f)
    catches = [
        float(r["catches"] or r["landings"])
        for r in recs
        if (r["catches"] or r["landings"]) and 2018 <= int(r["year"]) <= 2022
    ]
    rows = {r["species"]: r for r in derive_catch_targets(SNAP)}
    assert float(rows["sprat"]["target_tonnes"]) == pytest.approx(np.mean(catches), rel=1e-9)


def test_herring_catch_target_includes_central_stock():
    # her.27.25-2932 (central Baltic, dominant stock) has EMPTY `landings` for 2018-2022 —
    # only `catches` is populated. Regression guard: if the derivation regresses to reading
    # landings only, this stock's ~84-241 kt/yr drops out and the target collapses to ~20 kt.
    rows = {r["species"]: r for r in derive_catch_targets(SNAP)}
    assert float(rows["herring"]["target_tonnes"]) > 100_000


_FLOUNDER_STOCKS = ("fle.27.2223", "bzq.27.2425", "bzq.27.2628", "bwp.27.2729-32")


def _flounder_catch_mean() -> float:
    """Independent of the code under test: per-year catches (landings where catches are empty)
    summed over all four Baltic flounder stocks, then averaged over 2018-2022."""
    import json

    per_year: dict[int, float] = {}
    for stock in _FLOUNDER_STOCKS:
        with open(SNAP / f"{stock}.assessment.json") as f:
            for r in json.load(f):
                v = r["catches"] or r["landings"]
                if v and 2018 <= int(r["year"]) <= 2022:
                    per_year[int(r["year"])] = per_year.get(int(r["year"]), 0.0) + float(v)
    assert sorted(per_year) == [2018, 2019, 2020, 2021, 2022]
    return sum(per_year.values()) / len(per_year)


def test_flounder_catch_covers_all_four_baltic_stocks():
    # Until 2026-11 flounder mapped to fle.27.2223 alone (Belt Seas, ~5% of Baltic flounder
    # catch), so its catch target was 834 t against ~15 kt actually landed.
    rows = {r["species"]: r for r in derive_catch_targets(SNAP)}
    assert float(rows["flounder"]["target_tonnes"]) == pytest.approx(
        _flounder_catch_mean(), rel=1e-9
    )
    assert float(rows["flounder"]["target_tonnes"]) > 10_000


def test_committed_flounder_catch_row_matches_derivation():
    # The calibration reads the committed CSV, not the derivation: a mapping fix that never
    # reached the CSV would leave calibrations scoring against the old 834 t.
    from osmose.calibration.targets import load_targets

    targets, _ = load_targets(SNAP.parent / "biomass_targets.csv")
    (row,) = [t for t in targets if t.species == "flounder" and t.reference_point_type == "catch"]
    assert row.target == pytest.approx(_flounder_catch_mean(), rel=1e-9)


def test_version_bump_increments_rather_than_resetting():
    from scripts.derive_ices_targets import _bump_minor

    assert _bump_minor("#! version: 1.4") == "#! version: 1.5"
    assert _bump_minor("#! version: 2.19") == "#! version: 2.20"
