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
    assert set(rows) == {"cod", "herring", "sprat", "flounder"}
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

    recs = json.load(open(SNAP / "spr.27.22-32.assessment.json"))
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
