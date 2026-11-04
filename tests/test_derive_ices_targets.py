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
                # bzq.27.2628's SAG columns are swapped: its "landings" is the total catch.
                if stock == "bzq.27.2628":
                    v = r["landings"] or r["catches"]
                else:
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


def _snapshot_rows(stock: str) -> list[dict]:
    import json

    with open(SNAP / f"{stock}.assessment.json") as f:
        return json.load(f)


def _swapped() -> set[str]:
    import json

    return set(
        json.loads((SNAP / "index.json").read_text()).get("catch_landings_swapped_stocks", [])
    )


def test_catches_are_never_below_landings_after_corrections():
    # Catches = landings + discards, so a year with catches < landings means SAG published the
    # two columns the other way round (bzq.27.2628, 2015-2020: landings - catches == discards).
    # The derivation prefers `catches`, so an unlisted swap silently drops the discards.
    for path in sorted(SNAP.glob("*.assessment.json")):
        stock = path.name.removesuffix(".assessment.json")
        first, second = ("landings", "catches") if stock in _swapped() else ("catches", "landings")
        for r in _snapshot_rows(stock):
            c, la = r[first], r[second]
            if c not in (None, "") and la not in (None, ""):
                assert float(c) >= float(la) - 0.5, (stock, r["year"], c, la)


def test_every_listed_swap_is_still_needed():
    # A stale entry would flip a correct stock the wrong way after a refresh.
    for stock in _swapped():
        assert any(
            r["catches"] not in (None, "")
            and r["landings"] not in (None, "")
            and float(r["catches"]) < float(r["landings"]) - 0.5
            for r in _snapshot_rows(stock)
        ), stock


def test_every_committed_catch_row_matches_the_derivation():
    # The calibration reads the committed CSV. Until 2026-11 nothing tied it to the snapshots:
    # cod_west's row came from a different (2023) assessment than the snapshot (2022), flounder's
    # covered one stock of four, and cod_east's band had been rounded by hand. Every field must
    # now be exactly what derive_ices_targets.py --write produces.
    import csv

    with open(SNAP.parent / "biomass_targets.csv") as f:
        committed = {
            r["species"]: r
            for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))
            if r["reference_point_type"] == "catch"
        }
    derived = {r["species"]: r for r in derive_catch_targets(SNAP)}
    assert set(committed) == set(derived)
    for sp, row in derived.items():
        for col, val in row.items():
            assert committed[sp][col] == val, (sp, col, committed[sp][col], val)


def test_cod_west_catch_includes_discards_and_the_supplemented_2022():
    # Independent of the code under test. cod.27.22-24's pinned 2022 assessment has no `catches`
    # column (landings + discards are separate) and ends at 2021; 2022 (88 t landed + 27 t
    # discarded) comes from the 2023 assessment via index.json catch_supplements.
    per_year = {
        int(r["year"]): float(r["landings"]) + float(r["discards"] or 0)
        for r in _snapshot_rows("cod.27.22-24")
        if r["landings"] and 2018 <= int(r["year"]) <= 2022
    }
    assert 2022 not in per_year
    per_year[2022] = 88 + 27
    rows = {r["species"]: r for r in derive_catch_targets(SNAP)}
    expected = sum(per_year.values()) / 5
    assert float(rows["cod_west"]["target_tonnes"]) == pytest.approx(expected, rel=1e-9)


def _write_snapshot(tmp_path, rows: list[dict], index_extra: dict | None = None):
    import json

    (tmp_path / "s.assessment.json").write_text(json.dumps(rows))
    return tmp_path


def test_fallback_adds_discards_to_landings(tmp_path):
    from scripts.derive_ices_targets import _stock_catch

    snap = _write_snapshot(
        tmp_path,
        [
            {"year": "2020", "catches": "", "landings": "100", "discards": "7"},
            {"year": "2021", "catches": "", "landings": "50", "discards": ""},
            {"year": "2022", "catches": "90", "landings": "80", "discards": "10"},
        ],
    )
    assert _stock_catch(snap, "s") == {2020: 107.0, 2021: 50.0, 2022: 90.0}


def test_supplement_for_a_year_already_in_the_snapshot_raises(tmp_path):
    # A refresh that brings the year must not leave the hand-entered value silently in place.
    from scripts.derive_ices_targets import _stock_catch

    snap = _write_snapshot(
        tmp_path, [{"year": "2022", "catches": "", "landings": "88", "discards": "27"}]
    )
    with pytest.raises(ValueError, match="already in the snapshot"):
        _stock_catch(
            snap, "s", supplements={"2022": {"landings": 88, "discards": 27, "source": "x"}}
        )
