"""The ICES calibration targets are disaggregated into cod_east (cod.27.24-32)
and cod_west (cod.27.22-24), replacing the single aggregate cod target
(Phase 1 Task 5)."""

from pathlib import Path

from osmose.calibration.targets import load_targets

TARGETS = Path("data/baltic/reference/biomass_targets.csv")


def _by_species():
    targets, _ = load_targets(TARGETS)
    out: dict[str, list] = {}
    for t in targets:
        out.setdefault(t.species, []).append(t)
    return out


def test_aggregate_cod_replaced_by_east_and_west():
    by = _by_species()
    assert "cod" not in by, "aggregate cod target should be removed"
    assert "cod_east" in by
    assert "cod_west" in by


def test_cod_east_ssb_matches_post_collapse_range():
    by = _by_species()
    ssb = [t for t in by["cod_east"] if t.reference_point_type == "ssb"]
    assert len(ssb) == 1
    # cod.27.24-32 2018-2022 mean SSB ~70 kt (65-77 kt post-collapse)
    assert 60000 <= ssb[0].target <= 85000
    assert ssb[0].weight == 1.0


def test_cod_west_smaller_than_east():
    by = _by_species()
    east_ssb = next(t for t in by["cod_east"] if t.reference_point_type == "ssb")
    west_ssb = next(t for t in by["cod_west"] if t.reference_point_type == "ssb")
    # western Baltic cod is a smaller, also-depleted stock
    assert west_ssb.target < east_ssb.target


def test_both_stocks_have_catch_targets():
    by = _by_species()
    assert any(t.reference_point_type == "catch" for t in by["cod_east"])
    assert any(t.reference_point_type == "catch" for t in by["cod_west"])
