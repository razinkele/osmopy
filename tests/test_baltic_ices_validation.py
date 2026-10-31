"""Regression guards: Baltic model config stays consistent with ICES snapshots."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "baltic" / "reference" / "ices_snapshots"
MANIFEST = SNAPSHOT_DIR / "index.json"
VALIDATOR_SCRIPT = PROJECT_ROOT / "scripts" / "validate_baltic_vs_ices_sag.py"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="ICES SAG snapshots not yet pulled (run Tasks 2-3 first)",
)


@pytest.fixture(scope="module")
def validator_module():
    spec = importlib.util.spec_from_file_location("validate_baltic_vs_ices_sag", VALIDATOR_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_baltic_vs_ices_sag"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("validate_baltic_vs_ices_sag", None)


@pytest.fixture(scope="module")
def report(validator_module):
    return validator_module.run(write_report=False)


def test_manifest_exists_and_is_readable():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["advice_year"] == 2024
    assert "model_species_to_ices_stocks" in manifest
    assert manifest["model_species_to_ices_stocks"]["sprat"] == ["spr.27.22-32"]
    # The cod E/W split (2026-07-25) is reflected in the manifest; an aggregate 'cod' key
    # would silently drop the cod biomass row and leave fsh8 unchecked (#182).
    assert "cod" not in manifest["model_species_to_ices_stocks"]
    assert manifest["model_species_to_ices_stocks"]["cod_west"] == ["cod.27.22-24"]
    assert manifest["model_species_to_ices_stocks"]["cod_east"] == ["cod.27.24-32"]
    assert manifest["units_by_stock"]["spr.27.22-32"] == "tonnes"
    assert manifest["units_by_stock"]["cod.27.24-32"] == "index"


def test_every_assessed_species_has_snapshot_files():
    manifest = json.loads(MANIFEST.read_text())
    for species, stocks in manifest["model_species_to_ices_stocks"].items():
        for stock in stocks:
            assert (SNAPSHOT_DIR / f"{stock}.assessment.json").exists(), (
                f"Missing assessment snapshot for {species}/{stock}"
            )
            assert (SNAPSHOT_DIR / f"{stock}.reference_points.json").exists(), (
                f"Missing reference-points snapshot for {species}/{stock}"
            )


def test_validator_produces_report_for_sprat(report):
    """Sprat is the only single-stock, tonnes-scale, well-assessed Baltic
    species — it must always produce a numeric comparison on both axes.
    If either is None, the lowercase-key pathway or unit plumbing is
    silently broken."""
    assert "f_rates" in report
    assert "biomass_envelopes" in report
    assert "reference_points" in report
    f_rows = {r["species"]: r for r in report["f_rates"]}
    b_rows = {r["species"]: r for r in report["biomass_envelopes"]}
    assert "sprat" in f_rows, "F-rates table missing sprat"
    assert f_rows["sprat"]["ices_f_weighted"] is not None, (
        "sprat has linked stock but no F computed — check key casing in _series_by_year"
    )
    assert b_rows["sprat"]["ices_min_ssb"] is not None, (
        "sprat has linked stock but no SSB envelope computed"
    )


# Documented calibration choices that deliberately sit outside the ICES envelope.
# Bump the corresponding KNOWN_EXCEPTIONS set (and explain why in
# docs/baltic_ices_validation_2026-10-03.md Findings) when the model changes.
F_KNOWN_EXCEPTIONS = {
    # Western Baltic cod: model F=0.039 (fsh0) vs ICES ~0.91. Low F is intentional
    # against very-low post-collapse biomass to prevent extirpation in-model.
    # See docs/baltic_ices_validation_2026-10-03.md Findings.
    "cod_west",
    # cod_east and flounder were here until 2026-10-31 on "0.01 vs 0.16" and "6.4x above".
    # Both ICES series are RELATIVE (cod.27.24-32 "Fishing pressure: Relative",
    # fle.27.2223 "F/Fmsy"), so those ratios meant nothing; the validator now gives them
    # no verdict and the fence skips them. Flounder's real check is catch: model ~24 kt/yr
    # vs ICES 15.3 kt (2018-2022, all Baltic flounder stocks), ~1.6x.
}
# Empty since the cod split: cod_west's target [4000, 25000] t overlaps the western SSB
# envelope and cod_east is index-unit (no envelope). The pre-split aggregate 'cod' exception
# guarded a row that no longer existed (#182). (A bare `{}` would be a dict — keep `set()`.)
B_KNOWN_EXCEPTIONS: set[str] = set()


def test_no_severe_f_rate_drift(report):
    """Model F must stay within [0.25x, 4x] of ICES F — wider than the
    [0.5x, 1.5x] soft tolerance in the validator. Catches order-of-magnitude
    drift only. Known documented exceptions are allowlisted; bump
    F_KNOWN_EXCEPTIONS with a pointer to the findings doc when the set
    changes.
    """
    severe = []
    for r in report["f_rates"]:
        if r["ices_f_weighted"] is None or r["species"] in F_KNOWN_EXCEPTIONS:
            continue
        if r["ices_f_relative"]:
            # A relative index: model F / it is not an F ratio, so there is no drift to fence.
            continue
        if r["ices_f_weighted"] <= 0:
            # Moratorium or fishing-ban stock — no ratio to test. Drift fence
            # skips; the model-vs-zero comparison lives in a future scenario test.
            continue
        ratio = r["model_f"] / r["ices_f_weighted"]
        if ratio < 0.25 or ratio > 4.0:
            severe.append((r["species"], ratio))
    assert not severe, (
        f"Severe F-rate drift (order-of-magnitude) vs ICES: {severe}. "
        "Either refresh snapshots with a note, correct the model config, "
        "or — if this is a deliberate modeling choice — add the species to "
        "F_KNOWN_EXCEPTIONS with a rationale."
    )


def test_biomass_envelope_overlaps_ices_for_assessed_species(report):
    """Every species with linked tonnes-unit ICES stocks must have its
    biomass envelope overlap the ICES SSB envelope (2018-2022). Species
    where the envelope couldn't be computed (all stocks are index-unit
    or unassessed) are ignored. Known exceptions are allowlisted.
    """
    non_overlapping = [
        r["species"]
        for r in report["biomass_envelopes"]
        if r["envelopes_overlap"] is False and r["species"] not in B_KNOWN_EXCEPTIONS
    ]
    assert not non_overlapping, (
        f"Model biomass envelope does not overlap ICES SSB for: {non_overlapping}. "
        "Either broaden the target envelope, re-justify (e.g. total biomass "
        "vs SSB distinction — document in the report), or add the species "
        "to B_KNOWN_EXCEPTIONS with a rationale."
    )


# --- Branch coverage for the critical silent-None pathways -----------------
# These synthesize inputs so we verify the plumbing; the live `report`
# fixture only exercises whichever branch the current snapshots happen to hit.


def test_ssb_weighted_f_returns_none_when_window_empty(validator_module, monkeypatch):
    """If no stock has F+SSB overlap in 2018-2022, _ssb_weighted_f must return
    None — not 0.0, not crash. Without this the drift fence passes vacuously
    on a broken snapshot pull."""
    out_of_window = [{"year": "1990", "ssb": "100", "f": "0.5"}]
    monkeypatch.setattr(validator_module, "_load_assessment", lambda _s: out_of_window)
    assert validator_module._ssb_weighted_f(["fake_stock"]) is None


def test_ices_ssb_envelope_requires_full_coverage(validator_module, monkeypatch):
    """Partial-coverage years must drop from the envelope. Stock A reports
    2018-2022, stock B reports 2020-2022 only → envelope uses intersection
    (2020-2022), not the union (would undercount 2018-2019)."""
    a_data = [{"year": str(y), "ssb": "100"} for y in range(2018, 2023)]
    b_data = [{"year": str(y), "ssb": "50"} for y in range(2020, 2023)]
    store = {"A": a_data, "B": b_data}
    monkeypatch.setattr(validator_module, "_load_assessment", lambda s: store[s])
    lo, hi = validator_module._ices_ssb_envelope(["A", "B"])
    # 2020, 2021, 2022 each sum to 150 → min == max == 150
    assert lo == 150 and hi == 150


def test_ices_ssb_envelope_empty_intersection_returns_none(validator_module, monkeypatch):
    """If no year has coverage from all stocks, return (None, None) rather
    than silently producing a narrow envelope from the remainder."""
    a_data = [{"year": "2018", "ssb": "100"}]
    b_data = [{"year": "2022", "ssb": "200"}]
    store = {"A": a_data, "B": b_data}
    monkeypatch.setattr(validator_module, "_load_assessment", lambda s: store[s])
    assert validator_module._ices_ssb_envelope(["A", "B"]) == (None, None)


def test_both_cod_stocks_get_f_rows_from_their_own_fishery(report, validator_module):
    """Since the split each cod stock has its own fishery (fsh0 / fsh8) and its own
    ICES stock. cod_west compares against the tonnes stock with nothing excluded;
    cod_east has only the index-unit stock, so it takes the index-unit fallback
    (a dimensionless F is still an F) and excludes nothing."""
    f_rows = {r["species"]: r for r in report["f_rates"]}
    assert "cod" not in f_rows
    assert f_rows["cod_west"]["excluded_index_stocks"] == []
    assert f_rows["cod_west"]["ices_f_weighted"] is not None
    assert f_rows["cod_east"]["excluded_index_stocks"] == []
    assert f_rows["cod_east"]["ices_f_weighted"] is not None
    # Each stock reads its OWN fishery (fsh0 vs fsh8) — assert the mapping, not the values,
    # which could legitimately coincide.
    idx = validator_module._SPECIES_FSH_INDEX
    assert (idx["cod_west"], idx["cod_east"]) == (0, 8)


def test_both_cod_stocks_get_biomass_rows(report):
    """The aggregate manifest key silently dropped the cod biomass row for ten weeks
    (#182): biomass_targets.csv had cod_west/cod_east, the manifest had 'cod'.
    cod_west must now report an envelope; cod_east is index-unit (None, excluded)."""
    b_rows = {r["species"]: r for r in report["biomass_envelopes"]}
    assert "cod_west" in b_rows and "cod_east" in b_rows
    assert b_rows["cod_west"]["ices_min_ssb"] is not None
    assert b_rows["cod_west"]["envelopes_overlap"] is True
    assert b_rows["cod_east"]["ices_min_ssb"] is None
    assert "cod.27.24-32" in b_rows["cod_east"]["excluded_index_stocks"]


def test_herring_envelope_excludes_central_baltic_index_stock(report):
    """Mixed-unit herring must exclude her.27.25-2932 (index) from the SSB
    envelope sum. Otherwise a regression that accidentally sums indices into
    tonnes would produce an inflated but plausible-looking envelope."""
    b_rows = {r["species"]: r for r in report["biomass_envelopes"]}
    assert "her.27.25-2932" in b_rows["herring"]["excluded_index_stocks"]


def test_herring_f_reported_without_verdict_because_coverage_is_partial(report):
    """her.27.25-2932 publishes F only as F/FMSY, so the herring comparator is the
    other three stocks (~80% Bothnian). The F is still reported, but no verdict —
    while sprat, fully covered, keeps one: the rule must not silence every row."""
    f_rows = {r["species"]: r for r in report["f_rates"]}
    herring = f_rows["herring"]
    assert herring["ices_f_weighted"] is not None
    assert herring["in_tolerance"] is None
    assert "her.27.25-2932" in herring["no_verdict_reason"]
    assert f_rows["sprat"]["in_tolerance"] is not None
    assert f_rows["sprat"]["no_verdict_reason"] is None


def test_relative_ices_f_gets_no_verdict(report):
    """cod.27.24-32 and fle.27.2223 publish F as a relative index. Their rows report the
    value flagged relative with no verdict; absolute-F species keep theirs, so the rule
    cannot pass by silencing every row."""
    f_rows = {r["species"]: r for r in report["f_rates"]}
    for species, stock in (("cod_east", "cod.27.24-32"), ("flounder", "fle.27.2223")):
        row = f_rows[species]
        assert row["ices_f_relative"] is True, species
        assert row["in_tolerance"] is None, species
        assert stock in row["no_verdict_reason"], species
    for species in ("cod_west", "sprat"):
        assert f_rows[species]["ices_f_relative"] is False, species
        assert f_rows[species]["in_tolerance"] is not None, species
    # Herring's comparator stocks are absolute; its no-verdict is the partial-coverage one.
    assert f_rows["herring"]["ices_f_relative"] is False


def test_every_linked_stock_has_an_f_type():
    """`_f_is_relative` raises on a missing entry; pin coverage here so a new stock
    added to the manifest without its SAG F description fails at the manifest."""
    manifest = json.loads(MANIFEST.read_text())
    linked = {s for stocks in manifest["model_species_to_ices_stocks"].values() for s in stocks}
    assert linked <= set(manifest["f_type_by_stock"]), linked - set(manifest["f_type_by_stock"])


@pytest.mark.parametrize(
    ("description", "units", "relative"),
    [
        ("F", "", False),
        ("F", "ratio", True),  # her.27.25-2932
        ("F/Fmsy", "", True),  # fle.27.2223
        ("Fishing pressure: Relative", "", True),  # cod.27.24-32
        (" F ", " ", False),
    ],
)
def test_f_is_relative_classifies_sag_strings(validator_module, description, units, relative):
    manifest = {"f_type_by_stock": {"x": {"description": description, "units": units}}}
    assert validator_module._f_is_relative(manifest, "x") is relative


def test_f_is_relative_raises_on_missing_entry(validator_module):
    with pytest.raises(ValueError, match="f_type_by_stock"):
        validator_module._f_is_relative({"f_type_by_stock": {}}, "x")


def test_flounder_envelope_is_none_because_only_stock_is_index(report):
    """Flounder has a single index-unit stock. The envelope must be None,
    not zero and not a silent pass-through of the index values as tonnes."""
    b_rows = {r["species"]: r for r in report["biomass_envelopes"]}
    assert b_rows["flounder"]["ices_min_ssb"] is None
    assert "fle.27.2223" in b_rows["flounder"]["excluded_index_stocks"]


def test_f_known_exceptions_are_actually_outside_tolerance(report):
    """If a species is in F_KNOWN_EXCEPTIONS but its ratio is now within
    [0.25, 4.0], the allowlist has become dead code. Force the author to
    prune the allowlist rather than leave misleading comments in place.
    """
    present = {r["species"] for r in report["f_rates"]}
    # An allowlisted species that produces no row guards nothing (#182: the aggregate
    # 'cod' entry sat on a row the split had removed).
    assert F_KNOWN_EXCEPTIONS <= present, F_KNOWN_EXCEPTIONS - present
    # Nor does one whose ICES F is relative: the fence already skips it.
    relative = {r["species"] for r in report["f_rates"] if r["ices_f_relative"]}
    assert not F_KNOWN_EXCEPTIONS & relative, F_KNOWN_EXCEPTIONS & relative
    live_ratios: dict[str, float] = {}
    for r in report["f_rates"]:
        if r["species"] not in F_KNOWN_EXCEPTIONS:
            continue
        if r["ices_f_weighted"] is None or r["ices_f_weighted"] <= 0:
            continue
        live_ratios[r["species"]] = r["model_f"] / r["ices_f_weighted"]
    for species, ratio in live_ratios.items():
        assert ratio < 0.25 or ratio > 4.0, (
            f"{species} is allowlisted in F_KNOWN_EXCEPTIONS but ratio "
            f"{ratio:.3f} is within [0.25, 4.0] — prune the allowlist."
        )


def test_b_known_exceptions_are_actually_non_overlapping(report):
    """Same contract for biomass: an allowlisted species whose envelope
    now overlaps ICES must be pruned."""
    present = {r["species"] for r in report["biomass_envelopes"]}
    assert B_KNOWN_EXCEPTIONS <= present, B_KNOWN_EXCEPTIONS - present
    for r in report["biomass_envelopes"]:
        if r["species"] not in B_KNOWN_EXCEPTIONS:
            continue
        if r["envelopes_overlap"] is None:
            continue
        assert r["envelopes_overlap"] is False, (
            f"{r['species']} is allowlisted in B_KNOWN_EXCEPTIONS but its "
            "envelope now overlaps ICES — prune the allowlist."
        )


def test_write_report_preserves_hand_written_findings(validator_module, tmp_path, monkeypatch):
    """`--report` regenerates the tables; a hand-written 'Findings and Recommended
    Follow-ups' section already in the file must survive the rewrite (it is the document
    the allowlists cite), appended verbatim after the regenerated tables."""
    target = tmp_path / "report.md"
    target.write_text(
        "# old header\n\n| stale | table |\n\n## Findings and Recommended Follow-ups\n\n"
        "- keep me: cod_west 0.04x rationale\n"
    )
    monkeypatch.setattr(validator_module, "REPORT_MD", target)
    validator_module._write_markdown_report(validator_module.run(write_report=False))
    out = target.read_text()
    assert "stale" not in out  # tables regenerated
    assert "## Findings and Recommended Follow-ups" in out
    assert "keep me: cod_west 0.04x rationale" in out
    assert out.count("## Findings and Recommended Follow-ups") == 1
