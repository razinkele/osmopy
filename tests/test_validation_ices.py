"""Tests for osmose.validation.ices."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from osmose.validation.ices import (
    SpeciesBiomassComparison,
    compare_outputs_to_ices,
    format_markdown_report,
    load_snapshot,
    model_biomass_window_mean,
)


def _make_snapshot(tmp_path: Path) -> Path:
    """Build a minimal ICES snapshot bundle for tests."""
    d = tmp_path / "ices_snapshots"
    d.mkdir()
    manifest = {
        "advice_year": 2024,
        "model_species_to_ices_stocks": {
            "cod": ["cod.27.22-24"],
            "herring": ["her.27.20-24", "her.27.idx-only"],
            "flounder": [],
            "perch": ["per.27.idx-only"],
        },
        "units_by_stock": {
            "cod.27.22-24": "tonnes",
            "her.27.20-24": "tonnes",
            "her.27.idx-only": "index",
            "per.27.idx-only": "index",
        },
    }
    (d / "index.json").write_text(json.dumps(manifest))

    # Cod tonnes-unit assessment: SSB 100, 110, 120, 130, 140 over 2018-2022
    (d / "cod.27.22-24.assessment.json").write_text(
        json.dumps(
            [
                {"year": 2018, "ssb": 100.0, "f": 0.30},
                {"year": 2019, "ssb": 110.0, "f": 0.32},
                {"year": 2020, "ssb": 120.0, "f": 0.31},
                {"year": 2021, "ssb": 130.0, "f": 0.29},
                {"year": 2022, "ssb": 140.0, "f": 0.28},
            ]
        )
    )
    (d / "cod.27.22-24.reference_points.json").write_text(
        json.dumps(
            {
                "blim": 60.0,
                "bpa": 90.0,
                "fmsy": 0.30,
            }
        )
    )

    # Herring tonnes-unit assessment: SSB 1000-1500
    (d / "her.27.20-24.assessment.json").write_text(
        json.dumps(
            [
                {"year": 2018, "ssb": 1000.0, "f": 0.20},
                {"year": 2019, "ssb": 1100.0, "f": 0.22},
                {"year": 2020, "ssb": 1200.0, "f": 0.21},
                {"year": 2021, "ssb": 1400.0, "f": 0.19},
                {"year": 2022, "ssb": 1500.0, "f": 0.18},
            ]
        )
    )
    (d / "her.27.20-24.reference_points.json").write_text(json.dumps({}))

    # Index-unit assessments — exist but should be excluded from envelope
    (d / "her.27.idx-only.assessment.json").write_text(
        json.dumps(
            [
                {"year": 2018, "ssb": 0.85, "f": 0.20},
                {"year": 2022, "ssb": 0.95, "f": 0.18},
            ]
        )
    )
    (d / "her.27.idx-only.reference_points.json").write_text(json.dumps({}))
    (d / "per.27.idx-only.assessment.json").write_text(
        json.dumps(
            [
                {"year": 2020, "ssb": 1.2, "f": 0.10},
            ]
        )
    )
    (d / "per.27.idx-only.reference_points.json").write_text(json.dumps({}))

    return d


_SAME = object()


def _fake_results(
    biomass_by_species: dict[str, list[float]],
    ssb_by_species: dict[str, list[float]] | None | object = _SAME,
) -> MagicMock:
    """Build a mock OsmoseResults whose .biomass(species=...) / .ssb(species=...) return
    long-form dfs. By default SSB equals biomass; pass a dict for distinct SSB, or None for a run
    without SSB output (OsmoseResults(strict=False) then yields an empty frame)."""

    def _reader(by_species: dict[str, list[float]] | None):
        def _read(species: str | None = None) -> pd.DataFrame | None:
            if by_species is None:
                return pd.DataFrame()
            if species is None or species not in by_species:
                return None
            values = by_species[species]
            return pd.DataFrame(
                {
                    "species": [species] * len(values),
                    "time": list(range(len(values))),
                    "value": values,
                }
            )

        return _read

    mock = MagicMock()
    mock.biomass = _reader(biomass_by_species)
    mock.ssb = _reader(biomass_by_species if ssb_by_species is _SAME else ssb_by_species)  # type: ignore[arg-type]
    mock.output_dir = Path("/tmp/fake-results")
    return mock


class TestLoadSnapshot:
    def test_loads_manifest_and_assessments(self, tmp_path):
        d = _make_snapshot(tmp_path)
        snap = load_snapshot(d)
        assert snap.manifest["advice_year"] == 2024
        assert "cod.27.22-24" in snap.assessments
        assert "her.27.20-24" in snap.assessments
        assert "cod.27.22-24" in snap.reference_points

    def test_handles_missing_assessment_files_gracefully(self, tmp_path):
        # Create a manifest referencing a stock with no assessment file —
        # load_snapshot should silently skip it.
        d = tmp_path / "ices_snapshots"
        d.mkdir()
        (d / "index.json").write_text(
            json.dumps(
                {
                    "model_species_to_ices_stocks": {"cod": ["nonexistent.stock"]},
                    "units_by_stock": {"nonexistent.stock": "tonnes"},
                }
            )
        )
        snap = load_snapshot(d)
        assert "nonexistent.stock" not in snap.assessments


class TestModelBiomassWindowMean:
    def test_takes_mean_over_trailing_window(self):
        # 10 years of biomass output: mean of last 5 = mean(6,7,8,9,10) = 8.0
        results = _fake_results({"cod": list(range(1, 11))})
        assert model_biomass_window_mean(results, "cod", window_years=5) == 8.0

    def test_window_clamped_to_total(self):
        # Window > total: take mean of all
        results = _fake_results({"cod": [10.0, 20.0, 30.0]})
        assert model_biomass_window_mean(results, "cod", window_years=10) == 20.0

    def test_raises_on_missing_species(self):
        results = _fake_results({"cod": [100.0]})
        with pytest.raises(ValueError, match="no biomass time series"):
            model_biomass_window_mean(results, "hake", window_years=5)


def test_window_mean_reads_wide_all_species_frame():
    # eec/BoB OsmoseResults.biomass() returns a WIDE cross-species frame:
    # species are columns, and biomass(species=X) returns 0 rows because the
    # 'species' column is literally "all" (not the species name).
    wide = pd.DataFrame(
        {"Time": [0, 1, 2, 3, 4, 5], "Anchovy": [10, 10, 10, 20, 20, 20], "species": ["all"] * 6}
    )

    class R:
        output_dir = "x"

        def biomass(self, species=None):
            return wide[wide["species"] == species] if species is not None else wide

    assert model_biomass_window_mean(R(), "Anchovy", window_years=3) == 20.0


def test_window_mean_wide_frame_windows_by_calendar_year_not_row_count():
    # Distinguishes row-count windowing (WRONG, old bug) from calendar-year
    # windowing (correct): 2 rows/year, matching BoB/eec's sub-annual output
    # cadence (24 rows/year in the real config). window_years=1 must average
    # the last calendar YEAR of Time (Time > 1.0 -> rows at 1.5, 2.0 ->
    # mean(10, 100) == 55.0), NOT the last 1 ROW (which would be just 100.0).
    wide = pd.DataFrame(
        {
            "Time": [0.0, 0.5, 1.0, 1.5, 2.0],
            "Anchovy": [1, 1, 10, 10, 100],
            "species": ["all"] * 5,
        }
    )

    class R:
        output_dir = "x"

        def biomass(self, species=None):
            return wide[wide["species"] == species] if species is not None else wide

    assert model_biomass_window_mean(R(), "Anchovy", window_years=1) == 55.0


class TestCompareOutputsToIces:
    def test_in_range_species(self, tmp_path):
        # Cod ICES envelope from snapshot is [100, 140]. Model mean 120 → in range.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results(
            {
                "cod": [120.0] * 5,
                "herring": [1300.0] * 5,
                "flounder": [50.0] * 5,
                "perch": [10.0] * 5,
            }
        )
        comps = compare_outputs_to_ices(
            results, snap, window_years=5, ices_window=range(2018, 2023)
        )
        cod = next(c for c in comps if c.species == "cod")
        assert cod.model_mean_tonnes == 120.0
        assert cod.ices_min_tonnes == 100.0
        assert cod.ices_max_tonnes == 140.0
        assert cod.in_range is True
        assert cod.magnitude_factor is not None
        # 120 / sqrt(100*140) ≈ 120 / 118.32 ≈ 1.014
        assert 1.0 < cod.magnitude_factor < 1.05

    def test_overshoot_species(self, tmp_path):
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [500.0] * 5})  # well above the [100, 140] envelope
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        cod = next(c for c in comps if c.species == "cod")
        assert cod.in_range is False
        assert cod.magnitude_factor is not None
        assert cod.magnitude_factor > 4.0  # roughly 500 / 118.3

    def test_index_only_species_has_no_envelope(self, tmp_path):
        # perch is linked only to an index-unit stock — should be excluded
        # from envelope, but model mean still recorded.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"perch": [5.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        perch = next(c for c in comps if c.species == "perch")
        assert perch.model_mean_tonnes == 5.0
        assert perch.ices_min_tonnes is None
        assert perch.in_range is None
        assert "per.27.idx-only" in perch.excluded_index_stocks

    def test_no_linked_stocks_recorded_with_no_envelope(self, tmp_path):
        # flounder has empty stocks list in the manifest
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"flounder": [50.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        fl = next(c for c in comps if c.species == "flounder")
        assert fl.model_mean_tonnes == 50.0
        assert fl.in_range is None
        assert fl.excluded_index_stocks == []

    def test_mixed_units_excludes_index_stocks_from_envelope(self, tmp_path):
        # herring has both tonnes-unit (her.27.20-24) and index-unit
        # (her.27.idx-only) stocks — only the tonnes-unit one feeds the envelope.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"herring": [1300.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        her = next(c for c in comps if c.species == "herring")
        # Envelope from her.27.20-24 alone: [1000, 1500]
        assert her.ices_min_tonnes == 1000.0
        assert her.ices_max_tonnes == 1500.0
        assert her.in_range is True
        assert "her.27.idx-only" in her.excluded_index_stocks

    def test_missing_model_output_skips_species(self, tmp_path):
        # Snapshot includes cod, herring, flounder, perch — but results
        # only has cod. Other species should be silently skipped.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [120.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        species_present = {c.species for c in comps}
        assert species_present == {"cod"}

    def test_compares_model_ssb_with_the_ices_ssb_envelope(self, tmp_path):
        # The envelope is ICES SSB, so the model side must be SSB too. Biomass 500 would sit far
        # above the [100, 140] envelope; SSB 120 sits inside it. The verdict follows SSB.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [500.0] * 5}, ssb_by_species={"cod": [120.0] * 5})
        cod = next(
            c for c in compare_outputs_to_ices(results, snap, window_years=5) if c.species == "cod"
        )
        assert cod.model_quantity == "ssb"
        assert cod.model_mean_tonnes == 120.0
        assert cod.model_biomass_tonnes == 500.0  # kept as context, not compared
        assert cod.in_range is True

    def test_missing_ssb_output_gives_no_verdict_instead_of_comparing_biomass(self, tmp_path):
        # A run without output.ssb.enabled must not silently fall back to biomass-vs-SSB.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [120.0] * 5}, ssb_by_species=None)
        cod = next(
            c for c in compare_outputs_to_ices(results, snap, window_years=5) if c.species == "cod"
        )
        assert cod.in_range is None
        assert cod.magnitude_factor is None
        assert cod.model_quantity == "biomass"
        assert "output.ssb.enabled" in cod.note

    def test_biomass_quantity_is_an_explicit_opt_in(self, tmp_path):
        # quantity="biomass" keeps the old comparison for engine-vs-engine consistency checks.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [500.0] * 5}, ssb_by_species={"cod": [120.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5, quantity="biomass")
        cod = next(c for c in comps if c.species == "cod")
        assert cod.model_quantity == "biomass"
        assert cod.model_mean_tonnes == 500.0
        assert cod.in_range is False

    def test_in_memory_results_without_ssb_give_no_verdict(self, tmp_path):
        # In-memory OsmoseResults raises FileNotFoundError (not KeyError) for an output it never
        # produced; that must become a no-verdict row, not a crash of the whole report.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [120.0] * 5})
        results.ssb = MagicMock(side_effect=FileNotFoundError("no 'SSB' output"))
        cod = next(
            c for c in compare_outputs_to_ices(results, snap, window_years=5) if c.species == "cod"
        )
        assert cod.in_range is None
        assert "output.ssb.enabled" in cod.note

    def test_reads_ssb_from_real_output_files(self, tmp_path):
        # The Baltic writes both outputs as WIDE frames (Time + one column per species); the
        # biomass file carries a description line, the SSB file does not. Read them through the
        # real OsmoseResults, not a mock, so the SSB reader path itself is exercised.
        from osmose.results import OsmoseResults

        out = tmp_path / "output"
        out.mkdir()
        times = [i + 0.9583333333333334 for i in range(8)]
        (out / "osm_biomass_Simu0.csv").write_text(
            '"Mean biomass (tons), excluding first ages specified in input"\n'
            "Time,cod,herring\n" + "".join(f"{t},500.0,9000.0\n" for t in times)
        )
        (out / "osm_SSB_Simu0.csv").write_text(
            "Time,cod,herring\n" + "".join(f"{t},120.0,1300.0\n" for t in times)
        )
        snap = load_snapshot(_make_snapshot(tmp_path))
        with OsmoseResults(out, prefix="osm", strict=False) as results:
            comps = {c.species: c for c in compare_outputs_to_ices(results, snap, window_years=5)}
        assert comps["cod"].model_quantity == "ssb"
        assert comps["cod"].model_mean_tonnes == pytest.approx(120.0)
        assert comps["cod"].model_biomass_tonnes == pytest.approx(500.0)
        assert comps["cod"].in_range is True
        assert comps["herring"].model_mean_tonnes == pytest.approx(1300.0)
        assert comps["herring"].in_range is True


class TestFormatMarkdownReport:
    def test_includes_headers_and_summary(self):
        comps = [
            SpeciesBiomassComparison(
                species="cod",
                model_mean_tonnes=120.0,
                ices_min_tonnes=100.0,
                ices_max_tonnes=140.0,
                in_range=True,
                magnitude_factor=1.014,
            ),
            SpeciesBiomassComparison(
                species="sprat",
                model_mean_tonnes=50.0,
                ices_min_tonnes=200.0,
                ices_max_tonnes=300.0,
                in_range=False,
                magnitude_factor=0.20,
            ),
            SpeciesBiomassComparison(
                species="flounder",
                model_mean_tonnes=10.0,  # no envelope
            ),
        ]
        report = format_markdown_report(comps, window_years=5)
        assert "# OSMOSE outputs vs ICES SSB envelope" in report
        assert "Model window: last 5 years" in report
        # cod in range
        assert "| cod | 120 | [100, 140] | ✓ | 1.01 |" in report
        # sprat out of range
        assert "| sprat | 50 | [200, 300] | ✗ | 0.20 |" in report
        # Summary count
        assert "1/2 species in ICES SSB envelope" in report

    def test_no_verdict_row_is_not_scored_and_states_why(self):
        # A row with an envelope but no model SSB must print no verdict and not count as checked.
        comps = [
            SpeciesBiomassComparison(
                species="cod",
                model_mean_tonnes=120.0,
                ices_min_tonnes=100.0,
                ices_max_tonnes=140.0,
                in_range=True,
                magnitude_factor=1.014,
                model_quantity="ssb",
                model_biomass_tonnes=150.0,
            ),
            SpeciesBiomassComparison(
                species="herring",
                model_mean_tonnes=1500.0,
                ices_min_tonnes=800.0,
                ices_max_tonnes=900.0,
                model_quantity="biomass",
                model_biomass_tonnes=1500.0,
                note="model SSB unavailable (enable output.ssb.enabled); no verdict",
            ),
        ]
        report = format_markdown_report(comps, window_years=5)
        assert "| herring | 1,500 | [800, 900] | — | — | biomass |" in report
        assert "1/1 species in ICES SSB envelope" in report
        assert "output.ssb.enabled" in report
        assert "| cod | 120 | [100, 140] | ✓ | 1.01 | ssb | 150 |" in report

    def test_biomass_mode_footer_does_not_claim_an_ssb_comparison(self):
        # Under --quantity biomass the scored rows compare biomass with the SSB envelope; the
        # footer must say so instead of describing the SSB comparison that did not happen.
        comps = [
            SpeciesBiomassComparison(
                species="cod",
                model_mean_tonnes=120.0,
                ices_min_tonnes=100.0,
                ices_max_tonnes=140.0,
                in_range=True,
                magnitude_factor=1.014,
                model_quantity="biomass",
                model_biomass_tonnes=120.0,
            ),
        ]
        report = format_markdown_report(comps, window_years=5)
        assert "the model side is the model's SSB" not in report
        assert "not against ICES" in report

    def test_all_no_verdict_footer_does_not_claim_an_ssb_comparison(self):
        # With no SSB output anywhere, nothing was compared; the footer must not describe an SSB
        # comparison that every row contradicts (`compared` = biomass, no verdict).
        comps = [
            SpeciesBiomassComparison(
                species="cod",
                model_mean_tonnes=150.0,
                ices_min_tonnes=100.0,
                ices_max_tonnes=140.0,
                model_quantity="biomass",
                model_biomass_tonnes=150.0,
                note="model SSB unavailable (enable output.ssb.enabled); no verdict",
            ),
        ]
        report = format_markdown_report(comps, window_years=5)
        assert "the model side is the model's SSB" not in report
        assert "no model SSB output" in report


# -- CLI ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv_extra", "expected"), [([], "ssb"), (["--quantity", "biomass"], "biomass")]
)
def test_cli_passes_the_quantity_through_and_records_it(
    tmp_path, monkeypatch, argv_extra, expected
):
    import osmose.validation.ices as ices_mod
    from scripts.validate_outputs_vs_ices import main

    seen = {}

    def _compare(results, snapshot, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(ices_mod, "compare_outputs_to_ices", _compare)
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out.json"
    snap = _make_snapshot(tmp_path)
    main(
        [
            "--results-dir",
            str(results),
            "--snapshots-dir",
            str(snap),
            "--json",
            str(out),
            "--quiet",
            *argv_extra,
        ]
    )
    assert seen["quantity"] == expected
    assert json.loads(out.read_text())["quantity"] == expected


def test_cli_without_ssb_output_exits_1_and_says_how_to_fix_it(tmp_path, capsys):
    from scripts.validate_outputs_vs_ices import main

    out = tmp_path / "output"
    out.mkdir()
    (out / "osm_biomass_Simu0.csv").write_text(
        "Time,cod,herring\n" + "".join(f"{i + 0.5},120.0,1300.0\n" for i in range(6))
    )
    snap = _make_snapshot(tmp_path)
    rc = main(["--results-dir", str(out), "--snapshots-dir", str(snap), "--quiet"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No species got a verdict" in err
    assert "--quantity biomass" in err
