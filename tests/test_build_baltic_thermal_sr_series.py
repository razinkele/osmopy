import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_baltic_thermal_sr_series",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_thermal_sr_series.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_quarter_means():
    monthly = {mo: float(mo) for mo in range(1, 13)}
    assert m.quarter_mean(monthly, 3) == 8.0  # (7+8+9)/3
    assert m.quarter_mean(monthly, 4) == 11.0  # (10+11+12)/3


def test_assemble_series_layout():
    hist = {1993 + i: 10.0 + i * 0.1 for i in range(31)}
    rows = m.assemble_series(hist, tref=11.5)
    assert len(rows) == 50
    assert rows[0] == (1974, 11.5) and rows[18] == (1992, 11.5)
    assert rows[19] == (1993, 10.0) and rows[-1] == (2023, 13.0)
    years = [y for y, _ in rows]
    assert years == list(range(1974, 2024))  # contiguous — the loader requires it


def test_assemble_series_rejects_gap():
    import pytest

    hist = {1993: 10.0, 1995: 10.2}
    with pytest.raises(ValueError):
        m.assemble_series(hist, tref=10.0)


def test_write_series_csv_no_comments(tmp_path):
    p = tmp_path / "s.csv"
    rows = [(1974 + i, 7.0) for i in range(3)]
    m.write_series_csv(p, {0: rows, 1: [(y, 8.0) for y, _ in rows]})
    text = p.read_text()
    assert "#" not in text  # comments crash the loader
    assert text.splitlines()[0] == "year,temp_sp0,temp_sp1"
    assert text.splitlines()[1] == "1974,7.0,8.0"


def test_assemble_series_rejects_mismatched_first_year():
    import pytest

    hist = {1995: 10.0, 1996: 10.1, 1997: 10.2}  # first year is 1995, not 1993
    with pytest.raises(ValueError, match="does not match expected"):
        m.assemble_series(hist, tref=10.0, first_hist_year=1993)


def test_subset_bbox_3d_array():
    """Test _subset_bbox with synthetic 3D (time, lat, lon) array."""
    import numpy as np

    # Create synthetic 3D array (time=12, lat=7, lon=8)
    data = np.arange(12 * 7 * 8, dtype=float).reshape(12, 7, 8)

    # Coordinates within BBOX_SD (9.5-15.0E, 53.5-56.5N): 8 lon points (9-16E), 7 lat points (53-59N)
    lon = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    lat = np.array([53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0])

    # Subset to 9.5-15.0E, 53.5-56.5N (the hardcoded BBOX_SD)
    # Expected indices: lon 10,11,12,13,14,15 (indices 1,2,3,4,5,6) = 6 points
    #                   lat 54,55,56 (indices 1,2,3) = 3 points
    subset = m._subset_bbox(data, lon, lat)

    # Expected shape: (12, 3, 6) — 3 lat values, 6 lon values
    assert subset.shape == (12, 3, 6), f"expected (12, 3, 6), got {subset.shape}"

    # Verify values: subset[t, lat_idx, lon_idx] should correspond to original data values
    # Original data[0, 1, 1] = 1*8 + 1 = 9
    expected_val_0_0_0 = data[0, 1, 1]
    assert subset[0, 0, 0] == expected_val_0_0_0, (
        f"mismatch at [0,0,0]: {subset[0, 0, 0]} vs {expected_val_0_0_0}"
    )

    # Verify time axis is preserved: should have all 12 time slices
    assert subset.shape[0] == 12


def test_gap_detection_in_yearly_means():
    """Test that per-year gaps in means dict are detected."""
    # This simulates the gap detection logic from main()
    # If a variable's yearly means have gaps, it should be treated as DEGRADED

    # Simulate loading thetao for years 1993-2000, but 1995 and 1998 failed
    thetao_means = {
        1993: 10.0,
        1994: 10.1,
        # 1995 missing
        1996: 10.3,
        1997: 10.4,
        # 1998 missing
        1999: 10.6,
        2000: 10.7,
    }

    all_years = sorted(thetao_means.keys())
    first_year = all_years[0]
    last_year = all_years[-1]
    expected_years = list(range(first_year, last_year + 1))
    missing_years = [y for y in expected_years if y not in thetao_means]

    # Should detect missing years
    assert missing_years == [1995, 1998], f"expected [1995, 1998], got {missing_years}"
    assert len(missing_years) > 0, "gap detection failed"


def test_fallback_rows():
    """fallback_rows spans HIST_START-SPINUP_YEARS..hist_end (parameterized).

    hist_end must be an explicit argument, not a hardcoded year — the whole
    point of the parameter is that it can be made to match whatever the real
    (non-degraded) species actually loaded.
    """
    rows = m.fallback_rows(tref=8.5, hist_end=2021)

    assert len(rows) == 48, f"expected 48 rows (1974-2021), got {len(rows)}"

    years = [y for y, _ in rows]
    assert years == list(range(1974, 2022)), f"years mismatch: {years[:5]}...{years[-5:]}"

    assert all(t == 8.5 for _, t in rows), "not all temperatures equal to tref"

    assert rows[0] == (1974, 8.5)
    assert rows[-1] == (2021, 8.5)


def test_mixed_species_csv_assembly(tmp_path):
    """One real species + one fallback species, at the realistic 1993-2021 window.

    This is the scenario the round-3 bug broke: a successful loader spans
    HIST_START..DOWNLOAD_END (1993-2021, 29 historical years), so the
    degraded species' fallback_rows must be told hist_end=2021 too, not a
    hardcoded 2023 that would desync the two columns and abort the run.
    """
    # Species 0 (cod_west): real data from assemble_series, 1993-2021
    hist_sp0 = {1993 + i: 10.0 + i * 0.1 for i in range(29)}
    rows_sp0 = m.assemble_series(hist_sp0, tref=11.5)

    # Species 1 (herring): fallback (e.g., thetao succeeded but bottomT degraded)
    rows_sp1 = m.fallback_rows(tref=8.5, hist_end=2021)

    # Verify both have the same year set: 1974-2021
    years_sp0 = [y for y, _ in rows_sp0]
    years_sp1 = [y for y, _ in rows_sp1]
    assert years_sp0 == years_sp1 == list(range(1974, 2022))

    # Assemble and write
    csv_path = tmp_path / "series.csv"
    species_info = {
        0: {
            "variable": "thetao",
            "status": "OK (29 years)",
        },
        1: {
            "variable": "bottomT",
            "status": "DEGRADED: no cached files",
            "tref": "8.5 (UNSOURCED PLACEHOLDER)",
            "note": "UNSOURCED PLACEHOLDER — Task 4 must source real computed value",
        },
    }

    m.assemble_outputs(csv_path, {0: rows_sp0, 1: rows_sp1}, species_info)

    # Verify CSV
    csv_text = csv_path.read_text()
    lines = csv_text.splitlines()

    assert lines[0] == "year,temp_sp0,temp_sp1", f"header mismatch: {lines[0]}"
    assert "#" not in csv_text, "CSV contains comment lines"
    assert len(lines) == 49, f"expected 49 lines (1 header + 48 data), got {len(lines)}"

    # Verify first data row (1974)
    first_data = lines[1].split(",")
    assert first_data[0] == "1974"
    assert float(first_data[1]) == 11.5  # sp0 spinup value
    assert float(first_data[2]) == 8.5  # sp1 fallback constant

    # Verify a historical data row (1993)
    hist_line = lines[20]  # 1974 + 19 = 1993
    hist_data = hist_line.split(",")
    assert hist_data[0] == "1993"
    assert float(hist_data[1]) == 10.0  # sp0 first historical value
    assert float(hist_data[2]) == 8.5  # sp1 fallback (still constant)

    # Verify the last row lands on 2021, not a stale hardcoded 2023
    last_data = lines[-1].split(",")
    assert last_data[0] == "2021"
    assert float(last_data[2]) == 8.5

    # Verify README
    readme_path = Path(str(csv_path) + ".README.md")
    readme_text = readme_path.read_text()

    assert "Species 0" in readme_text
    assert "Species 1" in readme_text
    assert "DEGRADED" in readme_text
    assert "UNSOURCED PLACEHOLDER" in readme_text
    assert "8.5" in readme_text

    # UNSOURCED PLACEHOLDER must appear only on the degraded species (sp1),
    # not bleed into sp0's real-data block.
    sp0_block, sp1_block = readme_text.split("### Species 1")
    assert "UNSOURCED PLACEHOLDER" not in sp0_block
    assert "UNSOURCED PLACEHOLDER" in sp1_block
    assert "DEGRADED" in sp1_block


def test_mixed_species_csv_assembly_different_hist_end(tmp_path):
    """Pin that the fallback window is parameterized by hist_end, not hardcoded.

    Uses hist_end=2010 (deliberately not the production DOWNLOAD_END=2021)
    to guard against a hardcoded year constant creeping back into
    fallback_rows or the assembly path.
    """
    # Species 1 (herring) real, 1993-2010 (18 years)
    hist_sp1 = {1993 + i: 8.0 + i * 0.05 for i in range(18)}
    rows_sp1 = m.assemble_series(hist_sp1, tref=8.2)

    # Species 0 (cod_west) degraded, fallback matched to sp1's hist_end
    rows_sp0 = m.fallback_rows(tref=11.5, hist_end=2010)

    years_sp0 = [y for y, _ in rows_sp0]
    years_sp1 = [y for y, _ in rows_sp1]
    assert years_sp0 == years_sp1 == list(range(1974, 2011))

    csv_path = tmp_path / "series2.csv"
    species_info = {
        0: {
            "variable": "thetao",
            "status": "DEGRADED: no cached files",
            "tref": "11.5 (UNSOURCED PLACEHOLDER)",
        },
        1: {"variable": "bottomT", "status": "OK (18 years)"},
    }
    m.assemble_outputs(csv_path, {0: rows_sp0, 1: rows_sp1}, species_info)

    lines = csv_path.read_text().splitlines()
    assert lines[0] == "year,temp_sp0,temp_sp1"
    assert len(lines) == 1 + (2010 - 1974 + 1)  # header + 37 data rows
    assert lines[-1].split(",")[0] == "2010"
