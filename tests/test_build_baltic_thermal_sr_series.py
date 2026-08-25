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
    """Test fallback_rows returns 50 constant-value rows (1974-2023)."""
    rows = m.fallback_rows(tref=8.5)

    # Should have exactly 50 rows
    assert len(rows) == 50, f"expected 50 rows, got {len(rows)}"

    # Should span 1974-2023
    years = [y for y, _ in rows]
    assert years == list(range(1974, 2024)), f"years mismatch: {years[:5]}...{years[-5:]}"

    # All values should be the constant tref
    assert all(t == 8.5 for _, t in rows), "not all temperatures equal to tref"

    # First and last rows should be correct
    assert rows[0] == (1974, 8.5)
    assert rows[-1] == (2023, 8.5)


def test_mixed_species_csv_assembly(tmp_path):
    """Test CSV+README assembly with one real and one fallback species.

    This tests the realistic degradation scenario: one variable succeeds,
    the other uses PROVISIONAL fallback (no second assemble_series call).
    """
    # Species 0 (cod_west): real data from assemble_series
    hist_sp0 = {1993 + i: 10.0 + i * 0.1 for i in range(31)}  # 1993-2023
    rows_sp0 = m.assemble_series(hist_sp0, tref=11.5)

    # Species 1 (herring): fallback (e.g., thetao succeeded but bottomT degraded)
    rows_sp1 = m.fallback_rows(tref=8.5)

    # Verify both have same years (both should be 1974-2023)
    years_sp0 = [y for y, _ in rows_sp0]
    years_sp1 = [y for y, _ in rows_sp1]
    assert years_sp0 == years_sp1, f"species year mismatch: {len(years_sp0)} vs {len(years_sp1)}"

    # Assemble and write
    csv_path = tmp_path / "series.csv"
    species_info = {
        0: {
            "variable": "thetao",
            "status": "OK (31 years)",
        },
        1: {
            "variable": "bottomT",
            "status": "DEGRADED: no cached files",
            "tref": "8.5 (UNSOURCED PLACEHOLDER)",
        },
    }

    m.assemble_outputs(csv_path, csv_path, {0: rows_sp0, 1: rows_sp1}, species_info)

    # Verify CSV
    csv_text = csv_path.read_text()
    lines = csv_text.splitlines()

    # Header should have both species
    assert lines[0] == "year,temp_sp0,temp_sp1", f"header mismatch: {lines[0]}"

    # Should have 50 data rows (no '#' comments)
    assert "#" not in csv_text, "CSV contains comment lines"
    assert len(lines) == 51, f"expected 51 lines (1 header + 50 data), got {len(lines)}"

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

    # Verify README
    readme_path = Path(str(csv_path) + ".README.md")
    readme_text = readme_path.read_text()

    # Should mention both species
    assert "Species 0" in readme_text
    assert "Species 1" in readme_text

    # Should label sp1 as DEGRADED with UNSOURCED PLACEHOLDER
    assert "DEGRADED" in readme_text
    assert "UNSOURCED PLACEHOLDER" in readme_text
    assert "8.5" in readme_text
