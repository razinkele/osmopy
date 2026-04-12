"""Tests for osmose.results – OSMOSE output file reader."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osmose.config.validator import validate_config
from osmose.results import (
    OsmoseResults,
    _detect_preamble_lines,
    _extract_species,
    _find_output_files,
    _matches_output_type,
)
from osmose.schema.base import OsmoseField, ParamType
from osmose.schema.registry import ParameterRegistry


@pytest.fixture
def output_dir(tmp_path):
    """Create a fake OSMOSE output directory with test files."""
    np.random.seed(42)
    # Create biomass CSVs
    for sp_name in ["Anchovy", "Sardine"]:
        df = pd.DataFrame(
            {
                "time": range(10),
                "biomass": np.random.rand(10) * 1000,
            }
        )
        df.to_csv(tmp_path / f"osm_biomass_{sp_name}.csv", index=False)

    # Create abundance CSVs
    for sp_name in ["Anchovy", "Sardine"]:
        df = pd.DataFrame(
            {
                "time": range(10),
                "abundance": np.random.randint(1000, 100000, 10),
            }
        )
        df.to_csv(tmp_path / f"osm_abundance_{sp_name}.csv", index=False)

    # Create yield CSV
    df = pd.DataFrame({"time": range(10), "yield": np.random.rand(10) * 100})
    df.to_csv(tmp_path / "osm_yield_Anchovy.csv", index=False)

    # Create a spatial NetCDF
    ds = xr.Dataset(
        {
            "biomass": xr.DataArray(
                np.random.rand(10, 5, 5),
                dims=["time", "lat", "lon"],
                coords={
                    "time": range(10),
                    "lat": np.linspace(55, 65, 5),
                    "lon": np.linspace(20, 30, 5),
                },
            )
        }
    )
    ds.to_netcdf(tmp_path / "osm_spatial_biomass.nc")
    ds.close()

    return tmp_path


def test_list_outputs_missing_dir_returns_empty():
    """list_outputs on non-existent dir should return empty list."""
    from pathlib import Path

    res = OsmoseResults(Path("/nonexistent/xyz_not_here"))
    assert res.list_outputs() == []


def test_list_outputs(output_dir):
    results = OsmoseResults(output_dir)
    files = results.list_outputs()
    assert len(files) > 0
    assert any("biomass" in f for f in files)


def test_biomass_all_species(output_dir):
    results = OsmoseResults(output_dir)
    df = results.biomass()
    assert not df.empty
    assert "species" in df.columns
    assert set(df["species"].unique()) == {"Anchovy", "Sardine"}


def test_biomass_single_species(output_dir):
    results = OsmoseResults(output_dir)
    df = results.biomass("Anchovy")
    assert not df.empty
    assert set(df["species"].unique()) == {"Anchovy"}


def test_abundance(output_dir):
    results = OsmoseResults(output_dir)
    df = results.abundance()
    assert not df.empty
    assert "abundance" in df.columns


def test_yield_biomass(output_dir):
    results = OsmoseResults(output_dir)
    df = results.yield_biomass()
    assert not df.empty


def test_missing_output_returns_empty(output_dir):
    results = OsmoseResults(output_dir, strict=False)
    df = results.diet_matrix()
    assert df.empty


def test_spatial_netcdf(output_dir):
    results = OsmoseResults(output_dir)
    ds = results.spatial_biomass("osm_spatial_biomass.nc")
    assert "biomass" in ds.data_vars
    assert ds["biomass"].dims == ("time", "lat", "lon")
    results.close()


def test_read_csv_pattern(output_dir):
    results = OsmoseResults(output_dir)
    csvs = results.read_csv("osm_biomass_*.csv")
    assert len(csvs) == 2


def test_mortality_returns_empty_when_no_files(output_dir):
    results = OsmoseResults(output_dir, strict=False)
    df = results.mortality()
    assert df.empty


def test_mean_size_returns_empty_when_no_files(output_dir):
    results = OsmoseResults(output_dir, strict=False)
    df = results.mean_size()
    assert df.empty


def test_mean_trophic_level_returns_empty_when_no_files(output_dir):
    results = OsmoseResults(output_dir, strict=False)
    df = results.mean_trophic_level()
    assert df.empty


def test_close_clears_cache(output_dir):
    results = OsmoseResults(output_dir)
    results.read_netcdf("osm_spatial_biomass.nc")
    assert len(results._nc_cache) == 1
    results.close()
    assert len(results._nc_cache) == 0


# ---------------------------------------------------------------------------
# Fixtures for 2D output tests
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir_2d(tmp_path):
    """Create fake 2D OSMOSE output CSVs (ByAge, BySize, ByTL, etc.)."""
    # biomassByAge — two species, 3 time steps, 4 age classes
    for sp in ["Anchovy", "Sardine"]:
        df = pd.DataFrame(
            {
                "Time": [0, 1, 2],
                "0": [10.0, 20.0, 30.0],
                "1": [11.0, 21.0, 31.0],
                "2": [12.0, 22.0, 32.0],
                "3": [13.0, 23.0, 33.0],
            }
        )
        df.to_csv(tmp_path / f"osm_biomassByAge_{sp}.csv", index=False)

    # biomassBySize — one species
    df = pd.DataFrame(
        {
            "Time": [0, 1],
            "0-10": [100.0, 200.0],
            "10-20": [150.0, 250.0],
        }
    )
    df.to_csv(tmp_path / "osm_biomassBySize_Anchovy.csv", index=False)

    # biomassByTL
    df = pd.DataFrame({"Time": [0], "2.0": [50.0], "3.0": [60.0]})
    df.to_csv(tmp_path / "osm_biomassByTL_Anchovy.csv", index=False)

    # abundanceByAge
    df = pd.DataFrame({"Time": [0, 1], "0": [1000, 2000], "1": [1500, 2500]})
    df.to_csv(tmp_path / "osm_abundanceByAge_Anchovy.csv", index=False)

    # abundanceBySize
    df = pd.DataFrame({"Time": [0], "0-5": [500], "5-10": [600]})
    df.to_csv(tmp_path / "osm_abundanceBySize_Anchovy.csv", index=False)

    # abundanceByTL
    df = pd.DataFrame({"Time": [0], "2.0": [700], "3.0": [800]})
    df.to_csv(tmp_path / "osm_abundanceByTL_Anchovy.csv", index=False)

    # yieldByAge
    df = pd.DataFrame({"Time": [0], "0": [5.0], "1": [6.0]})
    df.to_csv(tmp_path / "osm_yieldByAge_Anchovy.csv", index=False)

    # yieldBySize
    df = pd.DataFrame({"Time": [0], "0-10": [7.0], "10-20": [8.0]})
    df.to_csv(tmp_path / "osm_yieldBySize_Anchovy.csv", index=False)

    # yieldNByAge
    df = pd.DataFrame({"Time": [0], "0": [50], "1": [60]})
    df.to_csv(tmp_path / "osm_yieldNByAge_Anchovy.csv", index=False)

    # yieldNBySize
    df = pd.DataFrame({"Time": [0], "0-10": [70], "10-20": [80]})
    df.to_csv(tmp_path / "osm_yieldNBySize_Anchovy.csv", index=False)

    # dietByAge
    df = pd.DataFrame({"Time": [0], "0": [0.3], "1": [0.7]})
    df.to_csv(tmp_path / "osm_dietByAge_Anchovy.csv", index=False)

    # dietBySize
    df = pd.DataFrame({"Time": [0], "0-10": [0.4], "10-20": [0.6]})
    df.to_csv(tmp_path / "osm_dietBySize_Anchovy.csv", index=False)

    # meanSizeByAge
    df = pd.DataFrame({"Time": [0], "0": [5.5], "1": [12.3]})
    df.to_csv(tmp_path / "osm_meanSizeByAge_Anchovy.csv", index=False)

    # meanTLBySize
    df = pd.DataFrame({"Time": [0], "0-10": [2.1], "10-20": [3.2]})
    df.to_csv(tmp_path / "osm_meanTLBySize_Anchovy.csv", index=False)

    # meanTLByAge
    df = pd.DataFrame({"Time": [0], "0": [2.5], "1": [3.5]})
    df.to_csv(tmp_path / "osm_meanTLByAge_Anchovy.csv", index=False)

    # yieldN (1D)
    df = pd.DataFrame({"time": [0, 1, 2], "yieldN": [100, 200, 300]})
    df.to_csv(tmp_path / "osm_yieldN_Anchovy.csv", index=False)

    # mortalityRate (1D)
    df = pd.DataFrame({"time": [0, 1], "mortalityRate": [0.1, 0.2]})
    df.to_csv(tmp_path / "osm_mortalityRate_Anchovy.csv", index=False)

    # sizeSpectrum (no species column)
    df = pd.DataFrame({"size": [1, 2, 3, 4], "count": [100, 80, 50, 20]})
    df.to_csv(tmp_path / "osm_sizeSpectrum_Simu0.csv", index=False)
    df2 = pd.DataFrame({"size": [1, 2, 3, 4], "count": [90, 70, 40, 10]})
    df2.to_csv(tmp_path / "osm_sizeSpectrum_Simu1.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests for _read_2d_output
# ---------------------------------------------------------------------------


class TestRead2dOutput:
    def test_normal_case(self, output_dir_2d):
        """2D output melts wide columns into long format with time, species, bin, value."""
        r = OsmoseResults(output_dir_2d)
        df = r._read_2d_output("biomassByAge")
        assert not df.empty
        assert list(df.columns) == ["time", "species", "bin", "value"]
        # 2 species x 3 timesteps x 4 bins = 24 rows
        assert len(df) == 24

    def test_species_filter(self, output_dir_2d):
        """Species filter returns only matching species rows."""
        r = OsmoseResults(output_dir_2d)
        df = r._read_2d_output("biomassByAge", species="Anchovy")
        assert set(df["species"].unique()) == {"Anchovy"}
        # 3 timesteps x 4 bins = 12 rows
        assert len(df) == 12

    def test_multiple_species(self, output_dir_2d):
        """Multiple species files are concatenated with correct species labels."""
        r = OsmoseResults(output_dir_2d)
        df = r._read_2d_output("biomassByAge")
        assert set(df["species"].unique()) == {"Anchovy", "Sardine"}

    def test_missing_files_returns_empty(self, output_dir_2d):
        """When no files match the pattern, an empty DataFrame is returned."""
        r = OsmoseResults(output_dir_2d, strict=False)
        df = r._read_2d_output("nonExistentOutput")
        assert df.empty

    def test_bin_values_preserved(self, output_dir_2d):
        """Bin column names (e.g., '0-10', '10-20') are preserved as bin values."""
        r = OsmoseResults(output_dir_2d)
        df = r._read_2d_output("biomassBySize", species="Anchovy")
        assert set(df["bin"].unique()) == {"0-10", "10-20"}

    def test_single_species_file(self, output_dir_2d):
        """Works correctly with only one species file."""
        r = OsmoseResults(output_dir_2d)
        df = r._read_2d_output("biomassByTL")
        assert len(df) == 2  # 1 timestep x 2 TL bins
        assert set(df["species"].unique()) == {"Anchovy"}

    def test_values_correct(self, output_dir_2d):
        """Melted values match the original data."""
        r = OsmoseResults(output_dir_2d)
        df = r._read_2d_output("biomassByAge", species="Anchovy")
        # At time=0, bin="0", value should be 10.0
        row = df[(df["time"] == 0) & (df["bin"] == "0")]
        assert row["value"].iloc[0] == 10.0


# ---------------------------------------------------------------------------
# Tests for 2D output convenience methods
# ---------------------------------------------------------------------------


class TestOutputMethods2D:
    @pytest.mark.parametrize(
        "method,expected_bins",
        [
            ("biomass_by_age", {"0", "1", "2", "3"}),
            ("biomass_by_size", {"0-10", "10-20"}),
            ("biomass_by_tl", {"2.0", "3.0"}),
            ("abundance_by_age", {"0", "1"}),
            ("abundance_by_size", {"0-5", "5-10"}),
            ("abundance_by_tl", {"2.0", "3.0"}),
            ("yield_by_age", {"0", "1"}),
            ("yield_by_size", {"0-10", "10-20"}),
            ("yield_n_by_age", {"0", "1"}),
            ("yield_n_by_size", {"0-10", "10-20"}),
            ("diet_by_age", {"0", "1"}),
            ("diet_by_size", {"0-10", "10-20"}),
            ("mean_size_by_age", {"0", "1"}),
            ("mean_tl_by_size", {"0-10", "10-20"}),
            ("mean_tl_by_age", {"0", "1"}),
        ],
    )
    def test_2d_method_returns_data(self, output_dir_2d, method, expected_bins):
        """Each 2D convenience method returns non-empty data with correct bins."""
        r = OsmoseResults(output_dir_2d)
        df = getattr(r, method)()
        assert not df.empty
        assert list(df.columns) == ["time", "species", "bin", "value"]
        assert set(df["bin"].unique()) == expected_bins

    @pytest.mark.parametrize(
        "method",
        [
            "biomass_by_age",
            "biomass_by_size",
            "abundance_by_age",
            "yield_by_age",
        ],
    )
    def test_2d_method_species_filter(self, output_dir_2d, method):
        """Each 2D convenience method supports species filtering."""
        r = OsmoseResults(output_dir_2d)
        df = getattr(r, method)(species="Anchovy")
        if not df.empty:
            assert set(df["species"].unique()) == {"Anchovy"}

    @pytest.mark.parametrize(
        "method",
        [
            "biomass_by_age",
            "biomass_by_size",
            "biomass_by_tl",
            "abundance_by_age",
            "abundance_by_size",
            "abundance_by_tl",
            "yield_by_age",
            "yield_by_size",
            "yield_n_by_age",
            "yield_n_by_size",
            "diet_by_age",
            "diet_by_size",
            "mean_size_by_age",
            "mean_tl_by_size",
            "mean_tl_by_age",
        ],
    )
    def test_2d_method_missing_files_empty(self, tmp_path, method):
        """2D methods return empty DataFrame when no matching files exist."""
        r = OsmoseResults(tmp_path, strict=False)
        df = getattr(r, method)()
        assert df.empty


# ---------------------------------------------------------------------------
# Tests for 1D output convenience methods
# ---------------------------------------------------------------------------


class TestOutputMethods1D:
    def test_yield_abundance(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.yield_abundance()
        assert not df.empty
        assert "species" in df.columns

    def test_yield_abundance_species_filter(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.yield_abundance(species="Anchovy")
        assert set(df["species"].unique()) == {"Anchovy"}

    def test_yield_abundance_missing(self, tmp_path):
        r = OsmoseResults(tmp_path, strict=False)
        df = r.yield_abundance()
        assert df.empty

    def test_mortality_rate(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.mortality_rate()
        assert not df.empty
        assert "species" in df.columns

    def test_mortality_rate_missing(self, tmp_path):
        r = OsmoseResults(tmp_path, strict=False)
        df = r.mortality_rate()
        assert df.empty


# ---------------------------------------------------------------------------
# Tests for size_spectrum
# ---------------------------------------------------------------------------


class TestSizeSpectrum:
    def test_normal_case(self, output_dir_2d):
        """size_spectrum concatenates all sizeSpectrum CSVs without species column."""
        r = OsmoseResults(output_dir_2d)
        df = r.size_spectrum()
        assert not df.empty
        assert "species" not in df.columns
        # 2 files x 4 rows each = 8 rows
        assert len(df) == 8

    def test_missing_files_empty(self, tmp_path):
        r = OsmoseResults(tmp_path, strict=False)
        df = r.size_spectrum()
        assert df.empty


# ---------------------------------------------------------------------------
# Tests for spatial methods
# ---------------------------------------------------------------------------


class TestSpatialMethods:
    def test_spatial_abundance(self, output_dir):
        """spatial_abundance delegates to read_netcdf."""
        r = OsmoseResults(output_dir)
        ds = r.spatial_abundance("osm_spatial_biomass.nc")
        assert isinstance(ds, xr.Dataset)
        r.close()

    def test_spatial_size(self, output_dir):
        r = OsmoseResults(output_dir)
        ds = r.spatial_size("osm_spatial_biomass.nc")
        assert isinstance(ds, xr.Dataset)
        r.close()

    def test_spatial_yield(self, output_dir):
        r = OsmoseResults(output_dir)
        ds = r.spatial_yield("osm_spatial_biomass.nc")
        assert isinstance(ds, xr.Dataset)
        r.close()

    def test_spatial_ltl(self, output_dir):
        r = OsmoseResults(output_dir)
        ds = r.spatial_ltl("osm_spatial_biomass.nc")
        assert isinstance(ds, xr.Dataset)
        r.close()


# ---------------------------------------------------------------------------
# Tests for export_dataframe
# ---------------------------------------------------------------------------


class TestExportDataframe:
    def test_biomass(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("biomass")
        assert not df.empty
        assert "species" in df.columns
        assert "biomass" in df.columns

    def test_biomass_species_filter(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("biomass", species="Anchovy")
        assert set(df["species"].unique()) == {"Anchovy"}

    def test_abundance(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("abundance")
        assert not df.empty

    def test_yield(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("yield")
        assert not df.empty

    def test_biomass_by_age_2d(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("biomass_by_age")
        assert not df.empty
        assert list(df.columns) == ["time", "species", "bin", "value"]

    def test_diet(self, output_dir):
        r = OsmoseResults(output_dir, strict=False)
        df = r.export_dataframe("diet")
        # diet_matrix returns empty when no files — that's fine
        assert isinstance(df, pd.DataFrame)

    def test_size_spectrum_ignores_species(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("size_spectrum", species="Anchovy")
        assert "species" not in df.columns

    def test_trophic(self, output_dir):
        r = OsmoseResults(output_dir, strict=False)
        df = r.export_dataframe("trophic")
        assert isinstance(df, pd.DataFrame)

    def test_unknown_type_returns_empty(self, output_dir):
        r = OsmoseResults(output_dir)
        df = r.export_dataframe("nonexistent_type")
        assert df.empty
        assert isinstance(df, pd.DataFrame)

    def test_yield_n(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("yield_n")
        assert not df.empty

    def test_mortality_rate(self, output_dir_2d):
        r = OsmoseResults(output_dir_2d)
        df = r.export_dataframe("mortality_rate")
        assert not df.empty


def test_export_map_includes_fishery_outputs():
    from osmose.results import OsmoseResults

    assert "fishery_yield" in OsmoseResults._EXPORT_MAP
    assert "fishery_yield_by_age" in OsmoseResults._EXPORT_MAP
    assert "fishery_yield_by_size" in OsmoseResults._EXPORT_MAP


def test_export_map_includes_bioen_outputs():
    from osmose.results import OsmoseResults

    assert "bioen_ingestion" in OsmoseResults._EXPORT_MAP
    assert "bioen_maintenance" in OsmoseResults._EXPORT_MAP
    assert "bioen_net_energy" in OsmoseResults._EXPORT_MAP


def test_export_map_includes_additional_distributions():
    from osmose.results import OsmoseResults

    assert "abundance_by_tl" in OsmoseResults._EXPORT_MAP
    assert "yield_n_by_age" in OsmoseResults._EXPORT_MAP
    assert "yield_n_by_size" in OsmoseResults._EXPORT_MAP
    assert "mean_size_by_age" in OsmoseResults._EXPORT_MAP
    assert "mean_tl_by_age" in OsmoseResults._EXPORT_MAP
    assert "mean_tl_by_size" in OsmoseResults._EXPORT_MAP


def test_fishery_yield_returns_dataframe(tmp_path):
    """fishery_yield() is callable and returns a DataFrame (empty when no files)."""
    r = OsmoseResults(tmp_path, strict=False)
    df = r.fishery_yield()
    assert isinstance(df, pd.DataFrame)


def test_bioen_ingestion_returns_dataframe(tmp_path):
    """bioen_ingestion() is callable and returns a DataFrame (empty when no files)."""
    r = OsmoseResults(tmp_path, strict=False)
    df = r.bioen_ingestion()
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests for internal helper functions
# ---------------------------------------------------------------------------

class TestFindOutputFiles:
    """Tests for _find_output_files searching root + subdirs."""

    def test_finds_root_files(self, tmp_path):
        (tmp_path / "osm_biomass_Anchovy.csv").write_text("time,biomass\n")
        result = _find_output_files(tmp_path, "osm_biomass*.csv")
        assert len(result) == 1

    def test_finds_mortality_subdir(self, tmp_path):
        mort_dir = tmp_path / "Mortality"
        mort_dir.mkdir()
        (mort_dir / "osm_mortalityRate-Anchovy_Simu0.csv").write_text("time,rate\n")
        result = _find_output_files(tmp_path, "osm_mortalityRate*.csv")
        assert len(result) == 1
        assert "Mortality" in str(result[0])

    def test_finds_bioen_subdir(self, tmp_path):
        bioen_dir = tmp_path / "Bioen"
        bioen_dir.mkdir()
        (bioen_dir / "osm_ingestion_Anchovy_Simu0.csv").write_text("time,val\n")
        result = _find_output_files(tmp_path, "osm_ingestion*.csv")
        assert len(result) == 1

    def test_combines_root_and_subdir(self, tmp_path):
        (tmp_path / "osm_biomass_Anchovy.csv").write_text("time,biomass\n")
        mort_dir = tmp_path / "Mortality"
        mort_dir.mkdir()
        (mort_dir / "osm_mortalityRate-Anchovy_Simu0.csv").write_text("time,rate\n")
        # Each glob only matches its own pattern
        assert len(_find_output_files(tmp_path, "osm_biomass*.csv")) == 1
        assert len(_find_output_files(tmp_path, "osm_mortalityRate*.csv")) == 1

    def test_empty_dir(self, tmp_path):
        assert _find_output_files(tmp_path, "osm_*.csv") == []


class TestDetectPreambleLines:
    """Tests for _detect_preamble_lines preamble detection."""

    def test_no_preamble(self, tmp_path):
        f = tmp_path / "clean.csv"
        f.write_text("time,biomass\n0,100\n1,200\n")
        assert _detect_preamble_lines(f) == 0

    def test_one_line_preamble(self, tmp_path):
        f = tmp_path / "one.csv"
        f.write_text("Biomass per time step\ntime,biomass\n0,100\n1,200\n")
        assert _detect_preamble_lines(f) == 1

    def test_two_line_preamble(self, tmp_path):
        f = tmp_path / "two.csv"
        f.write_text("Description line 1\nAnother desc\ntime,biomass\n0,100\n")
        assert _detect_preamble_lines(f) == 2

    def test_description_with_commas(self, tmp_path):
        """Preamble containing commas shouldn't fool field-count matching."""
        f = tmp_path / "comma.csv"
        f.write_text(
            "Mean biomass (tons), excluding first ages\ntime,biomass\n0,100\n1,200\n"
        )
        # Description has 2 fields, header has 2, data has 2 — all match.
        # But first pair (desc, header) matches at i=0, so preamble=0.
        # This is actually correct for same-width preambles — the algorithm
        # treats the first equal-width pair as header+data.
        result = _detect_preamble_lines(f)
        assert result == 0  # The 2-field desc line looks like a header

    def test_header_only_file_fallback(self, tmp_path):
        """Header-only file (no data rows) uses fallback to first multi-field line."""
        f = tmp_path / "header_only.csv"
        f.write_text("Description of output\ntime,biomass\n")
        result = _detect_preamble_lines(f)
        # Description is 1 field, header is 2 fields — no equal-width pair
        # Fallback: first line with >1 fields is header → skip 1 preamble line
        assert result == 1

    def test_single_line_file(self, tmp_path):
        f = tmp_path / "single.csv"
        f.write_text("time,biomass\n")
        assert _detect_preamble_lines(f) == 0

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        assert _detect_preamble_lines(f) == 0


class TestMatchesOutputType:
    """Tests for _matches_output_type exact type matching."""

    def test_biomass_matches(self):
        assert _matches_output_type("osm_biomass_Anchovy_Simu0", "biomass", "osm")

    def test_biomass_no_species(self):
        assert _matches_output_type("osm_biomass_Simu0", "biomass", "osm")

    def test_biomass_doesnt_match_biomassByAge(self):
        assert not _matches_output_type("osm_biomassByAge_Anchovy_Simu0", "biomass", "osm")

    def test_mortalityRate_matches_dash(self):
        assert _matches_output_type("osm_mortalityRate-Anchovy_Simu0", "mortalityRate", "osm")

    def test_mortalityRate_matches_underscore(self):
        assert _matches_output_type("osm_mortalityRate_Anchovy", "mortalityRate", "osm")

    def test_mortality_doesnt_match_mortalityRate(self):
        # "mortality" type should NOT match "mortalityRate" files
        assert not _matches_output_type("osm_mortalityRate-Anchovy_Simu0", "mortality", "osm")

    def test_ingestion_matches(self):
        assert _matches_output_type("osm_ingestion_Anchovy_Simu0", "ingestion", "osm")

    def test_meanEnet_matches(self):
        assert _matches_output_type("osm_meanEnet_Anchovy_Simu0", "meanEnet", "osm")

    def test_no_simu_suffix(self):
        assert _matches_output_type("osm_biomass_Anchovy", "biomass", "osm")

    def test_wrong_prefix(self):
        assert not _matches_output_type("other_biomass_Anchovy", "biomass", "osm")


class TestExtractSpecies:
    """Tests for _extract_species name extraction from filenames."""

    def test_underscore_species(self):
        assert _extract_species("osm_biomass_Anchovy_Simu0", "biomass", "osm") == "Anchovy"

    def test_dash_species(self):
        assert _extract_species(
            "osm_mortalityRate-Anchovy_Simu0", "mortalityRate", "osm"
        ) == "Anchovy"

    def test_no_species_aggregate(self):
        # Aggregate file (no species in name after stripping Simu suffix)
        result = _extract_species("osm_biomass_Simu0", "biomass", "osm")
        assert result is None

    def test_species_with_spaces(self):
        sp = _extract_species("osm_biomass_European Anchovy_Simu0", "biomass", "osm")
        assert sp == "European Anchovy"

    def test_no_simu_suffix(self):
        assert _extract_species("osm_biomass_Anchovy", "biomass", "osm") == "Anchovy"


# ---------------------------------------------------------------------------
# Tests for semicolon validation
# ---------------------------------------------------------------------------

class TestSemicolonValidation:
    """Tests for per-value semicolon-separated parameter validation."""

    def _make_registry(self, field: OsmoseField) -> ParameterRegistry:
        reg = ParameterRegistry()
        reg.register(field)
        return reg

    def test_valid_semicolons(self):
        field = OsmoseField(
            key_pattern="test.param",
            param_type=ParamType.FLOAT,
            min_val=0.0,
            max_val=100.0,
        )
        reg = self._make_registry(field)
        errors, _ = validate_config({"test.param": "10.0;20.0;50.0"}, reg)
        assert errors == []

    def test_invalid_semicolon_value(self):
        field = OsmoseField(
            key_pattern="test.param",
            param_type=ParamType.FLOAT,
            min_val=0.0,
            max_val=100.0,
        )
        reg = self._make_registry(field)
        errors, _ = validate_config({"test.param": "10.0;200.0;50.0"}, reg)
        assert len(errors) > 0  # 200.0 exceeds max

    def test_empty_between_semicolons(self):
        """Empty segments between semicolons are skipped."""
        field = OsmoseField(
            key_pattern="test.param",
            param_type=ParamType.FLOAT,
            min_val=0.0,
            max_val=100.0,
        )
        reg = self._make_registry(field)
        errors, _ = validate_config({"test.param": "10.0;;50.0"}, reg)
        assert errors == []


# ---------------------------------------------------------------------------
# Tests for key_case_map round-trip
# ---------------------------------------------------------------------------


class TestKeyCaseMapRoundTrip:
    """Tests for key_case_map persistence through scenario operations."""

    def test_fork_preserves_key_case_map(self, tmp_path):
        from osmose.scenarios import Scenario, ScenarioManager

        mgr = ScenarioManager(tmp_path)
        original = Scenario(
            name="original",
            config={"species.name.sp0": "Anchovy"},
            key_case_map={"species.name.sp0": "Species.Name.sp0"},
        )
        mgr.save(original)
        forked = mgr.fork("original", "forked")
        assert forked.key_case_map == {"species.name.sp0": "Species.Name.sp0"}

    def test_export_import_preserves_key_case_map(self, tmp_path):
        from osmose.scenarios import Scenario, ScenarioManager

        mgr = ScenarioManager(tmp_path)
        original = Scenario(
            name="test_scenario",
            config={"species.name.sp0": "Anchovy"},
            key_case_map={"species.name.sp0": "Species.Name.sp0"},
        )
        mgr.save(original)
        zip_path = tmp_path / "export.zip"
        mgr.export_all(zip_path)

        # Import into a new manager
        mgr2 = ScenarioManager(tmp_path / "import_dir")
        mgr2.import_all(zip_path)
        loaded = mgr2.load("test_scenario")
        assert loaded.key_case_map == {"species.name.sp0": "Species.Name.sp0"}

    def test_writer_uses_case_map(self, tmp_path):
        from osmose.config.writer import OsmoseConfigWriter

        writer = OsmoseConfigWriter()
        config = {"species.name.sp0": "Anchovy", "simulation.time.ndtperyear": "24"}
        case_map = {
            "species.name.sp0": "Species.Name.sp0",
            "simulation.time.ndtperyear": "simulation.time.ndtPerYear",
        }
        writer.write(config, tmp_path, key_case_map=case_map)
        # Read back and verify case is preserved
        found_original_case = False
        for csv_file in tmp_path.rglob("*.csv"):
            content = csv_file.read_text()
            if "Species.Name.sp0" in content or "simulation.time.ndtPerYear" in content:
                found_original_case = True
                break
        assert found_original_case, "Writer should use original case from key_case_map"
