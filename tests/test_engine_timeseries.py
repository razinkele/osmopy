"""Tests for OSMOSE time-series loading framework."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.timeseries import (
    SingleTimeSeries,
    GenericTimeSeries,
    ByYearTimeSeries,
    SeasonTimeSeries,
    ByClassTimeSeries,
    BySpeciesTimeSeries,
    ByRegimeTimeSeries,
    load_timeseries,
    TimeSeries,
)


def _write_csv(path: Path, header: list[str], rows: list[list[str]], sep: str = ";") -> None:
    """Write a simple CSV file with given separator."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


class TestSingleTimeSeries:
    """CSV reader with cycling."""

    def test_exact_length(self, tmp_path: Path) -> None:
        """CSV has exactly ndt_simu rows — no cycling needed."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "1.5"], ["1", "2.5"], ["2", "3.5"]])
        ts = SingleTimeSeries.from_csv(csv_file, ndt_per_year=3, ndt_simu=3)
        assert ts.get(0) == pytest.approx(1.5)
        assert ts.get(1) == pytest.approx(2.5)
        assert ts.get(2) == pytest.approx(3.5)

    def test_cycling(self, tmp_path: Path) -> None:
        """CSV has fewer rows than ndt_simu — values cycle."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "10.0"], ["1", "20.0"]])
        ts = SingleTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=6)
        # Expect: [10, 20, 10, 20, 10, 20]
        expected = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
        for i, v in enumerate(expected):
            assert ts.get(i) == pytest.approx(v)

    def test_truncation(self, tmp_path: Path) -> None:
        """CSV has more rows than ndt_simu — extra rows ignored."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "1.0"], ["1", "2.0"], ["2", "3.0"]])
        ts = SingleTimeSeries.from_csv(csv_file, ndt_per_year=3, ndt_simu=2)
        assert ts.get(0) == pytest.approx(1.0)
        assert ts.get(1) == pytest.approx(2.0)

    def test_values_array(self, tmp_path: Path) -> None:
        """values property returns the full pre-expanded array."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "5.0"]])
        ts = SingleTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=3)
        np.testing.assert_array_almost_equal(ts.values, [5.0, 5.0, 5.0])

    def test_rejects_partial_year(self, tmp_path: Path) -> None:
        """CSV with 3 rows but ndt_per_year=2 → not a multiple, raises ValueError."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "1.0"], ["1", "2.0"], ["2", "3.0"]])
        with pytest.raises(ValueError, match="multiple"):
            SingleTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=6)


class TestGenericTimeSeries:
    """CSV reader without cycling."""

    def test_reads_all_rows(self, tmp_path: Path) -> None:
        """Reads raw values, no cycling."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "1.0"], ["1", "2.0"], ["2", "3.0"]])
        ts = GenericTimeSeries.from_csv(csv_file)
        assert len(ts.values) == 3
        assert ts.get(0) == pytest.approx(1.0)
        assert ts.get(2) == pytest.approx(3.0)


class TestByYearTimeSeries:
    """One value per year, cycles if shorter."""

    def test_exact_years(self, tmp_path: Path) -> None:
        """CSV has exactly n_years rows."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["year", "value"], [["0", "100.0"], ["1", "200.0"]])
        ts = ByYearTimeSeries.from_csv(csv_file, n_years=2)
        assert ts.get(0) == pytest.approx(100.0)
        assert ts.get(1) == pytest.approx(200.0)

    def test_cycling_years(self, tmp_path: Path) -> None:
        """CSV has 2 years but simulation is 5 years."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["year", "value"], [["0", "10.0"], ["1", "20.0"]])
        ts = ByYearTimeSeries.from_csv(csv_file, n_years=5)
        # Cycles: [10, 20, 10, 20, 10]
        expected = [10.0, 20.0, 10.0, 20.0, 10.0]
        for i, v in enumerate(expected):
            assert ts.get(i) == pytest.approx(v)

    def test_get_for_step(self) -> None:
        """get_for_step maps simulation step to year index."""
        ts = ByYearTimeSeries(np.array([100.0, 200.0]))
        assert ts.get_for_step(0, ndt_per_year=24) == pytest.approx(100.0)  # year 0
        assert ts.get_for_step(23, ndt_per_year=24) == pytest.approx(100.0)  # still year 0
        assert ts.get_for_step(24, ndt_per_year=24) == pytest.approx(200.0)  # year 1


class TestSeasonTimeSeries:
    """Config array or CSV, repeated annually."""

    def test_from_array_cycles_annually(self) -> None:
        """Array of ndt_per_year values repeated over simulation."""
        seasonal = [0.1, 0.2, 0.3]  # 3 steps per year
        ts = SeasonTimeSeries.from_array(seasonal, ndt_per_year=3, ndt_simu=9)
        # 3 years: [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        for i in range(9):
            assert ts.get(i) == pytest.approx(seasonal[i % 3])

    def test_from_csv(self, tmp_path: Path) -> None:
        """Loads via SingleTimeSeries from CSV file."""
        csv_file = tmp_path / "season.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "0.5"], ["1", "0.3"], ["2", "0.2"]])
        ts = SeasonTimeSeries.from_csv(csv_file, ndt_per_year=3, ndt_simu=6)
        assert ts.get(0) == pytest.approx(0.5)
        assert ts.get(3) == pytest.approx(0.5)  # cycles

    def test_default_uniform(self) -> None:
        """No values provided → uniform 1.0 for all steps."""
        ts = SeasonTimeSeries.default(ndt_simu=6)
        for i in range(6):
            assert ts.get(i) == pytest.approx(1.0)


class TestByClassTimeSeries:
    """CSV with class thresholds in header, per-class values per dt."""

    def test_reads_classes_and_values(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ts.csv"
        # Header: step, class 0 (age 0), class 1 (age 2), class 2 (age 5)
        _write_csv(
            csv_file,
            ["step", "0", "2", "5"],
            [
                ["0", "0.1", "0.2", "0.3"],
                ["1", "0.4", "0.5", "0.6"],
            ],
        )
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=2)
        assert ts.get_by_class(0, 0) == pytest.approx(0.1)
        assert ts.get_by_class(0, 2) == pytest.approx(0.3)
        assert ts.get_by_class(1, 1) == pytest.approx(0.5)

    def test_cycling(self, tmp_path: Path) -> None:
        """Cycles when fewer rows than ndt_simu."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "0", "5"], [["0", "1.0", "2.0"]])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=3)
        # Row 0 repeated: [1.0, 2.0] for steps 0, 1, 2
        assert ts.get_by_class(0, 0) == pytest.approx(1.0)
        assert ts.get_by_class(2, 1) == pytest.approx(2.0)

    def test_class_of(self, tmp_path: Path) -> None:
        """classOf maps a value to its class index."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "0", "2", "5"], [["0", "0.1", "0.2", "0.3"]])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        assert ts.class_of(0.0) == 0  # >= 0, < 2
        assert ts.class_of(1.5) == 0  # >= 0, < 2
        assert ts.class_of(2.0) == 1  # >= 2, < 5
        assert ts.class_of(5.0) == 2  # >= 5 (last class)
        assert ts.class_of(-1.0) == -1  # below first threshold


class TestBySpeciesTimeSeries:
    """CSV with species names in header, per-species values per dt."""

    def test_reads_species_values(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ts.csv"
        _write_csv(
            csv_file,
            ["step", "cod", "herring"],
            [
                ["0", "100.0", "200.0"],
                ["1", "150.0", "250.0"],
            ],
        )
        ts = BySpeciesTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=2)
        assert ts.get_by_name(0, "cod") == pytest.approx(100.0)
        assert ts.get_by_name(1, "herring") == pytest.approx(250.0)

    def test_cycling(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "sp0"], [["0", "42.0"]])
        ts = BySpeciesTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=3)
        assert ts.get_by_name(2, "sp0") == pytest.approx(42.0)


class TestByRegimeTimeSeries:
    """Config arrays: shift years + values. Regime changes at year boundaries."""

    def test_two_regimes(self) -> None:
        """Value changes at year 2 (step 48 with ndt_per_year=24)."""
        ts = ByRegimeTimeSeries.from_config(
            shift_years=[2], values=[0.5, 1.0], ndt_per_year=24, ndt_simu=72
        )
        # Year 0-1: value 0.5, year 2: value 1.0
        assert ts.get(0) == pytest.approx(0.5)  # step 0 = year 0
        assert ts.get(47) == pytest.approx(0.5)  # step 47 = year 1
        assert ts.get(48) == pytest.approx(1.0)  # step 48 = year 2
        assert ts.get(71) == pytest.approx(1.0)  # last step

    def test_no_shift_constant(self) -> None:
        """No shift array → constant value for all steps."""
        ts = ByRegimeTimeSeries.from_config(
            shift_years=None, values=[3.14], ndt_per_year=24, ndt_simu=48
        )
        for i in range(48):
            assert ts.get(i) == pytest.approx(3.14)

    def test_shift_beyond_simulation(self) -> None:
        """Shift year beyond simulation length is ignored."""
        ts = ByRegimeTimeSeries.from_config(
            shift_years=[1, 100], values=[1.0, 2.0, 3.0], ndt_per_year=24, ndt_simu=48
        )
        # Only shift at year 1 matters. Year 100 is beyond ndt_simu=48 (2 years)
        assert ts.get(0) == pytest.approx(1.0)  # year 0
        assert ts.get(24) == pytest.approx(2.0)  # year 1


class TestLoadTimeSeries:
    """Factory function auto-detection."""

    def test_numeric_value_returns_single(self) -> None:
        """Scalar config value → SingleTimeSeries with constant."""
        config: dict[str, str] = {"mortality.fishing.rate.sp0": "0.3"}
        ts = load_timeseries(config, "mortality.fishing.rate", 0, ndt_per_year=24, ndt_simu=48)
        assert isinstance(ts, TimeSeries)
        assert ts.get(0) == pytest.approx(0.3)
        assert ts.get(47) == pytest.approx(0.3)

    def test_bydt_file_returns_single_with_cycling(self, tmp_path: Path) -> None:
        """byDt.file key → SingleTimeSeries (with cycling, matching Java)."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "1.0"], ["1", "2.0"]])
        config = {"mortality.fishing.rate.byDt.file.sp0": str(csv_file)}
        ts = load_timeseries(config, "mortality.fishing.rate", 0, ndt_per_year=2, ndt_simu=2)
        assert ts.get(0) == pytest.approx(1.0)
        assert ts.get(1) == pytest.approx(2.0)

    def test_byyear_file_returns_byyear(self, tmp_path: Path) -> None:
        """byYear.file key → ByYearTimeSeries."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["year", "value"], [["0", "10.0"], ["1", "20.0"]])
        config = {"mortality.fishing.rate.byYear.file.sp0": str(csv_file)}
        ts = load_timeseries(config, "mortality.fishing.rate", 0, ndt_per_year=24, ndt_simu=48)
        assert isinstance(ts, ByYearTimeSeries)

    def test_bytdt_typo_supported(self, tmp_path: Path) -> None:
        """Java typo 'bytDt' also detected (backward compat)."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "value"], [["0", "5.0"]])
        config = {"mortality.additional.rate.bytDt.file.sp0": str(csv_file)}
        ts = load_timeseries(config, "mortality.additional.rate", 0, ndt_per_year=1, ndt_simu=1)
        assert ts.get(0) == pytest.approx(5.0)

    def test_protocol_compliance(self) -> None:
        """All types satisfy the TimeSeries protocol."""
        single = SingleTimeSeries(np.array([1.0]))
        generic = GenericTimeSeries(np.array([1.0]))
        by_year = ByYearTimeSeries(np.array([1.0]))
        season = SeasonTimeSeries(np.array([1.0]))
        regime = ByRegimeTimeSeries(np.array([1.0]))
        for ts in [single, generic, by_year, season, regime]:
            assert isinstance(ts, TimeSeries)
