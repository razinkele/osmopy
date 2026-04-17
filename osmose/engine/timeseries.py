# osmose/engine/timeseries.py
"""Time-series loading framework for OSMOSE engine parameters.

Matches Java OSMOSE 4.3.3 util/timeseries/ classes. Provides 7 TimeSeries
types for loading parameter values that vary over time, with a factory
function for auto-detection from config keys.

Java reference: osmose-master/java/src/main/java/fr/ird/osmose/util/timeseries/
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


def _read_csv_column(path: Path, col: int = 1) -> list[float]:
    """Read a CSV file and return values from the specified column.

    Skips the header row. Auto-detects separator (;, comma, tab).
    """
    text = path.read_text()
    first_line = text.split("\n", 1)[0]
    for sep in [";", ",", "\t"]:
        if sep in first_line:
            break
    else:
        sep = ","

    values: list[float] = []
    reader = csv.reader(text.strip().splitlines(), delimiter=sep)
    next(reader)  # skip header
    for row in reader:
        if row and len(row) > col:
            values.append(float(row[col]))
    return values


def _cycle_to_length(values: list[float], target_len: int) -> NDArray[np.float64]:
    """Cycle values array to fill target_len, matching Java cycling behavior."""
    n = len(values)
    if n >= target_len:
        return np.array(values[:target_len], dtype=np.float64)
    result = np.empty(target_len, dtype=np.float64)
    result[:n] = values
    t = n
    while t < target_len:
        for k in range(n):
            result[t] = values[k]
            t += 1
            if t == target_len:
                break
    return result


@runtime_checkable
class TimeSeries(Protocol):
    """Protocol for all time-series types."""

    def get(self, step: int) -> float: ...


class SingleTimeSeries:
    """CSV time-series with cycling.

    Reads CSV (header + data rows, column[1]). If fewer rows than ndt_simu,
    cycles the values to fill the simulation length. Matches Java
    SingleTimeSeries behavior.
    """

    def __init__(self, values: NDArray[np.float64]) -> None:
        self.values = values

    @classmethod
    def from_csv(cls, path: Path, ndt_per_year: int, ndt_simu: int) -> SingleTimeSeries:
        raw = _read_csv_column(path)
        n = len(raw)
        # Java validation: series must be at least ndt_per_year OR exactly ndt_simu
        if n != ndt_simu and n < ndt_per_year:
            raise ValueError(f"Time series in {path} has {n} steps, need at least {ndt_per_year}")
        # Java validation: must be multiple of ndt_per_year (for clean cycling)
        if n != ndt_simu and n % ndt_per_year != 0:
            raise ValueError(
                f"Time series in {path} has {n} steps, must be a multiple of {ndt_per_year}"
            )
        # Truncate if longer than simulation
        if n > ndt_simu:
            raw = raw[:ndt_simu]
        # Cycle if shorter
        expanded = _cycle_to_length(raw, ndt_simu)
        return cls(expanded)

    def get(self, step: int) -> float:
        return float(self.values[step])


class GenericTimeSeries:
    """CSV time-series without cycling.

    Reads CSV (header + data rows, column[1]). Returns raw values as-is,
    no cycling or length validation. Matches Java GenericTimeSeries.
    """

    def __init__(self, values: NDArray[np.float64]) -> None:
        self.values = values

    @classmethod
    def from_csv(cls, path: Path) -> GenericTimeSeries:
        raw = _read_csv_column(path)
        return cls(np.array(raw, dtype=np.float64))

    def get(self, step: int) -> float:
        return float(self.values[step])


class ByYearTimeSeries:
    """Per-year CSV time-series with cycling.

    Reads CSV (header + data, column[1]) with one value per year. Cycles
    if fewer years than simulation. Matches Java ByYearTimeSeries.
    """

    def __init__(self, values: NDArray[np.float64]) -> None:
        self.values = values

    @classmethod
    def from_csv(cls, path: Path, n_years: int) -> ByYearTimeSeries:
        raw = _read_csv_column(path)
        n = len(raw)
        if n > n_years:
            raw = raw[:n_years]
        expanded = _cycle_to_length(raw, n_years)
        return cls(expanded)

    def get(self, step: int) -> float:
        # For by-year series, `step` is interpreted as the year index.
        # Use `get_for_step` for conversion from simulation-step units.
        return float(self.values[step])

    def get_for_step(self, step: int, ndt_per_year: int) -> float:
        """Get value for a simulation step by mapping to year index."""
        year = step // ndt_per_year
        return float(self.values[year])


class SeasonTimeSeries:
    """Seasonal time-series repeated annually.

    Three creation modes (matching Java SeasonTimeSeries):
    1. from_array: config array of ndt_per_year values, cycled annually
    2. from_csv: load via SingleTimeSeries (with cycling)
    3. default: uniform 1.0 for all steps

    Java: reads from config array key, or delegates to SingleTimeSeries for CSV.
    """

    def __init__(self, values: NDArray[np.float64]) -> None:
        self.values = values

    @classmethod
    def from_array(
        cls, seasonal_values: list[float], ndt_per_year: int, ndt_simu: int
    ) -> SeasonTimeSeries:
        n = len(seasonal_values)
        expanded = np.empty(ndt_simu, dtype=np.float64)
        if n == ndt_per_year:
            # Annual cycle — repeat pattern (Java: tempValues.length == nStepYear)
            for i in range(ndt_simu):
                expanded[i] = seasonal_values[i % ndt_per_year]
        else:
            # Full-length or other — use directly, truncate if needed (Java: else branch)
            for i in range(min(n, ndt_simu)):
                expanded[i] = seasonal_values[i]
            # Fill remainder with last value if shorter
            if n < ndt_simu:
                expanded[n:] = seasonal_values[-1]
        return cls(expanded)

    @classmethod
    def from_csv(cls, path: Path, ndt_per_year: int, ndt_simu: int) -> SeasonTimeSeries:
        sts = SingleTimeSeries.from_csv(path, ndt_per_year, ndt_simu)
        return cls(sts.values)

    @classmethod
    def default(cls, ndt_simu: int) -> SeasonTimeSeries:
        return cls(np.ones(ndt_simu, dtype=np.float64))

    def get(self, step: int) -> float:
        return float(self.values[step])


def _read_csv_multicolumn(
    path: Path,
) -> tuple[list[str], list[list[float]]]:
    """Read CSV returning header strings (col 1+) and data rows (col 1+) as floats."""
    text = path.read_text()
    first_line = text.split("\n", 1)[0]
    for sep in [";", ",", "\t"]:
        if sep in first_line:
            break
    else:
        sep = ","

    lines = text.strip().splitlines()
    reader = csv.reader(lines, delimiter=sep)
    header_row = next(reader)
    headers = header_row[1:]  # skip first column (step/time label)

    rows: list[list[float]] = []
    for row in reader:
        if row:
            rows.append([float(v) for v in row[1:]])
    return headers, rows


class ByClassTimeSeries:
    """Per-class per-dt CSV time-series with cycling.

    CSV format: header row has class thresholds (col 1+), data rows have
    per-class values. Cycles if fewer rows than ndt_simu.
    Matches Java ByClassTimeSeries.
    """

    def __init__(
        self,
        classes: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> None:
        self.classes = classes  # shape: (n_classes,)
        self.values = values  # shape: (ndt_simu, n_classes)

    @classmethod
    def from_csv(cls, path: Path, ndt_per_year: int, ndt_simu: int) -> ByClassTimeSeries:
        headers, rows = _read_csv_multicolumn(path)
        classes = np.array([float(h) for h in headers], dtype=np.float64)
        n_rows = len(rows)
        n_classes = len(classes)

        # Java validation: min length and multiple of ndt_per_year
        if n_rows != ndt_simu and n_rows < ndt_per_year:
            raise ValueError(
                f"ByClass time series in {path} has {n_rows} steps, need at least {ndt_per_year}"
            )
        if n_rows != ndt_simu and n_rows % ndt_per_year != 0:
            raise ValueError(
                f"ByClass time series in {path} has {n_rows} steps, "
                f"must be a multiple of {ndt_per_year}"
            )

        # Truncate if longer
        if n_rows > ndt_simu:
            rows = rows[:ndt_simu]
            n_rows = ndt_simu

        # Build values array
        raw = np.zeros((n_rows, n_classes), dtype=np.float64)
        for t in range(n_rows):
            for k in range(n_classes):
                raw[t, k] = rows[t][k]

        # Cycle if shorter
        if n_rows < ndt_simu:
            expanded = np.zeros((ndt_simu, n_classes), dtype=np.float64)
            expanded[:n_rows] = raw
            t = n_rows
            while t < ndt_simu:
                for k_row in range(n_rows):
                    expanded[t] = raw[k_row]
                    t += 1
                    if t == ndt_simu:
                        break
            return cls(classes, expanded)

        return cls(classes, raw)

    def get_by_class(self, step: int, class_idx: int) -> float:
        return float(self.values[step, class_idx])

    def class_of(self, value: float) -> int:
        """Map a value to its class index. Returns -1 if below first threshold."""
        if value < self.classes[0]:
            return -1
        for k in range(len(self.classes) - 1):
            if self.classes[k] <= value < self.classes[k + 1]:
                return k
        return len(self.classes) - 1

    @property
    def n_classes(self) -> int:
        return len(self.classes)


class BySpeciesTimeSeries:
    """Per-species per-dt CSV time-series with cycling.

    Like ByClassTimeSeries but header has species names instead of numeric
    thresholds. Matches Java BySpeciesTimeSeries.
    """

    def __init__(
        self,
        names: list[str],
        values: NDArray[np.float64],
    ) -> None:
        self.names = names
        self.values = values  # shape: (ndt_simu, n_species)
        self._name_to_idx = {n: i for i, n in enumerate(names)}

    @classmethod
    def from_csv(cls, path: Path, ndt_per_year: int, ndt_simu: int) -> BySpeciesTimeSeries:
        headers, rows = _read_csv_multicolumn(path)
        n_rows = len(rows)
        n_cols = len(headers)

        if n_rows > ndt_simu:
            rows = rows[:ndt_simu]
            n_rows = ndt_simu

        raw = np.zeros((n_rows, n_cols), dtype=np.float64)
        for t in range(n_rows):
            for k in range(n_cols):
                raw[t, k] = rows[t][k]

        if n_rows < ndt_simu:
            expanded = np.zeros((ndt_simu, n_cols), dtype=np.float64)
            expanded[:n_rows] = raw
            t = n_rows
            while t < ndt_simu:
                for k_row in range(n_rows):
                    expanded[t] = raw[k_row]
                    t += 1
                    if t == ndt_simu:
                        break
            return cls(headers, expanded)

        return cls(headers, raw)

    def get_by_name(self, step: int, species_name: str) -> float:
        idx = self._name_to_idx[species_name]
        return float(self.values[step, idx])

    def get_by_index(self, step: int, species_idx: int) -> float:
        return float(self.values[step, species_idx])


class ByRegimeTimeSeries:
    """Regime-switching time-series from config arrays.

    Values change at year boundaries specified by shift array. If no shift
    array, uses constant value. Matches Java ByRegimeTimeSeries.
    """

    def __init__(self, values: NDArray[np.float64]) -> None:
        self.values = values

    @classmethod
    def from_config(
        cls,
        shift_years: list[int] | None,
        values: list[float],
        ndt_per_year: int,
        ndt_simu: int,
    ) -> ByRegimeTimeSeries:
        expanded = np.empty(ndt_simu, dtype=np.float64)

        if shift_years is None or len(shift_years) == 0:
            expanded[:] = values[0]
            return cls(expanded)

        # Convert shift years to time steps, filter out-of-range
        shifts_dt = [y * ndt_per_year for y in shift_years if y * ndt_per_year < ndt_simu]

        # Java validation: need at least nShift + 1 values
        n_regimes = len(shifts_dt) + 1
        if len(values) < n_regimes:
            raise ValueError(
                f"ByRegime needs at least {n_regimes} values for {len(shifts_dt)} shifts, "
                f"got {len(values)}"
            )

        # Build step-indexed values
        i_rate = 0
        i_shift = 0
        next_shift = shifts_dt[i_shift] if i_shift < len(shifts_dt) else ndt_simu

        for step in range(ndt_simu):
            if step >= next_shift:
                i_shift += 1
                i_rate += 1
                next_shift = shifts_dt[i_shift] if i_shift < len(shifts_dt) else ndt_simu
            expanded[step] = values[i_rate]

        return cls(expanded)

    def get(self, step: int) -> float:
        return float(self.values[step])


def load_timeseries(
    config: dict[str, str],
    key_prefix: str,
    species_idx: int,
    ndt_per_year: int,
    ndt_simu: int,
) -> SingleTimeSeries | ByYearTimeSeries | ByClassTimeSeries:
    """Auto-detect and load a TimeSeries from config keys.

    Detection order (matches Java process readParameters() patterns):
    1. byYear.file → ByYearTimeSeries
    2. byDt.file or bytDt.file → SingleTimeSeries (with cycling)
    3. byDt.byAge.file or byDt.bySize.file → ByClassTimeSeries
    4. Numeric scalar → constant SingleTimeSeries
    """
    sp = f"sp{species_idx}"

    # Check byYear.file
    key = f"{key_prefix}.byYear.file.{sp}"
    if key in config:
        n_years = ndt_simu // ndt_per_year
        return ByYearTimeSeries.from_csv(Path(config[key]), n_years)

    # Check byDt.byAge.file or byDt.bySize.file
    for variant in ["byDt.byAge", "byDt.bySize"]:
        key = f"{key_prefix}.{variant}.file.{sp}"
        if key in config:
            return ByClassTimeSeries.from_csv(Path(config[key]), ndt_per_year, ndt_simu)

    # Check byDt.file (correct) or bytDt.file (Java typo)
    for variant in ["byDt", "bytDt"]:
        key = f"{key_prefix}.{variant}.file.{sp}"
        if key in config:
            return SingleTimeSeries.from_csv(Path(config[key]), ndt_per_year, ndt_simu)

    # Check scalar value
    key = f"{key_prefix}.{sp}"
    if key in config:
        try:
            val = float(config[key])
            return SingleTimeSeries(np.full(ndt_simu, val, dtype=np.float64))
        except ValueError:
            pass

    raise KeyError(f"No time-series config found for prefix '{key_prefix}' species {species_idx}")
