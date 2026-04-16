# SP-3: Time-Series Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a time-series loading framework matching Java OSMOSE 4.3.3's 7 `TimeSeries` classes, with a centralized factory for auto-detection.

**Architecture:** Single new file `osmose/engine/timeseries.py` containing 7 classes + factory function. All classes read CSV or config arrays and produce pre-expanded arrays indexed by simulation step. Three classes support cycling (repeating shorter series to fill the simulation). A `Protocol` type defines the shared interface.

**Tech Stack:** Python 3.12+, NumPy, csv stdlib, existing OSMOSE config reader separator detection.

**Spec:** `docs/superpowers/specs/2026-04-16-java-parity-full-design.md` (SP-3 section)

**Java reference:** `/home/razinka/osmose/osmose-master/java/src/main/java/fr/ird/osmose/util/timeseries/`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/engine/timeseries.py` | **create** | 7 TimeSeries classes + factory function |
| `tests/test_engine_timeseries.py` | **create** | Unit tests for all 7 types + factory |

---

### Task 1: SingleTimeSeries and GenericTimeSeries

These two are the simplest CSV readers. `SingleTimeSeries` reads CSV column[1] with cycling; `GenericTimeSeries` reads CSV column[1] without cycling.

**Files:**
- Create: `osmose/engine/timeseries.py`
- Create: `tests/test_engine_timeseries.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_timeseries.py`:

```python
"""Tests for OSMOSE time-series loading framework."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.timeseries import SingleTimeSeries, GenericTimeSeries


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.timeseries'`

- [ ] **Step 3: Implement SingleTimeSeries and GenericTimeSeries**

Create `osmose/engine/timeseries.py`:

```python
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
    def from_csv(
        cls, path: Path, ndt_per_year: int, ndt_simu: int
    ) -> SingleTimeSeries:
        raw = _read_csv_column(path)
        n = len(raw)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check osmose/engine/timeseries.py tests/test_engine_timeseries.py && .venv/bin/ruff format osmose/engine/timeseries.py tests/test_engine_timeseries.py`

```bash
git add osmose/engine/timeseries.py tests/test_engine_timeseries.py
git commit -m "feat(engine): add SingleTimeSeries and GenericTimeSeries with CSV loading"
```

---

### Task 2: ByYearTimeSeries and SeasonTimeSeries

`ByYearTimeSeries` reads one value per year from CSV, cycles if fewer years. `SeasonTimeSeries` reads from a config array OR a CSV file via SingleTimeSeries, cycles annually.

**Files:**
- Modify: `osmose/engine/timeseries.py`
- Modify: `tests/test_engine_timeseries.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_timeseries.py`:

```python
from osmose.engine.timeseries import ByYearTimeSeries, SeasonTimeSeries


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
        ts = SeasonTimeSeries.from_array(seasonal, ndt_simu=9)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py::TestByYearTimeSeries tests/test_engine_timeseries.py::TestSeasonTimeSeries -v`
Expected: FAIL — `cannot import name 'ByYearTimeSeries'`

- [ ] **Step 3: Implement ByYearTimeSeries and SeasonTimeSeries**

Add to `osmose/engine/timeseries.py`:

```python
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

    def get(self, year: int) -> float:
        return float(self.values[year])

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
        cls, seasonal_values: list[float], ndt_simu: int
    ) -> SeasonTimeSeries:
        ndt_per_year = len(seasonal_values)
        expanded = np.empty(ndt_simu, dtype=np.float64)
        for i in range(ndt_simu):
            expanded[i] = seasonal_values[i % ndt_per_year]
        return cls(expanded)

    @classmethod
    def from_csv(
        cls, path: Path, ndt_per_year: int, ndt_simu: int
    ) -> SeasonTimeSeries:
        sts = SingleTimeSeries.from_csv(path, ndt_per_year, ndt_simu)
        return cls(sts.values)

    @classmethod
    def default(cls, ndt_simu: int) -> SeasonTimeSeries:
        return cls(np.ones(ndt_simu, dtype=np.float64))

    def get(self, step: int) -> float:
        return float(self.values[step])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
git add osmose/engine/timeseries.py tests/test_engine_timeseries.py
git commit -m "feat(engine): add ByYearTimeSeries and SeasonTimeSeries"
```

---

### Task 3: ByClassTimeSeries and BySpeciesTimeSeries

`ByClassTimeSeries` reads CSV with class thresholds in header and per-class values per dt. `BySpeciesTimeSeries` is identical but with species names instead of numeric thresholds. Both cycle if shorter.

**Files:**
- Modify: `osmose/engine/timeseries.py`
- Modify: `tests/test_engine_timeseries.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_timeseries.py`:

```python
from osmose.engine.timeseries import ByClassTimeSeries, BySpeciesTimeSeries


class TestByClassTimeSeries:
    """CSV with class thresholds in header, per-class values per dt."""

    def test_reads_classes_and_values(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ts.csv"
        # Header: step, class 0 (age 0), class 1 (age 2), class 2 (age 5)
        _write_csv(csv_file, ["step", "0", "2", "5"], [
            ["0", "0.1", "0.2", "0.3"],
            ["1", "0.4", "0.5", "0.6"],
        ])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=2)
        assert ts.get_by_class(0, 0) == pytest.approx(0.1)
        assert ts.get_by_class(0, 2) == pytest.approx(0.3)
        assert ts.get_by_class(1, 1) == pytest.approx(0.5)

    def test_cycling(self, tmp_path: Path) -> None:
        """Cycles when fewer rows than ndt_simu."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "0", "5"], [
            ["0", "1.0", "2.0"],
        ])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=3)
        # Row 0 repeated: [1.0, 2.0] for steps 0, 1, 2
        assert ts.get_by_class(0, 0) == pytest.approx(1.0)
        assert ts.get_by_class(2, 1) == pytest.approx(2.0)

    def test_class_of(self, tmp_path: Path) -> None:
        """classOf maps a value to its class index."""
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "0", "2", "5"], [["0", "0.1", "0.2", "0.3"]])
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        assert ts.class_of(0.0) == 0    # >= 0, < 2
        assert ts.class_of(1.5) == 0    # >= 0, < 2
        assert ts.class_of(2.0) == 1    # >= 2, < 5
        assert ts.class_of(5.0) == 2    # >= 5 (last class)
        assert ts.class_of(-1.0) == -1  # below first threshold


class TestBySpeciesTimeSeries:
    """CSV with species names in header, per-species values per dt."""

    def test_reads_species_values(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "cod", "herring"], [
            ["0", "100.0", "200.0"],
            ["1", "150.0", "250.0"],
        ])
        ts = BySpeciesTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=2)
        assert ts.get_by_name(0, "cod") == pytest.approx(100.0)
        assert ts.get_by_name(1, "herring") == pytest.approx(250.0)

    def test_cycling(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ts.csv"
        _write_csv(csv_file, ["step", "sp0"], [["0", "42.0"]])
        ts = BySpeciesTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=3)
        assert ts.get_by_name(2, "sp0") == pytest.approx(42.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py::TestByClassTimeSeries tests/test_engine_timeseries.py::TestBySpeciesTimeSeries -v`
Expected: FAIL — `cannot import name 'ByClassTimeSeries'`

- [ ] **Step 3: Implement ByClassTimeSeries and BySpeciesTimeSeries**

Add to `osmose/engine/timeseries.py`:

```python
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
    def from_csv(
        cls, path: Path, ndt_per_year: int, ndt_simu: int
    ) -> ByClassTimeSeries:
        headers, rows = _read_csv_multicolumn(path)
        classes = np.array([float(h) for h in headers], dtype=np.float64)
        n_rows = len(rows)
        n_classes = len(classes)

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
    def from_csv(
        cls, path: Path, ndt_per_year: int, ndt_simu: int
    ) -> BySpeciesTimeSeries:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
git add osmose/engine/timeseries.py tests/test_engine_timeseries.py
git commit -m "feat(engine): add ByClassTimeSeries and BySpeciesTimeSeries"
```

---

### Task 4: ByRegimeTimeSeries

Reads from config arrays (not CSV). Shift years define regime boundaries; values array has one value per regime.

**Files:**
- Modify: `osmose/engine/timeseries.py`
- Modify: `tests/test_engine_timeseries.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_timeseries.py`:

```python
from osmose.engine.timeseries import ByRegimeTimeSeries


class TestByRegimeTimeSeries:
    """Config arrays: shift years + values. Regime changes at year boundaries."""

    def test_two_regimes(self) -> None:
        """Value changes at year 2 (step 48 with ndt_per_year=24)."""
        ts = ByRegimeTimeSeries.from_config(
            shift_years=[2], values=[0.5, 1.0], ndt_per_year=24, ndt_simu=72
        )
        # Year 0-1: value 0.5, year 2: value 1.0
        assert ts.get(0) == pytest.approx(0.5)    # step 0 = year 0
        assert ts.get(47) == pytest.approx(0.5)   # step 47 = year 1
        assert ts.get(48) == pytest.approx(1.0)   # step 48 = year 2
        assert ts.get(71) == pytest.approx(1.0)   # last step

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
        assert ts.get(0) == pytest.approx(1.0)    # year 0
        assert ts.get(24) == pytest.approx(2.0)   # year 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py::TestByRegimeTimeSeries -v`
Expected: FAIL — `cannot import name 'ByRegimeTimeSeries'`

- [ ] **Step 3: Implement ByRegimeTimeSeries**

Add to `osmose/engine/timeseries.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py -v`
Expected: All 19 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
git add osmose/engine/timeseries.py tests/test_engine_timeseries.py
git commit -m "feat(engine): add ByRegimeTimeSeries"
```

---

### Task 5: Factory Function and Protocol Verification

The `load_timeseries` factory auto-detects time-series type from config keys and returns the appropriate class.

**Files:**
- Modify: `osmose/engine/timeseries.py`
- Modify: `tests/test_engine_timeseries.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_timeseries.py`:

```python
from osmose.engine.timeseries import load_timeseries, TimeSeries


class TestLoadTimeSeries:
    """Factory function auto-detection."""

    def test_numeric_value_returns_single(self) -> None:
        """Scalar config value → SingleTimeSeries with constant."""
        config: dict[str, str] = {"mortality.fishing.rate.sp0": "0.3"}
        ts = load_timeseries(config, "mortality.fishing.rate", 0, ndt_per_year=24, ndt_simu=48)
        assert isinstance(ts, TimeSeries)
        assert ts.get(0) == pytest.approx(0.3)
        assert ts.get(47) == pytest.approx(0.3)

    def test_bydt_file_returns_generic(self, tmp_path: Path) -> None:
        """byDt.file key → GenericTimeSeries (or SingleTimeSeries for cycling)."""
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
        ts = load_timeseries(
            config, "mortality.additional.rate", 0, ndt_per_year=1, ndt_simu=1
        )
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py::TestLoadTimeSeries -v`
Expected: FAIL — `cannot import name 'load_timeseries'`

- [ ] **Step 3: Implement factory function**

Add to `osmose/engine/timeseries.py`:

```python
def load_timeseries(
    config: dict[str, str],
    key_prefix: str,
    species_idx: int,
    ndt_per_year: int,
    ndt_simu: int,
) -> TimeSeries:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_timeseries.py -v`
Expected: All 24 tests PASS

- [ ] **Step 5: Run full test suite for regression**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All existing tests still pass

- [ ] **Step 6: Lint and commit**

```bash
git add osmose/engine/timeseries.py tests/test_engine_timeseries.py
git commit -m "feat(engine): add load_timeseries() factory with auto-detection"
```
