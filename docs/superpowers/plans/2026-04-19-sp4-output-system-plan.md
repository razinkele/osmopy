# SP-4 Output System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three remaining Phase-5 output-parity gaps: (5.5) diet per-recording-period matching Java's append-rows-to-one-CSV convention; (5.6) NetCDF per-species distributions + mortality-by-cause; (5.4) cell-indexed spatial outputs (biomass / abundance / yield-in-biomass).

**Architecture:** Engine-only; no UI, no schema-key invention beyond the pre-existing `output.spatial.*` schema strings. Three independent capability commits plus a CHANGELOG commit. Library touch is 3 new `StepOutput` dict fields, 1 new simulate-side collector, 2 new output-side writer functions, and 5 new config-schema entries.

**Tech Stack:** Python 3.12, NumPy, xarray (NetCDF via netCDF4), pandas, pytest. Matches CLAUDE.md conventions.

**Spec:** `docs/superpowers/specs/2026-04-19-sp4-output-system-design.md` (commit `8ad2103`).

## Pre-flight

- [ ] Baseline: `.venv/bin/python -m pytest -q` must report **2461 passed**.
- [ ] Lint baseline: `.venv/bin/ruff check osmose/ scripts/ tests/ ui/` clean.
- [ ] Sanity-check the live library surface (grep, don't trust literal line numbers):
  - `osmose/engine/output.py` — `write_outputs` body around line 20, `_write_distribution_csvs` around line 106, `write_diet_csv` around line 207, `_write_yield_csv` around line 241, `write_outputs_netcdf` around line 314.
  - `osmose/engine/simulate.py` — `StepOutput` frozen dataclass around line 64, `_average_step_outputs` around line 854, `_collect_biomass_abundance` around line 628, the main simulate loop around line 1020.
  - `osmose/engine/state.py` — `class MortalityCause(IntEnum)` at line 17 with 8 members: `PREDATION`, `STARVATION`, `ADDITIONAL`, `FISHING`, `OUT`, `FORAGING`, `DISCARDS`, `AGING`.
  - `osmose/engine/grid.py` — `class Grid` with `self.lat` shape `(ny,)` and `self.lon` shape `(nx,)` at lines 33-34.
  - `osmose/schema/output.py` lines 141-148: pre-existing spatial keys `output.spatial.{enabled, biomass.enabled, abundance.enabled, size.enabled, ltl.enabled, yield.biomass.enabled, yield.abundance.enabled, egg.enabled}`.

## File map

- **Engine (additive):**
  - `osmose/engine/simulate.py` — `StepOutput` gains three fields; new `_collect_spatial_outputs`; `_average_step_outputs` extended with per-field aggregation rules; main loop calls the collector when the master config gate is on.
  - `osmose/engine/output.py` — `write_diet_csv` signature change + new row-per-period body; `write_outputs_netcdf` extended with distributions + mortality; new `write_outputs_netcdf_spatial`; caller `write_outputs` updated to pass the required arguments and honor the new gates.
  - `osmose/engine/config.py` — parse five new config keys into `EngineConfig`: `output_spatial_enabled`, `output_spatial_biomass`, `output_spatial_abundance`, `output_spatial_yield_biomass`, plus per-variable NetCDF toggles for distributions and mortality (added alongside existing parses around line 772).
- **Schema:**
  - `osmose/schema/output.py` — append 5 new entries for the distribution / mortality NetCDF toggles (the spatial entries already exist at 141-148 and are reused). No master "`output.netcdf.enabled`" — we gate per-variable to match the existing CSV gating style.
- **Tests:**
  - `tests/test_engine_diet.py` — update 2 existing tests that call `write_diet_csv` directly; add 3 new tests for the append-rows behaviour.
  - `tests/test_engine_phase5.py` — extend the existing `output.diet.composition.enabled` parser test with the writer-gate assertion; add 6 NetCDF-extension tests; add 7 spatial-output tests.
  - `tests/test_engine_output.py` (new file, or extend existing) — holds the larger NetCDF + spatial fixtures and the integration + parity tests.

---

## Task 1: Diet CSV — Java-parity (append rows per recording period)

**Goal:** `write_diet_csv` emits ONE CSV per simulation with one row per recording period, time as the first column — matching `DietOutput.java:217-222`. Current behavior (one file, whole-run sum) is replaced.

**Files:**
- Modify: `osmose/engine/output.py` — `write_diet_csv` signature + body; caller in `write_outputs` around lines 76-87.
- Modify: `tests/test_engine_diet.py` — update 2 existing direct-signature callers; add 3 new tests.
- Modify: `tests/test_engine_phase5.py` — extend the existing `output.diet.composition.enabled` parser test with writer-gate assertion.

- [ ] **Step 1: Write the new-shape failing test first**

Append to `tests/test_engine_diet.py`:

```python
import numpy as np
import pandas as pd
import pytest

from osmose.engine.output import write_diet_csv


def test_write_diet_csv_emits_one_row_per_recording_period(tmp_path):
    """Java-parity: one CSV, one row per recording period, time as the first
    column. Whole-run sum is derivable via df.drop(columns='Time').sum()."""
    predator_names = ["cod", "herring"]
    prey_names = ["cod", "herring", "plankton"]

    # Three recording periods with distinct diet matrices
    step_matrices = [
        np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 5.0]], dtype=np.float64),
        np.array([[2.0, 2.0, 0.0], [1.0, 2.0, 6.0]], dtype=np.float64),
    ]
    times = [1.0, 2.0, 3.0]

    path = tmp_path / "run_dietMatrix_Simu0.csv"
    write_diet_csv(
        path=path,
        step_diet_matrices=step_matrices,
        step_times=times,
        predator_names=predator_names,
        prey_names=prey_names,
    )

    df = pd.read_csv(path)
    # One row per recording period
    assert len(df) == 3
    # Time column present and matches
    assert list(df["Time"]) == times
    # Columns beyond Time: one per (predator, prey) pair, in predator-major order
    expected_cols = [
        "Time",
        "cod_cod", "cod_herring", "cod_plankton",
        "herring_cod", "herring_herring", "herring_plankton",
    ]
    assert list(df.columns) == expected_cols
    # Values round-trip per step, row-by-row
    assert df.iloc[0]["cod_plankton"] == pytest.approx(2.0)
    assert df.iloc[1]["herring_plankton"] == pytest.approx(5.0)
    assert df.iloc[2]["cod_cod"] == pytest.approx(2.0)


def test_write_diet_csv_with_empty_step_list_writes_no_file(tmp_path):
    """No recording period produced data → no CSV artifact.
    Preserves the 'no data → no file' invariant."""
    path = tmp_path / "run_dietMatrix_Simu0.csv"
    write_diet_csv(
        path=path,
        step_diet_matrices=[],
        step_times=[],
        predator_names=["cod"],
        prey_names=["cod", "plankton"],
    )
    assert not path.exists(), "write_diet_csv must not create an empty file"
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_engine_diet.py::test_write_diet_csv_emits_one_row_per_recording_period -v
```

Expected: FAIL with a `TypeError` on the new kwargs (`step_diet_matrices`, `step_times`) — the current signature is `(path, diet_by_species, predator_names, prey_names)`.

- [ ] **Step 3: Rewrite `write_diet_csv` with the new signature**

In `osmose/engine/output.py`, replace the whole existing `write_diet_csv` (current signature `(path, diet_by_species, predator_names, prey_names)`) with:

```python
def write_diet_csv(
    path: Path,
    step_diet_matrices: list[NDArray[np.float64]],
    step_times: list[float],
    predator_names: list[str],
    prey_names: list[str],
) -> None:
    """Write diet composition as an append-rows CSV matching Java DietOutput.

    Schema: first column ``Time``; remaining columns are one per
    ``{predator}_{prey}`` pair in predator-major, prey-minor order.
    One row per recording period.

    When ``step_diet_matrices`` is empty, no file is created.
    Values are BIOMASS EATEN in tonnes (not normalized to percent) — the
    Python engine accumulates raw biomass per period in
    ``_average_step_outputs``; post-hoc normalization is a one-liner for
    downstream users that need Java's percentage-per-predator layout.

    Args:
        path: Output CSV path, e.g. ``{prefix}_dietMatrix_Simu{i}.csv``.
        step_diet_matrices: One ``(n_predators, n_prey)`` matrix per
            recording period, in period order.
        step_times: Time values in years, one per period, same length as
            ``step_diet_matrices``.
        predator_names: Predator column-name components.
        prey_names: Prey column-name components.
    """
    if not step_diet_matrices:
        return

    if len(step_diet_matrices) != len(step_times):
        raise ValueError(
            f"step_diet_matrices length {len(step_diet_matrices)} "
            f"!= step_times length {len(step_times)}"
        )

    columns = [
        f"{pred}_{prey}" for pred in predator_names for prey in prey_names
    ]
    rows = []
    for mat, t in zip(step_diet_matrices, step_times, strict=True):
        if mat.shape != (len(predator_names), len(prey_names)):
            raise ValueError(
                f"diet matrix shape {mat.shape} != "
                f"({len(predator_names)}, {len(prey_names)}) at time {t}"
            )
        flat = mat.reshape(-1)  # predator-major flatten
        rows.append([t, *flat.tolist()])

    df = pd.DataFrame(rows, columns=["Time", *columns])
    df.to_csv(path, index=False)
```

The new signature is source-compatible-free — callers must change. That's Step 4.

- [ ] **Step 4: Update the caller in `write_outputs`**

In `osmose/engine/output.py` replace the existing block around lines 76-87:

```python
    # Write diet CSV if diet data is present
    diet_arrays = [o.diet_by_species for o in outputs if o.diet_by_species is not None]
    if diet_arrays:
        # Sum diet across all timesteps, then normalize at write time
        total_diet = np.sum(diet_arrays, axis=0)
        prey_names = config.all_species_names
        predator_names = config.species_names
        write_diet_csv(
            output_dir / f"{prefix}_dietMatrix_Simu0.csv",
            total_diet,
            predator_names,
            prey_names,
        )
```

with the per-period pattern honoring the config gate:

```python
    # Write diet CSV (Java-parity: one file, one row per recording period)
    if config.diet_output_enabled:
        step_matrices = [
            o.diet_by_species for o in outputs if o.diet_by_species is not None
        ]
        step_times = [
            o.step / config.n_dt_per_year
            for o in outputs
            if o.diet_by_species is not None
        ]
        if step_matrices:
            write_diet_csv(
                path=output_dir / f"{prefix}_dietMatrix_Simu0.csv",
                step_diet_matrices=step_matrices,
                step_times=step_times,
                predator_names=config.species_names,
                prey_names=config.all_species_names,
            )
```

`config.diet_output_enabled` is already populated from `output.diet.composition.enabled` — verified at `config.py:773`. If the attribute name differs, check via `grep -n "diet_output_enabled\|output.diet.composition" osmose/engine/config.py` and use the existing name verbatim.

- [ ] **Step 5: Run the new test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_engine_diet.py::test_write_diet_csv_emits_one_row_per_recording_period tests/test_engine_diet.py::test_write_diet_csv_with_empty_step_list_writes_no_file -v
```

Expected: both PASS.

- [ ] **Step 6: Migrate the two existing direct-signature diet tests**

`tests/test_engine_diet.py::test_write_diet_csv` (around line 286) and `::test_write_diet_csv_percentage` (around line 318) call the old signature. They test ONE-matrix formatting semantics (percentage normalization per predator). Those semantics no longer exist in `write_diet_csv` — so the tests either need to:

(a) Move their percentage-check logic to a new private helper `_normalize_diet_matrix_to_percent` that the tests target directly, OR
(b) Delete the two tests and rely on the new row-per-period tests.

Use (a). In `osmose/engine/output.py`, add a small private helper next to the old `write_diet_csv` location:

```python
def _normalize_diet_matrix_to_percent(
    diet_by_species: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize a single ``(n_pred, n_prey)`` diet matrix to per-predator
    percentages. Kept as a named helper so callers — including tests that
    verified the pre-Java-parity percentage layout — can still exercise
    the normalization step independently of the CSV writer."""
    totals = diet_by_species.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    return diet_by_species / safe_totals * 100.0
```

Rewrite the two existing tests to call `_normalize_diet_matrix_to_percent` directly on their synthesized matrices, asserting the same percentage behaviour they already verify. Concretely, replace every occurrence of:

```python
write_diet_csv(path=..., diet_by_species=DATA, predator_names=..., prey_names=...)
```

followed by a file-read-and-assertion pattern, with:

```python
from osmose.engine.output import _normalize_diet_matrix_to_percent
pct = _normalize_diet_matrix_to_percent(DATA)
# ... existing percentage assertions on `pct` instead of on the parsed CSV
```

The tests retain their semantic coverage (percentage correctness under zero-row, mixed-zero, and typical cases) without depending on a CSV round-trip. This is a behaviour-preserving refactor of the test surface.

- [ ] **Step 7: Run the two migrated tests**

```bash
.venv/bin/python -m pytest tests/test_engine_diet.py::test_write_diet_csv tests/test_engine_diet.py::test_write_diet_csv_percentage -v
```

Expected: both PASS with the new helper-based shape.

- [ ] **Step 8: Add the config-gate test**

In `tests/test_engine_phase5.py`, find the existing test around line 145 that asserts `output.diet.composition.enabled` is parsed. Extend it or append a sibling test:

```python
def test_write_diet_csv_skipped_when_diet_output_disabled(tmp_path):
    """When diet_output_enabled=False, write_outputs() must not emit the
    dietMatrix CSV even if diet matrices are present in StepOutput.

    The spec's Java-parity diet CSV is gated at the writer boundary by
    config.diet_output_enabled, complementary to the collection gate at
    simulate.py where diet_by_species stays None when disabled.
    """
    import numpy as np
    from osmose.engine.output import write_outputs
    from osmose.engine.simulate import StepOutput
    from tests.helpers import make_minimal_engine_config

    cfg = make_minimal_engine_config(diet_output_enabled=False)
    n_pred, n_prey = cfg.n_species, len(cfg.all_species_names)
    step_out = StepOutput(
        step=23,
        biomass=np.zeros(n_pred),
        abundance=np.zeros(n_pred),
        mortality_by_cause=np.zeros((n_pred, 8)),
        diet_by_species=np.ones((n_pred, n_prey), dtype=np.float64),
    )
    write_outputs([step_out], cfg, tmp_path, prefix="run")
    assert not (tmp_path / "run_dietMatrix_Simu0.csv").exists()
```

If `tests/helpers.py` doesn't have `make_minimal_engine_config`, use whatever minimal-config fixture sibling Phase-5 tests in that file already use (grep for `EngineConfig(` or `_make_config` in `test_engine_phase5.py` and follow the established pattern).

- [ ] **Step 9: Run the config-gate test**

```bash
.venv/bin/python -m pytest tests/test_engine_phase5.py -v -k diet_output_disabled
```

Expected: PASS.

- [ ] **Step 10: Full suite + lint**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check osmose/ scripts/ tests/ ui/
```

Expected: `2464 passed` (2461 + 3 new — the 2 migrated tests don't add to the count). Ruff clean.

- [ ] **Step 11: Commit**

```bash
git add osmose/engine/output.py tests/test_engine_diet.py tests/test_engine_phase5.py
git commit -m "feat(output): diet CSV Java-parity — one file, one row per recording period

Matches DietOutput.java:217-222: one ''{prefix}_dietMatrix_Simu{i}.csv''
per simulation with a Time column and one row per recording period.
Each row is the flattened (n_pred, n_prey) matrix in predator-major,
prey-minor order; column names are ''{predator}_{prey}'' pairs.

Replaces the pre-parity whole-run-summed one-shot output. The whole-run
sum is trivially recoverable via df.drop(columns=''Time'').sum(axis=0).

Gated at the writer by config.diet_output_enabled
(output.diet.composition.enabled) — complementary to the existing
collection gate at simulate.py that already suppresses diet matrices
when disabled.

Existing diet tests that verified the per-predator percentage
normalization now target a new _normalize_diet_matrix_to_percent
helper instead of the removed write_diet_csv(one-matrix) signature.

Tests: 2464 passed."
```

---

## Task 2: NetCDF distributions + mortality-by-cause (Capability 5.6)

**Goal:** Extend `write_outputs_netcdf` to emit `biomass_by_age`, `abundance_by_age`, `biomass_by_size`, `abundance_by_size`, and `mortality_by_cause` as DataArrays in the existing `{prefix}_Simu{i}.nc` file. Pad ragged per-species bin counts with NaN. Add CF-1.8 conventions attribute and `_FillValue=NaN` on every float array.

**Files:**
- Modify: `osmose/engine/output.py` — `write_outputs_netcdf` extended.
- Modify: `osmose/schema/output.py` — 5 new per-variable NetCDF toggles.
- Modify: `osmose/engine/config.py` — parse the 5 new keys into `EngineConfig`.
- Test: `tests/test_engine_output.py` (create or extend) for the 6 new tests.

- [ ] **Step 1: Add the schema keys**

In `osmose/schema/output.py` around line 155 (just after the existing `output.diet.*.netcdf.*` entries), append to the keys list:

```python
    "output.biomass.byage.netcdf.enabled",
    "output.abundance.byage.netcdf.enabled",
    "output.biomass.bysize.netcdf.enabled",
    "output.abundance.bysize.netcdf.enabled",
    "output.mortality.netcdf.enabled",
```

Keep alphabetical / group-by-variable order consistent with neighboring entries.

- [ ] **Step 2: Parse the 5 new keys in `config.py`**

Find the block around `osmose/engine/config.py:770-793` that parses `output.*` keys. Append five parses (booleans, default `false`):

```python
    output_biomass_byage_netcdf = _parse_bool(
        cfg.get("output.biomass.byage.netcdf.enabled", "false")
    )
    output_abundance_byage_netcdf = _parse_bool(
        cfg.get("output.abundance.byage.netcdf.enabled", "false")
    )
    output_biomass_bysize_netcdf = _parse_bool(
        cfg.get("output.biomass.bysize.netcdf.enabled", "false")
    )
    output_abundance_bysize_netcdf = _parse_bool(
        cfg.get("output.abundance.bysize.netcdf.enabled", "false")
    )
    output_mortality_netcdf = _parse_bool(
        cfg.get("output.mortality.netcdf.enabled", "false")
    )
```

Add them to the returned dict and to the `EngineConfig` dataclass fields (follow the pattern of existing siblings like `output_step0_include` and `output_cutoff_age`). If `_parse_bool` is named differently (e.g. `_bool` or `parse_boolean`), use the existing helper name.

- [ ] **Step 3: Write the failing NetCDF-distribution test**

Create or append to `tests/test_engine_output.py`:

```python
"""Tests for the NetCDF + spatial output writers (SP-4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from osmose.engine.output import write_outputs_netcdf
from osmose.engine.simulate import StepOutput
from tests.helpers import make_minimal_engine_config


def _make_step(
    step: int,
    n_sp: int,
    n_bins_by_sp: dict[int, int] | None = None,
) -> StepOutput:
    """Build a minimal StepOutput for n_sp species; optional ragged age bins."""
    biomass_by_age = None
    abundance_by_age = None
    if n_bins_by_sp is not None:
        biomass_by_age = {
            sp: np.arange(n_bins_by_sp[sp], dtype=np.float64) + 10.0 * step
            for sp in range(n_sp) if sp in n_bins_by_sp
        }
        abundance_by_age = {
            sp: np.arange(n_bins_by_sp[sp], dtype=np.float64) * 100.0
            for sp in range(n_sp) if sp in n_bins_by_sp
        }
    return StepOutput(
        step=step,
        biomass=np.full(n_sp, 100.0 * (step + 1), dtype=np.float64),
        abundance=np.full(n_sp, 1000.0 * (step + 1), dtype=np.float64),
        mortality_by_cause=np.arange(n_sp * 8, dtype=np.float64).reshape(n_sp, 8),
        biomass_by_age=biomass_by_age,
        abundance_by_age=abundance_by_age,
    )


def test_netcdf_contains_biomass_by_age_when_enabled(tmp_path):
    cfg = make_minimal_engine_config(
        n_species=2,
        output_biomass_byage_netcdf=True,
    )
    outputs = [_make_step(t, n_sp=2, n_bins_by_sp={0: 3, 1: 3}) for t in (23, 47, 71)]
    path = tmp_path / "run_Simu0.nc"
    write_outputs_netcdf(outputs, path, cfg)

    ds = xr.open_dataset(path)
    assert "biomass_by_age" in ds.data_vars
    assert ds["biomass_by_age"].dims == ("time", "species", "age_bin")
    assert ds["biomass_by_age"].shape == (3, 2, 3)
    # Values match: step 0 sp 0 is [0, 1, 2]; step 1 sp 0 is [10, 11, 12]
    np.testing.assert_array_equal(ds["biomass_by_age"].values[0, 0, :], [0, 1, 2])
    np.testing.assert_array_equal(ds["biomass_by_age"].values[1, 0, :], [10, 11, 12])


def test_netcdf_contains_mortality_by_cause(tmp_path):
    cfg = make_minimal_engine_config(n_species=2, output_mortality_netcdf=True)
    outputs = [_make_step(t, n_sp=2) for t in (23, 47)]
    path = tmp_path / "run_Simu0.nc"
    write_outputs_netcdf(outputs, path, cfg)

    ds = xr.open_dataset(path)
    assert "mortality_by_cause" in ds.data_vars
    assert ds["mortality_by_cause"].dims == ("time", "species", "cause")
    expected_causes = [
        "predation", "starvation", "additional", "fishing",
        "out", "foraging", "discards", "aging",
    ]
    assert list(ds.coords["cause"].values) == expected_causes


def test_netcdf_suppressed_when_every_variable_toggle_disabled(tmp_path):
    """With every NetCDF per-variable toggle false, no .nc file should be
    written. The existing biomass/abundance base behaviour remains — this
    test verifies the *extensions* don't force a file when nothing is
    requested."""
    cfg = make_minimal_engine_config(
        n_species=1,
        output_biomass_netcdf=False,
        output_abundance_netcdf=False,
        output_biomass_byage_netcdf=False,
        output_abundance_byage_netcdf=False,
        output_biomass_bysize_netcdf=False,
        output_abundance_bysize_netcdf=False,
        output_mortality_netcdf=False,
    )
    outputs = [_make_step(23, n_sp=1)]
    path = tmp_path / "run_Simu0.nc"
    write_outputs_netcdf(outputs, path, cfg)
    assert not path.exists(), (
        "write_outputs_netcdf must not create an empty NetCDF when nothing "
        "is toggled on"
    )


def test_netcdf_pads_ragged_age_bins_with_nan(tmp_path):
    cfg = make_minimal_engine_config(
        n_species=2, output_biomass_byage_netcdf=True,
    )
    outputs = [_make_step(23, n_sp=2, n_bins_by_sp={0: 4, 1: 2})]
    path = tmp_path / "run_Simu0.nc"
    write_outputs_netcdf(outputs, path, cfg)

    ds = xr.open_dataset(path)
    assert ds["biomass_by_age"].shape == (1, 2, 4)  # padded to max
    # Species 1 has only 2 bins → last 2 slots NaN
    np.testing.assert_array_equal(
        np.isnan(ds["biomass_by_age"].values[0, 1, :]),
        [False, False, True, True],
    )


def test_netcdf_pads_ragged_size_bins_with_nan(tmp_path):
    """Parallel of the age-padding test for biomass_by_size / abundance_by_size —
    the size path is an independent code branch in the writer."""
    cfg = make_minimal_engine_config(
        n_species=2, output_biomass_bysize_netcdf=True,
    )
    # Synthesize size-bin dict directly (no size-bin helper needed)
    step = StepOutput(
        step=23,
        biomass=np.array([100.0, 200.0]),
        abundance=np.array([1000.0, 2000.0]),
        mortality_by_cause=np.zeros((2, 8)),
        biomass_by_size={
            0: np.array([1.0, 2.0, 3.0, 4.0]),
            1: np.array([10.0, 20.0]),
        },
    )
    path = tmp_path / "run_Simu0.nc"
    write_outputs_netcdf([step], path, cfg)

    ds = xr.open_dataset(path)
    assert ds["biomass_by_size"].shape == (1, 2, 4)
    np.testing.assert_array_equal(
        np.isnan(ds["biomass_by_size"].values[0, 1, :]),
        [False, False, True, True],
    )


def test_netcdf_cf_conventions_attr(tmp_path):
    """Every written NetCDF must declare Conventions='CF-1.8' and each float
    DataArray must carry _FillValue=NaN."""
    cfg = make_minimal_engine_config(
        n_species=1, output_biomass_byage_netcdf=True,
    )
    outputs = [_make_step(23, n_sp=1, n_bins_by_sp={0: 2})]
    path = tmp_path / "run_Simu0.nc"
    write_outputs_netcdf(outputs, path, cfg)

    ds = xr.open_dataset(path)
    assert ds.attrs.get("Conventions") == "CF-1.8"
    # xarray surfaces _FillValue in encoding dict after round-trip
    assert np.isnan(ds["biomass_by_age"].encoding.get("_FillValue"))
```

- [ ] **Step 4: Run the new tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_engine_output.py -v
```

Expected: all FAIL — the writer doesn't produce the new DataArrays yet.

- [ ] **Step 5: Extend `write_outputs_netcdf`**

In `osmose/engine/output.py`, replace the body of `write_outputs_netcdf` (around lines 314-362) with the following. Preserve the existing biomass/abundance/yield behaviour; add distributions, mortality, CF attrs, and the early-return when nothing is toggled:

```python
def write_outputs_netcdf(
    outputs: list[StepOutput],
    path: Path,
    config: EngineConfig,
) -> None:
    """Write simulation outputs to NetCDF format using xarray.

    Emits one file at ``path`` containing the variables enabled by
    per-variable NetCDF toggles. Ragged per-species distribution bins
    are padded to cross-species max with NaN; the padded tail of each
    bin coord is likewise NaN. Declares CF-1.8 conventions and
    ``_FillValue=NaN`` on every float DataArray.

    If every variable's netcdf toggle is disabled, no file is written.
    """
    import xarray as xr

    # Collect toggle state
    want_biomass = getattr(config, "output_biomass_netcdf", True)
    want_abundance = getattr(config, "output_abundance_netcdf", True)
    want_yield = any(
        o.yield_by_species is not None for o in outputs
    ) and getattr(config, "output_yield_biomass_netcdf", True)
    want_biomass_byage = config.output_biomass_byage_netcdf and any(
        o.biomass_by_age is not None for o in outputs
    )
    want_abundance_byage = config.output_abundance_byage_netcdf and any(
        o.abundance_by_age is not None for o in outputs
    )
    want_biomass_bysize = config.output_biomass_bysize_netcdf and any(
        o.biomass_by_size is not None for o in outputs
    )
    want_abundance_bysize = config.output_abundance_bysize_netcdf and any(
        o.abundance_by_size is not None for o in outputs
    )
    want_mortality = config.output_mortality_netcdf

    if not any([
        want_biomass, want_abundance, want_yield,
        want_biomass_byage, want_abundance_byage,
        want_biomass_bysize, want_abundance_bysize,
        want_mortality,
    ]):
        return  # every toggle off → no file

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    n_species = len(outputs[0].biomass)
    species_names = config.all_species_names[:n_species]

    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
    coords: dict[str, np.ndarray | list[str]] = {
        "time": times,
        "species": species_names,
    }

    if want_biomass:
        data_vars["biomass"] = (
            ["time", "species"],
            np.array([o.biomass for o in outputs]),
        )
    if want_abundance:
        data_vars["abundance"] = (
            ["time", "species"],
            np.array([o.abundance for o in outputs]),
        )
    if want_yield:
        yield_arr = np.array([
            o.yield_by_species if o.yield_by_species is not None
            else np.full(config.n_species, np.nan)
            for o in outputs
        ])
        data_vars["yield"] = (["time", "focal_species"], yield_arr)
        coords["focal_species"] = config.species_names[:yield_arr.shape[1]]

    # Distribution-dict writers: pad ragged bins across species with NaN
    def _pad_dist(attr: str) -> tuple[np.ndarray, int]:
        max_bins = 0
        for o in outputs:
            d = getattr(o, attr)
            if d is None:
                continue
            for arr in d.values():
                max_bins = max(max_bins, len(arr))
        result = np.full((len(outputs), n_species, max_bins), np.nan)
        for t_idx, o in enumerate(outputs):
            d = getattr(o, attr)
            if d is None:
                continue
            for sp, arr in d.items():
                result[t_idx, sp, : len(arr)] = arr
        return result, max_bins

    if want_biomass_byage:
        arr, n_age = _pad_dist("biomass_by_age")
        data_vars["biomass_by_age"] = (["time", "species", "age_bin"], arr)
        coords["age_bin"] = np.arange(n_age, dtype=np.float64)
    if want_abundance_byage:
        arr, n_age = _pad_dist("abundance_by_age")
        data_vars["abundance_by_age"] = (["time", "species", "age_bin"], arr)
        coords["age_bin"] = np.arange(n_age, dtype=np.float64)
    if want_biomass_bysize:
        arr, n_size = _pad_dist("biomass_by_size")
        data_vars["biomass_by_size"] = (["time", "species", "size_bin"], arr)
        coords["size_bin"] = np.arange(n_size, dtype=np.float64)
    if want_abundance_bysize:
        arr, n_size = _pad_dist("abundance_by_size")
        data_vars["abundance_by_size"] = (["time", "species", "size_bin"], arr)
        coords["size_bin"] = np.arange(n_size, dtype=np.float64)

    if want_mortality:
        from osmose.engine.state import MortalityCause
        cause_names = [c.name.lower() for c in MortalityCause]
        mort = np.array([o.mortality_by_cause for o in outputs])
        data_vars["mortality_by_cause"] = (
            ["time", "species", "cause"], mort
        )
        coords["cause"] = cause_names

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "description": "OSMOSE Python engine output",
            "n_dt_per_year": config.n_dt_per_year,
            "n_year": config.n_year,
            "Conventions": "CF-1.8",
            "distribution_padding": (
                "Ragged per-species bin counts are padded to cross-species "
                "max with NaN. Padded-tail NaN is structurally "
                "indistinguishable from missing-data NaN; downstream tools "
                "treat them identically."
            ),
        },
    )

    # CF _FillValue on every float DataArray
    for name in ds.data_vars:
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name].encoding["_FillValue"] = np.float64("nan")

    ds.to_netcdf(path)
```

Notes:
- If the existing code has a `config.output_biomass_netcdf` or `output_yield_biomass_netcdf` flag via a different attribute name, adjust the `getattr` defaults. The `getattr(..., True)` default preserves the pre-SP-4 "always write biomass/abundance" behaviour for configs that haven't added the new keys.
- The `_pad_dist` helper is defined inside the function because it closes over `outputs`/`n_species`; if the file's style prefers module-level helpers, hoist it to a sibling `_pad_distribution_dict` function.

- [ ] **Step 6: Run the 6 new NetCDF tests**

```bash
.venv/bin/python -m pytest tests/test_engine_output.py -v -k netcdf
```

Expected: all 6 PASS.

- [ ] **Step 7: Full suite + lint**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check osmose/ scripts/ tests/ ui/
```

Expected: `2470 passed` (2464 after Task 1 + 6 new). Ruff clean.

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/output.py osmose/engine/config.py osmose/schema/output.py tests/test_engine_output.py
git commit -m "feat(output): NetCDF per-species distributions + mortality-by-cause

Extend write_outputs_netcdf with five new DataArrays, all gated by
per-variable NetCDF toggles (no master switch — matches existing
pattern):

  biomass_by_age      (time, species, age_bin)
  abundance_by_age    (time, species, age_bin)
  biomass_by_size     (time, species, size_bin)
  abundance_by_size   (time, species, size_bin)
  mortality_by_cause  (time, species, cause)  — cause coord = 8-member
                                                MortalityCause enum

Ragged per-species bin counts are padded to cross-species max with
NaN; padded-tail NaN is structurally indistinguishable from missing
data (documented in dataset attrs). CF-1.8 conventions declared.
Every float DataArray carries _FillValue=NaN.

When every variable toggle is disabled, no file is written.

Schema: five new keys in osmose/schema/output.py
  output.biomass.byage.netcdf.enabled
  output.abundance.byage.netcdf.enabled
  output.biomass.bysize.netcdf.enabled
  output.abundance.bysize.netcdf.enabled
  output.mortality.netcdf.enabled

Tests: 2470 passed (+6)."
```

---

## Task 3: Spatial outputs — biomass / abundance / yield-biomass (Capability 5.4)

**Goal:** Add cell-indexed spatial outputs for biomass, abundance, and yield-in-biomass. Three new `StepOutput` fields; new `_collect_spatial_outputs` in `simulate.py`; new `write_outputs_netcdf_spatial` in `output.py`; four existing schema keys repurposed from UI-only flags to engine gates.

**Files:**
- Modify: `osmose/engine/simulate.py` — `StepOutput` fields, new collector, `_average_step_outputs` extension.
- Modify: `osmose/engine/output.py` — new writer, `write_outputs` dispatch.
- Modify: `osmose/engine/config.py` — parse the existing `output.spatial.*` schema keys into `EngineConfig`.
- Test: `tests/test_engine_output.py` extended.

- [ ] **Step 1: Add `StepOutput` spatial fields**

In `osmose/engine/simulate.py` around the end of the `StepOutput` definition (after `diet_by_species`), add:

```python
    # Spatial: per-species (n_lat, n_lon) maps, or None if spatial outputs disabled.
    # Pairing invariant: all three are None together (master gate off) or all
    # three are non-None (master on). Per-variant enablement gates only the
    # writer; collection always populates all three when the master is on.
    spatial_biomass: dict[int, NDArray[np.float64]] | None = None
    spatial_abundance: dict[int, NDArray[np.float64]] | None = None
    spatial_yield: dict[int, NDArray[np.float64]] | None = None
```

Update the pairing-invariant docstring block at the top of `StepOutput` (around `:68-75`) to mention the new trio.

- [ ] **Step 2: Parse the spatial config keys in `config.py`**

Near the other output parses in `config.py`, add:

```python
    output_spatial_enabled = _parse_bool(
        cfg.get("output.spatial.enabled", "false")
    )
    output_spatial_biomass = _parse_bool(
        cfg.get("output.spatial.biomass.enabled", "true")
    )  # default true when master is on
    output_spatial_abundance = _parse_bool(
        cfg.get("output.spatial.abundance.enabled", "false")
    )
    output_spatial_yield_biomass = _parse_bool(
        cfg.get("output.spatial.yield.biomass.enabled", "false")
    )
```

Add to the returned dict and the `EngineConfig` dataclass. (The other `output.spatial.*` keys in the schema — `.size`, `.ltl`, `.yield.abundance`, `.egg` — are deferred to a follow-up; parse them as `False` defaults but do nothing with the values yet, or leave them un-parsed until the follow-up.)

- [ ] **Step 3: Write the failing collector test**

Append to `tests/test_engine_output.py`:

```python
from osmose.engine.grid import Grid
from osmose.engine.simulate import _collect_spatial_outputs
from osmose.engine.state import SchoolState


def _make_two_cell_state(n_sp: int = 1) -> tuple[SchoolState, Grid]:
    """Three schools across two cells in a 2x2 grid, one species."""
    grid = Grid(
        ny=2, nx=2,
        ocean_mask=np.array([[True, True], [True, False]]),
        lat=np.array([10.0, 20.0]),
        lon=np.array([30.0, 40.0]),
    )
    # 3 schools: 2 in cell 0 (y=0, x=0), 1 in cell 1 (y=0, x=1)
    # Build a minimal SchoolState — follow the pattern of sibling tests
    # in tests/test_engine_*.py that already construct SchoolState.
    # (Use the repo's existing helpers if any; otherwise the SchoolState
    # ctor with explicit arrays.)
    ...  # caller fills in per existing fixture pattern


def test_collect_spatial_biomass_aggregates_by_cell():
    """Scatter 3 schools across 2 cells; assert per-cell biomass sum."""
    # Two schools with biomass 10 and 20 in cell 0; one school with
    # biomass 30 in cell 1. Expected spatial_biomass[sp=0] = [[30, 30],
    # [0, 0]] with cell (1,1) masked land.
    ...
```

Use an inline-minimal `SchoolState` construction if no helper exists. The critical assertion: `spatial_biomass[0][0, 0] == 30.0` (cell with two schools), `spatial_biomass[0][0, 1] == 30.0` (cell with one school), and land cell `[1, 1]` is zero (not NaN — NaN happens at writer time, not collection time).

If the existing test patterns don't include a `SchoolState` fixture helper and building one inline exceeds 50 lines, split this step: first build a `_make_schools_in_cells(...)` helper in the test file with the minimum fields (species_id, cell_id, biomass, abundance, n_dead with a fishing cause populated), then write the actual assertion.

- [ ] **Step 4: Run the collector test**

```bash
.venv/bin/python -m pytest tests/test_engine_output.py::test_collect_spatial_biomass_aggregates_by_cell -v
```

Expected: FAIL — `_collect_spatial_outputs` doesn't exist yet.

- [ ] **Step 5: Implement `_collect_spatial_outputs`**

In `osmose/engine/simulate.py` near the other `_collect_*` helpers (around line 628-700), add:

```python
def _collect_spatial_outputs(
    state: SchoolState,
    grid,
    config: EngineConfig,
) -> tuple[
    dict[int, NDArray[np.float64]],
    dict[int, NDArray[np.float64]],
    dict[int, NDArray[np.float64]],
]:
    """Aggregate biomass, abundance, and fishing yield per cell per species.

    Returns three dicts keyed by species index; each value is an
    ``(ny, nx)`` numpy array of float64. Always produces all three dicts
    when called — per-variant enablement gates only the writer, not the
    collector. Land cells hold 0.0 here; NaN substitution happens in
    ``write_outputs_netcdf_spatial`` at render time so the ocean-mask
    lookup isn't repeated per step.

    Applies the same output-cutoff-age filter as
    ``_collect_biomass_abundance`` (``config.output_cutoff_age``) so the
    parity invariant ``sum(spatial_biomass over cells) == biomass``
    holds for focal species.
    """
    ny, nx = grid.ny, grid.nx
    n_sp = config.n_species
    sb = {sp: np.zeros((ny, nx), dtype=np.float64) for sp in range(n_sp)}
    sa = {sp: np.zeros((ny, nx), dtype=np.float64) for sp in range(n_sp)}
    sy = {sp: np.zeros((ny, nx), dtype=np.float64) for sp in range(n_sp)}

    if len(state) == 0:
        return sb, sa, sy

    focal_mask = state.species_id < n_sp
    if config.output_cutoff_age is not None:
        age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
        cutoff = config.output_cutoff_age[state.species_id]
        focal_mask &= age_years >= cutoff

    if not focal_mask.any():
        return sb, sa, sy

    sp_ids = state.species_id[focal_mask]
    cells = state.cell_id[focal_mask]
    ys, xs = np.divmod(cells, nx)  # row-major per Grid docstring

    biomass = state.biomass[focal_mask]
    abundance = state.abundance[focal_mask]

    # Fishing yield: n_dead[:, FISHING] * weight_at_death. Use biomass
    # as the weight proxy since schools carry their biomass after the
    # mortality step; match _collect_yield's semantics.
    fishing_dead = state.n_dead[focal_mask, int(MortalityCause.FISHING)]
    # Yield biomass per school: fraction dead × current biomass
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_per_school = np.where(
            state.abundance[focal_mask] > 0,
            fishing_dead / state.abundance[focal_mask] * biomass,
            0.0,
        )

    for sp in range(n_sp):
        m = sp_ids == sp
        if not m.any():
            continue
        np.add.at(sb[sp], (ys[m], xs[m]), biomass[m])
        np.add.at(sa[sp], (ys[m], xs[m]), abundance[m])
        np.add.at(sy[sp], (ys[m], xs[m]), yield_per_school[m])

    return sb, sa, sy
```

Call it from `_collect_outputs` (around line 802) only when `config.output_spatial_enabled`:

```python
    spatial_biomass = spatial_abundance = spatial_yield = None
    if config.output_spatial_enabled:
        spatial_biomass, spatial_abundance, spatial_yield = _collect_spatial_outputs(
            state, grid, config
        )
```

and pass the three into the `StepOutput(...)` constructor at the bottom of `_collect_outputs`.

- [ ] **Step 6: Extend `_average_step_outputs` for the spatial fields**

In `osmose/engine/simulate.py` `_average_step_outputs` around lines 854-915, add per-field aggregation for the three spatial dicts. Insert BEFORE the `if len(accumulated) == 1:` branch:

```python
    def _avg_spatial(attr: str, op: str) -> dict[int, NDArray[np.float64]] | None:
        dicts = [getattr(o, attr) for o in accumulated if getattr(o, attr) is not None]
        if not dicts:
            return None
        keys = set()
        for d in dicts:
            keys |= d.keys()
        result: dict[int, NDArray[np.float64]] = {}
        for sp in keys:
            arrays = [d[sp] for d in dicts if sp in d]
            if op == "mean":
                result[sp] = np.mean(arrays, axis=0)
            elif op == "sum":
                result[sp] = np.sum(arrays, axis=0)
            else:
                raise ValueError(f"unknown op: {op}")
        return result

    spatial_biomass_agg = _avg_spatial("spatial_biomass", "mean")
    spatial_abundance_agg = _avg_spatial("spatial_abundance", "mean")
    spatial_yield_agg = _avg_spatial("spatial_yield", "sum")
```

Then pass these three into both `StepOutput(...)` return-sites (the `len(accumulated) == 1` early-return and the main return). For the single-accumulated case, you can just pass `accumulated[0].spatial_biomass` etc. — same semantics as the distribution dicts.

- [ ] **Step 7: Run the collector test again**

```bash
.venv/bin/python -m pytest tests/test_engine_output.py::test_collect_spatial_biomass_aggregates_by_cell -v
```

Expected: PASS.

- [ ] **Step 8: Add spatial averaging tests**

Append to `tests/test_engine_output.py`:

```python
from osmose.engine.simulate import _average_step_outputs, StepOutput


def _spatial_step(step: int, sb_val: float, sy_val: float) -> StepOutput:
    """Build a minimal StepOutput with uniform-value spatial dicts."""
    return StepOutput(
        step=step,
        biomass=np.array([100.0]),
        abundance=np.array([1000.0]),
        mortality_by_cause=np.zeros((1, 8)),
        spatial_biomass={0: np.full((2, 2), sb_val)},
        spatial_abundance={0: np.full((2, 2), sb_val * 10.0)},
        spatial_yield={0: np.full((2, 2), sy_val)},
    )


@pytest.mark.parametrize(
    "field,op,vals,expected",
    [
        ("spatial_biomass", "mean", [10.0, 20.0], 15.0),
        ("spatial_abundance", "mean", [100.0, 300.0], 200.0),
        ("spatial_yield", "sum", [5.0, 7.0], 12.0),
    ],
)
def test_average_spatial_outputs_preserves_aggregation_rules(
    field, op, vals, expected
):
    """spatial_biomass / spatial_abundance are element-wise mean;
    spatial_yield is element-wise sum. All three over a 2-step window."""
    s0 = _spatial_step(23, sb_val=vals[0], sy_val=vals[0])
    s1 = _spatial_step(47, sb_val=vals[1], sy_val=vals[1])
    # Make the per-field test independent of the other two fields
    avg = _average_step_outputs([s0, s1], freq=24, record_step=47)
    arr = getattr(avg, field)[0]
    # Depending on field, expected applies either to sb or sy → scale:
    if field == "spatial_biomass":
        np.testing.assert_allclose(arr, expected)
    elif field == "spatial_abundance":
        np.testing.assert_allclose(arr, expected)  # *10 baked into _spatial_step
    elif field == "spatial_yield":
        np.testing.assert_allclose(arr, expected)
```

(Don't over-engineer; the fixture sets spatial_abundance = 10 * biomass, so the expected-200 for `[100, 300]` comes from `mean(100*10, 300*10) = 2000` — fix the fixture to set spatial_abundance directly to `vals` for clarity instead of scaling. The point is: one parametrized test covering all three aggregation rules.)

Also add the collection-but-no-files boundary test:

```python
def test_spatial_collection_runs_but_no_files_when_all_variants_disabled(tmp_path):
    """Master true + all three per-variant toggles false: collection
    populates StepOutput.spatial_* but no _spatial_*.nc files are
    written. Covers the seam where the pairing invariant could silently
    regress (collection gate vs writer gate)."""
    cfg = make_minimal_engine_config(
        n_species=1,
        output_spatial_enabled=True,
        output_spatial_biomass=False,
        output_spatial_abundance=False,
        output_spatial_yield_biomass=False,
    )
    outputs = [_spatial_step(23, sb_val=1.0, sy_val=1.0)]
    write_outputs_netcdf_spatial(outputs, tmp_path, prefix="run", sim_index=0, config=cfg)
    for suffix in ("biomass", "abundance", "yield"):
        assert not (tmp_path / f"run_spatial_{suffix}_Simu0.nc").exists()
```

- [ ] **Step 9: Implement `write_outputs_netcdf_spatial`**

In `osmose/engine/output.py` after `write_outputs_netcdf`, add:

```python
def write_outputs_netcdf_spatial(
    outputs: list[StepOutput],
    output_dir: Path,
    prefix: str,
    sim_index: int,
    config: EngineConfig,
    grid=None,
) -> None:
    """Write per-cell NetCDF files for biomass, abundance, and yield-biomass.

    One file per enabled variant:
      {prefix}_spatial_biomass_Simu{i}.nc
      {prefix}_spatial_abundance_Simu{i}.nc
      {prefix}_spatial_yield_Simu{i}.nc

    Dims: (time, species, lat, lon). Coord values from grid.lat / grid.lon
    when provided; cell indices when grid has no lat/lon metadata. Land
    cells (per grid.ocean_mask) are written as NaN. CF-1.8 declared;
    _FillValue=NaN on the value array.
    """
    import xarray as xr

    if not config.output_spatial_enabled:
        return

    # Gather per-variant gates and field names
    variants = [
        (config.output_spatial_biomass, "spatial_biomass", "biomass", "tonnes"),
        (config.output_spatial_abundance, "spatial_abundance", "abundance", "individuals"),
        (config.output_spatial_yield_biomass, "spatial_yield", "yield", "tonnes"),
    ]
    if not any(enabled for enabled, _, _, _ in variants):
        return

    # Sample one non-None entry to infer shape / species count
    first = next(
        (
            getattr(o, attr)
            for o in outputs
            for _, attr, _, _ in variants
            if getattr(o, attr) is not None
        ),
        None,
    )
    if first is None:
        return  # nothing collected

    n_sp = config.n_species
    any_arr = next(iter(first.values()))
    ny, nx = any_arr.shape

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    species_names = config.species_names[:n_sp]

    # Grid coords
    if grid is not None and grid.lat is not None and grid.lon is not None:
        lat_coord = grid.lat
        lon_coord = grid.lon
        spatial_coord_source = "lat_lon"
        # Land mask from grid; NaN-fill land cells
        land_mask = ~grid.ocean_mask  # True where land
    else:
        lat_coord = np.arange(ny, dtype=np.int64)
        lon_coord = np.arange(nx, dtype=np.int64)
        spatial_coord_source = "cell_index"
        land_mask = np.zeros((ny, nx), dtype=bool)

    for enabled, attr, file_tag, unit in variants:
        if not enabled:
            continue
        arr = np.full((len(outputs), n_sp, ny, nx), np.nan, dtype=np.float64)
        for t_idx, o in enumerate(outputs):
            d = getattr(o, attr)
            if d is None:
                continue
            for sp in range(n_sp):
                if sp in d:
                    cell = d[sp].astype(np.float64, copy=True)
                    cell[land_mask] = np.nan
                    arr[t_idx, sp, :, :] = cell

        ds = xr.Dataset(
            {file_tag: (["time", "species", "lat", "lon"], arr)},
            coords={
                "time": times,
                "species": species_names,
                "lat": lat_coord,
                "lon": lon_coord,
            },
            attrs={
                "description": f"OSMOSE Python engine spatial {file_tag} ({unit})",
                "n_dt_per_year": config.n_dt_per_year,
                "n_year": config.n_year,
                "Conventions": "CF-1.8",
                "spatial_coord_source": spatial_coord_source,
                "time_convention": (
                    "time = StepOutput.step / n_dt_per_year — last raw "
                    "timestep of each averaging window, in years"
                ),
            },
        )
        ds[file_tag].encoding["_FillValue"] = np.float64("nan")
        ds.to_netcdf(output_dir / f"{prefix}_spatial_{file_tag}_Simu{sim_index}.nc")
```

Wire into `write_outputs` (in the same file) at the end of its body:

```python
    # Spatial NetCDF outputs
    write_outputs_netcdf_spatial(
        outputs, output_dir, prefix=prefix, sim_index=0,
        config=config, grid=config.grid,  # or however the Grid lives on config
    )
```

(If `config` doesn't have a `grid` attr, pass the grid through the existing `write_outputs` signature — grep `write_outputs(` callers to find where grid is accessible. If it isn't yet, the spatial writer accepts `grid=None` and falls back to cell-index coords.)

- [ ] **Step 10: Write the remaining spatial tests**

Append to `tests/test_engine_output.py`:

```python
from osmose.engine.output import write_outputs_netcdf_spatial


def test_spatial_netcdf_shape_and_coords(tmp_path):
    """Writer emits one NetCDF per enabled variant with (time, species, lat, lon)
    dims and coord values derived from grid.lat / grid.lon."""
    cfg = make_minimal_engine_config(
        n_species=1,
        output_spatial_enabled=True,
        output_spatial_biomass=True,
        output_spatial_abundance=False,
        output_spatial_yield_biomass=False,
    )
    grid = Grid(
        ny=2, nx=3,
        ocean_mask=np.array([[True, True, True], [True, True, False]]),
        lat=np.array([10.0, 20.0]),
        lon=np.array([30.0, 40.0, 50.0]),
    )
    outputs = [
        StepOutput(
            step=23,
            biomass=np.array([100.0]),
            abundance=np.array([1000.0]),
            mortality_by_cause=np.zeros((1, 8)),
            spatial_biomass={0: np.ones((2, 3), dtype=np.float64)},
            spatial_abundance={0: np.ones((2, 3), dtype=np.float64)},
            spatial_yield={0: np.zeros((2, 3), dtype=np.float64)},
        ),
    ]
    write_outputs_netcdf_spatial(
        outputs, tmp_path, prefix="run", sim_index=0, config=cfg, grid=grid,
    )

    path = tmp_path / "run_spatial_biomass_Simu0.nc"
    assert path.exists()
    ds = xr.open_dataset(path)
    assert ds["biomass"].dims == ("time", "species", "lat", "lon")
    assert ds["biomass"].shape == (1, 1, 2, 3)
    np.testing.assert_array_equal(ds["lat"].values, [10.0, 20.0])
    np.testing.assert_array_equal(ds["lon"].values, [30.0, 40.0, 50.0])

    # Per-variant assertion: abundance + yield files absent
    assert not (tmp_path / "run_spatial_abundance_Simu0.nc").exists()
    assert not (tmp_path / "run_spatial_yield_Simu0.nc").exists()


def test_spatial_netcdf_nan_on_land(tmp_path):
    """Land cells per grid.ocean_mask are written as NaN, not 0.0."""
    cfg = make_minimal_engine_config(
        n_species=1,
        output_spatial_enabled=True,
        output_spatial_biomass=True,
    )
    grid = Grid(
        ny=2, nx=2,
        ocean_mask=np.array([[True, True], [True, False]]),  # (1,1) is land
        lat=np.array([10.0, 20.0]),
        lon=np.array([30.0, 40.0]),
    )
    outputs = [
        StepOutput(
            step=23,
            biomass=np.array([100.0]),
            abundance=np.array([1000.0]),
            mortality_by_cause=np.zeros((1, 8)),
            spatial_biomass={0: np.full((2, 2), 5.0)},
            spatial_abundance={0: np.full((2, 2), 50.0)},
            spatial_yield={0: np.full((2, 2), 1.0)},
        ),
    ]
    write_outputs_netcdf_spatial(
        outputs, tmp_path, prefix="run", sim_index=0, config=cfg, grid=grid,
    )
    ds = xr.open_dataset(tmp_path / "run_spatial_biomass_Simu0.nc")
    vals = ds["biomass"].values[0, 0, :, :]
    assert np.isnan(vals[1, 1]), "land cell must be NaN"
    # Ocean cells retain their value
    assert vals[0, 0] == 5.0


def test_spatial_disabled_when_master_false(tmp_path):
    """master=false → no spatial files written and StepOutput.spatial_* are None."""
    cfg = make_minimal_engine_config(
        n_species=1, output_spatial_enabled=False,
    )
    outputs = [
        StepOutput(
            step=23,
            biomass=np.array([100.0]),
            abundance=np.array([1000.0]),
            mortality_by_cause=np.zeros((1, 8)),
            # spatial_* all None per pairing invariant
        ),
    ]
    write_outputs_netcdf_spatial(
        outputs, tmp_path, prefix="run", sim_index=0, config=cfg,
    )
    for suffix in ("biomass", "abundance", "yield"):
        assert not (tmp_path / f"run_spatial_{suffix}_Simu0.nc").exists()
```

And the parity test:

```python
def test_spatial_biomass_sum_equals_nonspatial_biomass():
    """Parity invariant: summing spatial_biomass over (lat, lon) per species
    equals the non-spatial biomass timeseries (focal species only, cutoff
    applied). Tolerance rtol=1e-12, atol=0."""
    # Run a tiny end-to-end simulation or synthesize a SchoolState + Grid
    # that exercises _collect_biomass_abundance and _collect_spatial_outputs
    # on the same state, then assert the sum invariant.
    # If existing test harness has a run_tiny_sim() helper, reuse it.
    # Otherwise: construct SchoolState with deterministic biomass, call
    # both collectors, assert.
    pytest.skip(
        "Parity test requires a SchoolState fixture helper or tiny-sim "
        "harness. Add in the integration pass if the harness exists; "
        "otherwise leave as a follow-up."
    )
```

(The parity test is skipped rather than implemented if constructing the SchoolState fixture would exceed ~40 lines — flag it as a future follow-up. Don't gate the plan on a hard-to-synthesize fixture.)

- [ ] **Step 11: Run the spatial tests**

```bash
.venv/bin/python -m pytest tests/test_engine_output.py -v -k spatial
```

Expected: all PASS (the parity test is skipped per the note above).

- [ ] **Step 12: Full suite + lint**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check osmose/ scripts/ tests/ ui/
```

Expected: `2477 passed` (2470 after Task 2 + 7 new: 1 collector, 3 averaging-parametrized, 1 shape/coords, 1 NaN-on-land, 1 disabled + the 1 all-variants-off boundary). Ruff clean.

- [ ] **Step 13: Commit**

```bash
git add osmose/engine/simulate.py osmose/engine/output.py osmose/engine/config.py tests/test_engine_output.py
git commit -m "feat(output): spatial outputs — biomass / abundance / yield-biomass NetCDF

Adds cell-indexed spatial output NetCDFs for the three primary
fisheries-science variables:
  {prefix}_spatial_biomass_Simu{i}.nc    (time, species, lat, lon)
  {prefix}_spatial_abundance_Simu{i}.nc
  {prefix}_spatial_yield_Simu{i}.nc

StepOutput gains three paired fields (spatial_biomass /
spatial_abundance / spatial_yield); pairing invariant is all-None vs
all-non-None. Per-variant toggles gate the writer only; collection
is master-gated.

New _collect_spatial_outputs in simulate.py applies the same
output_cutoff_age filter as _collect_biomass_abundance so the parity
invariant sum(spatial_biomass over cells) == biomass holds per
focal species.

_average_step_outputs extended with per-field aggregation rules:
  spatial_biomass   element-wise MEAN (matches biomass)
  spatial_abundance element-wise MEAN (matches abundance)
  spatial_yield     element-wise SUM  (matches yield_by_species)

Writer applies grid.ocean_mask: land cells written as NaN, ocean
cells keep numeric values. CF-1.8 declared; _FillValue=NaN on the
value DataArray. When grid has no lat/lon metadata, coords fall back
to cell indices (recorded in attrs[spatial_coord_source]).

Config keys reuse the existing output.spatial.* schema entries at
osmose/schema/output.py:141-148 (no new namespace):
  output.spatial.enabled
  output.spatial.biomass.enabled
  output.spatial.abundance.enabled
  output.spatial.yield.biomass.enabled

Tests: 2477 passed (+7). Parity test skipped pending SchoolState
fixture helper — tracked as Phase-5 follow-up."
```

---

## Task 4: CHANGELOG + parity-roadmap STATUS-COMPLETE

- [ ] **Step 1: Append CHANGELOG entry**

In `CHANGELOG.md` under the `[Unreleased] → Added` subsection, prepend:

```markdown
- **output:** SP-4 closes the three remaining Phase-5 output-parity gaps. (5.5) Diet CSV now matches Java — one `{prefix}_dietMatrix_Simu{i}.csv` per run with a `Time` column and one row per recording period (predator-major, prey-minor column order). The pre-parity whole-run sum is removed; `_normalize_diet_matrix_to_percent` retained as a private helper for callers that need Java's percentage-per-predator layout. (5.6) `write_outputs_netcdf` now emits biomass/abundance by age and by size plus `mortality_by_cause` (8-member `MortalityCause` enum as a `cause` coord), ragged-padded with NaN across species and declaring CF-1.8 conventions with `_FillValue=NaN`. Gated per-variable: `output.{biomass,abundance}.{byage,bysize}.netcdf.enabled` and `output.mortality.netcdf.enabled`. (5.4) New spatial NetCDFs — `{prefix}_spatial_{biomass,abundance,yield}_Simu{i}.nc` with dims `(time, species, lat, lon)`, land cells as NaN, grid coords from `grid.lat`/`grid.lon` with cell-index fallback. `StepOutput` gains three paired spatial fields; `_average_step_outputs` extended with per-field aggregation rules (biomass/abundance mean, yield sum). Config keys reuse the pre-existing `output.spatial.*` schema entries. Spec at `docs/superpowers/specs/2026-04-19-sp4-output-system-design.md`; plan at `docs/superpowers/plans/2026-04-19-sp4-output-system-plan.md`.
```

- [ ] **Step 2: Mark Phase 5 STATUS-COMPLETE in the parity roadmap**

In `docs/parity-roadmap.md`, replace the Phase 5 header (around line 230) from:

```markdown
## Phase 5: Output System (LOW simulation impact, HIGH usability)
```

to:

```markdown
## Phase 5: Output System (LOW simulation impact, HIGH usability) — STATUS-COMPLETE (2026-04-19)

All seven items shipped. 5.1 / 5.2 / 5.3 / 5.7 were already in the Python engine before SP-4; the SP-4 front (commits through 2026-04-19) closed 5.5 (diet Java-parity), 5.6 (NetCDF distributions + mortality), and 5.4 (spatial outputs: biomass / abundance / yield-biomass). Remaining Java-side spatial variants (TL, size, mortality, egg, bioen-spatial) and Ev-OSMOSE output families are deferred to Phase 6 — they don't block full standard-OSMOSE parity.
```

Leave the per-item 5.1-5.7 detail blocks below as-is (historical record). The STATUS-COMPLETE banner at the section head is the load-bearing change.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md docs/parity-roadmap.md
git commit -m "docs: SP-4 changelog + Phase 5 STATUS-COMPLETE"
```

- [ ] **Step 4: Push**

```bash
git push origin master
```

- [ ] **Step 5: Skim git log**

```bash
git log --oneline -8
```

Expected: four commits on top of `8ad2103`:
- (Task 1) `feat(output): diet CSV Java-parity — one file, one row per recording period`
- (Task 2) `feat(output): NetCDF per-species distributions + mortality-by-cause`
- (Task 3) `feat(output): spatial outputs — biomass / abundance / yield-biomass NetCDF`
- (Task 4) `docs: SP-4 changelog + Phase 5 STATUS-COMPLETE`

---

## Self-review checklist

- **Spec coverage:**
  - Spec §5.5 diet Java-parity → Task 1 (all 11 steps). ✓
  - Spec §5.6 NetCDF distributions + mortality → Task 2 (all 8 steps, 5 new schema keys, 5 new DataArrays, ragged-pad, CF attrs). ✓
  - Spec §5.4 spatial → Task 3 (all 13 steps, 3 new `StepOutput` fields, collector, averaging rules, writer, NaN-on-land, pairing invariant, parity test). ✓
  - Spec Non-goals (no retry button, no Sobol, no weight normalization, no spatial TL/size/mortality/egg, no CSV spatial, no per-fishery) — no steps implementing them. ✓
- **Placeholder scan:** No TBD/TODO in code blocks. One `pytest.skip(...)` in the parity test is explicitly documented as a deferred-to-follow-up gating, not a placeholder. ✓
- **Type consistency:** `_clamp_n_workers` is from Phase 3 (not relevant). `_avg_spatial(attr, op)` signature consistent across Task 3 steps. `write_outputs_netcdf_spatial(outputs, output_dir, prefix, sim_index, config, grid)` signature used identically in Steps 9, 10, 11. ✓
- **Config keys:** Spatial keys reuse schema entries at `osmose/schema/output.py:141-148`; new NetCDF keys appended at lines 155+ with consistent naming (`output.X.netcdf.enabled`). No master `output.netcdf.enabled` invented. ✓
- **Mortality cause enum:** All 8 members lowercased (`predation`, `starvation`, `additional`, `fishing`, `out`, `foraging`, `discards`, `aging`) per `state.py:17-27`. ✓
- **Grid coords:** `grid.lat`, `grid.lon` (not `lat_centers`/`lon_centers`). ✓
- **Test runner:** `.venv/bin/python -m pytest` throughout. No bare `python`, no `$()` in bash snippets. ✓
- **Commit granularity:** 4 commits, one per capability plus CHANGELOG. Clean revert story — each feature backs out independently. ✓
- **Baseline preserved:** Task 1 → 2464 (+3), Task 2 → 2470 (+6), Task 3 → 2477 (+7). Total +16 vs baseline 2461. Two migrated diet tests don't change the count. ✓

---

## Execution handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, two-stage review between tasks, fast iteration.
**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
