# SP-4 Output System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three remaining Phase-5 output-parity gaps: (5.5) diet per-recording-period matching Java's one-file-with-append-rows convention; (5.6) NetCDF per-species distributions + mortality-by-cause; (5.4) cell-indexed spatial outputs (biomass / abundance / yield-in-biomass).

**Architecture:** Engine-only. Three capability commits + one CHANGELOG commit. Library surface change = 3 new `StepOutput` dict fields, 1 new simulate-side collector (`_collect_spatial_outputs`), 2 new output-side writers, 5 new config-schema entries plus 3 pre-existing schema entries that need parsing.

**Tech Stack:** Python 3.12, NumPy, xarray + netCDF4, pandas, pytest. Matches CLAUDE.md conventions.

**Spec:** `docs/superpowers/specs/2026-04-19-sp4-output-system-design.md` (commit `8ad2103`).

## Pre-flight

- [ ] Baseline test count. Run `.venv/bin/python -m pytest --co -q 2>&1 | tail -3`. The reviewer-verified baseline as of `e036338` is **2476 collected (41 deselected) / 2461 passed**. If your run shows a different number, update every `+N`/`→ M` arithmetic in this plan (`2464` → `2461 + Task1Δ`, etc.) from the actual baseline.
- [ ] Lint baseline: `.venv/bin/ruff check osmose/ scripts/ tests/ ui/` clean.
- [ ] Library surface anchors (grep, don't trust literal line numbers):
  - `osmose/engine/output.py` — `write_outputs(outputs: list[StepOutput], output_dir: Path, config: EngineConfig, prefix: str = "osm")` starts around line 24; `_write_distribution_csvs` ~106; `write_diet_csv(path, diet_by_species, predator_names, prey_names)` ~207; `_write_yield_csv` ~241; `write_outputs_netcdf(outputs, path, config)` ~314.
  - `osmose/engine/simulate.py` — `StepOutput` frozen dataclass ~64; `_collect_biomass_abundance` ~628; `_collect_yield` ~672; `_collect_outputs` ~802; `_average_step_outputs` ~854; two `_collect_outputs` call sites (one in the step-0 branch, one in the main loop).
  - `osmose/engine/state.py` — `class MortalityCause(IntEnum)` with 8 members (`PREDATION`, `STARVATION`, `ADDITIONAL`, `FISHING`, `OUT`, `FORAGING`, `DISCARDS`, `AGING`); `SchoolState` fields include `cell_x: NDArray[np.int32]`, `cell_y: NDArray[np.int32]`, `weight: NDArray[np.float64]`, `n_dead`, `age_dt` — **not** `cell_id`.
  - `osmose/engine/grid.py` — `Grid.lat` shape `(ny,)`, `Grid.lon` shape `(nx,)`, `Grid.ocean_mask` shape `(ny, nx)`.
  - `osmose/engine/config.py` — helper `_enabled(cfg: dict[str, str], key: str) -> bool` at line 133 is the canonical way to parse boolean output flags (**not** `_parse_bool`).
  - `osmose/schema/output.py:141-148` — spatial keys already declared: `output.spatial.{enabled, biomass.enabled, abundance.enabled, size.enabled, ltl.enabled, yield.biomass.enabled, yield.abundance.enabled, egg.enabled}`. Lines 149-151 declare `output.biomass.netcdf.enabled`, `output.abundance.netcdf.enabled`, `output.yield.biomass.netcdf.enabled` — these are currently unparsed in `EngineConfig`; Task 2 parses them.
  - `tests/helpers.py` — contains `_make_school(...)` and `_ScriptRunner` only. There is **no** `make_minimal_engine_config` helper today; Task 0 writes it.

## File map

- **Engine:** `osmose/engine/simulate.py` gains 3 `StepOutput` fields + `__post_init__` invariant + `_collect_spatial_outputs` + extensions to `_average_step_outputs` and `_collect_outputs`; `osmose/engine/output.py` gains modified `write_diet_csv`, extended `write_outputs_netcdf`, new `write_outputs_netcdf_spatial`, modified `write_outputs` dispatcher (threading Grid); `osmose/engine/config.py` parses 8 new/pre-existing output-netcdf + spatial keys.
- **Schema:** 5 new `output.*.netcdf.enabled` keys in `osmose/schema/output.py`.
- **Tests:** new `make_minimal_engine_config` in `tests/helpers.py`; `tests/test_engine_diet.py` migrations; new tests in `tests/test_engine_output.py` or extend existing; `tests/test_engine_phase5.py` gets the diet-config-gate test.

---

## Task 0: Test fixture — `make_minimal_engine_config`

**Goal:** Provide a keyword-arg-configurable `EngineConfig` builder that every SP-4 test depends on. `EngineConfig` has ~80 positional-no-default fields; tests must not instantiate it directly.

**Files:**
- Modify: `tests/helpers.py` — add `make_minimal_engine_config(**overrides) -> EngineConfig`.

- [ ] **Step 1: Read existing patterns**

  Grep sibling test files for how they construct `EngineConfig` today:

  ```bash
  grep -rn "EngineConfig(" tests/ | head -20
  ```

  Most tests use the real config reader with a small fixture dict (e.g. `config_reader.read_dict(MINIMAL_CONFIG_DICT)`). The SP-4 tests need programmatic kwarg overrides. The helper below uses the same reader path, then field-replaces any overrides the caller specified.

- [ ] **Step 2: Write the helper**

  Append to `tests/helpers.py`:

  ```python
  from dataclasses import replace

  from osmose.engine.config import EngineConfig


  # Minimum keys to satisfy EngineConfig.from_dict() for a 1-species, 0-background
  # test config. Expand as needed — the smoke test below will tell you which
  # species.* keys the reader still demands. Do NOT add grid.* keys: grid is
  # constructed separately (see Task 3 helper _make_grid_2x2_with_land) and is
  # NOT parsed from the cfg dict by EngineConfig.from_dict.
  _MINIMAL_CFG_DICT: dict[str, str] = {
      "simulation.nspecies": "1",
      "simulation.nbackground": "0",
      "simulation.time.ndtperyear": "24",   # NOT "simulation.ndtperyear"
      "simulation.time.nyear": "1",          # NOT "simulation.nyear"
      # Species-0 minimum (iterate via smoke test if reader demands more):
      "species.name.sp0": "sp0",
      "species.lifespan.sp0": "10",
      # ... add whatever else the reader raises KeyError on.
  }


  def make_minimal_engine_config(
      *,
      extra_cfg: dict[str, str] | None = None,
      **overrides,
  ) -> EngineConfig:
      """Build a small ``EngineConfig`` for unit tests with keyword overrides.

      ``extra_cfg`` injects/overrides raw config-dict keys before parsing
      (use this for any key the existing reader honors — e.g. ``output.X.enabled``).
      ``**overrides`` are applied via ``dataclasses.replace`` AFTER parsing,
      for fields not exposed via config keys.

      Raises AttributeError if an ``override`` names a non-existent
      ``EngineConfig`` field — this is intentional. Update the dataclass
      AND this helper together if you add a new field.
      """
      raw: dict[str, str] = dict(_MINIMAL_CFG_DICT)
      if extra_cfg:
          raw.update(extra_cfg)
      base = EngineConfig.from_dict(raw)
      if not overrides:
          return base
      # Validate overrides against declared fields
      declared = {f.name for f in base.__dataclass_fields__.values()}
      unknown = set(overrides) - declared
      if unknown:
          raise AttributeError(
              f"Unknown EngineConfig fields in overrides: {sorted(unknown)}"
          )
      return replace(base, **overrides)
  ```

  **Verified:** the public constructor is `EngineConfig.from_dict(cfg)` at `osmose/engine/config.py:1317` — a classmethod taking a single positional dict (no `cfg_dir` kwarg).

- [ ] **Step 3: Smoke test**

  ```bash
  .venv/bin/python -c "from tests.helpers import make_minimal_engine_config; c = make_minimal_engine_config(); print(type(c).__name__, c.n_species)"
  ```

  Expected: `EngineConfig 1`. If this errors, the `_MINIMAL_CFG_DICT` is missing required keys; iterate on it until the smoke test passes.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/helpers.py
  git commit -m "test: make_minimal_engine_config helper for SP-4 tests

  EngineConfig has ~80 positional-no-default fields; direct construction
  in tests is unwieldy. make_minimal_engine_config parses a minimal cfg
  dict via the real config reader, then applies dataclass-replace for
  any kwargs the caller wants overridden.

  Unknown kwargs raise AttributeError (typo-catching)."
  ```

---

## Task 1: Diet CSV — Java-parity (one file, appended rows)

**Goal:** `write_diet_csv` emits ONE CSV per simulation with one row per recording period, `Time` as the first column. Matches `DietOutput.java:217-222`.

**Files:**
- Modify: `osmose/engine/output.py` — `write_diet_csv` signature and body; caller in `write_outputs`; new `_normalize_diet_matrix_to_percent` helper for test compatibility.
- Modify: `tests/test_engine_diet.py` — migrate two existing tests to exercise `_normalize_diet_matrix_to_percent` instead of the old single-matrix writer.
- Modify or extend `tests/test_engine_phase5.py` — diet config-gate test.

- [ ] **Step 1: Verify baseline tests pass**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_diet.py tests/test_engine_phase5.py -q
  ```

  Record the current pass count; we'll compare after the migration.

- [ ] **Step 2: Write the new failing tests first**

  Append to `tests/test_engine_diet.py`:

  ```python
  import numpy as np
  import pandas as pd
  import pytest

  from osmose.engine.output import write_diet_csv


  def test_write_diet_csv_emits_one_row_per_recording_period(tmp_path):
      """Java-parity: one CSV per run, one row per recording period, Time
      in the first column. Whole-run sum recoverable via
      df.drop(columns='Time').sum(axis=0)."""
      predator_names = ["cod", "herring"]
      prey_names = ["cod", "herring", "plankton"]

      step_matrices = [
          np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]]),
          np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 5.0]]),
          np.array([[2.0, 2.0, 0.0], [1.0, 2.0, 6.0]]),
      ]
      step_times = [1.0, 2.0, 3.0]

      path = tmp_path / "run_dietMatrix_Simu0.csv"
      write_diet_csv(
          path=path,
          step_diet_matrices=step_matrices,
          step_times=step_times,
          predator_names=predator_names,
          prey_names=prey_names,
      )

      df = pd.read_csv(path)
      assert len(df) == 3
      assert list(df["Time"]) == step_times
      assert list(df.columns) == [
          "Time",
          "cod_cod", "cod_herring", "cod_plankton",
          "herring_cod", "herring_herring", "herring_plankton",
      ]
      assert df.iloc[0]["cod_plankton"] == pytest.approx(2.0)
      assert df.iloc[1]["herring_plankton"] == pytest.approx(5.0)
      assert df.iloc[2]["cod_cod"] == pytest.approx(2.0)


  def test_write_diet_csv_with_empty_step_list_writes_no_file(tmp_path):
      path = tmp_path / "run_dietMatrix_Simu0.csv"
      write_diet_csv(
          path=path,
          step_diet_matrices=[],
          step_times=[],
          predator_names=["cod"],
          prey_names=["cod", "plankton"],
      )
      assert not path.exists()
  ```

- [ ] **Step 3: Run the new tests to verify they fail**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_diet.py::test_write_diet_csv_emits_one_row_per_recording_period -v
  ```

  Expected: FAIL with `TypeError` on the new kwargs — current signature is `(path, diet_by_species, predator_names, prey_names)`.

- [ ] **Step 4: Rewrite `write_diet_csv`**

  In `osmose/engine/output.py`, replace the existing `write_diet_csv` (around line 207) with:

  ```python
  def write_diet_csv(
      *,
      path: Path,
      step_diet_matrices: list[NDArray[np.float64]],
      step_times: list[float],
      predator_names: list[str],
      prey_names: list[str],
  ) -> None:
      """Write diet composition as one CSV per simulation with one row per
      recording period, matching Java ``DietOutput.java:217-222``.

      Schema: first column ``Time``; remaining columns are one per
      ``{predator}_{prey}`` pair in predator-major, prey-minor order.
      Values are BIOMASS EATEN in tonnes. Per-predator percentage
      normalization is available via ``_normalize_diet_matrix_to_percent``
      when callers need Java's percentage layout.

      No-op when ``step_diet_matrices`` is empty.
      """
      if not step_diet_matrices:
          return
      if len(step_diet_matrices) != len(step_times):
          raise ValueError(
              f"step_diet_matrices length {len(step_diet_matrices)} "
              f"!= step_times length {len(step_times)}"
          )

      n_pred, n_prey = len(predator_names), len(prey_names)
      columns = [f"{pred}_{prey}" for pred in predator_names for prey in prey_names]
      rows: list[list[float]] = []
      for mat, t in zip(step_diet_matrices, step_times, strict=True):
          if mat.shape != (n_pred, n_prey):
              raise ValueError(
                  f"diet matrix shape {mat.shape} != ({n_pred}, {n_prey}) at time {t}"
              )
          rows.append([t, *mat.reshape(-1).tolist()])
      df = pd.DataFrame(rows, columns=["Time", *columns])
      df.to_csv(path, index=False)


  def _normalize_diet_matrix_to_percent(
      diet_by_species: NDArray[np.float64],
  ) -> NDArray[np.float64]:
      """Normalize a single (n_pred, n_prey) matrix to per-predator percentages."""
      totals = diet_by_species.sum(axis=1, keepdims=True)
      safe_totals = np.where(totals > 0, totals, 1.0)
      return diet_by_species / safe_totals * 100.0
  ```

  Keyword-only signature (`*,`) ensures callers update at compile time rather than silently passing the old positional order.

- [ ] **Step 5: Update the caller in `write_outputs`**

  Find the block at `osmose/engine/output.py` around lines 76-87 (grep `dietMatrix_Simu0`). Replace:

  ```python
      # Write diet CSV if diet data is present
      diet_arrays = [o.diet_by_species for o in outputs if o.diet_by_species is not None]
      if diet_arrays:
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

  with (gate honors `diet_output_enabled`, which is the attribute name on `EngineConfig` — grep `diet_output_enabled` in config.py to confirm):

  ```python
      # Write diet CSV (Java-parity: one file, one row per recording period)
      if config.diet_output_enabled:
          step_matrices: list[NDArray[np.float64]] = []
          step_times: list[float] = []
          for o in outputs:
              if o.diet_by_species is not None:
                  step_matrices.append(o.diet_by_species)
                  step_times.append(o.step / config.n_dt_per_year)
          write_diet_csv(
              path=output_dir / f"{prefix}_dietMatrix_Simu0.csv",
              step_diet_matrices=step_matrices,
              step_times=step_times,
              predator_names=config.species_names,
              prey_names=config.all_species_names,
          )
  ```

- [ ] **Step 6: Migrate the two existing diet tests**

  `tests/test_engine_diet.py::test_write_diet_csv` (~line 286) and `::test_write_diet_csv_percentage` (~line 318) are methods on a `class TestDietCSVOutput:` (preserve the class wrapper). Each calls the OLD single-matrix signature and checks CSV-round-trip percentages. Migrate them to exercise `_normalize_diet_matrix_to_percent` directly — same semantic coverage, no CSV round-trip.

  Replace each method body (keep method names, `self` arg, class indentation) with assertions over the result of:

  ```python
      from osmose.engine.output import _normalize_diet_matrix_to_percent
      pct = _normalize_diet_matrix_to_percent(DATA_ARRAY)
      # ... existing percentage-correctness assertions against `pct`
  ```

  Drop any `tmp_path`, `pd.read_csv(path)`, or file-I/O in these two methods; they're testing the percentage function, not disk format. Run them:

  ```bash
  .venv/bin/python -m pytest tests/test_engine_diet.py::TestDietCSVOutput -v
  ```

  Expected: both PASS.

- [ ] **Step 7: Run the new row-per-period tests**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_diet.py::test_write_diet_csv_emits_one_row_per_recording_period tests/test_engine_diet.py::test_write_diet_csv_with_empty_step_list_writes_no_file -v
  ```

  Expected: both PASS.

- [ ] **Step 8: Add the config-gate test**

  Append to `tests/test_engine_phase5.py` (or a new `tests/test_engine_output.py` if that's preferred — follow whatever sibling-test-file conventions dictate):

  ```python
  def test_write_outputs_skips_diet_csv_when_disabled(tmp_path):
      import numpy as np
      from osmose.engine.output import write_outputs
      from osmose.engine.simulate import StepOutput
      from tests.helpers import make_minimal_engine_config

      cfg = make_minimal_engine_config(diet_output_enabled=False)
      n_pred = cfg.n_species
      n_prey = len(cfg.all_species_names)
      step_out = StepOutput(
          step=23,
          biomass=np.zeros(n_pred + cfg.n_background),
          abundance=np.zeros(n_pred + cfg.n_background),
          mortality_by_cause=np.zeros((n_pred, 8)),
          diet_by_species=np.ones((n_pred, n_prey), dtype=np.float64),
      )
      # write_outputs signature: (outputs, output_dir, config, prefix="osm")
      write_outputs([step_out], tmp_path, cfg, prefix="run")
      assert not (tmp_path / "run_dietMatrix_Simu0.csv").exists()
  ```

  Note the positional order: `outputs, output_dir, config, prefix` (verified at `osmose/engine/output.py:24`).

- [ ] **Step 9: Run the config-gate test + full suite + lint**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_phase5.py -v -k diet_csv_when_disabled
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: new test PASSES; full suite is `baseline + 3` (the 2 migrated tests don't change the count); ruff clean.

- [ ] **Step 10: Commit**

  ```bash
  git add osmose/engine/output.py tests/test_engine_diet.py tests/test_engine_phase5.py
  git commit -m "feat(output): diet CSV Java-parity — one file, one row per recording period

  Replaces the pre-parity whole-run sum with one ''{prefix}_dietMatrix_Simu{i}.csv''
  per simulation, one row per recording period, Time in the first column.
  Columns are ''{predator}_{prey}'' pairs in predator-major, prey-minor order.
  Matches DietOutput.java:217-222.

  Gated at the writer boundary by config.diet_output_enabled (existing
  output.diet.composition.enabled). Complementary to the collection gate
  that already suppresses StepOutput.diet_by_species when disabled.

  Existing TestDietCSVOutput methods migrated to exercise the new
  _normalize_diet_matrix_to_percent helper — same semantic coverage,
  no CSV round-trip needed.

  Tests: +3 (2 new row-per-period + 1 config-gate)."
  ```

---

## Task 2: NetCDF distributions + mortality-by-cause (Capability 5.6)

**Goal:** Extend `write_outputs_netcdf` with 5 new DataArrays, padding ragged per-species bins with NaN, declaring CF-1.8 conventions, and gating each via a per-variable toggle. Also parse the 3 pre-existing netcdf schema keys (`output.biomass.netcdf.enabled`, `output.abundance.netcdf.enabled`, `output.yield.biomass.netcdf.enabled`) that are declared in the schema but unused in `EngineConfig`.

**Design note:** the spec's §5.6 was internally inconsistent — one line proposed a master `output.netcdf.enabled`, the test list said per-variable only. **This plan adopts per-variable only** (matches the schema's CSV-gating pattern: `output.biomass.enabled`, `output.biomass.byage.enabled`, etc., all per-variable). Each of the 8 NetCDF DataArrays has its own toggle; when all toggles are off, `write_outputs_netcdf` returns without creating a file.

**Files:**
- Modify: `osmose/engine/output.py` — `write_outputs_netcdf` extended.
- Modify: `osmose/engine/config.py` — parse 8 netcdf keys (3 pre-existing + 5 new).
- Modify: `osmose/schema/output.py` — 5 new schema entries.
- Test: `tests/test_engine_output.py` (create if absent) for 6 new tests.

- [ ] **Step 1: Add the 5 new schema keys**

  Append to `osmose/schema/output.py` (find the existing NetCDF-key block around lines 149-154, insert after):

  ```python
      "output.biomass.byage.netcdf.enabled",
      "output.abundance.byage.netcdf.enabled",
      "output.biomass.bysize.netcdf.enabled",
      "output.abundance.bysize.netcdf.enabled",
      "output.mortality.netcdf.enabled",
  ```

- [ ] **Step 2: Parse the 8 NetCDF keys in `config.py`**

  Find the output-flags parse block in `osmose/engine/config.py` (grep `output_record_freq` or `_enabled(cfg, "output.`). Add these parses, using the canonical helper `_enabled(cfg, key)` (at `config.py:133`). **Do NOT invent a `_parse_bool` helper — it doesn't exist; use `_enabled`.**

  ```python
      # Three pre-existing schema keys that were declared but not parsed
      "output_biomass_netcdf": _enabled(cfg, "output.biomass.netcdf.enabled"),
      "output_abundance_netcdf": _enabled(cfg, "output.abundance.netcdf.enabled"),
      "output_yield_biomass_netcdf": _enabled(
          cfg, "output.yield.biomass.netcdf.enabled"
      ),
      # Five new keys
      "output_biomass_byage_netcdf": _enabled(
          cfg, "output.biomass.byage.netcdf.enabled"
      ),
      "output_abundance_byage_netcdf": _enabled(
          cfg, "output.abundance.byage.netcdf.enabled"
      ),
      "output_biomass_bysize_netcdf": _enabled(
          cfg, "output.biomass.bysize.netcdf.enabled"
      ),
      "output_abundance_bysize_netcdf": _enabled(
          cfg, "output.abundance.bysize.netcdf.enabled"
      ),
      "output_mortality_netcdf": _enabled(cfg, "output.mortality.netcdf.enabled"),
  ```

  Add 8 matching `bool` fields with `= False` defaults to the `EngineConfig` dataclass — at the end of the defaulted-field region, adjacent to existing `output_step0_include`/`output_cutoff_age`-style flags. Grep to confirm the defaulted region; don't reorder positional-no-default fields.

- [ ] **Step 3: Write failing NetCDF tests**

  Create (or append to) `tests/test_engine_output.py`:

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


  def _make_step_with_age(
      step: int, n_sp: int, bins_by_sp: dict[int, int] | None = None
  ) -> StepOutput:
      biomass_by_age = None
      abundance_by_age = None
      if bins_by_sp is not None:
          biomass_by_age = {
              sp: np.arange(bins_by_sp[sp], dtype=np.float64) + 10.0 * step
              for sp in range(n_sp) if sp in bins_by_sp
          }
          abundance_by_age = {
              sp: np.arange(bins_by_sp[sp], dtype=np.float64) * 100.0
              for sp in range(n_sp) if sp in bins_by_sp
          }
      return StepOutput(
          step=step,
          biomass=np.full(n_sp, 100.0 * (step + 1)),
          abundance=np.full(n_sp, 1000.0 * (step + 1)),
          mortality_by_cause=np.arange(n_sp * 8, dtype=np.float64).reshape(n_sp, 8),
          biomass_by_age=biomass_by_age,
          abundance_by_age=abundance_by_age,
      )


  def test_netcdf_contains_biomass_by_age_when_enabled(tmp_path):
      cfg = make_minimal_engine_config(
          n_species=2, output_biomass_byage_netcdf=True,
      )
      outputs = [_make_step_with_age(t, 2, {0: 3, 1: 3}) for t in (23, 47, 71)]
      write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
      ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
      assert "biomass_by_age" in ds.data_vars
      assert ds["biomass_by_age"].dims == ("time", "species", "age_bin")
      assert ds["biomass_by_age"].shape == (3, 2, 3)
      np.testing.assert_array_equal(ds["biomass_by_age"].values[0, 0, :], [0, 1, 2])


  def test_netcdf_contains_mortality_by_cause(tmp_path):
      cfg = make_minimal_engine_config(n_species=2, output_mortality_netcdf=True)
      outputs = [_make_step_with_age(t, 2) for t in (23, 47)]
      write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
      ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
      assert "mortality_by_cause" in ds.data_vars
      assert ds["mortality_by_cause"].dims == ("time", "species", "cause")
      assert list(ds.coords["cause"].values) == [
          "Predation", "Starvation", "Additional", "Fishing",
          "Out", "Foraging", "Discards", "Aging",
      ]  # capitalize() to match existing CSV writer at output.py:161


  def test_netcdf_not_written_when_every_toggle_disabled(tmp_path):
      cfg = make_minimal_engine_config(
          n_species=1,
          output_biomass_netcdf=False,
          output_abundance_netcdf=False,
          output_yield_biomass_netcdf=False,
          output_biomass_byage_netcdf=False,
          output_abundance_byage_netcdf=False,
          output_biomass_bysize_netcdf=False,
          output_abundance_bysize_netcdf=False,
          output_mortality_netcdf=False,
      )
      path = tmp_path / "run_Simu0.nc"
      write_outputs_netcdf([_make_step_with_age(23, 1)], path, cfg)
      assert not path.exists()


  def test_netcdf_pads_ragged_age_bins_with_nan(tmp_path):
      cfg = make_minimal_engine_config(
          n_species=2, output_biomass_byage_netcdf=True,
      )
      outputs = [_make_step_with_age(23, 2, {0: 4, 1: 2})]
      write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
      ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
      assert ds["biomass_by_age"].shape == (1, 2, 4)
      np.testing.assert_array_equal(
          np.isnan(ds["biomass_by_age"].values[0, 1, :]),
          [False, False, True, True],
      )


  def test_netcdf_pads_ragged_size_bins_with_nan(tmp_path):
      cfg = make_minimal_engine_config(
          n_species=2, output_biomass_bysize_netcdf=True,
      )
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
      write_outputs_netcdf([step], tmp_path / "run_Simu0.nc", cfg)
      ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
      assert ds["biomass_by_size"].shape == (1, 2, 4)
      np.testing.assert_array_equal(
          np.isnan(ds["biomass_by_size"].values[0, 1, :]),
          [False, False, True, True],
      )


  def test_netcdf_cf_conventions_attr(tmp_path):
      cfg = make_minimal_engine_config(
          n_species=1, output_biomass_byage_netcdf=True,
      )
      outputs = [_make_step_with_age(23, 1, {0: 2})]
      write_outputs_netcdf(outputs, tmp_path / "run_Simu0.nc", cfg)
      ds = xr.open_dataset(tmp_path / "run_Simu0.nc")
      assert ds.attrs.get("Conventions") == "CF-1.8"
      # _FillValue surfaces on encoding after round-trip
      fv = ds["biomass_by_age"].encoding.get("_FillValue")
      assert fv is not None and np.isnan(fv)
  ```

- [ ] **Step 4: Run new tests, confirm they fail**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_output.py -v
  ```

  Expected: all FAIL (the writer doesn't emit the new DataArrays yet).

- [ ] **Step 5: Rewrite `write_outputs_netcdf`**

  In `osmose/engine/output.py`, replace `write_outputs_netcdf` (around line 314) with:

  ```python
  def write_outputs_netcdf(
      outputs: list[StepOutput],
      path: Path,
      config: EngineConfig,
  ) -> None:
      """Write simulation outputs to NetCDF.

      Each of the 8 possible variables is gated by its own
      ``output.{var}.netcdf.enabled`` config key. When every toggle is
      off, no file is written.

      Ragged per-species distribution bins are padded to cross-species
      max with NaN. Declares CF-1.8 conventions; every float DataArray
      carries ``_FillValue = NaN``.
      """
      import xarray as xr

      want = {
          "biomass":           config.output_biomass_netcdf,
          "abundance":         config.output_abundance_netcdf,
          "yield":             config.output_yield_biomass_netcdf
                               and any(o.yield_by_species is not None for o in outputs),
          "biomass_by_age":    config.output_biomass_byage_netcdf
                               and any(o.biomass_by_age is not None for o in outputs),
          "abundance_by_age":  config.output_abundance_byage_netcdf
                               and any(o.abundance_by_age is not None for o in outputs),
          "biomass_by_size":   config.output_biomass_bysize_netcdf
                               and any(o.biomass_by_size is not None for o in outputs),
          "abundance_by_size": config.output_abundance_bysize_netcdf
                               and any(o.abundance_by_size is not None for o in outputs),
          "mortality_by_cause": config.output_mortality_netcdf,
      }
      if not any(want.values()):
          return

      times = np.array([o.step / config.n_dt_per_year for o in outputs])
      n_species = len(outputs[0].biomass)
      species_names = config.all_species_names[:n_species]

      data_vars: dict = {}
      coords: dict = {"time": times, "species": species_names}

      if want["biomass"]:
          data_vars["biomass"] = (
              ["time", "species"],
              np.array([o.biomass for o in outputs]),
          )
      if want["abundance"]:
          data_vars["abundance"] = (
              ["time", "species"],
              np.array([o.abundance for o in outputs]),
          )
      if want["yield"]:
          yield_arr = np.array([
              o.yield_by_species if o.yield_by_species is not None
              else np.full(config.n_species, np.nan)
              for o in outputs
          ])
          data_vars["yield"] = (["time", "focal_species"], yield_arr)
          coords["focal_species"] = config.species_names[: yield_arr.shape[1]]

      def _pad(attr: str) -> tuple[np.ndarray, int]:
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

      if want["biomass_by_age"]:
          arr, n_age = _pad("biomass_by_age")
          data_vars["biomass_by_age"] = (["time", "species", "age_bin"], arr)
          coords["age_bin"] = np.arange(n_age, dtype=np.float64)
      if want["abundance_by_age"]:
          arr, n_age = _pad("abundance_by_age")
          data_vars["abundance_by_age"] = (["time", "species", "age_bin"], arr)
          coords.setdefault("age_bin", np.arange(n_age, dtype=np.float64))
      if want["biomass_by_size"]:
          arr, n_size = _pad("biomass_by_size")
          data_vars["biomass_by_size"] = (["time", "species", "size_bin"], arr)
          coords["size_bin"] = np.arange(n_size, dtype=np.float64)
      if want["abundance_by_size"]:
          arr, n_size = _pad("abundance_by_size")
          data_vars["abundance_by_size"] = (["time", "species", "size_bin"], arr)
          coords.setdefault("size_bin", np.arange(n_size, dtype=np.float64))

      if want["mortality_by_cause"]:
          from osmose.engine.state import MortalityCause
          data_vars["mortality_by_cause"] = (
              ["time", "species", "cause"],
              np.array([o.mortality_by_cause for o in outputs]),
          )
          coords["cause"] = [c.name.capitalize() for c in MortalityCause]
          # Match the existing CSV writer at osmose/engine/output.py:161
          # ("Predation", "Starvation", ..., "Aging"). Users comparing CSV
          # and NetCDF outputs see identical cause labels.

      dataset_attrs = {
          "description": "OSMOSE Python engine output",
          "n_dt_per_year": config.n_dt_per_year,
          "n_year": config.n_year,
          "Conventions": "CF-1.8",
          "distribution_padding": (
              "Ragged per-species bin counts are padded to cross-species "
              "max with NaN. Structural padding is indistinguishable from "
              "missing data — downstream tools treat both identically."
          ),
      }
      if "mortality_by_cause" in data_vars:
          # Glossary for opaque enum members (Foraging = bioen cost-of-foraging;
          # Out = advected-out-of-domain). Attached when mortality var is present.
          dataset_attrs["cause_descriptions"] = (
              "Predation: schools consumed by other schools; "
              "Starvation: failed energy budget; "
              "Additional: residual/M-other; "
              "Fishing: captured by fishing mortality; "
              "Out: advected out of domain; "
              "Foraging: bioenergetic cost-of-foraging (Ev-OSMOSE only); "
              "Discards: discarded catch; "
              "Aging: senescence at lifespan."
          )
      ds = xr.Dataset(data_vars, coords=coords, attrs=dataset_attrs)
      for name in ds.data_vars:
          if np.issubdtype(ds[name].dtype, np.floating):
              ds[name].encoding["_FillValue"] = np.float64("nan")
      ds.to_netcdf(path)
  ```

- [ ] **Step 6: Run the 6 NetCDF tests**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_output.py -v -k netcdf
  ```

  Expected: all 6 PASS.

- [ ] **Step 7: Full suite + lint**

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: `Task-1-result + 6 passed`. Ruff clean.

- [ ] **Step 8: Commit**

  ```bash
  git add osmose/engine/output.py osmose/engine/config.py osmose/schema/output.py tests/test_engine_output.py
  git commit -m "feat(output): NetCDF per-species distributions + mortality-by-cause

  write_outputs_netcdf extended with 5 new DataArrays:
    biomass_by_age / abundance_by_age    (time, species, age_bin)
    biomass_by_size / abundance_by_size  (time, species, size_bin)
    mortality_by_cause                   (time, species, cause)

  Cause coord uses the 8-member MortalityCause enum, capitalized
  to match the CSV writer (Predation, Starvation, ..., Aging).
  Ragged per-species bin counts padded to cross-species max with NaN.
  CF-1.8 conventions declared; _FillValue=NaN on every float DataArray.
  When every toggle is off, no file is written.

  Config parsing: adds 5 new keys plus the 3 pre-existing
  output.{biomass,abundance,yield.biomass}.netcdf.enabled schema keys
  that were declared but unparsed in EngineConfig.

  Tests: +6."
  ```

---

## Task 3: Spatial outputs — biomass / abundance / yield-biomass (Capability 5.4)

**Goal:** Cell-indexed spatial NetCDFs for biomass, abundance, yield-biomass. Three new `StepOutput` fields (pairing invariant via `__post_init__`); new `_collect_spatial_outputs` in `simulate.py` using `state.cell_x`/`state.cell_y`; per-field averaging rules in `_average_step_outputs`; new `write_outputs_netcdf_spatial`; `Grid` threaded through `write_outputs` as a new parameter.

**Files:**
- Modify: `osmose/engine/simulate.py` — StepOutput fields + invariant; `_collect_spatial_outputs`; `_average_step_outputs` extensions at both the single-accumulator and multi-accumulator branches; `_collect_outputs` signature gains `grid`; both call sites updated.
- Modify: `osmose/engine/output.py` — `write_outputs` signature gains `grid`; dispatches to new `write_outputs_netcdf_spatial`.
- Modify: `osmose/engine/config.py` — parse 4 of the pre-existing `output.spatial.*` schema keys into `EngineConfig` fields.
- Modify: all callers of `write_outputs` (grep — likely `osmose/engine/simulate.py`, `scripts/*`) to pass `grid`.
- Test: `tests/test_engine_output.py` extended; `tests/helpers.py` gets `_make_schools_in_cells`.

- [ ] **Step 1: StepOutput fields + pairing invariant**

  In `osmose/engine/simulate.py`, inside `StepOutput` (around `:64-96`), append three fields AFTER `diet_by_species`:

  ```python
      spatial_biomass: dict[int, NDArray[np.float64]] | None = None
      spatial_abundance: dict[int, NDArray[np.float64]] | None = None
      spatial_yield: dict[int, NDArray[np.float64]] | None = None

      def __post_init__(self) -> None:
          # Pairing invariant: all three spatial_* fields are None together
          # (spatial outputs disabled) or all non-None together (master gate
          # on). Per-variant enablement gates only the writer.
          trio = (self.spatial_biomass, self.spatial_abundance, self.spatial_yield)
          none_count = sum(x is None for x in trio)
          if none_count not in (0, 3):
              raise ValueError(
                  "StepOutput spatial_* trio must all be None or all non-None, "
                  f"got none_count={none_count}"
              )
  ```

  Note: existing `@dataclass(frozen=True)` allows `__post_init__`. If the distribution pairs (`biomass_by_age`/`abundance_by_age`, size pair) currently lack a similar check, don't add one here — keep the diff scoped.

  Update the class docstring block to mention the new trio.

- [ ] **Step 2: Parse spatial config keys**

  In `osmose/engine/config.py`, add to the output-flags block using `_enabled` (not `_parse_bool`):

  ```python
      "output_spatial_enabled": _enabled(cfg, "output.spatial.enabled"),
      "output_spatial_biomass": _enabled(cfg, "output.spatial.biomass.enabled"),
      "output_spatial_abundance": _enabled(cfg, "output.spatial.abundance.enabled"),
      "output_spatial_yield_biomass": _enabled(
          cfg, "output.spatial.yield.biomass.enabled"
      ),
  ```

  Add 4 matching `bool` fields with `= False` defaults to `EngineConfig`. (The schema's `.size`/`.ltl`/`.yield.abundance`/`.egg` spatial keys are deferred to a follow-up; don't parse them in this task.)

- [ ] **Step 3: Add `_make_schools_in_cells` test helper**

  Append to `tests/helpers.py`:

  ```python
  import numpy as np

  from osmose.engine.state import MortalityCause, SchoolState


  def make_schools_in_cells(
      *,
      cell_yx: list[tuple[int, int]],
      species_id: list[int],
      biomass: list[float],
      abundance: list[float],
      n_dead_fishing: list[float] | None = None,
      weight: list[float] | None = None,
      age_dt: list[int] | None = None,
  ) -> SchoolState:
      """Build a minimal SchoolState with schools placed in specified cells.

      All lists must have the same length (= number of schools).
      ``n_dead_fishing`` defaults to zeros; ``weight`` defaults to 1.0;
      ``age_dt`` defaults to a large value (above any plausible cutoff).
      Use SchoolState.create().replace(...) so default-valued fields stay
      populated.
      """
      n = len(cell_yx)
      assert all(len(x) == n for x in [species_id, biomass, abundance]), (
          "lists must have equal length"
      )
      ys = np.array([c[0] for c in cell_yx], dtype=np.int32)
      xs = np.array([c[1] for c in cell_yx], dtype=np.int32)
      sp = np.array(species_id, dtype=np.int32)
      base = SchoolState.create(n, sp)
      n_dead = np.zeros((n, len(MortalityCause)), dtype=np.float64)
      if n_dead_fishing is not None:
          n_dead[:, int(MortalityCause.FISHING)] = np.asarray(n_dead_fishing)
      return base.replace(
          cell_y=ys,
          cell_x=xs,
          biomass=np.asarray(biomass, dtype=np.float64),
          abundance=np.asarray(abundance, dtype=np.float64),
          weight=np.asarray(weight if weight is not None else [1.0] * n, dtype=np.float64),
          age_dt=np.asarray(age_dt if age_dt is not None else [10**6] * n, dtype=np.int32),
          n_dead=n_dead,
      )
  ```

  If `SchoolState.create` has a different public constructor (grep `def create` in `state.py`), adjust. The intent: minimal inline builder that every Task-3 test can call.

- [ ] **Step 4: Write the collector test**

  Append to `tests/test_engine_output.py`:

  ```python
  from osmose.engine.grid import Grid
  from osmose.engine.simulate import _collect_spatial_outputs
  from tests.helpers import make_schools_in_cells


  def _make_grid_2x2_with_land() -> Grid:
      return Grid(
          ny=2, nx=2,
          ocean_mask=np.array([[True, True], [True, False]]),  # (1,1) is land
          lat=np.array([10.0, 20.0]),
          lon=np.array([30.0, 40.0]),
      )


  def test_collect_spatial_biomass_aggregates_by_cell():
      """Three schools across two cells; per-cell biomass sums match
      np.add.at scatter."""
      grid = _make_grid_2x2_with_land()
      state = make_schools_in_cells(
          cell_yx=[(0, 0), (0, 0), (0, 1)],
          species_id=[0, 0, 0],
          biomass=[10.0, 20.0, 30.0],
          abundance=[100.0, 200.0, 300.0],
          weight=[1.0, 1.0, 1.0],
      )
      cfg = make_minimal_engine_config(n_species=1, output_spatial_enabled=True)

      sb, sa, sy = _collect_spatial_outputs(state, grid, cfg)

      # Cell (0,0) has 10 + 20 = 30 biomass; cell (0,1) has 30
      assert sb[0][0, 0] == 30.0
      assert sb[0][0, 1] == 30.0
      assert sb[0][1, 0] == 0.0  # ocean, no schools
      assert sb[0][1, 1] == 0.0  # land — collector writes 0, writer substitutes NaN
      # Abundance same shape
      assert sa[0][0, 0] == 300.0
      # Yield zero (no fishing deaths in fixture)
      assert sy[0][0, 0] == 0.0
  ```

- [ ] **Step 5: Implement `_collect_spatial_outputs`**

  In `osmose/engine/simulate.py`, near the other `_collect_*` helpers (between `_collect_yield` at ~:672 and `_collect_outputs` at ~:802), add:

  ```python
  def _collect_spatial_outputs(
      state: SchoolState,
      grid: "Grid",
      config: EngineConfig,
  ) -> tuple[
      dict[int, NDArray[np.float64]],
      dict[int, NDArray[np.float64]],
      dict[int, NDArray[np.float64]],
  ]:
      """Aggregate biomass, abundance, and fishing-yield-in-biomass per cell
      per focal species.

      Always returns all three dicts populated (per the StepOutput pairing
      invariant). Each dict is keyed by species index with
      ``(grid.ny, grid.nx)`` float64 value arrays.

      Ocean cells with no schools hold 0.0; land cells hold 0.0 here too,
      and the writer substitutes NaN at render time via ``grid.ocean_mask``.

      Applies the same ``output_cutoff_age`` filter as
      ``_collect_biomass_abundance`` (simulate.py ~:628-651) so the parity
      invariant ``sum(spatial_biomass over cells) == biomass[:n_species]``
      holds for focal species (with background species excluded since they
      have no per-cell location).

      Yield is ``n_dead[:, FISHING] * weight`` per school, matching
      ``_collect_yield``'s formula (simulate.py ~:672-683).
      """
      ny, nx = grid.ny, grid.nx
      n_sp = config.n_species
      sb = {sp: np.zeros((ny, nx), dtype=np.float64) for sp in range(n_sp)}
      sa = {sp: np.zeros((ny, nx), dtype=np.float64) for sp in range(n_sp)}
      sy = {sp: np.zeros((ny, nx), dtype=np.float64) for sp in range(n_sp)}

      if len(state) == 0:
          return sb, sa, sy

      focal = state.species_id < n_sp
      if config.output_cutoff_age is not None:
          age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
          cutoff = config.output_cutoff_age[state.species_id]
          focal &= age_years >= cutoff

      if not focal.any():
          return sb, sa, sy

      sp_ids = state.species_id[focal]
      ys = state.cell_y[focal]          # SchoolState has cell_y (not cell_id)
      xs = state.cell_x[focal]
      biomass = state.biomass[focal]
      abundance = state.abundance[focal]
      # Yield-in-biomass per school: n_dead * weight — matches _collect_yield
      yield_b = state.n_dead[focal, int(MortalityCause.FISHING)] * state.weight[focal]

      for sp in range(n_sp):
          m = sp_ids == sp
          if not m.any():
              continue
          np.add.at(sb[sp], (ys[m], xs[m]), biomass[m])
          np.add.at(sa[sp], (ys[m], xs[m]), abundance[m])
          np.add.at(sy[sp], (ys[m], xs[m]), yield_b[m])
      return sb, sa, sy
  ```

- [ ] **Step 6: Wire into `_collect_outputs` (signature change + 2 call sites)**

  `_collect_outputs` currently has signature `def _collect_outputs(state, config, step, bkg_output, diet_by_species)` around `:802`. Change to accept `grid`:

  ```python
  def _collect_outputs(
      state: SchoolState,
      config: EngineConfig,
      grid: "Grid",
      step: int,
      bkg_output,
      diet_by_species,
  ) -> StepOutput:
  ```

  Inside the body, before constructing `StepOutput`, add:

  ```python
      spatial_biomass = spatial_abundance = spatial_yield = None
      if config.output_spatial_enabled:
          spatial_biomass, spatial_abundance, spatial_yield = (
              _collect_spatial_outputs(state, grid, config)
          )
  ```

  Pass `spatial_biomass=spatial_biomass, spatial_abundance=spatial_abundance, spatial_yield=spatial_yield` into the `StepOutput(...)` constructor at the bottom.

  **Update both call sites** of `_collect_outputs`. Grep `_collect_outputs(` in `simulate.py` — two sites:

  - Step-0 snapshot (when `config.output_step0_include` is true, around `:1025-1028`).
  - Main loop recording (around `:1228-1233`).

  Both already have the grid object in scope (simulate receives `grid` as a top-level argument). Pass it through.

- [ ] **Step 7: Extend `_average_step_outputs` — BOTH branches**

  Current `_average_step_outputs` at `:854-915` has a single-accumulator early return at `:873-890` AND a multi-accumulator main path at `:891-914`. Both must receive the three new spatial fields.

  Insert near the top of the function (after the existing `_avg_bioen` helper):

  ```python
      def _avg_spatial(attr: str, op: str) -> dict[int, NDArray[np.float64]] | None:
          dicts = [
              getattr(o, attr) for o in accumulated if getattr(o, attr) is not None
          ]
          if not dicts:
              return None
          keys: set[int] = set()
          for d in dicts:
              keys |= d.keys()
          out: dict[int, NDArray[np.float64]] = {}
          for sp in keys:
              arrays = [d[sp] for d in dicts if sp in d]
              if op == "mean":
                  out[sp] = np.mean(arrays, axis=0)
              elif op == "sum":
                  out[sp] = np.sum(arrays, axis=0)
              else:
                  raise ValueError(f"unknown op: {op}")
          return out

      spatial_b_agg = _avg_spatial("spatial_biomass", "mean")
      spatial_a_agg = _avg_spatial("spatial_abundance", "mean")
      spatial_y_agg = _avg_spatial("spatial_yield", "sum")
  ```

  Then in BOTH return-sites — the `len(accumulated) == 1` early return AND the multi-step main return — pass the three aggregated kwargs into `StepOutput(...)`:

  ```python
          spatial_biomass=spatial_b_agg,
          spatial_abundance=spatial_a_agg,
          spatial_yield=spatial_y_agg,
  ```

  The pairing invariant enforced by `__post_init__` (from Step 1) will complain if you forget one branch — that's the trip-wire.

- [ ] **Step 8: Write the averaging tests**

  Append to `tests/test_engine_output.py`:

  ```python
  from osmose.engine.simulate import _average_step_outputs


  def _spatial_step(step: int, sb_val: float, sa_val: float, sy_val: float) -> StepOutput:
      return StepOutput(
          step=step,
          biomass=np.array([100.0]),
          abundance=np.array([1000.0]),
          mortality_by_cause=np.zeros((1, 8)),
          spatial_biomass={0: np.full((2, 2), sb_val)},
          spatial_abundance={0: np.full((2, 2), sa_val)},
          spatial_yield={0: np.full((2, 2), sy_val)},
      )


  @pytest.mark.parametrize(
      "field,op,vals,expected",
      [
          ("spatial_biomass",   "mean", [10.0, 20.0], 15.0),
          ("spatial_abundance", "mean", [100.0, 300.0], 200.0),
          ("spatial_yield",     "sum",  [5.0, 7.0], 12.0),
      ],
  )
  def test_average_spatial_outputs_rules(field, op, vals, expected):
      """spatial_biomass / spatial_abundance average element-wise;
      spatial_yield sums element-wise."""
      # Each field takes its own vals; the fixture sets all three but we
      # only assert on `field`. No cross-field scaling.
      s0 = _spatial_step(23, sb_val=vals[0], sa_val=vals[0], sy_val=vals[0])
      s1 = _spatial_step(47, sb_val=vals[1], sa_val=vals[1], sy_val=vals[1])
      avg = _average_step_outputs([s0, s1], freq=24, record_step=47)
      arr = getattr(avg, field)[0]
      np.testing.assert_allclose(arr, expected)


  @pytest.mark.parametrize(
      "field,expected",
      [
          ("spatial_biomass",   10.0),  # single-step mean == value
          ("spatial_abundance", 10.0),
          ("spatial_yield",     10.0),  # single-step sum == value
      ],
  )
  def test_average_spatial_outputs_single_accumulator_branch(field, expected):
      """Covers the early-return branch of _average_step_outputs (1 step).
      Iteration-1 review found both branches must be instrumented; this
      exercises the one the 2-step test above does NOT hit."""
      s0 = _spatial_step(23, sb_val=10.0, sa_val=10.0, sy_val=10.0)
      avg = _average_step_outputs([s0], freq=24, record_step=23)
      arr = getattr(avg, field)[0]
      np.testing.assert_allclose(arr, expected)


  def test_average_spatial_outputs_preserves_per_cell_variation():
      """Fixtures with np.full() reduce axis bugs to scalar equality.
      This test uses per-cell-varying arrays so an axis-wrong implementation
      (e.g. flatten-then-mean) gives a different answer than element-wise."""
      sb0 = np.array([[1.0, 2.0], [3.0, 4.0]])
      sb1 = np.array([[10.0, 20.0], [30.0, 40.0]])
      s0 = StepOutput(
          step=23, biomass=np.array([100.0]), abundance=np.array([1000.0]),
          mortality_by_cause=np.zeros((1, 8)),
          spatial_biomass={0: sb0},
          spatial_abundance={0: np.zeros((2, 2))},
          spatial_yield={0: np.zeros((2, 2))},
      )
      s1 = StepOutput(
          step=47, biomass=np.array([100.0]), abundance=np.array([1000.0]),
          mortality_by_cause=np.zeros((1, 8)),
          spatial_biomass={0: sb1},
          spatial_abundance={0: np.zeros((2, 2))},
          spatial_yield={0: np.zeros((2, 2))},
      )
      avg = _average_step_outputs([s0, s1], freq=24, record_step=47)
      np.testing.assert_allclose(
          avg.spatial_biomass[0],
          np.array([[5.5, 11.0], [16.5, 22.0]]),
      )


  def test_step_output_post_init_rejects_partial_spatial_trio():
      """Directly exercise the __post_init__ pairing invariant on StepOutput:
      spatial_biomass / spatial_abundance / spatial_yield must be all-None or
      all-set. Partial construction is rejected."""
      with pytest.raises(ValueError, match="spatial"):
          StepOutput(
              step=23,
              biomass=np.array([100.0]),
              abundance=np.array([1000.0]),
              mortality_by_cause=np.zeros((1, 8)),
              spatial_biomass={0: np.zeros((2, 2))},
              # spatial_abundance and spatial_yield omitted (None) → invariant violation
          )
  ```

- [ ] **Step 9: Run collector + averaging tests**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_output.py -v -k "collect_spatial or average_spatial or step_output_post_init"
  ```

  Expected: 9 PASS (1 collector + 3 parametrized 2-step + 3 parametrized 1-step single-accumulator + 1 per-cell-varying + 1 `__post_init__` invariant).

- [ ] **Step 10: Implement `write_outputs_netcdf_spatial` + thread `grid` through `write_outputs`**

  In `osmose/engine/output.py`, AFTER `write_outputs_netcdf`, add:

  ```python
  def write_outputs_netcdf_spatial(
      outputs: list[StepOutput],
      output_dir: Path,
      prefix: str,
      sim_index: int,
      config: EngineConfig,
      *,
      grid=None,  # osmose.engine.grid.Grid | None — None → cell-index fallback
  ) -> None:
      """Write per-cell NetCDF files for biomass, abundance, yield-biomass.

      One file per enabled variant:
        {prefix}_spatial_biomass_Simu{i}.nc
        {prefix}_spatial_abundance_Simu{i}.nc
        {prefix}_spatial_yield_Simu{i}.nc

      Dims: (time, species, lat, lon). Coord values from grid.lat / grid.lon
      when present; cell indices otherwise (recorded in attrs). Land cells
      (per grid.ocean_mask) written as NaN. CF-1.8 declared.
      """
      import xarray as xr

      if not config.output_spatial_enabled:
          return
      variants = [
          (config.output_spatial_biomass,        "spatial_biomass",   "biomass",   "tonnes"),
          (config.output_spatial_abundance,      "spatial_abundance", "abundance", "individuals"),
          (config.output_spatial_yield_biomass,  "spatial_yield",     "yield",     "tonnes"),
      ]
      if not any(enabled for enabled, *_ in variants):
          return
      # Sample to get shape
      sample = next(
          (getattr(o, attr) for o in outputs for _, attr, *_ in variants
           if getattr(o, attr) is not None),
          None,
      )
      if sample is None:
          return  # nothing collected
      any_arr = next(iter(sample.values()))
      ny, nx = any_arr.shape
      n_sp = config.n_species

      times = np.array([o.step / config.n_dt_per_year for o in outputs])
      species_names = config.species_names[:n_sp]

      if grid is not None and grid.lat is not None and grid.lon is not None:
          lat_coord, lon_coord = grid.lat, grid.lon
          coord_source = "lat_lon"
          land = ~grid.ocean_mask
      else:
          lat_coord = np.arange(ny, dtype=np.int64)
          lon_coord = np.arange(nx, dtype=np.int64)
          coord_source = "cell_index"
          land = np.zeros((ny, nx), dtype=bool)

      # Per-variant cell_methods (CF-1.8 convention for aggregation): biomass and
      # abundance are period means, yield is a period sum.
      cell_methods_by_tag = {
          "biomass":   "time: mean",
          "abundance": "time: mean",
          "yield":     "time: sum",
      }
      long_name_by_tag = {
          "biomass":   "spatial biomass per recording period (focal species only)",
          "abundance": "spatial abundance per recording period (focal species only)",
          "yield":     "spatial fishing yield summed over recording period (focal species only)",
      }

      for enabled, attr, tag, unit in variants:
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
                      cell[land] = np.nan  # ocean cells retain 0.0; land becomes NaN
                      arr[t_idx, sp, :, :] = cell

          ds = xr.Dataset(
              {tag: (["time", "species", "lat", "lon"], arr)},
              coords={
                  "time": times, "species": species_names,
                  "lat": lat_coord, "lon": lon_coord,
              },
              attrs={
                  "description": f"OSMOSE Python engine spatial {tag} ({unit})",
                  "n_dt_per_year": config.n_dt_per_year,
                  "n_year": config.n_year,
                  "Conventions": "CF-1.8",
                  "spatial_coord_source": coord_source,
                  "time_convention": (
                      "Each time coordinate value is the LAST raw timestep "
                      "of its averaging window (not window midpoint). "
                      "Consistent with the non-spatial _average_step_outputs."
                  ),
                  "nan_semantics": (
                      "NaN = land cell (outside ocean_mask). "
                      "0.0 = ocean cell with no schools this recording period. "
                      "The two are distinct states."
                  ),
                  # Ecological caveats per spec (attach so downstream tools preserve context)
                  "cutoff_age_note": (
                      "Spatial outputs apply the same output_cutoff_age filter "
                      "as the non-spatial biomass/abundance timeseries. "
                      "Young-of-year and other sub-cutoff schools are absent "
                      "from these maps even in nursery cells."
                  ),
                  "abundance_period_mean_note": (
                      "Biomass and abundance are per-period MEANS over the "
                      "averaging window (matching the non-spatial rule), "
                      "not end-of-period snapshots. Recruit pulses mid-window "
                      "are diluted by the mean."
                  ),
              },
          )
          # Per-DataArray CF-1.8 attrs: units, long_name, cell_methods.
          ds[tag].attrs["units"] = unit
          ds[tag].attrs["long_name"] = long_name_by_tag[tag]
          ds[tag].attrs["cell_methods"] = cell_methods_by_tag[tag]
          ds[tag].encoding["_FillValue"] = np.float64("nan")
          ds.to_netcdf(output_dir / f"{prefix}_spatial_{tag}_Simu{sim_index}.nc")
  ```

  Now thread `grid` through `write_outputs`. Update the signature (currently `write_outputs(outputs, output_dir, config, prefix="osm")`) to:

  ```python
  def write_outputs(
      outputs: list[StepOutput],
      output_dir: Path,
      config: EngineConfig,
      prefix: str = "osm",
      *,
      grid=None,
  ) -> None:
  ```

  At the end of the body, dispatch to spatial:

  ```python
      # Spatial outputs (NetCDF only; gated by config.output_spatial_enabled)
      write_outputs_netcdf_spatial(
          outputs, output_dir,
          prefix=prefix, sim_index=0,
          config=config, grid=grid,
      )
  ```

  Update every caller of `write_outputs` to pass `grid=grid`. Grep:

  ```bash
  grep -rn "write_outputs(" osmose/ scripts/ tests/ 2>&1 | head
  ```

  **Verified caller chain:** the ONE production caller is `osmose/engine/__init__.py:67` (`PythonEngine.run()`), which already has `grid` in scope (built at lines 42-56). Update that single call site to pass `grid=grid`. The remaining 15+ callers are all tests — since the new arg is keyword-only with a `None` default, tests don't need updating unless they exercise spatial paths. Scripts that construct configs and call `write_outputs` without a grid in scope pass `grid=None` (the writer falls back to cell-index coords).

- [ ] **Step 11: Write the remaining spatial tests**

  Append to `tests/test_engine_output.py`:

  ```python
  from osmose.engine.output import write_outputs_netcdf_spatial


  def test_spatial_netcdf_shape_and_coords(tmp_path):
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
              spatial_biomass={0: np.ones((2, 3))},
              spatial_abundance={0: np.ones((2, 3))},
              spatial_yield={0: np.zeros((2, 3))},
          ),
      ]
      write_outputs_netcdf_spatial(
          outputs, tmp_path, prefix="run", sim_index=0, config=cfg, grid=grid,
      )
      p = tmp_path / "run_spatial_biomass_Simu0.nc"
      assert p.exists()
      ds = xr.open_dataset(p)
      assert ds["biomass"].dims == ("time", "species", "lat", "lon")
      assert ds["biomass"].shape == (1, 1, 2, 3)
      np.testing.assert_array_equal(ds["lat"].values, [10.0, 20.0])
      np.testing.assert_array_equal(ds["lon"].values, [30.0, 40.0, 50.0])
      assert not (tmp_path / "run_spatial_abundance_Simu0.nc").exists()
      assert not (tmp_path / "run_spatial_yield_Simu0.nc").exists()


  def test_spatial_netcdf_nan_on_land(tmp_path):
      cfg = make_minimal_engine_config(
          n_species=1, output_spatial_enabled=True, output_spatial_biomass=True,
      )
      grid = Grid(
          ny=2, nx=2,
          ocean_mask=np.array([[True, True], [True, False]]),  # (1,1) land
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
      assert np.isnan(vals[1, 1])
      assert vals[0, 0] == 5.0


  def test_spatial_disabled_when_master_false(tmp_path):
      cfg = make_minimal_engine_config(
          n_species=1, output_spatial_enabled=False,
      )
      outputs = [
          StepOutput(
              step=23,
              biomass=np.array([100.0]),
              abundance=np.array([1000.0]),
              mortality_by_cause=np.zeros((1, 8)),
          ),
      ]
      write_outputs_netcdf_spatial(
          outputs, tmp_path, prefix="run", sim_index=0, config=cfg, grid=None,
      )
      for suffix in ("biomass", "abundance", "yield"):
          assert not (tmp_path / f"run_spatial_{suffix}_Simu0.nc").exists()


  def test_spatial_collection_runs_but_no_files_when_all_variants_off(tmp_path):
      """master=True + all three per-variant toggles False: collection populates
      StepOutput.spatial_* via the pairing invariant; no files written."""
      cfg = make_minimal_engine_config(
          n_species=1,
          output_spatial_enabled=True,
          output_spatial_biomass=False,
          output_spatial_abundance=False,
          output_spatial_yield_biomass=False,
      )
      outputs = [
          _spatial_step(23, sb_val=1.0, sa_val=1.0, sy_val=1.0),
      ]
      write_outputs_netcdf_spatial(
          outputs, tmp_path, prefix="run", sim_index=0, config=cfg, grid=None,
      )
      for suffix in ("biomass", "abundance", "yield"):
          assert not (tmp_path / f"run_spatial_{suffix}_Simu0.nc").exists()


  def test_spatial_netcdf_grid_none_fallback(tmp_path):
      """When grid=None is passed to the writer, coords fall back to cell
      indices (0..ny-1, 0..nx-1), no land masking is applied, and the
      spatial_coord_source attr records 'cell_index'. Covers the writer
      fallback branch exercised by scripts that don't have a Grid in scope."""
      cfg = make_minimal_engine_config(
          n_species=1, output_spatial_enabled=True, output_spatial_biomass=True,
      )
      outputs = [
          StepOutput(
              step=23,
              biomass=np.array([100.0]),
              abundance=np.array([1000.0]),
              mortality_by_cause=np.zeros((1, 8)),
              spatial_biomass={0: np.full((2, 3), 5.0)},
              spatial_abundance={0: np.full((2, 3), 50.0)},
              spatial_yield={0: np.full((2, 3), 1.0)},
          ),
      ]
      write_outputs_netcdf_spatial(
          outputs, tmp_path, prefix="run", sim_index=0, config=cfg, grid=None,
      )
      ds = xr.open_dataset(tmp_path / "run_spatial_biomass_Simu0.nc")
      np.testing.assert_array_equal(ds["lat"].values, [0, 1])
      np.testing.assert_array_equal(ds["lon"].values, [0, 1, 2])
      assert ds.attrs["spatial_coord_source"] == "cell_index"
      # No NaN anywhere — fallback omits land masking
      assert not np.isnan(ds["biomass"].values).any()


  def test_spatial_biomass_sum_equals_nonspatial_biomass():
      """Parity invariant: sum(spatial_biomass over cells) per focal species
      equals biomass[:n_species] (focal-only, cutoff applied)."""
      from osmose.engine.simulate import _collect_biomass_abundance
      grid = Grid(
          ny=2, nx=2,
          ocean_mask=np.ones((2, 2), dtype=bool),
          lat=np.array([10.0, 20.0]), lon=np.array([30.0, 40.0]),
      )
      cfg = make_minimal_engine_config(
          n_species=1, output_spatial_enabled=True,
      )
      state = make_schools_in_cells(
          cell_yx=[(0, 0), (0, 1), (1, 0), (1, 1)],
          species_id=[0, 0, 0, 0],
          biomass=[10.0, 20.0, 30.0, 40.0],
          abundance=[100.0, 200.0, 300.0, 400.0],
      )
      b_nonspatial, _ = _collect_biomass_abundance(state, cfg, bkg_output=None)
      sb, _, _ = _collect_spatial_outputs(state, grid, cfg)
      np.testing.assert_allclose(
          sb[0].sum(),
          b_nonspatial[: cfg.n_species][0],  # focal sp=0
          rtol=1e-12, atol=0.0,
      )
  ```

  The parity test is NOT skipped here — with `make_schools_in_cells` the fixture is cheap.

- [ ] **Step 12: Run all spatial tests**

  ```bash
  .venv/bin/python -m pytest tests/test_engine_output.py -v -k spatial
  ```

  Expected: 14 PASS — 1 collector + 3 averaging-2step + 3 averaging-1step (single-accumulator) + 1 per-cell-varying + 1 `__post_init__` invariant + 4 writer variants (shape, land-NaN, master-off, all-variants-off) + 1 writer grid-None fallback + 1 parity.

- [ ] **Step 13: Full suite + lint**

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: `baseline + 3 (Task 1) + 6 (Task 2) + 14 (Task 3) = baseline + 23`. Ruff clean.

- [ ] **Step 14: Commit**

  ```bash
  git add osmose/engine/simulate.py osmose/engine/output.py osmose/engine/config.py tests/test_engine_output.py tests/helpers.py
  git commit -m "feat(output): spatial outputs — biomass / abundance / yield-biomass NetCDF

  Cell-indexed spatial outputs with dims (time, species, lat, lon),
  one NetCDF per enabled variant. Land cells per grid.ocean_mask are
  written as NaN; ocean cells with no schools this period hold 0.0
  (distinct semantics, documented in attrs.nan_semantics). CF-1.8.

  StepOutput gains spatial_biomass / spatial_abundance / spatial_yield
  dict fields with an enforced pairing invariant via __post_init__
  (all None or all non-None).

  _collect_spatial_outputs applies the same output_cutoff_age filter
  as _collect_biomass_abundance and uses n_dead[:, FISHING] * weight
  for yield (matching _collect_yield). The parity invariant
  sum(spatial_biomass over cells) == biomass[:n_species] holds for
  focal species with rtol=1e-12.

  _average_step_outputs extended per-field: spatial_biomass and
  spatial_abundance averaged element-wise; spatial_yield summed.
  Both the single-accumulator early-return branch and the
  multi-accumulator main path pass the new kwargs.

  Config keys reuse existing output.spatial.* schema entries. Grid
  threads through write_outputs as a keyword-only arg; write_outputs
  dispatches to the new write_outputs_netcdf_spatial.

  Tests: +14 (collector, 3 averaging 2-step, 3 averaging 1-step single-accumulator,
  1 per-cell-varying, 1 __post_init__ invariant, 4 writer variants,
  1 grid-None fallback, 1 parity)."
  ```

---

## Task 4: CHANGELOG + parity-roadmap STATUS-COMPLETE

- [ ] **Step 1: Append CHANGELOG entry**

  In `CHANGELOG.md` under `[Unreleased] → Added`, prepend:

  ```markdown
  - **output:** SP-4 closes the three remaining Phase-5 output-parity gaps. (5.5) Diet CSV matches Java — one `{prefix}_dietMatrix_Simu{i}.csv` per run with `Time` column and one row per recording period (predator-major, prey-minor column order). `_normalize_diet_matrix_to_percent` retained as a private helper for callers needing Java's percentage-per-predator layout. (5.6) `write_outputs_netcdf` emits biomass/abundance by age and by size plus `mortality_by_cause` (8-member `MortalityCause` enum as `cause` coord); ragged-padded with NaN across species; CF-1.8 + `_FillValue=NaN`. Gated per-variable: `output.{biomass,abundance}.{byage,bysize}.netcdf.enabled` and `output.mortality.netcdf.enabled` — no master switch. (5.4) Spatial NetCDFs `{prefix}_spatial_{biomass,abundance,yield}_Simu{i}.nc` with dims `(time, species, lat, lon)`, land cells as NaN, grid coords from `grid.lat`/`grid.lon` with cell-index fallback. `StepOutput` gains three paired spatial dict fields with an enforced pairing invariant; `_average_step_outputs` extends with per-field rules (biomass/abundance mean, yield sum); `_collect_spatial_outputs` uses `state.cell_x`/`state.cell_y` and `n_dead * weight` for yield. Config keys reuse the pre-existing `output.spatial.*` schema entries. Spec at `docs/superpowers/specs/2026-04-19-sp4-output-system-design.md`; plan at `docs/superpowers/plans/2026-04-19-sp4-output-system-plan.md`.
  ```

- [ ] **Step 2: Mark Phase 5 STATUS-COMPLETE in `docs/parity-roadmap.md`**

  Replace the Phase 5 header (around line 230) with:

  ```markdown
  ## Phase 5: Output System (LOW simulation impact, HIGH usability) — STATUS-COMPLETE (2026-04-19)

  All seven items shipped. 5.1 / 5.2 / 5.3 / 5.7 were already in the Python engine before SP-4; the SP-4 front (commits through 2026-04-19) closed 5.5 (diet Java-parity), 5.6 (NetCDF distributions + mortality), and 5.4 (spatial outputs: biomass / abundance / yield-biomass). Remaining Java-side spatial variants (TL, size, mortality, egg, bioen-spatial) and Ev-OSMOSE output families are deferred to Phase 6 — they don't block full standard-OSMOSE parity.
  ```

  Leave the per-item 5.1-5.7 detail blocks intact (historical record).

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

  Expected (on top of `e036338` or the current plan-commit HEAD): 5 commits — Task 0 helper, three feature commits (Tasks 1-3), CHANGELOG commit.

---

## Self-review checklist

- **Spec coverage:**
  - Spec §5.5 → Task 1 (new signature, caller update, gate, 2 migrated + 3 new tests). ✓
  - Spec §5.6 → Task 2 (5 new schema keys + 3 pre-existing parses, ragged pad, CF-1.8, 6 tests). ✓
  - Spec §5.4 → Task 3 (3 StepOutput fields + pairing invariant, collector, averaging, writer, threaded Grid, 9 tests incl. parity). ✓
  - Spec Non-goals preserved: no spatial TL/size/mortality/egg; no Ev-OSMOSE; no per-fishery; no CSV spatial; no retroactive migration; no debug outputs. ✓
- **Placeholder scan:** No TBD/TODO. Every code block is complete. Parity test is not skipped. ✓
- **Type consistency:** `_enabled(cfg, key)` used throughout (not `_parse_bool`). `state.cell_x`/`state.cell_y` (not `cell_id`). `write_outputs(outputs, output_dir, config, prefix, *, grid)` signature consistent between definition and every caller-update instruction. `_collect_outputs(state, config, grid, step, ...)` consistent. `write_outputs_netcdf_spatial(outputs, output_dir, prefix, sim_index, config, grid)` consistent. ✓
- **Mortality cause coord:** 8 capitalized members (`Predation`, `Starvation`, ..., `Aging`) matching the existing CSV writer at `osmose/engine/output.py:161`. ✓
- **Test runner:** `.venv/bin/python -m pytest` throughout. No bare `python`, no `$()`. ✓
- **Commit granularity:** Task 0 (helper) + 3 feature commits + CHANGELOG = 5 commits. Task-0 helper is a prerequisite — reverting any feature commit leaves the helper and test file intact. ✓
- **Test count math:** baseline + 3 (Task 1) + 6 (Task 2) + 9 (Task 3) = baseline + 18. Re-run the baseline check in pre-flight to pin the concrete final number. ✓

---

## Execution handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task (starting with Task 0), two-stage review between tasks.
**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
