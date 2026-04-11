# Deep Review V2 — Phases 3-5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the remaining 6 real tasks from the 2026-04-05 deep review v2 plan — 4 medium-severity hardening fixes, 2 engine test-coverage additions, and 1 performance optimization — without breaking parity or existing tests.

**Architecture:** Single feature branch off `master`. One commit per task, TDD where behavior changes. Two groups: Phase 3 (4 medium fixes) followed by a pytest gate, then Phase 4-5 (engine tests + vectorization) followed by a final gate. No parity-migration run required (none of these tasks touch hot Numba paths; M13 changes how temperature is read but uses the same underlying values).

**Tech Stack:** Python 3.12, NumPy, pandas, Numba, xarray, pytest, ruff.

**Spec:** Derived from `docs/superpowers/plans/2026-04-05-deep-review-fixes-plan.md` (original v2 plan). This plan replaces Phases 3-5 of that document after the 2026-04-11 drift audit.

---

## Scope & drift audit findings

The original 2026-04-05 plan had 8 tasks across Phases 3-5. A main-thread spot-check on 2026-04-11 found 2 of them already resolved in master:

- **M8 (Close `OsmoseResults` in calibration)** — `osmose/calibration/problem.py` already uses `with OsmoseResults(output_dir, strict=False) as results:`. No action needed.
- **Task 26 (Diet tracking teardown + diet-in-mortality test)** — obsolete: the prescribed assertions reference `mort._pred_mod._diet_matrix`, but diet tracking was refactored to context-based (`ctx.diet_matrix`) in v0.6.0. No module-level global exists to test against.

This plan therefore contains the 6 remaining real tasks:

| # | ID | Title | Area | Commit type |
|---|---|---|---|---|
| 1 | M2 | Multi-year spawning season per-year normalization | engine config loader | fix |
| 2 | M7 | Reset `key_case_map` between reader calls | config reader | fix |
| 3 | M10 | Enforce 10 MB limit on ZIP entries in scenario import | scenarios.py | fix |
| 4 | M6 | Graceful skip on invalid scenario names in `import_all` | scenarios.py | fix |
| 5 | H18 | Add engine test coverage: `output.step0.include` + partial flush | tests | test |
| 6 | M13 | Vectorize temperature lookups in bioenergetics | simulate.py | perf |

---

## Phase 3 — Medium-Severity Hardening

### Task 1: Fix multi-year spawning season normalization (M2)

**Files:**
- Modify: `osmose/engine/config.py` — `_load_spawning_seasons` at lines ~397-446, specifically the normalization block around lines 433-437
- Test: `tests/test_engine_config.py`

**Context:** `_load_spawning_seasons(cfg, n_species, n_dt_per_year)` loads per-species season CSVs. When `reproduction.normalisation.enabled=true`, the loader currently computes `total = vals.sum(); vals = vals / total`, which divides the *entire* multi-year array by a single total. This collapses all years' weights into a single joint distribution summing to 1.0 over N years, not 1.0 per year. For a 2-year series `[1,2,3,4,5,6,7,8]` with `n_dt_per_year=4`, the buggy result is `[1,2,3,4,5,6,7,8] / 36 ≈ [0.028, 0.056, ..., 0.222]`, where each year's chunk sums to 0.278/0.722 rather than 1.0/1.0. Per-year normalization must divide each `n_dt_per_year`-sized chunk independently.

- [ ] **Step 0: Baseline pytest count**

Run: `.venv/bin/python -m pytest tests/ -q` and note the passing count. As of 2026-04-11 post-Phase-2 merge the baseline is 2091, but if master has moved since the plan was written, use whatever the current count is and treat each subsequent Phase 3 task as baseline+1, baseline+2, baseline+3, baseline+4. Record the observed baseline in your task report.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engine_config.py`:

```python
def test_spawning_season_normalization_per_year(tmp_path):
    """Normalization must divide each per-year chunk independently, not the total sum."""
    from osmose.engine.config import _load_spawning_seasons

    csv_path = tmp_path / "season_sp0.csv"
    csv_path.write_text("step;value\n0;1\n1;2\n2;3\n3;4\n4;5\n5;6\n6;7\n7;8\n")
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.season.file.sp0": "season_sp0.csv",
        "reproduction.normalisation.enabled": "true",
    }
    seasons = _load_spawning_seasons(cfg, n_species=1, n_dt_per_year=4)
    assert seasons is not None
    # Year 1 chunk [1,2,3,4] must sum to 1.0 after per-year normalization
    year1_sum = seasons[0, 0:4].sum()
    np.testing.assert_allclose(year1_sum, 1.0, atol=1e-10)
    # Year 2 chunk [5,6,7,8] must also sum to 1.0
    year2_sum = seasons[0, 4:8].sum()
    np.testing.assert_allclose(year2_sum, 1.0, atol=1e-10)
```

If `numpy as np` isn't imported at the top of `tests/test_engine_config.py`, add it.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py::test_spawning_season_normalization_per_year -v`

Expected: FAIL. With total-sum normalization, `year1_sum` will be about 0.278 (10/36), not 1.0.

- [ ] **Step 3: Fix the normalization block**

In `osmose/engine/config.py`, inside `_load_spawning_seasons`, locate the block (lines ~433-437):

```python
            if normalize:
                # Normalize per year: sum over each n_dt_per_year chunk
                total = vals.sum()
                if total > 0:
                    vals = vals / total
```

Replace it with per-year chunk normalization:

```python
            if normalize:
                # Normalize each n_dt_per_year-sized chunk independently so
                # every year's weights sum to 1.0, not the whole multi-year array.
                n_years_in_vals = max(1, n_vals // n_dt_per_year)
                for yr in range(n_years_in_vals):
                    s = yr * n_dt_per_year
                    e = min(s + n_dt_per_year, n_vals)
                    chunk_sum = vals[s:e].sum()
                    if chunk_sum > 0:
                        vals[s:e] = vals[s:e] / chunk_sum
```

Note: the comment above the old code already claimed per-year normalization; the comment was accurate, the implementation wasn't. Keep the intent visible in the new comment.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py::test_spawning_season_normalization_per_year -v`

Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: 2092 passed (2091 baseline + 1 new test), 15 skipped, 0 failed.

- [ ] **Step 6: Ruff check**

Run: `.venv/bin/ruff check osmose/engine/config.py tests/test_engine_config.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "fix: normalize spawning season per-year chunk instead of total (M2)"
```

---

### Task 2: Reset `key_case_map` between reader calls (M7)

**Files:**
- Modify: `osmose/config/reader.py` — `read()` method at line ~32
- Test: `tests/test_config_reader.py`

**Context:** `OsmoseConfigReader.key_case_map` is a dict initialized once in `__init__` and written by `read_file()` during every read. The `read()` entry point resets `self.skipped_lines = 0` but not `self.key_case_map`, so calling `reader.read(file_a)` followed by `reader.read(file_b)` leaves `file_a`'s keys lingering in `key_case_map`. If `file_b` uses different case for the same keys, the writer (which consults `key_case_map` for output case preservation) may round-trip `file_b` with `file_a`'s casing — a subtle data-corruption bug in reader reuse scenarios (e.g. the Shiny app, which holds one long-lived reader).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_reader.py`:

```python
def test_key_case_map_reset_between_reads(tmp_path):
    """key_case_map must not carry stale entries from previous read() calls."""
    from osmose.config.reader import OsmoseConfigReader

    f1 = tmp_path / "cfg1.csv"
    f1.write_text("Species.Name.sp0 ; Anchovy\n")
    f2 = tmp_path / "cfg2.csv"
    f2.write_text("simulation.time.nyear ; 5\n")

    reader = OsmoseConfigReader()
    reader.read(f1)
    assert "species.name.sp0" in reader.key_case_map

    reader.read(f2)
    assert "species.name.sp0" not in reader.key_case_map, (
        "Stale key from previous read() leaked into key_case_map"
    )
    assert "simulation.time.nyear" in reader.key_case_map
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py::test_key_case_map_reset_between_reads -v`

Expected: FAIL. The assertion `"species.name.sp0" not in reader.key_case_map` fails because the key persists across the second `read()` call.

- [ ] **Step 3: Reset `key_case_map` at the start of `read()`**

In `osmose/config/reader.py`, find `read()` (around line 32). The body begins with:

```python
    def read(self, master_file: Path) -> dict[str, str]:
        """Recursively read a master config and all referenced sub-configs."""
        self.skipped_lines = 0
        master_file = Path(master_file)
```

Add one line immediately after `self.skipped_lines = 0`:

```python
    def read(self, master_file: Path) -> dict[str, str]:
        """Recursively read a master config and all referenced sub-configs."""
        self.skipped_lines = 0
        self.key_case_map = {}
        master_file = Path(master_file)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py::test_key_case_map_reset_between_reads -v`

Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: 2093 passed, 15 skipped, 0 failed.

- [ ] **Step 6: Ruff check**

Run: `.venv/bin/ruff check osmose/config/reader.py tests/test_config_reader.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add osmose/config/reader.py tests/test_config_reader.py
git commit -m "fix: reset key_case_map between read() calls on same reader (M7)"
```

---

### Task 3: Enforce 10 MB limit on ZIP entries in scenario import (M10)

**Files:**
- Modify: `osmose/scenarios.py` — `ScenarioManager.import_all()` at lines ~177-201
- Test: `tests/test_scenarios.py`

**Context:** `import_all` reads every JSON entry in an uploaded ZIP via `zf.read(name)` with no size guard. A malicious or malformed ZIP can exhaust memory by declaring a small compressed size with a huge uncompressed size (ZIP bomb). Set a 10 MB per-entry limit — OSMOSE scenario JSONs are almost never larger than a few hundred KB, so 10 MB is a generous safety margin.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenarios.py` (create the file if it doesn't exist; check first with `ls tests/test_scenarios.py`):

```python
def test_import_all_rejects_oversized_zip_entries(tmp_path, caplog):
    """ZIP entries larger than 10 MB must be skipped with a warning, not read."""
    import json
    import zipfile
    import logging
    from osmose.scenarios import ScenarioManager

    storage = tmp_path / "scenarios"
    storage.mkdir()
    mgr = ScenarioManager(storage)

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # A legitimate small scenario
        small = {
            "name": "ok",
            "description": "",
            "config": {},
            "tags": [],
            "parent_scenario": None,
        }
        zf.writestr("ok.json", json.dumps(small))
        # An oversized entry: 11 MB of valid JSON (above the 10 MB cap)
        big = {"name": "big", "filler": "x" * (11 * 1024 * 1024)}
        zf.writestr("big.json", json.dumps(big))

    with caplog.at_level(logging.WARNING):
        count = mgr.import_all(zip_path)

    assert count == 1, f"Only the small scenario should import, got {count}"
    # ScenarioManager.save() writes to storage_dir/<name>/scenario.json
    assert (storage / "ok" / "scenario.json").exists(), (
        "Small scenario should have been saved to storage/ok/scenario.json"
    )
    assert any("oversized" in rec.message.lower() for rec in caplog.records), (
        "An oversize warning should have been logged"
    )
```

If the imports above aren't already present at the top of the test file, add them in the new test function locally (as shown) rather than at module level — this keeps the diff focused and avoids shadowing other tests' imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_import_all_rejects_oversized_zip_entries -v`

Expected: FAIL. Either the test passes the 11 MB entry to `Scenario(...)` (which will likely raise on validation) or the test asserts on a warning that doesn't yet exist.

- [ ] **Step 3: Add the size guard**

In `osmose/scenarios.py`, inside `import_all()`, find the current body:

```python
    def import_all(self, zip_path: Path) -> int:
        """Import scenarios from a ZIP file. Returns count of imported scenarios."""
        count = 0
        storage_resolved = self.storage_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name))
```

Insert the size guard as the first check inside the loop, before `zf.read(name)`:

```python
    def import_all(self, zip_path: Path) -> int:
        """Import scenarios from a ZIP file. Returns count of imported scenarios."""
        count = 0
        storage_resolved = self.storage_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                info = zf.getinfo(name)
                if info.file_size > 10 * 1024 * 1024:
                    _log.warning(
                        "Skipping oversized ZIP entry: %s (%d bytes)",
                        name,
                        info.file_size,
                    )
                    continue
                data = json.loads(zf.read(name))
```

`info.file_size` is the *uncompressed* size declared in the central directory — exactly the quantity that matters for memory bounding. (`info.compress_size` is the on-disk size, which a ZIP bomb misrepresents.)

`_log` is already the module-level logger in `osmose/scenarios.py` — confirm with a quick `grep -n "^_log" osmose/scenarios.py` before editing.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_import_all_rejects_oversized_zip_entries -v`

Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: 2094 passed, 15 skipped, 0 failed.

- [ ] **Step 6: Ruff check**

Run: `.venv/bin/ruff check osmose/scenarios.py tests/test_scenarios.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add osmose/scenarios.py tests/test_scenarios.py
git commit -m "fix: reject oversized ZIP entries in scenario import_all (M10)"
```

---

### Task 4: Graceful skip on invalid scenario names in `import_all` (M6)

**Files:**
- Modify: `osmose/scenarios.py` — `ScenarioManager.import_all()` (same method touched by Task 3; around lines ~192-200)
- Test: `tests/test_scenarios.py`

**Context:** After Task 3, `import_all` still calls `Scenario(name=scenario_name, ...)` unguarded. `Scenario`'s dataclass or `__post_init__` may raise `ValueError` on invalid characters, reserved names, or empty strings. One bad entry currently aborts the whole import, leaving later entries unprocessed. The existing path-escape check (`target.is_relative_to(storage_resolved)`) only handles one failure mode. Wrap the construction in `try/except ValueError` so one bad name skips that entry and lets the loop continue.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenarios.py`:

```python
def test_import_all_skips_invalid_scenario_names(tmp_path, caplog):
    """A single bad scenario name must not abort the import of the rest."""
    import json
    import zipfile
    import logging
    from osmose.scenarios import ScenarioManager, Scenario

    storage = tmp_path / "scenarios"
    storage.mkdir()
    mgr = ScenarioManager(storage)

    # Scenario(name="") raises ValueError("Scenario name must not be empty")
    # per osmose/scenarios.py:31-33. Verified 2026-04-11.

    zip_path = tmp_path / "mixed.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "bad.json",
            json.dumps({
                "name": "",
                "description": "invalid empty name",
                "config": {},
                "tags": [],
                "parent_scenario": None,
            }),
        )
        zf.writestr(
            "good.json",
            json.dumps({
                "name": "good_one",
                "description": "valid",
                "config": {},
                "tags": [],
                "parent_scenario": None,
            }),
        )

    with caplog.at_level(logging.WARNING):
        count = mgr.import_all(zip_path)

    assert count == 1, f"Only the valid scenario should import, got {count}"
    assert any(
        "invalid" in rec.message.lower() or "skipping" in rec.message.lower()
        for rec in caplog.records
    ), "A skip warning should have been logged for the bad entry"
```

The invalid-name premise was verified 2026-04-11: `Scenario(name="")` raises `ValueError("Scenario name must not be empty")` in `__post_init__` at `osmose/scenarios.py:31-33`. No additional probe needed.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_import_all_skips_invalid_scenario_names -v`

Expected: FAIL. The `ValueError` from constructing `Scenario(name="")` bubbles out of `import_all`, so `count` is never computed and the test crashes instead of asserting.

- [ ] **Step 3: Wrap `Scenario(...)` and `self.save(...)` in try/except**

In `osmose/scenarios.py`, inside `import_all()`, locate the block after the path-escape check:

```python
                scenario = Scenario(
                    name=scenario_name,
                    description=data.get("description", ""),
                    config=data.get("config", {}),
                    tags=data.get("tags", []),
                    parent_scenario=data.get("parent_scenario"),
                )
                self.save(scenario)
                count += 1
```

Replace with:

```python
                try:
                    scenario = Scenario(
                        name=scenario_name,
                        description=data.get("description", ""),
                        config=data.get("config", {}),
                        tags=data.get("tags", []),
                        parent_scenario=data.get("parent_scenario"),
                    )
                    self.save(scenario)
                    count += 1
                except ValueError as exc:
                    _log.warning(
                        "Skipping scenario with invalid name %r: %s",
                        scenario_name,
                        exc,
                    )
```

Only catch `ValueError` — `OSError` from `self.save()` should still bubble (disk errors are not one-bad-entry failures).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_import_all_skips_invalid_scenario_names -v`

Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: 2095 passed, 15 skipped, 0 failed.

- [ ] **Step 6: Ruff check**

Run: `.venv/bin/ruff check osmose/scenarios.py tests/test_scenarios.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add osmose/scenarios.py tests/test_scenarios.py
git commit -m "fix: gracefully skip invalid scenario names in import_all (M6)"
```

---

### Phase 3 gate

- [ ] **Run full suite**: `.venv/bin/python -m pytest tests/ -q`
  - Expected: 2095 passed, 15 skipped, 0 failed.
- [ ] **Run ruff**: `.venv/bin/ruff check osmose/ ui/ tests/`
  - Expected: `All checks passed!`
- [ ] **Run parity smoke**: `.venv/bin/python -m pytest tests/test_engine_parity.py -q`
  - Expected: 12 passed.

If any gate fails, stop and escalate. Do not proceed to Phase 4.

---

## Phase 4 — Engine test coverage

### Task 5: Add engine test coverage for `output.step0.include` and partial flush (H18)

**Files:**
- Modify: `tests/test_engine_simulate.py` — append two test functions after `test_age_distribution_uses_year_bins` (around line 160+)

**Context:** Two engine behaviors have no direct test coverage:

1. `config.output_step0_include` (from config key `output.step0.include=true`) prepends a `step=-1` snapshot output. Implementation at `osmose/engine/simulate.py:952-953`.
2. Non-divisible `output.recordfrequency.ndt` must still flush the tail accumulation (e.g. 12 sim steps with `freq=7` → 1 full group + 1 partial = ≥2 outputs). Implementation via `_average_step_outputs` and the main loop around `osmose/engine/simulate.py:1111-1124`.

Both tests use the existing `minimal_config` fixture defined at `tests/test_engine_simulate.py:12`. Because `minimal_config` is mutable dict, use `dict(minimal_config)` to avoid cross-test contamination (other tests in the same file follow this pattern).

- [ ] **Step 1: Add the first test — `output.step0.include`**

Append to `tests/test_engine_simulate.py`:

```python
def test_simulate_output_step0_include(minimal_config):
    """output.step0.include=true prepends a step=-1 snapshot to the output list."""
    cfg_dict = dict(minimal_config)
    cfg_dict["output.step0.include"] = "true"
    # Pin record frequency explicitly so this test is not coupled to the default.
    cfg_dict["output.recordfrequency.ndt"] = "1"
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)

    outputs = simulate(cfg, grid, rng)

    # First element must be the step=-1 snapshot
    assert outputs[0].step == -1, (
        f"Expected first output to be step=-1 snapshot, got step={outputs[0].step}"
    )
    # With record_freq=1 and n_dt_per_year*n_years=12, the normal run produces 12
    # regular outputs on top of the snapshot — total 13.
    assert len(outputs) == 13, f"Expected 13 outputs (12 + step0), got {len(outputs)}"
```

- [ ] **Step 2: Add the second test — partial flush**

```python
def test_simulate_partial_flush_non_divisible_record_freq(minimal_config):
    """A record frequency that doesn't divide n_steps must still flush the tail."""
    cfg_dict = dict(minimal_config)
    # 12 total steps, recording every 7 -> 1 full window (steps 0-6) + 1 partial window (7-11)
    cfg_dict["output.recordfrequency.ndt"] = "7"
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)

    outputs = simulate(cfg, grid, rng)

    # 12 / 7 = 1 remainder 5 -> at least 2 outputs (1 full, 1 partial flush)
    assert len(outputs) >= 2, f"Expected ≥2 outputs from partial flush, got {len(outputs)}"
    # Regression guard: the old buggy behavior was to drop the partial entirely,
    # producing exactly 1 output. Assert strictly more than 1.
    assert len(outputs) > 1, "Partial flush regression: tail accumulation was dropped"
```

- [ ] **Step 3: Run both new tests**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py::test_simulate_output_step0_include tests/test_engine_simulate.py::test_simulate_partial_flush_non_divisible_record_freq -v`

Expected: Both PASS. The implementations were verified 2026-04-11: `simulate.py:952-953` prepends the step=-1 snapshot when `output_step0_include` is true, and the main loop flushes accumulated tail in `_average_step_outputs`. If either test fails, do NOT modify production code — stop and escalate, because a failure means something has regressed since this plan was written.

- [ ] **Step 4: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: 2097 passed (2095 + 2 new), 15 skipped, 0 failed.

- [ ] **Step 5: Ruff check**

Run: `.venv/bin/ruff check tests/test_engine_simulate.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_simulate.py
git commit -m "test: add coverage for output.step0.include and partial flush (H18)"
```

---

## Phase 5 — Performance Optimization

### Task 6: Vectorize temperature lookups in bioenergetics (M13)

**Files:**
- Modify: `osmose/engine/simulate.py` — two blocks in `_bioen_step` (or its equivalent), around lines 244-259 and 283-289.

**Context:** The bioenergetics step currently reads per-school temperatures via a Python list comprehension over `temp_data.get_value(step, cell_y, cell_x)`, which is a scalar function call per school. For thousands of schools this is a measurable per-step cost. `PhysicalData.get_grid(step)` (in `osmose/engine/physical_data.py:69-75`) returns the full `(ny, nx)` grid for a timestep — NumPy fancy indexing with `grid[cell_y, cell_x]` gets the same values in one vectorized operation.

Verified prerequisites (all checked 2026-04-11):
- `get_grid()` exists at `physical_data.py:69-75` and returns the full `(ny, nx)` grid for a timestep.
- It raises `ValueError` for `is_constant=True` data, so it MUST only be called in the `else:` (spatially-explicit) branches that already guard on `is_constant`. The existing control flow at `simulate.py:233-259` and `simulate.py:281-292` does gate correctly.
- `sp_masks` is pre-filtered upstream at `simulate.py:207` to include only `(sp, mask)` pairs where `mask.any() == True`, so an explicit per-iteration empty-mask guard inside the loop is not needed. Do NOT add one.
- `PhysicalData._data` is always `float64` (forced via `raw.astype(np.float64)` in `from_netcdf` at `physical_data.py:48`), so fancy indexing returns `float64`, bitwise identical to what `float(get_value(...))` produces.
- `state.cell_x` and `state.cell_y` are `np.int32` arrays (declared at `state.py:52-53`), safe for NumPy fancy indexing.
- The current code iterates species-masks via `for sp, mask in sp_masks:`. The fix keeps this iteration pattern for the phi_t block but pulls `temp_grid` out of the loop (one lookup per timestep, not per species).
- The `temp_c_arr` block is already a flat loop across all schools and vectorizes trivially — no species iteration needed.

No new test: behavior is unchanged, only performance. The existing parity tests exercise both code paths and will catch any numerical divergence.

- [ ] **Step 1: Replace the phi_t spatial temperature lookup**

In `osmose/engine/simulate.py`, locate the block around line 244 (inside `_bioen_step`, under `if config.bioen_phit_enabled and temp_data is not None:` → `else:` branch):

```python
        else:
            # Spatially explicit: look up each school's cell
            phi_t_arr = np.empty(len(state), dtype=np.float64)
            for sp, mask in sp_masks:
                temps = np.array(
                    [
                        temp_data.get_value(step, int(state.cell_y[i]), int(state.cell_x[i]))
                        for i in np.where(mask)[0]
                    ]
                )
                phi_t_arr[mask] = phi_t_fn(
                    temps,
                    float(config.bioen_e_mobi[sp]),
                    float(config.bioen_e_d[sp]),
                    float(config.bioen_tp[sp]),
                )
```

Replace with:

```python
        else:
            # Spatially explicit: single vectorized grid lookup, then per-species phi_t.
            temp_grid = temp_data.get_grid(step)
            phi_t_arr = np.empty(len(state), dtype=np.float64)
            for sp, mask in sp_masks:
                temps = temp_grid[state.cell_y[mask], state.cell_x[mask]]
                phi_t_arr[mask] = phi_t_fn(
                    temps,
                    float(config.bioen_e_mobi[sp]),
                    float(config.bioen_e_d[sp]),
                    float(config.bioen_tp[sp]),
                )
```

The outer `temp_grid = temp_data.get_grid(step)` is hoisted outside the `for sp, mask in sp_masks:` loop so the grid is read once per timestep, not once per species.

- [ ] **Step 2: Replace the `temp_c_arr` spatial temperature lookup**

In the same file, locate the block around line 283 (a few lines further down):

```python
    elif temp_data is not None:
        temp_c_arr = np.array(
            [
                temp_data.get_value(step, int(state.cell_y[i]), int(state.cell_x[i]))
                for i in range(len(state))
            ]
        )
```

Replace with:

```python
    elif temp_data is not None:
        temp_grid = temp_data.get_grid(step)
        temp_c_arr = temp_grid[state.cell_y, state.cell_x]
```

Two lines instead of six, one vectorized lookup instead of one Python call per school.

- [ ] **Step 3: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: 2097 passed, 15 skipped, 0 failed.

Numerical parity check: the two code paths produce identical values because `temp_grid[cell_y, cell_x]` and `temp_data.get_value(step, cell_y, cell_x)` both return `temp_data._data[step % n_timesteps, cell_y, cell_x]` (verified by reading `physical_data.py:55-75`).

- [ ] **Step 4: Run parity suite**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q`

Expected: 12 passed. Parity baselines were frozen before this change; any divergence means the vectorization introduced a bug.

- [ ] **Step 5: Ruff check**

Run: `.venv/bin/ruff check osmose/engine/simulate.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "$(cat <<'EOF'
perf: vectorize temperature lookups in _bioen_step via get_grid (M13)

Replaces the per-school temp_data.get_value() list comprehension in both
the phi_t spatially-explicit branch and the temp_c_arr branch with a single
get_grid(step) lookup plus NumPy fancy indexing on (cell_y, cell_x). Same
values (verified: get_value returns _data[t_idx, cy, cx] and fancy indexing
returns _data[t_idx][cy, cx]), O(1) vectorized array ops per step instead
of O(n_schools) Python calls.
EOF
)"
```

---

## Phase 5 final gate

- [ ] **Run full suite**: `.venv/bin/python -m pytest tests/ -q`
  - Expected: 2097 passed, 15 skipped, 0 failed.
- [ ] **Run ruff**: `.venv/bin/ruff check osmose/ ui/ tests/`
  - Expected: `All checks passed!`
- [ ] **Run parity suite**: `.venv/bin/python -m pytest tests/test_engine_parity.py -q`
  - Expected: 12 passed.
- [ ] **Verify commit count**: `git log --oneline master..HEAD` on the feature branch
  - Expected: 6 commits, one per task (1 through 6).

If all gates pass, invoke `superpowers:finishing-a-development-branch` to merge/PR/cleanup.

---

## Out of scope (explicit non-goals)

These were in the original 2026-04-05 plan but are deliberately excluded from this one:

- **M8 (OsmoseResults close in calibration)** — already fixed on master (`with ... as results:` present).
- **Task 26 (diet-in-mortality test)** — obsolete after v0.6.0 refactor of diet tracking to context-based state.
- **H2/H15 `_tl_weighted_sum` Numba interface restructure** — deferred as architectural work; not a bug fix.
- **H9 ncell comment/code mismatch** — deferred pending Java reference verification.
- **M1-M5, M9, M11-M12, M14** — low-impact or latent-only issues per the 2026-04-05 deferrals list.
- **C3 phi_t semantic choice (ValueError vs silent Arrhenius fallback)** — deferred as a design decision, not a bug. Current silent-fallback behavior is acceptable.
