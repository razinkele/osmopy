# Codebase Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical and important inconsistencies, silent failures, and optimizations identified in the codebase analysis.

**Architecture:** 10 independent tasks grouped by domain. Each task is self-contained and can be committed independently. Tasks are ordered by severity (critical first) but have no inter-task dependencies.

**Tech Stack:** Python 3.12, Shiny for Python, pymoo, scipy, pandas, numpy, plotly, pytest

---

### Task 1: Fix Silent Calibration Error Handling

**Files:**
- Modify: `osmose/calibration/problem.py:84-99,117,129-132`
- Modify: `ui/pages/calibration.py:253-266,369-401,418-431,435-457`
- Modify: `ui/pages/setup.py:128-129`
- Modify: `ui/pages/grid.py:39-42`
- Test: `tests/test_calibration_problem.py`

**Step 1: Write failing tests for error logging in problem.py**

Add to `tests/test_calibration_problem.py`:

```python
def test_evaluate_logs_candidate_failure(tmp_path, caplog):
    """Silent except:pass should now log failures."""
    import logging
    from unittest.mock import patch
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=1.0)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    # Force _evaluate_candidate to raise
    with patch.object(problem, "_evaluate_candidate", side_effect=RuntimeError("boom")):
        import numpy as np
        X = np.array([[0.5]])
        out = {}
        with caplog.at_level(logging.WARNING):
            problem._evaluate(X, out)
        assert np.isinf(out["F"][0, 0])
        assert "boom" in caplog.text


def test_run_single_logs_subprocess_stderr(tmp_path, caplog):
    """Subprocess failures should log stderr content."""
    import logging
    from unittest.mock import patch, MagicMock
    from osmose.calibration.problem import OsmoseCalibrationProblem, FreeParameter

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=1.0)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = b"Java OutOfMemoryError"
    with patch("subprocess.run", return_value=mock_result):
        with caplog.at_level(logging.WARNING):
            result = problem._run_single({}, run_id=0)
        assert result == [float("inf")]
        assert "OutOfMemoryError" in caplog.text
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py::test_evaluate_logs_candidate_failure tests/test_calibration_problem.py::test_run_single_logs_subprocess_stderr -v`
Expected: FAIL — no logging output captured

**Step 3: Fix problem.py error handling**

In `osmose/calibration/problem.py`:

Move `import subprocess` to module level (line 5 area):
```python
import subprocess
```

Replace lines 84-89 (parallel path):
```python
                    try:
                        objectives = future.result()
                        for k, obj_val in enumerate(objectives):
                            F[i, k] = obj_val
                    except Exception as exc:
                        _log.warning("Candidate %d failed: %s", i, exc)
```

Replace lines 92-97 (sequential path):
```python
                try:
                    objectives = self._evaluate_candidate(i, params)
                    for k, obj_val in enumerate(objectives):
                        F[i, k] = obj_val
                except Exception as exc:
                    _log.warning("Candidate %d failed: %s", i, exc)
```

Remove `import subprocess` from line 117 inside `_run_single`.

Replace lines 131-132:
```python
        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors="replace")[:500] if result.stderr else ""
            _log.warning("OSMOSE run %d failed (exit %d): %s", run_id, result.returncode, stderr_msg)
            return [float("inf")] * self.n_obj
```

**Step 4: Fix calibration.py — add user notifications for silent returns**

In `ui/pages/calibration.py`, replace lines 254-266 in `handle_start_cal`:
```python
        selected = collect_selected_params(input, state)
        if not selected:
            cal_history.set([])
            ui.notification_show("Select at least one parameter to calibrate.", type="warning")
            return

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            ui.notification_show(f"JAR not found: {jar_path}", type="error")
            return

        obs_bio = input.observed_biomass()
        obs_diet = input.observed_diet()
        if not obs_bio and not obs_diet:
            ui.notification_show("Upload observed data (biomass or diet CSV).", type="warning")
            return
```

Replace lines 418-431 in `handle_sensitivity` with same pattern:
```python
        selected = collect_selected_params(input, state)
        if not selected:
            ui.notification_show("Select at least one parameter for sensitivity analysis.", type="warning")
            return

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            ui.notification_show(f"JAR not found: {jar_path}", type="error")
            return
```

And after line 431 where `obs_bio` is checked:
```python
        obs_bio = input.observed_biomass()
        if not obs_bio:
            ui.notification_show("Upload observed biomass CSV for sensitivity analysis.", type="warning")
            return
```

**Step 5: Wrap NSGA-II thread in try/except (lines 369-401)**

Replace the `run_optimization` function body:
```python
            def run_optimization():
                try:
                    from pymoo.algorithms.moo.nsga2 import NSGA2
                    from pymoo.optimize import minimize
                    from pymoo.termination import get_termination

                    algorithm = NSGA2(pop_size=pop_size)
                    termination = get_termination("n_gen", generations)

                    def append_history(val):
                        current = cal_history.get()
                        cal_history.set(current + [val])

                    callback = _make_progress_callback(
                        cal_history_append=append_history,
                        cancel_check=cancel_flag.get,
                    )

                    res = minimize(
                        problem,
                        algorithm,
                        termination,
                        seed=42,
                        verbose=False,
                        callback=callback,
                    )

                    if res.F is not None:
                        cal_F.set(res.F)
                        cal_X.set(res.X)
                except Exception as exc:
                    surrogate_status.set(f"Calibration failed: {exc}")
```

**Step 6: Fix sensitivity silent inf (lines 446-457)**

Replace the sensitivity eval loop exception handler:
```python
                try:
                    prob = OsmoseCalibrationProblem(
                        free_params=build_free_params(selected),
                        objective_fns=[lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df)],
                        base_config_path=base_config,
                        jar_path=jar_path,
                        work_dir=sens_work_dir / f"sens_{idx}",
                    )
                    result = prob._run_single(overrides, run_id=idx)
                    Y[idx] = result[0]
                except Exception as exc:
                    _log.warning("Sensitivity sample %d failed: %s", idx, exc)
                    Y[idx] = float("inf")
```

Add `from osmose.logging import setup_logging` and `_log = setup_logging("osmose.calibration.ui")` at the top of `calibration.py` (after existing imports).

**Step 7: Fix setup.py silent except (line 128-129)**

Replace:
```python
                except Exception:
                    pass
```
With:
```python
                except Exception as exc:
                    _log.debug("Could not update input %s: %s", input_id, exc)
```

Add `from osmose.logging import setup_logging` and `_log = setup_logging("osmose.setup")` at the top of `setup.py`.

**Step 8: Fix grid.py broad except (lines 39-42)**

Replace:
```python
    try:
        return pd.read_csv(full_path, header=None).values
    except Exception:
        return None
```
With:
```python
    try:
        return pd.read_csv(full_path, header=None).values
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    except Exception as exc:
        _log.warning("Failed to load mask %s: %s", full_path, exc)
        return None
```

Add `from osmose.logging import setup_logging` and `_log = setup_logging("osmose.grid.ui")` at the top of `grid.py`.

**Step 9: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -v`
Expected: PASS

**Step 10: Commit**

```bash
git add osmose/calibration/problem.py ui/pages/calibration.py ui/pages/setup.py ui/pages/grid.py tests/test_calibration_problem.py
git commit -m "fix: replace silent except:pass with logging and user notifications"
```

---

### Task 2: Fix Reactive Isolate Violations

**Files:**
- Modify: `ui/pages/movement.py:36,62`
- Modify: `ui/pages/advanced.py:88-93,136`
- Modify: `ui/pages/scenarios.py:72-74`

**Step 1: Fix movement.py — isolate config reads**

Replace `species_movement_panels` (line 34-40):
```python
    @render.ui
    def species_movement_panels():
        per_species = [f for f in MOVEMENT_FIELDS if f.indexed and "map" not in f.key_pattern]
        with reactive.isolate():
            n_species = int(state.config.get().get("simulation.nspecies", "3"))
        panels = []
        for i in range(n_species):
            panels.extend([render_field(f, species_idx=i) for f in per_species])
        return ui.div(*panels)
```

Replace `sync_species_movement_inputs` (lines 59-65):
```python
    @reactive.effect
    def sync_species_movement_inputs():
        per_species = [f for f in MOVEMENT_FIELDS if f.indexed and "map" not in f.key_pattern]
        with reactive.isolate():
            n_species = int(state.config.get().get("simulation.nspecies", "3"))
        for i in range(n_species):
            keys = [f.resolve_key(i) for f in per_species]
            sync_inputs(input, state, keys)
```

**Step 2: Fix advanced.py — remove side effect from render, isolate config read**

Replace `import_preview` (lines 82-128):
```python
    @render.ui
    def import_preview():
        pending = import_pending.get()
        if not pending:
            return ui.div()

        with reactive.isolate():
            current_cfg = state.config.get()
        diff = compute_import_diff(current_cfg, pending)
        if not diff:
            return ui.div(
                ui.p("No changes detected in imported file.", style=COLOR_MUTED),
            )

        rows = []
        for d in diff:
            old_display = d["old"] if d["old"] is not None else "(new)"
            rows.append(
                ui.tags.tr(
                    ui.tags.td(d["key"], style=STYLE_MONO_KEY),
                    ui.tags.td(
                        str(old_display),
                        style=COLOR_DANGER if d["old"] is not None else COLOR_MUTED,
                    ),
                    ui.tags.td(str(d["new"]), style=COLOR_SUCCESS),
                )
            )

        return ui.div(
            ui.h6(f"Import Preview: {len(diff)} change(s) detected"),
            ui.tags.div(
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("Key"),
                            ui.tags.th("Current"),
                            ui.tags.th("New Value"),
                        )
                    ),
                    ui.tags.tbody(*rows),
                    class_="table table-striped table-sm",
                ),
                style=STYLE_SCROLL_TABLE,
            ),
            ui.input_action_button(
                "confirm_import", "Confirm Import", class_="btn-success w-100 mt-2"
            ),
        )

    @reactive.effect
    def _clear_empty_import():
        """Clear import_pending when diff is empty (moved out of render)."""
        pending = import_pending.get()
        if not pending:
            return
        with reactive.isolate():
            current_cfg = state.config.get()
        diff = compute_import_diff(current_cfg, pending)
        if not diff:
            import_pending.set({})
```

Replace `confirm_import` (lines 130-139) — add isolate:
```python
    @reactive.effect
    @reactive.event(input.confirm_import)
    def confirm_import():
        pending = import_pending.get()
        if not pending:
            return
        with reactive.isolate():
            cfg = dict(state.config.get())
        cfg.update(pending)
        state.config.set(cfg)
        import_pending.set({})
```

**Step 3: Fix scenarios.py _bump — add isolate**

Replace lines 72-74:
```python
    def _bump():
        """Increment the refresh trigger to force re-render of scenario list."""
        with reactive.isolate():
            current = refresh_trigger.get()
        refresh_trigger.set(current + 1)
```

**Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ui/pages/movement.py ui/pages/advanced.py ui/pages/scenarios.py
git commit -m "fix: add reactive.isolate() to prevent infinite loop risks"
```

---

### Task 3: Consolidate Registry Construction

**Files:**
- Modify: `osmose/schema/__init__.py`
- Modify: `ui/state.py:10-43`
- Modify: `tests/test_schema_all.py` (find the `build_full_registry` function)
- Modify: `tests/test_integration.py` (find the `_build_full_registry` function)
- Test: `tests/test_schema_all.py`

**Step 1: Add ALL_FIELDS and build_registry to schema __init__**

Replace `osmose/schema/__init__.py`:
```python
from osmose.schema.base import OsmoseField, ParamType
from osmose.schema.registry import ParameterRegistry

from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from osmose.schema.grid import GRID_FIELDS
from osmose.schema.predation import PREDATION_FIELDS
from osmose.schema.fishing import FISHING_FIELDS
from osmose.schema.movement import MOVEMENT_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.output import OUTPUT_FIELDS
from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.economics import ECONOMICS_FIELDS

ALL_FIELDS: list[list[OsmoseField]] = [
    SIMULATION_FIELDS,
    SPECIES_FIELDS,
    GRID_FIELDS,
    PREDATION_FIELDS,
    FISHING_FIELDS,
    MOVEMENT_FIELDS,
    LTL_FIELDS,
    OUTPUT_FIELDS,
    BIOENERGETICS_FIELDS,
    ECONOMICS_FIELDS,
]


def build_registry() -> ParameterRegistry:
    """Build a ParameterRegistry with all OSMOSE parameter definitions."""
    reg = ParameterRegistry()
    for fields in ALL_FIELDS:
        for f in fields:
            reg.register(f)
    return reg


__all__ = ["OsmoseField", "ParamType", "ParameterRegistry", "ALL_FIELDS", "build_registry"]
```

**Step 2: Simplify ui/state.py**

Replace lines 1-43 of `ui/state.py`:
```python
"""Shared reactive application state for all UI pages."""

from __future__ import annotations

from pathlib import Path

from shiny import reactive

from osmose.runner import RunResult
from osmose.schema import build_registry


REGISTRY = build_registry()
```

(Remove the 10 individual field imports and `_build_registry` function.)

**Step 3: Update tests/test_schema_all.py**

Replace the `build_full_registry` function with:
```python
from osmose.schema import build_registry
```
And replace all calls to `build_full_registry()` with `build_registry()`.

**Step 4: Update tests/test_integration.py**

Replace the `_build_full_registry` function with:
```python
from osmose.schema import build_registry
```
And replace all calls to `_build_full_registry()` with `build_registry()`.

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_schema_all.py tests/test_integration.py tests/test_state.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add osmose/schema/__init__.py ui/state.py tests/test_schema_all.py tests/test_integration.py
git commit -m "refactor: consolidate registry construction into schema.__init__"
```

---

### Task 4: Fix Log Base Inconsistency in Size Spectrum

**Files:**
- Modify: `osmose/analysis.py:134-135`
- Test: `tests/test_analysis.py`

**Step 1: Write test verifying consistency**

Add to `tests/test_analysis.py`:
```python
def test_size_spectrum_slope_uses_log10():
    """Slope should be computed with log10 to match plotting convention."""
    import numpy as np
    from osmose.analysis import size_spectrum_slope
    import pandas as pd

    # Known power law: abundance = 1000 * size^-2
    sizes = np.array([1, 10, 100, 1000], dtype=float)
    abundances = 1000 * sizes ** -2.0

    df = pd.DataFrame({"size": sizes, "abundance": abundances})
    slope, intercept, r2 = size_spectrum_slope(df)

    # With log10: slope should be exactly -2.0
    assert abs(slope - (-2.0)) < 0.01
    assert r2 > 0.99
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analysis.py::test_size_spectrum_slope_uses_log10 -v`
Expected: FAIL — slope will be -2.0 * ln(10) ≈ -4.6 with natural log

**Step 3: Fix analysis.py to use log10**

Replace lines 134-135:
```python
    log_size = np.log10(spectrum_df["size"].values.astype(float))
    log_abundance = np.log10(spectrum_df["abundance"].values.astype(float))
```

**Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osmose/analysis.py tests/test_analysis.py
git commit -m "fix: standardize size spectrum slope to log10 (ecological convention)"
```

---

### Task 5: Fix Registry Performance

**Files:**
- Modify: `osmose/schema/registry.py:13-19,30-35,37-43`
- Test: `tests/test_registry.py`

**Step 1: Write performance assertion test**

Add to `tests/test_registry.py`:
```python
def test_match_field_uses_cache(registry):
    """match_field should be fast on repeated lookups (cached)."""
    # First call populates cache
    field1 = registry.match_field("simulation.time.ndtperyear")
    # Second call should hit cache
    field2 = registry.match_field("simulation.time.ndtperyear")
    assert field1 is field2


def test_categories_preserves_order(registry):
    """categories() should return unique categories in insertion order."""
    cats = registry.categories()
    assert len(cats) == len(set(cats))  # No duplicates
    assert len(cats) > 0
```

**Step 2: Fix registry.py — pre-compile regexes and add match cache**

Replace the entire `ParameterRegistry` class:
```python
class ParameterRegistry:
    """Collects all OSMOSE parameter definitions and provides lookup/validation."""

    def __init__(self):
        self._fields: list[OsmoseField] = []
        self._by_pattern: dict[str, OsmoseField] = {}
        self._compiled: list[tuple[re.Pattern, OsmoseField]] = []
        self._match_cache: dict[str, OsmoseField | None] = {}
        self._categories: list[str] = []

    def register(self, field: OsmoseField) -> None:
        self._fields.append(field)
        self._by_pattern[field.key_pattern] = field
        regex_str = re.escape(field.key_pattern).replace(r"\{idx\}", r"\d+")
        self._compiled.append((re.compile(regex_str), field))
        self._match_cache.clear()
        if field.category not in self._categories:
            self._categories.append(field.category)

    def all_fields(self) -> list[OsmoseField]:
        return list(self._fields)

    def fields_by_category(self, category: str) -> list[OsmoseField]:
        return [f for f in self._fields if f.category == category]

    def get_field(self, key_pattern: str) -> OsmoseField | None:
        return self._by_pattern.get(key_pattern)

    def categories(self) -> list[str]:
        return list(self._categories)

    def match_field(self, concrete_key: str) -> OsmoseField | None:
        """Match a concrete key like 'species.k.sp0' to its field pattern."""
        if concrete_key in self._match_cache:
            return self._match_cache[concrete_key]
        for compiled_re, field in self._compiled:
            if compiled_re.fullmatch(concrete_key):
                self._match_cache[concrete_key] = field
                return field
        self._match_cache[concrete_key] = None
        return None

    def validate(self, config: dict[str, object]) -> list[str]:
        """Validate a flat config dict against registered field constraints."""
        errors = []
        for key, value in config.items():
            field = self.match_field(key)
            if field:
                field_errors = field.validate_value(value)
                for e in field_errors:
                    errors.append(f"{key}: {e}")
        return errors
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_registry.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add osmose/schema/registry.py tests/test_registry.py
git commit -m "perf: pre-compile regexes and cache match_field lookups in registry"
```

---

### Task 6: Fix Config Reader — Add Cycle Guard and Missing File Warning

**Files:**
- Modify: `osmose/config/reader.py:31-38`
- Test: `tests/test_config_reader.py`

**Step 1: Write test for circular reference**

Add to `tests/test_config_reader.py`:
```python
def test_circular_reference_does_not_recurse(tmp_path):
    """Circular sub-file references should not cause infinite recursion."""
    from osmose.config.reader import OsmoseConfigReader

    # File A references File B, File B references File A
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    file_a.write_text(f"osmose.configuration.b ; {file_b.name}\nfoo ; bar\n")
    file_b.write_text(f"osmose.configuration.a ; {file_a.name}\nbaz ; qux\n")

    reader = OsmoseConfigReader()
    result = reader.read(file_a)
    assert result["foo"] == "bar"
    assert result["baz"] == "qux"


def test_missing_subfile_logs_warning(tmp_path, caplog):
    """Missing sub-config files should log a warning."""
    import logging
    from osmose.config.reader import OsmoseConfigReader

    master = tmp_path / "master.csv"
    master.write_text("osmose.configuration.sub ; nonexistent.csv\nfoo ; bar\n")

    reader = OsmoseConfigReader()
    with caplog.at_level(logging.WARNING):
        result = reader.read(master)
    assert result["foo"] == "bar"
    assert "nonexistent" in caplog.text
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py::test_circular_reference_does_not_recurse tests/test_config_reader.py::test_missing_subfile_logs_warning -v`
Expected: FAIL — first test hits RecursionError, second has no warning

**Step 3: Fix reader.py**

Replace `_read_recursive` (lines 31-38):
```python
    def _read_recursive(self, filepath: Path, flat: dict[str, str], _seen: set[Path] | None = None) -> None:
        if _seen is None:
            _seen = set()
        resolved = filepath.resolve()
        if resolved in _seen:
            _log.warning("Circular config reference skipped: %s", filepath)
            return
        _seen.add(resolved)
        file_params = self.read_file(filepath)
        flat.update(file_params)
        for key, value in file_params.items():
            if key.startswith("osmose.configuration."):
                sub_path = filepath.parent / value.strip()
                if sub_path.exists():
                    self._read_recursive(sub_path, flat, _seen)
                else:
                    _log.warning("Referenced sub-config not found: %s (from key %s)", sub_path, key)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osmose/config/reader.py tests/test_config_reader.py
git commit -m "fix: add cycle guard and missing sub-file warning to config reader"
```

---

### Task 7: Deduplicate Objective Functions and Theme Helpers

**Files:**
- Modify: `osmose/calibration/objectives.py:10-105`
- Modify: `ui/pages/results.py:18-23`
- Modify: `ui/pages/calibration.py:219-223`
- Test: `tests/test_objectives.py`

**Step 1: Refactor objectives.py — extract common RMSE helper**

Replace lines 10-105:
```python
def _timeseries_rmse(
    simulated: pd.DataFrame,
    observed: pd.DataFrame,
    value_col: str,
    species: str | None = None,
) -> float:
    """Generic RMSE for aligned time series with an optional species filter."""
    if species:
        simulated = simulated[simulated["species"] == species]
        observed = observed[observed["species"] == species]

    merged = pd.merge(simulated, observed, on="time", suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")

    diff = merged[f"{value_col}_sim"] - merged[f"{value_col}_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def biomass_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """Root mean square error of biomass time series."""
    return _timeseries_rmse(simulated, observed, "biomass", species)


def abundance_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """RMSE for abundance time series."""
    return _timeseries_rmse(simulated, observed, "abundance", species)


def yield_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """RMSE for yield time series."""
    return _timeseries_rmse(simulated, observed, "yield", species)


def diet_distance(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """Frobenius norm distance between diet composition matrices.

    Both DataFrames should be square matrices with predator rows and prey columns.
    """
    sim_vals = simulated.select_dtypes(include=[np.number]).values
    obs_vals = observed.select_dtypes(include=[np.number]).values

    if sim_vals.shape != obs_vals.shape:
        return float("inf")

    return float(np.linalg.norm(sim_vals - obs_vals, "fro"))


def _binned_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame
) -> float:
    """RMSE for 2D binned outputs (catch-at-size, size-at-age).

    Both DataFrames should have 'time', 'bin', and 'value' columns.
    """
    merged = pd.merge(simulated, observed, on=["time", "bin"], suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")

    diff = merged["value_sim"] - merged["value_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def catch_at_size_distance(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """RMSE between 2D catch-at-size outputs."""
    return _binned_rmse(simulated, observed)


def size_at_age_rmse(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """RMSE between 2D size-at-age outputs."""
    return _binned_rmse(simulated, observed)
```

**Step 2: Remove duplicate theme helpers from results.py and calibration.py**

In `ui/pages/results.py`, replace lines 18-23:
```python
def _tpl(input=None) -> str:
    """Return the Plotly template name for the current theme."""
    try:
        return "osmose" if input and input.theme_mode() == "dark" else "osmose-light"
    except (AttributeError, TypeError):
        return "osmose-light"
```

In `ui/pages/calibration.py`, replace lines 219-223:
```python
    def _tmpl() -> str:
        try:
            return "osmose" if input.theme_mode() == "dark" else "osmose-light"
        except (AttributeError, TypeError):
            return "osmose-light"
```

(Narrow the except from `Exception` to `(AttributeError, TypeError)` in both.)

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_objectives.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add osmose/calibration/objectives.py ui/pages/results.py ui/pages/calibration.py tests/test_objectives.py
git commit -m "refactor: deduplicate RMSE objectives and narrow theme except clauses"
```

---

### Task 8: Fix UI Cleanup — Debug Prints, Unused Imports, Temp File Safety

**Files:**
- Modify: `ui/pages/grid.py:12,185,237,250,260`
- Modify: `ui/pages/scenarios.py:49-64,208`
- Modify: `ui/components/help_modal.py:86`

**Step 1: Remove debug prints from grid.py**

Delete lines 237 and 250 (the `print(f"[GRID DEBUG]...")` statements).

**Step 2: Move `import math` to module level in grid.py**

Add `import math` near line 5 (with other stdlib imports). Remove `import math` from line 260.

**Step 3: Remove unused geojson_layer import in grid.py**

Change line 9-19 import to remove `geojson_layer`:
```python
from shiny_deckgl import (
    MapWidget,
    polygon_layer,
    CARTO_POSITRON,
    CARTO_DARK,
    zoom_widget,
    compass_widget,
    scale_widget,
    deck_legend_control,
)
```

**Step 4: Fix tooltip — add "type" to cell data in grid.py**

In `_build_grid_layers`, add `"type"` to cell dicts. Around line 101-104:
```python
                cell = {
                    "polygon": [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]],
                    "row": row, "col": col,
                    "type": "land" if is_land else "ocean",
                }
```

Move the `is_land` check before the cell dict construction:
```python
                is_land = (
                    mask is not None
                    and row < mask.shape[0]
                    and col < mask.shape[1]
                    and mask[row, col] <= 0
                )
                cell = {
                    "polygon": [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]],
                    "row": row, "col": col,
                    "type": "land" if is_land else "ocean",
                }
                if is_land:
                    land_cells.append(cell)
                else:
                    ocean_cells.append(cell)
```

**Step 5: Fix scenarios.py — replace mktemp with mkdtemp**

Replace lines 206-210:
```python
    @render.download(filename="osmose_scenarios.zip")
    def export_all_scenarios():
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        zip_path = tmp_dir / "osmose_scenarios.zip"
        mgr.export_all(zip_path)
        return str(zip_path)
```

**Step 6: Remove unnecessary layout_columns wrapper in scenarios.py**

Replace lines 49-64:
```python
        ui.card(
            ui.card_header("Bulk Operations"),
            ui.download_button(
                "export_all_scenarios",
                "Export All (ZIP)",
                class_="btn-primary w-100",
            ),
            ui.input_file(
                "import_scenarios_zip",
                "Import Scenarios (ZIP)",
                accept=[".zip"],
            ),
        ),
```

**Step 7: Fix stale GitHub URL in help_modal.py**

Change line 86:
```python
[View on GitHub](https://github.com/razinkele/osmopy)
```

**Step 8: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_grid.py tests/test_ui_scenarios.py tests/test_app_structure.py -v`
Expected: PASS

**Step 9: Commit**

```bash
git add ui/pages/grid.py ui/pages/scenarios.py ui/components/help_modal.py
git commit -m "fix: remove debug prints, fix temp file safety, update GitHub URL"
```

---

### Task 9: Fix Reporting and Results Edge Cases

**Files:**
- Modify: `osmose/reporting.py:12-44,51`
- Modify: `osmose/results.py:24-30`
- Modify: `osmose/grid.py:82-84`
- Test: `tests/test_reporting.py`
- Test: `tests/test_results.py`

**Step 1: Write test for generate_report unsupported format**

Add to `tests/test_reporting.py`:
```python
def test_generate_report_rejects_unsupported_format(tmp_path, mock_results):
    """Non-html format should raise NotImplementedError."""
    import pytest
    from osmose.reporting import generate_report

    with pytest.raises(NotImplementedError, match="csv"):
        generate_report(mock_results, {}, tmp_path / "report.csv", fmt="csv")
```

**Step 2: Write test for list_outputs on missing directory**

Add to `tests/test_results.py`:
```python
def test_list_outputs_missing_dir_returns_empty():
    """list_outputs on non-existent dir should return empty list."""
    from osmose.results import OsmoseResults

    res = OsmoseResults(Path("/nonexistent/xyz"))
    assert res.list_outputs() == []
```

**Step 3: Fix reporting.py — reject unsupported format**

Add at the top of `generate_report` (after line 61):
```python
    if fmt != "html":
        raise NotImplementedError(f"Report format '{fmt}' is not supported. Use 'html'.")
```

**Step 4: Fix reporting.py — rename summary_table to avoid collision**

Rename `summary_table` to `report_summary_table` at line 12 and update the call at line 64:
```python
def report_summary_table(results: OsmoseResults) -> pd.DataFrame:
```
And line 64:
```python
    table = report_summary_table(results)
```

**Step 5: Fix results.py — handle missing output_dir**

Replace `list_outputs` (lines 24-30):
```python
    def list_outputs(self) -> list[str]:
        """List all output files in the output directory."""
        if not self.output_dir.exists():
            return []
        files = []
        for f in sorted(self.output_dir.iterdir()):
            if f.suffix in (".csv", ".nc"):
                files.append(f.name)
        return files
```

**Step 6: Fix grid.py — log warning on shape mismatch**

Add logger to `osmose/grid.py`:
```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.grid")
```

Replace lines 82-84:
```python
        arr = np.loadtxt(csv_file, delimiter=",")
        if arr.shape != (nlat, nlon):
            _log.warning("Skipping %s: shape %s != (%d, %d)", csv_file.name, arr.shape, nlat, nlon)
            continue
```

**Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_reporting.py tests/test_results.py tests/test_grid_creation.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add osmose/reporting.py osmose/results.py osmose/grid.py tests/test_reporting.py tests/test_results.py
git commit -m "fix: handle missing dirs, reject unsupported report format, rename summary_table"
```

---

### Task 10: Fix Multiphase Objective Stub and Type Inconsistency

**Files:**
- Modify: `osmose/calibration/multiphase.py:32-34,64-97`
- Test: `tests/test_multiphase.py`

**Step 1: Write test for _optimize_phase with real objective**

Add to `tests/test_multiphase.py`:
```python
def test_optimize_phase_runs_real_objective(tmp_path):
    """_optimize_phase should use the provided objective_fn, not a stub."""
    from osmose.calibration.multiphase import MultiPhaseCalibrator, CalibrationPhase
    from osmose.calibration.problem import FreeParameter

    phase = CalibrationPhase(
        name="test",
        free_params=[FreeParameter(key="x", lower_bound=-5, upper_bound=5)],
        algorithm="Nelder-Mead",
        max_iter=50,
    )

    def real_objective(x):
        return float((x[0] - 2.0) ** 2)

    calibrator = MultiPhaseCalibrator(phases=[phase])
    result = calibrator._optimize_phase(phase, {}, str(tmp_path), objective_fn=real_objective)
    # Optimum should be near x=2.0
    assert abs(result["x"] - 2.0) < 0.5


def test_optimize_phase_differential_evolution(tmp_path):
    """differential_evolution branch should work with provided objective."""
    from osmose.calibration.multiphase import MultiPhaseCalibrator, CalibrationPhase
    from osmose.calibration.problem import FreeParameter

    phase = CalibrationPhase(
        name="test",
        free_params=[FreeParameter(key="x", lower_bound=-5, upper_bound=5)],
        algorithm="differential_evolution",
        max_iter=20,
    )

    def real_objective(x):
        return float((x[0] - 1.0) ** 2)

    calibrator = MultiPhaseCalibrator(phases=[phase])
    result = calibrator._optimize_phase(phase, {}, str(tmp_path), objective_fn=real_objective)
    assert abs(result["x"] - 1.0) < 1.0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_multiphase.py::test_optimize_phase_runs_real_objective -v`
Expected: FAIL — `_optimize_phase` doesn't accept `objective_fn`

**Step 3: Fix multiphase.py**

Change `run` signature (line 32-34) to accept `objective_fn` and fix `work_dir` type:
```python
    def run(
        self,
        work_dir: Path | str,
        objective_fn: Callable[[np.ndarray], float] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[dict[str, float]]:
```

Pass `objective_fn` through to `_optimize_phase` in line 55:
```python
            optimized = self._optimize_phase(phase, fixed_params, work_dir, objective_fn)
```

Change `_optimize_phase` signature (lines 64-68):
```python
    def _optimize_phase(
        self,
        phase: CalibrationPhase,
        fixed_params: dict[str, float],
        work_dir: Path | str,
        objective_fn: Callable[[np.ndarray], float] | None = None,
    ) -> dict[str, float]:
```

Replace the objective function definition (lines 83-85):
```python
        if objective_fn is None:
            raise ValueError(
                "objective_fn is required. Pass a callable that maps parameter vector to scalar."
            )

        def objective(x):
            return objective_fn(x)
```

Add `from pathlib import Path` to imports if not already present.

**Step 4: Update existing tests that mock _optimize_phase**

Existing tests that mock `_optimize_phase` should still pass since they patch the method entirely. No changes needed to existing mocked tests.

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_multiphase.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add osmose/calibration/multiphase.py tests/test_multiphase.py
git commit -m "fix: require objective_fn in MultiPhaseCalibrator, fix work_dir type"
```

---

## Summary

| Task | Domain | Severity | Key Changes |
|------|--------|----------|-------------|
| 1 | Error handling | Critical | Log calibration failures, show user notifications |
| 2 | Reactive patterns | Critical | Add isolate() to prevent infinite loops |
| 3 | Architecture | Important | Single registry construction point |
| 4 | Analysis | Important | Standardize log base to log10 |
| 5 | Performance | Important | Pre-compile regexes, cache match_field |
| 6 | Config | Important | Cycle guard + missing file warning |
| 7 | Code quality | Important | Deduplicate RMSE functions, narrow excepts |
| 8 | UI cleanup | Important | Remove debug prints, fix temp files, URL |
| 9 | Edge cases | Important | Handle missing dirs, reject bad format |
| 10 | Calibration | Important | Real objective_fn, fix type annotation |
