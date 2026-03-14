# OSMOSE-Python Enhancement Plan — Implementation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance OSMOSE-Python across 4 dimensions — production hardening, UX polish, new capabilities, and developer experience — in a single focused sprint of 18 tasks.

**Architecture:** Breadth-first approach — one high-impact item per dimension. Each task is self-contained. Wave 1 (11 parallel tasks) then Wave 2 (4 tasks) then Wave 3 (3 tasks). Breaking changes are allowed.

**Tech Stack:** Python 3.12+, Shiny, Plotly, pytest, asyncio, Jinja2, argparse, pyright, ruff, pre-commit

**Spec:** `docs/superpowers/specs/2026-03-11-enhancement-plan-design.md`

**Run tests:** `.venv/bin/python -m pytest`
**Run single test:** `.venv/bin/python -m pytest tests/test_file.py::test_name -v`
**Lint:** `.venv/bin/ruff check osmose/ ui/ tests/`
**Format:** `.venv/bin/ruff format osmose/ ui/ tests/`

---

## File Structure

### New files to create:
| File | Responsibility |
|------|---------------|
| `osmose/cli.py` | CLI entry point (argparse wrapper over runner/validator/reporter) |
| `osmose/history.py` | Run history tracking (JSON records per run) |
| `osmose/plotly_theme.py` | Canonical Plotly template + color constants |
| `osmose/templates/report.html` | Jinja2 HTML report template |
| `ui/pages/calibration_handlers.py` | Calibration event handlers (extracted from calibration.py) |
| `ui/pages/calibration_charts.py` | Calibration chart builders (extracted from calibration.py) |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_cli.py` | CLI tests |
| `tests/test_history.py` | Run history tests |
| `tests/test_plotly_theme.py` | Theme registration tests |
| `pyrightconfig.json` | pyright config |
| `.pre-commit-config.yaml` | Pre-commit hooks config |

### Files to modify:
| File | Changes |
|------|---------|
| `osmose/runner.py` | Add timeout_sec param, encoding safety |
| `osmose/analysis.py` | Add `_require_columns()` guards |
| `osmose/plotting.py` | Add column guards, 3 new chart functions, import theme from plotly_theme |
| `osmose/reporting.py` | Replace hardcoded HTML with Jinja2, return Path |
| `osmose/scenarios.py` | Atomic save/import with backup-rename pattern |
| `osmose/config/validator.py` | Add `validate_field()` for single-field validation |
| `ui/state.py` | Add `busy` and `dirty` reactive values |
| `ui/pages/run.py` | Pre-run validation gate, timeout input, history save |
| `ui/pages/results.py` | History tab, food web tab, dashboard tab |
| `ui/pages/calibration.py` | Slim down to layout + delegation |
| `ui/components/param_form.py` | Add min/max HTML attrs, field-level error rendering |
| `ui/charts.py` | Import from plotly_theme instead of defining own templates |
| `app.py` | Render loading overlay, dirty indicator, viewport meta |
| `www/osmose.css` | Loading overlay, dirty indicator, field-error, responsive styles |
| `pyproject.toml` | CLI entry point, pyright + pre-commit dev deps |
| `.github/workflows/ci.yml` | Coverage threshold, matrix, artifacts, pyright, Docker smoke |
| `Dockerfile` | HEALTHCHECK directive |

---

## Chunk 1: Production Hardening (Tasks 1-4)

### Task 1: Run timeout and cancellation

**Files:**
- Modify: `osmose/runner.py:56-114`
- Modify: `ui/pages/run.py:148-198`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test for timeout**

In `tests/test_runner.py`, add:

```python
import sys
import textwrap

async def test_run_timeout_kills_process(tmp_path):
    """A run that exceeds timeout should be killed and return code -1."""
    script = tmp_path / "slow.py"
    script.write_text(textwrap.dedent("""\
        import time
        time.sleep(60)
    """))
    runner = _ScriptRunner(jar_path=script, java_cmd=sys.executable)
    result = await runner.run(
        config_path=tmp_path / "fake.csv",
        timeout_sec=1,
    )
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()
```

Note: `_ScriptRunner` is defined in the existing test file. Import or reference it from there.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runner.py::test_run_timeout_kills_process -v`
Expected: FAIL because `run()` does not accept `timeout_sec`

- [ ] **Step 3: Implement timeout in OsmoseRunner.run()**

Edit `osmose/runner.py:56-114`. Add `timeout_sec: int | None = None` parameter. Wrap execution in `asyncio.wait_for()`. On `asyncio.TimeoutError`, kill the process and return `RunResult(returncode=-1, stderr="Run timed out after {timeout_sec}s")`. Also change `.decode()` to `.decode(errors="replace")` for encoding safety.

```python
async def run(
    self,
    config_path: Path,
    output_dir: Path | None = None,
    java_opts: list[str] | None = None,
    overrides: dict[str, str] | None = None,
    on_progress: Callable[[str], None] | None = None,
    timeout_sec: int | None = None,
) -> RunResult:
    cmd = self._build_cmd(config_path, output_dir, java_opts, overrides)
    _log.info("Starting OSMOSE: %s", " ".join(cmd))

    self._process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def read_stream(stream, lines_list):
        if stream is None:
            return
        async for line in stream:
            text = line.decode(errors="replace").rstrip()
            lines_list.append(text)
            if on_progress:
                on_progress(text)

    result_output_dir = output_dir or config_path.parent / "output"

    try:
        coro = asyncio.gather(
            read_stream(self._process.stdout, stdout_lines),
            read_stream(self._process.stderr, stderr_lines),
        )
        if timeout_sec is not None:
            await asyncio.wait_for(coro, timeout=timeout_sec)
        else:
            await coro
        await self._process.wait()
        _log.info("OSMOSE finished with exit code %d", self._process.returncode)
        return RunResult(
            returncode=self._process.returncode,
            output_dir=result_output_dir,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
        )
    except asyncio.TimeoutError:
        self._process.kill()
        await self._process.wait()
        _log.warning("OSMOSE run timed out after %ds", timeout_sec)
        return RunResult(
            returncode=-1,
            output_dir=result_output_dir,
            stdout="\n".join(stdout_lines),
            stderr=f"Run timed out after {timeout_sec}s",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_runner.py::test_run_timeout_kills_process -v`
Expected: PASS

- [ ] **Step 5: Run full runner test suite**

Run: `.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: All PASS (timeout=None is the default, no behavior change)

- [ ] **Step 6: Add timeout input to Run page**

Edit `ui/pages/run.py`. Add `ui.input_numeric("run_timeout", "Timeout (seconds)", value=3600, min=60, max=86400)` to `run_ui()`. In `handle_run()`, pass `timeout_sec=int(input.run_timeout())` to `runner.run()`.

- [ ] **Step 7: Commit**

```bash
git add osmose/runner.py ui/pages/run.py tests/test_runner.py
git commit -m "feat: add configurable timeout to OsmoseRunner.run()"
```

---

### Task 2: Pre-run config validation gate

**Files:**
- Modify: `ui/pages/run.py:147-198`
- Test: `tests/test_ui_run.py`

- [ ] **Step 1: Write tests for validation gate**

In `tests/test_ui_run.py`, add:

```python
from osmose.config.validator import validate_config, check_species_consistency
from osmose.schema import build_registry

def test_prerun_validation_blocks_on_errors():
    registry = build_registry()
    config = {"species.linf.sp0": "not_a_number", "simulation.nspecies": "2"}
    errors, warnings = validate_config(config, registry)
    assert len(errors) > 0
    assert "expected number" in errors[0]

def test_prerun_validation_passes_valid_config():
    registry = build_registry()
    config = {"simulation.nspecies": "3", "species.linf.sp0": "50.0"}
    errors, warnings = validate_config(config, registry)
    assert len(errors) == 0
```

- [ ] **Step 2: Run tests (should pass since they test existing validator)**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py::test_prerun_validation_blocks_on_errors tests/test_ui_run.py::test_prerun_validation_passes_valid_config -v`
Expected: PASS

- [ ] **Step 3: Wire validation into handle_run()**

Edit `ui/pages/run.py`. Add imports for `validate_config`, `check_file_references`, `check_species_consistency`. In `handle_run()`, before `status.set("Writing config...")`, add:

```python
config = state.config.get()
errors, warnings = validate_config(config, state.registry)
source_dir = state.config_dir.get()
if source_dir:
    file_errors = check_file_references(config, str(source_dir))
    errors.extend(file_errors)
species_warnings = check_species_consistency(config)
warnings.extend(species_warnings)

if errors:
    log_lines = ["--- VALIDATION ERRORS (run blocked) ---"]
    log_lines.extend(errors)
    if warnings:
        log_lines.append("--- WARNINGS ---")
        log_lines.extend(warnings)
    run_log.set(log_lines)
    status.set(f"Validation failed: {len(errors)} error(s)")
    return

if warnings:
    log_lines = ["--- WARNINGS (continuing anyway) ---"]
    log_lines.extend(warnings)
    run_log.set(log_lines)
```

- [ ] **Step 4: Run full run page tests**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ui/pages/run.py tests/test_ui_run.py
git commit -m "feat: validate config before run, block on errors"
```

---

### Task 3: DataFrame column guards in analysis

**Files:**
- Modify: `osmose/analysis.py:1-152`
- Modify: `osmose/plotting.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_analysis.py`, add:

```python
import pytest

def test_ensemble_stats_rejects_missing_value_col():
    df = pd.DataFrame({"time": [1, 2], "wrong_col": [10, 20]})
    with pytest.raises(ValueError, match="missing columns.*biomass"):
        ensemble_stats([df], value_col="biomass")

def test_shannon_diversity_rejects_missing_columns():
    df = pd.DataFrame({"time": [1], "wrong": [10]})
    with pytest.raises(ValueError, match="missing columns"):
        shannon_diversity(df)

def test_summary_table_rejects_missing_species_col():
    df = pd.DataFrame({"wrong": [1], "biomass": [10]})
    with pytest.raises(ValueError, match="missing columns.*species"):
        summary_table([df], value_col="biomass")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analysis.py::test_ensemble_stats_rejects_missing_value_col -v`
Expected: FAIL (raises KeyError, not ValueError)

- [ ] **Step 3: Add _require_columns() helper and guards**

Add to `osmose/analysis.py` after imports:

```python
def _require_columns(df: pd.DataFrame, *cols: str, context: str = "") -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{context}: missing columns {sorted(missing)}, got {sorted(df.columns)}"
        )
```

Add guards after empty checks in each function:
- `ensemble_stats()`: after `combined = pd.concat(...)`, add `_require_columns(combined, *group_cols, value_col, context="ensemble_stats")`
- `summary_table()`: after concat, add `_require_columns(combined, "species", value_col, context="summary_table")`
- `shannon_diversity()`: at top, add `_require_columns(biomass_df, "time", "biomass", context="shannon_diversity")`
- `mean_tl_catch()`: after merge, add `_require_columns(merged, "time", "species", "yield", "tl", context="mean_tl_catch")`
- `size_spectrum_slope()`: at top, add `_require_columns(spectrum_df, "size", "abundance", context="size_spectrum_slope")`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: All PASS

- [ ] **Step 5: Add guards to plotting functions**

Add the same `_require_columns()` to `osmose/plotting.py`. Read each `make_*` function to find its required columns, then add a guard after the empty-check. The required columns per function depend on the parameters — read the actual code first.

- [ ] **Step 6: Add guard to report_summary_table()**

In `osmose/reporting.py`, import `_require_columns` from `osmose.analysis` and add guards inside `report_summary_table()` after the `bio = results.biomass()` and `yld = results.yield_biomass()` calls:

```python
if not bio.empty:
    _require_columns(bio, "species", context="report_summary_table")
```

- [ ] **Step 7: Run plotting and reporting tests**

Run: `.venv/bin/python -m pytest tests/test_plotting.py tests/test_reporting.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add osmose/analysis.py osmose/plotting.py osmose/reporting.py tests/test_analysis.py
git commit -m "feat: add DataFrame column guards to analysis, plotting, and reporting"
```

---

### Task 4: Atomic scenario writes

**Files:**
- Modify: `osmose/scenarios.py:52-60`
- Test: `tests/test_scenarios.py`

- [ ] **Step 1: Write tests for atomic save**

In `tests/test_scenarios.py`, add:

```python
def test_save_overwrites_existing_scenario(tmp_path):
    manager = ScenarioManager(tmp_path)
    s1 = Scenario(name="test", config={"a": "1"})
    manager.save(s1)
    s2 = Scenario(name="test", config={"a": "2"})
    manager.save(s2)
    loaded = manager.load("test")
    assert loaded.config["a"] == "2"

def test_save_creates_new_scenario(tmp_path):
    manager = ScenarioManager(tmp_path)
    s = Scenario(name="brand_new", config={"x": "1"})
    manager.save(s)
    loaded = manager.load("brand_new")
    assert loaded.config["x"] == "1"

def test_save_backup_survives_rename_failure(tmp_path):
    """If os.rename fails putting new data in place, backup should survive."""
    import os
    from unittest.mock import patch

    manager = ScenarioManager(tmp_path)
    s1 = Scenario(name="test", config={"a": "original"})
    manager.save(s1)

    call_count = 0
    original_rename = os.rename

    def failing_rename(src, dst):
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # fail on the second rename (new -> target)
            raise OSError("Simulated failure")
        return original_rename(src, dst)

    s2 = Scenario(name="test", config={"a": "updated"})
    with patch("os.rename", side_effect=failing_rename):
        with pytest.raises(OSError, match="Simulated failure"):
            manager.save(s2)

    # Backup (.bak) should still exist with original data
    backup = tmp_path / "test.bak" / "scenario.json"
    assert backup.exists()
```

- [ ] **Step 2: Run tests (first two should pass, third should fail)**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_save_overwrites_existing_scenario tests/test_scenarios.py::test_save_creates_new_scenario -v`
Expected: PASS

- [ ] **Step 3: Implement atomic save with backup-rename**

Edit `osmose/scenarios.py`. Add `import os` and `import tempfile` to imports. Replace `save()` (lines 52-60):

```python
def save(self, scenario: Scenario) -> Path:
    """Save a scenario to disk using atomic write pattern."""
    scenario.modified_at = datetime.now().isoformat()
    target = self.storage_dir / scenario.name

    tmp_dir = Path(tempfile.mkdtemp(dir=self.storage_dir))
    data = asdict(scenario)
    try:
        with open(tmp_dir / "scenario.json", "w") as f:
            json.dump(data, f, indent=2)

        backup = None
        if target.exists():
            backup = target.with_suffix(".bak")
            os.rename(target, backup)
        os.rename(tmp_dir, target)
        if backup and backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    return target
```

- [ ] **Step 4: Run full scenario tests**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/scenarios.py tests/test_scenarios.py
git commit -m "feat: atomic scenario writes with backup-rename pattern"
```

---

## Chunk 2: UX Polish (Tasks 5-8)

### Task 5: Global loading overlay

**Files:**
- Modify: `ui/state.py:22-32`
- Modify: `app.py:35-131` and `app.py:134-147`
- Modify: `www/osmose.css`
- Modify: `ui/pages/run.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write failing test**

In `tests/test_state.py`, add:

```python
def test_appstate_has_busy_field():
    state = AppState()
    with reactive.isolate():
        assert state.busy.get() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_state.py::test_appstate_has_busy_field -v`
Expected: FAIL

- [ ] **Step 3: Add busy to AppState**

Edit `ui/state.py:32`. After `self.loading`, add:
```python
self.busy: reactive.Value[str | None] = reactive.Value(None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_state.py::test_appstate_has_busy_field -v`
Expected: PASS

- [ ] **Step 5: Add overlay UI to app.py**

In `app_ui`, add `ui.output_ui("loading_overlay"),` before the navset. In `server()`, add:

```python
@render.ui
def loading_overlay():
    msg = state.busy.get()
    if msg is None:
        return ui.div()
    return ui.div(
        ui.div(ui.div(class_="osm-spinner"), ui.p(msg), class_="osm-loading-content"),
        class_="osm-loading-overlay",
    )
```

- [ ] **Step 6: Add CSS for overlay**

Append to `www/osmose.css`:

```css
.osm-loading-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(15, 25, 35, 0.85);
    display: flex; align-items: center; justify-content: center;
    z-index: 9999;
}
.osm-loading-content { text-align: center; color: #e2e8f0; }
.osm-loading-content p { margin-top: 1rem; font-size: 1.1rem; }
.osm-spinner {
    width: 48px; height: 48px;
    border: 4px solid rgba(232, 168, 56, 0.2);
    border-top-color: #e8a838;
    border-radius: 50%;
    animation: osm-spin 0.8s linear infinite;
    margin: 0 auto;
}
@keyframes osm-spin { to { transform: rotate(360deg); } }
```

- [ ] **Step 7: Wire into run handler**

In `ui/pages/run.py` `handle_run()`, wrap run execution:

```python
state.busy.set("Running simulation...")
try:
    result = await runner.run(...)
finally:
    state.busy.set(None)
```

- [ ] **Step 8: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`

```bash
git add ui/state.py app.py www/osmose.css ui/pages/run.py tests/test_state.py
git commit -m "feat: global loading overlay for long operations"
```

---

### Task 6: Unsaved changes warning

**Files:**
- Modify: `ui/state.py:22-44`
- Modify: `app.py`
- Modify: `www/osmose.css`
- Test: `tests/test_state.py`

**Note:** Must run after Task 5 (both modify AppState and app.py).

- [ ] **Step 1: Write failing test**

In `tests/test_state.py`, add:

```python
def test_appstate_has_dirty_field():
    state = AppState()
    with reactive.isolate():
        assert state.dirty.get() is False

def test_update_config_sets_dirty():
    state = AppState()
    with reactive.isolate():
        state.update_config("simulation.nspecies", "5")
        assert state.dirty.get() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_state.py::test_appstate_has_dirty_field -v`
Expected: FAIL

- [ ] **Step 3: Add dirty to AppState and update_config**

In `ui/state.py`, add to `__init__`:
```python
self.dirty: reactive.Value[bool] = reactive.Value(False)
```

In `update_config()`, add `self.dirty.set(True)` after `self.config.set(cfg)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_state.py::test_appstate_has_dirty_field tests/test_state.py::test_update_config_sets_dirty -v`
Expected: PASS

- [ ] **Step 5: Add beforeunload JS handler**

In `app.py`'s JS `<script>` block (around line 40), add before the closing:

```javascript
window.addEventListener('beforeunload', function(e) {
    if (typeof Shiny !== 'undefined' && Shiny.shinyapp &&
        Shiny.shinyapp.$inputValues && Shiny.shinyapp.$inputValues.is_dirty) {
        e.preventDefault();
        e.returnValue = '';
    }
});
```

Also in `server()`, push the dirty state to JS:
```python
@reactive.effect
def _sync_dirty_to_js():
    from shiny import session as sess
    is_dirty = state.dirty.get()
    # Shiny auto-exposes reactive outputs as input values
```

- [ ] **Step 6: Add dirty banner to app.py**

In `app_ui`, add `ui.output_ui("dirty_banner"),` before the navset. In `server()`:

```python
@render.ui
def dirty_banner():
    if not state.dirty.get():
        return ui.div()
    return ui.div("You have unsaved changes", class_="osm-dirty-banner")
```

- [ ] **Step 6: Add CSS**

Append to `www/osmose.css`:

```css
.osm-dirty-banner {
    background: rgba(232, 168, 56, 0.15);
    border-left: 3px solid #e8a838;
    color: #e8a838;
    padding: 0.5rem 1rem;
    font-size: 0.85rem; font-weight: 500; text-align: center;
}
[data-theme="light"] .osm-dirty-banner {
    background: rgba(212, 148, 46, 0.1);
    border-left-color: #d4942e; color: #d4942e;
}
```

- [ ] **Step 7: Reset dirty on save/export**

Add `state.dirty.set(False)` in scenario save handler, config export handler, and example load handler.

- [ ] **Step 8: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`

```bash
git add ui/state.py app.py www/osmose.css ui/pages/scenarios.py ui/pages/advanced.py ui/pages/setup.py tests/test_state.py
git commit -m "feat: unsaved changes warning with dirty state tracking"
```

---

### Task 7: Field-level validation feedback

**Files:**
- Modify: `osmose/config/validator.py`
- Modify: `ui/components/param_form.py:31-154`
- Modify: `www/osmose.css`
- Test: `tests/test_param_form.py`

- [ ] **Step 1: Write failing test**

In `tests/test_param_form.py`, add:

```python
from osmose.config.validator import validate_field
from osmose.schema.base import OsmoseField, ParamType

def test_validate_field_rejects_out_of_bounds():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}", param_type=ParamType.FLOAT,
        default=50.0, min_val=1.0, max_val=200.0,
        description="L-infinity", category="growth",
    )
    error = validate_field("species.linf.sp0", "500.0", field)
    assert error is not None
    assert "above maximum" in error

def test_validate_field_accepts_valid_value():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}", param_type=ParamType.FLOAT,
        default=50.0, min_val=1.0, max_val=200.0,
        description="L-infinity", category="growth",
    )
    assert validate_field("species.linf.sp0", "100.0", field) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_param_form.py::test_validate_field_rejects_out_of_bounds -v`
Expected: FAIL

- [ ] **Step 3: Add validate_field() to validator.py**

Add to `osmose/config/validator.py` after `validate_config()`:

```python
def validate_field(key: str, value: str, field) -> str | None:
    """Validate a single field value. Returns error message or None."""
    if field.param_type in (ParamType.FLOAT, ParamType.INT):
        try:
            num = float(value)
        except (ValueError, TypeError):
            return f"Expected number, got '{value}'"
        if field.min_val is not None and num < field.min_val:
            return f"Value {num} below minimum {field.min_val}"
        if field.max_val is not None and num > field.max_val:
            return f"Value {num} above maximum {field.max_val}"
    elif field.param_type == ParamType.BOOL:
        if value.lower() not in ("true", "false", "0", "1"):
            return f"Expected boolean, got '{value}'"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_param_form.py::test_validate_field_rejects_out_of_bounds tests/test_param_form.py::test_validate_field_accepts_valid_value -v`
Expected: PASS

- [ ] **Step 5: Verify min/max HTML attributes exist**

Check `ui/components/param_form.py` — the FLOAT/INT branches already pass `min=field.min_val, max=field.max_val` to `ui.input_numeric()` (lines 93-94 and 109-110). No changes needed here. If they're missing, add them.

- [ ] **Step 6: Wire validate_field() into param_form rendering**

In `ui/components/param_form.py`, after creating the numeric widget, call `validate_field()` to check the initial value and render an error if invalid. This is a server-side concern — the full wiring into reactive effects happens at the page level. For now, `validate_field()` is available as a utility for page servers to call on input change and render `ui.div(error_msg, class_="field-error-msg")` below the field.

- [ ] **Step 7: Add CSS for field errors**

Append to `www/osmose.css`:

```css
.field-error-msg {
    color: #e74c3c; font-size: 0.75rem; margin-top: 0.25rem;
    padding-left: 0.5rem; border-left: 2px solid #e74c3c;
}
[data-theme="light"] .field-error-msg { color: #c0392b; border-left-color: #c0392b; }
```

- [ ] **Step 7: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_param_form.py -v`

```bash
git add osmose/config/validator.py ui/components/param_form.py www/osmose.css tests/test_param_form.py
git commit -m "feat: field-level validation with min/max bounds"
```

---

### Task 8: Responsive modals and mobile nav

**Files:**
- Modify: `app.py`
- Modify: `www/osmose.css`

- [ ] **Step 1: Add viewport meta tag**

Edit `app.py`. Add a new `ui.head_content()` line after the CSS include:

```python
ui.head_content(ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1")),
```

- [ ] **Step 2: Add responsive CSS**

Append to `www/osmose.css`:

```css
@media (max-width: 768px) {
    .nav-pills { flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch; white-space: nowrap; padding-bottom: 0.5rem; }
    .nav-pills .nav-item { flex-shrink: 0; }
    .osmose-badge { display: none; }
    .osmose-logo { font-size: 1.1rem; }
    .osmose-header { padding: 0.5rem 1rem; }
    .modal-dialog { width: 95vw; max-width: 95vw; margin: 1rem auto; }
    .modal-body { max-height: 70vh; overflow-y: auto; }
    .card-body { padding: 0.75rem; }
    .osmose-section-label { font-size: 0.7rem; }
}
```

- [ ] **Step 3: Manual verification**

Open app, resize to 768px, verify nav scrolls, badge hidden, modals full-width.

- [ ] **Step 4: Commit**

```bash
git add app.py www/osmose.css
git commit -m "feat: responsive modals and mobile nav at 768px breakpoint"
```

---

## Chunk 3: New Capabilities (Tasks 9-12)

### Task 9: CLI for batch runs

**Files:**
- Create: `osmose/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

**Dependencies:** Tasks 1, 2, 10

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
from unittest.mock import patch
import pytest
from osmose.cli import main

def test_cli_help():
    with patch("sys.argv", ["osmose", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

def test_cli_validate_missing_file():
    with patch("sys.argv", ["osmose", "validate", "/nonexistent.csv"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0

def test_cli_validate_valid_config(tmp_path):
    cfg = tmp_path / "test.csv"
    cfg.write_text("simulation.nspecies;3\n")
    with patch("sys.argv", ["osmose", "validate", str(cfg)]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code in (0, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Create osmose/cli.py**

Create `osmose/cli.py` with 4 subcommands: `validate`, `run`, `ensemble`, `report`. Each is a thin wrapper over existing library functions. Use `argparse`. See spec for full implementation. Key points:
- `validate` uses `OsmoseConfigReader`, `validate_config`, `check_file_references`, `check_species_consistency`
- `run` uses `asyncio.run(OsmoseRunner.run(...))`
- `ensemble` uses `asyncio.run(runner.run_ensemble(...))`
- `report` uses `generate_report(OsmoseResults(output_dir), ...)`
- Entry: `def main()` calls `parser.parse_args()` and dispatches to handler, exits with return code

- [ ] **Step 4: Add entry point to pyproject.toml**

Add after `[project.optional-dependencies]`:

```toml
[project.scripts]
osmose = "osmose.cli:main"
```

- [ ] **Step 5: Reinstall and test**

Run: `.venv/bin/pip install -e ".[dev]"` then `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 6: Verify CLI**

Run: `.venv/bin/osmose --help`
Expected: Shows subcommands

- [ ] **Step 7: Commit**

```bash
git add osmose/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: add CLI for batch runs, validation, and reporting"
```

---

### Task 10: Jinja2 HTML reports

**Files:**
- Modify: `osmose/reporting.py:48-122`
- Create: `osmose/templates/report.html`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_reporting.py`, add tests for: return type is Path, content uses Jinja2, custom template override works. Check existing test file for `mock_results` fixture.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reporting.py -v`
Expected: FAIL (returns None, not Path)

- [ ] **Step 3: Create Jinja2 template**

Create `osmose/templates/report.html` with sections: config summary, summary table, species details, charts placeholder. Uses `{{ metadata.nspecies }}`, `{{ summary_html | safe }}`, `{% for sp in species_details %}`, etc.

- [ ] **Step 4: Rewrite generate_report() to use Jinja2**

Change signature to accept `template_path: Path | None = None` and return `-> Path`. Use `jinja2.PackageLoader("osmose", "templates")` for default, `jinja2.FileSystemLoader` for custom. Keep `fmt` parameter name (not `format`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reporting.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/reporting.py osmose/templates/report.html tests/test_reporting.py
git commit -m "feat: Jinja2 HTML reports with custom template support"
```

---

### Task 11: Run history and experiment tracking

**Files:**
- Create: `osmose/history.py`
- Modify: `ui/pages/run.py`
- Test: `tests/test_history.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_history.py` with tests for: save and list records, sort by timestamp, compare runs.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_history.py -v`
Expected: FAIL

- [ ] **Step 3: Create osmose/history.py**

`RunRecord` dataclass with: timestamp, config_snapshot, duration_sec, output_dir, summary. `RunHistory` class with: save (JSON), list_runs (sorted newest first), load_run, compare_runs (returns DataFrame).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_history.py -v`
Expected: PASS

- [ ] **Step 5: Wire history save into run page**

In `ui/pages/run.py`, after successful run (returncode == 0), save a RunRecord with config snapshot and summary stats. Wrap in try/except (best-effort).

- [ ] **Step 6: Commit**

```bash
git add osmose/history.py tests/test_history.py ui/pages/run.py
git commit -m "feat: run history tracking with JSON records"
```

---

### Task 12: New chart types

**Files:**
- Modify: `osmose/plotting.py`
- Modify: `ui/pages/results.py`
- Test: `tests/test_plotting.py`

**Dependencies:** Task 11 (RunRecord), Task 18 (theme)

- [ ] **Step 1: Write failing tests**

In `tests/test_plotting.py`, add tests for `make_food_web()` (Sankey from diet matrix), `make_run_comparison()` (grouped bar chart from RunRecords), and `make_species_dashboard()` (small multiples with biomass + yield).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_plotting.py::test_make_food_web_returns_figure -v`
Expected: FAIL

- [ ] **Step 3: Implement make_food_web()**

Plotly Sankey diagram. Maps predator/prey to node indices, creates links weighted by proportion. Filters links below threshold (default 1%). Add `_require_columns` guard.

- [ ] **Step 4: Implement make_run_comparison()**

Grouped bar chart comparing summary stats across runs. Signature:
```python
def make_run_comparison(
    records: list[RunRecord],
    metrics: list[str] | None = None,
    template: str = TEMPLATE,
) -> go.Figure:
```

Uses `RunRecord.summary` dict from Task 11. X-axis = metric names, groups = runs (labeled by timestamp). If `metrics` is None, use all keys from the first record's summary. Returns `_empty_figure()` if no records.

- [ ] **Step 5: Implement make_species_dashboard()**

Uses `plotly.subplots.make_subplots()`. One row per species with biomass line and yield dashed line. Shared x-axis. Add `_require_columns` guards.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -v`
Expected: PASS

- [ ] **Step 7: Wire into Results page**

Add "Food Web", "Run Comparison", and "Dashboard" tabs to the results navset. Read existing tab pattern first.

- [ ] **Step 8: Commit**

```bash
git add osmose/plotting.py ui/pages/results.py tests/test_plotting.py
git commit -m "feat: add food web Sankey and species dashboard charts"
```

---

## Chunk 4: Developer Experience (Tasks 13-16)

### Task 13: Shared test fixtures via conftest.py

**Files:**
- Create: `tests/conftest.py`
- Modify: multiple test files

- [ ] **Step 1: Audit existing fixtures**

Read test files to find duplicated fixtures: sample configs, DataFrames, fake JARs, registry builds.

- [ ] **Step 2: Create tests/conftest.py**

Session-scoped: `registry` (build_registry). Function-scoped: `sample_config`, `sample_biomass_df`, `sample_yield_df`, `sample_diet_df`, `fake_jar`, `tmp_output_dir`.

- [ ] **Step 3: Update test files**

Replace local fixture definitions with conftest imports. Keep test-specific fixtures local.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All 400+ PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/*.py
git commit -m "refactor: consolidate shared test fixtures into conftest.py"
```

---

### Task 14: pyright type checking in CI

**Files:**
- Create: `pyrightconfig.json`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Create pyrightconfig.json**

```json
{
    "typeCheckingMode": "basic",
    "include": ["osmose", "ui"],
    "exclude": ["tests"],
    "pythonVersion": "3.12",
    "reportMissingImports": true,
    "reportMissingTypeStubs": false
}
```

- [ ] **Step 2: Add pyright to dev deps**

Add `"pyright>=1.1.350"` to `pyproject.toml` dev dependencies.

- [ ] **Step 3: Install and run locally**

Run: `.venv/bin/pip install -e ".[dev]"` then `.venv/bin/pyright`

- [ ] **Step 4: Fix type errors**

Fix each error. Common: Path|str coercion, optional returns, dict access. Estimate 10-20 fixes.

- [ ] **Step 5: Add pyright job to CI**

Add `type-check` job to `.github/workflows/ci.yml`.

- [ ] **Step 6: Verify zero errors**

Run: `.venv/bin/pyright`
Expected: 0 errors

- [ ] **Step 7: Commit**

```bash
git add pyrightconfig.json pyproject.toml .github/workflows/ci.yml osmose/ ui/
git commit -m "feat: add pyright type checking to CI"
```

---

### Task 15: Pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create .pre-commit-config.yaml**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 2: Add pre-commit to dev deps**

Add `"pre-commit>=3.5"` to `pyproject.toml`.

- [ ] **Step 3: Install and test**

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files
```

Expected: All hooks pass

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml
git commit -m "chore: add pre-commit hooks for ruff"
```

---

### Task 16: CI hardening

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `Dockerfile`

**Dependencies:** Task 14

- [ ] **Step 1: Add coverage threshold**

Change pytest line to: `pytest --cov=osmose --cov-report=term-missing --cov-report=html --cov-fail-under=95`

- [ ] **Step 2: Add coverage artifact upload**

Add `actions/upload-artifact@v4` step with `htmlcov/` path.

- [ ] **Step 3: Add Python version matrix**

Add `strategy.matrix.python-version: ["3.12", "3.13"]` to test job.

- [ ] **Step 4: Add Docker smoke test**

Add `docker` job with `docker build -t osmose-test .`

- [ ] **Step 5: Add HEALTHCHECK to Dockerfile**

Add curl install and `HEALTHCHECK CMD curl -f http://localhost:8000/ || exit 1`.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml Dockerfile
git commit -m "ci: coverage threshold, Python matrix, Docker smoke test"
```

---

## Chunk 5: Code Quality (Tasks 17-18)

### Task 17: Split calibration page

**Files:**
- Modify: `ui/pages/calibration.py` (545 LOC)
- Create: `ui/pages/calibration_handlers.py`
- Create: `ui/pages/calibration_charts.py`

- [ ] **Step 1: Read calibration.py fully**

Identify three sections: UI layout, event handlers, chart functions.

- [ ] **Step 2: Create calibration_charts.py**

Extract chart/visualization functions (convergence, Pareto, sensitivity plots).

- [ ] **Step 3: Create calibration_handlers.py**

Extract `register_calibration_handlers(input, output, session, state)` with all reactive handlers.

- [ ] **Step 4: Slim down calibration.py**

Keep only `calibration_ui()` and `calibration_server()` that delegates to handlers.

- [ ] **Step 5: Run existing tests**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration.py tests/test_ui_calibration_handlers.py -v`
Expected: All PASS

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add ui/pages/calibration.py ui/pages/calibration_handlers.py ui/pages/calibration_charts.py
git commit -m "refactor: split calibration page into layout, handlers, and charts"
```

---

### Task 18: Extract Plotly theme

**Files:**
- Create: `osmose/plotly_theme.py`
- Modify: `osmose/plotting.py:12-40`
- Modify: `ui/charts.py`
- Test: `tests/test_plotly_theme.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_plotly_theme.py`:

```python
import plotly.io as pio

def test_osmose_templates_registered():
    from osmose.plotly_theme import ensure_templates
    ensure_templates()
    assert "osmose" in pio.templates
    assert "osmose-light" in pio.templates

def test_osmose_colors_has_8_entries():
    from osmose.plotly_theme import OSMOSE_COLORS
    assert len(OSMOSE_COLORS) == 8

def test_get_plotly_template():
    from osmose.plotly_theme import get_plotly_template
    assert get_plotly_template("dark") == "osmose"
    assert get_plotly_template("light") == "osmose-light"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_plotly_theme.py -v`
Expected: FAIL

- [ ] **Step 3: Create osmose/plotly_theme.py**

Move the canonical template from `ui/charts.py` (more complete version with axis styling, tick fonts, borders). Export: `OSMOSE_COLORS`, `OSMOSE_COLORS_LIGHT`, `ensure_templates()`, `get_plotly_template()`, `PLOTLY_TEMPLATE`, `PLOTLY_TEMPLATE_LIGHT`.

- [ ] **Step 4: Update osmose/plotting.py**

Replace `_ensure_template()` and inline colorway with imports from `plotly_theme`.

- [ ] **Step 5: Update ui/charts.py**

Replace template definitions with re-exports from `osmose.plotly_theme`.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_plotly_theme.py tests/test_plotting.py -v`
Expected: All PASS

- [ ] **Step 7: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add osmose/plotly_theme.py osmose/plotting.py ui/charts.py tests/test_plotly_theme.py
git commit -m "refactor: extract Plotly theme to osmose/plotly_theme.py"
```

---

## Execution Order Summary

```
Wave 1 (parallel, 11 tasks):
  Task 1:  Run timeout
  Task 3:  DataFrame column guards
  Task 4:  Atomic scenario writes
  Task 5 -> Task 6:  Loading overlay -> Dirty state (sequential)
  Task 7:  Field-level validation
  Task 8:  Responsive CSS
  Task 13: conftest.py
  Task 15: Pre-commit hooks
  Task 17: Split calibration page
  Task 18: Extract Plotly theme

Wave 2 (4 tasks, after Wave 1):
  Task 2:  Pre-run validation gate
  Task 10: Jinja2 reports
  Task 11: Run history
  Task 14: pyright in CI

Wave 3 (3 tasks, after Wave 2):
  Task 9:  CLI
  Task 12: New chart types
  Task 16: CI hardening

Final verification:
  .venv/bin/python -m pytest -v
  .venv/bin/ruff check osmose/ ui/ tests/
  .venv/bin/pyright
```
