# OSMOSE-Python Enhancement Plan

**Date:** 2026-03-11
**Scope:** Comprehensive single-sprint enhancement (18 tasks)
**Approach:** Breadth-first — one high-impact item per dimension
**Breaking changes:** Allowed (clean slate)
**Audience:** Mixed — non-technical web UI users + technical CLI/scripting users
**Deployment:** Single server, small team

---

## Context

OSMOSE-Python is a mature codebase (~6,700 LOC core+UI, 400 tests, 98% coverage) with full R parity achieved. This plan addresses four dimensions simultaneously: production hardening, UX polish, new capabilities, and developer experience, plus two code quality tasks.

## Design Principles

- **Each task is self-contained** — no task depends on another unless explicitly noted
- **Preserve the schema-driven architecture** — all new features flow through `OsmoseField` and `ParameterRegistry`
- **CLI reuses core library** — thin wrapper over existing `runner`, `validator`, `reporting` modules
- **File-based state** — no database; JSON + filesystem for history and scenarios
- **CSS-first UI changes** — minimize Python changes for visual improvements

---

## Section 1: Production Hardening

### Task 1 — Run timeout & cancellation

**Files:** `osmose/runner.py`, `ui/pages/run.py`

Add `timeout_sec: int = 3600` parameter to `OsmoseRunner.run()`. Enforce via `asyncio.wait_for()`. On timeout:
- Kill the subprocess
- Return `RunResult(returncode=-1, stderr="Run timed out after {timeout_sec}s")`

In the Run page, add a numeric input for timeout (default 3600s, min 60, max 86400). Pass to `runner.run()`.

**Tests:** Async test with a sleep script that exceeds timeout; verify process killed and result reflects timeout.

### Task 2 — Pre-run config validation gate

**Files:** `ui/pages/run.py`, `osmose/config/validator.py`

Before calling `runner.run()`:
1. Call `validate_config(config, registry)` — type/bounds errors
2. Call `check_file_references(config, base_dir)` — missing data files
3. Call `check_species_consistency(config)` — nspecies vs species.name.spN

If errors: display in console with red styling, block the run.
If warnings only: display in amber, allow running.

**Tests:** Mock validator to return errors/warnings; verify run is blocked/allowed accordingly.

### Task 3 — DataFrame column guards in analysis

**Files:** `osmose/analysis.py`, `osmose/plotting.py`

Add helper:
```python
def _require_columns(df: pd.DataFrame, *cols: str, context: str = "") -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {missing}, got {list(df.columns)}")
```

Call at the top of: `ensemble_stats()`, `summary_table()` (in `analysis.py`), `report_summary_table()` (in `reporting.py`), `shannon_diversity()`, `mean_tl_catch()`, `size_spectrum_slope()`, and all 6 existing plotting functions. If Task 12 lands first, also guard the 3 new chart functions.

**Tests:** Pass DataFrames with wrong columns; verify clear ValueError messages.

### Task 4 — Atomic scenario writes

**File:** `osmose/scenarios.py`

Change `save()`:
1. Write to `tempfile.mkdtemp(dir=storage_dir)`
2. If `target` exists, rename it to `target_backup` via `os.rename(target, target_backup)`
3. `os.rename(tmp, target)` to put new data in place
4. `shutil.rmtree(target_backup)` to clean up old data

This minimizes the risk window: if the process dies between steps 2 and 3, the backup still exists. Note: directory rename is not fully atomic on Linux (unlike file rename), but this 3-step pattern avoids the data-loss window of rmtree-then-rename.

Same pattern for `import_all()` ZIP extraction.

**Tests:** Mock `os.rename` to raise mid-write; verify no partial state left.

---

## Section 2: UX Polish

### Task 5 — Global loading overlay

**Files:** `ui/state.py`, `app.py`, `www/osmose.css`

Add `state.busy: reactive.Value[str | None]` (default `None`).

When set, render a full-page overlay:
```html
<div class="osm-loading-overlay">
  <div class="osm-spinner"></div>
  <p>{busy_message}</p>
</div>
```

CSS: semi-transparent navy background (`rgba(15, 25, 35, 0.85)`), centered spinner (CSS-only animation), z-index above all content.

Wire into: example loading, run execution, scenario import, calibration start. Pattern:
```python
state.busy.set("Running simulation...")
try:
    result = await runner.run(...)
finally:
    state.busy.set(None)
```

**Tests:** Set busy → verify overlay renders; clear → verify removed.

### Task 6 — Unsaved changes warning

**Files:** `ui/state.py`, `app.py`, `www/osmose.css`

**Note:** Tasks 5 and 6 both modify `ui/state.py` and `app.py`. Implement sequentially or by the same implementer to avoid merge conflicts in `AppState.__init__()` and app layout.

Add `state.dirty: reactive.Value[bool]` (default `False`).

- Set `True` in `update_config()` (already the single mutation point)
- Reset on: config write/export, scenario save, example load
- Render amber dot (`.osm-dirty-indicator`) next to active nav pill
- Add JS `beforeunload` handler that checks `Shiny.shinyapp.$inputValues.dirty`
- Show inline banner at page top: "You have unsaved changes"

**Tests:** Update config → verify dirty=True; save scenario → verify dirty=False.

### Task 7 — Field-level validation feedback

**Files:** `ui/components/param_form.py`, `www/osmose.css`

For numeric fields in `render_field()`:
- Set `min`/`max` HTML attributes from `field.min_val`/`field.max_val` (browser-native enforcement)
- Add server-side `validate_field(key, value, registry)` → returns error string or `None`
- On input change, call validator and render error below field
- CSS: `.field-error` with red left border + red text, `.field-valid` with green checkmark

**Tests:** Render field with value outside bounds → verify error message; within bounds → verify no error.

### Task 8 — Responsive modals & mobile nav

**Files:** `www/osmose.css`, `ui/components/help_modal.py`

CSS-only changes at `@media (max-width: 768px)`:
- Nav pills: horizontal scrollable bar (`overflow-x: auto`, `flex-wrap: nowrap`)
- Modals: `width: 95vw`, `max-height: 80vh`, scrollable body
- Cards: reduced padding (`0.75rem`)
- Header: hide version badge, reduce title font

Add `<meta name="viewport" content="width=device-width, initial-scale=1">` to app head if missing.

**Tests:** None (CSS-only; manual verification).

---

## Section 3: New Capabilities

### Task 9 — CLI for batch runs

**New file:** `osmose/cli.py`
**Modify:** `pyproject.toml`

Commands via `argparse`:
```
osmose run <config> --jar <jar> --output <dir> --timeout 3600
osmose validate <config>
osmose ensemble <config> --replicates 10 --jar <jar>
osmose report <output_dir> --format html --output report.html
```

Each command is a thin wrapper:
- `run` → `asyncio.run(OsmoseRunner(...).run(...))`
- `validate` → `validate_config()` + `check_file_references()` + `check_species_consistency()`, print results, exit code 0/1
- `ensemble` → `asyncio.run(runner.run_ensemble(...))`
- `report` → `generate_report(OsmoseResults(output_dir), ...)`

Entry point in `pyproject.toml`:
```toml
[project.scripts]
osmose = "osmose.cli:main"
```

Note: `pyproject.toml` currently has no `[project.scripts]` section. The existing `include = ["osmose*"]` in `[tool.setuptools.packages.find]` already covers `osmose.cli`. The CLI `report` command should print the returned `Path` from `generate_report()` (see Task 10 return type change).

**Tests:** Call `main()` with mock args; verify correct runner/validator/reporter invocation.

**Dependency on:** Task 1 (timeout parameter), Task 2 (validate command uses same validation), Task 10 (report command uses Jinja2 reports).

### Task 10 — Jinja2 HTML reports

**Files:** `osmose/reporting.py`, new `osmose/templates/report.html`

Replace hardcoded HTML string with Jinja2 template:
```
osmose/templates/
  report.html      — main report template
```

Template receives context dict:
- `config`: flat config dict (HTML-escaped)
- `summary`: summary table as list of dicts
- `charts`: list of Plotly figure JSON strings (embedded via `plotly.io.to_html(fig, full_html=False)`)
- `metadata`: run timestamp, duration, output_dir, version

`generate_report()` signature change:
```python
def generate_report(
    results: OsmoseResults,
    config: dict[str, str],
    output_path: Path,
    fmt: str = "html",  # NOTE: preserves existing param name to avoid shadowing builtin format()
    template_path: Path | None = None,  # NEW: custom template override
) -> Path:  # NOTE: return type changes from None to Path (breaking change)
```

Use `jinja2.PackageLoader("osmose", "templates")` for bundled template. Fall back to `jinja2.FileSystemLoader` if `template_path` provided.

**Tests:** Generate report with sample data; verify HTML contains expected sections. Test custom template override.

### Task 11 — Run history & experiment tracking

**New file:** `osmose/history.py`
**Modify:** `osmose/runner.py`, `ui/pages/results.py`

After successful run in `OsmoseRunner.run()`, save JSON record:
```python
{
    "timestamp": "2026-03-11T14:30:00",
    "config_snapshot": {... flat config dict ...},
    "duration_sec": 42.5,
    "output_dir": "/path/to/output",
    "summary": {"mean_biomass": 1234.5, "total_yield": 567.8}
}
```

Storage: `{output_dir}/.osmose_history/{timestamp}.json`

`RunHistory` class:
- `list_runs(output_dir) -> list[RunRecord]` — sorted by timestamp
- `load_run(output_dir, timestamp) -> RunRecord` — single record
- `compare_runs(records) -> pd.DataFrame` — diff config + summary across runs

In Results page, add "History" tab:
- Table of past runs (timestamp, species count, duration, mean biomass)
- Click row → load that run's results into the existing charts
- Compare button → side-by-side summary table

**Tests:** Save 3 records, list, load, compare; verify sorting and diff output.

### Task 12 — New chart types

**Files:** `osmose/plotting.py` (or `plotting_extra.py`), `ui/pages/results.py`

Three new functions:

**`make_food_web(diet_matrix_df, template)`**
- Plotly Sankey diagram: predator → prey flows weighted by diet proportion
- Node colors from osmose palette; link opacity 0.4
- Filter links below threshold (default 1%) to reduce clutter

**`make_run_comparison(records: list[RunRecord], metrics: list[str], template)`**
- Grouped bar chart: x-axis = metric, groups = runs (by timestamp label)
- Uses `RunRecord.summary` from Task 11
- Dependency: Task 11

**`make_species_dashboard(results: OsmoseResults, species: list[str], template)`**
- Small multiples grid: one row per species
- Each row: biomass line, yield line, mortality stacked area (3 subplots sharing x-axis)
- Uses `plotly.subplots.make_subplots()`

Wire into Results page as new tabs: "Food Web", "Run Comparison", "Species Dashboard".

**Tests:** Generate each chart with sample data; verify trace counts and layout.

---

## Section 4: Developer Experience

### Task 13 — Shared test fixtures via conftest.py

**New file:** `tests/conftest.py`
**Modify:** Test files with duplicated fixtures, specifically:
- `test_analysis.py`, `test_plotting.py`, `test_reporting.py` — all create sample biomass/yield DataFrames
- `test_runner.py`, `test_calibration_problem.py`, `test_multiphase.py` — all create fake JAR scripts
- `test_registry.py`, `test_schema.py`, `test_schema_all.py` — all call `build_registry()`
- `test_results.py`, `test_ui_results.py` — both create sample output DataFrames
- `test_state.py`, `test_sync_config_pages.py`, `test_sync_setup.py` — all create sample config dicts

Shared fixtures:
- `sample_config` — Bay of Biscay subset (dict with ~20 keys covering species, simulation, grid)
- `registry` — `build_registry()` result (session-scoped for performance)
- `fake_jar` — Python script that writes dummy CSV output to stdout
- `sample_biomass_df` — DataFrame with species, time, biomass columns (3 species, 10 timesteps)
- `sample_diet_df` — DataFrame with predator, prey, proportion columns
- `tmp_output_dir` — tmp_path pre-populated with fake biomass.csv, yield.csv

Remove duplicated fixtures from individual test files. Keep test-specific fixtures local.

**Tests:** Run full suite; verify no regressions.

### Task 14 — pyright type checking in CI

**New files:** `pyrightconfig.json`
**Modify:** `pyproject.toml`, `.github/workflows/ci.yml`

`pyrightconfig.json`:
```json
{
    "typeCheckingMode": "basic",
    "include": ["osmose", "ui"],
    "exclude": ["tests"],
    "pythonVersion": "3.12"
}
```

Add `pyright>=1.1.350` as dev dependency (pin to avoid CI breakage from stricter future releases). Add CI job:
```yaml
type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e ".[dev]"
      - run: pyright
```

Fix initial type errors (estimated 10-20): likely `Path | str` coercion, unhandled `None` returns, dict access patterns.

**Tests:** CI passes with pyright; no new type errors introduced.

### Task 15 — Pre-commit hooks

**New file:** `.pre-commit-config.yaml`
**Modify:** `pyproject.toml` (add pre-commit as dev dep)

Config:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0  # Pin to latest stable at implementation time
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Keep it fast — ruff only (pyright is too slow for pre-commit). Document `pre-commit install` in README.

**Tests:** None (tooling config).

### Task 16 — CI hardening

**File:** `.github/workflows/ci.yml`, `Dockerfile`

Changes:
1. `pytest --cov-fail-under=95` — fail on coverage regression
2. Upload coverage HTML: `actions/upload-artifact@v4` with `htmlcov/`
3. Python matrix: `["3.12", "3.13"]`
4. Docker smoke test job: `docker build -t osmose-test .`
5. Add to Dockerfile:
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
     CMD curl -f http://localhost:8000/ || exit 1
   ```
   (requires `curl` in image — add to apt-get install)

**Tests:** CI pipeline succeeds on both Python versions.

---

## Section 5: Code Quality

### Task 17 — Split calibration page

**File:** `ui/pages/calibration.py` (545 LOC) → 3 files

Split into:
- `ui/pages/calibration.py` (~150 LOC) — `calibration_ui()` layout function + `calibration_server()` that delegates to handlers
- `ui/pages/calibration_handlers.py` (~200 LOC) — `register_calibration_handlers(input, output, session, state)`: start/stop NSGA-II, sensitivity analysis, GP surrogate, parameter collection
- `ui/pages/calibration_charts.py` (~150 LOC) — `make_convergence_chart()`, `make_pareto_chart()`, `make_sensitivity_chart()`

Public API unchanged: `from ui.pages.calibration import calibration_ui, calibration_server`.

**Tests:** Existing calibration tests pass unchanged.

### Task 18 — Extract Plotly theme

**Files:** `osmose/plotting.py` → extract to `osmose/plotly_theme.py`

Extract into new `osmose/plotly_theme.py` (~60 LOC):
- Create a named `OSMOSE_COLORS` constant from the inline colorway lists (currently defined inline in both `plotting.py`'s `_ensure_template()` and `ui/charts.py`'s template registration)
- Move `_ensure_template()` function and both template definitions ("osmose" dark, "osmose-light" light)
- **Important:** `ui/charts.py` has a more complete template (axis grid colors, tick fonts, border widths, title positioning) than `plotting.py`'s simpler version. Use the `ui/charts.py` version as the canonical one; drop `plotting.py`'s simpler duplicate.

`plotting.py` imports from `plotly_theme`. `ui/charts.py` also imports from `plotly_theme` instead of defining its own. Single source of truth for all Plotly styling.

New charts from Task 12 go directly into `plotting.py` since it's now lighter (~280 LOC without theme code + ~150 LOC new charts = ~430 LOC total — acceptable).

**Tests:** Import `plotly_theme`; verify templates registered. Existing plotting tests pass.

---

## Task Dependency Graph

```
Independent (can run in parallel):
  Tasks 1, 3, 4, 5, 6, 7, 8, 13, 15, 17, 18

Sequential chains:
  Task 1 → Task 9 (CLI uses timeout param)
  Task 2 → Task 9 (CLI validate uses same gate)
  Task 10 → Task 9 (CLI report uses Jinja2)
  Task 11 → Task 12 (run comparison chart needs history)
  Task 14 → Task 16 (CI includes pyright job)
  Task 18 → Task 12 (new charts use extracted theme)

Recommended execution order:
  Wave 1: Tasks 1, 3, 4, 5→6 (sequential, same files), 7, 8, 13, 15, 17, 18
  Wave 2: Tasks 2, 10, 11, 14
  Wave 3: Tasks 9, 12, 16
```

## Success Criteria

- All 400+ existing tests pass
- New tests added for Tasks 1-4, 5-7, 9-14, 17-18 (estimated +60-80 new tests)
- Coverage stays above 95%
- pyright passes with zero errors on `osmose/` and `ui/`
- CLI `osmose --help` works after `pip install -e .`
- Loading overlay visible during long operations
- Dirty indicator appears on config change
- Reports render from Jinja2 template
- Calibration page loads from split modules with no behavior change
