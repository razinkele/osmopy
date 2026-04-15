# Calibration UI Phase 2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Phase 1 calibration library features (banded loss, multi-seed validation, multi-objective sensitivity) in the Shiny UI, add a calibration history browser, and add parameter correlation visualization.

**Architecture:** Five UI features integrated into the existing calibration page. The results panel is restructured from a flat 4-tab layout to a two-level grouped navigation (Run / Results / History) using `navset_card_pill` + `navset_tab`. A new `history.py` library module provides JSON persistence. All chart functions follow the existing `tmpl` parameter pattern.

**Tech Stack:** Python 3.12+, Shiny for Python, Plotly Express, pandas, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-04-15-calibration-ui-phase2-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/calibration/history.py` | **create** | `save_run()`, `load_run()`, `list_runs()`, `delete_run()` |
| `tests/test_calibration_history.py` | **create** | Tests for history module |
| `ui/pages/calibration_charts.py` | **modify** | Add `make_correlation_chart()`, upgrade `make_sensitivity_chart()` for 2D |
| `ui/pages/calibration.py` | **modify** | Restructure tabs; add banded loss section; add validate controls; add history browser; add correlations tab |
| `ui/pages/calibration_handlers.py` | **modify** | Add banded loss objective builder; `_extract_species_stats()` helper; multi-seed validation handler; history save/load/delete handlers; sensitivity 2D Y support |
| `osmose/calibration/__init__.py` | **modify** | Export history module functions |

---

### Task 1: `history.py` — Calibration run persistence

**Files:**
- Create: `osmose/calibration/history.py`
- Create: `tests/test_calibration_history.py`

- [ ] **Step 1: Write tests for history module**

```python
# tests/test_calibration_history.py
"""Tests for calibration history persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osmose.calibration.history import delete_run, list_runs, load_run, save_run


@pytest.fixture()
def sample_run() -> dict:
    return {
        "version": 1,
        "timestamp": "2026-04-15T14:30:00",
        "algorithm": "nsga2",
        "settings": {"population_size": 50, "generations": 100, "n_parallel": 4},
        "parameters": [{"key": "species.k.sp0", "lower": 0.01, "upper": 1.0}],
        "objectives": {
            "biomass_rmse": True,
            "diet_distance": False,
            "banded_loss": {"enabled": False},
        },
        "results": {
            "best_objective": 0.342,
            "n_evaluations": 5000,
            "duration_seconds": 847,
            "objective_names": ["Biomass RMSE"],
            "convergence": [[0, 12.5], [1, 8.3], [2, 5.1]],
            "pareto_X": [[0.3, 100.0], [0.4, 120.0]],
            "pareto_F": [[0.34], [0.51]],
        },
    }


class TestSaveRun:
    def test_creates_json_file(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_filename_contains_timestamp_and_algorithm(
        self, tmp_path: Path, sample_run: dict
    ) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        assert "2026-04-15" in path.name
        assert "nsga2" in path.name

    def test_content_roundtrips(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        loaded = json.loads(path.read_text())
        assert loaded["version"] == 1
        assert loaded["algorithm"] == "nsga2"
        assert loaded["results"]["best_objective"] == 0.342

    def test_creates_directory_if_missing(self, tmp_path: Path, sample_run: dict) -> None:
        history_dir = tmp_path / "nested" / "history"
        path = save_run(sample_run, history_dir=history_dir)
        assert path.exists()
        assert history_dir.is_dir()


class TestLoadRun:
    def test_loads_saved_run(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        loaded = load_run(path)
        assert loaded["timestamp"] == "2026-04-15T14:30:00"
        assert loaded["results"]["pareto_X"] == [[0.3, 100.0], [0.4, 120.0]]

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run(tmp_path / "nonexistent.json")


class TestListRuns:
    def test_empty_directory(self, tmp_path: Path) -> None:
        assert list_runs(history_dir=tmp_path) == []

    def test_lists_runs_sorted_by_timestamp_desc(self, tmp_path: Path) -> None:
        run1 = {
            "version": 1,
            "timestamp": "2026-04-14T09:00:00",
            "algorithm": "surrogate",
            "settings": {},
            "parameters": [],
            "objectives": {},
            "results": {
                "best_objective": 0.5,
                "n_evaluations": 100,
                "duration_seconds": 60,
                "objective_names": [],
                "convergence": [],
                "pareto_X": [],
                "pareto_F": [],
            },
        }
        run2 = {
            "version": 1,
            "timestamp": "2026-04-15T14:30:00",
            "algorithm": "nsga2",
            "settings": {},
            "parameters": [{"key": "a", "lower": 0, "upper": 1}],
            "objectives": {},
            "results": {
                "best_objective": 0.3,
                "n_evaluations": 200,
                "duration_seconds": 120,
                "objective_names": [],
                "convergence": [],
                "pareto_X": [],
                "pareto_F": [],
            },
        }
        save_run(run1, history_dir=tmp_path)
        save_run(run2, history_dir=tmp_path)
        runs = list_runs(history_dir=tmp_path)
        assert len(runs) == 2
        assert runs[0]["timestamp"] == "2026-04-15T14:30:00"  # newest first
        assert runs[1]["timestamp"] == "2026-04-14T09:00:00"

    def test_list_entry_fields(self, tmp_path: Path, sample_run: dict) -> None:
        save_run(sample_run, history_dir=tmp_path)
        runs = list_runs(history_dir=tmp_path)
        entry = runs[0]
        assert "path" in entry
        assert entry["timestamp"] == "2026-04-15T14:30:00"
        assert entry["algorithm"] == "nsga2"
        assert entry["best_objective"] == 0.342
        assert entry["n_params"] == 1
        assert entry["duration_seconds"] == 847

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert list_runs(history_dir=tmp_path / "nonexistent") == []


class TestDeleteRun:
    def test_deletes_file(self, tmp_path: Path, sample_run: dict) -> None:
        path = save_run(sample_run, history_dir=tmp_path)
        assert path.exists()
        delete_run(path)
        assert not path.exists()

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            delete_run(tmp_path / "nonexistent.json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_history.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.history'`

- [ ] **Step 3: Implement history.py**

```python
# osmose/calibration/history.py
"""Calibration run persistence — save/load/list/delete JSON run files."""

from __future__ import annotations

import json
from pathlib import Path

HISTORY_DIR = Path("data/calibration_history")


def save_run(run_data: dict, history_dir: Path = HISTORY_DIR) -> Path:
    """Save a calibration run to JSON. Returns the path written."""
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = run_data["timestamp"].replace(":", "-")
    algo = run_data.get("algorithm", "unknown")
    filename = f"{ts}_{algo}.json"
    path = history_dir / filename
    path.write_text(json.dumps(run_data, indent=2))
    return path


def load_run(path: Path) -> dict:
    """Load a single run from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")
    return json.loads(path.read_text())


def list_runs(history_dir: Path = HISTORY_DIR) -> list[dict]:
    """List all runs, sorted by timestamp descending.

    Returns list of summary dicts with 'path', 'timestamp', 'algorithm',
    'best_objective', 'n_params', 'duration_seconds'.
    """
    if not history_dir.is_dir():
        return []
    entries = []
    for f in history_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entries.append(
            {
                "path": str(f),
                "timestamp": data.get("timestamp", ""),
                "algorithm": data.get("algorithm", "unknown"),
                "best_objective": data.get("results", {}).get("best_objective", float("inf")),
                "n_params": len(data.get("parameters", [])),
                "duration_seconds": data.get("results", {}).get("duration_seconds", 0),
            }
        )
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries


def delete_run(path: Path) -> None:
    """Delete a run file."""
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")
    path.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_history.py -v`
Expected: all PASS

- [ ] **Step 5: Run lint**

Run: `.venv/bin/ruff check osmose/calibration/history.py tests/test_calibration_history.py`
Expected: clean

- [ ] **Step 6: Commit**

```
git add osmose/calibration/history.py tests/test_calibration_history.py
git commit -m "feat(calibration): add history persistence module"
```

---

### Task 2: Upgrade `calibration_charts.py` — correlation chart + multi-objective sensitivity

**Files:**
- Modify: `ui/pages/calibration_charts.py`

- [ ] **Step 1: Add `make_correlation_chart()` and upgrade `make_sensitivity_chart()`**

Add to the end of `ui/pages/calibration_charts.py`:

```python
def make_correlation_chart(
    X: np.ndarray,
    F: np.ndarray,
    param_names: list[str],
    tmpl: str = "osmose",
) -> go.Figure:
    """Parallel coordinates plot of Pareto candidates."""
    import pandas as pd
    import plotly.express as px

    if X is None or len(X) == 0:
        return go.Figure().update_layout(
            title="Parameter Correlations (run calibration first)", template=tmpl
        )
    df = pd.DataFrame(X, columns=param_names)
    df["objective"] = F[:, 0] if F.shape[1] == 1 else np.sum(F, axis=1)
    fig = px.parallel_coordinates(
        df,
        color="objective",
        dimensions=param_names,
        color_continuous_scale="Viridis_r",
    )
    fig.update_layout(template=tmpl)
    return fig
```

Replace the existing `make_sensitivity_chart` function (lines 38-49) with:

```python
def make_sensitivity_chart(
    result: dict,
    tmpl: str = "osmose",
    selected_objective: int = 0,
) -> go.Figure:
    """Bar chart of Sobol sensitivity indices (1D or multi-objective)."""
    if "objective_names" in result:
        s1 = result["S1"][selected_objective]
        st = result["ST"][selected_objective]
        obj_name = result["objective_names"][selected_objective]
        title = f"Sobol Sensitivity — {obj_name}"
    else:
        s1 = result["S1"]
        st = result["ST"]
        title = "Sobol Sensitivity Indices"

    names = result["param_names"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="S1 (First-order)", x=names, y=s1))
    fig.add_trace(go.Bar(name="ST (Total-order)", x=names, y=st))
    fig.update_layout(title=title, barmode="group", template=tmpl)
    return fig
```

- [ ] **Step 2: Add import to calibration.py**

In `ui/pages/calibration.py`, update the import block (line 11-15) to include the new chart function:

```python
from ui.pages.calibration_charts import (
    make_convergence_chart,
    make_correlation_chart,
    make_pareto_chart,
    make_sensitivity_chart,
)
```

- [ ] **Step 3: Run lint**

Run: `.venv/bin/ruff check ui/pages/calibration_charts.py ui/pages/calibration.py`
Expected: clean

- [ ] **Step 4: Commit**

```
git add ui/pages/calibration_charts.py ui/pages/calibration.py
git commit -m "feat(calibration): add correlation chart, upgrade sensitivity for multi-objective"
```

---

### Task 3: Restructure `calibration.py` — two-level tab groups + new UI elements

**Files:**
- Modify: `ui/pages/calibration.py`

This is the largest UI task. It restructures the results panel and adds all new UI elements.

- [ ] **Step 1: Replace the `calibration_ui()` function**

Replace the entire `calibration_ui()` function (lines 45-120) in `ui/pages/calibration.py` with:

```python
def calibration_ui():
    return ui.div(
        expand_tab("Calibration Setup", "calibration"),
        ui.layout_columns(
            # Left: Configuration
            ui.card(
                collapsible_card_header("Calibration Setup", "calibration"),
                ui.input_select(
                    "cal_algorithm",
                    "Algorithm",
                    choices={
                        "nsga2": "NSGA-II (Direct)",
                        "surrogate": "GP Surrogate",
                    },
                ),
                ui.input_numeric("cal_pop_size", "Population size", value=50, min=10, max=500),
                ui.input_numeric(
                    "cal_generations", "Generations", value=100, min=10, max=1000
                ),
                ui.input_numeric(
                    "cal_n_parallel", "Parallel workers", value=4, min=1, max=32
                ),
                ui.hr(),
                ui.h5("Free Parameters"),
                ui.p("Select parameters to optimize:", style=STYLE_HINT_BLOCK),
                ui.output_ui("free_param_selector"),
                ui.hr(),
                ui.h5("Objectives"),
                ui.input_file(
                    "observed_biomass", "Upload observed biomass CSV", accept=[".csv"]
                ),
                ui.input_file(
                    "observed_diet", "Upload observed diet matrix CSV", accept=[".csv"]
                ),
                # Banded loss section
                ui.hr(),
                ui.input_checkbox("cal_banded_loss_enabled", "Enable Banded Loss Objective"),
                ui.panel_conditional(
                    "input.cal_banded_loss_enabled",
                    ui.input_select(
                        "cal_banded_source",
                        "Targets source",
                        choices={
                            "baltic": "Baltic defaults",
                            "upload": "Upload custom...",
                        },
                    ),
                    ui.panel_conditional(
                        "input.cal_banded_source === 'upload'",
                        ui.input_file(
                            "cal_banded_targets_file",
                            "Upload targets CSV",
                            accept=[".csv"],
                        ),
                    ),
                    ui.input_numeric(
                        "cal_w_stability", "Stability weight", value=5.0, min=0.0, max=100.0
                    ),
                    ui.input_numeric(
                        "cal_w_worst", "Worst-species weight", value=0.5, min=0.0, max=10.0
                    ),
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_start_cal", "Start Calibration", class_="btn-success w-100"
                    ),
                    ui.input_action_button(
                        "btn_stop_cal", "Stop", class_="btn-danger w-100"
                    ),
                    col_widths=[8, 4],
                ),
            ),
            # Right: Results — two-level grouped navigation
            ui.navset_card_pill(
                ui.nav_panel(
                    "Run",
                    ui.div(
                        ui.output_text("cal_status"),
                        output_widget("convergence_chart"),
                    ),
                ),
                ui.nav_panel(
                    "Results",
                    ui.navset_tab(
                        ui.nav_panel(
                            "Pareto Front",
                            ui.div(output_widget("pareto_chart")),
                        ),
                        ui.nav_panel(
                            "Correlations",
                            ui.div(output_widget("correlation_chart")),
                        ),
                        ui.nav_panel(
                            "Best Parameters",
                            ui.div(
                                ui.output_ui("best_params_table"),
                                ui.hr(),
                                ui.h6("Multi-Seed Validation"),
                                ui.layout_columns(
                                    ui.input_numeric(
                                        "cal_val_top_n", "Top N", value=5, min=1, max=50
                                    ),
                                    ui.input_numeric(
                                        "cal_val_seeds", "Seeds", value=5, min=2, max=20
                                    ),
                                    ui.input_action_button(
                                        "btn_validate",
                                        "Validate",
                                        class_="btn-info w-100",
                                    ),
                                    col_widths=[4, 4, 4],
                                ),
                                ui.output_ui("validation_table"),
                            ),
                        ),
                        ui.nav_panel(
                            "Sensitivity",
                            ui.div(
                                ui.input_action_button(
                                    "btn_sensitivity",
                                    "Run Sensitivity Analysis",
                                    class_="btn-info w-100",
                                ),
                                ui.output_ui("sensitivity_objective_selector"),
                                output_widget("sensitivity_chart"),
                            ),
                        ),
                        id="cal_results_tabs",
                    ),
                ),
                ui.nav_panel(
                    "History",
                    ui.div(
                        ui.output_ui("history_banner"),
                        ui.output_ui("history_list"),
                    ),
                ),
                id="cal_groups",
            ),
            col_widths=[4, 8],
        ),
        class_="osm-split-layout",
        id="split_calibration",
    )
```

- [ ] **Step 2: Update `calibration_server()` — add new reactive values and renderers**

Replace the entire `calibration_server()` function (lines 123-225) with:

```python
def calibration_server(input, output, session, state):
    cal_history = reactive.value([])
    cal_F = reactive.value(None)
    cal_X = reactive.value(None)
    sensitivity_result = reactive.value(None)
    cal_thread = reactive.value(None)
    surrogate_status = reactive.value("")
    validation_result = reactive.value(None)
    cal_param_names = reactive.value([])
    history_banner_text = reactive.value("")

    def _tmpl() -> str:
        mode = get_theme_mode(input)
        return "osmose" if mode == "dark" else "osmose-light"

    # Register event handlers
    register_calibration_handlers(
        input=input,
        output=output,
        session=session,
        state=state,
        cal_history=cal_history,
        cal_F=cal_F,
        cal_X=cal_X,
        sensitivity_result=sensitivity_result,
        cal_thread=cal_thread,
        surrogate_status=surrogate_status,
        copy_data_files=copy_data_files,
        validation_result=validation_result,
        cal_param_names=cal_param_names,
        history_banner_text=history_banner_text,
    )

    @render.text
    def cal_status():
        surr = surrogate_status.get()
        if surr:
            return surr
        hist = cal_history.get()
        if not hist:
            return "Ready. Configure parameters and objectives, then click Start."
        return f"Generation {len(hist)} — Best: {hist[-1]:.4f}"

    @render.ui
    def free_param_selector():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        n_str = cfg.get("simulation.nspecies", "3")
        try:
            n_species = int(float(n_str or "3"))
        except (ValueError, TypeError):
            n_species = 3
        params = get_calibratable_params(state.registry, n_species)
        checkboxes = [
            ui.input_checkbox(
                f"cal_param_{p['key'].replace('.', '_')}",
                p["label"],
                value=False,
            )
            for p in params
        ]
        return ui.div(*checkboxes)

    @render_plotly
    def convergence_chart():
        return make_convergence_chart(cal_history.get(), tmpl=_tmpl())

    @render_plotly
    def pareto_chart():
        F = cal_F.get()
        if F is None:
            return go.Figure().update_layout(
                title="Pareto Front (run calibration first)", template=_tmpl()
            )
        obj_labels = [f"Obj {i + 1}" for i in range(F.shape[1])]
        return make_pareto_chart(F, obj_labels, tmpl=_tmpl())

    @render_plotly
    def correlation_chart():
        X = cal_X.get()
        F = cal_F.get()
        names = cal_param_names.get()
        if X is None or F is None or not names:
            return go.Figure().update_layout(
                title="Parameter Correlations (run calibration first)", template=_tmpl()
            )
        return make_correlation_chart(X, F, names, tmpl=_tmpl())

    @render.ui
    def best_params_table():
        X = cal_X.get()
        F = cal_F.get()
        if X is None or F is None:
            return ui.div("Run calibration to see best parameters.", style=STYLE_EMPTY)
        order = np.argsort(F.sum(axis=1))[:10]
        names = cal_param_names.get()
        if not names:
            with reactive.isolate():
                cfg = state.config.get()
            selected = collect_selected_params(input, state, config=cfg)
            names = [p["key"].split(".")[-1] for p in selected]
        headers = [ui.tags.th(n) for n in names]
        headers.append(ui.tags.th("Total Obj"))
        rows = []
        for idx in order:
            cells = [ui.tags.td(f"{v:.4f}") for v in X[idx]]
            cells.append(ui.tags.td(f"{F[idx].sum():.4f}"))
            rows.append(ui.tags.tr(*cells))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render.ui
    def validation_table():
        result = validation_result.get()
        if result is None:
            return ui.div()
        rankings = result["rankings"]
        scores = result["scores"]
        headers = [
            ui.tags.th("Rank"),
            ui.tags.th("Mean"),
            ui.tags.th("Std"),
            ui.tags.th("CV"),
            ui.tags.th("Worst"),
        ]
        rows = []
        for i, idx in enumerate(rankings):
            s = scores[idx]
            rows.append(
                ui.tags.tr(
                    ui.tags.td(str(i + 1)),
                    ui.tags.td(f"{s['mean']:.4f}"),
                    ui.tags.td(f"{s['std']:.4f}"),
                    ui.tags.td(f"{s['cv']:.1%}"),
                    ui.tags.td(f"{s['worst_value']:.4f}"),
                )
            )
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render.ui
    def sensitivity_objective_selector():
        result = sensitivity_result.get()
        if result is None or "objective_names" not in result:
            return ui.div()
        choices = {
            str(i): name for i, name in enumerate(result["objective_names"])
        }
        return ui.input_select(
            "cal_sens_objective", "Objective", choices=choices
        )

    @render_plotly
    def sensitivity_chart():
        result = sensitivity_result.get()
        if result is None:
            return go.Figure().update_layout(
                title="Sensitivity (click Run)", template=_tmpl()
            )
        selected = 0
        if "objective_names" in result:
            try:
                selected = int(input.cal_sens_objective())
            except (AttributeError, ValueError, TypeError):
                selected = 0
        return make_sensitivity_chart(result, tmpl=_tmpl(), selected_objective=selected)

    @render.ui
    def history_banner():
        text = history_banner_text.get()
        if not text:
            return ui.div()
        return ui.div(
            ui.tags.em(text),
            class_="alert alert-info py-1 px-2 mb-2",
        )

    @render.ui
    def history_list():
        from osmose.calibration.history import list_runs

        runs = list_runs()
        if not runs:
            return ui.div("No calibration history yet.", style=STYLE_EMPTY)
        cards = []
        for i, run in enumerate(runs):
            ts = run["timestamp"]
            algo = run["algorithm"].upper()
            best = run["best_objective"]
            n_p = run["n_params"]
            dur = run["duration_seconds"]
            dur_str = f"{dur // 60}m {dur % 60}s" if dur >= 60 else f"{dur}s"
            cards.append(
                ui.div(
                    ui.div(
                        ui.tags.strong(f"{ts} — {algo}"),
                        ui.span(
                            f" Best: {best:.3f} | {n_p} params | {dur_str}",
                            style="margin-left: 8px; opacity: 0.8;",
                        ),
                        ui.div(
                            ui.input_action_button(
                                f"btn_load_run_{i}",
                                "Load",
                                class_="btn-sm btn-outline-primary me-1",
                            ),
                            ui.input_action_button(
                                f"btn_delete_run_{i}",
                                "Delete",
                                class_="btn-sm btn-outline-danger",
                            ),
                            style="display: inline-block; float: right;",
                        ),
                        style="padding: 8px;",
                    ),
                    class_="border rounded mb-2",
                )
            )
        return ui.div(
            ui.h6(f"Calibration History ({len(runs)} runs)"),
            *cards,
        )
```

- [ ] **Step 3: Update `__all__` export list**

Update the `__all__` list at the top of `calibration.py` (lines 29-39) to include new imports:

```python
from ui.pages.calibration_charts import (
    make_convergence_chart,
    make_correlation_chart,
    make_pareto_chart,
    make_sensitivity_chart,
)

__all__ = [
    "calibration_ui",
    "calibration_server",
    "get_calibratable_params",
    "collect_selected_params",
    "build_free_params",
    "make_convergence_chart",
    "make_correlation_chart",
    "make_pareto_chart",
    "make_sensitivity_chart",
    "_make_progress_callback",
]
```

- [ ] **Step 4: Run lint**

Run: `.venv/bin/ruff check ui/pages/calibration.py`
Expected: clean

- [ ] **Step 5: Commit**

```
git add ui/pages/calibration.py
git commit -m "feat(calibration): restructure tabs, add banded loss/validation/history/correlation UI"
```

---

### Task 4: Update `calibration_handlers.py` — banded loss, validation, history, sensitivity 2D

**Files:**
- Modify: `ui/pages/calibration_handlers.py`

This is the most complex handler task. It modifies `register_calibration_handlers` to accept the 3 new reactive values and adds 4 new handler functions.

- [ ] **Step 1: Update `register_calibration_handlers` signature**

Change the function signature (line 144) to accept new reactive values:

```python
def register_calibration_handlers(
    input,
    output,
    session,
    state,
    cal_history,
    cal_F,
    cal_X,
    sensitivity_result,
    cal_thread,
    surrogate_status,
    copy_data_files,
    validation_result,
    cal_param_names,
    history_banner_text,
):
```

- [ ] **Step 2: Add `_extract_species_stats` helper**

Add this helper function after the `build_free_params` function (after line 141), before `register_calibration_handlers`:

```python
def _extract_species_stats(results, species_names: list[str], n_eval_years: int = 10) -> dict:
    """Extract mean/cv/trend per species from simulation results.

    Returns dict with keys '{species}_mean', '{species}_cv', '{species}_trend'
    for each species. Same logic as calibrate_baltic.py run_simulation().
    """
    bio = results.biomass()
    total_years = len(bio)
    eval_data = bio.iloc[-n_eval_years:] if total_years > n_eval_years else bio

    stats: dict[str, float] = {}
    for sp in species_names:
        if sp not in eval_data.columns:
            continue
        vals = eval_data[sp].values.astype(float)
        mean_val = float(np.mean(vals))
        stats[f"{sp}_mean"] = mean_val

        if mean_val > 0:
            stats[f"{sp}_cv"] = float(np.std(vals) / mean_val)
        else:
            stats[f"{sp}_cv"] = 10.0

        if len(vals) >= 3:
            x = np.arange(len(vals), dtype=float)
            slope = np.polyfit(x, vals, 1)[0]
            stats[f"{sp}_trend"] = float(abs(slope) / (mean_val + 1.0))
        else:
            stats[f"{sp}_trend"] = 0.0

    return stats
```

- [ ] **Step 3: Add banded loss objective building in `handle_start_cal`**

Inside `handle_start_cal()`, after the existing objective building block (after line 232, where `objective_fns` list is built), add:

```python
        # Banded loss objective
        obj_names = []
        if obs_bio:
            obj_names.append("Biomass RMSE")
        if obs_diet:
            obj_names.append("Diet Distance")

        banded_enabled = False
        try:
            banded_enabled = bool(input.cal_banded_loss_enabled())
        except (SilentException, AttributeError):
            pass

        if banded_enabled:
            from osmose.calibration.losses import make_banded_objective
            from osmose.calibration.targets import load_targets

            banded_source = input.cal_banded_source()
            if banded_source == "baltic":
                targets_path = Path("data/baltic/reference/biomass_targets.csv")
            else:
                banded_file = input.cal_banded_targets_file()
                if not banded_file:
                    ui.notification_show(
                        "Upload a targets CSV or select Baltic defaults.",
                        type="warning",
                        duration=5,
                    )
                    return
                targets_path = Path(banded_file[0]["datapath"])

            targets, _ = load_targets(targets_path)
            species_names = [t.species for t in targets]
            w_stability = float(input.cal_w_stability())
            w_worst = float(input.cal_w_worst())
            banded_obj = make_banded_objective(
                targets, species_names, w_stability=w_stability, w_worst=w_worst
            )

            def banded_objective_fn(results, _banded=banded_obj, _sp=species_names):
                stats = _extract_species_stats(results, _sp)
                return _banded(stats)

            objective_fns.append(banded_objective_fn)
            obj_names.append("Banded Loss")

        if not objective_fns:
            ui.notification_show(
                "Enable at least one objective (upload data or enable banded loss).",
                type="warning",
                duration=5,
            )
            return

        # Store param names for charts
        cal_param_names.set([p["key"].split(".")[-1] for p in selected])
```

Also modify the existing objective validation block. Replace the lines that check `if not obs_bio and not obs_diet` (lines 200-203) with:

```python
        obs_bio = input.observed_biomass()
        obs_diet = input.observed_diet()
        # Don't require uploads if banded loss is enabled
        banded_check = False
        try:
            banded_check = bool(input.cal_banded_loss_enabled())
        except (SilentException, AttributeError):
            pass
        if not obs_bio and not obs_diet and not banded_check:
            ui.notification_show(
                "Upload observed data or enable banded loss objective.",
                type="warning",
                duration=5,
            )
            return
```

- [ ] **Step 4: Add history auto-save after calibration completes**

In both the `run_surrogate` and `run_optimization` thread functions, add history saving
after results are posted. In `run_surrogate`, after `msg_queue.post_results(...)` (around line 307):

```python
                    # Auto-save to history
                    import time as _time

                    from osmose.calibration.history import save_run

                    save_run(
                        {
                            "version": 1,
                            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "algorithm": "surrogate",
                            "settings": {
                                "population_size": pop_size,
                                "generations": 0,
                                "n_parallel": n_parallel,
                            },
                            "parameters": [
                                {"key": fp.key, "lower": fp.lower_bound, "upper": fp.upper_bound}
                                for fp in free_params
                            ],
                            "objectives": {
                                "biomass_rmse": bool(obs_bio),
                                "diet_distance": bool(obs_diet),
                                "banded_loss": {"enabled": banded_enabled},
                            },
                            "results": {
                                "best_objective": float(np.min(Y.sum(axis=1))),
                                "n_evaluations": n_samples,
                                "duration_seconds": 0,
                                "objective_names": obj_names,
                                "convergence": history,
                                "pareto_X": samples.tolist(),
                                "pareto_F": Y.tolist(),
                            },
                        }
                    )
```

Similarly in `run_optimization`, after `msg_queue.post_results(...)`.

Note: the NSGA-II thread cannot safely call `cal_history.get()` (reactive value, main thread only). Instead, reconstruct convergence from the pymoo result object. The callback tracks best objective per generation, but the result `res` contains the final Pareto front. We store an empty convergence list — the convergence chart is populated from the reactive value on the main thread, and history is primarily for loading Pareto data:

```python
                    if res.F is not None:
                        msg_queue.post_results(X=res.X, F=res.F)
                        # Auto-save to history
                        import time as _time

                        from osmose.calibration.history import save_run

                        save_run(
                            {
                                "version": 1,
                                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "algorithm": "nsga2",
                                "settings": {
                                    "population_size": pop_size,
                                    "generations": generations,
                                    "n_parallel": n_parallel,
                                },
                                "parameters": [
                                    {
                                        "key": fp.key,
                                        "lower": fp.lower_bound,
                                        "upper": fp.upper_bound,
                                    }
                                    for fp in free_params
                                ],
                                "objectives": {
                                    "biomass_rmse": bool(obs_bio),
                                    "diet_distance": bool(obs_diet),
                                    "banded_loss": {"enabled": banded_enabled},
                                },
                                "results": {
                                    "best_objective": float(np.min(res.F.sum(axis=1))),
                                    "n_evaluations": pop_size * generations,
                                    "duration_seconds": 0,
                                    "objective_names": obj_names,
                                    "convergence": [],
                                    "pareto_X": res.X.tolist(),
                                    "pareto_F": res.F.tolist(),
                                },
                            }
                        )
```

- [ ] **Step 5: Add validation handler**

Add after `handle_stop_cal` (after the cancel event handler):

```python
    @reactive.effect
    @reactive.event(input.btn_validate)
    def handle_validate():
        X = cal_X.get()
        F = cal_F.get()
        if X is None or F is None:
            ui.notification_show("Run calibration first.", type="warning", duration=5)
            return

        top_n = int(input.cal_val_top_n())
        n_seeds = int(input.cal_val_seeds())
        seeds = list(range(n_seeds))

        # Get top N candidates by total objective
        order = np.argsort(F.sum(axis=1))[:top_n]
        candidates = X[order]

        selected = collect_selected_params(input, state)
        param_keys = [p["key"] for p in selected]

        jar_path = Path(state.jar_path.get())
        with reactive.isolate():
            current_config = state.config.get()
            source_dir = state.config_dir.get()
            case_map = state.key_case_map.get()

        from osmose.calibration.problem import OsmoseCalibrationProblem
        from osmose.config.writer import OsmoseConfigWriter

        work_dir = Path(tempfile.mkdtemp(prefix="osmose_val_"))
        writer = OsmoseConfigWriter()
        config_dir = work_dir / "config"
        writer.write(current_config, config_dir, key_case_map=case_map)
        if source_dir and source_dir.is_dir():
            copy_data_files(current_config, source_dir, config_dir)
        base_config = config_dir / "osm_all-parameters.csv"

        # Minimal problem for running simulations (objectives computed manually)
        free_params = build_free_params(selected)
        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=[lambda r: 0.0],  # dummy — we compute scalar ourselves
            base_config_path=base_config,
            jar_path=jar_path,
            work_dir=work_dir,
        )

        validation_result.set(None)
        surrogate_status.set("Validating candidates...")

        def run_validation():
            try:
                from osmose.calibration.multiseed import rank_candidates_multiseed

                _counter = [0]

                def make_factory(seed):
                    def objective(x):
                        overrides = {
                            param_keys[j]: str(float(x[j]))
                            for j in range(len(param_keys))
                        }
                        overrides["simulation.random.seed"] = str(seed)
                        run_id = _counter[0]
                        _counter[0] += 1
                        obj_values = problem._run_single(overrides, run_id=run_id)
                        problem.cleanup_run(run_id)
                        return sum(obj_values)

                    return objective

                result = rank_candidates_multiseed(make_factory, candidates, seeds=seeds)
                msg_queue.post_status("")
                # Post validation result through queue
                msg_queue._q.put(("validation", result))
            except Exception as exc:
                _log.error("Validation failed: %s", exc, exc_info=True)
                msg_queue.post_error(f"Validation: {exc}")

        thread = threading.Thread(target=run_validation, daemon=True)
        thread.start()
```

- [ ] **Step 6: Add validation message handling in `_poll_cal_messages`**

In the `_poll_cal_messages` function, add a handler for the "validation" message type
(after the "sensitivity" handler around line 179):

```python
            elif kind == "validation":
                validation_result.set(payload)
```

- [ ] **Step 7: Add history load/delete handlers**

Add after the validation handler:

```python
    # History load/delete handlers — dynamic button IDs
    @reactive.effect
    def _handle_history_buttons():
        """Watch for dynamically created history load/delete buttons."""
        from osmose.calibration.history import delete_run, list_runs, load_run

        runs = list_runs()
        for i, run in enumerate(runs):
            load_id = f"btn_load_run_{i}"
            delete_id = f"btn_delete_run_{i}"
            try:
                if getattr(input, load_id)():
                    data = load_run(Path(run["path"]))
                    cal_X.set(np.array(data["results"]["pareto_X"]))
                    cal_F.set(np.array(data["results"]["pareto_F"]))
                    cal_param_names.set(
                        [p["key"].split(".")[-1] for p in data["parameters"]]
                    )
                    conv = data["results"].get("convergence", [])
                    cal_history.set([v for _, v in conv])
                    history_banner_text.set(
                        f"Viewing historical run from {data['timestamp']}"
                    )
                    validation_result.set(None)
            except (SilentException, AttributeError):
                pass
            try:
                if getattr(input, delete_id)():
                    delete_run(Path(run["path"]))
                    ui.notification_show("Run deleted.", type="message", duration=3)
            except (SilentException, AttributeError):
                pass
```

- [ ] **Step 8: Update sensitivity handler for multi-objective Y**

In `handle_sensitivity` (around line 410), modify the `run_sensitivity` function.
Replace the single-objective `Y = np.zeros(samples.shape[0])` block with:

```python
                # Build objective functions matching calibration config
                objective_fns_sens = []
                obj_names_sens = []
                if obs_bio:
                    objective_fns_sens.append(
                        lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df)
                    )
                    obj_names_sens.append("Biomass RMSE")

                n_obj_sens = max(1, len(objective_fns_sens))
                Y = np.zeros((samples.shape[0], n_obj_sens)) if n_obj_sens > 1 else np.zeros(
                    samples.shape[0]
                )

                prob = OsmoseCalibrationProblem(
                    free_params=build_free_params(selected),
                    objective_fns=objective_fns_sens or [lambda r: 0.0],
                    base_config_path=base_config,
                    jar_path=jar_path,
                    work_dir=sens_work_dir,
                )
```

Then update the Y assignment in the sample loop to handle multi-objective:

```python
                    try:
                        result = prob._run_single(overrides, run_id=idx)
                        if n_obj_sens > 1:
                            for k in range(n_obj_sens):
                                Y[idx, k] = result[k]
                        else:
                            Y[idx] = result[0]
                    except (
                        subprocess.TimeoutExpired,
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                        OSError,
                    ) as exc:
                        _log.warning("Sensitivity sample %d failed: %s", idx, exc)
                        if n_obj_sens > 1:
                            Y[idx, :] = float("inf")
                        else:
                            Y[idx] = float("inf")
```

- [ ] **Step 9: Run lint**

Run: `.venv/bin/ruff check ui/pages/calibration_handlers.py`
Expected: clean (fix any issues)

- [ ] **Step 10: Commit**

```
git add ui/pages/calibration_handlers.py
git commit -m "feat(calibration): add banded loss, validation, history, and 2D sensitivity handlers"
```

---

### Task 5: Update `__init__.py` — export history module

**Files:**
- Modify: `osmose/calibration/__init__.py`

- [ ] **Step 1: Add history exports**

Add to `osmose/calibration/__init__.py`:

```python
from osmose.calibration.history import save_run, load_run, list_runs, delete_run
```

And add to `__all__`:

```python
    "save_run",
    "load_run",
    "list_runs",
    "delete_run",
```

- [ ] **Step 2: Verify imports**

Run: `.venv/bin/python -c "from osmose.calibration import save_run, load_run, list_runs, delete_run; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```
git add osmose/calibration/__init__.py
git commit -m "feat(calibration): export history module from __init__.py"
```

---

### Task 6: Full regression test + lint

**Files:** None — validation only.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: all tests pass, no regressions

- [ ] **Step 2: Run lint on all modified files**

Run: `.venv/bin/ruff check osmose/calibration/ ui/pages/calibration*.py`
Expected: no errors

- [ ] **Step 3: Run format check**

Run: `.venv/bin/ruff format --check osmose/calibration/ ui/pages/calibration*.py`
Expected: clean (if not, format and commit)

- [ ] **Step 4: Verify all exports**

Run: `.venv/bin/python -c "import osmose.calibration; print(sorted(osmose.calibration.__all__)); print(f'{len(osmose.calibration.__all__)} exports')"`
Expected: 25 exports (21 from Phase 1 + 4 history functions)

- [ ] **Step 5: Commit any formatting fixes**

```
git add -u
git commit -m "style: format calibration UI Phase 2 files"
```
(Skip if no changes needed.)
