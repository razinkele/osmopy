# Compare Runs Output-Delta Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an output-delta section to the existing Shiny "Compare Runs" tab — pick exactly two runs and see the per-species ranked delta table + diverging-bar chart inline, reusing the shipped `run_delta`/`make_run_delta_chart`/`format_delta_report`.

**Architecture:** All changes are inside `ui/pages/results.py` (`results_ui()` + `results_server()`). A pure, unit-tested helper `_delta_for_selected(records, metric, window_years)` reconstructs `OsmoseResults` from two `RunRecord`s and calls `run_delta`. Two thin render functions (`@render_plotly run_delta_chart`, `@render.ui run_delta_table`) follow the existing `comparison_chart`/`config_diff_table` scaffolds. No `app.py` change, no engine changes, no new analysis/plotting code.

**Tech Stack:** Python 3.12, Shiny for Python, shinywidgets (`render_plotly`/`output_widget`), Plotly, pandas, pytest, ruff. Tests: `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-06-03-compare-runs-delta-ui-design.md`.

---

## Verified facts (from reconnaissance — use exactly)

- Compare Runs `nav_panel` is at `ui/pages/results.py:264-288` (line numbers approximate — anchor on STRUCTURE, not numbers): the controls `ui.div` (holding `ui.input_selectize("compare_runs_select", multiple=True, choices={})` + `ui.input_select("compare_metric", choices={"biomass","yield","abundance"})`) is wrapped inside a `ui.layout_columns(..., col_widths=[12])`; the `output_widget("comparison_chart")` + `ui.output_ui("config_diff_table")` are SIBLINGS of that `layout_columns` (not inside the controls div). Insert the slider inside the controls `ui.div` (after `compare_metric`); insert the 2 new outputs as siblings after `config_diff_table`.
- Existing render scaffolds: `@render_plotly def comparison_chart()` (599-621) and `@render.ui def config_diff_table()` (623-661). They use `_tpl(input)` (template), `_safe_output_dir(input.output_dir())`, `history_dir = out_dir.parent / ".osmose_history"`, `RunHistory(history_dir)`, `history.load_run(ts)` per selected timestamp, `STYLE_EMPTY` for empty prompts.
- `selected = input.compare_runs_select()` is a tuple of run timestamps (strings).
- `RunRecord` (osmose/history.py) has `.output_dir` (str), no `prefix` → `OsmoseResults(Path(rec.output_dir), strict=False)` (default prefix "osm").
- Shipped lib: `from osmose.analysis import run_delta, format_delta_report`; `from osmose.plotting import make_run_delta_chart`. `run_delta(baseline, variant, *, metric, window_years, top_n=None) -> list[SpeciesDelta]`.
- `tests/test_ui_results.py` exists; it unit-tests PURE helpers from results.py (e.g. `make_timeseries_chart`). Render decorators are not unit-tested (project convention).

## File Structure

- Modify: `ui/pages/results.py` — `_delta_for_selected` helper (module-level), the UI additions in `results_ui()`, the two render fns in `results_server()`.
- Modify: `tests/test_ui_results.py` — unit tests for `_delta_for_selected` + a page-imports/builds smoke.

---

## Task 1: `_delta_for_selected` pure helper

**Files:**
- Modify: `ui/pages/results.py`
- Test: `tests/test_ui_results.py`

- [ ] **Step 1: Write failing tests (monkeypatch OsmoseResults — no real files needed)**

Add to `tests/test_ui_results.py` (it already imports from `ui.pages.results`; match its import style — confirm with `grep -n "^from\|^import\|import results\|from ui" tests/test_ui_results.py`):

```python
import types
import pandas as pd
import pytest
from ui.pages import results as rp


class _FakeResults:
    """Mimics OsmoseResults' wide biomass()/yield_biomass()/abundance() accessors."""
    def __init__(self, frame):
        self._f = frame
    def biomass(self, species=None):
        return self._f
    def yield_biomass(self, species=None):
        return self._f
    def abundance(self, species=None):
        return self._f


def _wide(**species_to_series):
    n = len(next(iter(species_to_series.values())))
    d = {"Time": list(range(1, n + 1))}
    d.update(species_to_series)
    d["species"] = ["all"] * n
    return pd.DataFrame(d)


def _patch_osmose_results(monkeypatch, frames_by_dir):
    """Patch osmose.results.OsmoseResults so each output_dir maps to a fixed wide frame."""
    def _factory(output_dir, prefix="osm", strict=True):
        return _FakeResults(frames_by_dir[str(output_dir)])
    monkeypatch.setattr("osmose.results.OsmoseResults", _factory)


def test_delta_for_selected_ranks_and_signs(monkeypatch):
    frames = {
        "base": _wide(cod=[100.0, 100.0], herring=[50.0, 50.0]),
        "var": _wide(cod=[110.0, 110.0], herring=[100.0, 100.0]),
    }
    _patch_osmose_results(monkeypatch, frames)
    recs = [types.SimpleNamespace(output_dir="base"), types.SimpleNamespace(output_dir="var")]
    deltas = rp._delta_for_selected(recs, metric="biomass", window_years=2)
    by = {d.species: d for d in deltas}
    assert by["cod"].pct_delta == pytest.approx(0.10)
    assert by["herring"].pct_delta == pytest.approx(1.0)
    assert [d.species for d in deltas][0] == "herring"   # biggest |Δ%| first

def test_delta_for_selected_swap_flips_sign(monkeypatch):
    frames = {
        "base": _wide(cod=[100.0, 100.0]),
        "var": _wide(cod=[110.0, 110.0]),
    }
    _patch_osmose_results(monkeypatch, frames)
    recs = [types.SimpleNamespace(output_dir="base"), types.SimpleNamespace(output_dir="var")]
    fwd = rp._delta_for_selected(recs, metric="biomass", window_years=2)[0]
    rev = rp._delta_for_selected(list(reversed(recs)), metric="biomass", window_years=2)[0]
    assert fwd.pct_delta == pytest.approx(0.10)     # base→var: +10%
    assert rev.pct_delta == pytest.approx(-100 / 110)  # var→base: -9.09%

def test_delta_for_selected_requires_two(monkeypatch):
    recs = [types.SimpleNamespace(output_dir="only")]
    with pytest.raises(ValueError):
        rp._delta_for_selected(recs, metric="biomass", window_years=2)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k delta_for_selected -v`
Expected: FAIL (`module 'ui.pages.results' has no attribute '_delta_for_selected'`).

- [ ] **Step 3: Implement the helper**

In `ui/pages/results.py`, add at module level (near the other top-level helpers like `make_timeseries_chart`/`_tpl`; put the imports inside the function to match the page's lazy-import convention seen in `comparison_chart`):

```python
def _delta_for_selected(records, metric: str, window_years: int):
    """Per-species output delta between exactly two runs (baseline=records[0], variant=records[1]).

    Reconstructs each run as OsmoseResults(rec.output_dir) and returns run_delta's
    list[SpeciesDelta]. Raises ValueError unless exactly 2 records are given.
    """
    if len(records) != 2:
        raise ValueError("need exactly 2 runs for a pairwise output delta")
    from pathlib import Path

    from osmose.results import OsmoseResults
    from osmose.analysis import run_delta

    baseline = OsmoseResults(Path(records[0].output_dir), strict=False)
    variant = OsmoseResults(Path(records[1].output_dir), strict=False)
    return run_delta(baseline, variant, metric=metric, window_years=window_years)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k delta_for_selected -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add ui/pages/results.py tests/test_ui_results.py
git commit -m "feat(ui): _delta_for_selected helper for Compare Runs output delta"
```

---

## Task 2: UI controls + outputs + render functions

**Files:**
- Modify: `ui/pages/results.py`
- Test: `tests/test_ui_results.py`

- [ ] **Step 1: Add the window-years slider to the Compare Runs controls**

In `results_ui()`, in the controls `ui.div` (after the `compare_metric` `ui.input_select`, which ends at line ~282 — insert as a sibling before the `ui.div` closes at line 283):

```python
                        ui.input_slider(
                            "compare_window_years", "Window (years)", min=1, max=30, value=10
                        ),
```

- [ ] **Step 2: Add the two output placeholders**

After `ui.output_ui("config_diff_table")` (line ~287), add inside the same `nav_panel`:

```python
                output_widget("run_delta_chart"),
                ui.output_ui("run_delta_table"),
```

- [ ] **Step 3: Add the two render functions**

In `results_server()`, after the `config_diff_table` render fn closes (line ~661), add:

```python
    @render_plotly
    def run_delta_chart():
        tmpl = _tpl(input)
        selected = input.compare_runs_select()
        if not selected or len(selected) != 2:
            return go.Figure().update_layout(
                title="Select exactly 2 runs (1st = baseline, 2nd = variant)", template=tmpl
            )
        from osmose.history import RunHistory
        from osmose.plotting import make_run_delta_chart

        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return go.Figure().update_layout(title="Invalid output directory", template=tmpl)
        history_dir = out_dir.parent / ".osmose_history"
        if not history_dir.is_dir():
            return go.Figure().update_layout(title="No run history found", template=tmpl)

        metric = input.compare_metric()
        try:
            records = [RunHistory(history_dir).load_run(ts) for ts in selected]
            deltas = _delta_for_selected(records, metric, int(input.compare_window_years()))
        except Exception as e:  # noqa: BLE001 — UI guard: degrade to an error title, never crash the page
            return go.Figure().update_layout(title=f"Could not compute delta: {e}", template=tmpl)
        fig = make_run_delta_chart(deltas, metric=metric)
        fig.update_layout(template=tmpl)
        return fig

    @render.ui
    def run_delta_table():
        selected = input.compare_runs_select()
        if not selected or len(selected) != 2:
            return ui.div(
                "Select exactly 2 runs to see the per-species output delta "
                "(1st = baseline, 2nd = variant). The config diff above supports more than 2.",
                style=STYLE_EMPTY,
            )
        from osmose.history import RunHistory
        from osmose.analysis import format_delta_report

        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return ui.div("Invalid output directory.")
        history_dir = out_dir.parent / ".osmose_history"
        if not history_dir.is_dir():
            return ui.div("No run history found.")

        metric = input.compare_metric()
        try:
            records = [RunHistory(history_dir).load_run(ts) for ts in selected]
            window_years = int(input.compare_window_years())
            deltas = _delta_for_selected(records, metric, window_years)
        except Exception as e:  # noqa: BLE001 — UI guard: degrade to an error div, never crash the page
            return ui.div(f"Could not load run outputs: {e}")
        return ui.markdown(format_delta_report(deltas, metric=metric, window_years=window_years))
```

- [ ] **Step 4: Page-builds smoke test (catches wiring/syntax errors)**

Add to `tests/test_ui_results.py`:

```python
def test_results_ui_builds():
    # results_ui() must construct without error after the Compare Runs additions.
    from ui.pages.results import results_ui
    tag = results_ui()
    assert tag is not None
    html = str(tag)
    assert "compare_window_years" in html      # the new slider is wired
    assert "run_delta_chart" in html            # the new chart output
    assert "run_delta_table" in html            # the new table output
```

(If `results_ui()` needs args, inspect its signature: `grep -n "def results_ui" ui/pages/results.py` and call it accordingly. If the output IDs aren't literally in `str(tag)` for `output_widget`, assert on the slider id + that `str(tag)` contains the controls; adjust to what the tag tree actually serializes — the goal is "the page builds and includes the new controls".)

- [ ] **Step 5: Run smoke + the module imports**

Run: `.venv/bin/python -c "import ui.pages.results; print('import ok')"` (catches syntax/NameErrors).
Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v` (all pass — Task 1 + the smoke).
Run: `.venv/bin/ruff check ui/pages/results.py tests/test_ui_results.py && .venv/bin/ruff format --check ui/ tests/`.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/results.py tests/test_ui_results.py
git commit -m "feat(ui): Compare Runs output-delta chart + table (run_delta wired)"
```

---

## Task 3: Manual UI run-through + docs + finalize

**Files:**
- Modify: a usage doc (find: `grep -rln "Compare Runs\|compare_runs\|ui/pages/results" docs/ | head`; else add a line to `docs/baltic_example.md` near the UI/results discussion).

- [ ] **Step 1: Manual UI run-through (render fns aren't unit-tested — verify by running the app)**

Launch the app and confirm the wiring (the render decorators are only exercised live):
Run: `.venv/bin/shiny run app.py --host 127.0.0.1 --port 8000` (background), then open the Results → Compare Runs tab. Confirm: the **Window (years)** slider appears; selecting **0 or 1** run shows the "Select exactly 2 runs" prompts (chart title + table div); selecting **exactly 2** runs renders the diverging-bar delta chart + the markdown delta table; the **Metric** select switches the delta between biomass/yield/abundance. If no runs are available in the selector (the inherited history-dir quirk), note it — the wiring is still verified by the prompt states + the helper unit tests. Capture what you observed.

(If launching the full app is impractical in this environment, instead verify the render fns by importing the module and calling `results_server` is wired without error, and rely on the Task-1 helper tests + the Task-2 build smoke. State which path you used.)

- [ ] **Step 2: Docs**

Add a short note (in the doc found above) that the Results → Compare Runs tab now shows a per-species **output delta** when exactly 2 runs are selected (ranked table + diverging-bar chart; baseline = first selected, variant = second; metric + window-years controls). Mention it complements the `scripts/compare_runs.py` CLI (same `run_delta` engine). Note the inherited limitation: the run selector populates only when run history exists under `<output_dir>/../.osmose_history`.

- [ ] **Step 3: Full verification + lint**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v` (all pass; report count).
Run: `.venv/bin/python -m pytest tests/ -k "ui or results or analysis" -q` (no regressions; report counts, note pre-existing).
Run: `.venv/bin/ruff check ui/ tests/ && .venv/bin/ruff format --check ui/ tests/` (clean on touched files).

- [ ] **Step 4: Commit + finish**

```bash
git add docs/ ui/pages/results.py tests/test_ui_results.py
git commit -m "docs(ui): document Compare Runs output-delta section"
```

Use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** `_delta_for_selected` pure helper (pairwise, reconstruct OsmoseResults, run_delta, ValueError on !=2) → T1; window-years slider + reuse compare_metric + 2 outputs → T2 Steps 1-2; `run_delta_chart` (render_plotly, !=2 guard, error guard) + `run_delta_table` (render.ui, ui.markdown(format_delta_report), error guard) → T2 Step 3; tests (helper ranking/sign/raises + page-builds smoke) → T1/T2; manual run-through + docs + no-app.py-change → T3. Deferred (top_n, N-way, history-dir fix, per-period/cell) → not in plan, per spec. ✅

**Placeholder scan:** no TBD/TODO; every code step complete. T2 Step4 + T3 Step1 have explicit "adjust to what the tag serializes / which verification path you used" fallbacks — guarded instructions, not placeholders. ✅

**Type consistency:** `_delta_for_selected(records, metric, window_years)` signature identical T1↔T2 (both render fns call it the same way). Reuses shipped `run_delta`/`make_run_delta_chart`/`format_delta_report` signatures verbatim. Input IDs `compare_runs_select`/`compare_metric`/`compare_window_years` consistent across UI + server. Output IDs `run_delta_chart`/`run_delta_table` match between `results_ui()` placeholders and the `results_server()` render fn names. ✅
