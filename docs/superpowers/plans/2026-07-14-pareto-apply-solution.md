# Apply a selected Pareto solution to the config — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user apply a selected Pareto-front calibration solution to the live app config (not just download it), with a before-apply Current/New preview.

**Architecture:** A pure `apply_solution_overrides(config, params) → (new_config, n_changed)` in `osmose/calibration/pareto.py`; a `apply_picked_solution(state, params)` wiring function in `ui/pages/calibration.py` that merges via the pure helper and applies the app's standard config-write contract (`config.set` + `dirty.set(True)` + `load_trigger` bump, matching `advanced.py::confirm_import`); an "Apply to current config" button reusing the existing `_picked_solution()`; and an upgrade of the already-rendered `cal_selected_solution` table to a Current/New diff preview.

**Tech Stack:** Python 3.12, Shiny for Python (`reactive.Value`, `@reactive.effect`/`@reactive.event`, `reactive.isolate`), NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-14-pareto-apply-solution-design.md` (in-loop reviewed).

## Global Constraints

- The config-write contract is **`state.config.set(...)` + `state.dirty.set(True)` + a `state.load_trigger` bump under `reactive.isolate()`** — exactly `advanced.py:174-186`. Omitting `dirty` breaks the "modified" badge; omitting `load_trigger` leaves already-rendered pages showing stale values.
- Rendered override values use `str(value)` — identical to `solution_overrides_csv`'s `f"{k} ; {v}"`, so Apply and Download never diverge.
- The picked solution's keys are **full config keys already present in `state.config`** (free parameters); no `state.key_case_map` update is needed.
- Do NOT change the optimizers, the picker (`cal_pareto_picker`), the download button (`cal_export_params`), or the objective/chart code.
- ruff check + format clean on `osmose/ ui/ tests/`.

---

### Task 1: Pure `apply_solution_overrides` helper

**Files:**
- Modify: `osmose/calibration/pareto.py` (add function after `solution_overrides_csv`, ~line 75)
- Test: `tests/test_calibration_pareto.py` (create, or append if it exists)

**Interfaces:**
- Produces: `apply_solution_overrides(config: dict[str, str], params: dict[str, float]) -> tuple[dict[str, str], int]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibration_pareto.py
import numpy as np

from osmose.calibration.pareto import apply_solution_overrides, solution_overrides_csv


def test_apply_solution_overrides_merges_and_counts():
    cfg = {"a": "1", "b": "2"}
    new, n = apply_solution_overrides(cfg, {"a": 5.0, "c": 3.5})
    assert new == {"a": "5.0", "b": "2", "c": "3.5"}
    assert n == 2                        # "a" changed, "c" added
    assert cfg == {"a": "1", "b": "2"}   # input not mutated


def test_apply_solution_overrides_unchanged_value_not_counted():
    new, n = apply_solution_overrides({"a": "5.0"}, {"a": 5.0})
    assert new == {"a": "5.0"} and n == 0


def test_apply_solution_overrides_empty_params():
    cfg = {"a": "1"}
    new, n = apply_solution_overrides(cfg, {})
    assert new == cfg and n == 0


def test_apply_solution_overrides_matches_csv_rendering():
    """Apply and Download must render identical string values for the same params."""
    params = {"x.y": 0.125, "z": 3.0}
    new, _ = apply_solution_overrides({}, params)
    csv_vals = dict(
        line.split(" ; ") for line in solution_overrides_csv(params).strip().split("\n")
    )
    assert new == csv_vals
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_calibration_pareto.py -q`
Expected: FAIL — `ImportError: cannot import name 'apply_solution_overrides'`.

- [ ] **Step 3: Implement the helper**

Append to `osmose/calibration/pareto.py`:

```python
def apply_solution_overrides(config, params):
    """Merge a picked solution's ``{key: value}`` params into an OSMOSE config dict.

    OSMOSE config values are strings; solution params are floats — each is rendered with
    ``str(value)``, identical to :func:`solution_overrides_csv`, so Apply and Download never
    diverge. Returns ``(new_config, keys_changed)`` where ``keys_changed`` counts params whose
    stringified value differs from the config's current value (a not-yet-present key counts as
    changed). Does not mutate the input config.
    """
    new_config = dict(config)
    changed = 0
    for k, v in params.items():
        sv = str(v)
        if new_config.get(k) != sv:
            changed += 1
        new_config[k] = sv
    return new_config, changed
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_calibration_pareto.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + commit**

```bash
cd /home/razinka/osmopy && .venv/bin/ruff check osmose/calibration/pareto.py tests/test_calibration_pareto.py && .venv/bin/ruff format osmose/calibration/pareto.py tests/test_calibration_pareto.py
git add osmose/calibration/pareto.py tests/test_calibration_pareto.py
git commit -m "feat(calib): apply_solution_overrides — merge a Pareto solution into a config dict"
```

---

### Task 2: Calibration UI — apply wiring, button, and Current/New preview

**Files:**
- Modify: `ui/pages/calibration.py` (add `apply_picked_solution` + `_solution_diff_rows` module functions; add the "Apply" button after `cal_export_params` at ~line 334; upgrade `cal_selected_solution` at 589-605; add the `_apply_pareto_solution` effect in the server; add imports)
- Test: `tests/test_calibration_apply_solution.py` (create)

**Interfaces:**
- Consumes: `apply_solution_overrides` (Task 1); `_picked_solution()` (`calibration.py:574-587`); `AppState.{config,dirty,load_trigger}` (`ui/state.py:36,59`); `classify_config_diffs` (`ui/components/config_diff.py:19`).
- Produces: `apply_picked_solution(state, params: dict[str, float]) -> int`; `_solution_diff_rows(config, params) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibration_apply_solution.py
import pytest

pytest.importorskip("shiny")
from shiny import reactive

from ui.pages.calibration import _solution_diff_rows, apply_picked_solution
from ui.state import AppState


def test_solution_diff_rows_current_vs_new():
    rows = _solution_diff_rows({"a": "1"}, {"a": 2.0, "b": 3.0})
    by_key = {r["key"]: r for r in rows}
    assert by_key["a"]["value_a"] == "1"
    assert by_key["a"]["value_b"] == "2.0"
    assert by_key["a"]["change"] == "changed"
    assert by_key["b"]["value_a"] is None      # not in current config
    assert by_key["b"]["value_b"] == "3.0"
    assert by_key["b"]["change"] == "added"


def test_apply_picked_solution_wires_config_dirty_and_load_trigger():
    state = AppState()
    with reactive.isolate():
        state.config.set({"mortality.additional.rate.sp0": "0.5", "keep": "x"})
        t0 = state.load_trigger.get()
        n = apply_picked_solution(state, {"mortality.additional.rate.sp0": 0.8})
        assert n == 1
        assert state.config.get()["mortality.additional.rate.sp0"] == "0.8"
        assert state.config.get()["keep"] == "x"        # untouched key preserved
        assert state.dirty.get() is True                 # modified badge lights
        assert state.load_trigger.get() == t0 + 1        # pages re-read
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_calibration_apply_solution.py -q`
Expected: FAIL — `ImportError: cannot import name 'apply_picked_solution'` (and `_solution_diff_rows`).

- [ ] **Step 3: Add the module functions + import**

In `ui/pages/calibration.py`, add to the imports near the top (the file already imports `reactive` from shiny and `apply_solution_overrides`'s siblings from `osmose.calibration.pareto`):

```python
from osmose.calibration.pareto import apply_solution_overrides  # add to the existing pareto import block
from ui.components.config_diff import classify_config_diffs
```

Add these module-level functions (near the other `_render_*`/helper functions, before `calibration_ui`):

```python
def _solution_diff_rows(config, params):
    """Classified Current(value_a)/New(value_b) rows for a picked solution vs the live config."""
    diffs = [{"key": k, "value_a": config.get(k), "value_b": str(v)} for k, v in params.items()]
    return classify_config_diffs(diffs)


def apply_picked_solution(state, params) -> int:
    """Merge a picked solution's params into ``state.config`` with the app's standard config-write
    wiring (dirty flag + load_trigger bump so every already-rendered page re-reads). Returns the
    number of changed keys. Plain function (no Shiny event machinery) so it is unit-testable."""
    with reactive.isolate():
        cfg = dict(state.config.get())
    new_cfg, n = apply_solution_overrides(cfg, params)
    state.config.set(new_cfg)
    state.dirty.set(True)
    with reactive.isolate():
        state.load_trigger.set(state.load_trigger.get() + 1)
    return n
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_calibration_apply_solution.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the "Apply" button**

In `ui/pages/calibration.py`, immediately after the `cal_export_params` download button (`ui.download_button("cal_export_params", ...)` at ~line 330-334), add:

```python
ui.input_action_button(
    "cal_apply_solution",
    "Apply to current config",
    class_="btn-outline-success btn-sm",
),
```

- [ ] **Step 6: Upgrade `cal_selected_solution` to a Current/New preview**

Replace the body of the existing `cal_selected_solution` render (`calibration.py:589-605`) so its table shows Parameter / Current / New / Change instead of Parameter / Value:

```python
    @render.ui
    def cal_selected_solution():
        sol = _picked_solution()
        if sol is None:
            return ui.div()
        rows = _solution_diff_rows(state.config.get(), sol["params"])
        badge = {"changed": "bg-secondary", "added": "bg-success", "removed": "bg-danger"}
        body = [
            ui.tags.tr(
                ui.tags.td(r["key"]),
                ui.tags.td("(not set)" if r["value_a"] is None else r["value_a"]),
                ui.tags.td(r["value_b"]),
                ui.tags.td(ui.tags.span(r["change"], class_=f"badge {badge[r['change']]}")),
            )
            for r in rows
        ]
        obj_str = ", ".join(f"{v:.4g}" for v in sol["objectives"])
        return ui.div(
            ui.p(ui.tags.strong(f"Solution #{sol['index']} — objectives: "), obj_str),
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Parameter"),
                        ui.tags.th("Current"),
                        ui.tags.th("New"),
                        ui.tags.th("Change"),
                    )
                ),
                ui.tags.tbody(*body),
                class_="table table-sm table-striped",
            ),
        )
```

- [ ] **Step 7: Add the Apply effect in the server**

In the calibration server (near the `cal_export_params` download handler, ~line 607), add:

```python
    @reactive.effect
    @reactive.event(input.cal_apply_solution)
    def _apply_pareto_solution():
        sol = _picked_solution()
        if not sol or not sol.get("params"):
            ui.notification_show("Pick a Pareto solution first.", type="warning")
            return
        n = apply_picked_solution(state, sol["params"])
        ui.notification_show(
            f"Applied {n} parameter{'s' if n != 1 else ''} to the current config.",
            type="message",
            duration=5,
        )
```

(Confirm the calibration server function receives `state` — it does; the page reads `state.config.get()` at `calibration.py:459,535`.)

- [ ] **Step 8: Run the full relevant suite + lint**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_calibration_apply_solution.py tests/test_calibration_pareto.py -q && PYTHONPATH=. .venv/bin/python -m pytest tests/ -k "calibration or ui_state or config_diff" -q`
Expected: PASS — new tests green; existing calibration/state/config-diff tests unaffected.
Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
cd /home/razinka/osmopy && git add ui/pages/calibration.py tests/test_calibration_apply_solution.py
git commit -m "feat(calib): Apply picked Pareto solution to config (+ Current/New preview)"
```

---

### Task 3: Manual UI verification

**Files:** none (verification only).

- [ ] **Step 1: Drive the app and confirm the loop closes**

Launch the app (or use the running prod instance), go to Calibration, load a saved NSGA-II front from History (or run a tiny NSGA-II calibration), open the "Best Parameters" panel, and:
1. Pick a Pareto solution → confirm the selected-solution table now shows **Parameter / Current / New / Change** with changed/added badges.
2. Click **Apply to current config** → confirm the success toast reports the changed-key count.
3. Confirm the header **"modified" badge** now shows, and a free-parameter value on the **Species/Advanced** page reflects the applied value (proves `dirty` + `load_trigger` wiring).
4. With no solution picked, click Apply → confirm the "Pick a Pareto solution first." warning and the config is untouched.

Record the outcome. (No automated e2e — a real NSGA-II run is heavy and the emergent calibration e2e is CI-fragile; the Task-1/2 unit + reactive-state tests are the automated coverage.)

---

## Self-review

- **Spec coverage:** §1 pure helper → Task 1; §3 apply wiring (dirty + load_trigger) → Task 2 (`apply_picked_solution` + reactive-state test); §2 button → Task 2 Step 5; §4 Current/New preview → Task 2 Steps 3/6 (`_solution_diff_rows` + `cal_selected_solution`); testing strategy → Tasks 1–2 tests + Task 3 manual smoke; non-goals (Save-as-scenario, click-select, 3+obj) → untouched. All covered.
- **Placeholder scan:** every code step has full code; the only runtime unknown is the exact insertion line for the button/effect, given as a code anchor (`cal_export_params`) not a bare line number.
- **Type consistency:** `apply_solution_overrides(config, params) -> (dict, int)`, `apply_picked_solution(state, params) -> int`, `_solution_diff_rows(config, params) -> list[dict]` — used identically across tasks and tests; `classify_config_diffs` row shape (`value_a`/`value_b`/`change`) matches its definition.
- **Ordering:** Task 1 (pure helper) precedes Task 2 (which imports it). Task 2's reactive-state test catches the dirty/load_trigger wiring mechanically.
