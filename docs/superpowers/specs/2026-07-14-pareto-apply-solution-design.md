# Apply a selected Pareto solution to the config — design

**Date:** 2026-07-14
**Status:** Draft (brainstorming → spec)
**Branch:** `feat/pareto-apply-solution`

## Motivation

The multi-objective calibration **Pareto-front explorer already exists** in the calibration page
(`ui/pages/calibration.py`): both optimizers (NSGA-II Direct, GP surrogate) produce a front, and
the "Best Parameters" panel renders a Pareto scatter, a selectable Pareto table
(`_render_pareto_table`), parallel-coordinates, a solution picker (`select_solution`), and a
**Download parameters (CSV)** button (`cal_export_params` → `solution_overrides_csv`).

The one workflow break: a selected solution can **only be downloaded** as a `key ; value` overrides
CSV, which the user must then **merge into a config by hand**. There is no way to apply the picked
solution to the live config in-app — so the explorer dead-ends right before the payoff (pick a
compromise solution → *use it*).

## Goal

Add **"Apply to current config"** to the Pareto-solution panel: merge the selected solution's
parameter overrides into the app's live reactive config (`state.config`), so the user can
immediately run it (Run page) or persist it (existing Scenarios → "Save Current Config").

## Non-goals (YAGNI — the explorer is otherwise complete)

- A dedicated **"Save as scenario"** button on the calibration page — redundant with the existing
  Scenarios page flow: *Apply to config* then *Scenarios → Save Current Config* already saves it in
  two clicks, reusing `osmose/scenarios.py`'s store. Don't duplicate scenario-management UI here.
- Interactive point-selection on the Pareto scatter (click-to-select) — the table row picker
  already selects; plotly-click plumbing is fiddly for marginal gain.
- 3+ objective exploration — parallel-coordinates already covers ≥3; interactive high-dim view is a
  separate, larger build for uncertain demand.
- Any change to the optimizers, charts, the picker, or the download button (all stay as-is).

## Design

### 1. Pure merge helper (testable) — `osmose/calibration/pareto.py`

Add alongside `solution_overrides_csv` (same module, same shape of input):

```python
def apply_solution_overrides(config, params):
    """Merge a selected Pareto solution's {key: value} params into an OSMOSE config dict.

    OSMOSE config values are strings; solution params are floats — render each with the same
    formatting as ``solution_overrides_csv`` (str(value)). Returns (new_config, keys_changed) where
    keys_changed counts params whose stringified value differs from the config's current value
    (a new key counts as changed). Does not mutate the input config.
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

Rationale: the string rendering matches the download path (`solution_overrides_csv` emits
`f"{k} ; {v}"`), so *Apply* and *Download* produce identical values — no divergence between the two
ways of consuming a solution.

### 2. UI button — `ui/pages/calibration.py`, "Best Parameters" panel

Immediately after the existing `cal_export_params` download button, add:

```python
ui.input_action_button(
    "cal_apply_solution",
    "Apply to current config",
    class_="btn-outline-success btn-sm",
),
```

### 3. Handler — `ui/pages/calibration.py` server

The selected solution is already resolved by the existing helper **`_picked_solution()`**
(`calibration.py:574-587`) — it reads `cal_X`/`cal_F`/`input.cal_pareto_idx`/param-keys and returns
`select_solution(...)` → `{"index", "params", "objectives"}` or `None`. The download handler
`cal_export_params` (`calibration.py:607-610`) already consumes it, so the Apply handler reuses the
exact same picked solution — no divergence.

**This is the same operation as `advanced.py::confirm_import` (merge an external `{key: value}`
override dict into `state.config`), so it MUST follow that handler's full config-write wiring —
`state.config.set` + `state.dirty.set(True)` + a `state.load_trigger` bump** (`advanced.py:174-186`,
in-loop-review finding). Missing either flag is a real bug: without `dirty` the "modified" badge
(`app.py:627-636`) never lights; **without the `load_trigger` bump, already-rendered pages
(Domain/Species/Movement/…) keep showing stale values** because they read `state.config` inside
`reactive.isolate()` and re-render only on `load_trigger` (`setup.py:109-133` etc.). Extract the
apply into a plain function so it is unit-testable without the Shiny event machinery:

```python
def apply_picked_solution(state, params) -> int:
    """Merge a picked solution's params into state.config with the app's standard config-write
    wiring, so every page re-reads and the modified badge lights. Returns keys_changed."""
    with reactive.isolate():
        cfg = dict(state.config.get())
    new_cfg, n = apply_solution_overrides(cfg, params)
    state.config.set(new_cfg)
    state.dirty.set(True)                                     # "modified" badge
    with reactive.isolate():
        state.load_trigger.set(state.load_trigger.get() + 1)  # already-rendered pages re-read
    return n


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
        type="message", duration=5,
    )
```

No `state.key_case_map` update is needed (unlike `advanced.py`'s arbitrary imports): the solution's
keys are **pre-existing free-parameter config keys** already present in `state.config`, so Apply
only updates existing keys' values — it introduces none.

### 4. Preview the change before applying (in-loop-review finding)

A calibration solution can change many parameters at once, and there is no undo in the UI — a
silent whole-config mutation with only a count is risky. The `cal_selected_solution` render
(`calibration.py:589-605`) already draws a **Parameter / Value** table for the picked solution in
exactly this spot; upgrade it to **Parameter / Current / New** (current = `state.config.get(k)` or
"(not set)", new = `str(v)`, changed rows emphasized). This is a near-free preview-before-commit —
the user sees precisely what Apply will change, and a stale solution whose keys aren't in the live
config surfaces as "(not set)" current values. Reuse `ui/components/config_diff.py`
(`classify_config_diffs`/`render_config_diff_table`, already used by Scenarios Compare / Scenario
Diff) if its `[{key, value_a, value_b}]` shape fits; otherwise render the 3-column table inline.

### Behaviour

- No solution selected → warning notification, config untouched.
- Before applying, the Current/New preview (§4) shows exactly which keys change.
- Applied → `state.config` updated, `dirty` set (modified badge lights), `load_trigger` bumped so
  every already-rendered page re-reads and shows the new values; success notification with the
  changed-key count. The user then Runs it or saves it via Scenarios → Save Current Config.
- This is the same config-write contract as `advanced.py::confirm_import` (set + dirty +
  load_trigger); with all three, the applied values are consistent across Domain/Species/Run/etc.

## Testing strategy

1. **`apply_solution_overrides` unit tests** (pure, no Shiny):
   - Merges params into a config dict; unlisted keys untouched.
   - Float→string rendering matches `solution_overrides_csv` for the same params (assert the values
     in `new_config` equal the values parsed from `solution_overrides_csv`).
   - `keys_changed` counts only genuinely-changed/new keys (a param already equal to the config's
     value is not counted); does not mutate the input config.
   - Empty params → `(copy_of_config, 0)`.
2. **`apply_picked_solution` reactive-state test** (in-loop-review finding — this is the test that
   mechanically catches the two wiring bugs): follow the `tests/test_ui_state.py` +
   `tests/helpers.py` pattern — build an `AppState`, seed `state.config` with a small config,
   call `apply_picked_solution(state, params)` inside `reactive.isolate()`, and assert:
   `state.config.get()` has the merged values; `state.dirty.get() is True`; `state.load_trigger`
   incremented. No NSGA-II run, no browser — fully deterministic. (This is why the apply logic is
   extracted into a plain function rather than living only in the `@reactive.event` effect.)
3. **Preview helper** (if a new diff helper is added rather than reusing `config_diff.py`): unit-test
   that current-vs-new rows classify changed / unchanged / not-set correctly.
4. **Manual UI smoke** (verification, not CI): run/load a front → pick a solution → confirm the
   Current/New preview → Apply → confirm the "modified" badge lights and a free-parameter field on
   Species/Advanced shows the new value. An e2e is **not** warranted — a real NSGA-II run is heavy
   and the emergent calibration e2e is CI-fragile ([[feedback-ci-fragile-emergent-tests]]).
5. **No regression:** the existing download button and picker are unchanged; calibration tests green.

## Verification

Manual UI smoke on the running app: load a saved NSGA-II front (History) or run a tiny NSGA-II
calibration, pick a Pareto solution, click **Apply to current config**, and confirm (a) the success
toast shows the changed-key count and (b) a corresponding config value is updated on another page
(e.g. a free-parameter field on Species/Advanced). Plus `apply_solution_overrides` unit tests green
and `ruff` clean.

## Rollback

Mostly additive: one pure helper (`apply_solution_overrides`) in `pareto.py`; one button, one
`apply_picked_solution` function + effect in `calibration.py`; plus tests. The one modification to
existing UI is upgrading the `cal_selected_solution` table from 2 to 3 columns (Current/New). No
change to the optimizers, the picker, the download, data, or config format. Revertible.
