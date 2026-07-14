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
exact same picked solution — no divergence. On button click:

```python
@reactive.effect
@reactive.event(input.cal_apply_solution)
def _apply_pareto_solution():
    sol = _picked_solution()
    if not sol or not sol.get("params"):
        ui.notification_show("Pick a Pareto solution first.", type="warning")
        return
    new_cfg, n = apply_solution_overrides(state.config.get(), sol["params"])
    state.config.set(new_cfg)
    ui.notification_show(
        f"Applied {n} parameter{'s' if n != 1 else ''} to the current config.", type="message"
    )
```

### Behaviour

- No solution selected → warning notification, config untouched.
- Applied → `state.config` updated (live across Domain/Species/Run/etc.), success notification with
  the changed-key count. The user then Runs it or saves it via Scenarios.
- `state.config` is the same reactive the rest of the app reads/writes via `state.config.set()`
  (pattern: `advanced.py:175`, `forcing.py:132`, `setup.py:178`), so the applied values are
  immediately consistent everywhere.

## Testing strategy

1. **`apply_solution_overrides` unit tests** (pure, no Shiny):
   - Merges params into a config dict; unlisted keys untouched.
   - Float→string rendering matches `solution_overrides_csv` for the same params (assert the values
     in `new_config` equal the values parsed from `solution_overrides_csv`).
   - `keys_changed` counts only genuinely-changed/new keys (a param already equal to the config's
     value is not counted); does not mutate the input config.
   - Empty params → `(copy_of_config, 0)`.
2. **Handler smoke** (light): the `_apply_pareto_solution` logic is a 6-line effect; its only
   non-trivial piece is the pure helper (tested in 1) and the `state.config.set` call. An e2e is
   **not** warranted (a real NSGA-II run to populate the front is heavy and the emergent
   calibration e2e is CI-fragile) — the pure helper + a manual UI smoke (run/load a front → pick →
   Apply → confirm a config field on the Domain/Species page reflects the value) is the verification.
3. **No regression:** the existing download button and picker are unchanged; the calibration page's
   existing tests stay green.

## Verification

Manual UI smoke on the running app: load a saved NSGA-II front (History) or run a tiny NSGA-II
calibration, pick a Pareto solution, click **Apply to current config**, and confirm (a) the success
toast shows the changed-key count and (b) a corresponding config value is updated on another page
(e.g. a free-parameter field on Species/Advanced). Plus `apply_solution_overrides` unit tests green
and `ruff` clean.

## Rollback

Additive: one pure helper in `pareto.py`, one button + one effect in `calibration.py`, plus tests.
No change to existing behaviour, data, or config format. Revertible.
