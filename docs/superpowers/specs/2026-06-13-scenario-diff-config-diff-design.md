# Scenario Diff — Config Diff panel design

**Date:** 2026-06-13
**Status:** Approved (design phase)
**Feature:** Add a "Config differences" panel to the existing Scenario Diff tab that shows
which config keys differ between run A (baseline) and run B (variant), pairing the
*output* delta the tab already shows with the *config* delta that drove it.

## Motivation

The Scenario Diff tab (shipped in PR #60) answers "what did the run outputs do differently"
— overlaid per-species biomass curves and A/B/B−A spatial maps. It does not answer "what was
configured differently". A reader comparing two runs has to remember, or look up elsewhere,
which parameters changed. This panel closes that loop: the tab reads top-to-bottom as
**what you changed → what it did** (config delta → biomass overlay → spatial diff).

This folds in the separately-backlogged standalone "Config diff tool" — there is no need for a
separate page when the two-run selectors already live here.

## Reuse (do not rebuild)

`osmose/history.py` already implements the config diff. `RunHistory.compare_runs(ts_a, ts_b)`
returns, for the two runs' `config_snapshot` dicts, a list of every differing key:

```python
[{"key": str, "value_a": str | None, "value_b": str | None}, ...]
```

A key present in only one run appears with `None` on the missing side. `config_snapshot` is
populated for real runs (the Run page writes the run's full config into the `RunRecord`). The
Scenario Diff tab's `diff_run_a` / `diff_run_b` selectors already resolve to exactly the
timestamps `compare_runs` needs (`default_run_history().load_run(ts)` in `_results_for`).

So the only genuinely new logic is a small pure classifier that tags each diff row as
added / removed / changed and orders them for display. No new data path, no new diff algorithm.

The Compare Runs tab (`ui/pages/results.py:754` `config_diff_table`) already renders an N-run
config diff from `compare_runs_multi` as a Bootstrap table; this panel follows the same table
styling (`table table-sm table-striped`, `STYLE_MONO_KEY` for keys, `STYLE_EMPTY` for empty
states) for visual consistency, but is two-run, change-classified, and lives in the Scenario
Diff tab.

## Architecture

Single file touched for behaviour: `ui/pages/scenario_diff.py`.

1. **Pure classifier** — a module-level function in `scenario_diff.py`:

   ```python
   def _classify_config_diffs(diffs: list[dict]) -> list[dict]:
       """Tag each {key, value_a, value_b} row with a change type and sort for display.

       change is "added"   when value_a is None (key only in B),
                 "removed" when value_b is None (key only in A),
                 "changed" otherwise (both present, differ).
       Sort order: changed first, then added, then removed; alphabetical by key within
       each group (stable, deterministic — no reliance on input order).
       """
   ```

   Returns rows shaped `{"key", "value_a", "value_b", "change"}`. Pure (no I/O), so it is
   unit-tested directly. Input is exactly what `compare_runs` returns (`value_a`/`value_b` may
   be `None`); the classifier never raises on an empty list (returns `[]`).

2. **Render function** — a bare `@render.ui` `diff_config_table` (matching the tab's existing
   bare-render convention, no `@output`):
   - Reads `diff_run_a` / `diff_run_b` via the existing `_safe(...)` helper.
   - Empty / guard states (each returns a `ui.div(..., style=STYLE_EMPTY)` muted message):
     - either timestamp missing → "Select two runs to compare their configs."
     - `ts_a == ts_b` → "Same run selected — no config differences."
     - `compare_runs` raises (stale/missing run file) → "Could not load run configs."
       (caught broadly like `config_diff_table` does, `# noqa: BLE001`, so a bad history
       file degrades instead of crashing the page).
     - `compare_runs` returns `[]` → "Identical configuration — no differences."
   - Otherwise calls `_classify_config_diffs` and renders:
     - a header line: "N config keys differ" (N = row count).
     - a scrollable table with columns **Key | A | B**, one row per differing key:
       - Key cell uses `STYLE_MONO_KEY`.
       - A and B value cells show the value, or "—" when `None` (added/removed side).
       - a small change-type chip per row (a Bootstrap badge: `changed` = secondary,
         `added` = success, `removed` = danger) so the row's nature is scannable.
     - The table is wrapped in a scroll container (`max-height` with `overflow:auto`) so a
       large diff doesn't push the biomass chart off-screen.

3. **Placement** — wired into `scenario_diff_nav_panel()` directly under the A/B selector
   block and above the biomass chart, inside a collapsible `ui.accordion` /
   `ui.accordion_panel` ("Config differences") so it is tidy when not needed. The accordion is
   `open=True` by default (the diff is the point of the comparison); when configs are identical
   or no runs are selected the panel body just shows the muted empty state.

## Data flow

```
diff_run_a, diff_run_b (timestamps, existing selectors)
        │
        ▼
default_run_history().compare_runs(ts_a, ts_b)   ──► [{key, value_a, value_b}, ...]
        │
        ▼
_classify_config_diffs(...)                       ──► [{key, value_a, value_b, change}, ...]
        │
        ▼
diff_config_table (@render.ui)                     ──► Bootstrap table + count header
```

No reactive `Value`s are added — the render reads inputs directly and recomputes on selector
change, exactly like the tab's other `@render.ui` outputs (`diff_biomass_caption`). No NetCDF
handles are involved (this is config-only, independent of the spatial-dataset lifecycle).

## Error handling

- History read failure (corrupt/missing run JSON) → broad `except Exception` (`# noqa: BLE001`)
  → muted "Could not load run configs." message. The page never crashes on a bad record,
  matching the existing `config_diff_table` and `_populate_diff_runs` patterns.
- `None` values on either side are rendered as "—" (not the string "None").
- Empty diff and same-run cases are explicit muted states, not blank.

## Testing

1. **Unit — `tests/test_scenario_diff_config.py`** (new), against `_classify_config_diffs`:
   - changed row (both present, differ) → `change == "changed"`.
   - added row (`value_a is None`) → `change == "added"`.
   - removed row (`value_b is None`) → `change == "removed"`.
   - sort order: changed before added before removed; alphabetical within group.
   - empty input → `[]`.
   - rows preserve `key`/`value_a`/`value_b` verbatim.

2. **Structure test** — assert the panel is wired into the tab: `scenario_diff_nav_panel()`
   renders an element carrying the `diff_config_table` output id (string-search the rendered
   nav-panel HTML, the lightweight check this repo uses for static wiring).

3. **e2e — extend `tests/test_e2e_scenario_diff.py`**: the existing `_write_run` helper writes
   `config_snapshot: {}` for both runs. Give run A and run B **differing** `config_snapshot`s
   (e.g. A `{"predation.efficiency.sp0": "0.5", "mortality.natural.rate.sp0": "0.2"}`, B
   `{"predation.efficiency.sp0": "0.7", "movement.distance.sp0": "3"}`). Add a test that, after
   selecting A and B, the `#diff_config_table` region renders and contains the changed key
   `predation.efficiency.sp0` and both an added and a removed key. (The two existing e2e tests
   keep passing — adding keys to the snapshots does not affect biomass/spatial assertions; the
   same-run test still shows the identical-config empty state.)

All per-task gates run `ruff check`, `ruff format --check`, **and** `pyright` (lesson from the
Scenario Diff and Live Movement features: per-task gates that skipped pyright let type errors
reach the final review). CHANGELOG gets an entry in the same change set.

## Out of scope (YAGNI)

- Schema-registry human-readable labels / descriptions per key (raw dotted keys are already
  informative; deferred nice-to-have).
- Filtering to "meaningful" keys or grouping by config section.
- Highlighting which differing keys *plausibly drove* the observed output delta (a causal
  hint) — interesting, but speculative and out of scope for v1.
- Any change to `compare_runs` / `history.py` (reused as-is).
