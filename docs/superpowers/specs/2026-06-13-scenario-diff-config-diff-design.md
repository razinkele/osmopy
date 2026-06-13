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
styling (`table table-sm table-striped`, `STYLE_MONO_KEY` for keys) for visual consistency, but
is two-run, change-classified, and lives in the Scenario Diff tab. Its empty/guard states use
the tab-local `text-muted` convention (not `config_diff_table`'s `STYLE_EMPTY` block) to match
the other muted lines already in `scenario_diff.py` — see the Render function section.

## Architecture

Single **production** file touched: `ui/pages/scenario_diff.py` (tests and the CHANGELOG also
change — see Testing).

1. **Pure classifier** — a module-level function in `scenario_diff.py`:

   ```python
   def _classify_config_diffs(diffs: list[dict]) -> list[dict]:
       """Tag each {key, value_a, value_b} row with a change type and sort for display.

       change is "added"   when value_a is None (key only in B),
                 "removed" when value_b is None (key only in A),
                 "changed" otherwise (both present, differ).
       Sort: changed group first, then added, then removed; alphabetical by key
       within each group. The group priority is keyed explicitly (NOT by the change
       string's alphabetical order, which would wrongly put "added" first):

           key=lambda r: ({"changed": 0, "added": 1, "removed": 2}[r["change"]], r["key"])

       Deterministic and independent of input order.
       """
   ```

   Returns rows shaped `{"key", "value_a", "value_b", "change"}`. Pure (no I/O), so it is
   unit-tested directly. Input is exactly what `compare_runs` returns (`value_a`/`value_b` may
   be `None`); the classifier never raises on an empty list (returns `[]`).

   **Value contract:** only `None` drives added/removed — it means the key is absent on that
   side. Values are otherwise carried verbatim. A key present on both sides with values that
   differ (including an empty string `""` vs a non-empty string) is `"changed"`; an empty-string
   value renders as an empty cell, never as `"—"` (which is reserved for `None`).

2. **Render function** — a bare `@render.ui` `diff_config_table`, defined **inside**
   `scenario_diff_server` (like the tab's other `@render.ui` functions) so it closes over
   `input` and the `_safe` helper; `_classify_config_diffs` stays module-level. Matches the
   tab's bare-render convention (no `@output`):
   - Reads `diff_run_a` / `diff_run_b` via the existing `_safe(...)` helper. The unselected /
     not-yet-ready value is **falsy** — either `""` (selectors start with `choices={}`, so a
     ready-but-unselected select returns the empty string) or `None` (input not yet registered,
     which `_safe`'s `default=None` returns) — so the guard must use a falsy test
     (`not ts_a or not ts_b`) to cover both. This follows the falsy-guard pattern of
     `config_diff_table` (`results.py:757`) and the `if not ts` short-circuit in `_results_for`
     (`scenario_diff.py:116`). Note `diff_config_table` cannot route through `_results_for`
     (which returns an `OsmoseResults`): it must compare the raw timestamps directly to detect
     `ts_a == ts_b`, which is exactly why its own explicit falsy guard must come first.
   - Empty / guard states evaluated in **this exact order** (they are mutually exclusive only
     in order — e.g. `"" == ""` would wrongly trip the same-run branch if checked first):
     1. `not ts_a or not ts_b` → "Select two runs to compare their configs."
     2. `elif ts_a == ts_b` → "Same run selected — no config differences."
     3. `try` `compare_runs`, `except Exception` (`# noqa: BLE001`, like `config_diff_table`,
        so a stale/missing run file degrades instead of crashing) → "Could not load run configs."
     4. result `== []` → "Identical configuration — no differences."
     5. else → render the table.
     Each empty/guard message is a single compact line using the tab-local muted convention
     `ui.p(msg, class_="text-muted")` (consistent with `diff_biomass_caption:233` and
     `diff_spatial_status:328` in this same file — **not** `STYLE_EMPTY`, which is the
     centered padded block used in `results.py`). Compact so the open-but-empty accordion on
     first load is unobtrusive.
   - Otherwise calls `_classify_config_diffs` and renders:
     - a header line `ui.p(f"{n} differing config key{'s' if n != 1 else ''}", class_="text-muted")`
       (avoids the verb-agreement trap — reads "1 differing config key" / "3 differing config
       keys"; `n` = row count).
     - a Bootstrap table (`class_="table table-sm table-striped"` plus `style="font-size: 13px;"`,
       matching `config_diff_table` at `results.py:783-788`) with **four** columns —
       **Key | A | B | Change** — one `ui.tags.tr` per differing key:
       - Key cell: `ui.tags.td(row["key"], style=STYLE_MONO_KEY)`.
       - A / B cells: the value verbatim, or `"—"` when `None` (added/removed side); an
         empty-string value renders as an empty cell (only `None` becomes `"—"`).
       - Change cell: a Bootstrap badge `ui.tags.span(row["change"], class_=f"badge {cls}")`
         where `cls` comes from the fixed map
         `{"changed": "bg-secondary", "added": "bg-success", "removed": "bg-danger"}` — note the
         base `badge` class is required in addition to the `bg-*` colour. Scope cap (YAGNI line):
         bare change word + fixed class only; no tooltips, no legend, no per-type counts/filters.
     - The whole table is wrapped in a scroll container using the existing
       `STYLE_SCROLL_TABLE` constant (`ui.div(table, style=STYLE_SCROLL_TABLE)` →
       `max-height: 600px; overflow-y: auto;`) so a large diff doesn't push the biomass chart
       off-screen.
   - **Imports:** add `from ui.styles import STYLE_MONO_KEY, STYLE_SCROLL_TABLE` to
     `scenario_diff.py` (it currently imports neither). No new import is needed for the table /
     accordion / badge — `ui.tags.*`, `ui.accordion`, and `ui.accordion_panel` are all reached
     through the already-imported `shiny.ui` (`from shiny import reactive, render, ui`).

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
   - **order-independence:** feed rows in a deliberately scrambled order (removed, then changed,
     then added; keys not alphabetical) and assert the exact output ordering — proves the sort
     is deterministic and not accidentally relying on pre-sorted input.
   - empty input → `[]`.
   - rows preserve `key`/`value_a`/`value_b` verbatim.

2. **Structure test** (in the existing `tests/test_ui_results.py`, next to
   `test_scenario_diff_tab_wired_into_results`) — assert the panel is wired into the tab.
   `str(scenario_diff_nav_panel())` does NOT work (it returns a `NavPanel` repr; `.tagify()`
   raises "must appear within navset_*"). Use one of the two approaches that actually work:
   (a) the repo's established pattern — read the `scenario_diff.py` **source text** and assert
   `diff_config_table` and the accordion are present (mirrors the existing source-text wiring
   test); or (b) render the **panel body**: `str(scenario_diff_nav_panel().content)` tagifies
   the `Tag` and DOES contain the output ids (`"diff_config_table"`). Prefer (b) — it verifies
   the id is actually emitted in the panel content, not merely mentioned in source.

3. **e2e — extend `tests/test_e2e_scenario_diff.py`**: the shared `_write_run(name, idx, *, base)`
   helper hardcodes `config_snapshot: {}` and has no parameter for it. Add a
   `config_snapshot: dict | None = None` keyword param (default to `{}` inside the body) and
   pass differing dicts only from the new test, so the two existing tests' calls keep an empty
   snapshot. Give run A and run B **differing** `config_snapshot`s
   (e.g. A `{"predation.efficiency.sp0": "0.5", "mortality.natural.rate.sp0": "0.2"}`, B
   `{"predation.efficiency.sp0": "0.7", "movement.distance.sp0": "3"}`) — so there is one
   changed key (`predation.efficiency.sp0`), one removed (`mortality.natural.rate.sp0`, A-only)
   and one added (`movement.distance.sp0`, B-only). Add a test that, after selecting A and B,
   asserts on **content** (not visibility — `diff_config_table` is a bare `@render.ui` that
   renders as an empty zero-height `<div id="diff_config_table">` until the reactive recomputes,
   which Playwright would treat as not-visible):
   `expect(page.locator("#diff_config_table")).to_contain_text("predation.efficiency.sp0")`
   (and likewise for `movement.distance.sp0` and `mortality.natural.rate.sp0`). The accordion is
   `open=True`, so it renders expanded (`class "accordion-collapse collapse show"`) on load.
   The two existing e2e tests keep passing: they assert only on `#diff_biomass_chart`,
   `#diff_map_delta`, and `#diff_biomass_caption` ("Identical runs"), none of which read
   `config_snapshot`. (For the record, in the same-run test the config panel shows the
   "Same run selected — no config differences." state — the `ts_a == ts_b` branch, distinct
   from the "Identical configuration" branch — but that test does not assert on it.)

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
