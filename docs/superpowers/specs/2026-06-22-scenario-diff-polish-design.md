# Scenario-Diff Polish — Design Spec

**Date:** 2026-06-22
**Status:** Draft for review
**Topic:** Bring the saved-scenario comparison up to the run-comparison's quality, move it into a modal, and DRY the two diff tables into one shared component.

## Problem

The Scenarios page already has a "Compare Scenarios" feature (inline card: two
`input_select`s + a Compare button + a `compare_results` table). It is crude:

- Renders a bare 3-column table (Parameter / Value A / Value B) — **no change
  classification, no added/removed/changed badges, no diff count, no scroll
  container** (`ui/pages/scenarios.py:504-532`).
- Misleading edge states: selecting the same scenario (`a == b`) and comparing
  two **identical** configs both fall through to the generic empty message
  "Select two scenarios and click Compare." (`handle_compare` sets `[]` in both
  cases → `compare_results` shows the not-selected text).

Meanwhile the **run-comparison page** (`ui/pages/scenario_diff.py`) already has a
much nicer config-diff table — `_classify_config_diffs` (tags each row
added/removed/changed, sorts changed→added→removed then alphabetically) plus a
badged, counted, scrollable table render (`diff_config_table`, lines 287-334).
That logic is **private to that page** and duplicated in spirit by the crude
scenarios table.

## Goal

1. Lift the run-diff's classify + table-render into a **shared, pure UI
   component** that both pages consume — single source of truth for "what a
   config diff looks like."
2. Replace the inline scenarios Compare **card** with a **"Compare Scenarios"
   button → modal** (matches the wizard's modal pattern; declutters the page).
3. Give the scenario diff the run-diff's quality (badges, count, sort, scroll)
   **and** distinct, correct edge-state messages.

Non-goals (YAGNI): search/filter toggles, per-species grouping, exporting the
diff. Those are a clean follow-up once the shared component exists (the separate
"diff-table scannability" backlog item).

## Architecture

### New: `ui/components/config_diff.py` (pure, no I/O)

Two functions lifted verbatim-in-behavior from `scenario_diff.py`:

```python
_CHANGE_ORDER = {"changed": 0, "added": 1, "removed": 2}

def classify_config_diffs(diffs: list[dict]) -> list[dict]:
    """Tag each {key, value_a, value_b} row with change ∈
    {added (value_a is None), removed (value_b is None), changed}, then sort
    changed→added→removed, alphabetical by key within each group. Pure."""

def render_config_diff_table(diffs: list[dict]):
    """Given raw [{key, value_a, value_b}] dicts, classify and return a Shiny
    UI element: a leading count line ('N differing config key(s)') + a
    badged/sorted/scrollable table (Key / A / B / Change). Returns the count
    line + table wrapped in a div. Does NOT decide the empty case — callers
    pass only non-empty diff lists here (empty/edge states are caller-owned,
    because the wording differs per surface)."""
```

- `render_config_diff_table` contains exactly today's `diff_config_table` table
  body (badge classes `{changed: bg-secondary, added: bg-success, removed:
  bg-danger}`, `STYLE_MONO_KEY` keys, `—` for `None` cells, `table table-sm
  table-striped` at `font-size: 13px`, wrapped in `STYLE_SCROLL_TABLE`).
- Imports `STYLE_MONO_KEY`, `STYLE_SCROLL_TABLE` from `ui.styles`.
- **Pure**: no reactive, no history/scenario I/O — takes dicts, returns UI.
  Directly unit-testable on the classify half; the render half is exercised by
  the existing page tests + a light structural assertion.

### Changed: `ui/pages/scenario_diff.py`

- Delete the private `_CHANGE_ORDER` and `_classify_config_diffs`.
- `diff_config_table` keeps its **caller-owned** empty/edge branches (no runs
  selected / same run / identical config / load error) and, for the non-empty
  case, delegates to `render_config_diff_table(diffs)` instead of building the
  table inline.
- Net: behavior identical, ~45 fewer lines, presentation now shared.

### Changed: `ui/pages/scenarios.py`

**UI (`scenarios_ui`)**

- Remove the inline "Compare Scenarios" card (the middle column of the
  `osm-split-layout`) and its `compare_a`/`compare_b`/`btn_compare`/
  `compare_results` widgets from the page body.
- Add a **"Compare Scenarios" button** in the top action row, next to
  "+ New Scenario" (`btn_compare_open`, e.g. `class_="btn-warning mb-3"`).
- Re-flow the remaining cards: the split-layout drops from 3 columns to 2
  (Manage + Bulk Operations); update `col_widths` accordingly (e.g. `[6, 6]`).

**Server (`scenarios_server`)**

- `btn_compare_open` opens a modal (mirrors `_wizard_open`'s
  `ui.modal_show(ui.modal(...))` shape) containing:
  - `input_select("compare_a", "Scenario A", choices=<computed>)`
  - `input_select("compare_b", "Scenario B", choices=<computed>)`
  - `input_action_button("btn_compare", "Compare", class_="btn-warning")`
  - `ui.output_ui("compare_results")`
  - footer: a single Close button (`data-bs-dismiss="modal"`).
  The modal's selectors are created **fresh on each open**, so pass the
  current choices directly into `input_select(choices=...)` at modal-build
  time (compute from the saved-scenario list — the same data that fed the old
  `update_select` effect at `scenarios.py:486-487`). Do **not** rely on a
  separate post-render `update_select`: the inputs don't exist until the modal
  renders, so an update-by-id effect would be a no-op against a freshly built
  modal. Drop the now-dead `update_select(compare_a/compare_b ...)` effect.
- `handle_compare` (the `@reactive.event(input.btn_compare)` effect) — keep, but
  store enough to distinguish edge cases. Replace the current
  "set `[]` for both a==b and missing" with an explicit **state value** so
  `compare_results` can word each case:
  - nothing/one selected → "Select two scenarios and click Compare."
  - `a == b` → "Same scenario selected — no differences."
  - `mgr.compare(a, b)` empty → "Identical configuration — no differences."
  - non-empty → `render_config_diff_table(diffs_as_dicts)`.
- **Adapter**: `mgr.compare` returns `list[ParamDiff]` (dataclass: `key`,
  `value_a`, `value_b`). Convert to the dict shape the shared component wants:
  `[{"key": d.key, "value_a": d.value_a, "value_b": d.value_b} for d in diffs]`
  (via `dataclasses.asdict` or a literal comprehension).
- Delete the crude inline table in `compare_results` (lines 512-532).

## Data flow

```
Scenarios page: [Compare Scenarios] button
   → modal opens, selectors populated from saved-scenario list
   → user picks A, B → [Compare]
   → handle_compare: classify selection into a state
        (none | same | identical | diffs)
   → mgr.compare(A, B) -> list[ParamDiff]
   → asdict -> list[dict]  (adapter)
   → render_config_diff_table(dicts)  ← SHARED component
   → compare_results renders count + badged/sorted/scrollable table

Run-diff page (unchanged UX):
   compare_runs(ts_a, ts_b) -> list[dict]
   → diff_config_table: caller-owned edge messages
   → render_config_diff_table(diffs)  ← SAME shared component
```

Both surfaces converge on `render_config_diff_table`; only the **edge-state
wording** stays caller-local (a run says "Select two runs", a scenario says
"Select two scenarios").

## Edge cases

| Case | Scenario modal | Run-diff page |
|---|---|---|
| <2 scenarios saved | selectors still render; comparing yields the "select two" message | n/a |
| same A==B | "Same scenario selected — no differences." | already: "Same run selected — no config differences." |
| identical configs (empty diff) | "Identical configuration — no differences." | already: "Identical configuration — no differences." |
| missing/corrupt scenario file | `mgr.compare` may raise → guard with try/except, show "Could not load scenario configs." (mirror the run-diff page's `except` degrade) | already guarded |
| `None` value (key only on one side) | shared render shows `—` and tags added/removed | same |

## Testing

- **Unit (pure)** `tests/test_config_diff_component.py`:
  - `classify_config_diffs`: added/removed/changed tagging incl. empty-string
    value ≠ missing; sort order changed→added→removed then alphabetical;
    deterministic regardless of input order.
  - `render_config_diff_table`: returns a UI object; structural smoke (contains
    the count text and a `<table>`); `None` → `—` cell.
- **Adapter** in `tests/test_scenarios*.py` (or a new small test): `ParamDiff`
  list → dict list shape matches what `classify_config_diffs` consumes.
- **Regression**: existing `scenario_diff` page tests stay green (the page now
  delegates but renders the same structure). Run the scenarios/diff e2e if one
  touches the compare card (verify the modal opens and renders a diff).
- Full suite + ruff check/format + pyright on changed files (per repo gate
  conventions).

## Files touched

- **New** `ui/components/config_diff.py` (~55 lines, pure)
- **New** `tests/test_config_diff_component.py`
- **Mod** `ui/pages/scenario_diff.py` (remove private classify+inline render;
  delegate)
- **Mod** `ui/pages/scenarios.py` (inline card → button + modal; richer states;
  adapter; delegate render)
- Possibly **mod** a scenarios test if it asserted the old inline table markup.

## Risks

- Moving Compare into a modal removes the inline `compare_results` output id from
  the page body; any test/e2e that asserted that inline table must move its
  assertion behind the modal-open. Grep `compare_results`, `btn_compare`,
  `compare_a` in `tests/` before editing.
- The run-diff page's `diff_config_table` must keep byte-for-byte the same
  rendered structure so its visual baseline / tests don't flip — the shared
  render is a literal lift, so structure is preserved; verify with the existing
  page test.
