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

- Delete the private `_CHANGE_ORDER` and `_classify_config_diffs` (move both into
  the new component; the public name is `classify_config_diffs`, no leading `_`).
- `diff_config_table` keeps its **caller-owned** empty/edge branches (no runs
  selected / same run / identical config / load error) and, for the non-empty
  case, delegates to `render_config_diff_table(diffs)` instead of building the
  table inline.
- **Move the style imports out**: `STYLE_MONO_KEY` / `STYLE_SCROLL_TABLE`
  (`scenario_diff.py:27`) are used ONLY by the table-build block that moves into
  the new component. After delegation they are unused here → remove them from
  `scenario_diff.py`'s import or ruff F401 fails. (Verify nothing else in the
  file uses them first — current grep says only the table block does.)
- Net: behavior identical, ~45 fewer lines, presentation now shared.

### Changed: `tests/test_scenario_diff_config.py` (REQUIRED — not optional)

This file imports the deleted symbol directly:
`from ui.pages.scenario_diff import _classify_config_diffs` (line 5, 8 call
sites). Deleting/renaming the symbol breaks this file at **collection time**
(`ImportError`), not as a soft assertion. Repoint the import to
`from ui.components.config_diff import classify_config_diffs` and update the 8
call sites to the new (un-underscored) name. The test bodies otherwise stand
(they pin the classify behavior we are preserving).

### Changed: `ui/pages/scenarios.py`

**UI (`scenarios_ui`) — corrected layout facts**

The columns come from `ui.layout_columns(..., col_widths=[3, 5, 4])` holding
**three cards**: Left = *Save Scenario*, Middle = *Saved Scenarios* list, Right =
*Compare Scenarios* (`scenarios.py:35-72`). `osm-split-layout` is the class on the
**outer `ui.div`** (it wraps the action buttons, the `layout_columns`, AND the
*Bulk Operations* card — Bulk is a **separate sibling card, not in the column
grid**, `scenarios.py:73-85`).

- Remove the **Right** card (*Compare Scenarios*) and its
  `compare_a`/`compare_b`/`btn_compare`/`compare_results` widgets from
  `layout_columns`.
- Add a **"Compare Scenarios" button** in the top action row, next to
  "+ New Scenario" (`btn_compare_open`, e.g. `class_="btn-warning mb-3"`).
- Re-flow the **two remaining grid cards** (Save + List): change
  `col_widths=[3, 5, 4]` → a 2-card split (e.g. `[4, 8]` or `[5, 7]`). The Bulk
  Operations card is untouched (it was never in the grid).

**Server (`scenarios_server`)**

- `btn_compare_open` opens a modal (mirrors `_wizard_open`'s
  `ui.modal_show(ui.modal(...))` shape). **Reset the compare state to the
  untouched sentinel BEFORE `ui.modal_show`** (exactly as `_wizard_open` resets
  its four reactive values, `scenarios.py:148-151`) — otherwise a prior
  comparison's table flashes on the next open, because `compare_results` is
  defined once at server-build time and re-binds to its placeholder on every
  modal render. Use `size="l"` and `easy_close=True` (read-only view, backdrop
  dismiss is fine — matches the export/import modals, NOT the multi-step wizard
  which uses `easy_close=False`; because open always resets, backdrop-dismiss
  needs no close handler). The inner `STYLE_SCROLL_TABLE` div stays (modal body
  must not clip it — confirm no `overflow:hidden` fight; large-diff smoke test
  below). The modal contains:
  - `input_select("compare_a", "Scenario A", choices=<computed>)`
  - `input_select("compare_b", "Scenario B", choices=<computed>)`
  - `input_action_button("btn_compare", "Compare", class_="btn-warning")`
  - `ui.output_ui("compare_results")`
  - footer: a single Close button (`data-bs-dismiss="modal"`).
  The modal's selectors are created **fresh on each open**, so pass the
  current choices directly into `input_select(choices=...)` at modal-build
  time. Compute them from **`_scenario_names()`** (`ui/pages/scenarios.py:101-103`)
  — the same source the old `update_compare_choices` effect used. NOTE: that
  helper returns names in `mgr.list_scenarios()` order, **not** alphabetical
  (despite its docstring), so don't assert alphabetical order in tests. Do **not** rely on a
  separate post-render `update_select`: the inputs don't exist until the modal
  renders, so an update-by-id effect would be a no-op against a freshly built
  modal. Delete the now-dead `update_compare_choices` effect entirely
  (`scenarios.py:481-487`) — see the Server section; not just its two
  `update_select` calls.
- `handle_compare` (the `@reactive.event(input.btn_compare)` effect) — keep, but
  replace the current "set `[]` for both a==b and missing" (which conflates
  three distinct cases) with a **single tagged-union reactive value** so there is
  never a stale second list. Define one `compare_state = reactive.value(("none",
  None))` and set it to exactly one of:
  - `("none", None)` → "Select two scenarios and click Compare." *(also the
    reset-on-open and initial value)*
  - `("same", None)` (when `a == b`) → "Same scenario selected — no differences."
  - `("identical", None)` (non-empty selection, empty diff) → "Identical
    configuration — no differences."
  - `("error", None)` → "One or both scenarios could not be loaded — they may
    have been deleted." *(see guard below)*
  - `("diffs", dict_list)` → `render_config_diff_table(dict_list)`.

  `compare_results` switches on the tag only — no separate `compare_diffs` list
  to drift. The untouched sentinel `("none", None)` is distinct from
  `("identical", None)`, so a freshly opened modal shows the prompt, never
  "Identical".
- **Guard `mgr.compare`**: it is **unguarded today** — `compare()`
  (`osmose/scenarios.py:165`) → `load()` does a bare `open()`/`json.load`
  (`osmose/scenarios.py:120-121`) and can raise `FileNotFoundError` /
  `json.JSONDecodeError` / `ValueError` (e.g. a scenario deleted between
  modal-open and Compare-click). Wrap the call in
  `try/except Exception` inside `handle_compare` and set `("error", None)` on
  failure (mirror the run-diff page's broad `except` at
  `scenario_diff.py:294-297`; do NOT catch only `FileNotFoundError`).
- **Adapter**: `mgr.compare` returns `list[ParamDiff]` (flat dataclass: `key`,
  `value_a`, `value_b`; no nested mutables). Convert with the **literal
  comprehension** — NOT `dataclasses.asdict`:
  `[{"key": d.key, "value_a": d.value_a, "value_b": d.value_b} for d in diffs]`.
  Rationale: this is byte-identical to the run-diff's native dict shape
  (`history.py:84`), so both surfaces feed `render_config_diff_table` the same
  3-key dict; `asdict` deep-copies every leaf needlessly and would silently
  absorb a future 4th field.
- **Delete the crude inline table** in `compare_results` (lines 512-532) and the
  old `compare_diffs = reactive.value([])`.
- **Delete the whole `update_compare_choices` effect** (`scenarios.py:481-487`),
  not just its two `update_select` calls — it runs on every `refresh_trigger`
  bump and would target IDs that no longer live in the page body. Choices are
  computed fresh at modal-build time instead (see selector note above).

## Data flow

```
Scenarios page: [Compare Scenarios] button
   → modal opens, selectors populated from saved-scenario list
   → user picks A, B → [Compare]
   → handle_compare: classify selection into a state
        (none | same | identical | diffs)
   → mgr.compare(A, B) -> list[ParamDiff]
   → literal comprehension -> list[dict]  (adapter, matches history.py:84 shape)
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
| 0 scenarios saved | selectors empty → `("none", …)` prompt | n/a |
| exactly 1 scenario saved | both selectors default to it → `a == b` → "Same scenario selected" (NOT "select two") | n/a |
| same A==B | `("same", …)` → "Same scenario selected — no differences." | already: "Same run selected — no config differences." |
| identical configs (empty diff) | `("identical", …)` → "Identical configuration — no differences." | already: "Identical configuration — no differences." |
| scenario deleted between open and Compare | `load()` raises → caught → `("error", …)` → "One or both scenarios could not be loaded — they may have been deleted." | already guarded |
| corrupt scenario JSON | same `("error", …)` path | already guarded |
| `None` value (key only on one side) | shared render shows `—` and tags added/removed | same |
| empty-string value (`""`) | classified `changed` (not added/removed); render shows an **empty cell**, visually distinct intent from `—` — pin with a render test | same |
| very large diff (hundreds of keys) | `STYLE_SCROLL_TABLE` div scrolls inside a `size="l"` modal body; smoke-test ~300 rows | same (page card) |

### Key-normalization caveat (cross-surface)

The two surfaces do **not** normalize keys identically, and the shared component
must not assume they do:
- **Scenario compare** — `ScenarioManager.load` (`osmose/scenarios.py:116`) runs
  `canonicalize_config()` at `osmose/scenarios.py:122-124`, so `compare()`
  (`osmose/scenarios.py:165`) diffs **canonical (lower-cased / renamed) keys**,
  `key_case_map` dropped.
- **Run compare** (`osmose/history.py:74-85`) diffs the **raw** `config_snapshot`
  — no re-canonicalization.

`classify_config_diffs` sorts by `r["key"]` with Python's **case-sensitive** str
order, so mixed-case run keys sort differently than canonical scenario keys. This
is acceptable (each surface is internally consistent) but it means "single source
of truth" applies to *presentation*, not *key normalization*. The component takes
keys as-is. Pin the sort behavior with a mixed-case input test so any future
"normalize keys in the component" change is a conscious decision.

## Testing

- **Unit (pure)** `tests/test_config_diff_component.py`:
  - `classify_config_diffs`: added/removed/changed tagging incl. empty-string
    value ≠ missing; sort order changed→added→removed then alphabetical;
    deterministic regardless of input order; **mixed-case keys** sort
    case-sensitively (pins the normalization caveat).
  - `render_config_diff_table`: returns a UI object; structural smoke (contains
    the count text and a `<table>`); `None` → `—` cell; **`""` → empty cell**
    (distinct from `—`); ~300-row build does not error.
- **Adapter + state**: `ParamDiff` list → 3-key dict list matches what
  `classify_config_diffs` consumes; and a `scenarios_server`-level test that a
  successful compare followed by selecting `a == b` and re-comparing replaces
  the table with the "same scenario" message (no leftover table — proves the
  tagged-union state has no stale list).
- **Repoint** `tests/test_scenario_diff_config.py` imports (see Changed section)
  — its 8 cases keep covering the classify behavior, now via the component.
- **Regression**: existing `scenario_diff` page tests stay green (the page now
  delegates but renders the same structure); `tests/test_ui_results.py:360`
  asserts `"diff_config_table" in html` — unaffected (run page unchanged).
- **New e2e** `tests/test_e2e_scenario_compare.py`: no Scenarios-page Compare
  e2e exists today, so the modal-open path is otherwise uncovered.
  - **State setup (required — the modal needs ≥2 differing scenarios):** the
    selectors are empty in a fresh app, so the test must seed two scenarios
    before comparing. Two options, pick the simpler that fits the conftest:
    (a) write two `data/scenarios/<name>/scenario.json` files with differing
    `config` dicts in a fixture (with `shutil.rmtree` teardown — mirror the
    setup/teardown in `tests/test_e2e_scenario_wizard.py`), or (b) drive the
    "+ New Scenario" wizard twice from two different demos. Prefer (a) — it's
    deterministic and fast.
  - **Flow:** `dismiss_changelog_modal(page)` FIRST, nav to Scenarios, click
    "Compare Scenarios", pick A/B, click Compare, assert a badged diff table.
  - **Scope locators to the compare modal** (e.g. `page.locator(".modal
    #btn_compare")`, `.modal #compare_a`) — a bare `#compare_a` could match a
    detached/stale node. Mind **two stacked modals** (changelog dismissed before
    the compare modal opens).
- Full suite + ruff check/format + **pyright on changed files** (annotate the
  component's return as `list[dict[str, str | None]]` to avoid stub noise) per
  repo gate conventions.

## Files touched

- **New** `ui/components/config_diff.py` (~55 lines, pure)
- **New** `tests/test_config_diff_component.py`
- **New** `tests/test_e2e_scenario_compare.py` (modal-open path; was uncovered)
- **Mod** `ui/pages/scenario_diff.py` (remove private classify+inline render;
  delegate; **remove now-unused `STYLE_MONO_KEY`/`STYLE_SCROLL_TABLE` imports**)
- **Mod** `ui/pages/scenarios.py` (inline card → button + modal; tagged-union
  state + error guard; literal-comprehension adapter; delete
  `update_compare_choices` effect; **remove now-unused `STYLE_DIFF_ROW`
  import** — `STYLE_EMPTY` stays, used at `:285`,`:510`)
- **Mod** `tests/test_scenario_diff_config.py` (repoint import + symbol name)

## Risks

- **Pre-edit grep list** (confirm single occurrence / find all sites):
  `compare_results`, `btn_compare`, `compare_a`, `compare_b`,
  `update_compare_choices`, `_classify_config_diffs`, `_CHANGE_ORDER`,
  `STYLE_DIFF_ROW`. After editing `scenarios_ui`, grep the four widget IDs again
  to guarantee **exactly one** occurrence each (the modal builder) — a leftover
  duplicate id reintroduces the silent-binding race this repo has hit before
  (CLAUDE.md duplicate-id gotcha, commit `ca4a04c`).
- Moving Compare into a modal removes the inline `compare_results` output id from
  the page body; no existing test/e2e asserts it (grep confirmed — only
  `tests/test_ui_scenarios.py` tests `mgr.compare` at the model layer, which is
  unaffected), but the modal-open path is then **untested** until the new e2e is
  added.
- The run-diff page's `diff_config_table` must keep byte-for-byte the same
  rendered structure so its visual baseline / tests don't flip — verified the
  table-build block (`scenario_diff.py:301-333`) captures only locals + module
  style imports (no `input`/`session`/template closure), so the lift is
  genuinely structure-preserving; confirm with the existing page test.

## Gates (repo-specific call-outs)

- **Visual baseline: NO re-bless needed.** The only baseline is `nav_chrome`
  (clips `#main_nav`); this change adds a page-body button + a modal, touches no
  nav item. State this explicitly to preempt a wasted re-bless (the "new nav
  item → re-bless" gotcha does NOT apply here).
- **e2e**: the new test must `dismiss_changelog_modal` before nav and handle the
  two-stacked-modal sequence; default suite excludes e2e (`-m 'not e2e'`) so run
  it explicitly.
- **ruff** = check + format on `osmose/ ui/ tests/`; the two F401 removals above
  are the likely first failures if missed.
- **pyright** on changed files against the clean `[dev]` venv (annotate the
  component dict type).

## Acceptance (definition of done)

1. Full suite green, including the new/repointed test files
   (`test_config_diff_component.py`, repointed `test_scenario_diff_config.py`,
   the adapter/state test).
2. `ruff check` and `ruff format --check` clean on `osmose/ ui/ tests/` (the two
   F401 style-import removals done).
3. `pyright` clean on the changed files.
4. New e2e `tests/test_e2e_scenario_compare.py` passes when run explicitly
   (`-m e2e`); modal opens, seeded A/B compare renders a badged table.
5. Post-edit grep: exactly **one** occurrence each of `compare_a`, `compare_b`,
   `btn_compare`, `compare_results` (all inside the modal builder); the
   `update_compare_choices` effect and the crude inline `compare_results` table
   are fully gone.
6. Run-diff page (`scenario_diff.py`) renders an identical config-diff table to
   before (its existing tests stay green); no visual baseline re-bless needed.
