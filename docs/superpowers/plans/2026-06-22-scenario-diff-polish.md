# Scenario-Diff Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Scenarios-page "Compare Scenarios" feature to the run-diff page's quality (badged/sorted/scrollable table) inside a modal, by extracting one shared diff-table component both surfaces consume.

**Architecture:** A new pure `ui/components/config_diff.py` (classify + render) becomes the single source of truth for config-diff presentation. The run-diff page delegates to it (deleting its private duplicate). The Scenarios page replaces its crude inline compare card with a "Compare Scenarios" button → modal that uses the shared component, a tagged-union reactive state for edge messages, and a guarded `ParamDiff→dict` adapter.

**Tech Stack:** Python 3.12+, Shiny for Python 1.6.3, htmltools (`ui.tags`), pytest, Playwright (e2e), ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-06-22-scenario-diff-polish-design.md`

## Global Constraints

- Use `.venv/bin/python` (NOT `python`) for all pytest/python invocations.
- Ruff line length 100; lint = BOTH `ruff check` AND `ruff format --check` on `osmose/ ui/ tests/`.
- pyright must stay clean on changed files (run against the `[dev]` venv).
- Default pytest excludes e2e (`-m 'not e2e'`); run e2e explicitly with `-m e2e`.
- e2e files MUST be named `test_e2e_*.py` (conftest collect-ignore) and dismiss the changelog modal before nav (`tests._e2e_support.dismiss_changelog_modal`).
- Broad degrade-don't-crash guards use `except Exception:  # noqa: BLE001`.
- Commit after each task (frequent commits).

---

### Task 1: Shared config-diff component (pure)

**Files:**
- Create: `ui/components/config_diff.py`
- Test: `tests/test_config_diff_component.py`

**Interfaces:**
- Consumes: `ui.styles.STYLE_MONO_KEY`, `ui.styles.STYLE_SCROLL_TABLE` (existing).
- Produces:
  - `classify_config_diffs(diffs: list[dict[str, str | None]]) -> list[dict[str, str | None]]` — tags each `{key, value_a, value_b}` with `change ∈ {added, removed, changed}`, sorted changed→added→removed then alphabetical by key. Pure.
  - `render_config_diff_table(diffs: list[dict[str, str | None]])` -> a Shiny UI element (count line + badged/sorted/scrollable Key/A/B/Change table). Callers pass only NON-empty diff lists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_diff_component.py`:

```python
"""Unit tests for the shared config-diff component (classify + render)."""

from ui.components.config_diff import classify_config_diffs, render_config_diff_table


def test_changed_when_both_present_and_differ():
    out = classify_config_diffs([{"key": "a", "value_a": "1", "value_b": "2"}])
    assert out[0]["change"] == "changed"


def test_added_when_value_a_none():
    out = classify_config_diffs([{"key": "a", "value_a": None, "value_b": "2"}])
    assert out[0]["change"] == "added"


def test_removed_when_value_b_none():
    out = classify_config_diffs([{"key": "a", "value_a": "1", "value_b": None}])
    assert out[0]["change"] == "removed"


def test_empty_string_is_changed_not_added_or_removed():
    out = classify_config_diffs([{"key": "a", "value_a": "", "value_b": "x"}])
    assert out[0]["change"] == "changed"


def test_sort_changed_added_removed_then_alpha():
    diffs = [
        {"key": "z_changed", "value_a": "1", "value_b": "2"},
        {"key": "a_removed", "value_a": "1", "value_b": None},
        {"key": "m_added", "value_a": None, "value_b": "2"},
        {"key": "a_changed", "value_a": "1", "value_b": "2"},
    ]
    out = classify_config_diffs(diffs)
    assert [r["key"] for r in out] == ["a_changed", "z_changed", "m_added", "a_removed"]


def test_deterministic_regardless_of_input_order():
    diffs = [
        {"key": "b", "value_a": "1", "value_b": "2"},
        {"key": "a", "value_a": "1", "value_b": "2"},
    ]
    assert classify_config_diffs(diffs) == classify_config_diffs(list(reversed(diffs)))


def test_mixed_case_keys_sort_case_sensitively():
    # Python str sort is case-sensitive: uppercase 'S' (83) sorts before lowercase 's' (115).
    diffs = [
        {"key": "species.linf", "value_a": "1", "value_b": "2"},
        {"key": "Species.Linf", "value_a": "1", "value_b": "2"},
    ]
    out = classify_config_diffs(diffs)
    assert [r["key"] for r in out] == ["Species.Linf", "species.linf"]


def test_empty_input():
    assert classify_config_diffs([]) == []


def test_render_returns_count_and_table():
    html = str(
        render_config_diff_table(
            [{"key": "species.linf.sp0", "value_a": "0.5", "value_b": "0.7"}]
        )
    )
    assert "1 differing config key" in html
    assert "<table" in html
    assert "species.linf.sp0" in html


def test_render_none_cell_shows_exactly_one_dash():
    html = str(render_config_diff_table([{"key": "a", "value_a": None, "value_b": "x"}]))
    assert html.count("—") == 1  # exactly the one None cell, nothing else


def test_render_empty_string_cell_is_empty_not_dash():
    # value_a="" must render an EMPTY cell (<td></td>), NOT the em-dash reserved
    # for None. Assert the positive structure AND zero dashes (a strong pin, not
    # a global "no dash anywhere" negative).
    html = str(render_config_diff_table([{"key": "a", "value_a": "", "value_b": "x"}]))
    assert "<td></td>" in html
    assert html.count("—") == 0


def test_render_large_diff_builds_without_error():
    diffs = [{"key": f"k{i:04d}", "value_a": str(i), "value_b": str(i + 1)} for i in range(300)]
    html = str(render_config_diff_table(diffs))
    assert "300 differing config keys" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_diff_component.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.components.config_diff'`.

- [ ] **Step 3: Create the component**

Create `ui/components/config_diff.py`:

```python
"""Shared config-diff presentation: classify + render a Key/A/B/Change table.

Single source of truth for what a config diff looks like, consumed by both the
Scenario Diff run-comparison page and the Scenarios page's compare modal. Pure:
takes [{key, value_a, value_b}] dicts, returns classified rows / a Shiny UI
element. No reactive, no I/O. Callers own the empty/edge cases (the wording
differs per surface) and pass only NON-empty diff lists to the renderer.
"""

from __future__ import annotations

from shiny import ui

from ui.styles import STYLE_MONO_KEY, STYLE_SCROLL_TABLE

_CHANGE_ORDER = {"changed": 0, "added": 1, "removed": 2}


def classify_config_diffs(
    diffs: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    """Tag each {key, value_a, value_b} row with a change type and sort.

    change is "added"   when value_a is None (key only in B),
              "removed" when value_b is None (key only in A),
              "changed" otherwise (both present, differ — incl. an empty-string
              value, since only None means a missing key).
    Sorted changed-group-first, then added, then removed; alphabetical by key
    within each group. Deterministic and independent of input order. Pure.
    """
    rows: list[dict[str, str | None]] = []
    for d in diffs:
        va = d.get("value_a")
        vb = d.get("value_b")
        if va is None:
            change = "added"
        elif vb is None:
            change = "removed"
        else:
            change = "changed"
        rows.append({"key": d["key"], "value_a": va, "value_b": vb, "change": change})
    rows.sort(key=lambda r: (_CHANGE_ORDER[r["change"]], r["key"]))
    return rows


def render_config_diff_table(diffs: list[dict[str, str | None]]):
    """Classify raw diff dicts and return a count line + badged, sorted,
    scrollable Key/A/B/Change table. Pass only NON-empty diff lists."""
    rows = classify_config_diffs(diffs)
    n = len(rows)
    badge_cls = {"changed": "bg-secondary", "added": "bg-success", "removed": "bg-danger"}

    def _val_cell(v):
        return ui.tags.td("—" if v is None else v)

    body = [
        ui.tags.tr(
            ui.tags.td(r["key"], style=STYLE_MONO_KEY),
            _val_cell(r["value_a"]),
            _val_cell(r["value_b"]),
            ui.tags.td(ui.tags.span(r["change"], class_=f"badge {badge_cls[r['change']]}")),
        )
        for r in rows
    ]
    table = ui.tags.table(
        ui.tags.thead(
            ui.tags.tr(
                ui.tags.th("Key"),
                ui.tags.th("A"),
                ui.tags.th("B"),
                ui.tags.th("Change"),
            )
        ),
        ui.tags.tbody(*body),
        class_="table table-sm table-striped",
        style="font-size: 13px;",
    )
    return ui.div(
        ui.p(f"{n} differing config key{'s' if n != 1 else ''}", class_="text-muted"),
        ui.div(table, style=STYLE_SCROLL_TABLE),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_diff_component.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Lint + types**

Run: `.venv/bin/ruff check ui/components/config_diff.py tests/test_config_diff_component.py && .venv/bin/ruff format --check ui/components/config_diff.py tests/test_config_diff_component.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add ui/components/config_diff.py tests/test_config_diff_component.py
git commit -m "feat(config-diff): shared classify + render component"
```

---

### Task 2: Delegate the run-diff page to the shared component

**Files:**
- Modify: `ui/pages/scenario_diff.py` (remove `_CHANGE_ORDER` ~line 35, `_classify_config_diffs` ~38-60, the table-build block in `diff_config_table` ~301-333, and the `STYLE_MONO_KEY`/`STYLE_SCROLL_TABLE` import line 27)
- Modify: `tests/test_scenario_diff_config.py` (repoint import + rename 8 call sites)

**Interfaces:**
- Consumes: `ui.components.config_diff.render_config_diff_table` (Task 1).
- Produces: no new symbols. `diff_config_table` renders an identical structure to before.

- [ ] **Step 1: Repoint the existing classify test to the new module**

In `tests/test_scenario_diff_config.py`, change line 5 from:

```python
from ui.pages.scenario_diff import _classify_config_diffs
```

to:

```python
from ui.components.config_diff import classify_config_diffs
```

Then, **with the import line already repointed above**, rename the 8 call sites (lines ~9, 14, 19, 25, 36, 49, 54, 58): `_classify_config_diffs(` → `classify_config_diffs(`. ORDER MATTERS — fix the import FIRST, then a find-replace of the remaining bare token `_classify_config_diffs` → `classify_config_diffs` is safe (the import no longer contains the old token, so it won't be rewritten to the wrong module). Do NOT blanket-replace before fixing the import, or you'd produce `from ui.pages.scenario_diff import classify_config_diffs` — a deleted symbol → ImportError.

- [ ] **Step 2: Run the repointed test (it should pass immediately)**

Run: `.venv/bin/python -m pytest tests/test_scenario_diff_config.py -q`
Expected: PASS — Task 1 already created `classify_config_diffs`, so the repointed import resolves and the 8 cases pass unchanged. (This is a refactor, not a new behavior, so there is no red phase here.) If it FAILS with `ImportError`/`NameError`, the rename missed a call site — fix before proceeding.

- [ ] **Step 3: Edit `ui/pages/scenario_diff.py` — delegate render**

(a) Remove the style import at line 27:

```python
from ui.styles import STYLE_MONO_KEY, STYLE_SCROLL_TABLE
```

(b) Add an import (group with the other `from ui...` / `from osmose...` imports near the top, e.g. after line 26):

```python
from ui.components.config_diff import render_config_diff_table
```

(c) Delete the module-level `_CHANGE_ORDER` line (~35):

```python
_CHANGE_ORDER = {"changed": 0, "added": 1, "removed": 2}
```

(d) Delete the entire `_classify_config_diffs` function (~lines 38-60, the `def _classify_config_diffs(...)` through its `return rows`).

(e) In `diff_config_table`, replace ONLY the table-build block — everything from `rows = _classify_config_diffs(diffs)` (~line 301) through the final `return ui.div(...)` (~line 333) — with the single line `return render_config_diff_table(diffs)`. Leave every edge branch above it (including the `if not diffs:` "Identical configuration" return) exactly as-is. After the edit the tail of the function reads:

```python
        # ... (unchanged edge branches above: not-selected / same-run / load-error)
        if not diffs:
            return ui.p("Identical configuration — no differences.", class_="text-muted")

        return render_config_diff_table(diffs)
```

(The `if not diffs:` line is shown only for orientation — do NOT re-add it; it already exists.)

- [ ] **Step 4: Run the page + classify tests to verify green**

Run: `.venv/bin/python -m pytest tests/test_scenario_diff_config.py tests/test_ui_results.py -q`
Expected: PASS (the run-diff page renders the same structure; `test_ui_results.py` still finds `"diff_config_table"`).

- [ ] **Step 5: Lint + types**

Run: `.venv/bin/ruff check ui/pages/scenario_diff.py tests/test_scenario_diff_config.py && .venv/bin/ruff format --check ui/pages/scenario_diff.py tests/test_scenario_diff_config.py`
Expected: clean (this catches any leftover F401 from the removed style import).

Run: `.venv/bin/pyright ui/pages/scenario_diff.py`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/scenario_diff.py tests/test_scenario_diff_config.py
git commit -m "refactor(scenario-diff): delegate config table to shared component"
```

---

### Task 3: Scenarios page — button + modal, tagged-union state, guarded adapter

**Files:**
- Modify: `ui/pages/scenarios.py` (UI: add button, remove Compare card, reflow col_widths, drop `STYLE_DIFF_ROW` import; Server: pure `_resolve_compare_state`, modal, handler, render; delete `update_compare_choices` effect + old `compare_diffs`/`handle_compare`/`compare_results`)
- Modify: `ui/styles.py` (delete the now-dead `STYLE_DIFF_ROW` constant, line 30)
- Test: `tests/test_scenario_compare_state.py` (new — pure state-resolver)

**Interfaces:**
- Consumes: `ui.components.config_diff.render_config_diff_table` (Task 1); `ScenarioManager.compare` (returns `list[ParamDiff]` with `.key/.value_a/.value_b`); `_scenario_names()` (`ui/pages/scenarios.py:101-103`).
- Produces: module-level pure `_resolve_compare_state(name_a, name_b, compare) -> tuple[str, list[dict[str, str | None]] | None]` where the tag ∈ `{"none","same","identical","error","diffs"}` and the payload is `None` except for `"diffs"` (the adapted rows).

- [ ] **Step 1: Write the failing test for the pure state resolver**

Create `tests/test_scenario_compare_state.py`:

```python
"""Unit tests for the pure compare-state resolver used by the Scenarios modal."""

from ui.pages.scenarios import _resolve_compare_state


class _PD:
    """Stand-in for osmose.scenarios.ParamDiff (key/value_a/value_b)."""

    def __init__(self, key, a, b):
        self.key, self.value_a, self.value_b = key, a, b


def test_none_when_either_unselected():
    assert _resolve_compare_state("", "y", lambda a, b: [])[0] == "none"
    assert _resolve_compare_state("x", "", lambda a, b: [])[0] == "none"


def test_same_when_equal():
    assert _resolve_compare_state("x", "x", lambda a, b: [_PD("k", "1", "2")]) == ("same", None)


def test_identical_when_empty_diff():
    assert _resolve_compare_state("x", "y", lambda a, b: []) == ("identical", None)


def test_error_when_compare_raises():
    def boom(a, b):
        raise FileNotFoundError("deleted")

    assert _resolve_compare_state("x", "y", boom) == ("error", None)


def test_diffs_adapter_shape():
    tag, rows = _resolve_compare_state("x", "y", lambda a, b: [_PD("k", "1", "2")])
    assert tag == "diffs"
    assert rows == [{"key": "k", "value_a": "1", "value_b": "2"}]


def test_diffs_adapter_passes_none_through():
    # An added/removed key has a None side; the adapter must preserve it so
    # classify_config_diffs can later tag it added/removed (not coerce to "").
    tag, rows = _resolve_compare_state("x", "y", lambda a, b: [_PD("k", None, "5")])
    assert tag == "diffs"
    assert rows == [{"key": "k", "value_a": None, "value_b": "5"}]


def test_success_then_same_yields_no_stale_table():
    # The resolver is stateless, so a real diff followed by an a==b selection
    # returns ("same", None) with no leftover list — this is what keeps the
    # modal from flashing a stale table.
    diffs = [_PD("k", "1", "2")]
    assert _resolve_compare_state("x", "y", lambda a, b: diffs)[0] == "diffs"
    assert _resolve_compare_state("x", "x", lambda a, b: diffs) == ("same", None)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scenario_compare_state.py -q`
Expected: FAIL — `ImportError: cannot import name '_resolve_compare_state'`.

- [ ] **Step 3: Add the pure resolver to `ui/pages/scenarios.py`**

Add this module-level function near the top of `ui/pages/scenarios.py` (after the imports, before `scenarios_ui`):

```python
def _resolve_compare_state(name_a, name_b, compare):
    """Classify a scenario A/B selection into a tagged-union state for the
    compare modal. `compare` is ScenarioManager.compare (or a stub).

    Returns ("none"|"same"|"identical"|"error", None) or ("diffs", rows) where
    rows are the run-diff dict shape [{key, value_a, value_b}]. Pure."""
    if not name_a or not name_b:
        return ("none", None)
    if name_a == name_b:
        return ("same", None)
    try:
        diffs = compare(name_a, name_b)
    except Exception:  # noqa: BLE001 — missing/corrupt/deleted scenario: degrade
        return ("error", None)
    if not diffs:
        return ("identical", None)
    rows = [{"key": d.key, "value_a": d.value_a, "value_b": d.value_b} for d in diffs]
    return ("diffs", rows)
```

- [ ] **Step 4: Run the resolver test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scenario_compare_state.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Add the component import**

In `ui/pages/scenarios.py`, add to the import block (after the existing `from ui.components.collapsible import ...` line):

```python
from ui.components.config_diff import render_config_diff_table
```

- [ ] **Step 6: UI — add the button, remove the Compare card, drop the unused import**

(a) Change the style import (line 26) from:

```python
from ui.styles import STYLE_DIFF_ROW, STYLE_EMPTY
```

to:

```python
from ui.styles import STYLE_EMPTY
```

Then delete the now-orphaned constant definition `STYLE_DIFF_ROW = ...` at `ui/styles.py:30` — this was its only consumer. First confirm with `grep -rn "STYLE_DIFF_ROW" osmose/ ui/ tests/`: after the import change, the only remaining hit should be the definition line itself; delete it. (It's a module constant, not an import, so leaving it would not fail ruff — but it's dead code, so remove it.)

(b) In `scenarios_ui`, add the Compare button right after the `btn_new_scenario` button (line ~34):

```python
        ui.input_action_button("btn_new_scenario", "+ New Scenario", class_="btn-success mb-3"),
        ui.input_action_button(
            "btn_compare_open", "Compare Scenarios", class_="btn-warning mb-3"
        ),
```

(c) Remove the entire **Right: Compare** card from the `ui.layout_columns(...)` (the `ui.card(ui.card_header("Compare Scenarios"), ...)` block, ~lines 62-70 including its `compare_a`/`compare_b`/`btn_compare`/`compare_results` widgets).

(d) Change the layout's `col_widths=[3, 5, 4]` (~line 71) to two columns:

```python
            col_widths=[4, 8],
```

- [ ] **Step 7: Server — replace the inline compare with modal + handler + render**

(a) Delete the `update_compare_choices` effect entirely (`ui/pages/scenarios.py:481-487` — the `@reactive.effect` whose body calls `ui.update_select("compare_a", ...)` / `ui.update_select("compare_b", ...)`).

(b) Delete the old `compare_diffs = reactive.value([])` (~491), the old `handle_compare` effect (~493-502), and the old `compare_results` render (~504-532).

(c) Add the replacement block in the same region:

```python
    # --- Compare (modal) ---
    # Tagged-union state so untouched/same/identical/error/diffs never share a
    # stale list. ("none", None) is the untouched sentinel + reset-on-open value.
    compare_state: reactive.Value[tuple[str, list[dict[str, str | None]] | None]] = (
        reactive.value(("none", None))
    )

    @reactive.effect
    @reactive.event(input.btn_compare_open)
    def _compare_open():
        compare_state.set(("none", None))  # reset so no prior table flashes
        choices = _scenario_names()
        ui.modal_show(
            ui.modal(
                ui.input_select("compare_a", "Scenario A", choices=choices),
                ui.input_select("compare_b", "Scenario B", choices=choices),
                ui.input_action_button("btn_compare", "Compare", class_="btn-warning"),
                ui.hr(),
                ui.output_ui("compare_results"),
                title="Compare Scenarios",
                size="l",
                easy_close=True,
                footer=ui.tags.button(
                    "Close", class_="btn btn-secondary", **{"data-bs-dismiss": "modal"}
                ),
            )
        )

    @reactive.effect
    @reactive.event(input.btn_compare)
    def handle_compare():
        compare_state.set(
            _resolve_compare_state(input.compare_a(), input.compare_b(), mgr.compare)
        )

    @render.ui
    def compare_results():
        tag, payload = compare_state.get()
        # payload is the adapted rows list ONLY for the "diffs" tag; this guard
        # also narrows the type for pyright (None elsewhere).
        if payload is not None:
            return render_config_diff_table(payload)
        messages = {
            "none": "Select two scenarios and click Compare.",
            "same": "Same scenario selected — no differences.",
            "identical": "Identical configuration — no differences.",
            "error": "One or both scenarios could not be loaded — they may have been deleted.",
        }
        return ui.div(messages[tag], style=STYLE_EMPTY)
```

- [ ] **Step 8: Grep for stale references / duplicate ids**

Run: `grep -n "compare_a\|compare_b\|btn_compare\|compare_results\|update_compare_choices\|STYLE_DIFF_ROW" ui/pages/scenarios.py`
Expected: each of `compare_a`, `compare_b`, `btn_compare`, `compare_results` appears **exactly once** (inside the modal builder / its render); `update_compare_choices` and `STYLE_DIFF_ROW` appear **zero** times.

- [ ] **Step 9: Run the broader scenarios + import tests**

Run: `.venv/bin/python -m pytest tests/test_scenario_compare_state.py tests/test_ui_scenarios.py -q`
Expected: PASS (existing model-layer `mgr.compare` test unaffected; new resolver test green).

- [ ] **Step 10: Lint + types**

Run: `.venv/bin/ruff check ui/pages/scenarios.py tests/test_scenario_compare_state.py && .venv/bin/ruff format --check ui/pages/scenarios.py tests/test_scenario_compare_state.py`
Expected: clean (catches the `STYLE_DIFF_ROW` F401 if missed).

Run: `.venv/bin/pyright ui/pages/scenarios.py`
Expected: no new errors.

- [ ] **Step 11: Commit**

```bash
git add ui/pages/scenarios.py ui/styles.py tests/test_scenario_compare_state.py
git commit -m "feat(scenarios): compare in a modal with shared diff table + edge states"
```

---

### Task 4: e2e — the compare modal renders a diff

**Files:**
- Create: `tests/test_e2e_scenario_compare.py`

**Interfaces:**
- Consumes: `osmose.scenarios.Scenario`/`ScenarioManager` (seed two scenarios); `tests._e2e_support.dismiss_changelog_modal`; the `btn_compare_open`/`compare_a`/`compare_b`/`btn_compare` ids from Task 3.

- [ ] **Step 1: Write the e2e test**

Create `tests/test_e2e_scenario_compare.py`:

```python
"""End-to-end test for the Compare Scenarios modal.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_scenario_compare.py -v -m e2e

Excluded from the default suite (`-m 'not e2e'`). The compare edge logic is
covered purely by tests/test_scenario_compare_state.py; this asserts the modal
opens and renders the shared badged diff table end to end.
"""

import shutil
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from osmose.scenarios import Scenario, ScenarioManager
from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_SCENARIOS_DIR = Path("data/scenarios")  # state.scenarios_dir default (ui/state.py:35)


def test_compare_modal_renders_diff(page: Page, app: ShinyAppProc):
    # Seed two scenarios differing in exactly one key (deterministic + fast).
    mgr = ScenarioManager(_SCENARIOS_DIR)
    name_a = f"e2e_cmp_a_{uuid.uuid4().hex[:8]}"
    name_b = f"e2e_cmp_b_{uuid.uuid4().hex[:8]}"
    try:
        mgr.save(Scenario(name=name_a, config={"species.linf.sp0": "50.0"}))
        mgr.save(Scenario(name=name_b, config={"species.linf.sp0": "70.0"}))

        page.goto(app.url)
        page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
        dismiss_changelog_modal(page)
        page.locator(".nav-pills .nav-link[data-value='scenarios']").click()
        page.wait_for_selector("#btn_compare_open", timeout=_LOAD_TIMEOUT)

        page.click("#btn_compare_open")
        # Scope locators to the modal — a bare #compare_a could match a stale node.
        modal = page.locator(".modal")
        modal.locator("#compare_a").wait_for(timeout=_LOAD_TIMEOUT)
        page.select_option(".modal #compare_a", name_a)
        page.select_option(".modal #compare_b", name_b)
        page.click(".modal #btn_compare")

        expect(modal.locator("table").first).to_be_visible(timeout=_LOAD_TIMEOUT)
        expect(modal.locator(".badge")).to_have_count(1)
        expect(modal).to_contain_text("1 differing config key")
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name_a, ignore_errors=True)
        shutil.rmtree(_SCENARIOS_DIR / name_b, ignore_errors=True)
```

- [ ] **Step 2: Run the e2e test**

Run: `.venv/bin/python -m pytest tests/test_e2e_scenario_compare.py -v -m e2e`
Expected: PASS (1 passed). If Playwright browsers are missing, run `.venv/bin/python -m playwright install chromium` first.

- [ ] **Step 3: Lint**

Run: `.venv/bin/ruff check tests/test_e2e_scenario_compare.py && .venv/bin/ruff format --check tests/test_e2e_scenario_compare.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_scenario_compare.py
git commit -m "test(e2e): compare-scenarios modal renders a diff"
```

---

### Final verification (before finishing the branch)

- [ ] Full suite (default, no e2e): `.venv/bin/python -m pytest -n auto -q` → green.
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` and `.venv/bin/ruff format --check osmose/ ui/ tests/` → clean.
- [ ] `.venv/bin/pyright ui/components/config_diff.py ui/pages/scenario_diff.py ui/pages/scenarios.py` → no new errors.
- [ ] e2e explicitly: `.venv/bin/python -m pytest tests/test_e2e_scenario_compare.py -m e2e -q` → pass.
- [ ] Manual grep: `grep -rn "_classify_config_diffs\|update_compare_choices\|STYLE_DIFF_ROW" ui/ tests/` → no matches (old symbols fully gone; the repointed test uses `classify_config_diffs`).
- [ ] No visual-baseline re-bless needed (nav unchanged; only a page-body button + a modal). Do NOT re-bless.

## Spec coverage map

- Shared pure component (classify + render) → Task 1.
- Run-diff page delegates; private duplicate + style imports removed → Task 2.
- Test repoint (`test_scenario_diff_config.py`) → Task 2.
- Button + modal (size="l", easy_close=True, reset-on-open) → Task 3 (Steps 6-7).
- Tagged-union state (none/same/identical/error/diffs) + error guard + literal-comprehension adapter → Task 3 (`_resolve_compare_state`).
- Delete `update_compare_choices`; fresh choices from `_scenario_names()` → Task 3 (Steps 6c, 7a).
- `STYLE_DIFF_ROW` F401 removal → Task 3 (Step 6a).
- Edge cases (none/same/identical/error, empty-string cell, large diff) → Task 1 render tests + Task 3 resolver tests.
- New Scenarios-page modal e2e with seeded state + modal-scoped locators → Task 4.
- Gates (ruff check+format, pyright, e2e explicit, no visual re-bless) → per-task lint steps + Final verification.
