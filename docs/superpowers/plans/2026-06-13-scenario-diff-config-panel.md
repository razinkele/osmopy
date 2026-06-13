# Scenario Diff Config-Diff Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Config differences" panel to the Scenario Diff tab that shows which config keys differ between run A and run B (changed / added / removed), reusing `history.compare_runs`.

**Architecture:** One module-level pure classifier (`_classify_config_diffs`) plus one bare `@render.ui` (`diff_config_table`) in `ui/pages/scenario_diff.py`, wired into the tab via a `ui.accordion_panel`. The render calls the existing `default_run_history().compare_runs(ts_a, ts_b)` (no new diff logic, no new data path). No reactive `Value`s, no NetCDF handles — config-only.

**Tech Stack:** Python 3.12, Shiny for Python (bare `@render.ui`, `ui.accordion`/`ui.tags`), pytest, Playwright (e2e). Lint/format: ruff. Types: pyright.

**Spec:** `docs/superpowers/specs/2026-06-13-scenario-diff-config-diff-design.md`

---

## File Structure

- **Modify** `ui/pages/scenario_diff.py`:
  - new module-level constant `_CHANGE_ORDER` and function `_classify_config_diffs` (Task 1)
  - new `diff_config_table` `@render.ui` inside `scenario_diff_server` (Task 2)
  - new `ui.accordion`/`ui.accordion_panel` in `scenario_diff_nav_panel()` (Task 2)
  - new import `from ui.styles import STYLE_MONO_KEY, STYLE_SCROLL_TABLE` (Task 2)
- **Create** `tests/test_scenario_diff_config.py` — unit tests for `_classify_config_diffs` (Task 1)
- **Modify** `tests/test_ui_results.py` — structure test for the wired panel (Task 2)
- **Modify** `tests/test_e2e_scenario_diff.py` — `config_snapshot` param on `_write_run` + new e2e test (Task 3)
- **Modify** `CHANGELOG.md` — Unreleased → Added entry (Task 3)

Per-task gate (every task): `.venv/bin/ruff check osmose/ ui/ tests/`, `.venv/bin/ruff format --check osmose/ ui/ tests/`, and `.venv/bin/pyright ui/pages/scenario_diff.py tests/test_scenario_diff_config.py` (plus any file the task touched).

---

### Task 1: Pure classifier `_classify_config_diffs`

**Files:**
- Create: `tests/test_scenario_diff_config.py`
- Modify: `ui/pages/scenario_diff.py` (add module-level constant + function, after the
  `_SPATIAL_VAR_HINT` constant near line 30)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scenario_diff_config.py`:

```python
"""Unit tests for _classify_config_diffs (Scenario Diff config-diff panel)."""

from __future__ import annotations

from ui.pages.scenario_diff import _classify_config_diffs


def test_changed_row_both_present():
    out = _classify_config_diffs([{"key": "a", "value_a": "1", "value_b": "2"}])
    assert out == [{"key": "a", "value_a": "1", "value_b": "2", "change": "changed"}]


def test_added_row_value_a_none():
    out = _classify_config_diffs([{"key": "a", "value_a": None, "value_b": "2"}])
    assert out[0]["change"] == "added"


def test_removed_row_value_b_none():
    out = _classify_config_diffs([{"key": "a", "value_a": "1", "value_b": None}])
    assert out[0]["change"] == "removed"


def test_empty_value_strings_are_changed_not_added_or_removed():
    # "" is a present value, not a missing key — only None drives added/removed.
    out = _classify_config_diffs([{"key": "a", "value_a": "", "value_b": "x"}])
    assert out[0]["change"] == "changed"


def test_sort_changed_then_added_then_removed_alpha_within_group():
    diffs = [
        {"key": "z_changed", "value_a": "1", "value_b": "2"},
        {"key": "a_added", "value_a": None, "value_b": "2"},
        {"key": "m_removed", "value_a": "1", "value_b": None},
        {"key": "a_changed", "value_a": "1", "value_b": "2"},
    ]
    out = _classify_config_diffs(diffs)
    assert [r["key"] for r in out] == ["a_changed", "z_changed", "a_added", "m_removed"]
    assert [r["change"] for r in out] == ["changed", "changed", "added", "removed"]


def test_order_independence_scrambled_input():
    # Deliberately scrambled (removed, changed, added; keys not alphabetical).
    scrambled = [
        {"key": "y_removed", "value_a": "1", "value_b": None},
        {"key": "b_changed", "value_a": "1", "value_b": "2"},
        {"key": "x_added", "value_a": None, "value_b": "2"},
        {"key": "a_changed", "value_a": "3", "value_b": "4"},
    ]
    out = _classify_config_diffs(scrambled)
    assert [r["key"] for r in out] == ["a_changed", "b_changed", "x_added", "y_removed"]


def test_empty_input_returns_empty_list():
    assert _classify_config_diffs([]) == []


def test_rows_preserve_key_and_values_verbatim():
    out = _classify_config_diffs([{"key": "species.linf.sp0", "value_a": "0.5", "value_b": "0.7"}])
    assert out[0]["key"] == "species.linf.sp0"
    assert out[0]["value_a"] == "0.5"
    assert out[0]["value_b"] == "0.7"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scenario_diff_config.py -v`
Expected: FAIL with `ImportError: cannot import name '_classify_config_diffs'`.

- [ ] **Step 3: Implement the classifier**

In `ui/pages/scenario_diff.py`, after the `_SPATIAL_VAR_HINT = ...` line (near line 30), add:

```python
# Display priority for the config-diff panel: changed group first, then added, then
# removed (NOT the change string's alphabetical order, which would put "added" first).
_CHANGE_ORDER = {"changed": 0, "added": 1, "removed": 2}


def _classify_config_diffs(diffs: list[dict]) -> list[dict]:
    """Tag each {key, value_a, value_b} row with a change type and sort for display.

    change is "added"   when value_a is None (key only in B),
              "removed" when value_b is None (key only in A),
              "changed" otherwise (both present, differ — incl. an empty-string value,
              since only None means a missing key).
    Sorted changed-group-first, then added, then removed; alphabetical by key within
    each group. Deterministic and independent of input order. Pure (no I/O).
    """
    rows: list[dict] = []
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scenario_diff_config.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → no errors.
Run: `.venv/bin/ruff format --check osmose/ ui/ tests/` → all files formatted (run `.venv/bin/ruff format osmose/ ui/ tests/` if not).
Run: `.venv/bin/pyright ui/pages/scenario_diff.py tests/test_scenario_diff_config.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/scenario_diff.py tests/test_scenario_diff_config.py
git commit -m "feat(scenario-diff): add _classify_config_diffs classifier"
```

---

### Task 2: Render function + accordion wiring + structure test

**Files:**
- Modify: `ui/pages/scenario_diff.py` (import; `diff_config_table` inside `scenario_diff_server`; accordion in `scenario_diff_nav_panel`)
- Modify: `tests/test_ui_results.py` (structure test)

- [ ] **Step 1: Write the failing structure test**

In `tests/test_ui_results.py`, after `test_scenario_diff_tab_wired_into_results` (ends ~line 318), add:

```python
def test_scenario_diff_config_panel_wired():
    """The config-diff panel (accordion + output) is emitted in the Scenario Diff tab body."""
    from ui.pages.scenario_diff import scenario_diff_nav_panel

    # str(NavPanel) is only a repr and .tagify() raises outside a navset; render the BODY.
    html = str(scenario_diff_nav_panel().content)
    assert "diff_config_table" in html
    assert "Config differences" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py::test_scenario_diff_config_panel_wired -v`
Expected: FAIL (the assertions on `diff_config_table` / `Config differences` fail — not yet wired).

- [ ] **Step 3: Add the styles import**

In `ui/pages/scenario_diff.py`, add this import alongside the other `from ui...` imports (near
line 26, after `from ui.pages.grid_helpers import make_diff_map, make_spatial_map`):

```python
from ui.styles import STYLE_MONO_KEY, STYLE_SCROLL_TABLE
```

- [ ] **Step 4: Wire the accordion into the nav panel**

In `scenario_diff_nav_panel()`, insert the accordion between the `ui.layout_columns(...)`
selector block and `output_widget("diff_biomass_chart")`. Change:

```python
            col_widths=[12],
        ),
        output_widget("diff_biomass_chart"),
```

to:

```python
            col_widths=[12],
        ),
        ui.accordion(
            ui.accordion_panel(
                "Config differences",
                ui.output_ui("diff_config_table"),
            ),
            id="diff_config_accordion",
            open=True,
        ),
        output_widget("diff_biomass_chart"),
```

- [ ] **Step 5: Add the render function inside `scenario_diff_server`**

In `ui/pages/scenario_diff.py`, inside `scenario_diff_server`, after the `diff_biomass_caption`
render function (ends ~line 245), add:

```python
    # ── Config differences (which config keys differ between A and B) ──
    @render.ui
    def diff_config_table():
        ts_a, ts_b = _safe(input.diff_run_a), _safe(input.diff_run_b)
        # Falsy guard first: an unselected/not-yet-ready selector is "" or None.
        if not ts_a or not ts_b:
            return ui.p("Select two runs to compare their configs.", class_="text-muted")
        if ts_a == ts_b:
            return ui.p("Same run selected — no config differences.", class_="text-muted")
        try:
            diffs = default_run_history().compare_runs(ts_a, ts_b)
        except Exception:  # noqa: BLE001 — stale/missing run file: degrade, don't crash the render
            return ui.p("Could not load run configs.", class_="text-muted")
        if not diffs:
            return ui.p("Identical configuration — no differences.", class_="text-muted")

        rows = _classify_config_diffs(diffs)
        n = len(rows)
        badge_cls = {"changed": "bg-secondary", "added": "bg-success", "removed": "bg-danger"}

        def _val_cell(v):
            return ui.tags.td("—" if v is None else v)

        body = [
            ui.tags.tr(
                ui.tags.td(r["key"], style=STYLE_MONO_KEY),
                _val_cell(r["value_a"]),
                _val_cell(r["value_b"]),
                ui.tags.td(
                    ui.tags.span(r["change"], class_=f"badge {badge_cls[r['change']]}")
                ),
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

- [ ] **Step 6: Run the structure test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py::test_scenario_diff_config_panel_wired -v`
Expected: PASS.

- [ ] **Step 7: Run the broader UI + classifier tests (no regressions)**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py tests/test_scenario_diff_config.py -v`
Expected: PASS (all, including the pre-existing Results tests).

- [ ] **Step 8: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → no errors.
Run: `.venv/bin/ruff format --check osmose/ ui/ tests/` → formatted.
Run: `.venv/bin/pyright ui/pages/scenario_diff.py tests/test_ui_results.py` → 0 errors.

- [ ] **Step 9: Commit**

```bash
git add ui/pages/scenario_diff.py tests/test_ui_results.py
git commit -m "feat(scenario-diff): render config-diff panel + wire accordion"
```

---

### Task 3: e2e coverage + CHANGELOG

**Files:**
- Modify: `tests/test_e2e_scenario_diff.py` (`config_snapshot` param on `_write_run` + fixture + test)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a `config_snapshot` param to the shared `_write_run` helper**

In `tests/test_e2e_scenario_diff.py`, change the signature and the record dict. Change:

```python
def _write_run(name: str, idx: int, *, base: float) -> tuple[Path, Path]:
```

to:

```python
def _write_run(
    name: str, idx: int, *, base: float, config_snapshot: dict | None = None
) -> tuple[Path, Path]:
```

and change the record's snapshot line:

```python
        "config_snapshot": {},
```

to:

```python
        "config_snapshot": config_snapshot or {},
```

(The two existing tests call `_write_run` without `config_snapshot`, so they keep `{}` — unchanged behaviour.)

- [ ] **Step 2: Add a fixture writing two runs with differing config snapshots**

After the existing `two_runs` fixture (ends ~line 80), add:

```python
@pytest.fixture
def two_runs_config():
    """Two runs whose config_snapshots differ: one changed, one A-only, one B-only key."""
    _HISTORY.mkdir(parents=True, exist_ok=True)
    cfg_a = {"predation.efficiency.sp0": "0.5", "mortality.natural.rate.sp0": "0.2"}
    cfg_b = {"predation.efficiency.sp0": "0.7", "movement.distance.sp0": "3"}
    created = [
        _write_run("runC", 3, base=100.0, config_snapshot=cfg_a),
        _write_run("runD", 4, base=130.0, config_snapshot=cfg_b),
    ]
    yield [ts for ts, _ in created]
    import shutil

    for rec_path, _ in created:
        rec_path.unlink(missing_ok=True)
    shutil.rmtree(_SUBSTRATE, ignore_errors=True)
```

- [ ] **Step 3: Add the e2e test asserting the config panel renders the diff**

At the end of `tests/test_e2e_scenario_diff.py`, add:

```python
def test_scenario_diff_config_panel_shows_differences(
    page: Page, app: ShinyAppProc, two_runs_config
):
    """The config-diff panel lists changed, added, and removed keys for two runs."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    page.locator(".nav-pills .nav-link[data-value='results']").click()
    page.get_by_role("tab", name="Scenario Diff").click()

    expect(page.locator("#diff_run_a option[value='2026-06-13T03:00:00']")).to_have_count(
        1, timeout=_LOAD_TIMEOUT
    )
    page.locator("#diff_run_a").select_option("2026-06-13T03:00:00")
    page.locator("#diff_run_b").select_option("2026-06-13T04:00:00")

    # Assert on CONTENT (the bare @render.ui div is zero-height until it recomputes, so
    # to_be_visible would flake). to_contain_text waits for the render to populate.
    cfg = page.locator("#diff_config_table")
    expect(cfg).to_contain_text("predation.efficiency.sp0", timeout=_LOAD_TIMEOUT)  # changed
    expect(cfg).to_contain_text("mortality.natural.rate.sp0", timeout=_LOAD_TIMEOUT)  # removed
    expect(cfg).to_contain_text("movement.distance.sp0", timeout=_LOAD_TIMEOUT)  # added

    page.screenshot(path=str(_REPO / "screenshots" / "scenario_diff_config_e2e.png"))
```

- [ ] **Step 4: Run the e2e tests (explicit -m e2e)**

Run: `.venv/bin/python -m pytest tests/test_e2e_scenario_diff.py -v -m e2e`
Expected: PASS (3 passed — the two existing tests still pass with the new `_write_run` signature, plus the new config-panel test). If the screenshots dir is missing, the new test's `page.screenshot` needs `screenshots/` to exist — it already does (the existing test writes there).

- [ ] **Step 5: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, add a new bullet immediately after the
existing `**ui (scenario diff):**` bullet:

```markdown
- **ui (scenario diff):** a "Config differences" panel on the Scenario Diff tab lists the config
  keys that differ between the two compared runs (changed / added / removed, with a badge per
  row), so the tab reads top-to-bottom as *what you changed → what it did*. Reuses
  `RunHistory.compare_runs`; collapsible (open by default) and shown above the biomass overlay.
```

- [ ] **Step 6: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → no errors.
Run: `.venv/bin/ruff format --check osmose/ ui/ tests/` → formatted.
Run: `.venv/bin/pyright tests/test_e2e_scenario_diff.py` → 0 errors.

- [ ] **Step 7: Commit**

```bash
git add tests/test_e2e_scenario_diff.py CHANGELOG.md
git commit -m "test(scenario-diff): e2e config-diff panel + CHANGELOG"
```

---

## Final verification (after all tasks)

- [ ] Full non-e2e suite green: `.venv/bin/python -m pytest -m 'not e2e' -n auto -q`
- [ ] e2e green: `.venv/bin/python -m pytest tests/test_e2e_scenario_diff.py -v -m e2e`
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` and `.venv/bin/ruff format --check osmose/ ui/ tests/` clean
- [ ] `.venv/bin/pyright` clean on all touched files
- [ ] Dispatch a final whole-implementation code review before finishing the branch.

## Self-Review (plan author)

- **Spec coverage:** classifier (Task 1) ↔ spec §Architecture.1; render fn + accordion + imports + empty-state order + styling (Task 2) ↔ §Architecture.2/.3 + §Reuse + §Error handling; unit/structure/e2e tests (Tasks 1–3) ↔ §Testing.1/.2/.3; CHANGELOG (Task 3) ↔ §Testing note; pyright in every gate ↔ §Testing note. No spec requirement left without a task.
- **Type consistency:** `_classify_config_diffs(diffs: list[dict]) -> list[dict]` used identically in Task 1 (definition + tests) and Task 2 (call site); row keys `key/value_a/value_b/change` consistent across classifier, render, and tests; `badge_cls` keys match `_CHANGE_ORDER` keys (`changed/added/removed`); `STYLE_MONO_KEY`/`STYLE_SCROLL_TABLE` imported in Task 2 before first use.
- **No placeholders:** every code step shows complete code; commands have expected output.
