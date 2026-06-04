# Compare Runs decouple-from-output-dir Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Results → Compare Runs tab usable without first loading an output directory, by populating its run selector from `default_run_history()` on entry to the Results tab and removing the four vestigial `out_dir` guards in its readers.

**Architecture:** All changes in `ui/pages/results.py` (+ tests in `tests/test_ui_results.py`). Extract a pure `_compare_run_choices` helper (testable); add a nav-triggered `@reactive.effect` that populates `compare_runs_select` from run history with a changed-only guard (no re-navigation flicker) and selection preservation; move the populate out of `_do_load_results`; delete the four dead `out_dir` guards in `comparison_chart` / `config_diff_table` / `run_delta_chart` / `run_delta_table`.

**Tech Stack:** Python 3.12, Shiny for Python 1.5.1 (`reactive.effect`, `reactive.Value`, `reactive.isolate`, `ui.update_selectize`), pytest, ruff. Tests: `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-06-04-compare-runs-decouple-output-dir-design.md` (reviewed clean over 2 in-loop rounds).

---

## Verified facts (audit — use exactly)

- `ui/pages/results.py`:
  - Module-level helper `_safe_output_dir(raw)` defined at ~:34 (KEEP — still used by `_load_results` ~:441 and `download_results_csv` ~:742).
  - `results_server(input, output, session, state)` opens at :321; reactive values declared at :322-325:
    ```python
        results_obj: reactive.Value = reactive.Value(None)
        results_data: reactive.Value[dict[str, pd.DataFrame]] = reactive.Value({})
        rep_dirs: reactive.Value[list[Path]] = reactive.Value([])
        _prev_output_dir: reactive.Value[str] = reactive.Value("")
    ```
  - `_do_load_results` populate block (the part to remove), currently at ~:395-404 (body
    indented 12 spaces):
    ```python
            ui.update_select("result_species", choices=species_choices)

            # Populate run comparison choices from history
            from osmose.history import default_run_history

            runs = default_run_history().list_runs()
            choices = {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}
            ui.update_selectize("compare_runs_select", choices=choices)

            ui.notification_show("Results loaded successfully.", type="message", duration=3)
    ```
  - `_auto_load_results` (~:451) is the precedent nav effect — `nav = input.main_nav(); if nav != "results": return`. `main_nav` is the navset id (`app.py:286 id="main_nav"`); `"results"` the panel value (`app.py:271 value="results"`).
  - The four Compare Runs readers and their EXACT vestigial guards:
    - `comparison_chart` ~:629-631:
      ```python
              out_dir = _safe_output_dir(input.output_dir())
              if out_dir is None:
                  return go.Figure().update_layout(title="Invalid output directory", template=tmpl)
      ```
    - `config_diff_table` ~:650-652:
      ```python
              out_dir = _safe_output_dir(input.output_dir())
              if out_dir is None:
                  return ui.div("Invalid output directory.")
      ```
    - `run_delta_chart` ~:692-694: identical to `comparison_chart`'s guard (go.Figure form).
    - `run_delta_table` ~:718-720: identical to `config_diff_table`'s guard (ui.div form).
    - In all four, `out_dir` is NOT referenced after the guard (records come from `default_run_history()`).
  - `update_selectize` (Shiny 1.5.1) accepts `choices=` and `selected=` (a `list[str]` of keys); `selected=[]` clears.
- `tests/test_ui_results.py` exists; tests the page's pure helpers + has a page-build smoke `test_results_ui_builds`. Run baseline: `.venv/bin/python -m pytest tests/test_ui_results.py -q` → 16 passed (pre-change).
- CI "lint" runs `ruff check` + `ruff format --check` on `osmose/ ui/ tests/` — so `ui/pages/results.py` AND `tests/test_ui_results.py` must be ruff-clean.

## File Structure

- Modify: `ui/pages/results.py` — `_compare_run_choices` helper (module level); `_last_compare_choices` reactive value + `_populate_compare_runs` effect (in `results_server`); remove populate from `_do_load_results`; remove 4 guards.
- Modify: `tests/test_ui_results.py` — helper unit test; static-source regression guard; static-source wiring assertion.

---

## Task 1: Extract the `_compare_run_choices` helper

**Files:**
- Modify: `ui/pages/results.py`
- Test: `tests/test_ui_results.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ui_results.py`:
```python
def test_compare_run_choices_builds_label_map():
    import types

    from ui.pages.results import _compare_run_choices

    runs = [
        types.SimpleNamespace(timestamp="2026-06-03T12:00:00", duration_sec=42.0),
        types.SimpleNamespace(timestamp="2026-06-04T08:30:15", duration_sec=7.4),
    ]
    assert _compare_run_choices(runs) == {
        "2026-06-03T12:00:00": "2026-06-03T12:00:00 (42s)",
        "2026-06-04T08:30:15": "2026-06-04T08:30:15 (7s)",
    }


def test_compare_run_choices_empty():
    from ui.pages.results import _compare_run_choices

    assert _compare_run_choices([]) == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k compare_run_choices -v`
Expected: FAIL (`ImportError: cannot import name '_compare_run_choices'`).

- [ ] **Step 3: Implement the helper**

In `ui/pages/results.py`, add a module-level function immediately AFTER the `_safe_output_dir` function definition (keep it with the other module helpers, before `results_server`):
```python
def _compare_run_choices(runs) -> dict[str, str]:
    """Build the compare_runs_select choices dict from RunHistory.list_runs() records.

    {timestamp: "<first 19 chars of timestamp> (<duration>s)"} — the exact label
    format the Compare Runs selector has always used.
    """
    return {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k compare_run_choices -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/results.py tests/test_ui_results.py
git -C /home/razinka/osmose/osmose-python commit -m "refactor(ui): extract _compare_run_choices helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Remove the four vestigial `out_dir` guards

**Files:**
- Modify: `ui/pages/results.py`
- Test: `tests/test_ui_results.py`

- [ ] **Step 1: Write the failing regression test**

Append to `tests/test_ui_results.py`:
```python
def test_compare_runs_readers_have_no_output_dir_guard():
    """The 4 Compare Runs readers must not gate on input.output_dir() — they read
    run history from default_run_history(), not the active output dir. Asserts the
    two exact removed guard-return forms are gone (the surviving _load_results
    notification at ~:444 uses a different string and is intentionally untouched)."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py"
    ).read_text()
    assert src.count('return go.Figure().update_layout(title="Invalid output directory"') == 0
    assert src.count('ui.div("Invalid output directory.")') == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k no_output_dir_guard -v`
Expected: FAIL (counts are 2 and 2 today, not 0).

- [ ] **Step 3: Remove the four guards**

In `ui/pages/results.py`, delete the guard block (the two lines: the `out_dir = ...` assignment and the `if out_dir is None: return ...` it precedes — 3 lines total each) from each of the four readers. Exact deletions:

In `comparison_chart` (~:629-631), delete:
```python
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return go.Figure().update_layout(title="Invalid output directory", template=tmpl)
```
In `config_diff_table` (~:650-652), delete:
```python
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return ui.div("Invalid output directory.")
```
In `run_delta_chart` (~:692-694), delete:
```python
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return go.Figure().update_layout(title="Invalid output directory", template=tmpl)
```
In `run_delta_table` (~:718-720), delete:
```python
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return ui.div("Invalid output directory.")
```
Leave EVERYTHING else in each function unchanged (the `selected`-count guard above, the `default_run_history()` / `_delta_for_selected` calls and their `try/except` below). Do NOT touch `results_chart`, `diet_chart`, `download_results_csv`, or `_load_results` — those keep their output-dir handling.

- [ ] **Step 4: Run to verify it passes + page still imports/builds**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k "no_output_dir_guard or results_ui_builds" -v`
Expected: PASS (regression test + the page-build smoke).
Run: `.venv/bin/python -c "import ui.pages.results; print('import ok')"`
Expected: `import ok`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/results.py tests/test_ui_results.py
git -C /home/razinka/osmose/osmose-python commit -m "fix(ui): drop vestigial output_dir guards from Compare Runs readers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Populate the selector from a nav-triggered effect (decouple from load)

**Files:**
- Modify: `ui/pages/results.py`
- Test: `tests/test_ui_results.py`

- [ ] **Step 1: Write the failing wiring test**

Append to `tests/test_ui_results.py`:
```python
def test_compare_runs_selector_populated_independently_of_output_dir():
    """The selector is populated by a nav-triggered effect reading run history,
    not by _do_load_results. Assert the new wiring exists and the old populate
    block (its distinctive comment) is gone from _do_load_results."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py"
    ).read_text()
    assert "_populate_compare_runs" in src
    assert "_last_compare_choices" in src
    assert "Populate run comparison choices from history" not in src
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -k populated_independently -v`
Expected: FAIL (`_populate_compare_runs`/`_last_compare_choices` not present yet; the old comment still present).

- [ ] **Step 3a: Add the `_last_compare_choices` reactive value**

In `ui/pages/results.py`, in `results_server`, add the new reactive value right after `_prev_output_dir` (~:325):
```python
    _prev_output_dir: reactive.Value[str] = reactive.Value("")
    _last_compare_choices: reactive.Value[dict[str, str]] = reactive.Value({})
```

- [ ] **Step 3b: Remove the populate block from `_do_load_results`**

Delete these lines (currently ~:397-402), leaving the `ui.update_select("result_species", ...)` line above and the `ui.notification_show("Results loaded successfully." ...)` line below intact:
```python

            # Populate run comparison choices from history
            from osmose.history import default_run_history

            runs = default_run_history().list_runs()
            choices = {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}
            ui.update_selectize("compare_runs_select", choices=choices)
```
After this, `_do_load_results` populates only `result_species` (the `compare_runs_select` populate now lives in the new effect — Step 3c).

- [ ] **Step 3c: Add the nav-triggered populate effect**

Add this effect in `results_server` immediately AFTER the `_auto_load_results` effect (~:462, before `_reset_results_loaded`):
```python
    @reactive.effect
    def _populate_compare_runs():
        """Populate the Compare Runs selector from run history whenever the user is on
        the Results tab — independent of any loaded output dir. Changed-only guard avoids
        re-tearing-down the widget (and clobbering the selection) on every re-navigation;
        selection is preserved across a real refresh (a newly recorded run)."""
        if input.main_nav() != "results":
            return
        from osmose.history import default_run_history

        try:
            runs = default_run_history().list_runs()
        except Exception:  # noqa: BLE001 — never crash the page on a history-read error
            return
        choices = _compare_run_choices(runs)
        with reactive.isolate():
            if choices == _last_compare_choices.get():
                return
            current = input.compare_runs_select()
        _last_compare_choices.set(choices)
        keep = [ts for ts in (current or ()) if ts in choices]
        ui.update_selectize("compare_runs_select", choices=choices, selected=keep)
```

- [ ] **Step 4: Run the wiring test + full file tests + import**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v`
Expected: PASS (all — the wiring test, the Task-1/2 tests, and the existing page tests incl. `test_results_ui_builds`).
Run: `.venv/bin/python -c "import ui.pages.results; print('import ok')"`
Expected: `import ok`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/results.py tests/test_ui_results.py
git -C /home/razinka/osmose/osmose-python commit -m "fix(ui): populate Compare Runs selector from run history on Results-tab entry

Decouples the selector from loading an output dir (nav-triggered effect with a
changed-only guard + selection preservation); removes the load-gated populate
from _do_load_results.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Lint, full verification, manual UI run-through

**Files:** none (verification only) — plus a CHANGELOG note.

- [ ] **Step 1: Lint (must be clean — CI gate)**

Run: `.venv/bin/ruff check ui/pages/results.py tests/test_ui_results.py`
Run: `.venv/bin/ruff format --check ui/pages/results.py tests/test_ui_results.py`
Expected: both clean. If `ruff format --check` flags either file, run `.venv/bin/ruff format <file>` and re-run the tests.

- [ ] **Step 2: Broader regression check**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -q` (report count; expect the prior 16 + 4 new = 20 passed — the 4 new tests are 2 in Task 1, 1 in Task 2, 1 in Task 3).
Run: `.venv/bin/python -m pytest tests/ -k "results or ui" -q` (report pass/fail; classify any FAILURE as pre-existing/unrelated vs caused — if unsure, say so rather than asserting unrelated).

- [ ] **Step 3: Manual UI run-through (render fns/effects aren't unit-tested)**

Launch without loading an output dir and confirm Compare Runs works off history alone:
```bash
PYTHONPATH=/home/razinka/osmose/osmose-python timeout 30 .venv/bin/shiny run app.py --host 127.0.0.1 --port 8767
```
Then in a browser (or via a quick curl for HTTP 200 + a Playwright check if available): open the app, do NOT type/load an output directory, click the **Results** tab → **Compare Runs** sub-tab. Confirm: the "Select runs to compare" selector lists the committed run records (from `data/history`); selecting 2 renders the comparison chart, the config diff, and the output-delta table — with NO "Invalid output directory" message. If launching a browser is impractical, at minimum confirm HTTP 200 on the running app and rely on the import + page-build smoke + the static wiring tests; state which path was used.

- [ ] **Step 4: CHANGELOG note**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Fixed` (the section the run-history fix already uses), add:
```markdown
- **ui (Compare Runs):** the Results → Compare Runs tab now works without first loading
  an output directory — its run selector populates from the canonical run history
  (`data/history`) on entry to the Results tab, and the readers no longer require a
  loaded output dir. Previously the tab showed "Invalid output directory" / an empty
  selector until an unrelated output dir was loaded.
```

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add CHANGELOG.md
git -C /home/razinka/osmose/osmose-python commit -m "docs(changelog): Compare Runs works without a loaded output dir

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** remove 4 vestigial guards → Task 2; `_compare_run_choices` helper → Task 1; nav-triggered populate effect with changed-only guard + selection preservation + `_last_compare_choices` reactive value → Task 3; remove populate from `_do_load_results` (single writer, disjoint from `result_species`) → Task 3 Step 3b; helper unit test + empty case → Task 1; static regression guard (exact two removed forms == 0) → Task 2; wiring assertion → Task 3; manual run-through + lint + CHANGELOG → Task 4; out-of-scope items (rest of page's output-dir dependence, live refresh, history-format) → not in plan, per spec. ✅

**Placeholder scan:** no TBD/TODO; every code step shows exact before/after text taken from the audited source; the manual run-through gives a concrete launch command + an explicit fallback. ✅

**Type consistency:** `_compare_run_choices(runs) -> dict[str, str]` defined in Task 1 and used in Task 3's effect; `_last_compare_choices: reactive.Value[dict[str, str]]` declared in Task 3 Step 3a and read/written in Step 3c; the effect keys on `input.main_nav() == "results"` (matching the `_auto_load_results` precedent and `app.py`'s `main_nav`/`"results"`). `selected=keep` (a `list[str]`) matches Shiny 1.5.1's `update_selectize` signature. ✅
