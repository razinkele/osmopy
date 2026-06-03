# Run-History Canonical Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Run tab (writer) and the Results Compare Runs tab (5 readers) use one canonical run-history directory, so saved runs actually appear in the comparison selector.

**Architecture:** Add a `RUN_HISTORY_DIR` constant + `default_run_history()` helper to `osmose/history.py` (mirroring `osmose/calibration/history.py`'s `HISTORY_DIR` pattern), then point the single writer (`ui/pages/run.py`) and the 5 readers (`ui/pages/results.py`) at it. The helper's `RunHistory.__init__` already creates the dir, so the readers' `.is_dir()` guard is dropped; the `out_dir` guards stay (current "load a run first" UX preserved). Zero data migration — the existing records already live in `data/history`.

**Tech Stack:** Python 3.12, pytest, ruff. Tests: `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-06-03-run-history-canonical-dir-design.md`.

---

## Verified facts (audit — use exactly)

- `osmose/history.py`: imports `from pathlib import Path` (line 8); `RunHistory.__init__(self, history_dir)` does `self.history_dir = Path(history_dir); self.history_dir.mkdir(parents=True, exist_ok=True)`. `save(record)`, `list_runs()`, `load_run(ts)` all use `self.history_dir`. No canonical-dir constant exists.
- Pattern to mirror — `osmose/calibration/history.py:15-16`: `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` (3 parents, it's nested one deeper); `HISTORY_DIR = _PROJECT_ROOT / "data" / "calibration_history"`. For `osmose/history.py` (one level shallower) it's **`Path(__file__).resolve().parent.parent`** (osmose/history.py → osmose/ → repo root).
- Writer: `ui/pages/run.py:403-412` — `from osmose.history import RunRecord, RunHistory`; `history = RunHistory(Path("data/history"))`; `history.save(record)`; wrapped in `try/except (OSError, ValueError)`.
- Readers: `ui/pages/results.py`, 5 sites, each with `history_dir = out_dir.parent / ".osmose_history"` (the unique anchor string per site) then `if not history_dir.is_dir(): return <no history>` then `RunHistory(history_dir)`: `_do_load_results` (selector populate, ~:400), `comparison_chart` (~:635), `config_diff_table` (~:657), `run_delta_chart` (~:700), `run_delta_table` (~:729). Line numbers approximate — anchor on the `".osmose_history"` string.
- `tests/test_history.py` uses `RunHistory(tmp_path)`; no test asserts the real dir.

## File Structure

- Modify: `osmose/history.py` — add `_PROJECT_ROOT`, `RUN_HISTORY_DIR`, `default_run_history()`.
- Modify: `ui/pages/run.py` — writer uses `default_run_history()`.
- Modify: `ui/pages/results.py` — 5 readers use `default_run_history()` / `RUN_HISTORY_DIR`.
- Modify: `tests/test_history.py` — regression tests (canonical dir + writer/reader round-trip).

---

## Task 1: `RUN_HISTORY_DIR` + `default_run_history()` in `osmose/history.py`

**Files:**
- Modify: `osmose/history.py`
- Test: `tests/test_history.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_history.py` (it already imports `RunHistory`/`RunRecord` — confirm with `grep -n "^from\|^import" tests/test_history.py`; add `from osmose.history import RUN_HISTORY_DIR, default_run_history` to the imports it uses):

```python
from pathlib import Path
from osmose.history import RUN_HISTORY_DIR, default_run_history, RunHistory, RunRecord


def test_run_history_dir_is_canonical():
    assert isinstance(RUN_HISTORY_DIR, Path)
    assert RUN_HISTORY_DIR.is_absolute()
    assert RUN_HISTORY_DIR.parts[-2:] == ("data", "history")
    # repo-rooted: <repo>/data/history (repo root = osmose/history.py's parent.parent)
    assert RUN_HISTORY_DIR == Path(__file__).resolve().parent.parent / "data" / "history"

def test_default_run_history_uses_canonical_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("osmose.history.RUN_HISTORY_DIR", tmp_path)
    h = default_run_history()
    assert h.history_dir == tmp_path

def test_writer_reader_roundtrip_same_dir(monkeypatch, tmp_path):
    # The whole point: a record saved via one default_run_history() is found by another.
    monkeypatch.setattr("osmose.history.RUN_HISTORY_DIR", tmp_path)
    rec = RunRecord(timestamp="2026-06-03T12:00:00", output_dir="/x/output", summary={})
    default_run_history().save(rec)                       # writer side
    found = default_run_history().list_runs()             # reader side (fresh instance)
    assert [r.timestamp for r in found] == ["2026-06-03T12:00:00"]
    assert default_run_history().load_run("2026-06-03T12:00:00").output_dir == "/x/output"
```

(The `test_run_history_dir_is_canonical` assertion `Path(__file__).resolve().parent.parent` is from the TEST file's location — if `tests/` is at repo root, `tests/test_history.py`'s `parent.parent` is the repo root too. Confirm `tests/` sits directly under the repo root; if not, adjust that one assertion to compare against the repo root however the test suite already references it, OR just assert `RUN_HISTORY_DIR.parts[-2:] == ("data","history")` and `RUN_HISTORY_DIR.is_absolute()` and drop the exact-equality line.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_history.py -k "canonical or default_run_history or roundtrip" -v`
Expected: FAIL (`cannot import name 'RUN_HISTORY_DIR'`).

- [ ] **Step 3: Implement in `osmose/history.py`**

After the imports + `_log` (around line 12), add the constant:

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # osmose/history.py -> osmose/ -> repo root
RUN_HISTORY_DIR = _PROJECT_ROOT / "data" / "history"
```

At the END of the file (after the `RunHistory` class, which `default_run_history` references), add:

```python
def default_run_history() -> RunHistory:
    """The canonical RunHistory shared by the Run tab (writer) and the Results Compare Runs
    tab (readers), so saved runs are always found. Mirrors calibration/history.py's HISTORY_DIR.
    Reads RUN_HISTORY_DIR at call time (so tests can monkeypatch it)."""
    return RunHistory(RUN_HISTORY_DIR)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_history.py -v`
Expected: PASS (new tests + the file's existing tests). `.venv/bin/ruff check osmose/history.py tests/test_history.py`.

- [ ] **Step 5: Commit**

```bash
git add osmose/history.py tests/test_history.py
git commit -m "feat(history): RUN_HISTORY_DIR constant + default_run_history() helper"
```

---

## Task 2: Point the writer + 5 readers at the canonical dir

**Files:**
- Modify: `ui/pages/run.py`
- Modify: `ui/pages/results.py`

- [ ] **Step 1: Writer — `ui/pages/run.py`**

Change the import + construction (lines ~403, ~405). Replace:
```python
            from osmose.history import RunRecord, RunHistory

            history = RunHistory(Path("data/history"))
```
with:
```python
            from osmose.history import RunRecord, default_run_history

            history = default_run_history()
```
(If `Path` becomes unused in run.py after this, ruff will flag it — check `grep -n "Path(" ui/pages/run.py`; if it's used elsewhere, leave the import; if this was its only use, remove `Path` from run.py's imports to keep ruff clean.)

- [ ] **Step 2: Readers — `ui/pages/results.py` (all 5 sites)**

In EACH of the 5 reader sites (`_do_load_results`, `comparison_chart`, `config_diff_table`, `run_delta_chart`, `run_delta_table` — find them all with `grep -n '.osmose_history' ui/pages/results.py`), replace the history-dir derivation + existence guard + RunHistory construction. The current shape per render-fn site is:
```python
        from osmose.history import RunHistory
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return <invalid-output-dir result>
        history_dir = out_dir.parent / ".osmose_history"
        if not history_dir.is_dir():
            return <no-history result>
        history = RunHistory(history_dir)
```
Replace with (keep the `out_dir` guard — preserves "load a run first" UX; drop the `.osmose_history` derivation + the `.is_dir()` guard since `default_run_history()` creates the dir):
```python
        from osmose.history import default_run_history
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return <invalid-output-dir result>
        history = default_run_history()
```
For `_do_load_results` (the selector-population site, ~:398-405), the current code only populates `compare_runs_select` choices when `history_dir.is_dir()`. Replace the `history_dir = out_dir.parent / ".osmose_history"` + `if history_dir.is_dir():` gate with `default_run_history()` and populate from its `list_runs()` (empty list ⇒ empty selector, which is fine). Concretely, change:
```python
        history_dir = out_dir.parent / ".osmose_history"
        if history_dir.is_dir():
            runs = RunHistory(history_dir).list_runs()
            ...populate compare_runs_select...
```
to:
```python
        runs = default_run_history().list_runs()
        ...populate compare_runs_select... (same body; runs may be empty)
```
(Read the exact `_do_load_results` block first — `sed -n '395,410p' ui/pages/results.py` — and preserve its choice-dict construction; only swap the dir source. Keep the import style consistent with the other sites: `from osmose.history import default_run_history`.)

- [ ] **Step 3: Verify import + page builds + no stale `.osmose_history`**

Run: `.venv/bin/python -c "import ui.pages.run, ui.pages.results; print('import ok')"`
Run: `grep -n ".osmose_history" ui/pages/results.py` → expected: NO matches (all 5 removed).
Run: `grep -n "default_run_history" ui/pages/run.py ui/pages/results.py` → expected: 1 in run.py + 5 in results.py.
Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v` (the page still builds + the delta UI tests still pass; the `test_results_ui_builds` smoke must stay green).
Run: `.venv/bin/ruff check ui/pages/run.py ui/pages/results.py && .venv/bin/ruff format --check ui/`.

- [ ] **Step 4: Commit**

```bash
git add ui/pages/run.py ui/pages/results.py
git commit -m "fix(ui): Run tab + Compare Runs use canonical run-history dir (default_run_history)"
```

---

## Task 3: Wiring regression test + docs + finalize

**Files:**
- Modify: `tests/test_history.py` (wiring assertion)
- Modify: a doc/CHANGELOG note (find via `grep -rln "Compare Runs\|run history\|RunHistory" docs/ | head`)

- [ ] **Step 1: Wiring regression test (writer & readers can't silently diverge again)**

Add to `tests/test_history.py` a static-source assertion that both modules use the canonical helper (cheap, catches a future regression to a hardcoded path):

```python
def test_run_and_results_use_default_run_history():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    run_src = (repo / "ui" / "pages" / "run.py").read_text()
    res_src = (repo / "ui" / "pages" / "results.py").read_text()
    assert "default_run_history" in run_src
    assert "default_run_history" in res_src
    # the old mismatched paths must be gone
    assert 'RunHistory(Path("data/history"))' not in run_src
    assert ".osmose_history" not in res_src
```

Run: `.venv/bin/python -m pytest tests/test_history.py -k "use_default_run_history" -v` (PASS).

- [ ] **Step 2: Docs/CHANGELOG note**

Add a one-line note (in the doc found above, or `docs/baltic_example.md` near the Compare Runs / run-history discussion, or a CHANGELOG if one exists): "Fixed: Run-tab history and the Results Compare Runs selector now share a canonical directory (`data/history` via `osmose.history.RUN_HISTORY_DIR` / `default_run_history()`); previously the writer (`data/history`) and reader (`.osmose_history`) diverged so saved runs never appeared in the comparison selector."

- [ ] **Step 3: Full verification + lint**

Run: `.venv/bin/python -m pytest tests/test_history.py tests/test_ui_results.py -v` (all pass; report counts).
Run: `.venv/bin/python -m pytest tests/ -k "history or results or ui" -q` (no regressions; note pre-existing).
Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/` (clean on touched files).
Run (optional live check, if feasible): `timeout 25 .venv/bin/shiny run app.py --host 127.0.0.1 --port 8766` + curl → confirm HTTP 200 (the page still loads). If launching is impractical, rely on the import + build-smoke + helper tests; state which path used.

- [ ] **Step 4: Commit + finish**

```bash
git add tests/test_history.py docs/
git commit -m "test(history)+docs: lock writer==reader run-history dir invariant"
```

Use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** `RUN_HISTORY_DIR` + `default_run_history()` (mirroring calibration `HISTORY_DIR`) → T1; writer swap (`run.py`) → T2 Step1; 5 reader swaps (`results.py`, drop `.osmose_history` + `.is_dir()`, keep `out_dir` guard) → T2 Step2; zero migration (uses existing `data/history`) → inherent; regression tests (canonical dir, writer/reader round-trip, wiring) → T1 + T3 Step1; docs note → T3 Step2; out-of-scope (selector/out_dir decoupling, JSON format, calibration history) → not in plan, per spec. ✅

**Placeholder scan:** no TBD/TODO; every code step has before/after. The `_do_load_results` step says "read the exact block first + preserve the choice-dict body" with a `sed` to get it — a grounded instruction (the block's exact choice-dict construction wasn't captured in the audit), not a placeholder; the transformation (swap dir source to `default_run_history().list_runs()`) is fully specified. The test's exact-equality assertion has a documented fallback if `tests/` isn't directly under repo root. ✅

**Type consistency:** `RUN_HISTORY_DIR` (Path) + `default_run_history() -> RunHistory` used identically across T1 (def), T2 (writer + 5 readers), T3 (tests). `default_run_history()` returns a `RunHistory` whose `.save`/`.list_runs`/`.load_run` are the existing API. The `_PROJECT_ROOT` parent-count (2 for `osmose/history.py`) is distinct from calibration's (3) — verified against file depth. ✅
