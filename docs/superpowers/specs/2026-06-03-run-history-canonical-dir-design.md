# Run-history canonical directory — Design

**Date:** 2026-06-03
**Status:** Approved direction (brainstormed; audit-grounded). Bug fix.

## The bug

The Run tab saves run history to one directory; the Results "Compare Runs" tab reads from a
different one, so the run selector never populates — making the Compare Runs config-diff,
comparison chart, and the just-shipped output-delta section all unreachable.

- **Writer:** `ui/pages/run.py:405` → `RunHistory(Path("data/history"))` (CWD-relative).
- **Readers:** `ui/pages/results.py` (5 sites: `_do_load_results` ~:400, `comparison_chart` ~:635,
  `config_diff_table` ~:657, `run_delta_chart` ~:700, `run_delta_table` ~:729) →
  `out_dir.parent / ".osmose_history"`.
- Different base AND leaf → can never coincide. `.osmose_history` has never been created; all real
  run records (6, from 2026-05-17) live in `data/history/`.
- `osmose/history.py` has **no** canonical-dir constant (the dir is always caller-supplied). The
  mismatch is **untested** (`tests/test_history.py` uses `tmp_path`), which is why it shipped.

## Verified context (audit)

- `osmose/history.py`: `RunHistory(history_dir)` (required arg; `mkdir(parents=True, exist_ok=True)`
  on construct), `RunHistory.save(record)` → `history_dir/f"run_{ts}.json"`, `list_runs()`,
  `load_run(ts)`. `RunRecord(timestamp, config_snapshot, duration_sec, output_dir, summary)`.
- The **calibration** history system (`osmose/calibration/history.py`) already does this correctly:
  `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent`; `HISTORY_DIR = _PROJECT_ROOT /
  "data" / "calibration_history"`; writer-dir == reader-dir. We mirror that.
- Only ONE writer (`run.py`) and the readers are all in `results.py`. No scripts read run history.
  No other consumers; tests use `tmp_path` (won't constrain the change).

## Approach — Option C (canonical constant + helper, mirroring calibration)

### `osmose/history.py`

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # osmose/history.py -> osmose/ -> repo root
RUN_HISTORY_DIR = _PROJECT_ROOT / "data" / "history"


def default_run_history() -> "RunHistory":
    """The canonical RunHistory both the Run tab (writer) and the Results Compare Runs tab
    (reader) use, so saved runs are always found. Mirrors calibration/history.py's HISTORY_DIR."""
    return RunHistory(RUN_HISTORY_DIR)
```
(`RunHistory.__init__` already `mkdir`s the dir, so `default_run_history()` creates `data/history`
if absent.)

### `ui/pages/run.py`

Replace `RunHistory(Path("data/history"))` (line ~405) with `default_run_history()` (import it from
`osmose.history`). Fixes the CWD-relative fragility too (now repo-rooted).

### `ui/pages/results.py` (5 read sites)

Replace each `history_dir = out_dir.parent / ".osmose_history"` + `RunHistory(history_dir)` with
`default_run_history()` (and use `RUN_HISTORY_DIR` where a `.is_dir()` existence check is needed).
**Keep the existing surrounding guards** (the `out_dir`/`_safe_output_dir` checks stay — minimal
behavior change; the current "load a run, then compare" UX is preserved). The history path no
longer depends on `out_dir`.

Note: the two readers that lacked a pre-existing `except` (`comparison_chart`, `config_diff_table`)
gain a `try/except` around their `load_run`/`compare_runs_multi` call when `.is_dir()` is dropped —
that guard previously shielded them from a `FileNotFoundError` on a stale selection. This keeps the
"degrade, don't crash" behavior and mirrors the guard the delta readers already have.

Note (test hygiene): the writer becomes CWD-independent (absolute `RUN_HISTORY_DIR`). One existing
test (`test_handle_result_success_sets_output_dir`) used `monkeypatch.chdir(tmp_path)` to keep the
`history.save()` side effect out of the real `data/history/`; it must switch to
`monkeypatch.setattr("osmose.history.RUN_HISTORY_DIR", tmp_path)` (and the `.gitignore` comment that
references it is updated accordingly).

### Migration

**None.** The 6 existing records are already in `data/history/`, which becomes canonical — they
light up immediately.

### Tests (`tests/test_history.py`)

- `test_default_run_history_dir`: `default_run_history().history_dir == RUN_HISTORY_DIR` and
  `RUN_HISTORY_DIR == Path(__file__-derived repo root)/"data"/"history"` (assert it ends with
  `data/history` and is absolute).
- `test_writer_reader_same_dir_roundtrip` (the regression lock): construct two independent
  `default_run_history()` instances; `save(RunRecord(...))` via the first; the second's
  `list_runs()`/`load_run(ts)` finds it — proving writer and reader resolve the same dir. Use
  `monkeypatch.setattr("osmose.history.RUN_HISTORY_DIR", tmp_path)` so the test doesn't write into
  the real `data/history` (and `default_run_history` must read the module attr at call time so the
  patch takes effect — it does, since it references the module-level name).
- A wiring assertion that `ui/pages/run.py` and `ui/pages/results.py` both import/use
  `default_run_history` (grep-style or import check), so the writer/reader can't silently diverge
  again.

## Scope / YAGNI

- **In:** the constant + helper, the 1 writer + 5 reader swaps, the regression tests, a one-line
  docstring/comment.
- **Out:** decoupling the Compare Runs selector from `output_dir` (a separate UX nicety — readers
  keep their existing `out_dir` guard); changing the per-run JSON format; any calibration-history
  change (a separate, already-correct system); fixing the broader run-history-vs-output-dir UX.

## Honest limitations

- Fixes the *path*; the selector still populates only after an output dir is loaded (current UX),
  now finding history in the canonical `data/history`.
- `RUN_HISTORY_DIR` is repo-rooted (`_PROJECT_ROOT/data/history`), matching the calibration system;
  it assumes the app runs from the repo (the project's convention). Not a deployment-portability
  change — same assumption the calibration history already makes.

## Delivery

Single small PR: `osmose/history.py` (constant + helper), `ui/pages/run.py` (1 line),
`ui/pages/results.py` (5 sites), `tests/test_history.py` (regression tests), a doc/CHANGELOG note.
No engine changes, no calibration runs, no data migration.
