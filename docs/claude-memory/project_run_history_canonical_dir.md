---
name: project_run_history_canonical_dir
description: Run-history canonical-dir bug fix (writer/reader dir mismatch that made the Compare Runs tab dead) — SHIPPED to origin/master 2026-06-04 via subagent-driven dev.
metadata:
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

Fixed the run-history directory mismatch that made the **entire Results "Compare Runs" tab non-functional** (selector, config-diff, comparison chart, AND the just-shipped output-delta section). Merged fast-forward to master + **pushed to origin/master 2026-06-04** (`110b011..77c1324`, branch `fix/run-history-canonical-dir` deleted, origin synced).

## The bug
Writer `ui/pages/run.py` saved to `RunHistory(Path("data/history"))` (CWD-relative); the 5 readers in `ui/pages/results.py` read from `out_dir.parent / ".osmose_history"` — a different base AND leaf that never coincided, and `.osmose_history` was never created. The 6 real run records (2026-05-17) live in `data/history/`. Untested because `tests/test_history.py` used `tmp_path`. (This was the "pre-existing history-dir mismatch follow-on" flagged in [[project_result_delta_tracking]].)

## The fix (Option C — mirror calibration/history.py)
- `osmose/history.py`: `_PROJECT_ROOT = Path(__file__).resolve().parent.parent` (**2 parents** — one shallower than calibration's 3), `RUN_HISTORY_DIR = _PROJECT_ROOT/"data"/"history"`, `default_run_history() -> RunHistory` that reads the constant **at call time** (so monkeypatchable).
- Writer + all 5 readers swapped to `default_run_history()`; dropped the `.osmose_history` derivation + the `.is_dir()` guards (helper's `__init__` mkdirs); kept the `out_dir`/`_safe_output_dir` guards (preserves load-a-run-first UX).
- 6 reader sites across **3 shapes** (the review-critical decomposition): Shape A `comparison_chart`/`config_diff_table` got a NEW `try/except` because dropping `.is_dir()` newly exposed `FileNotFoundError`/`compare_runs_multi` raise on a stale selection; Shape B `run_delta_chart`/`run_delta_table` already had `except Exception`; Shape C `_do_load_results` positive-guard selector populate.
- Static wiring guard test + CHANGELOG `[Unreleased]/Fixed` note.

## Built subagent-driven; the review caught two MAJORs the happy path hid
THREE in-loop plan-review rounds before building, then per-task spec+quality review + a final integration review. Findings worth carrying:
- **R1 (HIGH+MAJOR):** a naive find/replace would miss the inline-comprehension reader shapes (Shape B); and dropping `.is_dir()` left an unguarded `load_run` → added the Shape-A try/except.
- **R2 (MAJOR — the key one):** making the writer path **absolute** silently defeats `tests/test_run_result_failure_invalidates_state.py::test_handle_result_success_sets_output_dir`, which used `monkeypatch.chdir(tmp_path)` *specifically* to keep `history.save()` out of the real `data/history/`. Fix: repoint that test to `monkeypatch.setattr("osmose.history.RUN_HISTORY_DIR", tmp_path)` + update the `.gitignore:6-9` comment built around it. **Lesson: when you change a property of a unit (relative→absolute), the risk lives wherever something LEANED ON the old property — usually a test — found only by grepping who depends on the seam you're moving.** R1 (which only read the reader/writer sites) could not have caught it.

## Status / follow-ons
- 3 implementation commits: `ba9cc28` (constant+helper), `b0b96cc` (writer+readers+test+gitignore), `77c1324` (wiring test+CHANGELOG). 38 focused tests pass; 609 pass in `-k "history or results or ui or run"` with ONE pre-existing/unrelated failure (`test_tutorial_3species` `ModuleNotFoundError` — editable-install/PYTHONPATH artifact, byte-identical+failing at base).
- **Open follow-on (Minor, deferred):** the `out_dir` guard is now vestigial for the 4 Compare-Runs readers — a user with valid run history still sees "Invalid output directory" if the active output-dir field is unset. Decoupling the selector from `output_dir` is a separate UX nicety (was explicitly out-of-scope per spec).
- **NEXT: pick a fresh backlog item.**

See [[project_result_delta_tracking]] (the delta UI this unblocks), [[feedback_in_loop_review_pattern]].
