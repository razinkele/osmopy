---
name: project_compare_runs_decouple
description: Compare Runs tab decoupled from output_dir (selector populates from run history on Results-tab entry; 4 vestigial guards removed). SHIPPED to origin/master 2026-06-04.
metadata:
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

The Results → **Compare Runs** tab now works WITHOUT first loading an output directory — the UX follow-on surfaced by the run-history canonical-dir fix ([[project_run_history_canonical_dir]]). Merged fast-forward to master + **pushed to origin/master 2026-06-04** (`c890e12..32347e4`, branch `fix/compare-runs-decouple-output-dir` deleted, origin synced). All in `ui/pages/results.py` + `tests/test_ui_results.py`.

## What it fixed (two layers of output_dir coupling)
1. **Vestigial guards:** the 4 Compare Runs readers (`comparison_chart`, `config_diff_table`, `run_delta_chart`, `run_delta_table`) computed `out_dir = _safe_output_dir(input.output_dir())` and early-returned "Invalid output directory" — but `out_dir` was unused (records come from `default_run_history()`). Removed all 4.
2. **Selector population gated on loading results:** `compare_runs_select` choices were populated ONLY inside `_do_load_results` (runs when an output dir loads). Moved to a nav-triggered `@reactive.effect _populate_compare_runs` that reads `default_run_history().list_runs()` whenever `input.main_nav() == "results"` — output-dir-independent. Extracted `_compare_run_choices(runs)` helper (testable).

## Key implementation facts
- The effect uses a **changed-only guard** (`_last_compare_choices` reactive.Value) + selection preservation (`selected=keep`): reads `input.main_nav()` OUTSIDE `reactive.isolate()` (the sole dependency), reads `_last_compare_choices.get()` + `input.compare_runs_select()` INSIDE isolate → no reactive loop with its own `update_selectize` write, and re-entering the tab doesn't tear down the widget / clobber selection unless a new run actually changed the choices.
- There is **NO auto-navigation to Results after a run** — `run.py`'s `update_navset` targets the in-page `run_engine_tabs` sub-tab, NOT `main_nav`. A just-finished run appears on the next manual entry to Results (which fires the effect). Deliberately did not add auto-nav (would pull the user off the Run console).
- selectize.js gotcha (cost a debug detour during runtime verification): server-populated options live in `el.selectize.options`, NOT the native `<select>.options` (which reads 0). Verify the selectize instance, not the underlying select.
- The 2 console errors seen on selection ("Error during model cleanup: Widget is not attached") are benign pre-existing shinywidgets/anywidget plotly-widget-teardown noise on re-render — orthogonal to this change.

## Process (full superpowers flow; 2 in-loop review rounds each on spec + plan)
brainstorm → spec → **spec in-loop review (2 executing reviewers)** found B1 (false "post-run auto-load flips main_nav" claim) + M1 (nav effect re-fire flicker → added the changed-only guard) → plan → **plan in-loop review (2 reviewers)** found the test-count typo (5→4) + a 16-vs-12-space indent inconsistency in a non-actionable quote → subagent-driven build (T1–T4, per-task spec+quality review) → final whole-impl review (READY TO MERGE) → **runtime-verified via Playwright** (selector populated with output_dir empty; 2 runs selected → readers render, NO "Invalid output directory"). 20 focused + 464 broad tests pass.

## Commits (4 impl + 4 doc)
`0fdf6c0` helper, `2958eee` remove guards, `7215809` populate effect, `32347e4` CHANGELOG; preceded by spec/plan/2×review-fix doc commits.

**Next: pick a fresh backlog item.** Render fns/effects aren't unit-tested by convention here → logic in the testable `_compare_run_choices`, behavior verified by static wiring tests + page-build smoke + the Playwright run-through.
