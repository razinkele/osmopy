# Decouple the Compare Runs tab from output_dir — Design

**Date:** 2026-06-04
**Status:** Approved direction (brainstormed; codebase-grounded). Bug fix / UX.

## The bug

The Results → **Compare Runs** tab is unusable unless an (unrelated) output directory has
been loaded, even though — after the run-history canonical-dir fix
(`docs/superpowers/specs/2026-06-03-run-history-canonical-dir-design.md`) — it reads run
history from `default_run_history()` and no longer needs the output dir at all. Two layers of
residual coupling cause this:

1. **Vestigial guards.** All four Compare Runs render functions in `ui/pages/results.py`
   (`comparison_chart`, `config_diff_table`, `run_delta_chart`, `run_delta_table`) compute
   `out_dir = _safe_output_dir(input.output_dir())` and early-return "Invalid output
   directory" when it is `None` — but `out_dir` is never used afterward (the records come from
   `default_run_history()`). The final review of the run-history fix flagged these as dead.
2. **Selector population is gated on loading results.** `compare_runs_select`'s choices are
   populated *only* inside `_do_load_results` (`ui/pages/results.py` ~:399-402), which runs on
   `btn_load_results` or the auto-load when navigating to Results *with* a loaded output dir.
   So with no output dir loaded the selector is empty — the deeper cause of the tab being
   unreachable.

## Verified context (audit)

- `ui/pages/results.py` `results_server(input, output, session, state)`:
  - `_do_load_results(out_dir)` (~:356) populates `result_species` AND `compare_runs_select`
    (~:399-402: `runs = default_run_history().list_runs(); choices = {r.timestamp:
    f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}; ui.update_selectize(
    "compare_runs_select", choices=choices)`).
  - Triggers: `_load_results` on `input.btn_load_results` (~:438); `_auto_load_results` on
    `input.main_nav() == "results"` with a loaded+unloaded dir (~:451).
  - The four readers each have the `out_dir = _safe_output_dir(input.output_dir()); if out_dir
    is None: return <invalid-dir result>` block (chart returns a `go.Figure` titled "Invalid
    output directory"; the `@render.ui` ones return `ui.div("Invalid output directory.")`).
  - `compare_runs_select` is defined in `results_ui()` (~:287) with `choices={}, multiple=True`.
- `_safe_output_dir` stays — still used by `_load_results` (~:441) and `download_results_csv`
  (~:742) directly (the results/diet charts go through the already-loaded `results_obj`). It
  must not be deleted.
- Render functions and `@reactive.effect`s are NOT unit-tested by project convention; the
  page's pure helpers are (`tests/test_ui_results.py`). `default_run_history().list_runs()`
  returns `[]` on an empty/absent history (it mkdir-creates the dir and globs).

## Approach

All changes in `ui/pages/results.py` (+ one test file). No new modules.

### 1. Remove the four vestigial `out_dir` guards

Delete the `out_dir = _safe_output_dir(input.output_dir())` + `if out_dir is None: return
<invalid-dir>` lines from `comparison_chart`, `config_diff_table`, `run_delta_chart`,
`run_delta_table`. The remaining guards (the `selected`-count check and the existing
try/except around `load_run`/`compare_runs_multi`/`_delta_for_selected`) fully cover the real
failure modes. Do NOT touch the output-dir guards in the genuinely output-dir-dependent
readers (`results_chart`, `diet_chart`, `download_results_csv`, etc.).

### 2. Populate the selector independently (single nav-triggered effect)

- **Extract** a pure helper near the other module helpers:
  ```python
  def _compare_run_choices(runs) -> dict[str, str]:
      """Build the compare_runs_select choices dict from RunHistory.list_runs() records."""
      return {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}
  ```
- **Add** a `_last_compare_choices = reactive.Value({})` near the other module-level reactive
  values in `results_server`, and a dedicated effect that populates the selector whenever the
  user enters the Results tab, independent of any output dir, **but only re-pushes when the
  choices actually changed** (avoids re-`update_selectize` flicker / momentary selection
  clobber on every re-navigation — see Honest limitations):
  ```python
  @reactive.effect
  def _populate_compare_runs():
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
              return  # nothing new — don't tear down the widget / clobber selection
          current = input.compare_runs_select()
      _last_compare_choices.set(choices)
      keep = [ts for ts in (current or ()) if ts in choices]
      ui.update_selectize("compare_runs_select", choices=choices, selected=keep)
  ```
  - `input.main_nav()` is the only reactive dependency (it's read outside `isolate()`); both
    `_last_compare_choices` and `input.compare_runs_select()` are read under `reactive.isolate()`
    so the effect does NOT re-fire on a selection change or on its own `update_selectize` write
    (confirmed no reactive loop in Shiny 1.5.1).
  - The changed-only guard makes re-entering the Results tab a no-op when no new run was
    recorded — preserving the widget's DOM state (today's run-once behavior) — while still
    picking up a newly-saved run (its timestamp changes the dict).
  - `selected=keep` preserves the user's current picks across a real re-populate (history only
    grows, so prior selections stay valid); `keep == []` is fine.
- **Remove** the `compare_runs_select` populate block from `_do_load_results` (the
  `result_species` populate there stays). `_populate_compare_runs` is then the SOLE writer of
  `compare_runs_select` (no double-populate). It and `_auto_load_results` both key on
  `input.main_nav() == "results"` and run in the same flush, but they touch DISJOINT widgets
  (`compare_runs_select` vs `result_species`), so their (non-deterministic) order is
  irrelevant.
- **Trigger reality (corrected):** the selector populates whenever the user *enters* the
  Results tab — output-dir-independent. There is **no** auto-navigation to Results after a run
  (`_handle_result` in `run.py` stays on the Run tab; `run.py`'s only `update_navset` targets
  the in-page `run_engine_tabs` sub-tab, not `main_nav`). A just-finished run therefore appears
  the next time the user navigates to Results (which is the natural action to view results). We
  deliberately do **not** add auto-navigation — it would pull the user off the Run console
  mid-review.

## Error handling

- History read error → the effect returns early, leaving the selector as-is (empty on first
  load); the readers already degrade to "No run history found." / "Select runs to compare".
- Empty history → `choices == {}` → selector empty → readers show their "Select runs…"
  prompt. No crash.
- A stale selected timestamp removed from history → dropped from `keep` (and the readers'
  try/except already handles a load miss).

## Testing

- `tests/test_ui_results.py`:
  - `_compare_run_choices` on a list of fake records (e.g. `types.SimpleNamespace(timestamp=
    "2026-06-03T12:00:00", duration_sec=42.0)`) → asserts the exact `{ts: "2026-06-03T12:00:00
    (42s)"}` mapping; and `[]` → `{}`.
  - A static-source regression guard. Real counts today: `"Invalid output directory"` appears
    **5×** in `ui/pages/results.py` — once in the `_load_results` notification (`~:444`, string
    `"Invalid output directory: must be inside the working directory."` — KEEPS) and once in
    each of the four Compare Runs guards being removed. A bare `grep -c` is fragile (the
    surviving notification shares the prefix). Use a form that targets ONLY the removed guards:
    read the source and assert both removed guard-return forms are gone —
    `src.count('return go.Figure().update_layout(title="Invalid output directory"') == 0` (the
    two chart guards) **and** `src.count('ui.div("Invalid output directory.")') == 0` (the two
    `@render.ui` guards). These two exact strings exist only in the four removed guards, so the
    surviving `:444` notification is untouched. (A function-slice assertion over the four reader
    bodies is an acceptable alternative; the two-exact-string form is simplest and robust.)
  - Optionally also assert the new `_populate_compare_runs` effect is wired:
    `"_populate_compare_runs" in src` and `"_last_compare_choices" in src`.
- Render fns/effects not unit-tested (convention) → a manual UI run-through: launch the app,
  do NOT load an output dir, open Results → Compare Runs, confirm the selector lists the
  committed run records and that selecting 2 renders the comparison chart + config diff +
  output-delta. Confirm the page still builds (`test_results_ui_builds`).

## Scope / YAGNI

- **In:** remove the 4 vestigial guards; the nav-triggered populate effect + the extracted
  choices helper; the tests; the manual run-through.
- **Out:** decoupling the rest of the Results page from `output_dir` (the results/diet charts
  and CSV download genuinely need it); auto-refresh of the selector while sitting on the tab
  (nav-triggered is sufficient); any history-format or `osmose/history.py` change; the
  separate pre-existing UX of the output-dir text box itself.

## Honest limitations

- The selector refreshes when you enter the Results tab, not live while you sit on it — a run
  finished elsewhere won't appear until you re-navigate. Acceptable: runs are launched from the
  Run tab in the same session, and viewing them means navigating to Results anyway (which fires
  the populate). The changed-only guard makes that re-entry cheap (no widget rebuild unless a
  new run was actually recorded).
- Compare Runs still lives under the Results page (same tab group); this fixes its data
  dependence, not its placement.

## Delivery

Single small PR: `ui/pages/results.py` (remove 4 guards, move/replace the populate, add the
helper) + `tests/test_ui_results.py` (helper test + regression guard) + a manual UI
run-through. No engine changes, no calibration runs.
