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
- `_safe_output_dir` stays — still used by `_do_load_results`, `download_results_csv`,
  `results_chart`, the diet/timeseries readers, etc.
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
- **Add** a dedicated effect in `results_server` that populates the selector whenever the user
  is on the Results tab, independent of any output dir:
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
          current = input.compare_runs_select()
      keep = [ts for ts in (current or ()) if ts in choices]
      ui.update_selectize("compare_runs_select", choices=choices, selected=keep)
  ```
  `selected=keep` preserves the user's current picks across a re-populate (history only grows,
  so prior selections stay valid); reading `current` under `reactive.isolate()` avoids making
  the effect re-fire on every selection change.
- **Remove** the `compare_runs_select` populate block from `_do_load_results` (the species
  populate there stays). The nav-triggered effect is now the single populate path. A
  just-finished run still appears: the post-run auto-load flips `main_nav` to "results", which
  fires the effect.

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
  - A static-source regression guard: the four Compare Runs reader functions' bodies no longer
    contain `"Invalid output directory"` (read `ui/pages/results.py` source, assert the count
    of that string dropped to only the genuinely-output-dir-dependent readers — or assert the
    four reader names are not followed by an output-dir guard; simplest robust form: assert
    `results.py` no longer early-returns "Invalid output directory" from the Compare Runs
    readers by checking the total occurrence count equals the known post-fix count).
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

- The selector refreshes on entering the Results tab, not live while you sit on it — a run
  finished in another browser session won't appear until you re-navigate. Acceptable: runs are
  launched from the Run tab in the same session, and the post-run auto-load re-navigates.
- Compare Runs still lives under the Results page (same tab group); this fixes its data
  dependence, not its placement.

## Delivery

Single small PR: `ui/pages/results.py` (remove 4 guards, move/replace the populate, add the
helper) + `tests/test_ui_results.py` (helper test + regression guard) + a manual UI
run-through. No engine changes, no calibration runs.
