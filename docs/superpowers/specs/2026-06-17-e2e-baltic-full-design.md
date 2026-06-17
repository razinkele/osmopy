# Full Baltic end-to-end Playwright test — design

**Date:** 2026-06-17
**Status:** approved (brainstorming), pending implementation plan

## Problem & goal

The unit suite never feeds a *real* Baltic engine run into the Results graphical
renderers — that gap is exactly how the diet-heatmap "(no prey data)" bug shipped (it
was caught only by an ad-hoc manual Playwright Baltic run, never a committed test). There
is a committed `test_e2e_live_movement.py` that loads Baltic and runs the Python engine 1
year with live movement, but it stops at run completion and asserts nothing on Results.

**Goal:** one committed full-journey e2e — load Baltic → Python run (+live movement) →
assert the key Results graphical outputs render with real data — so a regression in any of
those surfaces is caught by CI's e2e leg.

## Decisions (from brainstorming)

1. **Results coverage = key graphical outputs:** biomass time-series chart, the Diet
   Composition heatmap (explicit regression guard against the "(no prey data)" bug), and a
   spatial map. Not a broad every-tab sweep (brittle, mostly unit-covered); not minimal.
2. **Spatial output:** Baltic ships `output.spatial.enabled;false`, so force it on via a
   run override (`output.spatial.enabled=true`). Baltic has real ocean cells (unlike
   `minimal`, whose spatial is all-NaN), so this yields genuine, renderable spatial data
   and exercises the spatial-render path (which has had real bugs).
3. **One comprehensive test** alongside (not replacing) `test_e2e_live_movement.py`, which
   also covers the cancel path. The new test is the run + Results-outputs journey.

## Architecture

- New file **`tests/test_e2e_baltic.py`** — matches the `test_e2e_*.py` glob that
  `tests/conftest.py` collect-ignores when playwright is absent (CI `[dev]` has no
  playwright; e2e runs in the dedicated `[viztest]` leg). `pytestmark = pytest.mark.e2e`.
- Uses `create_app_fixture("../app.py")` + `tests/_e2e_support.dismiss_changelog_modal`
  (the established pattern; mirrors `test_e2e_live_movement.py`).
- One test function `test_baltic_full_run_and_outputs(page, app)`.

## Flow (data flow + steps)

1. **Load Baltic:** `nav[data-value='grid']` → `select_option("#load_example","baltic")`
   → `click("#btn_load_example")` → wait for `.shiny-notification` (load settled).
2. **Configure run:** `nav[data-value='run']` → `click("#engineBtnPython")` (Baltic is
   Python-only; default engine is Java). In the now-visible Python tab, fill
   `#py_param_overrides` with two lines: `simulation.time.nyear=1` and
   `output.spatial.enabled=true`.
3. **Run + live movement:** `click("#live_movement_view")` → `click("#btn_run")`.
   - Assert `#run_status` does NOT contain "Validation failed".
   - Assert `#live_movement_status` → contains "running", then "done" (`_RUN_TIMEOUT`).
   - Toggle `#live_movement_mode` → "Dots" (re-renders retained frame).
4. **Results outputs:** navigate `nav[data-value='results']` (auto-loads the run's output
   dir on completion). Assert:
   - **Biomass chart:** the results chart plot has rendered SVG trace content.
   - **Diet heatmap:** the diet plot renders AND the page does not show the "(no prey
     data)" / "no diet" empty-state text. (Selects the diet view first if it's a sub-tab.)
   - **Spatial map:** navigate to Spatial Results, pick a species, assert the map renders
     non-empty cells (deck.gl canvas or plotly heatmap trace present).
5. Save `screenshots/baltic_full_e2e.png`.

## Assertion strategy

- Plotly outputs: assert trace/`g.trace`/SVG path nodes exist inside the output container,
  not merely the container div (a bare container is zero-content). Use
  `expect(locator).to_be_visible` + a content check (e.g. `.plot .trace`, or
  `to_have_count > 0` on trace paths via `page.locator(...).count()`).
- Diet heatmap regression guard: `expect(diet_region).not_to_contain_text("no prey data")`
  AND a positive check that heatmap cells exist.
- Spatial map: assert the deck.gl `<canvas>` (or the plotly heatmap) is present and the
  "Expected 2D spatial slice" / empty-state text is absent.
- Timeouts: `_LOAD_TIMEOUT=20s`, `_RUN_TIMEOUT≈90s` (spatial I/O adds to the 1-year run).

## Risk to verify during implementation

That the Python engine emits a Results-renderable spatial file for Baltic when
`output.spatial.enabled=true` (the spatial viewer historically required synthetic
substrates because shipped configs emit no spatial output, and `minimal` is all-NaN). If
Baltic-with-override does NOT yield a renderable spatial map, fall back to asserting a
different always-emitted graphical output (e.g. biomass-by-size spectrum from
`output.biomass.bysize.enabled;true`) and document the substitution. This is checked by
running the test once locally before finalizing.

## Out of scope

- Java-engine Baltic (blocked by design — `java_engine_block_reason`).
- Every Results tab; multi-run compare / scenario-diff (separately unit + e2e covered).
- Asserting specific numeric values (a 1-year run isn't calibrated) — only that outputs
  render with real (non-empty) structure.
