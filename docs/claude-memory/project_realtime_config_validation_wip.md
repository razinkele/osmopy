---
name: project_realtime_config_validation_wip
description: "SHIPPED 2026-06-08 — real-time config validation panel on the Shiny Setup page. Live validation summary mirroring the Run gate via summarize_config_validation; merged to origin/master."
metadata:
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

**Real-time config validation in the Shiny form — SHIPPED to origin/master 2026-06-08** (`e7268bb..aebfb6b`, fast-forward, branch `feature/realtime-config-validation` deleted, pushed). 9 commits (4 spec/plan docs + 5 code).

## What shipped
A **live validation summary panel** on the Setup page (nav label "Species") that mirrors the Run gate as you edit — type/range/enum errors, missing species names, missing file references — surfaced BEFORE a run instead of only in the run log. UI-integration over EXISTING tested validators; NO new validation logic.
- **`osmose/config/validator.py`**: `summarize_config_validation(config, registry, config_dir: Path|None) -> (errors, warnings)` = the exact Run-gate sequence extracted (`validate_config` → `if config_dir: extend(check_file_references)` → `extend(check_species_consistency)`). The single source of truth.
- **DRY refactor of all 3 callers**: Run gate (`ui/pages/run.py`), the **CLI** (`osmose/cli.py:cmd_validate` — round-2 caught it was a 3rd hand-rolled copy), and the new panel — all route through the helper. Old imports pruned (ruff F401).
- **`ui/pages/setup.py`**: `@reactive.calc _config_validation()` (returns `(loaded, errors, warnings)`, `loaded = bool(config) and "simulation.nspecies" in config` → avoids false-green on `{}`; defensive try/except so the tab never crashes) + `@render.ui config_validation()` 3 states (grey "No configuration loaded" / green "✓ Configuration valid" / red-amber badge `N error(s) · M warning(s)` + capped lines). Round-2 folds: **cap each list at 10 + "… and N more"** (a moved config_dir yields 30-60+ file errors — movement.file.map{idx} alone is ~26 baltic/~56 eec FILE_PATH keys); `overflow-wrap:anywhere` on lines; `aria-live="polite"` wrapper (precedent calibration_handlers.py:837).
- **Tests**: 7 validator unit + DRY-lock + 3 source-string wiring (validator.py has no `__all__`). 3156 passed.

## The seam the final Playwright pass caught (per-task reviews missed it)
Task-3 placed `ui.output_ui("config_validation")` BETWEEN `expand_tab` and `layout_columns` inside the `osm-split-layout` flex row → (a) squished the panel to ~1/3 width (181/615px), and (b) **broke the expand_tab→grid adjacency** that the collapse CSS depends on (`.osm-split-layout > .osm-expand-tab.visible + .bslib-grid`, and the `~` collapse selectors). Fix (`aebfb6b`): lift the panel OUT of the flex container — wrap the split layout in an `osm-setup-root d-flex flex-column h-100`, panel full-width above it, + a scoped `.osm-setup-root > .osm-split-layout { flex:1 1 auto; min-height:0; height:auto; }` so the split still fills residual height. Playwright-verified: panel 615px full-width above columns, columns still 4:8 side-by-side, collapse toggles (194→0→194), live error fires + returns to green. **LESSON (same as trophic-network threshold seam): the whole-feature manual pass catches integration seams the per-task reviews can't — always drive the real render fn.** [[feedback_in_loop_review_pattern]]

## Method
Brainstorm (Approach A) → spec → **2 in-loop spec review rounds** (round-2 from fresh angles caught: cli.py 3rd copy, error-storm cap, overflow clip, aria-live; confirmed no reactive loop / panel updates on LOAD via every loader's direct state.config.set / config_dir staleness mirrors gate) → writing-plans (4 TDD tasks) → subagent-driven build (per-task spec+quality review) → final whole-feature review (Ready to merge) → Playwright → finish. Gotchas: shiny `output_ui` content renders only when its nav panel is activated (slot empty until you click the tab); `shiny run` without `--reload` needs restart to pick up edits; bypass tooltip-intercepted clicks by setting `.value` + dispatching `change` event.

## NEXT: pick a fresh backlog item
Queued (user "keep other options alive"): **pytest-xdist** suite parallelization (~8min serial / 486s, fast-test-dominated, only ~15/3146 files touch engine/numba → parallelizes well; audit xdist-safety: per-test temp dirs, RESULTS_DIR monkeypatch, no shared fixed-path writes); **scenario diff view** (side-by-side biomass + spatial maps on the shipped delta/Compare-Runs work). Backlog also: per-cell spatial NetCDF viewer, Pareto-front explorer UI, CI Python matrix, Sphinx API docs. See [[project_feature_improvements_backlog]]. Related: [[project_config_parser_diagnostics]].
