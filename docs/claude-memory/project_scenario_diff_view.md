---
name: project-scenario-diff-view
description: "Results 'Scenario Diff' tab comparing 2 runs (overlaid biomass + A/B/diff maps) + Config-diff panel; shipped 2026-06-13/14 (PR #60 eb7dcc0, config-diff 0624769)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Scenario Diff view — SHIPPED to origin/master 2026-06-13** (PR #60 rebase-merged, master `eb7dcc0`, branch deleted). New Results tab comparing 2 runs: overlaid biomass curves (A solid/B dashed) + 3 spatial maps (A, B, B−A diff). New code: `osmose/spatial_series.spatial_diff_2d`+`grid_latlon`, `grid_helpers.make_diff_map`+shared `_z_nan_to_none`, `osmose/analysis.biomass_long`, `osmose/plotting.make_biomass_overlay`, `ui/pages/scenario_diff.py` (tab embedded in Results navset via a **new sub-server-in-a-page pattern** — `scenario_diff_server` called from `results_server`). Built subagent-driven (6 tasks, per-task 3-angle review) under ultracode; spec 3 in-loop rounds, plan 5. **Final whole-impl review caught 5 pyright errors the per-task gates missed (per-task ran ruff but NOT pyright) → fixed before merge. LESSON: include pyright in per-task review gates.** 22 new unit tests + 2 e2e (live). Full suite 3203 passed.

**Config-Diff panel — SHIPPED 2026-06-14** (`0624769`). "Config differences" accordion atop the Scenario Diff tab (Key|A|B|Change, reuses `RunHistory.compare_runs`). New `ui/pages/scenario_diff.py` `_classify_config_diffs` + `diff_config_table`. Test gotcha: assert `@render.ui` CONTENT via `to_contain_text` (bare div is zero-height); `str(nav_panel().content)` not `str(NavPanel)`. Detail in `docs/superpowers/*/2026-06-14-scenario-diff-config*`.

**Scenario Diff design gotchas (durable):** (1) spatial diff aligns on lat/lon **coords** (`np.array_equal`), not shape — runs can share `(ny,nx)` but differ in coords; (2) **same run A=B** shares ONE open handle (opening same `.nc` twice → HDF5-lock); (3) **disjoint species** → empty-state, never `ds.sel(species=[])` (zeroes grid, destroys land mask); (4) time aligned by **value** (nearest-index per run), map titles show real time; (5) `make_spatial_map`/`make_diff_map` reuse shared `_z_nan_to_none` (land NaN→None for shinywidgets `allow_nan=False`).
