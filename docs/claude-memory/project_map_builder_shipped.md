---
name: project-map-builder-shipped
description: Map Builder page — author OSMOSE spatial grid maps (distribution/mask/zone) on the config grid; shipped 2026-06-19 PR
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Map Builder** — a Shiny page to author OSMOSE spatial grid maps by drawing polygons + click-to-painting cells on the loaded config's georeferenced grid, saving them as the engine's `;`-separated grid CSVs with type-aware config wiring. Shipped 2026-06-19, PR #81 (squash `2fd868d` on master), all CI green incl. CodeRabbit. 24 commits (spec→plan→subagent-driven TDD).

**Architecture (logic in a pure core, page is thin):**
- `osmose/maps/builder.py` (browser-free, unit-tested): `GridSpec` (cell_polygon scalar + `cell_polygons()` vectorized + cell_center + from_config + dx/dy), numpy ray-cast `rasterize_polygon`/`lonlat_to_cell`, `MapGrid` (blank/apply_cells/apply_polygon/erase/set_mask; `__init__` copies its buffer), `to_csv_text`/`from_csv_text`/`_fmt`, `validate`, `wire_map_into_config`, `save_map`/`_sanitize_filename`.
- `ui/pages/map_builder.py` over `shiny_deckgl`: MapWidget id `mb_map` (→ `input.mb_map_map_click`/`input.mb_map_drawn_features`), stage-then-Apply polygon draw + click-to-paint per cell, brush/eraser/mask tools, `dirty`-counter coalesced `partial_update` render (event-driven, not poll), Load-existing via `from_csv_text`.
- `app.py` 4 nav touch-points. DRY: `ui/pages/grid_helpers.py` `build_grid_layers` now derives cells from `GridSpec.cell_polygons()` (single source of truth; `tests/test_ui_grid.py` unchanged characterization guard).

**Load-bearing invariants (don't re-derive, they're locked by tests):**
- Map CSV `nlat`×`nlon`, `;`-sep, `-99` land. **File is SOUTH-row-0, memory is NORTH-row-0**; `to/from_csv_text` bridge with `np.flipud` (exactly once each). Round-trip verified THROUGH the engine's own `osmose.engine.movement_maps._load_csv_grid(path, ny, nx)` — never self-round-trip only.
- Real movement keys only: `movement.{species,file,steps,initialage,lastage,initialyear,lastyear}.map{N}` (discovery iterates `movement.species.map{N}`); NEVER the inverted `movement.map{N}.*`. `_next_map_index` = lowest-free index. Saved keys verified to survive 4.4.0 canonicalization (`aliases.py` doesn't touch them).
- Cell geom: `dx=(lr_lon-ul_lon)/nlon`, `dy=(ul_lat-lr_lat)/nlat`; cell `(r,c)` corners `[UL,UR,LR,LL]`. Distribution-on-base-mask-land WARNS (engine treats as absent), doesn't block.

**Tests:** `tests/test_maps_builder.py` (19 + Hypothesis prop), `tests/test_ui_map_builder.py` (import/nav/_species_choices/_existing_maps), `tests/test_e2e_map_builder.py` (3 Playwright, Baltic — render/toggle/save round-trip). Map Builder needs a REGULAR lat/lon grid (`grid.nlon/nlat/upleft.*/lowright.*`) → use **Baltic** for e2e, NOT eec_full (NcGrid has no bounds keys). Out of scope: multi-species-per-map, spatial-fishing mechanics, undo/redo, grid-bounds authoring, raster/shapefile import.

**Gotcha (visual gate):** adding a nav item changes the `#main_nav` rail → `test_visual_nav_chrome` (`nav_chrome` baseline) legitimately fails. Re-bless ONLY the genuinely-changed baseline (the gate's `F....` told us only nav_chrome failed; `advanced.png` also "changed" on regen but was sub-threshold AA noise — committing it = blessing noise). Regenerate in the pinned playwright container via `gh workflow run "Visual" --ref <branch> -f pages=all`, download the `visual-baselines` artifact, commit only the failed one. visual-gate is ADVISORY/non-required. See [[feedback-visual-harness-toast-gotcha]].

Spec+plan: `docs/superpowers/{specs,plans}/2026-06-19-map-based-scenario-builder*`. Backlog item from [[project_feature_improvements_backlog]].
