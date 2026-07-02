---
name: Movement visualization feature (v0.2.0)
description: Grid-page movement animation overlay — architecture, key helpers, config-key quirk, and the reactive loop fix. Non-obvious UI shape.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
**Feature:** Grid page overlay dropdown entry "Movement Animation" reveals species/speed/slider controls and animates per-timestep CSV movement maps.

**Spec:** `docs/superpowers/specs/2026-03-14-movement-visualization-design.md`
**Plan:** `docs/superpowers/plans/2026-03-14-movement-visualization-plan.md`
**Helpers live at:** bottom of `ui/pages/grid_helpers.py`.

**Key implementation details:**

- Pre-loads all CSV movement maps into reactive cache on species selection (partial-update optimization skips `_map.update()` when active map set unchanged).
- Helpers: `derive_map_label`, `parse_movement_steps`, `build_movement_cache`, `list_movement_species`.
- `MOVEMENT_PALETTE`: 8-color RGBA list for distinct map layers.

**Config-key quirk:** Config keys use `movement.{field}.map{N}` format — NOT the schema's `movement.map{idx}.{field}` layout. This is load-bearing and has tripped up refactors before.

**Reactive architecture:** `movement_controls` is a *separate* `@render.ui` output (NOT nested inside `grid_overlay_selector`). Nesting would trigger a reactive loop — the overlay dropdown depends on movement controls, which depend on the dropdown. `_rebuild_movement_cache` also isolates `_read_grid_values()` to prevent cache rebuilds on coordinate edits.
