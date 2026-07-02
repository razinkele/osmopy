---
name: Shiny reactive patterns — isolate, dirty flag, exception handling
description: Non-obvious gotchas when writing reactive code in this app. Failing to follow these causes infinite reactive loops or silent stale config.
type: feedback
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
**Rules for reactive code in ui/:**

1. `sync_inputs()` MUST set `state.dirty.set(True)` when the config actually changes — otherwise downstream Save/Run buttons lose their enable signal.
2. `update_config()` and `sync_inputs()` MUST use `reactive.isolate()` when reading config — reading it reactively creates a dependency on every config key, causing a rebuild on every keystroke.
3. `_rebuild_movement_cache` isolates `_read_grid_values()` to prevent cache rebuilds on coordinate edits.
4. `movement_controls` is a separate `@render.ui` output (not inside `grid_overlay_selector`) — nesting creates a reactive loop because the overlay dropdown depends on movement controls.
5. Exception handling uses `SilentException` (from `shiny.types`), NOT bare `except Exception`. Bare `except` swallows Shiny's own control-flow exceptions and breaks reactivity.
6. `render_field()` accepts an optional `config` dict for loaded values — pass it when rendering into a form that has a loaded scenario.

**Why:** Shiny's reactivity is implicit — reading any reactive value inside a reactive context subscribes to it. Without isolation, benign reads create invalidation cascades. Without the dirty flag, the UI can't tell the user there's unsaved work.

**How to apply:** When adding any new reactive effect or output, first ask: does this read config/input? If yes, either wrap in `isolate()` or accept that it will re-run on every change. When catching exceptions around Shiny calls, use `SilentException`, never bare `except`.
