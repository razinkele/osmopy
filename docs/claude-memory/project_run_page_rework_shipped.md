---
name: project-run-page-rework-shipped
description: 2026-06-21 Run-page rework — Python default + compact UI + live-stream crash fix — shipped + prod-deployed
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Run-page rework — SHIPPED 2026-06-21, merged master `70c9499` (local merge + push, no PR), prod-deployed + verified** (clone `70c9499`, NRestarts=0, :8838 & https://laguna.ku.lt/osmose/ 200).

User report: "make python default; run UI totally unacceptable — spinner + live-movement controls take too much space; app CRASHED displaying cod as dots at step 1037/1200, showing 5000 of 14846 schools."

**CRASH ROOT CAUSE (systematic-debugging, direct prod stack trace `DestroyedReactiveError: '_live_status_val' has been destroyed` + CancelledError cascade @ 12:50:22; server NRestarts=0 so the SERVER never died — the SESSION did):** sustained live **dots** streaming over a long run exhausted the browser tab → session died → the reactive polls (`_drain_run_done`/`_drain_live_queue`/`_drain_progress`), async `_render_live_map`, and the fire-and-forget daemon engine thread kept firing against the destroyed session → the cascade. Server-side snapshot/layer build proven robust (1200-frame stress, 0 exceptions, 4.5MB). Dots = heaviest renderer (per-point ScatterplotLayer) vs heatmap (texture).

**What shipped:**
- **Python default engine** (`ui/state.py:64` `reactive.Value("python")` + `app.py` localStorage fallback `'python'`; Java still selectable). Resolves the jar-swap-review "Java is the UI default" nuance.
- **Compact UI:** `loading_overlay` → slim corner pill (`osm-busy-pill`/`osm-spinner-sm`, www/osmose.css); Live Movement card **collapsed by default** with controls inside.
- **Crash fix (3 layers):** (1) **live card collapse = the stream gate** — replaced the `live_movement_view` switch + the same-day `_auto_enable_live_for_spatial` effect with a `live_view_expanded` input published by `toggleCardBody`/`restoreCardBodies` (Shiny.setInputValue, gated on `panelId==='run_live_movement'`); no stream unless expanded. (2) `choose_live_layer` (osmose/ui/pages/live_movement_render.py): dots→**heatmap fallback above 1500** filtered pts; `dot_cap` 5000→2000 (BOTH `build_snapshot` AND `make_step_observer` — the latter passes it through and OVERRIDES the former, so changing only build_snapshot is a no-op); live throttle 0.2→0.5s. (3) `session.on_ended` cancels the run (stops the daemon thread) + `_session_alive`/`_active_cancel_token` plain cells; the 3 polls + `_render_live_map` + `_populate_live_species` early-return on `not _session_alive[0]` and wrap bodies in `except BaseException` that **RE-RAISES when alive** (don't mask real bugs) and only debug-logs on teardown.

**KEY GOTCHAS (caught by the multi-agent workflow plan-review BEFORE execution; all verified):**
- `make_step_observer` has its OWN `dot_cap` default that overrides `build_snapshot`'s → must change both.
- **`asyncio.CancelledError` is a `BaseException`, NOT `Exception`** → guards must catch `BaseException` (the named cascade) and re-raise when the session is alive so genuine `_handle_result` bugs aren't masked.
- shiny_deckgl layer dicts key the kind under `"type"` (`ScatterplotLayer`/`HeatmapLayer`), NOT `@@type` (that prefix is only for accessor strings).
- The engine-default test lives in `tests/test_state_engine.py`, not `test_state.py`.
- Live-card collapse state is client-side localStorage; made server-readable via a `live_view_expanded` input on the toggle. Default-collapsed via `stored===null ? (panelId==='run_live_movement') : (stored==='1')` (only the live card; Run Config/Console stay expanded).
- e2e: removed the `#live_movement_view` `to_be_checked` asserts; expand the card race-safe (click only if collapsed, then assert `not_to_have_class("osm-body-collapsed")`).

**Supersedes part of [[project-python-run-feedback-shipped]]** (f1682ae): the `live_movement_view` switch + `_auto_enable_live_for_spatial` are removed; the live-progress bar/console from that feature are UNCHANGED (progress streams regardless of the live gate). Spec+plan `docs/superpowers/{specs,plans}/2026-06-21-run-page-rework*`. Verification: full suite 3536 pass, e2e 4 pass, ruff+pyright clean. NOTE: `app.py` is NOT in CI's ruff scope (`osmose/ ui/ tests/` only) — a pre-existing app.py:42 format nit is harmless.
