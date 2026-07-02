---
name: project-live-during-run-movement-viz
description: "Run-page deck.gl map streams living schools (heatmap/dots) during a Python-engine sim; shipped 2026-06-13 (PR #61, 3cb20a1). Durable: await run_in_executor in a Shiny effect BLOCKS live UI — use background thread + reactive.poll"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Live-during-run movement viz — SHIPPED to origin/master 2026-06-13** (PR #61 rebase-merged, master `3cb20a1`, branch deleted, local master synced; all CI legs green incl. test 3.12/3.13, CodeRabbit).

Run-page deck.gl map streams living schools as a biomass heatmap OR dots (toggle + species filter) WHILE a **Python-engine** sim runs (Java = prebuilt JAR, no hook → "Python only" note). New `osmose/live_movement.py` (`MovementSnapshot`/`build_snapshot`/`resolve_grid_latlon`/`make_step_observer`), `ui/pages/live_movement_render.py` (heatmap/dots, local species palette, `@@=d.*` row-dict accessors), engine **opt-in `step_observer` hook** in `simulate()` (parity-safe, off by default), run.py wiring (two-handle MapWidget + `reactive.poll` drain + async render effect).

**▶▶ CRITICAL LESSON (live e2e caught what 6 spec/plan review rounds missed):** the original `handle_run` did `await loop.run_in_executor(whole run)` → Shiny defers ALL flushes until the run ends → the live view never repainted mid-run. **Fix = fire-and-forget**: run the Python engine in a daemon thread (the calibration-dashboard pattern) + a main-thread completion `reactive.poll`; handler returns immediately so frames + run_log/status flush live. This ALSO fixed a pre-existing "Python run log/status don't stream mid-run" limitation. **Pattern: `await run_in_executor(long_task)` in a Shiny async effect BLOCKS live UI updates — use a background thread + reactive.poll instead.**

Built subagent-driven (5 tasks, per-task spec+quality+independent+**pyright** gates), spec 3 in-loop rounds + plan 3 rounds (all via sequential Workflow agents — parallel hit transient structured-output rate-limits). Full suite 3222 passed; e2e 2/2 live (twice, no flakiness).
