---
name: project-java-live-streaming
description: "Java engine run now streams console live in the UI (fire-and-forget, both engines off-thread)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**Java-engine run streams console LIVE in the UI — SHIPPED 2026-06-30, master `22eea67`, pushed.** Follows the silent-failure fix `b3b0574` (surface Java errors) [[project-c2-ui-java-440-background]].

**Root cause it fixed:** the Java branch in `ui/pages/run.py::handle_run` `await`ed `_run_java_engine`, which suspended handle_run so Shiny **deferred every reactive flush until the run finished** → no live console output during a Java run ("no text output"). The Python path had already been moved off-thread for exactly this reason (the dispatch comment said so).

**The pattern (now BOTH engines):** fire-and-forget background thread + reactive-poll drains. Java specifically:
- `_java_engine_setup(input, state, config, work_dir, source_dir)` — SYNC: jar check + `write_temp_config` + `stage_background_for_java`, builds `OsmoseRunner`. Returns a params dict OR an error STRING (no reactive side effects).
- `_java_engine_thread(runner, config_path, output_dir, java_opts, overrides, timeout, log_q, done_q)` — runs the jar OFF-thread via `asyncio.run(runner.run(on_progress=log_q.put_nowait))`, posts `("done", result, "")` / `("failed", None, msg)` to `_run_done_q`. Touches NO reactive state.
- handle_run: setup synchronously → launch thread → RETURN (Shiny keeps flushing).
- New `_run_log_q` + `_drain_run_log` reactive.poll (0.2s) stream jar lines → `run_log` live. The existing engine-agnostic `_drain_run_done` finishes the run (`_handle_result`, which now also shows the stdout tail on failure).

**KEY:** `_run_start_cell[0] = run_t0` is set at launch in BOTH engine branches (test_run_duration guards `count >= 2`). Cancel still works — `runner.cancel()` SIGTERMs the subprocess cross-thread. Regression test: `tests/test_java_engine_thread.py` runs Baltic on the 4.4.1 jar through `_java_engine_thread`, asserts lines stream to the queue + `("done", rc=0)` (skips if jar absent). 79 run-page tests pass; pre-existing `test_run_observer` failure unrelated.

**Still owed (user):** deploy (`sudo bash deploy.sh`) to get it live; the `grid.netcdf.file` Baltic error was config-specific and unreproduced on a fresh load (the live console + stdout-tail will now show the real cause if it recurs).
