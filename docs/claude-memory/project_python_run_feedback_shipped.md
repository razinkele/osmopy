---
name: project-python-run-feedback-shipped
description: 2026-06-21 Python-engine run-progress feedback (progress bar + console line + auto live map) — shipped + prod-deployed
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Python run-progress feedback — SHIPPED 2026-06-21, merged master `f1682ae` (local merge + push, no PR), prod-deployed + verified** (clone `f1682ae`, NRestarts=0, :8838 & https://laguna.ku.lt/osmose/ 200).

**Why:** user reported "running a Python simulation shows no indication, no live map animation, no console outputs." Systematic-debugging diagnosis: NOT a regression — handlers work, prod healthy, e2e + a plain-run probe confirmed `run_status` updates Idle→Running→Complete. Root cause = the Python run path gives almost no VISIBLE feedback by design: (a) `PythonEngine.run` has no console-streaming callback (only the JAVA path streams via `on_progress`→`run_log`); (b) the live map is opt-in via an off-by-default `live_movement_view` switch; (c) `py_threads`/`py_verbosity` were DEAD inputs (rendered, never read). "Python progress streaming" had been deferred/out-of-scope in the engine-capability work.

**What shipped:**
- **`osmose/live_movement.py`** (pure, unit-tested): `make_run_observer(progress_q, live_observer=None)` fans the engine's existing per-step `step_observer` into always-on progress `(done=step+1, n_steps, elapsed)` pushed to a queue + optional live snapshot (never raises into the engine); `config_is_spatial(config)` = `GridSpec.from_config` succeeds (Baltic regular-grid True; eec_full NcGrid False); `format_progress_label(done, n_steps, ndt)` — 1-based `done`, year = `(done-1)//ndt+1`, step-only fallback when ndt<=0.
- **`ui/pages/run.py`**: `_progress_q`/`_progress` + `_drain_progress` poll (mirrors `_drain_live_queue`); `run_progress` Bootstrap bar + in-place console "step" line in `run_console`; reset `_progress` at top of `handle_run`; drain `_progress_q`+set None in `_drain_run_done` AFTER `_handle_result`; auto-enable the live switch for spatial configs (changed-only guard); `py_threads` wired to `numba.set_num_threads` INSIDE `_python_engine_thread`; `py_verbosity` removed.

**KEY GOTCHAS (caught by 2 multi-agent workflow reviews after 4 in-loop spec rounds; all empirically verified):**
- **`numba.set_num_threads` is THREAD-LOCAL** (verified: set on session thread → child thread sees the default). So thread-count wiring MUST run inside `_python_engine_thread` (the thread that launches the `prange`), NOT in `handle_run`. Pass `n_threads` as a thread arg.
- **numba is an OPTIONAL extra** (not base dep; engine has a pure-Python fallback). A top-level `import numba` in run.py breaks `import app` on a numba-less install → import it LAZILY + guarded inside the thread.
- **pyright**: `numba.config.NUMBA_NUM_THREADS` needs `# type: ignore[attr-defined]` at the use site (the import-untyped ignore doesn't cover attr access); extract to a `cap` var (also dodges E501).
- **Stale 100% bar**: `_drain_run_done` must DRAIN `_progress_q` (not just `set(None)`) or the independent `_drain_progress` poll re-populates it post-completion. And place the drain AFTER `_handle_result` (the fn early-returns on empty queue; at the top it'd wipe progress every tick).
- **py_threads default flip 1→0** (=auto/all cores): wiring with the old default-1 would force single-threaded (slowdown).
- **e2e**: auto-enable makes the Baltic toggle already-on → the 3 existing manual `#live_movement_view` clicks would turn it OFF; replaced with `expect(...).to_be_checked()` sync. New plain-run progress test uses `nyear=3` (transient "step" is cleared on completion; long enough to sample, short enough for the 60s/120s budgets).

Spec+plan `docs/superpowers/{specs,plans}/2026-06-21-python-run-feedback*`. Verification: full suite 3530 pass, e2e 4 pass (incl. new plain-run progress test = behavioral proof), ruff+pyright clean. Subagent-driven execution; the workflow reviews caught the thread-local + optional-import + stale-bar issues the spec rounds missed (they reviewed design, not executable threading semantics). Related: [[project_run_page_engine_capability_shipped]] (engine-capability panel, prior), [[project_movement_visualization]].

## PARTIALLY SUPERSEDED 2026-06-21 by [[project-run-page-rework-shipped]] (master 70c9499)
The `live_movement_view` switch + the `_auto_enable_live_for_spatial` effect added here were REMOVED — the Live Movement card's collapse state is now the stream gate (`live_view_expanded` input; collapsed by default = no stream). The progress bar + in-place console line (the core of THIS feature) are UNCHANGED and still work regardless of the live gate. Reason: a live-stream crash (dots streaming → tab death → DestroyedReactiveError cascade) required gating streaming behind explicit expand + session-teardown hardening.
