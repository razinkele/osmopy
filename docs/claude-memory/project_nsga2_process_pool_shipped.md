---
name: project-nsga2-process-pool-shipped
description: NSGA-II ProcessPoolExecutor calibration speedup — shipped to origin/master 2026-06-15 (7a87c86); design gotchas for the process backend
metadata: 
  node_type: memory
  type: project
  originSessionId: c43bb8b2-9fc9-4f4c-a030-02009958769b
---

**NSGA-II ProcessPoolExecutor speedup — SHIPPED to origin/master 2026-06-15** (`7a87c86`, FF-merge `ab6e6ed..7a87c86` then `git push origin master`; branch `feat/nsga2-process-pool` deleted; local == origin/master). Escapes the GIL-capped ~3.02× thread-bound speedup by running per-individual OSMOSE sims in separate **forkserver** processes during a pymoo NSGA-II calibration run.

**What it is:** opt-in `parallel_backend: Literal["thread","process"]="thread"` kwarg on `OsmoseCalibrationProblem` (default thread = ZERO behavior change). The calibration dashboard auto-selects `"process"` when objectives are picklable (`_all_picklable` pickle.dumps probe in `calibration_handlers.py`), else falls back to `"thread"` (banded-loss closure case). Worker count via `OSMOSE_NSGA2_WORKERS` clamped [1,32] (fallback `n_parallel`). `scripts/benchmark_calibration.py --backend {thread,process}` compares.

**New/changed code:**
- `osmose/calibration/objectives.py`: pure `_biomass_long(df)` (melts WIDE engine biomass [Time + per-species cols] → long [time,species,biomass]; idempotent if already long) + picklable functors `BiomassRMSEObjective`/`DietDistanceObjective` (module-level, replace the old non-picklable UI lambdas).
- `osmose/calibration/problem.py`: frozen `_EvalSpec` dataclass; module-level `_worker_init(spec)` (rebuilds the problem once per worker with `n_parallel=1, parallel_backend="thread"` — workers must never nest a pool) + `_worker_eval(run_id, params)` (broad-except → inf) + `_resolve_worker_count`; `_evaluate` backend-aware dispatch (`if process and n_parallel>1 / elif n_parallel>1 / else`); `_evaluate_process` with per-future recovery; `_ensure_pool` (forkserver mp_context + pickle pre-check + discards `_broken`); `shutdown_pool(cancel_futures=False)`.
- `ui/pages/calibration_handlers.py`: `_all_picklable` helper; `parallel_backend=(...)` at the SHARED ctor (`:1112`); `finally: problem.shutdown_pool(cancel_futures=True)` around the run; lambdas→functors; removed orphaned `biomass_rmse/diet_distance` import.
- `scripts/benchmark_calibration.py`: module-level picklable `_BenchObjective` + `--backend` + `finally: shutdown_pool()`.

**▶▶ DURABLE GOTCHAS (design-critical):**
1. **forkserver context (NOT default fork):** `ProcessPoolExecutor(mp_context=multiprocessing.get_context("forkserver"), ...)`. Forking from the NSGA-II daemon thread + Numba `@njit(parallel=True)` deadlocks; forkserver starts a clean process.
2. **`BrokenProcessPool` subclasses `RuntimeError`** — it is deliberately NOT in `_expected_errors` (which would mis-score a dead worker as a per-candidate inf). Caught separately.
3. **PER-FUTURE broken-pool catch (the BLOCKER fix — verified flaky 50% without it):** in the `as_completed` drain, on `BrokenProcessPool` just record `broke = exc` and KEEP DRAINING (do NOT break) — futures that already finished still yield results into `F[i]` above; only still-`pending` candidates retry on a rebuilt pool (capped 3 attempts; re-running is safe — `seed=run_id` is deterministic). A whole-loop catch abandons finished candidates → flaky.
4. **submit() in its own try:** a pool that broke idle between generations raises `BrokenProcessPool` from `submit()` itself, not just from `.result()`.
5. **`use_registry=(self._registry is not None)` in `_EvalSpec`:** the registry is rebuilt per worker via `build_registry()` (not pickled) — required for thread==process parity (without it, `_validate_overrides` skips → different code path).
6. **Determinism = parity:** `seed=run_id` gives bit-equal `out["F"]` thread vs process (tested both with/without registry).

**Folded-in pre-existing bugfix (user chose "fix within this feature"):** the old biomass-RMSE objective fed the WIDE `results.biomass()` straight into `biomass_rmse` (which merges on [time,species]) → empty-merge → inf → >50% abort, OR KeyError. `_biomass_long` reshapes BOTH the observed CSV (in `__init__`, idempotent) and the simulated frame (in `__call__`).

**Verified:** full non-e2e suite 3296 passed / 19 skipped; broken-pool recovery 5/5 deterministic; `--backend process` real NSGA-II ran clean on forkserver (4 candidates, Baltic, each worker rebuilt its own config); ruff + pyright clean.

**Process:** built subagent-driven under ultracode. Spec 5 in-loop rounds (caught forkserver, the not-in-`_expected_errors` gap, per-future fix, use_registry parity, submit-outside-try). Plan 2 rounds (R1 the wide/long bug → user folded the fix in; R2 verified the per-future fix 8/8 deterministic by running it). Execution = ONE Workflow (3 tasks × impl+spec+quality+pyright, zero fix-loops, final review clean). See `docs/superpowers/{specs,plans}/2026-06-14-nsga2-process-pool*`.

Related: [[project_calibration_speedup_roadmap]] · [[feedback_de_workers_default]] · [[feedback_in_loop_review_pattern]]
