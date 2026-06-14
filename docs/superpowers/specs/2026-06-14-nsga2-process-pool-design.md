# NSGA-II ProcessPoolExecutor speedup — design

**Date:** 2026-06-14
**Status:** Approved (design phase)
**Feature:** Add a ProcessPoolExecutor evaluation backend to the NSGA-II calibration so per-candidate
OSMOSE simulations run in separate processes (escaping the GIL), recovering more than the current
3.02× thread-bound speedup.

## Motivation

NSGA-II evaluates each generation's population via a **ThreadPoolExecutor** inside
`OsmoseCalibrationProblem._evaluate` (`osmose/calibration/problem.py:142-147`). The per-candidate work
is a NumPy/Numba OSMOSE simulation, so at saturation the **GIL** throttles the threads — the measured
Baltic speedup is **3.02×** at `n_parallel=4` (a 1×2 smoke hit 4.34×). The CHANGELOG names the
ProcessPoolExecutor swap as the post-v0.10.0 follow-up. Processes give each worker its own interpreter
→ true parallelism on the compiled sim loops.

**The blocker that makes this non-trivial:** the objective functions are **lambdas / closures** built
in `ui/pages/calibration_handlers.py:1386-1435` (they capture observed-data DataFrames) and are stored
on the `OsmoseCalibrationProblem` instance (`problem.py:103`). A `ProcessPoolExecutor` submitting the
bound `self._evaluate_candidate` would pickle the whole problem → `PicklingError` on the first
generation. So the feature is **(1) make the objectives picklable** + **(2) add a process backend via
the initializer pattern** (already proven in `scripts/sensitivity_phase12.py:242-266`).

## Decisions (locked during brainstorming)

1. **Opt-in `parallel_backend`** on `OsmoseCalibrationProblem`, default `"thread"` (zero behavior
   change for existing callers/tests). The NSGA-II launch sites request `"process"`.
2. **Picklable objective functors** for the standard objectives (biomass RMSE, diet distance) in
   `osmose/calibration/objectives.py`, replacing the UI lambdas.
3. **Auto thread-fallback when objectives aren't picklable.** The launch site requests `"process"` but
   a `pickle.dumps` pre-check downgrades to `"thread"` (with a logged note) if any objective fails to
   pickle. **The banded-loss objective stays on thread in v1** (its closure + the UI-layer
   `_extract_species_stats` are gnarly to make picklable) — biomass/diet calibration (the common case)
   gets the process speedup; banded still works, just thread-bound. When banded is later made
   picklable it auto-upgrades to process with no code change.
4. **Persistent pool, reused across generations** (a fresh process pool per generation would re-spawn
   workers + recompile Numba every gen). **Worker count** from `OSMOSE_NSGA2_WORKERS` (fallback
   `n_parallel`), clamped `[1,32]`.

## Reuse (do not rebuild)

- `osmose/calibration/problem.py` — `_evaluate` (batch, `problem.py:133`), `_evaluate_candidate`
  (`:175`), `_run_single`/`_run_python_engine`/`_validate_overrides`/cache (`:186-256`). The process
  path reconstructs a problem in each worker and **reuses `_evaluate_candidate` unchanged** (parity).
- `osmose/calibration/objectives.py` — `biomass_rmse(sim, obs)` (`:41`), `diet_distance(sim, obs)`
  (`:55`). The functors wrap these; no new objective math.
- `scripts/sensitivity_phase12.py:242-266` — the `ProcessPoolExecutor(initializer=_pool_init,
  initargs=...)` worker-init pattern (build per-worker state once; tasks carry only small tuples).
- `osmose/schema` `build_registry()` — rebuilt per worker (registry is not pickled/shipped).
- `FreeParameter` (`problem.py:53`) is a picklable dataclass (its `Transform` enum is picklable).

## Architecture — four units

### 1. Picklable objective functors (`osmose/calibration/objectives.py`)

Module-level callable classes (picklable; hold only a DataFrame, which pandas pickles):

```python
class BiomassRMSEObjective:
    def __init__(self, observed: pd.DataFrame): self.observed = observed
    def __call__(self, results) -> float: return biomass_rmse(results.biomass(), self.observed)

class DietDistanceObjective:
    def __init__(self, observed: pd.DataFrame): self.observed = observed
    def __call__(self, results) -> float: return diet_distance(results.diet_matrix(), self.observed)
```

`ui/pages/calibration_handlers.py` builds `BiomassRMSEObjective(obs_bio_df)` /
`DietDistanceObjective(obs_diet_df)` instead of the lambdas (`:1386,:1389`). The banded objective is
left as-is (its closure path is unchanged; the auto-fallback keeps banded runs on thread).

### 2. `EvalSpec` (picklable) + worker functions (`osmose/calibration/problem.py`)

A frozen dataclass carrying exactly what a worker needs to reconstruct an evaluator — **everything the
`OsmoseCalibrationProblem.__init__` eval path uses except the registry** (rebuilt locally):

```python
@dataclass(frozen=True)
class _EvalSpec:
    free_params: list[FreeParameter]
    objective_fns: list[Callable]      # picklable functors
    base_config_path: Path
    work_dir: Path
    jar_path: Path | None
    java_cmd: str
    enable_cache: bool
    cache_dir: Path | None
    subprocess_timeout: int
    cleanup_after_eval: bool
    use_java_engine: bool
    use_registry: bool                 # = (self._registry is not None); see note below
```

`_EvalSpec` is built from the parent's **resolved** state — `self._enable_cache` and `self._cache_dir`
(the post-`__init__` value `cache_dir or work_dir/.cache`, `problem.py:110`), not the raw constructor
args — so every worker agrees on one shared cache dir. **`use_registry = (self._registry is not None)`
— load-bearing for parity:** the thread path skips override validation when `self._registry is None`
(`problem.py:374`), and the UI builds the problem with no registry. If workers unconditionally
`build_registry()` and validate, they could score a candidate `inf` that the thread path accepts →
breaking the thread==process parity. So workers rebuild the registry **iff the parent had one**.
`n_obj` is **not** an `_EvalSpec` field: pymoo's base `Problem.__init__` sets `self.n_obj` when the
worker reconstructs the problem, so `_worker_eval` reads it from there.

Module-level worker functions (so `ProcessPoolExecutor` pickles only the spec + small tuples):

```python
_WORKER_PROBLEM: OsmoseCalibrationProblem | None = None

def _worker_init(spec: _EvalSpec) -> None:
    global _WORKER_PROBLEM
    reg = build_registry() if spec.use_registry else None
    _WORKER_PROBLEM = OsmoseCalibrationProblem(
        free_params=spec.free_params, objective_fns=spec.objective_fns,
        base_config_path=spec.base_config_path, work_dir=spec.work_dir,
        jar_path=spec.jar_path, java_cmd=spec.java_cmd, n_parallel=1,
        parallel_backend="thread",  # workers never nest a pool
        enable_cache=spec.enable_cache, cache_dir=spec.cache_dir, registry=reg,
        subprocess_timeout=spec.subprocess_timeout,
        cleanup_after_eval=spec.cleanup_after_eval, use_java_engine=spec.use_java_engine,
    )

def _worker_eval(run_id: int, params: np.ndarray) -> list[float]:
    try:
        return _WORKER_PROBLEM._evaluate_candidate(run_id, params)
    except Exception:  # noqa: BLE001 — one bad candidate must not kill the generation
        return [float("inf")] * _WORKER_PROBLEM.n_obj
```

### 3. Process backend in `_evaluate` + pool lifecycle

`OsmoseCalibrationProblem.__init__` gains `parallel_backend: Literal["thread","process"] = "thread"`
and `self._executor = None` (add `from typing import Literal` + `import multiprocessing` +
`from concurrent.futures import ProcessPoolExecutor` and
`from concurrent.futures.process import BrokenProcessPool` + `from concurrent.futures import
as_completed` + `import pickle` + `from osmose.schema import build_registry` (module-level, since
`_worker_init` runs under `forkserver`) — `problem.py` currently imports only `ThreadPoolExecutor` and
`TYPE_CHECKING, Callable`; `import os` is already present).
`_evaluate`'s branch structure is **rewritten** (NOT kept verbatim — the existing `if self.n_parallel
> 1:` thread branch would shadow the process path since NSGA-II runs at `n_parallel=4`). The full
structure, backend-aware so the right branch is reached:

```python
if self.parallel_backend == "process" and self.n_parallel > 1:
    self._evaluate_process(X, F)          # §below
elif self.n_parallel > 1:                 # existing thread path, unchanged body
    <ThreadPoolExecutor ... as today>
else:
    <serial loop ... as today>
```

`_evaluate_process(X, F)` — must **not** let one dead worker discard already-finished candidates
(a single `fut.result()` raising `BrokenProcessPool` would otherwise abandon the rest of the
generation → `>50%` inf → the whole multi-hour run aborts on one transient OOM). So it tracks the
still-unscored candidates and, on a broken pool, rebuilds and **re-submits only those** within the
same generation (re-running a candidate is safe — `seed=run_id` is deterministic), capped at a small
number of rebuilds:

```python
pending = {i: params for i, params in enumerate(X)}   # candidates not yet scored
for _attempt in range(3):                             # 1 initial + 2 rebuild retries
    if not pending:
        break
    pool = self._ensure_pool()                        # lazy-create / rebuild (discards a broken pool)
    try:
        # submit INSIDE the try: a pool that broke idle between generations raises
        # BrokenProcessPool from submit() itself, which must hit the handler below.
        futures = {pool.submit(_worker_eval, i, p): i for i, p in pending.items()}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                for k, v in enumerate(fut.result()):
                    F[i, k] = v
                del pending[i]                        # scored OK
            except _expected_errors as exc:           # bad candidate → stays inf, done
                _log.warning("Candidate %d failed (expected): %s", i, exc)
                del pending[i]
    except BrokenProcessPool as exc:                  # RuntimeError, NOT in _expected_errors
        _log.warning("Pool broke (worker died): %s — rebuilding, retrying %d candidates",
                     exc, len(pending))
        self.shutdown_pool()                          # finished results stay in F; pending retried
# any still-pending after the retry cap stay inf; the >50% guard then applies
```

- `_ensure_pool()`: first **discard a broken persisted pool** —
  `if self._executor is not None and getattr(self._executor, "_broken", False): self.shutdown_pool()`
  (a pool can break while idle between generations). Then if `self._executor is None`, build
  `_EvalSpec` from `self`, **pre-check `pickle.dumps(spec)`** (raise a clear "objective not picklable;
  use parallel_backend='thread'" on failure), then
  `self._executor = ProcessPoolExecutor(max_workers=_resolve_worker_count(self.n_parallel),
  mp_context=multiprocessing.get_context("forkserver"), initializer=_worker_init, initargs=(spec,))`.
  Returns the executor. (`forkserver` per the Workers section — avoids the daemon-thread/Numba fork
  hazard.)
- `_resolve_worker_count(n_parallel)`: `int(os.environ.get("OSMOSE_NSGA2_WORKERS", n_parallel))`,
  clamped to `[1, 32]` (≥1).
- `shutdown_pool(cancel_futures=False)`: `if self._executor: self._executor.shutdown(cancel_futures=...)`;
  reset to `None`. Idempotent (no-op when no pool). The launch sites call it in a `finally`.
- The `>50% inf → abort` guard and `out["F"]` handling are unchanged.

### 4. Wiring at the NSGA-II launch sites

- `ui/pages/calibration_handlers.py`: after assembling `objective_fns`, choose the backend —
  `backend = "process" if _all_picklable(objective_fns) else "thread"` (`_all_picklable` is a small
  helper **in `calibration_handlers.py`** doing `pickle.dumps` on each + logging a UI-facing fallback
  note). Pass `parallel_backend=backend` to
  `OsmoseCalibrationProblem(...)`. Wrap the `minimize(...)` run so `problem.shutdown_pool()` runs in a
  `finally` (incl. the cancel path → `shutdown_pool(cancel_futures=True)`).
- `scripts/benchmark_calibration.py`: add a `--backend {thread,process}` switch so the benchmark can
  compare the two; ensure `shutdown_pool()` in a `finally`.

## Data flow

```
generation X (pop_size × n_var)  ──► OsmoseCalibrationProblem._evaluate
  parallel_backend == "process":
     _ensure_pool() → ProcessPoolExecutor(initializer=_worker_init, initargs=(_EvalSpec,))   [once]
        each worker: reconstruct OsmoseCalibrationProblem(n_parallel=1, registry=build_registry() iff parent had one)
     submit _worker_eval(i, params) per candidate  →  worker runs _evaluate_candidate (seed=i)
        → _run_single → _run_python_engine (in-process sim) → objective functors → [floats]
     collect into F → out["F"]
  (pool persists across generations; launch site shutdown_pool() in finally)
```

Pymoo stays serial generation-to-generation; only within-generation evaluation moves to processes.

## Picklability / determinism / workers

- **Picklable:** functors (DataFrame only), `FreeParameter` dataclasses, paths/bools/ints, the
  `_EvalSpec`. The registry is **rebuilt** per worker via `build_registry()` (not pickled). The pymoo
  `Problem` is never pickled — only `_EvalSpec` + `(run_id, params)` tuples cross the boundary.
- **Determinism / parity:** each candidate uses `seed=run_id` (its enumerate index, identical
  assignment in both backends) → deterministic regardless of worker/order; NSGA-II's algorithm RNG
  (`minimize(seed=42)`) lives in the parent. ⇒ process and thread backends produce **identical**
  `out["F"]` for the same population (the keystone test).
- **Workers:** persistent pool reused across generations; `OSMOSE_NSGA2_WORKERS` (clamp `[1,32]`,
  ~400 MB/sim memory caveat — 28 workers once exhausted 32 GB).
- **Start method = `forkserver` (NOT fork) — load-bearing.** The NSGA-II run executes on a **daemon
  thread** (`calibration_handlers.py:1315`), and the engine uses `@njit(parallel=True)`/`prange`
  (`osmose/engine/processes/mortality.py:1372,1433`), so by the time `_ensure_pool()` runs, a warmed
  Numba threadpool exists. Forking a `ProcessPoolExecutor` from a non-main thread with that live
  threadpool is the documented Numba deadlock hazard ("Attempted to fork from a non-main thread, the
  TBB library may be in an invalid state in the child"). Mitigation: create the pool with
  `mp_context=multiprocessing.get_context("forkserver")` — workers are forked from a clean, minimal
  forkserver process, never inheriting the warmed parent threadpool. (`forkserver`/`spawn` re-import
  the module, so `_worker_init`/`_worker_eval`/`_EvalSpec` MUST be module-level — they are.) `spawn`
  is the fallback if `forkserver` is unavailable.
- **Numba cache:** each worker recompiles its kernels on first eval (~seconds), paid **once per
  worker** because the pool is persistent — it amortizes across all generations. **Do NOT point
  workers at one shared `NUMBA_CACHE_DIR`** — this repo already learned (see
  `tests/_xdist_support.py` / `tests/conftest.py`) that parallel processes **race** writing the same
  cold `.nbi`/`.nbc` files; give workers no shared numba cache (accept the one-time recompile) or, if
  ever needed, a pre-warmed read-only cache. (The atomic-rename at `problem.py:239-243` is the
  objective-JSON cache, unrelated to numba's writer — it does keep the JSON eval cache process-safe.)

## Error handling

- **Two complementary picklability checks** (not redundant work): the launch-site `_all_picklable`
  is the *user-facing* graceful path (process→thread downgrade, logged — e.g. banded) and runs before
  the problem is built; the `_ensure_pool()` `pickle.dumps(_EvalSpec)` is the *in-core defensive*
  guard for direct API callers who set `parallel_backend="process"` with a non-picklable objective →
  clear error before any worker spawns.
- **Dead worker / `BrokenProcessPool`** (the most likely failure given the ~400 MB/sim caveat): caught
  in the process branch (it's a `RuntimeError`, not in `_expected_errors`) → `shutdown_pool()` tears
  down the poisoned pool so the next generation's `_ensure_pool()` rebuilds a fresh one; the affected
  candidates stay `inf` and the parent's `>50% inf` abort fires if a whole generation was lost.
- **Worker eval raises** → caught in `_worker_eval`, returns `inf` vector (parity with current failure
  handling).
- **Cancel** → pymoo's callback sets `force_termination` at the **generation boundary**, so cancel
  latency is *one generation* (the in-flight generation's process evals finish — same as the existing
  thread `with`-block). `shutdown_pool(cancel_futures=True)` in the launch-site `finally` is cleanup of
  an already-idle pool (with `cancel_futures=True` for the rare race where `finally` runs mid-flight).
- `_resolve_worker_count` clamps `<1 → 1`, `>32 → 32`.

## Testing

1. **Functors `tests/test_calibration_objectives.py`** (extend): `BiomassRMSEObjective` /
   `DietDistanceObjective` return the same value as `biomass_rmse`/`diet_distance` on the same frames;
   each functor survives `pickle.dumps`/`loads`.
2. **Parity `tests/test_nsga2_parallel.py`** (the keystone): a tiny `OsmoseCalibrationProblem` built
   from **`data/minimal/osm_all-parameters.csv`** (`nyear=10, ndtperyear=12, nspecies=2` → ~0.24 s
   warm/eval; do **NOT** use `data/examples/` — ~9.8 s/eval blows the budget), ≥1 cheap picklable
   objective, small `X` (2–4 rows, `n_obj=1`) → `_evaluate` with `parallel_backend="thread"` vs
   `"process"` yields **identical `out["F"]`** (verified live: bit-identical on 4 candidates / 2
   workers). Also: a worker eval that raises returns an `inf` vector; `_EvalSpec` round-trips
   `pickle.dumps`. (A few seconds — real `forkserver` spawn + one-time Numba compile; runs in the
   normal suite, not e2e.) **Run a second parity variant WITH a registry on both backends**
   (`registry=build_registry()`) so the `use_registry` mapping is exercised — a worker-vs-parent
   validation divergence would otherwise pass undetected (the no-registry variant can't catch it).
3. **Broken-pool recovery `tests/test_nsga2_parallel.py`**: cover **both** break points —
   `BrokenProcessPool` from `fut.result()` (a worker `os._exit(1)`s mid-eval) **and** from
   `pool.submit()` (a pool already broken when the next generation starts — e.g. inject a stub
   executor whose `submit` raises). Assert `_evaluate` does **not** propagate, that already-scored
   candidates keep their finite values (not clobbered to `inf`), still-pending candidates are retried
   on a rebuilt pool, and `_ensure_pool` discards a `_broken` pool.
4. **Worker count `tests/test_nsga2_parallel.py`**: `_resolve_worker_count` — env unset → `n_parallel`;
   set → clamped `[1,32]`; `<1 → 1` (monkeypatch `setenv`).
5. **Pool lifecycle**: `shutdown_pool()` is a no-op when no pool exists and idempotent after one
   `_evaluate`.

## Benchmark (not CI-gated)

Extend `scripts/benchmark_calibration.py` with `--backend {thread,process}` and print wall-clock +
the process/thread ratio. This is the *evidence* for the speedup — hardware-dependent, run on the real
multi-core box. **No CI ratio assertion** (CI is 2-core; measure perf, don't gate it).

## Out of scope (YAGNI)

- **Banded-loss objective picklability** — deferred; the auto thread-fallback keeps banded runs correct
  (just thread-bound). Revisit by moving `_extract_species_stats` + a `BandedBiomassObjective` to core.
- Parallelizing pymoo across generations (stays serial); touching DE / CMA-ES / surrogate-DE (already
  parallel via scipy/joblib); exposing the process backend in `scripts/calibrate_baltic.py` (no NSGA-II
  there); new objectives; distributed/multi-node/GPU; a CI perf-ratio gate; auto-selecting backend by
  eval count.
