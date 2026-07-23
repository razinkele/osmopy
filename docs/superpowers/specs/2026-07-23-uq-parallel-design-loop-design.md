# Spec — Parallelize the UQ design loop

> 2026-07-23. Enables the surrogate-Bayesian UQ real-data validation run
> (Phase 1b, 9 params) to complete in hours instead of a multi-day serial job.
> Context: `docs/superpowers/2026-07-23-uq-real-data-validation.md`.

## Problem

`run_design` (`osmose/calibration/uq/design.py`) runs `n_points × n_seeds`
independent engine calls serially, one `evaluator(x, seed)` at a time. At the
real `nyear=40` (~53 s/run), a ~150-point / n_seeds design is ~450–1500 runs =
6.6–22 h serial on `ncpu=1`. The runs are embarrassingly parallel; the design
loop is the natural fan-out point.

**Constraint:** `run_design` accepts an *arbitrary injectable* `Evaluator`
callable. Synthetic tests pass unpicklable closures that must stay serial.
Parallelism must be opt-in and encapsulated, not forced on every evaluator.

## Approach (B — batch-capable evaluator protocol)

`make_engine_evaluator(..., n_workers: int = 1)`:
- `n_workers == 1` → return the current serial closure, **unchanged**.
- `n_workers > 1` → return a small callable object that owns a persistent
  `ProcessPoolExecutor` and also exposes `evaluate_batch(tasks)`.

`run_design` builds a flat task list and calls `evaluate_batch` **iff** the
evaluator exposes it (`hasattr`); otherwise it serial-loops exactly as today.

Only two units change; `grow_until_calibrated` and `run_surrogate_bayes` are
untouched (the caller constructs the evaluator and passes it down).

### Interfaces

```
# design.py — task = (task_id, x, seed)
def run_design(evaluator, free_params, target_keys, n_points, n_seeds, *,
               seed=0, seed_offset=0, X=None) -> DesignResult
    # builds tasks; if hasattr(evaluator, "evaluate_batch"):
    #     results = evaluator.evaluate_batch(tasks)   # list[dict], aligned to tasks
    # else: results = [evaluator(x, s) for (_, x, s) in tasks]
    # scatter results back into Y/alpha by (point i, key) exactly as now.

class _ParallelEngineEvaluator:
    """Callable (x, seed)->dict AND batch (tasks)->list[dict]; owns a process pool."""
    def __init__(self, spec: _EngineEvaluatorSpec, n_workers: int): ...
    def __call__(self, x, seed) -> dict[str, float]:   # 1-element batch
    def evaluate_batch(self, tasks) -> list[dict[str, float]]:  # ordered like tasks
    def close(self) -> None
    def __enter__/__exit__                              # context-manager lifecycle

@dataclass(frozen=True)
class _EngineEvaluatorSpec:
    free_params: tuple[FreeParameter, ...]   # needed for point_to_overrides in-worker
    base_config_path: Path
    species: tuple[str, ...]
    enable_ssb: bool
    nyear: int | None
```

### Worker machinery (mirrors problem.py)

- Module-level `_worker_init(spec)` sets a process-global evaluator built once
  per worker via `make_engine_evaluator(list(spec.free_params),
  spec.base_config_path, spec.species, enable_ssb=spec.enable_ssb,
  nyear=spec.nyear)`. `FreeParameter` is a plain dataclass and pickles cleanly.
- Module-level `_worker_eval(task_id, x, seed) -> (task_id, dict)` calls the
  worker-global evaluator; only `(task_id, ndarray, int)` cross the boundary.
- `evaluate_batch` submits all tasks, collects by `task_id`, returns in input
  order.

### Determinism

Seeds are `seed_offset + i*n_seeds + k` — fixed per task, independent of
completion order. Parallel Y/alpha are bit-identical to serial. Results are
scattered back by task id, never by arrival order.

### Lifecycle

Pool is lazy-started on first `evaluate_batch`, persistent across grow rounds,
closed by the caller via context-manager (`with make_engine_evaluator(...,
n_workers=N) as ev:`). The full-run script owns this.

## Testing (TDD)

1. **Determinism** — a module-level *picklable* analytic batch-evaluator double
   (not the engine): `run_design` via `evaluate_batch` (real 2-worker pool)
   produces Y/alpha exactly equal to the serial callable path. Same X, same
   seeds.
2. **Batch order** — `evaluate_batch` returns results aligned to input task
   order even when workers finish out of order.
3. **Serial untouched** — `n_workers=1` returns a plain callable with no
   `evaluate_batch`; existing `run_design` behaviour unchanged (regression).
4. **Real-engine smoke** — `make_engine_evaluator(..., n_workers=2)` runs a tiny
   design (few points, nyear small) end-to-end; keys present, no crash.

## Out of scope (YAGNI)

- Broken-pool auto-rebuild (problem.py has it; the design run is one long job —
  let a worker failure propagate for now).
- Result caching.
- Parallelizing across grow rounds beyond pool reuse.
