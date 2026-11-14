# Worker × threads-per-worker sweep for the engine process pools, 2026-11-14

This is item 1 of `docs/osmopy-performance-optimization-plan.md`. Criteria were committed before any
run: `docs/diagnostics/worker_thread_sweep_prestated_2026-11-14.md` (`a2d9b1b6`). Harness:
`scripts/bench_pool_threads.py` (`a3404c12`).

**Result: the pinned policy is adopted.** Fourteen workers at one thread each give **492
evaluations/h**, against **124/h** for today's default of 8 workers at Numba's unrestricted thread
count: **3.96×**, in interleaved repeats with a spread of at most 5.3/h.
- **Encoded rule:** each of W workers now runs max(1, physical cores // W) threads.
- **Where it is applied:** the calibration pool, the UQ pool and `calibrate_baltic.py`'s DE path
  apply it themselves.

## Setup

- **Machine:** Intel i9-10940X, 14 physical / 28 logical cores, 31 GB.
- **Workload:** the production Baltic config at its default 15-yr horizon, evaluated through
  `OsmoseCalibrationProblem(parallel_backend="process")` (forkserver), with one free parameter.
- **Thread control:** threads per worker were set with `NUMBA_NUM_THREADS` / `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` in each cell's launch environment.
- **Timing:** one warm-up round of W evaluations was discarded, then 3 timed rounds of W.

## Grid: evaluations per hour (seconds per evaluation)

| workers | T = 1 | T = 2 | unrestricted | split rule T = max(1, 14 // W) |
|---:|---:|---:|---:|---:|
| 1 | 52 (70 s) | 58 (62 s) | 120 (30 s) | T=14: 122 (30 s) |
| 4 | 187 (77 s) | 205 (70 s) | 131 (110 s) | T=3: 245 (59 s) |
| 8 | 335 (86 s) | 357 (81 s) | **124 (233 s)**, the default | T=1: 335 (86 s) |
| 12 | 429 (101 s) | 404 (107 s) | 117 (369 s) | T=1: 429 (101 s) |
| 14 | **495 (102 s)** | 466 (108 s) | 125 (404 s) | T=1: 495 (102 s) |
| 16 | 460 (125 s) | **487 (118 s)** | 118 (487 s) | T=1: 460 (125 s) |

Peak pool memory was at most 6.5 GB (16 workers), against the pre-stated 24 GB limit. Every
evaluation returned a finite objective.

What the grid shows:
- **Unrestricted workers oversubscribe the machine.** At any worker count from 4 up, throughput sits
  at 117–131/h, about one fully threaded single run. Each worker spawns up to 28 threads, so 8
  workers contend for 14 cores and each evaluation takes 233 s instead of 86 s.
- **A single run scales poorly with threads.** One run at 14 threads is only 2.3× faster than at
  one thread (30 s vs 70 s per evaluation). The unrestricted default uses twice the CPU of 14
  threads (14.2 vs 6.8 cores) for the same speed.
- **Pinned workers scale until the physical cores run out.** Throughput rises to 495/h at 14
  workers × 1 thread, one worker per physical core. At 16 workers the extra workers spill onto
  hyperthreads (0.86 cores each at T = 1), and 2 threads per worker recover slightly (487/h).
  Per-evaluation time also rises with worker count (70 s alone, 102 s at 14 workers), most likely
  memory bandwidth (inferred, not measured).
- **The split rule is best or near-best everywhere.** It picks the best setting at 1, 4, 12 and 14
  workers and is within 6.5% of the best at 8 and 16. T = 1 everywhere would be much worse with few
  workers (52 vs 122/h at 1 worker, 187 vs 245/h at 4).

## Confirmation (interleaved, 3 repeats)

| cell | repeats (evals/h) | median | spread |
|---|---|---:|---:|
| 14 workers × 1 thread | 494.1, 491.5, 492.2 | **492.2** | 2.6 |
| 16 workers × 2 threads | 482.7, 485.9, 485.1 | 485.1 | 3.2 |
| 8 workers × unrestricted (default) | 124.3, 125.2, 119.9 | 124.3 | 5.3 |

The pre-stated criterion was that the winner's median must beat the default's by more than the
larger repeat spread, with peak RSS under 24 GB. **It passes**: 492.2 − 124.3 = 368 evaluations/h
against a spread of 5.3, at 5.7 GB.

## What changed

- **`osmose/engine/thread_policy.py`:**
  - `worker_thread_count(W)` returns the split rule. `OSMOSE_WORKER_THREADS` overrides it: a positive
    integer forces that count, capped to the logical budget, and `unrestricted` restores Numba's
    default.
  - `apply_worker_threads(W)` applies the rule **in the worker**, setting Numba's thread-local count
    and, when `threadpoolctl` is present, the BLAS/OpenMP pools.
- **`osmose/calibration/problem.py`:** `_EvalSpec` carries the resolved worker count, and the
  forkserver `_worker_init` applies the rule.
- **`osmose/calibration/uq/design.py`:** the spawn pool's `_worker_init` applies the rule.
- **`scripts/calibrate_baltic.py`, DE path:**
  - An integer `workers` now gets a fork `Pool` created here, with `apply_worker_threads` as its
    initializer, passed to scipy as `workers=pool.map`. `-1` resolves to all CPUs, as scipy does.
  - The isolated-eval map's children apply the rule themselves (`_eval_child`).
  - Nothing in the parent touches Numba's threads (see the correction below).
- **`tests/test_thread_policy.py`:**
  - The static guard asserting that calibration workers do *not* set a thread count is replaced by
    tests of the rule, the override and its cap, a forkserver worker, the spec's worker count, and
    the pool initialisers and DE path.
  - A fresh-interpreter test checks that a fork pool with the initializer runs a real `prange`
    kernel at the rule.
  - The single-run-cap leak test stays, with its docstring updated.
- **Not covered:** the CMA-ES and surrogate-DE optimizers parallelise through joblib's loky backend,
  which spawns fresh interpreters that do not inherit the parent's setting. They still depend on
  the launch environment. That is a follow-up.

## A correction during review: never set threads in a parent that forks

The first version set the rule in the DE parent, through a context manager around
`differential_evolution`, so that scipy's forked workers would inherit it. CodeRabbit flagged a real
hazard, which was reproduced here:
- **The cause:** this machine's Numba uses the GNU OpenMP threading layer. Any Numba thread call in
  a process (`get_num_threads`, `set_num_threads`, or a parallel kernel) starts the layer there.
- **The effect:** a child forked afterwards is killed when it runs a parallel kernel. It prints
  "Terminating: fork() called from a process already using GNU OpenMP, this is unsafe", and the pool
  hangs. Every engine evaluation runs the parallel mortality kernel.
- **Why the tests missed it:** they only had children read `get_num_threads()`, which never runs a
  parallel region.

The fix applies the rule only inside the children (pool initializer, isolated child). The
regression test runs a real `prange` kernel in fork children from a fresh interpreter. Re-introducing
a parent-side call makes it fail with the same termination message (checked by mutation). The
forkserver and spawn pools were never affected: their workers do not fork from a parent that has
used Numba.

## Verification of the encoded rule

The rule was re-measured through the pools themselves, with every thread variable unset
(`scripts/bench_pool_threads.py`, cells `policy`):

| cell | evals/h | s per eval | cores per worker |
|---|---:|---:|---:|
| 8 workers, policy | 338.2 | 85.2 | 0.99 |
| 8 × 1, env-pinned (grid) | 335.1 | 85.9 | 0.99 |
| 14 workers, policy (paired run 1) | 494.2 | 102.0 | 0.96 |
| 14 × 1, env-pinned (paired run 1) | 492.6 | 102.3 | 0.96 |
| 14 workers, policy (paired run 2) | 488.8 | 103.1 | 0.96 |
| 14 × 1, env-pinned (paired run 2) | 452.5 | 111.4 | 0.93 |

- **The built-in rule reproduces the env-pinned result:** 494.2 against 492.6 in the clean pair, and
  338 against 335 at 8 workers.
- **The two low readings (452.5 here, and 426.5 at 0.87 cores per worker in a first unpaired policy
  run) coincided with another project's test suite using one to two cores.** That is shown by the
  drop in cores per worker below 0.96 and by the processes seen running at the time. They are
  contention, not the policy; the first unpaired run is excluded for that reason.
- **The threading layer is Numba's OpenMP backend,** which requests the current count per parallel
  region. A runtime `set_num_threads(1)` therefore gives a one-thread team, as the environment
  cap does.

## Recommendations not done here

- **Run calibrations at 14 workers.** The default worker count in `calibrate_baltic.py`
  (`OSMOSE_DE_WORKERS`, 8) was not changed; that is a separate decision. With the rule, 8 workers
  give 335/h and 14 give 492/h. At 50 yr each worker holds about 400 MB, so 14 workers need about
  5.6 GB.
- **Throughput at 15 yr is assumed to transfer to the 50-yr calibration horizon.** This is not
  separately measured.
