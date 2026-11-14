# Pre-stated: worker × threads-per-worker sweep, 2026-11-14

Committed before any sweep run. This is item 1 of `docs/osmopy-performance-optimization-plan.md`.

## Question

Which combination of process-pool workers (W) and threads per worker (T) gives the most engine
evaluations per hour on this machine (Intel i9-10940X, 14 physical / 28 logical cores, 31 GB)? Does
pinning threads beat today's default, 8 workers at Numba's unrestricted thread count, by more than
run-to-run noise?

The 2026-07-15 thread-policy decision keeps workers unrestricted. It rests on a 4-worker
**thread-backend** measurement from April. The process-pool evidence against it is an
uncontrolled observation: a UQ run took 4.4 h uncapped against about 50 min pinned.

## Measurement

- **Harness:** `scripts/bench_pool_threads.py`, using `OsmoseCalibrationProblem` with
  `parallel_backend="process"` (forkserver) and `n_parallel=W`. The workload is the production
  Baltic config at its default 15-yr horizon, with one free parameter (the same objective and
  parameter as `scripts/benchmark_calibration.py`) and caching off.
- **Thread control:** `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
  `OPENBLAS_NUM_THREADS` are all set to T in the cell's launch environment, the convention the pinned
  UQ run used, and are inherited by the workers. "Unrestricted" means all four unset (Numba
  default: 28 logical cores). Each cell runs in its own process.
- **Timing:** one warm-up round of W evaluations is discarded (forkserver start, worker init, Numba
  cache load). Then R = 3 timed rounds of W evaluations run in one batch on the same pool:
  evaluations/h = 3W / t × 3600.
- **Also recorded:**
  - peak total RSS of the pool (all descendant processes, sampled every 0.5 s);
  - mean CPU cores used per worker during the timed batch;
  - the machine load average at the start.

## Grid

- **W ∈ {1, 4, 8, 12, 14, 16} × T ∈ {1, 2, unrestricted}.**
- **The split rule T = max(1, 14 // W)**, where it differs from those cells: (W=1, T=14) and
  (W=4, T=3).
- **20 cells, run sequentially, one engine job at a time.**

## Confirmation and decision

- **Confirmation:** the best two cells by evaluations/h and the default (W=8, unrestricted) are
  re-run interleaved (A B C, A B C, A B C), for 3 repeats each.
- **Adopt a pinned policy only if** the winner's median evaluations/h beats the default's median by
  more than the larger of the two cells' repeat spreads (max − min), **and** its peak total RSS is
  below 24 GB.
- **What would be encoded:** the thread rule the data support. That is either T = 1 per worker or the
  split rule, applied by the pools themselves, in:
  - `osmose/calibration/problem.py` `_worker_init`;
  - `osmose/calibration/uq/design.py` `_worker_init`;
  - the scipy-DE pool in `scripts/calibrate_baltic.py` if it does not inherit it.

  `tests/test_thread_policy.py:237,286` and the `osmose/engine/thread_policy.py` docstring change
  with it.
- **Otherwise:** record the negative in `docs/perf/` and leave the code alone.
- **Results do not depend on the thread count.** Mortality is bit-identical across thread counts
  (`tests/test_thread_policy.py:192`), so this is a throughput decision only.

## Known limits

- **Memory per worker at 15 yr** is below that of the 50-yr evaluations `calibrate_baltic.py` uses
  (about 400 MB). Peak RSS here is a lower bound for those runs.
- **The machine is shared with other projects' jobs.** The interleaved confirmation, not the grid
  pass alone, carries the decision.
- **Throughput at 15 yr is assumed to transfer to 50 yr** (per-step cost dominates). This is not
  separately measured.
