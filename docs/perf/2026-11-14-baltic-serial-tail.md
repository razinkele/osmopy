# Baltic serial tail: profile and the life-stage collector, 2026-11-14

This is item 4 of `docs/osmopy-performance-optimization-plan.md`: state initialisation, loop
overhead and output collection, put at about 14% of a threaded Baltic run.

**Result: nothing ships.**
- **The one sizeable bit-exact edit falls short.** That is the life-stage collector rewritten with
  `np.bincount`. It is bit-identical and saves 0.39 s per 10-yr run, but it misses the plan's bar
  (a median gain of at least 2% in every interleaved pair) in one pair at each thread count.
- **No other single bit-exact edit could clear the bar either.** In the calibration regime, one
  thread per worker since item 1, no other tail function is over 1.2% of the run.
  `_collect_outputs` is 2.2%, but it is a dozen collectors, none over 0.7%.
- **One lever is left:** an opt-in mode that skips the output collectors a calibration objective
  does not read. All output collection together is 5.1% of a one-thread run.

## Where the tail goes, at one thread

- **Setup:** warm 10-yr Baltic `run_in_memory`, `numba.set_num_threads(1)`, seed 42, master
  `40250aed`, under cProfile.
- **Timing:** 33.11 s profiled against 32.57 s unprofiled.
- **Shares are of the profiled `simulate()` time.**

| | time | share |
|---|---:|---:|
| `mortality` (Numba kernels) | 26.04 s | 78.6% |
| `movement` | 3.74 s | 11.3% |
| `_collect_outputs` | 0.72 s | 2.2% |
| of which `_collect_distributions` | 0.24 s | 0.7% |
| of which `_collect_mortality` | 0.21 s | 0.6% |
| `_collect_by_life_stage` | 0.58 s | 1.8% |
| `_growth` | 0.38 s | 1.2% |
| `aggregate_diet_by_species` | 0.36 s | 1.1% |
| `ResourceState.update` | 0.35 s | 1.0% |
| `_reproduction` | 0.26 s | 0.8% |
| `MovementMapSet.__init__` (9 calls, setup) | 0.16 s | 0.5% |
| `SchoolState.compact` | 0.14 s | 0.4% |
| `SchoolState.append` | 0.09 s | 0.3% |
| `BackgroundState.get_schools` | 0.08 s | 0.2% |

**What "state init + loop overhead" was.** The section 4.1 profile reports a 1.50 s remainder at
one thread. It is not loop overhead in the interpreter sense. It is mostly named functions outside
that profiler's phase list: the diet aggregation, `ResourceState.update`, the distribution
collectors, map loading, `compact` and `append`.

**Output collection as a whole** (`_collect_outputs`, `_collect_by_life_stage`,
`aggregate_diet_by_species`, background and averaging steps): 1.68 s, 5.1% of the one-thread run.
The share is larger in a threaded run, where the kernels shrink and these functions do not.

## Candidate: `_collect_by_life_stage` with `np.bincount`

- **Why it was slow.** The collector called `np.add.at` with tuple indices, nine calls per step,
  which misses numpy's 1-D fast path. `np.add.at` accounts for 0.43 s of its 0.58 s.
- **The change.** Flattened `(species, stage)` bins summed with `np.bincount`, as
  `aggregate_diet_by_species` already does. It is on branch `perf/life-stage-bincount` (commit
  `d535aa81`), not merged.
- **Bit-identical.** `bincount` adds each bin's weights in array order starting from zero, as
  `np.add.at` does. Checked three ways:
  - in-process on every step of a warm 10-yr Baltic run, 240 of 240, by byte comparison;
  - by `tests/test_engine_parity.py`;
  - by a new test against the original code on data spanning 32 orders of magnitude.
    Mutation-checked: reversed summation order or float32 accumulation fails it.
  - The new test also caught a numpy quirk: with no focal school, `bincount` returns int64 zeros
    even with float weights.
- **Saving, measured in-process on identical states:** 0.579 s → 0.190 s per 10-yr run, so
  0.389 s. That is 1.2% of a one-thread run and about 2.6% of a threaded one.

**The A/B protocol** (5-yr Baltic `benchmark_engine.py`, 7 repeats per arm, interleaved; control =
`40250aed`):

| | pair 1: control → bincount | change | pair 2: control → bincount | change |
|---|---|---:|---|---:|
| automatic threads (14) | 4.121 → 4.097 s | −0.6% | 4.116 → 3.960 s | −3.8% |
| `NUMBA_NUM_THREADS=1` | 9.913 → 9.525 s | −3.9% | 9.716 → 9.623 s | −1.0% |

- **The verdict.** Every pair favours the change, but one pair at each thread count is under 2%,
  so it does not meet the exit criterion and does not ship.
- **A limit of the protocol.** Years 1–5 carry fewer schools than years 6–10: 29.7% of the
  10-yr run's school-steps, from the item 3 counts. Scaling the saving by that share, assuming its
  cost is linear in schools, gives roughly 0.12 s at 5 yr: about 3% threaded and 1.2% at one
  thread. With a 4–8% spread per arm,
  two pairs cannot reliably resolve an effect that size against a 2% bar. The in-process paired
  number is the reliable measure of the edit; the A/B is the gate the plan set, and it was not
  met.

## What is left for item 4

- **Opt-in collector skipping for calibration.** `run_in_memory` computes every enabled output,
  including diet and size or age distributions, that a biomass objective never reads. Together
  these are up to about 5% of a one-thread run. This changes what `OsmoseResults` serves, so it
  must be an explicit opt-in. Its test must show that skipped outputs are absent, not silently
  zero.
- **Bundling bit-exact collector edits.** `d535aa81` plus the remaining 1-D `np.add.at` call sites
  (0.21 s of `ufunc.at` time over the run, mostly per-call overhead) could be measured together
  as one edit. Unmeasured; the expected total is about 0.4–0.5 s per 10-yr run.

## Files

`docs/diagnostics/serial_tail_2026-11-14/` holds the eight A/B result files (hostname removed).
- **A whole-run check:** all eight record identical final biomass for every species, control and
  treatment and both thread counts alike.
- **Reading the `numba_threads` field:** the one-thread files record 0. Under
  `NUMBA_NUM_THREADS=1`, `apply_single_run_threads` asks for 14, Numba refuses, and the policy falls
  back to Numba's default, which that variable sets to one thread. The 2.4× longer timings confirm
  it.
