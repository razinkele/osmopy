# Phase 1a length prefilter, 2026-11-17

Item 6 of `docs/osmopy-performance-optimization-plan.md`. The 2026-11-15 note
(`docs/perf/2026-11-15-phase1a-scan-cost.md`) put the size-window scan over school prey in
`_apply_predation_numba` at a quarter of one-thread mortality time. Most of that is spent rejecting
prey: on the Baltic 88% of scanned schools fail the size test. This note measures three versions of
a length prefilter that skips them.

**Result: v3 ships.** It is bit-identical to master and cuts a one-thread 10-yr Baltic run by
10.6% (`mortality()` −13.3%), the calibration worker regime since item 1. At the automatic thread
count and on EEC full it is neutral.

| version | what a predator visit does | one-thread `mortality()` | its session's control | change |
|---|---|---:|---:|---:|
| v1 | binary-searches a per-cell index rebuilt every sub-step, then sorts its candidates | 25.56 s | 25.37 s | +0.7% |
| v2 | the same, but the index is built once per step | 23.92 s | 25.06 s | −4.5% |
| **v3** | **walks its own candidate list, built once per step** | **21.95 s** | **25.32 s** | **−13.3%** |

Each row is the mean of four interleaved pairs, each arm in its own interpreter: a warm 10-yr Baltic
`run_in_memory`, seed 42, one Numba thread, against master `989eeb7d`. The control mean moved by up
to 0.43 s between sessions, so each version is compared with its own session's control. Every arm,
on both sides, gave the same biomass hash. v3's wall time: 28.49 s against 31.87 s, −10.6%, each
adjacent control/v3 pair between −9.4% and −11.7%.

## Why v1 saved nothing

A third arm built v1's machinery and then ran the original full scan anyway, so its cost over
control is the prefilter's own bookkeeping.

| arm | `mortality()` |
|---|---:|
| control | 25.49 s (3 runs) |
| v1 machinery + full scan | 30.56 s (3 runs) |

The bookkeeping cost about 5.1 s. So skipping the rejected pairs saved about 4.9 s
(25.37 + 5.07 − 25.56), in line with the 3.8–4.4 s the 2026-11-15 note estimated. The saving was
there all along; v1's bookkeeping ate it:
- It rebuilt each cell's length index on every sub-step (10 on the Baltic), with an argsort and
  four allocations each time.
- Each predator visit did two binary searches and an insertion sort of its candidates.

**v2** builds the index once per step: a predator's length does not change during mortality. That
recovered about 1.2 s.

**v3** notes that the candidate list itself does not change during the step either. A predator's
length, feeding stage and size window are all fixed. So `_cell_prey_candidates` builds every
predator's list once per step, in a flat array with per-school offsets, already sorted back into
cell order. A visit just walks its slice. The build costs 0.60 s over the 10-yr run (240 calls),
at any thread count.

## Why it stays bit-identical

The kernel's exact tests are unchanged. They run only on the predator's list, in the cell's own
order. Three properties make that identical to the full scan:
- **The list is a superset of the schools the size test accepts.** Its length bounds are
  `pred_len / r_max` and `pred_len / r_min`, widened by 1e-9 relative.
  - The widening is load-bearing. A search found prey one ULP above `pred_len / r_min` that the
    exact test still accepts, because `pred_len / prey_len` rounds onto `r_min`. An unwidened bound
    drops them. These cases are now in the test.
- **The list is in ascending cell order.** `total_available` is a running sum, and Phase 2 walks
  prey in buffer order.
- **Nothing else moves:**
  - the per-cell random streams, permutations and cause orders are drawn as before;
  - `n_local` is unchanged;
  - abundance is still tested at every visit.

Degenerate windows fall back to the exact tests:
- a NaN ratio lists every school;
- an `r_max` at or below zero lists none (no positive ratio can pass);
- an `r_min` at or below zero has no upper length bound.

A predator too young to feed gets an empty list, as the kernel returns before its scan anyway.

## Checks

- **Exact comparisons, all passing:** `tests/test_engine_parity.py`, `tests/test_engine_output_parity.py`, the
  kernel equivalence tests (`tests/test_engine_bioen_numba_kernel.py`, per-cell oracle vs both
  batch kernels, bioen on and off), `tests/test_mortality_rng.py` and
  `tests/test_engine_functional_response.py`. 204 passed.
- **New, `tests/test_phase1a_prefilter.py`:** the builder against a brute-force reference.
  - Every list is ascending and contains every accepted prey. This includes lists longer than 32
    (the `np.sort` branch), which no parity baseline produces.
  - The degenerate windows above, the rounding cases, and a check that the lists actually filter.
- **Mutations, each caught:**

  | mutation | caught by |
  |---|---|
  | a list missing its last candidate | the parity baselines (8 failures) and the new test (11) |
  | candidates in length order instead of cell order | the parity baselines (4 failures) |
  | long lists left unsorted | the new test (7) |
  | no widening of the upper bound | the new test (3, exactly the rounding cases) |
  | no filtering at all | the new test (1) |

- **Other configs and thread counts:**

  | run | control | v3 |
  |---|---:|---:|
  | Baltic 10-yr, automatic threads (28), wall | 13.85 / 14.49 / 13.94 s | 13.99 / 13.84 / 13.90 s |
  | EEC full 10-yr, one thread, wall | 6.96 / 7.13 / 6.99 s | 6.95 / 6.93 / 7.17 s |

  Both are neutral, with identical hashes.

## Left

- **At the automatic thread count the serial 0.60 s build cancels the scan saving.** A `prange`
  build could take most of it back (about 0.5 s of a 13.9 s Baltic run). It needs its own parallel
  variant: calibration's forked one-thread workers must not call a `prange` function (see the
  worker-thread note).
- **EEC gains nothing.** Its cells are sparser (median 15 schools), so there is less to skip.

## Reproduce

Each arm times one warm run in its own interpreter. Interleave control and prototype:

    .venv/bin/python scripts/baltic_phase1a_cost.py time <checkout>

`time` works on any checkout, and prints `mortality()` seconds and a biomass hash.
