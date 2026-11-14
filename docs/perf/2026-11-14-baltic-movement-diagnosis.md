# Baltic movement: where the time goes, 2026-11-14

This is item 2 of `docs/osmopy-performance-optimization-plan.md`. The 2026-11-14 profile put movement
at 25% of a threaded 10-yr Baltic run and 11% of a one-thread run. The plan's first step was to
diagnose before writing code. Its exit criterion: either a performance-preserving change worth at
least 2% on Baltic in every interleaved pair, or a note recording that the cost is inside the
serial kernel.

**Result: the cost is inside the serial seeded kernel.** No performance-preserving change of 2% or
more is available, so no code changes.

## Measurement

- **Setup:** production Baltic config, 5-yr run, warm (one warm-up run discarded), 14 threads
  (`apply_single_run_threads`), seed 42, current master (`f288de79`).
- **First pass:** cProfile by own time.
- **Second pass:** a wrapper timing the Numba map kernel `_map_move_batch_numba` separately from
  the rest of `movement()`.

| | time | share of the 4.5 s run |
|---|---:|---:|
| `movement()` total, 120 steps | 1.021 s | 23% |
| of which the Numba map kernel | **0.942 s** | **21%** |
| of which everything else in `movement()` | 0.079 s | 1.8% |

- **The kernel handles about 7,400 map-moved schools per step,** roughly 1 µs per school, serially.
- **The Python side is small:** the map-index precompute (`_precompute_map_indices`, 0.07 s), array
  copies and the salinity-weight lookup. Removing it entirely would save at most 1.8%, below the 2%
  bar.
- **A suspected hot spot did not show up.** `movement()` has a Python loop over every map-moved
  school that separates null-map migrations from placement failures. It runs only on steps where
  schools newly leave the domain, and on this run no step did (0 of 120), so it costs nothing here.

## Why the kernel is not a performance-preserving target

- **One serial random stream.** `_map_move_batch_numba` seeds NumPy's generator once per call
  (`np.random.seed(rng_seed)`) and draws positions school by school, in school order, with
  rejection sampling against each map.
- **What breaks the baselines.** Running schools in parallel, reordering them, or replacing the
  rejection sampler (for example with an alias table) changes which numbers each school draws. That
  moves the fixed-seed baselines in `tests/test_engine_parity.py`, so by the plan's classification it
  is a **statistically-equivalent** change, not a performance-preserving one.
- **What such a change would need:**
  - an independent stream per school or per species, for example counter-based, keyed by school
    and step, so the draws no longer depend on visit order;
  - then a `prange` over schools;
  - regenerated fixed-seed baselines and a multi-seed distribution comparison with pre-set
    tolerances.

  Expected gain is bounded by movement's share: about 21% of a threaded 5-yr Baltic run, and less
  at one thread, where mortality dominates.

## Not measured

- **Within-kernel split.** The time inside the kernel between same-map random walks and
  map-change rejection sampling was not measured. Per-school acceptance rates would show whether
  sparse maps drive the cost.
- **Other horizons and configs.** Only a 5-yr Baltic run was timed. The 10-yr profile's 25% share
  is consistent with the 23% here.
