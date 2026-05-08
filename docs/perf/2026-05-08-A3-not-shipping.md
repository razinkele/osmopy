# A3 — not shipping

## Measured delta

- Baseline (post-A2 master): 3.030 s eec_full 5-yr 7-repeat median
- After A3: 3.050 s
- Delta: **−0.7 % (within 100 ms / 2 % noise floor)**

## Why it didn't hit the gate

A3 fused the prey + pred double-call into a single sweep (`compute_school_indices_both`). The estimated 1-4 % saving came from sharing the per-species mask + age-conversion work between roles.

After A1 vectorised `compute_school_indices` (PR #35, 17.7 % wall-time saved), the remaining per-call cost dropped from ~1460 µs to ~290 µs. Halving the already-small overhead via fusion produced a saving below the 2 % gate's measurement resolution:

```
A2 baseline:   3.030 s  (606 ms/yr)
A3 candidate:  3.050 s  (610 ms/yr)
Difference:    +20 ms (0.7 %)  ← within noise floor
```

The 12/12 parity tests pass bit-exact and all 6 new tests pass — the fusion is **functionally correct** but the optimisation target was already collapsed by A1.

## Cumulative perf-plan outcome

A1 + A2 alone delivered **37.8 % wall-time reduction** on eec_full 5-yr (4.872 s → 3.030 s), already 1.6× the plan's top-of-range cumulative target (23 %) and 7.6× the floor target (5 %). A3's incremental contribution is in the noise — shipping it would add ~50 lines of code (fused method + tests) for no measurable gain.

| Item | Plan estimate | Measured | Status |
|---|---:|---:|---|
| A1 | 2-7 % | **17.7 %** | shipped (PR #35) |
| A2 | 2-12 % | **24.4 %** | shipped (PR #36) |
| A3 | 1-4 % | 0.7 % | **DROP** |
| Cumulative | 5-23 % | **37.8 %** | exceeded ceiling |

## Do not retry without

- A scenario where the per-species mask + age-conversion work is materially larger than on eec_full. After A1 the per-call cost is already 290 µs / call × 240 calls/run = 70 ms total — fusing this can save at most ~35 ms over a 3-second run, which is the noise floor.
- A different consumer (e.g., a calibration scenario doing many short runs where construction-time cost amortises poorly). The current eec_full 5-year is one long run.
- A profile re-take after A1 + A2 land — the post-A2 hot-path has shifted, and `compute_school_indices` is no longer in the top-3 by tottime. Confirm there's a real surface to optimise before re-attempting.

## Predation_for_cell update — also dropped

The plan added `predation.py:573-586` to A3's scope to keep the test/benchmark entry point in sync. With A3 dropped, the production path (`mortality.py:1726-1742`) keeps the two-call pattern. `predation_for_cell` is not in the production simulation hot path per its own module docstring, so divergence here has no measurable effect — the two call sites simply remain duplicated until a future perf round revisits the cumulative budget.

## Reference

- Plan: `docs/plans/2026-05-08-python-side-perf-plan.md` § A3
- Profile predecessor: `docs/perf/2026-05-08-eec_full-5yr-profile.md`
- A1: PR #35 — vectorised `compute_school_indices` (17.7 % gain)
- A2: PR #36 — vectorised `_precompute_map_indices` (24.4 % gain)
