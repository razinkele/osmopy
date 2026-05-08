# K2 — not shipping

## Measured delta
- Baseline: 5.025 s (eec_full 5 yr, 7-repeat median, post-PR #31 master)
- After K2: not implemented
- Estimated delta: **at best -1.0 %** (-50 ms), high cost

## Why it didn't hit the gate

K2's two surfaces are:
1. Per-cell `cause_orders` array allocation (4 ints) — `mortality.py:1216` (sequential) and `mortality.py:1366+` (parallel).
2. 4-element Fisher-Yates shuffle on that array.

Aggregate workload per 5-year run on eec_full:
```
~100 cells × 24 steps × 5 yr × 1 subdt × 5 sites = ~60 000 calls
```
At 0.5–1 µs per JIT'd allocation (optimistic): **30–60 ms total = 0.6–1.2 % of run time**.
The shuffle calls (length-4) are inlined to single-digit-nanosecond cost in JIT — negligible.

Aggregate K2 ceiling: **~1 % of run time**. Plan r2 K-gate threshold: **≥ 5 %** of `mortality()` self-time. K2's ceiling is 5× below the gate.

## High implementation cost

K2 has the highest **non-performance** cost of any item in the kernel-surgery plan:

1. **RNG-stream divergence.** Pooling `cause_orders` and pre-shuffling means the in-loop draw sequence changes. Output is no longer bit-identical to current Python-engine outputs.
2. **Calibration cache invalidation.** The cache key at `osmose/calibration/problem.py:341-354` is keyed on `f"python-{__version__}"`. K2 forces a `osmose/__version__.py` bump → all calibration caches drop → re-computation runs across the calibration fleet.
3. **Reproducibility tests.** `simulation.rng.fixed=true` reproducibility tests (PR-protected) would need to be updated to a new "frozen" baseline after K2 lands.

For ~50 ms of estimated saving against a 5 025 ms baseline, this is firmly negative cost-benefit.

## Profile excerpt

cProfile cannot resolve the JIT'd `mortality()` internals, but the warm-cache profile shows:

```
1.290s  120 calls   mortality.py:1665(mortality)  [Python wrapper]    17.6 %
0.026s  30 742 calls  numpy.zeros  [Python-side only — JIT-internal allocs invisible]
```

The 30 K Python-side `np.zeros` calls in 7.34s of total time give a per-call cost of ~0.85 µs in Python. JIT'd `np.zeros` at known-small fixed shape is materially faster (estimated 0.2–0.5 µs). At those rates K2's 60K-call aggregate sits well below the 5 % gate against `mortality()`'s 1.29 s tottime.

## Do not retry without

- A fixture with **substantially more cells** (≥ 1 000) and/or **more sub-steps** (≥ 4 subdt) — workload scales linearly with cell × step × subdt counts, so a 10× larger fixture could push K2 toward (but probably not past) the gate.
- A direct **JIT-internal profiling pass** (e.g. `numba.runtests` line-level profiling, or a one-shot Python-mode rerun) to confirm the kernel-internal cost rather than the Python-side estimate. The current evidence is bounded above by the Python-side micro-bench.
- A separate cost-benefit reset: the RNG-stream divergence + cache-invalidation tax cannot be hidden. K2 only makes sense if the measured saving is **≥ 5 %** of the total run, not just the gate against `mortality()` self-time.

## Recommendation if K1 ships

If K1 ships and K2 is reconsidered as a follow-on, **bundle them into a single PR**: K1 already pays the kernel-modification + parity-test review cost, and K2 alone cannot justify another round. But run the K1 PR's measurement first and decide K2 from the residual.

## Reference

- Plan: `docs/plans/2026-05-08-kernel-surgery-plan.md` § K2
- Profile: `docs/perf/2026-05-08-eec_full-5yr-profile.md`
- Cache key: `osmose/calibration/problem.py:341-354`
- RNG reproducibility contract: `osmose/engine/rng.py` module docstring
