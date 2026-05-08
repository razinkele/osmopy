# K3 — not shipping

## Measured delta
- Baseline: 5.025 s (eec_full 5 yr, 7-repeat median, post-#31 master)
- After K3: not implemented
- Estimated delta: **at best -0.4 %** (-20 ms)

## Why it didn't hit the gate

The K4 cProfile against eec_full 5-year shows `numpy.ndarray.copy` accounts for **30 ms across 2 480 calls = 0.41 % of run time**. That's the entire ceiling for K3 — even if pooling drove the copy cost to zero, the gain is unmeasurable against the established 100 ms / 2 % noise floor.

K3 was the smallest K-item by design (only the two function-local arrays `larva_deaths` and `inst_abd` were eligible; the other three escape to the caller via `state.replace`). The plan's r2 K-gate threshold for K3 was **≥ 2 %** — the cProfile measurement ceiling sits at 1/5 of that.

## Profile excerpt
```
0.030s  2 480 calls  numpy.ndarray.copy   (0.41 % of total)
0.026s 30 742 calls  numpy.zeros          (0.35 % of total — Python-side only)
```
For comparison, the top-3 hot spots:
```
1.290s  120 calls   mortality.py:1665(mortality)             17.6 %
1.269s  240 calls   accessibility.py:149(compute_school_indices)  17.3 %
1.230s  119 calls   movement.py:492(_precompute_map_indices)  16.7 %
```

The `.copy()` calls are negligible compared to the JIT'd kernel work and the accessibility/movement Python-side hot paths.

## Do not retry without
- A profiling pass against a fixture **larger than eec_full** (≥ 30 species, ≥ 5 000 schools steady-state). At eec_full's ~3 000-school scale, even a doubling of school count would not push K3 above the 2 % gate.
- A different baseline shape — e.g. a calibration scenario doing many short runs, where per-step setup overhead matters more than per-step kernel cost. K3's per-step copies might be relatively larger there. The current benchmark is one long run.
- Numba upgrade that exposes more JIT-internal work to cProfile, allowing the K3 surface to be re-evaluated against actually-measured kernel internals rather than the Python-side ceiling.

## Reference
- Plan: `docs/plans/2026-05-08-kernel-surgery-plan.md` § K3
- Profile: `docs/perf/2026-05-08-eec_full-5yr-profile.md`
- Benchmark baseline JSON: `/tmp/h6_baseline.json` (post-#31 master)
