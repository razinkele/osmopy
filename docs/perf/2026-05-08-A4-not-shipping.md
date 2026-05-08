# P3 / A4 — not shipping

`compute_feeding_stages` does not measurably appear in the post-P2 hot path on eec_full 5-yr.

## Profile measurement

cProfile of post-P2 master (`c59ae85`) on eec_full 5-yr (warm cache):

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
   120    0.004    0.000    0.008    0.000  feeding_stage.py:20(compute_feeding_stages)
```

- **tottime: 0.004 s** = 0.11 % of the 3.564 s profiled run = **0.14 % of the 2.881 s production run**.
- 67 µs per call — already inexpensive thanks to the existing per-species `np.searchsorted` vectorisation at `feeding_stage.py:88`.

Plan-r2 gate threshold: **≥ 2 % of run time**. Measured ceiling is **14× below** that. Even with full elimination, the saving is below the 100 ms / 2 % noise floor.

## Why it's already vectorised

`compute_feeding_stages` (`osmose/engine/processes/feeding_stage.py:20-89`) iterates ~15 species in a Python loop with vectorised inner work:

```python
for sp_idx, thresholds in self._stages_by_role.items():
    mask = species_id == sp_idx
    if not mask.any():
        continue
    sorted_thr = np.sort(thresholds)
    stages[mask] = np.searchsorted(sorted_thr, values, side="right").astype(np.int32)
```

Same template as A1's `compute_school_indices` — converts an n_schools-Python-loop into an n_species-Python-loop with vectorised inner ops. The python-side perf plan (PR #34) deferred this as A4 with a "needs profile pass first" criterion before re-attempting; the post-P2 profile makes the answer clear.

## Decision

**DROP.** No code change. The function has no measurable optimisation surface left on the eec_full benchmark.

## Do not retry without

- A scenario where per-species mask construction dominates — e.g. ≥ 100 species (eec_full has 14, baltic has ~16 focal + 5 background). At 100 species the per-species loop overhead might cross the gate.
- A profile against a different fixture that exercises the function more heavily (e.g. a tight calibration loop with many short runs where construction-time cost amortises poorly).
- A new functional requirement that adds complexity to the inner loop (e.g. multi-trait stages) — re-evaluate then.

## Reference

- Plan: `docs/plans/2026-05-08-post-v012-perf-plan.md` § P3
- Predecessor decision: `docs/plans/2026-05-08-python-side-perf-plan.md` § "Deferred (post-A3)" A4 candidate
- Profile data: `/tmp/post_v012.prof` (post-P2 master, eec_full 5-yr warm cache)
