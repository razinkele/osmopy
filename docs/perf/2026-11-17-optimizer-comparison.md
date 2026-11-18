# Optimizer comparison at a matched budget: results (perf plan item 5b)

The design and rule were committed before any run: `docs/perf/2026-11-17-optimizer-comparison-prestated.md`.
- Harness: `scripts/optimizer_comparison.py`.
- Raw results (every run's summary and the re-rank): `docs/diagnostics/optimizer_comparison_2026-11-17.json`.

**Setup:**
- Problem: calibration phase 1b, 9 log-space parameters, `make_objective` on the production Baltic config, 40 yr, training seed 42.
- Start point: x0 (R18). Its objective is 1156–1724, the collapse class; production's own values score 0.7190 ± 0.0033 on seeds 1–5.
- Budget: 224 real evaluations per run, 14 workers, 3 replicates per optimizer (seeds 1, 2, 3).
- Re-rank: each run's final best evaluated on seeds 1–5.
- Runs: 2026-11-17 16:55 to 2026-11-18 02:00.

## Results

| optimizer | rep | train best (seed 42) | best-so-far @74 / @149 / @224 | re-ranked mean ± SD (seeds 1–5) | nfev | sentinels |
|---|---:|---:|---|---:|---:|---:|
| cmaes | 1 | 0.9606 | 0.961 / 0.961 / 0.961 | 0.9543 ± 0.0039 | 224 | 0 |
| cmaes | 2 | 0.7166 | 0.940 / 0.822 / 0.717 | 0.7146 ± 0.0012 | 224 | 0 |
| cmaes | 3 | 0.7401 | 1.250 / 0.809 / 0.740 | 0.7328 ± 0.0034 | 224 | 0 |
| de | 1 | 1.1301 | 4.115 / 1.165 / 1.130 | 1.1288 ± 0.0042 | 224 | 0 |
| de | 2 | 0.9896 | 2.320 / 1.381 / 0.990 | 0.9845 ± 0.0080 | 224 | 0 |
| de | 3 | 1.2650 | 6.062 / 1.265 / 1.265 | 1.2784 ± 0.0279 | 224 | 0 |
| surrogate-de | 1 | 0.9901 | 2.410 / 2.213 / 0.990 | 0.9876 ± 0.0125 | 224 | 0 |
| surrogate-de | 2 | 2.5917 | 3.267 / 3.267 / 2.592 | **154.69 ± 297.07** | 224 | 0 |
| surrogate-de | 3 | 1.3443 | 4.446 / 2.043 / 1.344 | 1.3421 ± 0.0276 | 224 | 0 |

All 2,016 evaluations returned a real objective (no sentinels), and every run logged exactly 224.

**The rule's verdict: no ordering for any pair** (CMA-ES–DE, CMA-ES–surrogate-DE, DE–surrogate-DE).
The pre-stated threshold is 2 × the pooled re-rank seed SD, which is **198.0**. One run sets it.
Surrogate-DE replicate 2 picked a point that scores 2.59 on the training seed but re-ranks to
2.61, **82.97**, 2.58, 2.84 and **682.4** on seeds 1–5. It collapses on two of the five seeds.
Its seed SD of 297 dominates the pooled value.

## Reading (descriptive; none of this is a claim the rule supports)

1. **CMA-ES reached production's level in two of three replicates.** It re-ranked at 0.7146 and
   0.7328, against 0.7190 ± 0.0033 at production's values. Replicate 2 is slightly *below*
   production on all five seeds. Replicate 1 found 0.961 within its first 28 evaluations and
   never improved; its later generations kept sampling the collapse plateau.
2. **No DE replicate reached production's level** in 224 evaluations: 0.98 to 1.28, improving
   steadily, with the last gains still arriving at evaluation 224.
3. **Surrogate-DE** ended at 0.99 and 1.34, plus the seed-fragile point above.
4. **Exploratory only, decided after seeing the data, so not a result.** Without surrogate-DE
   replicate 2 the pooled SD would be 0.015, and 2 × 0.015 = 0.030. CMA-ES's worst replicate
   (0.954) would then beat DE's best (0.985) by 0.031, and surrogate-DE's best (0.988) by 0.033.
   That is barely over a threshold the rule did not specify, on three replicates. It motivates a
   larger CMA-ES study; it is not an ordering.

## Lessons for the next comparison

- **Training on one seed selects knife-edge points.** Surrogate-DE replicate 2's best was
  real on seed 42 and a collapse on seeds 2 and 5. A re-rank on held-out seeds is not
  optional for any optimizer that trains on a single seed. Item 5a's common-random-number mode
  (`OsmoseCalibrationProblem(crn_seeds=...)`) targets this; the scalar `make_objective` path
  used here has no equivalent yet.
- **A pooled SD lets one fragile run set the threshold for every pair.** A future rule should
  pool robustly (for example, the median of per-run SDs) or treat a run whose re-rank spread
  exceeds its mean as fragile and report it separately. This rule was fixed in advance and is
  applied as written.

## Wall time (descriptive only; no verdict uses it)

| run | seconds | note |
|---|---:|---|
| de_1 | 3271 | clean |
| cmaes_1 | 3980 | about 1–2 min of other load (a .NET SDK install) |
| surrogate-de_1 | 3253 | **contaminated:** C# builds and tests and a 4-minute Numba compile overlapped it |
| cmaes_2 | 4410 | **contaminated:** C# mutation loops and fixture captures, about 19:45–19:55 and 20:25–20:31 |
| surrogate-de_2 | 3014 | clean |
| de_2 | 3306 | clean |
| surrogate-de_3 | 3397 | clean |
| de_3 | 3254 | clean |
| cmaes_3 | 4262 | clean |

On the clean runs, CMA-ES takes about 1.3× DE's wall time for the same 224 evaluations: 4262 s
against 3254–3306 s. A CMA-ES generation is 14 evaluations on 14 workers, so each generation
waits for its slowest simulation. DE's population of 28 fills the pool twice per generation.
Results (`f`) are unaffected by the other load: there is no per-evaluation timeout that load
could turn into a sentinel.

## Decision (the plan's exit criterion)

The results table now exists. **It does not justify changing any default:**
- the pre-stated rule orders no pair;
- DE stays the default in `calibrate_baltic.py`;
- a staged screen → surrogate → re-rank workflow is not wired in.

**Worth doing next, if calibration throughput becomes the bottleneck:**
- a CMA-ES study with more replicates, because two of three reached production's level within 224 evaluations, where no DE run did;
- a robust pooling rule fixed in advance.

No evaluation-count target is set; item 5's "5–20× fewer evaluations" claim stays unjudged.
