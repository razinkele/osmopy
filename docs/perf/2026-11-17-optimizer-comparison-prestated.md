# Optimizer comparison at a matched budget: pre-stated design and rule (perf plan item 5b)

Written and committed **before** any comparison run. Harness: `scripts/optimizer_comparison.py`.

## Question

At the same number of real evaluations, on one real OSMOSE calibration objective, do DE, CMA-ES and
surrogate-assisted DE (the three choices of `calibrate_baltic.py --optimizer`) reach different
objective values? The repository's only comparison so far is synthetic
(`data/benchmarks/optimizer_comparison.json`).

## Design

- **Problem:** calibration phase 1b, 9 log-space parameters (larval and adult mortality of herring,
  sprat, flounder and stickleback, plus cod's maximum ingestion rate), with its real objective
  (`make_objective`) on the production Baltic config, 40 simulated years, simulation seed 42 for
  every candidate.
- **Start point:** phase 1b's own x0 (the R18 values), unless the pre-flight shows the objective
  at x0 within 2 seed SDs of the objective at production's values. In that case every optimizer
  starts from one fixed displaced point instead, recorded with the results.
- **Budget:** 224 real evaluations per run, 14 workers. The configurations, fixed in the harness:

  | optimizer | configuration | evaluations |
  |---|---|---|
  | DE | population 28 (x0, 30% within ±15% of it, 70% Latin hypercube, as `run_calibration` builds it), 7 generations, `tol = 0` | 28 × 8 |
  | CMA-ES | population 14, 16 generations, σ0 = 0.3, `tol = 0` | 14 × 16 |
  | surrogate-DE | 44 initial samples, 6 refinements of 30 | 44 + 180 |

  Every evaluation is logged from the worker that ran it. The logged count is the evaluation
  count. Best-so-far is read from the first 224 evaluations in time order, whatever an optimizer
  reports.
- **Replicates:** 3 per optimizer, with optimizer seeds 1, 2 and 3; 9 runs in all.
- **Re-rank:** each run's final best (the lowest logged objective) is evaluated on simulation
  seeds 1–5, which exclude the training seed 42.

## Instruments

1. **Primary: the re-ranked objective,** the mean over seeds 1–5 of each run's final best.
2. **Secondary, descriptive:** best-so-far on the training seed after 74, 149 and 224 evaluations.

## Rule

Fixed before the runs:

- **Optimizer A ranks above B** if the WORST of A's three re-ranked means beats the BEST of B's
  by more than 2 × the pooled re-rank seed SD.
- **Otherwise no ordering is claimed for that pair.** With three replicates, only gross
  differences are detectable, and the result says so.
- **Sentinels:** a run with any failed evaluation (objective ≥ 1e6) is flagged and reported. If
  its final best is a sentinel, it is excluded from the ranking.
- **Scope:** the table compares these three configurations at this budget on this problem, not the
  algorithms in general.

## Decision

The plan's exit criterion: a results table first. Only then decide whether a staged workflow
(screen, surrogate, re-rank) is worth wiring in as a default. The rule above decides nothing on its
own; it says which differences the table can support.

## Pre-flight

Run before the comparison (`optimizer_comparison.py preflight`, seeds 1–5, 40 yr):

- **Objective at x0 (R18): 1156–1724.** This is the collapse class.
- **Objective at production's own values: 0.716–0.723,** mean 0.7190, seed SD 0.0033.
- **So the start point stays x0.** x0 is thousands of SDs from production, so there is plenty to find and no optimizer starts on the answer. DE's 30% neighbourhood is centred on x0, not on production.
- **The optimum is reachable:** production's values sit inside all nine bounds. They are compared in the objective's resolved space, where larval rates are already divided by ndtperyear.
- **Bound corners:** lower builds, upper builds (`EngineConfig.from_dict`).
