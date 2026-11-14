# Baltic cell occupancy and predation pairs, 2026-11-14

This is item 3 of `docs/osmopy-performance-optimization-plan.md`. It repeats the 2026-09-12 EEC and
Bay of Biscay occupancy table (`docs/plans/2026-05-05-deep-review-remediation-plan.md`, "Exploring
the algorithmic improvement") on the Baltic, to decide whether work on the mortality kernel
(item 6) has anything to save.

**Result: Baltic cells are much fuller than EEC's.** The criterion the plan set fails: a median
of about 15 schools or fewer would have carried September's conclusion over, and the late-run
median is 26. The pair work is concentrated in a few crowded cells:
- 57.5% of scanned pairs sit in cells holding more than 50 schools;
- 29% sit in the roughly 9 cells per step that hold more than 100.

**What this changes, and what it does not:**
- **Weakened:** only September's "cells are tiny" objection to a smarter size-window scan.
- **Untouched:** the visit-order objection. `total_available` is a running sum, so a reordered
  scan changes the floating-point result.
- **Not re-measured:** the locality result, which was measured on EEC only.
- **For item 6:** the occupancy part of its entry criterion is met. The payoff is unmeasured.

## Measurement

- **Setup:** production Baltic config, 10-yr run, seed 42, master `40250aed`.
- **Instrument:** `scripts/baltic_cell_occupancy.py`. It reads the end-of-step state through
  `PythonEngine.run`'s `step_observer`, so the engine is not modified.
- **Raw output:** `docs/diagnostics/baltic_cell_occupancy_2026-11-14.json`.

**What is counted**, per step, over cells holding at least one live, in-domain school with positive
length:
- **schools per cell:** mean, median and maximum;
- **pairs scanned:** the sum of n_cell²;
- **eligible pairs:** pairs (p, q) with q ≠ p and `r_min ≤ L_p / L_q < r_max` at p's feeding stage.
  This is the kernel's Phase 1a size test (`mortality.py`, `ratio < r_min or ratio >= r_max` skips).

These are the quantities of the September table. September's probe was not committed, so its code
cannot be diffed against this one.

**How windows are summarised.**
- **Schools-per-cell figures:** the mean is a mean of per-step means, the median a median of
  per-step medians, and the maximum the largest cell seen.
- **Empty steps:** two steps in year 1 hold no live in-domain school. They count as zero in the
  per-step counts (schools, occupied cells, pairs). They are left out of the schools-per-cell
  statistics, which are undefined there: those use 22 of 24 steps in year 1 and 238 of 240 over the
  run.
- **Which Baltic window to compare:** years 6–10 is the one comparable with "EEC 5 yr", since both
  are past the first-year seeding transient.

| | Baltic yr 1 | Baltic yrs 6–10 | Baltic yrs 1–10 | EEC 1 yr | EEC 5 yr | BoB 1 yr |
|---|---:|---:|---:|---:|---:|---:|
| live schools per step | 1,901 | 17,730 | 12,603 | | | |
| occupied cells per step | 406 | 608 | 587 | | | |
| schools/cell, mean | 6.0 | **29.1** | 21.1 | 6.8 | 19.0 | 13.8 |
| schools/cell, median | 4 | **26** | 21 | 5 | 15 | 6 |
| schools/cell, max | 390 | 390 | 390 | 36 | 101 | 271 |
| pairs scanned | 1.17 M | 108.3 M | 132.8 M | 473 k | 27.1 M | 908 k |
| pairs eligible | 14.9 k | 12.8 M | 15.1 M | 64.5 k | 3.0 M | 457 k |
| **eligible fraction** | 1.3% | **11.8%** | 11.4% | 13.6% | 10.9% | 50.4% |

The EEC and BoB columns are copied from the September table. Baltic runs `mortality.subdt = 10`,
and the kernel repeats the scan every sub-step. Years 6–10 therefore come to about 1.08 billion
pair tests by this count, against 108.3 million per scan.

**Where the scanned pairs are** (years 6–10):

| cell size | ≤15 | 16–50 | 51–100 | >100 |
|---|---:|---:|---:|---:|
| share of pairs scanned | 1.6% | 40.9% | 28.3% | 29.2% |

On average 9.1 cells per step hold more than 100 schools.

**Cross-check against the plan's figure.** The plan asked for the school count to be checked
against the roughly 18.6k live schools per step recorded after the sub-one-fish floor
(2026-11-08). This run gives 17,730 per step over years 6–10, 4.7% lower. The two counts may
not use the same definition or window, and this note does not resolve the gap.

## Limits of the count

**"Pairs scanned" is an operation count, not a kernel counter.** The kernel's scan for predator p
runs over `n_local`, the schools in the cell mid-step. That includes zero-abundance schools that
have not yet been compacted, and the scan runs only for predators with `max_eatable > 0`. The
probe counts live schools at the end of the step. The two differ in both directions. The
eligible fraction is the better-defined number.

**The count is not a time share.** That 88% of scanned pairs fail the size test says how much of
the scan does no useful work. It does not say how much of the kernel's time the scan takes.

## What item 6 would need to measure first

**The missing number:** the share of one-thread mortality-kernel time spent in the Phase 1a scan
on Baltic. Only that bounds what a smarter scan could save; this note gives the operation count
it would be bounded by.

**Hypothesis, not measured: an order-preserving prefilter.** It would find the eligible prey
faster (for example, through a per-cell index sorted by length) and then visit them in their
original cell order. To be performance-preserving rather than statistically equivalent, it must
keep all of these unchanged:
- the eligible set;
- the visit order, and so the running sum `total_available`;
- `n_local`;
- the per-cell random stream, which indexes positions in the cell slice.

Whether the bookkeeping costs less than the scan it replaces is exactly what September found was
not true at EEC's occupancy. Baltic's crowded cells are where it would have to show a gain.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/baltic_cell_occupancy.py --years 10 --seed 42 \
    --out baltic_cell_occupancy.json
```
