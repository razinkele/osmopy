# Mortality thread oversubscription — investigation record

**Date:** 2026-07-15
**Box:** Intel i9-10940X — 14 physical cores × 2 hyperthreads = 28 logical, 1 socket, 1 NUMA node.

## Finding

The mortality cell-loop (`osmose/engine/processes/mortality.py::_mortality_all_cells_parallel`,
`@njit(cache=True, parallel=True)`, `prange` over cells) is ALREADY parallel and bit-identical at
every thread count. It does NOT scale past ~physical cores: a single run defaults to all 28 logical
(hyperthread) cores, which is the PESSIMAL point — ~1.5–2× slower than the ~physical-core optimum.

## Thread-scaling sweep (whole-engine wall-time)

eec_full (990 cells):

| threads | 2yr median (3 reps) | 3yr median (7 reps) |
|--------:|--------------------:|--------------------:|
| 1  | 1.608 s | 3.081 s |
| 4  | 0.913 s | 1.650 s |
| 8  | 0.714 s | 1.274 s |
| 12 | —       | **1.155 s (peak)** |
| 14 | —       | 1.172 s |
| 16 | 0.718 s | —       |
| 28 (default) | 1.485 s | 1.779 s |

baltic (2000 cells, 1yr, 5 reps): 1t 0.713 s → 8t 0.278 s → **14t 0.250 s (peak)** → 28t 0.412 s.

Determinism: final biomass was **bit-identical across all thread counts** in both configs (the
per-cell disjoint-slice + per-cell-seed prange is race-free).

## Root cause

A fork/join per timestep over ~1–2k uneven cells. Past the physical-core count, thread-team
coordination + hyperthread cache/port contention + memory bandwidth swamp the compute saved. Same
oversubscription pathology already documented for DE calibration workers (24→16 default).

## Conclusion

Cap single-run Numba threads to affinity-capped physical cores (~5-line change, bit-exact). The
backlog's "parallelize the cell-loop" and "native C+OpenMP port" levers are STALE: the loop is
already parallel, and a C+OpenMP port would hit the same fork/join + bandwidth + hyperthread walls.
Scope: single-run entry points only — calibration's nested-parallelism regime is unaffected.

Design: `docs/superpowers/specs/2026-07-15-single-run-thread-policy-design.md`.
