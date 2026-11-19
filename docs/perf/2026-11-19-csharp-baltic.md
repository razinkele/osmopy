# C# engine on Baltic: the Phase 2 performance measurement (task 2.8)

Date: 2026-11-19. Harness: `scripts/csharp/ab_eec.py --config baltic`. Spec 10.4's protocol:
Baltic, 10 years, seed 42, `output.minimal.enabled` on both engines, 7 interleaved pairs per
thread count, each engine in its own warmed-up process. Code state: this PR's head (tasks 2.1
to 2.7 merged, plus the collection at the end of a run described under "Memory").

This is the measurement for gate G2. The earlier note
(`2026-11-18-csharp-baltic-indicative.md`) was a first reading on task 2.1's branch.

## The gates

| Threads | Python median | C# median | Ratio | Gate / target (spec 10.4) | Smallest pair | Pairs agree | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 26.15 s | 10.57 s | **2.47x** | 1.15x / 1.4x | 2.47x | 7 / 7 | **PASS**, above the target |
| 14 | 11.48 s | 1.69 s | **6.81x** | 1.5x / 2.0x | 6.73x | 7 / 7 | **PASS**, above the target |

Load average at the start: 1.5 on 28 logical CPUs. The same measurement an hour earlier, on an
idle machine (load 0.6) and before the collection was added, gave 2.46x and 6.76x.

## The budgets

| Budget (spec 10.4) | 1 thread | 14 threads | Verdict |
|---|---|---|---|
| First call in a fresh process, at most 2 s | 0.51 s | 0.35 s | PASS |
| Bundle build plus marshalling, at most 5% of a run | 0.5% | 3.0% | PASS |
| Peak memory, no more than the Python engine's | 232 vs 308 MB | 238 vs 310 MB | PASS, by about 70 MB |

## Memory

**Before this PR the memory budget was not met for a process that runs many simulations.** The
harness had read the C# engine's peak at 14 threads as 336, 307, 292, 292 and 290 MB on five
occasions, against the Python engine's steady 309 MB. To see what a long-lived process does,
each engine ran Baltic (10 years) many times in one process, with resident memory read after
every run:

| Engine, threads | Runs | Resident memory after a run | Peak |
|---|---|---|---|
| Python, 1 | 12 | 299 MB, flat from the first run | 309 MB |
| C#, 14, before | 40 | about 265 MB from the fourth run on (236 to 282) | 291 MB |
| C#, 1, before | 60 | sawing between 260 and 312 MB from about run 20 | **324 MB** |

The cause, measured and not inferred:

- **The part of the process outside the .NET heap never moved**: resident memory minus what the
  collector had committed averaged 204 to 211 MB in every block of ten runs out of the 60. Nothing
  leaks in Python, NumPy or native memory.
- **The .NET heap held dead runs.** A run's own arrays (the school store, the scratch, the
  tables; about 30 MB) are dead when the run returns, but the collector, on its own schedule,
  cleared them only every second or third run. The managed heap sawed between about 21 and
  88 MB, and the resident memory followed it.

**The remedy is one collection when a run returns** (`OsmoseEngine.RunSimulation`). It takes
about 4 ms. Measured the same way after the change:

| Engine, threads, outputs | Runs | Resident memory after a run | Peak |
|---|---|---|---|
| C#, 1, minimal | 25 | 203 MB, flat | 235 MB |
| C#, 14, minimal | 30 | 204 MB, flat | 232 MB |
| C#, 14, full | 40 | 213 MB, flat | 246 MB |

The managed heap is back to 1.2 MB after every run. The C# engine now peaks 60 to 77 MB under
the Python engine at either thread count, with the full outputs too, and does not depend on
how many runs the process has done. The results do not change: the digests of EEC and Bay of
Biscay equal master's, the stock Baltic digests are the same before and after, and 16 Baltic
replicates of 15 years with every output on are bit-identical before and after on each engine
(`docs/baltic_csharp_crosscheck_2026-11-19.md`). A test holds
the managed heap after each of five Baltic runs under 5 MB; without the collection it reads 16
to 40 MB.

## Where the C# time goes

| Phase | 1 thread | 14 threads |
|---|---|---|
| mortality | 9.62 s (92%) | 1.09 s (67%) |
| movement | 0.55 s | 0.18 s |
| life cycle | 0.20 s | 0.21 s |
| regroup | 0.12 s | 0.12 s |
| outputs | 0.02 s | 0.02 s |

Mortality scales 8.8x on 14 threads and movement 3.0x. The life cycle and the regroup are serial.

## With the full outputs (not a gate)

The gates are defined on minimal outputs. With the config as shipped (every output family
Baltic asks for: life-stage, by age, by size, SSB, mean size, mean trophic level, yield in
numbers, diet), 5 pairs, `--outputs full`:

| Threads | Python | C# | Ratio |
|---|---|---|---|
| 1 | 28.49 s | 10.95 s | 2.60x |
| 14 | 12.96 s | 1.90 s | 6.83x |

This table was taken before the collection was added and on a machine that was not idle (load
6.4 at the start). The ratio is a paired measurement (the two engines interleaved); the absolute
times are not comparable with the gate table's.

**What the full outputs cost the C# engine**, measured on their own: one process, minimal and
full runs interleaved, the order alternating from pair to pair.

| Threads | Minimal | Full | Paired difference, median (range) | Cost | Pairs |
|---|---|---|---|---|---|
| 1 | 10.16 s | 10.63 s | +0.48 s (+0.45 to +0.56) | **4.8%** | 6 |
| 14 | 1.68 s | 1.89 s | +0.21 s (+0.16 to +0.26) | 12.7% | 12 |

At one thread the 0.48 s is mortality +0.34 s (the diet is booked inside the predation kernel),
the output pass +0.11 s and the life cycle +0.03 s. At 14 threads the serial output pass and
life cycle weigh more against a much shorter run.

**This replaces two earlier figures.** PR #319 gave 7.5%, and a comment on that PR gave 2.7%.
Neither was a paired measurement: each compared a minimal run and a full run taken at different
times under different load, and load alone moves a run by more than a percent. 4.8% is the one
to use. The Python engine's own cost was not measured in pairs; the two harness tables put it
near 2 s at one thread.

## Three readings of the gate

| | 2026-11-18, task 2.1's branch | 2026-11-19, idle, before the collection | 2026-11-19, this PR's head |
|---|---|---|---|
| 1 thread | 2.55x (27.46 s / 10.75 s) | 2.46x (26.22 s / 10.66 s) | 2.47x (26.15 s / 10.57 s) |
| 14 threads | 6.83x (12.19 s / 1.79 s) | 6.76x (11.45 s / 1.69 s) | 6.81x (11.48 s / 1.69 s) |
| C# peak memory, 1 / 14 threads | 286 / 336 MB | 296 / 307 MB | 232 / 238 MB |

The C# time did not grow with the output tasks (2.6a to 2.6c): the minimal path was checked
before and after each of them. The collection did not cost measurable time.

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
PYTHONPATH=. python scripts/csharp/ab_eec.py --config baltic --pairs 7 --threads 1 14 --years 10
PYTHONPATH=. python scripts/csharp/ab_eec.py --config baltic --outputs full --pairs 5 --threads 1 14 --years 10
```

About 7 minutes and 5 minutes. The many-runs series and the minimal-against-full pairs were
taken with a scratch script (a loop over `build_bundle`, `bridge.run` and `_step_outputs`,
reading `/proc/self/statm` and `System.GC.GetGCMemoryInfo()` after each run); the test named
above is what keeps the result.
