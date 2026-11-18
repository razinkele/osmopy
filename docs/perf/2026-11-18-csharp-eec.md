# C# engine: performance on EEC (task 1.11, spec 10.4)

Date: 2026-11-18. Harness: `scripts/csharp/ab_eec.py`. Config: EEC full, 10 years, seed 42,
`output.minimal.enabled` on both engines. Machine: 28 logical CPUs (14 physical cores).

## Verdict

| Threads | Python median | C# median | Ratio (gate 1.0x, target 1.2x) | Smallest pair ratio | Pairs agree |
|---|---|---|---|---|---|
| 1 | 6.586 s | 1.748 s | **3.77x** PASS | 3.74x | 7 / 7 |
| 14 | 2.998 s | 0.396 s | **7.57x** PASS | 6.36x | 7 / 7 |

**The EEC gate passes at both thread counts, about three to six times the 1.2x target.** An
earlier run the same day gave 3.79x and 7.61x (its table is below). The figures supersede the
indicative 3.4x / 4.6x quoted with tasks 1.8 and 1.9, which predate the map-table change
described under "Bundle build".

| Budget (spec 10.4) | 1 thread | 14 threads |
|---|---|---|
| First call in a fresh process, at most 2 s | 0.57 s, PASS | 0.40 s, PASS |
| Bundle build plus marshalling, at most 5% of a 10-yr EEC run | 2.2% (39 ms), PASS | **9.8%** (40 ms), OVER |
| Peak memory, no more than the Python engine's | 291.0 vs 289.1 MB, **1.9 MB over** | 289.6 vs 290.2 MB, PASS |

Two budget lines miss, by small absolute amounts. Both have an identified cause and a fix outside
this task (below). The owner decides whether they block anything. Neither changes the gate.

## Protocol, and where it departs from spec 10.4

As specified:
- Each engine runs in its own long-lived worker process, warmed once (a full 10-yr run) and then
  driven run by run.
- Pairs are interleaved, the engine going first alternates, and only one engine runs at a time.
- Python runs with a warm Numba cache and is measured fresh, on this branch; no figure from
  `docs/perf/` is used as its baseline.
- C# has one warm-up run in the same process. Its cold first call is measured separately: a cold
  1-year run minus a warm one, in a fresh process.
- The timed span is the same as `scripts/benchmark_engine.py`. For Python it is `simulate`; for
  C# it is the bundle, the result buffers, `RunSimulation` and the `StepOutput` list. Config
  parsing is outside both spans.

**Departure: the machine was not idle.** Other users' jobs ran throughout, and neither run could
wait for them:
- first run: load average 7.2 at the start (an `ecospace.out` loop on one core);
- second run: load 3.6 rising to 8.8 (a `java` job at about 1.6 cores and a `python` job at one).

The two runs agree to 1% on every median ratio. The second run's 14-thread pairs are the noisiest
(Python 2.91–3.38 s), which is where the smallest pair ratio, 6.36x, comes from. A load this
size cannot move either ratio near the 1.0x gate. It can move the 14-thread figure by a few
percent, so quote that one as "about 7.5x".

## Where the C# time goes

From the engine's `RunSummary` phase split, as medians over the 7 runs. The phases cover 99–100%
of the engine's own clock.

| Phase | 1 thread | 14 threads |
|---|---|---|
| mortality | 1.571 s (90%) | 0.229 s (58%) |
| movement | 0.065 s | 0.053 s |
| life cycle | 0.044 s | 0.047 s |
| regroup | 0.013 s | 0.016 s |
| outputs | 0.007 s | 0.007 s |

At 14 threads mortality scales 6.9x. The other phases stay near their one-thread cost:
- Movement runs in chunks of 2,048 schools (`SimulationRun.MoveChunk`). EEC peaks at 3,906
  schools, so it uses at most two workers.
- The life cycle and regroup are serial.

Those three are a third of the engine's 14-thread time, and the next target for Phase 2. The
spec's 14-thread Baltic row assumes movement runs about 8x parallel, and Baltic's ~18,600 schools
make ten chunks. EEC needs nothing more to pass.

## Bundle build

**Fixed in this task: the map tables.** `MovementMapSet` filled its `(age, step)` table in a
triple Python loop. On EEC at 10 years that is about 1.2 million iterations, and both engines pay
it at the start of every run: `simulate()` builds the same objects. Two changes:
- the fill is now one array assignment per map;
- the map CSV parser gained a NumPy fast path.

With a third change, the builder now reading each LTL forcing series once instead of frame by
frame, the EEC bundle build fell from 199 ms to 34 ms. The forcing change is covered by the
existing builder tests, which check the carrying capacity bit for bit against the live
`ResourceState`. The two map changes are tested bit for bit against the code they replaced:
- `tests/test_movement_maps_fill.py` runs EEC, Baltic and synthetic edge cases: overlapping maps,
  year and step lists, a negative age, a case-insensitive name;
- `tests/test_movement_map_loader.py` runs every shipped map file plus a file of special tokens.

Each test was mutation-checked. A float32 parse first survived, because every shipped map value
is float32-exact (0, 1, -99). The synthetic file now carries a fast-path row that float32 cannot
hold, and the mutation fails. The Python baseline above already includes the faster fill, which
makes the ratios conservative.

**What is left: 34–40 ms, nearly all input resolution.** Per build, averaged over 10:
- movement map sets: 22 ms (file reads 4 ms, path resolution and table work the rest);
- the LTL forcing (`ResourceState`): 12 ms;
- the rest of the movement section (flattening the tables, the random-walk domains): about 4 ms;
- every other section together: under 1 ms.

Marshalling proper is spread through these and is a few milliseconds at most.

The Python engine performs the same map-set and forcing work inside its own timed `simulate()`.
So this cost does not separate the engines; it only becomes a larger share as the C# run gets
shorter. At one thread it is 2.2%. At 14 threads the run is 0.4 s and the same 40 ms is 9.8%.

*Follow-up, not done here:* cache the static inputs (map grids, forcing frames) per process,
keyed by file identity. That would remove the cost from every run after the first, for both
engines. It changes shared engine code, so it deserves its own review.

## Peak memory

Both workers reach 205 MB after their imports and the config, before any run. Their peaks are
289–291 MB, so each engine's run adds 84–86 MB. At one thread C# came out 1.4–1.9 MB above
Python in both runs. At 14 threads it came out 0.9 MB above Python in the first run and 0.6 MB
below in the second.

**Where it comes from.** Measured in a separate process:
- Importing the C# bundle builder pulls in the Python engine's mortality and movement modules,
  and with them Numba and LLVM. That costs **64 MB**.
- The builder calls none of their jitted functions. It only uses plain helpers that live in
  those modules (`_precompute_resource_arrays`, `_flatten_all_map_sets`,
  `build_random_patches`).
- The build's own data adds 20 MB, and the CLR plus the run add 55–65 MB.

*Follow-up, not done here:* move those helpers to Numba-free modules. A process that runs only
the C# engine could then skip Numba, about 60 MB under the Python engine. Today
`osmose/engine/thread_policy.py` imports Numba as well, and calibration workers call it, so that
import would need to become lazy too before the saving applies there: about 60 MB per worker, or
0.8 GB across 14. The budget's real target is the 50-yr Baltic run of Phase 2, where the run's
data dominates. EEC's peaks are mostly runtime machinery.

## First run, same day

Load 7.2 at the start. Bundle build before the CSV fast path.

| Threads | Python median | C# median | Ratio | Smallest pair | Bundle share | First call | Peak RSS Python / C# |
|---|---|---|---|---|---|---|---|
| 1 | 6.535 s | 1.726 s | 3.79x | 3.70x | 2.4% | 0.60 s | 289.5 / 290.9 MB |
| 14 | 2.756 s | 0.362 s | 7.61x | 7.36x | 11.4% | 0.41 s | 289.2 / 290.1 MB |

## Per-pair data (second run)

| Threads | | Pair times (s) |
|---|---|---|
| 1 | Python | 6.540, 6.601, 6.576, 6.667, 6.590, 6.586, 6.577 |
| 1 | C# | 1.723, 1.764, 1.750, 1.742, 1.748, 1.760, 1.730 |
| 14 | Python | 2.908, 3.029, 2.962, 2.998, 2.978, 3.156, 3.382 |
| 14 | C# | 0.457, 0.408, 0.387, 0.392, 0.394, 0.396, 0.464 |

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
PYTHONPATH=. python scripts/csharp/ab_eec.py --pairs 7 --threads 1 14 --years 10 --json ab.json
# single runs, and where a run's time goes:
PYTHONPATH=. python scripts/benchmark_engine.py --engine csharp --config eec_full --years 10 --threads 14
PYTHONPATH=. python scripts/profile_engine.py --engine csharp --config eec_full --years 10 --threads 14
```

About 4 minutes for the A/B. `ab_eec.py` prints the load average and flags a busy machine.
