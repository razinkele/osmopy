# C# engine on Baltic: the Phase 2 performance measurement (task 2.8)

Date: 2026-11-19. Harness: `scripts/csharp/ab_eec.py --config baltic`. Spec 10.4's protocol:
Baltic, 10 years, seed 42, `output.minimal.enabled` on both engines, 7 interleaved pairs per
thread count, each engine in its own warmed-up process. Code state: master with tasks 2.1 to
2.6c merged, plus the UI branch (task 2.7), which does not touch either engine.

This is the measurement for gate G2. The earlier note
(`2026-11-18-csharp-baltic-indicative.md`) was a first reading on task 2.1's branch.

## The gates

| Threads | Python median | C# median | Ratio | Gate / target (spec 10.4) | Smallest pair | Pairs agree | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 26.22 s | 10.66 s | **2.46x** | 1.15x / 1.4x | 2.45x | 7 / 7 | **PASS**, above the target |
| 14 | 11.45 s | 1.69 s | **6.76x** | 1.5x / 2.0x | 6.68x | 7 / 7 | **PASS**, above the target |

The machine was idle when the measurement started (load average 0.61 on 28 logical CPUs).

## The budgets

| Budget (spec 10.4) | 1 thread | 14 threads | Verdict |
|---|---|---|---|
| First call in a fresh process, at most 2 s | 0.52 s | 0.37 s | PASS |
| Bundle build plus marshalling, at most 5% of a run | 0.5% | 3.0% | PASS |
| Peak memory, no more than the Python engine's | 296 vs 308 MB | 307 vs 309 MB | PASS here; see below |

**Peak memory at 14 threads varies from one measurement to the next.** Five readings:

| Reading | Pairs | Machine | C# peak | Python peak |
|---|---|---|---|---|
| 2026-11-18, task 2.1's branch | 7 | busy (load 2 to 7) | **336 MB** | 309 MB |
| This measurement | 7 | idle | 307 MB | 309 MB |
| Repeat 1 | 3 | lightly loaded | 292 MB | 309 MB |
| Repeat 2 | 3 | lightly loaded | 292 MB | 309 MB |
| Repeat 3 | 3 | lightly loaded | 290 MB | 309 MB |

Four readings are under the Python engine's and one is 27 MB over. The Python engine's peak does
not move; the C# engine's does, by up to 46 MB. The process keeps the .NET heap between runs,
and how far that heap grows before a collection depends on timing. The engine itself allocates
nothing per step in steady state (task 1.9). So the budget is met in the measurement taken for
the gate, but not with a margin that a busy machine cannot erase. Two ways to make it firm, not
taken here: a collection at the end of each run, or the runtime's memory-conserving setting.
Either needs its own measurement of the cost.

At one thread the C# engine is under in every reading taken during Phase 2 (284 to 296 MB
against 307 to 309 MB, five readings).

## Where the C# time goes

| Phase | 1 thread | 14 threads |
|---|---|---|
| mortality | 9.71 s (91%) | 1.09 s (65%) |
| movement | 0.56 s | 0.18 s |
| life cycle | 0.20 s | 0.21 s |
| regroup | 0.11 s | 0.12 s |
| outputs | 0.02 s | 0.02 s |

Mortality scales 8.9x on 14 threads and movement 3.1x. The life cycle and the regroup are serial.

## With the full outputs (not a gate)

The gates are defined on minimal outputs. With the config as shipped (every output family
Baltic asks for: life-stage, by age, by size, SSB, mean size, mean trophic level, yield in
numbers, diet), 5 pairs, `--outputs full`:

| Threads | Python | C# | Ratio | C# peak memory | Python peak memory |
|---|---|---|---|---|---|
| 1 | 28.49 s | 10.95 s | 2.60x | 296 MB | 315 MB |
| 14 | 12.96 s | 1.90 s | 6.83x | 310 MB | 316 MB |

The machine was not idle for this table (load 6.4 at the start). The ratio is a paired
measurement (the two engines interleaved); the absolute times are not comparable with the gate
table's, which was taken at another time.

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
to use. The Python engine's own cost was not measured in pairs; the two tables above put it
near 2 s at one thread.

## Against the first reading

| | First reading (2026-11-18) | This measurement |
|---|---|---|
| 1 thread | 2.55x (27.46 s / 10.75 s) | 2.46x (26.22 s / 10.66 s) |
| 14 threads | 6.83x (12.19 s / 1.79 s) | 6.76x (11.45 s / 1.69 s) |

Both engines are a little faster on the idle machine, the Python engine more so. The C# time
did not grow with tasks 2.6a to 2.6c: the minimal path was checked before and after each of them.

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
PYTHONPATH=. python scripts/csharp/ab_eec.py --config baltic --pairs 7 --threads 1 14 --years 10
PYTHONPATH=. python scripts/csharp/ab_eec.py --config baltic --outputs full --pairs 5 --threads 1 14 --years 10
```

About 7 minutes and 5 minutes.
