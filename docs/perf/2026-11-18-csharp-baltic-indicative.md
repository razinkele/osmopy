# C# engine on Baltic: an indicative speed reading (before task 2.8)

Date: 2026-11-18. Harness: `scripts/csharp/ab_eec.py --config baltic`. Baltic, 10 years, seed
42, `output.minimal.enabled` on both engines; 7 interleaved pairs per thread count; spec 10.4's
protocol. Branch state: task 2.1 (PR #315), the first on which Baltic runs on the C# engine.

**This is not the Phase 2 gate measurement.** That is task 2.8's, taken when Phase 2 is declared
done. It uses the same protocol, minimal outputs on both engines included (spec 10.4), so what
can still move the ratio is the code added between now and then. The reading was taken now
because the Baltic gate is go/no-go, and the outputs and UI work should not start on an engine
that might miss it.

## Reading

| Threads | Python median | C# median | Ratio | Gate / target (spec 10.4) | Smallest pair | Pairs agree |
|---|---|---|---|---|---|---|
| 1 | 27.46 s | 10.75 s | **2.55x** | 1.15x / 1.4x | 2.51x | 7 / 7 |
| 14 | 12.19 s | 1.79 s | **6.83x** | 1.5x / 2.0x | 6.71x | 7 / 7 |

| Budget (spec 10.4) | 1 thread | 14 threads |
|---|---|---|
| First call in a fresh process, at most 2 s | 0.52 s | 0.36 s |
| Bundle build plus marshalling, at most 5% of a run | 0.5% | 3.0% |
| Peak memory, no more than the Python engine's | 286 vs 308 MB | **336 vs 309 MB, 27 MB over** |

## Where the C# time goes

| Phase | 1 thread | 14 threads |
|---|---|---|
| mortality | 9.76 s (91%) | 1.17 s (68%) |
| movement | 0.58 s | 0.20 s |
| life cycle | 0.20 s | 0.21 s |
| regroup | 0.13 s | 0.12 s |
| outputs | 0.02 s | 0.02 s |

Mortality scales 8.3x on 14 threads, and movement 2.9x. The life cycle and the regroup are serial.

## Caveats

- **The outputs work is still to come.** The gate is measured with minimal outputs, as here, so
  the risk is task 2.6 slowing the minimal path (the stage snapshot, and the diet accumulation
  in the mortality kernel). Each 2.6 task must show the minimal run unchanged, in results and
  in time.
- **The machine was not idle** (load average 2.1 at the start, 7.1 at the end; other users'
  jobs). The pair ratios are tight all the same: 2.51 to 2.66, and 6.71 to 6.98.
- **Equivalence on Baltic is indicated, not established.** Task 2.8's ensemble (16 replicates,
  15 years) is the formal check. A short ensemble was run with the same harness and rule
  (`scripts/cross_engine_parity_440.py --engines python,python-control,csharp`, 8 replicates,
  5 years, 1 year of spin-up): **PASS**, with the C# arm inside the Python engines' own spread.

  | Pair | Largest difference (log10) | Median | Cells beyond the margin |
  |---|---|---|---|
  | Python vs C# | 0.028 (1.07x) | 0.0036 | 0 of 42 |
  | Python vs Python control | 0.044 (1.11x) | 0.0048 | 1 of 42 |

  40 of 42 cells are equivalent on both arms; the two that are not (mean weight of pikeperch
  and smelt) miss on the control arm as well, so they are underpowered at 8 replicates, not an
  engine difference. The background species' columns equal the Python engine's bit for bit.
- **Peak memory at 14 threads is over the budget by 27 MB.** The likely cause is the per-thread
  mortality scratch, which grows with the largest cell; it is allocated once per worker. To be
  measured and addressed in task 2.8.

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
PYTHONPATH=. python scripts/csharp/ab_eec.py --config baltic --pairs 7 --threads 1 14 --years 10
```

About 8 minutes.
