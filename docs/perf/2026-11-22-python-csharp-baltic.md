# Python and C# on the Baltic model — 2026-11-22

The Python and C# engines timed doing the same job on the production Baltic Sea config
(`data/baltic/baltic_all-parameters.csv`: 9 focal species plus the grey seal and the cormorant as
background predators, resources, the bottom-oxygen coupling of the benthos; 24 steps per year),
run as shipped for 15 years. Source of the README's Baltic table. The Java engine is not in it:
it refuses this config (`java_engine_block_reason`: the oxygen-to-benthos coupling has no Java
counterpart).

Raw record: [`2026-11-22-python-csharp-baltic.json`](2026-11-22-python-csharp-baltic.json).
Harness and method: as in [`2026-11-22-three-engines-eec_full.md`](2026-11-22-three-engines-eec_full.md)
(one fresh process per run that reads, simulates and writes; timed from outside; interleaved; one
warm-up round discarded, five measured).

## Results

Median of five runs (minimum–maximum). "Cores used" is CPU time divided by wall time.

| Engine | Threads set | Wall time | Speed vs Python, 1 thread | Cores used | Peak memory |
|---|---|---|---|---|---|
| Python | 1 | 64.0 s (62.1–66.5) | 1.0× | 1.00 | 326 MB |
| Python | 14 | 29.7 s (28.5–31.2) | 2.2× | 6.26 | 325 MB |
| C# | 1 | 25.9 s (25.0–26.7) | 2.5× | 1.01 | 256 MB |
| C# | 14 | 7.4 s (7.2–8.0) | 8.6× | 4.74 | 258 MB |

C# against Python at the same thread count: 2.5× on one thread, 4.0× on 14. On EEC the same
comparison gave 2.8× and 3.7×.

## Same job

- Each run left the 15 yearly biomass rows the config asks for.
- One extra round kept its outputs: Python and C# wrote the same 47 files, and for each engine
  the 1-thread and 14-thread runs wrote byte-identical files (47 of 47).
- The two engines do not draw the same random numbers, so their biomass differs run by run.

## A discarded first attempt

A first five-round measurement, taken minutes earlier, is not used. In its first three rounds
something else on the workstation took CPU: every arm ran 15–25% slower there than in rounds 4
and 5 (Python on one thread 80 s against 64 s), and a single idle round taken afterwards matched
the fast rounds. Its median would have overstated the times by about a quarter. The ratio of the
two engines within each round held steady through it (C# 2.4–2.7× faster on one thread), so the
comparison itself did not depend on it. The measurement above was taken next, on the quiet
machine, and its rounds agree within about ±5%. The discarded record is kept as
[`2026-11-22-python-csharp-baltic-discarded.json`](2026-11-22-python-csharp-baltic-discarded.json).

## Machine and versions

Intel Core i9-10940X (14 cores, 28 threads), Linux 6.8. Python 3.12.3, NumPy 2.4.6, Numba 0.65.1;
pythonnet 3.2.1 on .NET 10.0.15; osmopy 1.11.2 at `b6a07d7b`. CI and production run Numba 0.68.1
and NumPy 2.5.5; whether that moves the Python rows was not measured.

## To repeat

```bash
bash scripts/build_csharp_engine.sh     # the C# engine, once
.venv/bin/python scripts/benchmark_three_engines.py --config data/baltic/baltic_all-parameters.csv \
    --years 15 --repeats 5 --engines python,csharp --output out.json
```
