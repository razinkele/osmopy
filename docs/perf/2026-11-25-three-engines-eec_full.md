# Java, Python and C# on the same model — 2026-11-25 (v1.19.0)

A re-run of [2026-11-22](2026-11-22-three-engines-eec_full.md) with the same harness
(`scripts/benchmark_three_engines.py`), the same model (`data/eec_full`, 14 species, 10
resources, 24 steps per year, as shipped), seed and protocol (one fresh process per run doing the
whole job, five measured rounds after one discarded warm-up, arms interleaved), at git `d8e419bf`.

Raw records: [70 years](2026-11-25-three-engines-eec_full.json),
[5 years](2026-11-25-three-engines-eec_full-5yr.json), and the control
[70 years, Python and C# in the 2026-11-22 environment](2026-11-25-python-csharp-eec_full-venv.json).

## Results (production environment: micromamba `shiny`, Python 3.13, NumPy 2.5.5, Numba 0.68.1)

Median of five runs (minimum–maximum). "Cores used" is CPU time divided by wall time.

### 70 years

| Engine | Threads set | Wall time | Speed vs Java | Cores used | Peak memory |
|---|---|---|---|---|---|
| Java 4.4.1 | none (no such setting) | 211.9 s (192.5–216.9) | 1.0× | 1.08 | 575 MB |
| Python | 1 | 128.5 s (127.6–129.5) | 1.6× | 1.00 | 599 MB |
| Python | 14 | 49.1 s (48.9–50.5) | 4.3× | 6.66 | 586 MB |
| C# | 1 | 32.4 s (32.2–33.6) | 6.5× | 1.01 | 530 MB |
| C# | 14 | 14.0 s (13.7–14.1) | 15× | 3.24 | 538 MB |

C# against Python: 4.0× faster on one thread, 3.5× on 14.

### 5 years

| Engine | Threads set | Wall time | Speed vs Java | Cores used | Peak memory |
|---|---|---|---|---|---|
| Java 4.4.1 | none | 7.8 s (7.4–7.9) | 1.0× | 2.06 | 474 MB |
| Python | 1 | 6.9 s (6.9–7.0) | 1.1× | 1.00 | 550 MB |
| Python | 14 | 4.0 s (3.9–4.0) | 2.0× | 4.50 | 551 MB |
| C# | 1 | 5.9 s (5.9–6.0) | 1.3× | 1.05 | 494 MB |
| C# | 14 | 5.4 s (5.4–5.8) | 1.4× | 1.75 | 493 MB |

## Why it is slower than 2026-11-22: two separate causes

The 2026-11-22 run used the repository `.venv` (Python 3.12.3, NumPy 2.4.6, Numba 0.65.1,
pandas 3.0.6); this one uses the `shiny` environment that production runs and CLAUDE.md now
prescribes. The control re-ran Python and C# in `.venv` on today's code (70 years):

| Arm | 2026-11-22 (`.venv`, v1.12-era code) | today's code in `.venv` | today's code in `shiny` |
|---|---|---|---|
| Python @1 | 78.2 s | 88.3 s (**+13%** code) | 128.5 s (+46% environment) |
| Python @14 | 34.1 s | 43.5 s (**+27%** code) | 49.1 s (+13% environment) |
| C# @1 | 28.0 s | 30.8 s (**+10%** code) | 32.4 s (+5% environment) |
| C# @14 | 9.3 s | 12.3 s (**+32%** code) | 14.0 s (+14% environment) |
| peak memory | ~310–330 MB | ~325–355 MB | ~530–600 MB |

1. **Code (+10–13% on one thread, ~+30% on fourteen).** Since 2026-11-22 the engines honour
   output switches EEC sets that were ignored before (#123 work, v1.14.1–v1.19.0): Java-rule
   distributions, catch families, mortality by size and the staged diet (EEC now writes 141 files
   instead of 66). Writing them costs under 1 s (measured: `write_outputs` 0.9 s of a C# 70-year
   run); the cost is per-step collection, which is serial, so it weighs most at 14 threads.
   A 5-year stage timing in `.venv` agrees: simulation 3.81 s (v1.12.0) against 4.28 s (+12%),
   same memory.
2. **Environment (Python +46% on one thread; all memory +~200 MB).** On identical code the
   `shiny` environment's simulation is slower and config preparation takes 0.65 s and +200 MB
   against 0.20 s in `.venv` (5-year stage timing). The doubled peak memory is entirely this.
   Not investigated further here (NumPy 2.5 / Numba 0.68 / Python 3.13 are all candidates).

The Java engine is unchanged (−4%, within its run-to-run spread).
