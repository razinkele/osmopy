# Java, Python and C# on the same model — 2026-11-22

The three engines timed doing the same job: the Eastern English Channel reference model
(`data/eec_full`: 14 species, 10 resource groups, 24 steps per year), run as shipped for its 70
years, and for 5 years. Source of the README's Performance table.

Raw records: [`2026-11-22-three-engines-eec_full.json`](2026-11-22-three-engines-eec_full.json)
(70 years) and [`2026-11-22-three-engines-eec_full-5yr.json`](2026-11-22-three-engines-eec_full-5yr.json)
(5 years). Harness: `scripts/benchmark_three_engines.py`.

## Results

Median of five runs (minimum–maximum). "Cores used" is CPU time divided by wall time.

### 70 years (the model as shipped)

| Engine | Threads set | Wall time | Speed vs Java | Cores used | Peak memory |
|---|---|---|---|---|---|
| Java 4.4.1 | none (no such setting) | 220.0 s (204.7–225.7) | 1.0× | 1.08 | 584 MB |
| Python | 1 | 78.2 s (77.6–80.9) | 2.8× | 1.00 | 330 MB |
| Python | 14 | 34.1 s (33.3–37.0) | 6.5× | 6.95 | 330 MB |
| C# | 1 | 28.0 s (27.3–28.5) | 7.9× | 1.01 | 305 MB |
| C# | 14 | 9.3 s (9.0–9.8) | 24× | 4.54 | 312 MB |

C# against Python: 2.8× faster on one thread, 3.7× on 14.

### 5 years

| Engine | Threads set | Wall time | Speed vs Java | Cores used | Peak memory |
|---|---|---|---|---|---|
| Java 4.4.1 | none | 8.0 s (7.9–8.1) | 1.0× | 1.97 | 472 MB |
| Python | 1 | 5.0 s (4.9–5.1) | 1.6× | 1.00 | 296 MB |
| Python | 14 | 3.1 s (3.0–3.4) | 2.5× | 5.06 | 295 MB |
| C# | 1 | 4.6 s (4.4–4.8) | 1.7× | 1.06 | 222 MB |
| C# | 14 | 4.0 s (3.8–4.2) | 2.0× | 2.29 | 221 MB |

A short run is mostly start-up. On 14 threads the Python engine finishes a 5-year run **before**
the C# engine (3.1 s against 4.0 s): the C# run pays for starting the .NET runtime and handing
the model over to it. On the smallest shipped config (`data/minimal`, one year) the same harness
gave 1.1 s for Python and 4.0 s for C#, one run each. The C# engine's advantage is in long runs.

### The Python engine's first run

With an empty Numba cache the 5-year, one-thread run took **92.0 s**; the same run again took
5.1 s. The first run in a fresh checkout therefore pays about 87 s of compilation, once; the
compiled code is cached on disk. Every figure above is with the cache filled.

## Method

- **One fresh process per run, doing the whole job**: read the config, simulate, write the
  outputs. The Java engine is a separate program and can only be timed as a whole, so this is the
  only span the three have in common. `scripts/benchmark_engine.py` times the in-process core of
  the Python and C# engines only, and its numbers are not comparable with these.
- **Measured from outside**: wall clock, CPU time (user + system) and peak resident memory of
  that one child, from `os.wait4`.
- **Interleaved**: a round runs every arm once and the order rotates between rounds. One round
  is discarded as a warm-up, five are measured.
- **Threads**: the Python child calls `apply_single_run_threads(n)` (Numba) and caps the BLAS
  pools; the C# child passes `threads=n` to the engine. The Java engine has no thread setting
  for a single simulation and was run as the app runs it, `java -Xmx2g -jar ...`. Fourteen is
  the number of physical cores.
- **Same result at any thread count**: the biomass files of the 1-thread and 14-thread runs are
  byte-identical, for the Python engine and for the C# engine.

## What "the same job" does and does not mean

- **Same model.** All three read `data/eec_full/eec_all-parameters.csv`. Java gets it through
  `stage_config_for_java`, as in the app: that adds 40 keys naming each resource group's forcing
  file in the 4.4.1 jar's spelling, and changes or removes no value. Neither engine refuses the
  config.
- **Same run.** 70 (or 5) years, the config's own outputs, one row of biomass per year from each
  engine.
- **Not the same files.** Python and C# write the identical set (66 files). Java writes 48 and
  about twice the bytes: it puts the age and size distributions in one file per family where the
  other two write one per species, and it writes some families they do not (`meanSizeCatch`,
  `meanTLCatch`, `biomassPredPreyIni`, the additional-mortality distributions).
- **Not the same random numbers.** The engines use different generators, so single runs differ
  species by species; see the README's parity section for how they agree as ensembles. In the
  pilot run of each, the final total biomass was 0.87 Mt (Python), 0.94 Mt (C#) and 1.00 Mt
  (Java). The Python and C# runs are seeded (42) and repeat exactly; the Java run is not.
- **Not the Baltic.** The production Baltic config cannot be the common model: the Java engine
  refuses it (the bottom-oxygen coupling of the benthos has no Java counterpart).

## Machine and versions

Intel Core i9-10940X (14 cores, 28 threads), Linux 6.8, a workstation that was otherwise idle
but not isolated. Python 3.12.3, NumPy 2.4.6, Numba 0.65.1, pandas 3.0.6; pythonnet 3.2.1 on
.NET 10.0.15; OpenJDK 21.0.13 with the OSMOSE 4.4.1 jar; osmopy 1.11.1 at `c920fbb1`.

## To repeat

```bash
bash scripts/build_csharp_engine.sh     # the C# engine, once
.venv/bin/python scripts/benchmark_three_engines.py --years 70 --repeats 5 --output out.json
```

The Java arm needs `java` and `osmose-java/osmose-4.4.1-jar-with-dependencies.jar` (or `--jar`).
