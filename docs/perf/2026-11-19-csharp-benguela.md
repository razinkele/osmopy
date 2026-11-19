# C# engine: Benguela runs

Date: 2026-11-19. The C# engine refused the Southern Benguela demo for a feature the demo does
not use. This note records why it runs now and what was checked. No feature was ported.

## Why it was refused, and why that was wrong

Benguela sets `simulation.incoming.flux.enabled = TRUE`. The C# capability check took the
switch for a use of incoming flux, which the engine does not implement (plan task 3.2).

But the switch alone does nothing. Incoming flux needs a flux file per species
(`flux.incoming.byDt.byAge.file.sp#` or `.bySize.`; the config reader lowercases keys, and the
engine reads the lowercase form), and Benguela has none; no shipped config has one. With the switch on and no file the Python engine loads no flux, creates no school and
draws no random number. Checked three ways:

- on the engine's own state object: no flux loaded, no school in 48 steps, the generator not
  advanced;
- on the shipped config: a whole Benguela year is identical, frame for frame, with the switch
  on and with it off;
- the test of that fails when the two runs differ (checked with another seed).

The capability check now asks whether the feature is **in use**: the switch on and a flux file
set (`osmose.engine.incoming_flux.incoming_flux_in_use`). A config that sets a flux file is
still refused, by the feature's name.

## What Benguela needs from the engine

Ten focal species, no background species, four plankton groups, 24 steps a year, unfished
(`fisheries.enabled = FALSE`). With minimal outputs its required features are the core only; as
shipped it also needs the full outputs, of which it asks for few: biomass, yield, mortality and
the life-stage outputs.

## Ensemble equivalence (spec 10.2)

`scripts/cross_engine_parity_440.py --engines python,python-control,csharp`, the config as
shipped with minimal outputs, 32 replicates per arm, 15 years (the demo's own horizon), the
first 5 dropped. The rule and its calibration are those used for EEC, Bay of Biscay and Baltic.

| | Result |
|---|---|
| Verdict | **PASS** |
| Cells counted | 30: biomass, abundance and mean weight of 10 species |
| Equivalent cells, C# / control | 28 / 27 |
| Largest difference, Python vs C# | 0.0483 log10 (1.118x) |
| Largest difference, Python vs Python | 0.0636 (1.158x) |
| Median difference, Python vs C# / Python vs Python | 0.0045 / 0.0068 |
| Cells beyond the margin, Python vs C# / Python vs Python | 0 / 1 |
| Frozen rung (slack, ratio, variance) and its false-fail rate for identical engines | 3, 3.0, 5: 2.9% |

- **The ten yield cells are not counted.** Benguela is unfished, so every yield is zero on
  every arm; the rule lists such cells and leaves them out.
- **The cells that are not equivalent are all shallow-water hake's**, and they are not
  equivalent between the two Python arms either: abundance and mean weight on both arms, and
  biomass on the control arm only. It is the noisiest species of the demo, not an engine
  difference.

## Structure and determinism

- **Records, frames and files.** A Benguela run through `CSharpEngine` hands on exactly the
  records the engine computed, and gives the frames (keys, shapes, column labels, dtypes) and
  the files (names, headers, row counts) the Python engine gives for the same config.
- **Threads.** The records of a two-year run are byte-identical at 1, 2 and 14 threads, on two
  seeds.

Both are in `tests/test_csharp_engine_e2e.py`.

## Speed (indicative; Benguela has no gate)

`scripts/csharp/ab_eec.py --config data/benguela/benguela_all-parameters.csv`, 10 years, minimal
outputs, 5 interleaved pairs per thread count. The machine was not idle (load 7.8 at the
start, left over from the ensemble), so read these as indicative.

| Threads | Python median | C# median | Ratio | C# peak memory | Python peak memory |
|---|---|---|---|---|---|
| 1 | 10.87 s | 2.35 s | 4.6x | 214 MB | 290 MB |
| 14 | 6.65 s | 0.80 s | 8.4x | 213 MB | 289 MB |

At 14 threads the bundle build and marshalling are 11% of the run, over the 5% that spec 10.4
budgets on its gate configs: the build costs about 0.09 s whatever the run, and this run takes
0.8 s. The first call in a fresh process takes 0.46 to 0.55 s (budget 2 s).

## Limits

- **Incoming flux itself is still not implemented** on the C# engine.
- **The output families beyond the life stages were not compared on Benguela**: it ships with
  them switched off. They were compared on Baltic (`docs/baltic_csharp_crosscheck_2026-11-19.md`).
- **Fishing is not exercised by this config.**

## Which shipped configs the C# engine accepts, and which were run

"Accepted" means the capability check does not refuse the config. "Run" means it was executed
on the C# engine, here for one year at least.

| Config | Capability check | Run on the C# engine |
|---|---|---|
| EEC full, Bay of Biscay, Baltic, **Benguela** | accepted | yes: the end-to-end tests and an ensemble each |
| EEC (6 species), Bay of Biscay's 4.3.3 snapshot, minimal | accepted | yes, one year |
| Fine-grid Baltic, upsampled | accepted | yes, one year, with the grid and forcing files of a local checkout (they are not in the repository) |
| Fine-grid Baltic, real | accepted | no: it does not load on either engine in that checkout (its movement maps resolve to the coarse grid's files) |
| `baltic_a2` | refused: depletable resources (plan task 3.1) | |
| `baltic_depensation` | accepted since the depensation gate was ported (plan task 3.6, the same day); it was refused when this table was first written | yes: the end-to-end tests and a small ensemble (`docs/perf/2026-11-19-csharp-depensation.md`) |
| `examples_bioen`, `baltic_ev` | refused: bioenergetics or genetics, which stay on the Python engine | |

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --engines python,python-control,csharp \
  --config data/benguela/benguela_all-parameters.csv --n 32 --years 15 --spinup-years 5 \
  --persist-results equiv_benguela.json
PYTHONPATH=. .venv/bin/python scripts/csharp/ab_eec.py --config data/benguela/benguela_all-parameters.csv \
  --pairs 5 --threads 1 14 --years 10
```

About 25 minutes and 4 minutes.
