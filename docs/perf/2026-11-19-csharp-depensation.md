# C# engine: the depensation gate

Date: 2026-11-19. Plan task 3.6 lists four recruitment features. This note is about one of
them, the depensation (Allee) gate, which the shipped `baltic_depensation` demo uses. The
other three (the recruitment ceiling, the thermal gate, spatial reproductive-volume egg
survival) are not ported and are still refused by name.

## What the gate is

After the stock-recruitment curve and the reproductive-volume gate, the eggs of a gated species
are multiplied by

    A = S^theta / (S50^theta + S^theta)

where `S` is the species' spawning stock at that step. `A` is one half at `S = S50`, goes to 1
above it and to 0 below it. A species whose stock was seeded at that step is spared, so that
the start from an empty stock cannot be trapped. Config keys:
`reproduction.depensation.gate.enabled`, and per species `.species.enabled.sp#`, `.s50.sp#`,
`.theta.sp#`. The gate has no Java counterpart.

## What was ported

- The bundle carries three arrays for the focal species (`reproduction.depensation.enabled`,
  `.s50`, `.theta`), only when a species is gated.
- The C# egg count applies the term after the reproductive-volume gate, in the Python engine's
  order. The order matters to the last bit: two multiplications in another order need not
  round alike.
- The C# reader refuses the arrays when the bundle does not declare the feature, and refuses a
  gated species with `s50 <= 0` or `theta < 1`, as the Python loader does.

## Evidence

| Check | Result |
|---|---|
| The reproduction stage against the Python engine's, on a real EEC state: theta 1, 2.5 and 4, the stock above and below its S50, a seeded species gated as well | every field of every school bit for bit; the gated species' eggs are the Allee term times the ungated count; the seeded species is untouched |
| The same with the reproductive-volume gate on the same species | bit for bit |
| The shipped demo, five years, the gate on and off (same seed) | year 1 identical in every species (cod is still seeded); by year 5 the gated cod_west is under half the ungated one |
| The shipped demo at 1, 2 and 14 threads, two seeds | identical records |
| Result digests with no gate in the config, against master | identical, 8 of 8 |
| Ensemble on the shipped demo (below) | PASS |

## What the gate does on the shipped demo

The demo gates cod_west with `s50 = 90 000 t` and `theta = 4`. Its own config calls these
numbers a placeholder. The stock is a few thousand tonnes, far below that S50, so the gate all
but shuts recruitment as soon as cod spawns for real (from the third year). The stock then
falls to nothing, the seeding rule starts it again, and the cycle repeats about every nine
years. One run of the C# engine, cod_west biomass in tonnes at the end of the year:

| Year | Gate on | Gate off |
|---|---|---|
| 1 | 48 | 48 |
| 3 | 1 913 | 2 011 |
| 5 | 938 | 3 506 |
| 10 | 3 | 10 093 |
| 13 | 3 675 | 16 643 |
| 20 | 2 | 19 592 |

So a whole-run comparison on this demo is a comparison of a trajectory driven by the gate and
by seeding, which is what the demo is for.

## Ensemble equivalence (spec 10.2)

`scripts/cross_engine_parity_440.py --engines python,python-control,csharp` on the demo as
shipped, 16 replicates per arm, 15 years, the first 2 dropped. The rule is the one used for
EEC, Bay of Biscay, Baltic and Benguela, calibrated on this run's own Python replicates.

| | Result |
|---|---|
| Verdict | **PASS** |
| Frozen rung (slack, ratio, variance) and its false-fail rate for identical engines | 4, 3.0, 6: 4.9% |
| Equivalent cells, C# against Python | 35 of 36 counted |
| Equivalent cells, Python against Python | 35 of 36 counted |
| Largest difference, Python against C# | 0.0158 in log10 (1.037x) |
| Largest difference, Python against Python | 0.0321 in log10 (1.077x) |
| Not counted | 6 cells: the two background species, one same number on every arm |

cod_west, the gated species:

| Metric | Python against C# (log10) | Python against Python (log10) | Variance ratio |
|---|---|---|---|
| Abundance | 0.0021 | -0.0045 | 0.86 |
| Biomass | -0.0003 | -0.0062 | 0.80 |
| Mean weight | -0.0024 | -0.0016 | 0.83 |
| Yield | -0.0009 | -0.0062 | 0.82 |

The one cell that is not equivalent is pikeperch's mean weight, and it is not equivalent
between the two Python arms either (-0.032) at the tighter margin that metric has. It is the
same cell that missed in the Baltic ensemble.

The ensemble ran on the tree before the merge of master of the same evening. Nothing in that
merge changes this demo's numbers (two keys no shipped config sets, and two features the demo
does not use).

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
PYTHONPATH=. .venv/bin/python -c "from pathlib import Path; from osmose.demo import osmose_demo; print(osmose_demo('baltic_depensation', Path('dep_demo'))['config_file'])"
PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --engines python,python-control,csharp \
  --config dep_demo/config/baltic_depensation_all-parameters.csv --n 16 --years 15 --spinup-years 2 \
  --persist-results equiv_baltic_depensation.json
```

About 25 minutes.
