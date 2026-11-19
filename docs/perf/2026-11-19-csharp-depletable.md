# C# engine: depletable resources

Date: 2026-11-19. Plan task 3.1. The shipped `baltic_a2` demo uses depletable resources, and
the C# engine refused it for that. It runs now.

## What the feature is

By default a resource (a plankton group, the benthos) is reset to its carrying capacity K
every step, so grazing has no memory. With `ltl.depletable.enabled` the biomass that predation
leaves is carried to the next step and regrown toward K
(`osmose.engine.resources.logistic_regrow`):

    B = max(B_carried, floor * K)
    B_new = min(K, B + rate * B * (1 - B / K))        (0 where K <= 0)

`ltl.depletable.floor` (5% by default) keeps an emptied cell from staying empty. The rate is
per resource and per step (`species.regrowth.rate.sp#`, default `ltl.regrowth.rate.default`).
A run starts with empty resources, so its first update regrows from the floor, not from K.

## What was ported

- The bundle carries the floor and one rate per resource, only when the feature is on.
- The C# update regrows the content of the run's resource buffer where it used to overwrite it.
  The arithmetic is the Python function's, in its order. With bottom oxygen scaling a resource
  (Baltic's benthos), K is scaled first and the regrowth is toward that K.
- The C# reader refuses the two entries without the feature, a floor outside [0, 1] and a rate
  that is negative or not finite.

## What it unblocks besides the demo

`scripts/calibrate_baltic.py --engine csharp` checks the config its evaluations run, after the
phase's edits. The two calibrations on the depletion base (`--a2` and `--phase a2r`) were
refused for this feature and are accepted now (`tests/test_calibrate_baltic_engine.py`). Nothing
was calibrated here. Anyone who runs one should read the caveat below first: the search would
run on the regrowth as it is.

Those calibrations run on the production Baltic base, where bottom oxygen scales the benthos.
The demo leaves that coupling out, so the benthos there is depletable and not oxygen-scaled.
The combination is checked in "With the oxygen coupling too", below.

## A caveat that is not about the port: negative biomass (issue #337)

The function caps its result at K and does not bound it below. When a resource is carried
**above** K, which happens every time K falls from one step to the next, the logistic term is
negative, and with a steep rate the result is below zero. On the shipped `baltic_a2` demo, on
the Python engine, diatoms and dinoflagellates (rate 5 per step) are negative in up to 312 of
2000 cells in 60 of 72 steps. Both engines' predation kernels skip a resource that is not
positive, so those cells hold none of it for the step.

This is the model both engines share. **The C# engine reproduces it bit for bit, sign
included, and everything below is measured on the current arithmetic.** Whether to change the
function, and whether the depletable-plankton experiment needs another look, is the owner's
decision (issue #337). If the function changes, `StepPreparation.Regrow` must change with it,
and the tests that compare the two will say so.

## Evidence

Three things must hold, and each is tested on its own (`tests/test_csharp_depletable.py`).

| What | Check | Result |
|---|---|---|
| The update regrows what it is handed as the Python engine does | the biomass a real run carried into a step, EEC and Baltic (Baltic's includes negative cells and the oxygen-scaled benthos) | bit for bit |
| | made-up biomass: none, under the floor, in between, at K, above K, and in cells with K = 0 | bit for bit |
| | the first step, from an empty buffer | bit for bit; below K, as it should be |
| The kernel's eating is written to the buffer the next update reads | one whole step, from the carried biomass to the state it leaves: schools and outputs, and the resources **after** predation (predation took from more than 100 resource cells) | tier L (1e-12) |
| Nothing between two steps resets the buffer | a run without fish, 48 steps: its resources are then a pure sequence of updates | bit for bit; one update from an empty buffer is not the same number |
| A run does not start from the last run's buffer | six runs in one process, 1, 2 and 14 threads, two seeds | identical records per seed |
| The shipped demo | accepted; with the resources reset instead, the run differs from the first record | |
| Result digests with the feature off, against master | | identical, 8 of 8 |

## Ensemble equivalence (spec 10.2)

`scripts/cross_engine_parity_440.py --engines python,python-control,csharp` on `baltic_a2` as
shipped, 16 replicates per arm, 15 years, the first 2 dropped.

| | Result |
|---|---|
| Verdict | **PASS** |
| Frozen rung (slack, ratio, variance) and its false-fail rate for identical engines | 6, 5.0, 10: 4.5% |
| Equivalent cells, C# against Python | 30 of 36 counted |
| Equivalent cells, Python against Python | 29 of 36 counted |
| Largest difference, Python against C# | 0.116 in log10 (1.31x), cod_east abundance |
| Largest difference, Python against Python | 0.062 in log10 (1.15x), the same cell |
| Not counted | 6 cells: the two background species |

**This config is noisy, and 16 replicates do not settle its weakest species.** Every cell that
is not equivalent against the C# engine is not equivalent between the two Python arms either:

| Cell | Python against C# (log10, 90% half-width) | Python against Python |
|---|---|---|
| cod_east abundance | 0.116 (0.161) | -0.062 |
| cod_east biomass | 0.087 (0.126) | -0.048 |
| cod_east yield | 0.081 (0.118) | -0.045 |
| cod_east mean weight | -0.029 (0.036) | 0.014 |
| herring mean weight | -0.007 (0.063) | -0.045 |
| pikeperch mean weight | -0.010 (0.022) | 0.005 |

cod_east runs uncalibrated in this variant (the demo's own text says so), and its replicates
spread widely: the interval on its abundance is wider than the difference itself. The rule
passes because the C# differences are within what the Python engine shows against itself,
with the rule's allowance. It does not show that cod_east is the same to within a few percent;
more replicates would be needed for that.

The ensemble ran before the merge of the seeding-period fix of the same evening (#334). That
fix changes nothing on this demo: measured, the same seed gives the same numbers over 25 years
with and without it.

## With the oxygen coupling too (follow-up, the same evening)

The demo plus the one include it leaves out (`osmose.configuration.oxygen;baltic_param-oxygen.csv`
appended to its master file): the benthos is then both oxygen-scaled and depletable, as on the
base the depletion calibrations use.

**What the tests cover of this case.** The update, bit for bit, on Baltic with the coupling
(above), and a mutation that applies the oxygen factor after the regrowth fails there. The
whole-step and the no-fish tests run on EEC, which has no coupling. So for the combination the
deterministic checks stop at the update, and the rest is the ensemble below.

**The ensemble at 16 replicates failed, and the failure was the draw.**

| Sample | Verdict | Equivalent cells (C#, control, of 36) | Largest difference: C#, control | cod_east abundance, Python minus C# (log10, 90% half-width) |
|---|---|---|---|---|
| 16 per arm, 15 years | **FAIL** | 30, 33 | 0.086, 0.0135 | +0.086 (0.093) |
| 32 per arm (those 16 and 16 new seeds each) | **PASS** | 33, 34 | 0.020, 0.032 | +0.018 (0.067) |
| The same, with all 64 C# replicates | **PASS** | 33, 34 | 0.029, 0.032 | -0.011 (0.055) |

The 16-replicate run failed one criterion: its largest difference, cod_east abundance, was more
than 5 times the control's largest. That is the cell that was largest on the demo too, with the
same sign (+0.116). The rule passed there because the control's largest difference was on that
cell too (0.062), and the rule allows 5 times the control's. Two ensembles agreeing looked like
an engine difference. It was not:

1. **The two ensembles are one draw seen twice.** The harness gives each arm fixed seeds, so
   both ensembles ran the same seeds. Seed by seed, the C# arm's cod_east in the one config
   predicts its cod_east in the other (correlation 0.81; 0.40 and 0.22 for the two Python arms).
2. **On new seeds the sign reverses.** The same config, Python on 32 new seeds and C# on 48:
   cod_east abundance, Python minus C#, is -0.046 (standard error 0.035).
3. **Pooled, there is nothing.** All 64 Python replicates against all 64 C# replicates: -0.015,
   90% interval 0.046 either way. A rank test gives p = 0.71.

The 32-per-arm sample was fixed by seed order before any value was looked at (each arm keeps its
16 replicates and takes the next 16 new seeds), and judged by the rule itself (`--rejudge`). Its
frozen rung is 5, 4.0, 8, with a false-fail rate of 3.7% for identical engines; at 16 replicates
the rung was 6, 5.0, 10 and the rate 7.6%. Three cells are not equivalent at 32, all of them mean
weights, which have the narrowest margin: cod_east and herring, in the control as well, and
pikeperch against the C# engine only (-0.0175, half-width 0.0137, which is 4%).

**What to take from it.**

- No engine difference shows on the combination, at a resolution of about 1.15x on cod_east and
  far finer on the other species (cod_west, flounder and sprat agree within 0.005 in log10).
- **16 replicates are too few on the depletable configs.** At 16, the 90% half-width of the
  engine difference on cod_east abundance is 0.09 to 0.16 in log10 there, against 0.002 on the
  depensation demo. The demo's pass at 16 was as weak as this fail. Use 32.
- **Ensembles on related configs are not independent evidence**, because they share their seeds
  and the C# arm's outcome follows its seed across a small change of config. A second ensemble
  that agrees with the first needs other seeds before it counts as a replication.

How it was run: the 16 x 15 ensemble with the command below on the edited master file; the new
seeds with the harness's own per-replicate function (`in_memory_rep`, 15 years, the first 2
dropped), Python seeds 30000 to 30031 and C# seeds 40000 to 40047; then the rule with
`--rejudge` on the assembled replicates.

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
PYTHONPATH=. .venv/bin/python -c "from pathlib import Path; from osmose.demo import osmose_demo; print(osmose_demo('baltic_a2', Path('a2_demo'))['config_file'])"
PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --engines python,python-control,csharp \
  --config a2_demo/config/baltic_a2_all-parameters.csv --n 16 --years 15 --spinup-years 2 \
  --persist-results equiv_baltic_a2.json
```

About 30 minutes.
