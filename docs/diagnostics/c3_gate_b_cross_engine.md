# C3 Gate B — Java cross-engine parity of bioen-on

Date: 2026-09-04. Commit under test: `3aff2ca` (branch `c3-bioen-stage1`). `OSMOSE_JAR` set explicitly
to `/home/razinka/osmopy/osmose-java/osmose_4.3.3-jar-with-dependencies.jar` for both runs (unset by
default in this environment).

```
.venv/bin/python scripts/cross_engine_parity_440.py \
  --config data/examples_bioen/osm_all-parameters.csv \
  --n 16 --years 5 --engines python,4.3.3,4.4.1 --gate-engine 4.3.3 --require-nondegenerate

.venv/bin/python scripts/cross_engine_parity_440.py \
  --config data/examples/osm_all-parameters.csv \
  --n 16 --years 5 --engines python,4.3.3,4.4.1 --gate-engine 4.3.3 --require-nondegenerate
```

Horizon is 5 years per controller Ruling R37, not the 10 years in the task-9 brief's template — see
"Framing" below for why.

## Verdicts

- **bioen (gate-engine 4.3.3): `GATE (absolute Python<->4.3.3 equivalence + within 1 OoM): PASS`**
- **control, bioen off (gate-engine 4.3.3): `GATE (absolute Python<->4.3.3 equivalence + within 1 OoM): REVIEW: biomass:Hake, yield:Hake, mean_weight:Hake`**

Both runs actually executed the Java 4.3.3 arm — see "Java arm confirmation" below.

## Framing (mandatory)

**`data/examples_bioen` is a parity vehicle, not a calibrated ecosystem.** It is `data/examples` (the
bundled Bay-of-Biscay demo) plus an offline-fit bioen parameter set built to exercise the bioen code
paths in both engines, not a validated fishery. Established exhaustively before this task (a controlled
run flipping only `module.bioenergetics.enabled`): no species reaches maturity even under *classic*
(non-bioenergetic) growth on this fixture — HorseMackerel 7.21 cm, Mackerel 7.54, Sole 5.49, Hake
2.51 cm against maturity thresholds of 21/25/24/40 cm — so the demo's own growth/lifespan/maturity
parameters are internally inconsistent, independent of bioen. At the 5-year horizon Gate B runs at,
every population is still alive, `e_net` takes both signs, and growth accrues, so the bioen code paths
are genuinely exercised in both engines — that is all Gate B certifies. **No ecological conclusion may
be drawn from any number in this document.**

## mean_size: 4.3.3 agrees, 4.4.1 doesn't — and that's the correct result

In the bioen run, `mean_size` (individual length) is the one metric where the gate arm (4.3.3) and the
reported arm (4.4.1) tell different stories, and they tell them consistently:

| species | vs 4.3.3 (gated) | vs 4.4.1 (reported) |
|---|---|---|
| Anchovy | d=0.00, eq=Y | d=-0.35, eq=n |
| BlueWhiting | d=-0.00, eq=Y | d=-0.28, eq=n |
| Hake | d=0.00, eq=Y | d=-0.31, eq=n |
| HorseMackerel | d=-0.00, eq=Y | d=-0.37, eq=n |
| Mackerel | d=-0.00, eq=Y | d=-0.28, eq=n |
| Sardine | d=-0.00, eq=Y | d=-0.37, eq=n |
| Sole | d=-0.01, eq=Y | d=-0.39, eq=n |
| Sprat | d=-0.00, eq=Y | d=-0.34, eq=n |

Every species disagrees with 4.4.1 in the same direction (Python/4.3.3 length below 4.4.1's), by
0.28-0.39 log10 units (~1.9-2.4x), with a 90% CI half-width of 0.00 — i.e. this is a tight, deterministic
offset across all 16 reps, not noise. `biomass`, `abundance`, and `mean_weight` (the weight-based
metrics) agree with **both** Java versions to within 0.00-0.11 log10 for every species — so the
divergence is specific to length, not a general growth-trajectory mismatch.

**This is the correct outcome.** The C3 port was written against Java 4.3.3 (the spec's own line-number
citations are to 4.3.3), and it reproduces 4.3.3's length trajectory essentially exactly while correctly
disagreeing with a different, later engine version. It is also useful evidence about the harness itself:
Gate B can discriminate between the two Java versions rather than passing everything — the same
16-rep, 5-year ensemble that says "indistinguishable from 4.3.3" for every metric says "no" to 4.4.1 on
this one, at a threshold (`--delta-mean-weight`, log10(1.5) ~ 1.5x) five to eight times tighter than the
gap.

**What changed between 4.3.3 and 4.4.1 to cause this:** partially identified, not fully. No 4.4.1 Java
*source* exists anywhere in this environment (checked `/home/razinka/osmose-reference` — it stops at
the 4.3.3 release entry in `Releases.java`; checked `/srv/shiny-server/osmose-src` — jars only, no
source tree); only the two compiled jars are available. Decompiling method/field signatures with
`javap -p` on `fr.ird.osmose.process.bioen.EnergyBudget` in both jars shows a real structural change:

- 4.3.3's `EnergyBudget` has one energy-net computation path (`run()` calling `getMaintenance`,
  `getEgross`, `getDw`, `getDg`, `getRho`, ...).
- 4.4.1's `EnergyBudget` has **two** paths, `computeEnetLegacy` and `computeEnetLite` (plus
  `computeLiteTempFunction`), dispatched per-species through a new `EnetComputer[]` array of a new
  `GetENet` functional interface (two `init()`-time lambdas). It also carries new fields not present in
  4.3.3: `xi_crit`, `W0`, and inlined `temperature_tmin/tmax/topt/T1/T2`. `getMaintenance`/`getEgross`
  were renamed `updateMaintenance`/`updateEgross`, and a new `getDw_mig` method appears.

This is bytecode-level evidence (`javap` method/field listing, not a decompiled formula) of *what*
changed structurally, not *how* the formula changed — I did not decompile method bodies to find the
exact computation. Given weight-based metrics match both versions but length alone diverges by a
tight, deterministic, same-signed amount across every species, the most plausible locus is something
in this same rework that touches the weight-to-length conversion specifically (the new `W0` reference-
weight field is the best-fitting candidate) rather than the shared energy-budget/weight trajectory —
but that is an inference from the pattern, not a verified line of code. Full confirmation would require
either a 4.4.1 source checkout or a body-level bytecode trace, neither done here.

## Control's Hake REVIEW: a `data/examples` fixture artifact, not a port defect

The naive expectation would be that the untouched, "just classic growth" control config is *safer* than
the bioen fixture. It is the opposite: bioen passes cleanly (including Hake) while the control flags three
Hake metrics (`biomass` d=2.01-2.03, ~100x; `yield` d=-1.84 to -1.87, ~70x low; `mean_weight`
d=1.60-1.61, ~40x). `abundance` (d=0.30) and `mean_size` (d=-0.11) both stay within the gate for Hake
in the control run.

The reason traces to a known, *deliberately uncorrected* data defect. `b84d763` ("fix(data): correct
transposed accessibility matrix in examples_bioen") fixed the predation accessibility matrix for
`data/examples_bioen` only; its own message states `data/examples` and `data/examples_433_orig`
"carry the same defect and are NOT touched here: committed parity baselines were generated against
them, so correcting them would move Bay of Biscay parity outputs, which is a repo-wide decision."
`data/examples` is exactly the control config this task was told to run.

Reading `data/examples/predation/accessibility_matrix.csv` under the loader's own convention (row =
prey, column = predator — confirmed against `osmose/engine/accessibility.py`'s module docstring)
shows the Hake **column** (what Hake, as a predator, can access) is `0.0000` in every single row —
fish and plankton alike. That is a harder failure than the commit's own headline framing ("5 of 8
species get zero plankton access"): Hake specifically gets zero access to *everything*. The nonzero
values that were authored into the Hake *row* (0.6/0.5/0.5/0.3/0.3 under Anchovy/Sardine/Sprat/
HorseMackerel/BlueWhiting) land, under the loader's row=prey convention, as "how much each of those
species can eat Hake" — so the fixture's transposition turns the intended top piscivore into a species
that only gets eaten and cannot eat anything itself.

Both engines read this identical, broken input — the divergence is not from the config being wrong (it
is wrong for both), but from what each engine's code does with a species whose ingestion is
structurally always zero. Under complete, permanent food deprivation, `abundance` and `mean_size`
(length) still agree closely between Python and both Java versions; `biomass` and `mean_weight`
(both driven by per-fish weight, not length) do not — Java's chronically-starved Hake end up ~40x
heavier per fish than Python's for a similar headcount and a similar length. That points at a
zero-ingestion/starvation-numerics edge case that the two engines handle differently, but this task
did not trace the exact code path responsible — it is a genuinely out-of-scope debugging question for
an input class (permanent, total food lock) that no sane config should produce, only this
deliberately-uncorrected fixture does.

The load-bearing evidence that this is a control-fixture artifact and **not** a defect in the C3 bioen
port: the identical species, on the identical growth/predation code, passes the gate cleanly — every
metric, `eq=Y`, d in the same 0.09-0.11 range as every other species — in the bioen run, which uses
the *corrected* accessibility matrix. Nothing about Hake or its code path is unusual once it is given a
sane input.

## OUT-schools sub-question: could not resolve with this harness

`out_mortality` (`osmose/engine/processes/natural.py:184-208`) kills out-of-domain schools without
the survivor rescale Java's cause-agnostic `incrementNdead` is believed to apply at every other death
site; whether that matters for `e_net` (which does not reset every step, unlike `preyed_biomass`) was
unmeasured going into this task.

It remains unmeasured, but the reason is now concrete rather than "not gotten to yet": **this Gate-B
config never exercises the code path at all.** Instrumented directly (monkeypatching
`osmose.engine.processes.movement.movement` and observing `SchoolState.is_out` across the full
5-year, 120-step run against `data/examples_bioen`, seed 1234):

```
total movement-steps observed: 120
steps with >=1 out-of-domain school: 0
total (school,step) out-of-domain occurrences summed: 0
max out-of-domain schools in a single step: 0
out_mortality_rate per species: [0. 0. 0. 0. 0. 0. 0. 0.]
```

Two independent reasons, either one sufficient on its own: (1) all 8 species use
`movement.distribution.method.spN = random` (confirmed in `osm_param-movement.csv`), which
reassigns each school to a random cell within the grid's ocean cells every step — there is no
"outside the map" state to enter, so `is_out` is never set true. (2) `mortality.out.rate.spN` is absent
for every species in this config and defaults to 0.0 (`osmose/engine/config.py:757-758`), so even a
school that did go out would be killed at rate `1 - exp(-0) = 0`.

So Gate B — run on this fixture — is not evidence that the missing rescale is harmless; it is evidence
that this fixture cannot see it either way. Resolving the sub-question needs a config with genuine
area-restricted or seasonal movement maps and a nonzero `mortality.out.rate` (a Baltic-style config is
the natural candidate) run under bioen, instrumented the same way, with actual out-of-domain deaths
occurring during the run.

## Java arm confirmation

Both runs' logs show the 4.3.3 staging step firing and real per-species Java output feeding into the
gate — not a skip. From the bioen run:

```
[stage] 4.3.3: injected 24 .bioen key line(s) for bioen predation (predation.ingestion.rate.max.bioen and friends)
[stage] 4.3.3: injected species.biomass.nsteps.year (not bioen-specific — required by any NetCDF-file-forced resource species on this jar)
[run] done in 254s  (delta=0.48 log10 = 3.0x, gate-engine=4.3.3)
```

and a smoke-test run (`--n 1 --years 1`, same config, kept temp dir) of the exact staged 4.3.3 config
directly through `java -jar osmose_4.3.3-jar-with-dependencies.jar`, stdout:

```
osmose[info] Software version: 4.3.3
osmose[info] Configuration version: 4.3.3
osmose[info] Simulation 0 started...
```

Both `gate_b_bioen.log` and `gate_b_control.log` end with `exit=0` (see verbatim logs below), and the
per-species metric tables contain real, non-placeholder numbers for the `4.3.3` column in every metric
section — a skipped/absent arm would have printed `no python arm` / dropped the arm with a `[warn]`,
neither of which appears.

---

## bioen (gated 4.3.3, reported 4.4.1) — verbatim log

```
=== bioen (gated 4.3.3, reported 4.4.1) started 2026-09-04T17:37:35+03:00 ===
2026-09-04 17:37:36 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:37:36 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:37:36 [osmose.config] WARNING: 46 config key(s) are valid OSMOSE keys the Python engine does not implement; on this engine they have no effect. Use the Java engine if you need them: grid.lowright.lat, grid.lowright.lon, grid.mask.file, grid.upleft.lat, grid.upleft.lon, output.abundance.enabled, output.biomass.enabled, output.diet.stage.structure, output.diet.stage.threshold.sp0, output.diet.stage.threshold.sp1, and 36 more (see issue #123).
2026-09-04 17:37:42 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:37:47 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:37:55 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:38:04 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:38:13 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:38:22 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:38:31 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:38:40 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:38:49 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:38:58 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:39:07 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:39:15 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:39:24 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:39:33 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:39:42 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:39:51 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:40:00 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:40:09 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
2026-09-04 17:41:03 [osmose.config] INFO: Reading config from data/examples_bioen/osm_all-parameters.csv
[determinism] Python same-seed reproducible: True
[run] 16 reps x 3 engines x 5yr x 3 metrics ...
[stage] 4.3.3: injected 24 .bioen key line(s) for bioen predation (predation.ingestion.rate.max.bioen and friends)
[stage] 4.3.3: injected species.biomass.nsteps.year (not bioen-specific — required by any NetCDF-file-forced resource species on this jar)
[run] done in 254s  (delta=0.48 log10 = 3.0x, gate-engine=4.3.3)


[non-degeneracy check] (>=10% of 16 reps collapsed => FAIL)
  python   biomass    Anchovy                collapse_frac=0.00
  python   biomass    BlueWhiting            collapse_frac=0.00
  python   biomass    Hake                   collapse_frac=0.00
  python   biomass    HorseMackerel          collapse_frac=0.00
  python   biomass    Mackerel               collapse_frac=0.00
  python   biomass    Sardine                collapse_frac=0.00
  python   biomass    Sole                   collapse_frac=0.00
  python   biomass    Sprat                  collapse_frac=0.00
  python   abundance  Anchovy                collapse_frac=0.00
  python   abundance  BlueWhiting            collapse_frac=0.00
  python   abundance  Hake                   collapse_frac=0.00
  python   abundance  HorseMackerel          collapse_frac=0.00
  python   abundance  Mackerel               collapse_frac=0.00
  python   abundance  Sardine                collapse_frac=0.00
  python   abundance  Sole                   collapse_frac=0.00
  python   abundance  Sprat                  collapse_frac=0.00
  4.3.3    biomass    Anchovy                collapse_frac=0.00
  4.3.3    biomass    BlueWhiting            collapse_frac=0.00
  4.3.3    biomass    Hake                   collapse_frac=0.00
  4.3.3    biomass    HorseMackerel          collapse_frac=0.00
  4.3.3    biomass    Mackerel               collapse_frac=0.00
  4.3.3    biomass    Sardine                collapse_frac=0.00
  4.3.3    biomass    Sole                   collapse_frac=0.00
  4.3.3    biomass    Sprat                  collapse_frac=0.00
  4.3.3    abundance  Anchovy                collapse_frac=0.00
  4.3.3    abundance  BlueWhiting            collapse_frac=0.00
  4.3.3    abundance  Hake                   collapse_frac=0.00
  4.3.3    abundance  HorseMackerel          collapse_frac=0.00
  4.3.3    abundance  Mackerel               collapse_frac=0.00
  4.3.3    abundance  Sardine                collapse_frac=0.00
  4.3.3    abundance  Sole                   collapse_frac=0.00
  4.3.3    abundance  Sprat                  collapse_frac=0.00

==================== METRIC: biomass ====================
Anchovy                 4.3.3 d= -0.07 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.07 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d=  0.09 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.09 ci90=+-0.00 eq=Y
Hake                    4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.11 ci90=+-0.00 eq=Y
HorseMackerel           4.3.3 d=  0.09 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.09 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.11 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d=  0.10 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.10 ci90=+-0.00 eq=Y
Sole                    4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.11 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.10 ci90=+-0.00 eq=Y
==================== METRIC: yield ====================
Anchovy                 4.3.3 d= -0.24 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.24 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d= -0.06 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.06 ci90=+-0.00 eq=Y
Hake                    4.3.3 d= -0.07 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.06 ci90=+-0.00 eq=Y
HorseMackerel           4.3.3 d= -0.06 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.06 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d= -0.07 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.07 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d= -0.06 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.06 ci90=+-0.00 eq=Y
Sole                    4.3.3 d= -0.07 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.07 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d= -0.07 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.07 ci90=+-0.00 eq=Y
==================== METRIC: abundance ====================
Anchovy                 4.3.3 d= -0.07 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.07 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d=  0.09 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.09 ci90=+-0.00 eq=Y
Hake                    4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.11 ci90=+-0.00 eq=Y
HorseMackerel           4.3.3 d=  0.09 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.09 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.11 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d=  0.10 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.10 ci90=+-0.00 eq=Y
Sole                    4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.11 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d=  0.11 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.10 ci90=+-0.00 eq=Y
==================== METRIC: mean_weight ====================
Anchovy                 4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Hake                    4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
HorseMackerel           4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Sole                    4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
==================== METRIC: mean_size ====================
Anchovy                 4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.35 ci90=+-0.00 eq=n
BlueWhiting             4.3.3 d= -0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.28 ci90=+-0.00 eq=n
Hake                    4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=0.09  | 4.4.1 d= -0.31 ci90=+-0.00 eq=n
HorseMackerel           4.3.3 d= -0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.37 ci90=+-0.00 eq=n
Mackerel                4.3.3 d= -0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.28 ci90=+-0.00 eq=n
Sardine                 4.3.3 d= -0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.37 ci90=+-0.00 eq=n
Sole                    4.3.3 d= -0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.39 ci90=+-0.00 eq=n
Sprat                   4.3.3 d= -0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.34 ci90=+-0.00 eq=n
GATE (absolute Python<->4.3.3 equivalence + within 1 OoM): PASS
=== bioen finished 2026-09-04T17:42:02+03:00 exit=0 ===
```

## control (bioen off) — verbatim log

```
=== control (bioen off) started 2026-09-04T17:42:02+03:00 ===
2026-09-04 17:42:03 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:03 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:03 [osmose.config] WARNING: 46 config key(s) are valid OSMOSE keys the Python engine does not implement; on this engine they have no effect. Use the Java engine if you need them: grid.lowright.lat, grid.lowright.lon, grid.mask.file, grid.upleft.lat, grid.upleft.lon, output.abundance.enabled, output.biomass.enabled, output.diet.stage.structure, output.diet.stage.threshold.sp0, output.diet.stage.threshold.sp1, and 36 more (see issue #123).
2026-09-04 17:42:08 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:13 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:22 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:30 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:38 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:46 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:42:54 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:03 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:10 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:18 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:26 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:34 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:42 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:51 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:43:59 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:44:07 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:44:15 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:44:23 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
2026-09-04 17:45:05 [osmose.config] INFO: Reading config from data/examples/osm_all-parameters.csv
[determinism] Python same-seed reproducible: True
[run] 16 reps x 3 engines x 5yr x 3 metrics ...
[stage] 4.3.3: injected species.biomass.nsteps.year (not bioen-specific — required by any NetCDF-file-forced resource species on this jar)
[run] done in 217s  (delta=0.48 log10 = 3.0x, gate-engine=4.3.3)


[non-degeneracy check] (>=10% of 16 reps collapsed => FAIL)
  python   biomass    Anchovy                collapse_frac=0.00
  python   biomass    BlueWhiting            collapse_frac=0.00
  python   biomass    Hake                   collapse_frac=0.00
  python   biomass    HorseMackerel          collapse_frac=0.00
  python   biomass    Mackerel               collapse_frac=0.00
  python   biomass    Sardine                collapse_frac=0.00
  python   biomass    Sole                   collapse_frac=0.00
  python   biomass    Sprat                  collapse_frac=0.00
  python   abundance  Anchovy                collapse_frac=0.00
  python   abundance  BlueWhiting            collapse_frac=0.00
  python   abundance  Hake                   collapse_frac=0.00
  python   abundance  HorseMackerel          collapse_frac=0.00
  python   abundance  Mackerel               collapse_frac=0.00
  python   abundance  Sardine                collapse_frac=0.00
  python   abundance  Sole                   collapse_frac=0.00
  python   abundance  Sprat                  collapse_frac=0.00
  4.3.3    biomass    Anchovy                collapse_frac=0.00
  4.3.3    biomass    BlueWhiting            collapse_frac=0.00
  4.3.3    biomass    Hake                   collapse_frac=0.00
  4.3.3    biomass    HorseMackerel          collapse_frac=0.00
  4.3.3    biomass    Mackerel               collapse_frac=0.00
  4.3.3    biomass    Sardine                collapse_frac=0.00
  4.3.3    biomass    Sole                   collapse_frac=0.00
  4.3.3    biomass    Sprat                  collapse_frac=0.00
  4.3.3    abundance  Anchovy                collapse_frac=0.00
  4.3.3    abundance  BlueWhiting            collapse_frac=0.00
  4.3.3    abundance  Hake                   collapse_frac=0.00
  4.3.3    abundance  HorseMackerel          collapse_frac=0.00
  4.3.3    abundance  Mackerel               collapse_frac=0.00
  4.3.3    abundance  Sardine                collapse_frac=0.00
  4.3.3    abundance  Sole                   collapse_frac=0.00
  4.3.3    abundance  Sprat                  collapse_frac=0.00

==================== METRIC: biomass ====================
Anchovy                 4.3.3 d= -0.09 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.09 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Hake                    4.3.3 d=  2.01 ci90=+-0.06 eq=n KS=0.00  | 4.4.1 d=  2.03 ci90=+-0.05 eq=n
HorseMackerel           4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d=  0.02 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.02 ci90=+-0.00 eq=Y
Sole                    4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d=  0.03 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.03 ci90=+-0.00 eq=Y
==================== METRIC: yield ====================
Anchovy                 4.3.3 d= -0.13 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.13 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d= -0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.01 ci90=+-0.00 eq=Y
Hake                    4.3.3 d= -1.84 ci90=+-0.03 eq=n KS=0.00  | 4.4.1 d= -1.87 ci90=+-0.03 eq=n
HorseMackerel           4.3.3 d= -0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.01 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d= -0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.01 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d= -0.02 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.02 ci90=+-0.00 eq=Y
Sole                    4.3.3 d= -0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.01 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d= -0.03 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.03 ci90=+-0.00 eq=Y
==================== METRIC: abundance ====================
Anchovy                 4.3.3 d= -0.10 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.10 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Hake                    4.3.3 d=  0.30 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.30 ci90=+-0.00 eq=Y
HorseMackerel           4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d=  0.02 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.02 ci90=+-0.00 eq=Y
Sole                    4.3.3 d=  0.01 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.01 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d=  0.03 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.03 ci90=+-0.00 eq=Y
==================== METRIC: mean_weight ====================
Anchovy                 4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Hake                    4.3.3 d=  1.60 ci90=+-0.01 eq=n KS=0.00  | 4.4.1 d=  1.61 ci90=+-0.01 eq=n
HorseMackerel           4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Sole                    4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d=  0.00 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d=  0.00 ci90=+-0.00 eq=Y
==================== METRIC: mean_size ====================
Anchovy                 4.3.3 d= -0.02 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.02 ci90=+-0.00 eq=Y
BlueWhiting             4.3.3 d= -0.02 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.02 ci90=+-0.00 eq=Y
Hake                    4.3.3 d= -0.11 ci90=+-0.01 eq=Y KS=0.00  | 4.4.1 d= -0.11 ci90=+-0.01 eq=Y
HorseMackerel           4.3.3 d= -0.01 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.01 ci90=+-0.00 eq=Y
Mackerel                4.3.3 d= -0.02 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d= -0.02 ci90=+-0.00 eq=Y
Sardine                 4.3.3 d= -0.02 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.02 ci90=+-0.00 eq=Y
Sole                    4.3.3 d= -0.01 ci90=+-0.00 eq=Y KS=nan  | 4.4.1 d= -0.01 ci90=+-0.00 eq=Y
Sprat                   4.3.3 d= -0.03 ci90=+-0.00 eq=Y KS=0.00  | 4.4.1 d= -0.03 ci90=+-0.00 eq=Y
GATE (absolute Python<->4.3.3 equivalence + within 1 OoM): REVIEW: biomass:Hake, yield:Hake, mean_weight:Hake
=== control finished 2026-09-04T17:45:50+03:00 exit=0 ===
```
