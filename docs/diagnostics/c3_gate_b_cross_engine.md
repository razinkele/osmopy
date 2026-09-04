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

**What changed between 4.3.3 and 4.4.1 to cause this: not identified, and the previous version of this
document pointed at the wrong class.** Corrected in the Task 9 fix round (2026-09-04) after review
ruling R38 flagged the original `EnergyBudget`/`W0` inference as targeting a class that structurally
cannot own length. Reading the 4.3.3 **source** first (`/home/razinka/osmose-reference`) settles where
length actually lives: `School.length` is set **only** from `School.incrementWeight(dw)` ->
`species.computeLength(weight * 1e6f)` (`School.java`, `Species.computeLength`/`computeWeight` =
`(weight/c)^(1/bPower)` and its inverse) — never in `EnergyBudget`. `incrementWeight` itself is called
from exactly one place in bioen mode: `EnergyBudget.getDw()`. Classic length growth
(`GrowthProcess`/`AbstractGrowth`, von Bertalanffy) never runs at all when bioen is enabled
(`SimulationStep.java:112-114` only instantiates `GrowthProcess` when `!isBioenEnabled()`), so there is
no second length pathway to consider.

That fixes *where to look*, but a full re-investigation — `javap -c` (bytecode, with method bodies, on
every class in the actual causal chain, both jars) plus checking which of those bodies are even
*reachable* by this specific Gate-B run — still could not identify a cause, for a stronger reason than
"not tried yet": **every class in the chain is proven bytecode-identical between 4.3.3 and 4.4.1, and
the two genuinely new mechanisms 4.4.1 adds are both provably inert on this config.** In order:

- `Species.computeLength`/`computeWeight`, `School.incrementWeight`/`setLength`/`incrementLength`: byte-
  identical (only constant-pool indices differ) in both jars.
- `EnergyBudget.getDw`: byte-identical. It clamps `dgrowth` to 0 before any weight change when
  `e_net <= 0` (`if (school.getENet() > 0) ... else 0`), so weight loss never runs through this path in
  either version — ruling out an earlier "length freezes on starvation" hypothesis for the growth path.
- `EnergyBudget.run()` **does** genuinely change between versions: 4.4.1 adds an `if (school.isOut())`
  branch that calls a wholly new method, `getDw_mig` (parametric growth from new `W0`/`c_rateBioen`
  fields, independent of `e_net`) instead of the normal path. **This never fires on Gate B's config**:
  the "OUT-schools sub-question" section below already instrumented this exact run and found 0/120
  out-of-domain occurrences across the full horizon (`movement.distribution.method.spN=random` for all
  8 species — there is no "outside the map" state to enter).
- 4.4.1 also adds a second e_net formula, `computeEnetLite` (uses `xi_crit`, `Imax`, a new temperature
  function), dispatched per-species via `EnetComputer[]` at `init()`-time. The dispatch condition,
  read directly from `EnergyBudget.init()`'s bytecode, is config key `species.bioenergetics.model.sp%d`:
  null/absent or `"full"` -> the **legacy** path; anything else -> lite. That key does not appear
  anywhere under `data/` (grepped) — every species in this config takes the legacy path. Disassembling
  it (`computeEnetLegacy` = `updateEgross` + `updateMaintenance` + `setENet(eGross-eMaint)`) confirms it
  is functionally identical, instruction-for-instruction, to 4.3.3's inlined `getEgross`/`getMaintenance`
  sequence (`updateMaintenance`≡`getMaintenance`, `updateEgross`≡`getEgross`, just renamed).
- The output writer (`MeanSizeOutput_Netcdf`, abundance-weighted mean of `school.getLength()`) and the
  reproduction/recruitment path that seeds new schools' initial length (`BioenReproductionProcess` /
  `ReproductionProcess.create_reproduction_schools`, which the refactor moved to the shared superclass
  but left byte-identical) were also checked and are byte-identical between jars.
- The Python-side config staging (`write_temp_config`, target_version="4.3.3" vs "4.4.1") was checked
  too: `species.length2weight.condition.factor/allometric.power`, `species.beta`, and
  `species.egg.weight/size` come out numerically identical in both staged configs, so this is not a
  config-staging artifact either.

**Net: every code path that is actually reachable by this Gate-B run, on both the weight->length
conversion and the reproduction/recruitment path that seeds it, is bytecode-identical between 4.3.3 and
4.4.1.** The real structural changes 4.4.1 does carry in this area (`computeEnetLite`, `getDw_mig`,
the new `xi_crit`/`W0` fields) are confirmed dead code for this specific config, not "the most plausible
candidate" as the previous version of this section claimed. The true cause of the `mean_size` divergence
remains **unidentified** — the search space has been narrowed to "somewhere outside the classes checked
above" (e.g. population initialization/seeding, not yet checked), not closed. Full confirmation would
require either a 4.4.1 source checkout or a targeted instrumented run against the populator classes,
neither done here.

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

## Reading `KS=nan` / `ci90=+-0.00` in the tables

Several `mean_weight` rows (all 8 species in the bioen run; Mackerel/Sole in several metrics in the
control run) show `KS=nan`. This is `tost()`'s `se == 0` short-circuit (`scripts/cross_engine_parity_440.py`,
pre-existing, untouched by the fix round below): both engines' 16-rep samples had exactly zero measured
variance on the log scale for that species/metric. It is a **real comparison on real data**, not a
skipped one — `d`/`eq` are still computed from the actual per-rep values — it just can't run a t-test
against zero pooled variance, so `ks_2samp` is never invoked (`KS` stays `nan`, its initializer) and the
row's `ci90=+-0.00` is the literal `0.0` return value, not a computed confidence interval, and `eq=Y`
there is a bare `abs(d) <= delta` threshold check, not a p-value verdict. By contrast `mean_size` never
hits this branch (its KS values are real, 0.00-0.09), so the length-divergence argument above (a tight,
deterministic offset, not noise) is unaffected. `tost()`'s docstring now documents this explicitly.

## Fix round 1 (2026-09-04): harness hardening, no re-run needed

Reviewed under `.superpowers/sdd/2026-08-30-baltic-c3-bioen-stage1/task-9-review.md`. Two changes to
`scripts/cross_engine_parity_440.py` worth recording here because they touch what "PASS" means, even
though neither changes the verdicts above:

- **R39**: `main()`'s per-metric species-overlap computation (`sp_all`) is now `comparable_species()`,
  and an empty result for any metric now feeds `uncompared_metrics` into `gate_verdict()`, which FAILs
  the gate rather than silently printing zero rows for that metric. This closes a narrower, per-metric
  version of the whole-arm `empty_arms` check that already existed (a Java arm producing real data for
  4 metrics but an empty frame for a 5th would previously pass clean). Proven with a constructed
  degenerate case (`comparable_species` returns `[]` when one present arm has no entries for a metric;
  `gate_verdict` then reports `FAIL (metric(s) with zero comparable species ...)`) and confirmed the real
  Gate B data is unaffected: every metric in both logged runs above already had non-empty
  `comparable_species` for every present arm (see "Java arm confirmation"), so this fix is a no-op on
  the committed PASS/REVIEW verdicts — verified both by unit test and by re-running the harness's own
  smoke invocation (`--n 1 --years 1 --spinup-years 0`, same config) end to end after the change.
- Distinguished explicitly from the `se == 0` case above: that case still has a real entry in every
  arm's dict (so it still appears in `comparable_species`'s output) — R39 only hardens the case where a
  species/metric pair has no entry in some arm's dict at all.

See `tests/test_cross_engine_parity_bioen_staging.py` for the full RED/GREEN proof (16/16 pass).

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
