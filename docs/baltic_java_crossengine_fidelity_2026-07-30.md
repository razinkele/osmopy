# Baltic cross-engine fidelity — why the Java comparison was invalid

**Date:** 2026-07-30 · **Trigger:** investigating "Java drives cod_east extinct"

The investigation started from a false premise and ended at a structural incomparability. Five
findings, in dependency order; the first two invalidate prior conclusions, the third is a Java
limitation we cannot configure around.

## 1. The extinction never happened (harness bug, fixed)

`certify_java` read Java's biomass CSV by the RAW config species name. `reconcile_config_for_java`
rewrites `species.name.sp*` to Java's stripped internal form, so Java writes its columns as
`codwest`/`codeast`. The lookup missed and fell through to `series.get(sp, [0.0])`, fabricating a
zero series that scored as min 0, late-mean 0, "COLLAPSE".

Real 50-yr Java values: `cod_west` ≈ **354 kt**, `cod_east` ≈ **949 kt** — 14× and 11× *above*
envelope, not extinct. Fixed in `2250824`: names resolve through `sanitize_java_name`, and a missing
column now **raises** instead of substituting zero.

The zeros were byte-identical across the 2026-07-29 and 2026-07-30 runs because a failed dict lookup
is deterministic and independent of any config change. That invariance was misread as corroboration
for the `fsh8` fix having no effect on cod; in fact the number was never a measurement. Only species
whose names contain no `_`/`-` were ever read correctly.

**The old causal story is dead.** "Cod goes extinct → prey release → mid-trophic explosion" cannot
hold when the predator is itself inflated an order of magnitude.

## 2. The two engines report different quantities

`data/baltic/baltic_param-output.csv` sets `output.cutoff.enabled;true` and
`output.cutoff.age.sp0..sp8;0.5` — exclude schools younger than 0.5 yr from reported biomass.

- **Python** applies it (`osmose/engine/output.py:969`).
- **Java** does not: `osmose/java_background_staging.py:189,194` forces
  `output.cutoff.enabled=false`, commented only "belt-and-suspenders".

So Python reports adult biomass and Java reports everything including young-of-year. The
incomparability is real.

> **⚠ But it is numerically negligible — this does NOT explain the divergence. Measured, not
> assumed:** re-running Python with the 9 `output.cutoff.age.spN` keys removed moves its biomass by
> only **1.0–1.4×** (see the uncut table below). A 0.5-yr cutoff cannot account for 50–70×, because
> larvae and young-of-year carry almost no *mass* however numerous they are. An earlier draft of this
> note claimed the cutoff was "substantially" the cause and that the pattern matched the small,
> high-fecundity species; that claim was wrong and is retracted. Findings 3 and 4 below remain true
> as facts about the two engines — they just don't move the numbers.

## 3. Java 4.4.1 *cannot* apply the cutoff here (hard limitation)

The override is necessary, not sloppy. Re-enabling it crashes Java:

```
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: Index 9 out of bounds for length 9
	at fr.ird.osmose.output.OutputRegion.include(OutputRegion.java:168)
	at fr.ird.osmose.output.OutputRegion.getSelectivity(OutputRegion.java:163)
	at fr.ird.osmose.output.SpeciesOutput.update(SpeciesOutput.java:120)
```

Java sizes the cutoff-age array to the 9 focal species, then indexes it at 9 — the first background
species (runtime slot `n_focal + 0` = GreySeal). Python gets this right: `config.py:922` pads the
array with `[0.0] * n_bkg`. Java does not.

**Consequence: Java 4.4.1 cannot apply an output cutoff to any config with background species.**
The Baltic config has two (GreySeal, Cormorant), so this is unavoidable for this model.

## 4. `output.cutoff.enabled` is a dead toggle on the Python engine

Python decides the cutoff purely from the presence of `output.cutoff.age.spN` keys
(`osmose/engine/config.py:915-923`) and **never reads `output.cutoff.enabled`**. Across the whole
package the key appears only in the Java staging, the `config_validation` allowlist (as a
known-but-unread key), and `osmose/schema/output.py:59`.

Because it is a schema field with no engine-support marker, the UI renders an "Enable output cutoff
filtering" checkbox that has **no effect on Python runs** — set it false and biomass is still cut.
Combined with finding 3, there is **no value of `output.cutoff.enabled` that makes the two engines
report the same quantity**: Python always cuts, Java can never cut.

## 5. Four Python-only mechanisms are invisible to Java

Zero occurrences anywhere in `osmose-4.4.1-jar-with-dependencies.jar`:

| key | in jar | active in Baltic |
|---|---|---|
| `reproduction.rv.gate.*` | 0 hits | **true** — `raw_cap`, ref 150, **sp8 only** |
| `movement.salinity.gate.*` | 0 hits | **true** — **sp0 and sp8** |
| `reproduction.recruitment.ceiling.*` | 0 hits | absent (feature off) |
| `predation.accessibility.dynamic.*` | 0 hits | not set |
| `ltl.depletable.*` | 0 hits | not set |

The two active ones are cod-specific, so they bias cod upward on Java. But they are **not** the
dominant driver: the largest divergences are in *ungated* species.

## Quantified divergence (50 yr, `--params current`)

Python is 5-seed worst-case; Java is a single run (Java 4.4.1 has no numeric seed — see below).

| species | Python (cut) | Java (uncut) | J/P | Py/envHi | Ja/envHi | Python-only gate |
|---|---|---|---|---|---|---|
| cod_west | 14,343 | 353,850 | 24.7× | 0.6 | 14.2 | salinity |
| cod_east | 83,000 | 948,866 | 11.4× | 1.0 | 11.2 | salinity + RV |
| herring | 2,591,007 | 1,916,430 | 0.7× | 0.9 | 0.6 | — |
| sprat | 1,060,584 | **0** | extinct | 0.4 | 0.0 | — |
| flounder | 40,502 | 3,011,791 | 74.4× | 0.4 | 30.1 | — |
| perch | 45,089 | 2,279,394 | 50.6× | 0.9 | 45.6 | — |
| pikeperch | 1,400,444 | 8,139,264 | 5.8× | **56.0** | 325.6 | — |
| smelt | 682,441 | 4,019,851 | 5.9× | 0.5 | 2.7 | — |
| stickleback | 77,578 | 5,177,973 | 66.7× | 0.4 | 25.9 | — |

Python sits near or inside envelope for 8 of 9 species. The exception, pikeperch at 56× over, is the
known percid overshoot — pre-existing in Python and independent of everything here.

## Also corrected: the "single seed 42" claim

`certify_java` accepted a `seed` argument and never passed it to Java. Java 4.4.1 exposes only
`simulation.fixedseed.enabled` (a toggle), not a numeric seed, so no Java run can be pinned to a
seed. The parameter was removed rather than left as a silent no-op (`6a4e224`).

## The cutoff test (route 1), measured

Python re-run with all 9 `output.cutoff.age.spN` keys removed, 50 yr, seed 42:

| species | Py UNCUT | Py CUT | uncut/cut | Java UNCUT | Java/Py-uncut |
|---|---|---|---|---|---|
| cod_west | 13,913 | 14,343 | 1.0× | 353,850 | **25.4×** |
| cod_east | 83,665 | 83,000 | 1.0× | 948,866 | **11.3×** |
| herring | 3,691,056 | 2,591,007 | 1.4× | 1,916,430 | **0.52×** |
| sprat | 1,105,256 | 1,060,584 | 1.0× | 0 | extinct |
| flounder | 41,272 | 40,502 | 1.0× | 3,011,791 | **73.0×** |
| perch | 45,505 | 45,089 | 1.0× | 2,279,394 | **50.1×** |
| pikeperch | 1,476,229 | 1,400,444 | 1.1× | 8,139,264 | **5.5×** |
| smelt | 726,219 | 682,441 | 1.1× | 4,019,851 | **5.5×** |
| stickleback | 85,383 | 77,578 | 1.1× | 5,177,973 | **60.6×** |

**Route 1 is dead as an explanation.** Equalising the reported quantity leaves the divergence
essentially untouched. Whatever drives it is dynamical, not a reporting artifact.

### The pattern that remains

With the reporting mismatch eliminated, the residual signal is sharp and worth stating precisely:

- **Clupeids fail on Java.** sprat extinct (Python 1.1 Mt); herring at 0.52× (the only species where
  Java is *below* Python).
- **Everything else booms**, 5.5–73× above Python.

That is a trophic restructuring, not a uniform scaling — so a single global mortality or forcing
factor does not fit either. One tension to resolve: cod eats clupeids, yet cod is 11–25× *up* on
Java while its clupeid prey collapse. Any candidate mechanism has to explain both halves.

**Next diagnostic — attempted, and BLOCKED by finding 6.**

## 6. Python's `mortalityRate-*.csv` emits death COUNTS, not rates

Attempting the mortality-decomposition comparison surfaced a separate defect that makes it
impossible as-is.

| | Java 4.4.1 | Python engine |
|---|---|---|
| sprat `Mpred` (egg/juv/adult) | 2.8e-4 / 0.039 / 1.1e-4 | — |
| sprat `Madd` (egg/juv/adult) | 2.05 / 0.143 / 0.063 | — |
| sprat `Predation` (flat) | — | **5.46e9** |
| sprat `Additional` (flat) | — | **5.16e13** |

Roughly **13 orders of magnitude** apart. Confirmed from source, not inferred from magnitude:
`osmose/engine/simulate.py:805-813` (`_collect_mortality_by_cause`) sums `state.n_dead[:, cause]`
— the **number of individuals dead** — with no division by abundance-at-risk.
`osmose/engine/output.py:388-404` then writes that array to `Mortality/{prefix}_mortalityRate-{sp}_Simu0.csv`
under the header `"Mortality rates per time step for {sp}"`, with a docstring stating it is
"matching Java format".

Three things are wrong:

1. **The quantity is counts, not rates** — the filename, header line and column semantics all claim
   rates.
2. **The structure does not match Java.** Java writes a 3-line header with a (cause, stage)
   MultiIndex — `Mpred`/`Mstarv`/`Madd`/`F`/`Zout`/`Mfor`/`Mdis`/`Mage` × Eggs/Juvenil/Adult. Python
   writes a 1-line header with flat per-cause columns and no stage split.
3. **Any Python-side mortality analysis reading these files is wrong**, and the cross-engine
   mortality comparison — the diagnostic needed to explain the divergence in this note — cannot be
   done until the units are fixed.

Fixing it is not mechanical: converting counts to rates requires choosing the abundance-at-risk
denominator per cause and per life stage, and matching Java's structure means adding the stage
split. That is a modelling decision, recorded here rather than guessed at.

### #140 fixed, diagnostic run — and it is INCONCLUSIVE

`053742b` converts the Python output to instantaneous rates. Verified on real 12-yr Baltic output:
max rate per species is now **8.6 / 16.7 / 7.1** (sprat / cod_east / herring) against ~1e13 before.

Mean rates over 12 yr, Java per life stage vs Python whole-population:

| species | cause | Java Eggs | Java Juv | Java Adult | Python flat |
|---|---|---|---|---|---|
| sprat | Mpred | 0.0011 | **2.3612** | 1.3148 | 0.0074 |
| sprat | Madd | 2.0550 | 0.2085 | 0.2014 | 6.5322 |
| cod_east | Mpred | 0.0002 | **1.7233** | 0.1630 | 0.0002 |
| cod_east | Madd | 5.0974 | 0.8734 | 0.8318 | 11.5424 |
| herring | Mpred | 0.0277 | **1.9927** | 0.5800 | 0.0218 |
| herring | Madd | 2.4114 | 2.1819 | 2.0815 | 6.4683 |

**Why it does not answer the question.** Python's flat rate is a whole-population rate, and eggs
outnumber every other stage by orders of magnitude, so it is numerically dominated by the egg stage.
That shows in the data: Python's predation rate tracks Java's *egg* rate (sprat 0.0074 vs 0.0011;
herring 0.0218 vs 0.0277; cod_east 0.0002 vs 0.0002) and is 2–3 orders of magnitude below Java's
juvenile and adult rates. Comparing it against those is apples-to-oranges — the same
different-quantities trap as the cutoff, one level down.

**The one lead it did produce:** Java's *juvenile predation* is extreme and consistent across all
three species — 1.7–2.4, an order of magnitude above its own adult rates and far above its egg
rates. On Java, juveniles are being very heavily predated. Python has no per-stage figure to compare
against, so whether this is a genuine engine difference or normal for the model is untestable right
now. Python's additional mortality also runs 2–3× Java's egg-stage `Madd`, which is suggestive but
rests on the same confounded comparison.

**Blocking prerequisite, revised.** Completing the life-stage split (the unfinished half of #140) is
no longer optional polish — it is the gate on diagnosing the divergence at all. A whole-population
rate cannot be compared against a per-stage one.

### CAUSATION, FINAL: the predation-rate signal is a DENOMINATOR EFFECT — "cause" is refuted

Early-window test redone with **prey** biomass tracked, which the previous version omitted. The
correlation is unambiguous — the rate ratio collapses toward 1 exactly as the prey ratio rises to 1:

**herring**

| step | rate J/P | prey J/P | predator J/P |
|---|---|---|---|
| 8 | **16.23** | **0.45** | 0.56 |
| 9 | **8.99** | 0.66 | 0.71 |
| 10 | **6.00** | 0.88 | 0.82 |
| 11 | 1.66 | **1.03** | 0.93 |
| 12–17 | 0.22–2.40 (scatter) | 1.15–1.36 | 0.99–1.09 |

**sprat**

| step | rate J/P | prey J/P |
|---|---|---|
| 9 | 1.39 | 0.05 |
| 10 | **12.61** | 0.25 |
| 11 | 1.86 | 0.91 |
| 12–17 | 0.03–1.88 (scatter) | 2.16–4.81 |

**Every elevated rate sits where Java's prey stock is smaller.** Once prey biomass equalises (herring
step 11, prey J/P 1.03) the rate ratio drops to ~1 and thereafter scatters with no systematic
elevation. Since rate = eaten / **prey** biomass, this is the denominator moving, not the kernel.

**Conclusion: Java's predation process is NOT more intense.** That is independently consistent with the
corrected per-pair kernel, where Java takes *less* clupeid biomass per unit predator biomass
(J/P 0.15–1.29). Two independent measurements now agree, having previously appeared to conflict.

**The real upstream signal is early clupeid biomass.** Java starts the window with far less — sprat
J/P 0.01–0.25 at steps 8–10, herring 0.45–0.88 — while predator biomass is also *lower* (0.56–0.82), so
predation cannot explain it. The question moves upstream to **recruitment / early survival**, not
predation.

**Every earlier causation reading in this note is superseded by this one.** The sequence was: cause
(wrong — instrument), cause (wrong — no prey denominator), denominator effect (this). The prior sections
are retained for the audit trail.

### SUPERSEDED — DIRECTION OF CAUSATION (re-tested after `e7db9f6`): evidence favours CAUSE

Re-run with the fixed collector. **In-run control:** `Additional` J/P = **1.00 at every step** for both
species — the imposed rate agrees, so the instrument is verified within the run rather than assumed.

| step | predator biomass J/P | sprat juv predation J/P | herring juv predation J/P |
|---|---|---|---|
| 8 | 0.55 | 0.01 | **5.72** |
| 9 | — | 0.05 | **6.12** |
| 10 | 0.79 | **6.37** | 2.02 |
| 11 | — | **6.39** | 1.17 |
| 12 | 1.06 | 1.23 | 1.96 |
| 13 | — | 3.43 | 0.89 |
| 14 | 1.10 | 0.31 | 0.65 |
| 15 | — | 1.73 | 0.92 |

**The decisive asymmetry: Java predates clupeid juveniles ~6× harder while holding LESS predator
biomass.** At steps 10–11 Java's sprat-juvenile predation is 6.4× Python's with Java's predator field at
0.79× Python's; herring shows 5.7–6.1× at steps 8–9 with the predator field at 0.55×. More predation
from fewer predators cannot be a consequence of predator abundance — the abundance difference runs the
wrong way. By steps 13–15 the ratios scatter around 1 as the fields converge (J/P 1.06 → 1.10).

**Conclusion: the predation process itself differs, and it is upstream.** Java extracts more predation
per unit predator biomass. That is consistent with everything downstream — clupeids collapse on Java,
their predators are released and boom, and the adult-stage asymmetry (sprat 3.01×, herring 1.82×,
cod_east 0.82×) persists in steady state.

**Confidence: moderate, not high.** Absolute rates in the early window are small (sprat 0.0019 vs
0.0003), Java contributes a single unseeded stochastic draw, and the evidence is two species over a few
steps. What raises it above the earlier attempts is the in-run control plus the direction of the
predator-biomass asymmetry, which no consequence-side explanation accommodates.

**Next step attempted — BLOCKED. The per-pair kernel comparison cannot be done with current outputs.**

Discovery run (12 yr, both engines) shows the two trophic outputs are not comparable quantities:

| output | Java | Python |
|---|---|---|
| `predatorPressure` (absolute prey biomass eaten per pair) | `(444, 33)` | **absent** |
| `dietMatrix` | `(444, 33)` long: `Time`, `Prey` rows × predator **size-stage** columns (`codwest in [0,10[`, `[10,30[`, `[30,inf[`) | `(12, 100)` wide: flat `predator_prey` columns, one row per year |

Two hard obstacles:

1. **Python does not write `predatorPressure` at all.** That is the only output carrying absolute prey
   biomass eaten per predator-prey pair, and it is exactly what the standing hypothesis needs — "Java
   extracts more predation per unit predator biomass" requires biomass-eaten normalised by predator
   biomass. It cannot be measured on the Python side today.
2. **The `dietMatrix` outputs differ in orientation AND dimensionality.** Java resolves predator size
   stages; Python emits flat predator×prey pairs with no size dimension. Java is long-form
   (prey rows), Python wide-form (one row per year).

**What is still testable:** aggregating Java's size-stage columns to whole-predator totals allows a
comparison of diet **fractions** per predator-prey pair — i.e. kernel *shape*, which prey each predator
selects. That is a real test of accessibility/size-selectivity, but it does **not** test per-unit-biomass
intensity, which is the actual hypothesis.

Deliberately not attempted: computing a proxy by multiplying Python's diet fractions by an assumed
consumption rate. That would compare a modelled quantity against Java's measured one and reproduce the
same different-quantities error that produced three false conclusions earlier in this investigation.

**To unblock:** add `predatorPressure` to the Python engine's trophic outputs (absolute biomass eaten per
predator-prey pair, matching Java's semantics), and optionally the predator size-stage dimension to
`dietMatrix` for full parity. — **Done in `88f038d`.** No new computation was needed: `write_diet_csv`
already emitted absolute biomass, so this engine's `dietMatrix` held Java's *predatorPressure* quantity
while Java's `dietMatrix` holds percentages. Same filename, different quantity per engine.

### Kernel comparison, CORRECTED (`1bc7888`) — and it undercuts the predation-process conclusion

Per-pair predation per unit predator biomass, after fixing the `predatorPressure` convention:

| prey | predator | Java | Python | J/P |
|---|---|---|---|---|
| sprat | cod_west | 0.14055 | 0.14894 | 0.94 |
| sprat | cod_east | 0.31449 | 0.44207 | 0.71 |
| sprat | perch | 0.06028 | 0.04675 | 1.29 |
| sprat | pikeperch | 0.03819 | 0.07130 | 0.54 |
| herring | cod_west | 0.21269 | 0.34431 | 0.62 |
| herring | cod_east | 0.07357 | 0.15811 | 0.47 |
| herring | perch | 0.06448 | 0.37966 | 0.17 |
| herring | pikeperch | 0.05257 | 0.35997 | 0.15 |

The 17–100× artefact is gone. **But Java now predates clupeids at or BELOW Python per unit predator
biomass (0.15–1.29, mostly < 1)** — the opposite direction from the mortality-based measurement, which
had Java's adult clupeid predation at 1.8–3.0× Python's.

**The two reconcile, and the reconciliation is the important part.** They are different quantities:

* mortality rate on prey = biomass eaten **/ prey biomass**
* this kernel = biomass eaten **/ predator biomass**

Java's clupeid stocks are far smaller (sprat → 0, herring 0.52×). So the *same or less* absolute
predation, divided by a much smaller prey stock, yields a much higher per-prey mortality **rate**. Both
measurements can hold simultaneously with no contradiction.

**Consequence: the "Java's predation process is more aggressive" reading is not supported.** The
elevated mortality rate is consistent with a **denominator effect — depleted prey — rather than a more
intense kernel.** Per unit predator biomass Java's predators actually take *less* clupeid biomass.

**This reopens the direction of causation.** The earlier early-window test rested on Java's predation
*rate* being ~6× higher while predator biomass was lower; but prey biomass was never measured in that
test, and if Java's clupeid prey were already depleted at those steps the rate difference is again a
denominator artefact. The test needs redoing with prey biomass tracked alongside.

**Standing conclusion, downgraded:** the divergence is in emergent predation *outcomes*, but the
evidence no longer supports the predation kernel itself being more intense on Java. Both remaining
candidates — kernel intensity vs. prey-depletion feedback — are live.

### Superseded: kernel comparison run — result REJECTED (wrong `predatorPressure` convention)

Per-pair predation per unit predator biomass (12 yr, Java size stages summed):

| prey | predator | Java | Python | J/P |
|---|---|---|---|---|
| sprat | cod_west | 0.16103 | 3.57447 | 0.05 |
| sprat | cod_east | 0.32182 | 10.60964 | 0.03 |
| sprat | pikeperch | 0.04298 | 1.71115 | 0.03 |
| herring | cod_west | 0.21843 | 8.26356 | 0.03 |
| herring | perch | 0.06578 | 9.11193 | 0.01 |
| herring | pikeperch | 0.04990 | 8.63939 | 0.01 |

**This says Python predates 17–100× harder than Java — the direct opposite of the mortality-based
measurement**, which had Java's adult predation on clupeids at 1.8–3.0× Python's with the same config.
Two measurements of the same quantity cannot disagree by two orders of magnitude *and in sign*; at least
one is wrong, so neither is reportable until the discrepancy is resolved.

**Most likely cause — time normalisation, the same class of error already found and fixed for
mortality.** The ratios cluster around 0.01–0.06, and 1/24 = 0.042 sits inside that band with
`ndtperyear = 24`. This engine's `diet_by_species` is accumulated over the 24-step recording window,
whereas Java's `predatorPressure` is plausibly per saving interval or otherwise normalised — exactly the
mismatch that made the mortality rates non-comparable until `0de82b9`. The spread (0.01–0.06, a factor
of 6) argues it is not a single clean constant, so there may be a second effect.

**Not concluded:** that Java predates less. The `predatorPressure` time convention must be established
on both sides first — by the same method that settled the mortality case, i.e. running at
`recordfrequency=1` versus `24` and checking whether an interval row equals the sum of its per-step rows.
Until then the mortality-based adult comparison remains the better-supported measurement, because its
convention *was* verified that way and its imposed-cause control reads 1.00.

### ⚠ RETRACTED (2026-07-30, later): the causation finding below is an instrumentation artifact

The section below concluded that Python applies the larval additional-mortality rate to juveniles
while Java applies it to eggs, and built a causal story on a measured 690× difference. **Both the
mechanism and the causal conclusion are wrong.**

`osmose/engine/processes/natural.py:143` — `larva_mortality` gates on `state.is_egg`
(`n_dead[eggs] = state.abundance[eggs] * mortality_fraction[eggs]`), exactly matching Java's
`if (school.isEgg())`. The application is correct.

The defect is in the stage collector added in `22cdf22`. Per-step ordering:

1. mortality runs (incl. the `larva_mortality` pre-pass), recording `n_dead` on egg schools;
2. reproduction increments ages and clears `is_egg` at `first_feeding_age_dt` (`simulate.py:692-693`);
3. `_collect_outputs` runs **after reproduction** (`simulate.py:1804`), and `_collect_by_life_stage`
   classifies by the *post-ageing* `is_egg`.

So egg deaths are binned by the stage the school occupies at *collection* time, not at time of death —
producing Eggs ≈ 0 / Juvenil ≈ 6.15 (the larval rate) in this engine. The 690× was my instrument.

**The residual ~3× is also gone — re-measured after `e7db9f6`, the engines agree to within 1%:**

| species | Eggs (J/P) | Juvenil | Adult | stage-summed |
|---|---|---|---|---|
| sprat | 6.1740 / 6.1696 = **1.001** | 1.000 | 1.000 | 1.001 |
| herring | 4.1937 / 4.1566 = **1.009** | 1.000 | 1.000 | 1.009 |

Both engines apply the **full** configured larval rate per step to eggs
(`147.71283750854758 / 24 = 6.1547`, measured ~6.17 on both). `natural.py`'s docstring claim that Java
applies the full rate in one step rather than dividing by `n_dt_per_year` is correct and now confirmed.

The ~3× was my analysis error: Java's "49.3/yr" egg rate was the **mean over all steps**, while the
configured 147.7/yr is per-step-when-present. Eggs carry the mortality only in the steps they exist
(~1/3 of them), so the all-step mean is ~1/3 of the per-step value — and both engines' means agree at
2.0524 vs 2.0523. The `147.71 / 72 = 2.0516` "lead" was a coincidence between a time-average and an
invented divisor; there is no egg-stage-duration division in either engine.

[#142](https://github.com/razinkele/osmopy/issues/142) closed as not-a-bug. **Larval additional
mortality is eliminated as a cause of the divergence.** The imposed mortality terms demonstrably agree
at every life stage, which strengthens the conclusion that the divergence lives in emergent predation.
The adult-stage predation asymmetry is unaffected by the collector bug, since mature schools do not
change stage within a step.

**Direction of causation is reopened.** It cannot be re-tested until the collector attributes deaths to
the stage held at time of death.

### RETRACTED — DIRECTION OF CAUSATION: larval additional mortality is applied to a different stage

Juvenile-stage test (cutoff keys stripped from Python so early biomass is reported; stage boundaries
verified identical). Per step, sprat juvenile `Additional` mortality:

| step | Java | Python |
|---|---|---|
| 7 | 0.0000 | **6.1547** |
| 8 | 0.0089 | 5.8430 |
| 10 | 0.0089 | 5.3398 |
| 12 | 0.0089 | 4.5093 |
| 14 | 0.0089 | 2.8604 |
| 15 | 0.0089 | **0.0089** |

The numbers identify themselves exactly:

* `147.71283750854758 / 24 = 6.1547` — Python charges the **larval** rate
  (`mortality.additional.larva.rate.sp2`) to **JUVENILES**, decaying as cohorts age out of the larval
  window, then dropping to the general rate at step 15.
* `0.21435847448826884 / 24 = 0.008932` — Java charges the **general** rate
  (`mortality.additional.rate.sp2`) to juveniles throughout, and the larval rate to **EGGS** (measured
  earlier at 2.05/step).

So during each cohort's larval window Python applies **~690× Java's juvenile additional mortality**
(6.15 vs 0.0089). This is an *imposed* rate: it needs no feedback, and it is present from the very
first step juveniles exist. Herring shows the identical pattern (Python 4.11 → 0.0936 = `98.5/24`).

**Why this settles the direction.** At the steps where `Additional` already differs ~690×, total
predator biomass has Java *below* Python (J/P 0.00 → 0.17 → 0.57), only crossing 1.0 around step 12.
Predation cannot be driving a divergence that is already maximal while the predator field is smaller
on the diverging engine. Juvenile predation ratios in that window are dominated by near-zero absolute
values (0.002 vs 0.0000035) and converge to ~0.85–1.13 by steps 13–15 — i.e. the predation difference
*ramps*, exactly as an emergent consequence would.

**The magnitude differs too, not only the stage.** Java's egg `Madd` measured 49.3/yr against a
configured larval rate of 147.7/yr — a factor of ~3 that also explains the unexplained /3 noticed when
trying to identify the boundary arithmetically. So the two engines differ in *both* which stage carries
the larval rate and its effective size.

**Conclusion.** The Baltic Java/Python divergence originates in the **larval additional-mortality
application — stage assignment plus a ~3× magnitude difference — not in the predation process.** The
predation difference reported above is downstream. This is a concrete parity defect in imposed
mortality, and it is the thing to fix before any further cross-engine comparison of this config.

**Which engine is right: Java.** From `fr.ird.osmose.process.mortality.AdditionalMortality` bytecode:

```
School.isEgg()  ifeq 28
  ->  larvaAdditionalMortality[speciesIndex].getRate(school)
else
  ->  general additional mortality
```

The larval rate applies **if and only if the school is pre-first-feeding**; everything else gets the
general rate. This engine's `is_egg` predicate is already identical (`simulate.py:692`), so the
boundary is not the problem — the *selection* is. **This engine is wrong.**

Filed as [#142](https://github.com/razinkele/osmopy/issues/142), including the unresolved ~3×
(`AnnualLarvaMortality` conversion) that must be pinned alongside the stage fix, and the likelihood
that `mortality.additional.larva.rate.*` needs refitting since it was calibrated against the wrong
stage.

### Eggs/Juvenil boundary: RESOLVED — the engines already agree

Read from the Java 4.4.1 bytecode (`javap -c`), not inferred:

```
MortalityOutput.getStage(School):  isEgg() ? Eggs : (isMature() ? Adult : Juvenil)
School.isEgg():                    getAgeDt() < Species.getFirstFeedingAgeDt()
```

Python's is identical — `simulate.py:692`: `new_is_egg = new_age < state.first_feeding_age_dt`. So
"Eggs" means pre-first-feeding on both engines and the three-way split has the same shape.
**Cross-engine stage comparison is valid**, and the caveat recorded earlier in this note (and in the
`_collect_by_life_stage` docstring) was wrong — retracted.

**That converts an assumed artifact into a real finding.** The Baltic `Additional` mortality lands at
≈0 on Eggs / ≈7 on Juvenil in Python against Java's ≈2 on Eggs. I had attributed that to differing
stage bins. The bins are identical, so it is a genuine difference in **where each engine applies the
larval additional-mortality rate** — Python is charging it to post-first-feeding juveniles, Java to
pre-first-feeding eggs. `mortality.additional.larva.rate.sp2` is 147.7/yr for sprat, so this is a
large term being applied to a different stage on each engine, and it is a strong candidate for the
emergent-predation divergence: shifting a mortality of that size between stages changes how many
recruits survive to be eaten.

This also unblocks the juvenile-stage causation test that the section below could not run.

### Direction of causation: ATTEMPTED, design does not work — recorded so it is not repeated

The plan was to compare the earliest steps, where both engines still hold near-identical seeded
populations, and see whether Java's clupeid predation is already elevated *before* predator biomass
diverges (cause) or only after (consequence). Two hard obstacles killed it, both measured:

1. **The adult stage does not exist during the early window.** Adult predation rates first appear
   around step 16–17 of 24 — nothing has matured before that. The clean separation window and the
   only boundary-safe stage do not overlap.
2. **Python reports zero biomass for the first half-year.** Verified directly: `cod_west` and `sprat`
   are 0.0 at steps 0–3. This is the output cutoff (finding 2) — `output.cutoff.age.spN` is 0.5 yr and
   at seeding every school is younger than that, so all of it is excluded from reported biomass. Java,
   with the cutoff forcibly disabled (finding 3), reports non-zero from step ~4. So the driver
   variable is blanked on one engine exactly when it is needed.

In the window that does have data the ratios invert (Java's adult predation ≈ 0 while Python's is
small but non-zero, J/P ≈ 0.00–0.02), the opposite of the 12-yr average. That is **not** evidence for
"consequence": it reflects Java having essentially no adults yet, not a milder predation process.
Reading it either way would be over-interpretation.

**What a working design needs:** strip `output.cutoff.age.spN` from the Python side so early biomass is
reported at all, and compare at the **juvenile** stage, which does exist early — which in turn requires
resolving the Eggs/Juvenil boundary question (unverified, see finding 6) so the stages mean the same
thing on both engines. Until then the direction of causation is undetermined.

### CORRECTED adult-stage comparison (after `0de82b9`) — supersedes the table below

The table further down used Python values computed under the wrong saving-interval convention. Java's
convention was then verified empirically: **its annual row is the SUM of its 24 per-step rows**, exact
on the deterministic causes (`Madd/Juvenil` 0.14287 vs 0.14287). Its `Madd/Eggs` ratio came out at
0.042 ≈ 1/24, independently confirming the caveat in Java's own header — egg additional mortality is
per model step, everything else is interval-summed. `053742b` had instead derived one rate from
window-aggregated counts, which has no fixed relationship to that sum; `0de82b9` forms rates per step
and sums them, and Python now reproduces its own annual row from 24 per-step rows at ratio 1.000.

| species | cause | Java | Python | J/P |
|---|---|---|---|---|
| sprat | **Predation** | 1.2861 | 0.4266 | **3.01×** |
| sprat | Starvation | 0.0879 | 0.0563 | 1.56× |
| sprat | Additional | 0.2014 | 0.2022 | **1.00×** |
| sprat | Fishing | 0.1050 | 0.0939 | 1.12× |
| cod_east | Predation | 0.1681 | 0.2058 | **0.82×** |
| cod_east | Starvation | 0.0083 | 0.0032 | 2.58× |
| cod_east | Additional | 0.8319 | 0.8345 | **1.00×** |
| cod_east | Fishing | 0.0023 | 0.0018 | 1.30× |
| herring | **Predation** | 0.6477 | 0.3552 | **1.82×** |
| herring | Starvation | 0.0629 | 0.0678 | 0.93× |
| herring | Additional | 2.1124 | 2.1224 | **1.00×** |
| herring | Fishing | 0.1999 | 0.2037 | 0.98× |

**The fix is validated by a criterion it was not tuned for.** `Additional` mortality is an externally
imposed rate, and it now agrees to within 0.5% on all three species (1.00× / 1.00× / 1.00×), where
before it read 1.25× / 1.38× / 1.90×. `Fishing`, also imposed, collapsed from 1.58 / 2.05 / 2.03 to
1.12 / 1.30 / 0.98. The spurious offset was arithmetic, and it is gone.

**Because the imposed causes now agree, the divergence is localised to emergent predation.** That is a
much sharper claim than the earlier table supported: the engines apply the same externally specified
mortality identically, and differ in the predation process itself.

**The clupeid signal survives, smaller and less uniform than reported.** Adult predation runs
**3.01× on sprat** and **1.82× on herring** heavier on Java, against **0.82× on cod_east** — Java
predates adult cod slightly *less*. The direction and the prey/predator contrast hold. But
"clupeids ~4–5×" was overstated: sprat is the outlier, herring is modest, and the two clupeids differ
from each other by a factor of ~1.7.

**Still unresolved** (unchanged): whether elevated clupeid predation drives the divergence or follows
from an already-inflated predator field. Starvation also runs 1.6–2.6× on Java for sprat and cod_east
but 0.93× for herring — emergent, food-dependent, and not yet interpretable.

### Superseded: adult-stage comparison (after `22cdf22`)

> Wrong Python convention — see the corrected table above. Retained for the audit trail.

With the life-stage split in place, adults are comparable: identically defined on both engines
(mature fish, same maturity conjunction), each using its own stage's survivors as denominator. Mean
rate over 12 yr:

| species | cause | Java | Python | J/P |
|---|---|---|---|---|
| sprat | **Predation** | 1.4627 | 0.3074 | **4.76×** |
| sprat | Starvation | 0.0830 | 0.0462 | 1.80× |
| sprat | Additional | 0.2014 | 0.1610 | 1.25× |
| sprat | Fishing | 0.1042 | 0.0661 | 1.58× |
| cod_east | Predation | 0.1791 | 0.2154 | **0.83×** |
| cod_east | Starvation | 0.0074 | 0.0024 | 3.12× |
| cod_east | Additional | 0.8319 | 0.6013 | 1.38× |
| cod_east | Fishing | 0.0024 | 0.0012 | 2.05× |
| herring | **Predation** | 0.6876 | 0.1694 | **4.06×** |
| herring | Starvation | 0.0605 | 0.0328 | 1.84× |
| herring | Additional | 2.0809 | 1.0976 | 1.90× |
| herring | Fishing | 0.1941 | 0.0957 | 2.03× |

**The finding: adult predation on the clupeids is ~4–5× heavier on Java, and cod's is not.** Sprat
4.76× and herring 4.06× stand against cod_east at 0.83× — Java actually predates adult cod slightly
*less*. That matches the biomass pattern exactly: clupeids fail on Java while everything else booms.
The mechanism is elevated predation pressure on clupeids specifically, not a uniform difference.

**Caveat that keeps this from being conclusive.** There is a broad ~1.3–2× offset across *all* causes
including Fishing, which is an externally imposed rate that should not differ between engines. That
offset may be an artifact of the rate convention rather than dynamics: over a multi-step recording
window this code sums deaths and averages abundance, and Java's own convention for its saving
interval has not been verified against that. So the trustworthy quantity here is the clupeid
predation signal *relative to that background offset* — roughly 2–3× above it — not the absolute
ratios. Establishing Java's saving-interval convention would firm this up, and is the obvious next
step.

**What it does not resolve:** whether elevated clupeid predation is the cause of the divergence or a
consequence of an already-inflated predator field. Both directions fit these numbers. Distinguishing
them needs the first few years examined step-by-step, before the feedback establishes itself.

**Second blocker found:** `osmose/results.py::_read_mortality_rate_csv`, whose docstring says it
reads "a real `mortalityRate-{sp}` CSV", raises
`ParserError: Header rows must have an equal number of columns` on genuine Java 4.4.1 output.
Java's own header is inconsistent — row 1 has 25 fields (`Time` + 8 named causes × 3 stages), row 2
has 28 (blank + **9** stage-triples), and data rows have 29. Java emits 27 data columns but names
only 24, leaving one group of three unnamed. The reader has evidently never been run against Java
output. Needs its own fix; the table above was produced with a positional parser.

## What this means

**The Baltic cross-engine biomass comparison, as built, cannot be valid.** It differs in the
reported quantity (findings 2-4) *and* in the model being simulated (finding 5). A survivor-set
"DIFFER" between engines is therefore uninformative about engine correctness.

Making the *reported quantity* comparable is necessary for honesty but, as measured above, buys
almost nothing numerically:

1. ~~**Strip `output.cutoff.age.spN` from the Python side**~~ — **tested, does not work.** Moves
   Python by 1.0–1.4×; the 5.5–73× gap survives.
2. **Reconstruct adult-only biomass from Java's age-structured output** (`output.biomass.byage`) —
   still the cleanest way to compare like with like, but expect it to change little.
3. **Fix the Java `OutputRegion` bound** so the cutoff array covers background species. Correct, and
   worth reporting upstream, but not the cause of anything here.

None of these touch finding 5: the RV and salinity gates remain invisible to Java, so cod stays
biased upward there regardless. Any honest cross-check must disable those gates on the Python side or
exclude cod from the comparison.

**The real divergence is unexplained and is dynamical.** It is not the harness bug (fixed), not the
reported quantity (tested), and not wholly the missing gates (the largest gaps are in ungated
species). The clupeid-collapse / everything-else-booms pattern is the lead.

**Sprat is the one candidate genuine contradiction**: Python ~1.06 Mt, Java 0. `sprat` has no
underscore so it was always read correctly, and it is ungated, so neither finding 1 nor 5 explains
it. Its collapse is plausibly Java's inflated cod field (sprat is cod prey) — untested.

## Upstream signal: early clupeid ABUNDANCE, not growth (2026-07-30)

Decomposing `biomass = abundance × mean weight` over year 1 (`recordfrequency=1`, Python cutoff
stripped):

**sprat**

| step | Java abundance | Python abundance | abd J/P | Java wt (g) | Python wt (g) | wt J/P |
|---|---|---|---|---|---|---|
| 7 | 2.49e10 | 4.49e12 | 0.006 | 0.001 | 0.001 | 1.00 |
| 8 | 9.12e10 | 8.43e12 | 0.011 | 0.001 | 0.001 | 0.96 |
| 10 | 3.79e11 | 1.13e13 | 0.034 | 0.008 | 0.001 | 7.32 |
| 12 | 6.57e11 | 4.58e12 | 0.143 | 0.052 | 0.004 | 14.78 |
| 14 | 7.27e11 | 1.02e11 | **7.14** | 0.237 | 0.450 | 0.53 |

**herring** follows the same shape: abundance J/P 0.013 → 0.588 → 2.15, weight J/P 0.98 → 6.30 → 0.62.

**It is an abundance deficit, not a growth deficit.** In the early steps both engines carry animals of
essentially identical mean weight (~0.001 g — eggs/larvae), so they are counting the same kind of
individual; Java simply has **1–7% as many**. Growth is not the discriminator: mean weights match at
step 7–8 (J/P 0.96–1.00) and only diverge later as the cohorts age differently.

**The crossover is the striking part.** Java's abundance goes from 0.006× Python's at step 7 to
**7.1×** by step 14, while its mean weight falls from 14.8× to 0.53×. The two engines populate the
seeded stock on very different schedules — Python front-loads enormous numbers of near-weightless
individuals that then thin out; Java starts sparse and accumulates.

**So "Java's recruitment is lower" is the wrong framing.** Over year 1 Java ends with *more* clupeid
individuals, not fewer. What differs is the early-window composition — how each engine converts the
seeding target into schools, and how fast. Total seeded biomass is the same target
(`population.seeding.biomass.sp2 = 600000` t); neither engine is near it during this window.

**Next:** compare the seeding/initialisation path directly — number of schools created per species per
step and their initial size distribution — rather than inferring from standing stock. That is a
different subsystem from anything examined so far in this note, and none of the predation-side
conclusions carry over to it.
