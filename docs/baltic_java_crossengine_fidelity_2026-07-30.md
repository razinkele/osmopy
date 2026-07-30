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
