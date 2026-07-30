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

So Python reports adult biomass and Java reports everything including young-of-year. For
high-fecundity species that is a large multiplier, and it matches the pattern — the biggest
divergences are the small, fast, high-fecundity species.

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

## What this means

**The Baltic cross-engine biomass comparison, as built, cannot be valid.** It differs in the
reported quantity (findings 2-4) *and* in the model being simulated (finding 5). A survivor-set
"DIFFER" between engines is therefore uninformative about engine correctness.

To make it valid, one of:

1. **Strip `output.cutoff.age.spN` from the Python side of the cross-check** so both report uncut
   biomass. Cheapest, and testable immediately.
2. **Reconstruct adult-only biomass from Java's age-structured output** (`output.biomass.byage`),
   comparing like with like without touching either engine.
3. **Fix the Java `OutputRegion` bound** so the cutoff array covers background species. Correct but
   upstream.

Option 1 does not fix finding 5 — the RV and salinity gates remain invisible to Java, so cod will
stay biased upward there regardless. Any honest cross-check has to either disable those gates on the
Python side or exclude cod from the comparison.

**Sprat is the one candidate genuine contradiction**: Python ~1.06 Mt, Java 0. `sprat` has no
underscore so it was always read correctly, and it is ungated, so neither finding 1 nor 5 explains
it. Its collapse is plausibly Java's inflated cod field (sprat is cod prey) — untested.
