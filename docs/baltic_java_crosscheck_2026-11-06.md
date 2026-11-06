# Baltic Java 4.4.1 cross-check, re-run after the Cormorant fix — 2026-11-06

**Trigger:** #208 found that every Baltic Java arm had run Cormorant at **1×** while the Python engine
uses **2×** (`species.biomass.multiplier.sp16 = 2.0` never reached Java's `species.multiplier.sp16`).
The question here is whether earlier Java cross-checks involving the background predators have to be
re-read.

**Answer: no conclusion changes.** The fix moves Java's equilibria by at most about 5%, and only perch
moves clearly outside run-to-run noise. The Python↔Java divergence recorded on 2026-07-30
(`docs/baltic_java_crossengine_fidelity_2026-07-30.md`) is still there and essentially unchanged:
Java drives sprat extinct and runs most other species 5–100× above Python. The Cormorant defect was
real but does not explain that divergence.

One new open finding is in §3: Java's *reported* background-predator biomass collapses within each
step even though nothing in the staged config kills or fishes them.

Jar: `osmose-java/osmose-4.4.1-jar-with-dependencies.jar`. Config: the production `data/baltic` at
`125a0143` (v1.8.0), `--params current`, 50 yr. The Java arm pins off the two Python-only features it
cannot run (`ltl.oxygen.benthos.enabled`, `movement.salinity.gate.enabled`).

## 1. The standard cross-check (`baltic_stability_certify.py --java`)

Final-decade mean biomass, tonnes. Python is the 5-seed range. Java is a single run (4.4.1 has no
numeric seed).

| species | Python (5 seeds) | Java now | Java 2026-07-30 | Java / Python |
|---|---|---|---|---|
| cod_west | 12,284–14,330 | 362,263 | 353,850 | ~27× |
| cod_east | 64,245–66,315 | 895,316 | 948,866 | ~14× |
| herring | 2,473,162–2,608,156 | 1,838,303 | 1,916,430 | ~0.7× |
| sprat | 905,315–917,704 | **0** | **0** | extinct |
| flounder | 32,577–33,598 | 3,181,455 | 3,011,791 | ~96× |
| perch | 41,869–42,511 | 2,269,960 | 2,279,394 | ~54× |
| pikeperch | 1,352,059–1,502,868 | 8,532,195 | 8,139,264 | ~6× |
| smelt | 683,186–686,742 | 4,179,292 | 4,019,851 | ~6× |
| stickleback | 79,215–87,868 | 4,553,768 | 5,177,973 | ~55× |

- **Python: 5/5 assessed stocks persistent and in envelope** (7/9 all species; pikeperch and smelt
  are above their indicative envelopes, as before).
- **Java: 1/5 assessed** (herring only). The survivor sets differ by sprat.

Between 2026-07-30 and now, the Java arm received the `BG_ACCESS` cod-row fix (seal and cormorant
predation on both cod stocks), the Cormorant multiplier fix and every config change since July
(cod_east M, sprat F, hypoxia coupling pinned off on Java, …). Java's numbers moved by 2–12%. The
divergence pattern from July is unchanged: clupeids fail on Java and everything else booms.

The script's own summary line reads "Java cross-check: 1/9 persistent". That count is the *assessed*
pass count (1/5), not persistence: all species except sprat persist on Java.

## 2. The Cormorant A/B on Java

The same staged config run with `-Pspecies.multiplier.sp16=1` (the pre-#208 behaviour) against the
carried value of 2.0. The staged file was checked to contain `species.multiplier.sp16 ; 2.0`, and
Java's startup log confirms "Biomass for background group 16 will be multiplied by 2.0". There are two
runs per arm; the §1 Java run serves as a third 2× replicate.

Final-decade mean biomass, kt:

| species | 1× runs | 2× runs | 2×/1× of means | max/min within an arm |
|---|---|---|---|---|
| cod_west | 394, 354 | 353, 385, 362 | 0.980 | 1.110 |
| cod_east | 915, 893 | 898, 904, 895 | 0.994 | 1.024 |
| herring | 1,834, 1,775 | 1,888, 1,875, 1,838 | 1.035 | 1.033 |
| sprat | 0, 0 | 0, 0, 0 | — | — |
| flounder | 3,122, 3,409 | 3,111, 3,121, 3,181 | 0.961 | 1.092 |
| perch | 2,291, 2,310 | 2,174, 2,109, 2,270 | **0.949** | 1.076 |
| pikeperch | 8,810, 8,191 | 8,490, 8,359, 8,532 | 0.995 | 1.076 |
| smelt | 4,090, 4,301 | 4,091, 4,089, 4,179 | 0.982 | 1.052 |
| stickleback | 4,832, 4,688 | 4,828, 4,817, 4,554 | 0.994 | 1.060 |

Read against the within-arm spread (2–11%):

- **Perch −5%** is the only shift where every 2× run sits below every 1× run. It points the expected
  way: perch is a principal cormorant prey.
- **Herring +3.5%** also separates the arms, though only just (lowest 2× 1,838 kt against highest 1×
  1,834 kt). Treat it as suggestive.
- Everything else is inside the noise.

With 2–3 runs per arm these are effect sizes, not significance tests.

**Consequence for earlier Java cross-checks:** their seal/cormorant-sensitive numbers were off by
about 5% at most, against a cross-engine gap of 5–100×. None of the July conclusions depends on the
multiplier.

## 3. Open finding: Java's reported background biomass collapses within each step

Java's annual biomass output shows GreySeal and Cormorant at about 1–3% of their forcing after the
first year. The forcing is 4,500 t and 500 t ×2 = 1,000 t. The Python engine reports a constant
4,500 / 1,000 t.

A 2-yr Java run with per-step output (`output.recordfrequency.ndt=1`) shows:

- **The forcing is applied.** `biomassPredPreyIni` (start of step, before mortality) is exactly
  4,500 / 1,000 t at every step.
- **They do eat.** Their `dietMatrix` columns sum to 100% at every step from t = 0.54 yr on. They
  are all-NaN before that, which may be when Java's diet output starts recording rather than when
  they start eating.
- **But the end-of-step biomass in `biomass_Simu0.csv` falls.** It holds 4,500 / 1,000 t for three
  steps, then drops in whole-school chunks (multiples of 292.5 t for the seal). By year 2 it is zero
  in most steps.

Ruled out from the staged config:

- **Predation:** the GreySeal and Cormorant *prey* rows of the accessibility matrix are all zero.
- **Fishing:** both species have all-zero catchability and discard rows, and Java's yield output for
  them is zero.
- **Starvation and additional mortality:** v4.4.1's `MortalityProcess` skips background schools for
  both (`if (seq >= ns) break`).

One possible mechanism is that the reported end-of-step figure is computed per cell. v4.4.1 creates
background schools per cell on demand (`getBackgroundSchool(cell)`), so cells the mortality loop
never visits may drop out of the total. **That is not verified.** What is measured is that Java's
end-of-step background biomass output does not track the forcing, while start-of-step biomass and
diet do.

**Why it matters:** any reading of Java's background biomass output, or anything derived from it,
will be wrong. Whether the predation the background predators actually exert in Java is also reduced
is not settled. Java's `predatorPressure` omits background predator columns, so this run cannot
measure their consumption in tonnes.

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --java \
    --out docs/baltic_java_crosscheck_2026-11-06.md
```

The A/B and per-step probes were one-off scratch scripts. Each stages through `_prepare_java_cfg` +
`stage_config_for_java` (the same path `certify_java` uses). The A/B adds `-Pspecies.multiplier.sp16=1`
for the 1× arm. The probe adds `-Psimulation.time.nyear=2 -Poutput.recordfrequency.ndt=1`.

## Appendix — generated certification output

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** config default

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✓ | 1.21e+04 | [12284.302563377025, 14329.593105259864] |
| cod_east | ✓ | ✓ | 4.52e+04 | [64245.19524894278, 66314.57481584196] |
| herring | ✓ | ✓ | 2.31e+06 | [2473162.4985574908, 2608156.146229925] |
| sprat | ✓ | ✓ | 8.44e+05 | [905314.8246332215, 917704.4394325413] |
| flounder | ✓ | ✓ | 3.07e+04 | [32576.682403842908, 33598.32803701806] |
| perch | ✓ | ✓ | 3.93e+04 | [41869.41018758553, 42510.74007833198] |
| pikeperch | ✓ | ✗ | 1.21e+06 | [1352058.6917450833, 1502867.9733709753] |
| smelt | ✓ | ✗ | 6.46e+05 | [683186.1996137024, 686741.6253423884] |
| stickleback | ✓ | ✓ | 6.20e+04 | [79215.0778124692, 87868.20046005341] |

**Python verdict: 5/5 ASSESSED species persistent & in-envelope.**

**Java cross-check (as printed by the script): "1/9 persistent"**, which is the assessed pass count;
see §1. Survivor sets DIFFER with Python by sprat only.
