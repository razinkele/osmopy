# Herring compensation breakdown: result, 2026-11-12

**Question:** under B2's +2.9 °C knob (egg production × f = 0.228), herring's biomass falls by
26.5% in the v1.9.0 state (A0) but by 61.2% in the v1.8.8 state (A4). Through which channel does
A0 compensate with a smaller fall in density, and which cause's hazard moves? The criterion was
pre-stated in `herring_compensation_prestated_2026-11-12.md` (`6d40b5cc`). The raw output is
`herring_compensation_2026-11-12.json`. The run took 20 runs and 708 s.

**Answer.** In both states, almost all of the compensation comes from one place: **spawning
biomass per age-1+ fish** (mature fraction × weight) rises about 4.6×. **Survival per recruit does
not change.** What differs is how much density that rise costs.

- **The size of the rise is forced.** f is the same in both states, and the egg-stage channels
  barely move, so spawning biomass per fish has to rise by about 1/f in each.
- **A0 gets it with half the fish; A4 needs roughly a fifth.** In A0, age-1+ numbers fall to
  0.49×; in A4 they fall to 0.22×. The post-hoc secant (spawning biomass per fish against age-1+
  density) is **2.18 in A0 against 1.00 in A4**.
- **The early stage pushes the other way.** Egg-to-age-1 survival falls under warming, by more
  in A4. That coincides with a rise in age-0 predation hazard as young herring become scarcer.
  The run shows the association, not the cause.

The note stops at channel and cause, as pre-stated. *Why* spawning biomass per fish is twice as
density-elastic in A0 is not measured.

> **A separate finding outranks the warming question:** on v1.9.0 herring's **spawning stock is
> 22 kt, 1.5% of its 1.53 Mt biomass**. ICES puts central Baltic herring SSB alone at about
> 400–500 kt. See [the last section](#a-model-state-finding-herring-barely-matures-on-v190).

## Gates

| gate | result |
|---|---|
| G1: hooks read-only | **PASS.** All 20 herring final-decade biomasses are bit-identical to the matching attribution runs. |
| G2: knob exactness | **PASS.** In every year of the window (and the lag year) of every run, annual after-knob / after-curve eggs equal 0.22786543992898978 on warm runs and 1.0 on baseline runs. The maximum relative deviation is 4.4e-16, about 2 ULP. No seeded herring step falls in the window. Review caught that the first G2 instrument, a per-step sample capped at four values, filled during spin-up and never saw the window; this window-level check replaced it (`knob_window_max_rel_dev`). The annual check is equivalent to a per-step one here: `thermal_gate_factor` indexes its series by `year = step // n_dt_per_year`, so it is constant within a model year, and it is the only gate active for herring. Every non-seeded step in a year applies the same factor, so per-step deviations cannot offset inside the annual ratio. |
| G3: year-0 accounting | **PASS.** (recruits + egg and age-0 deaths) / eggs lies between 0.9956 and 1.0006 across all 20 runs. |
| Stationarity | The same-window and lagged decompositions agree to ≤ 0.006 in every log ratio, so the final decade is at equilibrium for this purpose. |

## The pre-stated η was mis-specified

The pre-stated comparison metric was η_k = ln r_k / ln r_SSB. In A4, SSB does not move under
warming (ln r_SSB = −0.005), so η is defined but unstable: dividing by a near-zero denominator
gives values in the hundreds that swing with the window. The normaliser was the wrong density. The channel that carries the compensation
(SPR) responds to **recruit or age-1+ density**, not to SSB. The pre-stated channel definitions
and the log-ratio table below stand. Both were pre-stated, and the table is identity-exact. Only
the η step fails. The recruit-normalised secant reported here is **post-hoc**.

## Channels (warm/base, ln ratios of 5-seed means, final decade)

| | biomass | SSB | lin | SR | knob | early | SPR | recruits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | −0.307 | **+0.838** | −0.006 | −0.028 | −1.479 | −0.027 | **+1.539** | −0.702 |
| A4 | −0.948 | −0.005 | +0.125 | −0.034 | −1.479 | −0.124 | **+1.512** | −1.517 |

- **knob** is exactly ln f in both states.
- **SR**, the Shepherd curve, is negligible in both, at ≤ 3.4%. The pre-stated Shepherd prediction
  is therefore moot: it is neither confirmed nor refuted.
- **lin** is bookkeeping, not fecundity. It is annual eggs over *annual-mean* SSB. A4's +0.125
  appears because spawning-step SSB shifted relative to the annual mean under warming. A
  season-weighted SSB would move that 0.125 into SPR without changing any conclusion.
- **SPR** carries the compensation in both states: about +1.5, a 4.6× rise.

### SPR split (measured from the stored runs, `spr_split` in the JSON)

SPR = (age-1+ standing stock per recruit) × (SSB per age-1+ fish).

| | age-1+ stock per recruit | SSB per age-1+ fish | age-1+ stock | immature biomass per recruit | mature share of biomass |
|---|---:|---:|---:|---:|---|
| A0 ln r | −0.004 | **+1.543** | −0.707 | +0.361 | 1.46% → 4.59% |
| A4 ln r | −0.008 | **+1.521** | −1.525 | +0.124 | 18.6% → 47.9% |

- **Survival and residence per recruit are flat in both states.** The SPR rise is entirely
  maturation and weight per fish. This is measured, not inferred by elimination.
- **Post-hoc secant**, ln r(SSB per fish) / −ln r(age-1+ stock): **2.18 (A0) against 1.00 (A4)**.
  The quantity rises by the same amount in both; A0 needs half the density reduction to get it.
  This is a two-point ratio between different operating points, not a density-dependence slope.
- **The chain to the B2 metric** (≥0.5-yr biomass, 98.5% immature in A0) runs as follows:
  - immature biomass ≈ recruits × immature biomass per recruit;
  - in A0 that is −0.702 + 0.361 = −0.341, against −1.517 + 0.124 = −1.393 in A4;
  - recruits fall less in A0 because SSB *rises* there (+0.838), while it stays flat in A4.

## Causes: two tables, two questions

**Which cause responded:** per-capita hazard, deaths over standing stock at the start of each
step, ×24 to give per year.

| class, cause | A0 base → warm | A4 base → warm |
|---|---|---|
| age-0 predation | 0.389 → **0.459** (+18%) | 0.607 → **0.838** (+38%) |
| age-0 starvation | 0.091 → 0.086 | 0.102 → 0.092 |
| age-0 additional | 2.124 → 2.121 | 2.111 → 2.098 |
| age-1+ predation | 0.362 → 0.387 (+7%) | 0.468 → 0.497 (+6%) |
| age-1+ starvation | 0.062 → 0.053 | 0.076 → 0.050 |
| age-1+ fishing | 0.355 → 0.355 | 0.354 → 0.354 |
| egg additional | 16.80 → 16.80 | 23.61 → 23.61 |

- **Age-0 predation hazard rises when young herring are scarcer:** +18% in A0 and +38% in A4.
  It moves in the same direction as the `early` channel (−0.03 in A0, −0.12 in A4). The run does
  not separate predation from exposure or the other causes, so this is an association, not a
  demonstrated cause of A4's larger early decline.
- **Post-recruit hazards barely move**, consistent with the flat stock-per-recruit factor above.
- **The egg-stage additional hazard differs by construction** (larval rate 28.89 vs 98.56 /yr).
  That is the state difference, not a response. Survival to age 1 is 2.0% in A0 against 0.097%
  in A4.

**How much each cause carries:** year-0 cohort fractions, deaths over eggs.
- In A0, additional takes 0.926 → 0.921 and predation 0.042 → 0.048.
- In A4, additional takes 0.995 → 0.994 and predation 0.0033 → 0.0043.

Under competing risks these fractions are not used to name the responding cause.

## Not measured: why spawning biomass per fish is twice as density-elastic in A0

The obvious reading is that in A0 the spawning stock is a thin tail: 1.5% of biomass sits above
the 18 cm maturity size. A small growth gain from lower density would then shift a large
*relative* fraction across the threshold. That is a hypothesis. It would be tested by an
intervention: perturb `species.maturity.size.sp1`, or record size at age per cohort.

## A model-state finding: herring barely matures on v1.9.0

The reproduction hook reads the engine's own `ssb` array, the one reproduction spawns from. The
engine's separate SSB output agrees: 21.5 kt for seed 42. On the certified v1.9.0 config:

- **Herring SSB is 22 kt out of 1.53 Mt** of ≥0.5-yr biomass, 1.46% mature.
- In the v1.8.8 state it was 503 kt out of 2.70 Mt (19%).
- For comparison, ICES central Baltic herring SSB alone is about 400–500 kt, the targets file's
  own note. The SAG snapshot envelope that `validate_outputs_vs_ices.py` uses is 790–868 kt.

The certification envelope for herring (0.8–3.0 Mt) is a *biomass* target, and certification
compares ≥0.5-yr biomass. So v1.9.0 certified herring at 1.5 Mt while its spawning stock is
about 20× below the central-Baltic SSB, and 36–39× below the SAG envelope. The outputs validator compared that total biomass with an
ICES **SSB** envelope. The certification also compares both cod stocks' ≥0.5-yr biomass with
envelopes whose `reference_point_type` is `ssb`. This finding is reported separately (issue
filed); it does not change the channel result above.

**Herring is the only such species.** One production run at v1.9.0 (seed 42, 50 yr, with
`output.ssb.enabled` added) gives the final-decade SSB against ≥0.5-yr biomass:

| species | biomass ≥0.5 yr | SSB | SSB share | maturity size |
|---|---:|---:|---:|---:|
| cod_west | 16,389 t | 14,099 t | 86.0% | 38 cm |
| cod_east | 77,149 t | 76,767 t | 99.5% | 22 cm |
| **herring** | **1,482,700 t** | **21,533 t** | **1.45%** | **18 cm** |
| sprat | 889,324 t | 913,716 t | 102.7% | 9 cm |
| flounder | 49,339 t | 43,152 t | 87.5% | 22 cm |
| perch | 62,639 t | 60,811 t | 97.1% | 18 cm |
| pikeperch | 618,112 t | 507,444 t | 82.1% | 40 cm |
| smelt | 654,144 t | 565,897 t | 86.5% | 10 cm |
| stickleback | 102,046 t | 103,405 t | 101.3% | 4.5 cm |

Shares above 100% occur because SSB counts mature fish younger than the 0.5-yr output cutoff.

- **Herring's size structure shows why.** Of its final-decade biomass, 49.8% sits in the 5–10 cm
  bin and 44.2% in 10–15 cm. Only 4.1% is in 15–20 cm, and 0.27% at 20 cm or more, against a
  maturity size of 18 cm.
- **The cod concern is small.** Because cod's biomass is 86–99.5% spawning stock, comparing it
  with an SSB envelope misstates it by at most about 14%.
- **The open question is herring's.** The state difference above shows how it arose. In the
  v1.8.8 state herring was 19% mature; v1.9.0 combined the corrected, poorer food field with a
  re-fit of larval mortality to the biomass target, and that combination is where the mature
  share fell. Two readings remain. Either herring growth on the corrected food field is too slow
  to reach maturity, or 18 cm is too high a maturity size for the model's Baltic herring. This
  run does not separate them.
