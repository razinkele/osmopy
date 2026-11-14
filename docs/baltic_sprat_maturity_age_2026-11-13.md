# Baltic sprat maturity age 1 yr, 2026-11-13

> **Not for auto-merge.** This changes the production Baltic model at the maintainer's request ("set
> a maturity age of 1 year for sprat, but be aware that only a part of the cohort - around 15% to
> 40%"). It waits for the maintainer's decision to merge.

**Why:** Baltic set no `species.maturity.age`, so sprat counted as mature from 9 cm at any age. Model
sprat pass 9 cm before 0.5 yr, so fish younger than a year spawned: 3.4% of sprat SSB came from fish
under 0.5 yr (#253), and all of age 1 was mature at spawning.

**The change (`data/baltic/baltic_param-species.csv`):**
- **`species.maturity.age.sp2 = 1.0`.**
- **`species.maturity.size.sp2` (9 cm) removed.** Sprat now matures by age only. The maintainer
  chose this form after a Java check showed it is the only one both engines read the same way (see
  "Java" below). In the Python engine it is bit-identical to keeping the 9 cm size as well.
- **Nothing else changes** in sprat (larval rate, stock-recruitment curve) or in any other species,
  apart from the SP1b cod rates re-solved below.

Criteria were committed before any candidate run:
`docs/diagnostics/sprat_maturity_age_prestated_2026-11-13.md` (`95e1b34e`, path fix `31658ded`).

## Result

| criterion | result | verdict |
|---|---|---|
| C1: age-1 mature share at spawning, in [0.15, 0.40] | **0.409** (seeds 0.407–0.411) | **just above the band** |
| C2: certification, 5 seeds × 50 yr | **5/5 assessed**, no range crosses an envelope | pass |
| C3: SP1b drift guard | tripped (cod moved ~1%); re-solved, 28/28 pass | pass |

- **C1 is 0.9 points over the stated ceiling,** not inside it. Following the pre-statement, it is
  reported, not tuned.
- **The age alone decides maturity.** Age-1 sprat were 10–13 cm at spawning in the master probe,
  above the 9 cm size this change removed. That is why the age-only form gives the same result.

### How the partial maturity arises

OSMOSE maturity is knife-edge, so with age only, each sprat matures on its first birthday. That
birthday falls inside its second spawning season.
Sprat spawn over steps 6–13 (April to mid-July, peak late May), so the share of last year's cohort
that has reached age 1 ramps up through the season:

| step | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| season weight | 0.03 | 0.09 | 0.16 | 0.22 | 0.22 | 0.16 | 0.09 | 0.03 |
| age-1 share mature | 0.00 | 0.03 | 0.11 | 0.27 | 0.49 | 0.72 | 0.88 | 0.97 |

Candidate config, mean of seeds 42, 123 and 7, final decade.

- **The 0.409 is the season-weighted mean,** the egg-weighted equivalent of a partial maturity at age
  1. It is not "41% of individuals are mature all season": every age-1 fish spawns from its birthday
  on. The plain mean over the eight steps is 0.433.
- **Measured on the candidate's own equilibrium.** A first probe of the rule on master's dynamics
  gave 0.382 (seed 1000, years 10–14). The candidate's 0.409 is consistent with the rule shifting
  egg output, and so next year's birthdays, later in the season. But the two numbers also differ in
  seed and horizon, so that shift is not measured here.
- **Age 2 and older:** 100% mature at every step, as expected.

### The lever, if the band is binding

The share depends on where the age threshold sits within the season. One step later, a maturity age
of 25 steps (1.04 yr), would shift the ramp by one step. Applying that shift to the measured ramp
gives about **0.26**, mid-band. That is an estimate: it ignores any feedback through birth dates, so
it would need its own run. A size threshold is not available as a lever: adding a sprat maturity size
would make Java ignore the age (see "Java").

## What it does to sprat (3 seeds, final decade, master → candidate)

| | master | candidate | change |
|---|---:|---:|---:|
| biomass (≥0.5 yr) | 0.913 Mt | 1.002 Mt | +9.8% |
| SSB | 0.939 Mt | 0.787 Mt | −16% |
| SSB / biomass | 1.03 | 0.785 | |
| egg output (after the SR curve) | | | **+13.5%** |

- **Fewer spawners, more eggs.** Sprat's Shepherd curve (`ssbhalf` 213 kt, shape 1.72) is domed, and
  at an SSB of about four times `ssbhalf` it sits on the descending side. Cutting SSB by 16% therefore
  raises egg output, and biomass rises 10%. This is the calibrated curve's behaviour, not something
  this change tuned.
- **Sprat's SSB now sits below its ICES SSB envelope.** The validator (#253) gives 0.79× the
  2018–2022 envelope (0.89–1.12 Mt), against 0.95–0.97× on master. Removing the spawners under a
  year old is what moves it out. This is reported, not re-fitted.
- **Other species move by at most 1.3%** (3-seed means): cod_west +1.0%, cod_east −1.2%,
  pikeperch +1.3%, every other species within ±1.1%.

## Certification: 5/5 assessed (50 yr × 5 seeds)

| species | master (#250, v1.10.0) | **this change** | envelope |
|---|---|---|---|
| cod_west | 18.7–19.3 kt | 18.9–19.4 kt | 4–25 kt |
| cod_east | 73.7–75.6 kt | 72.5–74.2 kt | 60–85 kt |
| herring | 1.51–1.59 Mt | 1.52–1.55 Mt | 0.8–3.0 Mt |
| sprat | 0.90–0.92 Mt | **1.00–1.01 Mt** | 0.8–2.5 Mt |
| flounder | 47.6–49.1 kt | 46.9–48.3 kt | 20–100 kt |
| *perch* | *33.9–36.4 kt* | *34.1–35.8 kt* | *8–50 kt* |
| *pikeperch* | *515–529 kt* | *515–532 kt* | *4–25 kt* |
| *smelt* | *655–663 kt* | *651–666 kt* | *20–120 kt* |
| *stickleback* | *82–98 kt* | *86–95 kt* | *50–500 kt* |

- **The cliff rule passes:** no assessed stock's 5-seed range crosses its envelope.
- **The indicative tier is unchanged at 2/4** (perch and stickleback).

## SP1b

The drift guard tripped as expected: cod_east's SP1-off baseline moved −1.6% (121,760 → 119,836 t)
and cod_west's +0.2% (11,231 → 11,253 t). `scripts/recalibrate_sp1b.py` re-solved both rates in one
sweep and 23 evaluations (record: `docs/diagnostics/sp1b_solve.json`):

| stock | rate | was | baseline | joint error |
|---|---:|---:|---:|---:|
| cod_east | 9.0292 | 8.9921 | 119,836 t | −0.08% |
| cod_west | 8.9402 | 8.9296 | 11,253 t | −0.75% |

Both are inside the 1-D tolerance (2%) and JOINT_TOL (5%); d0 is unchanged. The drift guard runs only
locally (it is skipped under `CI=true`), and passes on the final config: 28/28 in
`tests/test_sp1b_recalibration.py`.

## Age-only form: bit-identical to size + age

The runs above used `species.maturity.age.sp2 = 1.0` with the 9 cm size still set. The final config
drops the size. In the Python engine the two forms differ only for a sprat that is at least 1 yr old
and shorter than 9 cm, so they were checked to be the same, not assumed:
- **The SP1b drift guard** (which demands the 15-yr baselines and SP1-on means to 1e-6) passes on the
  age-only config with the rates solved on the size + age config.
- **The committed script on the age-only config** reproduces the size + age runs exactly for seeds 42,
  123 and 7: 20 values per seed (age-1 share, egg output, biomass and SSB of all nine species), no
  difference.
- **Certification on the age-only config** gives a table identical to full float precision.

The amendment to the pre-statement is therefore a re-expression, not a new treatment. Had any of
these differed, every run would have been repeated on the age-only config.

## Side effects to know

- **Bioen still needs a maturity length for sprat.** Bioen maturity is a length reaction norm (`length
  >= m0 + m1 * age`; Java rejects size/age maturity under bioen), and the tools that build a bioen
  config copied `m0` from the maturity size, which sprat no longer has. The production C3 bioen arm
  is unaffected: its JSON sets `species.maturity.m0.sp2 = 9.0` explicitly. For the tools,
  `osmose.calibration.bioen_offline.bioen_m0_cm` returns the maturity size, or for an age-based
  species the same 9.0 (`AGE_BASED_M0_CM`), and raises for any other species without a size. It is
  used by `scripts/fit_baltic_bioen_params.py` (which would otherwise have written `m0 = 0`, so
  every sprat matured at any length) and the test overlay `tests/_bioen_overlay.py` (which raised).
- **Sprat's "Juvenil" stage now runs to age 1.** Stage outputs use the same maturity rule, so sprat's
  stage mortality rates (and the SMS M2 comparison's per-stage rates for sprat) are not comparable
  across this change.
- **The v1.10.0 results this made stale were re-run on v1.10.1 (2026-11-14):**
  `docs/baltic_b2_scenarios_2026-11-14.md`, `docs/baltic_c4_salinity_2026-11-14.md`,
  `docs/baltic_java_crosscheck_2026-11-14.md`, and the v1.10.1 section of
  `docs/baltic_sms_m2_validation_2026-10-03.md`. None changed materially.
- **Both engines now read sprat's maturity the same way.** Disassembling the 4.4.1 jar
  (`Species.<init>`, `Species.isSexuallyMature`) shows Java treats size and age as ALTERNATIVES:

  ```
  if species.maturity.size.sp<i> is set:  sizeMaturity = size, ageMaturity = +inf
  else:                                   ageMaturity = age,   sizeMaturity = +inf
  mature = length >= sizeMaturity OR age >= ageMaturity    (latched: once mature, always mature)
  ```

  The Python engine applies `length >= size AND age >= age` (absent size and age both read as 0).
  The two agree exactly when a species sets one key, never both. With the 9 cm size kept, Java would
  have ignored the age and left sprat maturing at 9 cm. With age only, Java and Python both mature
  sprat at age 1. No Java run was made for this change.
- **The rule is now enforced by a test.** `tests/test_maturity_keys_parity.py` fails if any species in
  the Baltic, EEC or Bay of Biscay master config sets both keys, and if Baltic sprat regains a size.
- **The UI could have undone it.** The Setup species table writes every rendered value back into the
  config, so sprat's missing size would have been written as the schema default, 50 cm, stopping sprat
  maturing in both engines. The same mechanism also wrote the schema's 3.0 yr maturity age into every
  species of a UI-launched run. Both are fixed in a separate UI PR, which this one depends on.
- **The engine rule itself is unchanged.** Moving the Python engine to Java's either/or rule is a
  separate parity decision. It would not affect this config, which sets only one key.
- **Java has a maturity ogive that the Python engine lacks.** Java 4.4.1's `species.maturity.mode =
  stochastic` reads `species.maturity.l50` and `.l75` and matures each school with the conditional
  probability of a normal maturity length N(l50, (l75 − l50)/0.6745). That branch ignores the maturity
  age. It would express "only part of the cohort is mature" by size directly, but the Python engine
  would need it ported first.

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/sprat_age1_maturity_share.py \
    --config data/baltic/baltic_all-parameters.csv --seed 42 --years 50 --out /tmp/sprat_42
PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current \
    --out /tmp/sprat_certification.md
```

Run summaries (master and candidate, seeds 42, 123, 7):
`docs/diagnostics/sprat_maturity_age_runs_2026-11-13.json`.
