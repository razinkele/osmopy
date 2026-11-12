# Baltic herring recalibration: maturity 14.5 cm, adult mortality 0.2 /yr, 2026-11-12

> **Not for auto-merge.** This is a recalibration of the production Baltic model, requested by the
> maintainer ("run the herring recalibration with maturity 14.5 cm and lower mortality"). It waits
> for the maintainer's decision to merge.

**Why:** on v1.9.0, herring's spawning stock was 22 kt, 1.5% of its 1.5 Mt biomass (#247). The data
check (`docs/baltic_herring_growth_maturity_2026-11-12.md`) found two main causes:
- an adult additional mortality of 2.25 /yr, which left 95% of herring at age 1, against 32% in
  the WGSAM SMS assessment;
- a knife-edge maturity size of 18 cm, against a BITS L50 of 14.3 cm.

**Result: herring now has a realistic spawning stock, and the model still certifies 5/5.**

| | v1.9.0 | **this PR** | data |
|---|---|---|---|
| herring biomass (≥0.5 yr) | 1.48–1.56 Mt | 1.51–1.59 Mt | target 1.5 Mt (0.8–3.0) |
| herring SSB | 22 kt (1.5%) | **1.40–1.47 Mt (93%)** | ~85% mature by biomass (BITS ogive × SMS weights) |
| herring Q1 age-2+ share | 4.7% | **40.5%** | 68% (SMS) |
| herring Q1 age 2 share | 4.5% | **24.7%** | 24.4% (SMS) |

## What changes (`data/baltic/`)

| key | was | now |
|---|---|---|
| `species.maturity.size.sp1` | 18.0 cm | **14.5 cm** (BITS L50 14.3 cm Q1 / 15.0 cm Q4) |
| `mortality.additional.rate.sp1` | 2.247 /yr | **0.2 /yr** (grid choice, see below) |
| `mortality.additional.larva.rate.sp1` | 28.89 /yr (1.204 per step) | **192.0 /yr (8.0 per step)**, re-fitted to 1.5 Mt |
| `output.ssb.enabled` | absent | **true** (outputs only; verified bit-identical) |

**SP1b re-solved** (`osmose/calibration/larva_recal.py` `RECAL_RATES`;
record in `docs/diagnostics/sp1b_solve.json`). Cod moved, so the SP1b rates were re-solved. It
converged in one sweep and 27 evaluations.

| stock | rate | was | baseline | joint error |
|---|---:|---:|---:|---:|
| cod_east | 8.9921 | 9.1407 | 121,760 t | +0.52% |
| cod_west | 8.9296 | 9.3162 | 11,231 t | −1.98% |

Both are inside the 1-D tolerance (2%) and JOINT_TOL (5%). d0 is unchanged.

The pre-stated design is `docs/diagnostics/baltic_herring_recal_prestated_2026-11-12.md`
(`1cbba80f`), committed before any run.

## The search

- **Adult mortality had no sourced value.** The request said "lower" without a number, and no
  sourced value for herring's residual natural mortality was available.
- **So it was a grid:** M_add ∈ {0.2, 0.6, 1.1} /yr. For each value, herring's larval rate was
  re-fitted to 1.5 Mt.
- **Every run used** maturity 14.5 cm, seed 42 and 50 years, and records the final-decade mean.
  Raw output: `docs/diagnostics/baltic_herring_recal_{grid,refine,candidate_seeds}_2026-11-12.json`.
- **The control passed.** The production config with SSB output on gives herring biomass
  bit-identical to the attribution run (1,482,700.0836698213 t), so SSB output does not perturb
  the dynamics.
- **The reachability ceiling was stated before the runs.** The model's age-1+ predation, fishing
  and starvation already sum to about 0.78 /yr, above SMS's cohort total mortality (0.32–0.68 /yr).
  So the 68% age-2+ share cannot be reached through M_add. The stationary estimate is about 46% at
  M_add = 0 and 38% at 0.2; measured, it is 40.5%.

### Coarse grid

Herring final-decade biomass, kt, by larval rate (resolved, per step):

| M_add | 1.2 | 2.0 | 3.0 | 4.1 | 5.5 | 7.5 | 10.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 8,389 | 7,771 | 7,073 | 5,925 | 4,244 | 2,050 | **0** |
| 0.6 | 5,756 | 5,521 | 4,862 | 3,928 | 2,715 | 1,147 | **0** |
| 1.1 | 4,109 | 3,782 | 3,232 | 2,535 | 1,678 | 613 | **0** |

**Herring collapses at a larval rate of 10 at every M_add.** Near 1.5 Mt, SSB share rises steeply
with the larval rate, because suppressing recruitment leaves mostly mature adults.

### Refinement near 1.5 Mt

| M_add | larval rate | herring | SSB share | age 2+ | assessed in envelope |
|---|---:|---:|---:|---:|---|
| **0.2** | **8.0** | **1,586 kt** | **92.7%** | **40.7%** | **5/5** |
| 0.2 | 8.6 | 1,024 kt | 98.8% | 39.4% | 5/5 |
| 0.2 | 9.2 | 587 kt | 100.7% | 38.5% | 4/5 |
| 0.6 | 7.2 | 1,409 kt | 82.5% | 27.3% | 5/5 |
| 1.1 | 6.0 | 1,432 kt | 52.6% | 16.3% | 5/5 |

**Selection, by the pre-stated rule.** Among M_add values with a refined run inside ±10% of 1.5 Mt
and 5/5 assessed stocks in envelope, the candidate with herring's age-2+ share closest to SMS was
taken, provided SSB was at least 50% of biomass. That is **M_add 0.2, larval rate 8.0**.

**It sits on the steepest part of the curve.** The local slope is about −6 in ln B per ln L: a 7%
higher larval rate gives 1.02 Mt. The 5-seed spread below is tight, so the steepness does not
show up as seed sensitivity.

## Certification: 5/5 assessed (50 yr × 5 seeds)

| species | v1.9.0 | **this PR** | envelope |
|---|---|---|---|
| cod_west | 14.9–18.3 kt | 18.7–19.3 kt | 4–25 kt |
| cod_east | 76.1–80.0 kt | 73.7–75.6 kt | 60–85 kt |
| herring | 1.48–1.56 Mt | 1.51–1.59 Mt | 0.8–3.0 Mt |
| sprat | 0.88–0.90 Mt | 0.90–0.92 Mt | 0.8–2.5 Mt |
| flounder | 47.6–49.8 kt | 47.6–49.1 kt | 20–100 kt |
| *perch* | *60.0–62.9 kt (above)* | ***33.9–36.4 kt (inside)*** | *8–50 kt* |
| *pikeperch* | *598–619 kt* | *515–529 kt* | *4–25 kt* |
| *smelt* | *648–668 kt* | *655–663 kt* | *20–120 kt* |
| *stickleback* | *90–102 kt* | *82–98 kt* | *50–500 kt* |

- **The cliff rule passes:** no assessed stock's 5-seed range crosses its envelope.
- **The indicative tier improves from 1/4 to 2/4:** perch moves inside its envelope, and pikeperch
  falls by about 14%.

Herring per seed (`baltic_herring_recal_candidate_seeds_2026-11-12.json`):
- SSB 1.40–1.47 Mt, 92.6–93.3% of biomass;
- age-2+ share 40.2–40.7%;
- SSB / Shepherd ssbhalf ≈ 1.98, with ssbhalf kept at 725,836 t.

## What it does not fix

- **Age structure is still younger than assessed:** 59.5% age 1 against SMS 32%. The remaining
  gap is the model's predation and fishing on age-1+ herring, which this parameter cannot reach.
  They sum to about 0.72 /yr in the v1.9.0 state; they were not re-measured on this candidate.
- **Herring's growth curve was not touched.** Model L∞ is 27 cm against a BITS fit of 23.5 cm.
- **0.2 /yr is a grid value, not a sourced estimate.**

## Tests

- **Full local suite on the final branch** (master merged in): **5,030 passed**, 51 skipped (opt-in
  or data-gated), 2 xfailed (known gaps, strict), **0 failed**. No absolute-value fixture needed
  re-deriving.
- **The SP1b drift guard runs locally and passes on the new rates.** Mutation-checked: restoring
  the old cod_west baseline makes it fail.
- **The harness asserts the units on every run:** the adult rate is used as read, and the larval
  rate equals CSV / 24.

## What goes stale

These results were all computed on v1.9.0's herring:
- the B2 and C4 scenario re-runs;
- the B2 herring attribution and the compensation breakdown;
- the SMS M2 addendum;
- the Java cross-check (its arm reads the same data).

None was re-run here. Issue #247 is addressed if this is merged.
