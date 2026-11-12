# Baltic herring maturity + mortality recalibration: pre-stated design, 2026-11-12

Written and committed **before** any run. Requested by the maintainer: "run the herring
recalibration with maturity 14.5 cm and lower mortality" (issue #247, data check in
`docs/baltic_herring_growth_maturity_2026-11-12.md`).

## Fixed by the request

- **Maturity size:** `species.maturity.size.sp1` = 18.0 → **14.5 cm**. BITS L50 is 14.3 cm in Q1
  and 15.0 cm in Q4.
- **Adult additional mortality:** `mortality.additional.rate.sp1` is lowered from 2.247 /yr.
  - The request gives no number, and no sourced value for herring's residual natural mortality
    is available here.
  - So it is a **grid of three unsourced values: 0.2, 0.6 and 1.1 /yr**. 1.1 is about half the
    current rate.
  - The adult rate is per year and used as read; the reader does not migrate it. That is
    verified, and the harness asserts it.

## Re-fitted, as in #236

- **Herring larval mortality:** `mortality.additional.larva.rate.sp1` is the free parameter for
  each M_add, re-fitted to herring's biomass target of **1.5 Mt**, final-decade mean of a 50-yr
  run.
- **Units:** resolved per-step units, with CSV = resolved × 24. The harness asserts the resolved
  value.
- **cod_west larval rate:** re-fitted (target 10 kt, accept ≤ 20 kt) **only if** cod_west leaves
  its 4–25 kt envelope.
- **cod_east:** checked against 60–85 kt.
- **Order:** Gauss–Seidel, herring first.
- **Not re-fitted:** herring's Shepherd `ssbhalf` stays fixed at 725,836 t, and SSB/ssbhalf is
  reported for every candidate. If herring cannot reach 1.5 Mt at any larval rate for some M_add,
  ssbhalf is the first suspect, and that is reported rather than silently re-fitted.

## Reachability, stated before the runs

The model's age-1+ predation, fishing and starvation hazards already sum to about 0.78 /yr. That
is above the assessment's cohort total mortality, Z 0.32–0.68 /yr at ages 1–6
(`herring_compensation_2026-11-12.md`).

With constant annual survival s = e^−Z, the stationary age-2+ share of age-1+ numbers is s:
- about 46% at M_add = 0;
- about 38% at M_add = 0.2;
- about 25% at M_add = 0.6;
- about 15% at M_add = 1.1;
- about 5% at the current 2.25. This matches the measured 4.7%.

**So the SMS age structure (68% age 2+) cannot be reached through this parameter.** The other
hazards will also move as herring change size.

## Procedure

1. **Coarse grid**, 1 seed (42), 50 yr, in parallel: M_add ∈ {0.2, 0.6, 1.1} × herring larval
   rate (resolved) ∈ {1.2, 2.0, 3.0, 4.1, 5.5, 7.5, 10.0}.
   - A reference run uses maturity 14.5 with the current M_add (2.247) and larval rate (1.2036).
   - A control run uses the production config, seed 42, with `output.ssb.enabled` added. Its
     herring biomass must be **bit-identical** to the stored attribution run A0/baseline/42.
     Otherwise SSB output perturbs the dynamics, and nothing is interpreted.
2. **Interpolate** per M_add, log-log, to the larval rate giving 1.5 Mt. Then run a refinement wave
   at that rate, accepting herring within ±10%.
3. **cod_west:** if it leaves 4–25 kt in a refined run, re-fit its larval rate for that M_add.
4. **Select:**
   - Among M_add values whose refined run has all 5 assessed stocks inside their envelopes (one
     seed), take the one with herring's Q1 age-2+ share **closest to SMS's 68%**, the ceiling
     having been reported.
   - Herring SSB must be at least 50% of its ≥0.5-yr biomass. The data anchor is about 85% (BITS
     ogive × SMS weights).
5. **Certify the selected candidate:**
   - `baltic_stability_certify.py`, 50 yr × 5 seeds, with all 5 assessed stocks persistent and in
     envelope.
   - **Cliff rule** (from #236's rejected candidate A): if any assessed stock's 5-seed final-decade
     range crosses its envelope, the candidate is rejected. The next M_add, or a geometric midpoint
     larval rate, is tried instead.
6. **Then:**
   - re-solve SP1b if cod moved;
   - run the full test suite, re-deriving any absolute-value fixture with an isolated A/B;
   - enable `output.ssb.enabled` in `baltic_param-output.csv`. It is outputs-only; the step-1
     control shows that.

## Reported for every candidate

- herring biomass and SSB;
- SSB/biomass and SSB/ssbhalf;
- Q1 age shares;
- the larval rate (resolved and CSV);
- every assessed and indicative stock's final-decade biomass.

## Boundary

The PR is opened as **not for auto-merge**, since a recalibration that moves other stocks is the
maintainer's to merge. It lists what goes stale again:
- the B2 and C4 re-runs;
- the herring attribution and compensation notes;
- the SMS M2 addendum;
- the Java cross-check.
