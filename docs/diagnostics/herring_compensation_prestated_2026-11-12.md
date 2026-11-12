# Herring compensation breakdown: pre-stated criterion, 2026-11-12

Written and committed **before** any run. Harness: `scripts/herring_compensation_breakdown.py`.

## Question

B2's +2.9 °C knob multiplies herring's egg production by **f = 0.228** (C1 reproduction thermal
gate, after spin-up). Herring's final-decade biomass falls by:
- **26.5%** in the v1.9.0 state (A0);
- **61.2%** in the v1.8.8 state (A4: all of #236 reverted; `b2_herring_attribution_2026-11-12.md`).

Both states must offset the same f through density-dependent compensation. **Through which
channel does A0 compensate with a smaller fall in density, and which cause's hazard moves in that
channel?**

## Design

The design runs A0 and A4, each as `baseline` and `rcp85_ref` (built as in the attribution), over
the 5 house seeds for 50 yr: 20 runs. The config is byte-identical to the attribution runs. Two
read-only hooks record herring per step:

- `reproduction.regulate_recruitment` records:
  - the engine's spawning biomass (SSB, the same maturity rule reproduction uses);
  - linear eggs;
  - eggs after the Shepherd curve (ssbhalf 725,836 t, β 1.085);
  - eggs after the knob.
- `simulate._collect_by_life_stage` records:
  - deaths by cause (all 8 causes, incl. OUT and AGING), for egg, age-0 post-egg and age 1+ fish;
  - standing stock after mortality;
  - recruits, i.e. fish at `age_dt == 24`. Age advances by exactly 1 per step for every school,
    so each cohort is counted once.

## Channels and what each can say

With warm/base ratios r over the final decade (years 40–49), five channels make up the chain:

| channel | definition |
|---|---|
| lin | linear eggs / SSB |
| SR | after-curve / linear eggs |
| knob | after-knob / after-curve eggs |
| early | recruits / eggs (egg → age-1 survival) |
| SPR | SSB / recruits per year |

Two properties of that chain fix what it can and cannot show:

- **The product of the five ratios is 1 by construction.** That is an identity, **not a test**.
- **The only real instrument check in the chain is the knob channel.** On every non-seeded step,
  after-knob / after-curve must equal f exactly.

The comparison between states uses each compensating channel's **descriptive secant ratio**
η_k = ln r_k / ln r_SSB. These are two-point ratios between equilibria at different operating
points, **not density-dependence slopes**. Because ln r_SSB = −ln f / Σ_k η_k, a state with a
larger Σ η needs a smaller fall in SSB.

## Gates

- **G1, read-only hooks:** every run's herring final-decade biomass must be bit-identical to the
  matching run in `b2_herring_attribution_2026-11-11.json`. If any differs, the hooks perturbed the
  dynamics and nothing is interpreted.
- **G2, knob exactness:** after spin-up, after-knob / after-curve equals f = 0.22786543992898978 on
  non-seeded warm steps, and 1.0 on baseline steps. No seeded herring steps may fall inside the
  window.
- **G3, year-0 accounting:** (recruits + egg and age-0 deaths) / eggs must lie within 5% of 1, in
  every run. A shortfall is expected from `_floor_sub_one_fish`, which zeroes sub-one-fish schools
  without booking them in `n_dead`; for herring that should be tiny. A shortfall over 5% is an
  instrument fault.
- **Stationarity:** the decomposition is also computed on lagged windows (eggs and SSB over years
  39–48, recruits over 40–49). If the two windows disagree on which channel carries the
  difference, the final decade is not at equilibrium and the split is approximate. It is then
  reported as such.

## Criterion

1. **Channel.** The channel carrying A0's damping is the one whose η differs most between A0 and
   A4, in the direction of A0 compensating more per unit fall in SSB. If two channels move in
   opposite directions, both are reported with their sizes.
2. **Cause, two tables, two questions:**
   - **Which cause responded:** per-capita hazard, deaths_c over standing stock at the start of
     the step, per class. Comparing warm with base in each state shows which density-dependent
     cause's hazard falls when density falls. Larval *additional* mortality differs 3.4× between
     the states by construction (28.89 vs 98.56 /yr). That is the state difference, not a
     response.
   - **How much each cause carries:** year-0 cohort fractions, deaths_c / eggs. These sum to 1 −
     survival. Under competing risks a density-independent cause's fraction can rise when another
     falls, so this table is not used to name the responding cause.
3. **Recorded prediction (falsifiable):** the Shepherd channel alone should make the
   **higher-SSB state (A4)** compensate **more** per unit SSB. Its secant η_SR approaches −β·x/(1+x)
   with x = (SSB/ssbhalf)^β, which grows with SSB. So if A4 is the more sensitive state, the A0
   damping must come from the early or SPR channels, and must outweigh the SR effect. If instead
   η_SR is larger in A0, this reading of the Shepherd algebra is wrong for these operating points,
   and that is reported.

## Boundary

The note stops at **channel and cause**. It does not explain *why* that hazard moves, whether
through predator saturation, food competition or cannibalism. That would need an intervention.
