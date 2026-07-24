# Baltic stability — Phase 0 diagnostic finding (SP-A)

**Date:** 2026-07-01
**Script:** `scripts/baltic_stability_diagnostic.py` (50 yr × seeds 42/123/7, Python engine)

## What the data shows

| species | weight | ICES target (kt) | behaviour over 50 yr |
|---|---|---|---|
| herring | 1.0 | 1500 | **persists** (only survivor near/above ICES) |
| sprat | 1.0 | 1500 | starts below 0.1·lower then **overshoots** to ~6.9 Mt |
| cod | 1.0 | 120 | below 0.1·ICES-lower **from yr 0**; never establishes → ~0 |
| flounder | 0.5 | 50 | below 0.1·lower from yr 0 → ~0 |
| perch | 0.2 | 20 | below 0.1·lower from yr 0 → ~0 |
| pikeperch | 0.2 | 10 | below 0.1·lower from yr 0 → ~0 |
| smelt | 0.3 | 60 | below 0.1·lower from yr 0 → ~0 |
| stickleback | 0.2 | 200 | holds ~20 yr then collapses (~yr 19–21) |

Seed-independent (42/123/7 identical pattern).

## Interpretation (sharper than "slow collapse")

The "first year below 0.1·ICES-lower" metric conflates **seeded-below-target** with **collapsed-below-target**
(sprat reads "yr 0" yet later overshoots). Reading it together with the 50-yr trajectories, the real
diagnosis is:

- **This is not a healthy state that drifts to collapse — it is a config that never establishes 6 of 8
  species near their ICES targets.** Cod, flounder, perch, pikeperch, smelt sit 3–5 orders of magnitude
  below ICES essentially from the start; herring and sprat overshoot. This matches the project's known
  "1/8 ICES-in-range" status — the original calibration only ever placed one stock in range.
- The keystone framing (cod "first to collapse at yr 0") is therefore an **establishment/recruitment
  failure**, not a predation-during-decline cascade. The mortality-cause decomposition was moot (a
  species below the floor from yr 0 has no "pre-collapse window") and failed on a string-dtype column;
  not pursued — the establishment finding stands without it.

## Consequence for SP-A (the confirmed free-parameter set)

SP-A is effectively a **full ICES + stability recalibration**, exactly what the ε-constraint design
targets (minimise ICES_loss subject to bounded persistence). Because the failure is establishment +
recruitment, the free-parameter set is the `configure.py` baseline **plus the recruitment levers**:

- `mortality.additional.rate.sp{i}`, `mortality.additional.larva.rate.sp{i}`,
  `mortality.starvation.rate.max.sp{i}`, `predation.ingestion.rate.max.sp{i}` (baseline), **and**
- `stock.recruitment.ssbhalf.sp{i}` (Beverton-Holt half-saturation — density-dependence strength),
  `species.relativefecundity.sp{i}` (recruitment magnitude — the direct lever for under-establishing
  stocks), and `stock.recruitment.shape.sp{i}` for the Shepherd-type percids.

The ICES term must pull the 6 under-target species **up** while the stability term keeps herring/sprat
from overshooting and the rest from the late collapse — a genuine multi-criteria problem, well-suited
to the ε-constraint sweep.
