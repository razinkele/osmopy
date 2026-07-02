---
name: predation-architecture-parity
description: Predation must process school+resource prey in single proportional pass (not two-phase)
type: feedback
---

Java's `computePredation()` processes ALL prey (schools + resources) in a single accessibility/proportional distribution. Python previously had a two-phase approach: school prey eaten first, then resources filled remaining appetite. This caused two bugs:

1. **Schools-first bias**: Small forage fish (redMullet, sardine) got over-predated because predators exhausted appetite on schools before resources could buffer
2. **Remaining appetite miscalculation**: Used cumulative `pred_success_rate` (divided by n_subdt) instead of actual within-sub-timestep consumption, allowing predators to eat ~45% more than max_eatable per sub-timestep

**Why:** These two bugs combined caused redMullet (-2.0 OoM) and sardine (-1.4 OoM) divergence. After fix: 14/14 EEC species within 1 OoM.

**How to apply:** Any future predation changes must maintain the unified single-pass approach in `_apply_predation_for_school`. Never split school and resource predation into separate phases with independent success rate updates.
