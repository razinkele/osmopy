---
name: Background-species predation has zero effect on focal species (engine bug) [RESOLVED 2026-04-27]
description: 2026-04-27 — RESOLVED. Root cause was background.py:get_schools() never set length on background schools; size-ratio gate (pred_len/prey_len in [r_min, r_max)) rejected all prey when pred_len=0. See project_background_predation_length_fix.md for the fix and verification.
type: feedback
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
**STATUS: RESOLVED 2026-04-27.** Root cause was NOT the accessibility matrix or any of the 5 hypotheses below. It was simpler: `background.py:get_schools()` populated every SchoolState field except `length`, leaving it at the SchoolState.create() default of zeros. The size-ratio gate `pred_len/prey_len in [r_min, r_max)` then rejected all prey because `0 / anything < r_min`. Fix: 3-line patch in `get_schools()` to set length from `sp.lengths[cls_idx]`. Verified via mult=0/1/10 ramp showing dramatic, correctly-scaled predator effects. See `project_background_predation_length_fix.md`.

---
**Original investigation notes (kept for context):**

**Discovery method:** ran 5-year Baltic sims with `species.biomass.multiplier.sp14` and `.sp15` set to 0.0, 1.0, 10.0, 100.0. Final-year cod/herring/perch biomass identical to the byte across all four runs.

```
predators=0:    cod=1894254 herring=10265216 perch=7076262
predators=1:    cod=1894254 herring=10265216 perch=7076262
predators=10:   cod=1894254 herring=10265216 perch=7076262
predators=100:  cod=1894254 herring=10265216 perch=7076262
```

**What's working:**
- `species.biomass.multiplier.sp14/sp15` IS parsed (verified by inspecting `BackgroundSpeciesInfo.multiplier`).
- The multiplier IS applied at `osmose/engine/background.py:329` (`raw = raw * sp.multiplier`).
- Background schools ARE created and injected (`get_schools(0)` returns 2,464 schools totaling 50,000 t at 10× multiplier — confirmed correct biomass scaling).
- The reproduction.py fix (sex_ratio slicing, 2026-04-25) is intact.
- Schools are appended to the focal state via `state.append(bkg_schools)` at simulate.py:1179.

**What's broken:**
- Despite all the above, background schools contribute ZERO mortality to focal species via predation.
- Possible causes (need investigation in a future session):
  1. Background schools have wrong `species_id` mapping (engine assigns `species_id = n_focal + bkg_idx` per `background.py:378`, giving sp_id = 8 + 0 = 8 for seal — colliding numerically with LTL resource sp8 Diatoms in the config; but ResourceState and SchoolState are separate stores so this may not be the cause).
  2. Predator-prey accessibility matrix may default to 0 between background and focal species (need to check `osmose/engine/predation_access.py` or wherever the accessibility table is built).
  3. Background schools may be filtered out of predation eligibility (predators-only flag, age check, abundance check).
  4. Size-ratio matching may fail — but seal `predator_length` 110-170 cm × ratio 3-12 covers prey 9-57 cm which should overlap herring/sprat/cod/flounder.
  5. The `is_background` flag on schools may suppress them from being predators (only consumed, not consuming).

**Implications for prior work:**
- The 2026-04-26 phase 12 calibration ("with predators active") that achieved f=3.55 multi-seed mean — predators were NOT actually predating despite the engine running cleanly.
- The improvement vs phase 12 no-predators (f=5.24 → 3.55) cannot be attributed to predator pressure. It must come from:
  - Different DE random sampling / starting conditions
  - Subtle changes in init_pop seeding
  - The fact that this run started from `feature/tier1-predators` cherry-picked master state (different config history)
- The recommended next-steps from `project_phase12_with_predators.md` ("perch/pikeperch need density-dependent recruitment") still hold but for an additional reason: predator pressure can't help even in principle until the engine bug is fixed.

**To investigate (next session):**
1. Read `osmose/engine/processes/predation.py` carefully — find the loop over predators and check if schools where `is_background=True` (or species_id >= n_focal) are excluded.
2. Check the predator-prey accessibility matrix construction — is there a default-zero between background and focal that the config never overrides?
3. Run `scripts/calibrate_baltic.py` with diet output enabled, look at the diet matrix rows for background species (sp8, sp9 in the merged-id space) — are those rows all zero?
4. The diet output from a 5-year run was huge and hard to parse manually. Build a small script to extract per-predator-species diet totals.

**Workaround for the calibration:**
Until the bug is fixed, `species.biomass.multiplier` and the predator NetCDF biomass values have no effect on the calibration. The calibration is effectively just tuning focal-only mortality + fishing as if predators weren't there. To get realistic top-down pressure, either:
- Fix the predation pathway (proper)
- Use the existing `mortality.additional.rate.spN` + `mortality.additional.larva.rate.spN` parameters as the only handles (current effective state)

**Action items:**
- This memory documents the finding for future investigation.
- The `project_phase12_with_predators.md` claim that "background-species predation pathway is functional" should be qualified — predators are *injected as schools* but apparently not *predating focal species*. The pathway is half-functional.
