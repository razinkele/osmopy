---
name: Background-species predation length=0 fix
description: 2026-04-27 — fixed background.py:get_schools() to set length from sp.lengths[cls_idx]. Without this, predators had length=0, size-ratio gate rejected all prey, predators contributed zero mortality.
type: feedback
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
**Root cause:** `osmose/engine/background.py:get_schools()` populated `species_id`, `is_background`, `age_dt`, `first_feeding_age_dt`, `biomass`, `weight`, `abundance`, `trophic_level`, `cell_x`, `cell_y` — but never set `length`. `SchoolState.create()` defaults `length` to zeros. In predation:
```python
pred_len = length[p_idx]  # = 0 for background schools
ratio = pred_len / prey_len  # = 0
if ratio < r_min or ratio >= r_max:  # r_min ≥ 2.5
    continue  # all prey rejected
```
Predators had positive abundance/biomass/ingestion_rate but zero length, so the size-ratio check always failed → zero predation regardless of `species.biomass.multiplier`.

**Why:** OSMOSE config declares `species.length.spN=...` for background species (e.g., sp14 GreySeal `110;170`, sp15 Cormorant `70;85`), parsed into `BackgroundSpeciesInfo.lengths`. The data is there; `get_schools()` simply didn't propagate it.

**How to apply:** the fix is a 3-line change at `background.py:get_schools()` — declare `all_length` list, append `np.full(n_ocean, sp.lengths[cls_idx], dtype=np.float64)` per class, pass `length=np.concatenate(all_length)` to `state.replace()`.

**Verification (5y Baltic sim, multi-multiplier ramp):**
```
mult     cod  herring     sprat  flounder  perch  pikeperch  smelt  stickleback  Seal  Cormorant
0.0    53721  15734510  1749062    104007  10499      5549   1014      8072384     0          0
1.0    11544  15958566  1784812     48809   1349      1129      0      8325377  4500        499
10.0     212  15308296  1208254      1083      0       299      0      8686696 44999       4999
```
- Cod: 53,721 → 11,544 → 212 (strong seal predation)
- Flounder: 104k → 49k → 1k (seal)
- Perch: 10k → 1k → 0 (cormorant + seal)
- Pikeperch: 5.5k → 1k → 300 (cormorant)
- Stickleback slight rise: trophic cascade (herring/sprat decline → more zooplankton)
- Predator biomass scales correctly 1×/10× (4500/45000, 499/4999)

**Implications for prior calibration work:**
- The 2026-04-26 phase 12 result (f=3.55 multi-seed) was achieved with predators inert. Now predators are real, the calibration must be re-run.
- The next calibration will see strong cod/flounder/perch/pikeperch suppression even at biologically-realistic predator biomass — DE will likely back off natural-mortality and fishing rates more aggressively than before.
- The structural perch/pikeperch overshoot may be partly relieved by realistic cormorant predation (cormorant directed at perch zeros it out at 10×; at 1× the suppression is meaningful 10k → 1k).

**Regression test:** `tests/test_reproduction_background_compat.py` still passes (it asserts engine returncode==0, which the fix does not change).

**Action items:**
- Update `project_background_predation_zero_effect_bug.md` to note resolved.
- Re-run phase 12 calibration with active predators.
- Consider lowering predator NetCDF biomass — the 50,000 t × 10 case zeros stocks, suggesting 1× may be near upper bound of usable pressure. Current standing biomass values may be ecologically high.
