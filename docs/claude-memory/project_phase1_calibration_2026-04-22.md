---
name: Phase 1 calibration run 2026-04-22
description: Ran DE phase 1 (16 mortality params) for 6.7h wall-clock. Stopped extinctions but left 0/8 species in ICES range at 50-y equilibrium. 20-year eval window was too short — several species still growing at year 20.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Ran `scripts/calibrate_baltic.py --phase 1 --maxiter 15 --popsize 20 --popsize-mult 2 --years 20` on 2026-04-22. Fixed the hardcoded `whitefish` → `smelt` species-name bug in SPECIES_NAMES list before the run.

**Result summary:**
- DE objective: 12.19 (init) → **5.17** multi-seed mean (std 0.11, very stable)
- Plateau at 12.19 for 10 generations (eval 33-320); broke through at gen 11 → 9.37 → 6.78 → 5.35 (gens 11-13, then converged)
- 6.7h wall-clock for 512 evals
- Saved to `data/baltic/calibration_results/phase1_results.json`

**50-year validation verdict (via `scripts/report_calibration.py`):**
- ✅ **No extinctions** — smelt (0 → 5.06M t), stickleback (0 → 7.38M t). Combined with today's map fixes, the calibration successfully removed the extinction risk.
- ❌ **0/8 species in ICES range** at equilibrium (was 1/8 pre-cal; perch drifted from 11k in-range to 56k just-over).
- ❌ **Cod exploded** to 5.43M t (×22 upper target) — calibration set `mortality.additional.rate.sp1` (herring adult) to lower bound 0.001 + kept cod larval mortality maxed but adult mortality at defaults.
- ❌ **Herring collapsed** to 346k (×0.43 lower) — predated by over-abundant cod.
- ❌ **Pikeperch** ×173, **flounder** ×35, **smelt** ×42, **stickleback** ×15 — slow-equilibrium species all still growing at year 20.

**Root cause of imbalance:** 20-year evaluation window is too short to capture equilibrium for slow-dynamics species. Calibrator rewards parameters that look good mid-transient; at 50-year equilibrium several species continue to grow past target.

**Parameters at bounds (indicating DE wanted to go further but couldn't):**
- `mortality.additional.rate.sp1` = 0.001 (lower bound, log10=-3.0) — herring wanted lower mortality
- `mortality.additional.rate.sp3` = 1.574 (upper bound, log10=+0.2) — flounder mortality maxed
- `mortality.additional.larva.rate.sp0,sp4` = 15 (upper bound, log10=+1.18) — cod, perch larvae maxed
- `mortality.additional.larva.rate.sp1,sp5` = 8 (upper, log10=+0.9) — herring, pikeperch larvae maxed

**Recommended next action (future session):**
1. **Rerun phase 1 with `--years 50`** — single biggest quality improvement. ~16h wall-clock.
2. **Phase 2 calibration** (`--phase 2`, fishing mortality) — would add fishing-pressure control to reduce cod/pikeperch/flounder overshoots.
3. Consider loosening some upper bounds (cod larval mortality hit ceiling at 15) since DE clearly wanted more.

**Session-artifact files to keep:**
- `/tmp/osmose_baltic_50y/osm_biomass_Simu0.csv` — pre-cal 50y baseline (delete or preserve for comparison)
- `/tmp/osmose_calibration_full.log` — DE run log (512 evals, per-eval biomasses)
- `/tmp/osmose_postcal_report.log` — formatted pre/post report
- `data/baltic/calibration_results/phase1_results.json` — saved params (canonical)
- `scripts/report_calibration.py` — NEW script for pre/post analysis (this session)
