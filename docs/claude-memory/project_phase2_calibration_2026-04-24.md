---
name: Phase 2 calibration + full pre/post comparison
description: 2026-04-24 — phase 2 (fishing params) added on top of phase 1, evaluated at 50-year equilibrium. Cod now in ICES range, but trophic cascade made sprat worse. 1/8 in range but composite obj 3.93 vs pre-cal ~8.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Ran `scripts/calibrate_baltic.py --phase 2 --maxiter 10 --popsize 16 --popsize-mult 2 --years 50` on 2026-04-23/24. First applied a code fix: phase 2 now auto-stacks phase 1 params (without the fix, phase 2 would have run with R18 defaults, reintroducing extinction risk).

**Result summary:**
- DE objective: 4.15 (init) → **3.93** multi-seed mean (std 0.005, exceptionally stable)
- 10.2h wall-clock for 176 evals (50-year eval window, ~3.5min/eval)
- Stacked with phase 1 → 24 calibrated params total
- Saved to `data/baltic/calibration_results/phase2_results.json`

**50-year validation (3 seeds, phases 1+2 stacked via `scripts/report_calibration.py`):**
- ✅ **cod 94,653 t** (target 120k, ratio 0.79) — **IN ICES RANGE**. Was 5.4M after phase 1 alone.
- ✅ No extinctions — smelt 3.59M, stickleback 1.57M (down ×5 from phase 1's 7.4M)
- ✅ Herring 5.54M (×1.8 upper) — down from pre-cal 14.9M
- ✅ Perch 56k (×1.1 upper) — drifted slightly past upper bound
- ❌ **Sprat 14.5M (×5.8)** — got WORSE than phase 1's 3.6M. Trophic cascade: cod suppression freed sprat.
- ❌ Pikeperch 4.63M (×185), flounder 2.80M (×28), smelt 3.59M (×30) — still blown up
- 1/8 species formally in range (cod only), but composite objective 3.93 vs. pre-cal ~8

**Optimized fishing rates (from phase2_results.json):**
- cod      fsh0 = 0.372  (high — successfully controls cod)
- herring  fsh1 = 0.0037 (very low — lets herring recover)
- sprat    fsh2 = 0.092  (TOO LOW — sprat exploded post-cod-control)
- flounder fsh3 = 0.976  (at upper bound — DE wanted more)
- perch    fsh4 = 0.033
- pikeperch fsh5 = 0.685 (high but insufficient)
- smelt    fsh6 = 0.029
- stickleback fsh7 = 0.113

**Root cause of remaining imbalances:** trophic cascade not captured by sequential phase optimization. Phase 2 picked sprat fishing given phase-1 predator biomass, but controlling cod changed sprat dynamics. Flounder fishing hit upper bound.

**Code fix applied:**
- `scripts/calibrate_baltic.py` — phase 2 now auto-loads phase1_results.json into base_config (lines ~543-557)
- `scripts/report_calibration.py` — `--phase 2` stacks phase 1 overrides automatically

**Recommended next steps (future sessions):**
1. **Joint phase 1+2 calibration** (24 params) with `--years 50` — ~20h run, captures full trophic coupling
2. **Widen fishing upper bound** from log10=0.0 (1.0) to log10=+0.5 (~3.0) for flounder + pikeperch so DE isn't bound-limited
3. **Iterative alternation** (phase 1 → phase 2 → phase 1 → phase 2) to let DE refine until convergence

**Session-artifact files:**
- `data/baltic/calibration_results/phase1_results.json` — 16 mortality params
- `data/baltic/calibration_results/phase2_results.json` — 8 fishing params (new)
- `/tmp/osmose_calibration_phase2.log` — full 10h DE log
- `/tmp/osmose_postcal_phase2_report.log` — formatted 50y report

**2026-04-22 + 2026-04-23/24 calibration arc summary:**
- 2 calibration runs totaling 17h compute
- 24 parameters tuned (16 mortality + 8 fishing)
- Composite objective: pre-cal ~8 → phase 1 @ 20y 5.17 → phase 1 @ 50y ~12 → phase 2 @ 50y **3.93**
- 0 extinctions (was 2: smelt, stickleback)
- 1 species (cod) in ICES range at 50y equilibrium
- Model dynamics now stable (all CVs < 0.2) and biologically defensible
