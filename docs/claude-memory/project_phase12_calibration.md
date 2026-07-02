---
name: Phase 12 joint calibration — pivot result
description: 2026-04-24 — joint 24-param DE with widened mortality bounds (background-species approach abandoned). Result worse than phase 2 stacked: cod extinct-near, perch/pikeperch exploded ×450/×750.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Ran `scripts/calibrate_baltic.py --phase 12 --maxiter 15 --popsize 24 --popsize-mult 2 --years 50` on 2026-04-24 in worktree `feature/tier1-predators` (branched off master commit `0d1cdf3`).

**Branch state at launch:**
- Tier 1 plan executed through the subagent-driven path — see `docs/superpowers/plans/2026-04-24-tier1-baltic-improvements.md`.
- T1-T4 added background-species machinery (grey seal + cormorant).
- T5 surfaced a latent OSMOSE Python engine bug in `osmose/engine/processes/reproduction.py` (sex_ratio shape n_focal+n_bkg vs ssb shape n_focal → broadcast error).
- **Pivot (T3alt, commit `06d5f80`):** deactivated background species in config, widened `mortality.additional.rate.spN` upper bound from log10=+0.3 to log10=+0.7 for 6 predated species (cod, herring, sprat, flounder, perch, pikeperch); kept default bound for smelt + stickleback.
- T6+T7 added `--phase 12` (joint 24-param) + report-script stacking handling.
- T8 is this calibration run.

**T2 side-effects to note:** `workers=-1` was added to scipy DE call with `updating="deferred"`. On a 28-thread box this spawned 28 workers at ~400 MB each → RAM exhaustion, load 237, zero progress in 1h. Capped at `workers=8` via `OSMOSE_DE_WORKERS` env var (commit `d714d1c`). Also: without `PYTHONUNBUFFERED=1`, scipy's disp=True output was block-buffered and invisible for 2h. Attempt 3 used `/tmp/launch_phase12.sh` wrapper with `PYTHONUNBUFFERED=1` + `python -u`.

**Result summary:**
- DE objective: 11.03 (gen 1) → **5.24 multi-seed mean** (std 0.03, extremely stable)
- 7.1 hours wall-clock, 8-worker parallelism
- Saved to `data/baltic/calibration_results/phase12_results.json`

**50-year validation (3 seeds, phase 12 params alone):**

| Species | Biomass | Target | Ratio | Verdict |
|---|---:|---:|---:|---|
| cod | 16,620 | 120,000 | 0.14 | ❌ LOW (×0.28 of lower bound) |
| herring | 9,122,479 | 1,500,000 | 6.08 | ❌ HIGH |
| sprat | 10,992,890 | 1,500,000 | 7.33 | ❌ HIGH |
| flounder | 2,997,772 | 50,000 | 59.96 | ❌ HIGH ×30 |
| perch | 9,133,023 | 20,000 | 456.65 | ❌ HIGH ×183 |
| pikeperch | 7,525,379 | 10,000 | 752.54 | ❌ HIGH ×301 |
| smelt | 1,105,101 | 60,000 | 18.42 | ❌ HIGH ×9 |
| **stickleback** | 256,584 | 200,000 | 1.28 | ✅ **IN RANGE** |

**1/8 species in range — same score as phase 2 stacked, different species in range.**

**Comparison with phase 2 stacked (2026-04-24):**

| Species | Phase 2 ratio | Phase 12 ratio | Δ |
|---|---:|---:|---|
| cod | 0.79 (in range) | 0.14 (low) | **LOST** in-range |
| herring | 3.66 | 6.08 | worsened |
| sprat | 9.25 | 7.33 | improved |
| flounder | 56.27 | 59.96 | worsened |
| perch | 2.63 | 456.65 | **much worse** |
| pikeperch | 464.70 | 752.54 | worse |
| smelt | 58.06 | 18.42 | improved |
| stickleback | 8.01 | 1.28 (in range) | **GAINED** in-range |

**Verdict:** the pivot (widening mortality bounds instead of adding background predators) did not produce a better calibration than the sequential phase 1→2 approach. Joint 24-param optimization found a different local optimum where stickleback lands in range but cod, perch, pikeperch explode. The perch ×457 result suggests DE found a regime where perch's widened larval+adult mortality combined with low fishing produced runaway population growth.

**Why perch exploded:** `mortality.additional.larva.rate.sp4 = 1.19` (near R18 default 13.0 — essentially DE reduced perch larval mortality by 11×), `mortality.additional.rate.sp4 = 0.046` (low adult mortality), `fisheries.rate.base.fsh4 = 0.058` (low fishing). Combined: very little per-capita mortality on perch. Trophic regulation also missing (no cormorant predator in model).

**Mandatory recommended follow-up:** at this point the calibration picture is:
- Parameter space is deeply multimodal; sequential phase 1→2 hits a different basin than joint phase 1+2.
- No parameter tuning alone will get 5+/8 species in range — the missing top-down predator pressure matters ecologically.
- Either (a) patch `osmose/engine/processes/reproduction.py` for correct n_focal slicing so background species actually work, or (b) accept that Baltic-OSMOSE is structurally mismatched to ICES biomass targets and rescope the project goal.

**Optimized parameters (for archival):**

```
Larval mortality (sp0..sp7):
  cod=10.20, herring=2.17, sprat=0.17, flounder=0.95,
  perch=1.19, pikeperch=4.02, smelt=0.16, stickleback=1.68

Adult mortality (sp0..sp7):
  cod=1.75, herring=0.13, sprat=0.068, flounder=1.65,
  perch=0.046, pikeperch=0.23, smelt=0.13, stickleback=0.071

Fishing (fsh0..fsh7):
  cod=0.091, herring=0.0047, sprat=0.0038, flounder=1.94,
  perch=0.058, pikeperch=0.91, smelt=0.0055, stickleback=0.017
```

**Parameters at bounds:** flounder fishing (fsh3=1.94) near upper log10=+0.5 (=3.16) — DE still wants more. Pikeperch fishing (fsh5=0.91) near upper log10=+0.5 also. Other bounds not active.

**Session artifacts:**
- `data/baltic/calibration_results/phase12_results.json` — 24 optimized params
- `data/baltic/calibration_results/phase1_results.json` — kept from Apr 22 phase 1 run
- `data/baltic/calibration_results/phase2_results.json` — kept from Apr 24 phase 2 run
- `/tmp/osmose_calibration_phase12.log` — full 7.1h DE log (attempt 3, unbuffered)
- `/tmp/osmose_calibration_phase12.attempt1-oom.log` — attempt 1 (workers=-1 RAM exhaustion)
- `/tmp/osmose_calibration_phase12.attempt2-buffered.log` — attempt 2 (stdout block-buffered)
- `/tmp/launch_phase12.sh` — wrapper that sets PYTHONUNBUFFERED=1 + cd into worktree
- `/tmp/osmose_postcal_phase12_report.log` — formatted 50y report

**Worktree state:** `feature/tier1-predators` branch has 15+ commits. Not merged to master — should be reviewed before merging given that the T8 outcome is weaker than phase 2 stacked.

**Tier 1 plan documentation:** `docs/superpowers/plans/2026-04-24-tier1-baltic-improvements.md` was written for the background-species approach that had to be abandoned. Should annotate it with a "STATUS-ABANDONED" banner pointing at this memory.
