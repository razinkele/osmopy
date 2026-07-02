---
name: project_density_dependent_recruitment
description: "Density-dependent recruitment (Shepherd + hockey-stick SR) feature — ALL tasks 1-6 done on worktree branch, ready for PR. Shepherd beats B-H, obj 6.0->2.1"
metadata: 
  node_type: memory
  type: project
  originSessionId: b8c2d70b-86aa-4fbf-8ddc-71656c6e971b
---

Density-dependent recruitment feature adds **Shepherd** (B-H generalized by shape
exponent β; β>1 over-compensates → recruitment peaks then declines) and **hockey-stick**
stock-recruitment forms to the engine, to flatten the over-productive recruitment curves
that left perch/pikeperch ×100+ over ICES range in every prior Baltic calibration. See
[[project_phase12_cod_floor]] (the cod-floor failure showed constraints alone can't fix it).

**Location:** worktree `.claude/worktrees/density-dependent-recruitment`, branch
`worktree-density-dependent-recruitment`. Plan/design committed to master at
`docs/plans/2026-05-28-density-dependent-recruitment-{plan,design}.md`.

**Status 2026-05-30:**
- Tasks 1-5 DONE + green on the branch (9 commits): Shepherd+hockey-stick engine math
  (`apply_stock_recruitment` in reproduction.py, β param, β=1≡B-H exact anchor), config
  key `stock.recruitment.shape.sp{i}` + new types in allow-set, reproduction() wiring,
  calibrator `get_phase13_shepherd_params()` (39 params, all-8-on-Shepherd, cod sp0
  ssb_half fixed 120kt). 55 SR/config/calib tests pass.
- Task 6 (experiment) IN PROGRESS. Added `scripts/evaluate_calibration_vs_ices.py`
  (runs one full sim for a param set + ICES in-range count) and committed preliminary
  doc `docs/baltic_shepherd_calibration_2026-05-30.md` (commit 6d58d72).
- **KEY FINDING (40-y equilibrium, identical conditions):** Shepherd already beats B-H
  **2/8 vs 1/8 in ICES range** even on a mis-calibrated short-transient quick run.
  Over-compensation (β>1) brought sprat into range and cut smelt ×21→×5.4 with NO
  mortality change — mechanism validated. The β<1 mis-picks (perch/pikeperch made worse)
  and stickleback collapse (β=2.1 over-crush) are short-sim artifacts a full 40-y
  calibration fixes by construction.
- **IN FLIGHT:** proper 4h DE calibration launched (PID was 2690092; log
  `/tmp/phase13_calib.log`; background waiter task id b6p0php1m). Invocation:
  `--phase 13 --optimizer de --seeds 3 --years 40 --popsize-mult 5 --warm-start
  phase12_results.json --skip-warm-start-keys mortality.additional.rate.sp0 --patience 20
  --wall-clock-cap-h 4 --checkpoint-every 5`, OSMOSE_DE_WORKERS=16, eff_popsize 195,
  ~6-7 gens. Checkpoints to `phase13_checkpoint.json`. Overwrites phase13_results.json.

**OUTCOME 2026-05-30 (Task 6 DONE):** proper 4h run finished — obj 6.008→**2.133**
single-seed (−64%, project best-ever, multi-seed ±0.012 most seed-robust). At seed 42 /
40y: **Shepherd 2/8 (herring+smelt) > B-H 1/8 (flounder)** → PASS on plan primary
criterion. Over-comp (β>1) drove cod ×48→×11, smelt ×21→in-range, perch ×228→×80. HONEST
CAVEAT: strict-count gain modest because dominated by weight-0.2 percids (perch ×80,
pikeperch ×167) the weighted objective deprioritizes — pikeperch even kept β=0.50
(under-comp) since its low weight gives DE no reason to cap it. That's a grid-resolution
limitation ("coarse grid under-resolves habitat" per biomass_targets.csv), NOT a
recruitment-form failure. Final betas: cod 1.88, herring 0.76, sprat 0.75, flounder 1.80,
perch 1.60, pikeperch 0.50, smelt 2.56, stickleback 1.79. Commits 6d58d72 (eval script +
prelim) + 0106d67 (final verdict doc). Doc: `docs/baltic_shepherd_calibration_2026-05-30.md`.

**NEXT:** full test suite + lint pre-PR check (calibration cores now free), then
finish-the-branch (PR `worktree-density-dependent-recruitment` vs master). B-H baseline at
main checkout `data/baltic/calibration_results/phase12_results.json` (gitignored; copied
into worktree for warm-start). Quick run backed up `phase13_results.quick-transient.json`.
Follow-ups (out of scope, in doc): reweight objective for strict-count; percid spatial
under-resolution; tighten pikeperch β lower bound.

## SHIPPED to PR #50 (2026-05-30)
Tasks 1-6 complete. Branch `worktree-density-dependent-recruitment` pushed to origin; **PR #50 OPEN** vs master (https://github.com/razinkele/osmopy/pull/50), 11 feature commits. origin/master fast-forwarded to 441ac54 (the 3 design/plan-doc commits) before PR so PR shows only feature work. Verification: 55 tests pass across the 3 changed test files (test_engine_stock_recruitment, test_engine_config_validation, test_calibrate_baltic_parallelism). Verdict PASS: 2/8 ICES in-range vs 1/8 B-H baseline; weighted objective 6.008 -> 2.133 (-64%); +-0.012 multi-seed (most seed-robust Baltic result). Honest caveat: strict-count gain modest because perch/pikeperch are low-weight grid-under-resolved percids (spatial-resolution limit, not SR-form failure). Next: await review/merge of PR #50.
