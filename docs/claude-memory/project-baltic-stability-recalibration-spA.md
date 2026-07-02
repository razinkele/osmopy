---
name: project-baltic-stability-recalibration-spa
description: "SP-A Baltic long-term stability recalibration — IN PROGRESS on branch worktree-spec+baltic-stability-spA (T1-3 shipped, T4-6 remain)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**SP-A = recalibrate `data/baltic` free params for long-term (50yr+) bounded-equilibrium stability** — first of 3 sub-projects (SP-A params-only current grid → gate → SP-B grid re-resolution → SP-C recal on new grid). Follows the finding that Baltic collapses on both engines [[project-baltic-config-instability-diagnostic]]. Full brainstorm→spec→in-loop-review→plan→**Workflow plan-review (51 agents, 34 findings)**→reconcile→execute. **On branch `worktree-spec+baltic-stability-spA` (NOT merged/pushed).**

**Docs:** spec `docs/superpowers/specs/2026-07-01-baltic-stability-recalibration-spA-design.md`; plan `docs/superpowers/plans/2026-07-01-baltic-stability-spA.md`.

**Design (post-review, KEY decisions):**
- **eps-constraint** (minimise ICES_loss s.t. Stability≤eps, sweep eps) — NOT weighted-sum (only recovers convex hull). Reuses the EXISTING envelope-aware Baltic objective in `calibrate_baltic.py`.
- `calibrate_baltic.py` ALREADY had `w_stability` cv/trend penalties (default 5.0) that FAILED (no persistence term + short horizon) → **zeroed when eps finite** (avoid double-count). Premise "no stability term" was WRONG.
- `stability_penalty`: SMOOTH log10-below-floor persistence (commensurate w/ ICES log10^2, not flat +10) + envelope + **late-window** trend (max of full + final-third slope) + variability (0 for boom-bust stickleback).
- Multiseed: driver optimises single-seed then multi-seed RE-RANKS (no per-call two-factory needed — deviates from plan, correct for this driver).

**Phase 0 finding (T1, reframes SP-A):** NOT a drift-collapse — **6/8 species (cod/flounder/perch/pikeperch/smelt) never establish near ICES** (3-5 OoM below from yr0); herring persists, sprat overshoots, stickleback collapses ~yr19. Matches known "1/8 ICES-in-range". → establishment/recruitment failure → free-param set = configure.py baseline (additional/larva/starvation mortality, ingestion) + **recruitment levers** `stock.recruitment.ssbhalf`/`species.relativefecundity`/`stock.recruitment.shape`(percids).

**SHIPPED (committed on branch — T1-T5 ALL done):** T1 diagnostic+note (`8898e82`); T2 `osmose/calibration/stability.py`+7 tests (`f114121`); T3 eps-constraint in `calibrate_baltic.py` (`b4715b8`) — `run_simulation(return_biomass=)`, `_ObjectiveWrapper(epsilon=)` (inf=legacy), `--epsilon`, ices_loss/stability recorded; T4 `scripts/baltic_stability_sweep.py` + run_calibration records epsilon/ices_loss/stability in save_data (`be07a1b`); T5 `scripts/baltic_stability_certify.py` (`59637c5`). All calib tests pass (398/322), legacy unchanged at eps=inf. **T5 baseline VALIDATED: `--params current` → 0/8 (cod/flounder/perch/pikeperch/smelt collapse, herring/sprat/stickleback OVERSHOOT envelope).**

**KEY exec deviations from plan:** (1) driver optimises single-seed then multi-seed RE-RANKS → no per-call two-factory multiseed (matches driver). (2) sweep smoke = inject fake run_calibration (real calibration too slow: >12min even phase-1, since run_calibration always does 5-seed rerank). (3) T5 `--java` cross-check is a NOTE STUB (full staged 4.4.1 run deferred).

**▶▶ SP-A COMPLETE 2026-07-01 — GATE OUTCOME: parameters alone CANNOT stabilize Baltic in-ICES → SP-B (grid re-resolution) is the binding constraint. (`0d4a655`, branch NOT merged, data/baltic UNTOUCHED.)** T6 ran the full eps-sweep (phase 13 = 39 params: mortality+fishing+recruitment ssbhalf/shape, surrogate-DE, 35yr×3seed, ~5h). **Front: all finite eps (5→0.2) converge to the SAME config — min achievable instability = 8.11, far ABOVE all targets ≤5; reaching for stability WORSENS ICES (baseline ices_loss 2.70 → 5.88).** Cert (50yr×5seed, `docs/baltic_stability_certification_2026-07-01.md`): the least-unstable config gets **7/8 persisting (up from 2/8 baseline) but 0/8 IN-ENVELOPE — species OVERSHOOT ICES by 1-2 OoM instead of collapsing (cod ~63×, perch ~44×, pikeperch ~68×; only sprat collapses).** BOTH Python + Java 4.4.1 agree (sprat collapses, 7 persist). **No bounded middle between collapse and explosion on the coarse grid = the exact percid boom/bust limit, now shown for all 8 species.** Sweep front: `data/baltic/reference/stability_sweep.json`. **NEXT: SP-B (finer Baltic grid) is now evidence-justified.**

_(historical launch detail: detached PID 130857, subprocess `surrogate-de` workers=8, incremental JSON. Each eps ~1.5-2h; sims slower than est. due to numba-thread contention across 8 workers.)_ Output+log OUTSIDE the worktree (persist): `/home/razinka/osmose/osmose-python/.sp-a-t6/stability_sweep.json` (incremental per-eps) + `sweep.log`. **CAVEAT: the process runs FROM the worktree (PYTHONPATH=.) — DO NOT remove the worktree until T6 finishes.** Sweep FIX (`bdd0263`): subprocess calibrate_baltic (native __main__) not importlib — importlib-under-synthetic-name broke loky worker un-pickling (BrokenProcessPool). When done: `scripts/baltic_stability_certify.py --params .sp-a-t6/stability_sweep.json` → write data/baltic or SP-B gate.

**T6 command (for reference / re-launch):** `PYTHONPATH=. .venv/bin/python scripts/baltic_stability_sweep.py --phase <all-8> --epsilons inf 5 2 1 0.5 0.2 --years 35 --seeds 3 --out data/baltic/reference/stability_sweep.json` (days, or HPC) → `scripts/baltic_stability_certify.py --params <sweep.json>` → if all-8 persist+in-envelope: write data/baltic (VALUE round-trip check, NOT native_440_parity) + verify default-nyear healthy; else emit SP-B grid gate. **data/baltic UNTOUCHED until cert passes. Branch NOT merged (user: keep isolated until full 3-sub-project cutover).** Pick the calibration `--phase` that includes the recruitment levers (ssbhalf/relativefecundity/shape) per Phase-0.
