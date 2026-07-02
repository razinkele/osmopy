---
name: project_predator_functional_response
description: Predator functional-response feature — PR-A (engine) merged to LOCAL master unpushed 2026-06-02; PR-B (calibration) not started
metadata: 
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

Opt-in predator functional response (Holling type-I/II/III) for OSMOSE. Two-PR design (spec `docs/superpowers/specs/2026-05-31-predator-functional-response-design.md`, plan `docs/superpowers/plans/2026-06-01-predator-functional-response-plan.md`).

## PR-A (engine capability) — MERGED + PUSHED to origin/master 2026-06-02
HEAD `e1f4173` (`0b1fc27..e1f4173`); local+origin master synced. 8 design-doc/spec/plan/review commits + 11 implementation commits A0–A9. FF-merged to master then `git push origin master`. Feature branch `feat/predator-functional-response` deleted.

Built via subagent-driven-development (implementer→spec-review→quality-review per task) after 7 rounds of in-loop plan review. **103 FR tests pass; 12/12 Java parity bit-exact with FR off** (the 3 `@_exact_match_local_only` tests run only with `env -u CI` on Python 3.12). ruff check+format clean.

Config keys (default-off, bit-exact when off): `predation.functional.response.shape.sp{i}` (enum type1|type2|type3, default type1) + `…halfsat.sp{i}` (float K [0.1,5.0], required iff shape≠type1). Engine: `EngineConfig.fr_shape` int32[n_total] + `fr_halfsat` float64[n_total]; kernel branch at the single injection point in BOTH `mortality.py` kernels — `eaten_total = max_eatable·min(g_form(r), min(r,1))`, r=total_available/max_eatable, conservation clamp. type-I keeps the verbatim `min(total_available,max_eatable)` line.

## Hard-won findings (carry into PR-B + future)
- **numba=False toggle**: monkeypatch `osmose.engine.processes.mortality._HAS_NUMBA = False` to route to the Python fallback `_apply_predation_for_school`; else the engine always uses the numba kernel. Test helper `_run_short_sim(numba=...)` in `tests/test_engine_functional_response.py` does this.
- **Minimal config has ALL schools unlocated** (cell_x=-1) → predation kernel never runs → FR behavior tests are VACUOUS there. FR behavior/parity tests MUST use the Baltic config (`background=True`).
- **numba vs pure-Python full-sim diverge ~98%** even with FR OFF (different RNG-consumption paths) — documented pre-existing behavior. So cross-backend parity is tested at the KERNEL level (drive both kernels on identical inputs), not full-sim. For FR effect, compare same-backend FR-on vs FR-off.
- **"prey-only species is inert under FR" is IMPRECISE**: FR applies wherever a species predates, INCLUDING eating resources (plankton/benthos). No Baltic focal species is truly inert (all eat resources). Validation ACCEPTS a non-type1 shape on any species; runtime effect occurs only where it's a predator.
- `_FR_SHAPE_CODE`/`_FR_HALFSAT_SENTINEL` are DUPLICATED in `config.py` and `background.py` (importing would be circular: config.py imports background.py at module level). A parity test guards drift.
- `shepherd_beta` (the template FR mirrored) flows through 4 config.py layers: focal-dict (~:562) → `_repro` unpack (~:1539) → `_focal` rebuild (~:1601) → `_merge_focal_background` concat (~:776/:826). Background arrays built inline `np.array([b.x for b in background_list])` at ~:735.
- New required `EngineConfig` fields break `tests/test_engine_config_validation.py::_minimal_config` (direct `EngineConfig(**cfg)`); fixture must be updated. Only direct-construction site in the repo.
- `aggregate_diet_all_predators(diet_matrix, species_id, n_total)` added to `osmose/engine/output.py` (background-inclusive; `aggregate_diet_by_species` excludes background slots via focal_mask). Resource-prey diet columns start at `n_species` (=8 Baltic); production diet width is hardwired to n_species+n_background (=10) at `simulate.py:1436` → diagnostic needs width 16 via monkeypatching `enable_diet_tracking`.

## PR-B (Baltic phase-14 calibration + science) — MERGED + PUSHED to origin/master 2026-06-02→03 (HEAD `f04b2c2`)
6 commits (`e1f4173..f04b2c2`), scripts + tests + verdict doc only (no engine changes → no parity risk). Built via subagent-driven-development (PB1–PB4), each piece reviewed; final holistic code review = merge-ready. 121 FR+calibration tests pass.

**Verdict (`docs/baltic_fr_calibration_2026-06-02.md`): FR mechanism VALIDATED decisively, but NOT claimed as a Baltic calibration improvement.**
- Phase-14 (4 K's, type-III, frozen reconstructed phase-13 base, 720 evals/6.43h): obj 3.386±0.0155. Calibrated K: cod 4.98 (near upper bound = STRONG suppression), pikeperch 0.53 (near lower = near-neutral), GreySeal 1.62, Cormorant 3.95. **K direction: type-III g(r)=r²/(r²+K²) → K→small ≈ type-I (no refuge); K→LARGE = strong intake suppression.**
- FR-on vs FR-off (same base, 40y): cod overshoot ×33→×17.5 (halved); flounder recovered from EXTINCTION (×0→×0.37); percids worsened (perch ×166→×190, pikeperch ×127→×205); **strict ICES in-range UNCHANGED 1/8→1/8**.
- Process diagnostic (3 seeds, 40y, width-16): cod realized predation drops on ALL 8 prey (sprat −0.52±0.019, all >2σ); GreySeal→flounder −0.44, Cormorant→flounder −0.24 (the flounder-recovery mechanism); **28/32 pairs beyond multi-seed noise band**. Objective doesn't regress (FR-on ≤ FR-off by DE construction).
- **Why "not a claimed improvement": (1) approximate phase-13 base does NOT reproduce the documented 2.133 equilibrium (biomasses 2–3× over doc), so not comparable to shipped Shepherd; (2) strict in-range count unchanged — helped species (cod, flounder) don't cross thresholds, worsened ones are low-weight grid-under-resolved percids (the PR #50 spatial limit).**
- Ships as engine capability (PR-A) + documented exploration + reusable tooling: `--phase 14` in calibrate_baltic.py, `--mode shepherd-fr` in evaluate_calibration_vs_ices.py, `scripts/fr_process_diagnostic.py`, `scripts/reconstruct_phase13_results.py`.

**Predator FR feature line is now CLOSED (both PRs shipped).** Clean follow-up if revisiting: regenerate an EXACT phase-13 base (fresh ~4.5h phase-13 run, or commit phase-13's full 39-param result to a TRACKED path — calibration_results/ is gitignored), then re-run phase-14 + diagnostic to answer "FR vs the *validated* Baltic calibration."

### PR-B gotchas (carry forward)
- **Phase-14 result JSON is NOT self-sufficient for eval**: it holds ONLY the 4 free K's (the 39 frozen params live in base_config). To eval/diagnose, MERGE phase13_results.json (40 params) + phase14's 4 halfsat K's, else `from_dict` raises "ssbhalf.sp1 must be > 0 when type=shepherd". (Other phases store their full free set, so eval works directly — phase-14 is the exception.)
- **Running scripts directly needs `PYTHONPATH=<repo root>`**: `.venv/bin/python scripts/X.py` puts `scripts/` on sys.path, not the repo root, and osmose is NOT editable-installed → `ModuleNotFoundError: osmose`. Tests work (pytest adds rootdir); direct script runs don't. Prefix `PYTHONPATH=/home/razinka/osmose/osmose-python`.
- `data/baltic/calibration_results/` is **gitignored** (.gitignore:70) — no result JSON is ever tracked. That's why PR #50's phase-13 result was lost. `scripts/reconstruct_phase13_results.py` regenerates the (approximate) base on demand.

## (historical) PR-B was NOT STARTED at PR-A merge
Plan Part B (tasks B1–B6). Prereqs/steps: commit PR #50 phase-13 Shepherd result as `data/baltic/calibration_results/phase13_results.json`; `get_phase14_params()` returning `(keys,bounds,x0)` tuple (NOT a dict) + `phase=="14"` branch in `scripts/calibrate_baltic.py` using the **phase-2** freeze template (freeze 39 params, free 4 K); FR on cod(sp0)/pikeperch(sp5)/GreySeal(sp14→slot8)/Cormorant(sp15→slot9), type-III fixed, DE bound [0.5,5.0]; `--mode shepherd-fr` in `scripts/evaluate_calibration_vs_ices.py` (`make_objective` is at `calibrate_baltic.py:260`, sibling import); FR-on/FR-off process diagnostic at width 16 with a per-predator-prey mortality-delta noise band (mean±std across the same multi-seed set — NOT the objective std). Multi-hour DE run; honest go/no-go disposition (ship as Baltic improvement only if objective doesn't regress AND a predator's mortality reduction exceeds the noise band). See [[project_density_dependent_recruitment]] (PR #50 Shepherd is the phase-13 base).
