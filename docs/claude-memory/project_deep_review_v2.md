---
name: Deep Review V2 — Fix Plan Status
description: Status of the 2026-04-05 deep review fix plan — Phase 1 complete on branch fix/deep-review-v2-2026-04-11, 16 real tasks remaining in Phases 2-5 after drift audit
type: project
originSessionId: 077a9827-9ad8-4127-8ebe-28935f931333
---
## Deep Review V2 — Status as of 2026-04-11

**Plan file:** `docs/superpowers/plans/2026-04-05-deep-review-fixes-plan.md`
**ALL PHASES COMPLETE (Phase 1, 2, 3, 4, 5) and merged to local master (2026-04-11).** All three feature branches fast-forwarded and deleted. Master is **25 commits ahead of origin/master** — not yet pushed. Final state: **2098 tests passed** / 15 skipped / 0 failed, ruff clean, parity 12/12 bit-exact.

**Phase 1 commits (4):** b9ce68e (C5), 083ae5f+0f6f12a (C6), 96925fd (C7).
**Phase 2 commits (13):** f05c2fb (H1), 2dda4c4 (H3), c4cce5a+2d3596a (H4), d6d42ec+53b5690 (H5), 0b4407f (H6), f7e9319+4c8c622 (H7), 616efeb (H8), 6179e8d (H10+H11), adcbb38 (H12+H13), 3f981c0 (H14).
**Phase 3-5 commits (8):** a7b59e6 (plan doc), 55349dd+4d80db1 (M2 + partial-year followup), 1cdb77a (M7), de264ed (M10), 7c5af1a (M6), 00ff908 (H18 coverage tests), ea8e5c3 (M13 vectorize temp).

**Follow-up commits pattern (4 total):** C6, H4, H5, H7, M2 each produced a review-driven follow-up because the plan-specified minimal fixes left Important quality gaps — test strength (H4, H5), value-pin (C6), asymmetric case (H7), partial-year tail (M2). Pattern: plan-specified minimal fix → reviewer flags gap → one-commit targeted follow-up on the same branch.

**Plan for Phase 3-5 was written this session** via `superpowers:writing-plans` with 3 review loops (feature-dev:code-reviewer line-level, feature-dev:code-architect cross-task, general-purpose targeted probe on M13 numerical identity). Plan doc saved at `docs/superpowers/plans/2026-04-11-deep-review-v2-phase3-5-plan.md` (committed a7b59e6).

### Critical finding — plan drift

The v1 remediation branch (`fix/deep-review-remediation`) was merged into master and shipped as v0.6.0. The v2 plan document was NOT updated, so 11 of the 27 tasks are already-fixed. **Always spot-check task targets before dispatching implementer subagents.**

### Phase 1 commits on branch

- `b9ce68e` — C5: add `@reactive.effect _consume_cal_poll` in calibration_handlers.py
- `083ae5f` — C6: replace tautological W-L test with property-based assertions
- `0f6f12a` — C6 follow-up: add `math.pow` value pin, tighten `max_weight` bound strict, fix docstring/comment
- `96925fd` — C7: explicit NotImplementedError in JavaEngine.run_ensemble

### Drift audit — full 27-task status

**Phase 1 — 3 real / 7 tasks (all shipped this session):**
- C1, C2, C4, C3 — stale (absorbed into v0.6.0 via architecture refactor and guard additions)
- C5, C6, C7 — real, fixed this session
- C3 note: phi_t handles `e_d==e_m` via silent Arrhenius fallback, not ValueError as plan wanted. Semantic choice; left alone.

**Phase 2 — COMPLETE (10 real tasks shipped). Stale (2):** H16 (setup_logging already has `if not logger.handlers` guard), H17 (diet tracking refactored to context — no global state to tear down).

**Phase 3 — 4 real / 5 tasks (not started):**
- Real: M2 (config.py:433-437 spawning uses total-sum not per-year), M7 (reader.py:32 read() doesn't reset key_case_map), M10 (scenarios.py:185 no ZIP entry size guard), M6 (scenarios.py import_all has path-escape check but no try/except ValueError around Scenario ctor)
- Stale: M8 (problem.py already uses `with OsmoseResults(output_dir, strict=False) as results`)

**Phase 4 — 1 real / 2 tasks (not started):**
- Real: H18 (neither test_simulate_output_step0_include nor test_simulate_partial_flush exists)
- Stale: Task 26 diet-in-mortality test references `_pred_mod._diet_matrix` which does not exist post-C1 refactor

**Phase 5 — 1 real / 1 task (not started):**
- Real: M13 (simulate.py:250, 286 still use scalar `temp_data.get_value(...)` in list comprehensions). Verify `temp_data.get_grid(step)` method exists before dispatch.

**Remaining real work: NONE.** All 27 plan tasks are either shipped (16 real) or documented as stale/obsolete (11 dropped). The 2026-04-05 deep review v2 plan is fully resolved.

## How to apply

- Continue on branch `fix/deep-review-v2-2026-04-11` in a new session
- For each Phase 2-5 task, grep the target file BEFORE dispatching to confirm the bug still exists
- Phase 2 contains 5 security-cluster tasks (H10, H11, H12, H13, M10) — reasonable cherry-pick if user wants a partial follow-up session
- Engine-touching tasks (H1, H3, H4, H5, M2, M13) need `migration-check` skill run after each phase

## Why

Deep review v2 was planned as a bundled hardening pass. Shipping Phase 1 in isolation is reasonable because those tasks don't touch engine execution paths and carry zero Java-parity risk. Phases 2-5 can be run as one or more follow-up sessions.
