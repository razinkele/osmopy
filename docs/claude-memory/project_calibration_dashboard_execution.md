---
name: Calibration dashboard execution state (mid-flight)
description: Active subagent-driven execution of the calibration progress dashboard plan; 14/22 tasks complete on branch docs/spec-calibration-dashboard
type: project
originSessionId: 99666016-a3d1-4898-a6db-29c013e595aa
---
**Status as of 2026-05-16:** Mid-execution of the calibration progress dashboard feature. 14/22 implementation tasks complete. Continue in next session.

## Branch + commits
- Branch: `docs/spec-calibration-dashboard` (off origin/master ~`cd590c1`)
- HEAD: `fe97352` (T14 surrogate-DE checkpoint hook)
- Baseline commit `95039b2` — design spec (851 lines, 10 review rounds) + implementation plan (4,269 lines, 11 review rounds) committed before T1 dispatch
- One commit per task; no amends

## Done (T1-T14)
- **Phase 1** (T1-T7): `osmose/calibration/checkpoint.py` — `MAX_CHECKPOINT_BYTES`, `default_results_dir`, `RESULTS_DIR` (single source of truth), `CalibrationCheckpoint` frozen dataclass (16 `__post_init__` invariants — 14 from plan + 2 added by T2 code review for param_keys uniqueness and banded_loss-requires-banded_targets), `CheckpointReadResult` 4-kind discriminated union, `write_checkpoint` (atomic tmp+rename, numpy coercion, allow_nan=False), `read_checkpoint` (never raises; ok/no_run/partial/corrupt with enriched error_summary), `is_live`/`probe_writable`/`liveness_state`, `LiveSnapshot`.
- **Phase 2** (T8): `tests/conftest.py` fixtures — `tmp_results_dir` (patches 3 RESULTS_DIR locations with raising=False), `synthetic_two_species_targets`, `synthetic_stats_in_band`/`_sp_b_out_of_band`.
- **Phase 3** (T9): `_ObjectiveWrapper` in `scripts/calibrate_baltic.py` captures `last_per_species_residuals: list[tuple[str, float, float]] | None`. `__call__` iterates `self.targets`-derived species (was module `SPECIES_NAMES`); new `_simulate_and_compute_stats` seam. Load-bearing assign-at-end invariant.
- **Phase 4** (T10): `make_banded_objective` in `osmose/calibration/losses.py` returns `(callable, residuals_accessor)` tuple. Updated 1 production caller + 6 existing test callers.
- **Phase 5** (T11-T12): `_make_checkpoint_callback` extended with new kwargs (all defaulted; backward compat with 14 existing test call sites). `_write_progress_checkpoint` added to `osmose/calibration/checkpoint.py` (S1 layering — must not import from scripts/). DE save_run wiring at completion. `_save_run_safe` + `_save_run_fallback` centralised in `osmose/calibration/history.py` (S2). Thin `_save_run_for_de` wrapper in scripts. Tempfile.gettempdir() fallback with O_EXCL | 0o600.
- **Phase 6** (T13-T14): CMA-ES `run_cmaes` and surrogate-DE `surrogate_assisted_de` per-generation checkpoint hooks with triple-guard. Used existing parameter names (`n_iterations`/`n_initial`) not plan's (`max_outer_iter`/`n_init`) to preserve 7 existing callers.

## Test counts
- 2844 passed, 16 skipped, 41 deselected, 0 failed (full suite, excl. Java parity)
- 60+ tests in `tests/test_calibration_checkpoint.py`; 4 in `tests/test_runner_checkpoints.py`; 2 in `tests/test_history_wiring.py`; 8 in `tests/test_objective_evaluator_residuals.py`

## Remaining (T15-T22)
1. **T13+T14 review pending** — was mid-flight when rate-limited. `git diff 167e6f3..fe97352`. Apply two-stage review (spec compliance + code quality). Then `git -C` from working dir `/home/razinka/osmose/osmose-python`.
2. **T15 NSGA-II** (most complex remaining). Closure plumbing across 3 sites in `ui/pages/calibration_handlers.py`: declare `_shared_banded_residuals_accessor` + `_shared_banded_targets_dict` near line 487-499; assign in `handle_start_cal` ~line 742-746; consume in `_start_optimization_with_params` (`nonlocal`) at the existing `_make_progress_callback(...)` caller ~line 672. Plan section starts at line 2697.
3. **T16-T21 UI work** — Phase 8. Plan sections from line 2862. Load-bearing scoping note: `@output`/`@reactive.*` decorators must live INSIDE `register_calibration_handlers` (line 418), pure helpers at module scope. Printed code blocks use 0-indent; engineer adds +4 spaces when transcribing.
4. **T22** — manual UI smoke checklist; documentation only.

## Execution playbook
- Skill in use: `superpowers:subagent-driven-development`. Continuous execution; don't pause for check-ins between tasks.
- Per task: dispatch implementer (general-purpose), then spec reviewer (or combined spec+code), then code quality reviewer, then `TaskUpdate` to completed. Implementer fixes any reviewer findings.
- Implementer subagents have made sound deviations that reviewers ratified: `_ObjectiveWrapper` iteration source, scipy `functools.wraps` for callback signature dispatch, dict-shaped DE result (not OptimizeResult), surrogate-DE existing param names, T5 hard-coded `corrupt` for invariant violations (the plan's `partial` branch contradicted both the test and §5 spec).
- T2 code reviewer caught 2 real invariant gaps (param_keys uniqueness, banded_loss-without-targets); T11 reviewer caught scipy callback signature dispatch bug. Worth running code-quality reviewer per task.

## Gotchas
- `osmose/calibration/__init__.py` re-exports `make_banded_objective`. Its return shape changed to a tuple in T10 with no shim (per spec S2 decision). External callers (none known) would get a tuple.
- `_save_run_safe` reads `hist_mod.HISTORY_DIR` at call time, not via default arg. `save_run` (the underlying call) still captures it as a default, so the wrapper passes `history_dir=hist_mod.HISTORY_DIR` explicitly.
- `_write_progress_checkpoint` lives in `osmose/calibration/checkpoint.py` (the S1 layering fix); CMA-ES and surrogate-DE runners import it from there, NOT from `scripts/calibrate_baltic.py`.
- `tmp_results_dir` fixture patches three locations with `raising=False` since `ui.pages.calibration_handlers.RESULTS_DIR` doesn't exist yet (T16 will add it).
