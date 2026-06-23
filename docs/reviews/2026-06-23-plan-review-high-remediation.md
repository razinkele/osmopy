## Verdict

**Needs-rework** before execution. The plan's design and source-citation work is fundamentally sound — every targeted code site is real and the fix logic is correct — but Task 3's TDD red-phase test stub is broken in three independent, compounding ways (wrong class import, nonexistent `ResourceState.empty()`, wrong `mortality()` signature + invalid `n_subdt` kwarg) so the falsifiability gate is vacuous as written. Task 4's Java cross-check has four correctness blockers (wrong tolerance reference, suite-isolation failure, wrong output prefix, missing baseline base-path) plus an unupdated direct-call test (`test_engine_functional_response.py:780`) that will break the suite. None of these touch the underlying fix design; all are mechanically fixable in the plan text before any code is written.

## Required plan edits

### High

**1. Task 3 Step 1 test stub — three broken API calls (consolidates 6 duplicate findings).** The egg-retention test stub (plan lines ~345–361) cannot run; Step 2's "verify it fails first" would crash at import/call, not at the assertion, making the entire Task 3 red-phase vacuous. Fix all three:
- Line 345: `from tests.test_engine_mortality_loop import TestPredation` → `import TestUnifiedPredation` (class defined at `test_engine_mortality_loop.py:131`; also update the `TestPredation()._make_2sp_config()` call at line 349 and the template reference at line 355).
- Line 359: `ResourceState.empty(grid, cfg)` → `ResourceState(config=cfg.raw_config, grid=grid)` (`resources.py:49` has only `__init__`; pattern per `test_engine_mortality_loop.py:185`).
- Line 361: `mortality(state, cfg, grid, resources, n_subdt=4, step=0, rng=rng)` → `mortality(state, resources, cfg, rng, grid, step=0)` (real signature `mortality(state, resources, config, rng, grid, step=0, ...)` at `mortality.py:1759`). Drop `n_subdt`; to set it, add `"mortality.subdt": "4"` to the config dict (read internally at `mortality.py:1782`).

**2. Task 3 — `test_engine_functional_response.py:780` direct-call test not updated for the new `egg_retained` parameter.** Step 3 adds `egg_retained` as the final positional param to `_apply_predation_numba` (40→41 params); the existing direct `@njit` call at `test_engine_functional_response.py:780` passes only 40 args and will fail at JIT compile. The plan never names this file. **Edit:** add a step to update that call (pass a zero-initialized `egg_retained` array of the right length) AND add `tests/test_engine_functional_response.py` to the Task 3 Step 8 regression list so the breakage is caught before final verification, not at the end.

**3. Task 4 Step 1 — wrong tolerance reference for the Java cross-check.** Plan says "reuse the tolerance constant/asserts from `test_engine_parity.py`'s statistical/portable section," but that section is a Python-vs-Python 5% check (`rtol=0.05, atol=1.0`), far too tight for cross-engine and not the "1 OoM" band the plan's own prose names. **Edit:** reference `tests/test_calibration_problem_python_engine.py:135` (`0.1 <= py_val/java_val <= 10.0` ratio check) as the tolerance model.

**4. Task 4 Step 1 — Java parity test is not isolated from the default suite.** `pytestmark = pytest.mark.slow` provides no exclusion: `addopts` (`pyproject.toml:103`) is `-m 'not e2e and not visual'` (no `slow`), `slow` is unregistered (`pyproject.toml:95-98`), and the JAR is present so `skipif(_JAR is None)` → `skipif(False)`. The full EEC Java subprocess would therefore run on every default `pytest`/CI invocation. **Edit:** replace the `_JAR is None` skip guard with the repo's env-var pattern `@pytest.mark.skipif(not os.environ.get("OSMOSE_JAR"), ...)` (per `test_calibration_problem_python_engine.py:113-116`), AND register `slow` in `pyproject.toml` markers. Also note the Final-Verification `-m "not slow"` (plan line 518) overrides `addopts` and would re-admit e2e/visual — change it to `-m 'not e2e and not visual and not slow'`.

**5. Task 4 Step 1 — wrong `OsmoseResults` prefix for EEC.** EEC sets `output.file.prefix=eec` (`eec_param-output.csv:2`); `OsmoseResults.__init__` defaults to `prefix="osm"`, so `OsmoseResults(output_dir)` finds no files (glob `f"{prefix}_{type}*.csv"`, `results.py:658`) → empty/`FileNotFoundError`. **Edit:** the pseudocode must read `OsmoseResults(output_dir, prefix="eec")`.

### Medium

**6. Task 3 Step 1 — appetite/field setup placeholders are not trivially fillable from the cited template.** The two `...` blocks plus the cited template (`TestUnifiedPredation.test_total_eaten_never_exceeds_max_eatable`, line 183) won't produce predation: with `length=[40,10]`, `sizeRatio.min=1.0`, `sizeRatio.max=0.3`, the config parser swaps min/max (`config.py:662-669`) → `r_max=1.0`, ratio `4.0 >= 1.0` → prey skipped → `base=0`, so `assert base > 0` fails for the wrong reason. **Edit:** Step 1 must give concrete predator size/length/weight + appetite values that actually clear the egg cohort in one sub-dt (or explicitly state the size-ratio constraint the executor must satisfy), rather than "copy the template and increase appetite."

**7. Task 4 Step 3-4 — EEC baseline base-path underspecified.** `save_parity_baseline.py` hardcodes the `data/examples/` base for NetCDF/grid lookups; EEC's grid (`eec_grid-mask.nc`) lives under `data/eec_full/`. "Mirror the existing `save_baseline` body" will produce a `FileNotFoundError` unless the base dir is parameterized too. **Edit:** Step 3 must state explicitly that the EEC variant changes the base directory from `data/examples` to `data/eec_full` for grid/NetCDF resolution, not just the config CSV path.

### Low

**8. Wrong/off-by-8 line numbers (consolidates 4 findings).** Descriptions are correct; line numbers are off. Correct them so "jump-to-line" lands right:
- Task 2 Step 5: fleet block is `786-802`, not `778-794`.
- Task 2 Step 7: `_apply_fishing_for_school` call is at `1745` (FISHING branch starts 1740), not `1737`.
- Task 3 Step 3: `abd_q = inst_abd[q_idx]` is at `894`, not `886`.
- Task 3 Step 4 driver defs: `_mortality_in_cell_numba` `1077` (not 1069), `_mortality_all_cells_numba` `1219` (not 1211), `_mortality_all_cells_parallel` `1386` (not 1377). Call-site numbers (`1139/1306/1485`) are correct.

**9. Task 3 Step 5 — misleading "forward to `_mortality_in_cell_numba`" rationale.** From `mortality()`'s `else:` branch (`mortality.py:1931/1991`), `_mortality_in_cell`'s internal `use_full_numba` is always False (`1623`), so the `_mortality_in_cell_numba` forward is dead code. The real Python-fallback fix is Step 6 (`_apply_predation_for_school` via `state.egg_retained`, no new param). **Edit:** rephrase Step 5 to drop the dead-forwarding rationale and explicitly mark Step 6 as the active Python-path fix so the executor doesn't add an arg-count-mismatch risk or skip Step 6.

**10. Task 2 Step 8 — mixed `SchoolState` mutation idiom.** Direct `state.abundance[0] = ...` on the frozen dataclass works but diverges from the repo-wide `state.replace(...)` idiom and would silently break if arrays become read-only. **Edit (optional):** use `state = state.replace(...)` for consistency.

## Execution watch-items

- **JAR / Java availability (Task 4).** The cross-check depends on `osmose-java/osmose_4.3.3-jar-with-dependencies.jar` being present and Java being runnable; gate the test on the `OSMOSE_JAR` env var (edit #4) so it stays opt-in. The EEC Java run is multi-minute — keep it out of the default and CI suites.
- **Appetite-tuning the egg test (Task 3, edit #6).** Even after the API fixes, the executor must empirically tune predator size/length/appetite so the *buggy* code wipes the cohort (`base > 0`, survivors ~0) and the *fixed* code retains it — verify the red phase shows the semantic failure, not a TypeError or a `base==0` skip.
- **Baseline-staleness window (Task 3 Step 9 → Task 4 Steps 4-5).** The egg-retention fix shifts BoB biomass, so the bit-exact `TestBaselineParity` tests (`@_exact_match_local_only`, run on local CPython 3.12) will fail on the branch between the Task 3 commit and Task 4 baseline regeneration. Executor should either squash, defer the Task 3 commit until baselines are regenerated, or accept a known-broken intermediate commit — and must confirm the regenerated `parity_baseline_bob_1yr_seed42.npz` reflects the post-fix outputs before final verification.
- **EEC baseline NetCDF/grid paths (Task 4, edit #7).** Watch for `FileNotFoundError` on first `--config eec` run; confirm grid/background-species NetCDFs resolve under `data/eec_full/`.