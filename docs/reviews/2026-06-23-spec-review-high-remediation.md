# Deep Review: Bug-Fix Remediation Spec

## 1. Verdict

**Needs-rework.** The spec's core engineering analysis is sound and all three fixes address genuine bugs, but Fix 1's central threading instruction is *inverted* (it points the implementer at `_mortality_in_cell`, which is dead code on every Numba-enabled install — the real hot path is `mortality()` → `_mortality_all_cells_numba`/`_parallel` → `_apply_predation_numba`). Following the spec literally yields a silent no-op egg-retention fix in production *and* test suites that pass green before any fix is applied (Fix 2 and Fix 3 tests are vacuous under standard CI). These are correctness-defeating gaps in the spec's guidance, not the fix designs, so a focused rewrite of the three threading/test sections — plus the parity-gate clarification — makes it plan-ready.

## 2. Required spec edits

### HIGH

**H1 — Fix 1 threads `egg_retained` from the wrong function (dead code); the live Numba path is missed.** *(§Fix 1 / "Numba predation", spec lines ~83-90; mortality.py:1923-1982 vs 1564-1690)*
The spec says thread `egg_retained` "down from `_mortality_in_cell` … through every Numba kernel," citing `_mortality_in_cell_numba` (:1069). But `_mortality_in_cell` is only reached in the `else` branch at `mortality.py:1983` (fires when `_HAS_NUMBA=False`), and `_mortality_in_cell_numba` is then unreachable. The production dispatch is `mortality.py:1938`: `_mortality_all_cells_numba` / `_mortality_all_cells_parallel` called directly.
**Correction:** Rewrite the threading source to start at `mortality()` itself, where `work_state.egg_retained` is current after the per-sub-dt update (line 1920). Add `egg_retained` as a parameter to **both** `_mortality_all_cells_numba` (sig ~:1211) and `_mortality_all_cells_parallel` (sig ~:1377), pass it at the `_batch_fn(...)` call site (line ~1939), and forward it into `_apply_predation_numba` at the two internal call sites (~:1298, ~:1477). Add these three functions + the call site to "Files touched." Mark `_mortality_in_cell` / `_mortality_in_cell_numba` as the no-Numba fallback (still needs the param for that path, but is not the hot path). *(Consolidates four duplicate findings flagged critical/high — corrected severity high.)*

**H2 — Fix 2 test is vacuous in any Numba-enabled environment.** *(§Fix 2 / Test; mortality.py:1923)*
"Drive the Python path (set `bioen_enabled=True` or otherwise force the fallback)" does not work: the outer dispatch at `mortality.py:1923` (`if _HAS_NUMBA and len(valid_indices) > 0:`) has **no** `bioen_enabled` clause, so with Numba installed (CI included) execution routes through the batch path and `_apply_fishing_for_school` is never called. The test passes red-green-identical.
**Correction:** Specify the exact mechanism already used at `tests/test_engine_functional_response.py:207`: `mock.patch("osmose.engine.processes.mortality._HAS_NUMBA", False)`. State that the test must assert the fix fails first (fleet-effort absent) under that patch.

**H3 — Fix 2 problem statement mis-states the fallback's gating condition.** *(§Problem / Finding 2, spec lines ~28-31; §Fix 2 Change line ~139)*
The spec says the buggy Python path is "taken whenever `bioen_enabled` or Numba is absent." The `bioen_enabled` gate (lines 1615-1620) lives *inside* `_mortality_in_cell`, which is only reached when `_HAS_NUMBA=False`. The fleet-effort gap is a **no-Numba-only** issue, not a bioen issue.
**Correction:** Replace "whenever `bioen_enabled` or Numba is absent" with "only when Numba is absent (`_HAS_NUMBA=False`); `bioen_enabled` does not affect the outer dispatch." Downgrade the stated production impact accordingly.

### MEDIUM

**M1 — Fix 1 test assertion is non-falsifiable.** *(§Fix 1 / Test)*
Post-run `egg_retained` is 0 (all sub-dts released), so "surviving abundance ≥ retained fraction at sub-dt 1" reduces to `≥ 0`, trivially true in both buggy and fixed code given the abundance-clamp at line 2024.
**Correction:** Either (a) capture `inst_abd` mid-loop after exactly sub-dt 1 and assert deaths ≤ `abundance/n_subdt`; or (b) configure a predator whose appetite lies strictly between `abundance/n_subdt` and `abundance`, so the buggy path drives survivors to 0 at sub-dt 1 while the fix leaves survivors. State which.

**M2 — Fix 3 test omits `_WORKER_PROBLEM` setup; un-buildable as written.** *(§Fix 3 / Test; problem.py:68, 95)*
`_worker_eval` asserts `_WORKER_PROBLEM is not None` (line 95) before any logic. A direct-call test with "a stub problem whose objective raises a TypeError" hits `AssertionError`, not the intended path.
**Correction:** Direct-patch the module global: `monkeypatch.setattr("osmose.calibration.problem._WORKER_PROBLEM", stub)` where `stub` is a `MagicMock(n_obj=1)` whose `_evaluate_candidate.side_effect = TypeError(...)`. State that the TypeError must originate at/after `_evaluate_candidate` (it already swallows `_python_engine_errors` internally).

### LOW

**L1 — Fix 3's proposed `except _python_engine_errors` clause is dead code; description is wrong.** *(§Fix 3 / Change, spec lines ~162-169; problem.py:264-268)*
`_evaluate_candidate` already catches `_python_engine_errors` and returns `[inf]*n_obj`; those never reach `_worker_eval`. The proposed catch is unreachable, and the spec's second test case ("ValueError still returns `[inf]*n_obj`") passes before and after the fix.
**Correction:** State the real intent — let unexpected errors (TypeError/AttributeError) propagate. Simplest equivalent: remove the `except Exception:` block entirely (or `except BaseException: raise`). Drop or rewrite the vacuous second test case; note `_evaluate_candidate` is the actual `[inf]` source.

**L2 — Fix 3 docstring left false after the fix.** *(§Fix 3 / Change; problem.py:94)*
`"""…never raise into the pool."""` contradicts the new contract. Add an explicit instruction to update it (e.g. "raises for unexpected programming errors; returns `[inf]*n_obj` only via `_evaluate_candidate`").

**L3 — Fix 1 over-specifies a new parameter on `_apply_predation_for_school`.** *(§Fix 1 / Python predation, spec lines ~76-81; mortality.py:343, state.py:78)*
The function already receives `state: SchoolState`, and `state.egg_retained` is current at call time. No new parameter is needed for the Python path — read `state.egg_retained[q_idx]`.
**Correction:** Replace "add a parameter" with "read the existing `state.egg_retained` field"; avoids API churn and a second source of truth.

**L4 — Cell-lookup cross-reference off by one.** *(§Risks (Fix 2) and §Fix 2 Change docstring; mortality.py:786 vs 787)*
Line 786 is `if sp_id in targeted_species:`; the `cy, cx = work_state.cell_y[i], work_state.cell_x[i]` assignment is line 787.
**Correction:** Change `:786` → `:787` (or cite `:786-787`).

**L5 — Fleet-effort block range truncated (778-790 vs 778-794).** *(§Problem Finding 2 line ~26 and §Fix 2 Change opening line ~118)*
`778-790` covers only the `effort_factor` computation; the actual `eff_fishing[i] *= effort_factor[i]` application is at 792-794. The helper docstring and Fix 2 bullet already use the correct `778-794`.
**Correction:** Update both `778-790` citations to `778-794` for internal consistency.

**L6 — Undocumented per-sub-dt release approximation (fixed-quantum).** *(§Fix 1 / Model, spec lines ~55-61; mortality.py:1916)*
The release amount `work_state.abundance / n_subdt` uses the **original** step-start abundance, not the declining `inst_abd`; `egg_retained` can hit the 0-clamp early under heavy initial deaths. The fix is still correct (the `max(0,…)` clamp is safe), but the spec presents non-egg behaviour as "identically unchanged" without acknowledging this pre-existing approximation.
**Correction:** Add one sentence to §Model noting the release quantum is fixed at step-start abundance (pre-existing behaviour, not introduced here) so the parity gate's results are interpretable.

## 3. Open questions for the human

1. **Parity gate scope (blocking).** §Parity gate / §Testing claim "12/12 EEC + 8/8 BoB" and point at `tests/test_engine_parity.py`, but that file is **BoB-only, Python-vs-stored-baseline** (`baselines/parity_baseline_bob_1yr_seed42.npz`) — no EEC cases, no Python-vs-Java cross-check. CLAUDE.md cites "14/14 EEC" elsewhere, so the "12/12" number is also stale. **Decision needed:** (a) is the gate genuinely BoB-baseline-only (then fix the spec's wording and drop the EEC claim), or (b) must EEC and/or a Java cross-check be added before this fix can land? Fix 1 changes egg predation in *all* configs, so an EEC-side check may be wanted.

2. **Fleet out-of-bounds / no-Numba-fallback priority.** Given H3 — the fleet-effort bug only manifests when `_HAS_NUMBA=False` (an uncommon production configuration) — is fixing it still in scope for this remediation, or should it be re-prioritized? If kept, confirm the regression test should run under forced `_HAS_NUMBA=False` (per H2) rather than attempting to reproduce in the default Numba path.

3. **Fix 1 fallback completeness.** Should the no-Numba Python fallback (`_mortality_in_cell` / `_mortality_in_cell_numba`) also receive the egg-retention fix for behavioural parity with the Numba path, or is it acceptable to fix only the live hot path and leave the rarely-exercised fallback as a documented follow-up? (Affects how much of the dead-code chain the implementer must touch.)