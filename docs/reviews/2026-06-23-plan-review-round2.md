The mortality.py path differs from the cited location, but both defects concern the plan's correctness, not the live tree. Both verdicts confirm low/none severity with correct-by-default behavior. The source confirms the all-lowercase parser key at config.py:646-647 exactly as the defect states. No rework is warranted.

**Verdict:** Execution-ready. Both residual defects are confirmed real but rate low/none severity — each describes a cosmetic mismatch between test intent and operating values (camelCase keys silently falling back to parser defaults) or an inconsequential micro-cost on a non-hot path, and in both cases the plan's behavioral correctness claims hold. No defect blocks execution; the two edits below are optional hygiene improvements to prevent latent fragility.

**Residual edits:**

LOW — Task 3 egg-test config keys are wrong-case (silently ignored)
- Defect: Plan Task 3 Step 1 `_CFG` dict (plan lines 363-364) uses camelCase `predation.predPrey.sizeRatio.min.sp0/sp1` and `.max.sp0/sp1`. The parser at `osmose/engine/config.py:646-647` looks up all-lowercase `predation.predprey.sizeratio.min.sp{i}` / `.max.sp{i}`, so the four keys are discarded and defaults (`r_min=1.0, r_max=3.5`) apply. Predation proceeds only because `3.0 >= 3.5` is False — correct by accident. The comment "ratio 3.0, within [1.0, 1/0.3)" is misleading; with the intended `max=0.3` the guard would block predation and break `test_released_eggs_are_eaten`.
- Plan step: Task 3 Step 1.
- Fix: Lowercase the four keys to `predation.predprey.sizeratio.min.sp0/sp1` and `predation.predprey.sizeratio.max.sp0/sp1`, and set values that make the intent active and match the comment (e.g. `min=1.0`, `max=3.5`) so the test exercises the real guard rather than relying on default fallback. Same latent issue exists at `tests/test_engine_functional_response.py:604-607` (out of scope, but worth a note).

NONE — Task 2 Step 5 helper rebuilds the `targeted` set per school
- Defect: The Step 5 replacement calls `_fleet_effort_factor` for all n schools, and the Step 3 helper rebuilds `targeted: set[int]` per call, versus the original single build before the loops (orig at mortality.py:787-789). Behavior is preserved exactly (helper returns 1.0 for non-targeted → no-op).
- Plan step: Task 2 Steps 3 & 5.
- Fix: None required. `_precompute_effective_rates` runs once per time step, not in a hot kernel, so the cost is immeasurable. Optionally hoist the `targeted` set construction out of the helper if a one-line note is cheap, but no plan change is warranted.

Note: the cited `mortality.py` path does not exist at the repo root (the tree uses `osmose/engine/processes/`); both findings reference plan-internal line numbers and the verified `config.py:646-647`, which I confirmed exactly, so the line-number drift does not affect either verdict.