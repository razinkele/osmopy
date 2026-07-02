---
name: project-high-findings-remediation-2026-06
description: "Deep-review high-findings remediation shipped 2026-06-24 (master, pushed) — bioen double-starvation critical + 3 high fixes, Java-validated egg-retention."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9b0fdf28-14de-4aef-af1e-c62c175d61a7
---

# Deep-review high-findings remediation — shipped 2026-06-24

Merged to master `668021f` (+ recalibration `f2e1834`), **pushed to origin**. Driven from a workflow deep-review of v1.0.0 (`docs/reviews/2026-06-22-deep-review-v1.0.0.md`): 1 critical + 3 high, all fixed; medium/low + the 14-nonlocal calibration refactor deferred.

**Critical — bioen double-starvation** (`457ac55` fallback + `c273c8f` production). See [[reference-engine-mortality-dispatch]] for the both-paths nuance (this is the worked example).

**3 highs** (branch `fix/high-findings-remediation`, subagent-driven, 4 TDD tasks):
- **Fix 1 egg-retention** (`94f1bfb`): predation now gated on the released egg fraction — eatable egg prey = `max(inst_abd[q] - egg_retained[q], 0)`, threaded through `_apply_predation_numba` + 3 drivers + dispatch + the Python `_apply_predation_for_school` (reads `state.egg_retained`). `egg_retained = np.where(is_egg, abundance, 0)` so the clamp is a **no-op for non-egg prey**; deaths still decrement full `inst_abd`.
- **Fix 2 fleet-effort** (`41337c9`): shared `_fleet_effort_factor` applied on the pure-Python fishing fallback (was Numba-path-only). `_HAS_NUMBA=False`-only bug, low production impact.
- **Fix 3 worker_eval** (`8cef769`): removed the `except Exception → inf` swallow in `_worker_eval` (expected errors already caught in `_evaluate_candidate`; the swallow only hid programming bugs).

## Java cross-check is the ground truth (key pattern)

The egg fix changes ecosystem dynamics, so it was gated on a **Python-vs-Java cross-check** (`tests/test_egg_retention_java_parity.py`, opt-in via `OSMOSE_JAR=osmose-java/osmose_4.3.3-jar-with-dependencies.jar`, band `0.1<=py/java<=10.0`). The 4.3.3 jar **loaded eec_full fine** (despite prior worries) — all 14 EEC species within **0.807-1.724×** of Java → fix direction confirmed correct. Java implements graduated egg release; matching it is the validation. EEC + BoB `.npz` parity baselines regenerated only AFTER the Java check (never re-bless a baseline to make a behavior change pass).

## Equilibrium-sensitive tests reversing = a correct ecosystem fix

Two Baltic tests flipped because the (Java-validated) egg fix shifted recruitment — **not bugs**. Recalibrated `f2e1834` (xfails removed):
- `test_biomass_pyramid_emerges`: cod ~doubled (more eggs survive → cod recovers); re-measured `_PYRAMID_BOUNDS` (seed=42, yrs 5-25, ±20%); strict ordering sprat>stickleback>cod still holds.
- `test_fr_type3_reduces_greyseal_predation_on_top_prey`: type-III refuge **mechanism intact** — probing showed it holds at halfsat 0.3/2.0/5.0; the prior halfsat=1.0 was a fragile near-tie. Recalibrated to halfsat=2.0 (robust margin). **Principled recalibration, not a pass-the-test tweak** — distinguish "the measured value moved" from "the test was always fragile."

**Open follow-up:** none for this work. Deferred: 14-nonlocal calibration refactor, medium/low review findings, `.env` credential rotation.
