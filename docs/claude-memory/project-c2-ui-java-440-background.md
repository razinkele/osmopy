---
name: project-c2-ui-java-440-background
description: C2 SHIPPED — Baltic/baltic_ev run on Java 4.4.1 from the UI (version-aware block + extracted staging module)
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**C2 — "run nbackground>0 configs on Java 4.4.1 from the UI" — SHIPPED 2026-06-30, merged to master `de169f9`, pushed to origin** (4f0e57d→ac03be0→de169f9). Third piece of the 4.4.x cutover (after A + C1 [[project-c1-native-440-cutover]]). Baltic-specific scope (hand-authored GreySeal/Cormorant tables).

**What shipped:**
- **`osmose/java_background_staging.py`** (new) — sub-project A's Baltic Java-4.4.1 staging recipe extracted from `scripts/baltic_440_smoke.py` and **generalized over background species**: `stage_background_for_java(stage_dir, raw_config)` iterates `species.type==background`, deriving idx/name/nclass/file, and emits inline `species.biomass.spN` (from the predator NetCDF), `simulation.nschool.spN`, `output.diet.stage.threshold.spN`, movement maps, + augments the staged accessibility/catchability/discards matrices. **Returns `{"output.cutoff.enabled": "false"}`** for the runner `-P` override (the 4.4.1 `OutputRegion.include` OOB with nbackground>0). `background_staging_supported(config)` gates non-Baltic. Tables: `BG_ACCESS`, `BG_DIET_STAGE_THRESHOLD={GreySeal:90, Cormorant:65}`.
- **`java_engine_block_reason(config, jar_version=None)`** now version-aware: nbg=0→allow; nbg>0 + <4.4.0/None→block; nbg>0 + ≥4.4.0 + staging-supported→allow; nbg>0 + ≥4.4.0 + unknown species→block (names it). Threaded through `engine_capabilities.describe_engine/_describe_java` so the capability display is version-aware too.
- **UI** (`ui/pages/run.py`): the run gate + `describe_engine` pass `target_version_for_jar(selected jar)`; `_run_java_engine` stages (when nbg>0 + jar≥4.4.0) and merges the cutoff override into `runner.run(overrides=...)`.

**KEY GOTCHAS (dev-caught):**
- **Movement-map indices must be assigned in ONE `_write_background_movement_maps` call across ALL background species** — the helper computes `next_idx` from the master (not updated until after), so per-species calls collide → "No map assigned for Cormorant". (My first generalization had this bug; fixed.)
- **The module must NOT import `osmose.runner` or `ui.pages.run`** (`write_temp_config`) — else runner→staging→ui.run→runner cycle. The orchestrator takes an ALREADY-STAGED dir (caller runs `write_temp_config` first). Deps: reader/numpy/xarray/stdlib only.
- **Cutoff override must reach the jar as a `-P` arg** (A passes it on the CLI; the staged-config entry alone is unverified) → `stage_background_for_java` returns it, the UI passes it to `runner.run(overrides=)` (which emits `-Pkey=value`).
- A 5th workaround beyond the obvious four: `output.diet.stage.threshold.spN` (species-specific).

**Invariants:** staged-copy only (no `data/` change), no Python-engine change. Smoke harness (importing the module) runs the 4.4.1 jar: exit 0, predators feed. Full suite 3650 passed (only the 5 pre-existing non-C2 failures: docs `0.13.0`, `run_observer`, fmsy ordering-flake). **Only C3 (BoB 365-step NetCDF re-sample) remains** in the 4.4.x cutover.
