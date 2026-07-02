---
name: project-run-page-engine-capability-shipped
description: 2026-06-20 Run-page engine-capability transparency — shipped + prod-deployed
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Run-page engine-capability transparency — SHIPPED 2026-06-20, merged master `8560cc6` (no PR; merged locally then pushed), prod-deployed + verified.**

The Run page now honestly shows what each engine produces for the loaded config, and the misleading read-only Java/Python tabs are gone.

- **`osmose/engine_capabilities.py`** (new, pure, no Shiny) — `EngineCapability` dataclass + `describe_engine(engine, config) -> EngineCapability` (single source of truth: can_run, block_reason, pages_populated, pages_empty, notable_outputs). Only engine dep is `osmose.runner.java_engine_block_reason`. Total/never-raises (unknown-engine fallback). 12 tests `tests/test_engine_capabilities.py`.
- **Capability map** (verified vs real page gates): **Python** populates Results+Diagnostics always; Genetics/Economic/Spatial Results gated on `module.genetics.enabled`/`module.bioeconomics.enabled`/`output.spatial.enabled`. **Java** populates Results ONLY (carries the rich Java-only families); Diagnostics/Genetics/Economic/Spatial Results are all Python-gated (`diagnostics.py:57`, `genetics.py:27`, `economic.py:27`) → pages_empty. `notable_outputs` lists Java-only families (sizeSpectrum etc.) for Python and "statistically equivalent not bit-identical" for Java.
- **`ui/pages/run.py`** — removed `run_engine_tabs` navset_tab + `_sync_engine_tab` mirror observer; replaced per-engine inputs with TWO `ui.panel_conditional` blocks keyed on client-side `input.engine_mode` (`"input.engine_mode !== 'python'"` / `=== 'python'`); added `engine_indicator` + `engine_capability` `@render.ui` slots. `output.spatial.enabled` IS a real key (`schema/output.py:141`, read `engine/config.py:924`).
- Spec+plan `docs/superpowers/{specs,plans}/2026-06-20-run-page-engine-capability*`.

**KEY GOTCHA (caught in in-loop plan review — BLOCKER):** the original plan moved per-engine inputs into a DYNAMIC `@render.ui` slot rendering only the active engine. That makes `input.java_opts()`/`input.py_param_overrides()` conditionally-REGISTERED → races the `btn_run` click → reading an unregistered input raises `SilentException` which silently aborts `handle_run` ("Load does nothing" class). **Fix:** `ui.panel_conditional` keeps ALL widgets in the DOM (CSS-hide only, never unmounts — verified vs installed Shiny source), so every `input.*()` stays resolvable. `input.engine_mode` is a client-side input set on load via `Shiny.setInputValue('engine_mode', mode)` (`app.py:432`); undefined-initial → Java shows (matches `state.engine_mode` default "java"). Use this pattern for any "show only the active X's inputs" UI — NOT dynamic render.

Verification: full suite 3498 passed/19 skipped, e2e 2 passed (`test_e2e_live_movement.py` confirms `#py_param_overrides` reveals+drives a run under panel_conditional), ruff+pyright clean. Prod: clone HEAD `8560cc6`, NRestarts=0, :8838 + https://laguna.ku.lt/osmose/ both 200.
