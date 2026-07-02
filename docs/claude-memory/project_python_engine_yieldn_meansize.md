---
name: project-python-engine-yieldn-meansize
description: 2026-06-25 Python-engine yieldN + meanSize outputs (CSV+NetCDF) shipped; + the dormant-combined-NetCDF-writer discovery
metadata:
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

Shipped to local master 2026-06-25 (merge `af19fa0`, --no-ff; 5 TDD task commits `3d00653..9d76477` + spec/plan). Adds two Java-v4.4.1-faithful per-species Python-engine outputs that previously forced a Java run: **`yieldN`** (fishing catch in NUMBERS = Σ `n_dead[FISHING]`, no ×weight, no cutoff) and **`meanSize`** (ABUNDANCE-weighted mean length cm = Σ(abd×len)/Σ(abd), applies `output_cutoff_age`), in CSV + in-memory + NetCDF. Output-only, parity-safe (14/14 EEC `atol=0` + 8/8 BoB unchanged). Same gap-fill pattern as PR #75. Spec/plan: `docs/superpowers/{specs,plans}/2026-06-25-python-engine-yieldn-meansize*`.

## KEY durable discovery (non-obvious, useful for future NetCDF work)
**The combined per-species NetCDF writer `write_outputs_netcdf` was DORMANT** — never called by the engine's `write_outputs` entry point (which only wrote CSVs + `write_outputs_netcdf_spatial`). So the pre-existing `output.*.netcdf.enabled` flags (biomass/yield/etc.) were parsed but produced NOTHING on real runs. This feature **wires `write_outputs_netcdf(outputs, output_dir/f"{prefix}_Simu0.nc", config)` into `write_outputs`**, gated by the writer's own `if not any(want.values()): return` → inert for default/parity configs (they set no `.netcdf` flag), but it now also ACTIVATES all the other dormant per-variable NetCDF flags. Read back via the existing `read_netcdf(f"{prefix}_Simu0.nc")` + new `yield_abundance(source="netcdf")`/`mean_size(source="netcdf")`.

## Gotchas baked in (verified)
- Readers are `results.yield_abundance()` (→ "yieldN") and `results.mean_size()` (→ "meanSize") — there is NO `results.yield_n()` method.
- Both NetCDF vars use the **`focal_species`** dim/coord (`config.species_names`), NOT the shared `species` dim (`all_species_names` = focal+background) — else CSV cols ≠ NetCDF cols.
- New schema key added: `output.size.netcdf.enabled` (OSMOPY per-variable convention; Java uses a global toggle). yieldN's 3 keys already existed.
- `_CROSS_SPECIES_OUTPUT_TYPES` must include "yieldN"/"meanSize" or the in-memory cache key gets mangled by `partition("_")`.
- meanSize subdt-averaged via `_avg_scalar_dict` (mean-of-ratios, matches `meanTL`); yieldN summed. `_PYTHON_NOTABLE` no longer lists yieldN/meanSize.

## Process note
brainstorm → spec (2-round in-loop review, 3 source-verifiers; caught the dormant writer) → plan (5-lens **Workflow** review whose verifiers APPLIED the plan + ran tests → caught a missing `import numpy as np`, the meanSize dim mismatch, vacuous dead-sim tests, wrong `write_outputs`/`OsmoseResults` signatures) → subagent-driven TDD (5 tasks each reviewed) → final whole-branch review. The apply-and-run verification caught executability bugs pure-reasoning review missed. Related: [[reference-engine-mortality-dispatch]], the community-outputs gap-fill [[project-python-engine-community-outputs]].
