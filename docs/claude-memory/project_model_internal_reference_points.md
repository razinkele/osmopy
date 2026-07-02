---
name: project-model-internal-reference-points
description: 2026-06-25/26 model-internal Fmsy/Bmsy/Blim via yield-vs-F sweep shipped (v2 of the fisheries page) + the fishing-override-key + in-memory-reader traps
metadata:
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

Shipped to local master 2026-06-26 (merge `c78c15d`, --no-ff; 8 TDD commits `3bf5c0c..95d652a` + spec/plan). The deferred **v2 of the fisheries stock-status page** ([[project-fisheries-stock-status]]): derive per-species **Fmsy** (yield-maximising F), **Bmsy** (equilibrium SSB at Fmsy), **B0** (F=0 SSB), **Blim** (=0.2·B0) from a per-species **conditional** yield-vs-F sweep (others held at baseline). Library `osmose/validation/fmsy_sweep.py` + CLI `scripts/compute_model_reference_points.py` write `data/<eco>/reference/fisheries_model_reference_points.json`; the Fisheries page reads it (**precedence user > model > ICES**) so the Kobe auto-populates both axes with no user input. Parity-safe (sweep only varies fishing + output-flag config, like calibration; no engine-dynamics change). Travers-Trolet/Mackinson methodology. Spec/plan: `docs/superpowers/{specs,plans}/2026-06-25-model-internal-reference-points*`.

## KEY durable facts / traps (all caught by in-loop workflow reviews, verified vs source)
- **THE override-key trap:** to sweep a species' fishing F you must override the ACTIVE knob. `EngineConfig.from_dict` IGNORES the legacy `mortality.fishing.rate.sp{i}` when fisheries-mode is on (`module.multispecies.fisheries.enabled==true AND simulation.nfisheries>0`) — **BOTH bundled configs (eec_full=14, baltic=8) are fisheries-mode**. Override `fisheries.rate.base.fsh{j}` instead (map species→fishery = first catchability column >0; `config.py:296-313`). The runner **asserts `fishing_rate[i]` actually moved** before each run (no-op-trap guard) — a wrong key = a silent flat curve = garbage Fmsy.
- **In-memory readers differ from disk:** `results.fishery_yield()` reads a Java-only output and RAISES in-memory → use `results.yield_biomass()` (the "yield" output, tonnes). `results.ssb()` is GATED (force `output.ssb.enabled=true` per run; `output.yield.biomass.enabled` is INERT in-memory — yield is unconditional). **`results.mortality()` in-memory is a FLAT frame** (Time, Predation, …, **Fishing**, …, species) — NOT the `('F',stage)` MultiIndex of the on-disk CSV; read the flat `Fishing` column (a MultiIndex read silently returns 0.0 → every Fmsy 0.0).
- **Catchability file** resolves via `cfg["_osmose.config.dir"]` (injected by OsmoseConfigReader) + `osmose.engine.path_resolution.resolve_data_path(rel, config_dir=)` (globs `data/*/`). There is NO `__config_dir__` key.
- **Cost is real:** ~4.4 s/sim-year (measured) → the sweep (`n_species × grid7 × replicates3 × ~30yr`) is a **tens-of-minutes-to-hours offline CLI batch**, cached per config. Parallelize across runs on a ProcessPool with `numba.set_num_threads(1)` per worker (a single run already saturates cores via the `prange` mortality kernel); serial in-process when `max_workers<=1` (testability).
- `EngineConfig` field is `n_year` (singular). `fisheries_reference._to_float` (not `_float`). `b_ref_label` is now conditional (`Bmsy [model]`/`[user]`) — also fixed the parent feature's always-"[user]" minor.

## Process note
brainstorm → spec (2-round in-loop **workflow** review: caught the no-op override key + wrong yield reader + SSB-off + a 10–100× cost underestimate) → plan (apply-and-run workflow review: an agent applied all 6 tasks in an isolated worktree + ran them → caught the realized-F MultiIndex→silent-0.0 bug + the invented config-dir key) → subagent-driven TDD (6 tasks, each reviewed; 2 fix loops) → final whole-branch review MERGE-READY. Related: [[project-fisheries-stock-status]] (parent), [[project-python-engine-yieldn-meansize]].
