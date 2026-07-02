---
name: project-c1-native-440-cutover
description: C1 SHIPPED — bundled Python configs standardised to native OSMOSE 4.4.0; the reader/writer inverse-symmetry finding; C2/C3 still open
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**C1 — "standardise the Python setup to native OSMOSE 4.4.0" — SHIPPED 2026-06-30, merged to master `ac03be0`** (local; not pushed — push-on-request). Sub-project of the 4.4.x cutover; builds on Baltic-A [[project-baltic-440-background-adaptation]].

**What shipped:**
- **Bundled configs converted to native `osmose.version 4.4.1` on disk**: `data/{eec_full,minimal,baltic,baltic_ev}` (NOT BoB/`examples` — C3 365-step blocker). Each gated by **round-trip parity** (Python engine, fixed RNG, `<1e-9` — all **bit-exact 0.00**: old-4.3.3-source baseline vs new-native-source). `scripts/native_440_parity.py` (baselines gitignored in `scripts/_parity_baselines/`).
- **Conversion tool `scripts/migrate_bundled_to_440.py`**: per-param-file rewrite (ONLY `osmose.configuration.*`-reachable key-value files; matrices/maps excluded), RENAMES_440 forward + **skip-if-exists** (drop an OLD line whose NEW key already exists → fixes minimal `restart.enabled` dup); **EXCLUDES the lossy `predation.ingestion.rate.max.bioen` merge** (the engine reads the bioen value at config.py:1796; let the reader canonicalize it → baltic_ev parity stays 0.00); KEEPS `species.lmax`/`species.beta` (engine reads them, config.py:473/1976); does NOT bake Java-only resource-forcing keys (`species.biomass.{file,mode,varname}`) into source (they'd trip the validator; `write_temp_config` emits them at Java-stage time, H3).
- **Defaults**: default jar → 4.4.1 (`ui/state.py`); run/calibration/UI derive the write-target from the SELECTED jar via `aliases.target_version_for_jar(jar)` (4.4.1 → native, 4.3.3 → reverse-map). **BARE `to_target_keys`/writer/`write_temp_config` defaults stay 4.3.3** (USER decision) so a no-jar read→write→read round-trip is string-faithful (no rate ×ndt/÷ndt reformat) — avoids breaking `osmose_demo`'s internal round-trip check.
- `canonicalize_config` demoted to a documented **legacy 4.3.3 adapter** + one-time deprecation log (no-op on the now-native configs).

**KEY ARCHITECTURE FINDING — reader/writer inverse-symmetry (H1 was a phantom):** `OsmoseConfigReader`→`canonicalize_config` and `to_target_keys` are deliberate INVERSES. The reader ALWAYS normalizes to canonical (per-cohort larval rate via the source≥4.4.0 ÷ndt; stamps `osmose.version=4.4.0`). So `to_target_keys` always receives per-cohort and must `×ndt` for a 4.4.x target — native configs round-trip `reader(÷ndt)→writer(×ndt)` through the EXISTING code. There is **no double-scale path** (the original H1 concern was an artifact of feeding `to_target_keys` a raw rate/year dict, bypassing the reader). A "source-aware rate" fix was implemented then REVERTED (it broke the Java write path: fed Java per-cohort 15 instead of rate/year 360). **No Python-engine behaviour change — parity-gated to bit-exact.**

**Still open (separate cycles):** **C2** — UI `java_engine_block_reason` version-awareness so `nbackground>0` (Baltic) runs on Java 4.4.1 from the UI (wire in Baltic-A's staging). **C3** — BoB 365-step LTL-forcing fix (re-sample the NetCDF to a 24-step axis). Pre-existing non-C1 test failures (docs `0.13.0` vs actual `1.2.0`; `test_run_observer` live-movement lambda) are unaffected. [[project-config-key-migration-440]] [[reference-osmose-java-4-4-0]]
