---
name: python_engine_status
description: Python OSMOSE engine status — feature list, parity, historical bug fixes worth remembering, and pointer to live stats. Read before modifying osmose/engine/ or reasoning about parity.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
## Python Engine Status

**Phase:** 1-9 complete. Full Java feature parity including Ev-OSMOSE genetics (shipped in Front 6, v0.9.1). Only gap left is some Ev-OSMOSE bioen spatial outputs (see parity roadmap for exact list).

**Live counts:** do not trust hard-coded numbers here — see [project_current_status.md](project_current_status.md) for current version / test count / parity tables. Summary at hand-off (2026-04-19): v0.9.1, 2485 passing, EEC 14/14, Bay of Biscay 8/8, Python FASTER than Java on all benchmarks.

### Features Implemented (Complete List)
- Von Bertalanffy + Gompertz growth dispatch with predation gating
- Per-cell per-school interleaved mortality (matching Java computeMortality)
- Predation: Numba JIT with diet tracking, size-based, accessibility matrix, feeding stages
- Shared predation helpers (`compute_size_overlap`, `compute_appetite`)
- Fishing: v3 annual rate + v4 fisheries (seasonality, age/sigmoid selectivity, spatial maps, discards, MPA)
- Starvation: lagged predation success
- Natural mortality: additional + larva + aging + out-of-domain + time-varying BY_DT + spatial distribution
- Movement: random walk + map-based (CSV maps, rejection sampling, age/season indexing) + random patch constraint (Numba + prange parallel)
- Reproduction: SSB eggs, spawning seasons (normalized, multi-year), seeding, maturity age+size check
- Background species: inject/strip, forcing (uniform + NetCDF), predation participation
- Incoming flux: CSV time-series biomass injection
- Resources: legacy + `species.type` keys, multiplier/offset, time-varying accessibility, cap at 0.99
- Stage-indexed accessibility matrix (label-encoded CSV)
- Output: biomass, abundance, mortality, yield, biomassByAge/Size, abundanceByAge/Size CSVs + NetCDF per-species distributions + mortality-by-cause + diet CSV (Java-parity) + spatial biomass/abundance/yield-biomass NetCDF + cutoff age + recording frequency + step0 (Phase 5 SP-4 shipped v0.9.0)
- Per-species deterministic RNG via SeedSequence (movement + mortality flags)
- Bioenergetic (Ev-OSMOSE) module + Genetics (Genotype/Trait/Locus, weighted parent selection, evolving traits) + Economics (shipped Front 6, v0.9.1)

### Load-Bearing Historical Bug Fixes (keep — these are non-obvious)
- **Config-dir-aware file resolution** (`97ac300`): `_osmose.config.dir` is injected into the flat config dict during `reader.read()`, then used as the primary search path in all `_resolve_file()` / `_resolve_path()` functions. Before this fix, engine only searched `.` and `data/examples/`, so configs in other dirs silently fell back to 10×10 grid instead of correct geometry.
- **v4 fisheries seasonality parser** (`97ac300`): EEC uses both `fisheries.seasonality.fsh{i}` (inline) and `fisheries.seasonality.file.fsh{i}` (CSV). Parser handles both via catchability-matrix mapping.
- **Config case preservation** (`b9ce724`): config reader preserves original key case for Java writer — Java is case-sensitive. See [feedback_config_case.md](feedback_config_case.md).
- **Unified predation** (single-pass proportional): schools + resources in one pass, not sequential. See [feedback_predation_architecture.md](feedback_predation_architecture.md).
- **Larva mortality rate**: applied as FULL per-cohort rate (not divided by `n_dt_per_year`).

### Engine Layout
- Root: `osmose/engine/` — entry point `simulate.py`.
- Sub-packages: `processes/` (growth, predation, mortality, reproduction, movement, fishing), `outputs/`, plus top-level state/config helpers.
- Performance: see [performance_optimization.md](performance_optimization.md).
- Validation: `scripts/validate_engines.py` (Bay of Biscay); EEC via ad-hoc `/tmp` script.
