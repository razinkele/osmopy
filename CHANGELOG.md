# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/), generated from [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Changed

- **calibration:** preflight evaluator logs exceptions and aborts when Morris majority-of-samples fails (was: silent `except Exception: pass`). (4de10ae)
- **calibration:** surrogate `find_optimum(weights=...)` for weighted scalarization; default multi-objective call returns a Pareto (non-dominated) set instead of a naive unweighted sum. (6d75310)
- **calibration:** `OsmoseCalibrationProblem` now accepts `subprocess_timeout` and `cleanup_after_eval` options; on subprocess failure the full stderr lands in `run_dir/stderr.txt` instead of being truncated in logs. (8358725)
- **calibration:** preflight evaluation loop is parallelizable via `n_workers`; per-sample `output_dir/preflight_{i}` subdirs avoid concurrent-writer clobbering. (a454758)

### Refactored

- **calibration:** `run_preflight` split into `_run_morris_stage` / `_maybe_run_sobol_stage` / orchestrator. (13f9474)
- **calibration:** `detect_issues` split into per-category helpers (`_issues_blowup`, `_issues_negligible`, `_issues_all_negligible`, `_issues_flat`, `_issues_bound_tight`). (95db190)
- **scripts:** `calibrate_baltic.py` gains a `Callable[[np.ndarray], float]` return hint on `make_objective` and swaps the list-wrapped counter for `itertools.count(1)`. (7ace2f3)

## [0.8.1] - 2026-04-17

### Features

- auto-load .env in copernicus MCP server (9b8d197)
- **data:** add Baltic NetCDF grid and fix LTL trophic level key (f1bcd8f)
- **engine:** wire XorshiftRNG into simulation via java_compat_rng flag (9bf9d11)
- **engine:** add Java-compatible Xorshift RNG for bit-exact parity (70425ee)
- add dynamic accessibility, Python engine UI, Baltic example, and renderer badge (396e496)
- **engine:** wire bioen starvation into mortality loop with bioen_enabled switch (24e8296)
- **engine:** add foraging mortality function with constant and genetic modes (60dc41f)
- **engine:** add by-dt-by-class additional mortality variant (eb6f977)
- **engine:** add fishing scenario auto-detection from config keys (0579da2)
- **engine:** wire Gaussian/log-normal selectivity into fishing mortality (3e8c509)
- **engine:** add catch-based proportional allocation fishing variants (f713b7a)
- **engine:** add rate-by-dt-by-class fishing mortality variant (cdeb063)
- **engine:** add Gaussian and log-normal fishing selectivity types (496dbce)
- **engine:** add SingleTimeSeries and GenericTimeSeries with CSV loading (f976c7b)
- **ui:** add preflight screening checkbox, modal, and handler wiring (4f7b565)
- **calibration:** export preflight API from __init__ (47d0fc0)
- **calibration:** add run_preflight() two-stage orchestrator (fe6229a)
- **calibration:** add preflight issue detection logic (4ad9231)
- **calibration:** add Morris screening with multi-objective aggregation (36db8b4)
- **calibration:** add preflight data model — enums and dataclasses (165d1ba)
- **calibration:** export history module from __init__.py (a1a63d8)
- **calibration:** add banded loss, validation, history, and 2D sensitivity handlers (6a9bb06)
- **calibration:** restructure tabs, add banded loss/validation/history/correlation UI (1da6bdf)
- **calibration:** add history persistence module (2b42a0b)
- **calibration:** add correlation chart, upgrade sensitivity for multi-objective (aa242e7)
- **calibration:** export new public API from __init__.py (953405a)
- **calibration:** add evaluation cache and schema validation to problem (ec2b148)
- **calibration:** add surrogate cross-validation and fit_score_ (00c724d)
- **calibration:** extend sensitivity analysis for multi-objective (d1dd600)
- **calibration:** add multi-seed validation and candidate ranking (c1a27e2)
- **calibration:** add composable banded loss objectives (855b594)
- **calibration:** add BiomassTarget data model and CSV loader (9af1cd7)

### Bug Fixes

- **engine:** keep Numba enabled with java_compat_rng flag (3270bc1)
- **engine:** distinguish null-map migrations from placement failures in warning (679b70f)
- **tests:** update tests for new fishing config fields and sigmoid API (64c5676)
- apply test coverage + dependency ordering review to plan (653149d)
- apply code correctness + spec alignment review to plan (7027c98)
- **deploy:** add --root-path /osmose for reverse proxy WebSocket routing (b28db08)
- **calibration:** single-obj Pareto crash, NSGA-II convergence persistence, absolute HISTORY_DIR (6716e99)
- **calibration:** track duration_seconds, fix surrogate convergence format, add post_validation API (6861b26)
- **docs:** Phase 2 plan — fix except placeholder, thread-safe history save, dead code (d6a043e)
- **docs:** Phase 2 spec — add objective_names to history JSON, derive species_names (b5d1e27)
- **docs:** correct Phase 2 spec — navset names, tmpl params, run_id conflicts (74a07ee)
- **calibration:** guard worst_species_penalty against empty list, document first-objective-only metrics (87baf55)
- **engine:** align egg detection to first_feeding_age_dt (F4E) (6d7502c)
- **ui:** accessibility, UX safety, and consistency improvements (cb35aef)
- **config:** validation, writer routing, scenario persistence fixes (3b5601c)
- **engine:** safety and correctness fixes across engine processes (39ad713)

### Performance

- **calibration:** gate history buttons on tab selection + reactive trigger (d6d3f29)

### Chores

- gitignore operational dirs (.remember, logs, cmems_cache) (9b427ec)
- clean up ruff F401/F841 in non-CI-scoped scripts (d1ee962)
- register ICES MCP server in .mcp.json (ab3fd88)

### Documentation

- ship osmose-master java test/build-discipline patch (34707a4)
- add 2026-04-17 deep-review fixes plan (4525c92)
- apply 4-task review to SP-5 bioen activation plan (01b6fad)
- add SP-5 bioen activation implementation plan (123bdd1)
- apply 3-task review to SP-2 additional mortality plan (bb32dd7)
- add SP-2 additional mortality variants implementation plan (cc72bdb)
- apply 5-task review to SP-1 fishing plan (bec15e8)
- add SP-1 fishing system completion implementation plan (8011c4d)
- apply 5-task review to SP-3 timeseries plan (4bac915)
- add SP-3 time-series framework implementation plan (104b784)
- apply 5-iteration Java source review to parity spec (4f3863a)
- apply review fixes to Java parity spec (9 issues) (cb8ae1d)
- add full Java OSMOSE 4.3.3 parity design spec (3bcdd3e)
- apply 10-task review to preflight implementation plan (540c96f)
- add pre-flight sensitivity analysis implementation plan (c9e202a)
- apply 6-iteration review to pre-flight sensitivity spec (9c24e33)
- add pre-flight sensitivity analysis design spec (dcad6a7)
- add ICES data access implementation plan (b4cf65e)
- apply 4-iteration review to ICES data access spec (a6e1558)
- add ICES data access MCP server + skill spec (b1a78ef)
- add calibration UI Phase 2 implementation plan (9ab6b31)
- add calibration UI Phase 2 design spec (0b8cc86)
- add calibration library gaps Phase 1 implementation plan (b2b110d)
- fix 10 issues in calibration library gaps spec (f8c6f03)
- add calibration library gaps Phase 1 design spec (7dbb43b)

### Other

- 2026-04-17 deep-review fixes (credentials, types, docs) (2c45323)
- encode calibration preflight invariant via _require_preflight helper (5a50a36)
- widen param_names annotation for pandas DataFrame columns (0caf824)
- annotate reactive.value generics for cal_X/cal_F (b036a1f)
- broaden load_timeseries return, align ByYearTimeSeries param name (f23e733)
- narrow stage_accessibility and access_matrix for prey-scale path (b5e47c2)
- cast bioen eta to float to match bioen_starvation signature (4653d81)
- hoist fishing_catches to narrowed local in catch allocator (dde6d78)
- declare SchoolState.imax_trait and guard None in fields() iterators (59ccca1)
- remove committed CMEMS credentials from .mcp.json, document env setup (6e71a2e)
- require CMEMS credentials from env, remove hardcoded fallback (ddb742b)
- add reference_point_type and metadata to Baltic targets CSV (a8a56ef)

### Refactoring

- extract _require_creds helper, add behavioral guard test (3973069)
- **deploy:** switch from shiny-server to standalone Uvicorn service (2eb860a)

### Styling

- format calibration UI Phase 2 files (597b592)
- format calibration library gap files (b9c5881)

### Tests

- detect literal CMEMS credentials under any mcpServers env block (3b618e4)

## [0.8.0] - 2026-04-13

### Features

- **engine:** add movement.map.strict.coverage config key (M-7) (9911cc7)
- **engine:** add uncovered-slot fixture + strict parameter to MovementMapSet (M-7) (e3c218a)
- **ui:** extract parse_nspecies to shared _helpers.py (M-9) (2c5ba61)
- **engine:** MPAZone validates grid shape and binary values (I-8) (195b873)
- **engine:** validate bioen_* field coupling at EngineConfig construction (I-2) (ed009b3)
- **engine:** add SchoolState.validate() for biological invariants (I-1) (936eb33)
- add Java/Python tabbed layout to Run page (f6bc5c9)
- wire new pages with engine-gated nav items (578a2ac)
- add Ev-OSMOSE genetics stub page module (14d2332)
- add Python engine diagnostics page module (3d02fac)
- add economic module stub page module (2febf66)
- add Java/Python engine selector toggle in header (9991f74)
- add engine_mode reactive field to AppState (d20862d)
- register Map Viewer tab in app navigation (0d1b778)
- add Map Viewer page with file list and deck.gl preview (842594a)
- move model info into header bar, remove second row (e8a2f42)
- deep review fixes, CSV separator fix, automation setup, map display tests (634498e)
- add species distribution and fishing maps to grid overlay selector (b141fcd)
- wire Spatial Results page into nav with disabled pill state (b4f7cf6)
- implement spatial_results_server with data loading and map rendering (74ca06d)
- add spatial_results page skeleton with UI layout (6a32e18)
- **genetics:** seeding phase + config fields for transmission_year and neutral loci (579bf66)
- **genetics:** add neutral loci to GeneticState and inheritance (d6b6295)
- **genetics:** wire trait_overrides into _bioen_step and _bioen_reproduction (1b3384a)
- **economics:** add economic CSV output files (261d2fb)
- **economics:** add annual reset and catch memory update in simulation loop (c659736)
- **economics:** add full cost model, stock catchability, and catch memory (136a5b0)
- **economics:** add travel cost and stock-dependent revenue calculations (5b5afaf)
- **economics:** integrate effort map into fishing mortality scaling (b7c7ae1)
- **simulate:** wire genetics and economics into simulation loop (884cb21)
- **config:** add genetics_enabled and economics_enabled flags to EngineConfig (a564167)
- **economics:** add DSVM logit choice model and effort aggregation (b138f17)
- **economics:** add FleetConfig, FleetState, and config parsing (9612e98)
- **genetics:** add gametic inheritance with fecundity-weighted parent selection (8157bf4)
- **genetics:** add trait expression and phenotype override mechanism (3274519)
- **genetics:** add GeneticState with compact/append sync and initial genotype creation (3d5bc5e)
- **genetics:** add Trait dataclass and TraitRegistry with config parsing (7381795)

### Bug Fixes

- **ui:** log exc_info in _close_spatial_ds instead of bare pass (M-8) (04b5824)
- **engine:** apply _require_file to 4 adjacent silent-failure sites (26cf198)
- raise KeyError on NetCDF variable name mismatch for background species (C-8) (3d2d134)
- raise on missing config files instead of silent fallback (C-3..C-7) (fa0c5f9)
- pad fishing_seasonality/discard_rate for background species (C-1, C-2) (f177926)
- **tests:** register osmose plotly template in conftest to fix isolation failures (3bf4bf2)
- gracefully skip invalid scenario names in import_all (M6) (7c5af1a)
- reject oversized ZIP entries in scenario import_all (M10) (de264ed)
- reset key_case_map between read() calls on same reader (M7) (1cdb77a)
- normalize partial-year spawning tail with warning (M2 followup) (4d80db1)
- normalize spawning season per-year chunk instead of total (M2) (55349dd)
- sanitize exception messages in UI notifications (H14) (3f981c0)
- add path traversal guard to comparison_chart and config_diff_table (H12, H13) (adcbb38)
- register atexit cleanup for export and demo temp dirs (H10, H11) (6179e8d)
- move state.dirty.set inside reactive.isolate in forcing sync (H8) (616efeb)
- raise ValueError on asymmetric species column in RMSE merge (H7 followup) (4c8c622)
- include species in RMSE merge to prevent cross-product (H7) (f7e9319)
- filter internal _-prefixed keys from config writer output (H6) (0b4407f)
- aggregate movement map coverage warning instead of per-slot flooding (H5) (d6d42ec)
- use integer sampling instead of round() to avoid boundary cell bias (H4) (c4cce5a)
- preserve distribution dicts in _average_step_outputs (H3) (2dda4c4)
- use live abundance*weight in Python fallback predation (H1) (f05c2fb)
- explicit NotImplementedError in JavaEngine.run_ensemble (C7) (96925fd)
- **tests:** add W=c*L^b value pin and tighten bounds in weight-length test (C6 followup) (0f6f12a)
- replace tautological weight-length test with meaningful assertions (C6) (083ae5f)
- consume reactive.poll result so calibration UI updates (C5) (b9ce68e)
- CSV map orientation, colormap, layer toggle, multi-value config guard (247a232)
- restore hasattr fallback in make_legend for server compatibility (92ec20a)
- address round 3 review findings (9b004bd)
- add type annotations for reactive values, fix f-string lint warning (e8db4a7)
- handle all-NaN data in spatial scale info display (5c91e27)
- guard shiny:connected handler against duplicate registration (44473be)
- address review findings across spatial results implementation (c76ba79)
- remove unused math and _sdgl imports from grid.py (c2eb8ff)

### Performance

- **calibration:** reuse problem instance and add cleanup (a57f681)
- **ui:** add config value accessor and cache grid NetCDF loads (ea82aa8)
- **io:** add CSV/preamble caching and fix double stat() (83f5ea3)
- **engine:** vectorize hot-path loops and reduce allocations (7006418)
- vectorize temperature lookups in _bioen_step via get_grid (M13) (ea8e5c3)
- vectorize load_csv_overlay cell loop with NumPy (44e1e54)

### Chores

- delete unused JavaEngine stub + Engine Protocol + redundant Path import (I-10, M-2) (3f707ea)
- gitignore superpowers workspace + skill review HTML (41c8f29)
- fix remaining ruff errors (FleetState forward ref, ambiguous l) (d31fdcd)
- **tests:** ruff auto-fix unused imports and drop duplicate classes (d980e89)

### Documentation

- update README, CHANGELOG and bump version to 0.8.0 (cb2f447)
- **engine:** document population.seeding.year.max is global-only per Java parity (M-5) (48bb1c8)
- remove duplicate focal_starvation_rate_max in plan 1 task 3 (review loop 2) (9930818)
- fix 8 blockers in v3 deferred plans after thorough review loop (feb98ec)
- add implementation plans for 3 v3 deferred items (I-3, M-7, D-1+M-5+M-9) (be2e62f)
- add design specs for 3 v3 deferred item plans (I-3, M-7, D-1+M-5+M-9) (f914e51)
- **engine:** clarify cell_id expression when resources is None (M-6) (51e1c19)
- make output.py TODOs actionable / link to roadmap (M-4) (1eee064)
- revise v3 remediation plan after 5 review loops (e9bad01)
- add deep review v3 remediation plan (28 tasks, 7 phases) (e387de7)
- add fresh deep review v3 findings document (34 items) (441b378)
- **engine:** pin phi_t Arrhenius fallback behavior for e_d==e_m (C3) (e9dd7c0)
- add deep review v2 Phase 3-5 implementation plan (a7b59e6)
- enable bypass permissions and mark UI tightening plan complete (4cd9edd)
- add UI tightening + engine selector spec and plan (b0edf1b)
- add implementation plan for Grid / Spatial Results split (521d7e9)
- add spec for splitting Grid & Maps into Grid + Spatial Results (bab5d72)
- add Ev-OSMOSE + DSVM fleet dynamics design specification (0a33fe9)

### Other

- Merge branch 'refactor/from-dict-split-2026-04-12' — I-3 from_dict monolith split (e4100f3)
- docs+test: pin SimulationContext diet field two-way coupling (M-14) (592d0c6)
- docs+test: pin StepOutput age/size distribution pairing invariant (M-13) (9c3531b)
- Merge feat/ev-osmose-economic-mvp: Ev-OSMOSE genetics + DSVM fleet dynamics Phase 1 MVP (5b0e0aa)

### Refactoring

- **engine:** extract _parse_output_flags from from_dict (I-3 step 5/5) (05d1d03)
- **engine:** extract _merge_focal_background from from_dict (I-3 step 4/5) (3dd4551)
- **engine:** extract _parse_predation_params from from_dict (I-3 step 3/5) (52c65a4)
- **engine:** extract _parse_reproduction_params from from_dict (I-3 step 2/5) (a21c8c7)
- **engine:** extract _parse_growth_params from from_dict (I-3 step 1/5) (de71cde)
- **ui:** wire parse_nspecies into forcing.py (M-9) (034a1cc)
- **ui:** extract format_timing_pairs from diagnostics.py (M-9) (4fd32a4)
- **ui:** extract collect_resolved_keys from fishing.py (M-9) (ab2f327)
- **ui:** extract count_map_entries + wire parse_nspecies in movement.py (M-9) (4658092)
- **engine:** extract _accessibility_path_or_none helper (M-3) (b251f0d)
- **engine:** consolidate per-species timeseries CSV loaders into one helper (I-9) (b586642)
- extract _DEFAULT_VIEW_STATE in map_viewer.py (07fba24)
- consolidate FakeInput into shared test helpers (24dd1e4)
- replace string sentinels with named constants in grid.py (d8eaa48)
- extract _compute_half_extents to deduplicate finite-difference logic (e3fe543)
- extract _find_config_file helper to deduplicate file search (cf62ac0)
- extract _overlay_label and discover_spatial_files into grid_helpers (81a3394)
- remove Spatial Distribution tab from Results page (4c11a2b)
- rename Grid & Maps tab to Grid (f9c5ff1)
- move make_legend and make_spatial_map to grid_helpers for sharing (53af11a)

### Styling

- tighten nav pills, card headers, and content gap spacing (ede6e39)
- tighten header bar spacing and reduce element sizes (546b7e1)
- compact header padding, nav pill spacing, and section labels (b500011)
- add osm-disabled class for greyed-out nav pills (242cc17)
- ruff format remaining genetics and economics files (f185d95)
- fix ruff lint warnings in genetics and economics tests (dcc040b)

### Tests

- update tests for performance optimizations (8195aa2)
- **ui:** pin reactive.isolate write-propagation semantics (D-1) (23397ef)
- **ui:** add pure-helper unit tests for spatial_results._nc_label (M-9 partial) (4b5bfab)
- strengthen construction-only assertions in config validation tests (M-12) (bab8013)
- deduplicate test_parse_label (M-11) (42821ed)
- strengthen test_zero_rate_no_mortality with non-zero control school (M-10) (19d3129)
- pin additional_mortality_by_dt override step-rotation (I-7b) (ea827e9)
- pin out_mortality rate application when is_out=True (I-7a) (948078f)
- cover reproduction 'n_eggs < n_new' collapse branch (I-6) (6b50c83)
- pin _average_step_outputs multi-element branch contract (I-5) (8869743)
- add direct behavioral test for _predation_on_resources (I-4) (166d917)
- add coverage for output.step0.include and partial flush (H18) (00ff908)
- assert movement map warning fires once per species (H5 followup) (53b5690)
- add _map_move_school uniform placement regression test (H4 followup) (2d3596a)
- add E2E Playwright tests for Map Viewer page (6f9c070)
- add e2e tests for Spatial Results disabled pill and Grid rename (97278a1)
- **genetics:** 5 statistics tests for trait expression and neutral loci (78b9258)
- **economics:** add multi-fleet non-interference tests (4cced56)
- **economics:** add days-at-sea tracking and forced port tests (786431c)
- add integration tests for genetics and economics modules (b60f0c0)

## [0.7.0] - 2026-04-05

### Features

- add __post_init__ validation to MPAZone, ResourceSpeciesInfo, BackgroundSpeciesInfo, CalibrationPhase (H10) (a3465fa)
- add context manager protocol to OsmoseResults (H4) (4d58ee0)
- add __post_init__ validation to OsmoseField, Grid, and SchoolState (H3a-c) (ef96bbf)
- add EngineConfig __post_init__ validation for array lengths and biological constraints (dd56db5)
- FreeParameter uses Transform enum + bounds validation (81385dd)
- emit ImportWarning when Numba is unavailable in engine processes (079ad59)
- add strict mode to OsmoseResults to raise on missing data (399d485)

### Bug Fixes

- remove stale Phase 1 docstring and align output prefix to 'osm' (C6, H13) (9d59d9e)
- default strict=True for OsmoseResults, fix version fallback, narrow UI catches (C4, H7, H8) (72c1b5b)
- close OsmoseResults in calibration and handle malformed CSVs (C3, C5, M17) (d9e8216)
- add epsilon guards to f_o2 and phi_t to prevent NaN on edge cases (C2) (c3fab38)
- abort calibration when >50% of candidates fail (H6) (033de9a)
- use age_dt < first_feeding_age_dt for larvae check instead of is_egg (C2) (87e8ce9)
- wrap xr.open_dataset in context managers to prevent file handle leaks (H2) (0e0fa73)
- add logging for empty DataFrame returns and ncell injection skip (H5, SF3) (92786ab)
- use hasattr guard instead of broad except AttributeError in sync_inputs (C8) (5f7f231)
- validate CSV grid dimensions in movement map loader (C7) (f18370a)
- guard against division by zero when size_ratio_min is 0 (C1) (f63adf0)
- update bioen test to use lowercase sizeinf config key (C5 follow-up) (deddf6b)
- add missing calibration objective exports to __all__ (H8) (7778964)
- use _log.warning instead of warnings.warn for Numba fallback (H7) (979a4ac)
- replace assert with if/raise for bioenergetics validation (C3) (0f92a81)
- use mortality.additional key pattern in calibration auto-detect (C6) (286a4f4)
- lowercase config key lookups for ndtperyear and bioen sizeinf (C4, C5) (b746f37)
- add path traversal guard to _resolve_file (68cc9ba)
- address all remaining review findings (minor+medium) (a458236)
- add global_map_idx bounds check + clamp inf in nan_to_num (1462d6c)
- address review findings — guards, NaN clamping, tests (f833582)

### Performance

- vectorize fishing spatial map and MPA lookups (H12) (0d8e020)
- precompute species masks once in _bioen_step instead of 6 times (M1) (24770b1)

### Chores

- add .worktrees to .gitignore (deef65f)

### Documentation

- bump v0.7.0 — update CHANGELOG, README, and version for deep review #2 remediation (e167049)
- fix misleading comments across 7 files (M19-M25) (1add9fb)
- fix misleading comments across 7 files (M19-M25) (7768b82)
- fix comment quality issues from deep review (d1438a9)
- fix stale comments, schema descriptions, and docstrings (H9, M10-M14) (016a2d3)
- update README with latest benchmarks and EEC biomass parity table (16bd43d)

### Other

- Merge branch 'deep-review-2-remediation' (58b9740)

### Refactoring

- extract shared Numba mortality cause-dispatch, eliminating 3x duplication (H2) (d8dc189)
- split _collect_outputs into focused sub-functions with shared species-mean helper (H5) (ac3ca6c)
- make SchoolState and StepOutput frozen to enforce immutable-replacement pattern (H9) (367b777)
- consolidate 4 duplicated path resolvers into shared module (H6) (9729e8c)
- replace module-level mutable globals with SimulationContext (C1) (341e930)
- BFS deque, Scenario validation, batch append, mask UI warning, reader skip count (M4-M9) (1a34715)
- move _last_key_case_map from module global to AppState (H1 partial) (7d27ab3)
- add type annotations to config validator public API (M15) (5102268)

### Styling

- apply ruff format across codebase (9431c75)

### Tests

- add UI state and ensemble edge case tests (T7, T8) (bd286f2)
- add numerical edge case and ENUM validation tests, fix validate_field ENUM gap (T3, T6) (5544cd1)
- add config reader error path tests for circular refs, path escape, file size (T2) (cd83324)
- add _bioen_step orchestration tests for temperature branches and edge cases (T1) (302e1d4)
- add coverage for out_mortality formula, config errors, writer semicolon, resource depletion (T1,T4,T5,T7) (d955e1c)

## [0.6.0] - 2026-03-22

### Features

- wire Numba movement batch into movement() and simulate() (cd3224f)
- add _map_move_batch_numba for compiled movement (001355e)
- add _precompute_map_indices for movement Numba path (fd4fc46)
- add _flatten_all_map_sets for Numba movement data (1d13a44)
- wire batch cell loop into mortality() orchestration (Phase A) (c255eeb)
- add _mortality_all_cells_numba batch function (513720f)
- add _pre_generate_cell_rng for batch RNG pre-generation (0b6c81c)
- add --statistical mode to save_parity_baseline.py (b9edab6)

### Bug Fixes

- sp_ids[k] → sp_ids[idx] bug in movement Numba function (bfd02a0)

### Performance

- vectorize _precompute_effective_rates with NumPy (0b57e70)
- add prange parallel cell processing (Phase B) (8cce908)
- move RNG generation into Numba batch function (Phase A) (a549090)

### Documentation

- fix 3 review issues in scaling parity spec (round 3→4) (abf5447)
- add scaling parity spec (movement Numba + vectorized rates) (efb936d)
- add sync comments for duplicated Numba inner loops (5534d72)
- add parity tests to performance spec, fix review round 3 (2cbc34b)
- add Python engine performance optimization design spec (15cf94f)

### Other

- v0.6.0 — Python engine faster than Java (8509845)
- final baselines after scaling parity — Python 1.5x faster than Java on EEC 5yr (fcc8fd1)
- save EEC 5yr pre-scaling baseline (17.9s) (44b8d04)
- generate 10-seed statistical baseline for BoB 1yr (dab7e47)
- save 5yr pre-Phase-A timing baseline (69e30a7)

### Tests

- add TestStatisticalParity for cross-version RNG tolerance (6fed9fc)

## [0.5.0] - 2026-03-22

### Features

- **ui:** add ambient ocean atmosphere, sonar spinner, polished notifications (d117776)

### Bug Fixes

- preserve original key case when writing config back to Java (b9ce724)
- remove overly aggressive overlay failure notifications (9dc62a2)
- migrate from deck_legend_control to layer_legend_widget for shiny_deckgl 1.9 (c5229b1)
- **ui:** address code review findings across 11 UI files (854a7da)
- **engine:** apply larva mortality as full per-cohort rate, fix output headers (b6045bf)
- **engine:** larva mortality rate is per-timestep, not annual (c96ceaf)
- enable both biomass+abundance flags in distribution tests (4138a1e)
- warn on malformed fishing seasonality, narrow movement map exception (0aa569a)
- use utf-8 encoding with replace fallback, warn on unparseable lines (2aea6f1)
- narrow except Exception blocks to specific exceptions in UI sync code (279f470)
- register temp dir cleanup for scenario export downloads (85522d6)
- map None returncode to -1 instead of masking as success (4ddbe9b)
- remove duplicate growth_class parsing block (b74e902)
- remove double 1e-6 conversion on fishing yield (eacc5f9)
- remove duplicate age/size distribution block that overwrote correct results (8a7eaa4)
- **engine:** config-dir-aware file resolution + v4 fisheries seasonality (97ac300)

### Other

- v0.5.0 — full EEC parity (14/14) via unified predation architecture (c8de5fb)
- restore larva mortality /n_dt_per_year division (2531801)

### Tests

- add test_engine_accessibility.py for predation accessibility matrix (772b655)

## [0.4.0] - 2026-03-20

### Features

- **engine:** complete all 5 bioen output CSVs (ingestion, maintenance, rho, sizeInf) (d706678)
- **engine:** wire per-species RNG into movement and predation consumers (0ccf430)
- **engine:** wire O2 forcing into bioenergetic step (39c1d8b)
- **engine:** wire bioen reproduction — gonad-weight egg schools (96715bc)
- **engine:** add bioenergetic output CSV writers (a72c434)
- **engine:** wire bioenergetic processes into simulation loop (043c934)
- **engine:** parse all bioenergetic config keys + expand schema (a445816)
- **engine:** add energy budget process + bioen SchoolState fields (73cdb7a)
- **engine:** write size/age distribution CSV outputs matching Java format (c6f392f)
- **engine:** add bioen starvation with gonad-buffer deficit (ade8a5e)
- **engine:** add gonad-weight egg production for bioen mode (58725af)
- **engine:** add PhysicalData loader for temperature/oxygen forcing (3ee42c7)
- **engine:** add bioenergetic allometric ingestion cap (96ac549)
- **engine:** add Johnson thermal performance curve and O2 dose-response (92c48b4)
- **engine:** wire Gompertz growth dispatch with config-driven class selection (bfb5fe3)
- **engine:** add size/age distribution binning to StepOutput (4cb1db1)
- **engine:** add per-species deterministic RNG via SeedSequence (297fb01)
- **engine:** fix growth classname enum, add Gompertz schema fields, parse growth_class (fb74610)
- **engine:** add diet tracking to Numba predation path + extract shared helpers (df759f6)
- **engine:** Phases 4-5 — random patch, output frequency, yield, diet, NetCDF, step0 (65b3e21)
- **engine:** Phase 3 — spawning normalization, time-varying mortality, egg placement, seeding max (7120c2c)
- **engine:** Phase 2 — fishery seasonality, selectivity types, v3 scenarios, MPA, discards (6220046)
- **engine:** Phase 1 EEC parity fixes — maturity age, spatial fishing, lmax, resource multiplier, TL computation (d5b737e)
- **engine:** add output.cutoff.age filter — exclude young-of-year from biomass output (3360a3e)
- **engine:** per-cell per-school interleaved mortality matching Java computeMortality() (e2873e8)
- **engine:** add fisheries + stage-indexed accessibility + egg weight fix (92d97c3)
- **engine:** add feeding stages (B2) + diet matrix output (C2) (db1a93e)
- **engine:** add incoming flux — external biomass injection from CSV time-series (55ffc7d)
- **engine:** integrate map-based movement into simulation loop (42391e5)
- **engine:** add _map_move_school — per-school map-based movement algorithm (6bf860d)
- **engine:** add MovementMapSet — CSV map loading and index_maps construction (2861214)
- **engine:** update predation kernel for 2D size ratios with feeding stages (e00a9c5)
- **engine:** integrate background species into simulation loop with inject/strip pattern (0ed3e6e)
- **engine:** skip starvation/fishing/additional mortality for background species (095c9b4)
- **engine:** add NetCDF forcing support to BackgroundState (fa5b218)
- **engine:** add BackgroundState with uniform forcing and school generation (732ba88)
- add validation pipeline script (scripts/validate_engines.py) (894b2dd)
- add validation pipeline script comparing Python vs Java engines (ea28c70)
- **engine:** add fishing selectivity (knife-edge and sigmoid) (b1a2f64)
- **engine:** add mortality rate CSV output per species (20e0194)
- **engine:** interleaved mortality with egg release and cause shuffling (4ab8877)
- **engine:** add egg_retained field for retain/release mechanism (c5f1c44)
- **engine:** integrate resource species as prey in predation (37848f2)
- **engine:** implement LTL resource species with NetCDF forcing (786e01e)
- **engine:** add accessibility matrix, spawning seasons, real grid, fishing key fix for ecosystem parity (04c3718)
- **engine:** PythonEngine.run() now writes output CSV files (9b22ca8)
- **engine:** add CSV output writer matching Java format (5611ec1)
- **engine:** wire starvation and fishing into mortality sub-timestep loop (a8fee19)
- **engine:** add fishing mortality by rate (aeef341)
- **engine:** add starvation mortality with lagged predation success (241ebb1)
- **engine:** add starvation and fishing parameters to EngineConfig (6f22186)
- **engine:** wire predation into mortality sub-timestep loop (ba0b4fc)
- **engine:** add size-based predation with asynchronous prey updates (c4e59be)
- **engine:** add predation size ratio parameters to EngineConfig (59973b6)
- **engine:** wire movement and out-of-domain mortality into simulation loop (7447356)
- **engine:** add random walk movement and out-of-domain mortality (51d117a)
- **engine:** add movement parameters to EngineConfig (890c43e)
- **engine:** add larva mortality for eggs with separate rate (6046b7a)
- **engine:** wire reproduction and seeded initialization into simulation loop (7b2907d)
- **engine:** add reproduction process with egg production and seeding (6c0cd04)
- **engine:** add reproduction parameters to EngineConfig (0871d84)
- **engine:** wire growth and mortality processes into simulation loop (9ce5585)
- **engine:** add additional mortality and aging mortality processes (64e5a63)
- **engine:** add Gompertz expected length function (a60682d)
- **engine:** add Von Bertalanffy growth with predation-success gating (afa7fb9)
- **engine:** add delta_lmax_factor and additional_mortality_rate to EngineConfig (8cf2d5e)
- **engine:** wire PythonEngine.run() to simulation loop (29b77b1)
- **engine:** add simulation loop skeleton with stub processes and output collection (ca28ad2)
- **engine:** add ResourceState placeholder for LTL forcing (90571c5)
- **engine:** add Grid class with NetCDF loading and cell adjacency (01b37e1)
- **engine:** add EngineConfig for typed parameter extraction from flat config (15c7fbc)
- **engine:** add SchoolState SoA dataclass with create/replace/append/compact (8c460d8)
- **engine:** add Engine protocol with PythonEngine and JavaEngine stubs (bbb9de0)

### Bug Fixes

- remove duplicate StepOutput fields from merge artifact (70dcb78)
- resolve merge conflicts from parallel Tasks 5-12 + fix dataclass field ordering (d4d0179)
- **engine:** update stale diet test name + document predation helpers (69a4a01)
- **engine:** eggs cannot feed first timestep + fix larva mortality double-counting (719d950)
- **engine:** egg_weight_override config is in grams, convert to tonnes (* 1e-6) (b2b5849)
- **engine:** convert weight to tonnes (Java convention), remove double seeding (4864b26)
- **engine:** correct predation appetite (/n_dt_per_year), cap pred_success_rate, fix starvation rate (d23b31a)
- **engine:** resolve movement map and forcing file paths relative to config directory (14b4a96)
- **engine:** EEC config compatibility — resource keys, path resolution, grid dims, egg weight (c0b5f72)
- **engine:** correct config key case for size ratios + add selectivity (f926b1d)
- **engine:** correct predation size-ratio logic + resource species (52d50a4)
- **engine:** handle global simulation.nschool config key (77a6d70)
- **engine:** match Java behavior — egg skip, reproduction units, tests (2ca9b68)
- **engine:** correct mortality rate to match Java — D = M/(ndt*subdt) (403a4ee)
- **engine:** address code review — csr==1.0 bug, test precision, subdt guard (2694ed6)
- **engine:** address code review issues — shadowing, file leak, docs (d152fb6)
- defer shiny_deckgl import to avoid test collection errors (77e1f3c)
- filter non-spatial files from grid overlay dropdown (df30a58)
- collapsed panels now release space to siblings via CSS Grid override (89d54ea)

### Performance

- **engine:** Numba JIT predation + batch RNG — 5.9x faster, beats Java (202dbd4)

### Chores

- lint fixes + spatial bioen TODO + final cleanup (8e7a2c8)
- fix lint issues from sprint (unused imports + vars) (51aa833)
- remove worktree cache (348dca5)

### Documentation

- add engine gap closure implementation plan (ba001fa)
- fix 4 review issues in gap closure spec (26767e2)
- add engine gap closure design spec (6cee3b1)
- add Java parity sprint implementation plan (40e1de4)
- fix review round 3 — Linear references, starvation gonad-flush order (2542304)
- fix spec review issues — growth classnames, Gompertz keys, starvation formula (134edd2)
- add Java parity sprint design spec (676cff8)
- add comprehensive Java parity roadmap — 7 phases, 37 items (2679dc1)
- add B1 map movement implementation plan with review corrections (8a542b8)
- add B1 map movement design spec with Java parity fixes (38a7ba5)
- add B2 feeding stages implementation plan with review corrections (0afb2da)
- fix feeding stages spec — trailing semicolons, absent key default, indexing contract (ee9337d)
- add B2 feeding stages design spec (61d49dc)
- add complete implementation plan for remaining engine gaps (5506cfe)
- add Phase 4 movement implementation plan (9ac8dea)
- fix Phase 2 plan review issues (Gompertz, test fix, precision) (6a54f9c)
- add Python engine Phase 2 implementation plan (291a15f)
- fix 4 plan review issues (abundance snapshot, adjacency, biomass, TDD) (70b3bf9)
- add Python engine Phase 1 implementation plan (5d371b8)
- fix 3 remaining spec review issues (rev 3) (8c01f42)
- revise Python engine spec addressing 20 review issues (02b2038)
- add Python engine design specification (fb612f1)

### Other

- v0.4.0 (38c5b85)
- Merge branch 'worktree-agent-a161e3b4' (88608dc)
- Merge branch 'worktree-agent-a9a2eeb3' (487197a)
- Merge branch 'worktree-agent-abebd8cd' (e212eae)
- Merge branch 'worktree-agent-a4152a1a' (e27a122)
- Merge branch 'worktree-agent-a562217f' (bd77ce3)
- Merge branch 'worktree-agent-acf2a5a6' (a7ead1f)
- Merge branch 'worktree-agent-a87d9cd8' (Task 6: per-species RNG) (d08551a)
- Merge branch 'worktree-agent-aab44ec4' (Task 4: distribution output binning) (2f6faed)
- Merge branch 'worktree-agent-acf5f81d' (48e74b6)
- Merge C1+B3: mortality CSV output + fishing selectivity (149e189)
- Merge A1+A2+A3: interleaved mortality, egg retain/release, TL placeholder (e5dd2bb)
- Merge resource species + predation fix (eba886c)
- Merge performance optimization — Numba predation, 1.9x faster than Java (48bd1ba)
- Merge ecosystem parity — accessibility, spawning seasons, real grid (30130fb)
- Merge branch 'feature/python-engine-phase7' — output writer (64684ff)
- Merge branch 'feature/python-engine-phase6' — fishing + starvation (0cbfd78)
- Merge branch 'feature/python-engine-phase5' — predation (767fb1f)
- Merge branch 'feature/python-engine-phase4' — movement (1069d17)
- Merge branch 'feature/python-engine-phase3' — reproduction + initialization (1ce4437)
- Merge branch 'feature/python-engine-phase2' — growth + mortality (2db4f3b)
- Merge branch 'feature/python-engine-phase1' — Python engine foundation (915f075)

### Refactoring

- extract _make_bioen_config() for cross-test imports (ba64236)
- **engine:** delegate _mortality to new orchestrator module (3a44ae9)
- unify nav and panel collapse with same expand-tab pattern (43afd43)

### Tests

- **engine:** add background species stage + integration tests for feeding stages (7d8ebf0)
- **engine:** add integration tests for background species (5f29a23)
- **engine:** add background species predation participation tests (157b9bf)
- add Tier 1.5 Java comparison tests for growth and mortality (5efed58)
- add 503 full-model integration tests for all example studies (7b06427)

## [0.3.0] - 2026-03-15

### Features

- add OSMOSE Model scientific description to Help modal (bb0d019)

### Bug Fixes

- move popover init to end of body, use setInterval polling (d5eeadc)
- remove Show Help button, fix Bootstrap 5 popover initialization (7ad76d0)

### Documentation

- expand OSMOSE Model help with extensions, applications, and 30+ references (b9f699d)

### Other

- v0.3.0 — OSMOSE scientific docs, tooltip fix, Show Help removal (befbd0b)

## [0.2.0] - 2026-03-14

### Features

- add movement animation controls and cache logic to grid page (a2e478e)
- add build_movement_cache, MOVEMENT_PALETTE, and list_movement_species (19881ac)
- add derive_map_label and parse_movement_steps helpers (00db43f)
- make advanced page panel collapsible (a4e1d92)
- make scenarios page panel collapsible (3-column layout) (2dab172)
- make calibration page panel collapsible (01c4fce)
- make results page panel collapsible (7268a1c)
- make run page panel collapsible (5593c49)
- make movement page panel collapsible (93e84c4)
- make fishing page panel collapsible (ffdf6db)
- make forcing page panel collapsible (290c5c5)
- make setup page panel collapsible (9d77760)
- make grid page left panel collapsible (bc7f85f)
- add fullscreen toggle widget to grid preview map (f3d86a6)
- add CSS for collapsible nav sidebar, split-layout panels, and light theme (a89032e)
- add nav collapse JS + hamburger toggle button (a6469a4)
- add collapsible panel helpers (card header + expand tab) (97a7256)
- add layered tooltip system with hover popovers and Show Help toggle (8728cd4)
- add spatial file overlay selector to grid preview (c9376f0)
- smart results directory defaults and auto-load on tab switch (0079d4d)
- add manual tooltip text for non-schema UI fields (cd209de)
- populate species filter from config species names (218a5cd)
- wire spreadsheet species table into Setup page (37f866f)
- wire spreadsheet LTL table into Forcing page (f4e3b69)
- add render_species_table() spreadsheet component (258531f)
- rework example loading with Load button and config header (c1c0628)
- replace dirty_banner with persistent config header bar (2901508)
- add config_name, species_names, results_loaded to AppState (4d3dfb2)
- add UI scenario loading tests and fix demo dispatch bug (8971cbf)
- add eec_full demo from GhassenH/OSMOSE_EEC research config (6d907a8)
- add EEC and minimal demo scenarios (aa19d81)
- strengthen validator with schema-aware file refs, resource checks, and enum validation (95cbdfa)
- expand results parser with fishery, bioenergetics, and distribution outputs (966b106)
- expose Java CLI flags (-update, -verbose, -quiet, -Xmx) in runner (f1ecd44)
- expand version migration to cover v3.1 through v4.3.3 (57bb093)
- add CSV export, ensemble CI bands, and run comparison to Results page (e44651e)
- add Compare Runs tab with grouped bar chart and config diff table (1d37a4c)
- add compare_runs_multi() for N-way config diff (e79718c)
- add ensemble toggle with CI band rendering to Results page (796c317)
- add ensemble replicate aggregation with mean + 95% CI (c1fb675)
- add CSV download button to Results page (08f4ab2)
- add export_dataframe() to OsmoseResults for unified data export (5c40aa9)
- add food web Sankey, run comparison, and species dashboard charts (3fa57fa)
- add CLI for batch runs, validation, and reporting (29cac2d)
- add pyright type checking to CI (d26e40f)
- run history tracking with JSON records (a13615d)
- Jinja2 HTML reports with custom template support (84a0222)
- validate config before run, block on errors (f24c81a)
- responsive modals and mobile nav at 768px breakpoint (351d10d)
- field-level validation with min/max bounds (c3b2398)
- unsaved changes warning with dirty state tracking (731c85f)
- global loading overlay for long operations (7b15f5c)
- atomic scenario writes with backup-rename pattern (233e78f)
- add DataFrame column guards to analysis and plotting (74ecbff)
- add configurable timeout to OsmoseRunner.run() (2febf27)
- rename app to OSMOPY, add JAR selector, fix grid preview after demo load (1162847)

### Bug Fixes

- narrow exception handling, isolate reactive deps, add debug logging (2a8c0a2)
- nav hamburger stays visible when collapsed, expand tab uses sticky positioning (18410da)
- split movement controls into separate render output (bd50c95)
- broaden exception catches for SilentException, polish UI layout (c6e8ca4)
- add temp directory cleanup on startup and shutdown (7a40d1a)
- remove non-functional beforeunload guard (5cdffc2)
- add 10MB file size limit to config reader (778e370)
- reject path traversal in output_dir input (d24fee2)
- add path traversal check on overlay file paths (0022079)
- skip corrupt history files instead of crashing listing (0825ee7)
- notify user when run history save fails (f88dd55)
- protect stream reader from progress callback exceptions (70cda68)
- narrow surrogate/sensitivity sample exceptions to expected types (bdc700a)
- guard all nspecies int() parsing in UI pages against non-numeric values (9639dc7)
- batch low-severity fixes — cancel safety, test seeds, version fallback (d9b9d0f)
- add NetCDF cache eviction on directory switch (M14) (87411b1)
- raise error when csv_maps_to_netcdf finds no valid files (M18) (182340e)
- log unparseable config lines instead of silently skipping (M16) (6fe0bc5)
- validate calibration override keys against OSMOSE pattern (M2) (3b325cb)
- show sp0 values for indexed fields in advanced param table (M26) (17d03dd)
- use atomic write pattern for config files (M24) (dac571e)
- use load_trigger for advanced param_table rendering (M6) (a3c1b01)
- log warning for unknown export_dataframe output types (M27) (da237dd)
- catch ValueError in grid input parsing (M25) (d33da99)
- restore scenario backup on save failure (M23) (d7e7e43)
- show notification when results download has no data (M19) (e195b87)
- skip corrupt scenario files in listing instead of crashing (M17) (248251d)
- validate config reader sub-file paths stay within config dir (M4) (75d7e67)
- narrow get_theme_mode exception to expected types (2b7d026)
- add error handling to results loading (H6) (0abbe1e)
- narrow grid file-loading exceptions to expected types (H7-H10) (2d9e19b)
- handle 1D lat/lon coordinate arrays in grid preview (H18) (0ed761f)
- handle non-integer nspecies values gracefully (H17) (8889297)
- prevent results loading race condition (H4, M8) (3d8ba56)
- use load_trigger for calibration checkbox rendering (H3) (4e241fb)
- log run history save failures at warning level (H13) (584f772)
- narrow ensemble mode exception to expected types (H11) (6ec573f)
- HTML escape DataFrame output in report template (H2) (2a64c06)
- validate java_opts against safe JVM flag whitelist (4ce57d8)
- replace vacuous 'or True' assertion with real check (bcfc474)
- add path traversal protection to scenarios and history (af2bd9e)
- narrow calibration exceptions — propagate unexpected errors (7a172ef)
- add top-level exception handler to surrogate calibration thread (6980176)
- add state guards to scenario load and config import (b530c3b)
- thread-safe calibration communication via message queue (9afcce8)
- grid overlay selector scans config directly + CSV overlay support (0ab86c4)
- remove unused COLOR_MUTED import in results page (b1a1ca4)
- write flat master config to avoid duplicate parameters in run directory (547145b)
- copy entire source config directory before writing run config (270f71f)
- pass registry to check_file_references to avoid false file-path matches (769f7a9)
- config reader strips trailing separators, validator handles null/multi-values (6604dc4)
- grid preview working with deck.gl for both regular and NetCDF grids (ffad986)
- resolve dirty banner regression, button wrapping, and UI polish (202f244)
- update bundled example configs for OSMOSE v4.3.3 compatibility (badd5d5)
- migrate bundled eec_full config to v4.3.3 for out-of-box run (2d4270b)
- update remaining grid.ncolumn/nline references to nlon/nlat (b900c7a)
- align mortality schema keys and writer routing to post-v4.2.5 names (3254bf5)
- update grid schema keys to post-v3.3.3 names (nlon/nlat) (c9ebc04)
- update grid schema and example configs for OSMOSE 4.x parameter names (a2d6846)
- handle stale backup in atomic scenario save (9457257)
- harden timeout handling — race condition guard and cleaner UI wiring (b214655)
- require objective_fn in MultiPhaseCalibrator, fix work_dir type (15ad9c9)
- handle missing dirs, reject unsupported report format, rename summary_table (c051ee7)
- remove debug prints, fix temp file safety, update GitHub URL (c9d5b40)
- add cycle guard and missing sub-file warning to config reader (94e4425)
- standardize size spectrum slope to log10 (ecological convention) (fff4222)
- add reactive.isolate() to prevent infinite loop risks (8eb1b39)
- replace silent except:pass with logging and user notifications (f073a4a)

### Performance

- batch forcing page config updates (760445e)
- lazy-load result types on demand instead of all 16 eagerly (M13) (5b51c67)
- batch species parameter sync to single config update (M7) (0b658cc)
- pre-compile regexes and cache match_field lookups in registry (c4f0ee2)

### CI/CD

- coverage threshold, Python matrix, Docker smoke test, HEALTHCHECK (e321acc)

### Chores

- remove unused imports in new test files (e4e41d0)
- fix lint — move constant after imports, remove unused imports (0baa56e)
- remove unused imports in test_sync_config_pages (d851823)
- add pre-commit hooks for ruff (e0bed46)

### Documentation

- add movement visualization implementation plan (2f33f64)
- add movement visualization design spec (1582c0e)
- update codebase fixes plan with second-round analysis findings (7967985)
- add codebase fixes implementation plan (25 tasks, 4 phases) (7017df8)
- add comprehensive codebase analysis findings (2aa8daf)
- add codebase analysis design spec (c27f519)
- add UI improvements spec and implementation plan (02700fc)
- add results workflow implementation plan (7 tasks) (e4bbd4f)
- fix spec review issues — complete type map, N-way comparison, alignment strategy (b436c53)
- add results workflow enhancement design spec (1edec92)

### Other

- v0.2.0 — movement visualization, collapsible panels, codebase fixes (639ca24)
- Merge branch 'enhancement-sprint-2026-03-11': 18 enhancement tasks (production hardening, UX polish, new capabilities, dev experience) (ee1520c)

### Refactoring

- use explicit result method mapping instead of getattr (5907356)
- extract pure helper functions from grid.py into grid_helpers.py (M9, M12) (2056c84)
- standardize logging initialization across modules (M10) (97671b4)
- split calibration page into layout, handlers, and charts (1ae7fe0)
- extract Plotly theme to osmose/plotly_theme.py (c6d4888)
- consolidate shared test fixtures into conftest.py (24757b7)
- deduplicate RMSE objectives and narrow theme except clauses (55861a7)
- consolidate registry construction into schema.__init__ (75e10f8)

### Styling

- add movement animation controls CSS (e1dc098)
- format results workflow code (75d3c0b)

### Tests

- remove brittle source-inspection tests (54f7e27)
- add tests for copy_data_files and grid_helpers (H16 gaps) (3cd0e42)
- add Playwright E2E tests for reactive UI behavior (H16) (07c20ef)
- add CLI cmd_run and cmd_report error case tests (M29) (f8f36a9)
- replace vacuous hasattr tests with behavioral checks (M30) (0cc580a)
- add path traversal rejection test for scenario import (M28) (e6b56bc)
- add parallel calibration error isolation test (M20) (593bb49)
- replace brittle source inspection test with behavioral check (M21) (e2eddf4)
- add NaN/malformed input edge case tests (H14, H15) (41d2c0e)
- add comprehensive tests for all collapsible page panels (342bd64)
- add column guard tests for plotting functions (85382a9)

## [0.1.0] - 2026-03-07

### Features

- full R parity — 8 new modules, 22 output parsers, 146 new tests (33c798d)
- add Nautical Observatory theme with custom CSS and plotly template (be166cd)
- switch navigation to left-side pill list with grouped sections (6979b5f)
- add input validation tooltips showing field constraints (f6d4151)
- add scenario bulk export/import as ZIP (72f946e)
- add structured logging module with console output (007ca9b)
- add play/pause animation to spatial map time slider (b3a11e8)
- add config import preview with diff before merge (0694083)
- parallelize calibration objective evaluation (b2302cc)
- add per-generation progress callback for NSGA-II calibration (55d7af2)
- wire GP surrogate calibration into UI (ebbbf7c)
- wire Advanced page import/export handlers (cc6f4ad)
- wire Calibration Start/Stop/Sensitivity handlers (fbd5bde)
- sync Run page JAR path to AppState (d467c71)
- wire Grid, Forcing, Fishing, Movement input syncing (c89919a)
- wire Setup page input syncing to AppState (a1a6daf)
- add jar_path to AppState and sync_inputs utility (92f153b)
- add grid preview map with plotly (ecb4008)
- wire Calibration page with dynamic params and plotly charts (d9ecb33)
- wire Scenarios page to ScenarioManager (2c2d497)
- wire Results page with plotly charts (d7dd5b7)
- wire Run page buttons to OsmoseRunner (9513946)
- wire AppState into all page servers (93f1555)
- add AppState shared reactive state module (dec42bf)
- add Dockerfile, integration tests, and wire all UI pages (bde866a)
- add calibration module and UI (0e415b4)
- add run, results, scenarios, and advanced config pages (e6d51bc)
- add Shiny UI shell with param form and config pages (e258773)
- add runner, results reader, and scenario management (32ede0e)
- add config I/O (reader, writer) with roundtrip tests (3c0b5db)
- add schema-driven parameter system (base classes, registry, all modules) (eb74646)
- scaffold osmose-python project with dependencies (74c9b34)

### Performance

- lazy-import heavy dependencies for faster startup (d317b0f)

### CI/CD

- add GitHub Actions workflow for lint and test (76b276e)

### Chores

- format Phase 2 UI code with ruff (a1a7752)
- add shinywidgets dependency for plotly integration (67dbaa5)
- add CLAUDE.md, README, LICENSE, docs, lint fixes, formatting, and gitignore (cb7e743)

### Documentation

- add pill list refactoring plan (9578ce4)
- Phase 3 implementation plan (8 tasks) (eeaa0cc)
- Phase 3 design — complete Phase 2 gaps (ea59d91)
- add Phase 2 implementation plan (10 tasks) (dd2a9b4)
- add Phase 2 UI wiring design + deployment script (0b475d5)

### Refactoring

- extract inline styles to ui/styles.py constants (61684f7)
- remove redundant page_fluid wrappers from page UI functions (30f9da6)

### Tests

- add app structure tests for pill list navigation (f31335b)
- achieve 100% code coverage across all modules (a1b4e91)
