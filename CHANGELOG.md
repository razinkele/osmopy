# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/), generated from [Conventional Commits](https://www.conventionalcommits.org/).

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
