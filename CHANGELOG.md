# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/), generated from [Conventional Commits](https://www.conventionalcommits.org/).

## [1.1.0] - 2026-06-26

### Features

- **hpc:** Apptainer definition (Python+Java+numba, JAR + baked numba cache, read-only-friendly) (87c3e34)
- **cli:** osmose run --jar defaults to $OSMOSE_JAR (container-friendly) (bb1493d)
- **ui:** show reference-point source (model/ICES/user) on the Fisheries page (95d652a)
- **fmsy:** load model reference sidecar (precedence user>model>ICES; conditional label) (716d2db)
- **fmsy:** CLI to compute + write the model reference-point sidecar (d69ae70)
- **fmsy:** yield-vs-F sweep runner + compute_model_reference_points (ProcessPool, realized-F) (ff336e3)
- **fmsy:** derive_reference_points (peak/boundary/multi-peak/B0/Blim, pure) (ae3b680)
- **fmsy:** fishing-override resolver (mode detection + species->fishery map) (3bf5c0c)
- **ui:** indicative Fisheries stock-status page (Kobe + F/Fmsy + F/M bars) (e536b03)
- **plotting:** indicative Kobe + B/Bmsy & F/Fmsy ratio timeseries (b93c282)
- **validation:** indicative stock-status computation (SSB/Bmsy, exploited-stage F/Fmsy) (12d2b93)
- add fisheries_reference.py — per-species reference-point resolver (Task 4) (908e6e2)
- **fisheries:** annual_by_year aggregator (cadence-correct, absolute-year) (deb4df3)
- **engine:** SSB CSV/in-memory/NetCDF writers + results.ssb reader (4b3086b)
- **engine:** SSB collector + config flags + StepOutput (parity-safe output) (a9fe0ec)
- **output:** mark yieldN/meanSize produced; regression sweep green (9d76477)
- **output:** yieldN/meanSize NetCDF writer + run-path wiring + source=netcdf reader (1aaf672)
- **output:** yieldN/meanSize CSV writers + in-memory results wiring (82fa38a)
- **output:** yieldN/meanSize collectors + StepOutput fields + accumulation (175f00d)
- **output:** config flags + schema key for yieldN/meanSize (3d00653)

### Bug Fixes

- **hpc:** harden build-provided-JAR copy under set -eu (store glob result in a var) (cb9a531)
- **calibrate:** honor OSMOSE_RESULTS_DIR (container-friendly results dir) (cf85484)
- **fmsy-cli:** --grid nargs=+ (reject empty), honest n_years estimate, top-level import (0180e21)
- **ui:** wire ices_snapshot_dir, drop dead _refs, sanitize input IDs, log SSB-hint errors, add Results links (4a0e7d0)
- **validation:** narrow _exploited_f_by_year catch (drop AttributeError; stub carries output_dir) (07795aa)
- **test:** Task 3 test self-contained — drop premature stock_status/ReferencePoint imports (407858d)
- **engine:** gate egg predation on the released fraction (Numba + Python paths) (94f1bfb)
- **engine:** apply fleet-effort on the pure-Python fishing fallback path (41337c9)
- **calibration:** let _worker_eval propagate unexpected errors (no inf-swallow) (8cef769)
- **engine:** suppress standard starvation on the Numba path for bioen (c273c8f)
- **engine:** stop double-counting bioen starvation mortality (457ac55)

### Performance

- **engine:** vectorize movement method masks + bincount diet aggregation (c99097c)

### CI/CD

- **hpc:** build the Apptainer image with --fakeroot + smoke (CLI/engine/numba-cache/java/sweep) (be4f19f)

### Documentation

- **hpc:** fix SLURM config paths (real masters) + drop --jar under --contain (env default) (5e57de1)
- **hpc:** Apptainer build + run guide (read-only rule, SLURM job array, smoke) (557e2b8)
- implementation plan for OSMOSE HPC Apptainer container (3c9fae7)
- revise HPC Apptainer spec per in-loop review (3 reviewers) (d058fa8)
- spec for OSMOSE HPC Apptainer/Singularity container (82fb7b7)
- **fmsy:** clarify _FORCE_OUTPUTS — only output.ssb.enabled gates in-memory; yield flag inert (6564063)
- fix model-refpoints plan per apply-and-run workflow review (21 findings) (a1d6cf1)
- implementation plan for model-internal fishery reference points (188a3b3)
- revise model-refpoints spec per in-loop workflow review (19 findings, 5 critical) (e367ba7)
- spec for model-internal fishery reference points (Fmsy/Bmsy/Blim sweep) (3488090)
- fix fisheries plan per in-loop workflow review (9 findings) (8788990)
- implementation plan for fisheries stock-status diagnostics (e5dc3f6)
- revise fisheries spec per round-3 multi-angle review (22 findings) (b923bae)
- reframe fisheries spec per deep literature review (indicative; B=user Bmsy; ICES Fmsy only) (d728c31)
- revise fisheries stock-status spec per in-loop workflow review (77afc6e)
- spec for fisheries stock-status diagnostics (Kobe / B-ref / F-Fmsy) (07b8f79)
- split semicolon-joined test statements (ruff E702) in yieldN/meanSize plan (1cad736)
- fix yieldN/meanSize plan per multi-angle workflow review (4b07d1e)
- implementation plan for Python-engine yieldN + meanSize (CSV + NetCDF) (799c71a)
- revise yieldN/meanSize spec per in-loop review (b56b0f5)
- spec for Python-engine yieldN + meanSize outputs (CSV + NetCDF) (5d0bb80)
- **perf:** fix RNG-spike plan per in-loop workflow review (e6f2bd8)
- **perf:** implementation plan for Stage-0 RNG-reproduction spike (d2c7191)
- **perf:** spec for Stage-0 RNG-reproduction feasibility spike (0270755)
- **spike:** task-6 post-review fixes note (run_all defaults + boundary_probe hoist) (d65ef8c)
- **spike:** update task-6 report with run_all results (1dba643)
- **spike:** task-6 boundary-free bench report (cfd867b)
- **spike:** task-4 C kernel build report (7812235)
- **perf:** round-2 plan clarifications (non-blocking) (5f55b78)
- **perf:** fix kernel-spike plan per in-loop review round 1 (6c7aeab)
- **perf:** implementation plan for native-predation-kernel spike (7b93119)
- **perf:** revise kernel-spike spec per in-loop review (2766c05)
- **perf:** spec for native predation-kernel feasibility spike (83d65e6)
- round-2 plan review (execution-ready) + fix egg-test size-ratio keys (7bcbe76)
- rework remediation plan per 5-angle plan-review workflow (24 defects) (e44dfb6)
- implementation plan for high-findings remediation (08f16b7)
- resolve high-findings spec open questions (EEC/Java parity, keep Fix 2, fallback now) (6007478)
- rework high-findings spec against the production (Numba) dispatch path (b76aa89)
- design spec for deep-review high-findings remediation (f06190b)
- **readme:** fix dots screenshot (was a heatmap) + clearer caption (b163fa6)
- **readme:** add live-movement screenshots (heatmap + dots) from Baltic (90c2807)

### Other

- OSMOSE HPC Apptainer/Singularity container (ddc0d2c)
- model-internal fishery reference points (Fmsy/Bmsy/Blim sweep) (c78c15d)
- indicative fisheries stock-status diagnostics (ede0a77)
- Python-engine yieldN + meanSize outputs (CSV + NetCDF) (af19fa0)
- Merge Stage-0 RNG-reproduction feasibility spike (dbca0e1)
- **rng:** orchestrator + RNG-reproduction feasibility artifact + verdict (a552a31)
- **rng:** C-vs-Numba per-cell RNG-gen speed probe (3f42aff)
- **rng:** bit-exact parity gate (C vs njit oracle across n x seed grid) (57081bf)
- **rng:** C NumPy-legacy MT19937 (seed+permutation+shuffle) + cffi build (a9e9ae7)
- **rng:** scaffold + njit cell-RNG oracle (mirrors mortality.py:1479-1497) (4b2c925)
- Merge engine perf: vectorize movement masks + bincount diet aggregation (c415c19)
- Merge native-predation-kernel feasibility spike (610f517)
- **perf:** gitignore build artifacts + untrack SDD scratch (4a9dbb8)
- **perf:** orchestrator + native-predation-kernel spike artifact + verdict (b1a8d90)
- **perf:** boundary-free throughput bench + boundary-cost probes (4fb9d0d)
- **perf:** C-vs-Numba parity gate (<=1e-12 op-order rounding) (a75b830)
- **perf:** C leaf transcription + cffi build (portable & native) (eed83a6)
- **perf:** leaf-arg reconstruction + p10/p50/p95 cell selection (7a374c5)
- **perf:** capture cell-loop pre-state via monkeypatched parallel kernel (79c31c6)
- **perf:** native-predation harness scaffold + provenance guards (67dfdff)
- Merge fix/high-findings-remediation: deep-review high findings (668021f)

### Tests

- **engine:** recalibrate the two Baltic bands shifted by the egg-retention fix (f2e1834)
- **tutorial:** xfail biomass-pyramid band shifted by the Java-validated egg-retention fix (a4a2a00)
- **engine:** Java cross-check + EEC/BoB parity baselines for egg-retention fix (02b14b0)
- **engine:** xfail type-III FR invariant flipped by the egg-retention equilibrium shift (9b0d9ba)

## [1.0.0] - 2026-06-22

### Features

- **scenarios:** compare in a modal with shared diff table + edge states (95fef3b)
- **config-diff:** shared classify + render component (88138d7)
- **run:** live card collapse = stream gate; heatmap fallback; 0.5s throttle; drop switch/auto-enable (d4027c6)
- **run:** compact busy indicator (corner pill, not full overlay) (ab69f7f)
- **run:** default the UI engine to Python (Java still selectable) (a36f170)
- **run:** choose_live_layer (dots->heatmap fallback) + dot_cap 5000->2000 (acbe0e2)
- **run:** auto-enable live movement for spatial (regular-grid) configs (d363ee1)
- **run:** wire py_threads -> numba.set_num_threads (0=auto), drop dead py_verbosity (54ac7f6)
- **run:** Python progress bar + in-place console line (reuses step_observer) (c7b4bb6)
- **run:** pure run-observer + spatial predicate + progress label (af9617d)
- **forcing:** convert-only CLI (bring-your-own downloaded file) (a2f1a00)
- **forcing:** NetCDF writers + package public API (fee1ec7)
- **forcing:** phy_to_physics temperature/salinity conversion (5787f18)
- **forcing:** bgc_to_ltl 6-group conversion (Mode A/B), grid-general (00c1feb)
- **forcing:** grid-parameterized regrid/resample/mask helpers (a8605c9)
- **run:** capability panel render from describe_engine (65ed363)
- **run:** active-engine indicator slot (ab6e08d)
- **capabilities:** describe_engine for the Java engine + total fallback (26623d5)
- **capabilities:** describe_engine for the Python engine (c1e0e36)
- **capabilities:** EngineCapability dataclass + truthiness helper (3e3814a)
- **ui:** Scenario Wizard — guided New-Scenario flow on the Scenarios page (#82) (a8f3b80)
- **maps:** Map Builder page — author spatial grid maps on the config grid (#81) (2fd868d)
- **config:** canonicalize at validators + allowlist 4.4.0 keys (717934d)
- **engine:** canonicalize config at from_dict + unified-ingestion/new-key reads (1c93a4a)
- **config:** to_target_keys inverse (leaf-scoped, merge non-invertible) + drift guard (882f70f)
- **config:** canonicalize_config wrapper + malformed-version hardening (bc9f1cf)
- **config:** add 4.4.0 migration chain entry + skip-if-target-exists applier (e98b064)
- **config:** lock OSMOSE 4.4.0 rename set (ported from Releases.java $15) (cefad37)
- **engine:** enable meanTL in Baltic config + e2e community-output integration test (cb8d6b8)
- **engine:** write community biomass/abundanceDistribBySize CSVs (1839706)
- **engine:** write 1D meanTL CSV from captured per-species mean TL (915fd89)
- **engine:** capture biomass-weighted per-species meanTL into StepOutput (337179b)
- **engine:** add output.meanTL.enabled config flag (output_meantl) (77914a3)
- **results:** wire Sheldon spectrum, ABC chart + community-metrics panel into Diagnostics (ad43ab3)
- **plotting:** Sheldon NBSS + ABC dominance-curve charts (00ae25e)
- **community-metrics:** community_report orchestrator + markdown formatter (f11c4ba)
- **community-metrics:** Warwick ABC W-statistic + dominance curves (ed3f4e1)
- **community-metrics:** Mean Trophic Level + Marine Trophic Index (1cff03b)
- **community-metrics:** Sheldon NBSS mass spectrum + size diversity + totals (37102a4)
- **community-metrics:** shared helpers (window-mean, species cols, L-W coeffs) (966b884)
- **fishbase-ui:** surface inline bootstrap panel on the species setup page (683fe08)
- **fishbase-ui:** review/apply helpers + inline bootstrap panel server (4568075)
- **fishbase:** TRAIT_MAP + fetch_traits (median/range, Speccode quirk, a/b check) (6420e7f)
- **fishbase:** resolve_species (scientific/common, FB→SLB fallback) (6c0c03e)
- **fishbase:** client skeleton — _load_table with disk cache + typed errors (d1c26a9)
- **run:** make the three Run-page cards body-collapsible (22abf2d)
- **calibration:** select process backend at NSGA-II sites + benchmark switch (7a87c86)
- **calibration:** ProcessPoolExecutor backend for NSGA-II (forkserver, broken-pool recovery) (681bd7f)
- **calibration:** picklable BiomassRMSE/DietDistance objective functors (6cde163)
- **feedback:** header modal + token-gated read API route (insert before catch-all) (602cc3b)
- **ui:** feedback modal + submit handler (appends, no HTTP POST) (6d435ff)
- **feedback:** osmose/feedback store + token check (pure core) (ace5320)
- **ui:** wire startup changelog modal (shiny:connected, once per version) (1fb2a3b)
- **ui:** About modal renders README/Changelog + startup changelog modal (f802f42)
- **docs:** osmose/docs_content loader + changelog parser (9370244)
- **ui:** Parameter Sensitivity Explorer page + app registration (9d4690b)
- **calibration:** persist live sensitivity result via sobol_io (148c874)
- **ui:** make_sobol_tornado horizontal Sobol chart builder (22898ee)
- **calibration:** sobol_io artifact store + pure view helpers (b43baa2)
- **scenario-diff:** render config-diff panel + wire accordion (4685459)
- **scenario-diff:** add _classify_config_diffs classifier (50ff30c)
- **ui:** live movement view on the Run page (map + queue + poll + render) (46be06a)
- **engine:** optional step_observer hook in the Python simulate loop (6aeadb8)
- **live-movement:** add deck.gl heatmap + dots layer builders (b088c0e)
- **live-movement:** add MovementSnapshot + build_snapshot + queue observer (aa171a2)
- **ui:** embed Scenario Diff tab + sub-server in Results page (26ab806)
- **ui:** add scenario_diff tab module (nav panel + server) (782e1de)
- add biomass_long normalizer + make_biomass_overlay chart (fb4c536)
- **ui:** add make_diff_map + shared NaN→None serializer (a6b685a)
- **spatial:** add spatial_diff_2d + grid_latlon for scenario diff (78bb307)
- **calibration:** Pareto solution picker + config export (0ade369)
- **ui:** per-cell time-series panel on Spatial Results (14f0093)
- **spatial:** cell_timeseries backend for per-cell NetCDF extraction (14491a3)
- **ui:** live config validation panel on the Setup page (2a331b6)
- **config:** add summarize_config_validation helper (single source of truth) (dc198d8)
- **ui:** Results Trophic Network sub-tab (pyvis iframe, cached layout, index slider) (3c843ad)
- **trophic:** species_layout fixed positions + make_trophic_network_html pyvis builder (82e338d)
- **trophic:** diet_network_at per-timestep species aggregation (prey-sum, dead-stage-excluded predator mean) (e13ffac)
- **trophic:** pyvis dep + dietMatrix wildcard reader + label/time/universe helpers (7a3adc6)
- **config:** check_config.py CLI for parse diagnostics (2ed40a4)
- **config:** recursive-ref diagnostics + shipped-config regression (2b59546)
- **config:** line-numbered unparseable/empty_key/duplicate_key diagnostics (7261f73)
- **config:** ConfigDiagnostic dataclass + format/has-errors helpers (eaf0f7f)
- **size-spectrum:** compute_size_spectrum.py CLI (945978a)
- **plotting:** make_size_indicator_timeseries trend chart (873269f)
- **size-spectrum:** per-timestep indicator series + markdown report (70ec335)
- **size-spectrum:** compute_size_spectrum + LFI/slope/mean-size indicators (07ed094)
- **size-spectrum:** community by-size reader + reshape/window helpers (09d6dd5)
- **history:** RUN_HISTORY_DIR constant + default_run_history() helper (ba9cc28)
- **ui:** Compare Runs output-delta chart + table (run_delta wired) (908fc2a)
- **ui:** _delta_for_selected helper for Compare Runs output delta (320bd4b)
- **delta:** compare_runs CLI (5c17d7d)
- **delta:** diverging-bar run-delta chart (0e757e0)
- **delta:** markdown delta report (dac5f7e)
- **delta:** SpeciesDelta + run_delta (union, abs/pct, from-zero, ranked) (2a7927f)
- **delta:** per-species windowed-mean normalizer (wide + long output shapes) (bc160fe)
- **fisheries:** fix mortalityRate reader + compute_mortality_balance CLI (b55b15e)
- **fisheries:** F/M bar chart (d046c55)
- **fisheries:** F/M markdown report (7d32137)
- **fisheries:** MortalityBalance + compute_mortality_balance (F/M) (d00dedf)
- **fisheries:** annual_rate aggregation (per-step → annual, windowed) (b48beae)
- **fisheries:** mortalityRate CSV reader (2-row header + trailing comma) (23bd3b7)
- **fr:** shepherd-fr eval mode + FR-on/off process diagnostic (4c1f111)
- **fr:** phase-14 calibration scaffolding + reconstructed phase-13 base (a5846c1)
- **fr:** functional-response branch in numba kernel + njit arg threading (92efe26)
- **fr:** functional-response branch in Python predation kernel + kernel-vs-oracle value test (e909dee)
- **fr:** EngineConfig fr_shape/fr_halfsat 4-layer wiring + fixture fix (ddf84a9)
- **fr:** parse functional-response keys on background-predator path (8eb65f3)
- **fr:** parse functional-response keys with strict validation (focal path) (861610e)
- **fr:** add predator functional-response schema fields (d9465df)
- **tutorial:** pivot substrate to data/baltic/ subset (5 synthetic-3sp T4 attempts BLOCKED) (6a193ea)
- **tutorial:** stub _tutorial_config helper module (f69b435)
- **calibration-ui:** N-other-live-runs disclosure badge with [switch] stub (6523583)
- **calibration-ui:** current best parameters block with bound-distance hints (cb987ee)
- **calibration-ui:** convergence chart shows best-ever reference line from history (44de081)
- **calibration-ui:** per-species ICES proxy table with magnitude factor (3d5432b)
- **calibration-ui:** run header + liveness_state + pure-helper tests (e4dc2f7)
- **calibration-ui:** module-level RESULTS_DIR + scan helpers + LiveSnapshot reactive (3affb6e)
- **calibration:** NSGA-II callback writes checkpoint + save_run; chart-update regression-pinned (98d5596)
- **calibration:** surrogate-DE writes CalibrationCheckpoint per real-eval round (fe97352)
- **calibration:** CMA-ES writes CalibrationCheckpoint each generation (e518c3c)
- **calibration:** DE wires save_run on completion with tempfile 0o600 fallback (167e6f3)
- **calibration:** DE writes CalibrationCheckpoint with main-thread residual re-eval (4d0603c)
- **calibration:** make_banded_objective returns (callable, residuals_accessor) (d8f6631)
- **calibration:** _ObjectiveWrapper captures per-species residuals + sim_biomass (51b5723)
- **calibration:** LiveSnapshot for atomic per-tick scan results (ab9ed59)
- **calibration:** is_live + probe_writable + liveness_state classifier with boundary tests (1cd64b5)
- **calibration:** read_checkpoint with 4-kind discriminator, size guard, invariant-error catch (07c004b)
- **calibration:** write_checkpoint with atomic tmp+rename and numpy coercion (636623e)
- **calibration:** CheckpointReadResult discriminated union with two-sided invariants (e70ca55)
- **calibration:** CalibrationCheckpoint dataclass with 14 __post_init__ invariants (3cabd6c)
- **calibration:** scaffold checkpoint module with default_results_dir (b950754)
- **economics:** DSVM bioeconomic demo + spatial-output bugfix (#47) (cd590c1)
- **validation:** ICES output validation harness (#46) (969351f)

### Bug Fixes

- **config-diff:** use plain-dict impl per spec, drop unused typing imports (97b757e)
- **run:** session.on_ended cancels the run + _session_alive guards (no DestroyedReactiveError cascade) (b00f674)
- **forcing:** narrow phyc/zooc to non-None for pyright in Mode A (7d3abe2)
- complete the incomplete remediation edges (PR-C) (#93) (c0728e7)
- **scenarios:** harden scenario-store deserialization + validation (PR-B) (#92) (701d026)
- **maps:** Map Builder correctness (polygon mask-edit, lonlat edge, blank shape) (#91) (f3846ac)
- **packaging:** working Docker image + runtime CI gate + demo data resolution (#89) (dc65979)
- **config:** config/schema hardening (validate_value coercion + 4.4.0 migration gates) (#88) (286d375)
- **calibration:** one failure policy — a bad candidate scores inf on all backends (#87) (643aac9)
- **engine:** warn loudly on parsed-but-unapplied mortality features (#86) (f5b0fcf)
- PR-4 IO robustness (cancelled-run status + empty-CSV reader) (#85) (922dd59)
- **scenarios:** backup path appends '.bak' instead of replacing the suffix (#84) (ac5bdea)
- **plotting:** apply theme template + title dict to new charts (house style) (19f9f44)
- **community-metrics:** cast pd.to_numeric for pyright (08dd7df)
- **trophic:** parse the Python-engine diet matrix (<pred>_<prey>) — fixes KeyError 'Prey' (888a482)
- **results:** isolate _get_result_data memo cache to stop diet_chart recalc desync (578879d)
- **fishbase:** ruff format + rename e2e to test_e2e_fishbase.py (CI collect-ignore glob) (d184810)
- **fishbase:** pyright-clean reductions (numpy float array + casts for pandas stubs) (cb0071e)
- **fishbase:** defensive column guards in _match_in_db (review finding) (6747143)
- **deploy:** poll the HTTP health check instead of a single early probe (9ed8a7f)
- **deploy:** run prod from a git clone + guard Shiny OTel source extraction (1dcac33)
- **run:** block the Java engine for background-species configs (Baltic is Python-only) (9f45cff)
- **ui:** render diet heatmap from the engine's <predator>_<prey> matrix (9d1359f)
- **calibration:** OSMOSE_RESULTS_DIR override + StateDirectory so prod checkpoints are writable (8b48545)
- **types:** suppress reportPrivateImportUsage for layer_legend_widget under shiny_deckgl v1.9.2 (6182b4b)
- **ci:** declare httpx in [dev] for the feedback-endpoint TestClient tests (003336c)
- **tests:** skip copernicus mcp-env tests when fastmcp/copernicusmarine absent (6fa0769)
- **ui:** fire-and-forget Python run so the live view streams mid-run (476071b)
- **scenario-diff:** resolve pyright errors, show real time in map titles, add changelog (eb7dcc0)
- **ui:** render species-dimmed spatial output in Map/Flat views (81d6609)
- **types:** cast pandas reductions to silence real groupby unions; drop mis-diagnosed pyright pin (93e2413)
- **ci:** pin shiny<1.6 and pyright<1.1.410 to stop unpinned CI drift (6a56528)
- **types:** resolve pyright errors in trophic_network/size_spectrum/fisheries (2ab217a)
- **ui:** keep validation panel full-width above the split layout (aebfb6b)
- **trophic:** don't double-apply threshold (sub-5% slider was silently clamped) (5197f15)
- **ui:** populate Compare Runs selector from run history on Results-tab entry (7215809)
- **ui:** drop vestigial output_dir guards from Compare Runs readers (2958eee)
- **ui:** Run tab + Compare Runs use canonical run-history dir (default_run_history) (b0b96cc)
- **fisheries:** F/M on exploited life stage(s), excluding unfished-stage natural mortality (fec2172)
- **fisheries:** sum F and M across life stages (fishing is on Pre-recruits, not Recruits) (ccbcebb)
- **fr:** diagnostic preserves per-predator calibrated K + seeds=1 warning (b4fa377)
- **fr:** phase-14 raises (not warns) when frozen phase-13 base is missing (fcff417)
- scope bit-exact parity to Python 3.12 and omit JIT modules from coverage (2e9fcc9)
- **pyright:** silence 57 errors only visible against the CI-mirror venv (eda4273)
- resolve pyright errors (28 -> 0) — incl. one real runtime bug (2bc6305)
- **docker:** install git so pip can resolve shiny_deckgl git+https dep (d437597)
- **deps:** correct shiny_deckgl git URL and pin tag that exists (599dd1d)
- **ui:** Results tab now displays after a Run (df48294)
- **tutorial:** preserve Beat 6 edits across re-runs (conditional copytree) + enforce strict-key validation in regression test (75c2a32)
- **calibration:** reject duplicate param_keys + banded_loss-without-targets (8dbe54e)

### Performance

- **engine:** replace np.add.at with np.bincount in _collect_spatial_outputs (P2) (#41) (c59ae85)
- **engine:** hoist predation scratch buffers per-cell (K1/P1) (#40) (255f060)

### CI/CD

- add numba-disabled leg exercising pure-Python engine fallbacks (98d3e53)
- **visual:** target the visual test file so collection skips [dev]-only-dep modules (c1b1c16)
- **visual:** drop ini addopts in workflow (xdist's --dist not in [viztest]) (b2d0721)
- **visual:** runbook + opt-in digest-pinned gate with per-page baseline updates (a342100)
- **actions:** bump checkout/setup-python/upload-artifact to Node 24 majors (4d22db9)
- **type-check:** type-check the 3.13 path via pyright --pythonversion matrix (032a87e)
- install numba in dev deps so tests don't run the slow fallback path (f854978)

### Chores

- gitignore e2e screenshot + calibration-history artifacts (f623fc6)
- **run:** ruff format fixups for run-feedback UI changes (9ef440a)
- **config:** PR1 final gate fixes (001c530)
- **deploy:** silence benign pip noise (root-user warning + leftover ~dist dirs) (c608a06)
- **live-movement:** widen dot-jitter period, drop dead test branch, fix stale comment (dc97ad0)

### Documentation

- amend scenario-diff plan per 3-angle in-loop review (19b48fb)
- implementation plan for scenario-diff polish (19d8e1e)
- round-2 clarity fixes to scenario-diff spec (90e95d0)
- amend scenario-diff spec per 3-angle in-loop review (ce4768d)
- design spec for scenario-diff polish (shared component + modal) (4ba0f51)
- amend hardening plan per in-loop plan review (blocker + 2 major + minor) (380228d)
- implementation plan for deep-review latent-item hardening (dc15a7d)
- amend hardening spec per in-loop review (2 major + minors) (115b562)
- design spec for deep-review latent-item hardening (a1edda0)
- Sphinx + GitHub Pages API reference site (#94) (1f3a37e)
- harden Run-page-rework plan per workflow review (dot_cap override, layer key, stale tests, CancelledError, e2e race) (910e1ab)
- implementation plan for Run-page rework (547d112)
- spec for Run-page rework (python default, compact UI, live-stream crash fix) (be4c323)
- **jar-swap:** retarget resume plan to v4.4.1 (v4.4.0 resource-forcing bug fixed upstream) (d1aee23)
- run-feedback plan 2nd-pass review fixes (drain placement, e2e budget, comments, docstring) (73fa6c3)
- harden run-feedback plan per multi-angle workflow review (025b296)
- implementation plan for Python run-progress feedback (f50cf91)
- spec round-3 fixes (year off-by-one via 1-based done + pure label helper, py_threads min=0) (7bb1816)
- spec round-2 fix (e2e_baltic.py:57 also clicks the live toggle) (809da1d)
- spec round-1 review fixes (progress lifecycle, ndtperyear key, e2e toggle, threads idempotency) (5380229)
- spec for Python run-progress feedback (788c978)
- **forcing:** add osmose/forcing to CLAUDE.md architecture tree (ef6438b)
- harden CMEMS-forcing plan per multi-angle workflow review (984651f)
- plan round-1 review fixes (parity mask, package init, lint scope) (7a40c22)
- implementation plan for CMEMS forcing conversion core (sub-project A) (30b1d4f)
- spec for CMEMS forcing conversion core (sub-project A) (c3c7654)
- plan round-3 fix (update stale e2e comment, don't delete) (de825df)
- plan round-2 polish (stale-ref grep, files header, e2e id note) (416d766)
- plan round-1 review fixes (panel_conditional, e2e, citations) (156dcce)
- implementation plan for Run-page engine-capability transparency (bb15f0c)
- spec for Run-page engine-capability transparency (f227a49)
- deep-review remediation plan (2026-06-20) (#83) (271e2c7)
- OSMOSE 4.4.0 jar-swap resume plan (cross-engine + ICES validation) (#80) (64ba5ec)
- **plan:** PR1 2nd in-loop review fixes (CI/pitfalls/DRY angles) (eab16dd)
- **plan:** PR1 in-loop review fixes (3 angles, all verified) (ed86058)
- **plan:** PR1 (core config migration to 4.4.0) — 8 TDD tasks (9be003e)
- **spec:** config-migration v3 — ground in Releases.java, fix merge direction (1f28b6d)
- **spec:** config-key migration v2 after in-loop review (2 rounds) (a52bdfa)
- **spec:** config-key migration to OSMOSE 4.4.0 (canonical=new, read-aliases) (57a465d)
- in-loop review fixes for community-outputs spec+plan (8be75f7)
- **plan:** Python-engine community outputs (DistribBySize + meanTL) (c47bb31)
- **spec:** Python-engine community outputs (DistribBySize + meanTL) (396ac22)
- **changelog:** community size-spectrum extension (Sheldon NBSS + MTL/MTI/ABC) (b172c43)
- in-loop review fixes for community size-spectrum spec+plan (e3a4cae)
- **plan:** community size-spectrum extension implementation plan (e32bd24)
- **spec:** community size-spectrum extension (Sheldon NBSS + community metrics) (0644eb0)
- **plan:** full Baltic e2e test implementation plan (inline, 3 phases) (98e5fd3)
- **spec:** full Baltic e2e (run + live movement + Results outputs) design (8a3432a)
- **changelog:** add FishBase/SeaLifeBase trait bootstrap to Unreleased (33e00a1)
- **plan:** round-3 review fixes — SLB per-table degrade, cache evict, sentinel filter, multi-candidate picker, inline panel (drop render-in-modal) (2c700bd)
- **plan:** round-2 nit — correct Task 7 e2e header (modal-open smoke, no network) (4eccb08)
- **plan:** round-1 review fixes — pyarrow dep, field.description, Shiny-safe ids, modal reset, busy indicator (b4f6858)
- **plan:** FishBase trait bootstrap implementation plan (8 tasks, TDD) (810db98)
- **spec:** correct FishBase access to Source Cooperative parquet snapshots (4081c8a)
- **spec:** FishBase/SeaLifeBase species-trait bootstrap design (c53dc2f)
- fix visual plan/spec per live-run findings (#main_nav clip, gate-tolerance self-consistency) (369b936)
- revise visual-regression spec+plan per efficacy/maintenance/cost/operability review (b8a9553)
- revise visual-regression plan per in-loop review (upload-artifact@v6, branches/permissions, max_ratio doc) (c5a1241)
- implementation plan for UI visual regression tests (5cc4291)
- revise visual-regression spec per 3-round in-loop review (eb7e125)
- design spec for UI visual regression tests (1c8d73d)
- record shiny 1.6.x as the supported runtime (DEPLOY.md, CHANGELOG) (1e91671)
- implementation plan for the shiny 1.6.3 port (6996297)
- round-2 spec fixes (Bootstrap-match rationale, deploy.sh version-aware, cma np.Inf, floor-vs-lock) (2d71206)
- revise shiny-1.6 spec per round-1 review (6de1284)
- design spec for porting to shiny 1.6.3 everywhere (9d79645)
- DEPLOY.md — restart osmose-shiny.service after every pull (96b43d1)
- round-2 plan review of NSGA-II (blocker + major, verified by running) (729a60e)
- address plan review of NSGA-II speedup (blocker + majors) (ec21408)
- implementation plan for NSGA-II ProcessPoolExecutor speedup (b056407)
- round-4 review of NSGA-II spec (1 major guardrail + nit) (8df47d0)
- round-3 review of NSGA-II spec (1 major + nit) (9579e41)
- round-2 review of NSGA-II spec (blocker + 2 majors) (7e55f5d)
- address round-1 review of NSGA-II process-pool spec (3 majors) (2e13f64)
- spec for NSGA-II ProcessPoolExecutor speedup (8db6640)
- round-2 plan-review fix — read_feedback docstring <=100 cols (b78b7b1)
- plan-review nits for feedback system (all 3 angles converged) (81b169f)
- implementation plan for feedback system (sub-project 2) (436f9c3)
- address round-1 review of feedback-system spec (blocker + 2 majors) (b0c3375)
- spec for feedback system (sub-project 2) (826ac0e)
- freshen README + cut CHANGELOG v0.13.0; bump __version__ (86adb6b)
- plan-review nits for docs-in-app (all 3 angles converged) (e5e8969)
- implementation plan for docs-in-app (sub-project 1) (4acac33)
- round-2 review fix for docs-in-app spec (converged) (5db432b)
- address round-1 review of docs-in-app spec (ab06a22)
- spec for Docs-in-app (sub-project 1) (15b7bf9)
- plan-review fix — structure test asserts unique nav value (d40cf59)
- implementation plan for Parameter Sensitivity Explorer (c6ca9a4)
- round-4 review fixes for sensitivity-explorer spec (converged) (5155f83)
- address round-3 review of sensitivity-explorer spec (917f685)
- address round-2 review of sensitivity-explorer spec (4 majors) (736d985)
- address round-1 review of sensitivity-explorer spec (78b9c45)
- spec for Parameter Sensitivity Explorer page (f92a86e)
- plan-review nits (badge cell format-clean; spec table font-size) (1ca2790)
- implementation plan for config-diff panel (e67b7bf)
- round-2 review fixes for config-diff spec (converged) (42da847)
- address in-loop review of config-diff spec (e5b5149)
- spec for Scenario Diff config-diff panel (1929afd)
- **live-movement:** add CHANGELOG entry; clarify MovementSnapshot.status (3cb20a1)
- **live-movement:** add converged spec + implementation plan (d8644e5)
- **scenario-diff:** add converged spec + implementation plan (a1289a1)
- **usage:** add task-oriented usage guide + docs index; fix stale README API sketch (a1fc860)
- **plans:** add backlog infra+viewers plan (CI matrix, narrative docs, viewers) (b85de20)
- **changelog:** note live config validation panel (51869ab)
- implementation plan for real-time config validation (e1c1b43)
- fold round-2 in-loop spec review (fresh angles, executed vs real code) (79acb47)
- fold in-loop spec review (2 reviewers, executed vs real code) (d1fac36)
- real-time config validation in the form — design (5a06504)
- **changelog:** property-based tests via Hypothesis (e61407d)
- fold in-loop plan review (executed the plan's code) (18df305)
- property-based tests (Hypothesis) implementation plan — 7 tasks (2e81fea)
- fold round-3 in-loop review (empirical mutation + post-edit consistency) (26cd978)
- fold round-2 in-loop review (teeth + operational angles) (728f4b1)
- fold in-loop spec review (2 executing reviewers, prototyped vs real code) (f59a87d)
- property-based tests (Hypothesis) design — focused 4 targets (1cbc91f)
- **changelog:** trophic-network animation (pyvis) (42827db)
- trophic-network animation (pyvis) implementation plan (738a16a)
- round-2 clean — fix debounce primitive (throttle/debounce, not reactive.event) (4144166)
- fold pyvis in-loop review (2 BLOCKERs) into trophic-network spec (3eedc61)
- revise trophic-network spec to pyvis node-link (Sankey killed by review) (bdc5bde)
- **changelog:** config parser diagnostics (d335964)
- in-loop review fixes for config-parser-diagnostics plan (395cdd8)
- config parser diagnostics implementation plan (b13cbd2)
- round-2 clean — classify empty-key on the post-rstrip value (f892ef8)
- in-loop review fixes for config-parser-diagnostics spec (b115f95)
- CMEMS temperature forcing design spec (engine NetCDF-temp loader + baltic_ev) (7dc9627)
- **changelog:** community size-spectrum diagnostics (d7003de)
- in-loop review fixes for size-spectrum plan (2c4a6ac)
- size-spectrum diagnostics implementation plan (356c782)
- round-2 clean — note the private _read_output_csv coupling (bf62283)
- revise size-spectrum spec per round-1 in-loop review (b0945c3)
- community size-spectrum diagnostics design spec (1d38809)
- **changelog:** Compare Runs works without a loaded output dir (32347e4)
- in-loop review fixes for compare-runs decouple plan (c3401ac)
- compare-runs decouple-from-output-dir implementation plan (d62e850)
- in-loop review fixes for compare-runs decouple spec (4a4ba3a)
- compare-runs decouple-from-output-dir design spec (3ec348c)
- in-loop review fixes for run-history plan (126aaae)
- run-history canonical-dir implementation plan (e679f05)
- run-history canonical-dir reconciliation design spec (49ebd70)
- **ui:** document Compare Runs output-delta section (110b011)
- **ui:** in-loop review polish (broaden render guards, prompt clarity, structure note) (8a64263)
- **ui:** Compare Runs output-delta implementation plan (3d990f8)
- **ui:** Compare Runs output-delta section design spec (7070a0e)
- **delta:** document compare_runs CLI + deferred follow-ons (dfcb422)
- **delta:** round-2 deep-review fixes (fixture preamble + window guard) (8760cd9)
- **delta:** fix plan after deep 4-angle in-loop review (eb7dd6e)
- **delta:** implementation plan + spec wide-format correction (9773993)
- **delta:** result delta-tracking design spec (5de6746)
- **fisheries:** document F/M diagnostics CLI + deferred Kobe follow-up (6d8b2f5)
- **fisheries:** rescope to F/M diagnostics after 4-angle in-loop review (0c2c9f6)
- **fisheries:** fisheries-diagnostics implementation plan (535ea90)
- **fisheries:** fisheries stock-status diagnostics design spec (78f64cc)
- percid overshoot diagnostic — proven cause, fix deferred (9dad350)
- **fr:** phase-14 Baltic FR verdict + diagnostic per-predator K reporting (f04b2c2)
- **fr:** clarify PR-B review nits (predator-set sync note, override comment) (0370697)
- **fr:** config keys + cross-feature caveats + prey-predation-reduction consequence test (e1f4173)
- **fr:** final clean-pass polish (prey-survival test robustness) (89d274b)
- **fr:** round-7 confirmation-review fixes (a7054a0)
- **fr:** round-6 plan rewrite after 4-angle in-loop review (7bb68bc)
- **fr:** round-5 spec re-review fix + implementation plan (705525d)
- rewrite FR spec after round-4 review (2 CRITICALs) (93e88b8)
- fold round-3 diagnostic refinements into FR spec (44cb1e9)
- revise FR spec after round-1+2 in-loop review (e58c942)
- add predator functional response (aggregate, opt-in) design spec (7d18047)
- **plan:** refine density-dependent recruitment plan per review (441ac54)
- **plan:** density-dependent recruitment implementation plan (c01684c)
- **plan:** density-dependent recruitment design (hockey-stick + Shepherd) (5198ed9)
- revise PR #48 plan after second deep review (3b33118)
- implementation plan for PR #48 rebase + Task 11 (3dcd2cc)
- revise PR #48 design after deep multi-angle review (6fde858)
- design for PR #48 rebase + Task 11 resolution (c2122b9)
- **plan:** Ev-OSMOSE FIE-on-cod spec + r2 plan with review fixes (03941da)
- **tutorial:** polish — log_y axis + honest narrative about Baltic transient dynamics (c2ba915)
- **tutorial:** top-of-page README callout + doc-index row + tutorials index (735f350)
- **tutorial:** Beats 2-6 + closing + troubleshooting (Baltic substrate) (aa3631c)
- **tutorial:** preamble + Beat 1 paste-and-run script (Baltic substrate) (694bc99)
- **plan:** r3 of 30-min tutorial plan — fix duplicated suffix, wire numba_warmup, add Predator-extinct branch (3c582ba)
- **plan:** r2 of 30-min tutorial plan — apply round-1 review (TDD discipline, parameter pre-tuning, stopping criterion, conditional commits) (4b1654e)
- **plan:** 30-min 3-species tutorial implementation plan (12 tasks) (f9c3ee3)
- **spec:** r6 of 30-min tutorial — fix stale 0.8/0.99 reference in Task 9 (5502bf2)
- **spec:** r5 of 30-min tutorial — apply round-5 polish (CONFIG/build_config cleanup, accessibility 0.99, BASELINE_PERTURBATION used, ruff-clean skeleton, task ordering) (1585599)
- **spec:** r4 of 30-min tutorial — apply round-4 review (CONFIG/build_config naming, row count, relativefecundity, accessibility scaling, headless fallback) (2349b62)
- **spec:** r3 of 30-min tutorial — apply rounds 1-3 in-loop review (8bd86c9)
- **spec:** 30-minute 3-species tutorial design (e35625a)
- **calibration:** design spec + implementation plan for progress dashboard (95039b2)
- **perf:** 2026-05-08 perf arc overview (#45) (09bdb8a)
- **perf:** P4b not-shipping — __post_init__ bypass below 2% gate (#44) (45a0556)
- **perf:** P4 not-shipping — state.replace cache below noise floor (#43) (9b8ed62)
- **perf:** P3/A4 not-shipping — compute_feeding_stages already below gate (#42) (8a2f9da)
- **plan:** post-v0.12.0 perf plan — K1 + np.add.at + state.replace (r3) (#39) (d5e71d5)

### Other

- v1.0.0 (2849b96)
- Merge feat/scenario-diff-polish: shared config-diff component + compare modal (3624c11)
- record real run duration_sec across both engine paths (f77528d)
- sweep osmose_wizard_/osmose_maps_/osmose_val_ temp dirs (e4b411c)
- warn on unsupported fishing selectivity types 2/3 (silently knife-edged) (8f40eea)
- Merge feat/run-page-rework: Python default, compact Run UI, live-stream crash fix (70c9499)
- Merge feat/python-run-feedback: Python-engine run-progress feedback (f1682ae)
- Merge feat/cmems-forcing-core: CMEMS->OSMOSE forcing conversion core (sub-project A) (9d39da0)
- Merge feat/run-engine-capability: Run-page engine-capability transparency (8560cc6)
- redact leaked credential from tracked docs + scan whole tree (#90) (de6537d)
- OSMOSE 4.4.0 jar swap — PR-A: config value-migration layer (#79) (c9454d9)
- Config-key migration to OSMOSE 4.4.0 — PR2 (UI/writer/calibration wiring) (#78) (dc2d701)
- declare pyarrow runtime dep for fishbase parquet reads (6094c84)
- version-aware dependency upgrade (cma/shinyswatch/shinywidgets/deckgl) with floor check (d8eadad)
- promote shiny to 1.6.x; pin shinyswatch>=0.11, shinywidgets>=0.7, cma>=4.0, shiny_deckgl@v1.9.2 (1e71007)
- Apply ruff formatting to app.py (fix: code-quality) (c82d27c)
- **trophic:** fold in-loop plan-review fixes (5d46d25)
- **temperature-forcing:** close the line — no engine bug, blocked behind calibration (34c558b)
- **helcom:** freeze HOLAS-3 fish snapshot; no validator (rescoped after in-loop review) (c890e12)
- test(history)+docs: lock writer==reader run-history dir invariant (77c1324)
- **run-history:** round-3 clean — sync File Structure header with Step-1b files (818a059)
- **run-history:** fold round-2 findings — writer-test CWD-pollution fix + grep-count correction (77e8f5f)
- Density-dependent recruitment: Shepherd stock-recruitment (+ hockey-stick) (#50) (0b1fc27)
- Ev-OSMOSE FIE-on-cod: genetics plumbing + engine fixes + baltic_ev fixture (#48) (29b5206)
- Merge pull request #49 from razinkele/fix/shiny-deckgl-pin (3e58091)

### Refactoring

- **scenario-diff:** delegate config table to shared component (e2c8ba1)
- **run:** assert choose_live_layer (drop dead layer-builder re-exports + noqa) (e2c3ae7)
- **mcp:** delegate generate_osmose_* to osmose.forcing core (55b47fb)
- **run:** engine tabs -> client-side conditional settings (no input race) (4ef6746)
- **config:** route Run gate and CLI through summarize_config_validation (DRY) (a394bb1)
- **ui:** extract _compare_run_choices helper (0fdf6c0)

### Styling

- **e2e:** ruff format test_e2e_baltic.py (lint = check + format) (584979a)
- **fishbase:** apply ruff format (the format changes that the prior bad git-add aborted) (2603697)
- apply ruff format to 91 drifted files (6a7e941)
- **test:** drop unused pytest import in test_ices_proxy (2cfb168)
- **test:** hoist E402 imports (7e13ed0)

### Tests

- **e2e:** compare-scenarios modal renders a diff (e483786)
- assert a fresh session config_dir survives a normal-age cleanup sweep (c3d5bff)
- **e2e:** expand live card (new stream gate) instead of the removed switch (a0de6b8)
- **e2e:** drop manual live-toggle clicks (auto-on); assert plain-run progress (df47e80)
- **capabilities:** cover unknown-engine fallback, list isolation, truthiness edges (f9d34ce)
- **config:** bioen ingestion-unification parity gate (base-wins, intended change) (b6babf8)
- **engine:** parity note + final gates for community outputs (a9a3017)
- **community-metrics:** real-data eec smoke + final gate pass (9d176d3)
- **e2e:** assert Baltic spatial output renders (pill enables + map) + screenshot (ac92437)
- **e2e:** assert Baltic biomass chart + diet heatmap render (regression guard) (dc9daf3)
- **e2e:** Baltic full-run skeleton — load + Python run + live movement (1912dec)
- **visual:** refresh setup baseline for the FishBase bootstrap panel (3afae99)
- **fishbase:** e2e smoke — bootstrap panel renders on setup page (aa3a40e)
- **fishbase:** record cod + green-crab parquet fixtures (634b637)
- **otel-guard:** defer app import into test bodies to fix CI collection (b141bfa)
- **visual:** suppress Shiny toast to fix movement-page snapshot flake (b9179af)
- **visual:** authoritative container baselines (5 pages, inspected) (82428a2)
- **visual:** retry-until-stable capture (fixes Setup non-determinism in CI container) (46612f5)
- **visual:** use top-level import re instead of __import__ (final-review nit) (19f3d78)
- **visual:** Fishing, Movement, Advanced page snapshots (b55d81f)
- **visual:** Playwright harness + nav-chrome and Setup snapshots (4b50b32)
- **visual:** pure compare_images core (ratio + pixel-floor + mean-delta) with unit tests (d769c55)
- **visual:** add viztest extra, visual marker, collection guard, gitattributes (eb942f9)
- **e2e:** dismiss the startup changelog modal before nav clicks (4e740cb)
- **feedback:** e2e submit→store + CHANGELOG entry (ab6e6ed)
- **docs:** restore hardened e2e modal dismissal (focus-independent + localStorage poll) (044db54)
- **docs:** restore plan-verbatim e2e modal dismissal (fix: spec-compliance) (e5439bb)
- **docs:** e2e startup modal + About tabs; CHANGELOG entry (06e6541)
- **sensitivity:** e2e explorer page + CHANGELOG (3eb4873)
- **scenario-diff:** e2e config-diff panel + CHANGELOG (0624769)
- **e2e:** live movement view streams during a Python run + cancel path (da7a430)
- **e2e:** scenario diff tab renders overlay + spatial maps (cfaf086)
- harden csv-overlay perf guard against xdist contention (02c43d3)
- parallelize suite with pytest-xdist (~3.5x faster) (30efbca)
- skip eec/baltic data-dependent tests when gitignored artifacts absent (7d07de7)
- **hypothesis:** make window property two-sided (final-review teeth fix) (e7268bb)
- **hypothesis:** size-spectrum helper properties (convexity, LFI boundary, bin-width, window) (1c09159)
- **hypothesis:** diet_network_at properties (bounds, monotonicity, prey-sum, dead-stage) (512559b)
- **hypothesis:** preamble-detection properties (detect-k, no-raise, cache-invalidation) (2baea1b)
- **hypothesis:** config writer->reader round-trip properties (aa6d5a0)
- **hypothesis:** shared property-test strategies (f9c15bb)
- **hypothesis:** add hypothesis dev dep + deterministic ci profile (34f73a6)
- **fr:** background-inclusive diet aggregator + width-16 diagnostic test (4aefd36)
- **fr:** bit-exact parity-off gate + type1==absent + background-on + inert-prey-only (8356f54)
- **fr:** wire numba=False to the Python predation fallback via _HAS_NUMBA patch (57fa1ba)
- **fr:** shared run harness + FR config-injection helpers (cb8211e)
- **tutorial:** share Baltic sim across tutorial tests via module-scoped fixtures (06163ca)
- **conftest:** ignore e2e modules at collection when playwright missing (9e523d8)
- **tutorial:** encode measured equilibrium bounds (Baltic, seed=42) (3db8d5f)
- **tutorial:** replace numba_warmup stub with real session-scoped warmup (8e261f7)
- **tutorial:** regression test in final form (RED until Tasks 4-9) (3a23a3c)
- add tmp_results_dir + synthetic species fixtures for calibration tests (c3f573f)

## [0.12.0] - 2026-05-08

### Features

- **schema:** choice_labels for fisheries.selectivity.type ENUM (Phase 2 finisher) (#27) (f436be5)

### Bug Fixes

- **validation:** close broad schema-engine parity TODO from C1 (#28) (8b5908d)
- **engine:** average distribution dicts across recording window (M1) (#22) (f8dfd42)
- **ui+engine:** cooperative cancellation through simulate() (C4 Phase B) (#18) (f7d8e28)
- **ui:** render multi-value config entries as read-only text inputs (H12) (#17) (3958b41)
- **app:** defer cleanup_old_temp_dirs() until first session start (H11) (#15) (e0ead58)
- **engine:** Phase 3 — M4 Gompertz bounds, M5 accessibility/n_dead clamps, RNG docs (#11) (f16a057)
- **engine:** reproduction parameter bounds (H10) (#10) (8ff56a7)
- **schema:** field-quality fixes from Phase 2 plan (#9) (dc95f55)
- **validation:** close schema-coverage gaps on examples + minimal (H1, H4, H2) (#8) (dd55620)
- **schema:** movement keys (C1) — engine reads movement.{property}.map{N} (#5) (06b432e)
- **ui:** invalidate state.output_dir on failed/cancelled run (C4 Phase A) (#7) (65b0d7e)
- **ui:** path-traversal hardening in results page (C3) (#6) (de9bfa0)
- **schema:** output.bioen.sizeInf -> sizeinf; engine never read camelCase (C2) (#4) (b62ff8e)
- **validation:** allowlist baltic background-species keys (C5) (#3) (d7b9a6d)

### Performance

- **engine:** vectorise _precompute_map_indices per-species (A2) (#36) (1f717fe)
- **engine:** vectorise accessibility school-index lookup (A1) (#35) (70cc1da)
- **mortality:** drop redundant n_dead zeros allocation (H6 partial) (#31) (e0e1d94)
- **bench:** --config flag + per-fixture grid resolution + name-list guard (#30) (38dc8f3)
- **engine:** vectorise biomass_by_cell DSVM accumulator (H7) (#29) (8c6621e)

### Documentation

- **perf:** A3 not-shipping post-mortem (#37) (3c0b66c)
- **plan:** vectorise the non-JIT'd Python-side perf hot paths (#34) (ceb59f5)
- **plan:** kernel-surgery plan for remaining Phase 4 perf items (r2, post-review) (#32) (39725cf)
- **plan:** add deep-review remediation plan (9-iter loop-converged) (3851b9e)

### Other

- v0.12.0 — Python-side perf wins + deep-review remediation + B-H (#38) (31a2386)
- K4 — eec_full 5yr profile + go/no-go decisions for K1/K2/K3 (#33) (1839a1c)
- Merge origin/master into local master (ebe8343)
- bounded-runtime DE — patience + wall-clock cap (post-incident) (cf5cb8e)
- DE intermediate checkpointing — interrupt-safe long runs (a6c596b)
- optimizer benchmark on synthetic problems (fe0c04c)
- --optimizer flag wires CMA-ES + surrogate-DE into calibrate_baltic (135aa8f)
- **surrogate-de:** nfev accounting, GP fit recovery, warm-start warn (8edffa4)
- **cmaes:** correctness fixes for success flag, bounds, NaN handling (54b61d1)
- harden D4 sensitivity script against silent data loss (b7a5708)
- add GP surrogate-assisted DE runner (Tier C1 speedup) (8649e55)
- add CMA-ES runner (Tier C2 speedup) (6a90cf6)
- add Sobol sensitivity script for phase 12 param pruning (D4) (15e610f)
- launch wrapper for Tier A+B fast phase 12 calibration (f775df3)
- warm-start, tunable tol, and Tier A speedup knobs (1ae7630)
- **phase12:** enable Beverton-Holt density-dependent recruitment (816d5c1)
- **background:** set length on background schools (bg-predation fix) (92088c9)
- v0.10.0 — calibration on PythonEngine in-memory (d4eebe1)

### Refactoring

- **schema:** Phase 2 field-quality bundle — predation, bioen, ltl, fishing (#26) (936769c)
- **ui:** consolidate state.loading into state.busy (M10) (#19) (27fe142)
- **ui:** extract input_id_for_field/key helpers (M13) (#16) (6dda518)
- **ui:** make _inject_random_movement_ncell pure (H3) (#14) (7133fa3)

### Tests

- **engine:** NaN propagation through fishing_mortality (H8 — final) (#25) (c654f7d)
- **engine:** JIT determinism across thread counts (H9 — verified) (#24) (399339c)
- **engine:** NaN propagation extensions for reproduction/accessibility/starvation (H8) (#23) (30b04f6)
- **engine:** pin size-bin Java parity (M3 — verified, no code change) (#21) (6aa5cc4)
- **engine:** NaN/Inf propagation audit through mortality processes (H8) (#20) (9c85a8b)
- Phase 5a — L brittleness fix, M7 lifespan boundary, M8 runner failure modes (#13) (64a623a)
- **engine:** pin feeding-stage Java parity (M2 — verified, no code change) (#12) (a0574e4)

## [0.11.0] - 2026-04-28

### Bug Fixes

- **engine:** slice per-species arrays to focal-only in reproduction (37bc1d1)

### Documentation

- **plan:** Beverton-Holt + Ricker stock-recruitment implementation plan (c6b1b2f)
- **spec:** add .docx version of non-rectangular grids study (2daf029)
- **spec:** non-rectangular grid feasibility study (a5481b1)
- **plan:** round-3 cleanups — commit-message pattern + _expected_errors note (15673e7)
- **plan:** tighten __init__ signature decision (round-2) (d1202f3)
- **plan:** calibration plan round-1 review fixes (836625b)
- **plan:** calibration-python-engine implementation plan (dc5faae)
- **spec:** round-3 final-pass fixes (cb94146)
- **spec:** pin cache-key authoritative source (round-2 cleanup) (080fd78)
- **spec:** calibration spec round-1 review fixes (ea185fa)
- **spec:** calibration throughput — port NSGA-II to PythonEngine in-memory (d7fa894)

### Other

- bump 0.10.0 -> 0.11.0 (origin already has v0.10.0 tag at d4eebe1) (a24090d)
- docs+version: Beverton-Holt SR documented as post-parity divergence (e74326d)
- **reproduction:** apply stock-recruitment to per-step eggs (1c9cb28)
- **reproduction:** cleanup — NDArray annotations, diagnostic mismatch error, sharper zero-SSB test (57ad8b4)
- **reproduction:** add apply_stock_recruitment helper (B-H + Ricker) (db9338e)
- **config:** cleanup — docstring, schema desc trim, top-level pytest import, n_bkg consistency (bdaa398)
- **config:** parse stock.recruitment.{type,ssbhalf} per species (33c216f)
- add stock.recruitment.{type,ssbhalf} fields per species (05eed2f)
- compare two phase-12 calibration runs (d8df5db)
- activate grey seal + cormorant background species (f54bb26)
- widen mortality bounds instead of background species (edcea8b)
- handle --phase 12 (joint optimization) (a9b43d4)
- add joint phase 1+2 option (--phase 12, 24 params) (ef802c2)
- cap DE workers at 8 (from -1) to avoid RAM exhaustion (8b42497)
- enable DE process-pool parallelism (da7956d)
- widen flounder + pikeperch fishing upper bounds (8c8a349)
- Revert "baltic: add grey seal + cormorant as background species" (6541e9f)
- add grey seal + cormorant as background species (4d5e7fb)
- 2026-04-21/22/23/24 session work — map fixes, calibration, plan (0d1cdf3)

### Tests

- **reproduction:** multi-step integration smoke for B-H + Ricker (9e74dd8)
- **reproduction:** derive season from cfg.n_dt_per_year, isolate rng per call (c58f702)
- **reproduction:** pin linear-formula regression for type=none across SSB sweep (52653ab)
- **reproduction:** fix misleading SSB comment, isolate RNG per call (62c715c)

## [0.10.0] - 2026-04-21

### Documentation

- parity roadmap STATUS-COMPLETE audit — mark Phases 2/3/4 shipped (9f8af75)

### Other

- v0.10.0 — calibration on PythonEngine in-memory (d4eebe1)

## [0.9.3] - 2026-04-19

### Features

- **predation:** public predation_for_cell API (Phase 7.1 commit 1) (72f80a3)

### Documentation

- CHANGELOG entry for Phase 7.1 predation reconciliation (0b82e68)
- **tests:** fix stale predation() docstring in rng_consumers (20649e0)
- parity-roadmap §7.1 STATUS-COMPLETE (Phase 7.1) (7f9a3e7)
- **plan:** round-2 clarify cell_x/cell_y irrelevance for dispatch (af256ff)
- **plan:** Phase 7.1 plan round-1 review fixes (f4914d1)
- **plan:** Phase 7.1 predation-reconciliation implementation plan (e8d10cc)
- **spec:** Phase 7.1 round-4 review — bug-hunting pass (4fec5c8)
- **spec:** Phase 7.1 review-driven revisions (6c3aced)
- **spec:** Phase 7.1 predation-reconciliation design (0f16837)

### Other

- v0.9.3 (ea31ac8)

### Refactoring

- **predation:** update stale comments in predation_for_cell (Phase 7.1 task 3 follow-up) (009c781)
- **predation:** delete batch predation() orchestrator (Phase 7.1 commit 3) (37ceeb1)
- **tests:** expand single-school resource test to 2 schools (Phase 7.1 task 2 follow-up) (ed62a76)
- **tests:** migrate test_engine_rng_consumers to predation_for_cell (Phase 7.1 commit 2f) (2aa6018)
- **tests:** migrate test_engine_feeding_stages to predation_for_cell (Phase 7.1 commit 2e) (12c8255)
- **tests:** migrate test_engine_background to predation_for_cell (Phase 7.1 commit 2d) (5ca231a)
- **tests:** migrate test_engine_predation to predation_for_cell (Phase 7.1 commit 2c) (ec3edd3)
- **tests:** migrate test_engine_diet to predation_for_cell (Phase 7.1 commit 2b) (9461419)
- **tests:** migrate test_engine_predation_helpers to predation_for_cell (Phase 7.1 commit 2a) (0ca3c60)

## [0.9.2] - 2026-04-19

### Features

- **config:** unknown-key validation at EngineConfig.from_dict (Phase 7.3) (327a20c)

### Bug Fixes

- **config:** spec-review polish — restore UnknownKey export + strip empty-key artefact (66b3dcd)

### Documentation

- update README + CLAUDE.md for Phase 7.3 config validation (3af9f9a)
- CHANGELOG entry for Phase 7.3 config validation (f29c91d)
- **plan:** iter-4/5/6 review fixes — warn-mode integration, glob helper, arithmetic cleanup (008a8bb)
- **plan:** iter-3 Step 7 hint fixes (744ae1b)
- **plan:** iter-2 fixes — drop unused functools import + rewrite stale path prose (5761a72)
- **plan:** iter-1 review patches to config-validation plan (bad68cb)
- **plan:** config-validation implementation plan (Phase 7.3) (33b0478)
- **spec:** remove stale +13/d124de6 baseline paragraph (iter-2 fix) (557ee1b)
- **spec:** iteration-1 review patches to config-validation design (1a0c996)
- **spec:** self-review fixes to config-validation design (e8294ee)
- **spec:** config validation design (Phase 7.3) (4d273e6)
- **parity-roadmap:** Phase 7 honesty pass — 7.2 shipped, 7.1+7.3 scoped accurately (d124de6)
- **parity-roadmap:** Phase 6 STATUS-COMPLETE (bioenergetic module) (ede19d3)

### Other

- v0.9.2 (0e59258)
- **config:** align logging + canary loader with project conventions (d7181ca)

## [0.9.1] - 2026-04-19

### Features

- **economics:** wire write_economic_outputs into simulate end-of-run + STATUS-COMPLETE Front 6 (1dcffd7)

### Chores

- replace deprecated update_navs + gitignore ephemeral paths (b1e9328)

### Other

- v0.9.1 — Front 6 wire-up + post-release cleanups (7c68e7a)

### Tests

- **e2e:** sync Playwright assertions to current tab labels (259d3d1)

## [0.9.0] - 2026-04-18

### Features

- **output:** spatial outputs — biomass / abundance / yield-biomass NetCDF (901b3be)
- **output:** NetCDF per-species distributions + mortality-by-cause (d5947f0)
- **output:** diet CSV Java-parity — one file, one row per recording period (5e96916)
- **calibration-ui:** Pareto/Weighted toggle for surrogate optimum (84e5ab7)
- **calibration-ui:** n_workers numeric input for preflight (95e858d)
- **calibration-ui:** red-banner modal for PreflightEvalError (c8b2e32)
- **calibration:** ICES SAG validator for Baltic F rates and biomass envelopes (2be016f)
- **calibration:** problem.py robustness (timeout/stderr/cleanup) (8358725)

### Bug Fixes

- **baltic:** post-rename cleanup + Java-compat regression tests (89734a0)
- **calibration:** unit-aware F weighting + defensive snapshot pull + coverage tests (590240e)
- **calibration:** surrogate multi-obj returns Pareto or weighted-sum (6d75310)
- **calibration:** loud failures + abort threshold in preflight eval (4de10ae)

### Performance

- **calibration:** parallelize preflight eval with ThreadPoolExecutor (a454758)

### Chores

- gitignore .bak + calibration_results; archive sensitivity plan (355e7f4)

### Documentation

- post-session roadmap STATUS-COMPLETE for fronts 1-4 (7482b6f)
- SP-4 changelog + Phase 5 STATUS-COMPLETE (445f354)
- **plans:** fix stale test-count arithmetic in SP-4 plan self-review (e03a69c)
- **spec,plans:** iteration-2 spec + plan patches for SP-4 (47349a8)
- **spec:** SP-4 spec iteration-1 patches (internal consistency + Java parity) (fc7b4cc)
- **plans:** iteration-2 patches to SP-4 plan (542bcf9)
- **plans:** rewrite SP-4 plan after 4-reviewer ground-truth audit (0c74795)
- **plan:** SP-4 output system implementation plan (e036338)
- **spec:** scrub two residual output.spatialized.* config-key references (8ad2103)
- **spec:** SP-4 review-round-1 corrections (553ece4)
- **spec:** SP-4 output system design (Front 4) (b77e85d)
- changelog for calibration UI Phase 3 (62f628b)
- **plan,spec:** plan review-round-1 corrections (a476a0e)
- **plan:** calibration UI Phase 3 implementation plan (aef168f)
- **spec:** fix SilentException import path to shiny.types (was shiny.reactive) (0bbfdfb)
- **spec:** calibration UI Phase 3 review-round 1 corrections (38676a1)
- **spec:** calibration UI Phase 3 design (Front 3) (1cafed5)
- modernize README (TOC, Baltic example section, doc index) + changelog for review fixes and provenance doc (034f614)
- **baltic:** full provenance doc for the Baltic example (sources, scripts, per-parameter refs) (4031138)
- changelog for Baltic ICES cross-validation (9797e2f)
- **baltic:** document ICES snapshot refresh workflow (722f21f)
- **baltic:** summarize ICES cross-validation findings and known limits (2926436)
- **baltic:** scaffold ICES SAG snapshot directory + manifest (028beb7)
- roadmap for closing remaining fronts (ICES MCP → Java fishery reformat → UI Phase 3 → SP-4) (2fa0737)
- **plans:** STATUS-COMPLETE banners on three more shipped plans (fb08acc)
- **plans:** STATUS-COMPLETE banners on five already-shipped plans (934b46b)
- **plans:** loop-review patches to ICES SAG validation plan (518de4b)
- **plans:** mark UI Phase 2 complete; fix ICES plan MCP-shape drift (a313866)
- plan for ICES SAG cross-validation of Baltic calibration (d923ad0)
- changelog for calibration/sensitivity fixes (b8a58b4)

### Other

- v0.9.0 — full Phase 5 output-system parity + calibration UI Phase 3 + Baltic ICES validation (37bdafb)
- **baltic:** strip underscores from fishery names for Java-engine compatibility (98f478e)
- **baltic:** freeze ICES SAG snapshots for eight Baltic stocks (2024 advice, cod.27.22-24 on 2022) (a7d65a5)
- load-example picker on grid page + calibrate_baltic popsize_mult (fc5f271)
- **baltic:** grid-mask rebuild + ICES validation + whitefish→smelt swap (d61ea1c)

### Refactoring

- **scripts:** return-type hint + itertools counter in calibrate_baltic (7ace2f3)
- **calibration:** split detect_issues into per-category helpers (95db190)
- **calibration:** split run_preflight into Morris/Sobol stages (13f9474)

### Tests

- make_minimal_engine_config helper for SP-4 tests (0779581)
- **calibration:** regression fence for F-rate and biomass-envelope drift vs ICES (0435fde)

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

- v0.8.1 — deep-review fixes (credentials, types, docs) (e2bedc2)
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
