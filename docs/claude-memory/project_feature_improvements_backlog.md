---
name: Feature improvements backlog (post-v0.12.0)
description: 2026-05-08 — candidate next-direction features after the perf arc closed. Grouped by theme; top-5 picks called out.
type: project
originSessionId: d2b7f4a5-d107-4042-a473-f491e81f4df1
---
Compiled 2026-05-08 after the perf arc reached the 2 % gate's noise floor on eec_full / baltic. Pivoted from perf to feature direction. Listed here as durable institutional candidates for any future session that opens with "what should we build next?"

## Top-5 recommendations (highest value-per-effort) — ALL SHIPPED as of 2026-05-27

The entire top-5 is now consumed; the next session should pick from the thematic lists below, not here.

1. ~~**ICES validation harness**~~ — SHIPPED PR #46 (`osmose/validation/ices.py`).
2. ~~**Activate DSVM fleet economics**~~ — SHIPPED PR #47 (`scripts/run_dsvm_demo.py`).
3. ~~**Calibration progress dashboard (Shiny)**~~ — SHIPPED 2026-05-16.
4. ~~**Activate Ev-OSMOSE genetics**~~ — SHIPPED PR #48 (FIE-on-cod, merged 2026-05-27).
5. ~~**Tutorial: "Build a 3-species ecosystem in 30 minutes"**~~ — SHIPPED 2026-05-17 (Baltic-subset substrate).

## Science extensions (wired-but-inactive)

- Activate Ev-OSMOSE genetics (★ top-5)
- Activate DSVM fleet economics (★ top-5)
- Multi-species size-spectrum diagnostics (Sheldon spectra + community-level metrics)
- Climate-driven physical forcing via `mcp_servers/copernicus`

## Calibration

- ICES validation harness (★ top-5)
- ProcessPoolExecutor swap for NSGA-II (deferred from v0.10.0; recovers higher than 3.02× speedup)
- Density-dependent recruitment beyond B-H/Ricker (Hockey-stick, Shepherd)
- Multi-objective Pareto-front explorer UI
- Sub-stock aggregation for flounder

## UI / Shiny

- Calibration progress dashboard (★ top-5)
- ~~Scenario diff view (side-by-side biomass + spatial maps)~~ — SHIPPED PR #60 (2026-06-13), `ui/pages/scenario_diff.py`.
- **Live-during-run movement visualization** — stream actual school positions on a deck.gl map *while the simulation runs* (Play-button playback over the live run), distinct from the existing "Movement Animation" Grid overlay which only animates the configured input `movement.file.map{N}` distribution maps (`ui/pages/grid.py`), not live engine state. Non-trivial: needs the engine (Python `osmose/engine/simulate.py` and/or the Java subprocess) to EMIT per-step school positions during the run (a streaming hook / incremental output) + a Shiny consumer that polls/streams + renders. Today the engine runs to completion then writes outputs; there is no live feed. Could start Python-engine-only (in-process callback) and is a bigger lift for the Java path. Related: [[project_movement_visualization]].
- ~~Parameter sensitivity explorer (surface Sobol output as a Shiny page)~~ — SHIPPED to origin/master 2026-06-14 (`3eb4873`): `ui/pages/sensitivity_explorer.py` + `osmose/calibration/sobol_io.py` artifact store + persist hook on the live calibration run. Backend (`SensitivityAnalyzer`) already existed; this added persistence + a browse page.
- Map-based scenario builder (draw polygons → CSV)
- ~~Real-time config validation in the form~~ — SHIPPED 2026-06-08 (`summarize_config_validation` panel).
- Bootstrap config wizards (auto-generate Baltic-like / EEC-like scenarios)

## Output & analysis

- ~~**Python-engine by-size + 1D meanTL output**~~ — SHIPPED 2026-06-17 PR #75 (`a9a3017`), [[project-python-engine-community-outputs]]. Python engine now writes community `DistribBySize` + realized biomass-weighted 1D `meanTL`; Size Spectrum / Sheldon / MTL-MTI now populate on Python-only configs (Baltic). (`output.meanTL.bySize`/2D `meanTLByAge` NetCDF variants remain unbuilt — not needed by consumers.)
- Trophic-network animation (per-step diet-matrix graph)
- Standard fisheries diagnostics (F/M, B/Bmsy, F/Fmsy plots)
- Per-cell trajectory output (activate spatial + NetCDF time-series viewer)
- Result delta-tracking (highlight species / cells / periods that responded most to a config change)

## Configuration & UX

- Config presets (Baltic / EEC / BoB / Mediterranean clone-and-edit bundles)
- **Schema-driven config migration (auto-rewrite renamed keys with deprecation warning) — NOW CONCRETELY MOTIVATED by OSMOSE Java 4.4.0** ([[reference-osmose-java-4-4-0]]). 4.4.0 made these renames MANDATORY; OSMOPY's schema still emits the OLD keys, so 4.4.0-format configs won't round-trip and OSMOPY-written configs won't run on a 4.4.0 jar without migration. Concrete rename table to implement (old→new):
  - `simulation.bioen.enabled`→`module.bioenergetics.enabled`; `simulation.genetic.enabled`→`module.genetics.enabled`; `fisheries.enabled`→`module.multispecies.fisheries.enabled`; `economy.enabled`→`module.bioeconomics.enabled`
  - `output.restart.enabled`→`simulation.restart.enabled`; `output.restart.recordfrequency.ndt`→`simulation.restart.recordfrequency.ndt`; `output.restart.spinup`→`simulation.restart.spinup.nyear`
  - `predation.ingestion.rate.max.bioen.spX`→`predation.ingestion.rate.max.spX`; `predation.coef.ingestion.rate.max.larvae.bioen.sp`→`predation.larval.ingestion.rate.increase.ratio.spX`; `species.bioen.maturity.{eta,r,m0,m1}.spX`→`species.maturity.{eta,r,m0,m1}.spX`
  Approach: a bidirectional alias map in the config reader (read old OR new → canonical) + a writer mode/`--migrate` that emits 4.4.0 keys with a deprecation log; schema `key_pattern`s move to new names with old kept as read-aliases. NB: the Java engine's own `-update` migration (wrapped at `runner.py:_run_config_migration`) is an alternative bridge for the Java-engine path specifically. Direct dependency of the Java-core-4.4.0 update.
- Config diff tool
- Better parser error messages (structured line/column context)

## MCP / integrations

- HELCOM HOLAS-3 as calibration target (Baltic-specific assessments via `mcp__helcom`)
- Copernicus real-time forcing
- FishBase / SeaLifeBase auto-bootstrap species traits
- Scientific literature citation tracking via `mcp__scite`

## Engine / perf (post-noise-floor structural levers)

- `__slots__` on SchoolState (1-3 % + lower memory; needs invariant-coverage refactor)
- Mutable SchoolState refactor (remove per-construction validation; biggest remaining lever)
- C/Rust predation kernel (replace `_apply_predation_numba` native extension)
- GPU acceleration via Numba CUDA
- OpenMP-style spatial-locality cell scheduling

## Test infra

- Property-based tests via Hypothesis (SchoolState invariants, RNG reproducibility, vectorised-vs-loop parity)
- CI matrix on multiple Python versions (3.11 / 3.12 / 3.13)
- Faster test suite via pytest-xdist parallelisation
- Visual regression tests for UI via Playwright snapshots

## Documentation

- Tutorial: "Build a 3-species ecosystem in 30 minutes" (★ top-5)
- Auto-generated API reference via Sphinx + myst-parser
- Migration guide from R OSMOSE
- Scientific paper companion site (reproducibility-by-construction)

## Ops / reproducibility

- Container deployment (Docker / Apptainer)
- Snakemake / Nextflow workflows for calibration pipelines
- Cloud-runner integration
