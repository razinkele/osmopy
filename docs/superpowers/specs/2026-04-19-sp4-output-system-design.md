# SP-4 Output System — Design

> **Front 4** of the 2026-04-18 post-session roadmap. Spec for the plan at (forthcoming) `docs/superpowers/plans/2026-04-19-sp4-output-system-plan.md`.

**Goal:** Close the three remaining Phase-5 output gaps (per `docs/parity-roadmap.md`) so the Python engine achieves full standard-OSMOSE output parity (Ev-OSMOSE-specific outputs remain deferred). The gap-analysis pass confirmed Phase-5 items **5.1, 5.2, 5.3, 5.7 are already shipped** and only **5.4 (spatial), 5.5 (diet per-period), 5.6 (NetCDF completion)** remain.

## Scope choice

**C — Full SP-4** of the three scoping options. In order of effort:

1. **5.5 diet-per-recording-period** — the one small cleanup (whole-run diet sum is trivially derivable; Java writes one matrix per recording step).
2. **5.6 NetCDF completion** — extend the existing `write_outputs_netcdf` with per-species distribution variants (biomass/abundance by age and by size) and mortality-by-cause. All data is already in `StepOutput`; this is format-layer only.
3. **5.4 spatial outputs** — the architectural piece: cell-indexed `StepOutput` fields for biomass, abundance, yield; a new `_collect_spatial_outputs` step; a new spatial NetCDF writer. Config keys match Java's `output.spatialized.*` pattern.

**Out of scope.** Ev-OSMOSE outputs (genotype, bioen-spatial, diet-by-stage), debug/introspection outputs (school snapshots, spawn/death counts), per-fishery breakdowns (`FisheryOutput` — blocked by DSVM scope), spatial TL/size/mortality/egg maps (deferred — can be added without reshaping `StepOutput` once the spatial-biomass pattern is established), CSV variant of spatial outputs (NetCDF-only for v1). No change to existing output semantics (yield per species, biomass/abundance timeseries, mortality-by-cause CSV) — this spec only *adds* outputs and converts one (diet) from whole-run to per-period.

## Architecture

Three increments, each a single commit, in order of decreasing scope overlap:

```
┌────────────────────────────────────────────────────────────────────────┐
│ simulate.py                                                            │
│                                                                        │
│ per step:                                                              │
│   step_out = _collect_outputs(state, cfg, step)                        │
│              ├─ biomass, abundance, mortality (existing)               │
│              ├─ yield_by_species (existing)                            │
│              ├─ biomass/abundance _by_age/size (existing)              │
│              ├─ diet_by_species (existing)                             │
│              └─ spatial_biomass/abundance/yield ← NEW (5.4)            │
│                                                                        │
│ per recording period:                                                  │
│   averaged = _average_step_outputs(accumulated, freq, step)            │
│              └─ ndarray fields averaged element-wise ← extended (5.4)  │
│                                                                        │
│ end of run:                                                            │
│   write_outputs(outputs, cfg, ...)                                     │
│              ├─ species CSVs, yield CSV, distribution CSVs (existing)  │
│              ├─ mortality CSV, bioen CSVs (existing)                   │
│              ├─ write_diet_csv(outputs, ...) ← refactored (5.5)        │
│              ├─ write_outputs_netcdf(outputs, ...)                     │
│              │       └─ + distribution + mortality variants (5.6)      │
│              └─ write_outputs_netcdf_spatial(outputs, ...) ← NEW (5.4) │
└────────────────────────────────────────────────────────────────────────┘
```

No library-side changes beyond the engine-owned `simulate.py` / `output.py`. No schema-file changes beyond adding the new config-key `OsmoseField` entries. No UI changes.

---

## Capability 5.5 — Diet per-recording-period

### Current behaviour (whole-run sum)

`write_diet_csv(outputs, ...)` at `output.py:207` currently iterates `outputs: list[StepOutput]`, sums `step.diet_by_species` across all steps, and writes a single CSV `{prefix}_diet_Simu{i}.csv`. Java writes one CSV **per recording period**.

### Target behaviour (per-period)

- Input unchanged: `outputs: list[StepOutput]` already averaged per recording period (`_average_step_outputs` at `simulate.py:854`).
- Write one CSV per recording period, filename pattern `{prefix}_diet_Simu{i}_step{record_step}.csv` — the `record_step` matches the field already populated on the averaged `StepOutput`.
- Each CSV is the `(n_pred, n_prey)` matrix with predator names as rows and prey (focal species + LTL groups) as columns. Same column schema as the current single-file output.
- Remove the whole-run sum. Callers that want it can compute `sum(axis=time)` from the loaded NetCDF (or the per-step CSVs).

### Config gating

Tie writing to the existing `output.diet.composition.enabled` (currently parsed but not honored — `config.py:772`). When false, skip the writer entirely.

### Files touched

- Modify: `osmose/engine/output.py` — `write_diet_csv` body rewrite.
- Modify: one existing test in `tests/` that asserts on diet filename — update assertion.
- Modify: `osmose/engine/output.py:314` (`write_outputs_netcdf`) if it currently folds diet; confirm by reading.

### Tests

- `test_write_diet_csv_emits_one_file_per_recording_period`: construct a list of three `StepOutput`s with distinct `diet_by_species` matrices; call `write_diet_csv`; assert three files exist with the correct filename pattern and each file's matrix matches the per-step input.
- `test_write_diet_csv_skipped_when_config_disabled`: `output.diet.composition.enabled=false` → no diet files written.

---

## Capability 5.6 — NetCDF per-species distributions + mortality

### Current behaviour

`write_outputs_netcdf` at `output.py:314` emits biomass, abundance, and yield as a NetCDF dataset with dims `(time, species)`. Distribution data (`biomass_by_age/size`, `abundance_by_age/size`) and mortality-by-cause are written to CSVs but not NetCDF.

### Target behaviour (extend the existing writer)

Extend `write_outputs_netcdf` to emit three additional `DataArray`s into the same NetCDF dataset (no new file — same `{prefix}_Simu{i}.nc`):

- `biomass_by_age`: dims `(time, species, age_bin)`. From `StepOutput.biomass_by_age`, padded to the per-species max age-bin count with NaN for species with fewer bins.
- `abundance_by_age`: same dims.
- `biomass_by_size`: dims `(time, species, size_bin)`. Padded similarly.
- `abundance_by_size`: same.
- `mortality_by_cause`: dims `(time, species, cause)`. Cause names as string coords (`natural`, `predation`, `starvation`, `fishing`, `additional`, `out`).

Each emission gated by its existing config key (`output.biomass.byage.enabled`, etc.) PLUS a new master `output.netcdf.enabled`. When the master is `false`, no NetCDF file is written at all — the `.nc` file is suppressed in favour of the CSVs.

### Age- and size-bin coordinates

- Age bins: `age_bin_centers` from the species's age discretization (retrieved from `cfg` via the existing `_get_age_bins` helper or equivalent).
- Size bins: `size_bin_centers` similarly.

Bins may differ per species — for NetCDF this means we pad to the cross-species max and use NaN for unused slots. Document this in the `Dataset.attrs["distribution_padding"]` string.

### Files touched

- Modify: `osmose/engine/output.py` — `write_outputs_netcdf` body.
- Modify: `osmose/schema/output.py` — add `output.netcdf.enabled` `OsmoseField`.

### Tests

- `test_netcdf_contains_biomass_by_age_when_enabled`: synthesize 3 `StepOutput`s with `biomass_by_age` populated; write; open with xarray; assert `biomass_by_age` variable with dims `(time, species, age_bin)` and values matching input.
- `test_netcdf_contains_mortality_by_cause`: same pattern for mortality.
- `test_netcdf_suppressed_when_master_disabled`: `output.netcdf.enabled=false` → no `.nc` file in the output dir.
- `test_netcdf_pads_ragged_age_bins_with_nan`: two species with different bin counts → NetCDF shape matches max, shorter species has NaN in the tail slots.

---

## Capability 5.4 — Spatial outputs (NetCDF, biomass / abundance / yield)

### StepOutput extension

Add three optional fields, paired-presence per the existing `StepOutput` invariant pattern:

```python
spatial_biomass: dict[int, NDArray[np.float64]] | None = None   # sp_idx -> (n_lat, n_lon)
spatial_abundance: dict[int, NDArray[np.float64]] | None = None
spatial_yield: dict[int, NDArray[np.float64]] | None = None
```

Pairing invariant: all three are either `None` (spatial outputs disabled) or all non-`None` (any enabled — unused variants hold zeros rather than mixing None/non-None). The pairing matches the `biomass_by_age`/`abundance_by_age` convention already documented at `simulate.py:64-75`.

### Collection

New function `_collect_spatial_outputs(state, grid, config) -> tuple[dict, dict, dict]` in `simulate.py` that aggregates biomass / abundance / yield per cell per species:

- For each school in `SchoolState`, look up its cell via `state.cell_id`.
- Scatter biomass / abundance / fished-biomass into the per-species `(n_lat, n_lon)` arrays using `np.add.at` for correctness with duplicate cell writes.
- **Always produces all three dicts when the master is enabled** (no per-variant skip at collection time — the three arrays share the same school loop, so splitting them would save trivial CPU while complicating the pairing invariant). Per-variant enablement gates only the writer.
- Called from `_collect_outputs` only when `output.spatialized.enabled=true`. When the master is `false`, the function is not called at all and all three fields remain `None`.

### Averaging

`_average_step_outputs` already handles `dict[int, NDArray]` fields (via the distribution pairs). Reuse the same averaging pattern: element-wise mean across the accumulator list for each species. When any step has `spatial_biomass=None`, emit `None` for the averaged output (same rule as the existing distribution pairs).

### Writer

New function `write_outputs_netcdf_spatial(outputs, cfg, output_dir, prefix)` in `output.py`:

- One NetCDF file per enabled variant: `{prefix}_spatial_biomass_Simu{i}.nc`, `{prefix}_spatial_abundance_Simu{i}.nc`, `{prefix}_spatial_yield_Simu{i}.nc`.
- Dims: `(time, species, lat, lon)`.
- Coords: `time` from record step indices; `species` from species names; `lat` / `lon` from the grid's `baltic_grid.nc` coordinates (via `grid.lat_centers` / `grid.lon_centers`, which already exist for the grid widget).
- Values in tonnes (biomass), individuals (abundance), tonnes (yield) — matching the non-spatial variants.
- Cells outside the ocean mask written as NaN (not zero) so downstream plotting doesn't color land.

### Config keys (Java-compatible)

- `output.spatialized.enabled` (master; default `false`).
- `output.spatialized.biomass.enabled` (default `true` when master is `true`).
- `output.spatialized.abundance.enabled` (default `false`).
- `output.spatialized.yield.enabled` (default `false`).

Added as four new `OsmoseField` entries in `osmose/schema/output.py`. When the master is `false`, the collector short-circuits and the writer emits no files.

### Files touched

- Modify: `osmose/engine/simulate.py` — `StepOutput` gains three fields; `_collect_outputs` dispatches to new `_collect_spatial_outputs`; `_average_step_outputs` extended for the new dicts (element-wise mean, already supported pattern).
- Modify: `osmose/engine/output.py` — `write_outputs` calls the new spatial writer when the master is enabled; new `write_outputs_netcdf_spatial` function.
- Modify: `osmose/schema/output.py` — four new `OsmoseField` entries.

### Tests

- `test_collect_spatial_biomass_aggregates_by_cell`: 3 schools in 2 cells for 1 species; `_collect_spatial_outputs` produces a `(n_lat, n_lon)` array with the expected per-cell sums.
- `test_average_spatial_outputs_means_per_cell`: 2 step-outputs with different per-cell biomass for one species; averaged output is the elementwise mean.
- `test_spatial_netcdf_shape_and_coords`: write a 3-step, 2-species, 10×10-grid dataset; open with xarray; assert dims `(3, 2, 10, 10)` and the coord values.
- `test_spatial_netcdf_nan_on_land`: one cell in the input is masked as land; assert that cell is NaN in the written NetCDF (not 0.0).
- `test_spatial_disabled_when_master_false`: `output.spatialized.enabled=false` → `StepOutput.spatial_biomass` is `None` and no spatial NetCDF files are written.
- `test_spatial_per_variant_toggle`: `output.spatialized.enabled=true` + only `output.spatialized.biomass.enabled=true` → only the biomass NetCDF is written; abundance and yield files absent.

---

## Non-goals

Explicit exclusions so a future reader or plan reviewer doesn't think they were missed:

- **Spatial TL / size / mortality / egg** outputs. Can be added in a follow-up without reshaping `StepOutput` once the spatial-biomass pattern is established.
- **CSV variant of spatial outputs.** NetCDF-only. Export-to-CSV is a two-line xarray idiom for users who need it.
- **Ev-OSMOSE-specific outputs.** `MeanGenotypeOutput`, `VariableTraitOutput`, `BiomassDietStageOutput`, `SpatialEnetOutput*` — deferred to the Ev-OSMOSE front (roadmap Front 6).
- **Debug/introspection outputs.** `ModularSchoolSetSnapshot`, `NewSchoolOutput`, `NSchoolOutput`, `NDeadSchoolOutput`, `AgeAtDeathOutput`. Not in the user-facing config surface; skip.
- **Per-fishery outputs.** `FisheryOutput`, `FisheryOutputDistrib` — blocked by DSVM scope, separate thread.
- **Whole-run diet summary.** Replaced by per-period; existing downstream code does not depend on it (grep-verified in `Scope` above).
- **Retroactive migration.** Existing `output/` directories from prior runs keep the old diet file format; no retroactive re-write.

## Implementation order

1. **Capability 5.5 (diet per-period)** — smallest, doesn't touch `StepOutput`. Commit.
2. **Capability 5.6 (NetCDF distributions + mortality)** — extends existing writer; uses existing `StepOutput` fields only. Commit.
3. **Capability 5.4 (spatial)** — adds three `StepOutput` fields, new collector, new writer, four new config-schema entries. Commit.
4. **Parity-roadmap + CHANGELOG** — Phase 5 STATUS-COMPLETE banner in `docs/parity-roadmap.md`, CHANGELOG entry. Commit.

## Testing strategy

- **Unit tests for pure helpers**: `_collect_spatial_outputs`, the ndarray-average branch of `_average_step_outputs`, `write_outputs_netcdf_spatial`. These don't need a full engine run.
- **Integration tests via the existing engine-validate harness**: run a tiny 2-species, 5×5-grid config for 3 steps end-to-end; assert all four new NetCDF / CSV output shapes exist. One integration test covering all three capabilities.
- **Parity test**: compare spatial-biomass totals summed over cells against the non-spatial biomass timeseries. `sum(spatial_biomass, axis=(lat, lon)) == biomass` must hold per species per timestep (within rounding tolerance ~1e-9).

## References

- Spec: `docs/superpowers/specs/2026-04-19-sp4-output-system-design.md` (this file).
- Plan: `docs/superpowers/plans/2026-04-19-sp4-output-system-plan.md` (forthcoming).
- Gap analysis (2026-04-19, summarized in this session's conversation): 7 Phase-5 items, 4 shipped, 1 partial (5.5), 1 partial (5.6), 1 missing (5.4).
- Parity roadmap: `docs/parity-roadmap.md` Phase 5.
- Current `StepOutput` definition: `osmose/engine/simulate.py:64-96`.
- Current `write_outputs_netcdf`: `osmose/engine/output.py:314`.
- Current `write_diet_csv`: `osmose/engine/output.py:207`.
- Java references:
  - `SpatialBiomassOutput_Netcdf.java`, `SpatialAbundanceOutput.java`, `SpatialYieldOutput_Netcdf.java` at `osmose-master/java/src/main/java/fr/ird/osmose/output/spatial/`.
  - `DietOutput_Netcdf.java`, `DietOutput.java` at `osmose-master/java/src/main/java/fr/ird/osmose/output/`.
  - `MortalityOutput.java`, distribution variants — same directory.
