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

## Capability 5.5 — Diet per-recording-period (Java-parity)

### Current behaviour (whole-run sum)

`write_diet_csv(path, diet_by_species, predator_names, prey_names)` at `output.py:207` writes a single `(n_pred, n_prey)` diet matrix to one CSV. The *caller* at `output.py:76-87` iterates `outputs: list[StepOutput]` and sums `step.diet_by_species` across the whole run before invoking the writer. Result: one file per simulation, whole-run-integrated. The existing collection gate `output.diet.composition.enabled` is honored at `simulate.py:1094` (`enable_diet_tracking`) but the caller at `output.py:76-87` then writes regardless of the gate's intent — on a disabled run the gate short-circuits collection so the diet arrays are empty and the writer no-ops, but the gate is not explicit at the writer boundary.

### Target behaviour (Java-parity — append rows to one CSV)

Verified against `osmose-master/java/src/main/java/fr/ird/osmose/output/DietOutput.java:217-222`: Java writes **one** file `{prefix}_dietMatrix_Simu{rank}.csv` and appends rows each recording period, with `time` as the first column of every row. This is the same pattern as the biomass/yield/abundance CSVs already produced by `_write_species_csv`. My earlier spec proposed per-period files — that diverges from Java, not toward it. **Correcting the target**:

- Continue to emit one CSV per simulation (`{prefix}_dietMatrix_Simu{i}.csv`), matching Java's naming.
- Schema: first column `Time`, then one column per (predator, prey) pair in lexicographic order (equivalent to Java's column layout). One row per recording period. `n_periods` rows total.
- Rewrite `write_diet_csv` signature to accept the outputs list directly, preserving the existing CSV layout semantics:
  ```python
  def write_diet_csv(
      outputs: list[StepOutput],
      cfg: dict,
      output_dir: Path,
      prefix: str,
      sim_index: int,
      predator_names: list[str],
      prey_names: list[str],
  ) -> None:
  ```
  (The current caller at `output.py:76-87` collapses the per-step matrices by summing — replace that collapse with a per-row emission.)
- Remove the whole-run-summed matrix as the only output. The CSV that lands on disk now has one row per recording period, and the whole-run sum is trivially derivable by `df.drop(columns="Time").sum(axis=0)`.

### Config gating

Make the existing `output.diet.composition.enabled` (`config.py:772`) explicit at the writer boundary: if `cfg["output.diet.composition.enabled"] != "true"`, skip `write_diet_csv` entirely. Complementary to the collection gate at `simulate.py:1094`, not a replacement.

### Files touched

- Modify: `osmose/engine/output.py` — `write_diet_csv` signature + body; `write_outputs` call site at `:76-87`.
- Modify: **two** existing tests (not one — verified by reviewer):
  - `tests/test_engine_diet.py::test_write_diet_csv` (line 286) — old direct-ndarray signature.
  - `tests/test_engine_diet.py::test_write_diet_csv_percentage` (line 318) — same.
  Either migrate both to the new outputs-list signature, or keep a small private helper `_format_diet_row` with the old single-matrix shape and point the two existing tests at that helper instead of the public `write_diet_csv`.

### Tests

- `test_write_diet_csv_emits_one_row_per_recording_period`: three `StepOutput`s with distinct diet matrices; assert the written CSV has `Time` + expected pair columns, 3 data rows matching input step values.
- `test_write_diet_csv_skipped_when_config_disabled`: `output.diet.composition.enabled=false` → no file written.
- `test_write_diet_csv_empty_outputs_writes_no_file`: empty `outputs` list → no file (preserves the invariant "no data → no artifact").
- Update `tests/test_engine_phase5.py:145` (existing `output.diet.composition.enabled` parser test) to also assert the writer-gate behavior.

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
- `mortality_by_cause`: dims `(time, species, cause)`. Cause coord values are the **actual 8-member `MortalityCause` enum** (`osmose/engine/state.py:17-27`): `predation`, `starvation`, `additional`, `fishing`, `out`, `foraging`, `discards`, `aging`. (The earlier draft of this spec listed 6 causes with a `natural` label — that label does not exist in the enum; `additional` is the residual bucket, `aging` is senescence, `foraging` only populates when bioen is enabled. Match the enum verbatim, lowercase.)

**Ragged-bin handling is net-new, not an "extension".** `write_outputs_netcdf` currently has no per-species-ragged dict handling; it emits only `(time, species)` timeseries. The CSV distribution writer at `_write_distribution_csvs:121` takes `first_out`'s bin count only and is not robust to cross-species variation. The NetCDF extension computes per-step cross-species max bin count, preallocates `(time, species, max_bin)` float64 arrays filled with NaN, and scatters each species's row into the leading slots. Pad is NaN; the padded tail of the `age_bin` / `size_bin` coord is also NaN (not 0) so downstream plotters don't show spurious points at bin 0 for species with fewer bins.

**NetCDF conventions:** set `Dataset.attrs["Conventions"] = "CF-1.8"` and `_FillValue = NaN` on every float DataArray. Document in `Dataset.attrs["distribution_padding"]` that NaN represents both padded-tail (structural) and missing-data (runtime) — downstream tools treat them identically, which is the intended semantics.

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
- `test_netcdf_contains_mortality_by_cause`: same pattern for mortality; assert the `cause` coord has the 8 expected enum labels in the correct order.
- `test_netcdf_suppressed_when_master_disabled`: `output.netcdf.enabled=false` → no `.nc` file in the output dir.
- `test_netcdf_pads_ragged_age_bins_with_nan`: two species with different bin counts → NetCDF shape matches max, shorter species has NaN in the tail slots.
- `test_netcdf_pads_ragged_size_bins_with_nan`: parallel of the above for `biomass_by_size` / `abundance_by_size` — the size path is independent of age (flagged by reviewer).
- `test_netcdf_cf_conventions_attr`: assert `Dataset.attrs["Conventions"] == "CF-1.8"` and every float DataArray has `_FillValue = NaN`.

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

### Averaging (new per-field rules — not a simple reuse)

`_average_step_outputs` at `simulate.py:854-915` does **not** element-wise average dict fields today: distributions (`biomass_by_age` etc.) are **snapshot from `accumulated[-1]`** — the last step's values, not a mean (see the explicit comment at `:897`). The non-spatial scalar fields have heterogeneous rules: `biomass`/`abundance` are means, `mortality`/`yield` are sums, `diet_by_species` is summed. So there is no single "reuse existing pattern" for the three new spatial dicts — each one needs an explicit rule:

| Spatial field | Aggregation | Rationale |
|---|---|---|
| `spatial_biomass` | element-wise **mean** across accumulator | matches non-spatial `biomass` (line 891): instantaneous stock, averaged over the period |
| `spatial_abundance` | element-wise **mean** | matches non-spatial `abundance` (line 892) |
| `spatial_yield` | element-wise **sum** | matches non-spatial `yield_by_species` (line 893-896): fishing yield is a rate × interval integral |

Implementation: extend `_average_step_outputs` with a per-field dispatch that handles `dict[int, NDArray]` entries with the rule above. When any step has `spatial_biomass=None` (master disabled), emit `None` for the averaged output (same pairing-None rule as distributions).

### Writer

New function `write_outputs_netcdf_spatial(outputs, cfg, output_dir, prefix)` in `output.py`:

- One NetCDF file per enabled variant: `{prefix}_spatial_biomass_Simu{i}.nc`, `{prefix}_spatial_abundance_Simu{i}.nc`, `{prefix}_spatial_yield_Simu{i}.nc`.
- Dims: `(time, species, lat, lon)`.
- Coords: `time` from `StepOutput.step` values (which carry the last raw timestep of each averaging window — document this explicitly in `Dataset.attrs["time_convention"]`); `species` from species names; `lat` from `grid.lat` (1D `(ny,)` array, `osmose/engine/grid.py:33`) and `lon` from `grid.lon` (1D `(nx,)` array, `grid.py:34`). If either coord is `None` (grid without lat/lon metadata), write cell indices instead and record `attrs["spatial_coord_source"] = "cell_index"`.
- Values in tonnes (biomass), individuals (abundance), tonnes (yield) — matching the non-spatial variants.
- Cells outside the ocean mask written as NaN (not zero) so downstream plotting doesn't color land.

### Config keys (Java-compatible — use the existing schema keys, don't invent a parallel namespace)

**Correction from the earlier draft**: Java's `OutputManager` uses `output.spatial.*` for config keys (the `_spatialized_` suffix in Java is a *filename* fragment at `AbstractSpatialOutput.java:275`, not a config-key prefix). The Python schema at `osmose/schema/output.py:141-148` already declares `output.spatial.*` entries as UI-only flags — we repurpose those as the real engine gates, no new master key, no parallel namespace.

- `output.spatial.enabled` (master; already in schema) — gates `_collect_spatial_outputs` entirely.
- `output.spatial.biomass.enabled` (already in schema) — gates the biomass NetCDF writer.
- `output.spatial.abundance.enabled` (already in schema) — gates the abundance NetCDF writer.
- `output.spatial.yield.enabled` (already in schema) — gates the yield NetCDF writer.

When the master is `false`, the collector short-circuits and all three `spatial_*` fields on `StepOutput` are `None`. Per-variant toggles gate only the writer (always-collected-if-master-true, per the pairing invariant). If any Java reference configs happen to use `output.spatialized.*`, the Python config reader already normalizes key aliases at the reader layer — confirm in the plan before executing.

### Files touched

- Modify: `osmose/engine/simulate.py` — `StepOutput` gains three fields; `_collect_outputs` dispatches to new `_collect_spatial_outputs`; `_average_step_outputs` extended for the new dicts (element-wise mean, already supported pattern).
- Modify: `osmose/engine/output.py` — `write_outputs` calls the new spatial writer when the master is enabled; new `write_outputs_netcdf_spatial` function.
- Modify: `osmose/schema/output.py` — four new `OsmoseField` entries.

### Tests

- `test_collect_spatial_biomass_aggregates_by_cell`: 3 schools in 2 cells for 1 species; `_collect_spatial_outputs` produces a `(n_lat, n_lon)` array with the expected per-cell sums.
- `test_average_spatial_outputs_means_per_cell`: 2 step-outputs with different per-cell biomass for one species; averaged output is the elementwise mean.
- `test_spatial_netcdf_shape_and_coords`: write a 3-step, 2-species, 10×10-grid dataset; open with xarray; assert dims `(3, 2, 10, 10)` and the coord values.
- `test_spatial_netcdf_nan_on_land`: one cell in the input is masked as land; assert that cell is NaN in the written NetCDF (not 0.0).
- `test_spatial_disabled_when_master_false`: `output.spatial.enabled=false` → `StepOutput.spatial_biomass/abundance/yield` are all `None` (pairing invariant) and no spatial NetCDF files are written.
- `test_spatial_per_variant_toggle`: `output.spatial.enabled=true` + only `output.spatial.biomass.enabled=true` → the biomass NetCDF file **exists with expected `(time, species, lat, lon)` dims and coord values**, and the abundance and yield NetCDF files are **absent from `os.listdir(output_dir)`** (both positive and negative assertions explicit, per reviewer).
- `test_spatial_collection_runs_but_no_files_when_all_variants_disabled`: master `true` + all three per-variant toggles `false` → `StepOutput.spatial_*` populated (collection ran) but `os.listdir(output_dir)` contains zero `_spatial_*` files. Covers the seam where the pairing invariant could silently regress.
- `test_average_spatial_outputs_preserves_aggregation_rules`: parametrize over the three field names — `spatial_biomass` and `spatial_abundance` averaged element-wise (mean), `spatial_yield` summed element-wise. Each rule verified against a known 2-step input.

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
- **Integration tests via the existing engine-validate harness**: run a tiny 2-species, 5×5-grid config for 3 steps end-to-end; open each of the four new output files (`{prefix}_dietMatrix_Simu{i}.csv`, `{prefix}_Simu{i}.nc` with added distribution/mortality variables, `{prefix}_spatial_biomass_Simu{i}.nc`, …) and **assert dims/coord values match expectations** (file existence alone would miss an empty-or-malformed NetCDF). One integration test covering all three capabilities.
- **Parity test**: compare spatial-biomass totals summed over cells against the non-spatial biomass timeseries, with caveats that match the actual code paths:
  - `_collect_biomass_abundance` at `simulate.py:628` applies the `output.cutoff.age` filter and adds background-species biomass (which has no per-cell location). The spatial collector MUST replicate the same cutoff filter, and the parity assertion must exclude background species (restrict to focal species only).
  - Assert `np.allclose(spatial_biomass.sum(axis=(lat, lon)), biomass[focal_species_mask], rtol=1e-12, atol=0.0)` — relative tolerance, not absolute. The `1e-9` absolute in the earlier draft is too tight at tonnes-magnitude biomass (ulp accumulation scales with totals; 1e-9 of a 10⁶ tonnes total is below float64 precision).
  - If background-species biomass is ever nonzero in the test fixture, assert `spatial_biomass.sum() == (biomass - bkg_biomass).sum()` so the invariant is independent of background configuration.

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
