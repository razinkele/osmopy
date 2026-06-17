# Python-engine community outputs (DistribBySize + meanTL) — Design

**Date:** 2026-06-17
**Status:** Approved (brainstorming), pending implementation plan

## Goal

Make the Python simulation engine persist two outputs the Java engine writes but the Python engine
currently does not, so the community size-spectrum diagnostics light up on Python-only configs
(Baltic):

1. The **community size-distribution** files `{prefix}_biomassDistribBySize_Simu0.csv` and
   `{prefix}_abundanceDistribBySize_Simu0.csv` — wide `Time, Size, <species…>` — which the existing
   **Size Spectrum** diagnostic (`osmose/size_spectrum.py`) and the new **Sheldon spectrum**
   (`osmose/community_metrics.py`) read.
2. The **realized mean trophic level** 1D file `{prefix}_meanTL_Simu0.csv` — wide `Time, <species…>` —
   which `compute_trophic_indicators` (MTL / Marine Trophic Index) reads.

## Background — what already exists

- **Realized per-school TL is already computed.** In `osmose/engine/processes/mortality.py` (predation
  is a mortality source) the predation loop accumulates `ctx.tl_weighted_sum[p] += prey_TL · eaten`
  (school prey use prior-step `state.trophic_level[q]`; resource prey use `resources.species[r].trophic_level`),
  and post-loop sets `state.trophic_level = 1 + tl_weighted_sum / preyed_biomass` (mortality.py ≈2040-2054).
  This is the Java-faithful emergent TL. **No predation/hot-loop change is needed** — the value is
  maintained in `state.trophic_level`; this feature only surfaces it to an output file.
- **Per-size biomass/abundance is already computed.** `StepOutput` (`osmose/engine/simulate.py:76`)
  carries `biomass_by_size`/`abundance_by_size: dict[int, NDArray] | None` (keyed by species index →
  per-size-bin array), built in `_collect_outputs`/`_collect_distributions`. The engine already writes
  these as PER-SPECIES files `{prefix}_biomassBySize_<species>_Simu0.csv` (Time + size-bin columns) via
  `_build_distribution_dataframes`/`_write_distribution_csvs` in `osmose/engine/output.py`. The Java
  COMMUNITY layout (`Time, Size, <species>`) is a reshape of the same data and is what the readers want.
- Size-bin geometry: `config.output_size_min` (default 0.0) + k·`config.output_size_incr` (default
  10.0); the existing distribution writer labels size columns `f"{edge:.1f}"`.
- Config output flags live in `osmose/engine/config.py` (`_output` dict + `EngineConfig` fields):
  `output_biomass_bysize` ← `output.biomass.bysize.enabled`, `output_abundance_bysize` ←
  `output.abundance.bysize.enabled`. **There is no meanTL flag yet** — this feature adds
  `output_meantl` ← `output.meanTL.enabled`.
- The readers' expected formats (verified against `osmose/size_spectrum.py` + `osmose/results.py`):
  - `*DistribBySize*.csv`: wide `Time, Size, <species…>`; `_read_community_by_size` globs it (rglob,
    handles subdirs), sums species per (Time, Size).
  - `meanTL`: `OsmoseResults.mean_trophic_level()` reads `{prefix}_meanTL*.csv`, WIDE `Time` + one
    column per species. `_matches_output_type` keeps `meanTL` distinct from `meanTLBySize`.

## Architecture

All changes are **output-layer** (plus one captured field + one config flag). No predation/mortality
hot-loop edits. Files touched:

- `osmose/engine/config.py` — add the `output_meantl` flag (`_output` dict entry + `EngineConfig`
  field + `from_dict` wiring), matching the existing `output_biomass_bysize` pattern.
- `osmose/engine/simulate.py` — add `mean_tl: dict[int, float] | None = None` to `StepOutput`;
  populate it in `_collect_outputs` (and carry it through the averaging paths
  `_average_step_outputs`/spatial-average, consistent with how `biomass_by_size` is carried).
- `osmose/engine/output.py` — two new builder+writer pairs, wired into `write_outputs`:
  - `_build_distrib_bysize_community_dataframes(outputs, config)` → `{ "biomassDistribBySize": df,
    "abundanceDistribBySize": df }` (only the metrics whose bysize flag is set), each a wide
    `Time, Size, <species>` frame. `_write_distrib_bysize_community_csvs` writes
    `{prefix}_{key}_Simu0.csv`.
    - Build: for each output step and each size-bin index k, emit a row `(Time, Size=edge_k, sp0_val,
      sp1_val, …)` where `edge_k = output_size_min + k·output_size_incr` and `spX_val` is that
      species' per-size value (0.0 if the species has no entry for that step/bin). Result is the long
      `Time×Size` grid with one column per species — exactly the Java community layout.
  - `_build_meantl_dataframe(outputs, config)` → `{ "meanTL": df }` wide `Time` + one column per
    species, from each step's `mean_tl` dict (NaN where a species has no value that step).
    `_write_meantl_csv` writes `{prefix}_meanTL_Simu0.csv`. Gated by `config.output_meantl`.

### meanTL aggregation (in `_collect_outputs`)

Per output step, for each focal species compute the **abundance-weighted mean of
`state.trophic_level`** over that species' live schools with `trophic_level > 0` (exclude eggs /
never-fed schools whose TL is still the 0/baseline sentinel):

```
mean_tl[sp] = Σ_s (abundance_s · trophic_level_s) / Σ_s abundance_s     over schools s of sp with TL>0
```

A species with no qualifying school that step is omitted (→ NaN column cell). Background/resource
schools are excluded (focal species only, consistent with the other species outputs).

**Weighting choice (parity-determined):** the design uses **abundance-weighting** (mean TL of
individuals). This is the explicit default, but the **Java engine's `MeanTrophicLevel` output is the
arbiter** — the implementation MUST confirm whether Java weights by abundance or biomass and match it
(a one-line change to the weight); the parity test below is the source of truth. Likewise confirm
whether Java seeds a school's TL to the configured `species.trophic.level.spN` baseline (in which case
unfed schools contribute their baseline) or to 0 (excluded by the `TL>0` filter); match Java so the
early-step meanTL aligns.

## Data flow

`state.trophic_level` (already emergent) → `_collect_outputs` abundance-weighted per-species aggregate
→ `StepOutput.mean_tl` → `output.py` `_build_meantl_dataframe`/`_write_meantl_csv` → `{prefix}_meanTL_Simu0.csv`.

`state` per-size (existing) → `StepOutput.biomass_by_size`/`abundance_by_size` → `output.py`
`_build_distrib_bysize_community_dataframes`/writer → `{prefix}_{metric}DistribBySize_Simu0.csv`.

The existing per-species `{prefix}_biomassBySize_<sp>_Simu0.csv` files are **kept unchanged** (other
consumers/parity may use them); the community files are written alongside.

## Config gating

- `biomassDistribBySize` written only when `config.output_biomass_bysize`; `abundanceDistribBySize`
  only when `config.output_abundance_bysize` (reuse the existing flags — same data, community shape).
- `meanTL` written only when `config.output_meantl` (new flag, `output.meanTL.enabled`).
- When a flag is off, no file is written (the readers then degrade gracefully — already handled by
  `community_metrics`/`size_spectrum`).

## Error handling

- No `biomass_by_size`/`abundance_by_size` data in any step (feature disabled) → the community builder
  returns no entry for that metric; no file written.
- A step with an empty `mean_tl` dict → all-NaN row; a fully-absent species → omitted column.
- Size-bin count can vary across steps in principle; the builder uses the max bin count seen and pads
  shorter rows with 0.0 (matching `_build_distribution_dataframes`' zero-fill).

## Testing

- **Unit (output.py):** synthetic `StepOutput`s with known `biomass_by_size`/`abundance_by_size` →
  assert the community `DistribBySize` frame has columns `["Time","Size",*species]`, the right Size
  rows (`output_size_min + k·incr`), and correct per-species cells. Flag off → builder yields nothing.
- **Unit (meanTL aggregation):** a small `SchoolState` with two species, known abundances + trophic
  levels (incl. a TL=0 egg school that must be excluded) → assert `_collect_outputs` yields the
  hand-computed abundance-weighted mean per species, and `_build_meantl_dataframe` produces the wide
  frame; species absent that step → NaN.
- **Config:** `output.meanTL.enabled=true` round-trips to `config.output_meantl`.
- **Integration:** a short Python-engine run (Baltic subset or EEC) with the by-size + meanTL flags on
  → assert the three CSVs appear, and that `osmose.size_spectrum.compute_size_spectrum`,
  `osmose.community_metrics.compute_sheldon_spectrum`, and `compute_trophic_indicators` now succeed
  (Sheldon spectrum + MTL/MTI populate instead of degrading).
- **Parity:** on an EEC/BoB parity config run on BOTH engines, compare `meanTL` and the community
  `DistribBySize` against the Java outputs within the project's established parity tolerance (the
  realized-TL formula already matches Java, so meanTL should align closely; document any residual).

## Out of scope (YAGNI)

- 2D `meanTLBySize` / `meanTLByAge` outputs (the MTL/MTI consumer needs only 1D meanTL; the
  `OsmoseResults` 2D accessors already exist for Java output and aren't required here).
- NetCDF variants of the community DistribBySize (CSV is what the readers use).
- Removing or changing the existing per-species `biomassBySize_<sp>` files.
- Any change to the predation/mortality TL computation (already correct and Java-faithful).
