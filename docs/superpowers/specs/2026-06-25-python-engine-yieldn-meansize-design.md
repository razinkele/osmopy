# Python-engine `yieldN` + `meanSize` outputs — design

> Status: design (awaiting review) · 2026-06-25
> Fills two documented Python-engine output-parity gaps (same pattern as PR #75
> community outputs). Output-only, additive, parity-safe. Verified against OSMOSE
> Java v4.4.1 (tag `d91300a`) + OSMOPY's canonical 4.4.0 schema.

## 1. Why

`engine_capabilities.py:_PYTHON_NOTABLE` lists what the Python engine does NOT produce
(forcing a Java run): "sizeSpectrum, meanSize, meanTLByAge, **yieldN**, fishery-yield". The
Python engine is now the UI default, so every one of these forces users onto Java. Two of
them — **`yieldN`** (fishing catch in numbers) and **`meanSize`** (abundance-weighted mean
length) — are clean to add: simple per-species outputs with **existing `results.py` readers**
(`yield_n` → `_read_species_output("yieldN")`; `mean_size` → `_read_species_output("meanSize")`),
and they exercise on the bundled multi-fishery configs (eec_full, baltic). (Reader names
verified against source: `results.yield_abundance()` reads `yieldN`; `results.mean_size()`
reads `meanSize`. There is no `results.yield_n()` method.) This is the same "fill a
Python-engine output gap" pattern PR #75 used for `meanTL`/`DistribBySize`.

(Deliberately deferred: `fishery-yield` — a NetCDF species×fishery output whose existing CSV
reader is stale and whose fidelity is limited by the engine's one-fishery-per-species collapse;
`sizeSpectrum`/`meanTLByAge` — higher effort, partly speculative.)

## 2. Definitions (Java v4.4.1-verified)

- **`yieldN`** = fishing catch in **numbers** per focal species per step = Σ over schools of
  `n_dead[FISHING]` (the existing `_collect_yield` *without* the `× weight`). No age cutoff
  (catch is catch — matching `_collect_yield`, which applies no cutoff). Unit: individuals.
- **`meanSize`** = **abundance-weighted** mean length per focal species (Java
  `MeanSizeOutput`: `Σ(abundance × length) / Σ(abundance)`, cm), **applying the same
  `output_cutoff_age` young-of-year filter** the Python engine already uses for `meanTL`/
  biomass/abundance. The per-step collector omits a species with no qualifying abundance from
  its dict (like `_collect_mean_tl`), but the **wide output frame keeps ALL species columns,
  NaN-filled** for empty ones (the `_build_meantl_dataframe` convention) — so CSV, in-memory,
  and NetCDF are all the same shape, matching Java's NaN-on-empty. Unit: cm.

## 3. Gating keys (verified against OSMOPY's canonical 4.4.0 schema + v4.4.1)

| Output | CSV key | NetCDF key |
|---|---|---|
| `yieldN` | `output.yield.abundance.enabled` (canonical, in schema) | `output.yield.abundance.netcdf.enabled` (canonical, in schema) |
| `meanSize` | `output.size.enabled` (canonical, in schema) | `output.size.netcdf.enabled` — **NEW key, added to schema** |

None of the CSV/yieldN keys are in the 4.3→4.4 alias-rename map (stable across versions).
The one new key (`output.size.netcdf.enabled`) extends OSMOPY's **per-variable** NetCDF
convention (Java instead uses a global NetCDF toggle to pick `MeanSizeOutput_Netcdf`; OSMOPY
has chosen per-variable `.netcdf.enabled` keys for all its NetCDF outputs, so meanSize follows
suit). It is added to `schema/output.py`'s NetCDF list so config validation accepts it.

## 4. Architecture / components (each mirrors an existing template)

1. **`osmose/schema/output.py`** — add `output.size.netcdf.enabled` to the NetCDF key list
   (the only schema change; the other three keys already exist).
2. **`osmose/engine/config.py`** — parse four flags into `EngineConfig`:
   `output_yield_abundance` ← `output.yield.abundance.enabled`,
   `output_mean_size` ← `output.size.enabled`,
   `output_yield_abundance_netcdf` ← `output.yield.abundance.netcdf.enabled`,
   `output_mean_size_netcdf` ← `output.size.netcdf.enabled` (mirrors the existing
   `output_yield_biomass_netcdf` parsing + field + `from_dict` wiring).
3. **`osmose/engine/simulate.py`** — `StepOutput` gains `yield_n: NDArray|None` and
   `mean_size: dict[int,float]|None`. New collectors:
   - `_collect_yield_n(state, config)` — mirrors `_collect_yield` minus `× weight`.
   - `_collect_mean_size(state, config)` — mirrors `_collect_mean_tl` but weights by
     `abundance × length / abundance` and reads `state.length` (instead of biomass × TL),
     applying `output_cutoff_age`.
   Wire into the per-step build (gated on the CSV-or-NetCDF flag being on) and into the two
   subdt-accumulation paths exactly as `yield_by_species` (summed) and `mean_tl`
   (`_avg_scalar_dict`) are.
4. **`osmose/engine/output.py`** —
   - CSV: `_write_yieldn_csv` → `{prefix}_yieldN_Simu0.csv` (mirrors `_write_yield_csv`);
     `_write_meansize_csv` → `{prefix}_meanSize_Simu0.csv` (mirrors `_write_meantl_csv`);
     `_build_yieldn_dataframe` returns `{"yieldN": df}`, `_build_meansize_dataframe` returns
     `{"meanSize": df}` (both wide all-species, NaN-filled per the meanTL convention). Register
     the two CSV writers in `write_outputs`.
   - **In-memory wiring (results.py):** add `"yieldN"` and `"meanSize"` to
     `_CROSS_SPECIES_OUTPUT_TYPES` (so the in-memory cache key is the bare output_type with
     `species="all"`, exactly what `yield_abundance()`/`mean_size()` request), and add the two
     new build helpers to `_build_dataframes_from_outputs`'s helper list + `disk_shape` block.
   - NetCDF (combined per-species file): two new entries in `write_outputs_netcdf`'s `want`
     dict + `data_vars` — `yieldN` on `["time","focal_species"]` (like `yield`), `meanSize`
     on `["time","species"]`, NaN-filled where a step's field is absent.
   - **NetCDF run-path wiring (the dormant-writer fix):** `write_outputs_netcdf` is currently
     never called by the engine's `write_outputs` entry point (only the CSV writers + the
     spatial NetCDF are). Add a call `write_outputs_netcdf(outputs, output_dir /
     f"{prefix}_Simu0.nc", config)` at the end of `write_outputs`. This is **safe**: the
     writer's first line is `if not any(want.values()): return`, so when no `.netcdf.enabled`
     flag is set (all default/parity configs) it writes nothing — zero effect on existing runs
     or the EEC/BoB suites. As a bonus it activates the already-parsed-but-dormant NetCDF flags
     (biomass/yield/etc.) for configs that enable them.
5. **`osmose/results.py` (combined-NetCDF read-back):** the produced `{prefix}_Simu0.nc` is
   read via the **existing** `read_netcdf(f"{prefix}_Simu0.nc")` (returns the `xr.Dataset` with
   the `yieldN`/`meanSize` variables; disk-backed only). Add thin convenience accessors
   `yield_abundance(..., source="netcdf")` / `mean_size(..., source="netcdf")` that pull the
   variable from that Dataset and return a frame shaped like the CSV reader (mapping the NetCDF
   `focal_species`/`species` dim to the CSV frame's species column). These accessors are
   **disk-backed only** (`read_netcdf` raises in in-memory mode), so the format-parity test's
   NetCDF leg must use a disk-backed `OsmoseResults`.
6. **`osmose/engine_capabilities.py`** — drop `yieldN` and `meanSize` from the
   `_PYTHON_NOTABLE` "not produced" string.

## 5. Data flow

Per step → `_collect_yield_n` / `_collect_mean_size` populate `StepOutput` (only when the
respective output is enabled) → subdt accumulation (sum for yieldN, average for meanSize) →
`write_outputs` writes CSV + the in-memory cache (`yieldN`/`meanSize` keys); when a `.netcdf`
flag is on, `write_outputs_netcdf` adds the variable to the combined NetCDF Dataset. Read back
via the existing `results.yield_abundance()` / `results.mean_size()` (CSV) and the
combined-NetCDF reader (see §4.4 NetCDF resolution).

## 6. Parity & correctness

- **Java cross-engine bar is "within ~1 OoM"**, not bit-exact (PCG64 vs MT19937). The
  collectors must be *formula-faithful* to Java (abundance-weighted meanSize w/ cutoff;
  yieldN = fishing deaths in numbers), which these are.
- **Empty-species convention:** the wide output frame keeps ALL species columns (the
  `_build_meantl_dataframe` convention), NaN where a species has no qualifying abundance —
  identical shape across CSV, in-memory, and NetCDF, matching Java's NaN-on-empty. (The
  per-step collector dict omits empty species; the wide builder re-expands to NaN columns.)
- **Subdt accumulation:** `yieldN` is **summed** across the record-frequency window (catch
  accumulates; `n_dead` resets each step — matches Java). `meanSize` is **averaged** via
  `_avg_scalar_dict` (mean of per-step ratios), which differs slightly from Java's
  ratio-of-sums when abundance varies within the window — but is **identical to the engine's
  existing `meanTL` treatment** and within the ~1-OoM bar. Do NOT "fix" it into a sum.
- **Engine dynamics are untouched** — these are pure output collectors gated behind flags.
  The 14/14 EEC `atol=0` + 8/8 BoB Java-parity suites must stay green (they assert biomass,
  not these new outputs).
- **NetCDF run-path wiring risk:** adding the `write_outputs_netcdf` call into `write_outputs`
  activates a previously-dormant path for ALL its variables. Mitigated by the writer's
  `if not any(want.values()): return` guard — it only does work when a config sets a
  `.netcdf.enabled` flag. The bundled parity/default configs set none, so they write no `.nc`
  and stay green. The implementation must confirm this (run a default-config + a netcdf-enabled
  config and check the `.nc` appears only for the latter).

## 7. Testing

- Unit: `_collect_yield_n` = Σ fishing deaths in numbers per species (vs a reference);
  `_collect_mean_size` = abundance-weighted with `output_cutoff_age` applied, empty species
  omitted **from the per-step dict** (the wide output frame then NaN-fills them — don't assert
  an omitted *column*); both return `None` when their flag is off (gating).
- Format parity: for a small NetCDF-enabled run, **CSV value == in-memory value == NetCDF
  value** for both outputs (all wide all-species/NaN, so shape-identical — the project's
  disk↔in-memory standard, extended to NetCDF).
- Reader round-trip: `results.yield_abundance()` / `results.mean_size()` read the produced
  CSV; `results.yield_abundance(source="netcdf")` / `mean_size(source="netcdf")` (or
  `read_netcdf(f"{prefix}_Simu0.nc")`) read the combined `.nc`.
- Wiring-safety: a default (no-`.netcdf`) config run writes **no** `{prefix}_Simu0.nc`; a
  `.netcdf`-enabled run writes it with the expected variables (confirms the early-return guard).
- The existing EEC/BoB parity suites stay green (no dynamics change).
- Optional: a within-1-OoM cross-check vs a captured Java `meanSize`/`yieldN` if a fixture
  exists (not required — no Java fixture is bundled).

## 8. Out of scope

- `fishery-yield` (NetCDF species×fishery), `sizeSpectrum`, `meanTLByAge`,
  `output.meansize.byage` — separate gaps.
- Any change to the engine's one-fishery-per-species fishing model or the mortality dynamics.
- A global `output.netcdf.enabled` toggle (OSMOPY stays per-variable).
