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
and they exercise on the bundled multi-fishery configs (eec_full, baltic). This is the same
"fill a Python-engine output gap" pattern PR #75 used for `meanTL`/`DistribBySize`.

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
  biomass/abundance. A species with no qualifying abundance is **omitted** (Java emits NaN;
  matching `_collect_mean_tl`, which omits empty species — see §6 parity note). Unit: cm.

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
     both registered in `write_outputs` and in the in-memory `_build_*_dataframe` path so
     disk and in-memory match (`yieldN` and `meanSize` cache keys, matching the readers).
   - NetCDF: two new entries in `write_outputs_netcdf`'s `want` dict + `data_vars` —
     `yieldN` on `["time","focal_species"]` (like `yield`), `meanSize` on `["time","species"]`,
     NaN-filled where a step's field is absent.
5. **`osmose/engine_capabilities.py`** — drop `yieldN` and `meanSize` from the
   `_PYTHON_NOTABLE` "not produced" string.

## 5. Data flow

Per step → `_collect_yield_n` / `_collect_mean_size` populate `StepOutput` (only when the
respective output is enabled) → subdt accumulation (sum for yieldN, average for meanSize) →
`write_outputs` writes CSV + the in-memory cache (`yieldN`/`meanSize` keys); when a `.netcdf`
flag is on, `write_outputs_netcdf` adds the variable to the combined NetCDF Dataset. Read back
via the existing `results.yield_n()` / `results.mean_size()` (CSV) and the NetCDF reader.

## 6. Parity & correctness

- **Java cross-engine bar is "within ~1 OoM"**, not bit-exact (PCG64 vs MT19937). The
  collectors must be *formula-faithful* to Java (abundance-weighted meanSize w/ cutoff;
  yieldN = fishing deaths in numbers), which these are.
- **Empty-species convention:** `meanSize` omits species with no qualifying abundance
  (consistent with the engine's `_collect_mean_tl`); the CSV/in-memory frame therefore has
  no row/value for them, and the NetCDF fills NaN — matching Java's NaN semantics at the
  consumer level.
- **Engine dynamics are untouched** — these are pure output collectors gated behind flags.
  The 14/14 EEC `atol=0` + 8/8 BoB Java-parity suites must stay green (they assert biomass,
  not these new outputs).

## 7. Testing

- Unit: `_collect_yield_n` = Σ fishing deaths in numbers per species (vs a reference);
  `_collect_mean_size` = abundance-weighted with `output_cutoff_age` applied + empty-species
  omitted; both return `None` when their flag is off (gating).
- Format parity: for a small run, **CSV value == in-memory value == NetCDF value** for both
  outputs (the project's disk↔in-memory standard, extended to NetCDF).
- Reader round-trip: `results.yield_n()` / `results.mean_size()` read the produced CSV; the
  NetCDF reader reads the combined `.nc`.
- The existing EEC/BoB parity suites stay green (no dynamics change).
- Optional: a within-1-OoM cross-check vs a captured Java `meanSize`/`yieldN` if a fixture
  exists (not required — no Java fixture is bundled).

## 8. Out of scope

- `fishery-yield` (NetCDF species×fishery), `sizeSpectrum`, `meanTLByAge`,
  `output.meansize.byage` — separate gaps.
- Any change to the engine's one-fishery-per-species fishing model or the mortality dynamics.
- A global `output.netcdf.enabled` toggle (OSMOPY stays per-variable).
