# Calibration throughput — PythonEngine in-memory: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `OsmoseCalibrationProblem._run_single` off Java subprocess and onto `PythonEngine` in-memory, eliminating JVM startup and disk round-trip per candidate. Target ≥ 4× speedup on Baltic NSGA-II.

**Architecture:** Extract 5 `_build_*_dataframes` helpers from `osmose/engine/output.py` (shared between disk and in-memory paths). Add `OsmoseResults.from_outputs()` factory + dispatch at `_read_species_output` / `_read_2d_output` / `read_netcdf`. Add `PythonEngine.run_in_memory()` that returns `OsmoseResults` directly. Port `OsmoseCalibrationProblem` to call it by default; keep Java subprocess behind `use_java_engine=True`.

**Tech Stack:** Python 3.12, NumPy, pandas (existing), pymoo NSGA-II (existing), Numba JIT (existing). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md` (HEAD `cb94146`, 4-round review converged).

**Baseline:** master `cb94146`, **2510 tests passing** / 15 skipped / 41 deselected, ruff clean, v0.9.3 shipped.

**Target state:** 5 commits on branch `feature/calibration-python-engine`:
- Task 1 — output.py refactor (5 build-helpers extracted, disk behavior preserved)
- Task 2 — results.py factory + dispatch + 8 tests (test_results_in_memory.py)
- Task 3 — engine/__init__.py run_in_memory + grid-resolver extract + 4 tests (test_python_engine_in_memory.py)
- Task 4 — calibration/problem.py port + 3 tests (test_calibration_problem_python_engine.py)
- Task 5 — benchmark script + CHANGELOG entry

Final pass count: **≥ 2525** (2510 + 15 new tests). Ruff clean. Benchmark ≥ 4× on Baltic NSGA-II.

**Ship target:** v0.10.0 (minor bump — breaking change to `OsmoseCalibrationProblem.__init__`).

**Known non-goals (reaffirmed from spec):**
- `preflight.py` stays as-is (still calls `PythonEngine.run()` with disk); follow-up phase.
- No `ProcessPoolExecutor` switch; `ThreadPoolExecutor` retained.
- No AOT Numba precompile.
- No in-memory `spatial_biomass` / NetCDF outputs.

---

## Pre-flight

- [ ] Confirm baseline. Run `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`. Expected: **2510 passed, 15 skipped, 41 deselected**. If different, record the observed count and use `baseline+15` framing everywhere (15 = new tests this plan adds).
- [ ] Lint baseline. Run `.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2`. Expected: `All checks passed!`.
- [ ] Verify the code anchors the spec cites (grep rather than trust line numbers):
  - `osmose/engine/output.py` — `def write_outputs(` at `:23`; `def _write_species_csv(` at `:104`; `def _write_distribution_csvs(` at `:120`; `def _write_mortality_csvs(` at `:165`; `def _write_yield_csv(` at `:273`; `def _write_bioen_csvs(` at `:303`; `def write_diet_csv(` at `:221`; `def write_outputs_netcdf_spatial(` at `:492`.
  - `osmose/results.py` — `class OsmoseResults` at `:137`; `def __init__(` at `:145`; `def _read_2d_output(` at `:369`; `def _read_species_output(` at `:409`; `def read_netcdf(` at `:192`; `self._csv_cache: dict[str, pd.DataFrame]` initialized in `__init__`.
  - `osmose/engine/__init__.py` — `class PythonEngine` at `:12`; `def run(` at `:18`; grid-loading block at `:27-56`.
  - `osmose/calibration/problem.py` — `class OsmoseCalibrationProblem` (verify exact class name — the spec caught that this is NOT `OsmoseProblem`); `def _run_single(` around `:174`; `subprocess.run(cmd, ...)` call around `:213`; `_cache_key` at `:271`.
  - `osmose/engine/simulate.py` — `def simulate(` at `:1016`; `output_dir: Path | None = None` at `:1023`; `if output_dir is not None` guard at `:1357`.

---

## Task 1: Extract `_build_*_dataframes` helpers from `osmose/engine/output.py`

**Goal:** Refactor the 5 CSV-writing helpers into a pair of steps: build wide-form DataFrames (new), write them to disk (existing). DataFrame-building helpers become the shared source of truth for disk and in-memory paths.

**Behavior-preserving.** Every existing output test passes unchanged after this task.

**Files:**
- Modify: `osmose/engine/output.py`

---

- [ ] **Step 1: Read the existing write_outputs end-to-end**

Open `osmose/engine/output.py`. Read `write_outputs()` (`:23-104`) and each `_write_*_csv` helper. Note what each produces:

- `_write_species_csv(path, header, times, species, data)` is called TWICE from `write_outputs`: once for `biomass`, once for `abundance`. Each writes a single CSV with `Time` + one column per species.
- `_write_distribution_csvs` writes per-species CSVs for 4 output_types: `biomassByAge`, `abundanceByAge`, `biomassBySize`, `abundanceBySize`. One file per (output_type, species) pair.
- `_write_mortality_csvs` writes per-species mortality files.
- `_write_yield_csv` writes a single yield CSV.
- `_write_bioen_csvs` writes bioen CSVs (gated by `config.bioen_enabled`).
- `write_diet_csv` writes the diet matrix CSV (gated by `config.diet_output_enabled`).
- `write_outputs_netcdf_spatial` writes NetCDF spatial outputs — **stays disk-only** in this plan.

---

- [ ] **Step 2: Add `_build_species_dataframes` helper**

Insert this helper above `_write_species_csv` (around `:104`):

```python
def _build_species_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for per-species time series (biomass, abundance).

    Returns a dict with keys "biomass" and "abundance", each mapping to a
    wide-form DataFrame with columns ["Time"] + config.all_species_names.
    These are the in-memory equivalents of the CSVs written by
    _write_species_csv (one row per step, one column per species).
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    species = config.all_species_names
    biomass_data = np.array([o.biomass for o in outputs])
    abundance_data = np.array([o.abundance for o in outputs])

    bio_df = pd.DataFrame(biomass_data, columns=list(species))
    bio_df.insert(0, "Time", times)

    abd_df = pd.DataFrame(abundance_data, columns=list(species))
    abd_df.insert(0, "Time", times)

    return {"biomass": bio_df, "abundance": abd_df}
```

---

- [ ] **Step 3: Add `_build_distribution_dataframes` helper**

Insert above `_write_distribution_csvs` (around `:120`):

```python
def _build_distribution_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for per-species age/size distributions.

    Returns a dict keyed by f"{output_type}_{species_name}" (e.g.,
    "biomassByAge_cod"), each mapping to a wide-form DataFrame with
    columns ["Time"] + bin labels. One entry per (output_type, species)
    pair that has data in the outputs list.
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    result: dict[str, pd.DataFrame] = {}

    for label, attr_name in [
        ("biomassByAge", "biomass_by_age"),
        ("abundanceByAge", "abundance_by_age"),
        ("biomassBySize", "biomass_by_size"),
        ("abundanceBySize", "abundance_by_size"),
    ]:
        first_out = next((o for o in outputs if getattr(o, attr_name) is not None), None)
        if first_out is None:
            continue
        dist_data = getattr(first_out, attr_name)
        for sp_idx, sp_name in enumerate(config.species_names):
            if sp_idx not in dist_data:
                continue
            n_bins = len(dist_data[sp_idx])
            data_matrix = np.zeros((len(outputs), n_bins))
            for t_idx, o in enumerate(outputs):
                d = getattr(o, attr_name)
                if d is not None and sp_idx in d:
                    data_matrix[t_idx, : len(d[sp_idx])] = d[sp_idx]

            if "Age" in label:
                columns = [str(i) for i in range(n_bins)]
            else:
                edges = np.arange(
                    config.output_size_min,
                    config.output_size_min + n_bins * config.output_size_incr,
                    config.output_size_incr,
                )
                columns = [f"{e:.1f}" for e in edges]

            df = pd.DataFrame(data_matrix, columns=columns)
            df.insert(0, "Time", times)
            result[f"{label}_{sp_name}"] = df

    return result
```

---

- [ ] **Step 4: Add `_build_mortality_dataframes` helper**

Insert above `_write_mortality_csvs`. Ported verbatim from `_write_mortality_csvs` at `:165-189`:

```python
def _build_mortality_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for per-species mortality-rate outputs.

    Ported verbatim from _write_mortality_csvs (output.py:165-189);
    only change is returning DataFrames instead of writing them.

    Returns {f"mortalityRate_{species_name}": df} with columns
    ["Time"] + [cause.name.capitalize() for cause in MortalityCause].
    """
    from osmose.engine.state import MortalityCause

    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    cause_names = [c.name.capitalize() for c in MortalityCause]

    result: dict[str, pd.DataFrame] = {}
    for sp_idx, sp_name in enumerate(config.species_names):
        data = np.array([o.mortality_by_cause[sp_idx] for o in outputs])
        df = pd.DataFrame(data, columns=cause_names)
        df.insert(0, "Time", times)
        result[f"mortalityRate_{sp_name}"] = df
    return result
```

---

- [ ] **Step 5: Add `_build_yield_dataframes` helper**

Read `_write_yield_csv` at `:273`. Extract to a helper that returns `{"yield": df}` or similar per-species keys if the current writer produces multiple files. Template:

```python
def _build_yield_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for yield outputs.

    Ported verbatim from _write_yield_csv (output.py:273-295); returns
    {"yield": df} with columns ["Time"] + config.species_names. Only
    change: DataFrame returned instead of written.
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    yield_data = np.array(
        [
            o.yield_by_species if o.yield_by_species is not None
            else np.zeros(config.n_species)
            for o in outputs
        ]
    )
    species = config.species_names
    df = pd.DataFrame(yield_data, columns=list(species))
    df.insert(0, "Time", times)
    return {"yield": df}
```

---

- [ ] **Step 6: Add `_build_bioen_dataframes` helper**

Insert above `_write_bioen_csvs`. Ported verbatim from `_write_bioen_csvs` at `:303-332`:

```python
def _build_bioen_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build wide-form DataFrames for bioenergetic outputs.

    Caller must check config.bioen_enabled before calling. Ported
    verbatim from _write_bioen_csvs (output.py:303-332); only change
    is returning DataFrames instead of writing them.

    Returns {f"{label}_{species_name}": df} for each enabled bioen
    output, where df has columns ["Time", label].
    """
    times = np.array([o.step / config.n_dt_per_year for o in outputs])

    bioen_outputs = [
        ("bioen_e_net_by_species", "meanEnet", True),
        ("bioen_ingestion_by_species", "ingestion", config.output_bioen_ingest),
        ("bioen_maint_by_species", "maintenance", config.output_bioen_maint),
        ("bioen_rho_by_species", "rho", config.output_bioen_rho),
        ("bioen_size_inf_by_species", "sizeInf", config.output_bioen_sizeinf),
    ]

    result: dict[str, pd.DataFrame] = {}
    for attr, label, enabled in bioen_outputs:
        if not enabled:
            continue
        data_list = [getattr(o, attr) for o in outputs]
        if not any(d is not None for d in data_list):
            continue
        data = np.array([d if d is not None else np.zeros(config.n_species) for d in data_list])
        for sp_idx, sp_name in enumerate(config.species_names):
            df = pd.DataFrame({"Time": times, label: data[:, sp_idx]})
            result[f"{label}_{sp_name}"] = df
    return result
```

---

- [ ] **Step 7: Add `_build_diet_dataframe` helper**

Diet is different from the other 5 — it's a 3D matrix written as a long-form CSV via `write_diet_csv` (`:221`). Insert:

```python
def _build_diet_dataframe(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Build the diet matrix DataFrame (long-form: one row per recording period).

    Caller must check config.diet_output_enabled before calling.
    Keyed by "dietMatrix" to match the single-file CSV output.
    Ported verbatim from write_diet_csv (output.py:221-256); the only
    change is returning a DataFrame instead of writing it to disk.
    """
    step_matrices: list[NDArray[np.float64]] = []
    step_times: list[float] = []
    for o in outputs:
        if o.diet_by_species is not None:
            step_matrices.append(o.diet_by_species)
            step_times.append(o.step / config.n_dt_per_year)
    if not step_matrices:
        return {}

    predator_names = config.species_names
    prey_names = config.all_species_names
    n_pred, n_prey = len(predator_names), len(prey_names)
    columns = [f"{pred}_{prey}" for pred in predator_names for prey in prey_names]
    rows: list[list[float]] = []
    for mat, t in zip(step_matrices, step_times, strict=True):
        if mat.shape != (n_pred, n_prey):
            raise ValueError(
                f"diet matrix shape {mat.shape} != ({n_pred}, {n_prey}) at time {t}"
            )
        rows.append([t, *mat.reshape(-1).tolist()])
    df = pd.DataFrame(rows, columns=["Time", *columns])
    return {"dietMatrix": df}
```

Port `write_diet_csv` (`osmose/engine/output.py:221`) into the DataFrame-building part. Verify by comparing the existing CSV's header line to the produced DataFrame's columns.

---

- [ ] **Step 8: Rewrite each `_write_*_csv` helper to delegate to its build helper**

**Design:** preserve the existing file-writing structure (subdirectories, commentary headers, filename conventions), delegate the DataFrame-construction to the new build helpers. Each `_write_*_csv` becomes a thin wrapper.

Update `_write_mortality_csvs` (`:165`):

```python
def _write_mortality_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-species mortality rate CSVs matching Java format."""
    mort_dir = output_dir / "Mortality"
    mort_dir.mkdir(exist_ok=True)

    dfs = _build_mortality_dataframes(outputs, config)
    for key, df in dfs.items():
        # key is "mortalityRate_{sp_name}"
        sp_name = key.split("_", 1)[1]
        path = mort_dir / f"{prefix}_mortalityRate-{sp_name}_Simu0.csv"
        with open(path, "w") as f:
            f.write(f'"Mortality rates per time step for {sp_name}"\n')
            df.to_csv(f, index=False)
```

Update `_write_yield_csv` (`:273`):

```python
def _write_yield_csv(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write fishing yield CSV matching Java format."""
    dfs = _build_yield_dataframes(outputs, config)
    if "yield" not in dfs:
        return
    df = dfs["yield"]
    times = df["Time"].values
    species = [c for c in df.columns if c != "Time"]
    yield_data = df[species].values
    _write_species_csv(
        output_dir / f"{prefix}_yield_Simu0.csv",
        "Fishing yield (tons) per time step",
        times,
        species,
        yield_data,
    )
```

(The `_write_species_csv` primitive at `:104` is retained — it's the low-level CSV+header writer; `_write_yield_csv` now builds the DataFrame first via `_build_yield_dataframes`, then delegates to the primitive.)

Similarly for `write_outputs`'s direct biomass/abundance writes — replace the inline `_write_species_csv` calls at `:43-60` with DataFrame-path-aware calls:

```python
def write_outputs(
    outputs: list[StepOutput],
    output_dir: Path,
    config: EngineConfig,
    prefix: str = "osm",
    *,
    grid=None,
) -> None:
    """Write simulation outputs to CSV files matching Java format.

    Build-then-write: each family's CSV is built as a DataFrame via
    _build_*_dataframes, then _write_species_csv writes it to disk
    with the appropriate commentary header. The build helpers are the
    shared source of truth for disk and in-memory (OsmoseResults.from_outputs)
    paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Species-level time series (biomass, abundance): use _build_species_
    # dataframes, then pass through _write_species_csv for header writing.
    species_dfs = _build_species_dataframes(outputs, config)
    headers = {
        "biomass": "Mean biomass (tons), excluding first ages specified in input",
        "abundance": "Mean abundance (number of fish), excluding first ages specified in input",
    }
    for key in ("biomass", "abundance"):
        df = species_dfs[key]
        times = df["Time"].values
        species = [c for c in df.columns if c != "Time"]
        data = df[species].values
        _write_species_csv(
            output_dir / f"{prefix}_{key}_Simu0.csv",
            headers[key],
            times,
            species,
            data,
        )

    _write_mortality_csvs(output_dir, prefix, outputs, config)
    _write_yield_csv(output_dir, prefix, outputs, config)
    _write_distribution_csvs(output_dir, prefix, outputs, config)

    if config.bioen_enabled:
        _write_bioen_csvs(output_dir, prefix, outputs, config)

    if config.diet_output_enabled:
        step_matrices: list[NDArray[np.float64]] = []
        step_times: list[float] = []
        for o in outputs:
            if o.diet_by_species is not None:
                step_matrices.append(o.diet_by_species)
                step_times.append(o.step / config.n_dt_per_year)
        write_diet_csv(
            path=output_dir / f"{prefix}_dietMatrix_Simu0.csv",
            step_diet_matrices=step_matrices,
            step_times=step_times,
            predator_names=config.species_names,
            prey_names=config.all_species_names,
        )

    if config.output_spatial_enabled:
        write_outputs_netcdf_spatial(
            outputs,
            output_dir,
            prefix=prefix,
            sim_index=0,
            config=config,
            grid=grid,
        )
```

Update `_write_distribution_csvs` (`:120`) and `_write_bioen_csvs` (`:303`) the same way — each becomes: call the build helper, iterate over its returned dict, write each DataFrame to its canonical path. Update `_write_distribution_csvs`:

```python
def _write_distribution_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write per-species age/size distribution CSVs matching Java format."""
    dfs = _build_distribution_dataframes(outputs, config)
    for key, df in dfs.items():
        # key is "{output_type}_{species}"
        path = output_dir / f"{prefix}_{key}_Simu0.csv"
        df.to_csv(path, index=False)
```

Update `_write_bioen_csvs`:

```python
def _write_bioen_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write bioen-specific per-species CSVs into a Bioen/ subdirectory."""
    bioen_dir = output_dir / "Bioen"
    bioen_dir.mkdir(exist_ok=True)

    dfs = _build_bioen_dataframes(outputs, config)
    for key, df in dfs.items():
        # key is "{label}_{sp_name}"
        path = bioen_dir / f"{prefix}_{key}_Simu0.csv"
        df.to_csv(path, index=False)
```

The diet CSV stays routed through `write_diet_csv` for its current behavior in `write_outputs` — the build helper `_build_diet_dataframe` is used only by the in-memory path (Task 2).

---

- [ ] **Step 9: Run the existing output-tests suite — expect unchanged pass count**

```bash
.venv/bin/python -m pytest tests/ -q --no-header -k "output" 2>&1 | tail -5
```

Expected: all pre-existing output tests still pass. If any fail, a build helper is producing a different DataFrame than the inline code did. Fix the helper (not the test) — the tests are the regression net.

Also run the full suite:

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: **2510 passed, 15 skipped, 41 deselected** — unchanged from baseline.

---

- [ ] **Step 10: Run ruff**

```bash
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: `All checks passed!`.

---

- [ ] **Step 11: Commit**

Write the commit message to `/tmp/task1.msg` via the Write tool:

```
refactor(output): extract _build_*_dataframes helpers (calibration-python-engine task 1)

Split each _write_*_csv helper in osmose/engine/output.py into a pair
of steps: build wide-form DataFrames (new) + write them to disk
(existing). The build helpers are the shared source of truth for both
the disk path (write_outputs) and the upcoming in-memory path
(OsmoseResults.from_outputs in Task 2).

Six new module-level helpers:
  - _build_species_dataframes (biomass, abundance)
  - _build_distribution_dataframes (biomassByAge, abundanceByAge,
    biomassBySize, abundanceBySize — one per species per type)
  - _build_mortality_dataframes (mortalityRate per species)
  - _build_yield_dataframes (yield + per-fishery variants if present)
  - _build_bioen_dataframes (bioen CSVs; gated on config.bioen_enabled
    at the caller)
  - _build_diet_dataframe (dietMatrix; gated on
    config.diet_output_enabled at the caller)

write_outputs() rewritten as a dispatch loop over the dict returned
by each helper, with a small header-string table for the two
commentary-header CSVs (biomass, abundance).

Behavior-preserving. All existing output tests pass unchanged. Full
suite still 2510 passed / 15 skipped / 41 deselected. Ruff clean.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Then:

```bash
git add osmose/engine/output.py
git commit -F /tmp/task1.msg
```

---

## Task 2: Add `OsmoseResults.from_outputs()` factory + dispatch + 8 tests

**Goal:** Add the in-memory results factory and its tests in a single commit. After this task, `OsmoseResults.from_outputs()` is a complete, tested replacement for the disk round-trip on every CSV-backed getter.

**Files:**
- Modify: `osmose/results.py` (add factory + dispatch + `_build_dataframes_from_outputs` + `_melt_wide_to_long` helper)
- Create: `tests/test_results_in_memory.py` (8 tests)

---

- [ ] **Step 1: Add `_build_dataframes_from_outputs` to `osmose/results.py`**

Top of file, add the import from output.py:

```python
from osmose.engine.output import (
    _build_species_dataframes,
    _build_distribution_dataframes,
    _build_mortality_dataframes,
    _build_yield_dataframes,
    _build_bioen_dataframes,
    _build_diet_dataframe,
)
```

Then add a module-level private helper (above the `OsmoseResults` class). This helper produces the CACHE SHAPE directly — see Step 4 below for the shape rationale.

```python
def _build_dataframes_from_outputs(
    outputs: list[StepOutput],
    config: EngineConfig,
    grid: Grid,
) -> dict[str, pd.DataFrame]:
    """Build OsmoseResults._csv_cache from in-memory StepOutputs.

    Returns a dict mapping output_type (e.g., "biomass", "biomassByAge")
    to a LONG-form DataFrame with an added "species" column — exactly
    the shape that _read_species_output builds from on-disk per-file reads.

    Two steps:
    1. Call each _build_*_dataframes helper (disk-shape dicts).
    2. Adapt to cache shape: group by output_type, add species column,
       concatenate same-output_type entries.
    """
    # Step 1: gather disk-shape dicts
    disk_shape: dict[str, pd.DataFrame] = {}
    disk_shape.update(_build_species_dataframes(outputs, config))
    disk_shape.update(_build_mortality_dataframes(outputs, config))
    disk_shape.update(_build_yield_dataframes(outputs, config))
    disk_shape.update(_build_distribution_dataframes(outputs, config))
    if config.bioen_enabled:
        disk_shape.update(_build_bioen_dataframes(outputs, config))
    if config.diet_output_enabled:
        disk_shape.update(_build_diet_dataframe(outputs, config))
    _ = grid  # reserved for future NetCDF-in-memory work

    # Step 2: adapt disk shape to cache shape.
    # Cross-species disk entries keep species="all"; per-species entries
    # (f"{output_type}_{sp_name}") split back into (output_type, sp_name).
    _CROSS_SPECIES = {"biomass", "abundance", "yield", "dietMatrix"}
    cache_shape: dict[str, list[pd.DataFrame]] = {}
    for key, df in disk_shape.items():
        if key in _CROSS_SPECIES:
            output_type, sp_name = key, "all"
        else:
            output_type, _, sp_name = key.partition("_")
        annotated = df.copy()
        annotated["species"] = sp_name
        cache_shape.setdefault(output_type, []).append(annotated)

    return {
        ot: pd.concat(frames, ignore_index=True)
        for ot, frames in cache_shape.items()
    }
```

---

- [ ] **Step 2: Add `_melt_wide_to_long` module-level helper**

Extract the melt logic from `_read_2d_output:388-397` into a reusable helper (disk path will delegate to it in Step 4):

```python
def _melt_wide_to_long(
    wide_df: pd.DataFrame,
    species: str | None = None,
) -> pd.DataFrame:
    """Melt a wide-form DataFrame (Time + bin columns) into long-form.

    Returns columns: time, species, bin, value.
    If wide_df has a 'species' column, it's preserved; otherwise a
    'species' column is added with the value 'all'.
    """
    time_col = wide_df.columns[0]
    has_species = "species" in wide_df.columns
    id_cols = [time_col] + (["species"] if has_species else [])
    bin_cols = [c for c in wide_df.columns[1:] if c != "species"]

    melted = wide_df.melt(
        id_vars=id_cols,
        value_vars=bin_cols,
        var_name="bin",
        value_name="value",
    )
    melted = melted.rename(columns={time_col: "time"})
    if not has_species:
        melted["species"] = "all"
    melted = melted[["time", "species", "bin", "value"]]

    if species is not None:
        melted = melted[melted["species"] == species]
    return melted
```

---

- [ ] **Step 3: Add `self._in_memory` attribute + `from_outputs()` factory**

In `OsmoseResults.__init__`, add one line setting `self._in_memory = False` (existing disk-backed instances behave unchanged). Add the factory as a classmethod:

```python
@classmethod
def from_outputs(
    cls,
    outputs: list[StepOutput],
    engine_config: EngineConfig,
    grid: Grid,
    *,
    prefix: str = "osm",
) -> OsmoseResults:
    """Construct an OsmoseResults backed by in-memory StepOutputs.

    No disk I/O. Supports every CSV-backed getter (biomass, abundance,
    yield_biomass, mortality, diet_matrix, mean_size, mean_trophic_level,
    biomass_by_age, biomass_by_size, biomass_by_tl, abundance_by_age, etc.).
    Getters that return xr.Dataset from NetCDF (spatial_biomass, read_netcdf)
    raise FileNotFoundError in this mode.

    Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
    """
    obj = cls.__new__(cls)
    obj.output_dir = None
    obj.prefix = prefix
    obj.strict = True
    obj._csv_cache = _build_dataframes_from_outputs(outputs, engine_config, grid)
    # __init__ initializes _nc_cache = {} for NetCDF caching; since we're
    # bypassing __init__ via __new__, set it explicitly. close_cache()
    # iterates _nc_cache and would raise AttributeError otherwise.
    obj._nc_cache = {}
    obj._in_memory = True
    return obj
```

---

- [ ] **Step 4: Add dispatch to `_read_species_output`**

`_build_dataframes_from_outputs` (from Step 1) already produces the cache shape `_read_species_output` expects: key is `output_type`, value is a long-form DataFrame with a `species` column. So the dispatch is a clean early-return at the top of the method.

Edit `_read_species_output` (at `osmose/results.py:409`) — add the guard BEFORE the existing `if cache_key not in self._csv_cache:` block:

```python
def _read_species_output(self, output_type: str, species: str | None) -> pd.DataFrame:
    """Read CSV output files for a given output type.
    ...
    """
    if self._in_memory:
        if output_type not in self._csv_cache:
            raise FileNotFoundError(
                f"In-memory OsmoseResults has no '{output_type}' output. "
                f"Available: {sorted(self._csv_cache.keys())}"
            )
        combined = self._csv_cache[output_type]
        if species:
            combined = combined[combined["species"] == species]
        return combined

    # ...existing disk path unchanged...
```

**Verification query (run during implementation after Task 3 lands to confirm disk/memory agreement):**

```bash
.venv/bin/python -c "
from pathlib import Path
import tempfile
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults

cfg = OsmoseConfigReader().read(Path('data/examples/osm_all-parameters.csv'))
cfg['simulation.time.nyear'] = '1'
with tempfile.TemporaryDirectory() as d:
    PythonEngine().run(cfg, Path(d), seed=42)
    r_disk = OsmoseResults(Path(d))
    df_disk = r_disk.biomass()
r_mem = PythonEngine().run_in_memory(cfg, seed=42)
df_mem = r_mem.biomass()
print('disk cols:', list(df_disk.columns))
print('memory cols:', list(df_mem.columns))
print('equal:', df_disk.equals(df_mem))
"
```

Expected: same columns, `equal` is True. If False, inspect `df_disk.compare(df_mem)` — likely delta is column ordering or species-value strings (e.g., "all" vs empty).

---

- [ ] **Step 5: Add dispatch to `_read_2d_output`**

At the top of `_read_2d_output` (`:369`):

```python
def _read_2d_output(self, output_type: str, species: str | None = None) -> pd.DataFrame:
    if self._in_memory:
        if output_type not in self._csv_cache:
            raise FileNotFoundError(
                f"In-memory OsmoseResults has no '{output_type}' output. "
                f"Available: {sorted(self._csv_cache.keys())}"
            )
        return _melt_wide_to_long(self._csv_cache[output_type], species)

    # ...existing disk path unchanged; optionally refactor it to call
    # _melt_wide_to_long instead of the inline melt, for DRY.
```

---

- [ ] **Step 6: Add dispatch to `read_netcdf`**

At the top of `read_netcdf` (`:192`):

```python
def read_netcdf(self, filename: str) -> xr.Dataset:
    if self._in_memory:
        raise FileNotFoundError(
            f"In-memory OsmoseResults does not support NetCDF outputs "
            f"(requested: {filename}). Use the disk-backed OsmoseResults "
            f"constructor if you need spatial NetCDF outputs."
        )
    # ...existing disk path unchanged...
```

---

- [ ] **Step 7: Create `tests/test_results_in_memory.py` with 8 tests**

Create the test file verbatim:

```python
"""Unit tests for OsmoseResults.from_outputs() and in-memory dispatch.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.output import write_outputs
from osmose.engine.simulate import simulate
from osmose.engine.rng import build_rng
from osmose.results import (
    OsmoseResults,
    _build_dataframes_from_outputs,
)

EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _run_short_simulation(seed: int = 42):
    """Run a 1-year BoB simulation; return (outputs, config, grid)."""
    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    config = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid_path = EXAMPLE_CONFIG.parent / grid_file
        grid = Grid.from_netcdf(grid_path, mask_var=raw.get("grid.var.mask", "mask"))
    else:
        ny = int(raw.get("grid.nlon", raw.get("grid.ncol", "10")))
        nx = int(raw.get("grid.nlat", raw.get("grid.nrow", "10")))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)
    movement_rngs = build_rng(seed, config.n_species, config.movement_seed_fixed)
    mortality_rngs = build_rng(seed + 1, config.n_species, config.mortality_seed_fixed)

    outputs = simulate(
        config, grid, rng,
        movement_rngs=movement_rngs,
        mortality_rngs=mortality_rngs,
        output_dir=None,
    )
    return outputs, config, grid


@pytest.fixture(scope="module")
def _disk_and_memory_results():
    """Shared fixture: same simulation, two results — one disk, one memory."""
    outputs, config, grid = _run_short_simulation(seed=42)
    with tempfile.TemporaryDirectory() as d:
        write_outputs(outputs, Path(d), config)
        disk_results = OsmoseResults(Path(d))
        disk_results.biomass()  # force-cache everything before temp dir vanishes
        disk_results.abundance()
        try:
            disk_results.yield_biomass()
        except FileNotFoundError:
            pass
        try:
            disk_results.mortality()
        except FileNotFoundError:
            pass
        try:
            disk_results.diet_matrix()
        except FileNotFoundError:
            pass
    memory_results = OsmoseResults.from_outputs(outputs, config, grid)
    return disk_results, memory_results


def test_biomass_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    pd.testing.assert_frame_equal(
        disk.biomass().reset_index(drop=True),
        memory.biomass().reset_index(drop=True),
    )


def test_abundance_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    pd.testing.assert_frame_equal(
        disk.abundance().reset_index(drop=True),
        memory.abundance().reset_index(drop=True),
    )


def test_yield_biomass_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    try:
        disk_df = disk.yield_biomass()
    except FileNotFoundError:
        pytest.skip("yield output not produced by this fixture config")
    pd.testing.assert_frame_equal(
        disk_df.reset_index(drop=True),
        memory.yield_biomass().reset_index(drop=True),
    )


def test_mortality_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    try:
        disk_df = disk.mortality()
    except FileNotFoundError:
        pytest.skip("mortality output not produced by this fixture config")
    pd.testing.assert_frame_equal(
        disk_df.reset_index(drop=True),
        memory.mortality().reset_index(drop=True),
    )


def test_diet_matrix_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    try:
        disk_df = disk.diet_matrix()
    except FileNotFoundError:
        pytest.skip("diet_matrix output not produced by this fixture config")
    pd.testing.assert_frame_equal(
        disk_df.reset_index(drop=True),
        memory.diet_matrix().reset_index(drop=True),
    )


def test_spatial_biomass_raises_FileNotFoundError_in_memory_mode():
    outputs, config, grid = _run_short_simulation(seed=1)
    r = OsmoseResults.from_outputs(outputs, config, grid)
    with pytest.raises(FileNotFoundError, match="does not support NetCDF"):
        r.spatial_biomass("anything.nc")


def test_from_outputs_idempotent():
    """Calling the same getter twice returns the same DataFrame (cached)."""
    outputs, config, grid = _run_short_simulation(seed=7)
    r = OsmoseResults.from_outputs(outputs, config, grid)
    first = r.biomass()
    second = r.biomass()
    pd.testing.assert_frame_equal(first, second)


def test_from_outputs_populates_all_written_keys():
    """Cache keys from _build_dataframes_from_outputs equal keys that
    write_outputs would write to disk. Single source of truth for the
    supported-output-type set.
    """
    outputs, config, grid = _run_short_simulation(seed=42)

    memory_keys = set(_build_dataframes_from_outputs(outputs, config, grid).keys())

    with tempfile.TemporaryDirectory() as d:
        write_outputs(outputs, Path(d), config)
        disk_files = list(Path(d).glob(f"{config.__class__.__name__.lower()}_*.csv"))
        # Extract the {output_type}[_{species}] stem from filenames like
        # osm_biomass_Simu0.csv -> biomass
        # osm_biomassByAge_cod_Simu0.csv -> biomassByAge_cod
        disk_keys = set()
        for f in disk_files:
            stem = f.stem  # e.g., "osm_biomass_Simu0" or "osm_biomassByAge_cod_Simu0"
            # Remove "osm_" prefix and "_Simu0" suffix
            inner = stem.split("_", 1)[1].rsplit("_", 1)[0]
            disk_keys.add(inner)
    # Exclude NetCDF-only outputs from comparison
    assert memory_keys == disk_keys, (
        f"In-memory and disk paths disagree on output types.\n"
        f"In memory only: {memory_keys - disk_keys}\n"
        f"On disk only:   {disk_keys - memory_keys}"
    )
```

---

- [ ] **Step 8: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_results_in_memory.py -v 2>&1 | tail -20
```

Expected: 8 passed. Some may skip if the Bay of Biscay example doesn't produce certain outputs (yield, mortality, diet_matrix) at n_year=1 — that's fine; `pytest.skip` is built into those tests.

Full suite:

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: **2518 passed, ≥15 skipped, 41 deselected** (2510 + 8 new; some may skip). ≥ 2510 passing.

---

- [ ] **Step 9: Ruff**

```bash
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: `All checks passed!`.

---

- [ ] **Step 10: Commit**

Write to `/tmp/task2.msg`:

```
feat(results): OsmoseResults.from_outputs() in-memory factory (task 2)

Add OsmoseResults.from_outputs(outputs, engine_config, grid, *, prefix)
classmethod that wraps an in-memory list[StepOutput] and serves the
same CSV-backed getter API as the disk-backed instance. No disk I/O.

Implementation:
- _build_dataframes_from_outputs(outputs, config, grid) in results.py
  calls each _build_*_dataframes helper from osmose/engine/output.py
  and merges into a single dict keyed by output_type (or
  output_type_species for per-species outputs). Single source of truth:
  write_outputs (disk) and from_outputs (memory) consume the same
  build helpers.
- Dispatch at _read_species_output, _read_2d_output, and read_netcdf:
  when self._in_memory is True, serve from self._csv_cache or raise
  FileNotFoundError with actionable error message.
- New self._in_memory: bool attribute. False for existing disk-backed
  constructor; True when set by from_outputs().
- _melt_wide_to_long helper extracted from _read_2d_output's inline
  melt logic for reuse.

Tests (8): tests/test_results_in_memory.py covers
  - disk/memory getter parity for biomass, abundance, yield_biomass,
    mortality, diet_matrix (pd.testing.assert_frame_equal)
  - NetCDF getter raises FileNotFoundError in-memory mode
  - from_outputs idempotent
  - parity invariant: set of keys produced by _build_dataframes_from_
    outputs equals set of {output_type}[_{species}] stems written by
    write_outputs

Zero behavior change on disk-backed path. Full suite 2510 -> 2518 passed.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

```bash
git add osmose/results.py tests/test_results_in_memory.py
git commit -F /tmp/task2.msg
```

---

## Task 3: Add `PythonEngine.run_in_memory()` + grid-resolver extract + 4 tests

**Goal:** Give callers a direct in-memory `OsmoseResults`-returning API. Extract the grid-loading block so `run()` and `run_in_memory()` share it.

**Files:**
- Modify: `osmose/engine/__init__.py`
- Create: `tests/test_python_engine_in_memory.py` (4 tests)

---

- [ ] **Step 1: Extract `_resolve_grid` private helper**

In `osmose/engine/__init__.py`, add a new private method on `PythonEngine`:

```python
def _resolve_grid(self, config: dict[str, str]) -> Grid:
    """Resolve grid from config — shared between run() and run_in_memory()."""
    grid_file = config.get("grid.netcdf.file", "")
    mask_var = config.get("grid.var.mask", "mask")
    lat_var = config.get("grid.var.lat", "latitude")
    lon_var = config.get("grid.var.lon", "longitude")

    if grid_file:
        config_dir = config.get("_osmose.config.dir", "")
        search_bases = [Path(".")]
        if config_dir:
            search_bases.insert(0, Path(config_dir))
        search_bases.append(Path("data/examples"))
        for base in search_bases:
            path = base / grid_file
            if path.exists():
                return Grid.from_netcdf(
                    path, mask_var=mask_var, lat_dim=lat_var, lon_dim=lon_var
                )
        searched = [str(b / grid_file) for b in search_bases]
        raise FileNotFoundError(
            f"Grid file '{grid_file}' not found in search paths: {searched}. "
            "Set grid.netcdf.file to an existing file or remove the key "
            "to use a rectangular grid."
        )

    nx = int(config.get("grid.nlon", config.get("grid.ncol", "10")))
    ny = int(config.get("grid.nlat", config.get("grid.nrow", "10")))
    return Grid.from_dimensions(ny=ny, nx=nx)
```

Replace the existing grid-loading block in `run()` (lines 27-56) with:

```python
grid = self._resolve_grid(config)
```

---

- [ ] **Step 2: Add `run_in_memory` method**

After `run()`, add:

```python
def run_in_memory(
    self,
    config: dict[str, str],
    seed: int = 0,
) -> OsmoseResults:
    """Run the Python engine and return results as an in-memory OsmoseResults.

    Equivalent to run() except:
      - No output_dir argument.
      - No write_outputs() call.
      - Returns an OsmoseResults that serves DataFrames from the in-memory
        StepOutput list rather than from disk.

    Use this for calibration candidates, sensitivity analysis, or any other
    throughput-sensitive workflow where disk output is not needed.
    """
    from osmose.engine.config import EngineConfig
    from osmose.engine.rng import build_rng
    from osmose.engine.simulate import simulate
    from osmose.results import OsmoseResults

    engine_config = EngineConfig.from_dict(config)
    grid = self._resolve_grid(config)

    rng = np.random.default_rng(seed)
    movement_rngs = build_rng(seed, engine_config.n_species, engine_config.movement_seed_fixed)
    mortality_rngs = build_rng(
        seed + 1, engine_config.n_species, engine_config.mortality_seed_fixed
    )
    outputs = simulate(
        engine_config,
        grid,
        rng,
        movement_rngs=movement_rngs,
        mortality_rngs=mortality_rngs,
        output_dir=None,
    )
    return OsmoseResults.from_outputs(outputs, engine_config, grid)
```

Add the import at the top of the file if not already present:

```python
from osmose.results import OsmoseResults
```

---

- [ ] **Step 3: Audit `output_dir=None` call sites in `simulate()`**

Run:

```bash
grep -n "output_dir" osmose/engine/simulate.py | head -20
```

Expected hits include line 1023 (parameter definition) and line 1357 (guarded write). For any other line that references `output_dir`, confirm it's either:
- A pass-through (accepts None).
- Guarded by `if output_dir is not None:`.

If any unguarded write site appears, add a guard. The spec's risk table says this audit is Task 3 territory.

---

- [ ] **Step 4: Create `tests/test_python_engine_in_memory.py`**

```python
"""Tests for PythonEngine.run_in_memory().

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults

EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _short_config() -> dict[str, str]:
    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    return raw


def test_seed_determinism():
    """Same seed twice produces equal biomass."""
    cfg = _short_config()
    r1 = PythonEngine().run_in_memory(cfg, seed=42)
    r2 = PythonEngine().run_in_memory(cfg, seed=42)
    pd.testing.assert_frame_equal(
        r1.biomass().reset_index(drop=True),
        r2.biomass().reset_index(drop=True),
    )


def test_disk_vs_memory_same_biomass():
    """run() + OsmoseResults(dir) and run_in_memory() produce equal biomass
    within rtol=1e-12 (same RNG stream, same engine)."""
    cfg = _short_config()
    with tempfile.TemporaryDirectory() as d:
        PythonEngine().run(cfg, Path(d), seed=42)
        disk = OsmoseResults(Path(d)).biomass()
    memory = PythonEngine().run_in_memory(cfg, seed=42).biomass()
    # Reset indices to compare row-for-row
    disk_sorted = disk.reset_index(drop=True)
    memory_sorted = memory.reset_index(drop=True)
    pd.testing.assert_frame_equal(disk_sorted, memory_sorted, rtol=1e-12)


def test_missing_grid_file_raises_FileNotFoundError():
    """Same error contract as run() — config with a non-existent grid
    file raises FileNotFoundError from _resolve_grid."""
    cfg = _short_config()
    cfg["grid.netcdf.file"] = "no_such_file_exists.nc"
    with pytest.raises(FileNotFoundError, match="not found in search paths"):
        PythonEngine().run_in_memory(cfg, seed=0)


def test_no_disk_writes(tmp_path, monkeypatch):
    """run_in_memory must not leak CSV/properties/restart artefacts into cwd.

    Numba / Python __pycache__ lives with the package install, not in cwd,
    so it doesn't count here. We're checking that the engine's own output
    pipeline doesn't write to the current directory when output_dir is None.
    """
    cfg = _short_config()
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    PythonEngine().run_in_memory(cfg, seed=42)
    after = set(tmp_path.iterdir())
    new_entries = after - before
    # Filter out any __pycache__ or .pytest_cache that could appear due to
    # import-time side effects from OTHER packages (rare).
    suspicious = [p for p in new_entries if not p.name.startswith(".") and "__pycache__" not in p.name]
    assert not suspicious, (
        f"run_in_memory leaked files into cwd: {[str(p) for p in suspicious]}"
    )
```

---

- [ ] **Step 5: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_python_engine_in_memory.py -v 2>&1 | tail -10
```

Expected: 4 passed. If `test_disk_vs_memory_same_biomass` fails with a row-ordering issue, fix by sorting both DataFrames the same way before `assert_frame_equal` or by using `.compare()`.

Full suite:

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: **2522 passed** (2510 + 8 from Task 2 + 4 from Task 3).

---

- [ ] **Step 6: Ruff**

```bash
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: `All checks passed!`.

---

- [ ] **Step 7: Commit**

Write to `/tmp/task3.msg`:

```
feat(engine): PythonEngine.run_in_memory() + grid-resolver extract (task 3)

Add PythonEngine.run_in_memory(config, seed) -> OsmoseResults that
runs the simulation in-process and returns an in-memory
OsmoseResults with no disk I/O. Companion to PythonEngine.run()
which keeps disk-writing behavior unchanged.

Extracted the 30-line grid-loading block from run() into
PythonEngine._resolve_grid(config) -> Grid. Both run() and
run_in_memory() now share this helper — single source of truth
for NetCDF-vs-rectangular grid resolution.

Verified osmose/engine/simulate.py already accepts output_dir=None
(parameter default at :1023, end-of-run economic write guard at
:1357). run_in_memory passes output_dir=None through to simulate().

Tests (4): tests/test_python_engine_in_memory.py covers
  - seed determinism (same seed twice produces equal biomass)
  - disk vs memory parity (run + OsmoseResults(dir) equals run_in_
    memory within rtol=1e-12)
  - missing grid file raises FileNotFoundError (same contract as run)
  - no disk writes (run_in_memory leaves cwd untouched)

Full suite 2518 -> 2522 passed.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

```bash
git add osmose/engine/__init__.py tests/test_python_engine_in_memory.py
git commit -F /tmp/task3.msg
```

---

## Task 4: Port `OsmoseCalibrationProblem._run_single` + 3 tests

**Goal:** Replace the Java subprocess path with a call to `PythonEngine.run_in_memory()` by default; keep the Java path behind `use_java_engine=True` kwarg.

**Files:**
- Modify: `osmose/calibration/problem.py`
- Create: `tests/test_calibration_problem_python_engine.py` (3 tests)

---

- [ ] **Step 1: Update `OsmoseCalibrationProblem.__init__` signature — additive change only**

Open `osmose/calibration/problem.py`. The current `__init__` at `:76-119` has this signature (do NOT reorder these params — that would silently corrupt existing positional callers):

```python
def __init__(
    self,
    free_params: list[FreeParameter],
    objective_fns: list[Callable],
    base_config_path: Path,
    jar_path: Path,
    work_dir: Path,
    java_cmd: str = "java",
    n_parallel: int = 1,
    enable_cache: bool = False,
    cache_dir: Path | None = None,
    registry: "ParameterRegistry | None" = None,
    subprocess_timeout: int = 3600,
    cleanup_after_eval: bool = False,
):
```

**Make the change as follows.** Move `work_dir` from position 5 to position 4 (so it stays required positional — since `jar_path` is becoming optional, it cannot sit in front of a required `work_dir`). Move every Java-specific kwarg including `jar_path` behind a `*` keyword-only marker. Add `use_java_engine` as a new keyword-only arg at the end.

This is a **breaking positional-argument change** with the clearest semantics — callers passing all kwargs continue to work; callers that passed `jar_path` positionally (position 4) must now pass it as `jar_path=...` and add `use_java_engine=True` to preserve Java-subprocess behavior. Document explicitly in the Task 5 CHANGELOG.

Implement:

```python
def __init__(
    self,
    free_params: list[FreeParameter],
    objective_fns: list[Callable],
    base_config_path: Path,
    work_dir: Path,
    *,
    jar_path: Path | None = None,
    java_cmd: str = "java",
    n_parallel: int = 1,
    enable_cache: bool = False,
    cache_dir: Path | None = None,
    registry: "ParameterRegistry | None" = None,
    subprocess_timeout: int = 3600,
    cleanup_after_eval: bool = False,
    use_java_engine: bool = False,
):
    self.free_params = free_params
    self.objective_fns = objective_fns
    self.base_config_path = base_config_path
    self.jar_path = jar_path
    self.work_dir = work_dir
    self.java_cmd = java_cmd
    self.n_parallel = max(1, n_parallel)
    self._enable_cache = enable_cache
    self._cache_dir = cache_dir or (self.work_dir / ".cache")
    self._registry = registry
    self._cache_hits = 0
    self._cache_misses = 0
    self.subprocess_timeout = int(subprocess_timeout)
    self.cleanup_after_eval = bool(cleanup_after_eval)
    self.use_java_engine = use_java_engine

    if use_java_engine and jar_path is None:
        raise ValueError("use_java_engine=True requires jar_path")

    # Pre-compute base config hash for cache keys
    self._base_config_hash = ""
    if enable_cache and base_config_path.exists():
        self._base_config_hash = hashlib.sha256(base_config_path.read_bytes()).hexdigest()[:16]

    xl = np.array([fp.lower_bound for fp in free_params])
    xu = np.array([fp.upper_bound for fp in free_params])

    super().__init__(
        n_var=len(free_params),
        n_obj=len(objective_fns),
        n_constr=0,
        xl=xl,
        xu=xu,
    )
```

All existing attribute assignments preserved (`self.free_params`, `self.objective_fns`, `self.base_config_path`, `self.jar_path`, `self.work_dir`, `self.java_cmd`, `self.n_parallel`, `self._enable_cache`, `self._cache_dir`, `self._registry`, `self._cache_hits`, `self._cache_misses`, `self.subprocess_timeout`, `self.cleanup_after_eval`, `self._base_config_hash`). One new attribute: `self.use_java_engine`.

---

- [ ] **Step 2: Split `_run_single` into Python-engine + Java-subprocess paths**

Replace `_run_single`'s body with:

```python
def _run_single(self, overrides: dict[str, str], run_id: int) -> list[float]:
    """Run OSMOSE with overrides and return objective values.

    Uses PythonEngine.run_in_memory() when use_java_engine=False (default)
    or Java subprocess when use_java_engine=True.
    """
    # Validate override keys/values (unchanged)
    for key, value in overrides.items():
        if not _OSMOSE_KEY_PATTERN.match(key):
            raise ValueError(f"Invalid override key: {key!r}")
        val_str = str(value)
        if not _OSMOSE_VALUE_PATTERN.match(val_str):
            raise ValueError(
                f"Invalid override value for {key!r}: {val_str!r} — "
                "only alphanumeric, '.', '+', '-', 'e', 'E', '/' allowed"
            )

    # Schema validation (unchanged)
    self._validate_overrides(overrides)

    # Cache check (unchanged)
    if self._enable_cache:
        key = self._cache_key(overrides)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            self._cache_hits += 1
            cached = json.loads(cache_file.read_text())
            return cached["objectives"]

    # Dispatch
    if self.use_java_engine:
        results = self._run_java_subprocess(overrides, run_id)
    else:
        results = self._run_python_engine(overrides, run_id)

    if results is None:
        return [float("inf")] * self.n_obj

    obj_values = [float(fn(results)) for fn in self.objective_fns]

    # Cache write (unchanged)
    if self._enable_cache:
        key = self._cache_key(overrides)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / f"{key}.json"
        fd, tmp_file = tempfile.mkstemp(dir=str(self._cache_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"objectives": obj_values}, f)
            os.replace(tmp_file, str(cache_file))
        except OSError:
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
        self._cache_misses += 1

    # Cleanup Java subprocess artefacts AFTER objectives are computed
    # (only runs for the Java path; the Python path doesn't create
    # run_dir). Mirrors the pre-port placement — cleanup was always
    # the last step before returning.
    if self.use_java_engine and self.cleanup_after_eval:
        self.cleanup_run(run_id)

    return obj_values
```

---

- [ ] **Step 3: Add `_run_python_engine` helper**

Below `_run_single`. Note: `_expected_errors` is a module-level tuple already defined in `osmose/calibration/problem.py` (line ~30, used by the existing Java subprocess path). The new helper reuses it for consistent error handling across both engines.

```python
def _run_python_engine(
    self, overrides: dict[str, str], run_id: int
) -> "OsmoseResults | None":
    """Run the Python engine in-process; return OsmoseResults or None on failure."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    try:
        # Load base config + apply overrides (as string keys/values)
        base_cfg = OsmoseConfigReader().read(self.base_config_path)
        cfg = dict(base_cfg)
        cfg.update(overrides)
        return PythonEngine().run_in_memory(cfg, seed=run_id)
    except _expected_errors as exc:
        _log.warning(
            "Python-engine run %d failed (%s: %s)",
            run_id,
            type(exc).__name__,
            exc,
        )
        return None
```

---

- [ ] **Step 4: Move existing Java-subprocess body into `_run_java_subprocess` helper**

Move the remaining subprocess block (the existing `_run_single` body from `cmd = [self.java_cmd, ...]` through `return obj_values`) into:

```python
def _run_java_subprocess(
    self, overrides: dict[str, str], run_id: int
) -> "OsmoseResults | None":
    """Run Java subprocess; return OsmoseResults or None on failure.

    Retained as opt-in fallback for cross-engine validation
    (use_java_engine=True constructor flag). Behavior unchanged from
    pre-Phase v0.10.0 _run_single body.
    """
    if self.jar_path is None:
        raise RuntimeError("_run_java_subprocess called but jar_path is None")

    # Create isolated output directory
    run_dir = self.work_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "output"

    cmd = [self.java_cmd, "-jar", str(self.jar_path), str(self.base_config_path)]
    cmd.append(f"-Poutput.dir.path={output_dir}")
    for key, value in overrides.items():
        cmd.append(f"-P{key}={value}")

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=self.subprocess_timeout)
    except _expected_errors as exc:
        _log.warning("Java subprocess run %d raised: %s", run_id, exc)
        return None

    if result.returncode != 0:
        raw_stderr = result.stderr
        stderr_bytes = raw_stderr if isinstance(raw_stderr, (bytes, bytearray)) else b""
        stderr_file = run_dir / "stderr.txt"
        try:
            stderr_file.write_bytes(bytes(stderr_bytes))
        except OSError:
            pass
        stderr_msg = bytes(stderr_bytes).decode(errors="replace")[:500]
        _log.warning(
            "OSMOSE run %d failed (exit %d); full stderr at %s; head: %s",
            run_id, result.returncode, stderr_file, stderr_msg,
        )
        return None

    from osmose.results import OsmoseResults
    return OsmoseResults(output_dir, strict=False)

    # NOTE: cleanup_run(run_id) is NOT called here — _run_single calls
    # it after objective functions have consumed the results. Moving
    # cleanup into this helper would delete run_dir BEFORE objectives
    # are computed, breaking any objective that re-reads a CSV not yet
    # cached by OsmoseResults.
```

---

- [ ] **Step 5: Update `_cache_key` to handle None jar_path**

Find `_cache_key` around `:271`. Modify to substitute `osmose.__version__` when `jar_path` is None:

```python
def _cache_key(self, overrides: dict[str, str]) -> str:
    """Deterministic hash of overrides + engine identity + base config hash."""
    parts = sorted(overrides.items())
    if self.jar_path is not None:
        try:
            engine_id = str(self.jar_path.stat().st_mtime)
        except OSError:
            engine_id = "missing"
    else:
        from osmose import __version__
        engine_id = f"python-{__version__}"
    raw = f"{parts}|{engine_id}|{self._base_config_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

---

- [ ] **Step 6: Create `tests/test_calibration_problem_python_engine.py`**

```python
"""Tests for OsmoseCalibrationProblem Python-engine path.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
"""
from __future__ import annotations

import os
import subprocess as _subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from osmose.calibration import FreeParameter, Transform  # re-exported from .problem
from osmose.calibration.problem import OsmoseCalibrationProblem
from osmose.schema import build_registry

EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _simple_objective(results) -> float:
    """Minimal objective: total biomass at the last step."""
    df = results.biomass()
    last_row = df.iloc[-1]
    numeric = [v for v in last_row.values if isinstance(v, (int, float, np.floating))]
    return float(sum(numeric))


def _make_problem(tmp_path: Path, *, use_java_engine: bool = False, jar_path: Path | None = None):
    return OsmoseCalibrationProblem(
        free_params=[
            FreeParameter(
                key="mortality.fishing.rate.sp0",
                lower_bound=0.1,
                upper_bound=0.5,
                transform=Transform.LINEAR,
            ),
        ],
        base_config_path=EXAMPLE_CONFIG,
        objective_fns=[_simple_objective],
        registry=build_registry(),
        work_dir=tmp_path,
        use_java_engine=use_java_engine,
        jar_path=jar_path,
        n_parallel=1,
        enable_cache=False,
    )


def test_python_engine_default(tmp_path, monkeypatch):
    """Default OsmoseCalibrationProblem (no use_java_engine) evaluates
    via PythonEngine and does NOT invoke subprocess.run."""
    def _raise_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called in Python-engine mode")

    monkeypatch.setattr(_subprocess, "run", _raise_subprocess)
    problem = _make_problem(tmp_path, use_java_engine=False)
    # Run a single candidate
    x = np.array([[0.3]])  # pop_size=1, n_var=1
    out = {}
    problem._evaluate(x, out)
    assert "F" in out
    assert not np.isinf(out["F"][0, 0]), f"Candidate failed: {out['F']}"


def test_java_engine_opt_in(tmp_path, monkeypatch):
    """use_java_engine=True routes through _run_java_subprocess with the
    provided jar_path."""
    fake_jar = tmp_path / "fake.jar"
    fake_jar.write_bytes(b"")
    captured_argv = []

    def _fake_subprocess(cmd, *args, **kwargs):
        captured_argv.append(cmd)
        # Simulate a nonzero return code so OsmoseResults isn't constructed
        # (the fake jar doesn't actually run OSMOSE).
        result = mock.MagicMock()
        result.returncode = 1
        result.stderr = b"fake jar"
        return result

    monkeypatch.setattr(_subprocess, "run", _fake_subprocess)
    problem = _make_problem(tmp_path, use_java_engine=True, jar_path=fake_jar)
    x = np.array([[0.3]])
    out = {}
    problem._evaluate(x, out)
    # The fake subprocess returns nonzero, so objective is inf; but the
    # argv should contain the fake jar path.
    assert captured_argv, "subprocess.run not called"
    argv = captured_argv[0]
    assert str(fake_jar) in argv


@pytest.mark.skipif(
    not os.environ.get("OSMOSE_JAR"),
    reason="OSMOSE_JAR env var not set; cross-engine test skipped",
)
def test_objective_values_match_between_engines(tmp_path):
    """Run the same 3 candidates through both engines; assert objective
    values match within 1 OoM (the project's Python/Java parity tolerance).
    """
    jar_path = Path(os.environ["OSMOSE_JAR"])
    py_problem = _make_problem(tmp_path / "py", use_java_engine=False)
    java_problem = _make_problem(tmp_path / "java", use_java_engine=True, jar_path=jar_path)

    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 0.5, size=(3, 1))
    py_out = {}
    java_out = {}
    py_problem._evaluate(X, py_out)
    java_problem._evaluate(X, java_out)

    for i in range(3):
        py_val = py_out["F"][i, 0]
        java_val = java_out["F"][i, 0]
        # Within 1 OoM per documented parity
        assert 0.1 <= py_val / java_val <= 10.0, (
            f"Candidate {i}: Python objective {py_val:.4g} vs Java {java_val:.4g} "
            f"differs by >1 OoM"
        )
```

---

- [ ] **Step 7: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_calibration_problem_python_engine.py -v 2>&1 | tail -10
```

Expected: 3 passed (one skipped if `OSMOSE_JAR` not set — that's fine).

Full suite:

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: **2525 passed** (2510 + 15 new tests).

---

- [ ] **Step 8: Ruff**

```bash
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: `All checks passed!`.

---

- [ ] **Step 9: Commit**

Write to `/tmp/task4.msg`:

```
feat(calibration): port NSGA-II to PythonEngine by default (task 4)

OsmoseCalibrationProblem._run_single now routes through
PythonEngine.run_in_memory() by default. The Java subprocess path
is preserved as an opt-in fallback via use_java_engine=True +
jar_path=... kwargs. Breaking change: existing callers that passed
jar_path=p expecting Java subprocess behavior must now also pass
use_java_engine=True.

Implementation:
- __init__ signature change: jar_path goes from required to optional
  kwarg; new use_java_engine: bool = False flag gates the Java path.
- _run_single body: validation + cache check (unchanged) -> dispatch
  to _run_python_engine or _run_java_subprocess -> objective
  computation + cache write (unchanged).
- _run_python_engine: load base config + apply overrides, call
  PythonEngine.run_in_memory(cfg, seed=run_id), return OsmoseResults.
- _run_java_subprocess: prior _run_single body moved verbatim;
  behavior unchanged when use_java_engine=True.
- _cache_key: substitutes "python-{__version__}" for jar_mtime when
  jar_path is None. Cache keys differ between engines by design.

Tests (3): tests/test_calibration_problem_python_engine.py covers
  - default PythonEngine path: subprocess.run is NEVER called
  - use_java_engine=True: subprocess.run invoked with jar_path in argv
  - cross-engine objective parity within 1 OoM (OSMOSE_JAR-gated)

Full suite 2522 -> 2525 passed.

Breaking change documented in the Task 5 CHANGELOG entry.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

```bash
git add osmose/calibration/problem.py tests/test_calibration_problem_python_engine.py
git commit -F /tmp/task4.msg
```

---

## Task 5: Benchmark script + CHANGELOG entry

**Goal:** Ship the benchmark script that gates the v0.10.0 release, and record the ≥4× speedup target + breaking-change note in `CHANGELOG.md`.

**Files:**
- Create: `scripts/benchmark_calibration.py`
- Modify: `CHANGELOG.md`

---

- [ ] **Step 1: Create `scripts/benchmark_calibration.py`**

```python
#!/usr/bin/env python3
"""Benchmark NSGA-II calibration throughput: Python vs Java engine.

Runs a small NSGA-II problem on the Baltic example using both engines
and reports wall-clock ratio. Target: >=4x speedup on a 4-thread host.

Usage:
    .venv/bin/python scripts/benchmark_calibration.py [--java JAR_PATH]

If --java is omitted, only the Python-engine run is measured.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
BALTIC_CONFIG = PROJECT_DIR / "data" / "baltic" / "baltic_all-parameters.csv"


def _run_nsga2(use_java_engine: bool, jar_path: Path | None, n_gen: int, pop_size: int) -> float:
    """Run a small NSGA-II problem; return wall-clock seconds."""
    from osmose.calibration import FreeParameter, Transform  # re-exported from .problem
    from osmose.calibration.problem import OsmoseCalibrationProblem
    from osmose.schema import build_registry
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    def _objective(results) -> float:
        return float(results.biomass()["cod"].iloc[-1])

    with tempfile.TemporaryDirectory() as d:
        problem = OsmoseCalibrationProblem(
            free_params=[
                FreeParameter("mortality.fishing.rate.sp0", 0.1, 0.5, Transform.LINEAR),
            ],
            base_config_path=BALTIC_CONFIG,
            objective_fns=[_objective],
            registry=build_registry(),
            work_dir=Path(d),
            use_java_engine=use_java_engine,
            jar_path=jar_path,
            n_parallel=4,
            enable_cache=False,
        )

        algorithm = NSGA2(pop_size=pop_size)
        start = time.perf_counter()
        minimize(problem, algorithm, ("n_gen", n_gen), verbose=False, seed=42)
        return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--java", type=Path, help="Path to OSMOSE jar")
    parser.add_argument("--n-gen", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=20)
    args = parser.parse_args()

    if not BALTIC_CONFIG.exists():
        print(f"ERROR: Baltic config not found at {BALTIC_CONFIG}", file=sys.stderr)
        return 1

    print(f"Running {args.n_gen}-gen x {args.pop_size}-pop NSGA-II on Baltic")
    print()
    print("Python engine...")
    t_python = _run_nsga2(
        use_java_engine=False, jar_path=None,
        n_gen=args.n_gen, pop_size=args.pop_size,
    )
    print(f"  Wall-clock: {t_python:.2f}s")

    if args.java is None:
        print()
        print("Skipping Java comparison (--java not supplied).")
        return 0

    print()
    print(f"Java engine ({args.java})...")
    t_java = _run_nsga2(
        use_java_engine=True, jar_path=args.java,
        n_gen=args.n_gen, pop_size=args.pop_size,
    )
    print(f"  Wall-clock: {t_java:.2f}s")

    print()
    speedup = t_java / t_python if t_python > 0 else float("inf")
    print(f"Speedup (Java/Python): {speedup:.2f}x")
    if speedup < 4.0:
        print(f"WARNING: speedup {speedup:.2f}x is below the 4x v0.10.0 release gate")
        return 2
    print(f"OK: speedup >= 4x (v0.10.0 release gate met)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

- [ ] **Step 2: Run the benchmark smoke test (Python only — OSMOSE_JAR optional)**

```bash
.venv/bin/python scripts/benchmark_calibration.py --n-gen 2 --pop-size 5 2>&1 | tail -10
```

Expected: Reports a Python-engine wall-clock time; does not crash; `Skipping Java comparison` if no `--java` supplied.

---

- [ ] **Step 3: Run the full benchmark (v0.10.0 release gate)**

Only if `OSMOSE_JAR` is available. Record the output in the CHANGELOG entry below.

```bash
.venv/bin/python scripts/benchmark_calibration.py --java "$OSMOSE_JAR" 2>&1 | tail -15
```

Expected: Reports both Python and Java wall-clock times, plus speedup ratio. If < 4×, the v0.10.0 release is held per the spec's release gating — investigate the gap before tagging.

**If `OSMOSE_JAR` is not available**, skip this step but record in the CHANGELOG that the benchmark was not run. Release may proceed on the strength of the unit/integration tests alone, but the `>= 4x` claim must not be made.

---

- [ ] **Step 4: Prepend CHANGELOG entry**

Edit `CHANGELOG.md`. Prepend under `[Unreleased] → Changed` (create the section if absent):

```markdown
## [Unreleased]

### Changed

- **calibration:** port NSGA-II to PythonEngine in-memory by default (spec: `docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md`). `OsmoseCalibrationProblem._run_single` now evaluates candidates in-process via the Python engine instead of shelling out to a Java subprocess per candidate. Target wall-clock speedup ≥ 4× on a 10-generation × 20-candidate Baltic NSGA-II run (see `scripts/benchmark_calibration.py`). Java subprocess path preserved as an opt-in fallback via `OsmoseCalibrationProblem(use_java_engine=True, jar_path=...)`.
- **BREAKING:** `OsmoseCalibrationProblem.__init__` signature — `jar_path` is now an optional keyword argument (was required). Existing callers that constructed `OsmoseCalibrationProblem(..., jar_path=p)` expecting the Java subprocess path must now also pass `use_java_engine=True`. Without that flag, `jar_path` is ignored and the Python engine runs. Python/Java parity is complete (14/14 EEC, 8/8 BoB within 1 OoM), so accidental migration produces equivalent objective values in most configs.
- **BREAKING (runtime behavior):** NSGA-II Pareto fronts may differ slightly after this release, because the Python engine uses a different RNG stream than the Java engine. Numerical equivalence is within 1 OoM (documented parity). If bit-exact reproducibility with Java is required, set `use_java_engine=True`.

### Added

- `OsmoseResults.from_outputs(outputs, engine_config, grid)` classmethod — constructs an in-memory results object from a list of `StepOutput` returned by `simulate()`. No disk I/O.
- `PythonEngine.run_in_memory(config, seed)` — runs the Python engine and returns `OsmoseResults` directly, skipping the disk round-trip.
- `scripts/benchmark_calibration.py` — benchmarks NSGA-II calibration wall-clock for Python vs Java engines. Release gate for v0.10.x.

### Migration notes

- Long-lived calibration `work_dir` directories from v0.9.x will have cached objective values keyed by the Java `jar_mtime`. v0.10.0 uses `python-{__version__}` for the Python path, so old cache files never match new runs (no corruption, just cache misses). Optionally `rm -rf <work_dir>/cache` before running v0.10.0 NSGA-II to reclaim disk.
```

If the benchmark was run in Step 3, append one line with the observed speedup:

```markdown
- **Benchmark result (2026-04-20):** Baltic 10-gen × 20-cand NSGA-II, Python vs Java: *<observed_speedup>×* wall-clock speedup on *<host_description>*.
```

---

- [ ] **Step 5: Full-suite + ruff check**

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: **2525 passed**, ruff clean.

---

- [ ] **Step 6: Commit**

Write the commit message to `/tmp/task5.msg` via the Write tool (same pattern as Tasks 1-4):

```
docs: benchmark script + CHANGELOG for v0.10.0 calibration change (task 5)

Ship scripts/benchmark_calibration.py as the v0.10.0 release gate and
stage the [Unreleased] CHANGELOG entry with breaking-change notes and
migration guidance.

Benchmark: 10-gen x 20-candidate NSGA-II on Baltic. Targets >= 4x
wall-clock speedup Python-engine vs Java-subprocess. --java JAR_PATH
flag required for the comparison; Python-only mode skips the ratio.

CHANGELOG [Unreleased] Changed section documents:
- OsmoseCalibrationProblem default engine switch
- Breaking kwargs change (jar_path optional; use_java_engine required
  to preserve Java-subprocess behavior)
- Runtime-behavior note: NSGA-II Pareto fronts may differ by engine

Added section lists the three new public surfaces: OsmoseResults.
from_outputs, PythonEngine.run_in_memory, scripts/benchmark_
calibration.py.

Migration notes: v0.9.x cache files are stale (different _cache_key
scheme); user-decided whether to rm the cache dir.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Then:

```bash
git add scripts/benchmark_calibration.py CHANGELOG.md
git commit -F /tmp/task5.msg
```

---

## Self-review checklist

Before executing, the implementer verifies:

- [ ] Every spec section maps to at least one task — `_build_*_dataframes` refactor (Task 1), `OsmoseResults.from_outputs` factory (Task 2), `PythonEngine.run_in_memory` (Task 3), `OsmoseCalibrationProblem` port + cache key change (Task 4), benchmark + CHANGELOG (Task 5). ✓
- [ ] No placeholder text: no TBD, TODO, "fill in", "similar to task N". Every code block is pasteable. ✓
- [ ] Type / symbol consistency: `_build_species_dataframes`, `_build_distribution_dataframes`, `_build_mortality_dataframes`, `_build_yield_dataframes`, `_build_bioen_dataframes`, `_build_diet_dataframe`, `_build_dataframes_from_outputs`, `OsmoseResults.from_outputs`, `PythonEngine.run_in_memory`, `PythonEngine._resolve_grid`, `_run_python_engine`, `_run_java_subprocess`, `use_java_engine` — names used consistently across every task that references them. ✓
- [ ] Test runner: `.venv/bin/python -m pytest` throughout. No bare `python`. ✓
- [ ] Commit granularity: 5 commits as specified by the spec's Rollback section. Each commit is independently testable and revertible per the spec's "tests travel with the commit whose behavior they guard" rule. ✓
- [ ] Baseline check: pre-flight asserts 2510 passed / 15 skipped / 41 deselected; Step 11 / Step 8 / Step 5 expectations track +8, +12, +15 incrementally. Final target 2525. ✓
- [ ] Line-anchor notes: anchors are cited (`:23`, `:104`, `:369`, etc.) but the plan explicitly tells the implementer to grep rather than trust them. ✓
- [ ] Known iteration point: Task 2 Step 4 ("cache schema alignment between `_build_species_dataframes` wide-form and `_read_species_output`'s expected shape") is flagged as likely to require adjustment. Budget noted. ✓

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-calibration-python-engine-plan.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (5 tasks total), two-stage review (spec compliance + code quality) between tasks. Fast iteration in this session.

**2. Inline Execution** — I execute all tasks in this session using `superpowers:executing-plans` with checkpoints after Task 1 and Task 4 (the two largest).

Which approach?
