# Calibration throughput — port NSGA-II to PythonEngine in-memory (design)

**Status:** design / ready for implementation plan
**Target:** `osmose/engine/__init__.py`, `osmose/results.py`, `osmose/calibration/problem.py`, `osmose/engine/output.py`
**Ship target:** v0.10.0 (minor bump — `OsmoseCalibrationProblem.__init__` signature changes: `jar_path` goes from required to optional and existing callers must add `use_java_engine=True` to preserve current behavior)
**Baseline:** master `9f8af75`, 2510 tests passing, v0.9.3 shipped
**Prior-art reference:** `osmose/calibration/preflight.py:722-726` already uses PythonEngine in-process; this spec extends the same pattern to the main NSGA-II path.

## Problem statement

OSMOSE calibration has two execution paths with different performance profiles:

1. **pymoo NSGA-II optimizer** (`osmose/calibration/problem.py`) — spawns a Java JVM subprocess per candidate via `subprocess.run([java_cmd, "-jar", jar_path, ...])`. Reads outputs back from disk through `OsmoseResults(output_dir)`. Uses `ThreadPoolExecutor` for parallelism.
2. **Preflight / sensitivity** (`osmose/calibration/preflight.py`) — uses `PythonEngine().run(config, out_dir)` in-process, still reads back through `OsmoseResults(output_dir)`.

The NSGA-II path dominates wall-clock for any serious calibration: a 50-candidate × 20-generation run is 1000 JVM forks, and JVM startup alone is ~2–3s per call on a warm host. Even with parallel threads, subprocess overhead and disk I/O limit throughput far below what the Python engine is capable of.

Since Python engine parity is complete (EEC 14/14, BoB 8/8; v0.9.3 parity-roadmap STATUS-COMPLETE across all phases), nothing forces NSGA-II to shell out to Java. The Python engine can evaluate candidates in-process at a fraction of the cost.

The disk round-trip between `PythonEngine.run()` and `OsmoseResults(output_dir)` is also pure overhead in the calibration context — the engine writes CSVs, the objective function immediately reads them back. Cutting that loop adds another 15–20% per-candidate saving on top of the JVM removal.

## Goals

- Port `OsmoseCalibrationProblem._run_single` to evaluate candidates through `PythonEngine` in-process by default, not `subprocess.run([java_cmd, ...])`.
- Add an in-memory `OsmoseResults` factory so the engine can return results without writing to disk.
- Preserve the Java subprocess path as an opt-in fallback (`use_java_engine=True`) for cross-engine validation.
- Target ≥ 4× wall-clock speedup on a realistic 10-generation × 20-candidate NSGA-II run, measured on a 4-thread host against the Baltic example.

## Non-goals

- Process-based parallelism (`ProcessPoolExecutor`) for calibration — tracked as a separate follow-up phase.
- AOT Numba compilation — tracked as another follow-up.
- Surrogate warm-start / early rejection — separate feature.
- In-memory support for `spatial_biomass()` or any NetCDF output — calibration doesn't consume these today; disk-only for v1.
- Porting the Shiny UI calibration workflow — the UI path already goes through `preflight.py`, which already uses PythonEngine.

## Architecture

### Data flow change

Before:
```
[pymoo NSGA-II] → OsmoseCalibrationProblem._run_single()
                    → subprocess.run([java, -jar, jar_path, ..., -Poutput.dir.path=run_i/output])
                    → [Java writes output_dir]
                    → OsmoseResults(run_i/output, strict=False)
                    → [objective_fn(results)] → float
```

After (default):
```
[pymoo NSGA-II] → OsmoseCalibrationProblem._run_single()
                    → PythonEngine().run_in_memory(config, seed=run_id)
                    → OsmoseResults wrapping in-memory StepOutputs
                    → [objective_fn(results)] → float
```

After (opt-in for cross-engine validation):
```
[pymoo NSGA-II] → OsmoseCalibrationProblem(use_java_engine=True, jar_path=...)._run_single()
                    → [same Java subprocess path as today]
```

### New public surface

**`OsmoseResults.from_outputs(outputs, engine_config, grid, *, prefix="osm") -> OsmoseResults`** (factory classmethod):

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

    No disk I/O. Supports the same CSV-backed getter API as a disk-backed
    instance. All ~30 CSV-backed getter methods (biomass, abundance,
    yield_biomass, mortality, diet_matrix, mean_size, mean_trophic_level,
    biomass_by_age, biomass_by_size, biomass_by_tl, abundance_by_age,
    abundance_by_size, abundance_by_tl, yield_by_age, yield_by_size,
    yield_n_by_age, yield_n_by_size, diet_by_age, diet_by_size,
    mean_size_by_age, mean_tl_by_size, mean_tl_by_age, yield_abundance,
    mortality_rate, fishery_yield, fishery_yield_by_age,
    fishery_yield_by_size, bioen_ingestion, bioen_maintenance,
    bioen_net_energy, size_spectrum) work transparently via the in-memory
    cache. Getters that have no in-memory representation (spatial_biomass,
    anything returning xr.Dataset from NetCDF) raise FileNotFoundError in
    this mode — see "In-memory dispatch" below for the exact contract.
    """
```

**`PythonEngine.run_in_memory(config, seed=0) -> OsmoseResults`**:

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
```

### Changed public surface

**`OsmoseCalibrationProblem.__init__`** changes from requiring `jar_path` to making it an optional Python-engine-only kwarg, and adds a `use_java_engine` flag. Clean break — no deprecation shim.

Before (current):
```python
class OsmoseCalibrationProblem(ElementwiseProblem):
    def __init__(
        self,
        free_params: list[FreeParameter],
        base_config_path: Path,
        objective_fns: list[Callable[[OsmoseResults], float]],
        registry: ParameterRegistry,
        work_dir: Path,
        jar_path: Path,              # required keyword
        java_cmd: str = "java",
        ...
    ) -> None:
```

After:
```python
class OsmoseCalibrationProblem(ElementwiseProblem):
    def __init__(
        self,
        free_params: list[FreeParameter],
        base_config_path: Path,
        objective_fns: list[Callable[[OsmoseResults], float]],
        registry: ParameterRegistry,
        work_dir: Path,
        *,
        use_java_engine: bool = False,
        jar_path: Path | None = None,
        java_cmd: str = "java",
        n_parallel: int = 1,
        subprocess_timeout: int = 3600,
        enable_cache: bool = True,
        cleanup_after_eval: bool = True,
    ) -> None:
        ...
        if use_java_engine and jar_path is None:
            raise ValueError("use_java_engine=True requires jar_path")
        self.use_java_engine = use_java_engine
        self.jar_path = jar_path
```

**Breaking change.** Existing callers that constructed `OsmoseCalibrationProblem(..., jar_path=p)` expecting the Java subprocess path must now additionally pass `use_java_engine=True`. Without the flag, `jar_path` is ignored and the Python engine runs.

The rationale for NOT adding a soft-deprecation shim:

- The Python engine has full parity (14/14 EEC, 8/8 BoB within 1 OoM). Running existing calibrations through it produces equivalent objective values (within tolerance) plus ≥4× speedup. Accidental migration is the beneficial migration.
- A `DeprecationWarning` shim adds test surface (covering both the warning and the auto-flip logic) for no real user benefit. Anyone whose calibration genuinely REQUIRES the Java subprocess (e.g., comparing against upstream OSMOSE exactly) will set the flag on purpose — they're not the accidental case.
- No external callers exist outside this repo (the calibration module is not re-exported as a stable API; it's internal to the OSMOSE Python project).

If a reader's existing notebook breaks, the fix is a one-keyword addition (`use_java_engine=True`). The CHANGELOG entry documents this explicitly.

### Internal refactors

**`osmose/engine/output.py`** currently holds `write_outputs(outputs, output_dir, config, grid)` (at `:23`) plus seven per-family write helpers: `_write_species_csv` (:104), `_write_distribution_csvs` (:120), `_write_mortality_csvs` (:165), `_write_yield_csv` (:273), `_write_bioen_csvs` (:303), and the two NetCDF writers `write_outputs_netcdf` (:346), `write_outputs_netcdf_spatial` (:492). Each CSV helper iterates over `outputs`, builds wide-form per-species `pd.DataFrame`s, and writes them to `{prefix}_{output_type}_Simu0.csv` style files.

The refactor splits each `_write_*_csv` helper into two steps:

1. `_build_*_dataframes(outputs, config) -> dict[str, pd.DataFrame]` — returns a mapping from `output_type` (e.g., `"biomass"`, `"abundanceByAge"`, `"mortalityRate"`) to the wide-form DataFrame that would have been written. For family-per-species writers, the mapping key is `f"{output_type}_{species_name}"` to match the filename shape; for cross-species outputs (mortality, yield), the mapping has a single key per output_type.
2. Each `_write_*_csv` helper becomes: `dfs = _build_*_dataframes(outputs, config); for key, df in dfs.items(): df.to_csv(path_for(key), ...)`.

`write_outputs()` orchestrates all seven. Behavior preserved — the existing CSV output tests are the regression net.

`_build_dataframes_from_outputs(outputs, config, grid)` in `results.py` calls every `_build_*_dataframes` helper and collects the results into a single `dict[str, pd.DataFrame]` mapping (see the cache schema below). The two NetCDF writers stay disk-only in v1 — in-memory equivalents are out of scope per the non-goals.

This is the single most important invariant: disk output and in-memory output **must** produce the same DataFrame for the same simulation. The only way to guarantee that over time is to share the build code.

**`osmose/engine/__init__.py`** `PythonEngine.run()` has a 30-line grid-loading block (`:27-56`). Extract as `PythonEngine._resolve_grid(config) -> Grid` (private helper) and call from both `run()` and `run_in_memory()`.

### In-memory dispatch in `OsmoseResults`

The real read primitives in `OsmoseResults` are `_read_species_output(output_type, species)` (at `:409`) and `_read_2d_output(output_type, species)` (at `:369`) — NOT `read_csv`. Every CSV-backed getter (`biomass()`, `abundance()`, `biomass_by_age()`, etc.) routes through exactly one of these two helpers. `_read_species_output` already caches results in `self._csv_cache[output_type]`; `_read_2d_output` currently does not cache.

The dispatch design hooks at both primitives, not at `read_csv`.

**`from_outputs()` populates `self._csv_cache` directly.** The cache is a `dict[str, pd.DataFrame]` keyed by `output_type`:

```
Cache schema:
  self._csv_cache: dict[str, pd.DataFrame]
  Keys: output_type strings used by the build helpers. The illustrative
        subset includes "biomass", "abundance", "abundanceByAge",
        "abundanceBySize", "biomassByAge", "biomassBySize", "biomassByTL",
        "yield", "yieldByAge", "yieldBySize", "yieldN", "mortalityRate",
        "dietMatrix", "meanSize", "meanTL", "fisheryYield".
  Values: wide-form DataFrames with a 'species' column — exactly the shape
          _read_species_output() caches today.
```

The **complete** set of valid `output_type` keys is the union of keys produced by the five `_build_*_dataframes` helpers the plan introduces (one per existing `_write_*_csv` helper: `_build_species_dataframes`, `_build_distribution_dataframes`, `_build_mortality_dataframes`, `_build_yield_dataframes`, `_build_bioen_dataframes`). The authoritative enumeration is a plan-time task: Task 1 of the implementation plan extracts each helper, and the exact key set is a byproduct of that extraction. The parity test (`test_from_outputs_populates_all_written_keys`) asserts `set(build_dataframes_from_outputs(outputs, config, grid).keys()) == set(keys_that_write_outputs_would_write(outputs, config, grid))` — same-simulation, the two code paths agree on which output types exist. This test is the single source of truth for the cache key set; any future output family added to `write_outputs` automatically extends the cache and the parity test enforces it.

**Dispatch logic in the two primitives:**

```python
def _read_species_output(self, output_type: str, species: str | None) -> pd.DataFrame:
    # Already has: if cache_key not in self._csv_cache: load from disk.
    # After refactor: if self._in_memory and output_type not in self._csv_cache,
    # raise FileNotFoundError — the cache should have been populated by
    # from_outputs() for every supported output_type.
    if self._in_memory and output_type not in self._csv_cache:
        raise FileNotFoundError(
            f"In-memory OsmoseResults has no '{output_type}' output. "
            f"Available: {sorted(self._csv_cache.keys())}"
        )
    # ... existing cache-populate + species-filter logic unchanged.

def _read_2d_output(self, output_type: str, species: str | None = None) -> pd.DataFrame:
    # Currently uncached. Add short-circuit: if in-memory and cache hit,
    # melt the wide-form DataFrame on the fly.
    if self._in_memory:
        if output_type not in self._csv_cache:
            raise FileNotFoundError(
                f"In-memory OsmoseResults has no '{output_type}' output. "
                f"Available: {sorted(self._csv_cache.keys())}"
            )
        return _melt_wide_to_long(self._csv_cache[output_type], species)
    # ... existing disk-read path unchanged.
```

`_melt_wide_to_long` extracts the existing melt logic (lines `:388-397`) into a helper that takes a wide-form DataFrame and returns the long-form shape the getters expect.

**`spatial_biomass()` and `read_netcdf()`** bypass `_read_*_output` entirely — they go through `read_netcdf`. In-memory mode is NOT supported for these; `read_netcdf` gets a parallel dispatch:

```python
def read_netcdf(self, filename: str) -> xr.Dataset:
    if self._in_memory:
        raise FileNotFoundError(
            f"In-memory OsmoseResults does not support NetCDF outputs "
            f"(requested: {filename}). Use the disk-backed OsmoseResults "
            f"constructor if you need spatial NetCDF outputs."
        )
    # ... existing disk path unchanged.
```

This gives the clean `FileNotFoundError` contract the design promises: every in-memory getter either returns a valid DataFrame or raises `FileNotFoundError` with an actionable message — never silently returns an empty DataFrame or a `KeyError`.

**`self._in_memory: bool`** is a new instance attribute. Default `False` in `__init__`; set to `True` by `from_outputs()`. Disk-backed instances see zero behavior change.

## Success criteria

1. `grep -n "def run_in_memory" osmose/engine/__init__.py` → 1 hit.
2. `grep -n "def from_outputs" osmose/results.py` → 1 hit.
3. `grep -n "subprocess.run" osmose/calibration/problem.py` → exactly 1 hit (inside the opt-in `_run_java_subprocess` helper).
4. Default `OsmoseCalibrationProblem(..., jar_path=None)` evaluates without invoking any subprocess on the happy path — verified by a test that monkeypatches `subprocess.run` to raise.
5. `OsmoseCalibrationProblem(..., use_java_engine=True, jar_path=...)` still works — verified by a test that monkeypatches `subprocess.run` and asserts the argv contains `jar_path`.
6. Objective-value parity between engines on a small fixture: 3 candidates through Python engine match the same 3 candidates through Java engine within 1 OoM (the project's parity tolerance). Skipped in CI unless `OSMOSE_JAR` env var is set.
7. `pd.testing.assert_frame_equal(results_disk.biomass(), results_memory.biomass())` passes for every CSV-backed getter on a 2-year fixture run (same engine, same seed, two I/O paths).
8. Full suite: **≥ baseline passed + new tests, zero new failures**. Baseline 2510 → target ≥ 2522.
9. Ruff clean.
10. Benchmark script `scripts/benchmark_calibration.py` reports **≥ 4× wall-clock speedup** on a 10-generation × 20-candidate Baltic NSGA-II run, Python vs Java default.

## Testing strategy

Three new test files:

- **`tests/test_results_in_memory.py`** (~5 tests)
  - `test_biomass_getter_matches_disk_path`, `test_abundance_getter_matches_disk_path`, `test_yield_biomass_getter_matches_disk_path`, `test_mortality_getter_matches_disk_path`, `test_diet_matrix_getter_matches_disk_path` — all run the same config twice (disk + memory) and assert `pd.testing.assert_frame_equal`.
  - `test_spatial_biomass_raises_FileNotFoundError_in_memory_mode` — the one getter we explicitly don't support.
  - `test_from_outputs_idempotent` — calling the same getter twice returns cached result.
- **`tests/test_python_engine_in_memory.py`** (~4 tests)
  - `test_seed_determinism` — two `run_in_memory()` calls with same seed produce equal biomass.
  - `test_disk_vs_memory_same_biomass` — `run(cfg, tmp, seed=42)` and `run_in_memory(cfg, seed=42)` produce biomass within `rtol=1e-12`.
  - `test_missing_grid_file_raises_FileNotFoundError` — same error contract as `run()`.
  - `test_no_disk_writes` — pass `tmp_path` as `cwd` (monkeypatch via `chdir`) and run `run_in_memory()`. Assert that `tmp_path` contains zero files after the call returns. Numba's `__pycache__` lives under the package install directory, not the test sandbox, so it doesn't count; Python's own `__pycache__` likewise lives with the module, not the cwd. The test is checking that `run_in_memory()` doesn't leak CSV / properties / restart-file artefacts into the caller's working directory.
- **`tests/test_calibration_problem_python_engine.py`** (~3 tests)
  - `test_python_engine_default` — construct `OsmoseCalibrationProblem` without `use_java_engine`, monkeypatch `subprocess.run` to raise; assert `_run_single` completes without hitting subprocess.
  - `test_java_engine_opt_in` — construct with `use_java_engine=True, jar_path=<fake>`; monkeypatch `subprocess.run` to return a mock successful result; assert argv contains the fake jar_path.
  - `test_objective_values_match_between_engines` — run 3 candidates from a seeded random population through both engines; assert objective values match within 1 OoM. `pytest.skipif(not os.environ.get("OSMOSE_JAR"))`.

Total new tests: ~12. Target pass count becomes ≥ 2522.

Benchmark (not a test — runs on demand via `scripts/benchmark_calibration.py`):
- 10 generations × 20 candidates on the Baltic example.
- Run once with `use_java_engine=True` (requires `OSMOSE_JAR`), once with default Python engine.
- Report wall-clock ratio.
- Target: ≥ 4×.
- **Release gating:** the implementation plan's final task (before cutting v0.10.0) runs this benchmark once and records the observed speedup in the v0.10.0 CHANGELOG entry. If the speedup is < 4×, the release is held and the gap investigated. The benchmark is not part of CI.

## Risks and mitigations

| Risk | Manifestation | Mitigation |
|---|---|---|
| `_build_*_dataframes` refactor changes CSV output | Existing output tests fail | Each `_write_*_csv` helper becomes `dfs = _build_*_dataframes(...); for key, df in dfs.items(): df.to_csv(...)`. Same values written, same filenames. The existing output-tests suite is the regression net — we run it after each of the 7 helper refactors as part of the implementation plan's commit gating. |
| `simulate()` signature doesn't tolerate `output_dir=None` | `run_in_memory` crashes | **Verified already safe.** `simulate(..., output_dir: Path \| None = None)` at `osmose/engine/simulate.py:1023` already accepts None, and the end-of-run economic CSV write at `:1357` is already guarded by `if output_dir is not None`. Implementer runs `grep -n "output_dir\." osmose/engine/simulate.py` during plan Task 1 and confirms every usage is either a guarded write or a pass-through to a helper that ALSO accepts None. If any unguarded write site turns up, fix in place with a `None` guard and add a regression test. |
| In-memory cache too large on long runs | Memory balloon in calibration | The cache holds ~30 DataFrames per simulation, each roughly `n_steps × n_species` rows (typically < 100k rows total, ~10-30 MB). Workers release candidates after objective computation, so only `n_parallel` in flight. Memory is bounded at `n_parallel × per_run_cache_bytes` — well below typical calibration host RAM. |
| Java/Python objective values differ > 1 OoM on some configs | Cross-engine parity test fails | Python engine has documented parity at 14/14 EEC + 8/8 BoB within 1 OoM. The cross-engine test uses the same configs used for parity validation. If a new config shows drift, the cross-engine test is the canary, not a blocker — document it and let users opt into `use_java_engine=True`. |
| `OsmoseCalibrationProblem._cache_key` references `jar_mtime` (None in Python path) | Cache key inconsistency or crash | When `use_java_engine=False`, substitute `osmose.__version__` for `jar_mtime` in the key. Cache keys differ between engines by design — a Python run and a Java run with identical overrides produce different cache files, which is correct because the engines are different artifacts. |
| v0.9.x cache files stale after v0.10.0 engine switch | Calibration users with long-lived `work_dir` accumulate orphaned `.json` cache files | **Orphan files are left in place.** Cache key hash changes (Java jar_mtime → Python version), so old files stop matching and the new run regenerates. Nothing is silently corrupted. CHANGELOG entry will tell users: "Optionally `rm -rf <work_dir>/cache` before running v0.10.0 NSGA-II to reclaim disk if you had long-lived work directories under v0.9.x." No automatic cleanup; calibration work dirs are user-owned. |
| NSGA-II convergence changes because Python engine RNG stream differs from Java | Different Pareto front after the port | Expected. Not a bug — it's the engine switch surfacing as a stochastic difference. Users who need reproducibility with the Java path flip `use_java_engine=True`. Documented explicitly in the CHANGELOG as a runtime-behavior note, not a regression. |

## Rollback

Each of the four files touched is revertible independently:

- Revert `problem.py` — existing calibration returns to Java subprocess. Benign.
- Revert `results.py` — `OsmoseResults.from_outputs` goes away. `PythonEngine.run_in_memory` loses its return path, but `problem.py` was reverted first, so no caller remains.
- Revert `engine/__init__.py` — `run_in_memory` goes away. Same argument as above.
- Revert `engine/output.py` refactor — `_records_to_dataframe` helpers removed, `write_outputs` returns to inline CSV writing. Existing output tests still pass because CSV shapes are preserved.

Commit order (in the implementation plan) will be: output.py refactor → results.py factory → engine in-memory method → problem.py port → tests → benchmark. Any single-commit revert is safe as long as later commits haven't landed yet.

## Out-of-scope follow-ups

- **Process-based parallelism for calibration.** `ProcessPoolExecutor` would eliminate the Python GIL on the orchestration bits, but pays fork cost plus inter-process serialization overhead. Worth measuring once the in-memory path is in production; the current ThreadPool approach may already be good enough on the happy path.
- **AOT Numba compilation.** Would eliminate the ~20s cold-start, useful for short runs and CI. Orthogonal to this spec.
- **Surrogate warm-start.** Use the GP surrogate to reject clearly-bad candidates before running a full simulation. Separate feature, works for both engines.
- **In-memory `spatial_biomass` / NetCDF outputs.** Build when a calibration objective needs them.
- **Share Numba cache across `ProcessPoolExecutor` workers.** Non-trivial — probably an environment-variable tuning task rather than code.
