# Calibration throughput — port NSGA-II to PythonEngine in-memory (design)

**Status:** design / ready for implementation plan
**Target:** `osmose/engine/__init__.py`, `osmose/results.py`, `osmose/calibration/problem.py`, `osmose/engine/output.py`
**Ship target:** v0.10.0 (minor bump — includes a small breaking change to `OsmoseProblem`)
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

- Port `OsmoseProblem._run_single` to evaluate candidates through `PythonEngine` in-process by default, not `subprocess.run([java_cmd, ...])`.
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
[pymoo NSGA-II] → OsmoseProblem._run_single()
                    → subprocess.run([java, -jar, jar_path, ..., -Poutput.dir.path=run_i/output])
                    → [Java writes output_dir]
                    → OsmoseResults(run_i/output, strict=False)
                    → [objective_fn(results)] → float
```

After (default):
```
[pymoo NSGA-II] → OsmoseProblem._run_single()
                    → PythonEngine().run_in_memory(config, seed=run_id)
                    → OsmoseResults wrapping in-memory StepOutputs
                    → [objective_fn(results)] → float
```

After (opt-in for cross-engine validation):
```
[pymoo NSGA-II] → OsmoseProblem(use_java_engine=True, jar_path=...)._run_single()
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
    instance: biomass, abundance, yield_biomass, mortality, diet_matrix,
    mean_size, mean_trophic_level, biomass_by_age, biomass_by_size,
    biomass_by_tl, and list_outputs. Getters that have no in-memory
    representation (spatial_biomass, anything returning xr.Dataset from
    NetCDF) raise FileNotFoundError in this mode.
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

**`OsmoseProblem.__init__`** gains a keyword-only flag plus a `jar_path: Path | None = None` default:

```python
class OsmoseProblem(ElementwiseProblem):
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
        # Soft-migrate old callers: jar_path supplied without the flag still
        # works for one release, with a DeprecationWarning pointing at the
        # new flag.
        if jar_path is not None and not use_java_engine:
            import warnings
            warnings.warn(
                "OsmoseProblem now defaults to the Python engine. "
                "Passing jar_path without use_java_engine=True will stop "
                "triggering the Java subprocess path in v0.11.0. Set "
                "use_java_engine=True explicitly to keep current behavior.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.use_java_engine = True
        else:
            self.use_java_engine = use_java_engine
        self.jar_path = jar_path
```

### Internal refactors

**`osmose/engine/output.py`** currently holds `write_outputs(outputs, output_dir, config, grid)` which iterates once over `outputs` and writes per-family CSVs. Split each family's CSV build into a `_records_to_dataframe(records, config)` step that returns a `pd.DataFrame`, followed by a `df.to_csv(path, ...)` write. The DataFrame-building helpers become the shared source of truth for both paths:

- `write_outputs()` calls `_records_to_dataframe()` then writes to disk (current behavior preserved).
- `_build_dataframes_from_outputs(outputs, config, grid)` in `results.py` calls the same `_records_to_dataframe()` helpers, collects into `dict[str, pd.DataFrame]` keyed by the output-type pattern, and returns for `OsmoseResults` to cache.

This is the single most important invariant: disk output and in-memory output **must** produce the same DataFrame for the same simulation. The only way to guarantee that over time is to share the build code.

**`osmose/engine/__init__.py`** `PythonEngine.run()` currently has a 30-line grid-loading block. Extract as `PythonEngine._resolve_grid(config) -> Grid` (private helper) and call from both `run()` and `run_in_memory()`.

### In-memory dispatch in `OsmoseResults`

The existing `read_csv(pattern)` method is the single chokepoint every getter flows through. Add one dispatch line at the top:

```python
def read_csv(self, pattern: str) -> dict[str, pd.DataFrame]:
    """..."""
    if self._in_memory_cache is not None:
        return self._in_memory_cache.get(pattern, {})
    # existing disk path unchanged
    ...
```

`_in_memory_cache` is `None` for disk-backed instances (the existing `__init__` path) and a populated `dict[str, dict[str, pd.DataFrame]]` for in-memory instances built via `from_outputs()`.

## Success criteria

1. `grep -n "def run_in_memory" osmose/engine/__init__.py` → 1 hit.
2. `grep -n "def from_outputs" osmose/results.py` → 1 hit.
3. `grep -n "subprocess.run" osmose/calibration/problem.py` → exactly 1 hit (inside the opt-in `_run_java_subprocess` helper).
4. Default `OsmoseProblem(..., jar_path=None)` evaluates without invoking any subprocess on the happy path — verified by a test that monkeypatches `subprocess.run` to raise.
5. `OsmoseProblem(..., use_java_engine=True, jar_path=...)` still works — verified by a test that monkeypatches `subprocess.run` and asserts the argv contains `jar_path`.
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
  - `test_no_disk_writes` — run inside a `tmp_path` sandbox; assert the sandbox is empty after `run_in_memory()` returns.
- **`tests/test_calibration_problem_python_engine.py`** (~3 tests)
  - `test_python_engine_default` — construct `OsmoseProblem` without `use_java_engine`, monkeypatch `subprocess.run` to raise; assert `_run_single` completes without hitting subprocess.
  - `test_java_engine_opt_in` — construct with `use_java_engine=True, jar_path=<fake>`; monkeypatch `subprocess.run` to return a mock successful result; assert argv contains the fake jar_path.
  - `test_objective_values_match_between_engines` — run 3 candidates from a seeded random population through both engines; assert objective values match within 1 OoM. `pytest.skipif(not os.environ.get("OSMOSE_JAR"))`.

Total new tests: ~12. Target pass count becomes ≥ 2522.

Benchmark (not a test — runs on demand via `scripts/benchmark_calibration.py`):
- 10 generations × 20 candidates on the Baltic example.
- Run once with `use_java_engine=True` (requires `OSMOSE_JAR`), once with default Python engine.
- Report wall-clock ratio.
- Target: ≥ 4×.

## Risks and mitigations

| Risk | Manifestation | Mitigation |
|---|---|---|
| `_records_to_dataframe` refactor changes CSV output | Existing output tests fail | Parity test at test-suite level: disk path still passes every existing output test unchanged. `write_outputs()` after refactor is literally `df = _records_to_dataframe(...); df.to_csv(path)` — same values. |
| `simulate()` signature doesn't tolerate `output_dir=None` | `run_in_memory` crashes | Audit every `output_dir.<something>` call site in `simulate()`. Guard with `if output_dir is not None:` for any write-side operation. Add a regression test. |
| In-memory cache too large on long runs | Memory balloon in calibration | The cache holds ~12 DataFrames per simulation, each roughly `n_steps × n_species` rows (typically < 100k rows total). For 1000-candidate NSGA-II runs, workers release candidates after objective computation, so only `n_parallel` in flight at once. Memory is bounded and small. |
| Java/Python objective values differ > 1 OoM on some configs | Cross-engine parity test fails | Python engine has documented parity at 14/14 EEC + 8/8 BoB within 1 OoM. The cross-engine test uses the same configs used for parity validation. If a new config shows drift, the cross-engine test is the canary, not a blocker for shipping — document it and let users opt into `use_java_engine=True`. |
| `DeprecationWarning` on existing callers | Noisy log output for users of older API | Warning is emitted once per process (default `warnings.warn` dedup). Suppressed in the test suite via `pytest.warns` where we explicitly test the flag. Removal scheduled for v0.11.0 (one release's notice). |
| `OsmoseProblem._cache_key` references `jar_mtime` which may be None in Python path | Cache key inconsistency | When `use_java_engine=False`, substitute `osmose.__version__` for `jar_mtime`. Cache keys are different between the two engines by design — a Python run and a Java run with identical overrides produce different caches, which is correct because the engines are different artifacts. |
| NSGA-II convergence changes because Python engine RNG stream differs from Java | Different Pareto front after the port | Expected. Not a bug — it's the engine switch surfacing as a stochastic difference. Users who need reproducibility with the Java path flip `use_java_engine=True`. Document explicitly in CHANGELOG. |

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
