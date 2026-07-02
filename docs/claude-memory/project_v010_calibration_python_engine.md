---
name: v0.10.0 calibration-on-PythonEngine + release-gate lessons
description: 2026-04-21 release — NSGA-II calibration ported to in-memory PythonEngine, with three benchmark-surfaced fixes (HDF5 thread-safety, Java output-prefix mismatch, GIL contention insight). Measured 3.02× speedup, below aspirational 4× gate.
type: project
originSessionId: 61cc1c71-c262-410f-9cfd-f2eedf39023a
---
**SHIPPED 2026-04-21** (tag `v0.10.0`, merge commit `d4eebe1`, PR #1 closed).

## Marquee change
`OsmoseCalibrationProblem._run_single` now dispatches to `_run_python_engine` by default, which calls `PythonEngine.run_in_memory(cfg, seed)` — in-process, no disk round-trip. Java subprocess path retained behind `use_java_engine=True`. New `OsmoseResults.from_outputs(outputs, engine_config, grid)` factory; new `_build_*_dataframes` helpers extracted from `osmose/engine/output.py`.

**BREAKING:** `OsmoseCalibrationProblem.__init__` signature reordered (`work_dir` → position 4; `jar_path` demoted to optional kw-only; previously-positional kwargs now kw-only). Callers passing `jar_path` positionally for the Java path must add `use_java_engine=True`.

## Measured release-gate result: 3.02×, below 4× gate
Baltic 3-gen × 10-pop NSGA-II at `n_parallel=4` on 28-core host:
- Python: 4027.72s (~67 min)
- Java: 12149.11s (~202 min)
- Ratio: **3.02×**

**Why below the 4× aspiration:** half-saturated 1×2 smoke showed 4.34×. The drop at full saturation is **GIL contention among 4 ThreadPoolExecutor workers sharing one process**, not Numba oversubscription (empirical: thread caps via `NUMBA_NUM_THREADS=7` slowed it further because the mortality-phase `@njit(parallel=True) prange` benefits from unrestricted core use). Java subprocesses each have their own JVM with independent scheduling. **Post-v0.10.0 follow-up**: swap `ThreadPoolExecutor` → `ProcessPoolExecutor` in `OsmoseCalibrationProblem._evaluate` (concerns: problem picklability, Numba cache per-process warmup, NetCDF backend per-process repeat-load).

## Three benchmark-surfaced fixes
1. **HDF5 thread-safety (gen-2 crash)** — netCDF4-python C extension isn't thread-safe; under `ThreadPoolExecutor` (`n_parallel>1`), gen 1 succeeded but gen 2 aborted with `NetCDF: Can't open HDF5 attribute` + `double free or corruption`. Root cause: `osmose/engine/resources.py:_load_netcdf` held `xr.open_dataset` handles open for the whole simulation lifetime; gen-2 opens raced stale gen-1 handles. **Fix**: new `osmose/engine/_netcdf.open_dataset_safe(path)` helper serializes opens through a module-level `threading.Lock` and eagerly loads into memory so the handle closes before return. Applied at all four engine open sites (resources, grid, physical_data, background). Regression tests in `tests/test_thread_safety.py`.
2. **Java output-prefix mismatch** — `_run_java_subprocess` constructed `OsmoseResults(output_dir)` with default prefix `"osm"`, but Java writes `<output.file.name>_<type>_Simu*.csv` using the config's `output.file.name`. On Baltic that's `baltic_*`, so `biomass()` silently returned empty and objective crashed with `IndexError`. **Fix**: post-subprocess, glob `*_biomass_Simu*.csv` and pass actual prefix to `OsmoseResults`. Pre-existing bug, never triggered because no one ran parallel Java-calibration on Baltic before.
3. **Benchmark-script flags** — added `--skip-python` + `--python-wallclock` so a rerun can reuse a prior Python measurement instead of re-running the 67-min Python half.

## Key memory-worthy gotchas
- **`scripts/release.py` auto-regenerates CHANGELOG from git log** — would nuke any hand-written `[Unreleased]` prose. For releases with migration notes / breaking docs, do the release manually: edit `osmose/__version__.py`, edit CHANGELOG directly, commit `release: vX.Y.Z`, tag, push. v0.10.0 was done manually for this reason.
- **CI on this repo has been red since v0.9.3** due to `shiny_deckgl @ git+https://github.com/razinka/shiny_deckgl.git@v1.9.1` — private repo, CI runner has no GitHub auth so `pip install -e ".[dev]"` fails in every job before any user code runs. Releases ship with red CI as a consequence. Fix candidates: make shiny_deckgl public, add PAT to CI, or publish to PyPI.
- **Squash-merge of long-lived branches bundles docs/spec commits that were already on master** into the squash commit. Local master will then diverge from origin — the plan/spec commits still exist locally as separate commits, while origin has them inside the squash. Safe to `git reset --hard origin/master` on the main worktree, since the content is preserved in the merge commit.

## Key files (derivable, not in memory)
- Fix for HDF5 race: `osmose/engine/_netcdf.py`
- New in-memory API: `OsmoseResults.from_outputs`, `PythonEngine.run_in_memory`
- Benchmark: `scripts/benchmark_calibration.py`
- Spec: `docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md`
- Plan: `docs/superpowers/plans/2026-04-20-calibration-python-engine-plan.md`
