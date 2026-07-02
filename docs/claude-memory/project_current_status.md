---
name: OSMOSE Python current release + active plan status
description: Latest version, test count, release summary, and the active executable plan. Update on every release or when a major plan lands.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
**As of 2026-04-21 (post-v0.10.0 release):**

- **Version: v0.10.0 tagged and pushed to origin** (merge commit `d4eebe1`). PR #1 squash-merged. Marquee: NSGA-II calibration ported onto PythonEngine in-memory; measured 3.02× wall-clock speedup over Java on Baltic 3-gen × 10-pop at `n_parallel=4`. Three benchmark-surfaced fixes shipped with it (HDF5 thread-safety, Java output-prefix auto-detect, new DRY RNG helper).
- **2529 tests passing**, 20 skipped, 41 deselected, 0 failures, lint clean. Count grew from 2527 (added 2 regression tests for `open_dataset_safe` thread-safety).
- **v0.9.3** (`ea31ac8`) shipped Phase 7.1 predation reconciliation.
- **v0.9.2** (`0e59258`) shipped Phase 7.3 config validation.
- **Parity roadmap Phase 7: FULLY CLOSED.** 7.1 SHIPPED 2026-04-19, 7.2 SHIPPED pre-v0.9.0, 7.3 SHIPPED in v0.9.2.

**Engine parity milestones:**

- Python Engine Phases 1-9 COMPLETE — full Java parity except Ev-OSMOSE genetics (shipped separately in Front 6).
- EEC parity: 14/14. Bay of Biscay: 8/8 at year 1.
- Python FASTER than Java for single-sim: BoB 5yr 1.99s (Java 2.3s), EEC 5yr 5.2s (Java 7.2s, 1.4x faster).
- Calibration (v0.10.0): Python 3.02× faster than per-candidate Java subprocesses at `n_parallel=4` (limited by GIL among in-process workers; half-saturated shows 4.34×).

**Deploy:**

- App live at `https://laguna.ku.lt/osmose/` (osmose-shiny.service on port 8838, nginx proxy; symlink `/srv/shiny-server/osmose -> /home/razinka/osmose/osmose-python`).
- Git remote: SSH `git@github.com:razinkele/osmopy.git`.

**No active plan.** Known post-v0.10.0 follow-ups (see `project_v010_calibration_python_engine.md`):
- `ThreadPoolExecutor` → `ProcessPoolExecutor` in `OsmoseCalibrationProblem._evaluate` — expected to raise calibration speedup from 3× toward smoke-level 4-5× by giving each worker its own GIL. Needs: problem picklability audit, Numba per-process warmup measurement, NetCDF backend per-process load cost.
- CI red on master since v0.9.3 — `shiny_deckgl` is a private-repo git dependency that CI runners can't clone. Either publish to PyPI, make the repo public, or add a PAT to CI.

**Historical remediation threads (all closed):** Deep review v1, v2, v3 (Critical C-1..C-8, Important I-1..I-10, Minor M-2..M-14 + deferred I-3/M-5/M-7/D-1/M-9). See dedicated memory files for details.
