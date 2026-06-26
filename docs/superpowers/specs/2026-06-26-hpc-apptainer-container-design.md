# OSMOSE HPC Apptainer/Singularity container — design

> Status: design (awaiting review) · 2026-06-26
> A headless, batteries-included (Python + Java engines) Apptainer image for running OSMOSE
> batch work — the Fmsy yield-vs-F sweep, NSGA-II calibration, plain engine runs, and the Java
> reference engine — reproducibly on an HPC cluster. Built from a version-controlled definition
> file, NOT a conversion of the web-app Dockerfile.

## 1. Why

The expensive offline batches this project now ships — the model-internal Fmsy sweep
(`scripts/compute_model_reference_points.py`, tens-of-minutes-to-hours) and NSGA-II calibration
(`scripts/calibrate_baltic.py`) — are exactly the work HPC clusters exist for, but there is no
HPC artifact: the repo has only a `Dockerfile` that builds the **Shiny web app** (port 8000, a
`shiny run` entrypoint, bundled data, an interactive-UI dependency set). Clusters need the
opposite: a read-only, headless image with a CLI entrypoint, bind-mounted data/results, and the
JIT-accelerated engine. This feature adds that image + the HPC run docs.

## 2. Scope decisions (from brainstorming)

- **Batteries-included (Python + Java).** The image bundles the JIT Python engine (`numba`) AND a
  JRE 17 + the OSMOSE JAR, so all batch entry points work — including `osmose run` (the Java
  reference engine) for cross-validation.
- **Built from a native `.def`, not a Docker conversion.** A purpose-built batch image (no Shiny /
  web UI / `app.py`), so it stays lean and HPC-idiomatic; the web Dockerfile is unchanged.
- **Single-node multicore** (the engine's `ProcessPoolExecutor` / `--workers`); multi-node is
  user-side SLURM orchestration (job arrays), documented but not built in.
- **numba included** (the 5–10× simulation speedup that makes the sweep/calibration tractable).

## 3. Components

1. **`apptainer/osmose.def`** (new) — the Apptainer definition.
   - `Bootstrap: docker` / `From: python:3.12-slim` (same base + Debian-bookworm `libhdf5` ABI as
     the Dockerfile, so the build is known-good).
   - `%files`: copy only the batch surface into `/app/` — `pyproject.toml`, `osmose/`, `scripts/`,
     `data/` (the bundled-ecosystem baseline). **Omit** `ui/`, `www/`, `app.py` (web-only).
   - `%post`:
     - `apt-get install --no-install-recommends libhdf5-dev curl git openjdk-17-jre-headless`
       (the JRE replaces the Dockerfile's multi-stage Temurin copy — simpler in a single-stage def;
       `java` lands on `PATH`).
     - `pip install --no-cache-dir ".[numba]"` (core deps + numba; NOT the web extras — they come
       in via the base deps regardless, but no playwright/dev tools).
     - **Acquire the JAR** (not in the repo): download the pinned
       `osmose_4.3.3-jar-with-dependencies.jar` from the OSMOSE GitHub release to `/opt/osmose/osmose.jar`
       and verify its sha256 (pin both in the def). If the download is unavailable at build time, the
       def falls back to copying a build-provided `osmose-java/*.jar` (the Dockerfile's optional-wildcard
       behaviour) and logs a warning; `osmose run` then errors clearly at runtime if the JAR is absent.
     - `mkdir -p /results` (the default writable output mount point).
   - `%environment`: `JAVA_HOME` (the apt JRE path), `OSMOSE_DATA_DIR=/app/data` (the package is
     pip-installed, so the bundled data is found via this env var — matching the Dockerfile),
     `OSMOSE_RESULTS_DIR=/results` (the default writable mount), `OSMOSE_JAR=/opt/osmose/osmose.jar`.
   - `%runscript`: `exec osmose "$@"` (the `osmose` console script — validate/run/report — so
     `apptainer run osmose.sif validate <cfg>` works); batch scripts are run via `apptainer exec`.
   - `%help`: concise usage — the four batch invocations + the bind-mount pattern + the read-only gotcha.
   - `%labels`: `Maintainer`, `Version` (the OSMOSE/OSMOPY version), `Description`.
2. **`docs/hpc-apptainer.md`** (new) — the HPC guide.
   - **Build paths** (covering the common "no root on the login node" cases):
     `apptainer build --fakeroot osmose.sif apptainer/osmose.def`; a remote builder
     (`apptainer build --remote`); or build the Docker image and convert
     (`apptainer build osmose.sif docker-daemon://osmose:latest`).
   - **Run each batch entry point** via `apptainer exec`, with the bind-mounts:
     - Fmsy sweep: `apptainer exec -B $CFG:/cfg:ro -B $OUT:/results osmose.sif python
       scripts/compute_model_reference_points.py --config /cfg/<master>.csv --workers $SLURM_CPUS_PER_TASK --out /results/rp.json`.
     - Calibration: `apptainer exec -B ... osmose.sif python scripts/calibrate_baltic.py ...` with
       `OSMOSE_RESULTS_DIR=/results`.
     - Plain Python-engine run + `osmose run` (Java engine, `--jar $OSMOSE_JAR`).
   - **The read-only gotcha (prominent):** the image is read-only at runtime, so the bundled `data/`
     and the default `calibration_results` dir are NOT writable — all outputs MUST go to a
     bind-mounted writable host dir (`-B /scratch/$USER/results:/results` + `OSMOSE_RESULTS_DIR=/results`,
     and the sweep `--out /results/...`). Host configs bind-mount read-only.
   - **A SLURM `sbatch` example**: a job array fanning the sweep across nodes (embarrassingly parallel
     over species / replicates / seeds), with `--cpus-per-task` → `--workers` and per-task `--out`.
   - **A "keep in sync" note**: the `.def` and the `Dockerfile` share the apt + pip + Java steps; v1
     keeps them as two files (a shared `scripts/install-deps.sh` refactor is a deferred follow-up).
3. **CI build-smoke** (new GitHub Actions job, mirroring the existing Docker smoke) — install
   Apptainer (`eWaterCycle/setup-apptainer` or the apt path), `apptainer build --fakeroot osmose.sif
   apptainer/osmose.def`, then smoke: `apptainer exec osmose.sif osmose --help` and
   `apptainer exec osmose.sif python -c "import osmose.engine"`. So the recipe can't silently rot.

## 4. Data flow

build: `apptainer/osmose.def` → (apt JRE + libhdf5; pip `.[numba]`; download pinned JAR) → `osmose.sif`.
run (HPC): host config (RO bind) + host results dir (RW bind, `OSMOSE_RESULTS_DIR`) →
`apptainer exec osmose.sif python scripts/<sweep|calibrate>.py …` → outputs land in the bound results dir.

## 5. Error handling / edge cases

- **JAR download fails at build** → fall back to a build-provided `osmose-java/*.jar`; if neither,
  the image still builds (Python-engine batch works) and `osmose run` errors clearly that the JAR is
  missing (Java is the optional path).
- **Read-only filesystem** → results to a writable bind mount; the docs make this the first thing.
- **`--fakeroot` unavailable on the build host** → the docs give the remote-builder + docker-convert
  fallbacks.
- **numba/llvmlite wheel availability** for the base arch → `python:3.12-slim` on x86_64 has wheels;
  the def notes ARM/other arches may need a source build (out of scope).
- **Apptainer absent in CI** → the build-smoke is a separate job; if the setup action fails the job
  fails loudly (not silently skipped).

## 6. Testing

- **CI build-smoke** (the primary automated gate): the `.def` builds with `--fakeroot`;
  `apptainer exec osmose.sif osmose --help` exits 0; `import osmose.engine` succeeds.
- **Documented local smoke** (in `docs/hpc-apptainer.md`): a tiny real run inside the container —
  `apptainer exec -B /tmp:/results osmose.sif python scripts/compute_model_reference_points.py
  --config data/examples/<master>.csv --grid 0 0.5 --n-years 3 --replicates 1 --workers 1 --out
  /results/rp.json` — proving the sweep + readers + the writable bind-mount work end-to-end.
- **No application-code change** → the existing test suite + EEC/BoB parity are untouched (this is
  packaging/docs/CI only). The `.def` is validated by the build-smoke, not pytest.

## 7. Out of scope (deferred)

- **Multi-node MPI/Dask** parallelism (single-node multicore + SLURM job arrays is the model).
- **Refactoring the Dockerfile + `.def` to a shared install script** (v1 = two files + sync note).
- **Publishing the `.sif` / Docker image to a registry** (CI smoke-builds; no push).
- **A non-Java (Python-only) lean variant** (the chosen image is batteries-included; a slim variant
  is a future option).
- **ARM/non-x86_64 builds** (numba wheel availability not guaranteed).
