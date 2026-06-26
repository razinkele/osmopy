# OSMOSE HPC Apptainer/Singularity container — design

> Status: design (revised after in-loop review) · 2026-06-26
> A headless, batteries-included (Python + Java engines) Apptainer image for running OSMOSE
> batch work — the Fmsy yield-vs-F sweep, NSGA-II calibration, plain engine runs, and the Java
> reference engine — reproducibly on an HPC cluster. Built from a version-controlled definition,
> plus the **minimal container-friendliness app touch-ups** the review showed are required so the
> read-only-filesystem mitigation actually works for every entry point.

## 1. Why

The expensive offline batches this project ships — the Fmsy sweep
(`scripts/compute_model_reference_points.py`) and NSGA-II calibration
(`scripts/calibrate_baltic.py`) — are exactly the work HPC clusters exist for, but the repo has
only a `Dockerfile` that builds the **Shiny web app** (port 8000, `shiny run`, bundled data,
interactive-UI deps). Clusters need a read-only, headless image with a CLI entrypoint,
bind-mounted data/results, and the JIT-accelerated engine. This feature adds that image, the HPC
run docs, and three tiny code touch-ups so the container's conventions hold for all entry points.

## 2. Scope decisions (from brainstorming + review)

- **Batteries-included (Python + Java).** JIT Python engine (`numba`) AND a JRE 17 + the OSMOSE
  JAR, so all batch entry points work — including `osmose run` (Java reference engine).
- **Built from a native `.def`** (no Shiny/web), not a Docker conversion; the web Dockerfile is
  unchanged.
- **Single-node multicore** (`ProcessPoolExecutor` / `--workers`); multi-node is user-side SLURM
  **job-array fan-out of independent single-node runs** (in scope, documented), distinct from
  intra-run multi-node parallelism (deferred).
- **Minimal app touch-ups (the review's load-bearing finding):** the read-only-FS mitigation
  ("send writes to `OSMOSE_RESULTS_DIR`") was only partly true. Three small, independently-useful,
  unit-tested code changes make it true (see §3.1). This revises the original "packaging/docs only"
  framing.

## 3. Components

### 3.1 App touch-ups (small, each unit-tested)

These make the tools container-friendly (and better generally); each is ≤ a few lines + a test.

1. **`scripts/calibrate_baltic.py`** — its module-level `RESULTS_DIR =
   PROJECT_ROOT/"data"/"baltic"/"calibration_results"` (line 39) ignores the env var, so in a
   read-only image it crashes on first write. Make it honor **`OSMOSE_RESULTS_DIR`** (reuse
   `osmose.calibration.checkpoint.default_results_dir()` semantics, or a small env read). Test:
   with `OSMOSE_RESULTS_DIR` set, the resolved results dir points there.
2. **`osmose/cli.py`** — `osmose run --jar` is `required=True` with no fallback. Make `--jar`
   **default to `os.environ.get("OSMOSE_JAR")`** (drop `required=True`; keep the existing
   "JAR not found" runtime check so an unset/missing JAR still errors clearly). Test: with
   `OSMOSE_JAR` set and `--jar` omitted, the parsed jar path is the env value; unset+omitted →
   the clear "no JAR" error, not an argparse usage error.
3. **numba cache (not app code, but the same class of read-only fix)** — the engine's
   `@njit(cache=True)` writes to a `__pycache__` beside the source in (read-only) site-packages,
   so it silently falls back to per-process re-JIT, erasing the speedup §2 relies on. Handled in
   the `.def` (§3.2) via `NUMBA_CACHE_DIR` + a build-time warm-up.

### 3.2 `apptainer/osmose.def` (new) — the definition

- `Bootstrap: docker` / `From: python:3.12-slim` (Debian bookworm — same `libhdf5` ABI as the
  Dockerfile).
- `%files`: copy the batch surface into `/app/` — `pyproject.toml`, `osmose/`, `scripts/`, `data/`
  (~14 MB, negligible next to the JRE+numba footprint). Omit `ui/`, `www/`, `app.py`.
- `%post` (runs AFTER `%files`):
  - `apt-get install --no-install-recommends libhdf5-dev curl git openjdk-17-jre-headless`
    (the JRE replaces the Dockerfile's multi-stage Temurin copy; bookworm ships
    `openjdk-17-jre-headless`, and `update-alternatives` puts `java` on `PATH`).
  - `pip install --no-cache-dir ".[numba]"`.
  - **JAR (not in the repo): build-provided is the PRIMARY path** (hermetic/offline-friendly for
    locked-down HPC build nodes) — copy `osmose-java/*.jar` to `/opt/osmose/osmose.jar` if present;
    **else download** the pinned
    `https://github.com/osmose-model/osmose/releases/download/v4.3.3/osmose-4.3.3-jar-with-dependencies.jar`
    (hyphenated name; verify sha256). If neither, the image still builds (Python-engine batch works)
    and `osmose run` errors clearly that the JAR is absent.
  - **numba cache bake:** set `NUMBA_CACHE_DIR=/opt/numba-cache` and run a tiny warm-up
    (a 1-cell / few-step engine sim) so the `.nbi/.nbc` artifacts are baked into the image; numba
    reads a read-only cache fine at runtime → zero per-run JIT.
  - `mkdir -p /results`.
- `%environment`: `JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64` (x86_64; belt-and-suspenders — the
  runner only needs `java` on `PATH`), `OSMOSE_DATA_DIR=/app/data`, `OSMOSE_RESULTS_DIR=/results`,
  `OSMOSE_JAR=/opt/osmose/osmose.jar`, `NUMBA_CACHE_DIR=/opt/numba-cache`.
- `%runscript`: `exec osmose "$@"` (the console script — validate/run/report). **Note the
  asymmetry loudly in `%help`/docs:** the *primary* batch jobs (sweep, calibration) run via
  `apptainer exec … python scripts/…`, NOT `apptainer run`; `run` exposes the `osmose` CLI verbs.
- `%help`: the four batch invocations, the bind-mount pattern, the read-only gotcha, "always pass
  `--out`/`--jar $OSMOSE_JAR`".
- `%labels`: `Maintainer`, `Version` (OSMOPY version), `Description`.

### 3.3 `docs/hpc-apptainer.md` (new)

- **Build paths:** `apptainer build --fakeroot osmose.sif apptainer/osmose.def`; a remote builder
  (`--remote`); `docker-daemon://` convert; and a note that a pre-built `.sif` can be `apptainer
  pull`-ed if a registry image is published (currently not — §7). **Build needs PyPI + GitHub
  egress** (pip + the JAR fallback); air-gapped build nodes should build on a networked machine and
  `scp` the `.sif`, or provide the JAR via `osmose-java/` so only PyPI egress is needed.
- **Run each entry point** via `apptainer exec`, with bind-mounts:
  - Sweep: `apptainer exec -B $CFG:/cfg:ro -B $OUT:/results osmose.sif python
    scripts/compute_model_reference_points.py --config /cfg/osm_all-parameters.csv
    --workers $SLURM_CPUS_PER_TASK --out /results/rp.json`. **Always pass `--out`** (omitting it
    writes to the read-only bundled `data/<eco>/reference/`).
  - Calibration: `OSMOSE_RESULTS_DIR=/results apptainer exec -B $OUT:/results osmose.sif python
    scripts/calibrate_baltic.py …` (works thanks to §3.1.1).
  - Plain Python-engine run; `osmose run <cfg> --jar $OSMOSE_JAR` (Java engine).
- **The read-only gotcha (first thing):** the image is read-only at runtime; outputs MUST go to a
  writable bind-mount (`-B /scratch/$USER/results:/results` + `OSMOSE_RESULTS_DIR=/results` + the
  sweep `--out /results/…`). Host configs bind read-only to a separate `/cfg` mount (cleaner than
  binding over `/app/data`). Note Apptainer **auto-binds `$HOME`, `/tmp`, `$PWD`** — recommend
  `--contain` for reproducibility and to surface accidental writes (`/app` is NOT auto-bound).
- **SLURM `sbatch` example:** a job array of **independent whole-sweep runs** (one per config /
  scenario / seed), each a single-node `--workers N` job writing its own `--out`. (Partitioning a
  single sweep across array tasks would need a new `--species`/index flag + a merge step — §7.)

### 3.4 CI build-smoke (new GitHub Actions job)

Mirrors the existing Docker smoke. `eWaterCycle/setup-apptainer` → `apptainer build --fakeroot
osmose.sif apptainer/osmose.def` → smoke: (a) `apptainer exec osmose.sif osmose --help` exits 0;
(b) `apptainer exec osmose.sif python -c "import osmose.engine"`; (c) a tiny sweep into a writable
bind-mount (`-B /tmp:/results … --out /results/rp.json`) — exercises the engine + reader + writable
bind end-to-end and confirms the numba cache is used, not silently bypassed; (d) `apptainer exec
osmose.sif java -version` + assert `$OSMOSE_JAR` resolves (or the documented clear error). Fails
loudly if the setup action fails.

## 4. Data flow

build: `.def` → (apt JRE + libhdf5; pip `.[numba]`; JAR build-provided-or-download; numba warm-up)
→ `osmose.sif`. run (HPC): host config (RO `/cfg`) + host results (RW `/results`,
`OSMOSE_RESULTS_DIR`) → `apptainer exec osmose.sif python scripts/<sweep|calibrate>.py …` → outputs
in the bound results dir.

## 5. Error handling / edge cases

- **Read-only FS** → all writes to a writable bind mount; the §3.3 gotcha is the headline.
- **numba cache** baked at build (read RO at runtime); if the warm-up is skipped, set
  `NUMBA_CACHE_DIR` to writable scratch (re-JIT once per job).
- **calibration / `osmose run`** honor `OSMOSE_RESULTS_DIR` / `OSMOSE_JAR` via §3.1 — no per-script
  bind-over hacks needed.
- **JAR download fails / absent** → build-provided fallback; if neither, Python batch still works
  and `osmose run` errors clearly.
- **`--fakeroot` unavailable / air-gapped build** → remote builder, docker-convert, or
  build-elsewhere-and-`scp`; provide the JAR locally to avoid GitHub egress.
- **ARM/non-x86_64** → numba wheels + the `java-17-openjdk-amd64` path are x86_64 (out of scope).
- **Apptainer absent in CI** → the build-smoke is its own job; a failed setup fails loudly.

## 6. Testing

- **App touch-ups (pytest, in the default suite):** `calibrate_baltic` results-dir honors
  `OSMOSE_RESULTS_DIR`; `osmose run --jar` defaults to `$OSMOSE_JAR` (env set → used; unset+omitted →
  clear "no JAR" error). These run without a container.
- **CI build-smoke (§3.4):** the `.def` builds; `osmose --help`; `import osmose.engine`; a tiny
  sweep into a writable bind (proves the writable-mount + numba cache path); `java -version` + JAR
  resolves.
- **Documented local smoke** (`docs/hpc-apptainer.md`): the tiny sweep AND a 1-generation
  `calibrate_baltic` into `/results` (surfaces any residual read-only wall) AND `osmose run`
  against the bundled JAR.
- EEC/BoB parity untouched (the app touch-ups only change where outputs/JAR are read from, not
  dynamics).

## 7. Out of scope (deferred)

- **Intra-run multi-node** (MPI/Dask). Single-node multicore + job-array fan-out is the model.
- **Partitioning a single sweep across array tasks** (needs a new `--species`/seed-offset flag +
  a result-merge step).
- **Refactoring the Dockerfile + `.def` to a shared install script** (v1 = two files + a sync note).
- **Publishing the `.sif` / Docker image to a registry** (CI smoke-builds; no push).
- **A Python-only (no-Java) lean variant**; **ARM/non-x86_64** builds.
