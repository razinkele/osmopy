# OSMOSE HPC Apptainer container — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a headless batteries-included (Python + Java + numba) Apptainer image + HPC docs + a CI build-smoke, plus the two small app touch-ups that make the read-only-container conventions actually hold.

**Architecture:** Two TDD code touch-ups (calibration honors `OSMOSE_RESULTS_DIR`; `osmose run --jar` defaults to `$OSMOSE_JAR`) so all batch entry points write/find things correctly read-only; then a version-controlled `apptainer/osmose.def` (JRE 17 + JAR + numba cache baked at build), `docs/hpc-apptainer.md`, and a CI job that builds the `.sif` with `--fakeroot` and smoke-tests it.

**Tech Stack:** Apptainer/Singularity, Python 3.12, numba, OpenJDK 17 (JRE), GitHub Actions, pytest.

## Global Constraints

- **Read-only at runtime:** the `.sif` is read-only; all writes go to a bind-mounted host dir via `OSMOSE_RESULTS_DIR=/results` (+ `-B host:/results`) and the sweep `--out /results/…`. Configs bind read-only to a separate `/cfg`.
- **Base:** `Bootstrap: docker` / `From: python:3.12-slim` (Debian bookworm). JRE via apt `openjdk-17-jre-headless`; `JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64` (x86_64).
- **JAR (not in repo):** build-provided `osmose-java/*.jar` is the PRIMARY path; else download the pinned `https://github.com/osmose-model/osmose/releases/download/v4.3.3/osmose-4.3.3-jar-with-dependencies.jar` (hyphenated) with a sha256 check; else build still succeeds and `osmose run` errors clearly.
- **numba cache:** `@njit(cache=True)` (9 sites in `osmose/engine/`) must not write to read-only site-packages — set `NUMBA_CACHE_DIR=/opt/numba-cache` and bake the cache at build with a warm-up run.
- **`pip install ".[numba]"`** (`numba` is a real optional-dependency group; core deps include `netCDF4` needing `libhdf5-dev`).
- **App touch-ups must not change engine dynamics** — EEC/BoB parity untouched.
- **Run python** with `PYTHONPATH=.` from the worktree root via `.venv/bin/python`. Lint: `.venv/bin/ruff check` + `ruff format --check`.
- Spec: `docs/superpowers/specs/2026-06-26-hpc-apptainer-container-design.md`.

---

### Task 1: calibration honors `OSMOSE_RESULTS_DIR`

**Files:**
- Modify: `scripts/calibrate_baltic.py:39`
- Test: `tests/test_hpc_container_touchups.py` (new)

**Interfaces:**
- Consumes: `osmose.calibration.checkpoint.RESULTS_DIR` (already `= default_results_dir()`, env-aware).
- Produces: `scripts.calibrate_baltic.RESULTS_DIR` now equals the env-resolved dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hpc_container_touchups.py
import importlib
import os


def test_calibrate_baltic_honors_results_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMOSE_RESULTS_DIR", str(tmp_path / "rd"))
    import osmose.calibration.checkpoint as cp
    importlib.reload(cp)  # re-evaluate RESULTS_DIR = default_results_dir() with the env set
    cb = importlib.import_module("scripts.calibrate_baltic")
    cb = importlib.reload(cb)
    assert cb.RESULTS_DIR == (tmp_path / "rd")


def test_calibrate_baltic_results_dir_default_without_env(monkeypatch):
    monkeypatch.delenv("OSMOSE_RESULTS_DIR", raising=False)
    import osmose.calibration.checkpoint as cp
    importlib.reload(cp)
    cb = importlib.reload(importlib.import_module("scripts.calibrate_baltic"))
    assert cb.RESULTS_DIR.name == "calibration_results"  # package-root default
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_hpc_container_touchups.py -k calibrate -v`
Expected: FAIL (`test_calibrate_baltic_honors_results_dir_env`: `RESULTS_DIR` is the hardcoded `PROJECT_ROOT/data/...`, not the env path).

- [ ] **Step 3: Implement** — in `scripts/calibrate_baltic.py`, replace the hardcoded line 39
`RESULTS_DIR = PROJECT_ROOT / "data" / "baltic" / "calibration_results"` with an import of the
env-aware single-source-of-truth (which `checkpoint.py`'s own comment says this script should use):

```python
from osmose.calibration.checkpoint import RESULTS_DIR  # honors OSMOSE_RESULTS_DIR
```

Place the import with the other top-of-file imports; delete the line-39 redeclaration. (`PROJECT_ROOT`/`BALTIC_CONFIG`/`TARGETS_CSV` stay — only `RESULTS_DIR` changes.)

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_hpc_container_touchups.py -k calibrate -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_hpc_container_touchups.py
git commit -m "fix(calibrate): honor OSMOSE_RESULTS_DIR (container-friendly results dir)"
```

---

### Task 2: `osmose run --jar` defaults to `$OSMOSE_JAR`

**Files:**
- Modify: `osmose/cli.py` (`cmd_run` ~49, the `run` parser ~113)
- Test: `tests/test_hpc_container_touchups.py`

**Interfaces:**
- Produces: `osmose.cli._jar_from(args_jar) -> str | None` (helper: `--jar` value, else `$OSMOSE_JAR`, else None); `osmose run` with `--jar` optional.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_hpc_container_touchups.py
def test_jar_from_prefers_arg_then_env(monkeypatch):
    from osmose.cli import _jar_from
    monkeypatch.setenv("OSMOSE_JAR", "/env/osmose.jar")
    assert _jar_from("/cli/x.jar") == "/cli/x.jar"   # explicit --jar wins
    assert _jar_from(None) == "/env/osmose.jar"        # falls back to $OSMOSE_JAR
    monkeypatch.delenv("OSMOSE_JAR", raising=False)
    assert _jar_from(None) is None                      # neither -> None


def test_cmd_run_clear_error_when_no_jar(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from osmose.cli import cmd_run
    monkeypatch.delenv("OSMOSE_JAR", raising=False)
    cfg = tmp_path / "c.csv"; cfg.write_text("simulation.nspecies;1\n")
    rc = cmd_run(Namespace(config=str(cfg), jar=None, output=None, java_opts=None, timeout=None))
    assert rc == 1
    assert "jar" in capsys.readouterr().err.lower()  # clear error, NOT an argparse usage error
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_hpc_container_touchups.py -k "jar or cmd_run" -v`
Expected: FAIL (`_jar_from` not defined).

- [ ] **Step 3: Implement** — in `osmose/cli.py`:

(a) add `import os` at the top if absent, and the helper near `cmd_run`:
```python
def _jar_from(args_jar: str | None) -> str | None:
    """JAR path: explicit --jar, else $OSMOSE_JAR, else None."""
    return args_jar or os.environ.get("OSMOSE_JAR")
```
(b) in `cmd_run`, replace the `jar_path = Path(args.jar)` block (lines ~49-52) with:
```python
    jar = _jar_from(args.jar)
    if not jar:
        print("Error: no JAR specified (pass --jar or set OSMOSE_JAR)", file=sys.stderr)
        return 1
    jar_path = Path(jar)
    if not jar_path.exists():
        print(f"Error: JAR not found: {jar_path}", file=sys.stderr)
        return 1
```
(c) in the `run` parser (line ~113), drop `required=True`:
```python
    p_run.add_argument("--jar", help="Path to OSMOSE JAR (default: $OSMOSE_JAR)")
```

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_hpc_container_touchups.py -v`
Expected: PASS (Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/cli.py tests/test_hpc_container_touchups.py
git commit -m "feat(cli): osmose run --jar defaults to \$OSMOSE_JAR (container-friendly)"
```

---

### Task 3: `apptainer/osmose.def` — the container definition

**Files:**
- Create: `apptainer/osmose.def`
- Test: a local `apptainer build --fakeroot` if Apptainer is installed; otherwise the CI build-smoke (Task 5) is the gate. A non-Apptainer sanity check is in Step 4.

**Interfaces:**
- Consumes: the repo (`osmose/`, `scripts/`, `data/`, `pyproject.toml`); optionally a build-provided `osmose-java/*.jar`.
- Produces: `osmose.sif` exposing `osmose` (runscript) + the batch scripts (via `apptainer exec`), with `OSMOSE_RESULTS_DIR=/results`, `OSMOSE_JAR`, `NUMBA_CACHE_DIR` baked.

- [ ] **Step 1: Write the definition** — create `apptainer/osmose.def`:

```
Bootstrap: docker
From: python:3.12-slim

%files
    pyproject.toml /app/pyproject.toml
    osmose /app/osmose
    scripts /app/scripts
    data /app/data
    osmose-java /opt/osmose-build-jar

%post
    set -eu
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends libhdf5-dev curl git openjdk-17-jre-headless
    rm -rf /var/lib/apt/lists/*

    cd /app
    pip install --no-cache-dir ".[numba]"

    # OSMOSE JAR: build-provided is primary (offline-friendly); else download pinned 4.3.3.
    mkdir -p /opt/osmose
    JAR_URL="https://github.com/osmose-model/osmose/releases/download/v4.3.3/osmose-4.3.3-jar-with-dependencies.jar"
    JAR_SHA256="REPLACE_WITH_PINNED_SHA256"
    if ls /opt/osmose-build-jar/*.jar >/dev/null 2>&1; then
        cp "$(ls /opt/osmose-build-jar/*.jar | head -n1)" /opt/osmose/osmose.jar
        echo "OSMOSE JAR: used build-provided jar"
    elif curl -fsSL "$JAR_URL" -o /opt/osmose/osmose.jar; then
        echo "${JAR_SHA256}  /opt/osmose/osmose.jar" | sha256sum -c - \
            || { echo "WARN: JAR sha256 mismatch — removing"; rm -f /opt/osmose/osmose.jar; }
    else
        echo "WARN: no OSMOSE JAR (build-provided absent + download failed); 'osmose run' will error at runtime"
    fi

    # Bake the numba JIT cache so the read-only runtime reads it (no per-run re-JIT).
    mkdir -p /opt/numba-cache /results
    export NUMBA_CACHE_DIR=/opt/numba-cache OSMOSE_DATA_DIR=/app/data PYTHONPATH=/app
    python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
raw = dict(OsmoseConfigReader().read("/app/data/examples/osm_all-parameters.csv"))
raw["simulation.time.nyear"] = "1"
PythonEngine().run_in_memory(raw, seed=0)  # warms @njit(cache=True) artifacts into NUMBA_CACHE_DIR
print("numba cache warmed")
PY

%environment
    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
    export PATH="${JAVA_HOME}/bin:${PATH}"
    export PYTHONPATH=/app
    export OSMOSE_DATA_DIR=/app/data
    export OSMOSE_RESULTS_DIR=/results
    export OSMOSE_JAR=/opt/osmose/osmose.jar
    export NUMBA_CACHE_DIR=/opt/numba-cache

%runscript
    exec osmose "$@"

%help
    OSMOSE HPC batch image (Python + Java engines, numba).
    The image is READ-ONLY: bind a writable host dir to /results and send all output there.
      Fmsy sweep:
        apptainer exec -B $CFG:/cfg:ro -B $OUT:/results osmose.sif \
          python /app/scripts/compute_model_reference_points.py \
          --config /cfg/osm_all-parameters.csv --workers $SLURM_CPUS_PER_TASK --out /results/rp.json
      Calibration (honors OSMOSE_RESULTS_DIR=/results):
        apptainer exec -B $OUT:/results osmose.sif python /app/scripts/calibrate_baltic.py --phase 1
      Java reference engine:
        apptainer run osmose.sif run /cfg/osm_all-parameters.csv --jar $OSMOSE_JAR --output /results/java
    The primary batch jobs use 'apptainer exec ... python scripts/...', NOT 'apptainer run'.

%labels
    Maintainer OSMOSE/OSMOPY
    Description Headless HPC batch image (Python + Java engines, numba)
    Version 4.3.3
```

Note: replace `REPLACE_WITH_PINNED_SHA256` with the real sha256 of the v4.3.3 jar (`gh release download v4.4.1`-style fetch then `sha256sum`); if pinning is deferred, drop the `sha256sum -c` line and add a comment that the JAR is unverified.

- [ ] **Step 2: Add a `.dockerignore`-style exclude is unnecessary** — `%files` is explicit (only the batch surface is copied), so no extra ignore file is needed. (`osmose-java` may be absent; the `%files` line then copies nothing for it — Apptainer warns but continues; if that errors on the implementer's Apptainer version, guard it by creating an empty `osmose-java/.keep` in the repo OR move the build-jar copy to a bind-mount at build time. Verify during the local/CI build.)

- [ ] **Step 3: (no pytest)** — the `.def` is validated by a build, not pytest.

- [ ] **Step 4: Validate** — if Apptainer is installed locally:
Run: `apptainer build --fakeroot /tmp/osmose.sif apptainer/osmose.def && apptainer exec /tmp/osmose.sif osmose --help`
Expected: build succeeds; `osmose --help` prints usage.
If Apptainer is NOT installed locally, run a structural sanity check instead:
Run: `grep -qE '^Bootstrap:' apptainer/osmose.def && grep -qE '^%post' apptainer/osmose.def && grep -qE '^%runscript' apptainer/osmose.def && echo "def-structure-ok"`
Expected: `def-structure-ok` (the real build runs in CI — Task 5).

- [ ] **Step 5: Commit**

```bash
git add apptainer/osmose.def
git commit -m "feat(hpc): Apptainer definition (Python+Java+numba, JAR + baked numba cache, read-only-friendly)"
```

---

### Task 4: `docs/hpc-apptainer.md` — the HPC run guide

**Files:**
- Create: `docs/hpc-apptainer.md`

- [ ] **Step 1: Write the doc** — create `docs/hpc-apptainer.md` covering, with copy-pasteable commands:

````markdown
# Running OSMOSE on HPC with Apptainer/Singularity

A headless batch image (Python + Java engines, numba) for cluster runs of the Fmsy sweep,
NSGA-II calibration, and the Java reference engine.

## Build

The build needs **PyPI + GitHub egress** (pip deps + the OSMOSE JAR download). Pick the path your
cluster allows:

```bash
# Unprivileged build on a node that allows user namespaces:
apptainer build --fakeroot osmose.sif apptainer/osmose.def
# A remote builder (if you have an Apptainer/Sylabs account):
apptainer build --remote osmose.sif apptainer/osmose.def
# Or build the Docker image and convert:
docker build -t osmose -f Dockerfile . && apptainer build osmose.sif docker-daemon://osmose:latest
```
Air-gapped build node? Build on a networked machine and `scp` the `.sif`. To avoid GitHub egress,
drop the JAR into `osmose-java/` before building (the def uses a build-provided jar in preference to
downloading).

## The read-only rule (read this first)

The `.sif` is **read-only at runtime**. Every output must go to a **writable host dir** bound to
`/results`, and Apptainer auto-binds `$HOME`/`/tmp`/`$PWD` but **not** `/app`. Use `--contain` for
reproducible, leak-free runs.

```bash
mkdir -p /scratch/$USER/results
RES=/scratch/$USER/results
```

## Run the batch jobs (via `apptainer exec`)

```bash
# Fmsy yield-vs-F sweep (always pass --out to a /results path):
apptainer exec --contain -B $RES:/results -B $PWD/data:/cfg:ro osmose.sif \
  python /app/scripts/compute_model_reference_points.py \
  --config /cfg/baltic/baltic_all-parameters.csv --workers ${SLURM_CPUS_PER_TASK:-4} \
  --out /results/baltic_rp.json

# NSGA-II calibration (writes to /results via OSMOSE_RESULTS_DIR):
apptainer exec --contain -B $RES:/results osmose.sif \
  python /app/scripts/calibrate_baltic.py --phase 1 --maxiter 200

# Java reference engine:
apptainer run --contain -B $RES:/results osmose.sif \
  run /app/data/examples/osm_all-parameters.csv --jar $OSMOSE_JAR --output /results/java
```

## SLURM (job-array fan-out of independent runs)

Each array task is one **independent, single-node** sweep (e.g. one config/scenario/seed); the sweep
parallelizes internally over species via `--workers`. (Partitioning a *single* sweep across tasks
would need a new `--species` flag — not yet supported.)

```bash
#!/bin/bash
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
CONFIGS=(baltic eec_full minimal examples)
ECO=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
apptainer exec --contain -B /scratch/$USER/results:/results -B $PWD/data:/cfg:ro osmose.sif \
  python /app/scripts/compute_model_reference_points.py \
  --config /cfg/$ECO/${ECO}_all-parameters.csv --workers $SLURM_CPUS_PER_TASK \
  --out /results/${ECO}_rp.json
```

## Smoke test (a few minutes)

```bash
apptainer exec -B /tmp:/results osmose.sif python /app/scripts/compute_model_reference_points.py \
  --config /app/data/examples/osm_all-parameters.csv --grid 0 0.5 --n-years 3 --replicates 1 \
  --workers 1 --out /results/rp.json && echo OK
```

## Notes
- `data/` is bundled in the image (~14 MB) as a baseline; bind your own configs read-only to `/cfg`.
- The `.def` and the web `Dockerfile` share the apt/pip/Java steps — keep them in sync (a shared
  install script is a future refactor).
````

- [ ] **Step 2: Commit**

```bash
git add docs/hpc-apptainer.md
git commit -m "docs(hpc): Apptainer build + run guide (read-only rule, SLURM job array, smoke)"
```

---

### Task 5: CI build-smoke job

**Files:**
- Modify: `.github/workflows/ci.yml` (add a job)

- [ ] **Step 1: Add the job** — append a new job to `.github/workflows/ci.yml` (sibling of the existing test/docker jobs):

```yaml
  apptainer-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - name: Install Apptainer
        uses: eWaterCycle/setup-apptainer@v2
      - name: Build the HPC image
        run: apptainer build --fakeroot osmose.sif apptainer/osmose.def
      - name: Smoke — CLI + engine import + numba cache + java
        run: |
          apptainer exec osmose.sif osmose --help
          apptainer exec osmose.sif python -c "import osmose.engine"
          apptainer exec osmose.sif java -version
          apptainer exec osmose.sif sh -c 'ls -A "$NUMBA_CACHE_DIR" | grep -q . && echo numba-cache-baked'
      - name: Smoke — tiny sweep into a writable bind mount
        run: |
          mkdir -p "$RUNNER_TEMP/results"
          apptainer exec -B "$RUNNER_TEMP/results:/results" osmose.sif \
            python /app/scripts/compute_model_reference_points.py \
            --config /app/data/examples/osm_all-parameters.csv \
            --grid 0 0.5 --n-years 3 --replicates 1 --workers 1 --out /results/rp.json
          test -f "$RUNNER_TEMP/results/rp.json" && echo sweep-wrote-to-bind-mount
```

- [ ] **Step 2: Validate the YAML** — confirm the workflow parses:
Run: `PYTHONPATH=. .venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml-ok')"`
Expected: `yaml-ok`.

- [ ] **Step 3: Full check + commit** — run the app-touch-up tests + lint once more:
Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_hpc_container_touchups.py -q && .venv/bin/ruff check scripts/calibrate_baltic.py osmose/cli.py tests/test_hpc_container_touchups.py`
Expected: pass + clean.
```bash
git add .github/workflows/ci.yml
git commit -m "ci(hpc): build the Apptainer image with --fakeroot + smoke (CLI/engine/numba-cache/java/sweep)"
```

---

## Notes for the executor

- **The two app touch-ups are the only code changes** and are pytest-covered (Tasks 1–2); they must not touch engine dynamics (parity stays green). The `.def`/docs/CI are artifacts validated by the CI build-smoke (Task 5), not pytest.
- **numba cache:** the baked-cache warm-up in the `.def` (`%post`) is what preserves the speedup read-only — do not drop it; the CI smoke asserts the cache dir is non-empty.
- **JAR:** build-provided `osmose-java/*.jar` wins over the download. The CI runner has network, so the download path is exercised there. Pin the real sha256 if available; otherwise note it's unverified.
- **`osmose-java/` in `%files`:** the dir may not exist in the repo — verify the chosen Apptainer version tolerates a missing `%files` source (guard with an empty `osmose-java/.keep` if needed).
- Apptainer is an HPC tool and likely absent on the dev box — Task 3's real validation is the CI job; the local step falls back to a structural grep.
