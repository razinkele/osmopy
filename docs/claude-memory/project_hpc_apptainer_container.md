---
name: project-hpc-apptainer-container
description: 2026-06-26 OSMOSE HPC Apptainer/Singularity batch container shipped (image + docs + CI smoke + 2 read-only app touch-ups) + the read-only-FS landmine traps
metadata:
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

Shipped to local master 2026-06-26 (merge `ddc0d2c`, --no-ff; 7 deliverable commits `cf85484..be4f19f` + spec/plan). A **headless batteries-included (Python + Java + numba) batch image** for running OSMOSE on an HPC cluster — the Fmsy sweep, NSGA-II calibration, plain engine runs, and the Java reference engine. Built from a version-controlled `apptainer/osmose.def` (NOT a conversion of the web-app `Dockerfile`, which is Shiny/port-8000/bundled-data) + `docs/hpc-apptainer.md` (build paths, run commands, SLURM job-array, smoke) + a CI `apptainer-smoke` job (`--fakeroot` build + CLI/engine/numba-cache/java/sweep checks). Synergy with the expensive [[project-model-internal-reference-points]] sweep + calibration batches. Spec/plan: `docs/superpowers/{specs,plans}/2026-06-26-hpc-apptainer-container*`.

## KEY durable facts / traps (caught by in-loop reviews; verified vs source)
- **The "packaging-only" premise is FALSE — 3 read-only-FS landmines (the image is read-only at runtime):**
  1. **numba `@njit(cache=True)` (9 sites in `osmose/engine/`)** writes a `__pycache__` beside the source in read-only site-packages → silently falls back to **per-process re-JIT**, erasing the 5–10× speedup. Fix: **bake the cache at build** — `%post` sets `NUMBA_CACHE_DIR=/opt/numba-cache` + runs a 1-year warm-up sim; numba reads a read-only cache fine at runtime.
  2. **`scripts/calibrate_baltic.py` hardcoded `RESULTS_DIR = PROJECT_ROOT/data/...`** (ignored the env) → crashes read-only. Fix (app touch-up): `from osmose.calibration.checkpoint import RESULTS_DIR` (already env-aware via `OSMOSE_RESULTS_DIR`; checkpoint.py's own comment said calibrate_baltic should import it).
  3. **`osmose run --jar` was `required=True`** with no env fallback. Fix (app touch-up): a `_jar_from(args_jar)` helper → `--jar` else `$OSMOSE_JAR` else None; `--jar` now optional. Both touch-ups changed NO engine dynamics (EEC parity green).
- **`git` is REQUIRED in the `.def`** (`apt openjdk-17-jre-headless ... git`) — `pyproject.toml` has `git+https://` deps (`shiny_deckgl`, `pyvis`); `pip install` needs git to clone them. (A reviewer false-positively flagged git as bloat — rejected with evidence.)
- **The OSMOSE JAR is NOT in the repo** (the `Dockerfile` copies it optionally via `COPY osmose-java*` wildcard). The `.def` acquires it: build-provided `osmose-java/*.jar` PRIMARY (offline-friendly), else download pinned `https://github.com/osmose-model/osmose/releases/download/v4.3.3/osmose-4.3.3-jar-with-dependencies.jar` (HYPHEN, not underscore; sha256 dropped as unverified). `osmose-java/.keep` guards `%files` against the missing dir.
- **Config masters do NOT follow `<eco>_all-parameters.csv`:** `baltic/baltic_all-parameters.csv`, `eec_full/`**`eec_all-parameters.csv`**, `eec`/`examples`/`minimal`/ → `osm_all-parameters.csv`. (A SLURM-array doc example assuming the uniform pattern was a real bug — fixed to explicit paths.)
- **`--jar $OSMOSE_JAR` under `apptainer --contain` expands on the HOST** (unset → empty → "no JAR" error). Drop `--jar` entirely and rely on the container's `%environment OSMOSE_JAR` + the `_jar_from` fallback. Apptainer is **read-only at runtime**; auto-binds `$HOME`/`/tmp`/`$PWD` but NOT `/app` → outputs to a writable bind `-B host:/results` + `OSMOSE_RESULTS_DIR=/results`. JAVA_HOME=`/usr/lib/jvm/java-17-openjdk-amd64` (x86_64; the runner only needs `java` on PATH).
- Apptainer is absent on the dev box → the `.def`'s real validation is the **CI `apptainer-smoke` job** (`eWaterCycle/setup-apptainer` + `--fakeroot` build). `$NUMBA_CACHE_DIR` in the smoke must be inside `apptainer exec ... sh -c '...'` (single-quoted) so it expands in-container.

## Process note
brainstorm → spec (3-reviewer in-loop review: code-grounding + Apptainer/HPC-correctness + adversarial → found all 3 read-only landmines + the wrong JAR URL + the overstated SLURM job-array) → plan → subagent-driven TDD (5 tasks, each reviewed; 3 fix loops caught the wrong SLURM config paths, --jar-under-contain, the fragile `cp` under `set -eu`) → final whole-branch review MERGE-READY. Related: [[project-model-internal-reference-points]] (the batch this enables on HPC).
