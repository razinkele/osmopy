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
