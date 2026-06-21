# OSMOSE Python — Usage Guide

Task-oriented recipes for working with OSMOSE Python **outside the web UI** — scripting
runs, reading outputs, comparing scenarios, and calibrating from Python or the command
line. New to the model? Start with the hands-on
[30-minute tutorial](tutorials/30-minute-ecosystem.md); for install see the
[README](https://github.com/razinkele/osmopy/blob/master/README.md#quick-start).

Every Python snippet below runs against the bundled 2-species smoke config
`data/minimal/osm_all-parameters.csv` (finishes in seconds). Swap in a real config for
actual work:

| Config | Master file | Species |
|---|---|---|
| Minimal (smoke) | `data/minimal/osm_all-parameters.csv` | 2 |
| Eastern English Channel | `data/eec_full/eec_all-parameters.csv` | 14 |
| Baltic Sea | `data/baltic/baltic_all-parameters.csv` | 8 + 6 LTL |

Run snippets from the repository root in the project venv (`.venv/bin/python`), where the
editable install makes `osmose` importable and the `scripts/` are on hand.

---

## 1. Run a simulation

### Python engine (recommended — no Java required)

A config is a plain `dict[str, str]` produced by the config reader. Hand it to
`PythonEngine`, which has two run modes:

```python
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

config = OsmoseConfigReader().read(Path("data/minimal/osm_all-parameters.csv"))
engine = PythonEngine()  # backend="numpy" by default

# (a) In memory — no files written; returns an OsmoseResults you can query directly.
results = engine.run_in_memory(config=config, seed=42)

# (b) To disk — writes the usual OSMOSE CSV/NetCDF tree; returns a RunResult.
run = engine.run(config=config, output_dir=Path("output"), seed=42)
print(run.status, run.returncode)   # -> ok 0
```

Reproducible ensembles over seeds:

```python
runs = engine.run_ensemble(config=config, output_dir=Path("output_ens"), n=5, base_seed=0)
```

### Command line

The package installs an `osmose` console command:

```bash
osmose validate data/minimal/osm_all-parameters.csv      # config typo / range / reference check
osmose report  output/                                   # text summary of an output directory
osmose run     data/minimal/osm_all-parameters.csv --jar path/to/osmose.jar --output output/
```

`osmose run` is the **Java** path and requires `--jar`. To run the Python engine
non-interactively, call `PythonEngine` from a short script (above) — there is no
`osmose run` for the Python engine.

### Java engine (requires an OSMOSE JAR)

`OsmoseRunner.run` drives the Java subprocess. It is a coroutine — `await` it (or wrap in
`asyncio.run`), and it takes a config **path**, not a dict:

```python
import asyncio
from pathlib import Path
from osmose.runner import OsmoseRunner

runner = OsmoseRunner(jar_path=Path("osmose-java/osmose.jar"))
run = asyncio.run(runner.run(
    config_path=Path("data/minimal/osm_all-parameters.csv"),
    output_dir=Path("output_java"),
))
```

See [§6](#choose-an-engine-reproduce-results) for when to pick which engine.

---

## 2. Read outputs

`OsmoseResults` reads an output directory (or comes back from `run_in_memory`). Point it at
the directory and match the filename `prefix` (`osm` for the minimal/EEC configs):

```python
from pathlib import Path
from osmose.results import OsmoseResults

results = OsmoseResults(Path("output"), prefix="osm")
print(results.list_outputs()[:5])     # what's available
```

The time-series accessors return **wide** DataFrames — a `Time` column, one column per
species, and a constant `species` label column:

```python
bio = results.biomass()
#        Time       Anchovy        Hake species
# 0  0.000000    1833.333333  500.000000     all
# ...
```

Common accessors (all return a `pandas.DataFrame`):

| Method | Content |
|---|---|
| `biomass()`, `abundance()` | standing stock over time |
| `yield_biomass()`, `yield_abundance()` | catch over time |
| `mortality()`, `mortality_rate()` | mortality (by cause) |
| `mean_size()`, `mean_trophic_level()` | community indicators |
| `biomass_by_age()`, `biomass_by_size()` | structured distributions |

**Selecting one species:** pick the column — the `species=` argument filters the
row-label column (which is `"all"` on the standard wide files) and will return an empty
frame if you pass a species name:

```python
bio = results.biomass()
anchovy = bio[["Time", "Anchovy"]]          # correct: select the column
# anchovy_mean = bio["Anchovy"].mean()
```

Spatial / NetCDF outputs (when `output.spatial.enabled=true`) load as `xarray.Dataset`:

```python
ds = results.read_netcdf("osm_spatial_biomass_Simu0.nc")   # or results.spatial_biomass(...)
```

---

## 3. Compare two runs

`run_delta` ranks the per-species change between a baseline and a variant. It takes two
**`OsmoseResults` objects** (not paths):

```python
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.analysis import run_delta, format_delta_report
from pathlib import Path

config = OsmoseConfigReader().read(Path("data/minimal/osm_all-parameters.csv"))
baseline = PythonEngine().run_in_memory(config=config, seed=1)
variant  = PythonEngine().run_in_memory(config=config, seed=2)

deltas = run_delta(baseline, variant, metric="biomass", window_years=5)
for d in deltas:
    print(d.species, d.baseline_mean, d.variant_mean, d.pct_delta)

print(format_delta_report(deltas, metric="biomass", window_years=5))   # markdown report
```

Each `SpeciesDelta` carries `species`, `baseline_mean`, `variant_mean`, `abs_delta`,
`pct_delta` (`None` for a zero-baseline species), and `from_zero`. `metric` accepts
`"biomass"`, `"yield"`, or `"abundance"`.

From the command line, against two output directories:

```bash
python scripts/compare_runs.py --baseline output_a/ --variant output_b/ \
    --metric biomass --window-years 10 --report delta.md
```

---

## 4. Calibrate

The Baltic calibration driver wraps the optimizers (Differential Evolution, CMA-ES,
GP-surrogate DE) behind one CLI. It is long-running and writes results to
`data/calibration_history/`:

```bash
python scripts/calibrate_baltic.py --optimizer de --phase 12 --maxiter 100 --seeds 3
```

Read a finished run back into Python:

```python
from osmose.calibration import list_runs, load_run

runs = list_runs()                       # list[dict] from data/calibration_history
if runs:
    data = load_run(Path(runs[-1]["path"]))
    print(data["results"]["best_objective"])
```

The Calibration tab in the web UI shows a **live dashboard** for an in-flight run
(optimizer/phase/generation, a per-species ICES proxy table, convergence). For a custom
calibration problem, `osmose.calibration.OsmoseCalibrationProblem` is the programmatic
entry point (set `use_java_engine=True` for bit-exact Java evaluation — see §6).

---

## 5. Post-run diagnostics (CLI)

These read an existing output directory and emit a report / JSON / plot:

| Script | Purpose | Key args |
|---|---|---|
| `scripts/validate_outputs_vs_ices.py` | model biomass vs ICES SAG envelopes (in-range / overshoot) | `--results-dir`, `--ices-window YYYY-YYYY` |
| `scripts/compute_size_spectrum.py` | community size spectrum (slope, LFI, mean size) | `--results-dir`, `--lfi-threshold-cm` |
| `scripts/compute_mortality_balance.py` | fishing-vs-natural mortality (F/M) per species | `--results-dir`, `--species` |
| `scripts/check_config.py` | structured config parse diagnostics | `--config` |

```bash
python scripts/validate_outputs_vs_ices.py --results-dir output/ --report ices.md
python scripts/compute_size_spectrum.py    --results-dir output/ --plot spectrum.png
```

Pass `--help` to any script for its full argument list.

---

(choose-an-engine-reproduce-results)=
## 6. Choose an engine & reproduce results

| | Python engine | Java engine |
|---|---|---|
| Dependency | Python only | Java 17+ and an OSMOSE JAR |
| Entry point | `PythonEngine` | `OsmoseRunner` (async) |
| Use it for | everyday runs, calibration, the UI | bit-exact parity against upstream Java |

**Reproducibility.** Setting `simulation.rng.fixed=true` makes a single config + seed
reproducible **across Python-engine runs**. It does *not* make the Python engine
bit-equal to the Java engine: NumPy uses PCG64, the Java engine uses MT19937
(`java.util.Random`), and the streams diverge at the first draw. Cross-engine numerical
equivalence is "within 1 order of magnitude" per the parity suite (14/14 EEC, 8/8 Bay of
Biscay). If you need byte-exact reproducibility against Java, run the Java path — for
calibration, `OsmoseCalibrationProblem(use_java_engine=True)`. (Full caveat in the
`osmose/engine/rng.py` module docstring.)

---

## Where to go next

| Goal | Doc |
|---|---|
| Hands-on: run and perturb a Baltic ecosystem in 30 min | [`tutorials/30-minute-ecosystem.md`](tutorials/30-minute-ecosystem.md) |
| The Baltic example config and where its parameters come from | [`baltic_example.md`](https://github.com/razinkele/osmopy/blob/master/docs/baltic_example.md) |
| Model F/biomass vs ICES advice | [`baltic_ices_validation_2026-04-18.md`](https://github.com/razinkele/osmopy/blob/master/docs/baltic_ices_validation_2026-04-18.md) |
| Python-vs-Java port status | [`parity-roadmap.md`](https://github.com/razinkele/osmopy/blob/master/docs/parity-roadmap.md) |
| Short API reference (engine, config I/O, results, calibration) | [README → API sketch](https://github.com/razinkele/osmopy/blob/master/README.md#api-sketch) |
| Per-release change history | [`../CHANGELOG.md`](https://github.com/razinkele/osmopy/blob/master/CHANGELOG.md) |
