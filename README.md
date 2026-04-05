# OSMOSE Python Interface

Python orchestration layer, simulation engine, and Shiny web interface for the [OSMOSE](https://osmose-model.org/) marine ecosystem simulator. Provides both a pure-Python engine (with full Java parity) and the traditional Java engine as simulation backends.

## Features

- **Dual simulation engines** — run simulations via the built-in Python engine or the original Java engine (selectable in the UI)
- **Schema-driven parameter system** — 181 parameters defined once, UI auto-generated from metadata
- **Config I/O** — read/write OSMOSE's native `.csv`/`.properties` format with auto-detected separators and recursive sub-file loading
- **Python engine** — vectorized NumPy/Numba implementation with full process-level Java parity (growth, predation, reproduction, mortality, movement, fishing, starvation, bioenergetics)
- **Java engine** — async subprocess runner with real-time progress streaming
- **Results reader** — CSV and NetCDF output parsing via xarray (biomass, diet, spatial maps, mortality)
- **Calibration** — multi-objective optimization (pymoo NSGA-II), GP surrogate model, Sobol sensitivity analysis
- **Scenario management** — save, load, compare, and fork named configurations
- **10-tab Shiny UI** — Setup, Grid, Forcing, Fishing, Movement, Run, Results, Calibration, Scenarios, Advanced

## Simulation Engines

OSMOSE Python provides two interchangeable simulation backends via a common `Engine` protocol:

| | Python Engine | Java Engine |
|---|---|---|
| **Implementation** | Pure Python (NumPy + Numba JIT) | Java subprocess (OSMOSE 4.3.3 JAR) |
| **Parity** | Bay of Biscay 8/8, EEC 14/14 | Reference implementation |
| **Speed** | **Faster than Java** (see benchmarks below) | Production-ready |
| **Dependencies** | None beyond Python packages | Java 17+ and OSMOSE JAR |
| **Use case** | Production runs, calibration, development | Legacy compatibility |

Both engines produce identical output formats (CSV biomass, abundance, mortality, diet, trophic level, size/age distributions).

### Performance Benchmarks

The Python engine uses Numba JIT compilation and parallel execution to achieve performance faster than the Java reference implementation:

| Configuration | Python | Java | Speedup |
|---------------|--------|------|---------|
| Bay of Biscay 1yr (8 species) | **0.24s** | 0.80s | Python 3.3x faster |
| Bay of Biscay 5yr (8 species) | **1.99s** | 2.3s | Python 1.2x faster |
| English Channel 1yr (14 species) | **0.44s** | 2.5s | Python 5.7x faster |
| English Channel 5yr (14 species) | **5.2s** | 7.2s | Python 1.4x faster |

Benchmarked on Linux x86_64 with Numba 0.60, NumPy 1.26, Python 3.12. First run includes ~20s Numba compilation overhead (cached for subsequent runs).

Key optimizations:
- **Numba JIT mortality loop** — full interleaved predation/starvation/fishing compiled to native code with shared `_apply_single_cause` helper
- **Parallel cell processing** — `prange` over grid cells with per-cell deterministic seeding
- **Compiled movement** — map-based rejection sampling and random walk in Numba
- **Vectorized rate computation** — species-indexed NumPy operations replace per-school Python loops
- **Precomputed species masks** — bioenergetic step computes masks once instead of per-loop
- **Vectorized fishing** — spatial map and MPA lookups use per-species array indexing

Run benchmarks yourself:
```bash
.venv/bin/python scripts/benchmark_engine.py --years 5 --seed 42 --repeats 3
```

### Python Engine Validation

The Python engine has been validated against the Java reference on two real ecosystem configurations:

- **Bay of Biscay** (8 focal species + 6 resources): **8/8 species within 1 order of magnitude**
- **Eastern English Channel** (14 focal species, 1-year): **14/14 species within 1 order of magnitude**

EEC 1-year biomass parity (Python vs Java, tonnes):

| Species | Python | Java | Ratio |
|---------|--------|------|-------|
| lesserSpottedDogfish | 93 | 78 | 1.18 |
| redMullet | 95 | 121 | 0.79 |
| pouting | 336,738 | 314,956 | 1.07 |
| whiting | 101,094 | 86,767 | 1.17 |
| poorCod | 114,268 | 109,860 | 1.04 |
| cod | 1,441 | 1,466 | 0.98 |
| dragonet | 104,756 | 107,661 | 0.97 |
| sole | 43,117 | 53,184 | 0.81 |
| plaice | 15,040 | 18,483 | 0.81 |
| horseMackerel | 157,119 | 163,585 | 0.96 |
| mackerel | 13,982 | 17,326 | 0.81 |
| herring | 32,752,826 | 30,541,764 | 1.07 |
| sardine | 14,968 | 14,126 | 1.06 |
| squids | 131,280 | 119,896 | 1.09 |

Run the validation yourself:
```bash
.venv/bin/python scripts/validate_engines.py --years 5
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/razinkele/osmopy.git
cd osmopy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the app
shiny run app.py --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Requirements

- Python 3.12+
- Java 17+ (optional — only required for Java engine backend)

## Docker

```bash
docker build -t osmose-python .
docker run -p 8000:8000 osmose-python
```

Place `osmose.jar` in `osmose-java/` before building (required for Java engine).

## Project Structure

```
osmose/                  Core library (usable without Shiny)
  engine/                Python simulation engine (29 files, ~9000 LOC)
    config.py            Typed parameter extraction from flat config
    simulate.py          Main simulation loop (SimulationContext, frozen StepOutput)
    processes/           Growth, predation, mortality, reproduction, movement, fishing
    path_resolution.py   Consolidated file path resolver with traversal protection
    grid.py              Spatial grid with NetCDF loading
    resources.py         LTL plankton/resource forcing
    output.py            CSV/NetCDF output writer
  schema/                Parameter definitions + registry (181 params)
  config/                Config reader/writer
  calibration/           pymoo, GP surrogate, SALib sensitivity
  runner.py              Async Java subprocess manager
  results.py             CSV/NetCDF output reader
  scenarios.py           Save/load/compare/fork
ui/                      Shiny web interface
  pages/                 One module per tab
  components/            Reusable widgets (param form)
  theme.py               Shinyswatch superhero theme
data/
  examples/              Bay of Biscay example config (8 species)
  eec_full/              Eastern English Channel config (14 species)
tests/                   1864 tests
docs/
  parity-roadmap.md      Engine parity roadmap (7 phases, 37 items)
```

## Testing

```bash
.venv/bin/python -m pytest                      # run all tests
.venv/bin/python -m pytest --cov=osmose         # with coverage
.venv/bin/python -m pytest -v -k test_name      # specific test
.venv/bin/ruff check osmose/ ui/ tests/          # lint
.venv/bin/ruff format osmose/ ui/ tests/         # format
```

1864 tests covering schema, config I/O, all engine processes, performance parity, numerical edge cases, type invariants, thread safety, UI state, and integration scenarios.

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Shiny for Python + shinyswatch |
| Visualization | plotly |
| Simulation | NumPy, Numba (JIT mortality, parallel cells, compiled movement) |
| NetCDF | xarray, netCDF4 |
| Calibration | pymoo (NSGA-II), scikit-learn (GP) |
| Sensitivity | SALib |
| Config | pandas, jinja2 |
| Deployment | Docker (eclipse-temurin JRE + Python 3.12) |

## API Reference

### Engine Protocol

```python
from osmose.engine import PythonEngine, JavaEngine, Engine

# Python engine (no Java required)
engine = PythonEngine()
result = engine.run(config=config_dict, output_dir=Path("output"), seed=42)

# Java engine (requires OSMOSE JAR)
from osmose.runner import OsmoseRunner
runner = OsmoseRunner(jar_path=Path("osmose-java/osmose.jar"))
result = await runner.run(config_path=Path("config.csv"), output_dir=Path("output"))
```

### Config I/O

```python
from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter

# Read OSMOSE configuration
reader = OsmoseConfigReader()
config = reader.read("path/to/osm_all-parameters.csv")

# Write modified configuration
writer = OsmoseConfigWriter()
writer.write(config, "path/to/output.csv")
```

### Results Reader

```python
from osmose.results import OsmoseResults

# strict=True (default): raises FileNotFoundError if output files are missing
results = OsmoseResults("path/to/output/")
biomass = results.get_biomass()           # xarray Dataset
mortality = results.get_mortality()       # per-species mortality by cause
diet = results.get_diet()                 # diet composition matrix

# Use strict=False for partial/speculative loads (calibration, UI)
with OsmoseResults("path/to/output/", strict=False) as results:
    biomass = results.get_biomass()
```

### Thread Safety

The Python engine is re-entrant and thread-safe. Per-simulation state (diet matrix, trophic level tracking, config directory) is encapsulated in a `SimulationContext` dataclass passed through the call chain — no module-level globals. This makes parallel calibration safe:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(engine.run, config=cfg, seed=i) for i, cfg in enumerate(configs)]
```

### Calibration

```python
from osmose.calibration import CalibratorConfig, Calibrator

cal_config = CalibratorConfig(
    parameters={"species.k.sp0": (0.1, 0.5)},
    objectives=["biomass_rmse"],
    n_gen=50,
    pop_size=20,
)
calibrator = Calibrator(cal_config, base_config, jar_path)
result = calibrator.run()
```

## License

[MIT](LICENSE)
