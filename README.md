# OSMOSE Python

Python orchestration layer, simulation engine, and Shiny web interface for the [OSMOSE](https://osmose-model.org/) marine ecosystem simulator. Ships with a pure-Python engine (full Java parity) and a subprocess driver for the original Java engine.

<sub>**Python 3.12** · **NumPy + Numba** · **Shiny for Python** · **2510 tests** · **ruff clean** · **MIT**</sub>

## Status

| Surface | State |
|---|---|
| Python engine | Full Java parity on Bay of Biscay (8/8) and Eastern English Channel (14/14), within 1 order of magnitude. Faster than Java on every benchmarked config. |
| Java engine | Wrapped as async subprocess runner. OSMOSE 4.3.3 JAR supported. |
| Shiny UI | 10-tab end-to-end UI (Setup · Grid · Forcing · Fishing · Movement · Run · Results · Calibration · Scenarios · Advanced). |
| Calibration | pymoo NSGA-II, GP surrogate, SALib Morris/Sobol sensitivity; preflight stage + Pareto `find_optimum`. |
| Examples | Bay of Biscay (8 sp), Eastern English Channel (14 sp), **Baltic Sea (8 sp + 6 LTL)** with ICES SAG cross-validation. |
| Tests | 2510 passed, 15 skipped, 41 deselected. Pyright clean on `osmose/` and `ui/`. |
| Config validation | Opt-in typo catcher at `EngineConfig.from_dict` load time. Silent by default; `validation.strict.enabled=warn` logs `difflib` suggestions, `=error` raises with the full unknown-key list. |

## Contents

- [Quick start](#quick-start)
- [Simulation engines](#simulation-engines)
- [Performance](#performance)
- [Engine parity validation](#engine-parity-validation)
- [Examples](#examples)
  - [Baltic Sea](#baltic-sea)
- [Project layout](#project-layout)
- [Testing and linting](#testing-and-linting)
- [API sketch](#api-sketch)
- [MCP servers and credentials](#mcp-servers-and-credentials)
- [Docker](#docker)
- [Documentation index](#documentation-index)
- [License](#license)

## Quick start

```bash
git clone https://github.com/razinkele/osmopy.git
cd osmopy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The UI runs without Java — it defaults to the Python engine.

Requirements: **Python 3.12+**. Java 17+ only if you want to use the Java engine backend; the Python engine has no Java dependency.

## Simulation engines

Both engines implement the same `Engine` protocol and produce identical output shapes (biomass, abundance, mortality-by-cause, diet, trophic level, size/age distributions). The UI flips between them with one click.

|  | Python engine | Java engine |
|---|---|---|
| Implementation | Pure Python (NumPy + Numba JIT) | Java subprocess (OSMOSE 4.3.3 JAR) |
| Dependencies | Python packages only | Java 17+ and an OSMOSE JAR |
| Parity | Bay of Biscay 8/8, EEC 14/14 within 1 OoM | Reference implementation |
| Speed | 1.2×–5.7× faster than Java (below) | Production-ready |
| Use case | Production runs, calibration, development | Legacy compatibility, reference |

## Performance

Numba JIT + parallel cell processing + vectorized diet aggregation put the Python engine ahead of Java on every benchmarked config:

| Configuration | Python | Java | Speedup |
|---|---|---|---|
| Bay of Biscay, 1 yr (8 species) | **0.24 s** | 0.80 s | 3.3× |
| Bay of Biscay, 5 yr (8 species) | **1.99 s** | 2.3 s | 1.2× |
| Eastern English Channel, 1 yr (14 species) | **0.44 s** | 2.5 s | 5.7× |
| Eastern English Channel, 5 yr (14 species) | **5.2 s** | 7.2 s | 1.4× |

Benchmarked on Linux x86_64 with Numba 0.60, NumPy 1.26, Python 3.12. First run includes ~20 s Numba compilation overhead (cached for subsequent runs).

Key optimisations:

- Numba JIT mortality loop — full interleaved predation/starvation/fishing compiled to native code
- `prange` parallel cell processing with per-cell deterministic seeding
- Vectorised rate computation and diet aggregation (`np.add.at` per prey-species rollup)
- Pre-allocated diet buffers with capacity-based reuse
- Precomputed species masks; compiled movement (map rejection sampling + random walk)
- Results caching across repeated reads (CSV + NetCDF)

Run the suite yourself:

```bash
.venv/bin/python scripts/benchmark_engine.py --years 5 --seed 42 --repeats 3
```

## Engine parity validation

EEC 1-year biomass parity (Python vs Java, tonnes):

| Species | Python | Java | Ratio |
|---|---|---|---|
| lesserSpottedDogfish | 93 | 78 | 1.18 |
| redMullet | 95 | 121 | 0.79 |
| pouting | 336 738 | 314 956 | 1.07 |
| whiting | 101 094 | 86 767 | 1.17 |
| poorCod | 114 268 | 109 860 | 1.04 |
| cod | 1 441 | 1 466 | 0.98 |
| dragonet | 104 756 | 107 661 | 0.97 |
| sole | 43 117 | 53 184 | 0.81 |
| plaice | 15 040 | 18 483 | 0.81 |
| horseMackerel | 157 119 | 163 585 | 0.96 |
| mackerel | 13 982 | 17 326 | 0.81 |
| herring | 32 752 826 | 30 541 764 | 1.07 |
| sardine | 14 968 | 14 126 | 1.06 |
| squids | 131 280 | 119 896 | 1.09 |

All 14 species within 1 order of magnitude of Java. Re-run the validation:

```bash
.venv/bin/python scripts/validate_engines.py --years 5
```

Bay of Biscay (8 species) passes on all 8 with the same tolerance. See `docs/parity-roadmap.md` for the 7-phase closure history of the final parity gaps.

## Examples

Three ready-to-run configurations ship in-tree.

| Location | Species | Cells | Notes |
|---|---|---|---|
| `data/examples/` | 8 | Bay of Biscay | Reference parity config |
| `data/eec_full/` | 14 | Eastern English Channel | Full 14-species parity config |
| `data/baltic/` | 8 + 6 LTL | 612 (50 × 40) | Baltic calibration sandbox + ICES cross-validation |

### Baltic Sea

The Baltic example is the newest and most documented. It covers cod, herring, sprat, flounder, perch, pikeperch, smelt, stickleback plus six CMEMS-forced LTL groups, 50 × 40 cells over 10–30° E × 54–66° N, 24 dt/yr.

- **Full provenance** — every parameter family, every value source, every DOI: [`docs/baltic_example.md`](docs/baltic_example.md).
- **Validation vs ICES** — F rates and biomass envelopes cross-checked against the 2024 advice cycle: [`docs/baltic_ices_validation_2026-04-18.md`](docs/baltic_ices_validation_2026-04-18.md). Unit-aware (the ICES API mislabels some stocks' SSB units; the validator detects this via Blim magnitude).
- **Refresh workflow** — [`data/baltic/reference/ices_snapshots/README.md`](data/baltic/reference/ices_snapshots/README.md).
- **Calibration driver** — `scripts/calibrate_baltic.py` (phases: larval mortality, adult mortality + F) runs the Python engine directly, no Java needed.
- **Forcing** — `scripts/rebuild_baltic_mask.py` and `mcp_servers/copernicus/server.py` regenerate the mask + LTL NetCDF from CMEMS data.

```bash
.venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report
.venv/bin/python -m pytest tests/test_baltic_ices_validation.py -v
```

## Project layout

```
osmose/                     Core library (usable without Shiny)
  engine/                   Python simulation engine (44 files, ~11.5k LOC)
    simulate.py             Main simulation loop; SimulationContext, frozen StepOutput
    processes/              Growth, predation, mortality, reproduction, movement, fishing
    config.py               Typed parameter extraction
    grid.py                 Spatial grid with NetCDF loading
    resources.py            LTL plankton/resource forcing
    output.py               CSV + NetCDF output writer
    path_resolution.py      Consolidated resolver with traversal protection
  schema/                   Parameter definitions + registry (153 params)
  config/                   Config reader/writer (auto-detected separators; recursive includes)
  calibration/              pymoo NSGA-II + GP surrogate + SALib sensitivity + preflight
  runner.py                 Async Java subprocess manager
  results.py                CSV/NetCDF output reader (xarray)
  scenarios.py              Save/load/compare/fork
ui/                         Shiny web interface
  pages/                    One module per tab
  components/               Reusable widgets (param form)
  theme.py                  shinyswatch superhero theme
mcp_servers/                MCP servers
  copernicus/               CMEMS forcing downloader (env-based credentials)
data/
  examples/                 Bay of Biscay reference config (8 species)
  eec_full/                 Eastern English Channel config (14 species)
  baltic/                   Baltic Sea calibration sandbox (8 sp + 6 LTL); see docs/baltic_example.md
    reference/              Biomass targets + frozen ICES SAG snapshots
scripts/                    One-shot tools: benchmarks, validators, calibration, mask rebuild
tests/                      2510 tests (schema, config, engine processes, parity, calibration,
                            UI state, MCP hygiene, Baltic ICES cross-validation)
docs/
  baltic_example.md                           Baltic example full provenance
  baltic_ices_validation_2026-04-18.md        ICES cross-validation report (2024 advice)
  parity-roadmap.md                           Engine parity roadmap (7 phases)
  osmose-master-java-fixes.patch              Portable patch for upstream osmose-master
  plans/                                      Historical and active implementation plans
  superpowers/plans/                          Superpowers-skill-generated plans
```

## Testing and linting

```bash
.venv/bin/python -m pytest                   # run all tests (~50 s)
.venv/bin/python -m pytest --cov=osmose      # with coverage
.venv/bin/python -m pytest -v -k test_name   # specific test
.venv/bin/ruff check osmose/ ui/ tests/      # lint
.venv/bin/ruff format osmose/ ui/ tests/     # format
```

2510 tests across schema, config I/O, config-key validation, every engine process, parity regressions, numerical edge cases, type invariants, thread safety, UI state, calibration, scenario management, MCP credential hygiene, and integration scenarios. Pyright passes with zero errors on the `osmose/` and `ui/` trees.

## API sketch

### Engine protocol

```python
from osmose.engine import PythonEngine, JavaEngine, Engine

# Python engine — no Java needed
engine = PythonEngine()
result = engine.run(config=config_dict, output_dir=Path("output"), seed=42)

# Java engine — requires OSMOSE JAR
from osmose.runner import OsmoseRunner
runner = OsmoseRunner(jar_path=Path("osmose-java/osmose.jar"))
result = await runner.run(config_path=Path("config.csv"), output_dir=Path("output"))
```

### Config I/O

```python
from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter

reader = OsmoseConfigReader()
config = reader.read("path/to/osm_all-parameters.csv")

writer = OsmoseConfigWriter()
writer.write(config, "path/to/output.csv")
```

### Config-key validation

`EngineConfig.from_dict()` checks every key against an allowlist built from the `ParameterRegistry` plus an AST walk of `osmose/engine/config.py`. The allowlist knows all ~390 keys the engine or reader touches, including `{idx}`-patterned families (`species.linf.sp{idx}`, `movement.file.map{idx}`, …). Default is silent; opt in with `validation.strict.enabled`:

| Mode | Behavior on unknowns |
|---|---|
| `off` (default) | Single INFO-level nudge (`"Config has N unknown keys; set validation.strict.enabled=warn for details."`) — zero output on clean configs. |
| `warn` | One WARNING per unknown, with a `difflib` suggestion when the ratio passes 0.85. E.g. `species.liinf.sp0` → `"did you mean 'species.linf.sp{idx}'?"` |
| `error` | Collect **all** unknowns, then raise a single `ValueError` listing them (not fail-fast). |

```python
cfg = reader.read("path/to/osm_all-parameters.csv")
cfg["validation.strict.enabled"] = "warn"  # or "error"
EngineConfig.from_dict(cfg)  # emits per-key warnings
```

### Results reader

```python
from osmose.results import OsmoseResults

# strict=True (default): FileNotFoundError if outputs are missing
results = OsmoseResults("path/to/output/")
biomass = results.get_biomass()         # xarray Dataset
mortality = results.get_mortality()     # per-species mortality by cause
diet = results.get_diet()               # diet composition matrix

# strict=False for partial/speculative loads (calibration, UI)
with OsmoseResults("path/to/output/", strict=False) as results:
    biomass = results.get_biomass()
```

### Thread safety and parallel calibration

The Python engine is re-entrant. Per-simulation state is encapsulated in a `SimulationContext` dataclass passed through the call chain — no module-level globals. Safe for `ProcessPoolExecutor`-based calibration runs.

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

## MCP servers and credentials

MCP integrations ship under `mcp_servers/`. The Copernicus Marine server reads credentials from environment variables only — **no hardcoded fallbacks**. Populate `.env` (gitignored) at the repo root:

```bash
cp .env.example .env
$EDITOR .env   # fill in CMEMS_USERNAME / CMEMS_PASSWORD
```

`server.py` auto-loads `.env` via `python-dotenv`. Two enforcement tests guard against regressions:

- `tests/test_copernicus_mcp_env.py` — server source must not contain a hardcoded credential default; `_require_creds()` must raise `RuntimeError` on missing env.
- `tests/test_mcp_config_hygiene.py` — `.mcp.json` must not ship a CMEMS password literal under any server's `env` block.

The ICES data-access MCP server lives out-of-tree at `/home/razinka/ices-mcp-server/` (stdio-based, `uv` runner). Wire it via `.mcp.json` — see the Baltic ICES snapshots README for the refresh workflow.

## Docker

```bash
docker build -t osmose-python .
docker run -p 8000:8000 osmose-python
```

Place `osmose.jar` in `osmose-java/` before building if you need Java-engine support. The Python engine works without it.

## Documentation index

Start here depending on what you want:

| Goal | Doc |
|---|---|
| Run an existing example | [Quick start](#quick-start) above |
| Understand the Baltic example, its parameters, and their sources | [`docs/baltic_example.md`](docs/baltic_example.md) |
| See where model F and biomass sit vs ICES 2024 advice | [`docs/baltic_ices_validation_2026-04-18.md`](docs/baltic_ices_validation_2026-04-18.md) |
| Refresh ICES snapshots when a new advice year lands | [`data/baltic/reference/ices_snapshots/README.md`](data/baltic/reference/ices_snapshots/README.md) |
| Port progress vs Java (what was fixed, what's next) | [`docs/parity-roadmap.md`](docs/parity-roadmap.md) |
| Per-release change history | [`CHANGELOG.md`](CHANGELOG.md) |
| Build a Java-side fix patch | [`docs/osmose-master-java-fixes.patch`](docs/osmose-master-java-fixes.patch) |
| Implementation plans (historical and active) | [`docs/plans/`](docs/plans/) and [`docs/superpowers/plans/`](docs/superpowers/plans/) |
| Project conventions for contributors (tooling rules, gotchas) | [`CLAUDE.md`](CLAUDE.md) |

## License

[MIT](LICENSE)
