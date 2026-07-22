# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: layered monorepo with two executable surfaces: a Python application/library stack and a legacy R package that shells out to a Java engine.
- Why this classification: `osmose-python/app.py` wires UI pages onto a shared `AppState`; `osmose-python/osmose/engine/__init__.py` and `osmose-python/osmose/runner.py` expose two engine backends behind Python entrypoints; `osmose-master/R/osmose-main.R` delegates execution to the Java JAR; `osmose-master/java/src/main/java/fr/ird/osmose/Simulation.java` and `Configuration.java` own the simulation lifecycle and parameter loading.
- Primary constraints:
  - OSMOSE configuration is flat, file-based, and split recursively across CSV/properties-style files.
  - Both Python and R surfaces must stay compatible with Java-era config/output formats.
  - Large parts of the Python UI are schema-driven, so parameter metadata has to remain centralized.

### 2) System Flow

```text
config files -> Python UI / CLI or R wrapper -> config parsing and validation -> Python engine or Java JAR execution -> CSV/NetCDF outputs -> Python/R result readers and plots
```

1. `osmose-python/app.py` mounts the Shiny UI and shares state through `ui/state.py`.
2. Python config files are parsed by `osmose-python/osmose/config/reader.py`, which recursively follows `osmose.configuration.*` references and stores the config directory for later file resolution.
3. Python runtime chooses either the in-process `PythonEngine` (`osmose-python/osmose/engine/__init__.py`) or the Java subprocess adapter `OsmoseRunner` (`osmose-python/osmose/runner.py`).
4. The pure-Python engine runs the ordered simulation pipeline documented in `osmose-python/osmose/engine/simulate.py`; the Java path executes `java -jar ...` with validated options in `osmose-python/osmose/runner.py`.
5. On the legacy side, `osmose-master/R/osmose-main.R` builds arguments and calls `system2()` to launch the packaged Java engine.
6. Outputs are consumed by Python `osmose-python/osmose/results.py` or the R read helpers in `osmose-master/R/osmose-main.R` and `osmose-master/R/osmose2R.R`.

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| Python schema registry | Canonical parameter definitions and matching by concrete key | UI-specific rendering state | `osmose-python/osmose/schema/registry.py` |
| Python config layer | Recursive parsing, file reference checks, schema validation helpers | Simulation math | `osmose-python/osmose/config/reader.py`, `osmose-python/osmose/config/validator.py` |
| Python engine | Simulation state, step ordering, output aggregation | Shiny UI concerns | `osmose-python/osmose/engine/simulate.py`, `osmose-python/osmose/engine/__init__.py` |
| Python Java runner | Safe JVM command construction and async subprocess streaming | Domain-specific ecological calculations | `osmose-python/osmose/runner.py` |
| Python UI | Navigation, page wiring, reactive state, browser-facing affordances | Core config parsing and engine internals | `osmose-python/app.py`, `osmose-python/ui/state.py` |
| R wrapper | User-facing package API, Java process orchestration, plotting/reporting | Java simulation internals | `osmose-master/R/osmose-main.R`, `osmose-master/NAMESPACE` |
| Java core | Configuration loading, model lifecycle, process execution, NetCDF outputs | R package export logic | `osmose-master/java/src/main/java/fr/ird/osmose/Osmose.java`, `osmose-master/java/src/main/java/fr/ird/osmose/Simulation.java`, `osmose-master/java/src/main/java/fr/ird/osmose/Configuration.java` |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| Registry pattern | `osmose-python/osmose/schema/registry.py` | Centralize OSMOSE parameter definitions and validation metadata |
| Context object | `osmose-python/osmose/engine/simulate.py` (`SimulationContext`) | Carry mutable per-run state without module-level globals |
| Adapter/subprocess boundary | `osmose-python/osmose/runner.py`, `osmose-master/R/osmose-main.R` | Keep Java execution isolated from Python/R caller code |
| Layered page composition | `osmose-python/app.py` + `osmose-python/ui/pages/*` | Keep the large Shiny UI split by tab/page |
| S3 method-based R surface | `osmose-master/NAMESPACE`, `osmose-master/R/osmose-*.R` | Provide package-level polymorphism for plotting/reporting |
| Singleton entry object | `osmose-master/java/src/main/java/fr/ird/osmose/Osmose.java` | Centralize Java configuration/run lifecycle in one process object |

### 5) Known Architectural Risks

- Credentials for the Copernicus MCP helper are committed in both `.mcp.json` and default environment-variable fallbacks, which makes a development helper look like a production integration surface.
- The Python codebase is executable and well-tested, but type-check debt remains (`pyright` reports 22 errors) so the static contract is weaker than the runtime/test contract.
- The R/Java build path depends on Maven plus a local Maven repository entry; the current environment has Java and R installed but no Maven, which weakens reproducibility from a clean machine.
- Java tests are declared but skipped by default in `pom.xml`, so the default Maven package path does not exercise them.

### 6) Evidence

- `osmose-python/app.py`
- `osmose-python/osmose/schema/registry.py`
- `osmose-python/osmose/config/reader.py`
- `osmose-python/osmose/engine/simulate.py`
- `osmose-python/osmose/runner.py`
- `osmose-master/R/osmose-main.R`
- `osmose-master/java/src/main/java/fr/ird/osmose/Osmose.java`
- `osmose-master/java/src/main/java/fr/ird/osmose/Simulation.java`
- `osmose-master/java/src/main/java/fr/ird/osmose/Configuration.java`

