# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `osmose-python/` | Python orchestration layer, pure-Python engine, Shiny UI, tests, Docker packaging | `osmose-python/README.md`, `osmose-python/pyproject.toml`, `osmose-python/app.py` |
| `osmose-master/` | Legacy R package and Java reference engine | `osmose-master/README.md`, `osmose-master/DESCRIPTION`, `osmose-master/pom.xml` |
| `docs/` | Design notes and planning docs outside the shipped subprojects | `docs/plans/2026-02-21-osmose-python-port-design.md`, `docs/plans/2026-02-21-osmose-python-port-plan.md` |
| `.github/skills/` | Local Copilot skill definitions used during development/review | `.github/skills/acquire-codebase-knowledge/SKILL.md` |
| `docs/codebase/` | Generated codebase knowledge docs from this review | `docs/codebase/.codebase-scan.txt` |

### 2) Entry Points

- Main runtime entry: `osmose-python/app.py` for the Python UI; `osmose-master/R/osmose-main.R` (`run_osmose`) for the R package surface; `osmose-master/java/src/main/java/fr/ird/osmose/Osmose.java` for the Java executable entrypoint.
- Secondary entry points (worker/cli/jobs): `osmose-python/osmose/cli.py`; Python scripts under `osmose-python/scripts/`; Java tests under `osmose-master/java/src/test/java/fr/ird/osmose/`.
- How entry is selected (script/config): Python installs the `osmose` console script from `pyproject.toml`; the Docker image runs `shiny run app.py`; the R layer shells out to the packaged Java JAR with `system2()`; Maven assembles the Java JAR under `inst/java/`.

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `osmose-python/osmose/` | Python library code for config I/O, engines, calibration, results, scenarios | Page layout wiring that belongs in `ui/` |
| `osmose-python/osmose/engine/` | Simulation state, process pipeline, output writing, Java bridge helpers | Shiny page state and presentation logic |
| `osmose-python/ui/` | Shiny pages, reactive state, reusable UI components and theme hooks | Core numeric simulation logic |
| `osmose-python/tests/` | Pytest suites, fixtures, baseline data, E2E tests | Production runtime code |
| `osmose-master/R/` | R package API, configuration readers, plot/report helpers, Java invocation | Java simulation internals |
| `osmose-master/java/src/main/java/fr/ird/osmose/` | Java model core, processes, outputs, utilities | R package wrappers |
| `osmose-master/java/src/test/java/` | JUnit-based Java tests against fixture configurations | Shipped production logic |

### 4) Naming and Organization Rules

- File naming pattern: Python files are mostly `snake_case.py`; R files use `osmose-*.R` and `osmose_init-*.R`; Java classes are `PascalCase.java`.
- Directory organization pattern: both subprojects are layered/domain-grouped rather than feature-packaged at the repo root.
- Import aliasing or path conventions: Python uses absolute package imports such as `from osmose...` and `from ui...`; Java uses `fr.ird.osmose.*` packages; R exports are declared in `NAMESPACE`.

### 5) Evidence

- `osmose-python/app.py`
- `osmose-python/pyproject.toml`
- `osmose-python/osmose/`
- `osmose-python/ui/`
- `osmose-python/tests/`
- `osmose-master/R/osmose-main.R`
- `osmose-master/java/src/main/java/fr/ird/osmose/Osmose.java`
- `osmose-master/pom.xml`

