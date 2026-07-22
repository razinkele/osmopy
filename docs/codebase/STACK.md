# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python for `osmose-python/`; R package plus Java engine for `osmose-master/` | `osmose-python/pyproject.toml`, `osmose-master/DESCRIPTION`, `osmose-master/pom.xml` |
| Runtime + version | Python `>=3.12`; R `>=3.5.0`; Java runtime required by both subprojects (`>=17` for Python README path, `>=8` for R package system requirement) | `osmose-python/pyproject.toml`, `osmose-python/README.md`, `osmose-master/DESCRIPTION`, `osmose-master/Dockerfile` |
| Package manager | `pip`/setuptools for Python app; R package metadata via `DESCRIPTION`/`NAMESPACE`; Maven for Java build | `osmose-python/pyproject.toml`, `osmose-master/DESCRIPTION`, `osmose-master/pom.xml` |
| Module/build system | Python package build from `pyproject.toml`; R package build/check workflow; Maven JAR + assembly build for Java | `osmose-python/pyproject.toml`, `osmose-master/.github/workflows/r-compile.yml`, `osmose-master/pom.xml` |

### 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| `shiny` | `>=1.3.0` | Python web UI framework | `osmose-python/pyproject.toml` |
| `shinyswatch` | `>=0.7.0` | Shiny theme support | `osmose-python/pyproject.toml` |
| `shinywidgets` | `>=0.3` | Python UI widgets | `osmose-python/pyproject.toml` |
| `plotly` | `>=5.18` | Interactive charts | `osmose-python/pyproject.toml` |
| `pandas` | `>=2.2` | Config/data table handling | `osmose-python/pyproject.toml` |
| `xarray` | `>=2024.1` | NetCDF/results handling | `osmose-python/pyproject.toml` |
| `netCDF4` | `>=1.6` | NetCDF I/O | `osmose-python/pyproject.toml` |
| `jinja2` | `>=3.1` | Config/template generation | `osmose-python/pyproject.toml` |
| `pymoo` | `>=0.6` | Calibration/optimization | `osmose-python/pyproject.toml` |
| `scikit-learn` | `>=1.4` | Surrogate modelling | `osmose-python/pyproject.toml` |
| `SALib` | `>=1.5` | Sensitivity analysis | `osmose-python/pyproject.toml` |
| `numpy` / `scipy` | `>=1.26` / `>=1.13` | Numerical engine work | `osmose-python/pyproject.toml` |
| `shiny_deckgl` | Git dependency at `v1.9.1` | Map rendering integration | `osmose-python/pyproject.toml` |
| `opencsv` | `2.4` | Java CSV parsing | `osmose-master/pom.xml` |
| `netcdfAll` | `5.5.2` | Java NetCDF support | `osmose-master/pom.xml` |
| `commons-math3` | `3.6.1` | Java math utilities | `osmose-master/pom.xml` |
| `ml.options` | `1.0.0` | Java CLI option parsing from local Maven repo | `osmose-master/pom.xml`, `osmose-master/java/local/README.md` |
| R package imports (`ncdf4`, `mgcv`, `calibrar`, `rmarkdown`, `fields`, `stringr`) | `[TODO] per-package versions not pinned in repo` | R-side config, plotting, calibration, and report generation | `osmose-master/DESCRIPTION` |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| Ruff | Python linting and formatting | `osmose-python/pyproject.toml`, `osmose-python/.pre-commit-config.yaml`, `osmose-python/.github/workflows/ci.yml` |
| Pyright | Python type checking | `osmose-python/pyrightconfig.json`, `osmose-python/.github/workflows/ci.yml` |
| Pytest / pytest-cov / pytest-asyncio | Python test runner, coverage, async tests | `osmose-python/pyproject.toml`, `osmose-python/.github/workflows/ci.yml` |
| Playwright-marked pytest tests | Python browser/E2E tests (excluded by default) | `osmose-python/pyproject.toml`, `osmose-python/tests/test_e2e_grid_maps.py` |
| GitHub Actions | Per-subproject CI | `osmose-python/.github/workflows/ci.yml`, `osmose-master/.github/workflows/java-compile.yml`, `osmose-master/.github/workflows/r-compile.yml` |
| Maven | Java compilation, JAR assembly, Javadoc, PlantUML generation | `osmose-master/pom.xml` |
| JUnit 5 | Java tests | `osmose-master/pom.xml`, `osmose-master/java/src/test/java/fr/ird/osmose/TestAccess.java` |
| R CMD build/check | R package build and CRAN-style checks in workflow | `osmose-master/.github/workflows/r-compile.yml` |

### 4) Key Commands

```bash
cd osmose-python
pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check osmose/ ui/ tests/
.venv/bin/pyright

cd osmose-master
mvn package
R CMD build --no-build-vignettes .
R CMD check --as-cran --ignore-vignettes <tarball>
```

### 5) Environment and Config

- Config sources: `osmose-python/pyproject.toml`, `osmose-python/app.py`, `osmose-python/osmose/config/reader.py`, `osmose-master/DESCRIPTION`, `osmose-master/pom.xml`
- Required env vars: `CMEMS_USERNAME`, `CMEMS_PASSWORD` for the Copernicus MCP helper; runtime app env vars for the main Python UI are `[TODO]` because no `.env.example`-style template exists in the repo.
- Deployment/runtime constraints: the Python Docker image bundles Java into a `python:3.12-slim` image and exposes port `8000`; the R package expects Java on `PATH`; the Java build depends on a local Maven repository entry and remote UCAR repositories.

### 6) Evidence

- `osmose-python/pyproject.toml`
- `osmose-python/Dockerfile`
- `osmose-python/.github/workflows/ci.yml`
- `osmose-master/DESCRIPTION`
- `osmose-master/pom.xml`
- `osmose-master/.github/workflows/r-compile.yml`
- `osmose-master/.github/workflows/java-compile.yml`

