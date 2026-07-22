# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | Python uses `snake_case.py`; R uses `osmose-*.R` / `osmose_init-*.R`; Java uses `PascalCase.java` | `osmose-python/osmose/config/reader.py`, `osmose-master/R/osmose-readConfiguration.R`, `osmose-master/java/src/main/java/fr/ird/osmose/Simulation.java` | same |
| Functions/methods | Python and most R entrypoints are `snake_case`; Java methods are `camelCase` | `setup_logging`, `run_osmose`, `readConfiguration` | `osmose-python/osmose/logging.py`, `osmose-master/R/osmose-main.R`, `osmose-master/java/src/main/java/fr/ird/osmose/Osmose.java` |
| Types/interfaces | Python and Java types use `PascalCase`; Java interfaces are prefixed with `I` in places | `SimulationContext`, `ParameterRegistry`, `Configuration`, `IAggregation` | `osmose-python/osmose/engine/simulate.py`, `osmose-python/osmose/schema/registry.py`, `osmose-master/java/src/main/java/fr/ird/osmose/Configuration.java`, `osmose-master/java/src/main/java/fr/ird/osmose/IAggregation.java` |
| Constants/env vars | Uppercase with underscores | `CMEMS_USERNAME`, `CMEMS_PASSWORD`, `JAVA_HOME` | `osmose-python/.mcp.json`, `osmose-python/mcp_servers/copernicus/server.py`, `osmose-python/Dockerfile` |

### 2) Formatting and Linting

- Formatter: Ruff formatter via `ruff format`; Java/R formatter config is `[TODO]` because no repo-local formatter config was found for those subprojects.
- Linter: Ruff for Python; Pyright for Python type checking; Java/R static analysis tooling is `[TODO]` because no Checkstyle/SpotBugs/lintr config was found in-repo.
- Most relevant enforced rules: Python target version `py312`, line length `100`, `pytest` excludes `e2e` marker by default, Pyright uses `typeCheckingMode = "basic"`.
- Run commands: `.venv/bin/ruff check osmose/ ui/ tests/`, `.venv/bin/ruff format osmose/ ui/ tests/`, `.venv/bin/pyright`, `pytest --cov=osmose --cov-fail-under=90` in CI.

### 3) Import and Module Conventions

- Import grouping/order: Python relies on regular absolute imports from `osmose` and `ui`; no additional import-sorting config was found beyond Ruff usage.
- Alias vs relative import policy: Python prefers package-absolute imports (`from osmose...`, `from ui...`); Java uses `fr.ird.osmose.*`; R exports/imports are mediated through `NAMESPACE`.
- Public exports/barrel policy: Python exposes a console script and package modules directly; R package exports are explicitly enumerated in `NAMESPACE`.

### 4) Error and Logging Conventions

- Error strategy by layer: Python parsing/runner code raises explicit exceptions for invalid inputs and logs warnings for recoverable config issues; R frequently uses `warning()`/`stop()` and then shells to Java; Java uses `java.util.logging` wrappers and propagates checked exceptions to the top-level run path.
- Logging style and required context fields: Python log format is `%(asctime)s [%(name)s] %(levelname)s: %(message)s`; Java log format is `osmose[level] message`.
- Sensitive-data redaction rules: `[TODO]` No explicit redaction policy or secret-filtering helper was found in the repo.

### 5) Testing Conventions

- Test file naming/location rule: Python tests live in `tests/` and use `test_*.py`; Java tests live in `java/src/test/java/` and use `Test*.java`; R automated package tests are `[TODO]` because no `tests/testthat` directory was found.
- Mocking strategy norm: Python uses pytest fixtures from `tests/conftest.py` and `unittest.mock.patch` for subprocess-dependent tests; Java tests use fixture files plus JUnit lifecycle hooks such as `@BeforeAll`.
- Coverage expectation: Python CI enforces `--cov-fail-under=90`; Java/R coverage threshold is `[TODO]`.

### 6) Evidence

- `osmose-python/pyproject.toml`
- `osmose-python/.pre-commit-config.yaml`
- `osmose-python/pyrightconfig.json`
- `osmose-python/osmose/logging.py`
- `osmose-python/tests/conftest.py`
- `osmose-master/NAMESPACE`
- `osmose-master/R/osmose-main.R`
- `osmose-master/java/src/main/java/fr/ird/osmose/util/logging/OsmoseLogFormatter.java`
- `osmose-master/java/src/test/java/fr/ird/osmose/TestAccess.java`

