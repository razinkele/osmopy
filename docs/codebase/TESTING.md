# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- Primary test framework: Pytest for `osmose-python`; JUnit 5 for Java tests in `osmose-master`.
- Assertion/mocking tools: pytest assertions, `pytest-asyncio`, `pytest-cov`, `unittest.mock.patch`; JUnit assertions; R package checks through workflow commands rather than an in-repo `testthat` suite.
- Commands:

```bash
cd osmose-python
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m e2e
.venv/bin/python -m pytest --cov=osmose

cd osmose-master
mvn test
R CMD build --no-build-vignettes .
R CMD check --as-cran --ignore-vignettes <tarball>
```

### 2) Test Layout

- Test file placement pattern: Python tests live under `osmose-python/tests/`; Java tests live under `osmose-master/java/src/test/java/`; Java fixture data lives under `osmose-master/java/src/test/resources/`.
- Naming convention: Python `test_*.py`; Java `Test*.java`.
- Setup files and where they run: Python `tests/conftest.py` provides shared fixtures and plotly template registration; Java uses JUnit lifecycle methods like `@BeforeAll` inside test classes; R setup file for a package test framework is `[TODO]` because no `tests/testthat` tree was found.

### 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | yes | Python config/engine helpers, Java utilities and managers | `osmose-python/tests/test_config_reader.py`, `osmose-python/tests/test_engine_growth.py`, `osmose-master/java/src/test/java/fr/ird/osmose/TestAccess.java` |
| Integration | yes | Python full engine/results/UI flows; Java runs against bundled EEC fixture data | Python suite passed `2401` tests with `15` skipped in the current environment; Java test resources under `java/src/test/resources/osmose-eec/` suggest fixture-backed integration tests |
| E2E | yes (Python), `[TODO]` for R/Java | Browser/user flows through Shiny app | Python marks Playwright tests with `e2e` and excludes them by default; no separate R/Java E2E layer was identified |

### 4) Mocking and Isolation Strategy

- Main mocking approach: Python patches subprocess-heavy code and uses pytest fixtures/temp paths; Java tests use real files plus JUnit setup hooks.
- Isolation guarantees: Python fixtures are centralized in `tests/conftest.py`; Java tests can copy fixture files into temp locations (for example `TestAccess.java`).
- Common failure mode in tests: Python static typing can fail while the runtime suite remains green; Java tests are easy to miss because `pom.xml` sets `<skipTests>true</skipTests>`.

### 5) Coverage and Quality Signals

- Coverage tool + threshold: Python uses `pytest-cov` with CI threshold `90%`.
- Current reported coverage: `[TODO]` current coverage percentage was not captured during this review; only the CI threshold and full passing pytest run were verified.
- Known gaps/flaky areas:
  - Python `pyright` currently reports `22` type errors even though Ruff and pytest pass. The failures cluster in `osmose/engine/processes/fishing.py`, `osmose/engine/processes/mortality.py`, `osmose/engine/timeseries.py`, and calibration UI files under `ui/pages/`.
  - Java tests are disabled in the default Maven configuration via `<skipTests>true</skipTests>`.
  - R package automated tests beyond build/check were not found in an in-repo test directory.

### 6) Evidence

- `osmose-python/pyproject.toml`
- `osmose-python/.github/workflows/ci.yml`
- `osmose-python/tests/conftest.py`
- `osmose-python/tests/test_e2e_grid_maps.py`
- `osmose-master/pom.xml`
- `osmose-master/java/src/test/java/fr/ird/osmose/TestAccess.java`
- `osmose-master/.github/workflows/r-compile.yml`
