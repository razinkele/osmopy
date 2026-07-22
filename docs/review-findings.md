# Review findings

## Overall shape

- The repository is effectively **two nested project roots**:
  - `osmose-python/` — active Python app, pure-Python engine, Shiny UI, CI
  - `osmose-master/` — legacy R package plus Java reference engine
- Root-level scanning underreports the real stack because the important manifests and workflows live inside those subprojects.

## Current health

- **Python runtime health is good**
  - Ruff passes
  - Pytest passes: **2401 passed**, **15 skipped**
- **Python static typing is behind runtime health**
  - Pyright reports **22 errors**
  - Errors cluster in:
    - `osmose/engine/processes/fishing.py`
    - `osmose/engine/processes/mortality.py`
    - `osmose/engine/timeseries.py`
    - `ui/pages/calibration.py`
    - `ui/pages/calibration_charts.py`
    - `ui/pages/calibration_handlers.py`
- **Legacy Java/R path is weaker operationally**
  - Java is present locally
  - R is present locally
  - Maven is **not** installed locally
  - `osmose-master/pom.xml` skips tests by default
  - Java workflow also packages with `-DskipTests=true`

## Highest-risk findings

### 1. Committed credentials

- The same Copernicus credentials are committed in:
  - `osmose-python/.mcp.json`
  - `osmose-python/mcp_servers/copernicus/server.py`
- This is the clearest concrete security issue in the repository.

### 2. Type-check failures in hotspot code

- The Pyright failures are not random annotation noise.
- They sit in the same areas with recent heavy feature work:
  - engine mortality / fishing / timeseries
  - calibration UI state and path plumbing
- Several failures point to **optional-state and invariant encoding problems**, not just missing casts.

### 3. Legacy build reproducibility

- `osmose-master` depends on:
  - a **local Maven repository** under `java/local/`
  - remote UCAR artifact repositories
- That makes clean-machine builds more fragile than the Python project.

## Hotspots by recent churn (`osmose-python` nested git history, last 90 days)

- `osmose/engine/config.py` — **66**
- `osmose/engine/simulate.py` — **57**
- `ui/pages/grid.py` — **44**
- `app.py` — **37**
- `ui/pages/results.py` — **37**
- `osmose/engine/processes/mortality.py` — **35**
- `ui/pages/run.py` — **34**
- `ui/pages/calibration.py` — **28**

## What that churn means

- The riskiest active areas are:
  - **engine config + simulation core**
  - **large multi-purpose UI pages**
- Recent commits were concentrated on parity, RNG compatibility, fishing, bioenergetics, and dynamic accessibility, so breakage risk is highest where new parity logic meets old config/state assumptions.

## Architectural reality vs earlier intent

- Early Python port design centered on **keeping Java as the engine**.
- Current repo also contains a substantial **pure-Python simulation engine** that is actively developed.
- The current app/UI surface is broader and more complex than the early design docs suggest.

## Recommended priority order

1. Remove and rotate committed Copernicus credentials.
2. Fix the 22 Pyright errors, starting with engine optional-state invariants.
3. Make Java validation harder to skip:
   - stop default-skipping tests in `pom.xml`
   - stop packaging workflow from always using `-DskipTests=true`
4. Document or simplify the local Maven repository dependency.

## Pointers

- Full repo docs: `docs/codebase/`
- Risk summary: `docs/codebase/CONCERNS.md`
- Test posture: `docs/codebase/TESTING.md`
