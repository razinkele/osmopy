# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| ~~high~~ **RESOLVED 2026-09-27** (current state; history caveat below) | ~~Copernicus helper credentials are committed in source and MCP config~~ | `osmose-python/mcp_servers/copernicus/server.py`, `osmose-python/.mcp.json` | — | Done in the working tree: credentials are env-only (`load_dotenv` + `os.environ.get` + `_require_creds()`, `server.py:23-37`) and `tests/test_copernicus_mcp_env.py` + `tests/test_mcp_config_hygiene.py` fail CI if any reappear in tracked files — verified passing 2026-09-27. Swept this repo's history too: **14 revisions** of `.mcp.json`, `server.py` and `.env.example` hold no `CMEMS_(USER\|PASS\|USERNAME\|PASSWORD)` assigned a non-empty literal. Two caveats: (1) `6e71a2e8` reads "remove committed CMEMS credentials" but its diff is **purely additive** (54 insertions, 0 deletions — it added the env plumbing and the hygiene test), so its wording is not evidence that history was ever contaminated; (2) this file's `osmose-python/` path prefixes show it was authored against the **two-root workspace** (osmose-python + osmose-master), whose history is not visible from this repo — so the sweep cannot clear that one, and **rotation is not verifiable from any repo**. If the credentials were ever committed anywhere, treat rotation as open until an operator confirms it. |
| high | Java build path is not reproducible from a clean machine without Maven, and tests are skipped by default in both `pom.xml` and the workflow package command | `osmose-master/pom.xml`, `osmose-master/.github/workflows/java-compile.yml` | The reference engine can drift without routine local validation | Install/document Maven for contributors and stop skipping tests by default |
| ~~medium~~ **RESOLVED 2026-09-27** | ~~Python runtime/test posture is stronger than its static typing posture (`pyright` reports 22 errors)~~ | `osmose-python/pyrightconfig.json`, `osmose-python/.github/workflows/ci.yml` | — | Done: pyright reports **0 errors in CI**, verified 2026-09-27 at both v1.1.411 and v1.1.414. A local venv with the undeclared `pandas-stubs` package installed shows 21 stub-typing errors instead (proven by removal, 2026-09-29) — see CLAUDE.md Gotchas. Cleared over 2026-09-12/16 (`9199fec9`, `1832af92`); the CI type-check job is matrixed over 3.12/3.13 and green. Note a bare `.venv/bin/pyright` used to report ~295 *phantom* import errors from interpreter misresolution — see CLAUDE.md Gotchas before believing a large error count. |
| medium | Legacy Java code still contains unresolved TODO stubs in production classes | `docs/codebase/.codebase-scan.txt`, `osmose-master/java/src/main/java/fr/ird/osmose/background/BackgroundSchool.java` | Incomplete or fragile behavior can hide behind legacy paths | Review and either implement, remove, or explicitly deprecate stubbed methods |
| medium | Repo-root discovery is misleading because manifests and workflows live in nested subprojects | `docs/codebase/.codebase-scan.txt`, `osmose-python/pyproject.toml`, `osmose-master/pom.xml` | Automated tooling aimed at repo root can miss real build/test surfaces | Treat this repository as a two-root workspace in onboarding/docs/tooling |

### 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| Nested-subproject build assumptions | Root has no authoritative manifest, while both subprojects do | repo root vs `osmose-python/` and `osmose-master/` | New automation and contributors may run the wrong commands | Add root-level onboarding that points directly to both active roots |
| ~~Python type debt~~ **RESOLVED 2026-09-27** | ~~Pyright currently fails while tests pass; failures cluster around optional engine state and calibration UI plumbing~~ | `osmose-python/osmose/engine/processes/fishing.py`, `osmose-python/osmose/engine/processes/mortality.py`, `osmose-python/osmose/engine/timeseries.py`, `osmose-python/ui/pages/calibration_handlers.py` | — | Done: 0 pyright errors in CI as of 2026-09-27 (environment-dependent locally, see row above). The 22 were cleared over 2026-09-12/16. Residual caveat, not debt: `pyrightconfig.json` sets `exclude: ["tests"]`, so **`tests/` is not type-checked at all** — the 0 covers `osmose/` and `ui/` only. |
| Default-skipped Java tests | `pom.xml` sets `<skipTests>true</skipTests>` and the workflow packages with `-DskipTests=true` | `osmose-master/pom.xml`, `osmose-master/.github/workflows/java-compile.yml` | Packaging can succeed without validating the reference engine | Invert the default or create a separate fast/slow profile |
| Local Maven repository dependency | Build relies on `file:${project.basedir}/java/local/` plus UCAR repositories | `osmose-master/pom.xml`, `osmose-master/java/local/README.md` | Builds are more fragile on clean machines and outside the original maintainer workflow | Document, vendor, or publish the dependency chain more explicitly |
| ~~Secret handling in tooling~~ **RESOLVED 2026-09-27** | ~~Dev helper config contains real-looking credentials in committed files~~ | `osmose-python/.mcp.json`, `osmose-python/mcp_servers/copernicus/server.py` | — | Done: no credential literal in either file. `server.py:33-34` reads `os.environ.get("CMEMS_USERNAME"/"CMEMS_PASSWORD")` with **no fallback default**, and `_require_creds()` (`:37-45`) raises with operator guidance when either is unset. `tests/test_copernicus_mcp_env.py` + `tests/test_mcp_config_hygiene.py` fail CI if any reappear. **"Scrub history" is not actioned and is not known to be needed** — see the history caveat in Top Risks row 1. |
| Legacy TODO/stub concentration | Unfinished implementation remains in shipped Java classes | `docs/codebase/.codebase-scan.txt`, `osmose-master/java/src/main/java/fr/ird/osmose/background/BackgroundSchool.java` | Hard-to-predict behavior in edge features | Triage each TODO into fix/deprecate/document buckets |

### 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| ~~Hardcoded credentials in committed files~~ **RESOLVED 2026-09-27** (current state) | A07:2021 Identification and Authentication Failures | `osmose-python/.mcp.json`, `osmose-python/mcp_servers/copernicus/server.py` | Environment variables are **required**, not merely supported: `os.environ.get` with no fallback default (`server.py:33-34`) and `_require_creds()` raises when unset (`:37-45`); `tests/test_copernicus_mcp_env.py` + `tests/test_mcp_config_hygiene.py` enforce it in CI | **No committed secret and no unsafe fallback remains** in either file — the previous wording on both counts was stale (verified 2026-09-27). Remaining gap is historical, not current: **credential rotation cannot be verified from a repository**, and this file was authored against the two-root workspace whose history is not visible here. A sweep of 14 revisions of `.mcp.json`, `server.py` and `.env.example` in *this* repo found no `CMEMS_(USER\|PASS\|USERNAME\|PASSWORD)` literal. |
| Path traversal on overlay/data file selection | A01:2021 Broken Access Control | `osmose-python/osmose/engine/path_resolution.py`, `osmose-python/ui/pages/grid.py` | Python side rejects `..` path segments and validates overlay paths server-side | Protection exists, but there is no repo-level security policy or audit coverage |
| R wrapper shells out to Java with string-built arguments | A03:2021 Injection `[TODO] exact exploitability not fully established from reviewed files` | `osmose-master/R/osmose-main.R`, `osmose-master/R/update_config.R` | Uses `system2()` instead of a raw shell command execution path | Parameter construction is string-based and not centrally validated like Python JVM opts |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| Python acceleration relies on optional numerical stack | `osmose-python/README.md`, `osmose-python/pyproject.toml` | Engine supports pure Python plus optional Numba path | Performance can degrade sharply on environments missing the optimized stack | Make accelerated-path expectations explicit in deployment docs/tests |
| Large file/data footprint in repo | `docs/codebase/.codebase-scan.txt` | Multi-megabyte NetCDF, PDFs, and bundled JARs are present | Slower scans, clones, and local tooling | Separate heavy data artifacts from code when possible |
| Java build depends on local and remote artifact sources | `osmose-master/pom.xml`, `osmose-master/java/local/README.md` | Build requires both UCAR repos and a checked-in local Maven repo | Reproducibility and cache behavior vary by machine | Document and mirror required artifacts more explicitly |

### 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| `osmose-python/osmose/engine/config.py` | Central typed extraction layer for many recent parity features and file-loading rules | `66` path touches in `osmose-python` git history over the last 90 days | Keep changes narrow, preserve config-key compatibility, and validate through parity/config tests |
| `osmose-python/osmose/engine/simulate.py` and engine process modules | Central ordered simulation pipeline with shared state invariants | `57` touches for `simulate.py`; `35` for `processes/mortality.py`; `20` for `processes/movement.py`; `19` for `processes/predation.py` | Preserve step ordering and rely on engine parity/unit tests |
| `osmose-python/app.py` and major UI pages | Large UI wiring surface with many tabs and recent feature additions | `37` touches for `app.py`; `44` for `ui/pages/grid.py`; `37` for `ui/pages/results.py`; `34` for `ui/pages/run.py`; `28` for `ui/pages/calibration.py`; `19` for `ui/pages/calibration_handlers.py` | Change one page flow at a time and keep UI tests green |
| `osmose-master/R/osmose-main.R` | Main R-to-Java bridge and argument construction path | `[TODO]` root-level git history unavailable | Validate changes against packaged Java invocation and output reading |
| `osmose-master/java/src/main/java/fr/ird/osmose/Configuration.java` | Core config parser for legacy engine | `[TODO]` root-level git history unavailable; large central file | Use fixture-backed Java tests before changing parser behavior |

### 6) `[ASK USER]` Questions

1. [ASK USER] Is `osmose-python/mcp_servers/copernicus/` meant to be a committed, supported part of the product, or only local developer tooling? The answer changes whether the committed credentials are treated as a release blocker or a local-tooling cleanup.
2. [ASK USER] Should `osmose-master/` still be treated as an actively maintained product surface, or primarily as the legacy/reference engine for Python parity? That changes how aggressively to prioritize Java/R debt.
3. [ASK USER] Are the many `docs/superpowers/` plans/designs intended to represent active architecture decisions, or should onboarding/review docs treat them as historical implementation notes only?

### 7) Evidence

- `docs/codebase/.codebase-scan.txt`
- `osmose-python/.mcp.json`
- `osmose-python/mcp_servers/copernicus/server.py`
- `osmose-python/pyrightconfig.json`
- `osmose-python/.github/workflows/ci.yml`
- `osmose-python/osmose/engine/processes/fishing.py`
- `osmose-python/osmose/engine/processes/mortality.py`
- `osmose-python/osmose/engine/timeseries.py`
- `osmose-python/osmose/engine/path_resolution.py`
- `osmose-python/ui/pages/grid.py`
- nested git history from `osmose-python` (`git log --since=90.days --name-only`)
- `osmose-python/ui/pages/calibration_handlers.py`
- `osmose-master/pom.xml`
- `osmose-master/.github/workflows/java-compile.yml`
- `osmose-master/R/osmose-main.R`
- `osmose-master/R/update_config.R`
- `osmose-master/java/local/README.md`
- `osmose-master/java/src/main/java/fr/ird/osmose/background/BackgroundSchool.java`
