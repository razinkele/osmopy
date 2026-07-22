# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|---------------------------|---------|------------|-------------|----------|
| Java OSMOSE engine | Local subprocess/JAR | Alternate execution backend for Python and the only execution backend for the R package | Local executable + CLI args | high | `osmose-python/osmose/runner.py`, `osmose-master/R/osmose-main.R`, `osmose-master/pom.xml` |
| NetCDF/CSV model files | File-format boundary | Input configs, forcing files, movement maps, outputs, and restart/state exchange | File-system access | high | `osmose-python/osmose/config/reader.py`, `osmose-python/osmose/results.py`, `osmose-master/R/osmose-readConfiguration.R`, `osmose-master/java/src/main/java/fr/ird/osmose/Configuration.java` |
| Copernicus Marine / CMEMS helper | External data API via MCP helper | Browse/download Baltic forcing data for OSMOSE workflows | Environment variables, but defaults are hardcoded in-repo | medium | `osmose-python/mcp_servers/copernicus/server.py`, `osmose-python/.mcp.json` |
| UCAR NetCDF Maven repositories | Build-time artifact repository | Resolve Java NetCDF dependencies | Maven repository access | medium | `osmose-master/pom.xml` |
| Local Maven repository (`java/local/`) | Local artifact store | Supplies `ml.options` dependency | Local filesystem | medium | `osmose-master/pom.xml`, `osmose-master/java/local/README.md` |

### 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| Local configuration files | Source of simulation parameters and recursive sub-config references | Python `OsmoseConfigReader`; R `.readConfiguration`; Java `Configuration` | Path handling and missing-file errors can break runs | `osmose-python/osmose/config/reader.py`, `osmose-master/R/osmose-readConfiguration.R`, `osmose-master/java/src/main/java/fr/ird/osmose/Configuration.java` |
| Output directories (`output/`, NetCDF/CSV artifacts) | Persistent run results for later analysis | Python `results.py`, R `read_osmose`, Java output modules | Large output sets and format compatibility across versions | `osmose-python/osmose/results.py`, `osmose-master/R/osmose-main.R`, `osmose-master/java/src/main/java/fr/ird/osmose/Simulation.java` |
| Scenario/config snapshots | Saved model setups for reuse | Python scenario/config tooling `[TODO]`, R config readers/writers | `[TODO]` precise persistence shape not fully verified from source subset read | `osmose-python/README.md`, `osmose-master/R/osmose-write_osmose.R` |

### 3) Secrets and Credentials Handling

- Credential sources: environment variables in the Copernicus helper, plus a checked-in `.mcp.json` entry that also carries the same values.
- Hardcoding checks: hardcoded fallback credentials are present in `osmose-python/mcp_servers/copernicus/server.py`; the same credentials also appear in `osmose-python/.mcp.json`.
- Rotation or lifecycle notes: `[TODO]` no secret rotation or credential lifecycle process is documented in-repo.

### 4) Reliability and Failure Behavior

- Retry/backoff behavior: none found for the Java subprocess paths or the R wrapper; `[TODO]` Copernicus helper retry behavior was not fully inspected beyond credential/bootstrap code.
- Timeout policy: Python Java runner accepts `timeout_sec`; calibration subprocess calls use explicit subprocess timeouts in tests and implementation; R `system2()` calls wait synchronously with no timeout shown in the reviewed files.
- Circuit-breaker or fallback behavior: Python supports backend fallback at the architecture level (pure Python engine vs Java engine), but no circuit-breaker/fallback wrapper for external calls was found.

### 5) Observability for Integrations

- Logging around external calls: yes — Python runner/config layers log through `setup_logging`; Java uses `OLogger` formatting; R prints messages and routes stdout/stderr to log files.
- Metrics/tracing coverage: no in-repo metrics/tracing system was found.
- Missing visibility gaps: no structured tracing around file I/O or external dataset downloads; no secret-use audit trail; no repo-level monitoring configuration.

### 6) Evidence

- `osmose-python/osmose/runner.py`
- `osmose-python/osmose/config/reader.py`
- `osmose-python/osmose/results.py`
- `osmose-python/mcp_servers/copernicus/server.py`
- `osmose-python/.mcp.json`
- `osmose-master/R/osmose-main.R`
- `osmose-master/R/osmose-readConfiguration.R`
- `osmose-master/pom.xml`
- `osmose-master/java/local/README.md`

