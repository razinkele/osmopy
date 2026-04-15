# ICES Data Access — MCP Server + Claude Code Skill

> Date: 2026-04-15
> Status: Approved

## Summary

Two components for programmatic ICES data access:

1. **MCP Server** (`~/ices-mcp-server/`) — Standalone Python MCP server wrapping 4 ICES APIs (SAG, DATRAS, SD, Eggs & Larvae) into 8 tools. Stdio transport, `uv run`, no auth required.
2. **Claude Code Skill** (`~/.claude/skills/ices-data/`) — Multi-region skill encoding ICES area conventions, survey catalogs, and OSMOSE integration workflows.

## Motivation

The OSMOSE Baltic calibration pipeline uses hand-curated biomass targets in `data/baltic/reference/biomass_targets.csv` sourced from ICES stock assessments. Currently there is no programmatic way to:

- Validate targets against latest ICES assessment data
- Fetch survey CPUE for spatial distribution maps
- Derive Von Bertalanffy growth parameters from DATRAS age-length keys
- Compare reference points (Fmsy, Blim) against fishing config

ICES provides free, unauthenticated REST/SOAP APIs but has no Python client libraries (only R packages). A thin MCP wrapper makes this data accessible to Claude for any OSMOSE modeling task.

## MCP Server Design

### Location and Runtime

```
~/ices-mcp-server/
├── pyproject.toml          # deps: mcp, httpx, pandas
├── server.py               # MCP entry point, tool registration
├── ices/
│   ├── __init__.py
│   ├── sag.py              # SAG REST client
│   ├── datras.py           # DATRAS XML client
│   ├── sd.py               # Stock Database OData client
│   ├── eggs.py             # Eggs & Larvae REST client
│   └── aggregation.py      # DATRAS summary aggregation
```

- **Transport:** stdio
- **Runtime:** `uv run server.py` (zero env management)
- **Dependencies:** `mcp`, `httpx`, `pandas`
- **Python:** 3.12+

### `.mcp.json` Entry

```json
"ices": {
  "command": "uv",
  "args": ["run", "--directory", "/home/razinka/ices-mcp-server", "server.py"]
}
```

### Tools

#### `list_stocks`

List all ICES-assessed stocks for a given year.

- **API:** SAG `StockList?year={year}`
- **Input:**
  - `year` (int, required) — Assessment year (e.g. 2023)
  - `area_filter` (str, optional) — Regex filter on stock key labels (e.g. `"27\\.2[2-9]|27\\.3[0-2]"` for Baltic subdivisions)
- **Output:** List of `{stock_key, species_name, assessment_key, expert_group}`

#### `get_stock_assessment`

Full assessment time series for a stock.

- **API:** SAG `StockList` (to resolve key) + `StockDownload?assessmentKey={key}`
- **Input:**
  - `stock_key` (str, required) — e.g. `"cod.27.24-32"`
  - `year` (int, optional) — Assessment year, defaults to latest available
- **Output:** Time series rows: `{year, ssb, recruitment, f, catches, landings, discards, low_ssb, high_ssb}`

#### `get_reference_points`

Reference points for a stock assessment.

- **API:** SAG `FishStockReferencePoints?assessmentKey={key}`
- **Input:**
  - `stock_key` (str, required)
  - `year` (int, optional)
- **Output:** `{flim, fpa, fmsy, blim, bpa, msy_btrigger, f_age_range, recruitment_age}`

#### `get_stock_metadata`

Stock metadata from the ICES Stock Database.

- **API:** SD OData `StockListDWs3?$filter=...`
- **Input:**
  - `species_name` (str, required) — Scientific name (e.g. `"Gadus morhua"`) or common name (e.g. `"cod"`)
- **Output:** List of `{stock_key, species_scientific, species_common, area_name, expert_group, trophic_guild, size_guild}`

#### `get_survey_cpue`

CPUE data from DATRAS trawl surveys.

- **API:** DATRAS `getCPUELength` or `getCPUEAge`
- **Input:**
  - `survey` (str, required) — Survey code: BITS, NS-IBTS, EVHOE, CGFS, etc.
  - `year` (int, required)
  - `quarter` (int, required) — 1-4
  - `species` (str, optional) — Common name or WoRMS code. If omitted, returns all species.
  - `by` (str, optional) — `"length"` (default) or `"age"`
  - `mode` (str, optional) — `"summary"` (default) or `"raw"`
- **Output (summary):** Aggregated across hauls: `{species, length_or_age, mean_cpue, median_cpue, sd_cpue, n_hauls}`
- **Output (raw):** Per-haul rows: `{species, length_or_age, cpue, haul_id, lat, lon, depth}`

#### `get_survey_hauls`

Haul-level metadata from DATRAS surveys.

- **API:** DATRAS `getHHdata`
- **Input:**
  - `survey` (str, required)
  - `year` (int, required)
  - `quarter` (int, required)
- **Output:** List of `{haul_id, lat, lon, depth, gear, duration_min, datetime}`

#### `get_age_length_keys`

Age-length key data for Von Bertalanffy growth fitting.

- **API:** DATRAS `getCAdata` or `getCAdataSp`
- **Input:**
  - `survey` (str, required)
  - `year` (int, required)
  - `quarter` (int, required)
  - `species` (str, optional) — Common name or WoRMS code
- **Output:** List of `{species, age, length_cm, n_measured, sex, maturity}`

#### `search_eggs_larvae`

Ichthyoplankton summary data.

- **API:** Eggs & Larvae `getEggsAndLarvaeDataSummary`
- **Input:**
  - `species` (str, required) — Scientific name (e.g. `"Gadus morhua"`)
  - `year` (int, optional) — If omitted, returns all years
- **Output:** List of `{year, month, stage, survey, count}`

### Key Design Decisions

#### Stock Key Resolution

`get_stock_assessment` and `get_reference_points` accept human-readable stock key labels (e.g. `cod.27.24-32`) rather than numeric assessment keys. The server calls `list_stocks` internally to resolve the label to the latest assessment key. This means users never need to know about ICES internal IDs.

#### Species Code Resolution

DATRAS tools accept common names or WoRMS codes. The server maintains a built-in lookup table for ~30 common fish species (covering all typical OSMOSE focal species):

```python
COMMON_TO_WORMS = {
    "cod": 126436, "herring": 126417, "sprat": 126425,
    "flounder": 127141, "plaice": 127143, "sole": 127160,
    "whiting": 126438, "mackerel": 126735, "hake": 126484,
    "anchovy": 126426, "sardine": 126421, "horse mackerel": 126822,
    # ... ~20 more common species
}
```

For unlisted species, the server falls back to the ICES Vocabulary API (`vocab.ices.dk/services/rdf/collection/SpecWoRMS`).

#### DATRAS Aggregation

When `mode="summary"` (default), raw DATRAS XML is parsed into a DataFrame, then grouped by species + length/age class with mean/median/sd computed across hauls. This keeps MCP responses concise (tens of rows instead of thousands).

When `mode="raw"`, per-haul rows are returned as JSON. Raw mode is needed for:
- Spatial interpolation (haul positions matter)
- VB curve fitting (individual measurements)
- Custom aggregation

#### Error Handling

ICES APIs return empty XML or HTTP 500 for invalid year/quarter/survey combinations. The server catches these and returns structured error messages:
- `"No BITS data for Q3 2023 — BITS only runs in Q1 and Q4"`
- `"Stock key 'cod.27.99' not found in 2023 assessments"`
- `"DATRAS service temporarily unavailable (HTTP 503)"`

#### No Caching (v1)

ICES data updates annually (stock assessments) or per survey season. Caching adds complexity for minimal benefit. If latency becomes an issue, a simple file-based cache keyed on (endpoint, params) can be added later.

## Claude Code Skill Design

### Location

```
~/.claude/skills/ices-data/
├── ices-data.md            # Main skill file
```

### Trigger Conditions

Fire when the user:
- Asks about ICES stock assessments, survey data, or fish distribution
- Wants to validate or update OSMOSE calibration targets against ICES
- Asks for growth parameters, reference points, or biomass estimates from ICES
- Mentions DATRAS, SAG, BITS, or other ICES data products
- Wants to compare simulation output against observational data

### Skill Knowledge

#### ICES Area Conventions

The skill encodes how stock key labels map to regions:

| Key pattern | Region |
|-------------|--------|
| `*.27.22-*`, `*.27.24-32` | Baltic Sea |
| `*.27.8.*` | Bay of Biscay |
| `*.27.7.d-e` | Eastern English Channel |
| `*.27.4.*` | North Sea |
| `*.27.3.a` | Skagerrak-Kattegat |

#### Survey Catalog

| Survey | Region | Quarters | Years |
|--------|--------|----------|-------|
| BITS | Baltic | Q1, Q4 | 1991+ |
| NS-IBTS | North Sea | Q1, Q3 | 1965+ |
| EVHOE | Bay of Biscay | Q4 | 1997+ |
| CGFS | Eastern English Channel | Q4 | 1988+ |
| IBTS-Q3 | North Sea + Celtic | Q3 | 1991+ |

#### OSMOSE Integration Patterns

**Validate biomass targets:**
1. Call `list_stocks(year=2023, area_filter="27\\.2[2-9]|27\\.3[0-2]")` for Baltic stocks
2. For each stock, call `get_stock_assessment(stock_key)` to get SSB time series
3. Compute 5-year mean SSB, compare against `biomass_targets.csv` values
4. Report drift: "Cod SSB 2018-2022 mean = 72kt, your target = 120kt (total biomass ~ 1.5-2x SSB = 108-144kt — within range)"

**Update growth parameters:**
1. Call `get_age_length_keys(survey="BITS", year=2023, quarter=1, species="cod")`
2. Fit VB curve: `L(t) = L_inf * (1 - exp(-K * (t - t0)))` using `scipy.optimize.curve_fit`
3. Compare fitted L_inf, K, t0 against `species.lInf.sp{N}`, `species.K.sp{N}`, `species.t0.sp{N}`
4. Suggest updates if drift > 10%

**Generate distribution maps:**
1. Call `get_survey_cpue(survey="BITS", year=2023, quarter=1, species="cod", mode="raw")`
2. Call `get_survey_hauls(survey="BITS", year=2023, quarter=1)` for positions
3. Map haul lat/lon to OSMOSE grid cells (using grid config: nlon, nlat, lon/lat bounds)
4. Normalize CPUE per cell to create probability distribution maps
5. Write as CSV in OSMOSE movement map format

**Check reference points:**
1. Call `get_reference_points(stock_key="cod.27.24-32")`
2. Compare Fmsy against `fishing.mortality.rate.sp{N}` in fishing config
3. Flag if simulated F exceeds Fmsy

#### Config Key Patterns

```
species.lInf.sp{N}                    # VB L-infinity (cm)
species.K.sp{N}                       # VB K (year^-1)
species.t0.sp{N}                      # VB t0 (years)
species.length2weight.condition.factor.sp{N}  # L-W a coefficient
species.length2weight.allometric.power.sp{N}  # L-W b exponent
species.maturity.size.sp{N}           # Size at maturity (cm)
species.lifespan.sp{N}               # Maximum age (years)
mortality.additional.rate.sp{N}       # Natural mortality
predation.ingestion.rate.max.sp{N}    # Max ingestion rate
```

## Testing

### MCP Server Tests

- Unit tests for each API client module (mock httpx responses)
- Integration test that hits real ICES endpoints for one known stock (cod.27.24-32, year 2022)
- XML parsing tests with saved DATRAS response fixtures
- Species code resolution tests (common name -> WoRMS)
- Aggregation tests (raw -> summary mode)

### Skill Validation

- Manual testing: ask Claude to validate Baltic biomass targets, fetch cod growth params, generate a distribution map
- Verify skill triggers on ICES-related questions

## Out of Scope (v1)

- Write-back to OSMOSE config files (skill guides Claude, Claude does the edits)
- VMS/logbook data (requires JWT auth)
- Acoustic survey data (BIAS — no public API)
- Caching layer
- ICES Vocab API integration beyond species codes
- Automated calibration pipeline integration (future: skill could trigger `calibrate_baltic.py` with updated targets)
