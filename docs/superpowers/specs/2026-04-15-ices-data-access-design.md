# ICES Data Access — MCP Server + Claude Code Skill

> Date: 2026-04-15
> Status: Approved

## Summary

Two components for programmatic ICES data access:

1. **MCP Server** (`~/ices-mcp-server/`) — Standalone async Python MCP server wrapping 4 ICES APIs (SAG, DATRAS, SD, Eggs & Larvae) into 9 tools. Stdio transport, `uv run`, no auth required.
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
├── server.py               # MCP entry point, async tool registration
├── ices/
│   ├── __init__.py
│   ├── sag.py              # SAG REST client (async httpx)
│   ├── datras.py           # DATRAS XML client + summary aggregation
│   ├── sd.py               # Stock Database OData client
│   ├── eggs.py             # Eggs & Larvae REST client
│   └── vocab.py            # Species code resolution (COMMON_TO_WORMS + Vocab API fallback)
```

- **Transport:** stdio
- **Runtime:** `uv run server.py` (zero env management)
- **Dependencies:** `mcp`, `httpx`, `pandas`, `async-lru`
- **Async:** All tool handlers are `async def`, using `httpx.AsyncClient` with configured timeouts (connect=60s, read=120s for DATRAS large responses, 30s for SAG/SD/Eggs JSON). DATRAS streaming XML parsing runs via `asyncio.to_thread()` since `iterparse()` is synchronous.
- **Note:** `scipy` is NOT a server dependency. VB curve fitting runs in the Claude Code environment (user's `.venv`), not inside the MCP server. The server only fetches and returns data.
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
- **Note:** A species like cod may return 6+ stocks across regions. The tool returns all matches — the caller must pick the relevant stock key for downstream tools like `get_stock_assessment`.

#### `get_survey_cpue_length`

CPUE by length class from DATRAS trawl surveys.

- **API:** DATRAS `getCPUELength`
- **Input:**
  - `survey` (str, required) — Survey code: BITS, NS-IBTS, EVHOE, CGFS, etc.
  - `year` (int, required)
  - `quarter` (int, required) — 1-4
  - `species` (str, optional) — Common name or WoRMS code. If omitted, returns all species.
  - `mode` (str, optional) — `"summary"` (default) or `"raw"`
- **Output (summary):** Aggregated across hauls: `{species, aphia_id, length_cm, mean_cpue, median_cpue, sd_cpue, n_hauls}`
- **Output (raw):** Per-haul rows: `{species, aphia_id, length_cm, cpue_number_per_hour, haul_no, shoot_lat, shoot_lon, depth, gear, datetime}`

#### `get_survey_cpue_age`

CPUE by age class from DATRAS trawl surveys.

- **API:** DATRAS `getCPUEAge`
- **Input:**
  - `survey` (str, required) — Survey code: BITS, NS-IBTS, EVHOE, CGFS, etc.
  - `year` (int, required)
  - `quarter` (int, required) — 1-4
  - `species` (str, optional) — Common name or WoRMS code. If omitted, returns all species.
  - `mode` (str, optional) — `"summary"` (default) or `"raw"`
- **Output (summary):** Aggregated across hauls: `{species, aphia_id, age, mean_cpue, median_cpue, sd_cpue, n_hauls}`
- **Output (raw):** Per-haul rows: `{species, aphia_id, age, cpue_number_per_hour, haul_no, shoot_lat, shoot_lon, depth, gear, datetime}`

#### `get_survey_hauls`

Haul-level metadata from DATRAS surveys.

- **API:** DATRAS `getHHdata`
- **Input:**
  - `survey` (str, required)
  - `year` (int, required)
  - `quarter` (int, required)
- **Output:** List of `{haul_no, country, ship, shoot_lat, shoot_lon, haul_lat, haul_lon, depth, gear, haul_dur_min, day_night, sur_temp, bot_temp, datetime}`

#### `get_age_length_keys`

Age-length key data for Von Bertalanffy growth fitting.

- **API:** DATRAS `getCAdata` or `getCAdataSp`
- **Input:**
  - `survey` (str, required)
  - `year` (int, required)
  - `quarter` (int, required)
  - `species` (str, optional) — Common name or WoRMS code
- **Output:** List of `{species, aphia_id, age, length_cm, n_at_length, individual_weight_g, sex, maturity, maturity_scale}`

#### `search_eggs_larvae`

Ichthyoplankton summary data.

- **API:** Eggs & Larvae `getEggsAndLarvaeDataSummary`
- **Input:**
  - `species` (str, required) — Scientific name (e.g. `"Gadus morhua"`)
  - `year` (int, required) — Specific year to query
- **Output:** List of `{year, month, stage, survey, num_samples, num_measurements, aphia_id}`
- **Note:** If the result exceeds 500 rows, a `warning` field is included advising the caller to narrow the query.

### Key Design Decisions

#### Stock Key Resolution

`get_stock_assessment` and `get_reference_points` accept human-readable stock key labels (e.g. `cod.27.24-32`) rather than numeric assessment keys. The server calls `list_stocks` internally to resolve the label to the latest assessment key. This means users never need to know about ICES internal IDs.

**In-process async LRU cache:** The resolution function `resolve_stock_key(stock_key, year)` is wrapped with `alru_cache` from the `async-lru` package (not `functools.lru_cache`, which does not work with `async def` — it caches the coroutine object, not the awaited result). This eliminates repeated `StockList` fetches within a single MCP session (e.g. calling `get_stock_assessment` then `get_reference_points` for the same stock). The "no caching" policy applies to DATRAS data and cross-session persistence, not to this lightweight in-process lookup.

#### Species Code Resolution (vocab.py)

Lives in `ices/vocab.py`. DATRAS tools accept common names or WoRMS codes. The module maintains a built-in lookup table for ~30 common fish species (covering all typical OSMOSE focal species):

```python
COMMON_TO_WORMS = {
    "cod": 126436, "herring": 126417, "sprat": 126425,
    "flounder": 127141, "plaice": 127143, "sole": 127160,
    "whiting": 126438, "mackerel": 126735, "hake": 126484,
    "anchovy": 126426, "sardine": 126421, "horse mackerel": 126822,
    # ... ~20 more common species
}
```

The `resolve_species(name_or_code: str) -> int` function checks the table first, then falls back to the ICES Vocabulary API (`vocab.ices.dk/services/rdf/collection/SpecWoRMS`) for unlisted species.

#### DATRAS Aggregation

When `mode="summary"` (default), DATRAS XML is stream-parsed via `iterparse()` into a temporary CSV buffer (`io.StringIO` + `csv.writer`), then loaded once with `pd.read_csv()` for grouped aggregation (mean/median/sd by species + length/age class). This avoids materializing a list of dicts in memory (which triples memory usage for 31 MB XML). The aggregation runs in `datras.py` directly — no separate module needed.

When `mode="raw"`, per-haul rows are returned as JSON. Raw mode is needed for:
- Spatial interpolation (haul positions matter)
- VB curve fitting (individual measurements)
- Custom aggregation

For raw mode with large responses, results are streamed as JSON lines rather than a single array to avoid buffering the full result in memory.

#### DATRAS XML Parsing (verified 2026-04-15)

DATRAS responses are large (CPUE-by-length: ~31 MB, age-length: ~15.5 MB) and require careful parsing:

- **Streaming required:** Use `xml.etree.ElementTree.iterparse()` with `clear()` after each record — DOM parsing will exhaust memory.
- **XML namespace:** All elements use `xmlns="ices.dk.local/DATRAS"`. The parser must use namespace-aware lookups (e.g. `{ices.dk.local/DATRAS}LngtClas`) or strip the namespace prefix.
- **Element naming:** Root = `ArrayOfCls_Datras[Exchange]_{RecordType}`, record = `Cls_Datras[Exchange]_{RecordType}`. The `Exchange` infix appears on HH and CA but NOT on SurveyList or CPUE_Length.
- **NA sentinel:** `-9` throughout (not null/empty). Must convert to `None`/`NaN` on parse.
- **Length units:** `LngtClas` and `LngtClass` are in **mm**, not cm. Convert to cm (`/ 10.0`) in tool output.
- **Whitespace padding:** CA and HH string fields have trailing spaces — `.strip()` required.
- **DateTime format:** `DD/MM/YYYY HH:MM:SS` (European order, not ISO 8601).

#### Stock Key Cross-Mapping (verified 2026-04-15)

SAG and SD use different stock key formats for the same stocks:
- **SAG format:** `cod.27.24-32` (dot-separated, with ICES area prefix `27.`)
- **SD format:** `cod-2532` (dash-separated, no area prefix)

The server handles both formats via an explicit tested function `normalize_stock_key()` in `ices/sd.py`. A bidirectional lookup table is the primary path, covering all Baltic + common NE Atlantic stocks (~50 entries). There is NO regex fallback — unlisted keys raise `ValueError("Unknown stock key format: {key}. Add to STOCK_KEY_MAP in sd.py")`. This is intentional: silent regex normalization produces wrong keys for multi-area stocks like `her.27.20-24&253` or stocks with letter suffixes like `sol.27.4`. The mapping table must be covered by a test fixture with at least 10 edge-case stock keys verified against the live SD endpoint.

#### Error Handling

Three layers of error handling:

1. **HTTP-level:** Check status before parsing. Non-2xx responses (especially ICES 500 errors for invalid year/quarter/survey combos) raise a structured `IcesApiError` with a human-readable message before attempting XML/JSON parse:
   - `"No BITS data for Q3 2023 — BITS only runs in Q1 and Q4"`
   - `"Stock key 'cod.27.99' not found in 2023 assessments"`
   - `"DATRAS service temporarily unavailable (HTTP 503)"`

2. **Parse-level:** Wrap `iterparse()` in `try/except xml.etree.ElementTree.ParseError`. If HTTP 200 but XML is malformed or truncated mid-stream, return: `"DATRAS returned malformed XML after N records — partial data discarded"`. ICES sometimes returns HTML error pages with a 200 status code — detect by checking the first bytes for `<ArrayOf` prefix before parsing.

3. **Data-level:** SAG reference points can be null for specific stocks (e.g. Eastern Baltic cod has Bpa/Blim but null Flim/Fpa/Fmsy). The server returns these as `null` in JSON output with a `note` field (e.g. `"note": "FMSY not defined for this stock — ICES uses precautionary approach only"`).

4. **Transient retry:** For HTTP 429 (rate limit) and 503 (maintenance), retry with exponential backoff: 2s, 4s, 8s, max 3 attempts. ICES has undocumented soft rate limits — rapid looping (e.g. fetching 10 years of assessments) can trigger 429/503. Log each retry.

5. **Empty results:** When a valid request returns zero records (e.g. a species caught in zero hauls), return `{"result": [], "note": "No records found for BITS Q1 2023 cod"}` rather than an empty list with no context. This lets the caller distinguish "empty result" from a silent error.

#### No Caching (v1)

ICES data updates annually (stock assessments) or per survey season. Caching adds complexity for minimal benefit. If latency becomes an issue, a simple file-based cache keyed on (endpoint, params) can be added later.

For DATRAS large responses specifically, consider adding file-based caching in v2 to avoid re-downloading 30+ MB XML on repeated queries for the same survey/year/quarter.

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
3. Compute 5-year mean SSB
4. **Important:** `biomass_targets.csv` uses **total stock biomass** (not SSB) for 7/8 species. Only cod has `reference_point_type=ssb`. For biomass-type targets, apply a species-specific SSB-to-total scaling factor (typically 1.5-2x for gadoids, ~1.2x for small pelagics). For cod: compare SSB directly but note the target already accounts for the SSB→total conversion.
5. Report drift with units: "Cod SSB 2018-2022 mean = 72kt (ICES), your target = 120kt total biomass (total ~ 1.5-2x SSB = 108-144kt — within range)"

**Update growth parameters:**
1. Call `get_age_length_keys(survey="BITS", year=2023, quarter=1, species="cod")`
2. Fit VB curve: `L(t) = L_inf * (1 - exp(-K * (t - t0)))` using `scipy.optimize.curve_fit`
3. Compare fitted L_inf, K, t0 against config values
4. **Note on config key casing:** The schema uses lowercase (`species.linf`, `species.k`) but actual config files use camelCase (`species.lInf`, `species.K`). Java is case-sensitive. When writing config updates, match the casing in the existing config file, not the schema.
5. Suggest updates if drift > 10%

**Generate distribution maps:**
1. Call `get_survey_cpue_length(survey="BITS", year=2023, quarter=1, species="cod", mode="raw")`
2. Call `get_survey_hauls(survey="BITS", year=2023, quarter=1)` for positions
3. Map haul lat/lon to OSMOSE grid cells (using grid config: `grid.nlon`, `grid.nlat`, `grid.upleft.lat/lon`, `grid.lowright.lat/lon`)
4. Aggregate CPUE per cell. Two output options:
   - **Binary presence/absence** (matches existing Baltic maps): threshold CPUE → 1 (present) / 0 (absent) / -99 (land). This matches the format in `data/baltic/maps/` (semicolon-delimited, nlat x nlon grid).
   - **Continuous probability** (new): normalize CPUE to 0.0-1.0 distribution. OSMOSE accepts both formats but existing Baltic maps use binary.
5. Write as semicolon-delimited CSV, `nlat` rows x `nlon` columns.
6. **Grid cell mapping formula:** `col = floor((lon - upleft_lon) / cell_w)`, `row = floor((upleft_lat - lat) / cell_h)` where `cell_w = (lowright_lon - upleft_lon) / nlon`, `cell_h = (upleft_lat - lowright_lat) / nlat`. Discard hauls outside the grid bounds.
7. **Movement config keys** use `movement.{field}.map{N}` format in config files (e.g. `movement.file.map0`, `movement.species.map0`), NOT the schema format `movement.map{idx}.{field}`.

**Check reference points:**
1. Call `get_reference_points(stock_key="cod.27.24-32")`
2. Compare Fmsy against fishing config. Fishing rates are indexed by **fishery** (`fisheries.rate.base.fsh{N}`), not by species. There is also a legacy `mortality.fishing.rate.sp{N}` key in the species schema. Check which format the specific config uses.
3. Flag if simulated F exceeds Fmsy

#### Config Key Patterns

**Casing note:** Schema uses lowercase (`species.linf`), but config files use Java-compatible camelCase (`species.lInf`). Java is case-sensitive. Always match the casing in the existing config file.

```
# Species biology (config file casing shown)
species.lInf.sp{N}                    # VB L-infinity (cm) — schema: species.linf
species.K.sp{N}                       # VB K (year^-1) — schema: species.k
species.t0.sp{N}                      # VB t0 (years)
species.length2weight.condition.factor.sp{N}  # L-W a coefficient
species.length2weight.allometric.power.sp{N}  # L-W b exponent
species.maturity.size.sp{N}           # Size at maturity (cm)
species.lifespan.sp{N}               # Maximum age (years)

# Mortality & predation
mortality.additional.rate.sp{N}       # Natural mortality
predation.ingestion.rate.max.sp{N}    # Max ingestion rate

# Fishing (indexed by fishery, not species)
fisheries.rate.base.fsh{N}            # Base fishing rate per fishery
mortality.fishing.rate.sp{N}          # Legacy per-species F (check which format config uses)

# Movement maps (config file format, NOT schema format)
movement.file.map{N}                  # Map CSV file path
movement.species.map{N}              # Species index
movement.initialAge.map{N}           # Age threshold
movement.steps.map{N}                # Timesteps when active
```

## Testing

### MCP Server Tests

- **Unit tests** for each API client module (mock httpx responses with saved JSON/XML fixtures)
- **XML parsing tests** with saved DATRAS response fixtures (representative subsets of real responses, committed to `tests/fixtures/`)
- **Species code resolution tests** (common name -> WoRMS, including unknown species fallback)
- **Stock key cross-mapping tests** with at least 10 edge-case keys verified against live SD endpoint
- **Aggregation tests** (raw -> summary mode, verifying mean/median/sd computation)
- **Integration tests** (marked `@pytest.mark.integration`, excluded from default `pytest` run via `addopts = "-m 'not integration'"` in `pyproject.toml`):
  - Hit real ICES endpoints for one known stock (cod.27.24-32, year 2022)
  - Fetch one BITS quarter and verify parse pipeline end-to-end
  - These will fail during ICES quarterly maintenance — run manually only

### Skill Validation

- Manual testing: ask Claude to validate Baltic biomass targets, fetch cod growth params, generate a distribution map
- Verify skill triggers on ICES-related questions

## Out of Scope (v1)

- Write-back to OSMOSE config files (skill guides Claude, Claude does the edits)
- VMS/logbook data (requires JWT auth)
- Acoustic survey data (BIAS — no public API)
- Caching layer
- ICES Vocab API integration beyond species codes
- ICES Ecosystem Overviews and WGBIODIV indicator data (trophic level trends, biodiversity indices)
- Automated calibration pipeline integration (future: skill could trigger `calibrate_baltic.py` with updated targets)
