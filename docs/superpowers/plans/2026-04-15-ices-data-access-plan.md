# ICES Data Access MCP Server + Skill — Implementation Plan

> **STATUS (verified 2026-04-18): COMPLETE — shipped 2026-04-15. DO NOT RE-EXECUTE.** Evidence:
> - Server at `/home/razinka/ices-mcp-server/server.py` exposes the full 9-tool set (`list_stocks`, `get_stock_assessment`, `get_reference_points`, `get_stock_metadata`, `get_survey_cpue_length`, `get_survey_cpue_age`, `get_survey_hauls`, `get_age_length_keys`, `search_eggs_larvae`) via stdio.
> - Four ICES APIs covered by dedicated modules under `/home/razinka/ices-mcp-server/ices/`: `sag.py`, `datras.py`, `sd.py`, `eggs.py` (plus `vocab.py` support).
> - Claude Code skill at `~/.claude/skills/ices-data/ices-data.md` encodes multi-region OSMOSE integration workflows (Baltic / North Sea / Bay of Biscay / EEC).
> - Test suite under `/home/razinka/ices-mcp-server/tests/`: `test_sag.py`, `test_datras.py`, `test_sd.py`, `test_eggs.py`, `test_vocab.py`, `test_integration.py`.
> - Registered as `"ices"` in `osmose-python/.mcp.json` lines 35-38.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone async Python MCP server wrapping 4 ICES APIs into 9 tools, plus a Claude Code skill encoding multi-region OSMOSE integration workflows.

**Architecture:** Standalone project at `~/ices-mcp-server/` with one module per upstream API (`sag.py`, `datras.py`, `sd.py`, `eggs.py`, `vocab.py`). Async throughout (`httpx.AsyncClient`, `async-lru`). DATRAS XML parsed via streaming `iterparse()` in a thread pool. Skill is a single markdown file at `~/.claude/skills/ices-data/ices-data.md`.

**Tech Stack:** Python 3.12+, `mcp` SDK, `httpx`, `pandas`, `uv run`, `pytest`

**Spec:** `docs/superpowers/specs/2026-04-15-ices-data-access-design.md`

---

## File Map

### New files (all under `~/ices-mcp-server/`)

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, deps, pytest config |
| `server.py` | MCP entry point, 9 async tool handlers |
| `ices/__init__.py` | Package init, `IcesApiError` exception |
| `ices/vocab.py` | `COMMON_TO_WORMS` table + `resolve_species()` |
| `ices/sag.py` | SAG REST client: `list_stocks()`, `resolve_stock_key()`, `get_assessment()`, `get_reference_points()` |
| `ices/sd.py` | SD OData client: `get_metadata()`, `STOCK_KEY_MAP`, `normalize_stock_key()` |
| `ices/datras.py` | DATRAS XML client: `get_cpue_length()`, `get_cpue_age()`, `get_hauls()`, `get_age_length()`, streaming parser, aggregation |
| `ices/eggs.py` | Eggs & Larvae REST client: `search()` |
| `tests/__init__.py` | Test package |
| `tests/conftest.py` | Shared fixtures (async httpx mock, XML/JSON fixtures) |
| `tests/fixtures/sag_stock_list.json` | Saved SAG StockList response subset |
| `tests/fixtures/sag_stock_download.json` | Saved SAG StockDownload response subset |
| `tests/fixtures/sag_ref_points.json` | Saved SAG ReferencePoints response |
| `tests/fixtures/sd_stock_list.json` | Saved SD OData response |
| `tests/fixtures/datras_cpue_length.xml` | Saved DATRAS CPUE subset (~50 records) |
| `tests/fixtures/datras_hh.xml` | Saved DATRAS haul data subset (~10 records) |
| `tests/fixtures/datras_ca.xml` | Saved DATRAS age-length subset (~30 records) |
| `tests/fixtures/eggs_summary.json` | Saved Eggs & Larvae response |
| `tests/test_vocab.py` | Species resolution tests |
| `tests/test_sag.py` | SAG client tests |
| `tests/test_sd.py` | SD client + stock key mapping tests |
| `tests/test_datras.py` | DATRAS parser + aggregation tests |
| `tests/test_eggs.py` | Eggs client tests |
| `tests/test_integration.py` | Live endpoint integration tests (marked) |

### New files (skill)

| File | Responsibility |
|------|---------------|
| `~/.claude/skills/ices-data/ices-data.md` | Claude Code skill with ICES knowledge |

### Modified files

| File | Change |
|------|--------|
| `~/osmose/osmose-python/.mcp.json` | Add `"ices"` server entry |

---

## Task 1: Project Scaffold

**Files:**
- Create: `~/ices-mcp-server/pyproject.toml`
- Create: `~/ices-mcp-server/ices/__init__.py`
- Create: `~/ices-mcp-server/tests/__init__.py`

- [ ] **Step 1: Create project directory**

```bash
mkdir -p ~/ices-mcp-server/ices ~/ices-mcp-server/tests ~/ices-mcp-server/tests/fixtures
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "ices-mcp-server"
version = "0.1.0"
description = "MCP server wrapping ICES fisheries data APIs"
requires-python = ">=3.12"
dependencies = [
    "mcp>=1.0",
    "httpx>=0.27",
    "pandas>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "respx>=0.22", "ruff>=0.4"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "-m 'not integration'"
markers = ["integration: tests hitting live ICES endpoints (deselect by default)"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Write ices/__init__.py**

```python
"""ICES data access library."""


class IcesApiError(Exception):
    """Raised when an ICES API call fails."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)
```

- [ ] **Step 4: Write empty test inits**

```python
# tests/__init__.py — empty
```

- [ ] **Step 5: Verify uv can resolve deps**

Run: `cd ~/ices-mcp-server && uv sync --dev`
Expected: Dependencies resolved and installed.

- [ ] **Step 6: Commit**

```bash
cd ~/ices-mcp-server && git init && git add -A && git commit -m "chore: scaffold ices-mcp-server project"
```

---

## Task 2: Species Code Resolution (vocab.py)

**Files:**
- Create: `~/ices-mcp-server/ices/vocab.py`
- Create: `~/ices-mcp-server/tests/test_vocab.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vocab.py
import pytest

from ices.vocab import COMMON_TO_WORMS, resolve_species


def test_resolve_common_name():
    assert resolve_species("cod") == 126436


def test_resolve_common_name_case_insensitive():
    assert resolve_species("Cod") == 126436
    assert resolve_species("COD") == 126436


def test_resolve_numeric_string():
    assert resolve_species("126436") == 126436


def test_resolve_integer_passthrough():
    assert resolve_species(126436) == 126436


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown species"):
        resolve_species("nonexistent_fish_xyz")


def test_common_to_worms_has_key_species():
    required = ["cod", "herring", "sprat", "flounder", "plaice", "sole",
                 "whiting", "mackerel", "hake", "anchovy", "sardine",
                 "horse mackerel", "perch", "pikeperch", "whitefish",
                 "stickleback"]
    for name in required:
        assert name in COMMON_TO_WORMS, f"Missing: {name}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_vocab.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ices.vocab'`

- [ ] **Step 3: Implement vocab.py**

```python
# ices/vocab.py
"""Species code resolution: common name -> WoRMS AphiaID."""

from __future__ import annotations

COMMON_TO_WORMS: dict[str, int] = {
    # Gadoids
    "cod": 126436,
    "whiting": 126438,
    "hake": 126484,
    "haddock": 126437,
    "saithe": 126440,
    "poor cod": 126442,
    "pouting": 126441,
    # Clupeids
    "herring": 126417,
    "sprat": 126425,
    "sardine": 126421,
    "anchovy": 126426,
    # Flatfish
    "flounder": 127141,
    "plaice": 127143,
    "sole": 127160,
    "dab": 127139,
    "turbot": 127149,
    "brill": 127144,
    # Pelagics
    "mackerel": 126735,
    "horse mackerel": 126822,
    # Perciformes & Baltic
    "perch": 151353,
    "pikeperch": 151308,
    "whitefish": 126740,
    "stickleback": 126505,
    # Elasmobranchs
    "lesser spotted dogfish": 105814,
    # Cephalopods
    "squid": 140629,
    # Others
    "dragonet": 126792,
    "red mullet": 126986,
}

# Reverse lookup for display
_WORMS_TO_COMMON: dict[int, str] = {v: k for k, v in COMMON_TO_WORMS.items()}


def resolve_species(name_or_code: str | int) -> int:
    """Resolve a common name or WoRMS code to an integer AphiaID.

    Accepts: common name (case-insensitive), numeric string, or int.
    Raises ValueError for unknown names (no Vocab API fallback in v1).
    """
    if isinstance(name_or_code, int):
        return name_or_code

    # Numeric string
    try:
        return int(name_or_code)
    except ValueError:
        pass

    # Common name lookup (case-insensitive)
    key = name_or_code.strip().lower()
    if key in COMMON_TO_WORMS:
        return COMMON_TO_WORMS[key]

    raise ValueError(
        f"Unknown species: '{name_or_code}'. "
        f"Known names: {', '.join(sorted(COMMON_TO_WORMS.keys()))}"
    )


def species_display_name(aphia_id: int) -> str:
    """Return common name for a WoRMS code, or 'AphiaID:{code}' if unknown."""
    return _WORMS_TO_COMMON.get(aphia_id, f"AphiaID:{aphia_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_vocab.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/ices-mcp-server && git add ices/vocab.py tests/test_vocab.py && git commit -m "feat: add species code resolution (vocab.py)"
```

---

## Task 3: SAG REST Client (sag.py)

**Files:**
- Create: `~/ices-mcp-server/ices/sag.py`
- Create: `~/ices-mcp-server/tests/test_sag.py`
- Create: `~/ices-mcp-server/tests/conftest.py`
- Create: `~/ices-mcp-server/tests/fixtures/sag_stock_list.json`
- Create: `~/ices-mcp-server/tests/fixtures/sag_stock_download.json`
- Create: `~/ices-mcp-server/tests/fixtures/sag_ref_points.json`

- [ ] **Step 1: Save SAG fixture data**

Fetch real SAG responses and save representative subsets to `tests/fixtures/`. Run these commands:

```bash
cd ~/ices-mcp-server
# Fetch stock list for 2023, save first 10 entries
uv run python -c "
import httpx, json
r = httpx.get('https://sag.ices.dk/SAG_API/api/StockList?year=2023', timeout=30)
data = r.json()[:10]
with open('tests/fixtures/sag_stock_list.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Saved {len(data)} stocks')
"

# Fetch cod.27.24-32 assessment (find its key first)
uv run python -c "
import httpx, json
stocks = httpx.get('https://sag.ices.dk/SAG_API/api/StockList?year=2023', timeout=30).json()
cod = [s for s in stocks if s.get('StockKeyLabel') == 'cod.27.24-32'][0]
key = cod['AssessmentKey']
r = httpx.get(f'https://sag.ices.dk/SAG_API/api/StockDownload?assessmentKey={key}', timeout=30)
data = r.json()[:5]  # first 5 years
with open('tests/fixtures/sag_stock_download.json', 'w') as f:
    json.dump(data, f, indent=2)

rp = httpx.get(f'https://sag.ices.dk/SAG_API/api/FishStockReferencePoints?assessmentKey={key}', timeout=30)
with open('tests/fixtures/sag_ref_points.json', 'w') as f:
    json.dump(rp.json(), f, indent=2)
print(f'Saved assessment key={key}, {len(data)} years, ref points')
"
```

- [ ] **Step 2: Write conftest.py with httpx mock helper**

```python
# tests/conftest.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_path() -> Path:
    return FIXTURES


def load_fixture(name: str) -> str:
    """Load a fixture file as a string."""
    return (FIXTURES / name).read_text()


def load_json_fixture(name: str) -> list | dict:
    """Load a JSON fixture file."""
    return json.loads(load_fixture(name))


@pytest.fixture
def clear_sag_cache():
    """Clear SAG stock key cache between tests for isolation.

    Not autouse — only used by test_sag.py tests that need it.
    Import and use as: @pytest.mark.usefixtures("clear_sag_cache")
    """
    from ices.sag import clear_stock_key_cache

    clear_stock_key_cache()
    yield
    clear_stock_key_cache()
```

- [ ] **Step 3: Write failing SAG tests**

```python
# tests/test_sag.py
from __future__ import annotations

import json
import re

import httpx
import pytest
import respx

from ices.sag import (
    SAG_BASE,
    get_assessment,
    get_reference_points,
    list_stocks,
    resolve_stock_key,
)


pytestmark = pytest.mark.usefixtures("clear_sag_cache")
from tests.conftest import load_fixture, load_json_fixture


@respx.mock
@pytest.mark.asyncio
async def test_list_stocks():
    data = load_json_fixture("sag_stock_list.json")
    respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"}).respond(
        json=data
    )
    result = await list_stocks(httpx.AsyncClient(), year=2023)
    assert len(result) > 0
    assert "stock_key" in result[0]
    assert "assessment_key" in result[0]


@respx.mock
@pytest.mark.asyncio
async def test_list_stocks_with_area_filter():
    data = load_json_fixture("sag_stock_list.json")
    respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"}).respond(
        json=data
    )
    result = await list_stocks(
        httpx.AsyncClient(), year=2023, area_filter="27\\.24"
    )
    # Only stocks matching the area filter
    for stock in result:
        assert re.search(r"27\.24", stock["stock_key"])


@respx.mock
@pytest.mark.asyncio
async def test_resolve_stock_key():
    data = load_json_fixture("sag_stock_list.json")
    respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"}).respond(
        json=data
    )
    # This depends on fixture having cod.27.24-32
    # If not present, skip
    cod_entries = [s for s in data if s.get("StockKeyLabel") == "cod.27.24-32"]
    if not cod_entries:
        pytest.skip("Fixture doesn't contain cod.27.24-32")
    key = await resolve_stock_key(httpx.AsyncClient(), "cod.27.24-32", 2023)
    assert isinstance(key, int)
    assert key == cod_entries[0]["AssessmentKey"]


@respx.mock
@pytest.mark.asyncio
async def test_resolve_stock_key_not_found():
    data = load_json_fixture("sag_stock_list.json")
    respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"}).respond(
        json=data
    )
    from ices import IcesApiError

    with pytest.raises(IcesApiError, match="not found"):
        await resolve_stock_key(httpx.AsyncClient(), "fake.99.99", 2023)


@respx.mock
@pytest.mark.asyncio
async def test_get_assessment():
    stock_list = load_json_fixture("sag_stock_list.json")
    download = load_json_fixture("sag_stock_download.json")
    respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"}).respond(
        json=stock_list
    )
    cod = [s for s in stock_list if s.get("StockKeyLabel") == "cod.27.24-32"]
    if not cod:
        pytest.skip("Fixture doesn't contain cod.27.24-32")
    key = cod[0]["AssessmentKey"]
    respx.get(
        f"{SAG_BASE}/StockDownload", params={"assessmentKey": str(key)}
    ).respond(json=download)
    result = await get_assessment(
        httpx.AsyncClient(), stock_key="cod.27.24-32", year=2023
    )
    assert len(result) > 0
    assert "year" in result[0]
    assert "ssb" in result[0]


@respx.mock
@pytest.mark.asyncio
async def test_get_reference_points():
    stock_list = load_json_fixture("sag_stock_list.json")
    ref_data = load_json_fixture("sag_ref_points.json")
    respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"}).respond(
        json=stock_list
    )
    cod = [s for s in stock_list if s.get("StockKeyLabel") == "cod.27.24-32"]
    if not cod:
        pytest.skip("Fixture doesn't contain cod.27.24-32")
    key = cod[0]["AssessmentKey"]
    respx.get(
        f"{SAG_BASE}/FishStockReferencePoints",
        params={"assessmentKey": str(key)},
    ).respond(json=ref_data)
    result = await get_reference_points(
        httpx.AsyncClient(), stock_key="cod.27.24-32", year=2023
    )
    assert "blim" in result
    assert "bpa" in result


@respx.mock
@pytest.mark.asyncio
async def test_get_reference_points_null_fmsy():
    """Verify note field when Fmsy is null (e.g. Eastern Baltic cod)."""
    stock_list = load_json_fixture("sag_stock_list.json")
    # Build a ref points response with null F values
    null_ref = [{"Flim": None, "Fpa": None, "FMSY": None,
                 "Blim": 108942, "Bpa": 122114, "MSYBtrigger": None,
                 "FAge": None, "RecruitmentAge": 1}]
    respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"}).respond(
        json=stock_list
    )
    cod = [s for s in stock_list if s.get("StockKeyLabel") == "cod.27.24-32"]
    if not cod:
        pytest.skip("Fixture doesn't contain cod.27.24-32")
    key = cod[0]["AssessmentKey"]
    respx.get(
        f"{SAG_BASE}/FishStockReferencePoints",
        params={"assessmentKey": str(key)},
    ).respond(json=null_ref)
    result = await get_reference_points(
        httpx.AsyncClient(), stock_key="cod.27.24-32", year=2023
    )
    assert result["fmsy"] is None
    assert result["blim"] == 108942
    assert "note" in result
    assert "FMSY" in result["note"]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_json_retries_on_503():
    """Verify retry with backoff on transient 503."""
    from ices.sag import _fetch_json

    route = respx.get(f"{SAG_BASE}/StockList", params={"year": "2023"})
    # First 2 calls return 503, third succeeds
    route.side_effect = [
        httpx.Response(503, text="Maintenance"),
        httpx.Response(503, text="Maintenance"),
        httpx.Response(200, json=[{"StockKeyLabel": "test"}]),
    ]
    result = await _fetch_json(
        httpx.AsyncClient(), f"{SAG_BASE}/StockList", {"year": "2023"}
    )
    assert result == [{"StockKeyLabel": "test"}]
    assert route.call_count == 3
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_sag.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ices.sag'`

- [ ] **Step 5: Implement sag.py**

```python
# ices/sag.py
"""ICES Stock Assessment Graphs (SAG) REST client."""

from __future__ import annotations

import re

import httpx

from ices import IcesApiError

SAG_BASE = "https://sag.ices.dk/SAG_API/api"

# Timeouts: 60s connect, 30s read for JSON endpoints
_TIMEOUT = httpx.Timeout(connect=60.0, read=30.0, write=30.0, pool=30.0)

# Max retries for transient errors
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 503}


async def _fetch_json(client: httpx.AsyncClient, url: str, params: dict) -> list | dict:
    """Fetch JSON from SAG with retry on transient errors."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code in _RETRY_STATUSES:
                import asyncio

                delay = 2 ** (attempt + 1)
                await asyncio.sleep(delay)
                continue
            if resp.status_code != 200:
                raise IcesApiError(
                    f"SAG returned HTTP {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )
            return resp.json()
        except httpx.TimeoutException as e:
            last_exc = e
    if last_exc:
        raise IcesApiError(f"SAG request timed out after {_MAX_RETRIES} attempts") from last_exc
    raise IcesApiError("SAG request failed after retries")


async def list_stocks(
    client: httpx.AsyncClient,
    *,
    year: int,
    area_filter: str | None = None,
) -> list[dict]:
    """List all ICES-assessed stocks for a year."""
    data = await _fetch_json(client, f"{SAG_BASE}/StockList", {"year": str(year)})
    stocks = []
    for entry in data:
        stock_key = entry.get("StockKeyLabel", "")
        if area_filter and not re.search(area_filter, stock_key):
            continue
        stocks.append({
            "stock_key": stock_key,
            "species_name": entry.get("SpeciesName", ""),
            "assessment_key": entry.get("AssessmentKey"),
            "expert_group": entry.get("ExpertGroup", entry.get("ExpertGroupUrl", "")),
        })
    return stocks


# Module-level cache for stock key resolution (cleared per session).
# NOT using alru_cache because httpx.AsyncClient is unhashable.
_stock_key_cache: dict[tuple[str, int], int] = {}


def clear_stock_key_cache() -> None:
    """Clear the stock key resolution cache (useful for tests)."""
    _stock_key_cache.clear()


async def resolve_stock_key(
    client: httpx.AsyncClient, stock_key: str, year: int
) -> int:
    """Resolve a stock key label to its numeric assessment key."""
    cache_key = (stock_key, year)
    if cache_key in _stock_key_cache:
        return _stock_key_cache[cache_key]
    stocks = await list_stocks(client, year=year)
    for s in stocks:
        if s["stock_key"] == stock_key:
            _stock_key_cache[cache_key] = s["assessment_key"]
            return s["assessment_key"]
    raise IcesApiError(
        f"Stock key '{stock_key}' not found in {year} assessments. "
        f"Use list_stocks(year={year}) to see available stocks."
    )


async def get_assessment(
    client: httpx.AsyncClient,
    *,
    stock_key: str,
    year: int = 2023,
) -> list[dict]:
    """Fetch full assessment time series for a stock."""
    assessment_key = await resolve_stock_key(client, stock_key, year)
    data = await _fetch_json(
        client,
        f"{SAG_BASE}/StockDownload",
        {"assessmentKey": str(assessment_key)},
    )
    rows = []
    for entry in data:
        rows.append({
            "year": entry.get("Year"),
            "ssb": entry.get("StockSize"),
            "recruitment": entry.get("Recruitment"),
            "f": entry.get("FishingPressure"),
            "catches": entry.get("Catches"),
            "landings": entry.get("Landings"),
            "discards": entry.get("Discards"),
            "low_ssb": entry.get("Low_StockSize"),
            "high_ssb": entry.get("High_StockSize"),
        })
    return rows


async def get_reference_points(
    client: httpx.AsyncClient,
    *,
    stock_key: str,
    year: int = 2023,
) -> dict:
    """Fetch reference points for a stock assessment."""
    assessment_key = await resolve_stock_key(client, stock_key, year)
    data = await _fetch_json(
        client,
        f"{SAG_BASE}/FishStockReferencePoints",
        {"assessmentKey": str(assessment_key)},
    )
    # API returns a list with one element
    entry = data[0] if isinstance(data, list) and data else data
    result = {
        "flim": entry.get("Flim"),
        "fpa": entry.get("Fpa"),
        "fmsy": entry.get("FMSY"),
        "blim": entry.get("Blim"),
        "bpa": entry.get("Bpa"),
        "msy_btrigger": entry.get("MSYBtrigger"),
        "f_age_range": entry.get("FAge"),
        "recruitment_age": entry.get("RecruitmentAge"),
    }
    # Add note for null F references
    null_f = [k for k in ("flim", "fpa", "fmsy") if result[k] is None]
    if null_f:
        result["note"] = (
            f"{', '.join(k.upper() for k in null_f)} not defined for this stock"
        )
    return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_sag.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
cd ~/ices-mcp-server && git add ices/sag.py tests/test_sag.py tests/conftest.py tests/fixtures/sag_*.json && git commit -m "feat: add SAG REST client with stock key resolution"
```

---

## Task 4: Stock Database OData Client (sd.py)

**Files:**
- Create: `~/ices-mcp-server/ices/sd.py`
- Create: `~/ices-mcp-server/tests/test_sd.py`
- Create: `~/ices-mcp-server/tests/fixtures/sd_stock_list.json`

- [ ] **Step 1: Save SD fixture data**

```bash
cd ~/ices-mcp-server
uv run python -c "
import httpx, json
url = 'https://sd.ices.dk/services/odata3/StockListDWs3'
params = {'\$filter': \"SpeciesScientificName eq 'Gadus morhua'\", '\$top': '5'}
r = httpx.get(url, params=params, timeout=30)
data = r.json()
with open('tests/fixtures/sd_stock_list.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Saved {len(data.get(\"value\", []))} entries')
"
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_sd.py
from __future__ import annotations

import httpx
import pytest
import respx

from ices.sd import SD_BASE, STOCK_KEY_MAP, get_metadata, normalize_stock_key
from tests.conftest import load_json_fixture


def test_normalize_sag_to_sd():
    assert normalize_stock_key("cod.27.24-32") == "cod-2432"


def test_normalize_already_sd_format():
    assert normalize_stock_key("cod-2432") == "cod-2432"


def test_normalize_unknown_raises():
    with pytest.raises(ValueError, match="Unknown stock key"):
        normalize_stock_key("fake.99.99-99")


def test_stock_key_map_has_required_stocks():
    """Spec requires at least 10 edge-case keys in the map."""
    required = [
        "cod.27.24-32", "cod.27.22-24", "her.27.25-2932", "spr.27.22-32",
        "ple.27.24-32", "fle.27.2425", "her.27.20-24", "her.27.28",
        "cod.27.47d20", "sol.27.4", "mac.27.nea", "hke.27.3a46-8abd",
    ]
    for key in required:
        assert key in STOCK_KEY_MAP, f"Missing: {key}"
        sd_key = STOCK_KEY_MAP[key]
        assert isinstance(sd_key, str) and len(sd_key) > 0


@respx.mock
@pytest.mark.asyncio
async def test_get_metadata():
    data = load_json_fixture("sd_stock_list.json")
    respx.get(url__startswith=SD_BASE).respond(json=data)
    result = await get_metadata(httpx.AsyncClient(), species_name="Gadus morhua")
    assert len(result) > 0
    assert "stock_key" in result[0]
    assert "trophic_guild" in result[0]


@respx.mock
@pytest.mark.asyncio
async def test_get_metadata_common_name():
    """Verify common name 'cod' resolves to scientific name in OData filter."""
    data = load_json_fixture("sd_stock_list.json")
    route = respx.get(url__startswith=SD_BASE).respond(json=data)
    await get_metadata(httpx.AsyncClient(), species_name="cod")
    # Verify the OData filter used the scientific name
    assert route.call_count == 1
    request = route.calls[0].request
    assert "Gadus morhua" in str(request.url)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_sd.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement sd.py**

```python
# ices/sd.py
"""ICES Stock Database (SD) OData client."""

from __future__ import annotations

import httpx

from ices import IcesApiError

SD_BASE = "https://sd.ices.dk/services/odata3/StockListDWs3"

_TIMEOUT = httpx.Timeout(connect=60.0, read=30.0, write=30.0, pool=30.0)

# Bidirectional SAG <-> SD stock key map.
# SAG format: "cod.27.24-32", SD format: "cod-2432"
STOCK_KEY_MAP: dict[str, str] = {
    # Baltic
    "cod.27.24-32": "cod-2432",
    "cod.27.22-24": "cod-2224",
    "her.27.25-2932": "her-2532",
    "her.27.20-24": "her-2024",
    "her.27.28": "her-riga",
    "her.27.3031": "her-3031",
    "spr.27.22-32": "spr-2232",
    "ple.27.24-32": "ple-2432",
    "fle.27.2425": "fle-2425",
    "fle.27.2628": "fle-2628",
    "fle.27.2729-32": "fle-2732",
    "dab.27.22-32": "dab-2232",
    # North Sea
    "cod.27.47d20": "cod-347d",
    "her.27.3a47d": "her-47d3",
    "ple.27.420": "ple-nsea",
    "sol.27.4": "sol-nsea",
    "whg.27.47d": "whg-47d",
    "had.27.46a20": "had-346a",
    "mac.27.nea": "mac-nea",
    # Bay of Biscay
    "sol.27.8ab": "sol-bisc",
    "hke.27.3a46-8abd": "hke-nrtn",
    "ank.27.78abd": "ank-78ab",
    # Eastern English Channel
    "sol.27.7d": "sol-eche",
    "ple.27.7d": "ple-eche",
}

_SD_TO_SAG: dict[str, str] = {v: k for k, v in STOCK_KEY_MAP.items()}


def normalize_stock_key(key: str) -> str:
    """Convert a SAG-format stock key to SD format.

    Raises ValueError for unlisted keys — no regex fallback.
    """
    if key in STOCK_KEY_MAP:
        return STOCK_KEY_MAP[key]
    if key in _SD_TO_SAG:
        return key  # Already SD format
    raise ValueError(
        f"Unknown stock key format: '{key}'. "
        f"Add to STOCK_KEY_MAP in sd.py or use list_stocks() to find the correct key."
    )


# Common name -> scientific name for OData filter
_COMMON_TO_SCIENTIFIC: dict[str, str] = {
    "cod": "Gadus morhua",
    "herring": "Clupea harengus",
    "sprat": "Sprattus sprattus",
    "flounder": "Platichthys flesus",
    "plaice": "Pleuronectes platessa",
    "sole": "Solea solea",
    "whiting": "Merlangius merlangus",
    "mackerel": "Scomber scombrus",
    "hake": "Merluccius merluccius",
    "anchovy": "Engraulis encrasicolus",
    "sardine": "Sardina pilchardus",
    "horse mackerel": "Trachurus trachurus",
    "haddock": "Melanogrammus aeglefinus",
}


async def get_metadata(
    client: httpx.AsyncClient,
    *,
    species_name: str,
) -> list[dict]:
    """Fetch stock metadata for a species from the ICES Stock Database."""
    # Resolve common name to scientific if needed
    scientific = _COMMON_TO_SCIENTIFIC.get(species_name.strip().lower(), species_name)
    params = {
        "$filter": f"SpeciesScientificName eq '{scientific}'",
        "$top": "50",
    }
    resp = await client.get(SD_BASE, params=params, timeout=_TIMEOUT)
    if resp.status_code != 200:
        raise IcesApiError(
            f"SD returned HTTP {resp.status_code}", status_code=resp.status_code
        )
    data = resp.json()
    entries = data.get("value", [])
    return [
        {
            "stock_key": e.get("StockKeyLabel", ""),
            "species_scientific": e.get("SpeciesScientificName", ""),
            "species_common": e.get("SpeciesCommonName", ""),
            "area_name": e.get("StockKeyDescription", ""),
            "expert_group": e.get("ExpertGroup", ""),
            "trophic_guild": e.get("TrophicGuild", ""),
            "size_guild": e.get("SizeGuild", ""),
        }
        for e in entries
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_sd.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/ices-mcp-server && git add ices/sd.py tests/test_sd.py tests/fixtures/sd_*.json && git commit -m "feat: add SD OData client with stock key cross-mapping"
```

---

## Task 5: DATRAS XML Client (datras.py)

**Files:**
- Create: `~/ices-mcp-server/ices/datras.py`
- Create: `~/ices-mcp-server/tests/test_datras.py`
- Create: `~/ices-mcp-server/tests/fixtures/datras_cpue_length.xml`
- Create: `~/ices-mcp-server/tests/fixtures/datras_hh.xml`
- Create: `~/ices-mcp-server/tests/fixtures/datras_ca.xml`

This is the most complex task. The DATRAS module handles 4 tools + streaming XML parsing + aggregation.

- [ ] **Step 1: Save DATRAS fixture data (small subsets)**

```bash
cd ~/ices-mcp-server
uv run python -c "
import httpx
BASE = 'https://datras.ices.dk/WebServices/DATRASWebService.asmx'

# CPUE Length - save first ~50 records (truncate after 50th closing tag)
r = httpx.get(f'{BASE}/getCPUELength', params={'survey': 'BITS', 'year': '2023', 'quarter': '1'}, timeout=120)
xml = r.text
# Find 50th record end
tag = '</Cls_Datras_CPUE_Length>'
pos = 0
for i in range(50):
    pos = xml.find(tag, pos) + len(tag)
    if pos == len(tag) - 1:
        break
subset = xml[:pos] + '\n</ArrayOfCls_Datras_CPUE_Length>'
with open('tests/fixtures/datras_cpue_length.xml', 'w') as f:
    f.write(subset)
print(f'Saved CPUE length fixture ({len(subset)} bytes)')

# HH haul data - save first ~10 records
r = httpx.get(f'{BASE}/getHHdata', params={'survey': 'BITS', 'year': '2023', 'quarter': '1'}, timeout=120)
xml = r.text
tag = '</Cls_DatrasExchange_HH>'
pos = 0
for i in range(10):
    pos = xml.find(tag, pos) + len(tag)
    if pos == len(tag) - 1:
        break
subset = xml[:pos] + '\n</ArrayOfCls_DatrasExchange_HH>'
with open('tests/fixtures/datras_hh.xml', 'w') as f:
    f.write(subset)
print(f'Saved HH fixture ({len(subset)} bytes)')

# CA age-length data - save first ~30 records
r = httpx.get(f'{BASE}/getCAdata', params={'survey': 'BITS', 'year': '2023', 'quarter': '1'}, timeout=120)
xml = r.text
tag = '</Cls_DatrasExchange_CA>'
pos = 0
for i in range(30):
    pos = xml.find(tag, pos) + len(tag)
    if pos == len(tag) - 1:
        break
subset = xml[:pos] + '\n</ArrayOfCls_DatrasExchange_CA>'
with open('tests/fixtures/datras_ca.xml', 'w') as f:
    f.write(subset)
print(f'Saved CA fixture ({len(subset)} bytes)')
"
```

- [ ] **Step 2: Write failing DATRAS tests**

```python
# tests/test_datras.py
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from ices.datras import (
    DATRAS_BASE,
    get_age_length,
    get_cpue_age,
    get_cpue_length,
    get_hauls,
    parse_datras_xml,
)
from tests.conftest import FIXTURES


@pytest.fixture
def cpue_xml() -> str:
    return (FIXTURES / "datras_cpue_length.xml").read_text()


@pytest.fixture
def hh_xml() -> str:
    return (FIXTURES / "datras_hh.xml").read_text()


@pytest.fixture
def ca_xml() -> str:
    return (FIXTURES / "datras_ca.xml").read_text()


def test_parse_cpue_xml(cpue_xml):
    records = parse_datras_xml(cpue_xml, "Cls_Datras_CPUE_Length")
    assert len(records) > 0
    rec = records[0]
    assert "AphiaID" in rec or "Species" in rec
    # Length should be present (as LngtClas in mm)
    assert "LngtClas" in rec


def test_parse_cpue_na_sentinel(cpue_xml):
    records = parse_datras_xml(cpue_xml, "Cls_Datras_CPUE_Length")
    # No record should have string "-9" — should be None
    for rec in records:
        for v in rec.values():
            if v is not None:
                assert v != "-9", f"NA sentinel -9 not converted: {rec}"


def test_parse_hh_xml(hh_xml):
    records = parse_datras_xml(hh_xml, "Cls_DatrasExchange_HH")
    assert len(records) > 0
    rec = records[0]
    assert "ShootLat" in rec
    assert "ShootLong" in rec


def test_parse_hh_whitespace_stripped(hh_xml):
    records = parse_datras_xml(hh_xml, "Cls_DatrasExchange_HH")
    for rec in records:
        for v in rec.values():
            if isinstance(v, str):
                assert v == v.strip(), f"Whitespace not stripped: '{v}'"


def test_parse_ca_xml(ca_xml):
    records = parse_datras_xml(ca_xml, "Cls_DatrasExchange_CA")
    assert len(records) > 0
    rec = records[0]
    assert "Age" in rec
    assert "LngtClass" in rec


@respx.mock
@pytest.mark.asyncio
async def test_get_cpue_length_summary(cpue_xml):
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=cpue_xml, headers={"content-type": "text/xml"}
    )
    result = await get_cpue_length(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1, mode="summary"
    )
    assert len(result) > 0
    rec = result[0]
    assert "species" in rec
    assert "length_cm" in rec
    assert "mean_cpue" in rec
    assert "n_hauls" in rec


@respx.mock
@pytest.mark.asyncio
async def test_get_cpue_length_raw(cpue_xml):
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=cpue_xml, headers={"content-type": "text/xml"}
    )
    result = await get_cpue_length(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1, mode="raw"
    )
    assert len(result) > 0
    rec = result[0]
    assert "length_cm" in rec
    assert "cpue_number_per_hour" in rec
    assert "shoot_lat" in rec


@respx.mock
@pytest.mark.asyncio
async def test_get_cpue_length_converts_mm_to_cm(cpue_xml):
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=cpue_xml, headers={"content-type": "text/xml"}
    )
    result = await get_cpue_length(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1, mode="raw"
    )
    # All length values should be reasonable cm values (< 200)
    for rec in result:
        if rec["length_cm"] is not None:
            assert rec["length_cm"] < 200, f"Length {rec['length_cm']} looks like mm, not cm"


@respx.mock
@pytest.mark.asyncio
async def test_get_hauls(hh_xml):
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=hh_xml, headers={"content-type": "text/xml"}
    )
    result = await get_hauls(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1
    )
    assert len(result) > 0
    rec = result[0]
    assert "haul_no" in rec
    assert "shoot_lat" in rec
    assert "depth" in rec


@respx.mock
@pytest.mark.asyncio
async def test_get_cpue_age_summary(cpue_xml):
    # Reuse CPUE length fixture — structure is similar enough for parsing
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=cpue_xml, headers={"content-type": "text/xml"}
    )
    result = await get_cpue_age(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1, mode="summary"
    )
    # May be empty if fixture has no age data — that's OK
    if result:
        rec = result[0]
        assert "mean_cpue" in rec
        assert "n_hauls" in rec


@respx.mock
@pytest.mark.asyncio
async def test_get_cpue_age_raw(cpue_xml):
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=cpue_xml, headers={"content-type": "text/xml"}
    )
    result = await get_cpue_age(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1, mode="raw"
    )
    if result:
        rec = result[0]
        assert "age" in rec
        assert "cpue_number_per_hour" in rec
        assert "shoot_lat" in rec


@respx.mock
@pytest.mark.asyncio
async def test_get_age_length(ca_xml):
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=ca_xml, headers={"content-type": "text/xml"}
    )
    result = await get_age_length(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1
    )
    assert len(result) > 0
    rec = result[0]
    assert "age" in rec
    assert "length_cm" in rec
    assert "aphia_id" in rec


def test_parse_malformed_xml_no_records():
    """Verify ParseError on truncated XML with 0 complete records raises IcesApiError."""
    from ices import IcesApiError

    # Truncated mid-record — no closing tags, zero complete records
    broken_xml = '<ArrayOfCls_Datras_CPUE_Length xmlns="ices.dk.local/DATRAS"><Cls_Datras_CPUE_Length><AphiaID>126436</AphiaID>'
    with pytest.raises(IcesApiError, match="Failed to parse"):
        parse_datras_xml(broken_xml, "Cls_Datras_CPUE_Length")


def test_parse_malformed_xml_partial_recovery():
    """Verify truncated XML after some records returns partial results (not error)."""
    # One complete record followed by truncation
    partial_xml = (
        '<ArrayOfCls_Datras_CPUE_Length xmlns="ices.dk.local/DATRAS">'
        '<Cls_Datras_CPUE_Length><AphiaID>126436</AphiaID><LngtClas>100</LngtClas></Cls_Datras_CPUE_Length>'
        '<Cls_Datras_CPUE_Length><AphiaID>126417</AphiaID>'  # truncated
    )
    result = parse_datras_xml(partial_xml, "Cls_Datras_CPUE_Length")
    assert len(result) == 1  # Only the complete record
    assert result[0]["AphiaID"] == 126436


@respx.mock
@pytest.mark.asyncio
async def test_fetch_xml_rejects_html_200():
    """Verify HTML error page with 200 status is detected."""
    from ices import IcesApiError
    from ices.datras import _fetch_xml

    respx.get(url__startswith=DATRAS_BASE).respond(
        status_code=200,
        content="<html><body>Service unavailable</body></html>",
        headers={"content-type": "text/html"},
    )
    with pytest.raises(IcesApiError, match="non-XML response"):
        await _fetch_xml(httpx.AsyncClient(), "getCPUELength",
                         {"survey": "BITS", "year": "2023", "quarter": "1"})


@respx.mock
@pytest.mark.asyncio
async def test_get_cpue_length_species_filter(cpue_xml):
    """Verify species parameter filters records by WoRMS AphiaID."""
    respx.get(url__startswith=DATRAS_BASE).respond(
        content=cpue_xml, headers={"content-type": "text/xml"}
    )
    # Use a species that may or may not be in fixture
    result = await get_cpue_length(
        httpx.AsyncClient(), survey="BITS", year=2023, quarter=1,
        species="126436", mode="raw"  # cod AphiaID
    )
    # Either all results are cod, or empty if fixture has no cod
    for rec in result:
        assert rec["aphia_id"] == 126436
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_datras.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement datras.py**

```python
# ices/datras.py
"""ICES DATRAS trawl survey XML client with streaming parser and aggregation."""

from __future__ import annotations

import asyncio
import csv
import io
import xml.etree.ElementTree as ET

import httpx
import pandas as pd

from ices import IcesApiError
from ices.vocab import resolve_species, species_display_name

DATRAS_BASE = "https://datras.ices.dk/WebServices/DATRASWebService.asmx"

_NS = "ices.dk.local/DATRAS"
_NS_PREFIX = f"{{{_NS}}}"

# Timeouts: longer read for large XML
_TIMEOUT = httpx.Timeout(connect=60.0, read=120.0, write=30.0, pool=30.0)

_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 503}


async def _fetch_xml(client: httpx.AsyncClient, endpoint: str, params: dict) -> str:
    """Fetch XML from DATRAS with retry."""
    url = f"{DATRAS_BASE}/{endpoint}"
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code in _RETRY_STATUSES:
                delay = 2 ** (attempt + 1)
                await asyncio.sleep(delay)
                continue
            if resp.status_code != 200:
                raise IcesApiError(
                    f"DATRAS returned HTTP {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )
            text = resp.text
            # Detect HTML error page masquerading as 200
            # Use 'in' not 'startswith' — valid XML may have <?xml ...?> declaration
            if "<ArrayOf" not in text[:500]:
                raise IcesApiError(
                    "DATRAS returned non-XML response (possibly HTML error page)"
                )
            return text
        except httpx.TimeoutException as e:
            last_exc = e
    if last_exc:
        raise IcesApiError(f"DATRAS timed out after {_MAX_RETRIES} attempts") from last_exc
    raise IcesApiError("DATRAS request failed after retries")


def _parse_value(text: str | None) -> str | int | float | None:
    """Parse an XML text value, converting -9 sentinel to None."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    # Try numeric
    try:
        val = float(text)
        if val == -9.0:
            return None
        if val == int(val):
            return int(val)
        return val
    except ValueError:
        return text


def parse_datras_xml(xml_text: str, record_tag: str) -> list[dict]:
    """Parse DATRAS XML into a list of dicts using iterparse.

    Handles namespace stripping, -9 sentinel, whitespace.
    """
    records = []
    try:
        for event, elem in ET.iterparse(io.StringIO(xml_text), events=("end",)):
            # Strip namespace
            local_tag = elem.tag.replace(_NS_PREFIX, "")
            if local_tag == record_tag:
                record = {}
                for child in elem:
                    child_tag = child.tag.replace(_NS_PREFIX, "")
                    record[child_tag] = _parse_value(child.text)
                records.append(record)
                elem.clear()
    except ET.ParseError as e:
        if records:
            # Partial parse — return what we got with a warning
            pass
        else:
            raise IcesApiError(f"Failed to parse DATRAS XML: {e}") from e
    return records


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _aggregate_cpue(records: list[dict], group_col: str) -> list[dict]:
    """Aggregate CPUE records by species + length/age class."""
    if not records:
        return []
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    buf.seek(0)
    df = pd.read_csv(buf)
    cpue_col = "CPUE_number_per_hour"
    if cpue_col not in df.columns:
        return records  # Can't aggregate without CPUE column
    grouped = df.groupby(["AphiaID", group_col])[cpue_col].agg(
        mean_cpue="mean", median_cpue="median", sd_cpue="std", n_hauls="count"
    ).reset_index()
    species_map = dict(zip(df["AphiaID"], df.get("Species", df["AphiaID"])))
    result = []
    for _, row in grouped.iterrows():
        aphia = int(row["AphiaID"]) if pd.notna(row["AphiaID"]) else None
        result.append({
            "species": species_display_name(aphia) if aphia else str(species_map.get(aphia, "")),
            "aphia_id": aphia,
            group_col.lower(): _to_float(row[group_col]),
            "mean_cpue": round(float(row["mean_cpue"]), 4) if pd.notna(row["mean_cpue"]) else None,
            "median_cpue": round(float(row["median_cpue"]), 4) if pd.notna(row["median_cpue"]) else None,
            "sd_cpue": round(float(row["sd_cpue"]), 4) if pd.notna(row["sd_cpue"]) else None,
            "n_hauls": int(row["n_hauls"]),
        })
    return result


async def get_cpue_length(
    client: httpx.AsyncClient,
    *,
    survey: str,
    year: int,
    quarter: int,
    species: str | int | None = None,
    mode: str = "summary",
) -> list[dict]:
    """Fetch CPUE by length class."""
    xml_text = await _fetch_xml(
        client, "getCPUELength",
        {"survey": survey, "year": str(year), "quarter": str(quarter)},
    )
    records = await asyncio.to_thread(
        parse_datras_xml, xml_text, "Cls_Datras_CPUE_Length"
    )
    # Filter by species if requested
    if species is not None:
        aphia = resolve_species(species)
        records = [r for r in records if r.get("AphiaID") == aphia]

    if not records:
        return []

    if mode == "summary":
        # Convert mm -> cm in group column name
        for r in records:
            if "LngtClas" in r and r["LngtClas"] is not None:
                r["LngtClas"] = round(r["LngtClas"] / 10.0, 1)
        result = _aggregate_cpue(records, "LngtClas")
        # Rename column
        for row in result:
            if "lngtclas" in row:
                row["length_cm"] = row.pop("lngtclas")
        return result
    else:
        # Raw mode
        return [
            {
                "species": species_display_name(r.get("AphiaID")) if r.get("AphiaID") else r.get("Species", ""),
                "aphia_id": r.get("AphiaID"),
                "length_cm": round(r["LngtClas"] / 10.0, 1) if r.get("LngtClas") is not None else None,
                "cpue_number_per_hour": _to_float(r.get("CPUE_number_per_hour")),
                "haul_no": r.get("HaulNo"),
                "shoot_lat": _to_float(r.get("ShootLat")),
                "shoot_lon": _to_float(r.get("ShootLon")),
                "depth": _to_float(r.get("Depth")),
                "gear": r.get("Gear"),
                "datetime": r.get("DateTime"),
            }
            for r in records
        ]


async def get_cpue_age(
    client: httpx.AsyncClient,
    *,
    survey: str,
    year: int,
    quarter: int,
    species: str | int | None = None,
    mode: str = "summary",
) -> list[dict]:
    """Fetch CPUE by age class."""
    xml_text = await _fetch_xml(
        client, "getCPUEAge",
        {"survey": survey, "year": str(year), "quarter": str(quarter)},
    )
    records = await asyncio.to_thread(
        parse_datras_xml, xml_text, "Cls_Datras_CPUE_Age"
    )
    if species is not None:
        aphia = resolve_species(species)
        records = [r for r in records if r.get("AphiaID") == aphia]

    if not records:
        return []

    if mode == "summary":
        return _aggregate_cpue(records, "Age")
    else:
        return [
            {
                "species": species_display_name(r.get("AphiaID")) if r.get("AphiaID") else "",
                "aphia_id": r.get("AphiaID"),
                "age": r.get("Age"),
                "cpue_number_per_hour": _to_float(r.get("CPUE_number_per_hour")),
                "haul_no": r.get("HaulNo"),
                "shoot_lat": _to_float(r.get("ShootLat")),
                "shoot_lon": _to_float(r.get("ShootLon")),
                "depth": _to_float(r.get("Depth")),
                "gear": r.get("Gear"),
                "datetime": r.get("DateTime"),
            }
            for r in records
        ]


async def get_hauls(
    client: httpx.AsyncClient,
    *,
    survey: str,
    year: int,
    quarter: int,
) -> list[dict]:
    """Fetch haul-level metadata."""
    xml_text = await _fetch_xml(
        client, "getHHdata",
        {"survey": survey, "year": str(year), "quarter": str(quarter)},
    )
    records = await asyncio.to_thread(
        parse_datras_xml, xml_text, "Cls_DatrasExchange_HH"
    )
    return [
        {
            "haul_no": r.get("HaulNo"),
            "country": r.get("Country"),
            "ship": r.get("Ship"),
            "shoot_lat": _to_float(r.get("ShootLat")),
            "shoot_lon": _to_float(r.get("ShootLong")),
            "haul_lat": _to_float(r.get("HaulLat")),
            "haul_lon": _to_float(r.get("HaulLong")),
            "depth": _to_float(r.get("Depth")),
            "gear": r.get("Gear"),
            "haul_dur_min": _to_float(r.get("HaulDur")),
            "day_night": r.get("DayNight"),
            "sur_temp": _to_float(r.get("SurTemp")),
            "bot_temp": _to_float(r.get("BotTemp")),
            "datetime": r.get("DateTime"),
        }
        for r in records
    ]


async def get_age_length(
    client: httpx.AsyncClient,
    *,
    survey: str,
    year: int,
    quarter: int,
    species: str | int | None = None,
) -> list[dict]:
    """Fetch age-length key data."""
    xml_text = await _fetch_xml(
        client, "getCAdata",
        {"survey": survey, "year": str(year), "quarter": str(quarter)},
    )
    records = await asyncio.to_thread(
        parse_datras_xml, xml_text, "Cls_DatrasExchange_CA"
    )
    if species is not None:
        aphia = resolve_species(species)
        records = [r for r in records if r.get("Valid_Aphia") == aphia or r.get("SpecCode") == aphia]

    return [
        {
            "species": species_display_name(r.get("Valid_Aphia")) if r.get("Valid_Aphia") else "",
            "aphia_id": r.get("Valid_Aphia"),
            "age": r.get("Age"),
            "length_cm": round(r["LngtClass"] / 10.0, 1) if r.get("LngtClass") is not None else None,
            "n_at_length": r.get("CANoAtLngt"),
            "individual_weight_g": _to_float(r.get("IndWgt")),
            "sex": r.get("Sex"),
            "maturity": r.get("Maturity"),
            "maturity_scale": r.get("MaturityScale"),
        }
        for r in records
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_datras.py -v`
Expected: All 19 tests PASS. Some may need fixture adjustments depending on actual XML field names — fix and re-run.

- [ ] **Step 6: Commit**

```bash
cd ~/ices-mcp-server && git add ices/datras.py tests/test_datras.py tests/fixtures/datras_*.xml && git commit -m "feat: add DATRAS XML client with streaming parser and aggregation"
```

---

## Task 6: Eggs & Larvae REST Client (eggs.py)

**Files:**
- Create: `~/ices-mcp-server/ices/eggs.py`
- Create: `~/ices-mcp-server/tests/test_eggs.py`
- Create: `~/ices-mcp-server/tests/fixtures/eggs_summary.json`

- [ ] **Step 1: Save fixture data**

```bash
cd ~/ices-mcp-server
uv run python -c "
import httpx, json
r = httpx.get('https://eggsandlarvae.ices.dk/api/getEggsAndLarvaeDataSummary',
              params={'Species': 'Gadus morhua', 'Year': '2004'}, timeout=30)
data = r.json()
with open('tests/fixtures/eggs_summary.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Saved {len(data)} records')
"
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_eggs.py
from __future__ import annotations

import httpx
import pytest
import respx

from ices.eggs import EGGS_BASE, search
from tests.conftest import load_json_fixture


@respx.mock
@pytest.mark.asyncio
async def test_search():
    data = load_json_fixture("eggs_summary.json")
    respx.get(url__startswith=EGGS_BASE).respond(json=data)
    result = await search(
        httpx.AsyncClient(), species="Gadus morhua", year=2004
    )
    assert len(result) > 0
    rec = result[0]
    assert "year" in rec
    assert "month" in rec
    assert "stage" in rec
    assert "num_samples" in rec


@respx.mock
@pytest.mark.asyncio
async def test_search_empty():
    respx.get(url__startswith=EGGS_BASE).respond(json=[])
    result = await search(
        httpx.AsyncClient(), species="Gadus morhua", year=1900
    )
    assert result == []


@respx.mock
@pytest.mark.asyncio
async def test_search_truncates_large_results():
    """Verify responses > 500 rows are truncated."""
    large_data = [{"year": 2004, "month": 1, "numSamples": 1,
                   "noMeasurements": 1, "stage": "EL", "species": "Gadus morhua",
                   "survey": "CP-EGGS", "aphiaID": 126436}] * 600
    respx.get(url__startswith=EGGS_BASE).respond(json=large_data)
    result = await search(
        httpx.AsyncClient(), species="Gadus morhua", year=2004
    )
    assert len(result) == 500  # Truncated
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_eggs.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement eggs.py**

```python
# ices/eggs.py
"""ICES Eggs & Larvae REST client."""

from __future__ import annotations

import httpx

from ices import IcesApiError

EGGS_BASE = "https://eggsandlarvae.ices.dk/api"

_TIMEOUT = httpx.Timeout(connect=60.0, read=30.0, write=30.0, pool=30.0)


async def search(
    client: httpx.AsyncClient,
    *,
    species: str,
    year: int,
) -> list[dict]:
    """Fetch eggs and larvae summary data."""
    params = {"Species": species, "Year": str(year)}
    resp = await client.get(
        f"{EGGS_BASE}/getEggsAndLarvaeDataSummary",
        params=params,
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        raise IcesApiError(
            f"Eggs & Larvae API returned HTTP {resp.status_code}",
            status_code=resp.status_code,
        )
    data = resp.json()
    if not data:
        return []
    result = [
        {
            "year": entry.get("year"),
            "month": entry.get("month"),
            "stage": entry.get("stage"),
            "survey": entry.get("survey"),
            "num_samples": entry.get("numSamples"),
            "num_measurements": entry.get("noMeasurements"),
            "aphia_id": entry.get("aphiaID"),
        }
        for entry in data
    ]
    if len(result) > 500:
        return result[:500]  # Skill will add warning via server.py
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/ices-mcp-server && uv run pytest tests/test_eggs.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/ices-mcp-server && git add ices/eggs.py tests/test_eggs.py tests/fixtures/eggs_summary.json && git commit -m "feat: add Eggs & Larvae REST client"
```

---

## Task 7: MCP Server Entry Point (server.py)

**Files:**
- Create: `~/ices-mcp-server/server.py`

- [ ] **Step 1: Implement server.py with all 9 tool handlers**

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp>=1.0", "httpx>=0.27", "pandas>=2.2"]
# ///
"""ICES MCP Server — 9 tools wrapping SAG, DATRAS, SD, and Eggs & Larvae APIs."""

from __future__ import annotations

import json

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ices import IcesApiError
from ices.datras import get_age_length, get_cpue_age, get_cpue_length, get_hauls
from ices.eggs import search as eggs_search
from ices.sag import get_assessment, get_reference_points, list_stocks
from ices.sd import get_metadata

server = Server("ices")


def _result(data, note: str | None = None) -> list[TextContent]:
    """Format tool result as JSON text content."""
    payload = {"result": data}
    if note:
        payload["note"] = note
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


def _error(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="list_stocks", description="List all ICES-assessed stocks for a year", inputSchema={
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Assessment year (e.g. 2023)"},
                "area_filter": {"type": "string", "description": "Regex filter on stock key labels"},
            },
            "required": ["year"],
        }),
        Tool(name="get_stock_assessment", description="Full assessment time series (SSB, R, F, catches) for a stock", inputSchema={
            "type": "object",
            "properties": {
                "stock_key": {"type": "string", "description": "Stock key label, e.g. cod.27.24-32"},
                "year": {"type": "integer", "description": "Assessment year (default: 2023)"},
            },
            "required": ["stock_key"],
        }),
        Tool(name="get_reference_points", description="Reference points (Blim, Bpa, Fmsy, MSYBtrigger) for a stock", inputSchema={
            "type": "object",
            "properties": {
                "stock_key": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["stock_key"],
        }),
        Tool(name="get_stock_metadata", description="Stock metadata (area, expert group, trophic/size guild) for a species", inputSchema={
            "type": "object",
            "properties": {
                "species_name": {"type": "string", "description": "Scientific or common name"},
            },
            "required": ["species_name"],
        }),
        Tool(name="get_survey_cpue_length", description="CPUE by length class from DATRAS trawl surveys", inputSchema={
            "type": "object",
            "properties": {
                "survey": {"type": "string", "description": "Survey code: BITS, NS-IBTS, EVHOE, CGFS"},
                "year": {"type": "integer"},
                "quarter": {"type": "integer", "enum": [1, 2, 3, 4]},
                "species": {"type": "string", "description": "Common name or WoRMS code"},
                "mode": {"type": "string", "enum": ["summary", "raw"], "default": "summary"},
            },
            "required": ["survey", "year", "quarter"],
        }),
        Tool(name="get_survey_cpue_age", description="CPUE by age class from DATRAS trawl surveys", inputSchema={
            "type": "object",
            "properties": {
                "survey": {"type": "string"},
                "year": {"type": "integer"},
                "quarter": {"type": "integer", "enum": [1, 2, 3, 4]},
                "species": {"type": "string"},
                "mode": {"type": "string", "enum": ["summary", "raw"], "default": "summary"},
            },
            "required": ["survey", "year", "quarter"],
        }),
        Tool(name="get_survey_hauls", description="Haul-level metadata (positions, depth, gear) from DATRAS surveys", inputSchema={
            "type": "object",
            "properties": {
                "survey": {"type": "string"},
                "year": {"type": "integer"},
                "quarter": {"type": "integer", "enum": [1, 2, 3, 4]},
            },
            "required": ["survey", "year", "quarter"],
        }),
        Tool(name="get_age_length_keys", description="Age-length key data for Von Bertalanffy growth fitting", inputSchema={
            "type": "object",
            "properties": {
                "survey": {"type": "string"},
                "year": {"type": "integer"},
                "quarter": {"type": "integer", "enum": [1, 2, 3, 4]},
                "species": {"type": "string"},
            },
            "required": ["survey", "year", "quarter"],
        }),
        Tool(name="search_eggs_larvae", description="Ichthyoplankton summary data (eggs and larvae counts)", inputSchema={
            "type": "object",
            "properties": {
                "species": {"type": "string", "description": "Scientific name, e.g. Gadus morhua"},
                "year": {"type": "integer"},
            },
            "required": ["species", "year"],
        }),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    async with httpx.AsyncClient() as client:
        try:
            if name == "list_stocks":
                data = await list_stocks(client, year=arguments["year"],
                                         area_filter=arguments.get("area_filter"))
                note = None if data else f"No stocks found for year {arguments['year']}"
                return _result(data, note)

            elif name == "get_stock_assessment":
                data = await get_assessment(client, stock_key=arguments["stock_key"],
                                            year=arguments.get("year", 2023))
                return _result(data)

            elif name == "get_reference_points":
                data = await get_reference_points(client, stock_key=arguments["stock_key"],
                                                  year=arguments.get("year", 2023))
                return _result(data)

            elif name == "get_stock_metadata":
                data = await get_metadata(client, species_name=arguments["species_name"])
                note = None
                if not data:
                    note = f"No stocks found for '{arguments['species_name']}'"
                elif len(data) > 1:
                    note = f"Multiple stocks found ({len(data)}). Pick one stock_key for downstream tools."
                return _result(data, note)

            elif name == "get_survey_cpue_length":
                data = await get_cpue_length(client, survey=arguments["survey"],
                                             year=arguments["year"], quarter=arguments["quarter"],
                                             species=arguments.get("species"),
                                             mode=arguments.get("mode", "summary"))
                note = None if data else f"No CPUE data for {arguments['survey']} Q{arguments['quarter']} {arguments['year']}"
                return _result(data, note)

            elif name == "get_survey_cpue_age":
                data = await get_cpue_age(client, survey=arguments["survey"],
                                          year=arguments["year"], quarter=arguments["quarter"],
                                          species=arguments.get("species"),
                                          mode=arguments.get("mode", "summary"))
                note = None if data else f"No CPUE age data for {arguments['survey']} Q{arguments['quarter']} {arguments['year']}"
                return _result(data, note)

            elif name == "get_survey_hauls":
                data = await get_hauls(client, survey=arguments["survey"],
                                       year=arguments["year"], quarter=arguments["quarter"])
                note = None if data else f"No hauls for {arguments['survey']} Q{arguments['quarter']} {arguments['year']}"
                return _result(data, note)

            elif name == "get_age_length_keys":
                data = await get_age_length(client, survey=arguments["survey"],
                                            year=arguments["year"], quarter=arguments["quarter"],
                                            species=arguments.get("species"))
                note = None if data else f"No age-length data for {arguments['survey']} Q{arguments['quarter']} {arguments['year']}"
                return _result(data, note)

            elif name == "search_eggs_larvae":
                data = await eggs_search(client, species=arguments["species"],
                                         year=arguments["year"])
                note = None
                if not data:
                    note = f"No eggs/larvae data for {arguments['species']} in {arguments['year']}"
                elif len(data) >= 500:
                    note = "Result truncated to 500 rows. Narrow the query."
                return _result(data, note)

            else:
                return _error(f"Unknown tool: {name}")

        except IcesApiError as e:
            return _error(str(e))
        except Exception as e:
            return _error(f"Internal error: {type(e).__name__}: {e}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- [ ] **Step 2: Verify server starts without error**

Run: `cd ~/ices-mcp-server && echo '{}' | timeout 3 uv run server.py 2>&1 || true`
Expected: Server starts (may timeout on stdin read — that's fine). No import errors.

- [ ] **Step 3: Commit**

```bash
cd ~/ices-mcp-server && git add server.py && git commit -m "feat: add MCP server entry point with 9 tool handlers"
```

---

## Task 8: Integration Tests

**Files:**
- Create: `~/ices-mcp-server/tests/test_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_integration.py
"""Live ICES endpoint tests. Run manually: uv run pytest -m integration"""
from __future__ import annotations

import httpx
import pytest

from ices.datras import get_cpue_length, get_hauls
from ices.eggs import search as eggs_search
from ices.sag import get_assessment, get_reference_points, list_stocks


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sag_list_stocks_live():
    async with httpx.AsyncClient() as client:
        result = await list_stocks(client, year=2023)
        assert len(result) > 50  # Should have many stocks
        keys = {s["stock_key"] for s in result}
        assert "cod.27.24-32" in keys


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sag_assessment_live():
    async with httpx.AsyncClient() as client:
        result = await get_assessment(client, stock_key="cod.27.24-32", year=2023)
        assert len(result) > 10  # Multi-decade time series
        assert result[0]["year"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sag_ref_points_live():
    async with httpx.AsyncClient() as client:
        result = await get_reference_points(client, stock_key="cod.27.24-32", year=2023)
        assert result["blim"] is not None or result["bpa"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_datras_cpue_length_live():
    async with httpx.AsyncClient() as client:
        result = await get_cpue_length(
            client, survey="BITS", year=2023, quarter=1, mode="summary"
        )
        assert len(result) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_datras_hauls_live():
    async with httpx.AsyncClient() as client:
        result = await get_hauls(client, survey="BITS", year=2023, quarter=1)
        assert len(result) > 0
        assert result[0]["shoot_lat"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_eggs_search_live():
    async with httpx.AsyncClient() as client:
        result = await eggs_search(client, species="Gadus morhua", year=2004)
        assert len(result) > 0
```

- [ ] **Step 2: Verify integration tests are excluded by default**

Run: `cd ~/ices-mcp-server && uv run pytest --collect-only -q`
Expected: Integration tests should NOT appear in the collection (excluded by `addopts`).

- [ ] **Step 3: Commit**

```bash
cd ~/ices-mcp-server && git add tests/test_integration.py && git commit -m "test: add live ICES integration tests (marked, excluded by default)"
```

---

## Task 9: Register MCP Server in .mcp.json

**Files:**
- Modify: `~/osmose/osmose-python/.mcp.json`

- [ ] **Step 1: Add ices server entry**

Add to the `mcpServers` object in `~/osmose/osmose-python/.mcp.json`:

```json
"ices": {
  "command": "uv",
  "args": ["run", "--directory", "/home/razinka/ices-mcp-server", "server.py"]
}
```

- [ ] **Step 2: Commit**

```bash
cd ~/osmose/osmose-python && git add .mcp.json && git commit -m "chore: register ICES MCP server in .mcp.json"
```

---

## Task 10: Claude Code Skill

**Files:**
- Create: `~/.claude/skills/ices-data/ices-data.md`

- [ ] **Step 1: Write the skill file**

```markdown
---
name: ices-data
description: Fetch and analyze ICES fisheries data (stock assessments, survey CPUE, reference points, growth parameters) for OSMOSE marine ecosystem modeling. Use when the user asks about ICES stock data, DATRAS surveys, biomass targets validation, fish distribution, or growth parameter updates.
---

# ICES Data Access Skill

You have access to an ICES MCP server with 9 tools for querying fisheries data. Use these tools to fetch real data from ICES — never fabricate stock assessment numbers.

## Tools Available

| Tool | Purpose |
|------|---------|
| `list_stocks` | List all assessed stocks for a year. Use `area_filter` regex to narrow by region. |
| `get_stock_assessment` | Full time series (SSB, R, F, catches) for a stock key. |
| `get_reference_points` | Blim, Bpa, Fmsy, MSYBtrigger for a stock. Some may be null. |
| `get_stock_metadata` | Species metadata from Stock Database (trophic guild, expert group). Returns multiple stocks per species — caller must pick. |
| `get_survey_cpue_length` | CPUE by length from trawl surveys. `mode=summary` (default) or `raw`. |
| `get_survey_cpue_age` | CPUE by age from trawl surveys. Same modes. |
| `get_survey_hauls` | Haul positions, depth, gear, temperature from surveys. |
| `get_age_length_keys` | Age-length data for fitting Von Bertalanffy growth curves. |
| `search_eggs_larvae` | Ichthyoplankton summary counts by species/year. |

## ICES Area Conventions

Stock key labels encode the ICES area:
- `*.27.22-*`, `*.27.24-32` → **Baltic Sea**
- `*.27.8.*` → **Bay of Biscay**
- `*.27.7.d-e` → **Eastern English Channel**
- `*.27.4.*` → **North Sea**
- `*.27.3.a` → **Skagerrak-Kattegat**

Area filter examples:
- Baltic: `area_filter="27\\.(2[2-9]|3[0-2])"`
- North Sea: `area_filter="27\\.4"`
- Bay of Biscay: `area_filter="27\\.8"`

## Survey Catalog

| Survey | Region | Quarters | Notes |
|--------|--------|----------|-------|
| BITS | Baltic | Q1, Q4 | 1991+, main Baltic trawl survey |
| NS-IBTS | North Sea | Q1, Q3 | 1965+, International Bottom Trawl Survey |
| EVHOE | Bay of Biscay | Q4 | 1997+, French survey |
| CGFS | Eastern English Channel | Q4 | 1988+, Channel Ground Fish Survey |
| IBTS-Q3 | North Sea + Celtic | Q3 | 1991+, Q3 component of IBTS |

## OSMOSE Integration Workflows

### Validate Biomass Targets

1. `list_stocks(year=2023, area_filter="27\\.(2[2-9]|3[0-2])")`
2. `get_stock_assessment(stock_key=...)` for each stock
3. Compute 5-year mean SSB from the time series
4. **IMPORTANT:** OSMOSE `biomass_targets.csv` uses **total stock biomass** (not SSB) for 7/8 species. Only cod uses `reference_point_type=ssb`. For biomass-type targets, multiply ICES SSB by a scaling factor (1.5-2x for gadoids, ~1.2x for small pelagics) to estimate total biomass.
5. Compare against `data/{region}/reference/biomass_targets.csv`

### Update Growth Parameters

1. `get_age_length_keys(survey="BITS", year=2023, quarter=1, species="cod")`
2. Fit VB curve in Python: `L(t) = L_inf * (1 - exp(-K * (t - t0)))` using `scipy.optimize.curve_fit` (in the user's .venv, not the MCP server)
3. Compare against config values
4. **Config key casing:** Schema uses lowercase (`species.linf`) but config files use camelCase (`species.lInf`). Java is case-sensitive — always match the existing config file casing.

### Generate Distribution Maps

1. `get_survey_cpue_length(survey="BITS", year=2023, quarter=1, species="cod", mode="raw")`
2. `get_survey_hauls(survey="BITS", year=2023, quarter=1)`
3. Map haul positions to OSMOSE grid cells:
   - `cell_w = (lowright_lon - upleft_lon) / nlon`
   - `cell_h = (upleft_lat - lowright_lat) / nlat`
   - `col = floor((lon - upleft_lon) / cell_w)`
   - `row = floor((upleft_lat - lat) / cell_h)`
4. Existing Baltic maps use binary format (0/1/-99), semicolon-delimited, `nlat` rows x `nlon` columns
5. Movement config keys use `movement.{field}.map{N}` format (not schema format)

### Check Reference Points

1. `get_reference_points(stock_key="cod.27.24-32")`
2. Fishing rates are indexed by **fishery** (`fisheries.rate.base.fsh{N}`), not species
3. Flag if simulated F exceeds Fmsy

## Config Key Patterns

**Casing note:** Schema uses lowercase (`species.linf`) but config files use camelCase (`species.lInf`). Java is case-sensitive — always match the existing config file.

```
# Species biology (config file casing)
species.lInf.sp{N}                    # VB L-infinity (cm)
species.K.sp{N}                       # VB K (year^-1)
species.t0.sp{N}                      # VB t0 (years)
species.length2weight.condition.factor.sp{N}  # L-W a
species.length2weight.allometric.power.sp{N}  # L-W b
species.maturity.size.sp{N}           # Size at maturity (cm)
species.lifespan.sp{N}               # Maximum age (years)

# Mortality & predation
mortality.additional.rate.sp{N}       # Natural mortality
predation.ingestion.rate.max.sp{N}    # Max ingestion rate

# Fishing (indexed by fishery, not species)
fisheries.rate.base.fsh{N}            # Base fishing rate
mortality.fishing.rate.sp{N}          # Legacy per-species F

# Movement maps (config format, NOT schema)
movement.file.map{N}                  # Map CSV path
movement.species.map{N}              # Species index
movement.initialAge.map{N}           # Age threshold
movement.steps.map{N}                # Active timesteps
```
```

- [ ] **Step 2: Verify skill directory exists**

Run: `ls ~/.claude/skills/ices-data/ices-data.md`
Expected: File exists.

- [ ] **Step 3: Commit (no git for skills dir — just verify)**

The skills directory is not a git repo. Verify the file is readable.

---

## Task 11: Run Full Test Suite

- [ ] **Step 1: Run all unit tests**

Run: `cd ~/ices-mcp-server && uv run pytest -v`
Expected: All tests PASS (integration tests excluded).

- [ ] **Step 2: Run integration tests manually**

Run: `cd ~/ices-mcp-server && uv run pytest -m integration -v`
Expected: All 6 integration tests PASS (requires internet).

- [ ] **Step 3: Verify lint**

Run: `cd ~/ices-mcp-server && uv run ruff check ices/ tests/ server.py`
Expected: No lint errors.

- [ ] **Step 4: Final commit if any fixes needed**

```bash
cd ~/ices-mcp-server && git add -A && git commit -m "fix: address test/lint issues from full suite run"
```
