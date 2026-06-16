# FishBase/SeaLifeBase Trait Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user populate a focal species' life-history trait fields from FishBase/SeaLifeBase with a single fetch + per-trait review step.

**Architecture:** A pure `osmose/fishbase.py` client downloads the rfishbase-5 parquet snapshot tables from Source Cooperative (valid TLS) via stdlib `urllib`, reads them with pandas/pyarrow (already deps), and aggregates each trait to median + range. A Shiny component (`ui/components/fishbase_bootstrap.py`) drives resolve → fetch → review-table → apply, writing selected traits into `state.config` and refreshing the species forms via `state.load_trigger`.

**Tech Stack:** Python 3.12+, pandas (pyarrow parquet), stdlib `urllib`, Shiny for Python, pytest, Playwright (e2e).

**Spec:** `docs/superpowers/specs/2026-06-17-fishbase-trait-bootstrap-design.md`

---

## File structure

- Create `osmose/fishbase.py` — client: exceptions, dataclasses, `TRAIT_MAP`, `_load_table`, `resolve_species`, `fetch_traits`.
- Create `tests/fixtures/fishbase/` — tiny recorded parquet slices (cod + green crab).
- Create `scripts/_record_fishbase_fixtures.py` — one-off fixture recorder (documented, re-runnable).
- Create `tests/test_fishbase.py` — client unit tests.
- Create `ui/components/fishbase_bootstrap.py` — modal UI builder + `fishbase_bootstrap_server`.
- Modify `ui/pages/setup.py` — add the bootstrap control to the Species Configuration card + invoke the server helper.
- Create `tests/test_fishbase_bootstrap_ui.py` — controller-level UI test.
- Create `tests/test_fishbase_e2e.py` — Playwright e2e: **modal-open smoke only** (never clicks Fetch → no network; cross-process monkeypatch isn't possible since the app runs in a subprocess). `@pytest.mark.e2e`.
- Modify `pyproject.toml` — add `pyarrow>=14` to runtime `dependencies` (pandas does NOT pull it; it's only in `.venv` via the unmanaged copernicusmarine install).
- Modify `deploy.sh` — add `pyarrow` to the ensured-packages list (prod already has 21, defensive).

**UI integration note (refines spec decision #6):** the species form is a shared
component (`render_species_table`), so rather than N per-species buttons embedded in the
table, the feature is surfaced as **one control** in the Species Configuration card — a
species dropdown + scientific-name input + "Bootstrap from FishBase" button — opening a
modal scoped to the selected species. Same behavior (bootstrap a chosen species), lower
blast radius. Flag this to the user at plan review.

---

## Task 0: Declare the `pyarrow` runtime dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `deploy.sh`

Rationale: `pd.read_parquet`/`to_parquet` need a parquet engine. **pandas does not require
pyarrow** — the dev `.venv` only has it transitively via the unmanaged `copernicusmarine`
install, so a clean `[dev]`/CI venv would lack it and every fishbase test would error with
`ImportError: Missing optional dependency 'pyarrow'`. The prod shiny env already has
pyarrow 21 (verified), so declaring it won't break prod.

- [ ] **Step 1: Add to runtime dependencies**

In `pyproject.toml`, in `[project] dependencies`, add after the `pandas>=2.2` line:

```toml
    "pyarrow>=14",
```

- [ ] **Step 2: Add to deploy.sh ensured packages (defensive)**

In `deploy.sh`, find the version-floored install line:

```bash
"$SHINY_PIP" install --quiet --upgrade "cma>=4.0" "shinyswatch>=0.11" "shinywidgets>=0.7"
```

and append `"pyarrow>=14"` to it:

```bash
"$SHINY_PIP" install --quiet --upgrade "cma>=4.0" "shinyswatch>=0.11" "shinywidgets>=0.7" "pyarrow>=14"
```

- [ ] **Step 3: Verify it imports**

Run: `.venv/bin/python -c "import pyarrow; print(pyarrow.__version__)"`
Expected: a version ≥ 14 prints.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml deploy.sh
git commit -m "build: declare pyarrow runtime dep for fishbase parquet reads"
```

---

## Task 1: Client skeleton — exceptions, dataclasses, `_load_table`, cache

**Files:**
- Create: `osmose/fishbase.py`
- Test: `tests/test_fishbase.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fishbase.py
import io
import pandas as pd
import pytest
from osmose import fishbase


def _df(rows):
    return pd.DataFrame(rows)


def test_load_table_uses_cache_then_parses(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMOSE_FISHBASE_CACHE_DIR", str(tmp_path))
    calls = {"n": 0}
    sample = _df([{"SpecCode": 69, "Loo": 110.0}])
    raw = io.BytesIO()
    sample.to_parquet(raw)
    raw_bytes = raw.getvalue()

    def fake_fetch(url: str) -> bytes:
        calls["n"] += 1
        return raw_bytes

    monkeypatch.setattr(fishbase, "_http_get_bytes", fake_fetch)
    df1 = fishbase._load_table("popgrowth", "fb")
    df2 = fishbase._load_table("popgrowth", "fb")  # second call hits disk cache
    assert list(df1["SpecCode"]) == [69]
    assert calls["n"] == 1  # network called once; cache served the rest


def test_load_table_network_failure_raises_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMOSE_FISHBASE_CACHE_DIR", str(tmp_path))

    def boom(url: str) -> bytes:
        raise OSError("connection refused")

    monkeypatch.setattr(fishbase, "_http_get_bytes", boom)
    with pytest.raises(fishbase.FishBaseUnavailable):
        fishbase._load_table("popgrowth", "fb")


def test_http_get_bytes_reads_response(monkeypatch):
    """Cover the network seam itself (urlopen mocked — still no real network)."""

    class _FakeResp:
        def __init__(self, data): self._d = data
        def read(self): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    import urllib.request as _u
    monkeypatch.setattr(_u, "urlopen", lambda url, timeout=0: _FakeResp(b"ok"))
    assert fishbase._http_get_bytes("https://example/x.parquet") == b"ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fishbase.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.fishbase'`.

- [ ] **Step 3: Write minimal implementation**

```python
# osmose/fishbase.py
"""FishBase/SeaLifeBase trait bootstrap client.

Downloads the rfishbase-5 parquet-snapshot tables from Source Cooperative
(valid TLS, HTTP range) and queries them locally. Data is CC-BY-NC
(Carl Boettiger / FishBase.org); fetched at runtime (not redistributed).
"""

from __future__ import annotations

import io
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from osmose.logging import setup_logging

_log = setup_logging("osmose.fishbase")

_BASE = "https://data.source.coop/cboettig/fishbase/{db}/v24.07/parquet/{table}.parquet"
_CACHE_TTL_SEC = 7 * 24 * 3600  # a week; data snapshots are versioned/stable
_TIMEOUT_SEC = 30


class FishBaseError(Exception):
    """Base for FishBase client errors."""


class FishBaseUnavailable(FishBaseError):
    """Network / HTTP / parse failure reaching the data source."""


class FishBaseNoMatch(FishBaseError):
    """A name resolved to no record in either database."""


@dataclass(frozen=True)
class SpecMatch:
    spec_code: int
    scientific_name: str
    common_name: str
    db: str  # "fb" | "slb"


@dataclass(frozen=True)
class TraitEstimate:
    value: float  # median across studies
    n: int
    min: float
    max: float
    unit: str


def _cache_dir() -> Path:
    d = os.environ.get("OSMOSE_FISHBASE_CACHE_DIR", "").strip()
    base = Path(d) if d else Path.home() / ".cache" / "osmose" / "fishbase"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _http_get_bytes(url: str) -> bytes:
    """Single network seam (TLS-verified). Tests monkeypatch this."""
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SEC) as resp:  # noqa: S310 (https only)
        return resp.read()


def _load_table(table: str, db: str = "fb") -> pd.DataFrame:
    """Return a FishBase/SeaLifeBase parquet table, caching the raw file on disk."""
    cache = _cache_dir() / f"{db}_{table}.parquet"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < _CACHE_TTL_SEC
    if not fresh:
        url = _BASE.format(db=db, table=table)
        try:
            data = _http_get_bytes(url)
        except Exception as exc:  # noqa: BLE001 — any fetch failure is "unavailable"
            raise FishBaseUnavailable(f"could not fetch {url}: {exc}") from exc
        cache.write_bytes(data)
    try:
        return pd.read_parquet(cache)
    except Exception as exc:  # noqa: BLE001 — corrupt/changed payload
        raise FishBaseUnavailable(f"could not parse {table} parquet: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fishbase.py -q -o addopts=""`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add osmose/fishbase.py tests/test_fishbase.py
git commit -m "feat(fishbase): client skeleton — _load_table with disk cache + typed errors"
```

---

## Task 2: Record test fixtures (cod + green crab slices)

**Files:**
- Create: `scripts/_record_fishbase_fixtures.py`
- Create: `tests/fixtures/fishbase/{fb_species,fb_popgrowth,fb_poplw,fb_maturity,slb_species,slb_popgrowth,slb_poplw,slb_maturity}.parquet`

- [ ] **Step 1: Write the fixture recorder**

```python
# scripts/_record_fishbase_fixtures.py
"""Record tiny FishBase/SeaLifeBase parquet slices for tests (re-runnable).

Pulls only the rows for Gadus morhua (FishBase) and Carcinus maenas (SeaLifeBase)
so fixtures stay <50 KB. Run: .venv/bin/python scripts/_record_fishbase_fixtures.py
Requires network (one-off); CI never runs this.
"""
from pathlib import Path

from osmose import fishbase

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "fishbase"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = {"fb": ("Gadus", "morhua"), "slb": ("Carcinus", "maenas")}
TABLES = ["species", "popgrowth", "poplw", "maturity"]


def main() -> None:
    for db, (genus, species) in TARGETS.items():
        sp = fishbase._load_table("species", db)
        code = int(sp[(sp.Genus == genus) & (sp.Species == species)].SpecCode.iloc[0])
        for table in TABLES:
            df = fishbase._load_table(table, db)
            col = "Speccode" if "Speccode" in df.columns else "SpecCode"
            if table == "species":
                slice_ = df[(df.Genus == genus) & (df.Species == species)]
            else:
                slice_ = df[df[col] == code]
            slice_.to_parquet(OUT / f"{db}_{table}.parquet")
            print(f"{db}/{table}: {len(slice_)} rows (code={code})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to produce fixtures**

Run: `.venv/bin/python scripts/_record_fishbase_fixtures.py`
Expected output includes `fb/popgrowth: 108 rows (code=69)` and `slb/species: 1 rows (code=26397)`. Files appear under `tests/fixtures/fishbase/`.

- [ ] **Step 3: Verify fixtures are small and load**

Run: `.venv/bin/python -c "import pandas as pd, glob; [print(f, len(pd.read_parquet(f))) for f in sorted(glob.glob('tests/fixtures/fishbase/*.parquet'))]"`
Expected: 8 files, each loads, row counts small.

- [ ] **Step 4: Commit**

```bash
git add scripts/_record_fishbase_fixtures.py tests/fixtures/fishbase/
git commit -m "test(fishbase): record cod + green-crab parquet fixtures"
```

---

## Task 3: `resolve_species`

**Files:**
- Modify: `osmose/fishbase.py`
- Test: `tests/test_fishbase.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fishbase.py  (add)
from pathlib import Path

_FIX = Path(__file__).parent / "fixtures" / "fishbase"


@pytest.fixture
def fixture_tables(monkeypatch):
    """Serve fixtures from _load_table; no network."""
    def fake(table, db="fb"):
        return pd.read_parquet(_FIX / f"{db}_{table}.parquet")
    monkeypatch.setattr(fishbase, "_load_table", fake)


def test_resolve_scientific_name_fishbase(fixture_tables):
    matches = fishbase.resolve_species("Gadus morhua")
    assert len(matches) == 1
    m = matches[0]
    assert m.spec_code == 69 and m.db == "fb"
    assert m.common_name == "Atlantic cod"


def test_resolve_common_name(fixture_tables):
    matches = fishbase.resolve_species("atlantic cod")  # case-insensitive
    assert matches and matches[0].spec_code == 69


def test_resolve_falls_back_to_sealifebase(fixture_tables):
    matches = fishbase.resolve_species("Carcinus maenas")
    assert matches and matches[0].db == "slb" and matches[0].spec_code == 26397


def test_resolve_no_match_raises(fixture_tables):
    with pytest.raises(fishbase.FishBaseNoMatch):
        fishbase.resolve_species("Nonexistus fakus")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fishbase.py -q -o addopts=""`
Expected: FAIL — `AttributeError: module 'osmose.fishbase' has no attribute 'resolve_species'`.

- [ ] **Step 3: Implement**

```python
# osmose/fishbase.py  (add)
def _match_in_db(name: str, db: str) -> list[SpecMatch]:
    sp = _load_table("species", db)
    name = name.strip()
    out: list[SpecMatch] = []
    parts = name.split()
    if len(parts) >= 2:  # try scientific "Genus species"
        genus, species = parts[0], parts[1]
        hit = sp[
            sp.Genus.str.casefold().eq(genus.casefold())
            & sp.Species.str.casefold().eq(species.casefold())
        ]
        out += _rows_to_matches(hit, db)
    if not out:  # try common name (FBname)
        hit = sp[sp.FBname.fillna("").str.casefold().eq(name.casefold())]
        out += _rows_to_matches(hit, db)
    return out


def _rows_to_matches(hit: pd.DataFrame, db: str) -> list[SpecMatch]:
    return [
        SpecMatch(
            spec_code=int(r.SpecCode),
            scientific_name=f"{r.Genus} {r.Species}",
            common_name=("" if pd.isna(r.FBname) else str(r.FBname)),
            db=db,
        )
        for r in hit.itertuples()
    ]


def resolve_species(name: str, *, db: str | None = None) -> list[SpecMatch]:
    """Resolve a scientific or common name to candidate SpecMatch(es).

    Tries FishBase, then SeaLifeBase (unless ``db`` forces one). Raises
    FishBaseNoMatch when neither database has a record.
    """
    dbs = [db] if db else ["fb", "slb"]
    for d in dbs:
        matches = _match_in_db(name, d)
        if matches:
            return matches
    raise FishBaseNoMatch(f"no FishBase/SeaLifeBase record for {name!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_fishbase.py -q -o addopts=""`
Expected: PASS (all resolve tests + Task 1 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/fishbase.py tests/test_fishbase.py
git commit -m "feat(fishbase): resolve_species (scientific/common, FB→SLB fallback)"
```

---

## Task 4: `TRAIT_MAP` + `fetch_traits` (aggregation, Speccode quirk, a/b check)

**Files:**
- Modify: `osmose/fishbase.py`
- Test: `tests/test_fishbase.py`

- [ ] **Step 1: Write the failing tests (real cod medians)**

```python
# tests/test_fishbase.py  (add)
def test_fetch_traits_cod_medians(fixture_tables):
    t = fishbase.fetch_traits(69, "fb")
    assert t["species.linf"].value == pytest.approx(110.0, abs=1.0)
    assert t["species.linf"].n == 108
    assert t["species.k"].value == pytest.approx(0.163, abs=0.01)
    assert t["species.t0"].value == pytest.approx(-0.08, abs=0.05)
    assert t["species.lmax"].value == pytest.approx(200.0)
    assert t["species.maturity.size"].value == pytest.approx(63.79, abs=0.5)
    assert t["species.lifespan"].value == pytest.approx(25.0)
    assert t["species.linf"].min < t["species.linf"].value < t["species.linf"].max


def test_fetch_traits_ab_reproduce_weight(fixture_tables):
    """a/b convention W(g)=a*L(cm)^b: a 110 cm cod is a few-to-~15 kg."""
    t = fishbase.fetch_traits(69, "fb")
    a = t["species.length2weight.condition.factor"].value
    b = t["species.length2weight.allometric.power"].value
    weight_g = a * (110.0 ** b)
    assert 3_000 < weight_g < 25_000  # ~3–25 kg sanity band


def test_fetch_traits_partial_coverage_omits_missing(fixture_tables, monkeypatch):
    """A species with no poplw rows simply omits a/b — never errors."""
    real = fishbase._load_table

    def patched(table, db="fb"):
        df = real(table, db)
        return df.iloc[0:0] if table == "poplw" else df

    monkeypatch.setattr(fishbase, "_load_table", patched)
    t = fishbase.fetch_traits(69, "fb")
    assert "species.length2weight.condition.factor" not in t
    assert "species.linf" in t  # other traits still present
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fishbase.py -k fetch_traits -q -o addopts=""`
Expected: FAIL — no attribute `fetch_traits`.

- [ ] **Step 3: Implement**

```python
# osmose/fishbase.py  (add)
# OSMOSE key stem -> (table, column, speccode_column, unit)
TRAIT_MAP: dict[str, tuple[str, str, str, str]] = {
    "species.linf": ("popgrowth", "Loo", "SpecCode", "cm"),
    "species.k": ("popgrowth", "K", "SpecCode", "year^-1"),
    "species.t0": ("popgrowth", "to", "SpecCode", "year"),
    "species.lmax": ("species", "Length", "SpecCode", "cm"),
    "species.length2weight.condition.factor": ("poplw", "a", "SpecCode", ""),
    "species.length2weight.allometric.power": ("poplw", "b", "SpecCode", ""),
    "species.maturity.size": ("maturity", "Lm", "Speccode", "cm"),
    "species.lifespan": ("species", "LongevityWild", "SpecCode", "year"),
}


def fetch_traits(spec_code: int, db: str) -> dict[str, TraitEstimate]:
    """Aggregate each mapped trait to median/n/min-max for a species.

    Traits with no usable data are omitted (partial coverage is normal).
    """
    tables: dict[str, pd.DataFrame] = {}
    out: dict[str, TraitEstimate] = {}
    for key, (table, col, code_col, unit) in TRAIT_MAP.items():
        if table not in tables:
            tables[table] = _load_table(table, db)
        df = tables[table]
        if code_col not in df.columns or col not in df.columns:
            continue
        vals = pd.to_numeric(df.loc[df[code_col] == spec_code, col], errors="coerce").dropna()
        if vals.empty:
            continue
        out[key] = TraitEstimate(
            value=float(vals.median()),
            n=int(vals.size),
            min=float(vals.min()),
            max=float(vals.max()),
            unit=unit,
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_fishbase.py -q -o addopts=""`
Expected: PASS (all client tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/fishbase.py tests/test_fishbase.py
git commit -m "feat(fishbase): TRAIT_MAP + fetch_traits (median/range, Speccode quirk, a/b check)"
```

---

## Task 5: UI component — modal builder + `fishbase_bootstrap_server`

**Files:**
- Create: `ui/components/fishbase_bootstrap.py`
- Test: `tests/test_fishbase_bootstrap_ui.py`

- [ ] **Step 1: Write the failing controller-level test**

```python
# tests/test_fishbase_bootstrap_ui.py
from unittest.mock import patch

from osmose.fishbase import SpecMatch, TraitEstimate
from ui.components import fishbase_bootstrap as fb


def test_apply_writes_selected_traits_to_config():
    """apply_traits writes only ticked traits, into the right sp{idx} keys."""
    cfg = {"simulation.nspecies": "2", "species.name.sp1": "cod"}
    traits = {
        "species.linf": TraitEstimate(110.0, 108, 53.7, 226.0, "cm"),
        "species.k": TraitEstimate(0.163, 108, 0.048, 0.5, "year^-1"),
    }
    selected = {"species.linf"}  # user ticked only Linf
    new_cfg = fb.apply_traits(cfg, species_index=1, traits=traits, selected=selected)
    assert new_cfg["species.linf.sp1"] == "110.0"
    assert "species.k.sp1" not in new_cfg  # unticked -> untouched
    assert new_cfg is not cfg  # pure: returns a new dict


def test_review_rows_pairs_current_and_fetched():
    cfg = {"species.linf.sp0": "120"}
    traits = {"species.linf": TraitEstimate(110.0, 108, 53.7, 226.0, "cm")}
    rows = fb.review_rows(cfg, species_index=0, traits=traits)
    row = next(r for r in rows if r["key"] == "species.linf")
    assert row["current"] == "120" and row["fetched"] == 110.0 and row["n"] == 108
    # label resolves to the field's human description (NOT field.label, which doesn't
    # exist on OsmoseField) — building the row at all would AttributeError on a bad attr.
    assert row["label"] and row["label"] != "species.linf"


def test_pick_id_is_shiny_safe():
    """Checkbox ids must not contain dots (illegal in Shiny input ids)."""
    pid = fb._pick_id(2, "species.length2weight.condition.factor")
    assert "." not in pid
    assert pid == "fb_pick_2_species_length2weight_condition_factor"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fishbase_bootstrap_ui.py -q -o addopts=""`
Expected: FAIL — module/functions missing.

- [ ] **Step 3: Implement the pure helpers + UI/server**

```python
# ui/components/fishbase_bootstrap.py
"""'Bootstrap from FishBase' control: resolve -> fetch -> review -> apply.

Pure helpers (apply_traits, review_rows) are unit-tested; the server wires them
to a modal. Data is CC-BY-NC (FishBase via rOpenSci / Source Cooperative).
"""

from __future__ import annotations

from shiny import reactive, render, ui
from shiny.types import SilentException

from osmose import fishbase
from osmose.logging import setup_logging
from osmose.schema.species import SPECIES_FIELDS

_log = setup_logging("osmose.fishbase_ui")
_ATTRIB = "Data: FishBase/SeaLifeBase via rOpenSci / Source Cooperative (CC-BY-NC)."

# key stem -> OsmoseField (for resolve_key + description/unit). OsmoseField has
# `description` (the UI display string) — there is NO `.label` attribute.
_FIELD_BY_STEM = {f.key_pattern.replace(".sp{idx}", ""): f for f in SPECIES_FIELDS}


def _pick_id(species_index: int, key: str) -> str:
    """Shiny-safe checkbox id. Dots are ILLEGAL in Shiny input ids (repo convention,
    param_form.input_id_for_key); namespace by species index to avoid cross-species
    checkbox stickiness on modal re-open."""
    return f"fb_pick_{species_index}_" + key.replace(".", "_")


def review_rows(cfg: dict, species_index: int, traits: dict) -> list[dict]:
    """Build review-table rows pairing current config value with the fetched median."""
    rows = []
    for key, est in traits.items():
        field = _FIELD_BY_STEM.get(key)
        rows.append(
            {
                "key": key,
                "label": field.description if field else key,
                "current": cfg.get(field.resolve_key(species_index)) if field else None,
                "fetched": est.value,
                "n": est.n,
                "range": (est.min, est.max),
                "unit": est.unit,
            }
        )
    return rows


def apply_traits(cfg: dict, species_index: int, traits: dict, selected: set[str]) -> dict:
    """Return a NEW config with only the selected traits written to sp{index} keys."""
    out = dict(cfg)
    for key in selected:
        field = _FIELD_BY_STEM.get(key)
        est = traits.get(key)
        if field is None or est is None:
            continue
        out[field.resolve_key(species_index)] = str(est.value)
    return out


def fishbase_bootstrap_ui():
    """Control row to embed in the Species Configuration card."""
    return ui.div(
        ui.input_action_button(
            "fb_open", "Bootstrap from FishBase", class_="btn-outline-info btn-sm mt-2"
        ),
        ui.tags.small(_ATTRIB, class_="text-muted d-block mt-1"),
    )


def fishbase_bootstrap_server(input, output, session, state):
    _traits: reactive.Value = reactive.Value({})
    _match: reactive.Value = reactive.Value(None)

    def _species_choices() -> dict:
        # Source of truth = config's simulation.nspecies; names only label the slots.
        with reactive.isolate():
            cfg = state.config.get()
            names = state.species_names.get() or []
        try:
            n = int(float(cfg.get("simulation.nspecies", len(names)) or 0))
        except (TypeError, ValueError):
            n = len(names)
        return {
            str(i): (names[i] if i < len(names) and names[i] else f"Species {i}")
            for i in range(max(n, 0))
        }

    @reactive.effect
    @reactive.event(input.fb_open)
    def _open():
        _traits.set({})  # reset stale fetch state on every open (easy_close can leave it)
        _match.set(None)
        choices = _species_choices()
        first_idx = next(iter(choices), "0")
        prefill = choices.get(first_idx, "")
        ui.modal_show(
            ui.modal(
                ui.input_select("fb_species", "Species (config slot)", choices=choices),
                ui.input_text("fb_name", "Scientific or common name", value=prefill),
                ui.input_action_button("fb_fetch", "Fetch traits", class_="btn-primary btn-sm"),
                ui.output_ui("fb_review"),
                ui.input_action_button("fb_apply", "Apply selected", class_="btn-success btn-sm"),
                ui.tags.small(_ATTRIB, class_="text-muted d-block mt-2"),
                title="Bootstrap species traits from FishBase",
                easy_close=True,
                size="l",
            )
        )

    @reactive.effect
    @reactive.event(input.fb_fetch)
    def _fetch():
        name = (input.fb_name() or "").strip()
        if not name:
            ui.notification_show("Enter a species name.", type="warning", duration=5)
            return
        # First fetch downloads up to ~8 MB (4 parquet tables) synchronously, which briefly
        # blocks this session's flush (cf. the live-movement lesson). It's a one-shot user
        # action; show a busy notification so the freeze reads as progress, not a hang.
        # Subsequent fetches hit the week-long disk cache and are instant.
        ui.notification_show("Fetching from FishBase…", id="fb_busy", duration=None)
        try:
            matches = fishbase.resolve_species(name)
            m = matches[0]
            traits = fishbase.fetch_traits(m.spec_code, m.db)
        except fishbase.FishBaseNoMatch:
            _traits.set({}); _match.set(None)
            ui.notification_show(f"No FishBase/SeaLifeBase record for '{name}'.", type="error", duration=8)
            return
        except fishbase.FishBaseUnavailable:
            _traits.set({}); _match.set(None)
            ui.notification_show("FishBase unavailable — try again later.", type="error", duration=8)
            return
        finally:
            ui.notification_remove("fb_busy")
        _match.set(m)
        _traits.set(traits)

    @render.ui
    def fb_review():
        traits = _traits.get()
        m = _match.get()
        if not traits or m is None:
            return ui.div()
        with reactive.isolate():
            cfg = state.config.get()
        idx = int(input.fb_species())
        header = ui.tags.div(
            f"{m.scientific_name} ({m.common_name}) — {m.db.upper()}", class_="fw-bold mb-1"
        )
        rows = [
            ui.tags.tr(
                ui.tags.td(ui.input_checkbox(_pick_id(idx, r["key"]), "", value=True)),
                ui.tags.td(r["label"]),
                ui.tags.td("" if r["current"] is None else str(r["current"])),
                ui.tags.td(f"{r['fetched']:.4g} {r['unit']}"),
                ui.tags.td(str(r["n"])),
                ui.tags.td(f"{r['range'][0]:.4g}–{r['range'][1]:.4g}"),
            )
            for r in review_rows(cfg, idx, traits)
        ]
        return ui.div(
            header,
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(*[ui.tags.th(h) for h in ("✓", "Trait", "Current", "FishBase", "n", "Range")])
                ),
                ui.tags.tbody(*rows),
                class_="table table-sm",
            ),
        )

    @reactive.effect
    @reactive.event(input.fb_apply)
    def _apply():
        traits = _traits.get()
        if not traits:
            return
        idx = int(input.fb_species())
        selected = {k for k in traits if _checkbox(input, _pick_id(idx, k))}
        with reactive.isolate():
            cfg = dict(state.config.get())
        new_cfg = apply_traits(cfg, idx, traits, selected)
        if new_cfg != cfg:
            state.config.set(new_cfg)
            state.dirty.set(True)
            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)
        ui.notification_show(f"Applied {len(selected)} trait(s) to species {idx}.", type="message", duration=4)
        ui.modal_remove()


def _checkbox(input, input_id: str) -> bool:
    # Narrow except (repo precedent: calibration_handlers): only swallow absent /
    # not-yet-rendered inputs. A broad `except Exception` here would silently mask a
    # real id-resolution bug as "unchecked" -> "Applied 0 traits" with a success toast.
    try:
        return bool(getattr(input, input_id)())
    except (AttributeError, SilentException):
        return False
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_fishbase_bootstrap_ui.py -q -o addopts=""`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add ui/components/fishbase_bootstrap.py tests/test_fishbase_bootstrap_ui.py
git commit -m "feat(fishbase-ui): review/apply helpers + bootstrap modal server"
```

---

## Task 6: Wire the control into the setup page

**Files:**
- Modify: `ui/pages/setup.py`
- Test: `tests/test_fishbase_bootstrap_ui.py`

- [ ] **Step 1: Write the failing wiring test**

```python
# tests/test_fishbase_bootstrap_ui.py  (add)
def test_setup_ui_includes_bootstrap_control():
    from ui.pages.setup import setup_ui

    html = str(setup_ui())
    assert "fb_open" in html
    assert "Bootstrap from FishBase" in html
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fishbase_bootstrap_ui.py::test_setup_ui_includes_bootstrap_control -q -o addopts=""`
Expected: FAIL — `fb_open` not in setup_ui.

- [ ] **Step 3: Implement — add import, control, and server wiring**

In `ui/pages/setup.py`, add to the imports block:

```python
from ui.components.fishbase_bootstrap import fishbase_bootstrap_ui, fishbase_bootstrap_server
```

In `setup_ui`, inside the Species Configuration `ui.card(...)`, add `fishbase_bootstrap_ui()` right after `ui.output_ui("species_panels")`:

```python
                    ui.output_ui("species_panels"),
                    fishbase_bootstrap_ui(),
```

At the end of `setup_server`, add:

```python
    fishbase_bootstrap_server(input, output, session, state)
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `.venv/bin/python -m pytest tests/test_fishbase_bootstrap_ui.py tests/test_collapsible.py -q -o addopts=""`
Expected: PASS (the setup page still builds; bootstrap control present).

- [ ] **Step 5: Commit**

```bash
git add ui/pages/setup.py tests/test_fishbase_bootstrap_ui.py
git commit -m "feat(fishbase-ui): surface bootstrap control on the species setup page"
```

---

## Task 7: Playwright e2e (client monkeypatched, no network)

**Files:**
- Create: `tests/test_fishbase_e2e.py`

- [ ] **Step 1: Write the e2e test**

```python
# tests/test_fishbase_e2e.py
import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")


def test_bootstrap_modal_opens_and_lists_traits(page: Page, app: ShinyAppProc):
    """Smoke: the control opens the modal. (Fetch hits the live API only if reachable;
    this test asserts the modal + inputs render, not network results.)"""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=30_000)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='setup']").click()
    page.wait_for_selector("#fb_open", timeout=20_000)
    page.locator("#fb_open").click()
    expect(page.locator("#fb_name")).to_be_visible(timeout=10_000)
    expect(page.locator("#fb_fetch")).to_be_visible()
```

- [ ] **Step 2: Run it (browser required)**

Run: `.venv/bin/python -m pytest tests/test_fishbase_e2e.py -m e2e -o addopts="" -p no:cacheprovider`
Expected: PASS (1 passed). If the browser is unavailable, it errors at fixture setup — note and skip locally; CI's e2e leg runs it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fishbase_e2e.py
git commit -m "test(fishbase): e2e smoke — bootstrap modal opens on setup page"
```

---

## Task 8: Final verification + PR

- [ ] **Step 1: Lint + type-check the new/changed files**

Run: `.venv/bin/ruff check osmose/fishbase.py ui/components/fishbase_bootstrap.py ui/pages/setup.py tests/test_fishbase.py tests/test_fishbase_bootstrap_ui.py scripts/_record_fishbase_fixtures.py`
Then: `PYRIGHT_PYTHON_FORCE_VERSION=latest .venv/bin/pyright osmose/fishbase.py ui/components/fishbase_bootstrap.py`
Expected: ruff clean; pyright clean (the pre-existing `shiny_deckgl` resolution artifact in app.py is unrelated).

- [ ] **Step 2: Run the full unit suite (no network — fixtures only)**

Run: `.venv/bin/python -m pytest -q -m "not e2e"`
Expected: all pass (prior count + the new fishbase tests). Confirm no test reaches the network (fixtures + monkeypatch).

- [ ] **Step 3: Clean-venv check (the CI gotcha)**

Build a throwaway venv and **RUN** the parquet tests — `--collect-only` only imports modules
and would NOT execute `pd.read_parquet`/`to_parquet`, so it cannot catch a missing parquet
engine. `pandas` is a main runtime dep and `pyarrow` is now declared (Task 0); this proves a
clean install actually has a working parquet engine:
Run: `python3.12 -m venv /tmp/fbcheck && /tmp/fbcheck/bin/pip install -e ".[dev]" -q && /tmp/fbcheck/bin/python -m pytest tests/test_fishbase.py tests/test_fishbase_bootstrap_ui.py -q -o addopts=""`
Expected: all pass (parquet reads work → pyarrow is installed via the declared dep, not via the unmanaged `.venv` copernicusmarine chain).

- [ ] **Step 4: Push + open PR**

```bash
git push -u origin feat/fishbase-trait-bootstrap
gh pr create --base master --title "feat: FishBase/SeaLifeBase species-trait bootstrap" --body "Populate a focal species' life-history traits (Linf, K, t0, Lmax, length-weight a/b, maturity size, lifespan) from FishBase/SeaLifeBase parquet snapshots (Source Cooperative), with a per-trait review-and-apply modal on the species setup page. Pure osmose/fishbase.py client (urllib + pandas/pyarrow, no new dep), median+range resolution, FB->SLB fallback, fixtures-only tests (no network in CI). Spec: docs/superpowers/specs/2026-06-17-fishbase-trait-bootstrap-design.md. Data CC-BY-NC (FishBase via rOpenSci / Source Cooperative), attributed in-UI."
```
Expected: PR opens; watch CI to green.

---

## Notes for the implementer

- **No network in CI:** every client test monkeypatches `_load_table` (or `_http_get_bytes`) to fixtures. Only `scripts/_record_fishbase_fixtures.py` touches the network, and CI never runs it.
- **`Speccode` vs `SpecCode`:** the `maturity` table uses lowercase `Speccode`; `TRAIT_MAP` encodes the right column per table. Don't "normalize" it away.
- **Pure core:** `osmose/fishbase.py` imports no Shiny; `apply_traits`/`review_rows` are pure and unit-tested independently of the modal.
- **Apply refresh:** writing traits bumps `state.load_trigger` so `species_panels` re-renders (mirrors `grid.py: handle_load_example`).
- **Attribution:** the CC-BY-NC credit string is shown in the control and the modal.
- **OTel guard interplay:** unrelated, but the prod guard shipped in #67 means even a stray render error here won't crash the session.
