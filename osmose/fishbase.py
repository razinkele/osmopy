"""FishBase/SeaLifeBase trait bootstrap client.

Downloads the rfishbase-5 parquet-snapshot tables from Source Cooperative
(valid TLS, HTTP range) and queries them locally. Data is CC-BY-NC
(Carl Boettiger / FishBase.org); fetched at runtime (not redistributed).
"""

from __future__ import annotations

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
    req = urllib.request.Request(url, headers={"User-Agent": "osmose-python"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:  # noqa: S310 (https only)
        return resp.read()


def _load_table(table: str, db: str = "fb") -> pd.DataFrame:
    """Return a FishBase/SeaLifeBase parquet table, caching the raw file on disk.

    Atomic write (tmp + os.replace) so a crashed/interrupted download can't leave a
    truncated file behind. On a parse failure the (possibly corrupt) cache is EVICTED so
    the next call re-fetches, instead of re-raising for the whole TTL with no self-heal.
    """
    cache = _cache_dir() / f"{db}_{table}.parquet"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < _CACHE_TTL_SEC
    if not fresh:
        url = _BASE.format(db=db, table=table)
        try:
            data = _http_get_bytes(url)
        except Exception as exc:  # noqa: BLE001 — any fetch failure is "unavailable"
            raise FishBaseUnavailable(f"could not fetch {url}: {exc}") from exc
        tmp = cache.with_suffix(".parquet.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)  # atomic publish
    try:
        return pd.read_parquet(cache)
    except Exception as exc:  # noqa: BLE001 — corrupt/changed payload: evict + signal
        cache.unlink(missing_ok=True)
        raise FishBaseUnavailable(f"could not parse {table} parquet (cache evicted): {exc}") from exc


def _match_in_db(name: str, db: str) -> list[SpecMatch]:
    sp = _load_table("species", db)
    name = name.strip()
    out: list[SpecMatch] = []
    parts = name.split()
    if len(parts) >= 2 and {"Genus", "Species"}.issubset(sp.columns):  # scientific "Genus species"
        genus, species = parts[0], parts[1]
        hit = sp[
            sp.Genus.str.casefold().eq(genus.casefold())
            & sp.Species.str.casefold().eq(species.casefold())
        ]
        out += _rows_to_matches(hit, db)
    if not out and "FBname" in sp.columns:  # try common name (FBname)
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


# OSMOSE key stem -> (table, column, speccode_column, unit, positive_only)
# positive_only=True drops non-positive values: FishBase stores 0 as a "not recorded"
# sentinel for strictly-positive quantities (Loo/K/Lm/a/b/Length/longevity). `to` (t0)
# is legitimately negative, so it must NOT be positive-filtered.
TRAIT_MAP: dict[str, tuple[str, str, str, str, bool]] = {
    "species.linf": ("popgrowth", "Loo", "SpecCode", "cm", True),
    "species.k": ("popgrowth", "K", "SpecCode", "year^-1", True),
    "species.t0": ("popgrowth", "to", "SpecCode", "year", False),
    "species.lmax": ("species", "Length", "SpecCode", "cm", True),
    "species.length2weight.condition.factor": ("poplw", "a", "SpecCode", "", True),
    "species.length2weight.allometric.power": ("poplw", "b", "SpecCode", "", True),
    "species.maturity.size": ("maturity", "Lm", "Speccode", "cm", True),
    "species.lifespan": ("species", "LongevityWild", "SpecCode", "year", True),
}


def fetch_traits(spec_code: int, db: str) -> dict[str, TraitEstimate]:
    """Aggregate each mapped trait to median/n/min-max for a species.

    Per-table load failures degrade to "trait absent" (partial coverage is normal,
    especially on SeaLifeBase where some tables are missing) — a single missing table
    must NOT lose the traits that did load. Only a TOTAL outage (no table loads at all)
    re-raises FishBaseUnavailable so the UI shows its "unavailable" path.
    """
    tables: dict[str, pd.DataFrame | None] = {}
    out: dict[str, TraitEstimate] = {}
    for key, (table, col, code_col, unit, positive_only) in TRAIT_MAP.items():
        if table not in tables:
            try:
                tables[table] = _load_table(table, db)
            except FishBaseUnavailable:
                _log.warning("table %s unavailable for db=%s; skipping its traits", table, db)
                tables[table] = None
        df = tables[table]
        if df is None or code_col not in df.columns or col not in df.columns:
            continue
        vals = pd.to_numeric(df.loc[df[code_col] == spec_code, col], errors="coerce").dropna()
        if positive_only:
            vals = vals[vals > 0]  # drop the 0 "not recorded" sentinel
        if vals.empty:
            _log.debug("trait %s: no usable values for spec_code=%s db=%s", key, spec_code, db)
            continue
        out[key] = TraitEstimate(
            value=float(vals.median()),
            n=int(vals.size),
            min=float(vals.min()),
            max=float(vals.max()),
            unit=unit,
        )
    if tables and all(v is None for v in tables.values()):
        raise FishBaseUnavailable(f"no FishBase tables could be loaded for db={db}")
    return out
