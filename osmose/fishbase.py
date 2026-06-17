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
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SEC) as resp:  # noqa: S310 (https only)
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
