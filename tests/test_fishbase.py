import io
from pathlib import Path

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
    fishbase._load_table("popgrowth", "fb")  # second call hits disk cache
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


def test_load_table_evicts_corrupt_cache(tmp_path, monkeypatch):
    """A corrupt (fresh) cache must be deleted on parse failure so it can self-heal."""
    monkeypatch.setenv("OSMOSE_FISHBASE_CACHE_DIR", str(tmp_path))
    cache = tmp_path / "fb_popgrowth.parquet"
    cache.write_bytes(b"not a parquet file")  # fresh but corrupt
    monkeypatch.setattr(fishbase, "_http_get_bytes", lambda url: (_ for _ in ()).throw(AssertionError("should not fetch: cache is fresh")))
    with pytest.raises(fishbase.FishBaseUnavailable):
        fishbase._load_table("popgrowth", "fb")
    assert not cache.exists()  # evicted -> next call re-fetches


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
