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
        def __init__(self, data):
            self._d = data

        def read(self):
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request as _u

    monkeypatch.setattr(_u, "urlopen", lambda url, timeout=0: _FakeResp(b"ok"))
    assert fishbase._http_get_bytes("https://example/x.parquet") == b"ok"


def test_load_table_evicts_corrupt_cache(tmp_path, monkeypatch):
    """A corrupt (fresh) cache must be deleted on parse failure so it can self-heal."""
    monkeypatch.setenv("OSMOSE_FISHBASE_CACHE_DIR", str(tmp_path))
    cache = tmp_path / "fb_popgrowth.parquet"
    cache.write_bytes(b"not a parquet file")  # fresh but corrupt
    monkeypatch.setattr(
        fishbase,
        "_http_get_bytes",
        lambda url: (_ for _ in ()).throw(AssertionError("should not fetch: cache is fresh")),
    )
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
    weight_g = a * (110.0**b)
    assert 3_000 < weight_g < 25_000


def test_fetch_traits_partial_coverage_omits_missing(fixture_tables, monkeypatch):
    """A species with no poplw rows simply omits a/b — never errors."""
    real = fishbase._load_table

    def patched(table, db="fb"):
        df = real(table, db)
        return df.iloc[0:0] if table == "poplw" else df

    monkeypatch.setattr(fishbase, "_load_table", patched)
    t = fishbase.fetch_traits(69, "fb")
    assert "species.length2weight.condition.factor" not in t
    assert "species.linf" in t


def test_fetch_traits_missing_table_degrades_not_aborts(fixture_tables, monkeypatch):
    """A single table raising FishBaseUnavailable must NOT lose other-table traits."""
    real = fishbase._load_table

    def patched(table, db="fb"):
        if table == "maturity":
            raise fishbase.FishBaseUnavailable("maturity 404 on this db")
        return real(table, db)

    monkeypatch.setattr(fishbase, "_load_table", patched)
    t = fishbase.fetch_traits(69, "fb")
    assert "species.maturity.size" not in t
    assert "species.linf" in t


def test_fetch_traits_total_outage_raises(fixture_tables, monkeypatch):
    monkeypatch.setattr(
        fishbase,
        "_load_table",
        lambda table, db="fb": (_ for _ in ()).throw(fishbase.FishBaseUnavailable("down")),
    )
    with pytest.raises(fishbase.FishBaseUnavailable):
        fishbase.fetch_traits(69, "fb")


def test_fetch_traits_drops_zero_sentinel(monkeypatch):
    """positive_only traits drop 0 ('not recorded'); t0 keeps negatives."""
    pg = pd.DataFrame(
        {
            "SpecCode": [1, 1, 1],
            "Loo": [0.0, 100.0, 110.0],
            "K": [0.1, 0.2, 0.0],
            "to": [-0.5, -0.1, 0.0],
        }
    )
    monkeypatch.setattr(
        fishbase, "_load_table", lambda table, db="fb": pg if table == "popgrowth" else pg.iloc[0:0]
    )
    t = fishbase.fetch_traits(1, "fb")
    assert t["species.linf"].value == 105.0 and t["species.linf"].n == 2
    assert t["species.t0"].n == 3


def test_fetch_traits_sealifebase_partial(fixture_tables):
    """Green crab (SLB) returns a dict and never raises, even if growth tables are sparse."""
    t = fishbase.fetch_traits(26397, "slb")
    assert isinstance(t, dict)
