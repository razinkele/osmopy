"""cod_east is RV-gated (raw_cap) with elevated M and its own fishery; cod_west
keeps standard Shepherd recruitment and the western fishery (Phase 1 Task 6)."""

import pandas as pd

from osmose.config import OsmoseConfigReader

CFG = "data/baltic/baltic_all-parameters.csv"


def _cfg():
    return dict(OsmoseConfigReader().read(CFG))


def test_rv_gate_on_cod_east_only_raw_cap():
    cfg = _cfg()
    assert cfg["reproduction.rv.gate.mode"] == "raw_cap"
    assert float(cfg["reproduction.rv.gate.ref"]) > 1.0  # on the RV series scale
    assert cfg["reproduction.rv.gate.species.enabled.sp8"] == "true"  # cod_east
    assert cfg["reproduction.rv.gate.species.enabled.sp0"] == "false"  # cod_west standard SR


def test_cod_east_elevated_mortality():
    cfg = _cfg()
    m_east = float(cfg["mortality.additional.rate.sp8"])
    m_west = float(cfg["mortality.additional.rate.sp0"])
    assert m_east > 1.8 * m_west, f"cod_east M {m_east} not ~doubled vs cod_west {m_west}"


def test_cod_east_has_separate_fishery():
    cfg = _cfg()
    assert cfg["simulation.nfisheries"] == "9"
    assert cfg["fisheries.name.fsh8"] == "trawlcod_east"


def test_catchability_maps_each_stock_to_its_fishery():
    df = pd.read_csv("data/baltic/fishery-catchability.csv", index_col=0)
    assert "cod_west" in df.index and "cod_east" in df.index
    assert "cod" not in df.index
    # cod_west -> trawlcod (fsh0), cod_east -> trawlcod_east (fsh8)
    assert df.loc["cod_west", "trawlcod"] == 1
    assert df.loc["cod_east", "trawlcod_east"] == 1
    assert df.loc["cod_west", "trawlcod_east"] == 0
    assert df.loc["cod_east", "trawlcod"] == 0
