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


def test_cod_east_mortality_realistic_not_elevated():
    """cod_east's collapse is represented by its RV recruitment gate (low
    recruitment), NOT by elevated mortality. The original disaggregation used a
    doubled additional M (2.5) to stand in for the collapse, but that drove
    cod_east extinct; the persistence fix set it to a realistic eastern-Baltic-cod
    natural M (~0.9-1.1) and lets the RV gate carry the collapse. So cod_east's
    additional M is intentionally in the realistic band and is NOT elevated above
    cod_west's calibrated value. See test_rv_gate_on_cod_east_only_raw_cap for the
    collapse mechanism and docs/baltic_cod_east_fix_certification_2026-07-28.md."""
    cfg = _cfg()
    m_east = float(cfg["mortality.additional.rate.sp8"])
    assert 0.8 <= m_east <= 1.2, (
        f"cod_east additional M {m_east} outside the realistic eastern-cod band [0.8, 1.2] "
        "(the old doubled M=2.5 extincted it — the RV gate now represents the collapse)"
    )
    # the collapse is the RV gate, not an extreme mortality
    assert cfg["reproduction.rv.gate.species.enabled.sp8"] == "true"


def test_cod_east_has_separate_fishery():
    cfg = _cfg()
    assert cfg["simulation.nfisheries"] == "9"
    # java-safe canonical form (no underscore — see test_baltic_java_compat)
    assert cfg["fisheries.name.fsh8"] == "trawlcodeast"


def test_catchability_maps_each_stock_to_its_fishery():
    df = pd.read_csv("data/baltic/fishery-catchability.csv", index_col=0)
    assert "cod_west" in df.index and "cod_east" in df.index
    assert "cod" not in df.index
    # cod_west -> trawlcod (fsh0), cod_east -> trawlcodeast (fsh8)
    assert df.loc["cod_west", "trawlcod"] == 1
    assert df.loc["cod_east", "trawlcodeast"] == 1
    assert df.loc["cod_west", "trawlcodeast"] == 0
    assert df.loc["cod_east", "trawlcod"] == 0
