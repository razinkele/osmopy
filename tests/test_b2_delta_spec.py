"""B2 delta-spec schema validation (spec §1): every number cited, one referent,
no dead knobs, conversions self-consistent."""

import json
from pathlib import Path

SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "data/baltic/scenarios/b2_literature_deltas.json"
)


def _spec():
    return json.loads(SPEC_PATH.read_text())


def test_arms_and_matrix():
    arms = _spec()["arms"]
    assert [a["name"] for a in arms] == ["rcp45_bsap", "rcp45_ref", "rcp85_bsap", "rcp85_ref"]
    assert all(a["dT_C"] in (1.9, 2.9) for a in arms)
    assert {(a["rcp"], a["load"]) for a in arms} == {
        ("RCP4.5", "BSAP"),
        ("RCP4.5", "REF"),
        ("RCP8.5", "BSAP"),
        ("RCP8.5", "REF"),
    }


def test_every_number_cited_and_single_referent():
    for a in _spec()["arms"]:
        assert "Meier2022 Table 7" in a["dT_source"]
        d = a["dO2"]
        assert d["referent"] == "summer_bottom_o2"  # spec decision 4: the only accepted referent
        assert "Meier2022 Table 10" in d["source"]


def test_conversion_self_consistent():
    for a in _spec()["arms"]:
        d = a["dO2"]
        assert abs(d["value_mmol_m3"] - d["value_mL_L"] * d["conversion_mmol_per_mL_L"]) < 0.05


def test_no_dead_knobs():
    for a in _spec()["arms"]:
        for dead in ("ltl_scale", "salinity", "time_slice"):
            assert dead not in a  # spec §1: JSON carries only what machinery consumes


def test_provenance_records_relabel():
    assert "overstates end-century forcing" in _spec()["_provenance"]  # decision 5
