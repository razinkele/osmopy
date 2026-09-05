"""C4 delta-spec schema validation: arm structure, rationales, citations, and dead knob check."""

import json
from pathlib import Path

SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "data/baltic/scenarios/c4_salinity_sensitivity.json"
)


def _spec():
    return json.loads(SPEC_PATH.read_text())


def test_arms_and_salinity_deltas():
    arms = _spec()["arms"]
    assert [a["name"] for a in arms] == ["ds_m1", "ds_m2", "ds_m3"]
    assert [a["dS_PSU"] for a in arms] == [-1.0, -2.0, -3.0]


def test_every_arm_has_rationale_no_citation():
    for a in _spec()["arms"]:
        assert "rationale" in a and len(a["rationale"]) > 0
        assert "citation" not in a  # levers are chosen, not cited


def test_context_citations_contain_meier():
    citations = _spec()["context_citations"]
    assert all("Meier" in citations[key] for key in citations)


def test_provenance_records_mechanism_characterization():
    assert "not projections" in _spec()["_provenance"]


def test_no_dead_knobs():
    for a in _spec()["arms"]:
        for dead in ("dT_C", "dO2", "referent"):
            assert dead not in a  # spec: JSON carries only what machinery consumes
