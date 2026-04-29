"""Verify the calibration script passes workers=-1 to scipy DE."""
import ast
import json
from pathlib import Path


def test_de_call_uses_workers():
    source = (Path(__file__).parent.parent / "scripts" / "calibrate_baltic.py").read_text()
    tree = ast.parse(source)
    de_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "differential_evolution"
    ]
    assert de_calls, "no differential_evolution() call found"
    for call in de_calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "workers" in kw_names, (
            f"differential_evolution call at line {call.lineno} "
            f"does not pass 'workers' kwarg"
        )


def test_phase12_returns_expected_params():
    from scripts.calibrate_baltic import get_phase12_params
    keys, bounds, x0 = get_phase12_params()
    # 16 mortality (8 larval + 8 adult) + 8 fishing + 3 B-H ssb_half (sp3/4/5)
    assert len(keys) == 27, f"phase 12 should expose 27 params, got {len(keys)}"
    assert len(bounds) == 27
    assert len(x0) == 27
    # Must contain mortality + fishing + recruitment keys
    assert any("mortality.additional.larva.rate.sp0" in k for k in keys)
    assert any("mortality.additional.rate.sp0" in k for k in keys)
    assert any("fisheries.rate.base.fsh0" in k for k in keys)
    for sp_idx in [3, 4, 5]:
        assert f"stock.recruitment.ssbhalf.sp{sp_idx}" in keys, (
            f"sp{sp_idx} ssb_half should be DE-tunable in phase 12"
        )
    # Cod (sp0) ssb_half is fixed at the literature prior, NOT a DE param
    assert "stock.recruitment.ssbhalf.sp0" not in keys
    # Flounder + pikeperch fishing bounds should be widened (from Task 1)
    fsh3_idx = keys.index("fisheries.rate.base.fsh3")
    fsh5_idx = keys.index("fisheries.rate.base.fsh5")
    assert bounds[fsh3_idx] == (-2.5, 0.5)
    assert bounds[fsh5_idx] == (-2.5, 0.5)
    # Adult mortality upper bounds widened for predated species (from T3alt)
    for sp_idx in [0, 1, 2, 3, 4, 5]:
        adult_idx = keys.index(f"mortality.additional.rate.sp{sp_idx}")
        assert bounds[adult_idx] == (-3.0, 0.7), f"sp{sp_idx} adult mortality should be widened"
    for sp_idx in [6, 7]:
        adult_idx = keys.index(f"mortality.additional.rate.sp{sp_idx}")
        assert bounds[adult_idx] == (-3.0, 0.3), f"sp{sp_idx} adult mortality should be default"
    # B-H ssb_half bounds: flounder wider than perch/pikeperch
    sp3_idx = keys.index("stock.recruitment.ssbhalf.sp3")
    sp4_idx = keys.index("stock.recruitment.ssbhalf.sp4")
    sp5_idx = keys.index("stock.recruitment.ssbhalf.sp5")
    assert bounds[sp3_idx] == (3.7, 5.3)
    assert bounds[sp4_idx] == (2.7, 4.7)
    assert bounds[sp5_idx] == (2.7, 4.7)


def test_apply_warm_start_overrides_only_known_keys(tmp_path):
    """warm-start: known keys overridden, unknown kept default, skipped excluded."""
    from scripts.calibrate_baltic import apply_warm_start

    fixture = tmp_path / "prior.json"
    fixture.write_text(json.dumps({
        "log10_parameters": {
            "mortality.additional.rate.sp0": 0.57,
            "fisheries.rate.base.fsh3": -0.81,
            "mortality.additional.larva.rate.sp1": 0.62,
        }
    }))

    param_keys = [
        "mortality.additional.rate.sp0",          # in JSON, but skipped
        "fisheries.rate.base.fsh3",               # in JSON, applied
        "mortality.additional.larva.rate.sp1",    # in JSON, applied
        "stock.recruitment.ssbhalf.sp3",          # NOT in JSON, kept default
    ]
    x0 = [-1.301, -1.398, 0.903, 4.699]
    skip = {"mortality.additional.rate.sp0"}

    new_x0, applied, skipped = apply_warm_start(fixture, param_keys, x0, skip)

    # sp0 stays at default x0 (it's in skip set)
    assert new_x0[0] == -1.301
    # fsh3 + larva.sp1 are overridden from JSON
    assert new_x0[1] == -0.81
    assert new_x0[2] == 0.62
    # ssb_half stays at default x0 (not in JSON)
    assert new_x0[3] == 4.699
    # tracking lists
    assert applied == ["fisheries.rate.base.fsh3", "mortality.additional.larva.rate.sp1"]
    assert skipped == ["mortality.additional.rate.sp0"]
