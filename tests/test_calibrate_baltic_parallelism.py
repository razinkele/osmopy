"""Verify the calibration script passes workers=-1 to scipy DE."""
import ast
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


def test_phase12_returns_24_params():
    from scripts.calibrate_baltic import get_phase12_params
    keys, bounds, x0 = get_phase12_params()
    assert len(keys) == 24, f"phase 12 should expose 24 params, got {len(keys)}"
    assert len(bounds) == 24
    assert len(x0) == 24
    # Must contain both mortality + fishing keys
    assert any("mortality.additional.larva.rate.sp0" in k for k in keys)
    assert any("mortality.additional.rate.sp0" in k for k in keys)
    assert any("fisheries.rate.base.fsh0" in k for k in keys)
    # Flounder + pikeperch fishing bounds should be widened (from Task 1)
    fsh3_idx = keys.index("fisheries.rate.base.fsh3")
    fsh5_idx = keys.index("fisheries.rate.base.fsh5")
    assert bounds[fsh3_idx] == (-2.5, 0.5)
    assert bounds[fsh5_idx] == (-2.5, 0.5)
    # Adult mortality upper bounds widened for predated species (from T3alt)
    # mortality.additional.rate.sp0 is at index 8 (after 8 larval mortality)
    for sp_idx in [0, 1, 2, 3, 4, 5]:
        adult_idx = keys.index(f"mortality.additional.rate.sp{sp_idx}")
        assert bounds[adult_idx] == (-3.0, 0.7), f"sp{sp_idx} adult mortality should be widened"
    for sp_idx in [6, 7]:
        adult_idx = keys.index(f"mortality.additional.rate.sp{sp_idx}")
        assert bounds[adult_idx] == (-3.0, 0.3), f"sp{sp_idx} adult mortality should be default"
