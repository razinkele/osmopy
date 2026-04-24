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
