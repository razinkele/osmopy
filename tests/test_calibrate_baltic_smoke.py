"""Smoke test: the Baltic calibration CLI loads and `--help` succeeds."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def test_calibrate_baltic_module_loads_via_spec() -> None:
    """Load the script from disk (scripts/ has no __init__.py).

    The module must be inserted into ``sys.modules`` before ``exec_module``
    so that any ``@dataclass`` declarations it contains can find their own
    module in ``sys.modules[cls.__module__]`` during decoration.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_baltic.py"
    spec = importlib.util.spec_from_file_location("calibrate_baltic", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_baltic"] = module
    try:
        spec.loader.exec_module(module)
        assert hasattr(module, "run_simulation")
        assert hasattr(module, "make_objective")
    finally:
        sys.modules.pop("calibrate_baltic", None)


def test_calibrate_baltic_help() -> None:
    """``--help`` should succeed and mention the calibration role."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_baltic.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Baltic" in result.stdout or "calibrat" in result.stdout.lower()
