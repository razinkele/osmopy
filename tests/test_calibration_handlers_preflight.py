"""If optimization runs without preflight, we must fail loudly (not silently pass None)."""
import pytest


def test_preflight_invariant_raises():
    from ui.pages.calibration_handlers import _require_preflight

    with pytest.raises(RuntimeError, match="preflight"):
        _require_preflight(None, None, None)


def test_preflight_invariant_passes_with_all_paths():
    from pathlib import Path
    from ui.pages.calibration_handlers import _require_preflight

    a, b, c = _require_preflight(Path("/tmp/a"), Path("/tmp/b"), Path("/tmp/c"))
    assert (a, b, c) == (Path("/tmp/a"), Path("/tmp/b"), Path("/tmp/c"))
