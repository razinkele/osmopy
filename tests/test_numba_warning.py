"""Test that missing Numba triggers a warning."""

import importlib
import sys
import warnings


def test_predation_warns_without_numba() -> None:
    """Predation module emits ImportWarning when numba is unavailable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.predation")]
        for m in mods_to_remove:
            del sys.modules[m]
        orig = sys.modules.pop("numba", "NOT_SET")
        sys.modules["numba"] = None  # type: ignore[assignment]
        try:
            importlib.import_module("osmose.engine.processes.predation")
            numba_warnings = [x for x in w if "numba" in str(x.message).lower()]
            assert len(numba_warnings) >= 1, f"Expected numba warning, got: {[str(x.message) for x in w]}"
        finally:
            del sys.modules["numba"]
            if orig != "NOT_SET":
                sys.modules["numba"] = orig
            mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.predation")]
            for m in mods_to_remove:
                del sys.modules[m]


def test_mortality_warns_without_numba() -> None:
    """Mortality module emits ImportWarning when numba is unavailable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.mortality")]
        for m in mods_to_remove:
            del sys.modules[m]
        orig = sys.modules.pop("numba", "NOT_SET")
        sys.modules["numba"] = None  # type: ignore[assignment]
        try:
            importlib.import_module("osmose.engine.processes.mortality")
            numba_warnings = [x for x in w if "numba" in str(x.message).lower()]
            assert len(numba_warnings) >= 1, f"Expected numba warning, got: {[str(x.message) for x in w]}"
        finally:
            del sys.modules["numba"]
            if orig != "NOT_SET":
                sys.modules["numba"] = orig
            mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.mortality")]
            for m in mods_to_remove:
                del sys.modules[m]


def test_movement_warns_without_numba() -> None:
    """Movement module emits ImportWarning when numba is unavailable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.movement")]
        for m in mods_to_remove:
            del sys.modules[m]
        orig = sys.modules.pop("numba", "NOT_SET")
        sys.modules["numba"] = None  # type: ignore[assignment]
        try:
            importlib.import_module("osmose.engine.processes.movement")
            numba_warnings = [x for x in w if "numba" in str(x.message).lower()]
            assert len(numba_warnings) >= 1, f"Expected numba warning, got: {[str(x.message) for x in w]}"
        finally:
            del sys.modules["numba"]
            if orig != "NOT_SET":
                sys.modules["numba"] = orig
            mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.movement")]
            for m in mods_to_remove:
                del sys.modules[m]
