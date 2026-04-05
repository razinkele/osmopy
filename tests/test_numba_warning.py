"""Test that missing Numba triggers a warning."""

import importlib
import logging
import sys


def test_predation_warns_without_numba(caplog) -> None:
    """Predation module logs a warning when numba is unavailable."""
    mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.predation")]
    for m in mods_to_remove:
        del sys.modules[m]
    orig = sys.modules.pop("numba", "NOT_SET")
    sys.modules["numba"] = None  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.WARNING, logger="osmose.engine.processes.predation"):
            importlib.import_module("osmose.engine.processes.predation")
        numba_msgs = [r.message for r in caplog.records if "numba" in r.message.lower()]
        assert len(numba_msgs) >= 1, (
            f"Expected numba warning in log, got: {[r.message for r in caplog.records]}"
        )
    finally:
        del sys.modules["numba"]
        if orig != "NOT_SET":
            sys.modules["numba"] = orig
        mods_to_remove = [
            k for k in sys.modules if k.startswith("osmose.engine.processes.predation")
        ]
        for m in mods_to_remove:
            del sys.modules[m]


def test_mortality_warns_without_numba(caplog) -> None:
    """Mortality module logs a warning when numba is unavailable."""
    mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.mortality")]
    for m in mods_to_remove:
        del sys.modules[m]
    orig = sys.modules.pop("numba", "NOT_SET")
    sys.modules["numba"] = None  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.WARNING, logger="osmose.engine.processes.mortality"):
            importlib.import_module("osmose.engine.processes.mortality")
        numba_msgs = [r.message for r in caplog.records if "numba" in r.message.lower()]
        assert len(numba_msgs) >= 1, (
            f"Expected numba warning in log, got: {[r.message for r in caplog.records]}"
        )
    finally:
        del sys.modules["numba"]
        if orig != "NOT_SET":
            sys.modules["numba"] = orig
        mods_to_remove = [
            k for k in sys.modules if k.startswith("osmose.engine.processes.mortality")
        ]
        for m in mods_to_remove:
            del sys.modules[m]


def test_movement_warns_without_numba(caplog) -> None:
    """Movement module logs a warning when numba is unavailable."""
    mods_to_remove = [k for k in sys.modules if k.startswith("osmose.engine.processes.movement")]
    for m in mods_to_remove:
        del sys.modules[m]
    orig = sys.modules.pop("numba", "NOT_SET")
    sys.modules["numba"] = None  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.WARNING, logger="osmose.engine.processes.movement"):
            importlib.import_module("osmose.engine.processes.movement")
        numba_msgs = [r.message for r in caplog.records if "numba" in r.message.lower()]
        assert len(numba_msgs) >= 1, (
            f"Expected numba warning in log, got: {[r.message for r in caplog.records]}"
        )
    finally:
        del sys.modules["numba"]
        if orig != "NOT_SET":
            sys.modules["numba"] = orig
        mods_to_remove = [
            k for k in sys.modules if k.startswith("osmose.engine.processes.movement")
        ]
        for m in mods_to_remove:
            del sys.modules[m]
