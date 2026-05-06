"""C4 Phase B: cooperative cancellation of the Python engine.

The simulate() outer loop checks an optional `cancel_token` (a
threading.Event) once per step and raises SimulationCancelled when set.
Tests verify both the engine-level path (simulate raises) and the
PythonEngine.run path (which propagates the same exception).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine, SimulationCancelled
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate

REPO = Path(__file__).resolve().parents[1]


def _load_minimal_cfg() -> dict[str, str]:
    matches = sorted((REPO / "data" / "minimal").glob("*_all-parameters.csv"))
    cfg = OsmoseConfigReader().read(str(matches[0]))
    return {k: v for k, v in cfg.items() if k != ""}


def test_simulate_returns_normally_when_token_is_none() -> None:
    """Sanity: existing callers (token=None default) keep working."""
    cfg = _load_minimal_cfg()
    ec = EngineConfig.from_dict(cfg)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)
    outputs = simulate(ec, grid, rng)
    assert len(outputs) > 0


def test_simulate_raises_when_token_already_set() -> None:
    """Pre-set token: the loop catches it at step 0 and aborts."""
    cfg = _load_minimal_cfg()
    ec = EngineConfig.from_dict(cfg)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)

    token = threading.Event()
    token.set()  # signal cancellation BEFORE starting

    with pytest.raises(SimulationCancelled, match=r"step 0/"):
        simulate(ec, grid, rng, cancel_token=token)


def test_simulate_raises_when_token_set_mid_run() -> None:
    """Token set after a few steps: the loop catches it on next iteration."""
    cfg = _load_minimal_cfg()
    ec = EngineConfig.from_dict(cfg)
    grid = Grid.from_dimensions(ny=3, nx=3)
    rng = np.random.default_rng(42)

    token = threading.Event()

    # Schedule the token to fire from a worker thread shortly after the
    # simulation starts. Latency is bounded by step duration.
    def _fire() -> None:
        # short delay so simulate() has a chance to enter its loop
        import time
        time.sleep(0.01)
        token.set()

    t = threading.Thread(target=_fire)
    t.start()
    try:
        with pytest.raises(SimulationCancelled, match=r"step \d+/"):
            simulate(ec, grid, rng, cancel_token=token)
    finally:
        t.join()


def test_python_engine_run_in_memory_propagates_cancellation(tmp_path: Path) -> None:
    """End-to-end: a token set before run() must surface as SimulationCancelled."""
    cfg = _load_minimal_cfg()
    engine = PythonEngine()

    token = threading.Event()
    token.set()

    with pytest.raises(SimulationCancelled):
        engine.run_in_memory(cfg, seed=0, cancel_token=token)


def test_python_engine_run_to_disk_propagates_cancellation(tmp_path: Path) -> None:
    """run() (disk path) must also propagate cancellation."""
    cfg = _load_minimal_cfg()
    engine = PythonEngine()

    token = threading.Event()
    token.set()

    with pytest.raises(SimulationCancelled):
        engine.run(cfg, output_dir=tmp_path / "output", seed=0, cancel_token=token)


def test_engine_run_token_default_none_unchanged(tmp_path: Path) -> None:
    """run/run_in_memory must remain compatible with callers that don't pass a token."""
    cfg = _load_minimal_cfg()
    engine = PythonEngine()
    out = engine.run_in_memory(cfg, seed=0)
    assert out is not None
