"""Tests that simulation state is isolated per-run (no module globals)."""

import numpy as np

from osmose.engine.processes.predation import get_diet_matrix
from osmose.engine.simulate import SimulationContext


def test_diet_matrix_not_global():
    """get_diet_matrix without context should return None."""
    matrix = get_diet_matrix()
    assert matrix is None, "Diet matrix should not persist as module-level state"


def test_diet_matrix_with_context():
    """get_diet_matrix returns the matrix from the context."""
    ctx = SimulationContext()
    ctx.diet_matrix = np.zeros((3, 2), dtype=np.float64)
    matrix = get_diet_matrix(ctx=ctx)
    assert matrix is not None
    assert matrix.shape == (3, 2)


def test_contexts_are_independent():
    """Two SimulationContext instances should not share state."""
    ctx1 = SimulationContext()
    ctx2 = SimulationContext()

    ctx1.diet_tracking_enabled = True
    ctx1.diet_matrix = np.ones((2, 2), dtype=np.float64)
    ctx1.tl_weighted_sum = np.ones(5, dtype=np.float64)

    assert ctx2.diet_tracking_enabled is False
    assert ctx2.diet_matrix is None
    assert ctx2.tl_weighted_sum is None


def test_config_dir_isolated():
    """config_dir should be per-context, not shared."""
    ctx1 = SimulationContext(config_dir="/path/a")
    ctx2 = SimulationContext(config_dir="/path/b")
    assert ctx1.config_dir == "/path/a"
    assert ctx2.config_dir == "/path/b"
