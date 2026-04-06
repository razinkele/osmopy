# tests/test_economics_memory.py
"""Tests for vessel catch memory (exponential moving average)."""

import numpy as np
import pytest

from osmose.engine.economics.choice import update_catch_memory


class TestCatchMemory:
    def test_ema_update(self):
        """Memory = decay × old + (1 - decay) × new."""
        memory = np.array([[[100.0, 50.0], [0.0, 0.0]]])  # (1 fleet, 2, 2)
        realized = np.array([[[0.0, 200.0], [0.0, 0.0]]])
        decay = 0.7
        updated = update_catch_memory(memory, realized, decay)
        # Cell (0,0): 0.7*100 + 0.3*0 = 70
        assert updated[0, 0, 0] == pytest.approx(70.0)
        # Cell (0,1): 0.7*50 + 0.3*200 = 95
        assert updated[0, 0, 1] == pytest.approx(95.0)

    def test_zero_decay_replaces(self):
        """decay=0 → memory completely replaced by new observation."""
        memory = np.array([[[100.0]]])
        realized = np.array([[[42.0]]])
        updated = update_catch_memory(memory, realized, decay=0.0)
        assert updated[0, 0, 0] == pytest.approx(42.0)
