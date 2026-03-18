"""Fishing selectivity functions for the OSMOSE Python engine.

Selectivity determines what fraction of a species is vulnerable to fishing
based on body size.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def knife_edge(length: NDArray[np.float64], l50: float) -> NDArray[np.float64]:
    """Knife-edge selectivity: 0 below L50, 1 at or above."""
    return np.where(length >= l50, 1.0, 0.0)


def sigmoid(length: NDArray[np.float64], l50: float, slope: float = 1.0) -> NDArray[np.float64]:
    """Logistic sigmoid selectivity centered at L50."""
    return 1.0 / (1.0 + np.exp(-slope * (length - l50)))
