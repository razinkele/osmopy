"""Fishing selectivity functions for the OSMOSE Python engine.

Selectivity determines what fraction of a species is vulnerable to fishing
based on body size. Four types matching Java FisherySelectivity:
  0 = knife-edge (age or length)
  1 = sigmoid (logistic, parameterized by L50/L75)
  2 = Gaussian (normal, peak at L50)
  3 = log-normal (right-skewed, normalized by mode)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import lognorm, norm

# qnorm(0.75) — the 75th percentile of the standard normal
_Q75 = 0.674489750196082


def knife_edge(length: NDArray[np.float64], l50: float) -> NDArray[np.float64]:
    """Knife-edge selectivity: 0 below L50, 1 at or above."""
    return np.where(length >= l50, 1.0, 0.0)


def sigmoid(
    length: NDArray[np.float64], l50: float, l75: float, tiny: float = 1e-8
) -> NDArray[np.float64]:
    """Logistic sigmoid selectivity -- type 1 in Java.

    Parameterized by L50 (50% selectivity) and L75 (75% selectivity).
    Formula: 1 / (1 + exp(s1 - s2 * x)) where:
      s1 = (L50 * log(3)) / (L75 - L50)
      s2 = s1 / L50
    Matches Java FisherySelectivity.getSigmoidSelectivity().
    """
    s1 = (l50 * np.log(3)) / (l75 - l50)
    s2 = s1 / l50
    sel = 1.0 / (1.0 + np.exp(s1 - s2 * length))
    sel[sel < tiny] = 0.0
    return sel


def sigmoid_slope(
    length: NDArray[np.float64], l50: float, slope: float = 1.0
) -> NDArray[np.float64]:
    """Logistic sigmoid selectivity with slope parameter (legacy).

    Retained for backward compatibility with configs that provide slope
    instead of L75.
    """
    return 1.0 / (1.0 + np.exp(-slope * (length - l50)))


def gaussian(
    length: NDArray[np.float64], l50: float, l75: float, tiny: float = 1e-8
) -> NDArray[np.float64]:
    """Gaussian (normal) selectivity -- type 2 in Java.

    Peak at L50, normalized so selectivity(L50) = 1.0.
    sd = (L75 - L50) / qnorm(0.75).
    Matches Java FisherySelectivity.getGaussianSelectivity().
    """
    sd = (l75 - l50) / _Q75
    peak_density = norm.pdf(l50, loc=l50, scale=sd)
    sel = norm.pdf(length, loc=l50, scale=sd) / peak_density
    sel[sel < tiny] = 0.0
    return sel


def log_normal(
    length: NDArray[np.float64], l50: float, l75: float, tiny: float = 1e-8
) -> NDArray[np.float64]:
    """Log-normal selectivity -- type 3 in Java.

    Normalized by mode density. Parameters:
    mean = log(L50), sd = log(L75/L50) / qnorm(0.75).
    mode = exp(mean - sd**2).
    Matches Java FisherySelectivity.getLogNormalSelectivity().
    """
    mean = np.log(l50)
    sd = np.log(l75 / l50) / _Q75
    mode = np.exp(mean - sd**2)
    # scipy lognorm: s=sd, scale=exp(mean)
    scale = np.exp(mean)
    mode_density = lognorm.pdf(mode, s=sd, scale=scale)
    sel = lognorm.pdf(length, s=sd, scale=scale) / mode_density
    sel[sel < tiny] = 0.0
    return sel
