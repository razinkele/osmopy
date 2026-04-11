"""Johnson thermal performance curve and Arrhenius function.
Matches Java TempFunction class for bioenergetic module.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

K_B = 8.62e-5  # Boltzmann constant (eV/K), used in Arrhenius and Johnson thermal curves


def phi_t(temp_c: NDArray[np.float64], e_m: float, e_d: float, t_p: float) -> NDArray[np.float64]:
    """Johnson thermal performance curve, normalized so phi_T(T_P) = 1.0.

    Args:
        temp_c: Temperature in Celsius.
        e_m: Increasing activation energy (eV).
        e_d: Declining activation energy (eV).
        t_p: Peak temperature (Celsius).

    Degenerate case: when ``abs(e_d - e_m) < 1e-30`` the declining phase of
    the Johnson curve is ill-defined (division by the activation-energy
    gap), and this function falls back to the pure Arrhenius form
    ``exp(-e_m / (K_B * T))``. The fallback is silent by design — configs
    with ``e_d ≈ e_m`` are physically uncommon but not strictly invalid,
    and raising here would break any simulation that ends up on the edge
    through rounding. Callers that need strict Johnson semantics should
    validate their activation-energy parameters upstream.
    """
    t_k = temp_c + 273.15
    t_p_k = t_p + 273.15

    def _raw(t):
        num = np.exp(-e_m / (K_B * t))
        delta = e_d - e_m
        if abs(delta) < 1e-30:
            # Degenerate case: no declining phase, return Arrhenius-only
            return num
        ratio = e_m / delta
        denom = 1.0 + ratio * np.exp(e_d / K_B * (1.0 / t_p_k - 1.0 / t))
        return num / denom

    return _raw(t_k) / _raw(np.asarray(t_p_k))


def arrhenius(temp_c: NDArray[np.float64], e_m: float) -> NDArray[np.float64]:
    """Arrhenius function for maintenance metabolic rate.

    Args:
        temp_c: Temperature in Celsius.
        e_m: Activation energy (eV).
    """
    t_k = temp_c + 273.15
    return np.exp(-e_m / (K_B * t_k))
