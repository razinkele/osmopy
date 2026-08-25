"""Percid thermal recruitment gate — pure helpers.

Engine-state-free. The response curve + mode normalization are applied in the
config loader (osmose/engine/config.py:_load_thermal_gate); the per-step
multiplier is read back by thermal_gate_factor. Percid year-class strength is
temperature-gated (Pekcan-Hekim et al. 2011, Ambio; Olin et al. 2019,
Hydrobiologia).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig


def logistic_response(temp: NDArray[np.float64], t50: float, slope: float) -> NDArray[np.float64]:
    """Saturating year-class response to summer temperature, in (0, 1).

    0.5 at temp == t50; rises over ~slope degrees. Logistic (not linear)
    encodes that strong percid year-classes are the exception: cool years
    mostly fail, warm years above threshold succeed.
    """
    return 1.0 / (1.0 + np.exp(-(temp - t50) / slope))


def exponential_response(
    temp: NDArray[np.float64], beta: float, tref: float
) -> NDArray[np.float64]:
    """Voss & Quaas (2026, doi:10.1093/icesjms/fsag033) productivity factor
    exp(beta * (T - tref)). beta < 0 encodes warming-reduces-recruitment; the
    factor is exactly 1.0 at T == tref and is deliberately uncapped above 1
    (the paper's Beverton-Holt numerator has no cap). Scenario knob — see spec
    2026-08-25; NOT a validated mechanism.
    """
    return np.exp(beta * (temp - tref))


def normalize_factor(
    r: NDArray[np.float64],
    mode: str,
    r_ref: float,
    window_idx: list[int],
    floor: float,
) -> NDArray[np.float64]:
    """Turn a per-year response into a per-year egg multiplier.

    raw (exponential response only): the factor IS the response; tref
    anchoring replaces normalisation (C1 spec decision 8).
    thermal_cap (mean-reducing): clip(r / r_ref, 0, 1) — most years < 1.
    mean_preserving (realism only): r / mean(r over the sampled model years).
    All modes are then floored at ``floor``.

    ``floor`` > 0 is rejected under mean_preserving: a nonzero floor raises the
    small values after normalization and destroys the unit-mean property that
    is the entire point of the mode (review finding 4).
    """
    if mode == "raw":
        # exponential response only: the factor IS the response; tref anchoring
        # replaces normalisation (C1 spec decision 8).
        factor = r.copy()
    elif mode == "thermal_cap":
        factor = np.clip(r / r_ref, 0.0, 1.0)
    elif mode == "mean_preserving":
        if floor > 0.0:
            raise ValueError(
                "floor>0 is incompatible with mean_preserving (it breaks the "
                "unit-mean property); use floor=0.0 with mean_preserving."
            )
        denom = float(np.mean(r[window_idx]))
        if denom == 0.0:
            raise ValueError("mean_preserving denominator is 0 over the run window.")
        factor = r / denom
    else:
        raise ValueError(f"unknown thermal gate mode: {mode!r}")
    return np.maximum(factor, floor)


def thermal_gate_factor(config: "EngineConfig", step: int) -> NDArray[np.float64]:
    """Per-species egg-production multiplier for this timestep.

    1.0 for every species when the gate is off or the species is disabled;
    otherwise the current model year's per-species factor (constant within a
    model year), with the series index wrapping if the run outlasts the series.
    """
    out = np.ones(config.n_species, dtype=np.float64)
    factor = config.thermal_gate_factor_by_index
    if factor is None:
        return out
    year = step // config.n_dt_per_year
    idx = (config.thermal_gate_offset + year) % factor.shape[0]
    mask = config.thermal_gate_enabled
    out[mask] = factor[idx, mask]
    return out
