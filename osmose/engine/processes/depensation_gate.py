"""Recruitment depensation / Allee gate — pure helper.

Engine-state-free. The multiplier is a function of the CURRENT per-species SSB
(state-dependent, unlike the step-driven RV/thermal gates). Applied to egg
production in reproduction() after apply_stock_recruitment. A depensatory Allee
term creates a low-SSB recruitment trap (Liermann-Hilborn form).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def depensation_factor(
    ssb: NDArray[np.float64],
    s50: NDArray[np.float64],
    theta: NDArray[np.float64],
    enabled: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Per-species Allee multiplier A(SSB)=SSB^theta/(S50^theta+SSB^theta), in (0, 1].

    1.0 where disabled. A->0 as SSB->0, A=0.5 at SSB==S50, A->1 as SSB->inf.
    All arguments are length n_sp.
    """
    out = np.ones(ssb.shape[0], dtype=np.float64)
    for sp in range(ssb.shape[0]):
        if not enabled[sp]:
            continue
        s = ssb[sp]
        if s <= 0.0:
            out[sp] = (
                0.0  # full suppression at SSB=0; harmless (n_eggs already 0) + skipped-when-seeded
            )
            continue
        with np.errstate(over="ignore"):
            top = s ** theta[sp]
            half = s50[sp] ** theta[sp]
        if np.isinf(top) or np.isinf(half):
            # A power that does not fit a double (theta in the dozens): inf / (inf + inf) would
            # be NaN, and the NaN would go into the egg count. The same term written as ratios,
            # each at most 1. For powers that fit, the line below is untouched, so an ordinary
            # result does not move by a bit (the C# engine is compared with it bit for bit).
            if s >= s50[sp]:
                out[sp] = 1.0 / (1.0 + (s50[sp] / s) ** theta[sp])
            else:
                low = (s / s50[sp]) ** theta[sp]
                out[sp] = low / (1.0 + low)
            continue
        out[sp] = top / (half + top)
    return out
