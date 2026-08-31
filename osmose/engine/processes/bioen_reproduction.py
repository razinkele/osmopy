"""Gonad-weight-based reproduction for bioenergetic mode.
Matches Java BioenReproductionProcess.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bioen_egg_release(
    gonad_weight: NDArray[np.float64],
    abundance: NDArray[np.float64],
    is_mature: NDArray[np.bool_],
    season: float,
    sex_ratio: float,
    egg_weight_t: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Java ``BioenReproductionProcess.run()``, per school (verified at source, 4.3.3)::

        if (!school.isMature()) continue;
        float wEgg = school.getGonadWeight() * (float) season;
        if (wEgg <= 0) continue;
        school.incrementGonadWeight(-wEgg);
        double nEgg = wEgg * sexRatio / species.getEggWeight() * 1000000
                      * school.getInstantaneousAbundance();

    Java's ``1000000`` converts the gonad from tonnes to grams because
    ``species.egg.weight`` is in grams. Here BOTH weights are in tonnes
    (``egg_weight_t`` is ``config.egg_weight_override`` or the allometric fallback
    ``c * L^b * 1e-6``), so the ratio already has the right units and there is no 1e6.

    Returns
    -------
    (n_eggs, w_egg)
        ``n_eggs`` is the egg count contributed by each school (already multiplied by its
        abundance); ``w_egg`` is the per-FISH gonad mass released, to be subtracted from
        ``gonad_weight``. Both are 0 for immature schools and for schools whose release
        would be non-positive (Java's ``wEgg <= 0 -> continue``, which also leaves the
        gonad untouched — this is a PARTIAL decrement, never a flush).
    """
    released = gonad_weight * season
    w_egg = np.where(is_mature & (released > 0.0), released, 0.0)
    safe_ew = max(float(egg_weight_t), 1e-20)
    n_eggs = w_egg * sex_ratio / safe_ew * abundance
    return n_eggs, w_egg
