"""Bioenergetic predation — allometric ingestion cap.
Matches Java BioenPredationMortality.

**Unit hazard for whoever writes the bioen overlay.** Java's
``getMaxPredationRate`` (``BioenPredationMortality.java:241-276``) early-returns
``predationRateBioen[speciesIndex]`` for background predators
(``speciesIndex >= nSpecies``) *before* the ``/ getNStepYear()`` that the focal
branch applies. ``per_fish_ingestion_cap`` replicates that verbatim, so the
FOCAL half of ``config.bioen_i_max_all`` is consumed as a per-YEAR rate while the
BACKGROUND half is consumed as a per-TIME-STEP rate. Both halves are currently
filled from the same per-year key family (``config.py``: focal
``predation.ingestion.rate.max.sp{i}``, background
``BackgroundSpeciesInfo.ingestion_rate``), so a background Imax authored on the
focal convention overshoots by a factor of ``n_dt_per_year``. Background entries
must be supplied already in per-time-step units.

A second known gap on the same path: Java reads ``species.beta.sp{fileindex}``
for background species (``BackgroundSpecies.java:130-133``) via ``cfg.getDouble``,
which has **no default** — a missing key is a fatal
``error("Could not find parameter ...")`` in ``Configuration.getParameter``
(``:998-1006``). The port has no background beta at all and silently substitutes
0.8, which is the PORT's focal default (``config.py:2499``), not a Java default.
So where Java refuses to start, the port runs with an invented exponent. Both
gaps are Gate-B prerequisites for the overlay, not defects of the cap formula.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bioen_ingestion_cap(
    weight: NDArray[np.float64],
    i_max: float,
    beta: float,
    n_dt_per_year: int,
    n_subdt: int,
    is_larvae: NDArray[np.bool_],
    theta: float = 1.0,
    c_rate: float = 0.0,
) -> NDArray[np.float64]:
    """Compute max ingestion per sub-timestep for bioen mode.

    Adults: I_max * w_g^beta / (n_dt * subdt)
    Larvae: (I_max + (theta-1)*c_rate) * w_g^beta / (n_dt * subdt)

    Weight is converted from tonnes to grams internally.

    Superseded by :func:`per_fish_ingestion_cap` on the live mortality path — this
    per-species scalar form is retained only for the standalone unit tests.
    """
    w_grams = weight * 1e6
    i_eff = np.where(is_larvae, i_max + (theta - 1.0) * c_rate, i_max)
    return i_eff * np.power(w_grams, beta) / (n_dt_per_year * n_subdt)


def per_fish_ingestion_cap(
    weight: NDArray[np.float64],
    species_id: NDArray[np.int32],
    age_dt: NDArray[np.int32],
    i_max_all: NDArray[np.float64],
    beta: NDArray[np.float64],
    larvae_thres_dt: NDArray[np.int32],
    theta: NDArray[np.float64],
    c_rate: NDArray[np.float64],
    n_species: int,
    n_dt_per_year: int,
    n_subdt: int,
) -> NDArray[np.float64]:
    """Java ``getMaxPredationRate`` x ``(w*1e6)^beta / subdt``, per FISH, in tonnes.

    ``max_eatable`` for a school is ``cap[p] * instantaneous_abundance[p]``: Java
    multiplies by ``getInstantaneousAbundance()`` at every predator visit
    (``BioenPredationMortality.java:140-145``), so the cap must stay per-fish here and
    be scaled at the use site rather than post-hoc against a per-school total.

    Focal species (``species_id < n_species``)::

        Imax_eff = (Imax + (theta - 1) * c_rate) / n_dt_per_year   if ageDt < larvaeThresDt
                 =  Imax / n_dt_per_year                            otherwise

    Background predators (``species_id >= n_species``) take ``i_max_all[species_id]``
    with **no ``n_dt_per_year`` division** (Java's early return) and beta 0.8. See this
    module's docstring for the unit hazard that creates.
    """
    sp = np.asarray(species_id)
    is_focal = sp < n_species
    sp_f = np.where(is_focal, sp, 0)
    b = np.where(is_focal, beta[sp_f], 0.8)
    imax = i_max_all[sp]
    larval = is_focal & (np.asarray(age_dt) < larvae_thres_dt[sp_f])
    i_focal = np.where(larval, imax + (theta[sp_f] - 1.0) * c_rate[sp_f], imax) / n_dt_per_year
    i_eff = np.where(is_focal, i_focal, imax)
    return i_eff * np.power(weight * 1e6, b) / n_subdt * 1e-6
