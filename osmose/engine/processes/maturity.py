"""Which schools are mature: fixed thresholds, or OSMOSE 4.4.0's stochastic maturity ogive.

By default a school is mature when its length and age reach the species' thresholds
(``species.maturity.size`` and ``species.maturity.age``). With
``species.maturity.mode.sp# = stochastic`` (OSMOSE 4.4.0; Java's ``Species.isSexuallyMature``)
maturity is drawn instead. Length at maturity follows a normal distribution with mean
``species.maturity.l50`` and a standard deviation such that 75% of fish are mature at
``species.maturity.l75``. A school that grew from ``L0`` to ``L`` this step and is not yet mature
becomes mature with probability

    (F(L) - F(L0)) / (1 - F(L0)),    F = the normal CDF,

the chance of maturing in this step given that it had not matured at ``L0``. Once mature, a
school stays mature. Java ignores the mode under bioenergetics, and so does this engine.

The draw happens in reproduction (``mature_stochastically``), as in Java, and only for species
that reproduce at all (sex ratio and fecundity above 0). Reproduction, the life-stage output and
the SSB output all ask ``mature_mask``, so they agree on which schools are mature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig
    from osmose.engine.state import SchoolState

#: The standard normal quantile at 0.75: sd = (l75 - l50) / Z75 (Java's Species constructor).
Z75 = 0.674489750196082


def mature_mask(
    state: SchoolState, config: EngineConfig, where: NDArray[np.bool_] | None = None
) -> NDArray[np.bool_]:
    """Mature schools: the drawn flag for a stochastic species, the thresholds otherwise.

    ``where`` restricts the test to some schools (the live-movement view tests only those it
    draws); the result then has one entry per selected school.
    """

    def pick(a: NDArray) -> NDArray:
        return a if where is None else a[where]

    sid = pick(state.species_id)
    fixed = (pick(state.length) >= config.maturity_size[sid]) & (
        pick(state.age_dt) >= config.maturity_age_dt[sid]
    )
    # getattr: the live-movement view also passes light stand-ins for the config.
    stochastic = getattr(config, "maturity_stochastic", None)
    if stochastic is None:
        return fixed
    drawn = pick(state.is_mature) if state.is_mature is not None else np.zeros(len(sid), dtype=bool)
    return np.where(stochastic[sid], drawn, fixed)


def maturation_probability(
    length: NDArray[np.float64],
    length_start: NDArray[np.float64],
    l50: NDArray[np.float64],
    sd: NDArray[np.float64],
) -> NDArray[np.float64]:
    """P(maturing this step | not mature at the start of it). NaN where F(L0) is 1."""
    f_now = ndtr((length - l50) / sd)
    f_start = ndtr((length_start - l50) / sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (f_now - f_start) / (1.0 - f_start)


def mature_stochastically(
    state: SchoolState, config: EngineConfig, rng: np.random.Generator
) -> SchoolState:
    """Draw maturity for the schools of stochastic species that are not mature yet.

    One uniform per such school, in state order; nothing is drawn when no species is
    stochastic, so the random stream of every other config is untouched.
    """
    if config.maturity_stochastic is None or len(state) == 0:
        return state
    assert config.maturity_l50 is not None and config.maturity_sd is not None  # set together
    sid = state.species_id
    drawn = (
        state.is_mature.copy() if state.is_mature is not None else np.zeros(len(state), dtype=bool)
    )
    reproduces = (config.sex_ratio > 0) & (config.relative_fecundity > 0)
    candidates = np.flatnonzero(config.maturity_stochastic[sid] & reproduces[sid] & ~drawn)
    if len(candidates):
        s = sid[candidates]
        prob = maturation_probability(
            state.length[candidates],
            state.length_start[candidates],
            config.maturity_l50[s],
            config.maturity_sd[s],
        )
        drawn[candidates] = rng.random(len(candidates)) < prob  # NaN compares False
    return state.replace(is_mature=drawn)
