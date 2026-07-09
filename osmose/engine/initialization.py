"""Warm-start standing-stock initialization (opt-in, inert by default).

Given a per-species initial biomass, seed an age-structured standing adult
population at t=0 (numbers-at-age ~ exp(-Z*age), Von Bertalanffy length,
allometric weight), so a genuine adult community — including a clupeid-dominated
alternative state — can exist at t=0 for a reciprocal-invasion / hysteresis test.
Mirrors osmose/engine/incoming_flux.py. Gated by the canonical config flag
``module.population.initialisation.enabled`` (default false => empty init,
byte-identical to the current Java-convention empty population).

When the flag is on, egg-seeding is disabled (config sets seeding_max_step=0) so
the standing stock evolves under fixed parameters without the SSB==0 egg-rescue
continuously re-injecting a suppressed species — see config._load_reproduction.

The age structure is a COARSE starting shape: the decay rate is a life-history
proxy (max of the residual additional mortality and 1.5*K), not a true total Z,
so the seeded stock re-equilibrates under the model's own dynamics within a few
years. It is an initial condition, not an equilibrium the model must preserve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.state import SchoolState

_ENABLE_KEY = "module.population.initialisation.enabled"  # canonical (post-4.4.0) key
_ENABLE_KEY_LEGACY = "population.initialization.relativebiomass.enabled"  # pre-4.4.0 alias
_MIN_DECAY = 0.05  # floor on the age-decay rate so the structure isn't flat when the proxy is ~0


def _flag_enabled(raw: dict) -> bool:
    val = raw.get(_ENABLE_KEY, raw.get(_ENABLE_KEY_LEGACY, "false"))
    return str(val).lower() == "true"


def age_structured_population(
    target_biomass: float,
    linf: float,
    k: float,
    t0: float,
    cf: float,
    ap: float,
    mortality: float,
    lifespan_years: float,
    n_dt_per_year: int,
    min_length: float = 0.001,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Distribute target_biomass (tonnes) across integer age classes 0..floor(lifespan).

    Returns (ages_dt, lengths_cm, weights_tonnes, abundances) aligned per age class,
    with sum(abundances*weights) == target_biomass. Empty arrays if nothing to seed.
    Uses floor(lifespan) classes at mid-year ages so no cohort is created at/beyond the
    aging-death threshold (lifespan_dt).
    """
    empty = (
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )
    if target_biomass <= 0.0 or lifespan_years <= 0.0:
        return empty
    n_ages = max(1, int(lifespan_years))  # floor
    ages_years = np.arange(n_ages, dtype=np.float64) + 0.5  # mid-year; max = n_ages-0.5 < lifespan
    lengths = linf * (1.0 - np.exp(-k * (ages_years - t0)))
    lengths = np.maximum(lengths, min_length)
    weights = cf * lengths**ap * 1e-6  # grams -> tonnes
    decay = max(float(mortality), 1.5 * abs(float(k)), _MIN_DECAY)  # coarse total-Z proxy
    numbers = np.exp(-decay * ages_years)
    rel_biomass = numbers * weights
    total = float(rel_biomass.sum())
    if total <= 0.0:
        return empty
    abundances = (target_biomass / total) * numbers
    lifespan_dt = int(lifespan_years * n_dt_per_year)
    ages_dt = np.round(ages_years * n_dt_per_year).astype(np.int64)
    ages_dt = np.clip(ages_dt, 0, max(0, lifespan_dt - 1)).astype(np.int32)
    return ages_dt, lengths, weights, abundances


def build_initial_population(config, grid, rng) -> SchoolState:
    """Age-structured standing population at t=0, or an empty SchoolState if the flag is off.

    The flag-off path returns SchoolState.create(n_schools=0), identical to the engine's
    default empty initialization (parity).
    """
    empty = SchoolState.create(n_schools=0)
    raw = getattr(config, "raw_config", {}) or {}
    if not _flag_enabled(raw):
        return empty
    ys, xs = np.where(grid.ocean_mask)
    if len(ys) == 0:
        return empty
    ys = ys.astype(np.int32)
    xs = xs.astype(np.int32)
    parts: list[SchoolState] = []
    for sp in range(config.n_species):
        target = float(config.seeding_biomass[sp])
        if target <= 0.0:
            continue
        ages_dt, lengths, weights, abund = age_structured_population(
            target,
            float(config.linf[sp]),
            float(config.k[sp]),
            float(config.t0[sp]),
            float(config.condition_factor[sp]),
            float(config.allometric_power[sp]),
            float(config.additional_mortality_rate[sp]),
            float(config.lifespan_dt[sp]) / config.n_dt_per_year,
            config.n_dt_per_year,
        )
        n_schools_sp = int(config.n_schools[sp])
        for c in range(len(ages_dt)):
            if abund[c] <= 0.0 or weights[c] <= 0.0:
                continue
            n_new = n_schools_sp if (abund[c] >= n_schools_sp and n_schools_sp > 0) else 1
            abund_per = abund[c] / n_new
            idx = rng.integers(0, len(ys), size=n_new)
            new = SchoolState.create(n_schools=n_new, species_id=np.full(n_new, sp, dtype=np.int32))
            new = new.replace(
                abundance=np.full(n_new, abund_per, dtype=np.float64),
                biomass=np.full(n_new, abund_per * weights[c], dtype=np.float64),
                length=np.full(n_new, lengths[c], dtype=np.float64),
                length_start=np.full(n_new, lengths[c], dtype=np.float64),
                weight=np.full(n_new, weights[c], dtype=np.float64),
                age_dt=np.full(n_new, ages_dt[c], dtype=np.int32),
                cell_x=xs[idx],
                cell_y=ys[idx],
                is_egg=np.zeros(n_new, dtype=np.bool_),
            )
            parts.append(new)
    if not parts:
        return empty
    result = parts[0]
    for p in parts[1:]:
        result = result.append(p)
    return result
