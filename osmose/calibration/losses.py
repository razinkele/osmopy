"""Composable banded loss objectives for OSMOSE calibration."""

from __future__ import annotations

import math
from collections.abc import Callable

from osmose.calibration.targets import BiomassTarget


def banded_log_ratio_loss(sim_biomass: float, lower: float, upper: float) -> float:
    """Per-species loss: 0 inside [lower, upper], squared log-distance outside."""
    if sim_biomass <= 0:
        return 100.0
    if sim_biomass < lower:
        return math.log10(lower / sim_biomass) ** 2
    if sim_biomass > upper:
        return math.log10(sim_biomass / upper) ** 2
    return 0.0


def stability_penalty(
    cv: float,
    trend: float,
    cv_threshold: float = 0.2,
    trend_threshold: float = 0.05,
) -> float:
    """Penalty for oscillations (CV) and non-equilibrium (trend)."""
    penalty = 0.0
    if cv > cv_threshold:
        penalty += (cv - cv_threshold) ** 2
    if trend > trend_threshold:
        penalty += (trend - trend_threshold) ** 2
    return penalty


def worst_species_penalty(species_errors: list[float]) -> float:
    """Max of weighted per-species errors."""
    return max(species_errors)


def make_banded_objective(
    targets: list[BiomassTarget],
    species_names: list[str],
    w_stability: float = 5.0,
    w_worst: float = 0.5,
) -> Callable[[dict[str, float]], float]:
    """Factory: returns callable(species_stats) -> scalar objective.

    species_stats keys: ``{species}_mean``, ``{species}_cv``, ``{species}_trend``.
    Missing species keys receive a penalty of 100.0, weighted by species weight.
    (Note: the Baltic script applies the missing-species penalty unweighted.
    Weighting it here is intentional — see spec for rationale.)
    """
    target_dict = {t.species: t for t in targets}

    def objective(species_stats: dict[str, float]) -> float:
        total_error = 0.0
        weighted_errors: list[float] = []

        for sp in species_names:
            mean_key = f"{sp}_mean"
            cv_key = f"{sp}_cv"
            trend_key = f"{sp}_trend"

            if mean_key not in species_stats:
                sp_error = 100.0
            else:
                sp_error = banded_log_ratio_loss(
                    species_stats[mean_key], target_dict[sp].lower, target_dict[sp].upper
                )

            w = target_dict[sp].weight
            weighted_error = w * sp_error
            total_error += weighted_error
            weighted_errors.append(weighted_error)

            cv = species_stats.get(cv_key, 0.0)
            trend = species_stats.get(trend_key, 0.0)
            total_error += w_stability * w * stability_penalty(cv, trend)

        total_error += w_worst * worst_species_penalty(weighted_errors)
        return total_error

    return objective
