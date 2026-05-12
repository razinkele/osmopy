"""Composable banded loss objectives for OSMOSE calibration."""

from __future__ import annotations

import math

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
    if not species_errors:
        return 0.0
    return max(species_errors)


def make_banded_objective(
    targets: list[BiomassTarget],
    species_names: list[str],
    w_stability: float = 5.0,
    w_worst: float = 0.5,
):
    """Factory returning (objective_callable, residuals_accessor).

    objective_callable(species_stats) -> float
        Same scalar contract as the previous signature (unchanged values and
        defaults).

    residuals_accessor() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]] | None
        Returns (species_labels, residuals, sim_biomass) from the most-recent
        objective call. Cleared to None at START of each call (mid-call raise
        leaves None — spec §6.5.2 parity with Path A). Re-populated as LAST
        statement before return.
    """
    target_dict = {t.species: t for t in targets}
    state: dict[str, tuple] = {"residuals": None}

    def objective(species_stats: dict[str, float]) -> float:
        state["residuals"] = None  # clear at start

        residuals_local: list[tuple[str, float, float]] = []
        total_error = 0.0
        worst_error = 0.0
        for sp in species_names:
            mean_key = f"{sp}_mean"
            cv_key = f"{sp}_cv"
            trend_key = f"{sp}_trend"

            if mean_key not in species_stats or sp not in target_dict:
                total_error += 100.0
                worst_error = max(worst_error, 100.0)
                residuals_local.append((sp, 100.0, 0.0))
                continue

            sim_biomass = species_stats[mean_key]
            target = target_dict[sp]
            recorded_biomass = sim_biomass

            if sim_biomass <= 0:
                sp_error = 100.0
                recorded_biomass = 0.0
            elif sim_biomass < target.lower:
                sp_error = float(math.log10(target.lower / sim_biomass) ** 2)
            elif sim_biomass > target.upper:
                sp_error = float(math.log10(sim_biomass / target.upper) ** 2)
            else:
                sp_error = 0.0

            weighted_error = target.weight * sp_error
            total_error += weighted_error
            worst_error = max(worst_error, weighted_error)
            residuals_local.append((sp, weighted_error, float(recorded_biomass)))

            cv = species_stats.get(cv_key, 0.0)
            if cv > 0.2:
                total_error += w_stability * target.weight * (cv - 0.2) ** 2
            trend = species_stats.get(trend_key, 0.0)
            if trend > 0.05:
                total_error += w_stability * target.weight * (trend - 0.05) ** 2

        total_error += w_worst * worst_error

        state["residuals"] = (
            tuple(sp for sp, _, _ in residuals_local),
            tuple(r for _, r, _ in residuals_local),
            tuple(b for _, _, b in residuals_local),
        )
        return total_error

    def residuals_accessor():
        return state["residuals"]

    return objective, residuals_accessor
