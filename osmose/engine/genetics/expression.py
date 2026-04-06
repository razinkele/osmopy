# osmose/engine/genetics/expression.py
"""Genotype-to-phenotype expression and config override application."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.genetics.genotype import GeneticState
from osmose.engine.genetics.trait import TraitRegistry


def express_traits(
    gs: GeneticState,
    species_id: NDArray[np.int32],
) -> dict[str, NDArray[np.float64]]:
    """Compute phenotypic values for all traits, all schools.

    phenotype[i] = species_mean[sp_i] + sum(alleles[i, :, :]) + env_noise[i]
    """
    phenotypes: dict[str, NDArray[np.float64]] = {}
    for name, trait in gs.registry.traits.items():
        sp = species_id
        g = trait.species_mean[sp] + gs.alleles[name].sum(axis=(1, 2))
        phenotypes[name] = g + gs.env_noise[name]
    return phenotypes


def apply_trait_overrides(
    overrides: dict[str, NDArray[np.float64]],
    phenotypes: dict[str, NDArray[np.float64]],
    registry: TraitRegistry,
) -> None:
    """Store per-school phenotypic values keyed by target parameter name."""
    for trait_name, values in phenotypes.items():
        target = registry.traits[trait_name].target_param
        overrides[target] = values
