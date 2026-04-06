"""Ev-OSMOSE eco-evolutionary genetics module."""

from osmose.engine.genetics.genotype import GeneticState, compact_genetic_state, create_initial_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry

__all__ = [
    "GeneticState",
    "Trait",
    "TraitRegistry",
    "compact_genetic_state",
    "create_initial_genotypes",
]
