"""Ev-OSMOSE eco-evolutionary genetics module."""

from osmose.engine.genetics.expression import apply_trait_overrides, express_traits
from osmose.engine.genetics.genotype import (
    GeneticState,
    compact_genetic_state,
    create_initial_genotypes,
)
from osmose.engine.genetics.inheritance import (
    create_offspring_genotypes,
    form_gamete,
    select_parents,
)
from osmose.engine.genetics.trait import Trait, TraitRegistry

__all__ = [
    "GeneticState",
    "Trait",
    "TraitRegistry",
    "apply_trait_overrides",
    "compact_genetic_state",
    "create_initial_genotypes",
    "create_offspring_genotypes",
    "express_traits",
    "form_gamete",
    "select_parents",
]
