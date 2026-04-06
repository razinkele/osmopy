# osmose/engine/genetics/genotype.py
"""GeneticState: parallel allele/noise arrays for all schools."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from osmose.engine.genetics.trait import TraitRegistry


@dataclass
class GeneticState:
    """Per-school genetic data, parallel to SchoolState."""

    alleles: dict[str, NDArray[np.float64]]
    env_noise: dict[str, NDArray[np.float64]]
    registry: TraitRegistry
    neutral_alleles: NDArray[np.int32] | None = field(default=None)

    def append(self, other: GeneticState) -> GeneticState:
        new_alleles = {}
        new_noise = {}
        for name in self.alleles:
            new_alleles[name] = np.concatenate([self.alleles[name], other.alleles[name]], axis=0)
            new_noise[name] = np.concatenate([self.env_noise[name], other.env_noise[name]], axis=0)
        neutral = None
        if self.neutral_alleles is not None and other.neutral_alleles is not None:
            neutral = np.concatenate([self.neutral_alleles, other.neutral_alleles], axis=0)
        return GeneticState(
            alleles=new_alleles,
            env_noise=new_noise,
            registry=self.registry,
            neutral_alleles=neutral,
        )


def compact_genetic_state(gs: GeneticState, alive_mask: NDArray[np.bool_]) -> GeneticState:
    new_alleles = {name: arr[alive_mask] for name, arr in gs.alleles.items()}
    new_noise = {name: arr[alive_mask] for name, arr in gs.env_noise.items()}
    neutral = gs.neutral_alleles[alive_mask] if gs.neutral_alleles is not None else None
    return GeneticState(
        alleles=new_alleles, env_noise=new_noise, registry=gs.registry, neutral_alleles=neutral
    )


def create_initial_genotypes(
    registry: TraitRegistry,
    species_id: NDArray[np.int32],
    rng: np.random.Generator,
    n_neutral: int = 0,
    n_neutral_val: int = 50,
) -> GeneticState:
    n_schools = len(species_id)
    alleles: dict[str, NDArray[np.float64]] = {}
    env_noise: dict[str, NDArray[np.float64]] = {}

    for name, trait in registry.traits.items():
        max_loci = int(trait.n_loci.max())
        arr = np.zeros((n_schools, max_loci, 2), dtype=np.float64)
        noise = np.zeros(n_schools, dtype=np.float64)

        for i in range(n_schools):
            sp = species_id[i]
            n_loc = int(trait.n_loci[sp])
            pool = trait.allele_pool[sp]
            for loc in range(n_loc):
                arr[i, loc, :] = rng.choice(pool[loc], size=2, replace=True)
            if trait.env_var[sp] > 0:
                noise[i] = rng.normal(0.0, np.sqrt(trait.env_var[sp]))

        alleles[name] = arr
        env_noise[name] = noise

    neutral = None
    if n_neutral > 0:
        neutral = rng.integers(0, n_neutral_val, size=(n_schools, n_neutral, 2), dtype=np.int32)

    return GeneticState(
        alleles=alleles, env_noise=env_noise, registry=registry, neutral_alleles=neutral
    )
