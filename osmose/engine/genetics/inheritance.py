# osmose/engine/genetics/inheritance.py
"""Gametic inheritance: parent selection, meiotic segregation, offspring genotype creation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.genetics.genotype import GeneticState


def select_parents(
    gonad_weight: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Select two parents with probability proportional to gonad weight (fecundity proxy)."""
    weights = gonad_weight / gonad_weight.sum()
    parent_a, parent_b = rng.choice(len(gonad_weight), size=2, replace=True, p=weights)
    return int(parent_a), int(parent_b)


def form_gamete(
    parent_alleles: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Meiotic segregation: pick one allele per locus (free recombination).

    Args:
        parent_alleles: Shape (n_loci, 2) — diploid alleles for one parent, one trait.
        rng: Random generator.

    Returns:
        Gamete array of shape (n_loci,).
    """
    n_loci = parent_alleles.shape[0]
    picks = rng.integers(0, 2, size=n_loci)
    return parent_alleles[np.arange(n_loci), picks]


def create_offspring_genotypes(
    parent_gs: GeneticState,
    gonad_weight: NDArray[np.float64],
    species_id: NDArray[np.int32],
    offspring_species: int,
    n_offspring: int,
    rng: np.random.Generator,
    seeding: bool = False,
) -> GeneticState:
    """Create genotypes for n_offspring new schools of one species."""
    sp_mask = (species_id == offspring_species) & (gonad_weight > 0)
    sp_indices = np.where(sp_mask)[0]

    alleles: dict[str, NDArray[np.float64]] = {}
    env_noise: dict[str, NDArray[np.float64]] = {}

    for name, trait in parent_gs.registry.traits.items():
        max_loci = parent_gs.alleles[name].shape[1]
        n_loc = int(trait.n_loci[offspring_species])
        off_alleles = np.zeros((n_offspring, max_loci, 2), dtype=np.float64)

        if seeding:
            # During seeding: draw alleles from the full population pool at random
            pool_alleles = parent_gs.alleles[name]  # (N, max_loci, 2)
            for j in range(n_offspring):
                for a in range(2):
                    donor = rng.integers(0, len(pool_alleles))
                    allele_pick = rng.integers(0, 2)
                    off_alleles[j, :n_loc, a] = pool_alleles[donor, :n_loc, allele_pick]
        elif len(sp_indices) > 0:
            sp_gonad = gonad_weight[sp_indices]
            sp_alleles = parent_gs.alleles[name][sp_indices]

            for j in range(n_offspring):
                pa, pb = select_parents(sp_gonad, rng)
                off_alleles[j, :n_loc, 0] = form_gamete(sp_alleles[pa, :n_loc, :], rng)
                off_alleles[j, :n_loc, 1] = form_gamete(sp_alleles[pb, :n_loc, :], rng)

        ev = float(trait.env_var[offspring_species])
        if ev > 0:
            noise = rng.normal(0.0, np.sqrt(ev), size=n_offspring)
        else:
            noise = np.zeros(n_offspring)

        alleles[name] = off_alleles
        env_noise[name] = noise

    # Neutral loci
    neutral = None
    if parent_gs.neutral_alleles is not None:
        n_neutral = parent_gs.neutral_alleles.shape[1]
        n_neutral_val = int(parent_gs.neutral_alleles.max()) + 1
        off_neutral = np.zeros((n_offspring, n_neutral, 2), dtype=np.int32)

        if seeding:
            pool = parent_gs.neutral_alleles  # (N, n_neutral, 2)
            for j in range(n_offspring):
                for a in range(2):
                    donor = rng.integers(0, len(pool))
                    allele_pick = rng.integers(0, 2)
                    off_neutral[j, :, a] = pool[donor, :, allele_pick]
        elif len(sp_indices) > 0:
            sp_gonad = gonad_weight[sp_indices]
            sp_neutral = parent_gs.neutral_alleles[sp_indices]
            for j in range(n_offspring):
                pa, pb = select_parents(sp_gonad, rng)
                picks_a = rng.integers(0, 2, size=n_neutral)
                picks_b = rng.integers(0, 2, size=n_neutral)
                off_neutral[j, :, 0] = sp_neutral[pa, np.arange(n_neutral), picks_a]
                off_neutral[j, :, 1] = sp_neutral[pb, np.arange(n_neutral), picks_b]
        else:
            # No eligible parents → random draw
            off_neutral = rng.integers(
                0, n_neutral_val, size=(n_offspring, n_neutral, 2), dtype=np.int32
            )

        neutral = off_neutral

    return GeneticState(
        alleles=alleles, env_noise=env_noise, registry=parent_gs.registry, neutral_alleles=neutral
    )
