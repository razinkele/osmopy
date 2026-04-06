# tests/test_genetics_seeding.py
"""Tests for seeding phase in genetics inheritance."""

from __future__ import annotations

import numpy as np

from osmose.engine.genetics.genotype import create_initial_genotypes
from osmose.engine.genetics.inheritance import create_offspring_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry


def _make_registry() -> TraitRegistry:
    trait = Trait(
        name="imax",
        target_param="bioen_i_max",
        n_loci=np.array([2], dtype=np.int32),
        n_alleles=np.array([3], dtype=np.int32),
        species_mean=np.array([4.0]),
        species_var=np.array([0.1]),
        env_var=np.array([0.0]),
        allele_pool=[[np.array([3.0, 4.0, 5.0]), np.array([3.0, 4.0, 5.0])]],
    )
    reg = TraitRegistry()
    reg.register(trait)
    return reg


class TestSeedingPhase:
    def test_seeding_offspring_alleles_from_pool(self):
        """During seeding, offspring alleles must come from the full parental pool."""
        reg = _make_registry()
        rng = np.random.default_rng(42)
        species_id = np.zeros(20, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=3, n_neutral_val=10)

        gonad = np.ones(20)
        offspring_gs = create_offspring_genotypes(
            gs,
            gonad,
            species_id,
            offspring_species=0,
            n_offspring=8,
            rng=np.random.default_rng(10),
            seeding=True,
        )
        assert offspring_gs.neutral_alleles is not None
        assert offspring_gs.neutral_alleles.shape == (8, 3, 2)
        # All neutral allele values should come from the parental pool
        parent_vals = set(gs.neutral_alleles.flatten().tolist())
        offspring_vals = set(offspring_gs.neutral_alleles.flatten().tolist())
        assert offspring_vals.issubset(parent_vals)

    def test_seeding_does_not_require_eligible_parents(self):
        """Seeding should not fail when gonad_weight is zero for all parents."""
        reg = _make_registry()
        rng = np.random.default_rng(7)
        species_id = np.zeros(5, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=2, n_neutral_val=5)

        # All gonad weights zero → normal inheritance would produce no offspring
        gonad = np.zeros(5)
        offspring_gs = create_offspring_genotypes(
            gs,
            gonad,
            species_id,
            offspring_species=0,
            n_offspring=4,
            rng=np.random.default_rng(8),
            seeding=True,
        )
        # Should produce offspring genotypes (drawn from pool) rather than crash
        assert offspring_gs.alleles["imax"].shape[0] == 4
