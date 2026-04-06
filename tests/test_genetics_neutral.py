# tests/test_genetics_neutral.py
"""Tests for neutral loci: creation, inheritance, compaction, and append."""

from __future__ import annotations

import numpy as np

from osmose.engine.genetics.genotype import (
    compact_genetic_state,
    create_initial_genotypes,
)
from osmose.engine.genetics.inheritance import create_offspring_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry


def _make_registry() -> TraitRegistry:
    """Minimal one-trait registry for testing."""
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


class TestNeutralLociCreation:
    def test_neutral_alleles_none_when_zero(self):
        """create_initial_genotypes with n_neutral=0 must leave neutral_alleles=None."""
        reg = _make_registry()
        rng = np.random.default_rng(0)
        species_id = np.zeros(10, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=0)
        assert gs.neutral_alleles is None

    def test_neutral_alleles_shape(self):
        """neutral_alleles shape must be (n_schools, n_neutral, 2)."""
        reg = _make_registry()
        rng = np.random.default_rng(1)
        species_id = np.zeros(20, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=5, n_neutral_val=10)
        assert gs.neutral_alleles is not None
        assert gs.neutral_alleles.shape == (20, 5, 2)

    def test_neutral_allele_values_in_range(self):
        """All neutral allele values must be in [0, n_neutral_val)."""
        reg = _make_registry()
        rng = np.random.default_rng(2)
        species_id = np.zeros(50, dtype=np.int32)
        n_val = 7
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=3, n_neutral_val=n_val)
        assert gs.neutral_alleles.min() >= 0
        assert gs.neutral_alleles.max() < n_val


class TestNeutralLociInheritance:
    def test_offspring_inherit_neutral_alleles(self):
        """Offspring neutral_alleles must be inherited from the parental pool."""
        reg = _make_registry()
        rng = np.random.default_rng(3)
        species_id = np.zeros(10, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=4, n_neutral_val=6)

        gonad = np.ones(10)
        offspring_gs = create_offspring_genotypes(
            gs,
            gonad,
            species_id,
            offspring_species=0,
            n_offspring=5,
            rng=np.random.default_rng(4),
        )
        assert offspring_gs.neutral_alleles is not None
        assert offspring_gs.neutral_alleles.shape == (5, 4, 2)
        # All values must come from the same integer range
        parent_vals = set(gs.neutral_alleles.flatten().tolist())
        offspring_vals = set(offspring_gs.neutral_alleles.flatten().tolist())
        assert offspring_vals.issubset(parent_vals)

    def test_compact_preserves_neutral(self):
        """compact_genetic_state must correctly filter neutral_alleles with the alive mask."""
        reg = _make_registry()
        rng = np.random.default_rng(5)
        species_id = np.zeros(6, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=3, n_neutral_val=8)

        mask = np.array([True, False, True, False, True, True])
        cgs = compact_genetic_state(gs, mask)
        assert cgs.neutral_alleles is not None
        assert cgs.neutral_alleles.shape[0] == 4
        np.testing.assert_array_equal(cgs.neutral_alleles, gs.neutral_alleles[mask])
