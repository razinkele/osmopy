# tests/test_genetics_statistics.py
"""Tests for genetic summary statistics: trait expression, allelic diversity."""

from __future__ import annotations

import numpy as np

from osmose.engine.genetics.expression import apply_trait_overrides, express_traits
from osmose.engine.genetics.genotype import create_initial_genotypes
from osmose.engine.genetics.trait import Trait, TraitRegistry


def _make_registry(n_loci: int = 4, env_var: float = 0.0) -> TraitRegistry:
    trait = Trait(
        name="imax",
        target_param="bioen_i_max",
        n_loci=np.array([n_loci], dtype=np.int32),
        n_alleles=np.array([5], dtype=np.int32),
        species_mean=np.array([3.5]),
        species_var=np.array([1.0]),
        env_var=np.array([env_var]),
        allele_pool=[[np.array([-0.5, -0.25, 0.0, 0.25, 0.5])] * n_loci],
    )
    reg = TraitRegistry()
    reg.register(trait)
    return reg


class TestTraitExpressionStatistics:
    def test_phenotype_mean_near_species_mean(self):
        """With symmetric allele pool and large sample, phenotype mean ≈ species_mean."""
        reg = _make_registry()
        rng = np.random.default_rng(0)
        species_id = np.zeros(500, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng)
        pheno = express_traits(gs, species_id)
        assert abs(pheno["imax"].mean() - 3.5) < 0.5  # within 0.5 of species mean

    def test_phenotype_variance_positive_when_var_gt_zero(self):
        """Trait phenotype must have nonzero variance when species_var > 0."""
        reg = _make_registry()
        rng = np.random.default_rng(1)
        species_id = np.zeros(200, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng)
        pheno = express_traits(gs, species_id)
        assert pheno["imax"].var() > 0.0

    def test_env_noise_increases_variance(self):
        """Adding env_var > 0 must increase phenotypic variance relative to env_var = 0."""
        reg_no_env = _make_registry(env_var=0.0)
        reg_with_env = _make_registry(env_var=2.0)
        rng_a = np.random.default_rng(2)
        rng_b = np.random.default_rng(2)
        species_id = np.zeros(300, dtype=np.int32)
        gs_no = create_initial_genotypes(reg_no_env, species_id, rng_a)
        gs_env = create_initial_genotypes(reg_with_env, species_id, rng_b)
        pheno_no = express_traits(gs_no, species_id)
        pheno_env = express_traits(gs_env, species_id)
        assert pheno_env["imax"].var() > pheno_no["imax"].var()

    def test_apply_trait_overrides_keys_match_target_param(self):
        """apply_trait_overrides must key results by target_param, not trait name."""
        reg = _make_registry()
        rng = np.random.default_rng(3)
        species_id = np.zeros(10, dtype=np.int32)
        gs = create_initial_genotypes(reg, species_id, rng)
        pheno = express_traits(gs, species_id)
        overrides: dict = {}
        apply_trait_overrides(overrides, pheno, reg)
        assert "bioen_i_max" in overrides
        assert "imax" not in overrides
        assert overrides["bioen_i_max"].shape == (10,)

    def test_neutral_loci_allelic_richness(self):
        """Neutral loci allelic richness should be close to n_neutral_val with large samples."""
        reg = _make_registry()
        rng = np.random.default_rng(4)
        species_id = np.zeros(200, dtype=np.int32)
        n_val = 10
        gs = create_initial_genotypes(reg, species_id, rng, n_neutral=5, n_neutral_val=n_val)
        assert gs.neutral_alleles is not None
        # Count unique allele values across all loci and haplotypes
        unique_vals = np.unique(gs.neutral_alleles)
        # With 200 schools × 5 loci × 2 haplotypes = 2000 draws, expect most of 10 values
        assert len(unique_vals) >= n_val * 0.8
