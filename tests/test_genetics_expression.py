# tests/test_genetics_expression.py
"""Tests for genotype-to-phenotype expression and config override."""

import numpy as np
import pytest

from osmose.engine.genetics.expression import apply_trait_overrides, express_traits
from osmose.engine.genetics.genotype import GeneticState
from osmose.engine.genetics.trait import Trait, TraitRegistry


class TestExpressTraits:
    def test_phenotype_is_mean_plus_alleles_plus_noise(self):
        """Phenotype = species_mean + sum(alleles) + env_noise."""
        registry = TraitRegistry()
        trait = Trait(
            name="imax",
            species_mean=np.array([3.5]),
            species_var=np.array([0.1]),
            env_var=np.array([0.0]),
            n_loci=np.array([2], dtype=np.int32),
            n_alleles=np.array([10], dtype=np.int32),
            target_param="ingestion_rate",
        )
        registry.register(trait)

        alleles = np.array([[[0.1, 0.2], [0.3, -0.1]]])  # shape (1, 2, 2)
        noise = np.array([0.05])
        gs = GeneticState(
            alleles={"imax": alleles},
            env_noise={"imax": noise},
            registry=registry,
        )
        species_id = np.array([0], dtype=np.int32)

        phenotypes = express_traits(gs, species_id)
        # 3.5 + (0.1 + 0.2 + 0.3 + (-0.1)) + 0.05 = 4.05
        assert phenotypes["imax"][0] == pytest.approx(4.05)

    def test_multi_school_expression(self):
        registry = TraitRegistry()
        trait = Trait(
            name="imax",
            species_mean=np.array([0.0, 10.0]),
            species_var=np.array([1.0, 1.0]),
            env_var=np.array([0.0, 0.0]),
            n_loci=np.array([1, 1], dtype=np.int32),
            n_alleles=np.array([10, 10], dtype=np.int32),
            target_param="ingestion_rate",
        )
        registry.register(trait)

        alleles = np.array([[[0.5, 0.5]], [[-1.0, 1.0]]])  # shape (2, 1, 2)
        noise = np.zeros(2)
        gs = GeneticState(alleles={"imax": alleles}, env_noise={"imax": noise}, registry=registry)
        species_id = np.array([0, 1], dtype=np.int32)

        phenotypes = express_traits(gs, species_id)
        assert phenotypes["imax"][0] == pytest.approx(1.0)   # 0 + 0.5 + 0.5
        assert phenotypes["imax"][1] == pytest.approx(10.0)  # 10 + (-1) + 1


class TestApplyTraitOverrides:
    def test_override_stored_in_dict(self):
        """apply_trait_overrides stores per-school values in the overrides dict."""
        registry = TraitRegistry()
        trait = Trait(
            name="imax",
            species_mean=np.array([3.5]),
            species_var=np.array([0.1]),
            env_var=np.array([0.0]),
            n_loci=np.array([2], dtype=np.int32),
            n_alleles=np.array([10], dtype=np.int32),
            target_param="ingestion_rate",
        )
        registry.register(trait)

        phenotypes = {"imax": np.array([4.0, 3.8])}
        overrides: dict[str, np.ndarray] = {}
        apply_trait_overrides(overrides, phenotypes, registry)
        assert "ingestion_rate" in overrides
        assert overrides["ingestion_rate"][0] == pytest.approx(4.0)
        assert overrides["ingestion_rate"][1] == pytest.approx(3.8)
