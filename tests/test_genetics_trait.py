"""Tests for Trait and TraitRegistry."""

import numpy as np
import pytest

from osmose.engine.genetics.trait import Trait, TraitRegistry


class TestTrait:
    def test_trait_frozen(self):
        t = Trait(
            name="imax",
            species_mean=np.array([3.5]),
            species_var=np.array([0.1]),
            env_var=np.array([0.05]),
            n_loci=np.array([10], dtype=np.int32),
            n_alleles=np.array([20], dtype=np.int32),
            target_param="ingestion_rate",
        )
        assert t.name == "imax"
        assert t.target_param == "ingestion_rate"
        with pytest.raises(AttributeError):
            t.name = "other"


class TestTraitRegistry:
    def test_from_config_single_trait(self):
        cfg = {
            "simulation.genetic.enabled": "true",
            "simulation.nspecies": "2",
            "evolution.trait.imax.mean.sp0": "3.5",
            "evolution.trait.imax.mean.sp1": "4.0",
            "evolution.trait.imax.var.sp0": "0.1",
            "evolution.trait.imax.var.sp1": "0.2",
            "evolution.trait.imax.envvar.sp0": "0.05",
            "evolution.trait.imax.envvar.sp1": "0.08",
            "evolution.trait.imax.nlocus.sp0": "10",
            "evolution.trait.imax.nlocus.sp1": "10",
            "evolution.trait.imax.nval.sp0": "20",
            "evolution.trait.imax.nval.sp1": "20",
            "evolution.trait.imax.target": "ingestion_rate",
        }
        registry = TraitRegistry.from_config(cfg, n_species=2)
        assert "imax" in registry.traits
        trait = registry.traits["imax"]
        assert trait.species_mean[0] == pytest.approx(3.5)
        assert trait.species_mean[1] == pytest.approx(4.0)
        assert trait.n_loci[0] == 10

    def test_allele_pool_variance(self):
        """Allele pool values should produce correct total genotypic variance."""
        cfg = {
            "evolution.trait.imax.mean.sp0": "0.0",
            "evolution.trait.imax.var.sp0": "1.0",
            "evolution.trait.imax.envvar.sp0": "0.0",
            "evolution.trait.imax.nlocus.sp0": "50",
            "evolution.trait.imax.nval.sp0": "1000",
            "evolution.trait.imax.target": "ingestion_rate",
        }
        registry = TraitRegistry.from_config(cfg, n_species=1)
        trait = registry.traits["imax"]
        pool = trait.allele_pool[0]  # species 0
        assert len(pool) == 50  # n_loci pools
        per_locus_var = np.var(pool[0])
        expected_per_locus = 1.0 / (2 * 50)
        assert per_locus_var == pytest.approx(expected_per_locus, rel=0.3)

    def test_empty_when_no_traits(self):
        registry = TraitRegistry.from_config({}, n_species=1)
        assert len(registry.traits) == 0
