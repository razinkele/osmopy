# tests/test_genetics_inheritance.py
"""Tests for gametic inheritance: parent selection, gamete formation, offspring creation."""

import numpy as np

from osmose.engine.genetics.genotype import GeneticState
from osmose.engine.genetics.inheritance import (
    create_offspring_genotypes,
    form_gamete,
    select_parents,
)
from osmose.engine.genetics.trait import Trait, TraitRegistry


class TestSelectParents:
    def test_fecundity_weighted(self):
        """Parents should be drawn proportional to gonad_weight."""
        gonad_weight = np.array([0.0, 0.0, 100.0])
        rng = np.random.default_rng(42)
        counts = np.zeros(3, dtype=int)
        for _ in range(200):
            a, b = select_parents(gonad_weight, rng)
            counts[a] += 1
            counts[b] += 1
        assert counts[0] == 0
        assert counts[1] == 0
        assert counts[2] == 400


class TestFormGamete:
    def test_picks_one_allele_per_locus(self):
        """Gamete should contain exactly one allele from each locus."""
        parent_alleles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rng = np.random.default_rng(42)
        gamete = form_gamete(parent_alleles, rng)
        assert gamete.shape == (3,)
        for i in range(3):
            assert gamete[i] in [parent_alleles[i, 0], parent_alleles[i, 1]]


class TestCreateOffspringGenotypes:
    def _make_registry(self) -> TraitRegistry:
        registry = TraitRegistry()
        registry.register(
            Trait(
                name="imax",
                species_mean=np.array([3.5]),
                species_var=np.array([0.1]),
                env_var=np.array([0.05]),
                n_loci=np.array([3], dtype=np.int32),
                n_alleles=np.array([20], dtype=np.int32),
                target_param="ingestion_rate",
            )
        )
        return registry

    def test_offspring_has_correct_shape(self):
        registry = self._make_registry()
        parent_alleles = np.zeros((2, 3, 2))
        parent_alleles[0] = [[1, 2], [3, 4], [5, 6]]
        parent_alleles[1] = [[7, 8], [9, 10], [11, 12]]
        parent_noise = np.array([0.1, 0.2])
        parent_gs = GeneticState(
            alleles={"imax": parent_alleles},
            env_noise={"imax": parent_noise},
            registry=registry,
        )

        gonad_weight = np.array([50.0, 50.0])
        rng = np.random.default_rng(42)

        offspring_gs = create_offspring_genotypes(
            parent_gs=parent_gs,
            gonad_weight=gonad_weight,
            species_id=np.array([0, 0], dtype=np.int32),
            offspring_species=0,
            n_offspring=5,
            rng=rng,
        )
        assert offspring_gs.alleles["imax"].shape == (5, 3, 2)
        assert offspring_gs.env_noise["imax"].shape == (5,)

    def test_offspring_alleles_come_from_parents(self):
        """Each offspring allele at each locus must be from one of the two parents' alleles."""
        registry = self._make_registry()
        parent_alleles = np.zeros((2, 3, 2))
        parent_alleles[0] = [[1, 2], [3, 4], [5, 6]]
        parent_alleles[1] = [[7, 8], [9, 10], [11, 12]]
        parent_gs = GeneticState(
            alleles={"imax": parent_alleles},
            env_noise={"imax": np.zeros(2)},
            registry=registry,
        )

        rng = np.random.default_rng(99)
        offspring_gs = create_offspring_genotypes(
            parent_gs=parent_gs,
            gonad_weight=np.array([50.0, 50.0]),
            species_id=np.array([0, 0], dtype=np.int32),
            offspring_species=0,
            n_offspring=10,
            rng=rng,
        )
        parent_vals = set(parent_alleles.flatten())
        for val in offspring_gs.alleles["imax"].flatten():
            assert val in parent_vals
