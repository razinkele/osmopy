"""Tests for Trait and TraitRegistry."""

import numpy as np
import pytest

from osmose.engine.genetics.trait import Trait, TraitRegistry
from osmose.engine.genetics.genotype import (
    compact_genetic_state,
    create_initial_genotypes,
)


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


class TestGeneticState:
    def _make_registry(self) -> TraitRegistry:
        cfg = {
            "evolution.trait.imax.mean.sp0": "3.5",
            "evolution.trait.imax.var.sp0": "0.1",
            "evolution.trait.imax.envvar.sp0": "0.05",
            "evolution.trait.imax.nlocus.sp0": "5",
            "evolution.trait.imax.nval.sp0": "20",
            "evolution.trait.imax.target": "ingestion_rate",
        }
        return TraitRegistry.from_config(cfg, n_species=1)

    def test_create_initial_genotypes(self):
        registry = self._make_registry()
        species_id = np.array([0, 0, 0], dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)
        assert "imax" in gs.alleles
        assert gs.alleles["imax"].shape == (3, 5, 2)
        assert "imax" in gs.env_noise
        assert gs.env_noise["imax"].shape == (3,)

    def test_compact_removes_dead(self):
        registry = self._make_registry()
        species_id = np.array([0, 0, 0], dtype=np.int32)
        rng = np.random.default_rng(42)
        gs = create_initial_genotypes(registry, species_id, rng)
        alive = np.array([True, False, True])
        compacted = compact_genetic_state(gs, alive)
        assert compacted.alleles["imax"].shape[0] == 2
        assert compacted.env_noise["imax"].shape[0] == 2

    def test_append_concatenates(self):
        registry = self._make_registry()
        rng = np.random.default_rng(42)
        gs1 = create_initial_genotypes(registry, np.array([0, 0], dtype=np.int32), rng)
        gs2 = create_initial_genotypes(registry, np.array([0], dtype=np.int32), rng)
        merged = gs1.append(gs2)
        assert merged.alleles["imax"].shape[0] == 3
        assert merged.env_noise["imax"].shape[0] == 3


def test_apply_trait_overrides_routes_imax_to_bioen_i_max_target() -> None:
    """Sanity: `apply_trait_overrides` writes under the declared target."""
    import numpy as np
    from osmose.engine.genetics import (
        TraitRegistry,
        apply_trait_overrides,
        create_initial_genotypes,
        express_traits,
    )

    cfg = {
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.mean.sp0": "3.0",
        "evolution.trait.imax.var.sp0": "0.0",
        "evolution.trait.imax.envvar.sp0": "0.0",
        "evolution.trait.imax.nlocus.sp0": "0",
        "evolution.trait.imax.nval.sp0": "0",
    }
    registry = TraitRegistry.from_config(cfg, n_species=1)
    rng = np.random.default_rng(42)
    species_id = np.zeros(4, dtype=np.int32)
    gs = create_initial_genotypes(registry, species_id, rng, n_neutral=0, n_neutral_val=0)
    phenotypes = express_traits(gs, species_id)

    overrides: dict[str, np.ndarray] = {}
    apply_trait_overrides(overrides, phenotypes, registry)

    # (a) Target name is the declared one, NOT the trait name
    assert set(overrides.keys()) == {"bioen_i_max"}
    # (b) Shape matches n_schools — bioen step indexes per-school via this
    assert overrides["bioen_i_max"].shape == species_id.shape
    # Zero variance + zero env noise → all phenotype values equal mean
    assert np.allclose(overrides["bioen_i_max"], 3.0)


def test_bioen_i_max_inherits_genetic_and_env_variance() -> None:
    """Nonzero var + envvar must produce non-degenerate per-school phenotypes
    with empirical SD ≈ sqrt(var + envvar). Catches a silent-zero regression
    where the trait reads as registered but doesn't actually draw values."""
    import numpy as np
    from osmose.engine.genetics import (
        TraitRegistry,
        apply_trait_overrides,
        create_initial_genotypes,
        express_traits,
    )

    cfg = {
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.mean.sp0": "3.0",
        "evolution.trait.imax.var.sp0": "0.018",
        "evolution.trait.imax.envvar.sp0": "0.054",
        # Use a large pool (50 loci × 200 alleles) to suppress finite-sample
        # allele-pool bias so the empirical mean stays within ±0.02 of 3.0.
        "evolution.trait.imax.nlocus.sp0": "50",
        "evolution.trait.imax.nval.sp0": "200",
    }
    registry = TraitRegistry.from_config(cfg, n_species=1)
    rng = np.random.default_rng(0)
    species_id = np.zeros(2_000, dtype=np.int32)
    gs = create_initial_genotypes(registry, species_id, rng, n_neutral=0, n_neutral_val=0)
    phenotypes = express_traits(gs, species_id)

    overrides: dict[str, np.ndarray] = {}
    apply_trait_overrides(overrides, phenotypes, registry)
    expected_sd = np.sqrt(0.018 + 0.054)
    empirical = overrides["bioen_i_max"]
    # 2,000 draws is enough that empirical SD lands within ±15% of expected
    assert abs(float(empirical.std()) - expected_sd) / expected_sd < 0.15
    # Mean stays near the species mean (±0.1 accounts for finite-sample allele-pool bias)
    assert abs(float(empirical.mean()) - 3.0) < 0.1


def test_bioen_i_max_pins_zero_variance_species_to_species_mean() -> None:
    """Multi-species: cod (sp0) evolves, herring (sp1) does not. Herring's
    phenotype must remain pinned to its species_mean exactly."""
    import numpy as np
    from osmose.engine.genetics import (
        TraitRegistry,
        apply_trait_overrides,
        create_initial_genotypes,
        express_traits,
    )

    cfg = {
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.imax.mean.sp0": "3.0",
        "evolution.trait.imax.var.sp0": "0.018",
        "evolution.trait.imax.envvar.sp0": "0.054",
        "evolution.trait.imax.nlocus.sp0": "10",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.mean.sp1": "5.0",
        "evolution.trait.imax.var.sp1": "0.0",
        "evolution.trait.imax.envvar.sp1": "0.0",
        "evolution.trait.imax.nlocus.sp1": "0",
        "evolution.trait.imax.nval.sp1": "0",
    }
    registry = TraitRegistry.from_config(cfg, n_species=2)
    rng = np.random.default_rng(0)
    species_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    gs = create_initial_genotypes(registry, species_id, rng, n_neutral=0, n_neutral_val=0)
    phenotypes = express_traits(gs, species_id)

    overrides: dict[str, np.ndarray] = {}
    apply_trait_overrides(overrides, phenotypes, registry)
    # sp1 phenotypes pinned to species_mean.sp1 = 5.0 exactly
    assert np.allclose(overrides["bioen_i_max"][3:], 5.0)
    # sp0 phenotypes vary
    assert overrides["bioen_i_max"][:3].std() > 0.0
