"""Trait definitions and registry for evolving parameters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trait:
    """One evolving trait with per-species genetic parameters.

    Attributes:
        name: Trait identifier (e.g., "imax").
        species_mean: Initial genotypic mean per species, shape (n_species,).
        species_var: Additive genetic variance per species, shape (n_species,).
        env_var: Environmental noise variance per species, shape (n_species,).
        n_loci: Number of loci per species, shape (n_species,).
        n_alleles: Number of possible allelic values per locus per species, shape (n_species,).
        target_param: EngineConfig field this trait overrides.
        allele_pool: Pre-drawn allelic values — allele_pool[species][locus] is a 1-D array
            of n_alleles possible values. Populated by TraitRegistry.
    """

    name: str
    species_mean: NDArray[np.float64]
    species_var: NDArray[np.float64]
    env_var: NDArray[np.float64]
    n_loci: NDArray[np.int32]
    n_alleles: NDArray[np.int32]
    target_param: str
    allele_pool: list[list[NDArray[np.float64]]] = field(default_factory=list)


class TraitRegistry:
    """Collection of evolving traits, parsed from OSMOSE config."""

    def __init__(self) -> None:
        self.traits: dict[str, Trait] = {}

    def register(self, trait: Trait) -> None:
        self.traits[trait.name] = trait

    @classmethod
    def from_config(cls, cfg: dict[str, str], n_species: int) -> TraitRegistry:
        """Parse all evolution.trait.<name>.* keys from config."""
        registry = cls()

        trait_names: set[str] = set()
        for key in cfg:
            m = re.match(r"evolution\.trait\.(\w+)\.target", key)
            if m:
                trait_names.add(m.group(1))

        for name in sorted(trait_names):
            prefix = f"evolution.trait.{name}"
            target = cfg[f"{prefix}.target"]

            means = np.array(
                [float(cfg.get(f"{prefix}.mean.sp{i}", "0.0")) for i in range(n_species)]
            )
            variances = np.array(
                [float(cfg.get(f"{prefix}.var.sp{i}", "0.0")) for i in range(n_species)]
            )
            env_vars = np.array(
                [float(cfg.get(f"{prefix}.envvar.sp{i}", "0.0")) for i in range(n_species)]
            )
            n_loci = np.array(
                [int(cfg.get(f"{prefix}.nlocus.sp{i}", "10")) for i in range(n_species)],
                dtype=np.int32,
            )
            n_alleles = np.array(
                [int(cfg.get(f"{prefix}.nval.sp{i}", "20")) for i in range(n_species)],
                dtype=np.int32,
            )

            rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
            allele_pool: list[list[NDArray[np.float64]]] = []
            for sp in range(n_species):
                sp_pools: list[NDArray[np.float64]] = []
                if variances[sp] > 0 and n_loci[sp] > 0:
                    per_locus_sd = np.sqrt(variances[sp] / (2 * n_loci[sp]))
                    for _loc in range(n_loci[sp]):
                        values = rng.normal(0.0, per_locus_sd, size=int(n_alleles[sp]))
                        sp_pools.append(values)
                allele_pool.append(sp_pools)

            trait = Trait(
                name=name,
                species_mean=means,
                species_var=variances,
                env_var=env_vars,
                n_loci=n_loci,
                n_alleles=n_alleles,
                target_param=target,
                allele_pool=allele_pool,
            )
            registry.register(trait)

        return registry
