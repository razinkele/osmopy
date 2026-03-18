"""EngineConfig: typed parameter extraction from flat OSMOSE config dicts.

Converts the flat string key-value config (as read by OsmoseConfigReader)
into typed NumPy arrays indexed by species, ready for vectorized computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _get(cfg: dict[str, str], key: str) -> str:
    """Get a config value, raising KeyError with a clear message."""
    val = cfg.get(key)
    if val is None:
        raise KeyError(f"Required OSMOSE config key missing: {key!r}")
    return val


def _species_float(cfg: dict[str, str], pattern: str, n: int) -> NDArray[np.float64]:
    return np.array([float(_get(cfg, pattern.format(i=i))) for i in range(n)])


def _species_int(cfg: dict[str, str], pattern: str, n: int) -> NDArray[np.int32]:
    return np.array([int(_get(cfg, pattern.format(i=i))) for i in range(n)], dtype=np.int32)


def _species_str(cfg: dict[str, str], pattern: str, n: int) -> list[str]:
    return [_get(cfg, pattern.format(i=i)) for i in range(n)]


def _species_float_optional(
    cfg: dict[str, str], pattern: str, n: int, default: float
) -> NDArray[np.float64]:
    """Extract a per-species float array, using default if key is missing."""
    return np.array([float(cfg.get(pattern.format(i=i), str(default))) for i in range(n)])


def _species_int_optional(
    cfg: dict[str, str], pattern: str, n: int, default: int
) -> NDArray[np.int32]:
    """Extract a per-species int array, using default if key is missing."""
    return np.array(
        [int(cfg.get(pattern.format(i=i), str(default))) for i in range(n)], dtype=np.int32
    )


@dataclass
class EngineConfig:
    """Typed engine configuration extracted from a flat OSMOSE config dict."""

    n_species: int
    n_dt_per_year: int
    n_year: int
    n_steps: int
    n_schools: NDArray[np.int32]
    species_names: list[str]

    linf: NDArray[np.float64]
    k: NDArray[np.float64]
    t0: NDArray[np.float64]
    egg_size: NDArray[np.float64]
    condition_factor: NDArray[np.float64]
    allometric_power: NDArray[np.float64]
    vb_threshold_age: NDArray[np.float64]
    lifespan_dt: NDArray[np.int32]

    mortality_subdt: int
    ingestion_rate: NDArray[np.float64]
    critical_success_rate: NDArray[np.float64]

    # Growth
    delta_lmax_factor: NDArray[np.float64]  # max growth scaling factor (default 2.0)

    # Natural mortality
    additional_mortality_rate: NDArray[np.float64]  # annual additional mortality rate per species

    # Reproduction
    sex_ratio: NDArray[np.float64]  # fraction female per species
    relative_fecundity: NDArray[np.float64]  # eggs per gram of mature female
    maturity_size: NDArray[np.float64]  # length at maturity (cm)
    seeding_biomass: NDArray[np.float64]  # initial biomass for seeding (tonnes)
    larva_mortality_rate: NDArray[np.float64]  # additional mortality for eggs/larvae

    # Predation
    size_ratio_min: NDArray[np.float64]  # max pred/prey ratio per species (Java naming)
    size_ratio_max: NDArray[np.float64]  # min pred/prey ratio per species

    # Starvation
    starvation_rate_max: NDArray[np.float64]  # max starvation mortality rate

    # Fishing
    fishing_enabled: bool  # global fishing toggle
    fishing_rate: NDArray[np.float64]  # annual fishing mortality rate per species

    # Movement
    movement_method: list[str]
    random_walk_range: NDArray[np.int32]
    out_mortality_rate: NDArray[np.float64]

    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
        n_sp = int(_get(cfg, "simulation.nspecies"))
        n_dt = int(_get(cfg, "simulation.time.ndtperyear"))
        n_yr = int(_get(cfg, "simulation.time.nyear"))
        lifespan_years = _species_float(cfg, "species.lifespan.sp{i}", n_sp)

        return cls(
            n_species=n_sp,
            n_dt_per_year=n_dt,
            n_year=n_yr,
            n_steps=n_dt * n_yr,
            n_schools=_species_int_optional(
                cfg, "simulation.nschool.sp{i}", n_sp,
                default=int(cfg.get("simulation.nschool", "20")),
            ),
            species_names=_species_str(cfg, "species.name.sp{i}", n_sp),
            linf=_species_float(cfg, "species.linf.sp{i}", n_sp),
            k=_species_float(cfg, "species.k.sp{i}", n_sp),
            t0=_species_float(cfg, "species.t0.sp{i}", n_sp),
            egg_size=_species_float(cfg, "species.egg.size.sp{i}", n_sp),
            condition_factor=_species_float(
                cfg, "species.length2weight.condition.factor.sp{i}", n_sp
            ),
            allometric_power=_species_float(
                cfg, "species.length2weight.allometric.power.sp{i}", n_sp
            ),
            vb_threshold_age=_species_float(
                cfg, "species.vonbertalanffy.threshold.age.sp{i}", n_sp
            ),
            lifespan_dt=(lifespan_years * n_dt).astype(np.int32),
            mortality_subdt=max(1, int(cfg.get("mortality.subdt", "10"))),
            ingestion_rate=_species_float(cfg, "predation.ingestion.rate.max.sp{i}", n_sp),
            critical_success_rate=_species_float(cfg, "predation.efficiency.critical.sp{i}", n_sp),
            delta_lmax_factor=_species_float_optional(
                cfg, "species.delta.lmax.factor.sp{i}", n_sp, default=2.0
            ),
            additional_mortality_rate=_species_float_optional(
                cfg, "mortality.additional.rate.sp{i}", n_sp, default=0.0
            ),
            sex_ratio=_species_float_optional(cfg, "species.sexratio.sp{i}", n_sp, default=0.5),
            relative_fecundity=_species_float_optional(
                cfg, "species.relativefecundity.sp{i}", n_sp, default=500.0
            ),
            maturity_size=_species_float_optional(
                cfg, "species.maturity.size.sp{i}", n_sp, default=0.0
            ),
            seeding_biomass=_species_float_optional(
                cfg, "population.seeding.biomass.sp{i}", n_sp, default=0.0
            ),
            larva_mortality_rate=_species_float_optional(
                cfg, "mortality.additional.larva.rate.sp{i}", n_sp, default=0.0
            ),
            size_ratio_min=_species_float_optional(
                cfg, "predation.predPrey.sizeRatio.min.sp{i}", n_sp, default=1.0
            ),
            size_ratio_max=_species_float_optional(
                cfg, "predation.predPrey.sizeRatio.max.sp{i}", n_sp, default=3.5
            ),
            starvation_rate_max=_species_float_optional(
                cfg, "mortality.starvation.rate.max.sp{i}", n_sp, default=0.0
            ),
            fishing_enabled=cfg.get("simulation.fishing.mortality.enabled", "true").lower()
            == "true",
            fishing_rate=_species_float_optional(cfg, "fishing.rate.sp{i}", n_sp, default=0.0),
            movement_method=[
                cfg.get(f"movement.distribution.method.sp{i}", "random") for i in range(n_sp)
            ],
            random_walk_range=_species_int_optional(
                cfg, "movement.randomwalk.range.sp{i}", n_sp, default=1
            ),
            out_mortality_rate=_species_float_optional(
                cfg, "mortality.out.rate.sp{i}", n_sp, default=0.0
            ),
        )
