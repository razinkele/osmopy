"""EngineConfig: typed parameter extraction from flat OSMOSE config dicts.

Converts the flat string key-value config (as read by OsmoseConfigReader)
into typed NumPy arrays indexed by species, ready for vectorized computation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from osmose.engine.accessibility import AccessibilityMatrix
from osmose.engine.background import (
    BackgroundSpeciesInfo,
    _parse_floats,
    parse_background_species,
)


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


def _search_dirs() -> list[Path]:
    """Build a list of directories to search for data files."""
    import glob as _glob

    dirs = [Path("."), Path("data/examples")]
    dirs += [Path(d) for d in _glob.glob("data/*/")]
    return dirs


def _resolve_file(file_key: str) -> Path | None:
    """Resolve a relative file path against multiple search directories."""
    if not file_key:
        return None
    p = Path(file_key)
    if p.is_absolute() and p.exists():
        return p
    for base in _search_dirs():
        path = base / file_key
        if path.exists():
            return path
    return None


def _load_accessibility(
    cfg: dict[str, str], n_species: int
) -> NDArray[np.float64] | None:
    """Load predation accessibility matrix from CSV if available.

    Returns matrix with shape (n_total, n_total) where index [predator, prey] = coefficient.
    Used only when no stage structure is configured.
    """
    file_key = cfg.get("predation.accessibility.file", "")
    path = _resolve_file(file_key)
    if path is not None:
        df = pd.read_csv(path, sep=";", index_col=0)
        return df.values.astype(np.float64)
    return None


def _load_stage_accessibility(
    cfg: dict[str, str], all_species_names: list[str]
) -> AccessibilityMatrix | None:
    """Load stage-indexed accessibility matrix when age/size stages are used.

    Returns an AccessibilityMatrix instance, or None if no accessibility file exists.
    """
    file_key = cfg.get("predation.accessibility.file", "")
    path = _resolve_file(file_key)
    if path is None:
        return None
    return AccessibilityMatrix.from_csv(path, all_species_names)


def _parse_fisheries(
    cfg: dict[str, str], species_names: list[str], n_species: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    """Parse fisheries-based fishing config (OSMOSE v4).

    Returns
    -------
    fishing_rate: NDArray[np.float64]
        Annual fishing rate per species.
    fishing_selectivity_a50: NDArray[np.float64]
        Age at 50% selectivity (years) per species. NaN if not applicable.
    fishing_selectivity_type: NDArray[np.int32]
        0 = age-based (knife-edge), 1 = length-based, -1 = no fishing.
    """
    fishing_rate = np.zeros(n_species, dtype=np.float64)
    fishing_a50 = np.full(n_species, np.nan, dtype=np.float64)
    fishing_sel_type = np.full(n_species, -1, dtype=np.int32)

    n_fisheries = int(cfg.get("simulation.nfisheries", "0"))
    if n_fisheries == 0:
        return fishing_rate, fishing_a50, fishing_sel_type

    # Read catchability CSV to map species → fishery
    catch_file = cfg.get("fisheries.catchability.file", "")
    catch_path = _resolve_file(catch_file)
    if catch_path is None:
        return fishing_rate, fishing_a50, fishing_sel_type

    catch_df = pd.read_csv(catch_path, index_col=0)
    # Row labels = species names, column labels = fishery names
    # Build species_name → fishery_index mapping
    species_to_fishery: dict[str, int] = {}
    for row_idx in range(len(catch_df)):
        row_name = str(catch_df.index[row_idx]).strip()
        for col_idx in range(len(catch_df.columns)):
            val = float(catch_df.iloc[row_idx, col_idx])
            if val > 0:
                species_to_fishery[row_name.lower()] = col_idx
                break

    # Map species names to their fishing parameters
    for sp_idx in range(n_species):
        sp_name = species_names[sp_idx].strip().lower()
        fsh_idx = species_to_fishery.get(sp_name)
        if fsh_idx is None:
            continue

        # Base rate
        rate_key = f"fisheries.rate.base.fsh{fsh_idx}"
        rate_val = cfg.get(rate_key, "0.0")
        fishing_rate[sp_idx] = float(rate_val)

        # Selectivity type: 0 = knife-edge by age
        sel_type = int(cfg.get(f"fisheries.selectivity.type.fsh{fsh_idx}", "0"))
        if sel_type == 0:
            # Age-based knife-edge
            a50_key = f"fisheries.selectivity.a50.fsh{fsh_idx}"
            a50_val = cfg.get(a50_key, "0.0")
            fishing_a50[sp_idx] = float(a50_val)
            fishing_sel_type[sp_idx] = 0
        else:
            fishing_sel_type[sp_idx] = 1

    return fishing_rate, fishing_a50, fishing_sel_type


def _load_spawning_seasons(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int
) -> NDArray[np.float64] | None:
    """Load spawning season CSV files for each species.

    Returns array of shape (n_species, n_dt_per_year) with season weights.
    """
    seasons = np.ones((n_species, n_dt_per_year), dtype=np.float64) / n_dt_per_year
    found_any = False

    for i in range(n_species):
        file_key = cfg.get(f"reproduction.season.file.sp{i}", "")
        if not file_key:
            continue
        path = _resolve_file(file_key)
        if path is not None:
            df = pd.read_csv(path, sep=";")
            values = df.iloc[:, 1].values.astype(np.float64)
            if len(values) == n_dt_per_year:
                seasons[i] = values
                found_any = True

    return seasons if found_any else None


@dataclass
class EngineConfig:
    """Typed engine configuration extracted from a flat OSMOSE config dict."""

    n_species: int
    n_dt_per_year: int
    n_year: int
    n_steps: int
    n_schools: NDArray[np.int32]
    species_names: list[str]

    # Background species
    n_background: int
    background_file_indices: list[int]
    all_species_names: list[str]

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

    # Predation — 2D arrays of shape (n_total, max_stages)
    size_ratio_min: NDArray[np.float64]  # min pred/prey ratio per species per stage
    size_ratio_max: NDArray[np.float64]  # max pred/prey ratio per species per stage

    # Feeding stages
    feeding_stage_thresholds: list  # per-species list of threshold floats
    feeding_stage_metric: list[str]  # per-species metric name ("size"/"age"/"weight"/"tl")
    n_feeding_stages: NDArray[np.int32]  # number of feeding stages per species

    # Starvation
    starvation_rate_max: NDArray[np.float64]  # max starvation mortality rate

    # Fishing
    fishing_enabled: bool  # global fishing toggle
    fishing_rate: NDArray[np.float64]  # annual fishing mortality rate per species
    fishing_selectivity_l50: NDArray[np.float64]  # length at 50% selectivity

    # Fishing — fisheries-based selectivity
    fishing_selectivity_a50: NDArray[np.float64]  # age at 50% selectivity (years), NaN = unused
    fishing_selectivity_type: NDArray[np.int32]  # 0=age, 1=length, -1=none

    # Predation accessibility
    accessibility_matrix: NDArray[np.float64] | None  # (n_pred, n_prey) or None
    stage_accessibility: AccessibilityMatrix | None  # stage-indexed accessibility, or None

    # Reproduction
    spawning_season: NDArray[np.float64] | None  # (n_species, n_dt_per_year) or None

    # Movement
    movement_method: list[str]
    random_walk_range: NDArray[np.int32]
    out_mortality_rate: NDArray[np.float64]

    # Egg weight override: if species.egg.weight.sp{i} is set, use it instead of allometry
    egg_weight_override: NDArray[np.float64] | None  # shape (n_species,) or None

    # Raw config dict for subsystems that need unparsed access (e.g. ResourceState)
    raw_config: dict[str, str]

    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
        n_sp = int(_get(cfg, "simulation.nspecies"))
        n_dt = int(_get(cfg, "simulation.time.ndtperyear"))
        n_yr = int(_get(cfg, "simulation.time.nyear"))
        lifespan_years = _species_float(cfg, "species.lifespan.sp{i}", n_sp)

        # Fishing rate: try fisheries-based (v4), then per-species patterns
        fisheries_enabled = cfg.get("fisheries.enabled", "false").lower() == "true"
        n_fisheries = int(cfg.get("simulation.nfisheries", "0"))

        # Build species names early for fisheries parsing
        _focal_names = _species_str(cfg, "species.name.sp{i}", n_sp)

        if fisheries_enabled and n_fisheries > 0:
            fishing, focal_fishing_a50, focal_fishing_sel_type = _parse_fisheries(
                cfg, _focal_names, n_sp
            )
        else:
            # Legacy per-species rate
            fishing = _species_float_optional(
                cfg, "mortality.fishing.rate.sp{i}", n_sp, default=0.0
            )
            if fishing.sum() == 0:
                fishing = _species_float_optional(cfg, "fishing.rate.sp{i}", n_sp, default=0.0)
            focal_fishing_a50 = np.full(n_sp, np.nan, dtype=np.float64)
            focal_fishing_sel_type = np.full(n_sp, -1, dtype=np.int32)

        # Parse background species
        background_list: list[BackgroundSpeciesInfo] = parse_background_species(
            cfg, n_focal=n_sp, n_dt_per_year=n_dt
        )
        n_bkg = len(background_list)

        # Build focal-only arrays first
        focal_species_names = _focal_names
        focal_linf = _species_float(cfg, "species.linf.sp{i}", n_sp)
        focal_k = _species_float(cfg, "species.k.sp{i}", n_sp)
        focal_t0 = _species_float(cfg, "species.t0.sp{i}", n_sp)
        focal_egg_size = _species_float(cfg, "species.egg.size.sp{i}", n_sp)
        focal_condition_factor = _species_float(
            cfg, "species.length2weight.condition.factor.sp{i}", n_sp
        )
        focal_allometric_power = _species_float(
            cfg, "species.length2weight.allometric.power.sp{i}", n_sp
        )
        focal_vb_threshold_age = _species_float(
            cfg, "species.vonbertalanffy.threshold.age.sp{i}", n_sp
        )
        focal_lifespan_dt = (lifespan_years * n_dt).astype(np.int32)
        focal_ingestion_rate = _species_float(cfg, "predation.ingestion.rate.max.sp{i}", n_sp)
        focal_critical_success_rate = _species_float(
            cfg, "predation.efficiency.critical.sp{i}", n_sp
        )
        focal_delta_lmax_factor = _species_float_optional(
            cfg, "species.delta.lmax.factor.sp{i}", n_sp, default=2.0
        )
        focal_additional_mortality_rate = _species_float_optional(
            cfg, "mortality.additional.rate.sp{i}", n_sp, default=0.0
        )
        focal_sex_ratio = _species_float_optional(cfg, "species.sexratio.sp{i}", n_sp, default=0.5)
        focal_relative_fecundity = _species_float_optional(
            cfg, "species.relativefecundity.sp{i}", n_sp, default=500.0
        )
        focal_maturity_size = _species_float_optional(
            cfg, "species.maturity.size.sp{i}", n_sp, default=0.0
        )
        focal_seeding_biomass = _species_float_optional(
            cfg, "population.seeding.biomass.sp{i}", n_sp, default=0.0
        )
        focal_larva_mortality_rate = _species_float_optional(
            cfg, "mortality.additional.larva.rate.sp{i}", n_sp, default=0.0
        )
        # --- Feeding stages: thresholds, metrics, multi-value size ratios ---
        _VALID_METRICS = {"age", "size", "weight", "tl"}
        global_metric = cfg.get("predation.predprey.stage.structure", "size").strip().lower()
        if global_metric not in _VALID_METRICS:
            raise ValueError(
                f"Unrecognized feeding stage metric: {global_metric!r}. "
                f"Must be one of {sorted(_VALID_METRICS)}."
            )

        all_thresholds: list[list[float]] = []
        all_metrics: list[str] = []
        all_ratio_min: list[list[float]] = []
        all_ratio_max: list[list[float]] = []

        for i in range(n_sp):
            # Per-species metric override
            sp_metric = cfg.get(f"predation.predprey.stage.structure.sp{i}", "").strip().lower()
            if not sp_metric:
                sp_metric = global_metric
            elif sp_metric not in _VALID_METRICS:
                raise ValueError(
                    f"Unrecognized feeding stage metric for sp{i}: {sp_metric!r}. "
                    f"Must be one of {sorted(_VALID_METRICS)}."
                )
            all_metrics.append(sp_metric)

            # Thresholds
            thresh_raw = cfg.get(f"predation.predprey.stage.threshold.sp{i}", "")
            if not thresh_raw or thresh_raw.strip().lower() == "null":
                sp_thresholds: list[float] = []
            else:
                sp_thresholds = _parse_floats(thresh_raw)
            all_thresholds.append(sp_thresholds)
            n_stages = len(sp_thresholds) + 1

            # Size ratios (multi-value)
            rmin_raw = cfg.get(f"predation.predprey.sizeratio.min.sp{i}", "1.0")
            rmax_raw = cfg.get(f"predation.predprey.sizeratio.max.sp{i}", "3.5")
            rmin_list = _parse_floats(rmin_raw)
            rmax_list = _parse_floats(rmax_raw)

            # Validate length matches n_stages
            if len(rmin_list) != n_stages:
                raise ValueError(
                    f"Size ratio min count mismatch for sp{i}: "
                    f"got {len(rmin_list)}, expected {n_stages} stages"
                )
            if len(rmax_list) != n_stages:
                raise ValueError(
                    f"Size ratio max count mismatch for sp{i}: "
                    f"got {len(rmax_list)}, expected {n_stages} stages"
                )

            # Swap validation: if parsed min > parsed max (Java convention), swap + warn
            for s in range(n_stages):
                if rmin_list[s] > rmax_list[s]:
                    warnings.warn(
                        f"Swapping size ratios for sp{i} stage {s}: "
                        f"min={rmin_list[s]}, max={rmax_list[s]}",
                        stacklevel=2,
                    )
                    rmin_list[s], rmax_list[s] = rmax_list[s], rmin_list[s]

            all_ratio_min.append(rmin_list)
            all_ratio_max.append(rmax_list)

        # Background species: thresholds, metrics, ratios
        for b in background_list:
            b_idx = b.file_index
            b_metric = cfg.get(f"predation.predprey.stage.structure.sp{b_idx}", "").strip().lower()
            if not b_metric:
                b_metric = global_metric
            all_metrics.append(b_metric)

            thresh_raw = cfg.get(f"predation.predprey.stage.threshold.sp{b_idx}", "")
            if not thresh_raw or thresh_raw.strip().lower() == "null":
                b_thresholds: list[float] = []
            else:
                b_thresholds = _parse_floats(thresh_raw)
            all_thresholds.append(b_thresholds)
            n_stages = len(b_thresholds) + 1

            rmin_list = list(b.size_ratio_min)
            rmax_list = list(b.size_ratio_max)

            # If background species has 1 ratio but multiple stages, pad
            if len(rmin_list) == 1 and n_stages > 1:
                rmin_list = rmin_list * n_stages
            if len(rmax_list) == 1 and n_stages > 1:
                rmax_list = rmax_list * n_stages

            # Swap validation for background
            for s in range(min(len(rmin_list), len(rmax_list))):
                if rmin_list[s] > rmax_list[s]:
                    rmin_list[s], rmax_list[s] = rmax_list[s], rmin_list[s]

            all_ratio_min.append(rmin_list)
            all_ratio_max.append(rmax_list)

        # Build 2D arrays with padding
        n_total = n_sp + n_bkg
        max_stages = max((len(r) for r in all_ratio_min), default=1)
        n_feeding_stages = np.array([len(r) for r in all_ratio_min], dtype=np.int32)

        size_ratio_min_2d = np.zeros((n_total, max_stages), dtype=np.float64)
        size_ratio_max_2d = np.zeros((n_total, max_stages), dtype=np.float64)
        for sp_i in range(n_total):
            n_st = len(all_ratio_min[sp_i])
            for s in range(n_st):
                size_ratio_min_2d[sp_i, s] = all_ratio_min[sp_i][s]
                size_ratio_max_2d[sp_i, s] = all_ratio_max[sp_i][s]
            # Pad remaining columns with last valid value
            if n_st > 0 and n_st < max_stages:
                size_ratio_min_2d[sp_i, n_st:] = all_ratio_min[sp_i][-1]
                size_ratio_max_2d[sp_i, n_st:] = all_ratio_max[sp_i][-1]

        focal_starvation_rate_max = _species_float_optional(
            cfg, "mortality.starvation.rate.max.sp{i}", n_sp, default=0.0
        )
        focal_fishing_selectivity_l50 = _species_float_optional(
            cfg, "fishing.selectivity.l50.sp{i}", n_sp, default=0.0
        )
        focal_movement_method = [
            cfg.get(f"movement.distribution.method.sp{i}", "random") for i in range(n_sp)
        ]
        focal_random_walk_range = _species_int_optional(
            cfg, "movement.randomwalk.range.sp{i}", n_sp, default=1
        )
        focal_out_mortality_rate = _species_float_optional(
            cfg, "mortality.out.rate.sp{i}", n_sp, default=0.0
        )
        focal_n_schools = _species_int_optional(
            cfg,
            "simulation.nschool.sp{i}",
            n_sp,
            default=int(cfg.get("simulation.nschool", "20")),
        )

        # Extend all per-species arrays with background species values
        if n_bkg > 0:
            bkg_names = [b.name for b in background_list]
            bkg_ingestion = np.array([b.ingestion_rate for b in background_list])
            bkg_condition_factor = np.array([b.condition_factor for b in background_list])
            bkg_allometric_power = np.array([b.allometric_power for b in background_list])
            bkg_zeros_f = np.zeros(n_bkg, dtype=np.float64)
            bkg_zeros_i = np.zeros(n_bkg, dtype=np.int32)

            all_species_names = focal_species_names + bkg_names
            linf = np.concatenate([focal_linf, bkg_zeros_f])
            k = np.concatenate([focal_k, bkg_zeros_f])
            t0 = np.concatenate([focal_t0, bkg_zeros_f])
            egg_size = np.concatenate([focal_egg_size, bkg_zeros_f])
            condition_factor = np.concatenate([focal_condition_factor, bkg_condition_factor])
            allometric_power = np.concatenate([focal_allometric_power, bkg_allometric_power])
            vb_threshold_age = np.concatenate([focal_vb_threshold_age, bkg_zeros_f])
            lifespan_dt = np.concatenate([focal_lifespan_dt, bkg_zeros_i])
            ingestion_rate = np.concatenate([focal_ingestion_rate, bkg_ingestion])
            critical_success_rate = np.concatenate([focal_critical_success_rate, bkg_zeros_f])
            delta_lmax_factor = np.concatenate([focal_delta_lmax_factor, bkg_zeros_f])
            additional_mortality_rate = np.concatenate(
                [focal_additional_mortality_rate, bkg_zeros_f]
            )
            sex_ratio = np.concatenate([focal_sex_ratio, bkg_zeros_f])
            relative_fecundity = np.concatenate([focal_relative_fecundity, bkg_zeros_f])
            maturity_size = np.concatenate([focal_maturity_size, bkg_zeros_f])
            seeding_biomass = np.concatenate([focal_seeding_biomass, bkg_zeros_f])
            larva_mortality_rate = np.concatenate([focal_larva_mortality_rate, bkg_zeros_f])
            starvation_rate_max = np.concatenate([focal_starvation_rate_max, bkg_zeros_f])
            fishing_rate = np.concatenate([fishing, bkg_zeros_f])
            fishing_selectivity_l50 = np.concatenate([focal_fishing_selectivity_l50, bkg_zeros_f])
            fishing_selectivity_a50 = np.concatenate(
                [focal_fishing_a50, np.full(n_bkg, np.nan, dtype=np.float64)]
            )
            fishing_selectivity_type = np.concatenate(
                [focal_fishing_sel_type, np.full(n_bkg, -1, dtype=np.int32)]
            )
            movement_method = focal_movement_method + ["none"] * n_bkg
            random_walk_range = np.concatenate([focal_random_walk_range, bkg_zeros_i])
            out_mortality_rate = np.concatenate([focal_out_mortality_rate, bkg_zeros_f])
            n_schools = np.concatenate([focal_n_schools, bkg_zeros_i])
        else:
            all_species_names = focal_species_names[:]
            linf = focal_linf
            k = focal_k
            t0 = focal_t0
            egg_size = focal_egg_size
            condition_factor = focal_condition_factor
            allometric_power = focal_allometric_power
            vb_threshold_age = focal_vb_threshold_age
            lifespan_dt = focal_lifespan_dt
            ingestion_rate = focal_ingestion_rate
            critical_success_rate = focal_critical_success_rate
            delta_lmax_factor = focal_delta_lmax_factor
            additional_mortality_rate = focal_additional_mortality_rate
            sex_ratio = focal_sex_ratio
            relative_fecundity = focal_relative_fecundity
            maturity_size = focal_maturity_size
            seeding_biomass = focal_seeding_biomass
            larva_mortality_rate = focal_larva_mortality_rate
            starvation_rate_max = focal_starvation_rate_max
            fishing_rate = fishing
            fishing_selectivity_l50 = focal_fishing_selectivity_l50
            fishing_selectivity_a50 = focal_fishing_a50
            fishing_selectivity_type = focal_fishing_sel_type
            movement_method = focal_movement_method
            random_walk_range = focal_random_walk_range
            out_mortality_rate = focal_out_mortality_rate
            n_schools = focal_n_schools

        # Egg weight override: use species.egg.weight.sp{i} if provided
        egg_weight_vals = [cfg.get(f"species.egg.weight.sp{i}", "") for i in range(n_sp)]
        if any(v for v in egg_weight_vals):
            egg_weight_override = np.array(
                [float(v) if v else float("nan") for v in egg_weight_vals], dtype=np.float64
            )
        else:
            egg_weight_override = None

        return cls(
            n_species=n_sp,
            n_dt_per_year=n_dt,
            n_year=n_yr,
            n_steps=n_dt * n_yr,
            n_schools=n_schools,
            species_names=focal_species_names,
            n_background=n_bkg,
            background_file_indices=[b.file_index for b in background_list],
            all_species_names=all_species_names,
            linf=linf,
            k=k,
            t0=t0,
            egg_size=egg_size,
            condition_factor=condition_factor,
            allometric_power=allometric_power,
            vb_threshold_age=vb_threshold_age,
            lifespan_dt=lifespan_dt,
            mortality_subdt=max(1, int(cfg.get("mortality.subdt", "10"))),
            ingestion_rate=ingestion_rate,
            critical_success_rate=critical_success_rate,
            delta_lmax_factor=delta_lmax_factor,
            additional_mortality_rate=additional_mortality_rate,
            sex_ratio=sex_ratio,
            relative_fecundity=relative_fecundity,
            maturity_size=maturity_size,
            seeding_biomass=seeding_biomass,
            larva_mortality_rate=larva_mortality_rate,
            size_ratio_min=size_ratio_min_2d,
            size_ratio_max=size_ratio_max_2d,
            feeding_stage_thresholds=all_thresholds,
            feeding_stage_metric=all_metrics,
            n_feeding_stages=n_feeding_stages,
            starvation_rate_max=starvation_rate_max,
            accessibility_matrix=_load_accessibility(cfg, n_sp),
            stage_accessibility=_load_stage_accessibility(cfg, all_species_names),
            spawning_season=_load_spawning_seasons(cfg, n_sp, n_dt),
            fishing_enabled=(
                cfg.get("simulation.fishing.mortality.enabled", "true").lower() == "true"
                or fisheries_enabled
            ),
            fishing_rate=fishing_rate,
            fishing_selectivity_l50=fishing_selectivity_l50,
            fishing_selectivity_a50=fishing_selectivity_a50,
            fishing_selectivity_type=fishing_selectivity_type,
            movement_method=movement_method,
            random_walk_range=random_walk_range,
            out_mortality_rate=out_mortality_rate,
            egg_weight_override=egg_weight_override,
            raw_config=cfg,
        )
