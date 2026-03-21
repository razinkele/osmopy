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



_GROWTH_MAP: dict[str, str] = {
    # Current canonical classnames
    "fr.ird.osmose.process.growth.VonBertalanffyGrowth": "VB",
    "fr.ird.osmose.process.growth.GompertzGrowth": "GOMPERTZ",
    # Legacy backward compat
    "fr.ird.osmose.growth.VonBertalanffy": "VB",
    "fr.ird.osmose.growth.Gompertz": "GOMPERTZ",
    "fr.ird.osmose.growth.Linear": "VB",  # Linear was never real, map to VB
}


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


_config_dir: str = ""


def _set_config_dir(config_dir: str) -> None:
    """Set the config base directory for file resolution."""
    global _config_dir
    _config_dir = config_dir


def _search_dirs() -> list[Path]:
    """Build a list of directories to search for data files."""
    import glob as _glob

    dirs: list[Path] = []
    if _config_dir:
        dirs.append(Path(_config_dir))
    dirs.append(Path("."))
    dirs.append(Path("data/examples"))
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


def _load_spatial_csv(path: Path) -> np.ndarray:
    """Load a semicolon-separated spatial grid CSV and flip rows (south-to-north → north-to-south)."""
    df = pd.read_csv(path, sep=";", header=None)
    data = df.values.astype(np.float64)
    return np.flipud(data)


def _load_accessibility(cfg: dict[str, str], n_species: int) -> NDArray[np.float64] | None:
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


@dataclass
class MPAZone:
    """A Marine Protected Area definition."""

    grid: NDArray[np.float64]  # 2D spatial grid (1 = protected, 0 = not)
    start_year: int
    end_year: int
    percentage: float  # reduction factor (0-1)


def _parse_fisheries(
    cfg: dict[str, str], species_names: list[str], n_species: int
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Parse fisheries-based fishing config (OSMOSE v4).

    Returns
    -------
    fishing_rate: NDArray[np.float64]
        Annual fishing rate per species.
    fishing_selectivity_a50: NDArray[np.float64]
        Age at 50% selectivity (years) per species. NaN if not applicable.
    fishing_selectivity_type: NDArray[np.int32]
        0 = age-based (knife-edge), 1 = sigmoidal size, -1 = no fishing.
    fishing_selectivity_l50: NDArray[np.float64]
        Length at 50% selectivity for sigmoidal type. 0 if not applicable.
    fishing_selectivity_slope: NDArray[np.float64]
        Slope of sigmoid selectivity. 0 if not applicable.
    """
    fishing_rate = np.zeros(n_species, dtype=np.float64)
    fishing_a50 = np.full(n_species, np.nan, dtype=np.float64)
    fishing_sel_type = np.full(n_species, -1, dtype=np.int32)
    fishing_l50 = np.zeros(n_species, dtype=np.float64)
    fishing_slope = np.zeros(n_species, dtype=np.float64)

    n_fisheries = int(cfg.get("simulation.nfisheries", "0"))
    if n_fisheries == 0:
        return fishing_rate, fishing_a50, fishing_sel_type, fishing_l50, fishing_slope

    # Read catchability CSV to map species -> fishery
    catch_file = cfg.get("fisheries.catchability.file", "")
    catch_path = _resolve_file(catch_file)
    if catch_path is None:
        return fishing_rate, fishing_a50, fishing_sel_type, fishing_l50, fishing_slope

    catch_df = pd.read_csv(catch_path, index_col=0)
    # Row labels = species names, column labels = fishery names
    # Build species_name -> fishery_index mapping
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

        # Selectivity type: 0 = knife-edge by age, 1 = sigmoidal size
        sel_type = int(cfg.get(f"fisheries.selectivity.type.fsh{fsh_idx}", "0"))
        if sel_type == 0:
            # Age-based knife-edge
            a50_key = f"fisheries.selectivity.a50.fsh{fsh_idx}"
            a50_val = cfg.get(a50_key, "0.0")
            fishing_a50[sp_idx] = float(a50_val)
            fishing_sel_type[sp_idx] = 0
        elif sel_type == 1:
            # Sigmoidal size selectivity
            fishing_sel_type[sp_idx] = 1
            fishing_l50[sp_idx] = float(cfg.get(f"fisheries.selectivity.l50.fsh{fsh_idx}", "0.0"))
            fishing_slope[sp_idx] = float(
                cfg.get(f"fisheries.selectivity.slope.fsh{fsh_idx}", "1.0")
            )
        else:
            fishing_sel_type[sp_idx] = sel_type

    return fishing_rate, fishing_a50, fishing_sel_type, fishing_l50, fishing_slope


def _load_fishing_seasonality(
    cfg: dict[str, str],
    n_species: int,
    n_dt_per_year: int,
    species_names: list[str] | None = None,
) -> NDArray[np.float64] | None:
    """Load fishing seasonality for each species (v3 sp{i} or v4 fsh{i} format).

    Supports:
    - ``fisheries.seasonality.file.sp{i}`` — per-species CSV file (v3)
    - ``fisheries.seasonality.file.fsh{i}`` — per-fishery CSV file (v4)
    - ``fisheries.seasonality.fsh{i}`` — inline semicolon-separated values (v4)

    Returns array of shape (n_species, n_dt_per_year) with season weights.
    None if no seasonality found.
    """
    seasons = np.ones((n_species, n_dt_per_year), dtype=np.float64) / n_dt_per_year
    found_any = False

    # Build fishery-to-species mapping for v4 fisheries
    fsh_to_sp: dict[int, int] = {}
    catch_file = cfg.get("fisheries.catchability.file", "")
    if catch_file and species_names:
        catch_path = _resolve_file(catch_file)
        if catch_path is not None:
            catch_df = pd.read_csv(catch_path, index_col=0)
            for row_idx in range(len(catch_df)):
                row_name = str(catch_df.index[row_idx]).strip().lower()
                for col_idx in range(len(catch_df.columns)):
                    if float(catch_df.iloc[row_idx, col_idx]) > 0:
                        for sp_idx, name in enumerate(species_names):
                            if name.strip().lower() == row_name:
                                fsh_to_sp[col_idx] = sp_idx
                                break
                        break

    def _set_season(sp_idx: int, values: NDArray[np.float64]) -> None:
        nonlocal found_any
        if len(values) >= n_dt_per_year:
            vals = values[:n_dt_per_year]
            total = vals.sum()
            if total > 0:
                seasons[sp_idx] = vals / total
                found_any = True

    # Try v3 per-species file keys first
    for i in range(n_species):
        file_key = cfg.get(f"fisheries.seasonality.file.sp{i}", "")
        if not file_key:
            continue
        path = _resolve_file(file_key)
        if path is not None:
            df = pd.read_csv(path, sep=";")
            _set_season(i, df.iloc[:, 1].values.astype(np.float64))

    # Try v4 per-fishery keys (file or inline)
    n_fisheries = int(cfg.get("simulation.nfisheries", "0"))
    for fsh in range(n_fisheries):
        sp_idx = fsh_to_sp.get(fsh)
        if sp_idx is None:
            continue

        # Try file reference
        file_key = cfg.get(f"fisheries.seasonality.file.fsh{fsh}", "")
        if file_key:
            path = _resolve_file(file_key)
            if path is not None:
                df = pd.read_csv(path, sep=";")
                _set_season(sp_idx, df.iloc[:, 1].values.astype(np.float64))
                continue

        # Try inline semicolon-separated values
        inline_val = cfg.get(f"fisheries.seasonality.fsh{fsh}", "")
        if inline_val:
            try:
                vals = np.array(
                    [float(v.strip()) for v in inline_val.split(";") if v.strip()],
                    dtype=np.float64,
                )
                _set_season(sp_idx, vals)
            except (ValueError, TypeError) as exc:
                warnings.warn(
                    f"Invalid fisheries.seasonality.fsh{fsh} value: {inline_val!r} — {exc}",
                    stacklevel=2,
                )

    return seasons if found_any else None


def _load_fishing_rate_by_year(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load time-varying annual fishing rate CSV for each species.

    Returns list of arrays (one per species), or None if no files found.
    """
    result: list[NDArray[np.float64] | None] = [None] * n_species
    found_any = False

    for i in range(n_species):
        file_key = cfg.get(f"mortality.fishing.rate.byyear.file.sp{i}", "")
        if not file_key:
            continue
        path = _resolve_file(file_key)
        if path is not None:
            values = np.loadtxt(path, dtype=np.float64)
            result[i] = values.flatten()
            found_any = True

    return result if found_any else None


def _parse_mpa_zones(cfg: dict[str, str]) -> list[MPAZone] | None:
    """Parse Marine Protected Area configurations."""
    zones: list[MPAZone] = []
    i = 0
    while True:
        file_key = cfg.get(f"mpa.file.mpa{i}", "")
        if not file_key:
            break
        path = _resolve_file(file_key)
        if path is None:
            i += 1
            continue
        grid = _load_spatial_csv(path)
        start_year = int(cfg.get(f"mpa.start.year.mpa{i}", "0"))
        end_year = int(cfg.get(f"mpa.end.year.mpa{i}", "999"))
        percentage = float(cfg.get(f"mpa.percentage.mpa{i}", "1.0"))
        zones.append(
            MPAZone(grid=grid, start_year=start_year, end_year=end_year, percentage=percentage)
        )
        i += 1

    return zones if zones else None


def _load_discard_rates(
    cfg: dict[str, str], species_names: list[str], n_species: int
) -> NDArray[np.float64] | None:
    """Load fishery discard rates from CSV.

    Returns per-species discard rate array, or None if no discard file.
    """
    file_key = cfg.get("fisheries.discards.file", "")
    path = _resolve_file(file_key)
    if path is None:
        return None

    df = pd.read_csv(path, index_col=0)
    discard_rate = np.zeros(n_species, dtype=np.float64)

    for sp_idx in range(n_species):
        sp_name = species_names[sp_idx].strip()
        if sp_name in df.index:
            row = df.loc[sp_name]
            # Take the max discard rate across fisheries (species is caught by one fishery)
            vals = row.values.astype(np.float64)
            nonzero = vals[vals > 0]
            if len(nonzero) > 0:
                discard_rate[sp_idx] = nonzero[0]

    return discard_rate


def _load_spawning_seasons(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int
) -> NDArray[np.float64] | None:
    """Load spawning season CSV files for each species.

    Returns array of shape (n_species, n_columns) with season weights.
    n_columns equals n_dt_per_year for single-year data, or n_dt_per_year * n_years
    for multi-year time series.
    """
    normalize = cfg.get("reproduction.normalisation.enabled", "false").lower() == "true"

    # First pass: load all values to determine max column count
    all_values: list[NDArray[np.float64] | None] = [None] * n_species
    max_cols = n_dt_per_year
    found_any = False

    for i in range(n_species):
        file_key = cfg.get(f"reproduction.season.file.sp{i}", "")
        if not file_key:
            continue
        path = _resolve_file(file_key)
        if path is not None:
            df = pd.read_csv(path, sep=";")
            values = df.iloc[:, 1].values.astype(np.float64)
            if len(values) >= n_dt_per_year:
                all_values[i] = values
                max_cols = max(max_cols, len(values))
                found_any = True

    if not found_any:
        return None

    seasons = np.ones((n_species, max_cols), dtype=np.float64) / n_dt_per_year
    for i in range(n_species):
        if all_values[i] is not None:
            vals = all_values[i]
            n_vals = len(vals)
            if normalize:
                # Normalize per year: sum over each n_dt_per_year chunk
                total = vals.sum()
                if total > 0:
                    vals = vals / total
            seasons[i, :n_vals] = vals
            # Pad remaining columns with uniform if multi-year array is shorter
            if n_vals < max_cols:
                seasons[i, n_vals:] = 1.0 / n_dt_per_year

    return seasons


def _load_additional_mortality_by_dt(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load time-varying additional mortality CSV (BY_DT scenario).

    Returns a list of arrays (one per species), or None if no files found.
    """
    result: list[NDArray[np.float64] | None] = [None] * n_species
    found_any = False

    for i in range(n_species):
        file_key = cfg.get(f"mortality.additional.rate.bytdt.file.sp{i}", "")
        if not file_key:
            continue
        path = _resolve_file(file_key)
        if path is not None:
            values = np.loadtxt(path, dtype=np.float64)
            result[i] = values.flatten()
            found_any = True

    return result if found_any else None


def _load_additional_mortality_spatial(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load spatial additional mortality distribution maps.

    Returns a list of 2D arrays (one per species), or None if no files found.
    """
    result: list[NDArray[np.float64] | None] = [None] * n_species
    found_any = False

    for i in range(n_species):
        file_key = cfg.get(f"mortality.additional.spatial.distrib.file.sp{i}", "")
        if not file_key:
            continue
        path = _resolve_file(file_key)
        if path is not None:
            result[i] = _load_spatial_csv(path)
            found_any = True

    return result if found_any else None


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
    additional_mortality_by_dt: list[NDArray[np.float64] | None] | None  # BY_DT per-step rates
    additional_mortality_spatial: list[NDArray[np.float64] | None] | None  # spatial multiplier maps

    # Reproduction
    sex_ratio: NDArray[np.float64]  # fraction female per species
    relative_fecundity: NDArray[np.float64]  # eggs per gram of mature female
    maturity_size: NDArray[np.float64]  # length at maturity (cm)
    seeding_biomass: NDArray[np.float64]  # initial biomass for seeding (tonnes)
    seeding_max_step: NDArray[np.int32]  # max step for seeding (default: lifespan_dt)
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
    fishing_selectivity_type: NDArray[np.int32]  # 0=age, 1=sigmoidal size, -1=none
    fishing_selectivity_slope: NDArray[np.float64]  # sigmoid slope (type 1 only)

    # Fishing seasonality: (n_species, n_dt_per_year) normalized weights, or None
    fishing_seasonality: NDArray[np.float64] | None

    # Fishing rate by year: per-species array of annual rates, or None
    fishing_rate_by_year: list[NDArray[np.float64] | None] | None

    # Marine Protected Areas
    mpa_zones: list[MPAZone] | None

    # Fishery discards: per-species discard fraction (0-1), or None
    fishing_discard_rate: NDArray[np.float64] | None

    # Predation accessibility
    accessibility_matrix: NDArray[np.float64] | None  # (n_pred, n_prey) or None
    stage_accessibility: AccessibilityMatrix | None  # stage-indexed accessibility, or None

    # Reproduction
    spawning_season: NDArray[np.float64] | None  # (n_species, n_dt_per_year) or None

    # Movement
    movement_method: list[str]
    random_walk_range: NDArray[np.int32]
    out_mortality_rate: NDArray[np.float64]

    # Maturity age in timesteps (0 = no age threshold, only size-based)
    maturity_age_dt: NDArray[np.int32]

    # Maximum length cap (may differ from linf)
    lmax: NDArray[np.float64]

    # Spatial fishing distribution maps: one 2D grid per species, or None
    fishing_spatial_maps: list  # list[NDArray[np.float64] | None]

    # Egg weight override: if species.egg.weight.sp{i} is set, use it instead of allometry
    egg_weight_override: NDArray[np.float64] | None  # shape (n_species,) or None

    # Output cutoff age: exclude schools younger than this from biomass/abundance output
    output_cutoff_age: NDArray[np.float64] | None  # shape (n_total,) or None

    # Output recording frequency (steps between output records)
    output_record_frequency: int

    # Diet composition output
    diet_output_enabled: bool

    # Initial state output (step -1)
    output_step0_include: bool

    # RNG seeding flags
    movement_seed_fixed: bool  # per-species independent RNG for movement
    mortality_seed_fixed: bool  # per-species independent RNG for mortality

    # Random distribution patch constraint: per-species ncell values, or None
    random_distribution_ncell: NDArray[np.int32] | None

    # Growth class per species: "VB" or "GOMPERTZ"
    growth_class: list[str]

    # Raw config dict for subsystems that need unparsed access (e.g. ResourceState)
    raw_config: dict[str, str]

    # Gompertz growth parameters (None when no GOMPERTZ species)
    gompertz_ke: NDArray[np.float64] | None = None
    gompertz_lstart: NDArray[np.float64] | None = None
    gompertz_kg: NDArray[np.float64] | None = None
    gompertz_tg: NDArray[np.float64] | None = None
    gompertz_linf: NDArray[np.float64] | None = None
    gompertz_thr_age_exp_dt: NDArray[np.int32] | None = None
    gompertz_thr_age_gom_dt: NDArray[np.int32] | None = None

    # Bioenergetic model toggle
    bioen_enabled: bool = False

    # Bioenergetic global flags
    bioen_phit_enabled: bool = True
    bioen_fo2_enabled: bool = True

    # Bioenergetic per-species parameters (None when bioen disabled)
    bioen_beta: NDArray[np.float64] | None = None           # allometric exponent
    bioen_zlayer: NDArray[np.int32] | None = None           # depth layer index
    bioen_assimilation: NDArray[np.float64] | None = None   # assimilation efficiency
    bioen_c_m: NDArray[np.float64] | None = None            # maintenance coefficient
    bioen_eta: NDArray[np.float64] | None = None            # energy density ratio
    bioen_r: NDArray[np.float64] | None = None              # reproductive allocation
    bioen_m0: NDArray[np.float64] | None = None             # LMRN intercept
    bioen_m1: NDArray[np.float64] | None = None             # LMRN slope
    bioen_e_mobi: NDArray[np.float64] | None = None         # Johnson e_M (eV)
    bioen_e_d: NDArray[np.float64] | None = None            # Johnson e_D (eV)
    bioen_tp: NDArray[np.float64] | None = None             # peak temperature (°C)
    bioen_e_maint: NDArray[np.float64] | None = None        # Arrhenius maintenance energy (eV)
    bioen_o2_c1: NDArray[np.float64] | None = None          # O2 dose-response asymptote
    bioen_o2_c2: NDArray[np.float64] | None = None          # O2 half-saturation
    bioen_i_max: NDArray[np.float64] | None = None          # max ingestion rate (bioen)
    bioen_theta: NDArray[np.float64] | None = None          # larvae ingestion multiplier
    bioen_c_rate: NDArray[np.float64] | None = None         # larvae correction coefficient
    bioen_k_for: NDArray[np.float64] | None = None          # foraging mortality

    # Distribution output flags
    output_biomass_byage: bool = False
    output_biomass_bysize: bool = False
    output_abundance_byage: bool = False
    output_abundance_bysize: bool = False
    output_size_min: float = 0.0
    output_size_max: float = 205.0
    output_size_incr: float = 10.0

    # Bioenergetic output flags (default False; meanEnet is always written when bioen enabled)
    output_bioen_ingest: bool = False
    output_bioen_maint: bool = False
    output_bioen_rho: bool = False
    output_bioen_sizeinf: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
        _set_config_dir(cfg.get("_osmose.config.dir", ""))
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
            (
                fishing,
                focal_fishing_a50,
                focal_fishing_sel_type,
                focal_fishing_l50_fsh,
                focal_fishing_slope,
            ) = _parse_fisheries(cfg, _focal_names, n_sp)
        else:
            # Legacy per-species rate
            fishing = _species_float_optional(
                cfg, "mortality.fishing.rate.sp{i}", n_sp, default=0.0
            )
            if fishing.sum() == 0:
                fishing = _species_float_optional(cfg, "fishing.rate.sp{i}", n_sp, default=0.0)
            focal_fishing_a50 = np.full(n_sp, np.nan, dtype=np.float64)
            focal_fishing_sel_type = np.full(n_sp, -1, dtype=np.int32)
            focal_fishing_l50_fsh = np.zeros(n_sp, dtype=np.float64)
            focal_fishing_slope = np.zeros(n_sp, dtype=np.float64)

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
        # Seeding max step: explicit override or default to lifespan
        seeding_max_year_str = cfg.get("population.seeding.year.max", "")
        if seeding_max_year_str:
            seeding_max_years = float(seeding_max_year_str)
            focal_seeding_max_step = np.full(n_sp, int(seeding_max_years * n_dt), dtype=np.int32)
        else:
            focal_seeding_max_step = (lifespan_years * n_dt).astype(np.int32)
        focal_larva_mortality_rate = _species_float_optional(
            cfg, "mortality.additional.larva.rate.sp{i}", n_sp, default=0.0
        )
        # Maturity age (years → timesteps); default 0 means no age threshold
        focal_maturity_age_years = _species_float_optional(
            cfg, "species.maturity.age.sp{i}", n_sp, default=0.0
        )
        focal_maturity_age_dt = (focal_maturity_age_years * n_dt).astype(np.int32)
        # Lmax: separate cap that may differ from linf
        focal_lmax = _species_float_optional(cfg, "species.lmax.sp{i}", n_sp, default=0.0)
        # Default to linf if not set (value 0.0 treated as absent)
        for i in range(n_sp):
            if focal_lmax[i] <= 0:
                focal_lmax[i] = focal_linf[i]
        # Fishing spatial distribution maps
        focal_fishing_spatial_maps: list[np.ndarray | None] = []
        # Try shared fisheries map first (v4)
        shared_fishing_map_file = cfg.get("fisheries.movement.file.map0", "")
        shared_fishing_map: np.ndarray | None = None
        if shared_fishing_map_file:
            shared_path = _resolve_file(shared_fishing_map_file)
            if shared_path is not None:
                shared_fishing_map = _load_spatial_csv(shared_path)
        for i in range(n_sp):
            sp_map_file = cfg.get(f"mortality.fishing.spatial.distrib.file.sp{i}", "")
            if sp_map_file:
                sp_path = _resolve_file(sp_map_file)
                if sp_path is not None:
                    focal_fishing_spatial_maps.append(_load_spatial_csv(sp_path))
                else:
                    focal_fishing_spatial_maps.append(shared_fishing_map)
            else:
                focal_fishing_spatial_maps.append(shared_fishing_map)

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
        # Merge fisheries-parsed L50 (for sigmoid type) into selectivity L50
        for i in range(n_sp):
            if focal_fishing_l50_fsh[i] > 0 and focal_fishing_selectivity_l50[i] == 0:
                focal_fishing_selectivity_l50[i] = focal_fishing_l50_fsh[i]
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
            seeding_max_step = np.concatenate([focal_seeding_max_step, bkg_zeros_i])
            larva_mortality_rate = np.concatenate([focal_larva_mortality_rate, bkg_zeros_f])
            maturity_age_dt = np.concatenate([focal_maturity_age_dt, bkg_zeros_i])
            lmax = np.concatenate([focal_lmax, bkg_zeros_f])
            starvation_rate_max = np.concatenate([focal_starvation_rate_max, bkg_zeros_f])
            fishing_rate = np.concatenate([fishing, bkg_zeros_f])
            fishing_selectivity_l50 = np.concatenate([focal_fishing_selectivity_l50, bkg_zeros_f])
            fishing_selectivity_a50 = np.concatenate(
                [focal_fishing_a50, np.full(n_bkg, np.nan, dtype=np.float64)]
            )
            fishing_selectivity_type = np.concatenate(
                [focal_fishing_sel_type, np.full(n_bkg, -1, dtype=np.int32)]
            )
            fishing_selectivity_slope = np.concatenate([focal_fishing_slope, bkg_zeros_f])
            movement_method = focal_movement_method + ["none"] * n_bkg
            random_walk_range = np.concatenate([focal_random_walk_range, bkg_zeros_i])
            out_mortality_rate = np.concatenate([focal_out_mortality_rate, bkg_zeros_f])
            n_schools = np.concatenate([focal_n_schools, bkg_zeros_i])
            fishing_spatial_maps = focal_fishing_spatial_maps + [None] * n_bkg
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
            seeding_max_step = focal_seeding_max_step
            larva_mortality_rate = focal_larva_mortality_rate
            maturity_age_dt = focal_maturity_age_dt
            lmax = focal_lmax
            starvation_rate_max = focal_starvation_rate_max
            fishing_rate = fishing
            fishing_selectivity_l50 = focal_fishing_selectivity_l50
            fishing_selectivity_a50 = focal_fishing_a50
            fishing_selectivity_type = focal_fishing_sel_type
            fishing_selectivity_slope = focal_fishing_slope
            movement_method = focal_movement_method
            random_walk_range = focal_random_walk_range
            out_mortality_rate = focal_out_mortality_rate
            n_schools = focal_n_schools
            fishing_spatial_maps = focal_fishing_spatial_maps

        # Egg weight override: use species.egg.weight.sp{i} if provided
        # Java convention: config value is in GRAMS, convert to tonnes (* 1e-6)
        egg_weight_vals = [cfg.get(f"species.egg.weight.sp{i}", "") for i in range(n_sp)]
        if any(v for v in egg_weight_vals):
            egg_weight_override = np.array(
                [float(v) * 1e-6 if v else float("nan") for v in egg_weight_vals],
                dtype=np.float64,
            )
        else:
            egg_weight_override = None

        # Output cutoff age: output.cutoff.age.sp{i} (years, default 0 = include all)
        n_total = n_sp + n_bkg
        cutoff_vals = []
        found_any = False
        for i in range(n_sp):
            val = cfg.get(f"output.cutoff.age.sp{i}", "")
            if val and val.lower() not in ("null", "none", ""):
                cutoff_vals.append(float(val))
                found_any = True
            else:
                cutoff_vals.append(0.0)
        # Pad for background species (no cutoff)
        cutoff_vals.extend([0.0] * n_bkg)
        output_cutoff_age = np.array(cutoff_vals, dtype=np.float64) if found_any else None

        # Phase 2 fishing features
        fishing_seasonality = _load_fishing_seasonality(cfg, n_sp, n_dt, focal_species_names)
        fishing_rate_by_year = _load_fishing_rate_by_year(cfg, n_sp)
        mpa_zones = _parse_mpa_zones(cfg)
        fishing_discard_rate = _load_discard_rates(cfg, focal_species_names, n_sp)

        # Phase 4: Random distribution patch constraint
        ncell_vals = []
        ncell_found = False
        for i in range(n_sp):
            val = cfg.get(f"movement.distribution.ncell.sp{i}", "")
            if val:
                ncell_vals.append(int(val))
                ncell_found = True
            else:
                ncell_vals.append(0)
        random_distribution_ncell = (
            np.array(ncell_vals, dtype=np.int32) if ncell_found else None
        )

        # Phase 5: Output recording frequency (default: 1 = every step)
        output_record_freq = int(cfg.get("output.recordfrequency.ndt", "1"))

        # Phase 5: Diet output
        diet_output_enabled = cfg.get("output.diet.composition.enabled", "false").lower() == "true"

        # Phase 5: Initial state output
        output_step0 = cfg.get("output.step0.include", "false").lower() == "true"

        # Phase 4: Random seed flags
        movement_seed_fixed = cfg.get("movement.randomseed.fixed", "false").lower() == "true"
        mortality_seed_fixed = (
            cfg.get("stochastic.mortality.randomseed.fixed", "false").lower() == "true"
        )

        # Growth class dispatch: parse classname for each focal species
        growth_class = [
            _GROWTH_MAP.get(
                cfg.get(f"growth.java.classname.sp{i}", "").strip(),
                "VB",
            )
            for i in range(n_sp)
        ]

        # Gompertz parameters: only parsed when at least one species uses GOMPERTZ
        gompertz_ke = gompertz_lstart = gompertz_kg = gompertz_tg = gompertz_linf = None
        gompertz_thr_age_exp_dt = gompertz_thr_age_gom_dt = None
        if "GOMPERTZ" in growth_class:
            gompertz_ke = _species_float_optional(cfg, "growth.exponential.ke.sp{i}", n_sp, 0.0)
            gompertz_lstart = _species_float_optional(
                cfg, "growth.exponential.lstart.sp{i}", n_sp, 0.1
            )
            gompertz_kg = _species_float_optional(cfg, "growth.gompertz.kg.sp{i}", n_sp, 0.0)
            gompertz_tg = _species_float_optional(cfg, "growth.gompertz.tg.sp{i}", n_sp, 0.0)
            gompertz_linf = _species_float_optional(
                cfg, "growth.gompertz.linf.sp{i}", n_sp, 0.0
            )
            exp_yrs = _species_float_optional(
                cfg, "growth.exponential.thr.age.sp{i}", n_sp, 0.0
            )
            gom_yrs = _species_float_optional(
                cfg, "growth.gompertz.thr.age.sp{i}", n_sp, 0.0
            )
            gompertz_thr_age_exp_dt = (exp_yrs * n_dt).astype(np.int32)
            gompertz_thr_age_gom_dt = (gom_yrs * n_dt).astype(np.int32)

        # Bioenergetic parameters: only parsed when simulation.bioen.enabled=true
        _bioen_enabled = cfg.get("simulation.bioen.enabled", "false").lower() == "true"
        _bioen_phit_enabled = cfg.get("simulation.bioen.phit.enabled", "true").lower() == "true"
        _bioen_fo2_enabled = cfg.get("simulation.bioen.fo2.enabled", "true").lower() == "true"
        bioen_beta = bioen_zlayer = bioen_assimilation = bioen_c_m = None
        bioen_eta = bioen_r = bioen_m0 = bioen_m1 = None
        bioen_e_mobi = bioen_e_d = bioen_tp = bioen_e_maint = None
        bioen_o2_c1 = bioen_o2_c2 = bioen_i_max = bioen_theta = bioen_c_rate = bioen_k_for = None
        if _bioen_enabled:
            bioen_beta = _species_float_optional(cfg, "species.beta.sp{i}", n_sp, 0.8)
            bioen_zlayer = _species_int_optional(cfg, "species.zlayer.sp{i}", n_sp, 0)
            bioen_assimilation = _species_float_optional(
                cfg, "species.bioen.assimilation.sp{i}", n_sp, 0.7
            )
            bioen_c_m = _species_float_optional(
                cfg, "species.bioen.maint.energy.c_m.sp{i}", n_sp, 0.0
            )
            bioen_eta = _species_float_optional(
                cfg, "species.bioen.maturity.eta.sp{i}", n_sp, 1.0
            )
            bioen_r = _species_float_optional(
                cfg, "species.bioen.maturity.r.sp{i}", n_sp, 0.0
            )
            bioen_m0 = _species_float_optional(
                cfg, "species.bioen.maturity.m0.sp{i}", n_sp, 0.0
            )
            bioen_m1 = _species_float_optional(
                cfg, "species.bioen.maturity.m1.sp{i}", n_sp, 0.0
            )
            bioen_e_mobi = _species_float_optional(
                cfg, "species.bioen.mobilized.e.mobi.sp{i}", n_sp, 0.65
            )
            bioen_e_d = _species_float_optional(
                cfg, "species.bioen.mobilized.e.D.sp{i}", n_sp, 1.5
            )
            bioen_tp = _species_float_optional(
                cfg, "species.bioen.mobilized.Tp.sp{i}", n_sp, 20.0
            )
            bioen_e_maint = _species_float_optional(
                cfg, "species.bioen.maint.e.maint.sp{i}", n_sp, 0.65
            )
            bioen_o2_c1 = _species_float_optional(
                cfg, "species.oxygen.c1.sp{i}", n_sp, 1.0
            )
            bioen_o2_c2 = _species_float_optional(
                cfg, "species.oxygen.c2.sp{i}", n_sp, 1.0
            )
            bioen_i_max = _species_float_optional(
                cfg, "predation.ingestion.rate.max.bioen.sp{i}", n_sp, 0.0
            )
            bioen_theta = _species_float_optional(
                cfg, "predation.coef.ingestion.rate.max.larvae.bioen.sp{i}", n_sp, 1.0
            )
            bioen_c_rate = _species_float_optional(
                cfg, "predation.c.bioen.sp{i}", n_sp, 0.0
            )
            bioen_k_for = _species_float_optional(
                cfg, "species.bioen.forage.k_for.sp{i}", n_sp, 0.0
            )

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
            additional_mortality_by_dt=_load_additional_mortality_by_dt(cfg, n_sp),
            additional_mortality_spatial=_load_additional_mortality_spatial(cfg, n_sp),
            sex_ratio=sex_ratio,
            relative_fecundity=relative_fecundity,
            maturity_size=maturity_size,
            maturity_age_dt=maturity_age_dt,
            lmax=lmax,
            fishing_spatial_maps=fishing_spatial_maps,
            seeding_biomass=seeding_biomass,
            seeding_max_step=seeding_max_step,
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
            fishing_selectivity_slope=fishing_selectivity_slope,
            fishing_seasonality=fishing_seasonality,
            fishing_rate_by_year=fishing_rate_by_year,
            mpa_zones=mpa_zones,
            fishing_discard_rate=fishing_discard_rate,
            movement_method=movement_method,
            random_walk_range=random_walk_range,
            out_mortality_rate=out_mortality_rate,
            egg_weight_override=egg_weight_override,
            output_cutoff_age=output_cutoff_age,
            output_record_frequency=output_record_freq,
            diet_output_enabled=diet_output_enabled,
            output_step0_include=output_step0,
            movement_seed_fixed=movement_seed_fixed,
            mortality_seed_fixed=mortality_seed_fixed,
            random_distribution_ncell=random_distribution_ncell,
            growth_class=growth_class,
            gompertz_ke=gompertz_ke,
            gompertz_lstart=gompertz_lstart,
            gompertz_kg=gompertz_kg,
            gompertz_tg=gompertz_tg,
            gompertz_linf=gompertz_linf,
            gompertz_thr_age_exp_dt=gompertz_thr_age_exp_dt,
            gompertz_thr_age_gom_dt=gompertz_thr_age_gom_dt,
            raw_config=cfg,
            bioen_enabled=_bioen_enabled,
            bioen_phit_enabled=_bioen_phit_enabled,
            bioen_fo2_enabled=_bioen_fo2_enabled,
            bioen_beta=bioen_beta,
            bioen_zlayer=bioen_zlayer,
            bioen_assimilation=bioen_assimilation,
            bioen_c_m=bioen_c_m,
            bioen_eta=bioen_eta,
            bioen_r=bioen_r,
            bioen_m0=bioen_m0,
            bioen_m1=bioen_m1,
            bioen_e_mobi=bioen_e_mobi,
            bioen_e_d=bioen_e_d,
            bioen_tp=bioen_tp,
            bioen_e_maint=bioen_e_maint,
            bioen_o2_c1=bioen_o2_c1,
            bioen_o2_c2=bioen_o2_c2,
            bioen_i_max=bioen_i_max,
            bioen_theta=bioen_theta,
            bioen_c_rate=bioen_c_rate,
            bioen_k_for=bioen_k_for,
            output_biomass_byage=cfg.get("output.biomass.byage.enabled", "false").lower() == "true",
            output_biomass_bysize=cfg.get("output.biomass.bysize.enabled", "false").lower() == "true",
            output_abundance_byage=cfg.get("output.abundance.byage.enabled", "false").lower() == "true",
            output_abundance_bysize=cfg.get("output.abundance.bysize.enabled", "false").lower() == "true",
            output_size_min=float(cfg.get("output.distrib.bysize.min", "0")),
            output_size_max=float(cfg.get("output.distrib.bysize.max", "205")),
            output_size_incr=float(cfg.get("output.distrib.bysize.incr", "10")),
            output_bioen_ingest=cfg.get("output.bioen.ingest.enabled", "false").lower() == "true",
            output_bioen_maint=cfg.get("output.bioen.maint.enabled", "false").lower() == "true",
            output_bioen_rho=cfg.get("output.bioen.rho.enabled", "false").lower() == "true",
            output_bioen_sizeinf=cfg.get("output.bioen.sizeInf.enabled", "false").lower() == "true",
        )
