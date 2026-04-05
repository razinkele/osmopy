"""Background species support for the OSMOSE Python engine.

Background species are an intermediate tier: they have size classes with biomass
from external forcing, participate in predation as both predators and prey, but
don't grow or reproduce.
"""

from __future__ import annotations

import glob as _glob
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _resolve_path(filepath_str: str, config_dir: str = "") -> Path:
    """Resolve a relative file path against multiple candidate directories.

    Tries (in order):
    1. The path as-is (works for absolute paths or paths relative to CWD).
    2. Relative to the config directory (if provided).
    3. Relative to ``data/examples/``.
    4. Relative to any ``data/*/`` subdirectory (sorted for determinism).

    Returns the first existing candidate, or the original path if none found
    (so that the subsequent open/xr.open_dataset call raises a clear error).
    """
    p = Path(filepath_str)
    if p.exists():
        return p
    search_dirs: list[Path] = []
    if config_dir:
        search_dirs.append(Path(config_dir))
    search_dirs.append(Path("data/examples"))
    search_dirs.extend(Path(d) for d in sorted(_glob.glob("data/*/")))
    for base in search_dirs:
        candidate = base / p
        if candidate.exists():
            return candidate
    return p  # fall through — caller will raise FileNotFoundError


def _parse_floats(value: str) -> list[float]:
    """Parse a semicolon- or comma-separated string into a list of floats."""
    return [float(x.strip()) for x in re.split(r"[;,]", value) if x.strip()]


def _strip_name(name: str) -> str:
    """Strip underscores and hyphens from a species name."""
    return name.replace("_", "").replace("-", "")


@dataclass
class BackgroundSpeciesInfo:
    """Typed parameters for one background species."""

    name: str
    """Display name (underscores/hyphens stripped)."""

    species_index: int
    """0-based index among background species (i.e. sorted position)."""

    file_index: int
    """File index as used in config keys, e.g. 10 for sp10."""

    n_class: int
    """Number of size/age classes."""

    lengths: list[float]
    """Mid-length (cm) of each size class."""

    trophic_levels: list[float]
    """Trophic level of each size class."""

    ages_dt: list[int]
    """Age of each class in time-steps: int(age_years) * n_dt_per_year."""

    condition_factor: float
    """Length-weight condition factor (a in W = a*L^b)."""

    allometric_power: float
    """Allometric power (b in W = a*L^b)."""

    size_ratio_min: list[float]
    """Minimum predator/prey size ratio(s) per feeding stage."""

    size_ratio_max: list[float]
    """Maximum predator/prey size ratio(s) per feeding stage."""

    ingestion_rate: float
    """Maximum ingestion rate (per year)."""

    multiplier: float
    """Biomass forcing multiplier (default 1.0)."""

    offset: float
    """Biomass forcing offset in tonnes (default 0.0)."""

    forcing_nsteps_year: int
    """Number of forcing time steps per year for biomass interpolation."""

    proportions: list[float]
    """Fraction of total biomass in each size class (sums to 1.0)."""

    proportion_ts: "NDArray[np.float64] | None" = field(default=None)
    """Time-varying proportion overrides (shape: n_steps x n_class). None = use proportions."""


def parse_background_species(
    cfg: dict[str, str],
    n_focal: int,
    n_dt_per_year: int,
) -> list[BackgroundSpeciesInfo]:
    """Parse all background species from a flat OSMOSE config dict.

    Parameters
    ----------
    cfg:
        Flat string key-value OSMOSE config.
    n_focal:
        Number of focal species (used only for logging/validation).
    n_dt_per_year:
        Simulation time steps per year.

    Returns
    -------
    list[BackgroundSpeciesInfo]
        Background species sorted by file index, 0-based species_index assigned.
    """
    # Scan for species.type.sp* = "background"
    file_indices: list[int] = []
    for key, val in cfg.items():
        m = re.fullmatch(r"species\.type\.sp(\d+)", key)
        if m and val.strip().lower() == "background":
            file_indices.append(int(m.group(1)))

    if not file_indices:
        return []

    # Sort numerically
    file_indices.sort()

    # Validate count against simulation.nbackground (warning only)
    expected = cfg.get("simulation.nbackground")
    if expected is not None:
        n_expected = int(expected)
        if len(file_indices) != n_expected:
            logger.warning(
                "simulation.nbackground=%d but found %d background species in config",
                n_expected,
                len(file_indices),
            )

    # Global forcing nsteps fallback
    global_nsteps = int(cfg.get("simulation.nsteps.year", str(n_dt_per_year)))

    result: list[BackgroundSpeciesInfo] = []
    for species_index, file_idx in enumerate(file_indices):
        i = file_idx  # shorthand

        name_raw = cfg.get(f"species.name.sp{i}", f"background_{i}")
        name = _strip_name(name_raw.strip())

        n_class = int(cfg.get(f"species.nclass.sp{i}", "1"))

        lengths_raw = cfg.get(f"species.length.sp{i}", "0")
        lengths = _parse_floats(lengths_raw)

        proportions_raw = cfg.get(f"species.size.proportion.sp{i}", "1")
        proportions = _parse_floats(proportions_raw)

        trophic_raw = cfg.get(f"species.trophic.level.sp{i}", "2")
        trophic_levels = _parse_floats(trophic_raw)

        ages_raw = cfg.get(f"species.age.sp{i}", "0")
        age_floats = _parse_floats(ages_raw)
        # Truncate FIRST, then multiply (not round)
        ages_dt = [int(age) * n_dt_per_year for age in age_floats]

        condition_factor = float(cfg.get(f"species.length2weight.condition.factor.sp{i}", "0.01"))
        allometric_power = float(cfg.get(f"species.length2weight.allometric.power.sp{i}", "3.0"))

        size_ratio_max = _parse_floats(cfg.get(f"predation.predprey.sizeratio.max.sp{i}", "3.5"))
        size_ratio_min = _parse_floats(cfg.get(f"predation.predprey.sizeratio.min.sp{i}", "1.0"))
        ingestion_rate = float(cfg.get(f"predation.ingestion.rate.max.sp{i}", "3.5"))

        multiplier = float(cfg.get(f"species.biomass.multiplier.sp{i}", "1.0"))
        offset = float(cfg.get(f"species.biomass.offset.sp{i}", "0.0"))

        # Per-species override, then global fallback, then n_dt_per_year
        forcing_nsteps_year = int(cfg.get(f"species.biomass.nsteps.year.sp{i}", str(global_nsteps)))

        result.append(
            BackgroundSpeciesInfo(
                name=name,
                species_index=species_index,
                file_index=file_idx,
                n_class=n_class,
                lengths=lengths,
                trophic_levels=trophic_levels,
                ages_dt=ages_dt,
                condition_factor=condition_factor,
                allometric_power=allometric_power,
                size_ratio_min=size_ratio_min,
                size_ratio_max=size_ratio_max,
                ingestion_rate=ingestion_rate,
                multiplier=multiplier,
                offset=offset,
                forcing_nsteps_year=forcing_nsteps_year,
                proportions=proportions,
            )
        )

    return result


class BackgroundState:
    """Manages background species forcing and generates SchoolState instances.

    For uniform forcing (``species.biomass.total.sp{N}`` present), biomass is
    distributed evenly across all ocean cells.  Species without a total biomass
    key are flagged as NetCDF-mode (per_cell_biomass == -1.0) and handled in
    Task 3.
    """

    def __init__(
        self,
        config: dict[str, str],
        grid: "Grid",  # noqa: F821 — Grid imported lazily to avoid circular deps
        engine_config: "EngineConfig",  # noqa: F821
    ) -> None:
        n_focal = engine_config.n_species
        n_dt_per_year = engine_config.n_dt_per_year
        self._n_dt_per_year: int = n_dt_per_year

        self._species: list[BackgroundSpeciesInfo] = parse_background_species(
            config, n_focal=n_focal, n_dt_per_year=n_dt_per_year
        )
        self._n_focal = n_focal

        # Pre-compute ocean cell coordinates (y, x) from grid mask
        ys, xs = np.where(grid.ocean_mask)
        self._ocean_ys: NDArray[np.int32] = ys.astype(np.int32)
        self._ocean_xs: NDArray[np.int32] = xs.astype(np.int32)
        n_ocean = len(self._ocean_ys)

        # Pre-compute per-class weights and per-cell biomass for each species
        self._weights: list[NDArray[np.float64]] = []
        self._per_cell_biomass: list[float] = []

        # NetCDF forcing data: one 3D array (n_forcing_steps, ny, nx) per species,
        # or None if the species uses uniform mode.
        self._forcing_data: list[NDArray[np.float64] | None] = []
        self._forcing_nsteps: list[int] = []

        # Load proportion time-series CSVs when referenced in config
        for sp in self._species:
            prop_file_key = f"species.size.proportion.file.sp{sp.file_index}"
            if prop_file_key in config and not sp.proportion_ts:
                import pandas as pd

                csv_path = _resolve_path(
                    config[prop_file_key],
                    config.get("_osmose.config.dir", ""),
                )
                df = pd.read_csv(csv_path, sep=";")
                sp.proportion_ts = df.values.astype(np.float64)

        for sp in self._species:
            # w[cls] = condition_factor * length[cls]^allometric_power, convert grams to tonnes
            weights = np.array(
                [sp.condition_factor * (ln**sp.allometric_power) * 1e-6 for ln in sp.lengths],
                dtype=np.float64,
            )
            self._weights.append(weights)

            # Determine per-cell biomass
            total_key = f"species.biomass.total.sp{sp.file_index}"
            if total_key in config:
                total_biomass = float(config[total_key])
                per_cell = sp.multiplier * (total_biomass / n_ocean + sp.offset)
                self._per_cell_biomass.append(per_cell)
                self._forcing_data.append(None)
                self._forcing_nsteps.append(0)
            else:
                per_cell = -1.0  # NetCDF mode
                self._per_cell_biomass.append(per_cell)

                # Load NetCDF forcing file
                nc_path_str = config.get(f"species.file.sp{sp.file_index}")
                if nc_path_str is not None:
                    nc_path = _resolve_path(nc_path_str, config.get("_osmose.config.dir", ""))
                    with xr.open_dataset(nc_path) as ds:
                        # Try species name as variable, fall back to first variable
                        stripped = sp.name
                        if stripped in ds:
                            da = ds[stripped]
                        else:
                            first_var = list(ds.data_vars)[0]
                            da = ds[first_var]
                            logger.debug(
                                "NetCDF variable '%s' not found for species %s; using '%s'",
                                stripped,
                                sp.name,
                                first_var,
                            )
                        raw: NDArray[np.float64] = da.values.astype(np.float64)
                        # Apply multiplier
                        raw = raw * sp.multiplier
                    # Regrid to model grid if shapes differ
                    target_ny = int(
                        np.sum(grid.ocean_mask.any(axis=1) | ~grid.ocean_mask.any(axis=1))
                    )
                    target_ny = grid.ocean_mask.shape[0]
                    target_nx = grid.ocean_mask.shape[1]
                    if raw.shape[1] != target_ny or raw.shape[2] != target_nx:
                        raw = BackgroundState._regrid(raw, target_ny, target_nx)
                    self._forcing_data.append(raw)
                    self._forcing_nsteps.append(raw.shape[0])
                else:
                    logger.warning(
                        "Species %s is in NetCDF mode but no 'species.file.sp%d' found in config",
                        sp.name,
                        sp.file_index,
                    )
                    self._forcing_data.append(None)
                    self._forcing_nsteps.append(0)

    def get_schools(self, step: int) -> "SchoolState":  # noqa: F821
        """Return a SchoolState with one school per species per class per ocean cell.

        Parameters
        ----------
        step:
            Current simulation time step (unused for uniform forcing; reserved
            for NetCDF time-interpolation in Task 3).
        """
        from osmose.engine.state import SchoolState

        n_ocean = len(self._ocean_ys)

        if not self._species:
            return SchoolState.create(n_schools=0)

        # Build per-species-per-class school data, then concatenate
        all_species_id: list[NDArray[np.int32]] = []
        all_is_background: list[NDArray[np.bool_]] = []
        all_age_dt: list[NDArray[np.int32]] = []
        all_first_feeding_age_dt: list[NDArray[np.int32]] = []
        all_biomass: list[NDArray[np.float64]] = []
        all_weight: list[NDArray[np.float64]] = []
        all_abundance: list[NDArray[np.float64]] = []
        all_trophic_level: list[NDArray[np.float64]] = []
        all_cell_x: list[NDArray[np.int32]] = []
        all_cell_y: list[NDArray[np.int32]] = []

        for bkg_idx, sp in enumerate(self._species):
            species_id = self._n_focal + bkg_idx
            per_cell = self._per_cell_biomass[bkg_idx]
            weights = self._weights[bkg_idx]
            forcing = self._forcing_data[bkg_idx]

            # Determine per-cell biomass array for this species at this step
            if forcing is not None:
                # NetCDF mode: map simulation step to forcing index.
                # step_in_year uses the simulation n_dt_per_year; ratio then
                # maps into the declared forcing resolution (forcing_nsteps_year).
                n_forcing = self._forcing_nsteps[bkg_idx]
                sim_n_dt = self._n_dt_per_year
                step_in_year = step % sim_n_dt
                # Java mapping: use declared forcing_nsteps_year as temporal resolution
                n_declared = sp.forcing_nsteps_year
                forcing_idx = min(
                    int(step_in_year * n_declared / sim_n_dt),
                    n_forcing - 1,
                )
                # Extract spatial slice at ocean cells: shape (n_ocean,)
                ocean_y = self._ocean_ys
                ocean_x = self._ocean_xs
                sp_cell_biomass_base: NDArray[np.float64] = forcing[forcing_idx][ocean_y, ocean_x]
                netcdf_mode = True
            else:
                netcdf_mode = False

            for cls_idx in range(sp.n_class):
                # Use time-varying proportion when available
                if sp.proportion_ts is not None and len(sp.proportion_ts) > 0:
                    ts_idx = min(step, len(sp.proportion_ts) - 1)
                    prop = float(sp.proportion_ts[ts_idx, cls_idx])
                else:
                    prop = sp.proportions[cls_idx]
                cls_weight = weights[cls_idx]
                cls_tl = sp.trophic_levels[cls_idx] if cls_idx < len(sp.trophic_levels) else 2.0
                cls_age_dt = sp.ages_dt[cls_idx] if cls_idx < len(sp.ages_dt) else 0

                if netcdf_mode:
                    # Per-cell array from NetCDF
                    cell_biomass: NDArray[np.float64] = sp_cell_biomass_base * prop
                else:
                    # Uniform: scalar spread evenly across all ocean cells
                    scalar_biomass = per_cell * prop
                    cell_biomass = np.full(n_ocean, scalar_biomass, dtype=np.float64)

                cls_weight_arr = np.full(n_ocean, cls_weight, dtype=np.float64)
                if cls_weight > 0.0:
                    cls_abundance_arr: NDArray[np.float64] = cell_biomass / cls_weight
                else:
                    cls_abundance_arr = np.zeros(n_ocean, dtype=np.float64)

                all_species_id.append(np.full(n_ocean, species_id, dtype=np.int32))
                all_is_background.append(np.ones(n_ocean, dtype=np.bool_))
                # CRITICAL: Java convention — -1 means always eligible to predate
                all_first_feeding_age_dt.append(np.full(n_ocean, -1, dtype=np.int32))
                all_age_dt.append(np.full(n_ocean, cls_age_dt, dtype=np.int32))
                all_biomass.append(cell_biomass)
                all_weight.append(cls_weight_arr)
                all_abundance.append(cls_abundance_arr)
                all_trophic_level.append(np.full(n_ocean, cls_tl, dtype=np.float64))
                all_cell_x.append(self._ocean_xs.copy())
                all_cell_y.append(self._ocean_ys.copy())

        n_total = sum(len(a) for a in all_species_id)
        state = SchoolState.create(n_schools=n_total)
        return state.replace(
            species_id=np.concatenate(all_species_id),
            is_background=np.concatenate(all_is_background),
            first_feeding_age_dt=np.concatenate(all_first_feeding_age_dt),
            age_dt=np.concatenate(all_age_dt),
            biomass=np.concatenate(all_biomass),
            weight=np.concatenate(all_weight),
            abundance=np.concatenate(all_abundance),
            trophic_level=np.concatenate(all_trophic_level),
            cell_x=np.concatenate(all_cell_x),
            cell_y=np.concatenate(all_cell_y),
        )

    @staticmethod
    def _regrid(data: NDArray[np.float64], target_ny: int, target_nx: int) -> NDArray[np.float64]:
        """Nearest-neighbor regrid a 3D array (time, lat, lon) to (time, target_ny, target_nx)."""
        fy = data.shape[1] / target_ny
        fx = data.shape[2] / target_nx
        rows = np.clip((np.arange(target_ny) * fy).astype(int), 0, data.shape[1] - 1)
        cols = np.clip((np.arange(target_nx) * fx).astype(int), 0, data.shape[2] - 1)
        return data[:, rows][:, :, cols]
