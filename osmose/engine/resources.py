"""Resource species (LTL) for the OSMOSE Python engine.

Resources are plankton/detritus that serve as prey. Their biomass comes
from external forcing (NetCDF or uniform) and regenerates each timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from osmose.engine.grid import Grid
from osmose.engine.path_resolution import resolve_data_path


@dataclass
class ResourceSpeciesInfo:
    """Metadata for one resource species."""

    name: str
    size_min: float  # cm
    size_max: float  # cm
    trophic_level: float
    accessibility: float  # fraction available per timestep [0, 0.99]
    multiplier: float = 1.0  # biomass scaling multiplier
    offset: float = 0.0  # biomass offset (for uniform distribution)
    accessibility_ts: NDArray[np.float64] | None = None  # time-varying accessibility


class ResourceState:
    """Container for LTL resource biomass per grid cell.

    Loads biomass from NetCDF forcing or distributes a uniform total.
    Biomass resets from forcing each timestep (resources regenerate).
    """

    def __init__(self, config: dict[str, str], grid: Grid) -> None:
        self.config = config
        self.grid = grid
        self.n_resources = int(config.get("simulation.nresource", "0"))
        self.species: list[ResourceSpeciesInfo] = []
        # Per-cell biomass: shape (n_resources, n_cells) where n_cells = ny * nx
        self.biomass: NDArray[np.float64] = np.zeros(
            (max(1, self.n_resources), grid.ny * grid.nx), dtype=np.float64
        )
        self._forcing_data: xr.Dataset | None = None
        self._n_forcing_steps: int = 0
        self._forcing_var_names: list[str] = []
        self._uniform_biomass: NDArray[np.float64] = np.zeros(
            max(1, self.n_resources), dtype=np.float64
        )

        if self.n_resources > 0:
            self._load_config()

    def _load_config(self) -> None:
        """Parse resource species metadata and load forcing data."""
        cfg = self.config

        # Try the legacy ltl.*.rsc{i} pattern first
        has_ltl_keys = any(k.startswith("ltl.name.rsc") for k in cfg)

        if has_ltl_keys:
            self._load_config_ltl()
        else:
            # Fallback: species.type.sp{N} = resource pattern (EEC-style)
            self._load_config_species_type()

    def _load_config_ltl(self) -> None:
        """Load resource config from legacy ltl.*.rsc{i} keys."""
        cfg = self.config

        for i in range(self.n_resources):
            name = cfg.get(f"ltl.name.rsc{i}", f"Resource{i}")
            raw_access = float(cfg.get(f"ltl.accessibility2fish.rsc{i}", "0.01"))
            self.species.append(
                ResourceSpeciesInfo(
                    name=name,
                    size_min=float(cfg.get(f"ltl.size.min.rsc{i}", "0.001")),
                    size_max=float(cfg.get(f"ltl.size.max.rsc{i}", "0.01")),
                    trophic_level=float(cfg.get(f"ltl.tl.rsc{i}", "1.0")),
                    accessibility=min(raw_access, 0.99),  # Cap at 0.99
                )
            )
            self._forcing_var_names.append(name)

        # Load NetCDF forcing
        self._load_netcdf(cfg.get("ltl.netcdf.file", ""))

        # Check for uniform biomass fallback
        for i in range(self.n_resources):
            total = cfg.get(f"ltl.biomass.total.rsc{i}", "")
            if total:
                n_ocean = self.grid.n_ocean_cells
                self._uniform_biomass[i] = float(total) / max(1, n_ocean)

    def _load_config_species_type(self) -> None:
        """Load resource config from species.type.sp{N} = resource keys (EEC-style)."""
        cfg = self.config

        # Discover resource species file indices
        resource_indices: list[int] = []
        for key, val in cfg.items():
            if key.startswith("species.type.sp") and val.strip().lower() == "resource":
                fi = int(key.rsplit("sp", 1)[1])
                resource_indices.append(fi)
        resource_indices.sort()

        if not resource_indices:
            return

        # Override n_resources if discovery found more/fewer
        self.n_resources = len(resource_indices)
        self.biomass = np.zeros((self.n_resources, self.grid.ny * self.grid.nx), dtype=np.float64)
        self._uniform_biomass = np.zeros(self.n_resources, dtype=np.float64)

        nc_file = ""
        for i, fi in enumerate(resource_indices):
            name = cfg.get(f"species.name.sp{fi}", f"Resource{i}")
            raw_access = float(cfg.get(f"species.accessibility2fish.sp{fi}", "0.01"))
            # Cap accessibility at 0.99 (Fix 1.6)
            capped_access = min(raw_access, 0.99)
            multiplier = float(cfg.get(f"species.multiplier.sp{fi}", "1.0"))
            offset = float(cfg.get(f"species.offset.sp{fi}", "0.0"))

            # Time-varying accessibility (Fix 1.5)
            access_ts: NDArray[np.float64] | None = None
            access_file = cfg.get(f"species.accessibility2fish.file.sp{fi}", "")
            if access_file:
                access_path = self._resolve_data_file(access_file)
                if access_path is not None:
                    import pandas as _pd

                    df = _pd.read_csv(access_path, sep=";")
                    # Cap all values at 0.99
                    access_ts = np.minimum(df.iloc[:, -1].values.astype(np.float64), 0.99)

            self.species.append(
                ResourceSpeciesInfo(
                    name=name,
                    size_min=float(cfg.get(f"species.size.min.sp{fi}", "0.001")),
                    size_max=float(cfg.get(f"species.size.max.sp{fi}", "0.01")),
                    trophic_level=float(cfg.get(f"species.trophic.level.sp{fi}", "1.0")),
                    accessibility=capped_access,
                    multiplier=multiplier,
                    offset=offset,
                    accessibility_ts=access_ts,
                )
            )
            self._forcing_var_names.append(name)

            # Per-species NetCDF file (EEC: all point to the same file)
            if not nc_file:
                nc_file = cfg.get(f"species.file.sp{fi}", "")

        # Temporal resolution from config
        self._n_forcing_steps = int(
            cfg.get("species.biomass.nsteps.year", str(self._n_forcing_steps))
        )

        # Load NetCDF forcing
        self._load_netcdf(nc_file)

    def _resolve_data_file(self, file_key: str) -> Path | None:
        """Resolve a data file path against multiple search directories."""
        config_dir = self.config.get("_osmose.config.dir", "")
        return resolve_data_path(file_key, config_dir=config_dir)

    def _load_netcdf(self, nc_file: str) -> None:
        """Load a NetCDF forcing file, trying multiple search paths."""
        if not nc_file:
            return
        config_dir = self.config.get("_osmose.config.dir", "")
        path = resolve_data_path(nc_file, config_dir=config_dir)
        if path is not None:
            self._forcing_data = xr.open_dataset(path)
            first_var = list(self._forcing_data.data_vars)[0]
            self._n_forcing_steps = self._forcing_data[first_var].shape[0]

    def update(self, step: int) -> None:
        """Load resource biomass for the given simulation timestep.

        Resources regenerate from forcing each timestep (predation effects
        are temporary -- biomass resets from the forcing data).
        """
        if self.n_resources == 0:
            return

        n_dt_per_year = int(self.config.get("simulation.time.ndtperyear", "24"))
        grid = self.grid

        for i in range(self.n_resources):
            rsc = self.species[i]

            # Determine current accessibility: time-varying or constant
            if rsc.accessibility_ts is not None and len(rsc.accessibility_ts) > 0:
                ts_idx = step % len(rsc.accessibility_ts)
                access = float(rsc.accessibility_ts[ts_idx])
            else:
                access = rsc.accessibility

            if self._forcing_data is not None and rsc.name in self._forcing_data:
                # Map simulation timestep to forcing timestep
                # Forcing has _n_forcing_steps per year, simulation has n_dt_per_year
                step_in_year = step % n_dt_per_year
                forcing_idx = int(step_in_year * self._n_forcing_steps / n_dt_per_year)
                forcing_idx = min(forcing_idx, self._n_forcing_steps - 1)

                # Get spatial data for this timestep
                data = self._forcing_data[rsc.name].isel(time=forcing_idx).values
                # data shape might be (lat_forcing, lon_forcing) which may differ from grid

                # Regrid to model grid using nearest-neighbor
                biomass_2d = self._regrid_to_model(data)

                # Apply multiplier and accessibility coefficient, then flatten
                cell_biomass = biomass_2d.flatten() * rsc.multiplier * access
                self.biomass[i, : len(cell_biomass)] = cell_biomass[: grid.ny * grid.nx]

            elif self._uniform_biomass[i] > 0:
                # Uniform distribution: multiplier * (per_cell + offset) * accessibility
                per_cell = rsc.multiplier * (self._uniform_biomass[i] + rsc.offset) * access
                self.biomass[i, :] = per_cell
            else:
                self.biomass[i, :] = 0.0

    def _regrid_to_model(self, data: NDArray) -> NDArray:
        """Regrid forcing data to model grid via nearest-neighbor index mapping."""
        target = (self.grid.ny, self.grid.nx)
        if data.shape == target:
            return data.astype(np.float64)

        # Simple nearest-neighbor via numpy index arrays
        fy = data.shape[0] / target[0]
        fx = data.shape[1] / target[1]
        rows = (np.arange(target[0]) * fy).astype(int)
        cols = (np.arange(target[1]) * fx).astype(int)
        rows = np.clip(rows, 0, data.shape[0] - 1)
        cols = np.clip(cols, 0, data.shape[1] - 1)
        return data[np.ix_(rows, cols)].astype(np.float64)

    def get_cell_biomass(self, resource_idx: int, cell_y: int, cell_x: int) -> float:
        """Get biomass for a specific resource at a specific cell."""
        if resource_idx >= self.n_resources:
            return 0.0
        cell_id = cell_y * self.grid.nx + cell_x
        if cell_id >= self.biomass.shape[1]:
            return 0.0
        return float(self.biomass[resource_idx, cell_id])

    def close(self) -> None:
        """Close any open NetCDF datasets."""
        if self._forcing_data is not None:
            self._forcing_data.close()
            self._forcing_data = None
