"""Grid: spatial grid representation for the engine simulation.

Loads grid topology from NetCDF or creates simple rectangular grids.
Provides cell indexing, ocean/land masking, and cell adjacency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class Grid:
    """2D spatial grid for OSMOSE simulation.

    Cells are indexed in row-major order: cell_id = y * nx + x.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        ocean_mask: NDArray[np.bool_],
        lat: NDArray[np.float64] | None = None,
        lon: NDArray[np.float64] | None = None,
    ) -> None:
        self.ny = ny
        self.nx = nx
        self.ocean_mask = ocean_mask
        self.lat = lat
        self.lon = lon
        if ny < 1 or nx < 1:
            raise ValueError(f"Grid dimensions must be >= 1, got ny={ny}, nx={nx}")
        if ocean_mask.shape != (ny, nx):
            raise ValueError(
                f"ocean_mask shape {ocean_mask.shape} does not match (ny={ny}, nx={nx})"
            )
        if lat is not None and lat.shape != (ny,):
            raise ValueError(f"lat shape {lat.shape} does not match ny={ny}")
        if lon is not None and lon.shape != (nx,):
            raise ValueError(f"lon shape {lon.shape} does not match nx={nx}")

    @property
    def n_cells(self) -> int:
        return self.ny * self.nx

    @property
    def n_ocean_cells(self) -> int:
        return int(self.ocean_mask.sum())

    def cell_to_yx(self, cell_id: int) -> tuple[int, int]:
        """Convert flat cell ID to (y, x) grid coordinates."""
        return divmod(cell_id, self.nx)

    def yx_to_cell(self, y: int, x: int) -> int:
        """Convert (y, x) grid coordinates to flat cell ID."""
        return y * self.nx + x

    def neighbors(self, y: int, x: int) -> list[tuple[int, int]]:
        """Return (y, x) coordinates of all valid neighboring cells (8-connected)."""
        result = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb_y, nb_x = y + dy, x + dx
                if 0 <= nb_y < self.ny and 0 <= nb_x < self.nx:
                    result.append((nb_y, nb_x))
        return result

    @classmethod
    def from_netcdf(
        cls,
        path: Path,
        mask_var: str = "mask",
        lat_dim: str | None = None,
        lon_dim: str | None = None,
    ) -> Grid:
        """Load grid from a NetCDF file.

        Dimension/variable names are auto-detected if not specified.
        Supports both 1D coordinate arrays and 2D lat/lon variables.
        """
        with xr.open_dataset(path) as ds:
            mask_data = ds[mask_var].values
            ny, nx = mask_data.shape

            # Auto-detect lat dimension/variable
            if lat_dim is None:
                for candidate in ["latitude", "lat", "y"]:
                    if candidate in ds.dims or candidate in ds.coords or candidate in ds:
                        lat_dim = candidate
                        break

            # Auto-detect lon dimension/variable
            if lon_dim is None:
                for candidate in ["longitude", "lon", "x"]:
                    if candidate in ds.dims or candidate in ds.coords or candidate in ds:
                        lon_dim = candidate
                        break

            lat = ds[lat_dim].values.astype(np.float64) if lat_dim and lat_dim in ds else None
            lon = ds[lon_dim].values.astype(np.float64) if lon_dim and lon_dim in ds else None

            # Handle 2D lat/lon arrays (e.g. curvilinear grids)
            if lat is not None and lat.ndim == 2:
                lat = lat[:, 0]
            if lon is not None and lon.ndim == 2:
                lon = lon[0, :]

        ocean_mask = mask_data > 0
        return cls(ny=ny, nx=nx, ocean_mask=ocean_mask, lat=lat, lon=lon)

    @classmethod
    def from_dimensions(cls, ny: int, nx: int) -> Grid:
        """Create a simple rectangular grid with all ocean cells."""
        ocean_mask = np.ones((ny, nx), dtype=np.bool_)
        return cls(ny=ny, nx=nx, ocean_mask=ocean_mask)
