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
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
    ) -> Grid:
        """Load grid from a NetCDF file."""
        with xr.open_dataset(path) as ds:
            mask_data = ds[mask_var].values
            lat = ds[lat_dim].values.astype(np.float64)
            lon = ds[lon_dim].values.astype(np.float64)
        ny, nx = mask_data.shape
        ocean_mask = mask_data > 0
        return cls(ny=ny, nx=nx, ocean_mask=ocean_mask, lat=lat, lon=lon)

    @classmethod
    def from_dimensions(cls, ny: int, nx: int) -> Grid:
        """Create a simple rectangular grid with all ocean cells."""
        ocean_mask = np.ones((ny, nx), dtype=np.bool_)
        return cls(ny=ny, nx=nx, ocean_mask=ocean_mask)
