"""Generic NetCDF/constant physical data loader for temperature and oxygen forcing."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


class PhysicalData:
    """Physical forcing data (temperature or oxygen).

    Two modes:
    - Constant: single value applied everywhere.
    - NetCDF: 3D array (time, y, x) with periodic cycling.
    """

    def __init__(self, data: NDArray[np.float64] | None, constant: float | None, nsteps_year: int) -> None:
        self._data = data
        self._constant = constant
        self._nsteps_year = nsteps_year

    @classmethod
    def from_constant(cls, value: float, factor: float = 1.0, offset: float = 0.0) -> PhysicalData:
        """Create constant-mode physical data: factor * (value + offset)."""
        return cls(data=None, constant=factor * (value + offset), nsteps_year=1)

    @classmethod
    def from_netcdf(cls, path: Path, varname: str = "temp", nsteps_year: int = 12,
                    factor: float = 1.0, offset: float = 0.0) -> PhysicalData:
        """Load from NetCDF file."""
        import xarray as xr
        ds = xr.open_dataset(path)
        raw = ds[varname].values
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        data = factor * (raw.astype(np.float64) + offset)
        return cls(data=data, constant=None, nsteps_year=nsteps_year)

    @property
    def is_constant(self) -> bool:
        return self._constant is not None

    def get_value(self, step: int, cell_y: int, cell_x: int) -> float:
        """Get value at a specific cell and timestep."""
        if self._constant is not None:
            return self._constant
        assert self._data is not None
        t_idx = step % self._data.shape[0]
        return float(self._data[t_idx, cell_y, cell_x])

    def get_scalar(self) -> float:
        """Get the constant value. Raises ValueError if not constant mode."""
        if self._constant is None:
            raise ValueError("PhysicalData is not in constant mode")
        return self._constant

    def get_grid(self, step: int) -> NDArray[np.float64]:
        """Return full (ny, nx) grid for a timestep."""
        if self._constant is not None:
            raise ValueError("Constant PhysicalData has no spatial grid")
        assert self._data is not None
        t_idx = step % self._data.shape[0]
        return self._data[t_idx]
