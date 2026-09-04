"""Generic NetCDF/constant physical data loader for temperature and oxygen forcing."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


class PhysicalData:
    """Physical forcing data (temperature or oxygen).

    Two modes:
    - Constant: single value applied everywhere.
    - NetCDF: (time, y, x) or (time, z, y, x) array with periodic time cycling. This
      class draws no distinction between temperature and oxygen -- neither does Java's
      ForcingFile.readVariable, which branches on shape length alone for any variable
      name. `get_grid(step, layer=...)` plumbs the z-axis when the loaded file has one.
      In Java, both TempFunction.java:162 and OxygenFunction.java:130-133 resolve that
      layer per-species (`Species.getDepthLayer()`, from `species.zlayer.sp{idx}`)
      through this same `getValue(int, Cell)` mechanism. This Python port currently only
      wires `species.zlayer.sp{idx}` into the temperature branch of `_bioen_step`
      (osmose/engine/simulate.py); a 4-D oxygen file is read at layer 0 regardless of
      zlayer. `_load_oxygen_data` raises if a 4-D oxygen file is combined with any
      nonzero `species.zlayer.sp{idx}` so that gap can't produce a silent wrong answer.
    """

    def __init__(
        self,
        data: NDArray[np.float64] | None,
        constant: float | None,
        nsteps_year: int,
    ) -> None:
        self._data = data
        self._constant = constant
        self._nsteps_year = nsteps_year
        self.rv_ref: float | None = None

    @classmethod
    def from_constant(cls, value: float, factor: float = 1.0, offset: float = 0.0) -> PhysicalData:
        """Create constant-mode physical data: factor * (value + offset)."""
        return cls(data=None, constant=factor * (value + offset), nsteps_year=1)

    @classmethod
    def from_netcdf(
        cls,
        path: Path,
        varname: str = "temp",
        nsteps_year: int = 12,
        factor: float = 1.0,
        offset: float = 0.0,
    ) -> PhysicalData:
        """Load from NetCDF file. Accepts (time, y, x) or (time, z, y, x) -- the
        latter enables per-species depth sampling via `get_grid(step, layer=...)`."""
        from osmose.engine._netcdf import open_dataset_safe

        ds = open_dataset_safe(path)
        raw = ds[varname].values
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        if raw.ndim not in (3, 4):
            raise ValueError(
                f"{path}:{varname} must be (time,y,x) or (time,z,y,x); got shape {raw.shape}"
            )
        data = factor * (raw.astype(np.float64) + offset)
        return cls(data=data, constant=None, nsteps_year=nsteps_year)

    @classmethod
    def from_netcdf_field(cls, path: Path, varname: str, rv_ref: float) -> PhysicalData:
        """Load a per-cell field NetCDF (no factor/offset), carrying an rv_ref scalar."""
        from osmose.engine._netcdf import open_dataset_safe

        ds = open_dataset_safe(path)
        raw = ds[varname].values
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        obj = cls(data=raw.astype(np.float64), constant=None, nsteps_year=raw.shape[0])
        obj.rv_ref = rv_ref
        return obj

    @property
    def is_constant(self) -> bool:
        return self._constant is not None

    @property
    def n_layers(self) -> int:
        """Number of depth layers: 1 for constant mode or 3-D (time,y,x) data."""
        return 1 if self._data is None or self._data.ndim == 3 else int(self._data.shape[1])

    def _frame(self, step: int, layer: int) -> NDArray[np.float64]:
        """Return the (ny, nx) grid for `step` (cycled modulo the loaded frame count)
        at `layer`. `step % frame_count` -- NOT `_nsteps_year`, which is metadata only
        (see `_load_temperature_data`/`_load_oxygen_data` in simulate.py)."""
        assert self._data is not None
        t_idx = step % self._data.shape[0]
        if self._data.ndim == 3:
            if layer != 0:
                raise IndexError(f"layer {layer} requested from a single-layer field")
            return self._data[t_idx]
        if not 0 <= layer < self._data.shape[1]:
            raise IndexError(f"layer {layer} out of range (n_layers={self._data.shape[1]})")
        return self._data[t_idx, layer]

    def get_value(self, step: int, cell_y: int, cell_x: int, layer: int = 0) -> float:
        """Get value at a specific cell and timestep (and depth layer, if 4-D)."""
        if self._constant is not None:
            return self._constant
        return float(self._frame(step, layer)[cell_y, cell_x])

    def get_scalar(self) -> float:
        """Get the constant value. Raises ValueError if not constant mode."""
        if self._constant is None:
            raise ValueError("PhysicalData is not in constant mode")
        return self._constant

    def get_grid(self, step: int, layer: int = 0) -> NDArray[np.float64]:
        """Return full (ny, nx) grid for a timestep (and depth layer, if 4-D)."""
        if self._constant is not None:
            raise ValueError("Constant PhysicalData has no spatial grid")
        return self._frame(step, layer)
