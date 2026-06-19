from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    nlon: int
    nlat: int
    upleft_lat: float
    upleft_lon: float
    lowright_lat: float
    lowright_lon: float

    @classmethod
    def from_config(cls, cfg: dict[str, str]) -> "GridSpec":
        return cls(
            nlon=int(float(cfg["grid.nlon"])),
            nlat=int(float(cfg["grid.nlat"])),
            upleft_lat=float(cfg["grid.upleft.lat"]),
            upleft_lon=float(cfg["grid.upleft.lon"]),
            lowright_lat=float(cfg["grid.lowright.lat"]),
            lowright_lon=float(cfg["grid.lowright.lon"]),
        )

    @property
    def dx(self) -> float:
        return (self.lowright_lon - self.upleft_lon) / self.nlon

    @property
    def dy(self) -> float:
        return (self.upleft_lat - self.lowright_lat) / self.nlat

    def cell_polygon(self, row: int, col: int) -> list[list[float]]:
        lo0 = self.upleft_lon + col * self.dx
        la0 = self.upleft_lat - row * self.dy
        lo1, la1 = lo0 + self.dx, la0 - self.dy
        return [[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1]]

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        return (self.upleft_lat - (row + 0.5) * self.dy, self.upleft_lon + (col + 0.5) * self.dx)

    def cell_polygons(self) -> np.ndarray:
        """Vectorized corners for ALL cells, shape (nlat, nlon, 4, 2) in [UL,UR,LR,LL] / [lon,lat]."""
        cols = np.arange(self.nlon)
        rows = np.arange(self.nlat)
        lo0 = self.upleft_lon + cols * self.dx
        la0 = self.upleft_lat - rows * self.dy
        lo0g, la0g = np.meshgrid(lo0, la0)
        lo1g, la1g = lo0g + self.dx, la0g - self.dy
        return np.stack(
            [
                np.stack([lo0g, la0g], -1),
                np.stack([lo1g, la0g], -1),
                np.stack([lo1g, la1g], -1),
                np.stack([lo0g, la1g], -1),
            ],
            axis=2,
        )
