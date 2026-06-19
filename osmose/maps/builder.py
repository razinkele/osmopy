from __future__ import annotations

from collections.abc import Iterable
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


def _point_in_ring(px: float, py: float, ring: list[list[float]]) -> bool:
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _open_ring(ring: list[list[float]]) -> list[list[float]]:
    return ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring


def rasterize_polygon(grid: GridSpec, polygon_lonlat, mask=None, *, mask_edit: bool = False):
    ring = _open_ring([list(p) for p in polygon_lonlat])
    out: list[tuple[int, int]] = []
    for r in range(grid.nlat):
        for c in range(grid.nlon):
            if mask is not None and not mask_edit and mask[r, c] == -99:
                continue
            lat, lon = grid.cell_center(r, c)
            if _point_in_ring(lon, lat, ring):
                out.append((r, c))
    return out


def lonlat_to_cell(grid: GridSpec, lon: float, lat: float):
    c = int((lon - grid.upleft_lon) / grid.dx)
    r = int((grid.upleft_lat - lat) / grid.dy)
    if 0 <= r < grid.nlat and 0 <= c < grid.nlon:
        return (r, c)
    return None


class MapGrid:
    def __init__(self, array: np.ndarray):
        self._a = array

    @classmethod
    def blank(cls, grid: GridSpec, base_mask: np.ndarray | None = None) -> "MapGrid":
        a = np.zeros((grid.nlat, grid.nlon), dtype=float)
        if base_mask is not None:
            a[base_mask == -99] = -99
        return cls(a)

    @property
    def array(self) -> np.ndarray:
        return self._a

    def apply_cells(self, cells: Iterable[tuple[int, int]], value: float) -> None:
        for r, c in cells:
            self._a[r, c] = value

    def apply_polygon(
        self, grid: GridSpec, polygon_lonlat, value: float, *, mask_edit: bool = False
    ) -> None:
        self.apply_cells(
            rasterize_polygon(grid, polygon_lonlat, self._a, mask_edit=mask_edit), value
        )

    def erase(self, cells: Iterable[tuple[int, int]]) -> None:
        self.apply_cells(cells, 0.0)

    def set_mask(self, cells: Iterable[tuple[int, int]], masked: bool) -> None:
        self.apply_cells(cells, -99.0 if masked else 0.0)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.10g}"


def to_csv_text(mg: "MapGrid") -> str:
    south_first = np.flipud(mg.array)
    lines = []
    for row in south_first:
        lines.append(";".join(_fmt(v) for v in row))
    return "\n".join(lines) + "\n"


def from_csv_text(text: str, grid: GridSpec) -> "MapGrid":
    rows = [ln for ln in text.splitlines() if ln.strip()]
    data = [[float(x) for x in ln.split(";")] for ln in rows]
    if len(data) != grid.nlat or any(len(r) != grid.nlon for r in data):
        raise ValueError(
            f"CSV dims {len(data)}x{len(data[0]) if data else 0} != grid {grid.nlat}x{grid.nlon}"
        )
    return MapGrid(np.flipud(np.array(data, dtype=float)))
