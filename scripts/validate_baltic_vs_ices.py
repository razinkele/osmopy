"""Validate Baltic grid mask + movement maps against ICES BITS survey data.

Fetches haul positions (DATRAS HH records) for recent BITS Q1/Q4 surveys and
compares against the Baltic grid mask and per-species movement distributions.
Intended as a ground-truth cross-check — BITS hauls are the authoritative
spatial sample of where fish actually occur in the Baltic.

Run from repo root:
    .venv/bin/python scripts/validate_baltic_vs_ices.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import xarray as xr

ICES_MCP_PATH = Path("/home/razinka/ices-mcp-server")
if str(ICES_MCP_PATH) not in sys.path:
    sys.path.insert(0, str(ICES_MCP_PATH))

from ices.datras import get_cpue_length, get_hauls  # noqa: E402

GRID_NC = Path("data/baltic/baltic_grid.nc")
MAPS_DIR = Path("data/baltic/maps")

SURVEY_YEARS = [2021, 2022, 2023]
QUARTERS = [1, 4]

SPECIES_APHIA = {
    "cod": 126436,
    "herring": 126417,
    "sprat": 126425,
    "flounder": 127141,
}


def load_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(GRID_NC)
    return ds.latitude.values, ds.longitude.values, ds.mask.values > 0


def haul_cell(lat: float, lon: float, grid_lat: np.ndarray, grid_lon: np.ndarray) -> tuple[int, int] | None:
    lat_step = abs(grid_lat[1] - grid_lat[0])
    lon_step = abs(grid_lon[1] - grid_lon[0])
    top_edge = grid_lat[0] + lat_step / 2
    left_edge = grid_lon[0] - lon_step / 2
    r = int((top_edge - lat) / lat_step)
    c = int((lon - left_edge) / lon_step)
    if 0 <= r < len(grid_lat) and 0 <= c < len(grid_lon):
        return (r, c)
    return None


async def fetch_all_hauls() -> pd.DataFrame:
    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=180) as client:
        for year in SURVEY_YEARS:
            for q in QUARTERS:
                try:
                    hh = await get_hauls(client, survey="BITS", year=year, quarter=q)
                except Exception as exc:
                    print(f"  ! BITS {year}Q{q}: {exc}")
                    continue
                for r in hh:
                    if r.get("shoot_lat") is not None and r.get("shoot_lon") is not None:
                        rows.append(r)
                print(f"  BITS {year}Q{q}: {len(hh)} hauls")
    return pd.DataFrame(rows)


async def fetch_species_cpue(species: str, aphia_id: int) -> pd.DataFrame:
    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=180) as client:
        for year in SURVEY_YEARS:
            for q in QUARTERS:
                try:
                    data = await get_cpue_length(
                        client, survey="BITS", year=year, quarter=q,
                        mode="raw", aphia_id=aphia_id,
                    )
                except Exception as exc:
                    print(f"  ! CPUE {species} {year}Q{q}: {exc}")
                    continue
                for r in data:
                    if r.get("shoot_lat") is not None and r.get("shoot_lon") is not None:
                        rows.append(r)
                print(f"  CPUE {species} {year}Q{q}: {len(data)} records")
    return pd.DataFrame(rows)


def validate_hauls(hauls: pd.DataFrame, mask: np.ndarray,
                   grid_lat: np.ndarray, grid_lon: np.ndarray) -> None:
    print("\n=== Haul-position validation ===")
    print(f"Total hauls:              {len(hauls)}")

    cells = hauls.apply(
        lambda row: haul_cell(row["shoot_lat"], row["shoot_lon"], grid_lat, grid_lon),
        axis=1,
    )
    outside = cells.isna().sum()
    inside = cells.dropna()
    rows_cols = np.array([list(c) for c in inside])
    if len(rows_cols) == 0:
        print("  No hauls inside grid!")
        return
    in_mask = mask[rows_cols[:, 0], rows_cols[:, 1]]
    n_in_ocean = int(in_mask.sum())
    n_in_land = int((~in_mask).sum())

    print(f"  outside grid bbox:      {outside}")
    print(f"  inside grid, in ocean:  {n_in_ocean}  ({100*n_in_ocean/len(hauls):.1f}%)")
    print(f"  inside grid, on LAND:   {n_in_land}   ← mask misses real fishing ground")

    if n_in_land > 0:
        land_hits = rows_cols[~in_mask]
        by_cell = pd.Series([tuple(rc) for rc in land_hits]).value_counts()
        print("  Top mask-land cells with hauls:")
        for (r, c), n in by_cell.head(10).items():
            print(f"    (r={r}, c={c}) lat={grid_lat[r]:.2f} lon={grid_lon[c]:.2f}: {n} hauls")

    haul_cells = set(map(tuple, rows_cols[in_mask]))
    mask_cells = {(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if mask[r, c]}
    unvisited = mask_cells - haul_cells
    print(f"  ocean cells with hauls: {len(haul_cells)} / {len(mask_cells)} "
          f"({100*len(haul_cells)/len(mask_cells):.1f}%)")
    print(f"  ocean cells never sampled by BITS: {len(unvisited)} (mostly shallow/coastal)")


def validate_species(name: str, cpue: pd.DataFrame, mask: np.ndarray,
                     grid_lat: np.ndarray, grid_lon: np.ndarray) -> None:
    print(f"\n=== Species footprint: {name} ===")
    if cpue.empty:
        print("  (no data)")
        return
    cpue_col = "cpue_number_per_hour"
    positive = cpue[pd.to_numeric(cpue[cpue_col], errors="coerce").fillna(0) > 0]
    print(f"  positive-CPUE records: {len(positive)} of {len(cpue)}")

    cells = positive.apply(
        lambda r: haul_cell(r["shoot_lat"], r["shoot_lon"], grid_lat, grid_lon),
        axis=1,
    ).dropna()
    rows_cols = np.array([list(c) for c in cells])
    if len(rows_cols) == 0:
        print("  (no records inside grid)")
        return
    ices_footprint = np.zeros_like(mask, dtype=bool)
    ices_footprint[rows_cols[:, 0], rows_cols[:, 1]] = True

    # Load our movement maps for this species
    model_cells = np.zeros_like(mask, dtype=bool)
    species_files = list(MAPS_DIR.glob(f"{name}*.csv"))
    for f in species_files:
        arr = np.flipud(pd.read_csv(f, sep=";", header=None).values)
        model_cells |= arr > 0
    print(f"  model uses {int(model_cells.sum())} cells across {len(species_files)} maps")
    print(f"  BITS saw positive catches in {int(ices_footprint.sum())} cells")
    both = int((ices_footprint & model_cells).sum())
    ices_only = int((ices_footprint & ~model_cells).sum())
    model_only = int((model_cells & ~ices_footprint).sum())
    print(f"  overlap: {both}, BITS-only (model missing): {ices_only}, "
          f"model-only (BITS absent): {model_only}")


async def main() -> None:
    grid_lat, grid_lon, mask = load_grid()
    print(f"Grid: {mask.shape}, {int(mask.sum())} ocean cells")
    print(f"Grid bbox: lat {grid_lat[-1]:.2f}–{grid_lat[0]:.2f}, lon {grid_lon[0]:.2f}–{grid_lon[-1]:.2f}")

    print("\nFetching BITS hauls…")
    hauls = await fetch_all_hauls()
    validate_hauls(hauls, mask, grid_lat, grid_lon)

    for sp, aphia in SPECIES_APHIA.items():
        print(f"\nFetching BITS CPUE for {sp} (Aphia {aphia})…")
        cpue = await fetch_species_cpue(sp, aphia)
        if not cpue.empty:
            validate_species(sp, cpue, mask, grid_lat, grid_lon)


if __name__ == "__main__":
    asyncio.run(main())
