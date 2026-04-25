#!/usr/bin/env python3
"""Build the Baltic predator (grey seal + cormorant) biomass NetCDF.

Produces data/baltic/baltic_predator_biomass.nc with per-class, per-timestep,
per-cell STANDING BIOMASS for two background species. OSMOSE multiplies this
by `predation.ingestion.rate.max` (annual turnover) to get consumption — do
NOT put consumption-equivalent biomass here.

Spatial distribution: weighted by HELCOM sub-basin per literature (Galatius
et al. 2020 for seal; Östman et al. 2013 + Heikinheimo 2021 for cormorant).
Temporal: constant across 24 biweekly steps. Cormorant biomass is
presence-weighted (×0.48) to account for ~65% Oct-Apr absence from Baltic.

Usage: .venv/bin/python scripts/build_baltic_predator_nc.py
"""
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_NC = PROJECT_ROOT / "data" / "baltic" / "baltic_grid.nc"
OUT_NC = PROJECT_ROOT / "data" / "baltic" / "baltic_predator_biomass.nc"

# STANDING biomass (tonnes of predator flesh)
# Seal:      30,000 ind × 150 kg/ind = 4,500 t  (HELCOM Seal DB 2019 / Galatius 2020)
# Cormorant: ~520,000 ind × 2 kg × 0.48 presence = 500 t  (Östman 2013; seasonal)
SEAL_STANDING_T = 4_500.0
CORMORANT_STANDING_T = 500.0

# Size classes: "length" here is the PREDATOR's own body length (cm),
# used by OSMOSE for (a) length→weight via condition factor, and
# (b) size-ratio predation (predator_length / prey_length in [min, max]).
# For a 150 cm seal with size_ratio 3-12 → prey range 12.5-50 cm (herring,
# sprat, medium cod, flounder). For an 80 cm cormorant with ratio 2.5-8 →
# prey 10-32 cm (small perch, herring, juvenile cod).
SEAL_CLASSES = {
    "length": [110.0, 170.0],            # sub-adult and adult body length (cm)
    "trophic_level": [4.5, 4.8],
    "biomass_fraction": [0.35, 0.65],    # Baltic seal population skewed adult
    "age_years": [2, 6],
}
CORMORANT_CLASSES = {
    "length": [70.0, 85.0],              # body length (cm)
    "trophic_level": [4.2, 4.5],
    "biomass_fraction": [0.4, 0.6],
    "age_years": [1, 3],
}


def _classify(lat: float, lon: float) -> str:
    if lat >= 63.4: return "BothnianBay"
    if lat >= 60.4: return "BothnianSea"
    if lat >= 59.0 and lon >= 22.9: return "GulfOfFinland"
    if lat >= 58.5 and lon < 21.0:  return "NBalticProper"
    if lat >= 57.0 and 21.0 <= lon <= 24.5: return "GulfOfRiga"
    if lat >= 58.0 and lon < 19.5:  return "WGotland"
    if lat >= 56.8 and 18.0 <= lon <= 21.5: return "EGotland"
    if lat >= 54.8 and lon >= 16.0: return "BornholmGdansk"
    if lat >= 54.8 and 13.0 <= lon < 16.0: return "Arkona"
    if lat >= 54.8 and lon < 13.0:  return "BeltSoundKiel"
    return "Mecklenburg"


# Spatial weights per HELCOM sub-region, normalized to 1.0.
# Seal: Gulf of Bothnia + Stockholm archipelago dominate (~80% of 30k pop),
# Kalmarsund ~10%, SW Baltic ~5%, others <3% (Galatius 2020 + HELCOM 2019).
SEAL_WEIGHTS = {
    "BothnianBay": 0.15, "BothnianSea": 0.25, "GulfOfFinland": 0.03,
    "NBalticProper": 0.25, "GulfOfRiga": 0.02, "WGotland": 0.15,
    "EGotland": 0.08, "BornholmGdansk": 0.03, "Arkona": 0.02,
    "BeltSoundKiel": 0.01, "Mecklenburg": 0.01,
}
# Cormorant: coastal-weighted; Gulf of Riga + Gulf of Finland + Stockholm
# archipelago + Kalmar Sound are peak density (Östman 2013, Heikinheimo 2021).
CORMORANT_WEIGHTS = {
    "BothnianBay": 0.04, "BothnianSea": 0.10, "GulfOfFinland": 0.20,
    "NBalticProper": 0.15, "GulfOfRiga": 0.22, "WGotland": 0.05,
    "EGotland": 0.10, "BornholmGdansk": 0.05, "Arkona": 0.05,
    "BeltSoundKiel": 0.02, "Mecklenburg": 0.02,
}


def _build_spatial_field(total_t: float, weights: dict[str, float],
                         lat1d: np.ndarray, lon1d: np.ndarray,
                         ocean: np.ndarray) -> np.ndarray:
    """Allocate `total_t` across ocean cells proportional to sub-basin weights."""
    ny, nx = ocean.shape
    weight_per_cell = np.zeros((ny, nx), dtype=float)
    for r in range(ny):
        for c in range(nx):
            if not ocean[r, c]:
                continue
            region = _classify(lat1d[r], lon1d[c])
            weight_per_cell[r, c] = weights[region]
    total_weight = weight_per_cell.sum()
    if total_weight == 0:
        raise RuntimeError("No ocean cells matched any region weight")
    return total_t * weight_per_cell / total_weight  # tonnes per cell


def main() -> None:
    grid = xr.open_dataset(GRID_NC)
    lat = grid["latitude"].values
    lon = grid["longitude"].values
    ocean = grid["mask"].values == 1

    n_time = 24
    time_coord = np.arange(n_time, dtype=np.int32)

    def _build(total_t: float, weights: dict[str, float]) -> np.ndarray:
        """(time, lat, lon) TOTAL standing biomass; constant across time.

        OSMOSE's background-species loader expects (time, lat, lon) per
        variable and splits the per-cell biomass across size classes via
        `species.size.proportion.spN` (set in baltic_param-background.csv).
        """
        field_2d = _build_spatial_field(total_t, weights, lat, lon, ocean)
        out = np.zeros((n_time, len(lat), len(lon)), dtype=float)
        out[:, :, :] = field_2d[np.newaxis, :, :]
        return out

    seal_data = _build(SEAL_STANDING_T, SEAL_WEIGHTS)
    cormorant_data = _build(CORMORANT_STANDING_T, CORMORANT_WEIGHTS)

    ds = xr.Dataset(
        {
            "GreySeal": (["time", "latitude", "longitude"], seal_data),
            "Cormorant": (["time", "latitude", "longitude"], cormorant_data),
        },
        coords={
            "time": time_coord,
            "latitude": lat,
            "longitude": lon,
        },
        attrs={
            "title": "Baltic OSMOSE background-species biomass (top predators)",
            "description": (
                "Grey seal + cormorant STANDING biomass. OSMOSE computes "
                "consumption as biomass × predation.ingestion.rate.max "
                "(set in baltic_param-background.csv)."
            ),
            "seal_standing_tonnes": SEAL_STANDING_T,
            "cormorant_standing_tonnes": CORMORANT_STANDING_T,
            "references": (
                "Galatius et al. 2020 doi:10.2981/wlb.00711 (seal pop + distrib); "
                "Lundström et al. 2010 doi:10.7557/3.2733 (seal diet composition); "
                "Gårdmark et al. 2012 doi:10.1093/icesjms/fss099 (seal herring predation); "
                "Östman et al. 2013 doi:10.1371/journal.pone.0083763 (cormorant pop + consumption); "
                "Heikinheimo et al. 2021 doi:10.1093/icesjms/fsab258 (cormorant perch mortality); "
                "Heikinheimo et al. 2016 doi:10.1139/cjfas-2015-0033 (cormorant pikeperch mortality)"
            ),
        },
    )

    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT_NC)
    print(f"Wrote {OUT_NC}")
    print(f"  GreySeal per-timestep total: {seal_data[0].sum():.1f} t (expected {SEAL_STANDING_T:.1f})")
    print(f"  Cormorant per-timestep total: {cormorant_data[0].sum():.1f} t (expected {CORMORANT_STANDING_T:.1f})")
    grid.close()


if __name__ == "__main__":
    main()
