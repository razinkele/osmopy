"""Build the C1 thermal series (SST Q3 / bottom-T Q4) from CMEMS data.

cod_west (sp0) ← surface thetao Q3 mean over SD22–24 (bbox 9.5–15.0E, 53.5–56.5N)
herring (sp1) ← bottom-T Q4 mean, same box

Download CMEMS data with scripts/download_baltic_rv_forcing.py:
  .venv/bin/python scripts/download_baltic_rv_forcing.py --vars thetao bottomT \\
    --depth-min 0 --depth-max 5 --start 1993 --end <product-end>

The script downloads missing CMEMS files, caches to data/cmems_cache/cmems_downloads/,
and writes the series to data/baltic/forcing/baltic_thermal_sr_series.csv with
provenance in data/baltic/forcing/baltic_thermal_sr_series.csv.README.md.

On download failure (no credentials, network error, product gap), prints DEGRADED:
and uses a PROVISIONAL fallback constant for that species (marked in README).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_thermal_sr_series.csv"

# SD22-24 sub-basin (review area)
BBOX_SD = {"lon_min": 9.5, "lon_max": 15.0, "lat_min": 53.5, "lat_max": 56.5}

# Provisional fallback constants (used if download fails; Task 4 must source real values)
PROVISIONAL_TREF = {0: 11.5, 1: 8.5}  # cod_west, herring
PROVISIONAL_LABEL = {0: "11.5 (PROVISIONAL)", 1: "8.5 (PROVISIONAL)"}


def quarter_mean(monthly_temps: dict[int, float], quarter: int) -> float:
    """Compute mean temperature for a quarter (Q1=1,Q2=2,Q3=3,Q4=4).

    monthly_temps: dict with keys 1-12 (Jan-Dec)
    quarter: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

    Raises ValueError if any required month is missing.
    """
    q_months = {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9), 4: (10, 11, 12)}
    if quarter not in q_months:
        raise ValueError(f"quarter must be 1-4, got {quarter}")
    months = q_months[quarter]
    try:
        values = [monthly_temps[m] for m in months]
    except KeyError as e:
        raise ValueError(f"missing month in quarter {quarter}: {e}") from e
    return sum(values) / len(values)


def assemble_series(
    hist: dict[int, float], tref: float, first_hist_year: int = 1993, spinup: int = 19
) -> list[tuple[int, float]]:
    """Build series with spin-up years + historical years.

    hist: dict of {year: temperature} for historical period
    tref: reference temperature for spin-up years
    first_hist_year: first year in hist (default 1993)
    spinup: number of spin-up years to prepend (default 19)

    Returns list of (year, temp) tuples, with years spanning
    first_hist_year-spinup through max(hist.keys()).

    Raises ValueError if historical years are not contiguous or if the first year
    in hist does not match first_hist_year.
    """
    if not hist:
        raise ValueError("hist must not be empty")

    hist_years = sorted(hist.keys())
    first_year = hist_years[0]
    last_year = hist_years[-1]

    # Validate first year matches expected
    if first_year != first_hist_year:
        raise ValueError(
            f"first historical year {first_year} does not match expected {first_hist_year}"
        )

    # Check contiguity
    expected = list(range(first_year, last_year + 1))
    if hist_years != expected:
        raise ValueError(f"historical years not contiguous: {hist_years}")

    # Build spin-up rows
    spinup_start = first_year - spinup
    rows = [(y, tref) for y in range(spinup_start, first_year)]

    # Add historical rows
    for y in hist_years:
        rows.append((y, hist[y]))

    return rows


def write_series_csv(path: Path, rows_by_species: dict[int, list[tuple[int, float]]]) -> None:
    """Write series CSV with columns year, temp_sp0, temp_sp1, ...

    path: output file path
    rows_by_species: dict of {species_idx: [(year, temp), ...]}

    No comment lines (loader rejects them). Years must be the same and
    contiguous across all species.
    """
    # Ensure path is a Path object
    if not isinstance(path, Path):
        path = Path(path)

    # Extract years from first species
    sp_indices = sorted(rows_by_species.keys())
    if not sp_indices:
        raise ValueError("rows_by_species must not be empty")

    first_sp = sp_indices[0]
    all_years = [y for y, _ in rows_by_species[first_sp]]

    # Validate all species have same years
    for sp in sp_indices:
        sp_years = [y for y, _ in rows_by_species[sp]]
        if sp_years != all_years:
            raise ValueError(
                f"species have mismatched years: sp{first_sp}={all_years} vs sp{sp}={sp_years}"
            )

    # Build header
    header = "year," + ",".join(f"temp_sp{sp}" for sp in sp_indices)

    # Build rows
    lines = [header]
    for year in all_years:
        values = [str(year)]
        for sp in sp_indices:
            temp_dict = {y: t for y, t in rows_by_species[sp]}
            values.append(str(temp_dict[year]))
        lines.append(",".join(values))

    # Write file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_readme(path: Path, species_info: dict[int, dict[str, str]]) -> None:
    """Write provenance README to path.README.md.

    species_info: dict of {species_idx: {key: value}} with fields like
      'variable', 'quarters', 'bbox', 'tref_mean', 'year_range', 'status', etc.
    """
    readme_path = Path(str(path) + ".README.md")

    lines = [
        "# Baltic Thermal Series (C1)",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Series specification",
        "",
        f"Output file: `{path.name}`",
        "",
        "## Per-species metadata",
        "",
    ]

    for sp in sorted(species_info.keys()):
        info = species_info[sp]
        lines.append(f"### Species {sp}")
        lines.append("")
        for key, value in sorted(info.items()):
            lines.append(f"  - {key}: {value}")
        lines.append("")

    readme_path.write_text("\n".join(lines))


def _months_from_time(ds: xr.Dataset) -> dict[int, np.ndarray]:
    """Extract monthly data arrays and month numbers from time coordinate.

    Returns dict of {month: data_array} for all 12 months in the dataset.
    Raises ValueError if months are incomplete or duplicated.
    """
    if "time" not in ds.dims:
        raise ValueError("Dataset has no 'time' dimension")

    times = ds["time"].values
    months_dict = {}

    for i, t in enumerate(times):
        # Parse month from datetime64 timestamp
        dt64 = np.datetime64(t)
        # Convert to pandas Timestamp for easy month extraction
        import pandas as pd

        ts = pd.Timestamp(dt64)
        month = ts.month
        if month in months_dict:
            raise ValueError(
                f"duplicate data for month {month} at indices {months_dict[month]}, {i}"
            )
        months_dict[month] = i

    if len(months_dict) != 12:
        raise ValueError(
            f"expected 12 months, got {len(months_dict)}: {sorted(months_dict.keys())}"
        )

    return months_dict


def _subset_bbox(data: np.ndarray, coords_lon: np.ndarray, coords_lat: np.ndarray) -> np.ndarray:
    """Subset spatial data to SD22-24 bbox (9.5-15.0E, 53.5-56.5N).

    data: array of shape (lat, lon) or (time, lat, lon) or (time, depth, lat, lon)
    coords_lon, coords_lat: 1D coordinate arrays

    Returns subsetted array.
    """
    # Find indices in bbox
    lon_mask = (coords_lon >= BBOX_SD["lon_min"]) & (coords_lon <= BBOX_SD["lon_max"])
    lat_mask = (coords_lat >= BBOX_SD["lat_min"]) & (coords_lat <= BBOX_SD["lat_max"])

    lon_indices = np.where(lon_mask)[0]
    lat_indices = np.where(lat_mask)[0]

    if len(lon_indices) == 0 or len(lat_indices) == 0:
        raise ValueError(
            f"no data in bbox {BBOX_SD}; lon indices {lon_indices}, lat indices {lat_indices}"
        )

    # Subset depending on data shape
    if data.ndim == 2:  # (lat, lon)
        return data[np.ix_(lat_indices, lon_indices)]
    elif data.ndim == 3:  # (time, lat, lon)
        return data[:, np.ix_(lat_indices, lon_indices)]
    elif data.ndim == 4:  # (time, depth, lat, lon)
        return data[:, :, np.ix_(lat_indices, lon_indices)]
    else:
        raise ValueError(f"unexpected data shape {data.shape}")


def _load_thetao_series(files: list[Path]) -> dict[int, float] | None:
    """Load thetao (surface temp) Q3 means from cached files.

    Returns dict of {year: mean_temp} or None on failure.
    """
    thetao_means = {}
    for f in files:
        try:
            ds = xr.open_dataset(f)

            # Extract year from filename (e.g., "baltic_phy_monthly_reanalysis_thetao_1993-01_1993-12.nc")
            fname_parts = f.stem.split("_")
            year_str = fname_parts[-2].split("-")[0]
            year = int(year_str)

            # Get month mapping
            months_idx = _months_from_time(ds)

            # Load thetao data
            theta = ds["thetao"].values
            if theta.ndim == 4:
                # (time, depth, lat, lon) - take surface
                theta = theta[:, 0, :, :]
            elif theta.ndim != 3:
                raise ValueError(f"expected 3D or 4D thetao, got shape {theta.shape}")

            # Get coordinates
            if "longitude" in ds.coords:
                lon = ds["longitude"].values
            elif "lon" in ds.coords:
                lon = ds["lon"].values
            else:
                raise ValueError("no longitude/lon coordinate")

            if "latitude" in ds.coords:
                lat = ds["latitude"].values
            elif "lat" in ds.coords:
                lat = ds["lat"].values
            else:
                raise ValueError("no latitude/lat coordinate")

            # Subset to SD22-24 bbox
            theta_subset = _subset_bbox(theta, lon, lat)

            # Extract Q3 months and average
            q3_indices = [months_idx[m] for m in (7, 8, 9)]
            q3_data = theta_subset[q3_indices, :, :]
            q3_mean = float(q3_data.mean())

            thetao_means[year] = q3_mean
            ds.close()

        except Exception as e:
            print(f"  warning: failed to read {f.name}: {e}")
            continue

    return thetao_means if thetao_means else None


def _load_bottomt_series(files: list[Path]) -> dict[int, float] | None:
    """Load bottomT (bottom temp) Q4 means from cached files.

    Returns dict of {year: mean_temp} or None on failure.
    """
    bottomt_means = {}
    for f in files:
        try:
            ds = xr.open_dataset(f)

            # Extract year from filename
            fname_parts = f.stem.split("_")
            year_str = fname_parts[-2].split("-")[0]
            year = int(year_str)

            # Get month mapping
            months_idx = _months_from_time(ds)

            # Load bottomT data
            bt = ds["bottomT"].values
            if bt.ndim == 4:
                # (time, depth, lat, lon) - take bottom (last depth)
                bt = bt[:, -1, :, :]
            elif bt.ndim != 3:
                raise ValueError(f"expected 3D or 4D bottomT, got shape {bt.shape}")

            # Get coordinates
            if "longitude" in ds.coords:
                lon = ds["longitude"].values
            elif "lon" in ds.coords:
                lon = ds["lon"].values
            else:
                raise ValueError("no longitude/lon coordinate")

            if "latitude" in ds.coords:
                lat = ds["latitude"].values
            elif "lat" in ds.coords:
                lat = ds["lat"].values
            else:
                raise ValueError("no latitude/lat coordinate")

            # Subset to SD22-24 bbox
            bt_subset = _subset_bbox(bt, lon, lat)

            # Extract Q4 months and average
            q4_indices = [months_idx[m] for m in (10, 11, 12)]
            q4_data = bt_subset[q4_indices, :, :]
            q4_mean = float(q4_data.mean())

            bottomt_means[year] = q4_mean
            ds.close()

        except Exception as e:
            print(f"  warning: failed to read {f.name}: {e}")
            continue

    return bottomt_means if bottomt_means else None


def _download_variable(var: str, start_year: int, end_year: int) -> bool:
    """Download missing years for a variable using download_baltic_rv_forcing machinery.

    Returns True if download succeeded (at least one year), False otherwise.
    """
    try:
        # Import download machinery
        from download_baltic_rv_forcing import FIELDS, _creds, _download_year

        info = FIELDS.get(var)
        if not info:
            print(f"  error: variable {var} not in FIELDS")
            return False

        # Get credentials
        try:
            u, p = _creds()
        except SystemExit as e:
            print(f"  DEGRADED: {e}")
            return False

        import copernicusmarine as cm

        cm.login(username=u, password=p, force_overwrite=True)

        # Download missing years
        downloaded = False
        for year in range(start_year, end_year + 1):
            out_file = DL / f"baltic_{info['tag']}_{var}_{year}-01_{year}-12.nc"
            if out_file.exists() and out_file.stat().st_size > 0:
                continue  # Already cached
            try:
                _download_year(cm, var, year, 0.0, 5.0)
                downloaded = True
            except Exception as e:
                print(f"  warning: failed to download {var} {year}: {e}")

        return downloaded

    except Exception as e:
        print(f"  DEGRADED: failed to download {var}: {e}")
        return False


def main() -> int:
    """Download CMEMS thetao/bottomT and build thermal series.

    Pulls surface thetao (Q3) for cod_west and bottom-T (Q4) for herring
    from CMEMS. Attempts download of missing years; on failure prints DEGRADED:
    and uses PROVISIONAL fallback constants (marked in README).

    Returns 0 on success (with or without degradation).
    """
    # Try to download missing files
    print("[download] attempting thetao for 1993-2021 ...")
    try:
        _download_variable("thetao", 1993, 2021)
    except Exception as e:
        print(f"  warning: download attempt failed: {e}")

    print("[download] attempting bottomT for 1993-2021 ...")
    try:
        _download_variable("bottomT", 1993, 2021)
    except Exception as e:
        print(f"  warning: download attempt failed: {e}")

    # Glob cached files
    thetao_files = sorted(DL.glob("baltic_phy_monthly_reanalysis_thetao_*.nc"))
    bottomt_files = sorted(DL.glob("baltic_phy_monthly_reanalysis_bottomT_*.nc"))

    rows_by_species = {}
    species_info = {}
    degradations = []

    # Process thetao (cod_west, sp0) - Q3 mean
    if thetao_files:
        print(f"[thetao] loading {len(thetao_files)} files for Q3 (Jul-Sep) ...")
        thetao_means = _load_thetao_series(thetao_files)
        if thetao_means:
            tref = sum(thetao_means.values()) / len(thetao_means)
            rows_by_species[0] = assemble_series(thetao_means, tref=tref)
            year_range = f"{min(thetao_means.keys())}-{max(thetao_means.keys())}"
            species_info[0] = {
                "variable": "thetao (surface temperature)",
                "quarters": "Q3 (Jul-Sep)",
                "bbox": "SD22-24 (9.5-15.0E, 53.5-56.5N)",
                "tref_mean": f"{tref:.2f}",
                "year_range": year_range,
                "status": f"OK ({len(thetao_means)} years)",
            }
            print(f"  thetao Q3: {len(thetao_means)} years, tref={tref:.2f}, range={year_range}")
        else:
            degradations.append("thetao (no valid data after load attempt)")
    else:
        degradations.append("thetao (no cached files after download attempt)")

    # Process bottomT (herring, sp1) - Q4 mean
    if bottomt_files:
        print(f"[bottomT] loading {len(bottomt_files)} files for Q4 (Oct-Dec) ...")
        bottomt_means = _load_bottomt_series(bottomt_files)
        if bottomt_means:
            tref = sum(bottomt_means.values()) / len(bottomt_means)
            rows_by_species[1] = assemble_series(bottomt_means, tref=tref)
            year_range = f"{min(bottomt_means.keys())}-{max(bottomt_means.keys())}"
            species_info[1] = {
                "variable": "bottomT (bottom temperature)",
                "quarters": "Q4 (Oct-Dec)",
                "bbox": "SD22-24 (9.5-15.0E, 53.5-56.5N)",
                "tref_mean": f"{tref:.2f}",
                "year_range": year_range,
                "status": f"OK ({len(bottomt_means)} years)",
            }
            print(f"  bottomT Q4: {len(bottomt_means)} years, tref={tref:.2f}, range={year_range}")
        else:
            degradations.append("bottomT (no valid data after load attempt)")
    else:
        degradations.append("bottomT (no cached files after download attempt)")

    # Use PROVISIONAL fallback for missing species (maintain fixed CSV shape)
    if 0 not in rows_by_species:
        print(f"DEGRADED: {degradations[0] if degradations else 'thetao missing'}")
        if 1 in rows_by_species:
            # Use historical years from herring
            hist_years = {y: PROVISIONAL_TREF[0] for y, _ in rows_by_species[1]}
            rows_by_species[0] = (
                hist_years
                if len(hist_years) == 1
                else assemble_series(hist_years, tref=PROVISIONAL_TREF[0])
            )
            species_info[0] = {
                "variable": "thetao (surface temperature)",
                "quarters": "Q3 (Jul-Sep)",
                "status": f"DEGRADED: {degradations[0]}; using PROVISIONAL constant {PROVISIONAL_LABEL[0]}",
                "note": "Task 4 must download and recompute if this path fires",
            }

    if 1 not in rows_by_species:
        print(f"DEGRADED: {degradations[-1] if degradations else 'bottomT missing'}")
        if 0 in rows_by_species:
            # Use historical years from cod_west
            hist_years = {y: PROVISIONAL_TREF[1] for y, _ in rows_by_species[0]}
            rows_by_species[1] = (
                hist_years
                if len(hist_years) == 1
                else assemble_series(hist_years, tref=PROVISIONAL_TREF[1])
            )
            species_info[1] = {
                "variable": "bottomT (bottom temperature)",
                "quarters": "Q4 (Oct-Dec)",
                "status": f"DEGRADED: {degradations[-1]}; using PROVISIONAL constant {PROVISIONAL_LABEL[1]}",
                "note": "Task 4 must download and recompute if this path fires",
            }

    if not rows_by_species:
        print("ERROR: no thermal data available after degradation")
        return 1

    # Ensure both species have same years (for CSV shape)
    all_sps = sorted(rows_by_species.keys())
    first_sp = all_sps[0]
    first_years = {y for y, _ in rows_by_species[first_sp]}
    for sp in all_sps[1:]:
        sp_years = {y for y, _ in rows_by_species[sp]}
        if sp_years != first_years:
            print(f"ERROR: species have different year ranges: sp{first_sp} vs sp{sp}")
            return 1

    # Write output
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_series_csv(OUT, rows_by_species)
    write_readme(OUT, species_info)

    print(f"wrote {OUT} ({len(rows_by_species)} species)")
    print(f"wrote {OUT}.README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
