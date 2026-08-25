"""Build the C1 thermal series (SST Q3 / bottom-T Q4) from CMEMS data.

cod_west (sp0) ← surface thetao Q3 mean over SD22–24 (bbox 9.5–15.0E, 53.5–56.5N)
herring (sp1) ← bottom-T Q4 mean, same box

Download CMEMS data with scripts/download_baltic_rv_forcing.py:
  .venv/bin/python scripts/download_baltic_rv_forcing.py --vars thetao bottomT \\
    --depth-min 0 --depth-max 5 --start 1993 --end <product-end>

The script downloads missing CMEMS files, caches to data/cmems_cache/cmems_downloads/,
and writes the series to data/baltic/forcing/baltic_thermal_sr_series.csv with
provenance in data/baltic/forcing/baltic_thermal_sr_series.README.md.

On download failure (no credentials, network error, product gap), prints DEGRADED:
and uses PROVISIONAL fallback constants (marked in README for Task 4 to source real values).
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_thermal_sr_series.csv"

# SD22-24 sub-basin (review area)
BBOX_SD = {"lon_min": 9.5, "lon_max": 15.0, "lat_min": 53.5, "lat_max": 56.5}

# Provisional fallback constants (UNSOURCED PLACEHOLDERS — Task 4 must replace)
PROVISIONAL_TREF = {0: 11.5, 1: 8.5}  # cod_west, herring

# --- single source of truth for the year window --------------------------
# Every path that needs a year span (download requests, assemble_series's
# spin-up math, fallback_rows's row count) derives from these three. Do NOT
# hardcode a year window anywhere else in this file — three independently
# hardcoded windows (download calls, assemble_series defaults, fallback_rows)
# disagreeing with each other was the root cause of every regression across
# three prior review rounds on this file.
SPINUP_YEARS = 19
HIST_START = 1993
DOWNLOAD_END = 2021  # current CMEMS product-end request; move forward as the product extends


class DataUnavailable(Exception):
    """Data genuinely absent after a download attempt (creds/network/product gap/yearly gaps).

    This is the ONLY condition that may trigger the PROVISIONAL fallback.
    """

    pass


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
    hist: dict[int, float],
    tref: float,
    first_hist_year: int = HIST_START,
    spinup: int = SPINUP_YEARS,
) -> list[tuple[int, float]]:
    """Build series with spin-up years + historical years.

    hist: dict of {year: temperature} for historical period
    tref: reference temperature for spin-up years
    first_hist_year: first year in hist (default HIST_START)
    spinup: number of spin-up years to prepend (default SPINUP_YEARS)

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


def fallback_rows(tref: float, hist_end: int) -> list[tuple[int, float]]:
    """Return constant-temperature rows spanning HIST_START-SPINUP_YEARS..hist_end.

    tref: constant temperature for all rows
    hist_end: last year of the series. In a mixed degradation (one species
      real, one degraded) this MUST be the real species' last historical
      year, so both columns line up. When both species are degraded, pass
      DOWNLOAD_END. Never hardcode this — it is the one thing that made the
      three prior rounds on this file disagree with the real data's span.

    Returns list of (year, temp) tuples for years
    (HIST_START - SPINUP_YEARS) .. hist_end, inclusive.
    """
    start = HIST_START - SPINUP_YEARS
    return [(y, tref) for y in range(start, hist_end + 1)]


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
    """Write provenance README alongside path, with its final suffix replaced by .README.md.

    e.g. baltic_thermal_sr_series.csv -> baltic_thermal_sr_series.README.md.
    Previously this APPENDED ".README.md" (producing
    baltic_thermal_sr_series.csv.README.md), which disagreed with the
    plan/Task-4 contract's expected filename — fixed to replace the .csv
    suffix instead of stacking onto it.

    species_info: dict of {species_idx: {key: value}} with fields like
      'variable', 'quarters', 'bbox', 'tref_mean', 'year_range', 'status', etc.
    """
    readme_path = path.with_suffix(".README.md")

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

    # Subset depending on data shape — use sequential subsetting for clarity and correctness
    if data.ndim == 2:  # (lat, lon)
        return data[lat_indices, :][:, lon_indices]
    elif data.ndim == 3:  # (time, lat, lon)
        return data[:, lat_indices, :][:, :, lon_indices]
    elif data.ndim == 4:  # (time, depth, lat, lon)
        return data[:, :, lat_indices, :][:, :, :, lon_indices]
    else:
        raise ValueError(f"unexpected data shape {data.shape}")


def _months_from_time(ds: xr.Dataset) -> dict[int, np.ndarray]:
    """Extract monthly data arrays and month numbers from time coordinate.

    Returns dict of {month: index} for all months in the dataset.
    Raises ValueError if months are incomplete or duplicated.
    """
    if "time" not in ds.dims:
        raise ValueError("Dataset has no 'time' dimension")

    times = ds["time"].values
    months_dict = {}

    for i, t in enumerate(times):
        # Parse month from datetime64 timestamp
        import pandas as pd

        dt64 = np.datetime64(t)
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


def _spatial_nanmean(arr: np.ndarray) -> float:
    """Mean over a spatial array, excluding masked/land cells (NaN) instead of propagating them.

    SD22-24 is ~54% land/masked in the CMEMS grids. A plain `.mean()` over
    the bbox subset averages the NaNs in, so almost every yearly mean came
    out NaN while the surrounding code still reported status: OK. This
    returns the mean of only the finite cells.

    Raises ValueError if EVERY cell in `arr` is non-finite (masked) — there
    is nothing to average. This is checked HERE, at the one place that
    actually knows how many cells went into the mean, rather than letting
    `np.nanmean` return NaN silently (numpy only warns: "Mean of empty
    slice") and relying on that NaN to survive arithmetic two call frames
    downstream (through quarter_mean's sum, then _reject_nonfinite_means)
    before anything notices. Catching it at the source also names the
    actual fully-masked month/slice, instead of a NaN which the caller
    can no longer trace back to.
    """
    finite_count = np.count_nonzero(np.isfinite(arr))
    if finite_count == 0:
        raise ValueError(f"spatial subset is fully masked (0 of {arr.size} cells finite)")
    return float(np.nanmean(arr))


def _reject_nonfinite_means(var_name: str, means: dict[int, float]) -> None:
    """Raise ValueError naming var_name and the offending years if any yearly mean is non-finite.

    A NaN (or inf) mean with cached files present is a data/parsing
    problem — e.g. every cell in the bbox subset was masked for that
    year — and must never be silently written out as an OK series. This is
    deliberately a ValueError, not a DataUnavailable: a DataUnavailable
    would silently route into the PROVISIONAL fallback, disguising a
    code/data bug as ordinary data absence. It must crash main() loudly
    instead.
    """
    bad_years = sorted(y for y, v in means.items() if not np.isfinite(v))
    if bad_years:
        raise ValueError(f"{var_name}: non-finite yearly mean(s) for years {bad_years}")


def _load_thetao_series(files: list[Path]) -> dict[int, float]:
    """Load thetao (surface temp) Q3 means from cached files.

    Returns dict of {year: mean_temp}.
    Raises DataUnavailable if data is genuinely absent.
    Propagates other exceptions (code bugs).
    """
    thetao_means = {}
    for f in files:
        # I/O errors only in file open
        try:
            ds = xr.open_dataset(f)
        except (FileNotFoundError, OSError, IOError) as e:
            print(f"  warning: failed to open {f.name}: {e}")
            continue

        # Parsing errors and code bugs propagate — they are NOT caught
        # Extract year from filename (e.g., "baltic_phy_monthly_reanalysis_thetao_1993-01_1993-12.nc")
        fname_parts = f.stem.split("_")
        year_str = fname_parts[-2].split("-")[0]
        year = int(year_str)

        # Get month mapping — parsing error, must propagate
        months_idx = _months_from_time(ds)

        # Load thetao data — I/O error only
        try:
            theta = ds["thetao"].values
        except KeyError as e:
            print(f"  warning: no thetao variable in {f.name}: {e}")
            ds.close()
            continue

        if theta.ndim == 4:
            # (time, depth, lat, lon) - take surface
            theta = theta[:, 0, :, :]
        elif theta.ndim != 3:
            raise ValueError(f"expected 3D or 4D thetao, got shape {theta.shape}")

        # Get coordinates — parsing error, must propagate
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

        # Subset to SD22-24 bbox — parsing error, must propagate
        theta_subset = _subset_bbox(theta, lon, lat)

        # Extract Q3 months using quarter_mean — parsing error, must propagate.
        # _spatial_nanmean excludes masked/land cells rather than averaging
        # them in as NaN (SD22-24 is ~54% land/masked in the CMEMS grids).
        q3_dict = {m: _spatial_nanmean(theta_subset[months_idx[m], :, :]) for m in (7, 8, 9)}
        q3_mean = quarter_mean(q3_dict, quarter=3)

        thetao_means[year] = q3_mean
        ds.close()

    # Check for genuine absence: either no cache files, or files loaded but empty
    if not thetao_means:
        if not files:
            raise DataUnavailable("thetao: no cached files (download failed or not run)")
        else:
            raise DataUnavailable(f"thetao: {len(files)} files loaded but all failed or empty")

    # NaN guard: a non-finite yearly mean with files present is a data/parsing
    # problem (e.g. every cell in the bbox was masked that year), never OK.
    _reject_nonfinite_means("thetao", thetao_means)

    # Check for yearly gaps
    all_years = sorted(thetao_means.keys())
    expected_years = list(range(all_years[0], all_years[-1] + 1))
    missing_years = [y for y in expected_years if y not in thetao_means]
    if missing_years:
        raise DataUnavailable(f"thetao: yearly gaps in data: {missing_years}")

    return thetao_means


def _load_bottomt_series(files: list[Path]) -> dict[int, float]:
    """Load bottomT (bottom temp) Q4 means from cached files.

    Returns dict of {year: mean_temp}.
    Raises DataUnavailable if data is genuinely absent.
    Propagates other exceptions (code bugs).
    """
    bottomt_means = {}
    for f in files:
        # I/O errors only in file open
        try:
            ds = xr.open_dataset(f)
        except (FileNotFoundError, OSError, IOError) as e:
            print(f"  warning: failed to open {f.name}: {e}")
            continue

        # Parsing errors and code bugs propagate — they are NOT caught
        # Extract year from filename
        fname_parts = f.stem.split("_")
        year_str = fname_parts[-2].split("-")[0]
        year = int(year_str)

        # Get month mapping — parsing error, must propagate
        months_idx = _months_from_time(ds)

        # Load bottomT data — I/O error only
        try:
            bt = ds["bottomT"].values
        except KeyError as e:
            print(f"  warning: no bottomT variable in {f.name}: {e}")
            ds.close()
            continue

        if bt.ndim == 4:
            # (time, depth, lat, lon) - take bottom (last depth)
            bt = bt[:, -1, :, :]
        elif bt.ndim != 3:
            raise ValueError(f"expected 3D or 4D bottomT, got shape {bt.shape}")

        # Get coordinates — parsing error, must propagate
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

        # Subset to SD22-24 bbox — parsing error, must propagate
        bt_subset = _subset_bbox(bt, lon, lat)

        # Extract Q4 months using quarter_mean — parsing error, must propagate.
        # _spatial_nanmean excludes masked/land cells rather than averaging
        # them in as NaN (SD22-24 is ~54% land/masked in the CMEMS grids).
        q4_dict = {m: _spatial_nanmean(bt_subset[months_idx[m], :, :]) for m in (10, 11, 12)}
        q4_mean = quarter_mean(q4_dict, quarter=4)

        bottomt_means[year] = q4_mean
        ds.close()

    # Check for genuine absence: either no cache files, or files loaded but empty
    if not bottomt_means:
        if not files:
            raise DataUnavailable("bottomT: no cached files (download failed or not run)")
        else:
            raise DataUnavailable(f"bottomT: {len(files)} files loaded but all failed or empty")

    # NaN guard: a non-finite yearly mean with files present is a data/parsing
    # problem (e.g. every cell in the bbox was masked that year), never OK.
    _reject_nonfinite_means("bottomT", bottomt_means)

    # Check for yearly gaps
    all_years = sorted(bottomt_means.keys())
    expected_years = list(range(all_years[0], all_years[-1] + 1))
    missing_years = [y for y in expected_years if y not in bottomt_means]
    if missing_years:
        raise DataUnavailable(f"bottomT: yearly gaps in data: {missing_years}")

    return bottomt_means


def _download_variable(var: str, start_year: int, end_year: int) -> str | None:
    """Download missing years for a variable using download_baltic_rv_forcing machinery.

    A FIELDS lookup miss is a code bug (typo'd --vars name) and is left to
    raise KeyError — it must not be swallowed into a silent DEGRADED.

    Credential lookup (_creds()) and the `copernicusmarine` import are each
    given their own narrow, specific handling — an expired/missing .env or
    an uninstalled package is a clear, anticipated condition, not something
    that should be caught by (and indistinguishable from) a catch-all around
    the whole function.

    The broad catches here wrap only the actual network calls — `cm.login`
    and, per year, `_download_year` — because the remote client library can
    raise a wide variety of exception types for a wide variety of transient
    failures (bad/expired credentials, auth service down, connection reset,
    etc.). Their tracebacks are captured (not just str(e)) so a real failure
    is never invisible even though it's caught. `cm.login` in particular
    used to be unguarded here: a stale password would raise straight out of
    this function, through main() (which has no wrapper), and abort the
    whole run with a raw traceback instead of degrading gracefully — the
    same failure shape as the fallback-window bug this round otherwise
    fixes, just one call earlier in the pipeline.

    Returns None if every requested year ended up cached (already present or
    freshly downloaded this run). Otherwise returns a human-readable
    diagnostic string — naming the variable and embedding the traceback of
    the failure(s) — for the caller to fold into the DEGRADED record
    (printed and written to the README).
    """
    from download_baltic_rv_forcing import FIELDS, _creds, _download_year

    info = FIELDS[var]

    try:
        u, p = _creds()
    except SystemExit as e:
        return f"{var}: credentials unavailable ({e})"

    try:
        import copernicusmarine as cm
    except ImportError as e:
        return f"{var}: copernicusmarine not installed ({e})"

    try:
        cm.login(username=u, password=p, force_overwrite=True)
    except Exception:
        tb = traceback.format_exc()
        print(f"  warning: CMEMS login failed for {var}:\n{tb}")
        return f"{var}: CMEMS login failed:\n{tb}"

    failures: list[str] = []
    for year in range(start_year, end_year + 1):
        out_file = DL / f"baltic_{info['tag']}_{var}_{year}-01_{year}-12.nc"
        if out_file.exists() and out_file.stat().st_size > 0:
            continue  # already cached
        try:
            _download_year(cm, var, year, 0.0, 5.0)
        except Exception:
            tb = traceback.format_exc()
            print(f"  warning: failed to download {var} {year}:\n{tb}")
            failures.append(f"{var} {year} failed:\n{tb}")

    if not failures:
        return None
    return f"{var}: {len(failures)} year(s) failed to download\n" + "\n".join(failures)


def assemble_outputs(
    csv_path: Path,
    rows_by_species: dict[int, list[tuple[int, float]]],
    species_info: dict[int, dict[str, str]],
) -> None:
    """Assemble and write CSV + README from species data.

    csv_path: output CSV file path. The README is always written alongside
      it, by write_readme, at csv_path with its suffix replaced by
      .README.md — there is no separate configurable README path.
    rows_by_species: dict of {species_idx: [(year, temp), ...]}
    species_info: dict of {species_idx: metadata dict}
    """
    write_series_csv(csv_path, rows_by_species)
    write_readme(csv_path, species_info)


def _compose_degradation_reason(base_reason: str, download_note: str | None) -> str:
    """Fold the download attempt's diagnostic note into a degradation reason.

    Without this, the traceback captured by _download_variable only reached
    the README when the glob found zero cached files at all. The more likely
    real case — some years downloaded, some failed, and the loader's own
    yearly-gap check raises DataUnavailable — discarded the download note
    entirely, silently dropping the one piece of information that explains
    *why* the fallback fired. Called from every branch that sets
    degradation_reason[...] so the two species can't drift out of sync again.
    """
    if download_note:
        return f"{base_reason}\ndownload diagnostics: {download_note}"
    return base_reason


def main() -> int:
    """Download CMEMS thetao/bottomT and build thermal series.

    Pulls surface thetao (Q3) for cod_west and bottom-T (Q4) for herring
    from CMEMS over HIST_START..DOWNLOAD_END. Attempts download of missing
    years; on genuine data absence (download failure, missing files, yearly
    gaps) uses the PROVISIONAL fallback for that species only.

    No `except Exception` appears at this level, deliberately: a code bug
    (bad FIELDS entry, a broken loader) must crash loudly here rather than
    be silently reported as DEGRADED alongside genuine data-absence cases.

    Returns 0 on success (with or without degradation).
    """
    print(f"[download] attempting thetao for {HIST_START}-{DOWNLOAD_END} ...")
    thetao_download_note = _download_variable("thetao", HIST_START, DOWNLOAD_END)
    if thetao_download_note:
        print(f"  {thetao_download_note}")

    print(f"[download] attempting bottomT for {HIST_START}-{DOWNLOAD_END} ...")
    bottomt_download_note = _download_variable("bottomT", HIST_START, DOWNLOAD_END)
    if bottomt_download_note:
        print(f"  {bottomt_download_note}")

    # Glob cached files
    thetao_files = sorted(DL.glob("baltic_phy_monthly_reanalysis_thetao_*.nc"))
    bottomt_files = sorted(DL.glob("baltic_phy_monthly_reanalysis_bottomT_*.nc"))

    rows_by_species: dict[int, list[tuple[int, float]]] = {}
    species_info: dict[int, dict[str, str]] = {}
    last_hist_year: dict[int, int] = {}
    degradation_reason: dict[int, str] = {}

    # Process thetao (cod_west, sp0) - Q3 mean
    if thetao_files:
        print(f"[thetao] loading {len(thetao_files)} files for Q3 (Jul-Sep) ...")
        try:
            thetao_means = _load_thetao_series(thetao_files)
            tref = sum(thetao_means.values()) / len(thetao_means)
            rows_by_species[0] = assemble_series(thetao_means, tref=tref)
            last_hist_year[0] = max(thetao_means)
            year_range = f"{min(thetao_means)}-{max(thetao_means)}"
            species_info[0] = {
                "variable": "thetao (surface temperature)",
                "quarters": "Q3 (Jul-Sep)",
                "bbox": "SD22-24 (9.5-15.0E, 53.5-56.5N)",
                "tref_mean": f"{tref:.2f}",
                "year_range": year_range,
                "status": f"OK ({len(thetao_means)} years)",
            }
            print(f"  thetao Q3: {len(thetao_means)} years, tref={tref:.2f}, range={year_range}")
        except DataUnavailable as e:
            degradation_reason[0] = _compose_degradation_reason(str(e), thetao_download_note)
            print(f"  DEGRADED: {degradation_reason[0]}")
    else:
        degradation_reason[0] = _compose_degradation_reason(
            "thetao: no cached files after download attempt", thetao_download_note
        )

    # Process bottomT (herring, sp1) - Q4 mean
    if bottomt_files:
        print(f"[bottomT] loading {len(bottomt_files)} files for Q4 (Oct-Dec) ...")
        try:
            bottomt_means = _load_bottomt_series(bottomt_files)
            tref = sum(bottomt_means.values()) / len(bottomt_means)
            rows_by_species[1] = assemble_series(bottomt_means, tref=tref)
            last_hist_year[1] = max(bottomt_means)
            year_range = f"{min(bottomt_means)}-{max(bottomt_means)}"
            species_info[1] = {
                "variable": "bottomT (bottom temperature)",
                "quarters": "Q4 (Oct-Dec)",
                "bbox": "SD22-24 (9.5-15.0E, 53.5-56.5N)",
                "tref_mean": f"{tref:.2f}",
                "year_range": year_range,
                "status": f"OK ({len(bottomt_means)} years)",
            }
            print(f"  bottomT Q4: {len(bottomt_means)} years, tref={tref:.2f}, range={year_range}")
        except DataUnavailable as e:
            degradation_reason[1] = _compose_degradation_reason(str(e), bottomt_download_note)
            print(f"  DEGRADED: {degradation_reason[1]}")
    else:
        degradation_reason[1] = _compose_degradation_reason(
            "bottomT: no cached files after download attempt", bottomt_download_note
        )

    # Fallback window: the surviving species' last historical year in a
    # mixed degradation (so both columns line up), or DOWNLOAD_END if both
    # variables failed. This is the fix for the round-3 bug: fallback_rows
    # used to hardcode 1974-2023 while a real loader spans 1974-2021,
    # producing mismatched year sets and an aborted run in exactly the
    # one-real-one-degraded scenario this fallback exists for.
    hist_end = next(iter(last_hist_year.values()), DOWNLOAD_END)

    fallback_meta = {
        0: ("thetao (surface temperature)", "Q3 (Jul-Sep)"),
        1: ("bottomT (bottom temperature)", "Q4 (Oct-Dec)"),
    }
    for sp, (var_name, quarter_label) in fallback_meta.items():
        if sp in rows_by_species:
            continue
        reason = degradation_reason.get(sp, f"sp{sp} missing")
        print(f"DEGRADED: {reason}")
        tref = PROVISIONAL_TREF[sp]
        rows_by_species[sp] = fallback_rows(tref, hist_end)
        species_info[sp] = {
            "variable": var_name,
            "quarters": quarter_label,
            "status": f"DEGRADED: {reason}",
            "tref": f"{tref} (UNSOURCED PLACEHOLDER)",
            "note": "UNSOURCED PLACEHOLDER — Task 4 must source real computed value or literature source",
        }

    # Final assertion, not a routine branch: with a shared hist_end this must
    # now be unreachable for the mixed-degradation case the fallback exists
    # for. It stays as a guard for a real-vs-real mismatch (e.g. two loaders
    # that both "succeeded" but over different completion windows).
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
    assemble_outputs(OUT, rows_by_species, species_info)

    print(f"wrote {OUT} ({len(rows_by_species)} species)")
    print(f"wrote {OUT.with_suffix('.README.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
