"""Build the C1 thermal series (SST Q3 / bottom-T Q4) from CMEMS data.

cod_west (sp0) ← surface thetao Q3 mean over SD22–24 (bbox 9.5–15.0E, 53.5–56.5N)
herring (sp1) ← bottom-T Q4 mean, same box

Download CMEMS data with scripts/download_baltic_rv_forcing.py:
  .venv/bin/python scripts/download_baltic_rv_forcing.py --vars thetao bottomT \\
    --depth-min 0 --depth-max 5 --start 1993 --end <product-end>

The script reads cached files from data/cmems_cache/cmems_downloads/ and
writes the series to data/baltic/forcing/baltic_thermal_sr_series.csv with
provenance in data/baltic/forcing/baltic_thermal_sr_series.csv.README.md.

On missing downloads, prints a DEGRADED: line and continues with what exists.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_thermal_sr_series.csv"


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

    Raises ValueError if historical years are not contiguous.
    """
    if not hist:
        raise ValueError("hist must not be empty")

    hist_years = sorted(hist.keys())
    first_year = hist_years[0]
    last_year = hist_years[-1]

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


def main() -> int:
    """Download CMEMS thetao/bottomT and build thermal series.

    Pulls surface thetao (Q3) for cod_west and bottom-T (Q4) for herring
    from cached CMEMS downloads. Prints DEGRADED: line if any variable
    is missing, then continues with what exists.

    Returns 0 on success (with or without degradation).
    """
    try:
        import xarray as xr
    except ImportError:
        print("DEGRADED: xarray not installed")
        return 1

    # Try to load cached CMEMS files
    # Look for thetao files (surface temperature) for Q3 (Jul-Sep)
    # Look for bottomT files (bottom temperature) for Q4 (Oct-Dec)

    thetao_files = sorted(DL.glob("baltic_phy_monthly_reanalysis_thetao_*.nc"))
    bottomt_files = sorted(DL.glob("baltic_phy_monthly_reanalysis_bottomT_*.nc"))

    degraded = []
    if not thetao_files:
        degraded.append("thetao (surface temperature for cod_west Q3)")
    if not bottomt_files:
        degraded.append("bottomT (bottom temperature for herring Q4)")

    if degraded:
        print(f"DEGRADED: missing CMEMS downloads: {', '.join(degraded)}")
        if not thetao_files and not bottomt_files:
            print(
                "  Download with: .venv/bin/python scripts/download_baltic_rv_forcing.py "
                "--vars thetao bottomT --depth-min 0 --depth-max 5 "
                "--start 1993 --end <product-end>"
            )
            return 1

    rows_by_species = {}
    species_info = {}

    # Process thetao (cod_west, sp0) - Q3 mean
    if thetao_files:
        print(f"[thetao] loading {len(thetao_files)} files for Q3 (Jul-Sep) ...")
        thetao_means = {}
        for f in thetao_files:
            try:
                ds = xr.open_dataset(f)
                # Extract year from filename
                fname_parts = f.stem.split("_")
                year_str = fname_parts[-2].split("-")[0]  # e.g., "1993-01" -> "1993"
                year = int(year_str)

                # Get thetao data, select surface, select Q3 months (7, 8, 9 = indices 6, 7, 8)
                theta = ds["thetao"].values
                if theta.ndim == 4:
                    # (time, depth, lat, lon) - take surface
                    theta = theta[:, 0, :, :]
                # theta should be (12, lat, lon) if 1 file/year
                if theta.shape[0] >= 9:
                    q3_data = theta[6:9, :, :]  # months 7, 8, 9
                    q3_mean = q3_data.mean()
                    thetao_means[year] = float(q3_mean)
                ds.close()
            except Exception as e:
                print(f"  warning: failed to read {f.name}: {e}")

        if thetao_means:
            tref = sum(thetao_means.values()) / len(thetao_means)
            rows_by_species[0] = assemble_series(thetao_means, tref=tref)
            year_range = f"{min(thetao_means.keys())}-{max(thetao_means.keys())}"
            species_info[0] = {
                "variable": "thetao (surface temperature)",
                "quarters": "Q3 (Jul-Sep)",
                "tref_mean": f"{tref:.2f}",
                "year_range": year_range,
                "status": f"OK ({len(thetao_means)} years)",
            }
            print(f"  thetao Q3: {len(thetao_means)} years, tref={tref:.2f}, range={year_range}")

    # Process bottomT (herring, sp1) - Q4 mean
    if bottomt_files:
        print(f"[bottomT] loading {len(bottomt_files)} files for Q4 (Oct-Dec) ...")
        bottomt_means = {}
        for f in bottomt_files:
            try:
                ds = xr.open_dataset(f)
                # Extract year from filename
                fname_parts = f.stem.split("_")
                year_str = fname_parts[-2].split("-")[0]
                year = int(year_str)

                # Get bottomT data
                # bottomT should be (time, lat, lon) or (time, depth, lat, lon)
                bt = ds["bottomT"].values
                if bt.ndim == 4:
                    # (time, depth, lat, lon) - take bottom (last depth)
                    bt = bt[:, -1, :, :]
                # Select Q4 months (10, 11, 12 = indices 9, 10, 11)
                if bt.shape[0] >= 12:
                    q4_data = bt[9:12, :, :]
                    q4_mean = q4_data.mean()
                    bottomt_means[year] = float(q4_mean)
                ds.close()
            except Exception as e:
                print(f"  warning: failed to read {f.name}: {e}")

        if bottomt_means:
            tref = sum(bottomt_means.values()) / len(bottomt_means)
            rows_by_species[1] = assemble_series(bottomt_means, tref=tref)
            year_range = f"{min(bottomt_means.keys())}-{max(bottomt_means.keys())}"
            species_info[1] = {
                "variable": "bottomT (bottom temperature)",
                "quarters": "Q4 (Oct-Dec)",
                "tref_mean": f"{tref:.2f}",
                "year_range": year_range,
                "status": f"OK ({len(bottomt_means)} years)",
            }
            print(f"  bottomT Q4: {len(bottomt_means)} years, tref={tref:.2f}, range={year_range}")

    if not rows_by_species:
        print("ERROR: no thermal data available after degradation")
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
