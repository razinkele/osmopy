#!/usr/bin/env python
"""Download CMEMS reanalysis fields for the Baltic cod reproductive-volume test.

Pulls full-depth, multi-decade Baltic reanalysis so the RV-vs-overshoot
diagnostic (scripts/baltic_rv_overshoot_diagnostic.py) can measure the real
reproductive volume: deep-basin water with salinity >= 11 PSU AND O2 >= 2 mL/L.

  - salinity `so`  from cmems_mod_bal_phy_my_P1M-m  (PHY multiyear monthly)
  - oxygen   `o2`  from cmems_mod_bal_bgc_my_P1M-m  (BGC multiyear monthly)

Both are depth-resolved; the diagnostic extracts the deepest valid level per
cell. We download from `--depth-min` (default 40 m — the deep-basin cod
spawning sills are all deeper, so shallow levels are not needed and dropping
them roughly halves the download) to `--depth-max` (default 260 m, past the
Gotland Deep floor).

Filenames match the diagnostic's auto-glob (`*phy*so*.nc`, `*bgc*o2*.nc`).

Usage:
  # measure one year first (recommended before the full pull):
  PYTHONPATH=. .venv/bin/python scripts/download_baltic_rv_forcing.py --probe 2010
  # full multi-decade pull (year-by-year, resumable):
  PYTHONPATH=. .venv/bin/python scripts/download_baltic_rv_forcing.py --start 1993 --end 2021
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "cmems_cache" / "cmems_downloads"

# Baltic OSMOSE bounding box (matches mcp_servers/copernicus/server.py).
BBOX = dict(
    minimum_longitude=9.5, maximum_longitude=30.5, minimum_latitude=53.5, maximum_latitude=66.5
)

FIELDS = {
    "so": {"dataset_id": "cmems_mod_bal_phy_my_P1M-m", "tag": "phy_monthly_reanalysis"},
    "thetao": {"dataset_id": "cmems_mod_bal_phy_my_P1M-m", "tag": "phy_monthly_reanalysis"},
    "o2": {"dataset_id": "cmems_mod_bal_bgc_my_P1M-m", "tag": "bgc_monthly_reanalysis"},
}


def _creds() -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    u, p = os.environ.get("CMEMS_USERNAME"), os.environ.get("CMEMS_PASSWORD")
    if not u or not p:
        raise SystemExit("CMEMS_USERNAME/CMEMS_PASSWORD not set (see .env / copernicus README).")
    return u, p


def _download_year(cm, var: str, year: int, depth_min: float, depth_max: float) -> Path:
    info = FIELDS[var]
    last_day = calendar.monthrange(year, 12)[1]
    fname = f"baltic_{info['tag']}_{var}_{year}-01_{year}-12.nc"
    out = OUT_DIR / fname
    if out.exists() and out.stat().st_size > 0:
        print(f"  [skip] {fname} already present ({out.stat().st_size / 1e6:.0f} MB)")
        return out
    cm.subset(
        dataset_id=info["dataset_id"],
        variables=[var],
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-{last_day}T23:59:59",
        minimum_depth=depth_min,
        maximum_depth=depth_max,
        output_directory=str(OUT_DIR),
        output_filename=fname,
        overwrite=True,
        disable_progress_bar=False,
        **BBOX,
    )
    return out


def _report_file(path: Path) -> None:
    import xarray as xr

    if not path.exists():
        print(f"  !! not written: {path.name}")
        return
    mb = path.stat().st_size / 1e6
    with xr.open_dataset(path) as ds:
        v = list(ds.data_vars)
        nt = len(ds["time"]) if "time" in ds.dims else "?"
        nd = len(ds["depth"]) if "depth" in ds.dims else "?"
        dmax = float(ds["depth"].max()) if "depth" in ds.coords else float("nan")
        print(
            f"  {path.name}: {mb:.0f} MB  vars={v}  t={nt}  depth_levels={nd}  max_depth={dmax:.0f} m"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=1993, help="first year (inclusive)")
    ap.add_argument("--end", type=int, default=2021, help="last year (inclusive)")
    ap.add_argument(
        "--probe", type=int, default=None, help="download ONLY this one year, then report size"
    )
    ap.add_argument("--vars", nargs="+", default=["so", "o2"], choices=list(FIELDS))
    ap.add_argument("--depth-min", type=float, default=40.0)
    ap.add_argument("--depth-max", type=float, default=260.0)
    args = ap.parse_args(argv)

    u, p = _creds()
    import copernicusmarine as cm

    cm.login(username=u, password=p, force_overwrite=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    years = [args.probe] if args.probe is not None else list(range(args.start, args.end + 1))
    print(
        f"Downloading {args.vars} for years {years[0]}-{years[-1]} "
        f"(depth {args.depth_min:.0f}-{args.depth_max:.0f} m) into {OUT_DIR}"
    )

    written: list[Path] = []
    for var in args.vars:
        for year in years:
            print(f"[{var} {year}] ...")
            try:
                written.append(_download_year(cm, var, year, args.depth_min, args.depth_max))
            except Exception as exc:  # noqa: BLE001 — report and continue to next year
                print(f"  !! failed {var} {year}: {exc}")

    print("\n=== downloaded files ===")
    total = 0.0
    for path in written:
        _report_file(path)
        if path.exists():
            total += path.stat().st_size / 1e6
    print(f"total: {total:.0f} MB across {len(written)} files")
    if args.probe is not None and written:
        per_var_year = total / max(1, len([f for f in written if f.exists()]))
        span = args.end - args.start + 1
        full = per_var_year * span * len(args.vars)
        print(
            f"\nPROJECTION: ~{per_var_year:.0f} MB/variable/year -> full {args.start}-{args.end} "
            f"({span} yr x {len(args.vars)} vars) ~= {full / 1000:.1f} GB"
        )
        print(
            "If acceptable, run without --probe to pull the full range (year-by-year, resumable)."
        )
        print("To cut size ~2x, add --depth-min 60 or restrict to spawning months in a follow-up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
