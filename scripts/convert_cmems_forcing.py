# scripts/convert_cmems_forcing.py
"""Convert a downloaded CMEMS NetCDF into OSMOSE forcing (convert-only, no download).

Usage:
  convert_cmems_forcing.py --source bgc.nc --config data/baltic --kind ltl --out ltl.nc
  convert_cmems_forcing.py --source phy.nc --config data/baltic --kind physics --out out_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from osmose.config.reader import OsmoseConfigReader
from osmose.forcing import (
    bgc_to_ltl,
    load_ocean_mask,
    phy_to_physics,
    write_ltl,
    write_physics,
)
from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec

_log = setup_logging("osmose.forcing.cli")


def _load_config(config: str | dict) -> dict:
    """Accept a pre-resolved dict (tests) or a path to a config DIR or master file."""
    if isinstance(config, dict):
        return config
    p = Path(config)
    if p.is_dir():
        masters = sorted(p.glob("*all-parameters*.csv"))
        if len(masters) != 1:
            raise ValueError(
                f"expected exactly one *all-parameters*.csv in {p}, found {len(masters)}: "
                f"{[m.name for m in masters]}; pass the master file directly"
            )
        p = masters[0]
    reader = OsmoseConfigReader()
    return reader.read(p)  # read(master_file: Path) -> dict[str, str]


def _run(
    *,
    source: str,
    config: str | dict,
    kind: str,
    out: str,
    grid_file: str | None = None,
    year: int = 0,
    depth_integrate_m: float = 50.0,
    depth_surface_m: float = 10.0,
    prefix: str = "baltic",
    force: bool = False,
) -> int:
    """Core CLI logic; returns a process exit code."""
    try:
        cfg = _load_config(config)
        grid = GridSpec.from_config(cfg)
        # A produces 24 biweekly steps; warn (don't fail) if the config wants otherwise,
        # so a future non-24 target is visible rather than silently mismatched.
        ndt = cfg.get("simulation.time.ndtPerYear")
        if ndt is not None and str(ndt).strip() not in ("", "24"):
            _log.warning(
                "config simulation.time.ndtPerYear=%s but forcing is generated with 24 "
                "biweekly steps (sub-project A is 24-step only)",
                ndt,
            )
        src = Path(source)
        if not src.exists():
            _log.error("source file not found: %s", source)
            return 1
        mask = load_ocean_mask(Path(grid_file)) if grid_file else None
        ds = xr.open_dataset(src)
        try:
            if kind == "ltl":
                result = bgc_to_ltl(
                    ds, grid, year=year, depth_integrate_m=depth_integrate_m, ocean_mask=mask
                )
                path = write_ltl(result, out, overwrite=force)
                _log.info("wrote LTL forcing: %s", path)
                for g in result.data_vars:
                    _log.info("  %s: total=%.0f t", g, float(result[g].sum(skipna=True)))
            elif kind == "physics":
                dsets = phy_to_physics(ds, grid, year=year, depth_surface_m=depth_surface_m)
                if not dsets:
                    _log.error("no physics variables (thetao/so) found in source")
                    return 1
                paths = write_physics(dsets, out, prefix=prefix, overwrite=force)
                _log.info(
                    "wrote physics forcing: %s",
                    ", ".join(str(p) for p in paths.values()),
                )
            else:
                _log.error("unknown --kind %r (use 'ltl' or 'physics')", kind)
                return 2
        finally:
            ds.close()
    except (ValueError, OSError, KeyError) as exc:  # FileExistsError ⊂ OSError
        _log.error("conversion failed: %s", exc)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert a downloaded CMEMS file to OSMOSE forcing")
    p.add_argument("--source", required=True, help="downloaded CMEMS NetCDF (BGC or PHY)")
    p.add_argument(
        "--config",
        required=True,
        help="config directory or master config file (for the target grid)",
    )
    p.add_argument("--kind", required=True, choices=["ltl", "physics"])
    p.add_argument("--out", required=True, help="output file (ltl) or directory (physics)")
    p.add_argument("--grid-file", default=None, help="grid NetCDF for the ocean mask (optional)")
    p.add_argument("--year", type=int, default=0, help="year to extract (0 = all available)")
    p.add_argument(
        "--depth-integrate",
        type=float,
        default=50.0,
        help="LTL integration depth (m); default 50 is Baltic-tuned, set per region",
    )
    p.add_argument(
        "--depth-surface",
        type=float,
        default=10.0,
        help="physics surface-layer depth (m); default 10 is Baltic-tuned",
    )
    p.add_argument("--prefix", default="baltic", help="physics output filename prefix")
    p.add_argument("--force", action="store_true", help="overwrite existing output file(s)")
    a = p.parse_args(argv)
    return _run(
        source=a.source,
        config=a.config,
        kind=a.kind,
        out=a.out,
        grid_file=a.grid_file,
        year=a.year,
        depth_integrate_m=a.depth_integrate,
        depth_surface_m=a.depth_surface,
        prefix=a.prefix,
        force=a.force,
    )


if __name__ == "__main__":
    sys.exit(main())
