# osmose/forcing/io.py
"""NetCDF writers for OSMOSE forcing datasets."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def write_ltl(ds: xr.Dataset, path: Path | str, *, overwrite: bool = False) -> Path:
    """Write an LTL forcing dataset to NetCDF; returns the path.

    Refuses to clobber an existing file unless overwrite=True (the convert CLI
    is a new write surface; the in-tree Baltic forcing asset must not be
    silently destroyed). The MCP wrapper passes overwrite=True to preserve its
    always-regenerate behavior.
    """
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True (CLI: --force) to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(out))
    return out


def write_physics(
    dsets: dict[str, xr.Dataset],
    out_dir: Path | str,
    prefix: str = "baltic",
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write each physics dataset to f'{prefix}_{name}.nc'; returns {name: path}.

    Returns a name->path mapping (not a bare list) so consumers get the
    explicit pairing. Refuses to clobber unless overwrite=True.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, ds in dsets.items():
        fpath = out / f"{prefix}_{name}.nc"
        if fpath.exists() and not overwrite:
            raise FileExistsError(
                f"{fpath} exists; pass overwrite=True (CLI: --force) to replace it"
            )
        ds.to_netcdf(str(fpath))
        paths[name] = fpath
    return paths
