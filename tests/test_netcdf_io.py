"""NetCDF is opened without xarray's backend-plugin scan (osmose/netcdf_io.py).

Naming no engine, or naming it by string, makes xarray list every installed backend plugin; in
the production environment that imported rioxarray -> rasterio/GDAL on the first open, about
0.85 s and 200 MB per process (measured 2026-11-26). The netCDF4 backend is passed as a class.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from osmose.engine._netcdf import open_dataset_safe
from osmose.netcdf_io import NC_BACKEND, open_dataset


@pytest.fixture
def nc(tmp_path):
    path = tmp_path / "t.nc"
    xr.Dataset({"v": (("y", "x"), np.arange(6.0).reshape(2, 3))}).to_netcdf(path, engine="netcdf4")
    return path


@pytest.fixture
def no_plugin_scan(monkeypatch):
    from xarray.backends import plugins

    def boom(*_a, **_k):
        raise AssertionError("xarray scanned its backend plugins")

    monkeypatch.setattr(plugins, "list_engines", boom)


def test_the_scan_is_what_this_guards(nc, no_plugin_scan):
    """The control arm: a plain open does scan the plugins (so the guard below can fail)."""
    with pytest.raises(AssertionError, match="scanned"):
        xr.open_dataset(nc)


def test_the_osmose_openers_do_not_scan(nc, no_plugin_scan):
    assert NC_BACKEND is not None
    with open_dataset(nc) as ds:
        assert float(ds["v"].sum()) == 15.0
    assert float(open_dataset_safe(nc)["v"].sum()) == 15.0


def test_no_osmose_module_opens_netcdf_by_name():
    """Every reader goes through osmose.netcdf_io (a new bare xr.open_dataset would scan)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "osmose"
    bare = [
        str(p.relative_to(root.parent))
        for p in root.rglob("*.py")
        if p.name != "netcdf_io.py" and "xr.open_dataset(" in p.read_text(encoding="utf-8")
    ]
    assert bare == []
