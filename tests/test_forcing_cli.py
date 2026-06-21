# tests/test_forcing_cli.py
import numpy as np
import xarray as xr

from scripts.convert_cmems_forcing import _run


def _write_bgc(path):
    ds = xr.Dataset(
        {
            "phyc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 10.0),
            "zooc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 5.0),
        },
        coords={
            "time": np.arange(12),
            "latitude": np.linspace(66, 54, 4),
            "longitude": np.linspace(10, 30, 5),
        },
    )
    ds.to_netcdf(str(path))


def _grid_cfg():
    return {
        "grid.nlon": "10",
        "grid.nlat": "8",
        "grid.upleft.lat": "66",
        "grid.upleft.lon": "10",
        "grid.lowright.lat": "54",
        "grid.lowright.lon": "30",
    }


def test_cli_run_ltl(tmp_path, monkeypatch):
    src = tmp_path / "bgc.nc"
    _write_bgc(src)
    out = tmp_path / "ltl.nc"
    # _run takes a pre-resolved config dict + grid_file to stay unit-testable
    rc = _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(out), grid_file=None)
    assert rc == 0
    assert out.exists()
    reopened = xr.open_dataset(out)
    assert "Diatoms" in reopened.data_vars
    reopened.close()


def test_cli_run_missing_vars_returns_nonzero(tmp_path):
    src = tmp_path / "bad.nc"
    xr.Dataset(
        {"o2": (["time", "latitude", "longitude"], np.ones((12, 4, 5)))},
        coords={
            "time": np.arange(12),
            "latitude": np.linspace(66, 54, 4),
            "longitude": np.linspace(10, 30, 5),
        },
    ).to_netcdf(str(src))
    rc = _run(
        source=str(src), config=_grid_cfg(), kind="ltl", out=str(tmp_path / "x.nc"), grid_file=None
    )
    assert rc != 0


def test_cli_resolves_config_directory(tmp_path):
    # A directory containing a single *all-parameters*.csv master resolves cleanly.
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    (cfgdir / "x_all-parameters.csv").write_text(
        "grid.nlon ; 10\ngrid.nlat ; 8\ngrid.upleft.lat ; 66\ngrid.upleft.lon ; 10\n"
        "grid.lowright.lat ; 54\ngrid.lowright.lon ; 30\n"
    )
    src = tmp_path / "bgc.nc"
    _write_bgc(src)
    rc = _run(
        source=str(src),
        config=str(cfgdir),
        kind="ltl",
        out=str(tmp_path / "ltl.nc"),
        grid_file=None,
    )
    assert rc == 0


def test_cli_refuses_clobber_without_force(tmp_path):
    src = tmp_path / "bgc.nc"
    _write_bgc(src)
    out = tmp_path / "ltl.nc"
    assert _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(out), grid_file=None) == 0
    # second run without force -> FileExistsError caught -> nonzero
    assert _run(source=str(src), config=_grid_cfg(), kind="ltl", out=str(out), grid_file=None) != 0
    # with force -> succeeds
    assert (
        _run(
            source=str(src),
            config=_grid_cfg(),
            kind="ltl",
            out=str(out),
            grid_file=None,
            force=True,
        )
        == 0
    )
