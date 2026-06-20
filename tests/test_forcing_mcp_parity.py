# tests/test_forcing_mcp_parity.py
import importlib.util

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from osmose.forcing import bgc_to_ltl, load_ocean_mask
from osmose.maps.builder import GridSpec

# The MCP wrapper module imports fastmcp/copernicusmarine, absent in the clean venv.
_HAS_MCP = importlib.util.find_spec("fastmcp") is not None and (
    importlib.util.find_spec("copernicusmarine") is not None
)
BALTIC = GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
_BALTIC_GRID_NC = Path("data/baltic/baltic_grid.nc")


def _bgc(tmp_path):
    lat = np.linspace(66, 54, 6)
    lon = np.linspace(10, 30, 7)
    ds = xr.Dataset(
        {
            "phyc": (
                ["time", "latitude", "longitude"],
                np.abs(np.random.default_rng(0).random((12, 6, 7))) * 10,
            ),
            "zooc": (
                ["time", "latitude", "longitude"],
                np.abs(np.random.default_rng(1).random((12, 6, 7))) * 5,
            ),
            "nppv": (
                ["time", "latitude", "longitude"],
                np.abs(np.random.default_rng(2).random((12, 6, 7))) * 100,
            ),
        },
        coords={"time": np.arange(12), "latitude": lat, "longitude": lon},
    )
    path = tmp_path / "bgc.nc"
    ds.to_netcdf(str(path))
    return path


@pytest.mark.skipif(not _HAS_MCP, reason="MCP deps (fastmcp/copernicusmarine) not installed")
def test_mcp_wrapper_matches_core(tmp_path):
    import mcp_servers.copernicus.server as srv

    src = _bgc(tmp_path)
    out_file = tmp_path / "mcp_ltl.nc"
    srv.generate_osmose_ltl(source_bgc_file=str(src), output_file=str(out_file))
    mcp_ds = xr.open_dataset(out_file)

    # Reuse the wrapper's OWN Baltic grid so the two can never silently disagree.
    # (BALTIC above must equal it; cell_volume_m3's mid-lat = (66+54)/2 = 60.0, which
    # is exactly the old hardcoded cos(radians(60)) — so this Baltic-only parity check
    # cannot detect the cos-factor GENERALIZATION, only Baltic-grid value drift.)
    assert srv._baltic_grid() == BALTIC
    # The MCP wrapper applies the Baltic ocean mask (land -> NaN). Pass the SAME
    # mask to the core call so both sides NaN identical land cells; otherwise the
    # core leaves land cells as real biomass and nan_to_num(0) != real value fails.
    mask = load_ocean_mask(_BALTIC_GRID_NC)
    with xr.open_dataset(src) as core_src:
        core_ds = bgc_to_ltl(core_src, srv._baltic_grid(), ocean_mask=mask)
    for g in [
        "Diatoms",
        "Dinoflagellates",
        "Microzooplankton",
        "Mesozooplankton",
        "Macrozooplankton",
        "Benthos",
    ]:
        assert np.allclose(
            np.nan_to_num(mcp_ds[g].values), np.nan_to_num(core_ds[g].values), rtol=1e-6
        )
    mcp_ds.close()
