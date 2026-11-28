"""Unit tests for the Spatial Results page."""


def test_spatial_results_ui_returns_div():
    from ui.pages.spatial_results import spatial_results_ui

    result = spatial_results_ui()
    # Should return a Shiny Tag (div)
    assert hasattr(result, "attrs"), "spatial_results_ui should return a Shiny Tag"


def test_spatial_results_server_callable():
    from ui.pages.spatial_results import spatial_results_server

    assert callable(spatial_results_server)


def test_nc_label_biomass():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_biomass_Sp0.nc") == "Biomass"


def test_nc_label_ltl():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_ltlBiomass.nc") == "LTL"


def test_nc_label_abundance():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_abundance_Sp0.nc") == "Abundance"


def test_nc_label_yield():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_yield_Sp0.nc") == "Yield"


def test_nc_label_size():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_sizeSpectrum.nc") == "Size"


def test_nc_label_meantl():
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_meanTL.nc") == "Trophic Level"


def test_nc_label_unknown():
    from ui.pages.spatial_results import _nc_label

    label = _nc_label("osm_custom_output.nc")
    assert label == "Osm Custom Output"


def test_nc_label_ltl_before_biomass():
    """ltlBiomass should match LTL, not Biomass — ordering invariant."""
    from ui.pages.spatial_results import _nc_label

    assert _nc_label("osm_ltlBiomass.nc") == "LTL"


def test_spatial_nc_files_keeps_only_maps(tmp_path):
    """Only NetCDF files with a (time, lat, lon) variable are offered as maps:
    yieldByFishery and the combined per-species file are .nc but not spatial."""
    import numpy as np
    import xarray as xr

    from ui.pages.spatial_results import _spatial_nc_files_in

    xr.Dataset(
        {"spatial_biomass": (("time", "species", "lat", "lon"), np.zeros((2, 1, 3, 4)))}
    ).to_netcdf(tmp_path / "baltic_spatial_biomass_Simu0.nc")
    xr.Dataset({"yield": (("time", "fishery"), np.zeros((2, 3)))}).to_netcdf(
        tmp_path / "baltic_yieldByFishery_Simu0.nc"
    )
    (tmp_path / "broken.nc").write_bytes(b"not a netcdf file")

    names = ["baltic_spatial_biomass_Simu0.nc", "baltic_yieldByFishery_Simu0.nc", "broken.nc"]
    assert _spatial_nc_files_in(tmp_path, names) == ["baltic_spatial_biomass_Simu0.nc"]
