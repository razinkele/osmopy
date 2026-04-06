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
