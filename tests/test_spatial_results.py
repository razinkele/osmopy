"""Unit tests for the Spatial Results page."""


def test_spatial_results_ui_returns_div():
    from ui.pages.spatial_results import spatial_results_ui

    result = spatial_results_ui()
    # Should return a Shiny Tag (div)
    assert hasattr(result, "attrs"), "spatial_results_ui should return a Shiny Tag"


def test_spatial_results_server_callable():
    from ui.pages.spatial_results import spatial_results_server

    assert callable(spatial_results_server)
