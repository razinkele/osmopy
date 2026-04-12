"""Tests for pure helpers in ui/pages/spatial_results.py."""

from ui.pages.spatial_results import _nc_label


# ---------------------------------------------------------------------------
# _nc_label: known _NC_LABELS keys → exact mapped labels
# ---------------------------------------------------------------------------


def test_nc_label_biomass():
    """Filename containing 'biomass' returns the mapped label 'Biomass'."""
    assert _nc_label("biomass_sp0.nc") == "Biomass"


def test_nc_label_ltl():
    """Filename containing 'ltl' returns the mapped label 'LTL'."""
    assert _nc_label("ltl_plankton.nc") == "LTL"


def test_nc_label_abundance():
    """Filename containing 'abundance' returns the mapped label 'Abundance'."""
    assert _nc_label("abundance_sp1.nc") == "Abundance"


def test_nc_label_yield():
    """Filename containing 'yield' returns the mapped label 'Yield'."""
    assert _nc_label("yield_total.nc") == "Yield"


def test_nc_label_size():
    """Filename containing 'size' returns the mapped label 'Size'."""
    assert _nc_label("size_at_age.nc") == "Size"


def test_nc_label_meantl():
    """Filename containing 'meantl' returns the mapped label 'Trophic Level'."""
    assert _nc_label("meantl_sp3.nc") == "Trophic Level"


# ---------------------------------------------------------------------------
# _nc_label: key matching is case-insensitive on the stem
# ---------------------------------------------------------------------------


def test_nc_label_case_insensitive_upper():
    """Key matching is case-insensitive: 'BIOMASS' stem matches 'biomass' key."""
    assert _nc_label("BIOMASS_sp0.nc") == "Biomass"


def test_nc_label_case_insensitive_mixed():
    """Key matching is case-insensitive: 'Abundance_SP2' stem matches 'abundance' key."""
    assert _nc_label("Abundance_SP2.nc") == "Abundance"


# ---------------------------------------------------------------------------
# _nc_label: full paths — only the filename stem is used
# ---------------------------------------------------------------------------


def test_nc_label_full_path_known():
    """Full path with known key in filename returns the correct mapped label."""
    assert _nc_label("/tmp/data/biomass_2020.nc") == "Biomass"


def test_nc_label_full_path_unknown():
    """Full path with unknown key falls back to title-cased stem."""
    result = _nc_label("/tmp/data/mortality_rate.nc")
    assert result == "Mortality Rate"


# ---------------------------------------------------------------------------
# _nc_label: fallback — unknown stems are title-cased with underscores replaced
# ---------------------------------------------------------------------------


def test_nc_label_unknown_stem_titlecase():
    """Unknown filename: underscores replaced with spaces and title-cased."""
    assert _nc_label("ocean_temperature.nc") == "Ocean Temperature"


def test_nc_label_unknown_stem_single_word():
    """Unknown single-word filename without underscores: title-cased stem."""
    assert _nc_label("salinity.nc") == "Salinity"


def test_nc_label_no_extension():
    """Filename with no .nc extension: stem is the full string, fallback applies."""
    result = _nc_label("some_output")
    assert result == "Some Output"


# ---------------------------------------------------------------------------
# _nc_label: edge cases
# ---------------------------------------------------------------------------


def test_nc_label_empty_string():
    """Empty string input: returns an empty string (stem of '' is '')."""
    assert _nc_label("") == ""


def test_nc_label_return_type_is_str():
    """Return type is always str regardless of input."""
    assert isinstance(_nc_label("anything.nc"), str)


def test_nc_label_key_as_substring():
    """'meantl' embedded inside a longer stem is still matched."""
    assert _nc_label("spatial_meantl_weighted.nc") == "Trophic Level"
