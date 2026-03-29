"""Tests for OsmoseResults strict mode."""

from pathlib import Path

import pytest

from osmose.results import OsmoseResults


def test_strict_raises_on_missing_dir(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError when output_dir does not exist."""
    r = OsmoseResults(tmp_path / "nonexistent", strict=True)
    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        r.biomass()


def test_strict_raises_on_no_matching_files(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError when no files match the pattern."""
    r = OsmoseResults(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="No files matching"):
        r.biomass()


def test_strict_raises_on_no_spectrum_files(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError for size_spectrum when no files match."""
    r = OsmoseResults(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="No files matching"):
        r.size_spectrum()


def test_strict_raises_on_2d_missing(tmp_path: Path) -> None:
    """strict=True raises FileNotFoundError for 2D output methods."""
    r = OsmoseResults(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="No files matching"):
        r.biomass_by_age()


def test_non_strict_returns_empty_df(tmp_path: Path) -> None:
    """Default (strict=False) still returns empty DataFrame for backwards compat."""
    r = OsmoseResults(tmp_path / "nonexistent")
    df = r.biomass()
    assert df.empty


def test_strict_default_is_false(tmp_path: Path) -> None:
    """Verify strict defaults to False."""
    r = OsmoseResults(tmp_path / "nonexistent")
    assert r.strict is False
