"""Doc-freshening assertions: version bump + CHANGELOG 0.13.0 cut + README currency."""

from __future__ import annotations

import pathlib

_REPO = pathlib.Path(__file__).resolve().parent.parent


def test_version_is_0_13_0():
    from osmose import __version__

    assert __version__ == "0.13.0"


def test_changelog_has_dated_0_13_0_section():
    cl = (_REPO / "CHANGELOG.md").read_text()
    assert "## [0.13.0] - 2026-06-14" in cl
    # A fresh empty [Unreleased] remains above it.
    assert "## [Unreleased]" in cl
    assert cl.index("## [Unreleased]") < cl.index("## [0.13.0]")


def test_readme_names_sensitivity_page():
    rm = (_REPO / "README.md").read_text()
    assert "Sensitivity" in rm
