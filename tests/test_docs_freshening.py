"""Doc-freshening assertions: version is semver + CHANGELOG has the released section + README currency."""

from __future__ import annotations

import pathlib
import re

_REPO = pathlib.Path(__file__).resolve().parent.parent


def test_version_is_semver():
    from osmose import __version__

    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), f"non-semver version: {__version__}"


def test_changelog_has_dated_section_for_current_version():
    from osmose import __version__

    cl = (_REPO / "CHANGELOG.md").read_text()
    # the released version has a dated section, and it is the latest (top) one
    assert re.search(rf"## \[{re.escape(__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}", cl)
    assert cl.index(f"## [{__version__}]") == cl.index("## [")


def test_readme_names_sensitivity_page():
    rm = (_REPO / "README.md").read_text()
    assert "Sensitivity" in rm
