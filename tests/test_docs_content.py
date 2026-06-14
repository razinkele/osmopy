"""Unit tests for osmose.docs_content (doc loader + changelog parser)."""

from __future__ import annotations

import pytest

from osmose.docs_content import latest_changelog_entry, read_doc

_SAMPLE = """# Changelog

## [Unreleased]

## [0.13.0] - 2026-06-14

### Added
- Thing one
- Thing two

## [0.12.0] - 2026-05-08

### Added
- Old thing
"""


def test_read_doc_readme_has_marker():
    assert "OSMOSE" in read_doc("readme")


def test_read_doc_changelog_has_marker():
    text = read_doc("changelog")
    assert "Changelog" in text and "## [0.13.0]" in text


def test_read_doc_unknown_kind_raises():
    with pytest.raises(ValueError):
        read_doc("bogus")


def test_read_doc_missing_file_fallback(tmp_path):
    assert read_doc("readme", root=tmp_path) == "_Documentation unavailable._"


def test_latest_entry_parses_first_dated_section():
    e = latest_changelog_entry(_SAMPLE)
    assert e["version"] == "0.13.0"
    assert e["date"] == "2026-06-14"
    assert "Thing one" in e["body"] and "Thing two" in e["body"]
    assert "Old thing" not in e["body"]  # stops at the next ## section
    assert not e["body"].startswith("## [")  # heading line excluded


def test_latest_entry_date_optional():
    e = latest_changelog_entry("## [0.13.0]\n\n### Added\n- x\n")
    assert e["version"] == "0.13.0" and e["date"] is None


def test_latest_entry_unreleased_only_fallback():
    e = latest_changelog_entry("# Changelog\n\n## [Unreleased]\n- foo\n")
    assert e["version"] is None and "foo" in e["body"]


def test_latest_entry_empty_fallback():
    e = latest_changelog_entry("# Changelog\n\nnothing here\n")
    assert e["body"] == "No release notes yet."
