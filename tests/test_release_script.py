import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "release", Path(__file__).parent.parent / "scripts" / "release.py"
)
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)

_HEADER = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n"
_RELEASED = (
    "## [1.0.1] - 2026-01-02\n\n### Bug Fixes\n\n- hand-edited note (abc1234)\n\n"
    "## [1.0.0] - 2026-01-01\n\n### Features\n\n- first (def5678)\n"
)
_NEW = "## [1.0.2] - 2026-02-03\n\n### Bug Fixes\n\n- **x:** y (0123456)\n\n"


def test_splice_section_inserts_above_newest_release_and_changes_nothing_else():
    existing = _HEADER + "\n" + _RELEASED
    out = release.splice_section(existing, _NEW)
    assert out.index("## [1.0.2]") < out.index("## [1.0.1]")
    assert out.replace(_NEW, "", 1) == existing  # insertions only


def test_splice_section_without_any_release_appends_after_header():
    out = release.splice_section(_HEADER, _NEW)
    assert out.startswith(_HEADER) and out.endswith(_NEW)


def test_splice_section_refuses_a_duplicate_version():
    existing = release.splice_section(_HEADER + "\n" + _RELEASED, _NEW)
    with pytest.raises(ValueError, match=r"1\.0\.2"):
        release.splice_section(existing, _NEW.replace("2026-02-03", "2026-02-04"))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    version = tmp_path / "__version__.py"
    version.write_text('__version__ = "1.0.1"\n')
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_HEADER + "\n" + _RELEASED)
    monkeypatch.setattr(release, "VERSION_FILE", version)
    monkeypatch.setattr(release, "CHANGELOG_FILE", changelog)
    monkeypatch.setattr(release, "git_tags", lambda: ["v1.0.0", "v1.0.1"])
    # Every range returns the same commit: the old whole-file regeneration would then rewrite the
    # released sections (their hand-edited text is not in git), which is what this guards against.
    commit = {"sha": "0123456", "type": "fix", "scope": "x", "description": "y"}
    monkeypatch.setattr(release, "git_log_between", lambda *a, **k: [commit])
    return version, changelog


def test_bump_leaves_released_sections_byte_identical(staged):
    version, changelog = staged
    before = changelog.read_text()
    release.main(["patch"])
    after = changelog.read_text()
    assert '"1.0.2"' in version.read_text()
    assert after.endswith(_RELEASED)
    assert after.startswith(before.split("## [", 1)[0] + "## [1.0.2] - ")
    assert len(after.splitlines()) > len(before.splitlines())


def test_rerun_refuses_and_writes_nothing(staged):
    version, changelog = staged
    release.main(["patch"])
    version.write_text('__version__ = "1.0.1"\n')  # as if the bump were reverted, not the log
    before = changelog.read_text()
    with pytest.raises(SystemExit):
        release.main(["patch"])
    assert changelog.read_text() == before
    assert '"1.0.1"' in version.read_text()
