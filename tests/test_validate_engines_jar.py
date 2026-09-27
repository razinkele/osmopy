"""JAR discovery for the Python-vs-Java validation script.

``scripts/validate_engines.py`` is the Python-vs-Java cross-check, and
``resolve_jar`` is the gate in front of it: get the discovery wrong and the
script is simply unrunnable, which is how it sat for a while. It used to hard-code
``osmose_4.3.3-jar-with-dependencies.jar`` (underscore), while
``apptainer/osmose.def`` downloads ``osmose-4.3.3-...`` (hyphen) and the 4.4.1
call sites elsewhere in the repo use a hyphen too. The JAR is gitignored, so
**nothing in-tree pins the true spelling** and no fixture can assert it — which
is exactly why the precedence chain, rather than any filename, is what gets
tested here.

These tests never need a real JAR: ``resolve_jar`` does path resolution only and
deliberately does not check existence (``main()`` does that separately, so it can
report the offending path). An empty file with a ``.jar`` suffix is therefore a
faithful fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_engines as ve


@pytest.fixture(autouse=True)
def _no_ambient_jar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise a developer's real $OSMOSE_JAR.

    Without this, anyone who has the env var set (the documented way to run the
    cross-check) would see the glob and None cases resolve to their own JAR, and
    the suite would pass or fail depending on the shell it was launched from.
    """
    monkeypatch.delenv("OSMOSE_JAR", raising=False)


def _touch_jar(directory: Path, name: str) -> Path:
    """Create an empty file with a ``.jar`` name and return its path.

    Empty is faithful: ``resolve_jar`` does path resolution only and never opens the
    archive, so a zero-byte file exercises it exactly as a real 24 MB jar would.
    """
    directory.mkdir(parents=True, exist_ok=True)
    jar = directory / name
    jar.touch()
    return jar


# ---------------------------------------------------------------------------
# Precedence: explicit > $OSMOSE_JAR > glob
# ---------------------------------------------------------------------------


def test_explicit_wins_over_env_and_glob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--jar beats both lower-precedence sources, even when all three exist."""
    monkeypatch.setenv("OSMOSE_JAR", str(_touch_jar(tmp_path / "env", "from-env.jar")))
    monkeypatch.setattr(ve, "JAR_DIR", tmp_path / "dir")
    _touch_jar(tmp_path / "dir", "from-glob.jar")

    explicit = tmp_path / "explicit" / "chosen.jar"
    assert ve.resolve_jar(str(explicit)) == explicit


def test_env_wins_over_glob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """$OSMOSE_JAR beats the directory glob when no --jar is passed."""
    from_env = _touch_jar(tmp_path / "env", "from-env.jar")
    monkeypatch.setenv("OSMOSE_JAR", str(from_env))
    monkeypatch.setattr(ve, "JAR_DIR", tmp_path / "dir")
    _touch_jar(tmp_path / "dir", "from-glob.jar")

    assert ve.resolve_jar(None) == from_env


def test_glob_used_when_nothing_else_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The last resort in the chain: glob ``JAR_DIR`` when no --jar and no env var.

    This is the path a developer who simply drops a jar into ``osmose-java/`` takes,
    and the only one that works with no arguments and no environment setup.
    """
    jar = _touch_jar(tmp_path / "dir", "osmose-4.4.1-jar-with-dependencies.jar")
    monkeypatch.setattr(ve, "JAR_DIR", tmp_path / "dir")

    assert ve.resolve_jar(None) == jar


# ---------------------------------------------------------------------------
# Filename-agnosticism — the actual regression this function fixed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "osmose_4.3.3-jar-with-dependencies.jar",  # underscore (the old hard-coded name)
        "osmose-4.3.3-jar-with-dependencies.jar",  # hyphen (apptainer/osmose.def)
        "osmose-4.4.1-jar-with-dependencies.jar",  # hyphen, the 4.4.1 call sites
        "osmose.jar",  # bare
    ],
)
def test_glob_is_filename_agnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Any *.jar is found. The old hard-coded filename is why this matters."""
    jar = _touch_jar(tmp_path / "dir", name)
    monkeypatch.setattr(ve, "JAR_DIR", tmp_path / "dir")

    assert ve.resolve_jar(None) == jar


def test_non_jar_files_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory holding only non-JARs resolves to None, not to a stray file."""
    d = tmp_path / "dir"
    d.mkdir()
    (d / "README.md").touch()
    (d / "osmose.jar.sha256").touch()
    monkeypatch.setattr(ve, "JAR_DIR", d)

    assert ve.resolve_jar(None) is None


# ---------------------------------------------------------------------------
# Multiple candidates and the empty case
# ---------------------------------------------------------------------------


def test_multiple_jars_pick_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With several JARs the choice is sorted-first, so repeat runs agree.

    Asserted because ``Path.glob`` order is filesystem-dependent: without the
    ``sorted()`` a developer with both a 4.3.3 and a 4.4.1 JAR could get a
    different engine on different machines and never be told which one ran.
    ``main()`` prints the resolved path for exactly that reason.
    """
    monkeypatch.setattr(ve, "JAR_DIR", tmp_path / "dir")
    for name in ("osmose-4.4.1.jar", "osmose-4.3.3.jar", "aaa.jar"):
        _touch_jar(tmp_path / "dir", name)

    first = ve.resolve_jar(None)
    assert first == tmp_path / "dir" / "aaa.jar"
    assert ve.resolve_jar(None) == first


def test_missing_dir_resolves_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """osmose-java/ is gitignored, so on a fresh clone it does not exist at all.

    ``Path.glob`` on a missing directory yields nothing rather than raising, so
    this must return None — the fresh-clone path into ``main()``'s error message.
    """
    monkeypatch.setattr(ve, "JAR_DIR", tmp_path / "does-not-exist")

    assert ve.resolve_jar(None) is None


def test_empty_string_explicit_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--jar ""`` is falsy and must fall through rather than return Path("").

    Path("") normalises to Path("."), which exists as a directory — so returning
    it would sail past ``main()``'s ``.exists()`` check and hand a directory to
    ``java -jar``.
    """
    from_env = _touch_jar(tmp_path / "env", "from-env.jar")
    monkeypatch.setenv("OSMOSE_JAR", str(from_env))

    assert ve.resolve_jar("") == from_env


def test_resolve_does_not_require_existence(tmp_path: Path) -> None:
    """A non-existent explicit path comes back as-is, for main() to report.

    The existence check lives in ``main()``, not here, so that its error message
    can name the path the user actually asked for.
    """
    missing = tmp_path / "nope.jar"
    assert ve.resolve_jar(str(missing)) == missing
    assert not missing.exists()
