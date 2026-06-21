"""Fast guard tests for the Sphinx docs build (run in the normal [dev] CI leg).

These are the cheap early signal; the authoritative check is the `-W` Sphinx
build in .github/workflows/docs.yml.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import osmose

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONF = _REPO_ROOT / "docs" / "conf.py"


def test_all_osmose_modules_import():
    """autodoc imports every module it documents; an unimportable module breaks
    the docs build. Catch it here in the fast CI leg, before the slow Sphinx job.

    Caveat: this leg runs under the [dev] extra (a superset of the docs job's
    [docs] + numba). It cannot catch a module that gains a top-level import of a
    dev-only dep (e.g. httpx, pillow) -- that is caught only by the -W docs build.
    """
    failures: list[str] = []
    # walk_packages imports each sub-PACKAGE itself (to read __path__ for
    # recursion); that import is outside the loop's try, so a raising package
    # __init__ would abort the walk with a raw traceback. onerror collects it.
    walker = pkgutil.walk_packages(
        osmose.__path__,
        "osmose.",
        onerror=lambda name: failures.append(f"{name}: package import error during walk"),
    )
    for mod in walker:
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - collect all, not just the first
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "osmose modules failed to import:\n" + "\n".join(failures)


def test_conf_py_is_import_safe_and_defines_data_globals():
    """conf.py must exec WITHOUT the [docs] extra installed (no top-level sphinx/
    furo import) and define the core data globals."""
    namespace: dict[str, object] = {}
    exec(compile(_CONF.read_text(), str(_CONF), "exec"), namespace)
    for key in ("project", "author", "release", "html_theme", "extensions"):
        assert key in namespace, f"conf.py missing data global: {key}"
    assert namespace["html_theme"] == "furo"
    assert "myst_parser" in namespace["extensions"]  # type: ignore[operator]
