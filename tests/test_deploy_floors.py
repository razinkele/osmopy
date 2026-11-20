"""deploy.sh's Step 0 changes a package only when it is missing or below its floor.

The prod env is shared with the server's other apps and managed with micromamba (DEPLOY.md, "The
shared environment"). Measured 2026-11-21, the Step 0 of v1.11.0 would have replaced packages that
already met their floors: ``pip install --upgrade "cma>=4.0" ...`` moves to the newest release
(cma 4.0.0 -> 4.6.0), and a "not exactly 1.9.2" guard would have reinstalled shiny_deckgl 1.9.2
over the 1.16.1 that two other apps need.

The guards are tested by running the exact snippets from deploy.sh against faked installed
versions, so a guard that stops being a floor test fails here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy.sh"


def _step0() -> list[str]:
    text = DEPLOY.read_text()
    return text[text.index("# --- Step 0:") : text.index("# --- Step 1:")].splitlines()


def _code(lines: list[str]) -> list[str]:
    return [s for s in (ln.strip() for ln in lines) if s and not s.startswith("#")]


def test_every_upgrade_in_step0_is_behind_a_guard():
    depth = 0
    upgrades = 0
    for line in _code(_step0()):
        if line.startswith("if ") and line.endswith("then"):
            depth += 1
        elif line == "fi":
            depth -= 1
        elif "PIP_INSTALL[@]" in line and "--upgrade" in line:
            upgrades += 1
            assert depth > 0, f"unconditional --upgrade in Step 0: {line}"
    assert upgrades, "found no guarded --upgrade: the test no longer sees Step 0's pip lines"


def test_the_floors_are_installed_without_upgrade():
    floors = [ln for ln in _code(_step0()) if "PIP_INSTALL[@]" in ln and '"cma>=4.0"' in ln]
    assert len(floors) == 1, floors
    assert "--upgrade" not in floors[0]
    for spec in ('"shinyswatch>=0.11"', '"shinywidgets>=0.7"', '"pyarrow>=14"', '"packaging"'):
        assert spec in floors[0]


def test_pip_never_falls_back_to_a_user_install():
    # Without --no-user, pip installs into shiny's home when site-packages is not writable, and
    # the floor check then passes on that copy instead of the deploy stopping.
    code = _code(DEPLOY.read_text().splitlines())
    definition = [ln for ln in code if ln.startswith("PIP_INSTALL=(")]
    assert len(definition) == 1 and "--no-user" in definition[0], definition
    pip_install = re.compile(r"(\bpip3?|SHINY_PIP)[\"']?\s+install\b")
    raw = [ln for ln in code if pip_install.search(ln) and "PIP_INSTALL" not in ln]
    assert not raw, f"pip install outside PIP_INSTALL: {raw}"


def _guard(marker: str) -> str:
    """The Python of the Step 0 guard that names ``marker``."""
    prefix = 'if ! as_shiny "$SHINY_PYTHON" -c \''
    hits = [ln for ln in _code(_step0()) if ln.startswith(prefix) and marker in ln]
    assert len(hits) == 1, (marker, hits)
    return hits[0][len(prefix) :].rsplit("' 2>/dev/null", 1)[0]


_SHIM = """
import importlib.metadata as md, sys, types
installed = {installed!r}
def version(name):
    if name not in installed:
        raise md.PackageNotFoundError(name)
    return installed[name]
md.version = version
sys.modules["pyvis"] = (
    types.SimpleNamespace(__version__=installed["pyvis"]) if "pyvis" in installed else None
)
"""


def _skips_install(snippet: str, installed: dict[str, str]) -> bool:
    """deploy.sh installs when the guard fails (``if ! guard; then install``)."""
    code = _SHIM.format(installed=installed) + "\n" + snippet
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


@pytest.mark.parametrize(
    ("installed", "skips"),
    [
        ({"shiny_deckgl": "1.16.1"}, True),  # what the other apps installed: never downgraded
        ({"shiny_deckgl": "1.9.2"}, True),
        ({"shiny_deckgl": "1.10.0"}, True),  # a string comparison would call this older
        ({"shiny_deckgl": "1.9.1"}, False),
        ({}, False),
    ],
)
def test_the_deckgl_guard_installs_only_below_its_floor(installed, skips):
    assert _skips_install(_guard("shiny_deckgl"), installed) is skips


@pytest.mark.parametrize(
    ("installed", "skips"),
    [
        ({"pyvis": "5.0.0"}, True),
        ({"pyvis": "4.2"}, True),
        ({"pyvis": "0.3.2"}, False),  # PyPI's unrelated lineage
        ({}, False),
    ],
)
def test_the_pyvis_guard_installs_only_below_its_floor(installed, skips):
    assert _skips_install(_guard("pyvis"), installed) is skips
