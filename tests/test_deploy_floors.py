"""deploy.sh's Step 0 changes a package only when it is missing or below its floor.

The prod env is shared with the server's other apps and managed with micromamba (DEPLOY.md, "The
shared environment"). Measured 2026-11-21, the Step 0 of v1.11.0 would have replaced packages that
already met their floors: ``pip install --upgrade "cma>=4.0" ...`` moves to the newest release
(cma 4.0.0 -> 4.6.0), and a "not exactly 1.9.2" guard would have reinstalled shiny_deckgl 1.9.2
over the 1.16.1 that two other apps need.

The guards are tested by running the exact snippets from deploy.sh against faked installed
versions, so a guard that stops being a floor test fails here. The floors themselves must be the
versions CI tests (``pyproject.toml``), which since 2026-11-21 are the ones production runs.
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


def _pyproject_dependencies() -> list[str]:
    import tomllib

    pyproject = DEPLOY.parent / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["project"]["dependencies"]


def test_deploy_floors_match_the_versions_ci_tests():
    """CI tests what pyproject.toml pins; the deploy's floors must name the same versions.

    The shiny_deckgl version appears in four places (pyproject's tag, the guard's floor, the tag
    the guard installs, the final floor check) and shiny's in two; a bump that misses one leaves
    the deploy installing or accepting a version CI never ran.
    """
    deps = _pyproject_dependencies()
    (deckgl,) = [d for d in deps if d.startswith("shiny_deckgl")]
    (shiny,) = [d for d in deps if re.match(r"shiny[<>=]", d)]
    tested_deckgl = re.search(r"@v(\d+\.\d+\.\d+)$", deckgl)
    tested_shiny = re.search(r">=(\d+\.\d+\.\d+)", shiny)
    assert tested_deckgl and tested_shiny, (deckgl, shiny)

    step0 = "\n".join(_step0())
    guard_floor = re.search(r'Version\("(\d+\.\d+\.\d+)"\)', _guard("shiny_deckgl"))
    install_tags = re.findall(r"shiny_deckgl\.git@v(\d+\.\d+\.\d+)", step0)
    check_deckgl = re.search(r'"shiny_deckgl": "(\d+\.\d+\.\d+)"', step0)
    check_shiny = re.search(r'"shiny": "(\d+\.\d+\.\d+)"', step0)
    assert guard_floor and check_deckgl and check_shiny and install_tags

    assert guard_floor.group(1) == tested_deckgl.group(1)
    assert install_tags == [tested_deckgl.group(1)]
    assert check_deckgl.group(1) == tested_deckgl.group(1)
    assert check_shiny.group(1) == tested_shiny.group(1)


@pytest.mark.parametrize(
    ("installed", "skips"),
    [
        ({"shiny_deckgl": "1.16.1"}, True),  # prod's, and the floor: never reinstalled
        ({"shiny_deckgl": "1.17.0"}, True),  # newer, e.g. from another app: never downgraded
        ({"shiny_deckgl": "1.100.0"}, True),  # a string comparison would call this older
        ({"shiny_deckgl": "1.16.0"}, False),
        ({"shiny_deckgl": "1.9.2"}, False),  # the floor until 2026-11-21
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
