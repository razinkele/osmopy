"""CMEMS credentials must come from env; no default fallback is allowed."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp_servers" / "copernicus" / "server.py"


def test_server_py_has_no_hardcoded_credentials():
    src = SERVER_PATH.read_text()
    assert "Razinka@2026" not in src
    pw_line = next(
        (line for line in src.splitlines() if "CMEMS_PASSWORD" in line and "os.environ" in line),
        None,
    )
    assert pw_line is not None, "CMEMS_PASSWORD env lookup not found"
    assert not re.search(r"os\.environ\.get\(\s*['\"]CMEMS_PASSWORD['\"]\s*,\s*['\"]", pw_line), (
        f"CMEMS_PASSWORD must not have a hardcoded string default: {pw_line!r}"
    )
    user_line = next(
        (line for line in src.splitlines() if "CMEMS_USERNAME" in line and "os.environ" in line),
        None,
    )
    assert user_line is not None
    assert not re.search(r"os\.environ\.get\(\s*['\"]CMEMS_USERNAME['\"]\s*,\s*['\"]", user_line)


def _neuter_dotenv(monkeypatch):
    """Prepare to exec server.py, skipping the calling test if any of its undeclared
    MCP deps are absent.

    server.py imports copernicusmarine, fastmcp, and python-dotenv at module top, and
    none are in [dev] (kept out intentionally — copernicusmarine is heavy). On CI / the
    deploy env one or more is missing, so exec_module would ERROR. importorskip makes the
    test SKIP instead, matching test_copernicus_ltl_mask.py's module-level guard. (Guarding
    only dotenv was insufficient: once dotenv arrived transitively via fastmcp /
    pydantic-settings, the test ran and hit the missing fastmcp -> CI red.) Then stop
    load_dotenv from reading a developer's local .env."""
    pytest.importorskip("copernicusmarine")
    pytest.importorskip("fastmcp")
    dotenv = pytest.importorskip("dotenv")

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)


def test_server_py_module_globals_reflect_env(monkeypatch):
    """Load server.py via spec_from_file_location (mcp_servers is not a package)."""
    _neuter_dotenv(monkeypatch)
    monkeypatch.delenv("CMEMS_USERNAME", raising=False)
    monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
    spec = importlib.util.spec_from_file_location("_copernicus_server_test", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert getattr(mod, "CMEMS_USER", "sentinel") is None
    assert getattr(mod, "CMEMS_PASS", "sentinel") is None


def test_require_creds_raises_on_missing_env(monkeypatch):
    """The guard must raise RuntimeError, not silently pass."""
    _neuter_dotenv(monkeypatch)
    monkeypatch.delenv("CMEMS_USERNAME", raising=False)
    monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
    spec = importlib.util.spec_from_file_location("_copernicus_server_raises", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(RuntimeError, match="CMEMS_USERNAME"):
        mod._require_creds()
