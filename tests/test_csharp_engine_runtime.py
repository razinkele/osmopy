"""Runtime and assembly resolution for the C# engine (spec section 9.4). Pure Python, no CLR."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

import osmose.engine_cs as cs


def _fake_dotnet(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    exe = root / "dotnet"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return root


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No DOTNET_ROOT, an empty HOME and a PATH without dotnet."""
    monkeypatch.delenv("DOTNET_ROOT", raising=False)
    monkeypatch.delenv(cs.ENGINE_DIR_ENV, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return home


def test_dotnet_root_env_wins(clean_env, tmp_path, monkeypatch):
    env_root = _fake_dotnet(tmp_path / "envroot")
    _fake_dotnet(clean_env / ".dotnet")
    monkeypatch.setenv("DOTNET_ROOT", str(env_root))
    assert cs.resolve_dotnet_root() == env_root


def test_home_dotnet_used_when_env_unset(clean_env):
    home_root = _fake_dotnet(clean_env / ".dotnet")
    assert cs.resolve_dotnet_root() == home_root


def test_env_root_without_dotnet_falls_through(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DOTNET_ROOT", str(tmp_path / "nothing-here"))
    home_root = _fake_dotnet(clean_env / ".dotnet")
    assert cs.resolve_dotnet_root() == home_root


def test_dotnet_on_path_resolved_through_symlink(clean_env, tmp_path, monkeypatch):
    real = _fake_dotnet(tmp_path / "opt" / "dotnet")
    link_dir = tmp_path / "usrbin"
    link_dir.mkdir()
    (link_dir / "dotnet").symlink_to(real / "dotnet")
    monkeypatch.setenv("PATH", str(link_dir))
    assert cs.resolve_dotnet_root() == real


def test_none_when_no_runtime(clean_env):
    assert cs.resolve_dotnet_root() is None


def test_assembly_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv(cs.ENGINE_DIR_ENV, str(tmp_path))
    assert cs.assembly_dir() == tmp_path


def test_assembly_dir_default_is_package_bin(monkeypatch):
    monkeypatch.delenv(cs.ENGINE_DIR_ENV, raising=False)
    assert cs.assembly_dir() == Path(cs.__file__).resolve().parent / "bin"


def test_unavailable_names_the_missing_assembly_and_the_build_command(
    clean_env, monkeypatch, tmp_path
):
    monkeypatch.setenv(cs.ENGINE_DIR_ENV, str(tmp_path / "nobuild"))
    ok, why = cs.csharp_engine_available()
    if "pythonnet" in why:
        pytest.skip("pythonnet not installed in this environment")
    assert not ok
    assert "Osmose.Engine.dll" in why and cs.BUILD_COMMAND in why


def test_unavailable_without_runtime(clean_env, monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("Osmose.Engine.dll", "Osmose.Engine.runtimeconfig.json"):
        (bin_dir / name).write_text("")
    monkeypatch.setenv(cs.ENGINE_DIR_ENV, str(bin_dir))
    ok, why = cs.csharp_engine_available()
    if "pythonnet" in why:
        pytest.skip("pythonnet not installed in this environment")
    assert not ok
    assert "no .NET runtime" in why
