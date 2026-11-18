"""The C# engine backend: a .NET library behind one ``RunSimulation`` entry point.

Design: ``docs/superpowers/specs/2026-11-17-csharp-engine-design.md``. This package is the
Python side. **Importing it never loads the CLR** (spec section 9.4): the resolution helpers
below are pure Python, and only :mod:`osmose.engine_cs.bridge` imports pythonnet, on first use.
That keeps the capability check usable in the UI and in processes that later fork, since
CoreCLR's runtime threads do not survive ``fork``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

ASSEMBLY_NAME = "Osmose.Engine"
ENGINE_DIR_ENV = "OSMOSE_CSHARP_ENGINE_DIR"
BUILD_COMMAND = "bash scripts/build_csharp_engine.sh"


def assembly_dir() -> Path:
    """Directory holding the built assembly: ``$OSMOSE_CSHARP_ENGINE_DIR``, else ``bin/`` here."""
    override = os.environ.get(ENGINE_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "bin"


def resolve_dotnet_root() -> Path | None:
    """The .NET installation to host: ``$DOTNET_ROOT``, else ``~/.dotnet``, else ``dotnet`` on
    ``PATH`` (resolved through symlinks). ``None`` when none holds a ``dotnet`` executable.

    The bridge passes the result to pythonnet explicitly, together with the runtime config:
    clr_loader 0.3.1 ignores ``dotnet_root`` when no runtime config is given (spec section 9.4),
    so nothing may depend on the process environment happening to carry ``DOTNET_ROOT``.
    """
    env_root = os.environ.get("DOTNET_ROOT")
    if env_root and (Path(env_root) / "dotnet").is_file():
        return Path(env_root)
    home_root = Path.home() / ".dotnet"
    if (home_root / "dotnet").is_file():
        return home_root
    on_path = shutil.which("dotnet")
    if on_path:
        return Path(on_path).resolve().parent
    return None


def csharp_engine_available() -> tuple[bool, str]:
    """Whether the C# engine could be loaded here, and why not. Never loads the CLR."""
    if importlib.util.find_spec("pythonnet") is None:
        return False, 'pythonnet is not installed (pip install -e ".[csharp]")'
    bin_dir = assembly_dir()
    for name in (f"{ASSEMBLY_NAME}.dll", f"{ASSEMBLY_NAME}.runtimeconfig.json"):
        if not (bin_dir / name).is_file():
            return False, f"{name} not found in {bin_dir}; build it with: {BUILD_COMMAND}"
    if resolve_dotnet_root() is None:
        return False, "no .NET runtime found ($DOTNET_ROOT, ~/.dotnet or dotnet on PATH)"
    return True, ""
