"""The only module that imports pythonnet (spec sections 9.1 and 9.4).

``load()`` hosts CoreCLR once per process and returns the ``Osmose.Engine`` namespace module.
Never call it in a process that later forks: CoreCLR's runtime threads do not survive
``fork``. The calibration and UQ pools use spawn/forkserver, so each worker loads its own.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Any

from osmose.engine_cs import (
    ASSEMBLY_NAME,
    BUILD_COMMAND,
    assembly_dir,
    csharp_engine_available,
    resolve_dotnet_root,
)


class CSharpEngineUnavailable(RuntimeError):
    """The engine cannot be loaded here; the message says why and how to fix it."""


_lock = threading.Lock()
_engine: ModuleType | None = None


def load() -> ModuleType:
    """Host the CLR (first call only) and return the ``Osmose.Engine`` namespace module."""
    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        ok, reason = csharp_engine_available()
        if not ok:
            raise CSharpEngineUnavailable(reason)
        bin_dir = assembly_dir()
        runtime_config = bin_dir / f"{ASSEMBLY_NAME}.runtimeconfig.json"
        import pythonnet

        if pythonnet.get_runtime_info() is None:
            # Both arguments, always: without runtime_config clr_loader re-discovers the root
            # from the environment and ignores dotnet_root (spec section 9.4).
            pythonnet.load(
                "coreclr",
                runtime_config=str(runtime_config),
                dotnet_root=str(resolve_dotnet_root()),
            )
        clr: Any = importlib.import_module("clr")
        try:
            clr.AddReference(str(bin_dir / f"{ASSEMBLY_NAME}.dll"))
        except Exception as exc:  # a stale or foreign build: say how to rebuild
            raise CSharpEngineUnavailable(
                f"could not load {bin_dir / ASSEMBLY_NAME}.dll ({exc}); rebuild with: {BUILD_COMMAND}"
            ) from exc
        _engine = importlib.import_module(ASSEMBLY_NAME)
        return _engine
