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


#: How often the waiting thread wakes to forward a cancel request and let signals through.
POLL_SECONDS = 0.05


def run(bundle: Any, control: Any, cancel_token: threading.Event | None = None) -> tuple:
    """Call ``OsmoseEngine.RunSimulation`` on ``bundle`` and return its summary as
    ``(status, steps, records, peak_schools, elapsed_seconds, rng_scheme)``.

    The engine runs on a worker thread; this thread waits in a ``join(timeout)`` loop, where a
    ``KeyboardInterrupt`` or a ``SIGALRM``-raised timeout is delivered normally, and forwards
    ``cancel_token`` to slot 0 of ``control`` (spec 8.5).

    **Buffer lifetime (spec 8.5).** The engine writes into memory ``bundle`` borrows. If anything
    is raised in the waiting thread, the engine is asked to stop (slot 0), the worker is JOINED,
    and only then is the exception re-raised: nothing may unwind while C# can still write.
    """
    engine = load()
    config = bundle.to_csharp(engine)
    outcome: dict[str, Any] = {}
    returned = threading.Event()  # set once RunSimulation has returned, whatever happened

    def work() -> None:
        try:
            s = engine.OsmoseEngine.RunSimulation(config)
            outcome["summary"] = (
                int(s.Status), int(s.StepsCompleted), int(s.RecordsWritten),
                int(s.PeakSchools), float(s.ElapsedSeconds), int(s.RngScheme),
            )  # fmt: skip
        except BaseException as exc:  # carried to the caller
            outcome["error"] = exc
        finally:
            returned.set()

    def forward_cancel() -> None:
        if cancel_token is not None and cancel_token.is_set():
            control[0] = 1

    forward_cancel()  # before the start: a token already set stops the engine at step 0
    worker = threading.Thread(target=work, name="osmose-csharp-run", daemon=False)
    worker.start()
    # Wait on our own Event, NOT Thread.join: measured 2026-11-18, after a KeyboardInterrupt
    # interrupted join(timeout), the next join() returned at once while the engine went on
    # writing (CPython's interrupted-join handling marks the still-running thread stopped).
    try:
        while not returned.wait(POLL_SECONDS):
            forward_cancel()
    except BaseException:
        control[0] = 1  # the engine checks slot 0 once per step: the wait is bounded by one step
        while True:  # a second signal must not unwind past a running engine either
            try:
                returned.wait()
                break
            except BaseException:
                continue
        raise  # the first exception: the inner handler above has finished
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["summary"]
