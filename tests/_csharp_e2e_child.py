"""Child process for tests/test_csharp_engine_e2e.py: the CLR is hosted here, never in pytest.

Usage: ``python tests/_csharp_e2e_child.py <repo root> <mode> <raw config json> <out> [arg]``.
Every mode writes its result to ``<out>`` (a pickle or a JSON file).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])  # the worktree's osmose, not the editable install's

import json
import os
import pickle
import signal
import threading
import time

import numpy as np

FRAMES = ("biomass", "abundance", "yield_biomass", "mortality")


def _raw() -> dict:
    return json.loads(Path(sys.argv[3]).read_text())


def memory(out: Path) -> None:
    from osmose.engine_cs import CSharpEngine

    res = CSharpEngine().run_in_memory(_raw(), seed=int(sys.argv[5]))
    out.write_bytes(pickle.dumps({name: getattr(res, name)() for name in FRAMES}))


def outputs(out: Path) -> None:
    """The ``StepOutput`` list of one run, as ``run`` and ``run_in_memory`` hand it on, with the
    resource names the run published on its config."""
    from osmose.engine_cs import CSharpEngine

    config, _grid, records = CSharpEngine()._execute(_raw(), int(sys.argv[5]), None)
    out.write_bytes(pickle.dumps({"records": records, "resource_names": config.resource_names}))


def frames(out: Path) -> None:
    """Every frame ``run_in_memory`` builds, by its key."""
    from osmose.engine_cs import CSharpEngine

    res = CSharpEngine().run_in_memory(_raw(), seed=int(sys.argv[5]))
    out.write_bytes(pickle.dumps(res._csv_cache))


def files(out: Path) -> None:
    from osmose.engine_cs import CSharpEngine

    CSharpEngine().run(_raw(), out, seed=3)


def cancel_at(out: Path) -> None:
    """Cancel through slot 0 once slot 1 (steps completed) reaches k; result rows past the
    records written are prefilled with NaN so they are provably untouched."""
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig
    from osmose.engine_cs import bridge
    from osmose.engine_cs.build import allocate_results, build_bundle

    raw, k = _raw(), int(sys.argv[5])
    config = EngineConfig.from_dict(raw)
    grid = PythonEngine()._resolve_grid(raw)
    bundle = build_bundle(config, raw, grid, seed=5)
    buffers = allocate_results(bundle, config)
    for a in (buffers.biomass, buffers.abundance, buffers.mortality, buffers.yield_):
        a[:] = np.nan

    def watcher():
        while buffers.control[1] < k:
            time.sleep(0.001)
        buffers.control[0] = 1

    threading.Thread(target=watcher, daemon=True).start()
    status, steps, records, *_ = bridge.run(bundle, buffers.control)
    rows = np.isfinite(buffers.biomass).all(axis=1)
    out.write_text(json.dumps({
        "status": status, "steps": steps, "records": records, "n_steps": int(config.n_steps),
        "finite_rows": rows.tolist(), "record_step": buffers.record_step.tolist(),
        "control": buffers.control.tolist(),
    }))  # fmt: skip


def cancel_from_thread(out: Path) -> None:
    """The UI's pattern (task 2.7): the engine runs on a WORKER thread, and another thread sets
    the cancel token once the engine has completed k steps. The run must stop, raise
    ``SimulationCancelled`` on the worker, and write nothing."""
    import osmose.engine_cs.bridge as br
    from osmose.engine.config import EngineConfig
    from osmose.engine_cs import CSharpEngine

    raw, k = _raw(), int(sys.argv[5])
    seen: dict = {}
    real_run = br.run

    def run(bundle, control, cancel_token=None):
        seen["control"] = control
        return real_run(bundle, control, cancel_token)

    br.run = run
    token = threading.Event()
    result: dict = {}
    output_dir = out.parent / "cancelled_run"

    def worker():
        try:
            CSharpEngine().run(raw, output_dir, seed=1, cancel_token=token)
            result["raised"] = None
        except BaseException as exc:
            result.update(raised=type(exc).__name__, message=str(exc))

    thread = threading.Thread(target=worker, name="ui-run")
    thread.start()
    while thread.is_alive() and ("control" not in seen or seen["control"][1] < k):
        time.sleep(0.001)
    token.set()
    set_at = time.monotonic()
    thread.join(timeout=120)
    result["joined"] = not thread.is_alive()
    result["seconds_to_stop"] = time.monotonic() - set_at
    control = seen["control"]
    at_join = int(control[1])
    time.sleep(0.3)
    result.update(
        steps_at_join=at_join,
        steps_later=int(control[1]),
        cancel_slot=int(control[0]),
        n_steps=int(EngineConfig.from_dict(raw).n_steps),
        files=sorted(p.name for p in output_dir.rglob("*")) if output_dir.exists() else [],
    )
    out.write_text(json.dumps(result))


def expect_error(out: Path) -> None:
    """``precancel``: a cancel token already set; ``deadline``: a deadline already past."""
    from osmose.engine_cs import CSharpEngine

    kind = sys.argv[5]
    token = threading.Event()
    engine = CSharpEngine(deadline_seconds=1e-9) if kind == "deadline" else CSharpEngine()
    if kind == "precancel":
        token.set()
    try:
        engine.run_in_memory(_raw(), seed=1, cancel_token=token)
        result = {"raised": None}
    except Exception as exc:
        result = {"raised": type(exc).__name__, "message": str(exc)}
    out.write_text(json.dumps(result))


class _SimTimeout(Exception):
    """Stands in for scripts/calibrate_baltic.py's SIGALRM-raised timeout."""


def interrupted(out: Path) -> None:
    """A signal raised in the waiting (main) thread once the engine has completed two steps:
    the wrapper must stop the engine and JOIN the worker before re-raising."""
    import osmose.engine_cs.bridge as br
    from osmose.engine_cs import CSharpEngine

    kind = sys.argv[5]
    seen: dict = {}
    real_run = br.run

    def handler(signum, frame):
        raise _SimTimeout("alarm")

    if kind == "alarm":
        signal.signal(signal.SIGALRM, handler)

    def run(bundle, control, cancel_token=None):
        seen["control"] = control

        def trigger():
            while control[1] < 2:
                time.sleep(0.001)
            os.kill(os.getpid(), signal.SIGINT if kind == "int" else signal.SIGALRM)

        threading.Thread(target=trigger, daemon=True).start()
        return real_run(bundle, control, cancel_token)

    br.run = run
    raw = _raw()
    try:
        CSharpEngine().run_in_memory(raw, seed=1)
        result = {"raised": None}
    except BaseException as exc:  # KeyboardInterrupt included
        control = seen["control"]
        at_catch = int(control[1])
        alive = [t.name for t in threading.enumerate() if t.name == "osmose-csharp-run"]
        time.sleep(0.3)
        result = {
            "raised": type(exc).__name__, "worker_alive": bool(alive), "cancel_slot": int(control[0]),
            "steps_at_catch": at_catch, "steps_later": int(control[1]),
        }  # fmt: skip
    from osmose.engine.config import EngineConfig

    result["n_steps"] = int(EngineConfig.from_dict(raw).n_steps)
    out.write_text(json.dumps(result))


def summary(out: Path) -> None:
    """One run straight through the bridge; writes the RunSummary fields (task 1.11)."""
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig
    from osmose.engine_cs import bridge
    from osmose.engine_cs.build import allocate_results, build_bundle

    raw = _raw()
    cfg = EngineConfig.from_dict(raw)
    bundle = build_bundle(cfg, raw, PythonEngine()._resolve_grid(raw), seed=5)
    buffers = allocate_results(bundle, cfg)
    status, steps, _records, _peak, elapsed, _scheme, phases = bridge.run(bundle, buffers.control)
    out.write_text(
        json.dumps({"status": str(status), "steps": int(steps), "n_steps": int(cfg.n_steps),
                    "elapsed": float(elapsed), "phases": phases})
    )  # fmt: skip


if __name__ == "__main__":
    {
        "memory": memory,
        "outputs": outputs,
        "frames": frames,
        "files": files,
        "cancel_at": cancel_at,
        "cancel_thread": cancel_from_thread,
        "error": expect_error,
        "interrupted": interrupted,
        "summary": summary,
    }[sys.argv[2]](Path(sys.argv[4]))
