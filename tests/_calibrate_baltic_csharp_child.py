"""Child process for tests/test_csharp_calibrate_baltic.py: ``scripts/calibrate_baltic.py`` on
the real C# engine. The .NET runtime is hosted in this process and in the workers it forks,
never in pytest.

Usage: ``python tests/_calibrate_baltic_csharp_child.py <repo root> <mode> <out dir>``.
Every mode writes ``<out dir>/summary.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])  # the worktree's osmose and scripts, not the editable install's

import json
import os
import time


def run(out: Path, isolated: bool) -> None:
    """A whole calibration, as small as scipy's DE allows: five candidates, two generations,
    two workers, three-year runs. Results and history go under ``out``."""
    from osmose.calibration import history
    from osmose.engine_cs import clr_hosted
    from scripts import calibrate_baltic as cb

    history.HISTORY_DIR = out / "history"
    cb.RESULTS_DIR = out / "results"
    # No evaluation may fall back to the Python engine, in this process or in the workers
    # forked from it (they inherit this). Its constructor stays: the C# engine uses it to
    # resolve the grid.
    from osmose.engine import PythonEngine

    def refuse(*_args, **_kwargs):
        raise AssertionError("an evaluation ran on the Python engine")

    PythonEngine.run = refuse
    PythonEngine.run_in_memory = refuse
    os.environ["OSMOSE_DE_WORKERS"] = "2"
    hosted_before = clr_hosted()
    saved = cb.run_calibration(
        phase="1b", maxiter=1, popsize=5, popsize_mult=0, n_seeds=1, n_years=3,
        checkpoint_every=1, sim_timeout_s=300.0, isolated_eval=isolated, engine="csharp",
    )  # fmt: skip
    # The optimizer is done: the re-rank and the final residuals ran HERE, on the C# engine.
    refused = ""
    try:
        cb._assert_may_fork("a second run in this process")
    except RuntimeError as exc:
        refused = str(exc)
    (out / "summary.json").write_text(
        json.dumps({
            "saved": saved,
            "hosted_before": hosted_before,
            "hosted_after": clr_hosted(),
            "fork_refused_after": refused,
            "numba_imported": "numba" in sys.modules,
        })
    )  # fmt: skip


def timeout(out: Path) -> None:
    """The script's SIGALRM-raised timeout through the bridge, between two ordinary runs in the
    same process. The first run loads the runtime and warms everything up, so the alarm of the
    second lands inside the engine and not in the start-up; the third must equal the first."""
    from osmose.config.reader import OsmoseConfigReader
    from scripts import calibrate_baltic as cb

    config = dict(OsmoseConfigReader().read(cb.BALTIC_CONFIG))

    def short():
        return cb.run_simulation(config, {}, n_years=2, seed=1, engine="csharp", engine_threads=1)

    before = short()
    t0 = time.perf_counter()
    timed_out = cb.run_simulation(
        config, {}, n_years=40, seed=1, timeout_s=2.0, engine="csharp", engine_threads=1
    )
    elapsed = time.perf_counter() - t0
    after = short()
    (out / "summary.json").write_text(
        json.dumps({
            "before": before, "timed_out": timed_out, "timed_out_seconds": elapsed, "after": after,
        })
    )  # fmt: skip


if __name__ == "__main__":
    mode, out_dir = sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "pool":
        run(out_dir, isolated=False)
    elif mode == "isolated":
        run(out_dir, isolated=True)
    elif mode == "timeout":
        timeout(out_dir)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
