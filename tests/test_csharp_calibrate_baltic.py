"""``scripts/calibrate_baltic.py --engine csharp`` on the real C# engine, end to end, in a child
process (the owner's G2 decision, 2026-11-19).

The unit tests (tests/test_calibrate_baltic_engine.py) stub the engine. Here it runs: the DE
workers are FORKED from a parent that has not loaded the .NET runtime, each loads its own, and
the parent loads it only after the last fork, for the re-rank. Needs the built engine: skips
without it, unless ``OSMOSE_REQUIRE_CSHARP=1`` (the csharp-engine CI job), where it fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from osmose.engine_cs import csharp_engine_available

ROOT = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).with_name("_calibrate_baltic_csharp_child.py")


def _need_engine() -> None:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")


def _child(mode: str, out: Path, timeout: int = 900) -> dict:
    proc = subprocess.run(
        [sys.executable, str(CHILD), str(ROOT), mode, str(out)],
        capture_output=True, text=True, timeout=timeout, cwd=ROOT,
    )  # fmt: skip
    assert proc.returncode == 0, (proc.stdout[-3000:], proc.stderr[-3000:])
    return json.loads((out / "summary.json").read_text()) | {"stdout": proc.stdout}


@pytest.mark.parametrize("mode", ["pool", "isolated"])
def test_a_small_calibration_runs_on_the_csharp_engine(mode, tmp_path):
    """Both ways the script parallelises DE: its fork pool, and a killable child per evaluation.
    Every evaluation must have COMPLETED: a worker that could not run the engine scores the
    1e6 penalty, and a run made of penalties would still finish and save a file."""
    _need_engine()
    got = _child(mode, tmp_path)
    saved = got["saved"]
    assert saved["engine"] == "csharp" and saved["phase"] == "1b" and saved["optimizer"] == "de"
    assert saved["n_evaluations"] >= 10  # five candidates, two generations
    assert 0.0 <= saved["objective_single_seed"] < 1e5, "the optimizer's best is a penalty"
    assert len(saved["objective_per_seed"]) == 5
    assert all(0.0 <= v < 1e5 for v in saved["objective_per_seed"]), saved["objective_per_seed"]
    assert "C# preflight: the starting point evaluates to" in got["stdout"]
    # Files: under the engine's name, so a Python run's results cannot be replaced.
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == [
        "phase1b_csharp_checkpoint.json", "phase1b_csharp_results.json",
    ]  # fmt: skip
    (run_file,) = (tmp_path / "history").iterdir()
    assert json.loads(run_file.read_text())["engine"] == "csharp"
    # The order of operations that makes forked workers safe: no runtime in the parent while it
    # forks, and once it has run the engine (the re-rank), the guard refuses any further fork.
    assert got["hosted_before"] is False and got["hosted_after"] is True
    assert "must not fork" in got["fork_refused_after"]
    # Nothing in this process ran the Python engine: its Numba was never imported.
    assert got["numba_imported"] is False


def test_a_timed_out_candidate_does_not_take_the_engine_with_it(tmp_path):
    """The script stops a run with an exception raised from SIGALRM. Through the bridge that
    must stop the engine promptly, score the candidate as failed, and leave the process able to
    run the next one, with the same result as before."""
    _need_engine()
    got = _child("timeout", tmp_path)
    biomass = [f"{sp}_mean" for sp in (
        "cod_west", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt", "stickleback",
        "cod_east",
    )]  # fmt: skip
    assert all(got["before"][key] > 0 for key in biomass)
    assert got["timed_out"] == {}, "a timed-out run is a failed evaluation"
    assert 2.0 <= got["timed_out_seconds"] < 15.0, "stopped at the alarm, not after 40 years"
    assert got["after"] == got["before"], "the run after the timeout differs from the one before"
