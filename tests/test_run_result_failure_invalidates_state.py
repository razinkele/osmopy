"""C4 Phase A: failed/cancelled RunResult must invalidate state.output_dir.

Pre-C4 the Results page would auto-load from `state.output_dir` regardless
of whether the previous run succeeded — silent stale-result display. C4
Phase A gates `state.output_dir.set` on `result.returncode == 0` and clears
it (sets None) on failure or cancellation, so the Results page's existing
`if out and ...` guard short-circuits.

Phase B will plumb a cancellation token through `simulate.py` and wire
the UI Cancel button. This file's tests cover the Phase A surface only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from osmose.engine import SimulationCancelled
from osmose.runner import RunResult
from ui.pages.run import _handle_result


@pytest.fixture
def state_stub():
    """A minimal AppState-shaped stub with `.set` / `.get` reactive-value mocks."""
    state = MagicMock()
    state.run_result = MagicMock()
    state.output_dir = MagicMock()
    return state


@pytest.fixture
def status_stub():
    return MagicMock()


@pytest.fixture
def run_log_stub():
    log = MagicMock()
    log.get = MagicMock(return_value=[])
    return log


def test_runresult_default_status_ok():
    """Default constructor (no status / message kwargs) must still work — backwards-compat."""
    r = RunResult(returncode=0, output_dir=Path("out"), stdout="", stderr="")
    assert r.status == "ok"
    assert r.message == ""


def test_runresult_failed_construction():
    r = RunResult(
        returncode=1, output_dir=Path(""), stdout="", stderr="boom",
        status="failed", message="boom",
    )
    assert r.status == "failed"
    assert r.message == "boom"


def test_runresult_cancelled_construction():
    r = RunResult(
        returncode=-1, output_dir=Path(""), stdout="", stderr="",
        status="cancelled", message="user cancelled",
    )
    assert r.status == "cancelled"


def test_handle_result_success_sets_output_dir(
    state_stub, status_stub, run_log_stub, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """On a successful run, state.output_dir must be set to the result's output_dir.

    monkeypatch.chdir(tmp_path) so the unrelated history.save() side effect
    (which writes to Path('data/history')) lands inside the test's tmp_path
    instead of polluting the repo's working tree.
    """
    monkeypatch.chdir(tmp_path)
    result = RunResult(returncode=0, output_dir=tmp_path, stdout="", stderr="")
    _handle_result(result, config={}, state=state_stub, run_log=run_log_stub, status=status_stub)
    state_stub.output_dir.set.assert_called_once_with(tmp_path)


def test_handle_result_failure_clears_output_dir(state_stub, status_stub, run_log_stub):
    """On a failed run, state.output_dir must be cleared to None — no stale auto-load."""
    result = RunResult(
        returncode=1, output_dir=Path(""), stdout="", stderr="oops",
        status="failed", message="oops",
    )
    _handle_result(result, config={}, state=state_stub, run_log=run_log_stub, status=status_stub)
    # The bug pre-C4: state.output_dir.set was called with the (empty/missing)
    # output_dir from the failed run. Post-C4: it's called with None.
    state_stub.output_dir.set.assert_called_once_with(None)


def test_handle_result_cancelled_clears_output_dir(state_stub, status_stub, run_log_stub):
    """Same as failed — cancelled runs must also clear state.output_dir."""
    result = RunResult(
        returncode=-1, output_dir=Path(""), stdout="", stderr="",
        status="cancelled", message="user cancelled",
    )
    _handle_result(result, config={}, state=state_stub, run_log=run_log_stub, status=status_stub)
    state_stub.output_dir.set.assert_called_once_with(None)


def test_handle_result_cancelled_status_message(state_stub, status_stub, run_log_stub):
    """Cancelled status text must mention cancellation, not 'Failed'."""
    result = RunResult(
        returncode=-1, output_dir=Path(""), stdout="", stderr="",
        status="cancelled", message="user cancelled",
    )
    _handle_result(result, config={}, state=state_stub, run_log=run_log_stub, status=status_stub)
    # status_stub.set was called with a string starting with 'Cancelled'
    args, _ = status_stub.set.call_args
    assert args[0].startswith("Cancelled"), f"expected Cancelled-prefixed status, got: {args[0]!r}"


def test_simulation_cancelled_is_subclass_of_exception():
    """The exception must be a vanilla Exception so generic handlers can catch it."""
    assert issubclass(SimulationCancelled, Exception)
