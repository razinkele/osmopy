"""The DE parent must stay OpenMP-free while its workers are forked.

On Numba's GNU OpenMP threading layer, a process that has run a parallel kernel (or touched
Numba's thread count) cannot fork children that run parallel kernels: each child is terminated
("fork() called from a process already using GNU OpenMP, this is unsafe"). With
``--isolated-eval`` every DE candidate runs in a freshly forked child, and the checkpoint callback
re-evaluates the best candidate every ``--checkpoint-every`` generations. Done in the parent, that
re-evaluation started OpenMP there and every later candidate was scored the 1e6 penalty, so the
optimizer stopped learning without an error (reproduced 2026-11-14). The re-evaluation now runs in
a forked child (``_ForkedReeval``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_PROBE = """
import json, sys
import numba
sys.path.insert(0, {scripts!r})
from calibrate_baltic import _ForkedReeval, isolated_eval_map

@numba.njit(parallel=True)
def kernel(n):
    s = 0.0
    for i in numba.prange(n):
        s += i
    return s

class _Evaluator:
    seed = 7
    targets = []

    def __init__(self):
        self.last_per_species_residuals = None

    def __call__(self, x):
        v = float(kernel(1000)) + float(x[0])
        self.last_per_species_residuals = [("cod", v, 1.0)]
        return v

def objective(x):
    return float(kernel(1000)) + float(x)

if __name__ == "__main__":
    reeval = _ForkedReeval(_Evaluator(), timeout_s=120.0)
    value = reeval([2.0])  # the checkpoint re-evaluation
    after = isolated_eval_map(timeout_s=120.0, n_workers=2, penalty=1e6)(objective, [1.0, 2.0])
    print(json.dumps({{"value": value, "residuals": reeval.last_per_species_residuals,
                      "seed": reeval.seed, "after": after}}))
"""


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores")
def test_checkpoint_reeval_keeps_later_isolated_evals_alive(tmp_path):
    """A checkpoint re-evaluation followed by isolated evaluations that run a parallel kernel,
    in a fresh interpreter (this pytest process may already have started OpenMP)."""
    pytest.importorskip("numba")
    script = tmp_path / "probe.py"
    script.write_text(_PROBE.format(scripts=str(ROOT / "scripts")))
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300, env=env
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rec = json.loads(proc.stdout.strip().splitlines()[-1])
    assert rec["value"] == 499502.0
    assert rec["residuals"] == [["cod", 499502.0, 1.0]]  # handed back from the child
    assert rec["seed"] == 7
    assert rec["after"] == [499501.0, 499502.0], (
        "isolated evals were penalized after the checkpoint"
    )


class _CallbackCaptured(Exception):
    pass


@pytest.mark.parametrize(
    ("isolated_eval", "de_workers", "forked"),
    [
        (True, "1", True),  # --isolated-eval: a forked child per candidate
        (False, "4", True),  # scipy's own fork pool
        (False, "-1", True),  # scipy's pool on every core
        (False, "1", False),  # serial DE: the parent is the only process
    ],
)
def test_run_calibration_hands_the_checkpoint_a_forked_reeval_only_with_forked_workers(
    monkeypatch, tmp_results_dir, isolated_eval, de_workers, forked
):
    """Drive run_calibration up to the checkpoint callback and capture the evaluator it is
    given. Forked worker modes must get _ForkedReeval wrapping the DE objective; serial DE must
    get the objective itself (re-evaluating in the parent is safe there and needs no fork)."""
    import scripts.calibrate_baltic as cal

    monkeypatch.setenv("OSMOSE_DE_WORKERS", de_workers)
    seen = {}

    def capture(*args, **kwargs):
        seen["evaluator"] = kwargs["evaluator"]
        raise _CallbackCaptured

    monkeypatch.setattr(cal, "_make_checkpoint_callback", capture)
    with pytest.raises(_CallbackCaptured):
        cal.run_calibration(phase="1", n_years=1, isolated_eval=isolated_eval)

    evaluator = seen["evaluator"]
    if forked:
        assert isinstance(evaluator, cal._ForkedReeval)
        assert hasattr(evaluator._evaluator, "targets")  # wraps the DE objective
        assert evaluator.targets == evaluator._evaluator.targets
    else:
        assert not isinstance(evaluator, cal._ForkedReeval)
        assert hasattr(evaluator, "targets")


class _RaisingEvaluator:
    seed = 1
    targets: tuple = ()
    last_per_species_residuals = None

    def __call__(self, x):
        raise ValueError("bad candidate")


class _HangingEvaluator:
    seed = 1
    targets: tuple = ()
    last_per_species_residuals = None

    def __call__(self, x):
        import time

        time.sleep(60)
        return 0.0


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_forked_reeval_reports_a_failure_instead_of_returning_nothing():
    """The checkpoint logs a failed re-evaluation when the evaluator raises; the child's error
    must come back as an exception, not as a silent empty result."""
    from scripts.calibrate_baltic import _ForkedReeval

    with pytest.raises(RuntimeError, match="ValueError: bad candidate"):
        _ForkedReeval(_RaisingEvaluator(), timeout_s=60.0)([1.0])


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_forked_reeval_times_out_and_kills_the_child():
    from scripts.calibrate_baltic import _ForkedReeval

    with pytest.raises(TimeoutError):
        _ForkedReeval(_HangingEvaluator(), timeout_s=1.0)([1.0])
