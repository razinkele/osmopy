import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import calibrate_baltic as cb  # noqa: E402


def test_expand_param_overrides_passthrough_and_logspace():
    keys = ["mortality.additional.larva.rate.sp0", "mortality.additional.rate.sp1"]
    x = np.array([1.0, -2.0])  # log10 -> 10.0, 0.01
    ov = cb.expand_param_overrides(keys, x, use_log_space=True)
    assert ov == {
        "mortality.additional.larva.rate.sp0": str(10.0),
        "mortality.additional.rate.sp1": str(0.01),
    }


def test_expand_param_overrides_zoo_sentinel_expands_to_four_keys():
    keys = ["mortality.additional.larva.rate.sp0", "species.regrowth.rate.zoo"]
    x = np.array([0.0, np.log10(0.6)])  # -> mort 1.0, zoo 0.6
    ov = cb.expand_param_overrides(keys, x, use_log_space=True)
    for r in (10, 11, 12, 13):
        assert ov[f"species.regrowth.rate.sp{r}"] == str(0.6)
    assert "species.regrowth.rate.zoo" not in ov
    assert ov["mortality.additional.larva.rate.sp0"] == str(1.0)


def test_get_a2_params_appends_zoo_param():
    keys, bounds, x0 = cb.get_a2_params()
    base_n = len(cb.get_phase1_params()[0])
    assert len(keys) == base_n + 1
    assert keys[-1] == "species.regrowth.rate.zoo"
    assert bounds[-1] == (-1.0, float(np.log10(2.0)))
    assert abs(x0[-1] - float(np.log10(0.6))) < 1e-12


def test_enable_a2_base_config_sets_keys_without_mutating_input():
    base = {"simulation.time.nyear": "15"}
    out = cb.enable_a2_base_config(base)
    assert out["ltl.depletable.enabled"] == "true"
    assert out["ltl.depletable.floor"] == "0.05"
    assert out["species.regrowth.rate.sp8"] == "5.0"
    assert out["species.regrowth.rate.sp9"] == "5.0"
    assert "ltl.depletable.enabled" not in base  # input untouched


# ---------------------------------------------------------------- sim timeout guard
def test_run_with_timeout_returns_when_fast():
    assert cb._run_with_timeout(lambda: 42, 5.0) == 42


def test_run_with_timeout_no_limit_passthrough():
    assert cb._run_with_timeout(lambda: 7, None) == 7


def test_run_with_timeout_raises_on_slow():
    import time

    import pytest

    with pytest.raises(cb._SimTimeout):
        cb._run_with_timeout(lambda: time.sleep(3), 0.3)


# ---------------------------------------------------------------- isolated-eval executor
def _fake_ok(x):
    return float(x[0]) * 10.0


def _fake_crash(x):
    if int(x[0]) == 99:
        import os

        os._exit(1)  # simulate a native worker crash (process dies, no result)
    return float(x[0])


def _fake_hang(x):
    if int(x[0]) == 99:
        import time as _t

        _t.sleep(30)  # hang far past the timeout
    return float(x[0])


def test_isolated_map_normal_returns_ordered_results():
    m = cb.isolated_eval_map(timeout_s=15.0, n_workers=4, penalty=1e6)
    tasks = [np.array([i]) for i in range(6)]
    assert m(_fake_ok, tasks) == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]


def test_isolated_map_penalizes_crashed_eval():
    m = cb.isolated_eval_map(timeout_s=15.0, n_workers=4, penalty=1e6)
    tasks = [np.array([1]), np.array([99]), np.array([2])]  # 99 hard-crashes
    res = m(_fake_crash, tasks)
    assert res[0] == 1.0 and res[2] == 2.0
    assert res[1] == 1e6  # crash -> penalty, others unaffected


def test_isolated_map_penalizes_hung_eval():
    m = cb.isolated_eval_map(timeout_s=0.5, n_workers=4, penalty=1e6)
    tasks = [np.array([1]), np.array([99]), np.array([2])]  # 99 hangs
    res = m(_fake_hang, tasks)
    assert res[0] == 1.0 and res[2] == 2.0
    assert res[1] == 1e6  # killed at timeout -> penalty
