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
