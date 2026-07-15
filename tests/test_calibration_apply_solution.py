import pytest

pytest.importorskip("shiny")
from shiny import reactive

from ui.pages.calibration import _solution_diff_rows, apply_picked_solution
from ui.state import AppState


def test_solution_diff_rows_current_vs_new():
    rows = _solution_diff_rows({"a": "1"}, {"a": 2.0, "b": 3.0})
    by_key = {r["key"]: r for r in rows}
    assert by_key["a"]["value_a"] == "1"
    assert by_key["a"]["value_b"] == "2.0"
    assert by_key["a"]["change"] == "changed"
    assert by_key["b"]["value_a"] is None  # not in current config
    assert by_key["b"]["value_b"] == "3.0"
    assert by_key["b"]["change"] == "added"


def test_apply_picked_solution_wires_config_dirty_and_load_trigger():
    state = AppState()
    with reactive.isolate():
        state.config.set({"mortality.additional.rate.sp0": "0.5", "keep": "x"})
        t0 = state.load_trigger.get()
        n = apply_picked_solution(state, {"mortality.additional.rate.sp0": 0.8})
        assert n == 1
        assert state.config.get()["mortality.additional.rate.sp0"] == "0.8"
        assert state.config.get()["keep"] == "x"  # untouched key preserved
        assert state.dirty.get() is True  # modified badge lights
        assert state.load_trigger.get() == t0 + 1  # pages re-read
