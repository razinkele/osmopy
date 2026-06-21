# tests/test_run_observer.py
import queue
import types

import pytest

from osmose.live_movement import config_is_spatial, format_progress_label, make_run_observer


def _cfg(n_steps):
    return types.SimpleNamespace(n_steps=n_steps)


def test_run_observer_pushes_one_based_done_every_step():
    q: queue.Queue = queue.Queue(maxsize=1)
    obs = make_run_observer(q)
    obs(0, None, None, _cfg(24))  # step 0 -> done 1
    done, n, elapsed = q.get_nowait()
    assert done == 1 and n == 24 and elapsed >= 0.0
    obs(5, None, None, _cfg(24))  # step 5 -> done 6
    done, n, elapsed = q.get_nowait()
    assert done == 6 and n == 24 and elapsed >= 0.0


def test_run_observer_delegates_to_live_observer():
    q: queue.Queue = queue.Queue(maxsize=1)
    seen = []
    obs = make_run_observer(q, live_observer=lambda s, st, g, c: seen.append(s))
    obs(3, "st", "g", _cfg(10))
    assert seen == [3]
    assert q.get_nowait()[0] == 4  # done = step+1


def test_run_observer_without_live_observer_still_pushes_progress():
    q: queue.Queue = queue.Queue(maxsize=1)
    obs = make_run_observer(q, live_observer=None)
    obs(0, None, None, _cfg(10))
    assert q.get_nowait() == (1, 10, pytest.approx(0.0, abs=5.0))


def test_run_observer_swallows_live_observer_exception():
    q: queue.Queue = queue.Queue(maxsize=1)

    def boom(s, st, g, c):
        raise RuntimeError("live boom")

    obs = make_run_observer(q, live_observer=boom)
    obs(0, None, None, _cfg(10))  # must NOT raise
    assert q.get_nowait()[0] == 1  # progress still pushed (pushed before delegate)


def test_config_is_spatial():
    baltic = {
        "grid.nlon": "50",
        "grid.nlat": "40",
        "grid.upleft.lat": "66",
        "grid.upleft.lon": "10",
        "grid.lowright.lat": "54",
        "grid.lowright.lon": "30",
    }
    assert config_is_spatial(baltic) is True
    assert config_is_spatial({"grid.netcdf.file": "eec_grid-mask.nc"}) is False
    assert config_is_spatial({}) is False


def test_format_progress_label_year_off_by_one():
    # 1-year run, final tick: done == ndt must give "Year 1/1" (NOT "Year 2/1")
    assert "Year 1/1" in format_progress_label(24, 24, 24)
    # first step of year 2 in a 2-year run
    assert format_progress_label(25, 48, 24).startswith("Year 2/2")
    # first step
    assert format_progress_label(1, 24, 24).startswith("Year 1/1")
    # ndt <= 0 -> step-only label, no ZeroDivisionError, no "Year"
    s = format_progress_label(3, 10, 0)
    assert s.startswith("Step 3/10") and "Year" not in s
