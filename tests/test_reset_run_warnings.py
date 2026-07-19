"""reset_run_warnings clears all three engine warning-dedup sets so each UI run re-emits."""

import logging

import pytest


@pytest.fixture(autouse=True)
def _isolate_engine_warning_state():
    """MANDATORY hygiene: the three warning-dedup sets are PROCESS-GLOBAL. Clear them before AND
    after each test, and restore osmose logger levels, so these tests neither pollute nor get
    polluted by the rest of the suite (e.g. tests/test_issue_120's leaked 'snap.nc' dedup key under
    `pytest --dist loadfile`). Mirrors tests/test_issue_123_known_but_unread_keys.py's clear fixture."""
    from osmose.engine import config as cfg_mod
    from osmose.engine import config_validation as cv

    def _clear() -> None:
        cv._WARNED_JAVA_ONLY_KEYS.clear()
        cfg_mod._WARNED_UNSUPPORTED_RESTART.clear()
        cfg_mod._WARNED_UNSUPPORTED_MORTALITY.clear()

    _levels = {
        n: logging.getLogger(n).level for n in ("osmose", "osmose.config", "osmose.engine.config")
    }
    _clear()
    yield
    _clear()
    for name, lvl in _levels.items():
        logging.getLogger(name).setLevel(lvl)


def test_reset_run_warnings_clears_all_three_dedup_sets():
    from osmose.engine import reset_run_warnings
    from osmose.engine import config as cfg_mod
    from osmose.engine import config_validation as cv

    cv._WARNED_JAVA_ONLY_KEYS.add("fingerprint-a")
    cfg_mod._WARNED_UNSUPPORTED_RESTART.add("restart-msg")
    cfg_mod._WARNED_UNSUPPORTED_MORTALITY.add("mortality-msg")

    reset_run_warnings()

    assert cv._WARNED_JAVA_ONLY_KEYS == set()
    assert cfg_mod._WARNED_UNSUPPORTED_RESTART == set()
    assert cfg_mod._WARNED_UNSUPPORTED_MORTALITY == set()


def test_reset_run_warnings_lets_a_warning_re_emit():
    from osmose.engine import reset_run_warnings
    from osmose.engine.config_validation import warn_unread_java_only_keys

    cfg = {"simulation.ncpu": "8"}  # a java-only key -> #123 warning

    caplog_records = []
    h = logging.Handler()
    h.emit = lambda r: caplog_records.append(r.getMessage())
    logging.getLogger("osmose.config").addHandler(h)
    logging.getLogger("osmose.config").setLevel(logging.WARNING)
    try:
        warn_unread_java_only_keys(cfg)  # first emit populates the dedup set
        first = sum("issue #123" in m for m in caplog_records)
        warn_unread_java_only_keys(cfg)  # deduped -> no second emit
        deduped = sum("issue #123" in m for m in caplog_records)
        reset_run_warnings()  # clear -> next call re-emits
        warn_unread_java_only_keys(cfg)
        after_reset = sum("issue #123" in m for m in caplog_records)
    finally:
        logging.getLogger("osmose.config").removeHandler(h)

    assert first == 1
    assert deduped == 1  # the 2nd call was suppressed by dedup
    assert after_reset == 2  # reset let it re-emit
