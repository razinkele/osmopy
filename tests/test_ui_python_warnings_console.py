"""The Python run path streams osmose WARNING+ logs into the run console queue (thread-isolated)."""

import logging
import queue
import threading

import pytest


@pytest.fixture(autouse=True)
def _isolate_engine_warning_state():
    """MANDATORY hygiene: the three warning-dedup sets are PROCESS-GLOBAL and these tests add
    handlers / set levels on the global osmose loggers. Clear the dedup sets before AND after each
    test and restore logger levels, so this file neither pollutes nor is polluted by the suite
    (e.g. tests/test_issue_120's leaked 'snap.nc' dedup key under `pytest --dist loadfile` would
    otherwise suppress the #120 assertion below). Mirrors tests/test_issue_123's clear fixture."""
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


def test_queue_log_handler_forwards_warning_on_matching_thread():
    from ui.pages.run import _QueueLogHandler

    q: queue.Queue = queue.Queue()
    h = _QueueLogHandler(q, threading.get_ident())
    log = logging.getLogger("osmose.config")
    log.addHandler(h)
    log.setLevel(logging.INFO)  # INFO so the INFO record IS created -> only the HANDLER's WARNING
    try:  # level can drop it (the production gate; WARNING here would be vacuous)
        log.warning("hello from the run")
        log.info(
            "info is below WARNING"
        )  # created, propagates to handler, dropped by handler level
    finally:
        log.removeHandler(h)

    lines = []
    while not q.empty():
        lines.append(q.get_nowait())
    assert lines == ["WARNING: hello from the run"]  # exactly the WARNING, formatted, no INFO


def test_queue_log_handler_drops_records_from_a_different_thread():
    from ui.pages.run import _QueueLogHandler

    q: queue.Queue = queue.Queue()
    h = _QueueLogHandler(q, threading.get_ident())  # bound to THIS thread
    log = logging.getLogger("osmose.config")
    log.addHandler(h)
    log.setLevel(logging.WARNING)
    try:
        # emit from a DIFFERENT thread -> record.thread differs -> must be dropped (no leak)
        t = threading.Thread(target=lambda: log.warning("from another session's run"))
        t.start()
        t.join()
        log.warning("from this run")  # same thread -> forwarded
    finally:
        log.removeHandler(h)

    lines = []
    while not q.empty():
        lines.append(q.get_nowait())
    assert lines == ["WARNING: from this run"]  # the other thread's warning was filtered out


def test_queue_log_handler_never_raises_on_broken_queue():
    from ui.pages.run import _QueueLogHandler

    class _Full:
        def put_nowait(self, _):
            raise RuntimeError("queue exploded")

    h = _QueueLogHandler(_Full(), threading.get_ident())
    rec = logging.LogRecord("osmose.config", logging.WARNING, __file__, 1, "x", None, None)
    h.emit(rec)  # must NOT raise (a log line can't break a run)


def test_python_engine_thread_streams_120_and_123_and_detaches(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import _QueueLogHandler, _python_engine_thread

    cfg = dict(OsmoseConfigReader().read("data/minimal/osm_all-parameters.csv"))
    cfg["_osmose.config.dir"] = "data/minimal"  # so the engine resolves the bundled data files
    cfg["simulation.ncpu"] = "8"  # java-only -> #123
    cfg["oxygen.factor"] = "1.0"  # java-only -> #123
    cfg["simulation.restart.file"] = "snap.nc"  # -> #120
    cfg["simulation.time.nyear"] = "1"  # keep the run short

    out = tmp_path / "output"
    out.mkdir()
    log_q: queue.Queue = queue.Queue()
    done_q: queue.Queue = queue.Queue()

    # Runs synchronously here (a thread in prod). The #120/#123 warnings fire in _prepare_run on
    # THIS thread, so the thread-filtered handler forwards them regardless of the run's outcome.
    _python_engine_thread(cfg, out, threading.Event(), None, done_q, 0, log_q)

    lines = []
    while not log_q.empty():
        lines.append(log_q.get_nowait())
    assert any("see issue #123" in ln for ln in lines), f"no #123 warning streamed; got {lines}"
    assert any("see issue #120" in ln for ln in lines), f"no #120 warning streamed; got {lines}"

    # handler was detached in the finally -> no leak on the global osmose logger
    assert not any(isinstance(h, _QueueLogHandler) for h in logging.getLogger("osmose").handlers), (
        "the _QueueLogHandler leaked onto the osmose logger"
    )


def test_python_engine_thread_without_log_q_attaches_no_handler(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import _QueueLogHandler, _python_engine_thread

    cfg = dict(OsmoseConfigReader().read("data/minimal/osm_all-parameters.csv"))
    cfg["_osmose.config.dir"] = "data/minimal"
    cfg["simulation.time.nyear"] = "1"
    done_q: queue.Queue = queue.Queue()
    out = tmp_path / "out"
    out.mkdir()

    _python_engine_thread(cfg, out, threading.Event(), None, done_q, 0, None)  # log_q=None

    assert not any(isinstance(h, _QueueLogHandler) for h in logging.getLogger("osmose").handlers), (
        "no handler should be attached when log_q is None"
    )
