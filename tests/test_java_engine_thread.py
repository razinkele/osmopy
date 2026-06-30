"""Integration: the off-thread Java runner streams console lines + posts a result (live-streaming)."""

import queue
import tempfile
from pathlib import Path

import pytest

_JAR = Path("osmose-java/osmose-4.4.1-jar-with-dependencies.jar")


@pytest.mark.skipif(not _JAR.exists(), reason="OSMOSE 4.4.1 jar not present")
def test_java_engine_thread_streams_lines_and_posts_done():
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.java_background_staging import stage_background_for_java
    from osmose.runner import OsmoseRunner
    from ui.pages.run import _java_engine_thread, write_temp_config

    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    config = dict(OsmoseConfigReader().read(str(res["config_file"])))
    work = tmp / "work"
    cp = write_temp_config(
        config, work, source_dir=res["config_file"].parent, target_version="4.4.1"
    )
    overrides = stage_background_for_java(cp.parent, config)
    overrides = {**overrides, "simulation.time.nyear": "1"}

    log_q: queue.Queue = queue.Queue()
    done_q: queue.Queue = queue.Queue()
    runner = OsmoseRunner(jar_path=_JAR)
    # runs synchronously here (in a thread in prod); blocks until the jar finishes
    _java_engine_thread(runner, cp, work / "output", None, overrides, 600, log_q, done_q)

    # the console was streamed line-by-line through the queue (the live-streaming mechanism)
    lines = []
    while not log_q.empty():
        lines.append(log_q.get_nowait())
    assert lines, "no console lines were streamed to the queue"
    assert any("OSMOSE" in ln or "4.4.1" in ln for ln in lines), (
        "expected the jar banner in the stream"
    )

    kind, result, _msg = done_q.get_nowait()
    assert kind == "done"
    assert result.returncode == 0, f"Baltic Java run failed: {result.stderr[-500:]}"
