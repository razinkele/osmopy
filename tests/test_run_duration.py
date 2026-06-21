"""Item 3: _handle_result must persist the real elapsed time, not 0."""

import types

import ui.pages.run as run_mod


def test_handle_result_records_real_duration(monkeypatch, tmp_path):
    captured = {}

    class _FakeHistory:
        def save(self, record):
            captured["record"] = record

    monkeypatch.setattr("osmose.history.default_run_history", lambda: _FakeHistory())
    # Freeze the "now" end time; start is passed in as 95.0 -> elapsed 5.0s.
    monkeypatch.setattr(run_mod.time, "monotonic", lambda: 100.0)

    result = types.SimpleNamespace(returncode=0, output_dir=str(tmp_path), status="ok", message="")
    state = types.SimpleNamespace(
        run_result=types.SimpleNamespace(set=lambda v: None),
        output_dir=types.SimpleNamespace(set=lambda v: None),
    )
    status = types.SimpleNamespace(set=lambda v: None)

    run_mod._handle_result(result, {"k": "v"}, state, None, status, start_monotonic=95.0)

    assert "record" in captured
    assert captured["record"].duration_sec == 5.0


def test_both_engine_paths_thread_start_time():
    """Guard the wiring: the start time is threaded as a parameter through BOTH
    engine paths (not read from a single Python-only cell). A bare
    'start_monotonic in src' would pass on just the signature add, so assert the
    Java call-site literal and that both functions accept the parameter."""
    import inspect
    import pathlib

    # Both functions must accept the parameter (catches a dropped signature).
    assert "start_monotonic" in inspect.signature(run_mod._handle_result).parameters
    assert "start_monotonic" in inspect.signature(run_mod._run_java_engine).parameters

    src = pathlib.Path(run_mod.__file__).read_text()
    assert "_run_start_cell" in src  # Python fire-and-forget path
    # Java call site forwards t0 (run.py ~L830) — fails if that site is not wired.
    assert "start_monotonic=run_t0" in src
