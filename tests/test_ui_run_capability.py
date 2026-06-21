# tests/test_ui_run_capability.py
import ui.pages.run as run_page


def test_run_page_has_no_engine_tabs_navset():
    # The misleading read-only navset and its mirror observer must be gone.
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "run_engine_tabs" not in text
    assert "_sync_engine_tab" not in text


def test_run_page_uses_panel_conditional_for_engine_settings():
    text = open(run_page.__file__, encoding="utf-8").read()
    # Per-engine inputs stay always-registered, visibility via client-side condition.
    assert "panel_conditional" in text
    assert "input.engine_mode" in text
    # All per-engine input ids still present (registered).
    for input_id in (
        "java_opts",
        "run_timeout",
        "param_overrides",
        "py_threads",
        "py_param_overrides",
    ):
        assert input_id in text


def test_py_threads_wired_and_verbosity_removed():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "py_verbosity" not in text  # widget removed
    assert "set_num_threads" in text  # py_threads now wired
    assert "py_threads" in text  # input still present (wired, not dead)


def test_run_page_source_has_indicator_and_capability_slots():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert 'output_ui("engine_indicator")' in text
    assert 'output_ui("engine_capability")' in text


def test_run_page_imports_describe_engine_and_renders_capability():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "from osmose.engine_capabilities import describe_engine" in text
    assert "def engine_capability" in text


def test_run_page_has_progress_machinery():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert 'output_ui("run_progress")' in text
    assert "make_run_observer" in text
    assert "_progress_q" in text  # discriminating: NOT matched by existing "on_progress"
    assert "_progress.set(" in text  # the new reactive value, not the Java on_progress fn
    assert "format_progress_label" in text


def test_live_view_uses_expand_gate_not_switch():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "input.live_movement_view" not in text  # switch gate removed
    assert "live_view_expanded" in text  # expand gate present
    assert "_auto_enable_live_for_spatial" not in text  # superseded
    assert "choose_live_layer" in text  # heatmap fallback wired


def test_run_server_hardens_against_session_teardown():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "session.on_ended" in text
    assert "_session_alive" in text
