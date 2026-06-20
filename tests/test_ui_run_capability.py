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
        "py_verbosity",
        "py_param_overrides",
    ):
        assert input_id in text
