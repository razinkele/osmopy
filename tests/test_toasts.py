"""Completion toasts: success links to a tab, failure is a longer-lived danger toast."""

from ui.components.toasts import completion_toast


def _html(t) -> str:
    return str(t.tagify())


def test_success_toast_links_to_tab():
    t = completion_toast("Simulation complete", "Output: /x", tab="results", action="Open Results")
    html = _html(t)
    assert "Simulation complete" in html and "Output: /x" in html
    assert "nav-link[data-value=&apos;results&apos;]" in html and "Open Results" in html
    assert "text-bg-success" in html or "success" in html


def test_failure_toast_has_no_link_and_danger_type():
    html = _html(completion_toast("Simulation failed", "boom", ok=False))
    assert "boom" in html
    assert "osm-toast-action" not in html
    assert "danger" in html
