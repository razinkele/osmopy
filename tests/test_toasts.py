"""Completion toasts: success links to a tab, failure is a longer-lived danger toast."""

import numpy as np

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


def test_pareto_front_size_counts_non_dominated_rows():
    from ui.components.toasts import pareto_front_size

    assert pareto_front_size(np.arange(1.0, 51.0).reshape(-1, 1)) == 1  # 1-objective: one best
    front = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    assert pareto_front_size(front) == 3
    assert pareto_front_size(np.empty((0, 2))) == 0


def test_toast_link_can_open_nested_tabs():
    html = _html(
        completion_toast(
            "Calibration finished",
            "ok",
            tab="calibration",
            subtabs=(("cal_groups", "Results"),),
        )
    )
    assert "#cal_groups .nav-link[data-value=&apos;Results&apos;]" in html
