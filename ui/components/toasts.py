"""Actionable completion toasts (shiny >= 1.9 ``ui.toast``).

Reserved for the few events a user waits on — a run or a calibration finishing — where the
toast carries a next step. Routine confirmations stay on ``ui.notification_show``.
"""

from __future__ import annotations

from shiny import ui


def _goto_tab_link(tab: str, label: str, subtabs: tuple[tuple[str, str], ...] = ()) -> ui.Tag:
    """A link that switches the sidebar to ``tab``, then each nested navset ``(id, value)``."""
    steps = [f"document.querySelector(\".nav-pills .nav-link[data-value='{tab}']\")?.click();"]
    for nav_id, value in subtabs:
        steps.append(
            f"document.querySelector(\"#{nav_id} .nav-link[data-value='{value}']\")?.click();"
        )
    # nested navsets render inside the tab just shown: give each click a tick
    js = "".join(f"setTimeout(function(){{{s}}}, {i * 150});" for i, s in enumerate(steps))
    return ui.tags.a(label, href="#", onclick=js + "return false;", class_="osm-toast-action")


def pareto_front_size(F) -> int:
    """Number of non-dominated rows of an objective matrix (all objectives minimised)."""
    import numpy as np

    F = np.atleast_2d(np.asarray(F, dtype=float))
    if F.size == 0:
        return 0
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        others = F[np.arange(n) != i]
        dominated[i] = bool(np.any(np.all(others <= F[i], axis=1) & np.any(others < F[i], axis=1)))
    return int((~dominated).sum())


def completion_toast(
    title: str,
    body: str,
    *,
    tab: str | None = None,
    subtabs: tuple[tuple[str, str], ...] = (),
    action: str = "Open",
    ok: bool = True,
):
    """Build the toast; ``tab`` (+ nested ``subtabs``) adds a link that navigates there."""
    children: list = [ui.tags.div(body)]
    if tab:
        children.append(ui.tags.div(_goto_tab_link(tab, action, subtabs), class_="mt-2"))
    return ui.toast(
        *children,
        header=title,
        type="success" if ok else "danger",
        duration_s=12 if ok else 20,
        position="bottom-right",
    )


def show_completion_toast(*args, **kwargs) -> None:
    ui.show_toast(completion_toast(*args, **kwargs))
