"""Actionable completion toasts (shiny >= 1.9 ``ui.toast``).

Reserved for the few events a user waits on — a run or a calibration finishing — where the
toast carries a next step. Routine confirmations stay on ``ui.notification_show``.
"""

from __future__ import annotations

from shiny import ui


def _goto_tab_link(tab: str, label: str) -> ui.Tag:
    """A link that switches the sidebar to ``tab`` (by clicking its nav pill)."""
    js = (
        f"document.querySelector(\".nav-pills .nav-link[data-value='{tab}']\")?.click();"
        "return false;"
    )
    return ui.tags.a(label, href="#", onclick=js, class_="osm-toast-action")


def completion_toast(
    title: str, body: str, *, tab: str | None = None, action: str = "Open", ok: bool = True
):
    """Build the toast; ``tab`` adds a link to that sidebar tab."""
    children: list = [ui.tags.div(body)]
    if tab:
        children.append(ui.tags.div(_goto_tab_link(tab, action), class_="mt-2"))
    return ui.toast(
        *children,
        header=title,
        type="success" if ok else "danger",
        duration_s=12 if ok else 20,
        position="bottom-right",
    )


def show_completion_toast(*args, **kwargs) -> None:
    ui.show_toast(completion_toast(*args, **kwargs))
