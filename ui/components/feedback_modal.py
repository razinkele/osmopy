"""Feedback modal (bug report / suggestion) + submit handler.

Reuses help_modal._bs_modal (static Bootstrap, header-triggered). Submission is a server-side
reactive.effect that appends to the feedback store — no HTTP POST. The read side is a
token-gated GET wired in app.py.
"""

from __future__ import annotations

from shiny import reactive, ui
from shiny.types import SilentException

from osmose import __version__
from osmose.feedback import append_feedback, build_feedback_record
from osmose.logging import setup_logging
from ui.components.help_modal import _bs_modal

_log = setup_logging("osmose.feedback_modal")


def feedback_modal():
    """The Send-feedback modal (header-triggered, static Bootstrap)."""
    body = ui.TagList(
        ui.input_radio_buttons(
            "feedback_type",
            "Type",
            {"bug": "Bug report", "suggestion": "Suggestion", "other": "Other"},
            selected="bug",
        ),
        ui.input_text_area(
            "feedback_message",
            "Message",
            rows=5,
            placeholder="What happened, or what would you like to see?",
            width="100%",
        ),
        ui.input_text("feedback_contact", "Contact (optional)", width="100%"),
        ui.input_action_button("feedback_submit", "Send", class_="btn-primary"),
        ui.tags.p(
            "Stored locally with the app version and current tab. Contact is optional.",
            class_="text-muted small mt-2",
        ),
    )
    return _bs_modal("feedbackModal", "Send feedback", body, size="lg")


def _safe_nav(input) -> str:
    try:
        return input.main_nav() or ""
    except (SilentException, AttributeError):
        return ""


def feedback_server(input, output, session, state):
    """Wire the submit handler. `output`/`state` unused; kept for call-signature uniformity."""

    @reactive.effect
    @reactive.event(input.feedback_submit)
    def _submit():
        msg = (input.feedback_message() or "").strip()
        if not msg:
            ui.notification_show("Enter a message before sending.", type="warning", duration=5)
            return
        try:
            rec = build_feedback_record(
                input.feedback_type(),
                msg,
                contact=(input.feedback_contact() or "").strip(),
                version=__version__,
                nav_tab=_safe_nav(input),
            )
            append_feedback(rec)
        except Exception:  # noqa: BLE001 — never crash the session on a save failure
            _log.error("feedback save failed", exc_info=True)
            ui.notification_show("Couldn't save feedback — try again.", type="error", duration=8)
            return
        ui.notification_show("Thanks — feedback saved.", type="message", duration=4)
        ui.update_text_area("feedback_message", value="")
        ui.update_text("feedback_contact", value="")
