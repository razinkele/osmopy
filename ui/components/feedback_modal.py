"""Feedback modal (bug report / suggestion) + submit handler.

Reuses help_modal._bs_modal (static Bootstrap, header-triggered). Submission is a server-side
reactive.effect that appends to the feedback store — no HTTP POST. The read side is a
token-gated GET wired in app.py.

Because the modal is STATIC markup opened client-side by ``data-bs-toggle``, it cannot be
dismissed with ``ui.remove_modal()`` — that only removes modals shown via ``ui.modal_show()``
and would silently do nothing here. Success sends a ``hide-modal`` custom message instead,
handled by the JS block in ``app.py`` (registered next to ``toggle-spatial-pill``).
"""

from __future__ import annotations

import os
import time

from shiny import reactive, ui
from shiny.types import SilentException

from osmose import __version__
from osmose.feedback import (
    append_feedback,
    build_feedback_record,
    looks_like_email,
    save_contact,
)
from osmose.feedback_limits import RateLimiter
from osmose.logging import setup_logging
from ui.components.help_modal import _bs_modal

_log = setup_logging("osmose.feedback_modal")

MODAL_ID = "feedbackModal"

# 5 submissions per hour per client. Module-level on purpose: the handler reads it by global
# name at call time (a module-attribute read), so a test can monkeypatch
# ``ui.components.feedback_modal._LIMITER`` without the handler having closed over the old one.
_LIMITER = RateLimiter(max_per_window=5, window_s=3600)

# Set this when the app sits behind a reverse proxy you control, so X-Forwarded-For is
# trustworthy. Unset, the header is ignored entirely — see _client_key.
_TRUSTED_PROXY_ENV = "OSMOSE_TRUSTED_PROXY"

# The fallback bucket when no client address is available at all. Deliberately a CONSTANT and
# deliberately not ``session.id`` — see _client_key.
_SHARED_KEY = "_no-client-address_"

_SUCCESS_MSG = "Thanks — feedback saved."
# tests/test_e2e_feedback.py waits for the substring "saved" in the notification. Keep it.

# Warn-once flags for the two degraded rate-limiting modes. Silent degradation becomes a
# mystery ticket; one warning becomes a config fix.
_warned_untrusted_proxy = False
_warned_no_client = False


def _honeypot_field():
    """The offscreen "Website" input, hardened against browser autofill and the tab order.

    ``ui.input_text`` already emits ``autocomplete="off"`` and does not accept ``tabindex``,
    so the rendered ``<input>`` is post-processed. Both attributes matter: autofill happily
    fills an offscreen field labelled "Website", and a keyboard user tabbing through the form
    would otherwise land in it — either way a real bug report gets silently dropped, which is
    exactly the failure the explicit honeypot check exists to avoid.
    """
    tag = ui.input_text("feedback_website", "Website", width="100%")
    for child in tag.children:
        if getattr(child, "name", None) == "input":
            child.attrs.update({"autocomplete": "off", "tabindex": "-1"})
    return tag


def feedback_modal():
    """The Send-feedback modal (header-triggered, static Bootstrap)."""
    body = ui.TagList(
        ui.input_radio_buttons(
            "feedback_type",
            "What kind of feedback is this?",
            {"bug": "Bug report", "suggestion": "Feature request", "other": "Other"},
            selected="bug",
            inline=True,
        ),
        ui.input_text_area(
            "feedback_message",
            "Tell us what happened, or what you'd like to see (required)",
            rows=6,
            placeholder="For a bug: what you did, what you expected, what happened instead.",
            width="100%",
        ),
        ui.input_text(
            "feedback_contact",
            "Email (optional) — only used to follow up on this report",
            placeholder="you@example.org",
            width="100%",
        ),
        # Honeypot: real users never see it, bots fill every field they find.
        ui.tags.div(
            _honeypot_field(),
            style="position:absolute; left:-10000px; top:auto; width:1px; height:1px; overflow:hidden;",
            **{"aria-hidden": "true"},
        ),
        ui.input_action_button("feedback_submit", "Send feedback", class_="btn-primary"),
        ui.tags.p(
            "Stored on this server with the app version and the tab you were on. "
            "Your email is kept separately and never published.",
            class_="text-muted small mt-2",
        ),
    )
    return _bs_modal(MODAL_ID, "Send feedback", body, size="lg")


def _safe_nav(input) -> str:
    try:
        return input.main_nav() or ""
    except (SilentException, AttributeError):
        return ""


def _read_honeypot(input) -> str:
    """The honeypot value, or "" when it cannot be read.

    Fails OPEN on purpose: treating an unreadable honeypot as empty risks letting one bot
    through, while the other direction silently discards a real user's bug report.
    """
    try:
        return (input.feedback_website() or "").strip()
    except (SilentException, AttributeError):
        return ""


def _client_key(session) -> str:
    """Best-effort per-client bucket key for the rate limiter. Never raises.

    1. ``X-Forwarded-For``, but ONLY when ``OSMOSE_TRUSTED_PROXY`` is set, and then the
       RIGHTMOST comma-separated entry — with one trusted proxy in front that is the address
       the proxy itself observed. The leftmost entry is attacker-supplied; taking it is the
       classic XFF spoof and hands every request a fresh bucket.
    2. Otherwise ``session.http_conn.client.host``.
    3. Otherwise ONE shared constant key. Never ``session.id``: a new websocket is a new id,
       so that would let a shell loop fill the limiter's 10 000-key table — and the limiter
       fails closed on new keys when full, locking out every first-time submitter for an
       hour. A constant degrades to a single shared bucket instead: visible and bounded.
    """
    global _warned_untrusted_proxy, _warned_no_client

    conn = getattr(session, "http_conn", None)
    headers = getattr(conn, "headers", None)
    xff = ""
    if headers is not None:
        try:
            # Lowercase key via .get: Starlette's Headers is case-insensitive and a plain
            # dict fake then behaves identically.
            xff = (headers.get("x-forwarded-for", "") or "").strip()
        except Exception:  # noqa: BLE001 — an exotic headers object must not kill submission
            xff = ""
    if xff:
        if os.environ.get(_TRUSTED_PROXY_ENV):
            candidate = xff.split(",")[-1].strip()
            if candidate:
                return candidate
            # Trailing comma: no usable entry. Fall through rather than key on "", which
            # would merge every client into one bucket.
        elif not _warned_untrusted_proxy:
            _warned_untrusted_proxy = True
            _log.warning(
                "X-Forwarded-For present but %s is unset — ignoring it and rate-limiting all "
                "clients as one bucket. Set %s when behind a reverse proxy you control.",
                _TRUSTED_PROXY_ENV,
                _TRUSTED_PROXY_ENV,
            )

    client = getattr(conn, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    if host:
        return str(host)

    if not _warned_no_client:
        _warned_no_client = True
        _log.warning(
            "No client address available — rate-limiting every submission as one shared "
            "bucket. Feedback throttling is effectively global until this is fixed."
        )
    return _SHARED_KEY


def _rate_limit_notice(limiter: RateLimiter, key: str, now: float) -> str | None:
    """None when the submission is allowed; the user-facing refusal copy otherwise.

    ``at_capacity`` narrows the cause in ONE direction only (see its docstring): False proves
    the refusal was this caller's own per-key cap, so the message may say so. True is
    ambiguous — an established key over its own cap while the table happens to be full reads
    both — so the message must stay true either way. Getting this wrong tells a first-time
    submitter who has sent nothing that they have already sent several.
    """
    if limiter.allow(key, now=now):
        return None
    if not limiter.at_capacity:
        return "You've sent several already — please wait a little before sending more."
    return "We couldn't accept this right now — please try again shortly."


async def _finish_success(session) -> None:
    """The one and only success path — the honeypot hit takes it too.

    A bot must not be able to tell it was caught from anything it can observe: same
    notification, same cleared fields, same dismissed modal. Leaving the modal open would be
    a tell.
    """
    ui.notification_show(_SUCCESS_MSG, type="message", duration=4)
    ui.update_text_area("feedback_message", value="")
    ui.update_text("feedback_contact", value="")
    ui.update_text("feedback_website", value="")
    await session.send_custom_message("hide-modal", {"id": MODAL_ID})


def feedback_server(input, output, session, state):
    """Wire the submit handler. `output`/`state` unused; kept for call-signature uniformity."""

    @reactive.effect
    @reactive.event(input.feedback_submit)
    async def _submit():
        msg = (input.feedback_message() or "").strip()
        if not msg:
            ui.notification_show("Enter a message before sending.", type="warning", duration=5)
            return

        # Honeypot is checked EXPLICITLY here rather than by catching the ValueError
        # build_feedback_record raises for it: that same exception type also signals an empty
        # message and an unknown type, so a broad catch would show a thank-you for a record
        # that was never stored. Silently discarding a real bug report is far worse than a
        # bot learning something. build_feedback_record keeps its own guard (below, the value
        # is still passed through) as defence in depth.
        honeypot = _read_honeypot(input)
        if honeypot:
            # Before the limiter on purpose: in the shared-bucket fallback, bot hits would
            # otherwise burn the 5/hour that real users have.
            _log.info("feedback honeypot filled — dropping submission silently")
            await _finish_success(session)
            return

        contact = (input.feedback_contact() or "").strip()
        if contact and not looks_like_email(contact):
            ui.notification_show(
                "That email address doesn't look right — correct it or leave it blank.",
                type="warning",
                duration=6,
            )
            return

        notice = _rate_limit_notice(_LIMITER, _client_key(session), time.time())
        if notice is not None:
            ui.notification_show(notice, type="warning", duration=8)
            return

        try:
            rec = build_feedback_record(
                input.feedback_type(),
                msg,
                contact=contact,
                version=__version__,
                nav_tab=_safe_nav(input),
                honeypot=honeypot,  # provably "" here; keeps the library guard live, not inert
            )
            append_feedback(rec)
            # The address goes to the private side-store only, keyed by record id — it is
            # deliberately absent from the record itself, which is designed to be copyable
            # into a public issue.
            save_contact(rec["id"], contact)
        except Exception:  # noqa: BLE001 — never crash the session on a save failure
            # Cannot swallow the empty-message or honeypot paths: both returned above.
            _log.error("feedback save failed", exc_info=True)
            ui.notification_show("Couldn't save feedback — try again.", type="error", duration=8)
            return

        await _finish_success(session)
