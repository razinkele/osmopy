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

# Post-write steps in _finish_success: notification, three field clears, modal dismiss. Used
# only to recognise a total failure, which escalates from WARNING to ERROR.
_FINISH_STEPS = 5

# Warn-once flags for the three degraded rate-limiting modes. Silent degradation becomes a
# mystery ticket; one warning becomes a config fix.
_warned_untrusted_proxy = False
_warned_no_client = False
_warned_empty_xff = False


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


def _safe_text(input, name: str) -> str:
    """Read a text input as a stripped string, tolerating one that is not registered yet.

    Every input this handler reads goes through here, so the tolerance is uniform rather than
    applied to whichever field someone remembered. A `SilentException` escaping a reactive
    effect aborts the whole submission silently, which is the failure mode this prevents.
    """
    try:
        return (getattr(input, name)() or "").strip()
    except (SilentException, AttributeError):
        return ""


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
    return _safe_text(input, "feedback_website")


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

    Every path that ends up NOT keying per client warns once. There are three of them, and the
    third (2 reached from a trusted proxy) is the easiest to miss.
    """
    global _warned_untrusted_proxy, _warned_no_client, _warned_empty_xff

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
            # Trailing comma / empty entry: nothing usable here. Falling through does NOT
            # recover per-client keying — behind this same trusted proxy, client.host is the
            # PROXY's own address, so every user lands in one bucket just as an empty key
            # would. It is chosen only because a real address is a saner key than "", and it
            # warns because the result is a global throttle that nothing else would reveal.
            if not _warned_empty_xff:
                _warned_empty_xff = True
                _log.warning(
                    "%s is set but X-Forwarded-For (%r) has no usable rightmost entry — "
                    "falling back to the connecting address, which behind that proxy is the "
                    "PROXY, so all clients share one rate-limit bucket.",
                    _TRUSTED_PROXY_ENV,
                    xff,
                )
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


REJECT = "reject"  # show the message, store nothing
DROP = "drop"  # look exactly like success, store nothing
ACCEPT = "accept"  # store it, then look like success


def _classify_and_consume(
    msg: str,
    contact: str,
    honeypot: str,
    limiter: RateLimiter,
    key: str,
    now: float,
) -> tuple[str, str | None]:
    """Decide a submission's fate, CONSUMING a rate-limit slot in the process.

    The name says "and consume" because this is not a predicate: reaching the rate-limit check
    charges ``limiter`` for ``key``. Calling it twice to re-read a decision would double-charge
    the caller and eventually refuse them. There is one call site today; the name is for the
    next person, who will not read this docstring first.

    Returns ``(outcome, message)``. ``message`` is the user-facing copy for REJECT and is
    **None for DROP and ACCEPT** — those two share the handler's ``_SUCCESS_MSG``, which is the
    single source of that copy. Returning a second copy here that only tests could observe
    made the indistinguishability tests assert against a value the shipped handler never read.

    **THE ORDER OF THESE CHECKS IS SECURITY-RELEVANT — do not reshuffle for tidiness.**

    The honeypot is checked LAST, after every other rejection. Checking it earlier turns the
    handler into a one-probe oracle: a bot submits the same malformed email twice, once with
    the hidden field filled and once without, and the two different answers name the trap
    field. A honeypot a bot can identify is not a honeypot. So every rejection reachable
    without the honeypot must be decided BEFORE the honeypot is consulted, and the honeypot's
    own outcome must be indistinguishable from plain success.

    It also sits after the rate limiter so bot traffic is metered like anyone else's. Putting
    it first left honeypot requests unbounded, and the "it protects the shared bucket"
    argument does not hold: an attacker draining the bucket simply leaves the field empty.

    R5 is unaffected — the honeypot is still checked explicitly here, before
    ``build_feedback_record``, rather than by catching the ValueError that function raises for
    it (that same exception also signals an empty message and an unknown type, so a broad
    catch would thank the user for a record that was never stored).
    """
    if not msg:
        return REJECT, "Enter a message before sending."
    if contact and not looks_like_email(contact):
        return REJECT, "That email address doesn't look right — correct it or leave it blank."
    notice = _rate_limit_notice(limiter, key, now)
    if notice is not None:
        return REJECT, notice
    if honeypot:
        return DROP, None
    return ACCEPT, None


def _store_submission(
    *,
    type: str,
    msg: str,
    contact: str,
    honeypot: str,
    version: str,
    nav_tab: str,
) -> str | None:
    """Persist one submission. Returns an error message for the user, or None on success.

    The address is stored under a SEPARATE guard on purpose. Once ``append_feedback`` has
    returned, the feedback IS stored; telling the user otherwise because the contacts
    side-store failed makes them resubmit, which yields a duplicate record plus an orphan
    ``has_contact: true`` record with no contacts row. A lost address is worth a log line, not
    a false failure.
    """
    try:
        rec = build_feedback_record(
            type,
            msg,
            contact=contact,
            version=version,
            nav_tab=nav_tab,
            honeypot=honeypot,  # provably "" here; keeps the library guard live, not inert
        )
        append_feedback(rec)
    except Exception:  # noqa: BLE001 — never crash the session on a save failure
        _log.error("feedback save failed", exc_info=True)
        return "Couldn't save feedback — try again."

    if contact:
        try:
            save_contact(rec["id"], contact)
        except Exception:  # noqa: BLE001 — the feedback itself is already safely stored
            _log.error(
                "feedback %s was stored but its contact address was LOST — the record says "
                "has_contact=true and no contacts row exists for it",
                rec["id"],
                exc_info=True,
            )
    return None


async def _finish_success(session) -> None:
    """The one and only success path — the honeypot hit takes it too.

    A bot must not be able to tell it was caught from anything it can observe: same
    notification, same cleared fields, same dismissed modal. Leaving the modal open would be
    a tell.

    Every step is best-effort AND INDEPENDENTLY GUARDED. By the time this runs the record is
    already written, so a failure here must not undo that, must not surface as an error the
    user would act on by resubmitting, and must not crash the effect.

    One try around the whole block would have been worse than none: a ``notification_show``
    raise would have skipped the clear AND the dismiss AND said nothing, leaving the user
    looking at an unchanged form with no message — which is precisely the "assume it failed,
    submit again" duplicate that giving ``save_contact`` its own guard removed. Each step
    therefore fails alone, and a clean sweep of failures escalates to ERROR so the silence is
    never total.
    """
    failed: list[str] = []

    def _step(label: str, fn) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001 — one dead step must not take the others with it
            failed.append(label)
            _log.warning("feedback: %s failed after the record was written", label, exc_info=True)

    _step(
        "success notification",
        lambda: ui.notification_show(_SUCCESS_MSG, type="message", duration=4),
    )
    _step("clear message field", lambda: ui.update_text_area("feedback_message", value=""))
    _step("clear contact field", lambda: ui.update_text("feedback_contact", value=""))
    _step("clear honeypot field", lambda: ui.update_text("feedback_website", value=""))

    try:
        await session.send_custom_message("hide-modal", {"id": MODAL_ID})
    except Exception:  # noqa: BLE001 — a closed socket cannot undo a completed save
        failed.append("modal dismiss")
        _log.warning("feedback: modal dismiss failed after the record was written", exc_info=True)

    if len(failed) == _FINISH_STEPS:
        # The user was shown nothing at all. They will assume it failed and submit again, so
        # the operator has to be able to see that from the log alone.
        _log.error(
            "feedback: record written but EVERY post-write step failed (%s) — the user saw "
            "no confirmation and will probably resubmit",
            ", ".join(failed),
        )


def feedback_server(input, output, session, state):
    """Wire the submit handler. `output`/`state` unused; kept for call-signature uniformity."""

    @reactive.effect
    @reactive.event(input.feedback_submit)
    async def _submit():
        msg = _safe_text(input, "feedback_message")
        contact = _safe_text(input, "feedback_contact")
        honeypot = _read_honeypot(input)

        # _LIMITER is read by global name at call time (a module-attribute read), so a test
        # can monkeypatch it without this closure having captured the old one.
        outcome, message = _classify_and_consume(
            msg, contact, honeypot, _LIMITER, _client_key(session), time.time()
        )

        if outcome == REJECT:
            ui.notification_show(message, type="warning", duration=6)
            return

        if outcome == DROP:
            _log.info("feedback honeypot filled — dropping submission silently")
            await _finish_success(session)
            return

        error = _store_submission(
            type=_safe_text(input, "feedback_type"),
            msg=msg,
            contact=contact,
            honeypot=honeypot,
            version=__version__,
            nav_tab=_safe_nav(input),
        )
        if error is not None:
            ui.notification_show(error, type="error", duration=8)
            return

        await _finish_success(session)
