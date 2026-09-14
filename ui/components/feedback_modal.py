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

import errno
import ipaddress
import os
import time
from collections.abc import Callable

from shiny import reactive, ui
from shiny.types import SilentException

from osmose import __version__
from osmose.feedback import (
    VALID_TYPES,
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

# Two save-failure messages, because "try again" is a lie for half the failures. A full store
# (RuntimeError from osmose.feedback._append_json_line) and an unwritable path both persist until
# an OPERATOR acts, so inviting a retry just produces a user hammering a button that cannot work.
_RETRY_SAVE_MSG = "Couldn't save feedback — try again."
_TERMINAL_SAVE_MSG = "The server can't store feedback right now — this has been logged."
# Both terminal causes share ONE message on purpose. Task 5 measured a variant that let error
# detail reach the caller and found the TOKEN in the response; the rule that came out of it is
# that detail goes to the log and never to the user. Saying "the disk is full" or "permission
# denied" would also hand an anonymous submitter a probe into the server's state. The log line
# (`feedback save failed`, with traceback) is where an operator finds out which it was.
_TERMINAL_SAVE_ERRNOS = frozenset(
    {errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSPC, errno.EDQUOT}
)


def _is_terminal_save_failure(exc: BaseException) -> bool:
    """Whether retrying this save could never succeed.

    ``RuntimeError`` is the store-full guard. The errnos are the filesystem states only an
    operator can clear: no permission, a read-only mount, a full disk, an exceeded quota.
    Anything else is treated as possibly transient — when we genuinely do not know, "try again"
    is the safer of the two wrong answers, because it does not strand a user whose next attempt
    would have worked.
    """
    if isinstance(exc, RuntimeError):
        return True
    return isinstance(exc, OSError) and exc.errno in _TERMINAL_SAVE_ERRNOS


# Warn-once flags for the four degraded rate-limiting modes. Silent degradation becomes a
# mystery ticket; one warning becomes a config fix.
_warned_untrusted_proxy = False
_warned_no_client = False
_warned_empty_xff = False
_warned_proxy_no_xff = False

# Loopback + RFC1918 only. Deliberately NOT ``ipaddress.is_private``, which is also True for the
# RFC5737 documentation ranges (198.51.100.0/24, 203.0.113.0/24) and for 0.0.0.0/8, 169.254/16 and
# 240.0.0.0/4. Those are not evidence of a reverse proxy, and 198.51.100.7 is the address the
# existing "direct connection, no degradation" test uses — keying the heuristic off `is_private`
# would warn about a healthy direct deployment. Verified 2026-09-14.
_PROXY_HINT_NETS = tuple(
    ipaddress.ip_network(n)
    for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "::1/128", "fc00::/7")
)


def _looks_like_a_proxy_hop(host: str) -> bool:
    """Whether ``host`` is the kind of address a reverse proxy connects from.

    A heuristic, and only ever used to decide whether to LOG — never to choose the key. A
    hostname or a unix-socket path is unparseable and answers False rather than guessing.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(ip in net for net in _PROXY_HINT_NETS)


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

    Every path that ends up NOT keying per client warns once. There are **four** of them, not
    three — an earlier version of this docstring undercounted, and the missing one is the mode
    that matches a default nginx deployment:

    - a. XFF present, ``OSMOSE_TRUSTED_PROXY`` unset — the header is ignored.
    - b. XFF present and trusted, but no usable rightmost entry.
    - c. No client address at all — the shared constant key.
    - d. **No XFF header at all**, while ``client.host`` is loopback/RFC1918. Path 2 returns the
      PROXY's own address, so every user shares one bucket. This one is invisible to the other
      three checks because the whole ``if xff:`` block is skipped, and it cannot be told apart
      from a genuine direct connection from localhost — hence a heuristic warning, and only a
      warning. The key is deliberately unchanged: a constant bucket is the correct behaviour
      here; the SILENCE was the defect.
    """
    global _warned_untrusted_proxy, _warned_no_client, _warned_empty_xff, _warned_proxy_no_xff

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
        # Degraded mode (d). Only reachable with NO X-Forwarded-For at all — cases (a) and (b)
        # already warned above, and re-warning here would double-report one misconfiguration.
        if not xff and not _warned_proxy_no_xff and _looks_like_a_proxy_hop(str(host)):
            _warned_proxy_no_xff = True
            _log.warning(
                "No X-Forwarded-For header and the connecting address (%s) is loopback or "
                "RFC1918 — that is almost always a reverse proxy which is not forwarding the "
                "header, in which case this address is the PROXY and every client shares one "
                "rate-limit bucket. Configure the proxy to send X-Forwarded-For and set %s. "
                "(If this really is a direct local connection, ignore it.)",
                host,
                _TRUSTED_PROXY_ENV,
            )
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
    feedback_type: str,
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
    field. A honeypot a bot can identify is not a honeypot.

    The rule, stated generally because the narrow version of it let the oracle back in: **EVERY
    validation capable of producing a distinguishable response must run BEFORE the honeypot
    check** — not merely the rejections someone once enumerated. ``feedback_type`` is the case
    that proved it. It was validated only inside ``build_feedback_record``, which runs in
    ``_store_submission``, i.e. AFTER the DROP branch has already returned. So a crafted type
    with an empty honeypot produced the save-failure copy with the modal still open, while the
    same crafted type with the honeypot filled produced the success copy and dismissed the
    modal — neither storing anything. That difference names the trap field just as precisely as
    the email probe this ordering was introduced to close. Validating the type here, ahead of
    the honeypot, is what actually closes it; ``build_feedback_record``'s own guard stays as
    defence in depth.

    When adding a field, ask what response an invalid value produces and where that decision is
    made. If the answer is "after the honeypot", it is an oracle.

    It also sits after the rate limiter so bot traffic is metered like anyone else's. Putting
    it first left honeypot requests unbounded, and the "it protects the shared bucket"
    argument does not hold: an attacker draining the bucket simply leaves the field empty.

    R5 is unaffected — the honeypot is still checked explicitly here, before
    ``build_feedback_record``, rather than by catching the ValueError that function raises for
    it (that same exception also signals an empty message and an unknown type, so a broad
    catch would thank the user for a record that was never stored).
    """
    if feedback_type not in VALID_TYPES:
        # Unreachable from the shipped radio buttons, so this is a crafted client. It must
        # still be decided HERE rather than in build_feedback_record — see the docstring.
        return REJECT, "Choose a feedback type before sending."
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

    The error is one of TWO messages, chosen by ``_is_terminal_save_failure``: a retryable
    failure invites a retry, a terminal one (full store, unwritable path) does not, because
    retrying it can never work. Neither discloses which condition occurred.

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
    except Exception as exc:  # noqa: BLE001 — never crash the session on a save failure
        _log.error("feedback save failed", exc_info=True)
        return _TERMINAL_SAVE_MSG if _is_terminal_save_failure(exc) else _RETRY_SAVE_MSG

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
    # Every synchronous best-effort step, as (label, call). The ERROR escalation below derives
    # its total from this list rather than from a hand-maintained constant: a literal would
    # silently stop matching the moment a step was added, and the escalation would cease to
    # exist with nothing going red to say so.
    steps: list[tuple[str, Callable[[], None]]] = [
        (
            "success notification",
            lambda: ui.notification_show(_SUCCESS_MSG, type="message", duration=4),
        ),
        ("clear message field", lambda: ui.update_text_area("feedback_message", value="")),
        ("clear contact field", lambda: ui.update_text("feedback_contact", value="")),
        ("clear honeypot field", lambda: ui.update_text("feedback_website", value="")),
    ]

    failed: list[str] = []
    for label, call in steps:
        try:
            call()
        except Exception:  # noqa: BLE001 — one dead step must not take the others with it
            failed.append(label)
            _log.warning("feedback: %s failed after the record was written", label, exc_info=True)

    # The dismiss is awaited, so it cannot sit in `steps`. It is still a post-write step, so it
    # is counted on BOTH sides — +1 here and a `failed` entry below — because special-casing it
    # out of either side would make a genuine total failure uncountable.
    attempted = len(steps) + 1
    try:
        await session.send_custom_message("hide-modal", {"id": MODAL_ID})
    except Exception:  # noqa: BLE001 — a closed socket cannot undo a completed save
        failed.append("modal dismiss")
        _log.warning("feedback: modal dismiss failed after the record was written", exc_info=True)

    if len(failed) == attempted:
        # The user was shown nothing at all. They will assume it failed and submit again, so
        # the operator has to be able to see that from the log alone.
        _log.error(
            "feedback: record written but EVERY post-write step failed (%s) — the user saw "
            "no confirmation and will probably resubmit",
            ", ".join(failed),
        )


def _notify(message: str, *, type: str, duration: int) -> None:
    """Show one notification, best-effort. Never raises.

    ``_finish_success`` already guards its own ``notification_show`` because a transport
    failure there must not crash the effect after a record has been written. The two refusal
    notifications in ``_submit`` had no such guard, so a dead socket on THOSE paths raised
    straight out of the reactive effect -- an inconsistency with no reason behind it. A user
    who cannot be told why their submission was refused is no worse off for the log line, and
    the session stays alive.
    """
    try:
        ui.notification_show(message, type=type, duration=duration)
    except Exception:  # noqa: BLE001 — a dead socket must not kill the submit effect
        _log.warning("feedback: could not show the %r notification", type, exc_info=True)


def feedback_server(input, output, session, state):
    """Wire the submit handler. `output`/`state` unused; kept for call-signature uniformity."""

    @reactive.effect
    @reactive.event(input.feedback_submit)
    async def _submit():
        # Read the type HERE, not at the _store_submission call below: it is validated inside
        # _classify_and_consume, which must decide it before the honeypot branch returns.
        ftype = _safe_text(input, "feedback_type")
        msg = _safe_text(input, "feedback_message")
        contact = _safe_text(input, "feedback_contact")
        honeypot = _read_honeypot(input)

        # _LIMITER is read by global name at call time (a module-attribute read), so a test
        # can monkeypatch it without this closure having captured the old one.
        outcome, message = _classify_and_consume(
            ftype, msg, contact, honeypot, _LIMITER, _client_key(session), time.time()
        )

        if outcome == REJECT:
            _notify(message, type="warning", duration=6)
            return

        if outcome == DROP:
            _log.info("feedback honeypot filled — dropping submission silently")
            await _finish_success(session)
            return

        error = _store_submission(
            type=ftype,
            msg=msg,
            contact=contact,
            honeypot=honeypot,
            version=__version__,
            nav_tab=_safe_nav(input),
        )
        if error is not None:
            _notify(error, type="error", duration=8)
            return

        await _finish_success(session)
