"""Unit tests for the feedback submit modal (markup + client-key + rate-limit copy).

The submit effect itself is glue over these seams and is exercised end-to-end by
``tests/test_e2e_feedback.py`` (opt-in: ``-m e2e``). What is unit-tested here is everything
that has a decision in it: the markup contract the brief pins down, the per-client key
derivation (which is security-relevant -- see ``_client_key``) and the refusal copy (which
must never assert a cause the limiter cannot prove).
"""

from __future__ import annotations

import asyncio
import logging
import re
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

import ui.components.feedback_modal as fm
from osmose.feedback_limits import MAX_KEYS, RateLimiter
from ui.components.feedback_modal import feedback_modal


class _AncestryOf(HTMLParser):
    """Collect the chain of open tags above ``<input id=...>``.

    Used instead of substring searches so "is the honeypot hidden" is answered about the
    honeypot's own ancestors, not about the document containing a hiding rule somewhere.
    """

    _VOID = {"input", "br", "hr", "img", "meta", "link", "source", "track", "wbr"}

    def __init__(self, target_id: str) -> None:
        super().__init__()
        self._target = target_id
        self._stack: list[tuple[str, dict]] = []
        self.ancestors: list[tuple[str, dict]] | None = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "input" and d.get("id") == self._target:
            self.ancestors = list(self._stack)
        if tag not in self._VOID:
            self._stack.append((tag, d))

    def handle_endtag(self, tag):
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                return


def test_modal_has_honeypot_and_is_visually_hidden():
    """The brief's own version searched the WHOLE document for the hiding rule.

    That is the vacuity class this branch has already been bitten by twice: any unrelated
    ``position:absolute`` in the modal would have satisfied it while the honeypot sat in plain
    sight. Scoped here to the honeypot input's actual ancestor chain.
    """
    html = str(feedback_modal())
    assert "feedback_website" in html  # honeypot field present

    parser = _AncestryOf("feedback_website")
    parser.feed(html)
    assert parser.ancestors is not None, 'no <input id="feedback_website"> in the modal'

    def _hides(attrs: dict) -> bool:
        style = (attrs.get("style") or "").replace(" ", "")
        return "position:absolute" in style or "d-none" in (attrs.get("class") or "")

    hiding = [(tag, a) for tag, a in parser.ancestors if _hides(a)]
    assert hiding, (
        "honeypot has no hiding ancestor — a position:absolute or d-none elsewhere in the "
        "document does not count. Ancestor chain: "
        f"{[(t, a.get('style'), a.get('class')) for t, a in parser.ancestors]}"
    )


def test_modal_offers_bug_and_feature_and_other():
    html = str(feedback_modal())
    for label in ("Bug report", "Feature request", "Other"):
        assert label in html


def test_message_field_label_says_it_is_required():
    """The brief's UX list asks for a required-message hint (R10).

    Scoped to the label bound to ``feedback_message`` rather than the whole modal, so the word
    appearing anywhere else on the page cannot satisfy it.
    """
    html = str(feedback_modal())
    m = re.search(r'<label[^>]*for="feedback_message"[^>]*>(.*?)</label>', html, re.S)
    assert m is not None, "no <label> bound to feedback_message in the modal"
    label = m.group(1)
    assert "(required)" in label, f"message label carries no required hint: {label!r}"


def test_honeypot_input_is_hidden_from_autofill_and_the_tab_order():
    """Scoped to the honeypot ``<input>`` itself, not the whole modal.

    A bare ``'tabindex="-1"' in html`` would pass with zero changes -- ``_bs_modal`` already
    puts ``tabindex="-1"`` on the modal wrapper div, and every ``ui.input_text`` already
    renders ``autocomplete="off"``. Browser autofill happily fills an offscreen "Website"
    field, and by the same reasoning that keeps the honeypot check explicit (R5), a false
    positive silently discards a real bug report.
    """
    html = str(feedback_modal())
    m = re.search(r"<input[^>]*id=\"feedback_website\"[^>]*>", html)
    assert m is not None, 'no <input> tag carrying id="feedback_website" in the modal'
    tag = m.group(0)
    assert 'tabindex="-1"' in tag, f"honeypot input is still tab-reachable: {tag}"
    assert 'autocomplete="off"' in tag, f"honeypot input is still autofillable: {tag}"


# ── _client_key ──────────────────────────────────────────────────────────────────


def _fake_session(*, headers: dict | None = None, host: str | None = None):
    """A session stand-in shaped like Starlette's HTTPConnection (shiny 1.6.3)."""
    conn = SimpleNamespace(
        headers=headers if headers is not None else {},
        client=SimpleNamespace(host=host) if host is not None else None,
    )
    # `id` is present precisely so a test can prove it is NOT used as the key.
    return SimpleNamespace(http_conn=conn, id="session-abc123")


def _reset_warn_flags(monkeypatch):
    """Re-arm the module-level warn-once flags so these tests are order-independent."""
    monkeypatch.setattr(fm, "_warned_untrusted_proxy", False, raising=False)
    monkeypatch.setattr(fm, "_warned_no_client", False, raising=False)
    monkeypatch.setattr(fm, "_warned_empty_xff", False, raising=False)


def test_client_key_takes_the_rightmost_xff_entry_when_a_proxy_is_trusted(monkeypatch):
    """Rightmost, not leftmost -- the leftmost entry is attacker-supplied.

    With one trusted proxy in front, the rightmost comma-separated entry is the address
    that proxy itself observed. Taking the leftmost is the classic X-Forwarded-For spoof
    and makes the limiter useless: a client sends ``X-Forwarded-For: <anything>`` and gets
    a fresh bucket per request.
    """
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    monkeypatch.setenv("OSMOSE_TRUSTED_PROXY", "1")
    sess = _fake_session(headers={"x-forwarded-for": "1.2.3.4, 203.0.113.9"}, host="10.0.0.1")
    assert key_fn(sess) == "203.0.113.9"


def test_client_key_falls_through_when_the_rightmost_xff_entry_is_empty(monkeypatch):
    """A trailing comma must not yield an empty key that merges every client into one."""
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    monkeypatch.setenv("OSMOSE_TRUSTED_PROXY", "1")
    sess = _fake_session(headers={"x-forwarded-for": "1.2.3.4,  "}, host="10.0.0.1")
    assert key_fn(sess) == "10.0.0.1"


def test_client_key_warns_once_when_a_trusted_proxy_yields_no_usable_entry(monkeypatch, caplog):
    """The third degraded path, and the easiest to miss.

    Falling through to ``client.host`` here does NOT restore per-client keying: behind the very
    proxy we just trusted, that address IS the proxy, so every user shares one bucket exactly
    as an empty key would have. Silent, consequential, and invisible to every other signal --
    so it warns, once, like the other two.
    """
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    monkeypatch.setenv("OSMOSE_TRUSTED_PROXY", "1")
    sess = _fake_session(headers={"x-forwarded-for": "1.2.3.4,  "}, host="10.0.0.1")
    logger_name = fm._log.name
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert key_fn(sess) == "10.0.0.1"
        assert key_fn(sess) == "10.0.0.1"
    warnings = [r for r in caplog.records if r.name == logger_name and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warn-once record, got {len(warnings)}"
    assert "PROXY" in warnings[0].getMessage(), (
        f"warning does not say the bucket is now shared: {warnings[0].getMessage()!r}"
    )


def test_client_key_ignores_xff_when_no_proxy_is_trusted_and_warns_once(monkeypatch, caplog):
    """An untrusted XFF must be ignored, and the degradation must be audible exactly once."""
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    monkeypatch.delenv("OSMOSE_TRUSTED_PROXY", raising=False)
    sess = _fake_session(headers={"x-forwarded-for": "1.2.3.4"}, host="10.0.0.1")
    logger_name = fm._log.name
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert key_fn(sess) == "10.0.0.1"  # the spoofable header is NOT the key
        assert key_fn(sess) == "10.0.0.1"
    warnings = [r for r in caplog.records if r.name == logger_name and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warn-once record, got {len(warnings)}"
    assert "OSMOSE_TRUSTED_PROXY" in warnings[0].getMessage()


def test_client_key_uses_client_host_when_present(monkeypatch, caplog):
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    monkeypatch.delenv("OSMOSE_TRUSTED_PROXY", raising=False)
    logger_name = fm._log.name
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert key_fn(_fake_session(host="198.51.100.7")) == "198.51.100.7"
    assert [r for r in caplog.records if r.name == logger_name] == []  # no degradation


def test_client_key_never_falls_back_to_the_session_id(monkeypatch, caplog):
    """No address available -> ONE shared constant key, never ``session.id``.

    A new websocket is a new ``session.id``, so keying on it makes the limiter's 10 000-key
    table fillable by a shell loop -- and since the limiter fails closed on new keys when
    full, that would lock out every first-time submitter for an hour. A single constant
    degrades to one shared bucket instead: visible, bounded, and warned about.
    """
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    monkeypatch.delenv("OSMOSE_TRUSTED_PROXY", raising=False)
    sess = _fake_session()  # no headers, no client
    logger_name = fm._log.name
    with caplog.at_level(logging.WARNING, logger=logger_name):
        k1 = key_fn(sess)
        k2 = key_fn(_fake_session())
    assert "session-abc123" not in k1, f"session id leaked into the rate-limit key: {k1!r}"
    assert k1 == k2, "fallback key must be CONSTANT, not per-session"
    warnings = [r for r in caplog.records if r.name == logger_name and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warn-once record, got {len(warnings)}"


def test_client_key_survives_a_session_with_no_http_conn(monkeypatch):
    """Never raise out of the key derivation -- a crash here kills the whole submission.

    The raise is converted to an AssertionError deliberately: an escaping AttributeError would
    red this test as an ERROR with no statement of what was expected, and "red for the wrong
    reason" is exactly what this suite is trying not to accept.
    """
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    try:
        key = key_fn(SimpleNamespace())
    except Exception as exc:  # noqa: BLE001 — any raise at all is the failure under test
        raise AssertionError(f"_client_key raised {exc!r} on a session with no http_conn") from exc
    assert isinstance(key, str)


# ── _rate_limit_notice ───────────────────────────────────────────────────────────


def test_rate_limit_notice_is_none_while_under_the_cap():
    notice_fn = getattr(fm, "_rate_limit_notice", None)
    assert notice_fn is not None, "_rate_limit_notice is not implemented"
    rl = RateLimiter(max_per_window=2, window_s=60)
    assert notice_fn(rl, "ip1", 0.0) is None
    assert notice_fn(rl, "ip1", 0.0) is None


def test_rate_limit_notice_blames_the_caller_only_when_the_cause_is_provable():
    """``at_capacity`` False proves the refusal was this caller's own per-key cap."""
    notice_fn = getattr(fm, "_rate_limit_notice", None)
    assert notice_fn is not None, "_rate_limit_notice is not implemented"
    rl = RateLimiter(max_per_window=1, window_s=60)
    assert notice_fn(rl, "ip1", 0.0) is None
    msg = notice_fn(rl, "ip1", 0.0)
    assert msg is not None, "second submission inside the window should have been refused"
    assert "several" in msg.lower(), f"provable per-key refusal should say so, got {msg!r}"


def test_rate_limit_notice_asserts_no_cause_when_the_table_is_full():
    """At capacity the cause is AMBIGUOUS, so the copy must be true either way.

    An established key over its own per-key cap reads ``allow() -> False`` AND
    ``at_capacity -> True`` at the same time, so naming saturation there would be a guess --
    and naming the per-key cap would tell a first-time submitter who has sent nothing that
    they have sent several already.
    """
    notice_fn = getattr(fm, "_rate_limit_notice", None)
    assert notice_fn is not None, "_rate_limit_notice is not implemented"
    rl = RateLimiter(max_per_window=5, window_s=60)
    for i in range(MAX_KEYS):
        rl.allow(f"k{i}", now=0.0)
    assert rl.at_capacity
    msg = notice_fn(rl, "brand-new-key", 0.0)
    assert msg is not None, "a new key must be refused once the table is full"
    assert "several" not in msg.lower(), f"must not assert a cause at capacity, got {msg!r}"


# ── success copy ─────────────────────────────────────────────────────────────────


def test_success_copy_still_says_saved():
    """``tests/test_e2e_feedback.py`` waits for the substring "saved" in the notification."""
    msg = getattr(fm, "_SUCCESS_MSG", None)
    assert msg is not None, "_SUCCESS_MSG is not defined"
    assert "saved" in msg, f"e2e test waits for 'saved' in the notification, got {msg!r}"


# ── _classify_and_consume: check ORDER, which is security-relevant ───────────────


def _limiter(*, exhausted: bool) -> RateLimiter:
    """A fresh limiter, optionally already at its per-key cap for "ip1"."""
    rl = RateLimiter(max_per_window=1, window_s=3600)
    if exhausted:
        rl.allow("ip1", now=0.0)
    return rl


# label -> (message, contact, limiter already exhausted)
_ORACLE_PROBES = {
    "empty_message": ("", "", False),
    "malformed_email": ("a real bug report", "not-an-email", False),
    "rate_limited": ("a real bug report", "", True),
    "everything_valid": ("a real bug report", "me@example.org", False),
}


@pytest.mark.parametrize("label", sorted(_ORACLE_PROBES))
def test_honeypot_is_not_a_one_probe_oracle(label):
    """The response must not reveal whether the hidden field was filled.

    Otherwise the handler is a ONE-PROBE ORACLE: send the same malformed email twice, once
    with the hidden field filled and once without, and the two different answers name the trap
    field. A honeypot a bot can identify is not a honeypot. So every rejection reachable
    without the honeypot has to be decided BEFORE the honeypot is consulted, and the honeypot's
    own outcome has to be indistinguishable from plain success.

    This covers the REJECT copy only, which is the copy the handler actually ships. ACCEPT and
    DROP return no message at all — their shared wording lives in the handler's ``_SUCCESS_MSG``
    — so their indistinguishability cannot be judged here and is gated instead by
    ``test_accept_and_drop_are_indistinguishable_to_the_client``, which drives the real handler.
    """
    msg, contact, exhausted = _ORACLE_PROBES[label]
    clean = fm._classify_and_consume(msg, contact, "", _limiter(exhausted=exhausted), "ip1", 0.0)
    baited = fm._classify_and_consume(
        msg, contact, "http://spam.example", _limiter(exhausted=exhausted), "ip1", 0.0
    )

    assert clean[1] == baited[1], (
        f"{label}: honeypot is a one-probe oracle — with the hidden field filled the client "
        f"sees {baited[1]!r}, without it {clean[1]!r}. The difference names the trap."
    )
    if clean[0] == fm.REJECT:
        assert baited[0] == fm.REJECT, (
            f"{label}: filling the honeypot turned a rejection into {baited[0]!r}"
        )
        assert clean[1] is not None, f"{label}: a REJECT must carry copy to show the user"
    else:
        assert (clean[0], baited[0]) == (fm.ACCEPT, fm.DROP), (
            f"{label}: expected accept/drop, got {clean[0]!r}/{baited[0]!r}"
        )
        assert (clean[1], baited[1]) == (None, None), (
            "ACCEPT/DROP must carry no message — _SUCCESS_MSG in the handler is the single "
            f"source of that copy, got {clean[1]!r}/{baited[1]!r}"
        )


def test_honeypot_submissions_are_rate_limited_like_everyone_else():
    """Honeypot traffic must consume the bucket, or it is unmetered handler work per bot hit.

    "It protects the shared bucket for real users" does not survive contact with an attacker,
    who simply leaves the hidden field empty to drain it anyway.
    """
    rl = RateLimiter(max_per_window=1, window_s=3600)
    first = fm._classify_and_consume("bait", "", "http://spam.example", rl, "ip1", 0.0)
    second = fm._classify_and_consume("bait", "", "http://spam.example", rl, "ip1", 0.0)
    assert first[0] == fm.DROP
    assert second[0] == fm.REJECT, "a second honeypot hit was not rate-limited"


# ── _store_submission ────────────────────────────────────────────────────────────


def test_store_reports_success_when_only_the_contact_side_store_fails(monkeypatch):
    """append_feedback succeeded, so the report IS stored — never say otherwise.

    Telling the user it failed makes them resubmit, which yields a duplicate record plus an
    orphan has_contact=true record with no contacts row.
    """
    appended = []
    monkeypatch.setattr(fm, "append_feedback", lambda rec: appended.append(rec))

    def _contacts_boom(feedback_id, email):
        raise OSError("contacts store is read-only")

    monkeypatch.setattr(fm, "save_contact", _contacts_boom)

    err = fm._store_submission(
        type="bug",
        msg="it broke",
        contact="me@example.org",
        honeypot="",
        version="0.0.0",
        nav_tab="run",
    )
    assert err is None, f"user was told the report failed, but it was stored: {err!r}"
    assert len(appended) == 1, "the feedback record was not appended"


def test_store_reports_failure_when_the_feedback_store_itself_fails(monkeypatch):
    """The other direction: a genuine loss must surface, not be swallowed into a thank-you."""

    def _store_boom(rec):
        raise OSError("disk full")

    monkeypatch.setattr(fm, "append_feedback", _store_boom)
    monkeypatch.setattr(fm, "save_contact", lambda *a, **k: None)

    err = fm._store_submission(
        type="bug",
        msg="it broke",
        contact="",
        honeypot="",
        version="0.0.0",
        nav_tab="run",
    )
    assert err is not None, "a failed append was reported to the user as success"


# ── _finish_success ──────────────────────────────────────────────────────────────


def test_finish_success_survives_a_dead_socket(monkeypatch):
    """A transport failure must not undo a completed save, nor crash the effect.

    By the time this runs the record is written. If the dismissal raises, the user must not
    see an error they would act on by resubmitting.
    """
    shown: list[str] = []
    cleared: list[str] = []
    monkeypatch.setattr(fm.ui, "notification_show", lambda m, **kw: shown.append(m))
    monkeypatch.setattr(fm.ui, "update_text_area", lambda i, **kw: cleared.append(i))
    monkeypatch.setattr(fm.ui, "update_text", lambda i, **kw: cleared.append(i))

    class _DeadSession:
        async def send_custom_message(self, name, payload):
            raise RuntimeError("websocket is closed")

    try:
        asyncio.run(fm._finish_success(_DeadSession()))
    except Exception as exc:  # noqa: BLE001 — any escape at all is the failure under test
        raise AssertionError(
            f"_finish_success let {exc!r} escape after the record was already stored"
        ) from exc

    # Proves the run actually reached the raising step rather than bailing out early, which
    # would make the no-raise assertion above vacuous.
    assert shown == [fm._SUCCESS_MSG], f"success notification was not shown: {shown!r}"
    assert "feedback_message" in cleared, f"form was not cleared before the dismiss: {cleared!r}"


# ── the SHIPPED handler: ACCEPT vs DROP must be indistinguishable ────────────────
#
# This is a plain unit test on purpose -- no e2e marker. `addopts` excludes `e2e` from a
# default pytest AND from CI, so an e2e-only gate on this property is, in practice, no gate.
# Driving the real _submit also means the assertions are about values the shipped code
# actually emits, rather than about a return value only a test can see.


class _FakeInput:
    """Shiny `input` stand-in: every attribute is a zero-arg getter."""

    def __init__(self, **values: str) -> None:
        self._values = values

    def __getattr__(self, name: str):
        def _get() -> str:
            return self._values.get(name, "")

        return _get


class _FakeSession:
    """Records custom messages instead of writing to a websocket."""

    def __init__(self) -> None:
        self.custom_messages: list[tuple[str, dict]] = []
        self.http_conn = SimpleNamespace(headers={}, client=SimpleNamespace(host="198.51.100.7"))
        self.id = "fake-session"

    async def send_custom_message(self, name: str, payload: dict) -> None:
        self.custom_messages.append((name, payload))


def _capture_submit(monkeypatch, input_obj, session):
    """Register the real feedback_server and hand back its `_submit` coroutine.

    `reactive.effect` / `reactive.event` are swapped for identity decorators so the handler
    can be awaited directly. Nothing about the handler's body is stubbed -- this is the
    shipped code path.
    """
    captured = {}

    def _fake_effect(fn):
        captured["fn"] = fn
        return fn

    def _fake_event(*_args, **_kwargs):
        def _deco(fn):
            return fn

        return _deco

    monkeypatch.setattr(fm.reactive, "effect", _fake_effect)
    monkeypatch.setattr(fm.reactive, "event", _fake_event)
    fm.feedback_server(input_obj, None, session, None)
    assert "fn" in captured, "feedback_server did not register a reactive effect"
    return captured["fn"]


def _observe_submission(monkeypatch, *, honeypot: str) -> dict:
    """Run one real submission and return everything the client could observe."""
    notifications: list[tuple] = []
    clears: list[tuple] = []
    stored: list[dict] = []

    monkeypatch.setattr(fm.ui, "notification_show", lambda m, **kw: notifications.append((m, kw)))
    monkeypatch.setattr(fm.ui, "update_text_area", lambda i, **kw: clears.append((i, kw)))
    monkeypatch.setattr(fm.ui, "update_text", lambda i, **kw: clears.append((i, kw)))
    monkeypatch.setattr(fm, "append_feedback", lambda rec: stored.append(rec))
    monkeypatch.setattr(fm, "save_contact", lambda feedback_id, email: None)
    # A fresh limiter per run, so the second submission is not refused by the first.
    monkeypatch.setattr(fm, "_LIMITER", RateLimiter(max_per_window=5, window_s=3600))

    session = _FakeSession()
    submit = _capture_submit(
        monkeypatch,
        _FakeInput(
            feedback_message="a real bug report",
            feedback_contact="me@example.org",
            feedback_type="bug",
            feedback_website=honeypot,
        ),
        session,
    )
    asyncio.run(submit())

    return {
        "notifications": notifications,
        "clears": clears,
        "custom_messages": session.custom_messages,
        "stored": stored,
    }


def test_accept_and_drop_are_indistinguishable_to_the_client(monkeypatch):
    """A bot must not learn it was caught from ANY observable the handler emits.

    Notification text and kwargs, field clears, and the dismiss payload must match exactly.
    The single permitted difference is invisible from the browser: whether a record was
    stored.
    """
    accept = _observe_submission(monkeypatch, honeypot="")
    drop = _observe_submission(monkeypatch, honeypot="http://spam.example")

    assert accept["notifications"] == drop["notifications"], (
        "honeypot hit is distinguishable by its notification — "
        f"accept={accept['notifications']!r} drop={drop['notifications']!r}"
    )
    assert accept["clears"] == drop["clears"], (
        f"honeypot hit clears different fields — accept={accept['clears']!r} "
        f"drop={drop['clears']!r}"
    )
    assert accept["custom_messages"] == drop["custom_messages"], (
        "honeypot hit is distinguishable by the dismiss message — "
        f"accept={accept['custom_messages']!r} drop={drop['custom_messages']!r}"
    )

    # And the one difference that IS intended, asserted so the test cannot pass by both sides
    # doing nothing at all.
    assert len(accept["stored"]) == 1, "the genuine submission was not stored"
    assert drop["stored"] == [], "the honeypot submission was stored"
    assert accept["notifications"], "no notification was emitted by either arm"
    assert accept["custom_messages"] == [("hide-modal", {"id": fm.MODAL_ID})], (
        f"unexpected dismiss payload: {accept['custom_messages']!r}"
    )


# ── _finish_success: steps must fail INDEPENDENTLY ───────────────────────────────


def test_a_failed_notification_still_clears_and_dismisses(monkeypatch, caplog):
    """One dead step must not take the others with it.

    A single try around the whole block meant a notification_show raise produced no message,
    no clear, no dismiss and no surfaced error, after the record was already written — the
    same "assume it failed, submit again" duplicate that giving save_contact its own guard
    removed.
    """
    cleared: list[str] = []

    def _notification_boom(_msg, **_kw):
        raise RuntimeError("notification channel is gone")

    monkeypatch.setattr(fm.ui, "notification_show", _notification_boom)
    monkeypatch.setattr(fm.ui, "update_text_area", lambda i, **kw: cleared.append(i))
    monkeypatch.setattr(fm.ui, "update_text", lambda i, **kw: cleared.append(i))

    session = _FakeSession()
    with caplog.at_level(logging.WARNING, logger=fm._log.name):
        try:
            asyncio.run(fm._finish_success(session))
        except Exception as exc:  # noqa: BLE001 — any escape at all is the failure under test
            raise AssertionError(f"_finish_success let {exc!r} escape") from exc

    assert "feedback_message" in cleared, (
        f"a failed notification prevented the field clear: {cleared!r}"
    )
    assert session.custom_messages == [("hide-modal", {"id": fm.MODAL_ID})], (
        f"a failed notification prevented the modal dismiss: {session.custom_messages!r}"
    )
    assert any("success notification" in r.getMessage() for r in caplog.records), (
        "the failed step left no log evidence"
    )


def test_a_total_post_write_failure_is_logged_as_an_error(monkeypatch, caplog):
    """If the user is shown nothing at all, the operator must be able to see that.

    They will assume the submission failed and send it again, so silence here becomes a
    duplicate record with no trace of why.
    """

    def _boom(*_a, **_kw):
        raise RuntimeError("session is gone")

    monkeypatch.setattr(fm.ui, "notification_show", _boom)
    monkeypatch.setattr(fm.ui, "update_text_area", _boom)
    monkeypatch.setattr(fm.ui, "update_text", _boom)

    class _DeadSession:
        async def send_custom_message(self, name, payload):
            raise RuntimeError("websocket is closed")

    with caplog.at_level(logging.WARNING, logger=fm._log.name):
        try:
            asyncio.run(fm._finish_success(_DeadSession()))
        except Exception as exc:  # noqa: BLE001 — any escape at all is the failure under test
            raise AssertionError(f"_finish_success let {exc!r} escape") from exc

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, (
        "every post-write step failed and nothing was logged at ERROR — the user saw no "
        "confirmation and the operator has no way to know"
    )
    assert "resubmit" in errors[0].getMessage(), (
        f"the error does not say why it matters: {errors[0].getMessage()!r}"
    )
