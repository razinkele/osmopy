"""Unit tests for the feedback submit modal (markup + client-key + rate-limit copy).

The submit effect itself is glue over these seams and is exercised end-to-end by
``tests/test_e2e_feedback.py`` (opt-in: ``-m e2e``). What is unit-tested here is everything
that has a decision in it: the markup contract the brief pins down, the per-client key
derivation (which is security-relevant -- see ``_client_key``) and the refusal copy (which
must never assert a cause the limiter cannot prove).
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import re
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

import ui.components.feedback_modal as fm
import osmose.feedback as osmose_feedback
from osmose.feedback import MAX_NAV_TAB
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
    monkeypatch.setattr(fm, "_warned_proxy_no_xff", False, raising=False)


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


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.1", "172.16.0.5", "192.168.1.1", "::1"])
def test_client_key_warns_once_when_a_proxy_sends_no_xff_at_all(monkeypatch, caplog, host):
    """The FOURTH degraded path, and the only one that used to be completely silent.

    A reverse proxy that forwards no ``X-Forwarded-For`` skips the whole ``if xff:`` block, so
    none of the other three warnings can fire -- yet ``client.host`` is then the PROXY's own
    address and every client shares one bucket. Measured live on this code 2026-09-14: the key
    came back ``'127.0.0.1'`` for two different sessions and nothing at all was logged.

    The key must NOT change -- a constant bucket is the correct behaviour once the address is
    unavailable. It is the silence that is the defect.
    """
    _reset_warn_flags(monkeypatch)
    monkeypatch.delenv("OSMOSE_TRUSTED_PROXY", raising=False)
    sess = _fake_session(host=host)  # no headers at all -> no XFF
    logger_name = fm._log.name
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert fm._client_key(sess) == host, "the warning must not change the key"
        assert fm._client_key(_fake_session(host=host)) == host
    warnings = [r for r in caplog.records if r.name == logger_name and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"expected exactly one warn-once record for a silent proxy hop from {host!r}, "
        f"got {len(warnings)}: {[r.getMessage() for r in warnings]}"
    )
    text = warnings[0].getMessage()
    assert "X-Forwarded-For" in text, f"warning does not name the missing header: {text!r}"
    assert "OSMOSE_TRUSTED_PROXY" in text, f"warning does not name the fix: {text!r}"


@pytest.mark.parametrize("host", ["198.51.100.7", "8.8.8.8", "not-an-ip-address"])
def test_client_key_stays_silent_for_a_genuine_direct_connection(monkeypatch, caplog, host):
    """Positive control for the warning above: it must NOT fire on a real direct deployment.

    ``198.51.100.7`` is the address the existing direct-connection test uses and, critically,
    ``ipaddress.ip_address('198.51.100.7').is_private`` is **True** -- Python counts the RFC5737
    documentation ranges as private. Keying the heuristic off ``is_private`` would therefore
    warn about a perfectly healthy direct deployment (and redden that test). This asserts the
    narrower loopback+RFC1918 rule that avoids it. An unparseable host must also stay silent
    rather than guess.
    """
    _reset_warn_flags(monkeypatch)
    monkeypatch.delenv("OSMOSE_TRUSTED_PROXY", raising=False)
    logger_name = fm._log.name
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert fm._client_key(_fake_session(host=host)) == host
    noise = [r.getMessage() for r in caplog.records if r.name == logger_name]
    assert noise == [], f"a direct connection from {host!r} was reported as a proxy hop: {noise}"


def test_client_key_does_not_double_warn_when_xff_was_already_reported(monkeypatch, caplog):
    """Modes (a) and (d) must not both fire for one misconfiguration.

    An untrusted XFF arriving from a loopback proxy satisfies the address heuristic too. Without
    the ``not xff`` gate the operator gets two warnings describing the same problem, which reads
    as two problems.
    """
    _reset_warn_flags(monkeypatch)
    monkeypatch.delenv("OSMOSE_TRUSTED_PROXY", raising=False)
    sess = _fake_session(headers={"x-forwarded-for": "1.2.3.4"}, host="127.0.0.1")
    logger_name = fm._log.name
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert fm._client_key(sess) == "127.0.0.1"
    warnings = [r for r in caplog.records if r.name == logger_name and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"one misconfiguration produced {len(warnings)} warnings: "
        f"{[r.getMessage() for r in warnings]}"
    )
    assert "OSMOSE_TRUSTED_PROXY is unset" in warnings[0].getMessage(), (
        f"the wrong one of the two warnings fired: {warnings[0].getMessage()!r}"
    )


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


# label -> (feedback type, message, contact, limiter already exhausted)
_ORACLE_PROBES = {
    "empty_message": ("bug", "", "", False),
    "malformed_email": ("bug", "a real bug report", "not-an-email", False),
    "rate_limited": ("bug", "a real bug report", "", True),
    "everything_valid": ("bug", "a real bug report", "me@example.org", False),
    # A crafted client can send any type. This one reopened the oracle once already: type
    # was validated only inside build_feedback_record, which runs AFTER the DROP branch
    # returns, so honeypot-empty gave the save-failure copy with the modal open while
    # honeypot-filled gave success and dismissed it.
    "unknown_type": ("bogus", "a real bug report", "", False),
    "empty_type": ("", "a real bug report", "", False),
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
    ftype, msg, contact, exhausted = _ORACLE_PROBES[label]
    clean = fm._classify_and_consume(
        ftype, msg, contact, "", _limiter(exhausted=exhausted), "ip1", 0.0
    )
    baited = fm._classify_and_consume(
        ftype, msg, contact, "http://spam.example", _limiter(exhausted=exhausted), "ip1", 0.0
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
    first = fm._classify_and_consume("bug", "bait", "", "http://spam.example", rl, "ip1", 0.0)
    second = fm._classify_and_consume("bug", "bait", "", "http://spam.example", rl, "ip1", 0.0)
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


def _store_with_append_raising(monkeypatch, exc: BaseException) -> str | None:
    """Drive ``_store_submission`` with ``append_feedback`` raising ``exc``; return the copy."""

    def _boom(rec):
        raise exc

    monkeypatch.setattr(fm, "append_feedback", _boom)
    monkeypatch.setattr(fm, "save_contact", lambda *a, **k: None)
    return fm._store_submission(
        type="bug", msg="it broke", contact="", honeypot="", version="0.0.0", nav_tab="run"
    )


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("feedback store is full (52428800 bytes) — rotate /x"),
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EROFS, "Read-only file system"),
        OSError(errno.ENOSPC, "No space left on device"),
    ],
    ids=["store-full", "eacces", "erofs", "enospc"],
)
def test_store_does_not_invite_a_retry_that_cannot_work(monkeypatch, exc):
    """A full store and an unwritable path both persist until an OPERATOR acts.

    "try again" is not merely unhelpful here, it is false: the user retries, it fails
    identically, and they conclude the app is broken.

    An earlier version of this docstring justified the EACCES case by claiming the production
    source tree is read-only to the service user, citing DEPLOY.md. That was measured and
    REFUTED (see DEPLOY.md's writable-state bullet, which now records the retraction): the tree
    is service-user writable and the default store path works. The BEHAVIOUR pinned here is
    unaffected and still correct -- EACCES, EROFS, ENOSPC and a full store are all terminal
    wherever they occur, and a read-only mount or a wrong-owner StateDirectory can still
    produce them. Only the "this is what production does" claim was wrong.
    """
    err = _store_with_append_raising(monkeypatch, exc)
    assert err == fm._TERMINAL_SAVE_MSG, f"terminal failure {exc!r} produced retry copy: {err!r}"
    assert "try again" not in err.lower(), f"terminal copy still invites a retry: {err!r}"


@pytest.mark.parametrize(
    "exc",
    [OSError("disk gremlins"), ValueError("something odd"), OSError(errno.EINTR, "Interrupted")],
    ids=["errno-less-oserror", "valueerror", "eintr"],
)
def test_store_still_invites_a_retry_when_the_failure_may_be_transient(monkeypatch, exc):
    """Positive control for the test above: the terminal copy must not swallow everything.

    A classifier that answered "terminal" for every exception would pass the terminal test
    while stranding users whose next attempt would have succeeded. An unknown failure keeps the
    retry invitation.
    """
    err = _store_with_append_raising(monkeypatch, exc)
    assert err == fm._RETRY_SAVE_MSG, (
        f"possibly-transient {exc!r} was reported as terminal: {err!r}"
    )


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("feedback store is full (52428800 bytes) — rotate /srv/osmose/feedback.jsonl"),
        PermissionError(errno.EACCES, "Permission denied: '/opt/app/data/feedback/feedback.jsonl'"),
    ],
    ids=["store-full", "eacces"],
)
def test_save_failure_copy_never_leaks_the_cause_to_the_user(monkeypatch, exc):
    """Task 5 measured error detail reaching the caller and found the TOKEN in the response.

    The rule that came out of it applies here too: detail goes to the log, never to the user.
    Both terminal causes must be indistinguishable in the UI -- otherwise an anonymous
    submitter has a probe into the server's filesystem state.
    """
    err = _store_with_append_raising(monkeypatch, exc)
    lowered = err.lower()
    for leak in ("errno", "permission", "denied", "full", "space", "read-only", "rotate", "/"):
        assert leak not in lowered, f"user-facing copy leaked {leak!r} from {exc!r}: {err!r}"


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


def _observe_submission(monkeypatch, *, honeypot: str, ftype: str = "bug") -> dict:
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
            feedback_type=ftype,
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


@pytest.mark.parametrize("ftype", ["bogus", "", "BUG"], ids=["unknown", "empty", "wrong-case"])
def test_a_crafted_feedback_type_is_not_a_honeypot_oracle(monkeypatch, ftype):
    """The oracle that came back through an unvalidated field.

    ``feedback_type`` is client-settable and was validated only inside
    ``build_feedback_record``, which runs in ``_store_submission`` -- AFTER the DROP branch has
    returned. So with a crafted type, honeypot-EMPTY produced the save-failure notification
    with the modal still open, while honeypot-FILLED produced the success notification and
    dismissed the modal. Neither stored anything; the difference named the trap field exactly
    as the email probe once did.

    This must be asserted at the HANDLER, not on ``_classify_and_consume``: at that level an
    unvalidated bad type returns ACCEPT/DROP, which the unit oracle test reads as correctly
    indistinguishable. The divergence only becomes observable once the ACCEPT path reaches the
    failing store call. A unit-level probe alone would pass with the bug present.
    """
    clean = _observe_submission(monkeypatch, honeypot="", ftype=ftype)
    baited = _observe_submission(monkeypatch, honeypot="http://spam.example", ftype=ftype)

    assert clean["notifications"] == baited["notifications"], (
        f"type={ftype!r}: the honeypot is distinguishable by notification — "
        f"clean={clean['notifications']!r} baited={baited['notifications']!r}"
    )
    assert clean["clears"] == baited["clears"], (
        f"type={ftype!r}: the honeypot is distinguishable by which fields get cleared — "
        f"clean={clean['clears']!r} baited={baited['clears']!r}"
    )
    assert clean["custom_messages"] == baited["custom_messages"], (
        f"type={ftype!r}: the honeypot is distinguishable by modal dismissal — "
        f"clean={clean['custom_messages']!r} baited={baited['custom_messages']!r}"
    )
    assert clean["stored"] == [] and baited["stored"] == [], (
        f"type={ftype!r}: an invalid type was stored — clean={clean['stored']!r} "
        f"baited={baited['stored']!r}"
    )


def test_a_refusal_notification_failure_does_not_kill_the_effect(monkeypatch):
    """A dead socket on a REFUSAL path must not raise out of the reactive effect.

    ``_finish_success`` has always guarded its own ``notification_show`` -- a transport failure
    after a record is written must not crash the session. The two refusal notifications in
    ``_submit`` had no such guard, for no reason anyone recorded: the same dead socket that is
    survivable one branch later was fatal here. Both now go through ``_notify``.

    Driven through the real handler with an empty message, so it is the shipped REJECT path that
    is exercised and not ``_notify`` in isolation.
    """
    boom = []

    def _dead_socket(message, **kw):
        boom.append(message)
        raise RuntimeError("websocket is closed")

    monkeypatch.setattr(fm.ui, "notification_show", _dead_socket)
    monkeypatch.setattr(fm, "_LIMITER", RateLimiter(max_per_window=5, window_s=3600))
    session = _FakeSession()
    submit = _capture_submit(
        monkeypatch,
        _FakeInput(feedback_message="", feedback_type="bug", feedback_website=""),
        session,
    )

    asyncio.run(submit())  # must not raise

    assert boom, "the refusal notification was never attempted — the test proved nothing"


# nav_tab: the only client-settable field that had no cap.
# Each payload is large enough that an UNCAPPED store would take ~4 requests to fill 50 MiB,
# which fits inside the 5-per-hour budget. `int` and `dict` additionally break the NAIVE fix
# `(nav_tab or "")[:200]` (TypeError / KeyError); `list` defeats it a third way, since slicing
# keeps 200 *elements* which can be 200 x 100 kB; `str` is the only one the naive fix handles,
# which is exactly why a str-only test would pass over a live hole.
_NAV_PAYLOADS = {
    "str": "x" * 2_000_000,
    "list": ["y" * 100_000] * 300,
    "dict": {"k" * 100_000: "v" * 100_000},
    "int": 10**4000,
}


@pytest.mark.parametrize("kind", sorted(_NAV_PAYLOADS))
def test_nav_tab_cannot_fill_the_store_or_reopen_the_oracle(monkeypatch, tmp_path, kind):
    """`nav_tab` is client-settable, was stored verbatim, and had no cap.

    Two consequences, both closed by capping it:

    * **The store is a shared, permanent resource.** ``MAX_STORE_BYTES`` is a whole-FILE guard
      with no per-record limit, and once tripped it stays tripped until an operator rotates. A
      handful of oversized ``nav_tab`` values inside the ordinary rate-limit budget therefore
      DoS the feature for everyone, permanently.
    * **A full store reopens the honeypot oracle.** Clean arm: terminal save error, modal stays
      open. Baited arm: success copy, modal dismissed. All three observables differ again.

    Driven against the REAL store on disk, not a monkeypatched ``append_feedback``: what is
    being asserted is how many BYTES reach the file, which a fake cannot show.
    """
    store = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(store))
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(tmp_path / "contacts.jsonl"))
    monkeypatch.setattr(fm, "_LIMITER", RateLimiter(max_per_window=5, window_s=3600))
    monkeypatch.setattr(fm.ui, "notification_show", lambda m, **kw: None)
    monkeypatch.setattr(fm.ui, "update_text_area", lambda i, **kw: None)
    monkeypatch.setattr(fm.ui, "update_text", lambda i, **kw: None)

    submit = _capture_submit(
        monkeypatch,
        _FakeInput(
            feedback_message="a real bug report",
            feedback_type="bug",
            feedback_website="",
            main_nav=_NAV_PAYLOADS[kind],
        ),
        _FakeSession(),
    )
    asyncio.run(submit())

    assert store.is_file(), f"{kind}: nothing was stored — the test proved nothing"
    size = store.stat().st_size
    assert size < 10_000, (
        f"{kind}: one submission wrote {size} bytes; ~{fm_max_store() // size} of them would "
        f"fill the {fm_max_store()} byte store permanently"
    )
    rec = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(rec["nav_tab"], str), (
        f"{kind}: nav_tab was stored as {type(rec['nav_tab']).__name__}, not str — an "
        "uncoerced value keeps its full size no matter what the cap says"
    )
    assert len(rec["nav_tab"]) <= MAX_NAV_TAB, (
        f"{kind}: nav_tab stored {len(rec['nav_tab'])} chars, cap is {MAX_NAV_TAB}"
    )


@pytest.mark.parametrize("kind", sorted(_NAV_PAYLOADS))
def test_a_crafted_nav_tab_cannot_exhaust_the_store_within_the_rate_limit(
    monkeypatch, tmp_path, kind
):
    """The exploit itself: spend the whole hourly budget on oversized `nav_tab` values.

    This is the assertion that matters, and it is deliberately NOT "the two honeypot arms look
    the same once the store is full". Once the store IS full, ACCEPT and DROP *are*
    distinguishable — the clean arm gets the terminal save error, the baited arm gets success —
    and no amount of capping changes that. The oracle is a CONSEQUENCE of a full store; the
    defence is preventing a client from filling one. So what is pinned here is that the whole
    5-per-hour budget cannot fill it.

    ``MAX_STORE_BYTES`` is patched down to 1 MiB so one uncapped submission (payloads are ~2 MiB
    and up) would blow it. Patching the constant rather than writing 50 MiB per parameter keeps
    the test honest about the ratio while staying fast.
    """
    store = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(store))
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(tmp_path / "contacts.jsonl"))
    monkeypatch.setattr(osmose_feedback, "MAX_STORE_BYTES", 1_000_000)
    monkeypatch.setattr(fm, "_LIMITER", RateLimiter(max_per_window=5, window_s=3600))
    notes: list = []
    monkeypatch.setattr(fm.ui, "notification_show", lambda m, **kw: notes.append(m))
    monkeypatch.setattr(fm.ui, "update_text_area", lambda i, **kw: None)
    monkeypatch.setattr(fm.ui, "update_text", lambda i, **kw: None)

    submit = _capture_submit(
        monkeypatch,
        _FakeInput(
            feedback_message="a real bug report",
            feedback_type="bug",
            feedback_website="",
            main_nav=_NAV_PAYLOADS[kind],
        ),
        _FakeSession(),
    )
    for _ in range(5):  # the entire per-client hourly budget
        asyncio.run(submit())

    assert store.is_file(), f"{kind}: nothing was ever stored — the test proved nothing"
    lines = [ln for ln in store.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 5, (
        f"{kind}: {len(lines)}/5 submissions stored — the store filled part-way through and "
        f"the feature is now broken for every other user. Notifications seen: {notes!r}"
    )
    size = store.stat().st_size
    assert size < osmose_feedback.MAX_STORE_BYTES, (
        f"{kind}: one client filled the store to {size} bytes against a "
        f"{osmose_feedback.MAX_STORE_BYTES} byte cap using nothing but nav_tab"
    )
    assert fm._TERMINAL_SAVE_MSG not in notes, (
        f"{kind}: the store hit its cap inside one client's hourly budget — {notes!r}"
    )


def fm_max_store() -> int:
    from osmose.feedback import MAX_STORE_BYTES

    return MAX_STORE_BYTES


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
    # The ERROR escalation is for a TOTAL failure only. Firing it here would make it noise and
    # would also let a `>= 1` comparison masquerade as the `== attempted` one.
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, (
        "one failed step out of several escalated to ERROR — that level is reserved for the "
        f"case where the user saw nothing at all: {[r.getMessage() for r in errors]}"
    )


def test_a_total_post_write_failure_is_logged_as_an_error(monkeypatch, caplog):
    """If the user is shown nothing at all, the operator must be able to see that.

    They will assume the submission failed and send it again, so silence here becomes a
    duplicate record with no trace of why.

    **This test pins the PROPERTY, not the step count.** It never says "five"; it makes the
    underlying calls raise and asserts the escalation fired. Add a sixth best-effort step that
    goes through the same ``ui`` functions and this keeps working, while an escalation compared
    against a hand-maintained literal would silently stop firing. A step added through some
    *other* mechanism will red this test rather than pass it — which is the correct signal to
    extend the patching below, not a reason to loosen the assertion.
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

    # Guard against the test silently stopping to force failures at all: without this, a
    # refactor that routed the steps elsewhere would leave the ERROR assertion below failing
    # for a reason nobody could read off the message.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "no step was made to fail — this test is no longer forcing the scenario"

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, (
        "every post-write step failed and nothing was logged at ERROR — the user saw no "
        f"confirmation and the operator has no way to know ({len(warnings)} steps failed)"
    )
    assert "resubmit" in errors[0].getMessage(), (
        f"the error does not say why it matters: {errors[0].getMessage()!r}"
    )
