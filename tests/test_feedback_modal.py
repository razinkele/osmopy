"""Unit tests for the feedback submit modal (markup + client-key + rate-limit copy).

The submit effect itself is glue over these seams and is exercised end-to-end by
``tests/test_e2e_feedback.py`` (opt-in: ``-m e2e``). What is unit-tested here is everything
that has a decision in it: the markup contract the brief pins down, the per-client key
derivation (which is security-relevant -- see ``_client_key``) and the refusal copy (which
must never assert a cause the limiter cannot prove).
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import ui.components.feedback_modal as fm
from osmose.feedback_limits import MAX_KEYS, RateLimiter
from ui.components.feedback_modal import feedback_modal


def test_modal_has_honeypot_and_is_visually_hidden():
    html = str(feedback_modal())
    assert "feedback_website" in html  # honeypot field present
    assert "position:absolute" in html or "d-none" in html  # and hidden from real users


def test_modal_offers_bug_and_feature_and_other():
    html = str(feedback_modal())
    for label in ("Bug report", "Feature request", "Other"):
        assert label in html


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
    """Never raise out of the key derivation -- a crash here kills the whole submission."""
    key_fn = getattr(fm, "_client_key", None)
    assert key_fn is not None, "_client_key is not implemented"
    _reset_warn_flags(monkeypatch)
    assert isinstance(key_fn(SimpleNamespace()), str)


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
