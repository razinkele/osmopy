"""Integration tests for the token-gated maintainer review page (`GET /feedback/review`).

Every value this page renders arrives from a PUBLIC form, so the tests below are written as
injection probes: each negative assertion ("no raw markup", "no address") is paired with a
positive control proving the very record it is about actually reached the page. A negative
assertion on its own passes against a blank page, which is not evidence of anything.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

from osmose.feedback import (
    append_feedback,
    build_feedback_record,
    lookup_contact,
    save_contact,
)

_MARKER = "BENIGN_MARKER_42"
# Closes the <pre> the message is rendered in, then injects a script. If the message is not
# escaped, `_pre_blocks` below stops at the injected `</pre>` and the marker falls outside the
# block -- which is exactly how the positive control catches an unescaped render.
_XSS_MESSAGE = f"</pre><script>alert('xss')</script>{_MARKER}"
_EMAIL = "maintainer.probe@example.org"
# Spelled out rather than imported from app: this asserts the page links to THE repository,
# not merely to whatever string app.py happened to pass in.
_REPO_URL = "https://github.com/razinkele/osmopy"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Same shape as tests/test_feedback_api.py: env first, `app` imported INSIDE the fixture.

    `OSMOSE_CONTACTS_FILE` is redirected too -- without it `save_contact` would write a real
    address into the repo's gitignored `data/feedback/contacts.jsonl`, where `git status` would
    never show it.
    """
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(tmp_path / "fb.jsonl"))
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(tmp_path / "contacts.jsonl"))
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    from app import app

    return TestClient(app.starlette_app)


def _pre_blocks(body: str) -> list[str]:
    """Contents of every `<pre>` element -- lets a test assert on the message, not the page."""
    return re.findall(r"<pre[^>]*>(.*?)</pre>", body, re.S)


def _get(client, token: str | None = "secret"):
    headers = {} if token is None else {"x-feedback-token": token}
    return client.get("/feedback/review", headers=headers)


def test_review_requires_token(client):
    append_feedback(build_feedback_record("bug", "token gate probe"))
    anon = _get(client, token=None)
    wrong = _get(client, token="nope")
    assert anon.status_code == 403
    assert wrong.status_code == 403
    for resp in (anon, wrong):
        assert "secret" not in resp.text  # the token itself must never come back
        assert "token gate probe" not in resp.text  # no record content without auth


def test_review_renders_records_and_escapes_html(client):
    append_feedback(build_feedback_record("bug", _XSS_MESSAGE, version="1.2.3", nav_tab="Results"))
    resp = _get(client)
    assert resp.status_code == 200
    body = resp.text

    blocks = _pre_blocks(body)
    assert len(blocks) == 1, f"expected exactly one message block, got {len(blocks)}"
    msg = blocks[0]
    assert _MARKER in msg  # positive control: this record's message really rendered...
    assert "&lt;script&gt;" in msg  # ... and rendered escaped
    assert "&lt;/pre&gt;" in msg  # ... including the block-breakout attempt
    assert "<script" not in msg
    assert "<script>alert" not in body  # page-wide: no raw markup anywhere


def test_review_links_to_the_repository(client):
    """R15: `repo_url` must be a live parameter, not a dead one the page ignores."""
    append_feedback(build_feedback_record("other", "repo link probe"))
    body = _get(client).text
    assert "repo link probe" in body  # positive control: the page rendered at all
    assert f'href="{_REPO_URL}"' in body


def test_review_escapes_record_metadata(client):
    # Hand-built (not via build_feedback_record) so that `ts`, `type` and `id` carry markup too:
    # append_feedback takes any dict, and a corrupt/hostile line on disk must render safely.
    append_feedback(
        {
            "id": 'abc"><b>ID_MARK</b>',
            "ts": '2026-09-14T12:00:00"><b>TS_MARK</b>',
            "type": 'bug"><b>TYPE_MARK</b>',
            "message": "metadata probe MSG_MARK",
            "has_contact": False,
            "version": '9.9"><img src=x onerror=alert(1)>VER_MARK',
            "nav_tab": 'Results"><b>TAB_MARK</b>',
        }
    )
    body = _get(client).text

    # Positive controls: every interpolated field actually reached the page.
    assert "metadata probe MSG_MARK" in body
    for mark in ("ID_MARK", "TS_MARK", "TYPE_MARK", "VER_MARK", "TAB_MARK"):
        assert mark in body, f"{mark} never rendered -- the negatives below would be vacuous"

    assert "&quot;&gt;&lt;b&gt;" in body  # the attribute-breakout sequence, escaped
    assert '"><b>' not in body  # ... and never raw
    assert "&lt;img src=x" in body  # the img payload survives only as inert text
    assert "<img" not in body  # ... never as a tag ("onerror=..." as text is harmless)


def test_review_never_renders_an_email_address(client):
    rec = build_feedback_record("suggestion", "please add EMAIL_PROBE_MSG", contact=_EMAIL)
    append_feedback(rec)
    save_contact(rec["id"], _EMAIL)
    # The address IS on disk and IS retrievable by id -- so this test can genuinely fail.
    assert lookup_contact(rec["id"]) == _EMAIL
    assert rec["has_contact"] is True

    body = _get(client).text
    assert "EMAIL_PROBE_MSG" in body  # positive control: the record rendered...
    assert 'data-has-contact="yes"' in body  # ... and is flagged as having a contact
    assert _EMAIL not in body
    assert "maintainer.probe" not in body
    assert "@example.org" not in body


def test_review_error_path_returns_bare_internal(client, monkeypatch):
    import app as app_module

    def _boom(*args, **kwargs):
        raise RuntimeError("store exploded while holding token secret")

    monkeypatch.setattr(app_module, "read_feedback", _boom)
    resp = _get(client)
    assert resp.status_code == 500
    assert resp.text.strip() == "internal"
    assert "Traceback" not in resp.text
    assert "store exploded" not in resp.text
