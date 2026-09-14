"""Integration test for the token-gated read-only feedback API route."""

from __future__ import annotations

import logging

import pytest
from starlette.testclient import TestClient

from osmose.feedback import append_feedback, build_feedback_record


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(tmp_path / "fb.jsonl"))
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    append_feedback(build_feedback_record("bug", "api round trip"))  # writes to env path
    from app import app

    return TestClient(app.starlette_app)


def test_no_token_forbidden(client):
    assert client.get("/api/feedback").status_code == 403


def test_wrong_token_forbidden(client):
    assert client.get("/api/feedback", headers={"X-Feedback-Token": "nope"}).status_code == 403


def test_correct_token_returns_records(client):
    r = client.get("/api/feedback", headers={"X-Feedback-Token": "secret"})
    assert r.status_code == 200
    assert any(rec["message"] == "api round trip" for rec in r.json())


def test_api_error_path_logs_server_side_and_discloses_nothing(client, monkeypatch, caplog):
    """Sibling of the review page's 500: a total failure of this route must not be silent either.

    The response stays bare (the raiser's message embeds the token on purpose, and it must not come
    back); the traceback goes to the server log. Asserted, not assumed -- the review route carried
    a comment claiming it failed loudly while logging nothing at all.
    """
    import app as app_module

    def _boom(*args, **kwargs):
        raise RuntimeError("store exploded while holding token secret")

    monkeypatch.setattr(app_module, "read_feedback", _boom)
    with caplog.at_level(logging.ERROR, logger="osmose.app"):
        resp = client.get("/api/feedback", headers={"X-Feedback-Token": "secret"})

    assert resp.status_code == 500  # positive control: we really took the failure path
    logged = [r for r in caplog.records if r.name == "osmose.app"]
    assert logged, "the 500 path logged NOTHING"
    assert any(r.levelno >= logging.ERROR and r.exc_info for r in logged), (
        "the log line must carry the traceback (`_log.exception`)"
    )
    assert resp.json() == {"error": "internal"}
    assert "store exploded" not in resp.text
    assert "secret" not in resp.text
