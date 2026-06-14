"""Integration test for the token-gated read-only feedback API route."""

from __future__ import annotations

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
