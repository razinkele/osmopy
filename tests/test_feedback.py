"""Unit tests for osmose.feedback (store + token check)."""

from __future__ import annotations

import pytest

from osmose.feedback import (
    MAX_CONTACT,
    MAX_STORE_BYTES,
    append_feedback,
    build_feedback_record,
    check_feedback_token,
    looks_like_email,
    read_feedback,
)


def test_build_record_fields():
    r = build_feedback_record(
        "bug", "  it broke  ", contact="me@x.io", version="0.13.0", nav_tab="run"
    )
    assert r["type"] == "bug" and r["message"] == "it broke"  # stripped
    assert r["contact"] == "me@x.io" and r["version"] == "0.13.0" and r["nav_tab"] == "run"
    assert r["id"] and r["ts"]


def test_build_record_unknown_type_raises():
    with pytest.raises(ValueError):
        build_feedback_record("spam", "x")


def test_build_record_empty_message_raises():
    with pytest.raises(ValueError):
        build_feedback_record("bug", "   ")


def test_build_record_truncates_long_message():
    r = build_feedback_record("bug", "x" * 6000)
    assert len(r["message"]) == 5000


def test_append_read_round_trip_newest_first(tmp_path):
    p = tmp_path / "fb.jsonl"
    append_feedback(build_feedback_record("bug", "first"), path=p)
    append_feedback(build_feedback_record("suggestion", "second"), path=p)
    out = read_feedback(path=p)
    assert [r["message"] for r in out] == ["second", "first"]  # newest-first


def test_read_missing_file_is_empty(tmp_path):
    assert read_feedback(path=tmp_path / "nope.jsonl") == []


def test_read_skips_corrupt_line(tmp_path):
    p = tmp_path / "fb.jsonl"
    append_feedback(build_feedback_record("bug", "ok"), path=p)
    with open(p, "a", encoding="utf-8") as f:
        f.write("{ not json\n")
    out = read_feedback(path=p)
    assert len(out) == 1 and out[0]["message"] == "ok"


def test_check_token_unset_env_is_false(monkeypatch):
    monkeypatch.delenv("OSMOSE_FEEDBACK_TOKEN", raising=False)
    assert check_feedback_token("anything") is False


def test_check_token_matching_and_mismatch(monkeypatch):
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    assert check_feedback_token("secret") is True
    assert check_feedback_token("wrong") is False
    assert check_feedback_token(None) is False


def test_check_token_non_ascii_is_false_not_raise(monkeypatch):
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    assert check_feedback_token("café") is False  # must not raise TypeError


def test_env_override_path(tmp_path, monkeypatch):
    p = tmp_path / "override.jsonl"
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(p))
    append_feedback(build_feedback_record("bug", "via env"))  # no path= → uses env
    assert [r["message"] for r in read_feedback()] == ["via env"]


def test_contact_is_capped():
    r = build_feedback_record("bug", "m", contact="c" * 5000)
    assert len(r["contact"]) == MAX_CONTACT


def test_honeypot_non_empty_is_rejected():
    with pytest.raises(ValueError, match="honeypot"):
        build_feedback_record("bug", "m", honeypot="i am a bot")


@pytest.mark.parametrize(
    "s,ok",
    [
        ("a@b.co", True),
        ("first.last+tag@sub.example.org", True),
        ("", False),
        ("no-at-sign", False),
        ("a@", False),
        ("@b.co", False),
        ("a b@c.co", False),
    ],
)
def test_looks_like_email(s, ok):
    assert looks_like_email(s) is ok


def test_append_refuses_when_store_is_over_cap(tmp_path, monkeypatch):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b"x" * (MAX_STORE_BYTES + 1))
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(p))
    with pytest.raises(RuntimeError, match="store is full"):
        append_feedback(build_feedback_record("bug", "m"))
