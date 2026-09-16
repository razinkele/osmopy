"""Unit tests for osmose.feedback (store + token check)."""

from __future__ import annotations

import json

import pytest

from osmose.feedback import (
    MAX_CONTACT,
    MAX_STORE_BYTES,
    append_feedback,
    build_feedback_record,
    check_feedback_token,
    looks_like_email,
    lookup_contact,
    read_feedback,
    save_contact,
)


def test_build_record_fields():
    r = build_feedback_record(
        "bug", "  it broke  ", contact="me@x.io", version="0.13.0", nav_tab="run"
    )
    assert r["type"] == "bug" and r["message"] == "it broke"  # stripped
    assert r["has_contact"] is True and r["version"] == "0.13.0" and r["nav_tab"] == "run"
    assert r["id"] and r["ts"]
    assert "contact" not in r  # address never lives on the record, only the boolean flag
    assert "me@x.io" not in json.dumps(r)


def test_build_record_no_contact_has_contact_false():
    r = build_feedback_record("bug", "m")
    assert r["has_contact"] is False
    assert "contact" not in r


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


def test_read_skips_non_object_line(tmp_path):
    """Valid JSON that is not an object (`null`, `[]`) must be skipped, not raise.

    It used to raise AttributeError on `.setdefault`, which failed the whole read -- and so
    took the maintainer review page down entirely, hiding every good record behind one bad line.
    """
    p = tmp_path / "fb.jsonl"
    append_feedback(build_feedback_record("bug", "before"), path=p)
    with open(p, "a", encoding="utf-8") as f:
        f.write('null\n[]\n"just a string"\n')
    append_feedback(build_feedback_record("bug", "after"), path=p)
    try:
        out = read_feedback(path=p)
    except Exception as exc:
        raise AssertionError(f"read_feedback raised on a non-object line: {exc!r}") from exc
    assert [r["message"] for r in out] == ["after", "before"]  # newest first, both survive


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


def test_contact_is_capped(tmp_path, monkeypatch):
    # Task 2 moved the address out of the record; the record only ever carries a boolean.
    # The cap is still a real requirement -- it now applies where the address is actually
    # stored, i.e. the contacts side-store reached via save_contact/lookup_contact.
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(tmp_path / "c.jsonl"))
    r = build_feedback_record("bug", "m", contact="c" * 5000)
    assert r["has_contact"] is True
    save_contact(r["id"], "c" * 5000)
    stored = lookup_contact(r["id"])
    assert stored is not None
    assert len(stored) == MAX_CONTACT


def test_contact_is_not_in_the_main_record(tmp_path, monkeypatch):
    feedback_file = tmp_path / "f.jsonl"
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(feedback_file))
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(tmp_path / "c.jsonl"))
    rec = build_feedback_record("bug", "m", contact="user@example.org")
    append_feedback(rec)
    save_contact(rec["id"], "user@example.org")
    stored = read_feedback()[0]
    assert stored["has_contact"] is True
    assert "user@example.org" not in json.dumps(stored)
    assert lookup_contact(rec["id"]) == "user@example.org"
    assert lookup_contact("nope") is None
    # The reader is NOT a witness for the file: read_feedback() does
    # `rec.pop("contact", "")` on every line, so it would scrub an address out of its own
    # OUTPUT and report a clean record while the address sat in the store. Measured
    # 2026-09-14: with `"contact": contact` put back into build_feedback_record, every
    # assertion above still passed. Assert the bytes.
    raw = feedback_file.read_text(encoding="utf-8")
    # Control on the negative below: the id is unique to this record, so this fails if the
    # line is not there at all. (A message of "m" would not be a control -- every record's
    # "message" key contains an m.)
    assert rec["id"] in raw, f"the line being asserted about is not in the file: {raw!r}"
    assert "user@example.org" not in raw, f"the address is in the feedback store: {raw!r}"


def test_legacy_contact_record_is_normalised_on_read(tmp_path):
    # D6 backwards compatibility: data/feedback/feedback.jsonl may already hold records
    # written by the OLD code, which carry a literal "contact" key with a real email in it.
    # After normalisation, read_feedback() must NEVER return an address, even for these.
    p = tmp_path / "legacy.jsonl"
    legacy = {
        "id": "legacy-id-1",
        "ts": "2026-01-01T00:00:00",
        "type": "bug",
        "message": "old-style record",
        "contact": "old@example.org",
        "version": "0.1.0",
        "nav_tab": "run",
    }
    p.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    out = read_feedback(path=p)
    assert len(out) == 1
    rec = out[0]
    assert rec["has_contact"] is True
    assert "contact" not in rec
    assert "old@example.org" not in json.dumps(rec)


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


def test_contacts_store_refuses_when_over_cap(tmp_path, monkeypatch):
    """The PII store gets the SAME size cap as the public one.

    Until 2026-09-14 the cap and the flock lived only in ``append_feedback``, so the file
    designed to be pasted into a public issue had both protections while ``contacts.jsonl`` --
    the one holding email addresses -- had neither. The asymmetry ran backwards.
    """
    p = tmp_path / "c.jsonl"
    p.write_bytes(b"x" * (MAX_STORE_BYTES + 1))
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(p))
    with pytest.raises(RuntimeError, match="contacts store is full"):
        save_contact("some-id", "user@example.org")


def test_contacts_store_under_cap_still_writes(tmp_path, monkeypatch):
    """Positive control: the guard must refuse an OVER-cap file, not every existing file.

    The file is pre-created and non-empty on purpose. A guard mutated to trip on any existing
    file (or on any size >= 0) would still pass against a fresh path, so testing the empty case
    alone would not constrain the comparison at all.
    """
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "older", "email": "prior@example.org"}\n', encoding="utf-8")
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(p))
    save_contact("some-id", "user@example.org")
    assert lookup_contact("some-id") == "user@example.org", "append to an under-cap file was lost"
    assert lookup_contact("older") == "prior@example.org", "the pre-existing line was clobbered"


def test_contacts_write_takes_an_exclusive_lock(tmp_path, monkeypatch):
    """A concurrent append must not be able to interleave a half-written PII line."""
    import fcntl

    locks: list[int] = []
    real_flock = fcntl.flock
    monkeypatch.setattr(
        fcntl, "flock", lambda fd, op: (locks.append(op), real_flock(fd, op))[1], raising=True
    )
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(tmp_path / "c.jsonl"))
    save_contact("some-id", "user@example.org")
    assert locks == [fcntl.LOCK_EX], (
        f"save_contact did not take an exclusive lock (flock ops seen: {locks})"
    )
