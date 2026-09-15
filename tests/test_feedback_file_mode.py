"""The feedback stores must be 0600 on disk, enforced in code rather than by deployment.

`contacts.jsonl` holds reporter EMAIL ADDRESSES. Until this guard existed, `osmose/feedback.py`
set no mode at all — no chmod, no umask, no `mode=` — so the files inherited the process umask and
landed 0644. Measured in production 2026-09-15: both stores world-readable, any local user on the
host able to read the addresses. That was patched on that host with systemd's
`StateDirectoryMode=0700`, but a directory mode is a property of one deployment; these tests make
the file mode a property of the code, so it travels to the next host.

Both stores are asserted, not just the contacts one. `feedback.jsonl` is only PII-free when written
entirely by CURRENT code — a legacy v1 line can still carry an inline `contact` key (see CLAUDE.md;
`read_feedback` scrubs its OUTPUT, not the file). And `_append_json_line` is shared by both stores
precisely so protections cannot be added to one side only; its own docstring records the last time
that asymmetry happened, with the size cap and the flock. Parametrizing here keeps that honest.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from osmose.feedback import append_feedback, build_feedback_record, save_contact


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def _write_one(kind: str, p: Path) -> None:
    """Drive one record into whichever store `kind` names."""
    if kind == "feedback":
        append_feedback(build_feedback_record("bug", "a message"), path=p)
    else:
        save_contact("some-id", "user@example.org", path=p)


@pytest.mark.parametrize("kind", ["feedback", "contacts"])
def test_new_store_is_created_0600(tmp_path: Path, kind: str) -> None:
    """A store the code creates itself is never group- or world-readable.

    Asserted as an exact mode, not `not & S_IRWXO`: 0640 would also pass a
    world-readable-only check while still exposing the addresses to the host's
    shiny/adm-style groups, which is most of the risk on a shared box.
    """
    p = tmp_path / f"{kind}.jsonl"
    _write_one(kind, p)
    assert p.is_file()
    assert _mode(p) == 0o600, f"{kind} store created as {oct(_mode(p))}, expected 0o600"


@pytest.mark.parametrize("kind", ["feedback", "contacts"])
def test_existing_loose_store_is_tightened_on_next_append(tmp_path: Path, kind: str) -> None:
    """Self-healing: a store left 0644 by an older version is fixed on the next write.

    This is the case that actually occurred in production, so it is the one most worth
    gating. Creation-time mode alone would leave such a file loose forever.
    """
    p = tmp_path / f"{kind}.jsonl"
    p.write_text("")
    p.chmod(0o644)
    assert _mode(p) == 0o644  # precondition: the test is exercising the loose case

    _write_one(kind, p)
    assert _mode(p) == 0o600, f"{kind} store left at {oct(_mode(p))} after append"


@pytest.mark.parametrize("kind", ["feedback", "contacts"])
def test_tightening_does_not_lose_existing_content(tmp_path: Path, kind: str) -> None:
    """Fixing the mode must not truncate or drop what is already stored.

    A chmod-then-rewrite implementation could silently discard prior records; assert the
    earlier line survives alongside the new one.
    """
    p = tmp_path / f"{kind}.jsonl"
    _write_one(kind, p)
    p.chmod(0o644)
    _write_one(kind, p)

    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2, f"expected both records, got {len(lines)}"
    assert _mode(p) == 0o600


def test_chmod_failure_still_stores_the_record_and_warns(tmp_path, monkeypatch, caplog) -> None:
    """A store we cannot chmod must still accept the record, loudly.

    The realistic cause is a store file owned by another user (the app was run as root once),
    where the mode is both unfixable and the thing most worth knowing about. Raising there would
    trade a permissions problem for silent data loss, and a reporter cannot resubmit a record they
    already believe was sent. So the write must survive and the log must say so.
    """
    import osmose.feedback as fb

    p = tmp_path / "fb.jsonl"

    def _boom(*_a, **_k):
        raise PermissionError("not the owner")

    monkeypatch.setattr(fb.os, "chmod", _boom)

    with caplog.at_level("WARNING"):
        append_feedback(build_feedback_record("bug", "survives a chmod failure"), path=p)

    stored = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(stored) == 1, "record was lost when chmod failed"
    assert "survives a chmod failure" in stored[0]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("0600" in w for w in warnings), f"no warning naming the mode; saw {warnings}"
    assert any(str(p) in w for w in warnings), f"warning does not name the file; saw {warnings}"
