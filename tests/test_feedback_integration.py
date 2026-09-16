"""Submit path -> real store, WITHOUT a browser, so it runs in a bare pytest and in CI.

``tests/test_e2e_feedback.py`` carries ``pytestmark = pytest.mark.e2e`` and ``addopts`` in
pyproject.toml is ``-m 'not e2e and not visual'``, so the browser->disk path has never run in
CI. These tests close the handler->disk half of it: they drive the SHIPPED ``_submit`` effect
(the same ``_capture_submit`` harness ``tests/test_feedback_modal.py`` uses) against REAL
files under ``tmp_path``, and assert on the bytes that land there.

Why real files. ``tests/test_feedback_modal.py`` monkeypatches ``append_feedback`` and
``save_contact`` away -- correct for the decisions it tests (classification, refusal copy,
post-write step independence) and blind to everything about the store. The PII split in
particular cannot be judged from a fake, and cannot be judged from ``read_feedback()``
either: that reader pops a literal ``contact`` key out of its OUTPUT while leaving it in the
FILE (``osmose/feedback.py:165``), so a reader-side "the address is not there" assertion
stays green with the address sitting on disk. Every negative here reads the raw file.

What this file deliberately does NOT re-assert, because it is already covered at equal or
greater strength:

* the full accept/drop observable equality (notification, kwargs, field clears, dismiss
  payload) -- ``test_feedback_modal.py::test_accept_and_drop_are_indistinguishable_to_the_client``;
* the rate-limit refusal wording -- ``test_feedback_modal.py::test_rate_limit_notice_*``;
* the limiter's own window/eviction behaviour -- ``tests/test_feedback_limits.py``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import ui.components.feedback_modal as fm
from osmose import __version__
from osmose.feedback import lookup_contact
from osmose.feedback_limits import RateLimiter
from tests.test_feedback_modal import _capture_submit, _FakeInput, _FakeSession

# The SHIPPED limiter configuration, read once at import -- before any fixture swaps the
# module global. Derived rather than hardcoded: a literal 5 here would silently stop matching
# production the day the cap changes, and the cap test would then prove nothing.
_CAP = fm._LIMITER.max_per_window
_WINDOW_S = fm._LIMITER.window_s


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """Route BOTH stores into tmp_path and hand the handler a fresh rate-limit bucket.

    ``autouse`` on purpose: no test in this file can write to the real ``data/feedback/`` by
    forgetting the fixture. Setting ``OSMOSE_CONTACTS_FILE`` matters as much as the feedback
    one -- ``tests/test_e2e_feedback.py`` assigns ``OSMOSE_FEEDBACK_FILE`` at module IMPORT
    time (so it is set for the whole session once that module is collected, deselected or
    not) and never touches the contacts variable, which would otherwise resolve to the
    default repo path and put a real address there.

    The limiter is replaced by a fresh one carrying the shipped configuration, so these tests
    neither inherit a partly-spent bucket nor leak submissions into one.
    """
    feedback = tmp_path / "feedback.jsonl"
    contacts = tmp_path / "contacts.jsonl"
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(feedback))
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(contacts))
    monkeypatch.setattr(fm, "_LIMITER", RateLimiter(max_per_window=_CAP, window_s=_WINDOW_S))
    return SimpleNamespace(feedback=feedback, contacts=contacts)


def _records(path: Path) -> list[dict]:
    """Every JSON line in ``path``, in FILE order (oldest first) -- not ``read_feedback``.

    ``read_feedback`` reverses, and more importantly normalises: it would hide exactly the
    defect the privacy test exists to catch.
    """
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _submit(
    monkeypatch,
    *,
    message: str,
    contact: str = "",
    honeypot: str = "",
    kind: str = "bug",
    nav_tab: str = "run",
) -> list[tuple[str, dict]]:
    """Run one submission through the real handler; return what the user was shown.

    Only the ``ui`` side effects are stubbed (they need a live Shiny session). The store
    functions are deliberately left alone -- they are the subject.
    """
    notifications: list[tuple[str, dict]] = []
    monkeypatch.setattr(fm.ui, "notification_show", lambda m, **kw: notifications.append((m, kw)))
    monkeypatch.setattr(fm.ui, "update_text_area", lambda i, **kw: None)
    monkeypatch.setattr(fm.ui, "update_text", lambda i, **kw: None)

    input_obj = _FakeInput(
        feedback_message=message,
        feedback_contact=contact,
        # Always explicit: an unset type reads as "" and build_feedback_record raises, which
        # turns every positive control in this file into a save-failure red.
        feedback_type=kind,
        feedback_website=honeypot,
        main_nav=nav_tab,
    )
    submit = _capture_submit(monkeypatch, input_obj, _FakeSession())
    asyncio.run(submit())
    return notifications


def test_a_valid_submission_lands_on_disk_as_a_readable_record(monkeypatch, store):
    """The whole point of the feature: the report survives the process, and can be read back."""
    notes = _submit(
        monkeypatch,
        message="integration bug report alpha",
        kind="suggestion",
        nav_tab="calibration",
    )

    recs = _records(store.feedback)
    assert len(recs) == 1, f"expected exactly one record on disk, got {recs!r}"
    rec = recs[0]
    assert rec["message"] == "integration bug report alpha"
    assert rec["type"] == "suggestion", f"the radio choice was not carried through: {rec!r}"
    assert rec["nav_tab"] == "calibration", f"the tab context was not carried through: {rec!r}"
    assert rec["version"] == __version__, f"the app version was not recorded: {rec!r}"
    assert rec["has_contact"] is False, "no address was supplied but the flag says otherwise"
    assert rec["id"] and rec["ts"], f"record is missing its id/timestamp: {rec!r}"

    # Guard: without this, "a record is on disk" would also pass if the handler had written it
    # and then told the user it failed -- which is the resubmit-and-duplicate failure.
    assert notes and notes[0][0] == fm._SUCCESS_MSG, (
        f"the record was stored but the user was not told so: {notes!r}"
    )


def test_the_address_lands_in_contacts_and_never_in_the_feedback_record(monkeypatch, store):
    """The feature's core privacy split, asserted against real file contents.

    ``test_feedback.py::test_contact_is_not_in_the_main_record`` makes the same claim at the
    library level, but its file-side assertion goes through ``read_feedback()``, whose
    ``rec.pop("contact", "")`` scrubs the OUTPUT and not the file. Here the negative reads the
    raw bytes, and the whole thing is driven through the shipped handler rather than by
    calling the store functions in the order the test itself chose.
    """
    addr = "privacy-probe@example.org"
    _submit(monkeypatch, message="with an address", contact=addr)
    _submit(monkeypatch, message="without an address")

    # POSITIVE first: the address really did reach disk, in the side-store, keyed to the
    # record that carried it. Without this the negative below could pass simply because
    # nothing was stored anywhere.
    contact_rows = _records(store.contacts)
    assert len(contact_rows) == 1, (
        f"expected exactly one contacts row (one of the two submissions carried an address), "
        f"got {contact_rows!r}"
    )
    assert contact_rows[0]["email"] == addr, f"the address was not stored intact: {contact_rows!r}"

    recs = _records(store.feedback)
    assert [r["message"] for r in recs] == ["with an address", "without an address"], (
        f"both submissions should be stored, oldest first: {recs!r}"
    )
    assert contact_rows[0]["id"] == recs[0]["id"], (
        "the contacts row is not keyed to the record that carried the address — the operator "
        f"cannot link them back: {contact_rows[0]!r} vs {recs[0]!r}"
    )
    assert recs[0]["has_contact"] is True, "an address was supplied but the record says no"
    assert recs[1]["has_contact"] is False, "no address was supplied but the record says yes"
    assert lookup_contact(recs[0]["id"]) == addr
    assert lookup_contact(recs[1]["id"]) is None

    # NEGATIVE, against the raw file: the record is designed to be pasted into a public issue.
    raw = store.feedback.read_text(encoding="utf-8")
    assert "with an address" in raw, f"the line being asserted about is not in the file: {raw!r}"
    assert addr not in raw, (
        f"the email address is sitting in the feedback store, which is the file designed to be "
        f"copied into a public GitHub issue: {raw!r}"
    )


def test_a_filled_honeypot_writes_nothing_while_still_looking_like_success(monkeypatch, store):
    """A bot's submission is dropped before the store, and it cannot tell from the response.

    The success-notification assertion here is NOT a duplicate of
    ``test_accept_and_drop_are_indistinguishable_to_the_client`` (which owns the full
    cross-arm equality of every observable): it is the guard that gives "nothing was written"
    a meaning. Without it, an empty store also passes when the handler failed outright and
    showed ``_RETRY_SAVE_MSG``.
    """
    bait = "bot payload zulu"
    notes = _submit(monkeypatch, message=bait, honeypot="http://spam.example")

    assert _records(store.feedback) == [], "the honeypot submission was written to the store"
    assert notes and notes[0][0] == fm._SUCCESS_MSG, (
        f"the drop did not look like success to the client, which names the trap field: {notes!r}"
    )
    assert notes[0][1].get("type") == "message", (
        f"the drop's notification style differs from a real success: {notes!r}"
    )

    # Positive control, in the SAME store: a clean submission does land, so the emptiness
    # above is the honeypot's doing and not a dead store path.
    _submit(monkeypatch, message="a genuine report")
    assert [r["message"] for r in _records(store.feedback)] == ["a genuine report"]
    assert bait not in store.feedback.read_text(encoding="utf-8"), (
        "the honeypot payload reached the store after all"
    )


def test_the_submission_after_the_cap_is_refused_within_the_window(monkeypatch, store):
    """Submission ``_CAP + 1`` from one client is refused, and nothing more reaches disk.

    Only the *outcome* is asserted -- the refusal wording belongs to
    ``test_feedback_modal.py::test_rate_limit_notice_*`` and the limiter's own semantics to
    ``tests/test_feedback_limits.py``. What is new here is that the shipped handler, with the
    shipped cap, stops the record from being written.
    """
    started = time.time()
    for i in range(_CAP):
        notes = _submit(monkeypatch, message=f"under the cap {i}")
        assert notes and notes[0][0] == fm._SUCCESS_MSG, (
            f"submission {i + 1} of {_CAP} was refused before the cap was reached: {notes!r}"
        )
    assert [r["message"] for r in _records(store.feedback)] == [
        f"under the cap {i}" for i in range(_CAP)
    ], f"the submissions under the cap did not all land: {_records(store.feedback)!r}"

    over = "over the cap"
    notes = _submit(monkeypatch, message=over)

    assert notes, "the refused submission showed the user nothing at all"
    msg, kwargs = notes[0]
    assert msg != fm._SUCCESS_MSG, "a submission over the cap was reported as saved"
    assert kwargs.get("type") == "warning", f"a refusal was not styled as one: {notes!r}"
    assert len(_records(store.feedback)) == _CAP, (
        f"the refused submission was written anyway: {_records(store.feedback)!r}"
    )
    assert over not in store.feedback.read_text(encoding="utf-8")

    # "Within the window" is a claim about elapsed time, so assert it rather than assume it:
    # if these submissions ever took longer than the window, the refusal above would be
    # testing nothing.
    assert time.time() - started < _WINDOW_S, (
        f"the {_CAP + 1} submissions took longer than the rate-limit window ({_WINDOW_S}s)"
    )
