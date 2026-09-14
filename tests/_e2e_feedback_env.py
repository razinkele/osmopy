"""Shared scaffolding for the two feedback e2e modules (env isolation + modal dismissal).

WHY THIS EXISTS. Both e2e modules used to set ``OSMOSE_FEEDBACK_FILE`` with a bare
``os.environ[...] = ...`` at MODULE IMPORT time, because the ``create_app_fixture`` subprocess
must inherit the variable at launch and pytest imports every collected module before running
anything. Three problems came out of that:

1. **Merely collecting the module mutated the session environment**, and nothing ever restored
   it — `pytest tests/` with no `-m e2e` still imported these files (the marker filters
   execution, not collection) and left the variable set for every other test in the run.
2. The two modules fought over one variable, "solved" with ``setdefault`` in one of them and a
   plain assignment in the other — so which store each app wrote to depended on collection
   order.
3. **Neither module ever set ``OSMOSE_CONTACTS_FILE``.** The e2e submissions fill in an email
   address, so the first run wrote a real address into the developer's own
   ``data/feedback/contacts.jsonl``. Gitignored, so it would never have shown up in a diff.

A module-scoped fixture using ``pytest.MonkeyPatch()`` as a context manager fixes all three:
it runs at SETUP rather than import, it is undone at teardown, it gives each module its own
pair of files, and it sets both variables in one place so the contacts store cannot be
forgotten again. The app fixture must depend on it by name so the ordering is guaranteed
rather than inherited from autouse-ordering rules -- see the ``app`` fixtures in the two modules.

``playwright`` is imported lazily inside ``dismiss_modal`` rather than at module scope, so the
env-isolation half of this file can be unit-tested in a default (non-e2e) run on a machine with no
browser driver installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover — annotations only, see the lazy import in dismiss_modal
    from playwright.sync_api import Page

REPO = Path(__file__).resolve().parent.parent
LOAD_TIMEOUT = 15_000


def feedback_env_fixture(stem: str):
    """A module-scoped fixture pointing the app subprocess at dedicated feedback stores.

    ``stem`` names the files, so each module gets its own pair and the two can never write to
    each other's store regardless of collection order. Both are removed at teardown.

    Set at fixture SETUP, which is still before ``create_app_fixture`` launches the subprocess
    provided the app fixture depends on this one (it does -- see ``app_fixture``).
    """

    @pytest.fixture(scope="module")
    def feedback_env():
        directory = REPO / "data" / "feedback"
        feedback = directory / f"_{stem}.jsonl"
        contacts = directory / f"_{stem}_contacts.jsonl"
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("OSMOSE_FEEDBACK_FILE", str(feedback))
            mp.setenv("OSMOSE_CONTACTS_FILE", str(contacts))
            try:
                yield feedback, contacts
            finally:
                feedback.unlink(missing_ok=True)
                contacts.unlink(missing_ok=True)

    return feedback_env


def store_fixture():
    """A function-scoped fixture giving each test empty stores, and cleaning up after it.

    Reads the paths from the environment rather than capturing them at import, so it always
    names the files the app subprocess actually inherited. A constant captured at import could
    name a file the app never writes -- and then a "nothing was stored" assertion would pass
    because the file does not exist, not because the feature worked.
    """

    @pytest.fixture
    def e2e_store(feedback_env):
        feedback = Path(os.environ["OSMOSE_FEEDBACK_FILE"])
        contacts = Path(os.environ["OSMOSE_CONTACTS_FILE"])
        assert (feedback, contacts) == feedback_env, (
            "the environment the app inherited does not match the module's own fixture — "
            f"env={(feedback, contacts)!r} fixture={feedback_env!r}"
        )
        for p in (feedback, contacts):
            p.unlink(missing_ok=True)
        yield feedback
        for p in (feedback, contacts):
            p.unlink(missing_ok=True)

    return e2e_store


def dismiss_modal(page: Page, selector: str) -> None:
    """Close a Bootstrap modal, retrying through its fade-in.

    Bootstrap ignores ``hide()`` while a show transition is still running, and
    ``to_be_visible`` is satisfied the instant the element becomes visible — which is DURING
    that transition. A dismiss click landing in that window is swallowed, the modal stays open
    forever, and its backdrop then intercepts every later click. Measured at 1 failure in 24
    runs in ``test_e2e_feedback.py`` and 2 in 6 in ``test_e2e_feedback_modal.py`` (which loads
    the page twice per run) before this retry was added. A fixed sleep would paper over it;
    retrying until the modal is actually gone does not.
    """
    from playwright.sync_api import expect  # lazy: keeps this module importable without it

    modal = page.locator(selector)
    for _ in range(10):
        if not modal.is_visible():
            return
        modal.locator("[data-bs-dismiss='modal']").click()
        try:
            expect(modal).not_to_be_visible(timeout=1_500)
            return
        except AssertionError:
            continue
    expect(modal).not_to_be_visible(timeout=LOAD_TIMEOUT)  # final attempt, real failure text
