"""End-to-end test for the feedback modal (submit → JSONL store).

WHEN TO RUN THIS. It is `e2e`, and `addopts` is `-m 'not e2e and not visual'`, so it does not
run in a bare pytest or in CI — the marker here really does gate execution (unlike `slow`,
see the marker list in pyproject.toml). The handler→disk half of this path is covered by
default in ``tests/test_feedback_integration.py``; what only THIS test can see is the
browser→handler half. Run it before merging any change to:

* the modal markup or the input ids a browser actually fills
  (``ui/components/feedback_modal.py``);
* the ``feedback_server`` call in ``app.py`` — nothing else asserts the handler is wired into
  the running app at all.

A break in either leaves every default-run test green, because none of them goes through a
browser. It does NOT see the modal dismissing itself (the ``hide-modal`` JS in ``app.py``) or
honeypot indistinguishability in the browser — ``tests/test_e2e_feedback_modal.py`` owns both,
and says in its own docstring that this file must not be edited to assert them.

    .venv/bin/python -m pytest tests/test_e2e_feedback.py -v -m e2e

RATE-LIMIT BUDGET — read before adding a submitting test.
``ui.components.feedback_modal._LIMITER`` is process-global and allows 5 submissions per hour
per client key. Every session in one app subprocess keys on 127.0.0.1, so all tests in this
FILE share ONE bucket of 5 (a separate subprocess, and so a separate bucket, per test module).
This file spends **1 of 5**; ``tests/test_e2e_feedback_modal.py`` spends 3 of its own 5. Note
that a honeypot submission consumes a slot too — it is rate-limited before it is dropped.
A sixth submission here would fail with a rate-limit notice instead of "saved", for a reason
nobody would guess from the failure. Adding one needs a plan, not just a new test.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_feedback_env import (
    LOAD_TIMEOUT as _LOAD_TIMEOUT,
)
from tests._e2e_feedback_env import (
    dismiss_modal as _dismiss_modal,
)
from tests._e2e_feedback_env import (
    feedback_env_fixture,
    store_fixture,
)

pytestmark = pytest.mark.e2e

# Dedicated stores for this module, set at fixture SETUP and undone at teardown -- never at
# import, which used to leak the variable into every other test in the session. Sets the
# CONTACTS store too, without which an e2e submission carrying an email wrote a real address
# into the developer's own gitignored contacts.jsonl.
feedback_env = feedback_env_fixture("e2e_feedback")
e2e_store = store_fixture()

_app = create_app_fixture("../app.py")


@pytest.fixture(scope="module")
def app(feedback_env, _app):
    """The app subprocess, launched only AFTER feedback_env has set the store variables.

    The dependency is explicit rather than relying on autouse ordering: the subprocess
    inherits the environment at launch, so a fixture that ran afterwards would be useless and
    the failure -- records going to the wrong file -- would be silent.
    """
    return _app


def test_feedback_submit_writes_record(page: Page, app: ShinyAppProc, e2e_store):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # A startup changelog modal overlays the header and must go before the Feedback link is
    # clickable.
    changelog = page.locator("#changelogModal")
    expect(changelog).to_be_visible(timeout=_LOAD_TIMEOUT)
    _dismiss_modal(page, "#changelogModal")

    page.get_by_role("link", name="Feedback").click()
    expect(page.locator("#feedbackModal")).to_be_visible(timeout=_LOAD_TIMEOUT)

    page.locator("#feedbackModal #feedback_message").fill("e2e bug report alpha")
    page.locator("#feedback_submit").click()

    # Success notification appears.
    expect(page.locator(".shiny-notification")).to_contain_text("saved", timeout=_LOAD_TIMEOUT)

    # The record landed in the dedicated store. `e2e_store` is the path the app subprocess
    # actually inherited, re-read from the environment rather than captured at import.
    def _written():
        return e2e_store.is_file() and "e2e bug report alpha" in e2e_store.read_text()

    page.wait_for_timeout(500)
    assert _written(), "feedback record was not written to the store"
