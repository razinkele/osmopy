"""End-to-end tests for the two feedback-modal behaviours nothing else can observe.

``tests/test_e2e_feedback.py`` proves a submission reaches the store. It does not — and must
not be edited to — assert either of these, both of which fail SILENTLY:

* the modal dismissing itself on success. It is static Bootstrap markup opened client-side by
  ``data-bs-toggle``, so ``ui.remove_modal()`` would be a no-op; the success path sends a
  ``hide-modal`` custom message to the JS handler in ``app.py`` instead. If that message name,
  that handler, or the ``getOrCreateInstance`` call ever drifts, the notification still
  appears and the record still lands — only the modal stays open, and no unit test can see it.
* a honeypot hit being INDISTINGUISHABLE from real success. Any difference a bot can observe
  (a different message, a modal left open) tells it which field is the trap.

RATE-LIMIT BUDGET — read before adding a submitting test.
``ui.components.feedback_modal._LIMITER`` is process-global and allows 5 submissions per hour
per client key. Every session in one app subprocess keys on 127.0.0.1, so all tests in this
FILE share ONE bucket of 5 (a separate subprocess, and so a separate bucket, per test module).
This file spends **3 of 5**: one in the dismiss test, and two in the honeypot test — the
honeypot hit consumes a slot as well, because it is rate-limited before it is dropped, and the
positive control after it spends the third. A sixth submission would fail with a rate-limit
notice instead of "saved", for a reason nobody would guess from the failure. Adding one needs a
plan, not just a new test.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_feedback_modal.py -v -m e2e
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

# Dedicated stores for this module. The old import-time `os.environ.setdefault` dance existed
# only because the sibling module assigned the same variable at ITS import time and collection
# order decided the winner; a module-scoped fixture gives each module its own pair and removes
# the race entirely. It also sets OSMOSE_CONTACTS_FILE, which neither module used to set --
# the submissions below fill in an email address.
feedback_env = feedback_env_fixture("e2e_feedback_modal")
e2e_store = store_fixture()

_app = create_app_fixture("../app.py")


@pytest.fixture(scope="module")
def app(feedback_env, _app):
    """The app subprocess, launched only AFTER feedback_env has set the store variables.

    Explicit dependency, not autouse ordering: the subprocess inherits the environment at
    launch, so a fixture running afterwards would be useless and the failure -- records going
    to the wrong file, or a "nothing was stored" assertion passing vacuously -- would be
    silent.
    """
    return _app


def _open_feedback_modal(page: Page, app: ShinyAppProc) -> None:
    """Load the app, clear the startup changelog modal, open the feedback modal."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # The startup changelog modal overlays the header and must go before the Feedback link
    # is clickable.
    expect(page.locator("#changelogModal")).to_be_visible(timeout=_LOAD_TIMEOUT)
    _dismiss_modal(page, "#changelogModal")

    page.get_by_role("link", name="Feedback").click()
    expect(page.locator("#feedbackModal")).to_be_visible(timeout=_LOAD_TIMEOUT)


def test_success_dismisses_the_modal_and_clears_the_form(page: Page, app: ShinyAppProc, e2e_store):
    _open_feedback_modal(page, app)

    page.locator("#feedbackModal #feedback_message").fill("e2e dismiss check")
    page.locator("#feedback_submit").click()

    expect(page.locator(".shiny-notification")).to_contain_text("saved", timeout=_LOAD_TIMEOUT)
    # The assertion this file exists for: the custom-message dismiss actually fired.
    expect(page.locator("#feedbackModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)
    expect(page.locator("#feedbackModal #feedback_message")).to_have_value("")


def test_honeypot_hit_looks_like_success_but_stores_nothing(
    page: Page, app: ShinyAppProc, e2e_store
):
    _open_feedback_modal(page, app)

    page.locator("#feedbackModal #feedback_message").fill("e2e honeypot bait")
    page.locator("#feedbackModal #feedback_website").fill("http://spam.example")
    page.locator("#feedback_submit").click()

    # Same copy, same dismiss as a real success — a bot must learn nothing from either.
    expect(page.locator(".shiny-notification")).to_contain_text("saved", timeout=_LOAD_TIMEOUT)
    expect(page.locator("#feedbackModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)

    # A REAL submission into the same store, as a positive control. Without it the absence
    # assertion below would pass just as happily if the app were writing somewhere else
    # entirely — proving the file we read is the file the app writes is what makes an absence
    # mean anything.
    page.get_by_role("link", name="Feedback").click()
    expect(page.locator("#feedbackModal")).to_be_visible(timeout=_LOAD_TIMEOUT)
    page.locator("#feedbackModal #feedback_message").fill("e2e control record")
    page.locator("#feedback_submit").click()
    expect(page.locator("#feedbackModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)

    page.wait_for_timeout(500)
    stored = e2e_store.read_text() if e2e_store.is_file() else ""
    assert "e2e control record" in stored, (
        f"positive control missing — {e2e_store} is not the store this app writes to, "
        "so the honeypot assertion below would be vacuous"
    )
    assert "e2e honeypot bait" not in stored, "honeypot submission was stored anyway"
