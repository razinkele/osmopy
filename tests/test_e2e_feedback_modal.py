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

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_feedback_modal.py -v -m e2e
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

_REPO = Path(__file__).resolve().parent.parent
# A DEDICATED store file (never the real feedback.jsonl), set at import time so the
# create_app_fixture subprocess inherits it at launch.
#
# setdefault, NOT assignment: tests/test_e2e_feedback.py sets this same variable at ITS import
# time, and pytest imports every collected module before running anything. A plain assignment
# here would win and silently redirect that test's app to a store it never reads, failing it.
# Verified the hard way. With setdefault the two modules share one store file when both are
# collected, which is safe because each test unlinks it first -- safe SEQUENTIALLY; do not run
# these two files concurrently across xdist workers.
os.environ.setdefault(
    "OSMOSE_FEEDBACK_FILE",
    str(_REPO / "data" / "feedback" / "_e2e_feedback_modal.jsonl"),
)

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000


@pytest.fixture
def e2e_store():
    """The store the app subprocess actually writes to, resolved AFTER every module import.

    Deliberately NOT a module-level constant. setdefault above runs at import time, but the
    sibling module's plain assignment may run after it (collection order decides), and the app
    subprocess launches later still with whatever env survived. A constant captured at import
    could therefore name a file the app never writes -- and then the "nothing was stored"
    assertion below would pass because the file does not exist, not because the honeypot
    worked. Reading the env here, post-import, is the same value the app inherited.
    """
    p = Path(os.environ["OSMOSE_FEEDBACK_FILE"])
    p.unlink(missing_ok=True)
    yield p
    p.unlink(missing_ok=True)


def _dismiss_modal(page: Page, selector: str) -> None:
    """Close a Bootstrap modal, retrying through its fade-in.

    Bootstrap ignores ``hide()`` while a show transition is still running, and
    ``to_be_visible`` is satisfied the instant the element becomes visible — which is DURING
    that transition. A dismiss click landing in that window is swallowed, the modal stays
    open forever, and its backdrop then intercepts every later click. Measured at 2 failures
    in 6 runs before this retry was added. A fixed sleep would paper over it; retrying until
    the modal is actually gone does not.
    """
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
    expect(modal).not_to_be_visible(timeout=_LOAD_TIMEOUT)  # final attempt, real failure text


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
