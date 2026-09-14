"""End-to-end test for the feedback modal (submit → JSONL store).

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_feedback.py -v -m e2e
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
# Point the app subprocess at a DEDICATED file (never the real feedback.jsonl). Set at import
# time so the create_app_fixture subprocess inherits it at launch.
_E2E_FILE = _REPO / "data" / "feedback" / "_e2e_feedback.jsonl"
os.environ["OSMOSE_FEEDBACK_FILE"] = str(_E2E_FILE)

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000


@pytest.fixture
def clean_e2e_file():
    _E2E_FILE.unlink(missing_ok=True)
    yield _E2E_FILE
    _E2E_FILE.unlink(missing_ok=True)


def _dismiss_modal(page: Page, selector: str) -> None:
    """Close a Bootstrap modal, retrying through its fade-in.

    Bootstrap ignores ``hide()`` while a show transition is still running, and ``to_be_visible``
    is satisfied the instant the element becomes visible — which is DURING that transition. A
    dismiss click landing in that window is swallowed, the modal stays open, and its backdrop
    then intercepts every later click. Measured here at 1 failure in 24 runs before this retry
    (and 2 in 6 in tests/test_e2e_feedback_modal.py, which loads the page twice per run). A
    fixed sleep would paper over it; retrying until the modal is actually gone does not.
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


def test_feedback_submit_writes_record(page: Page, app: ShinyAppProc, clean_e2e_file):
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

    # The record landed in the dedicated store.
    def _written():
        return _E2E_FILE.is_file() and "e2e bug report alpha" in _E2E_FILE.read_text()

    page.wait_for_timeout(500)
    assert _written(), "feedback record was not written to the store"
