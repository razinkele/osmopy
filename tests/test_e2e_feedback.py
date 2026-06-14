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


def test_feedback_submit_writes_record(page: Page, app: ShinyAppProc, clean_e2e_file):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # A startup changelog modal overlays the header — wait for it to finish fading in, then
    # dismiss it via its close button (focus-independent) before clicking the header Feedback
    # link. A one-shot is_visible() check races the fade-in: the modal can still be invisible
    # at check time and then animate in to intercept the Feedback click, so wait for it first.
    changelog = page.locator("#changelogModal")
    expect(changelog).to_be_visible(timeout=_LOAD_TIMEOUT)
    changelog.locator("[data-bs-dismiss='modal']").click()
    expect(changelog).not_to_be_visible(timeout=_LOAD_TIMEOUT)

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
