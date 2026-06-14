"""End-to-end test for the in-app docs (startup changelog modal + About tabs).

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_docs.py -v -m e2e
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000


def test_startup_modal_shows_once_then_suppressed(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Fresh context (empty localStorage) → the what's-new modal auto-shows.
    expect(page.locator("#changelogModal")).to_be_visible(timeout=_LOAD_TIMEOUT)

    # Dismiss it (Esc closes a Bootstrap modal) and confirm the guard recorded the version.
    page.keyboard.press("Escape")
    expect(page.locator("#changelogModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)
    seen = page.evaluate("() => localStorage.getItem('osmose_seen_changelog_version')")
    assert seen and seen != ""

    # Reload (same context preserves localStorage) → modal does NOT reappear.
    page.reload()
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    expect(page.locator("#changelogModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)


def test_about_modal_changelog_tab(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Dismiss the startup modal first (it covers the header).
    if page.locator("#changelogModal").is_visible():
        page.keyboard.press("Escape")
        expect(page.locator("#changelogModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)

    # Open About from the header, switch to the Changelog tab, assert rendered content.
    page.get_by_role("link", name="About").click()
    expect(page.locator("#aboutModal")).to_be_visible(timeout=_LOAD_TIMEOUT)
    page.locator("#aboutModal").get_by_role("tab", name="Changelog").click()
    expect(page.locator("#aboutModal")).to_contain_text("0.13.0", timeout=_LOAD_TIMEOUT)
