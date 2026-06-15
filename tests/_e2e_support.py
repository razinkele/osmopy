"""Shared helpers for Playwright e2e tests.

Not a test module (underscore prefix) — imported by the ``test_e2e_*`` files.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

_CHANGELOG_TIMEOUT = 20_000


def dismiss_changelog_modal(page: Page, timeout: int = _CHANGELOG_TIMEOUT) -> None:
    """Dismiss the once-per-version "What's new" startup modal (docs-in-app feature).

    A fresh Playwright browser context has empty ``localStorage``, so the modal auto-shows on
    ``shiny:connected`` and overlays the header, intercepting the nav/header clicks that most
    e2e tests perform right after load. Call this right after ``wait_for_selector('.nav-pills')``
    and before the first click.

    Dismiss via the close button (``[data-bs-dismiss='modal']``) — NOT Escape, which needs the
    modal to hold keyboard focus, which it intermittently lacks after navigation (the
    test_e2e_docs.py / test_e2e_feedback.py precedent). The modal is reliably present on a fresh
    load, so this requires it visible (matching those siblings); if a future version-tracking
    change stops it showing, update this one helper.
    """
    changelog = page.locator("#changelogModal")
    expect(changelog).to_be_visible(timeout=timeout)
    changelog.locator("[data-bs-dismiss='modal']").click()
    expect(changelog).not_to_be_visible(timeout=timeout)
