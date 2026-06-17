import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")


def test_bootstrap_panel_renders_on_setup(page: Page, app: ShinyAppProc):
    """Smoke: the inline bootstrap panel renders on the Setup page. Never clicks Fetch,
    so no network — asserts the panel + inputs are present, not network results."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=30_000)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='setup']").click()
    page.wait_for_selector("#fb_fetch", timeout=20_000)
    expect(page.locator("#fb_name")).to_be_visible(timeout=10_000)
    expect(page.locator("#fb_fetch")).to_be_visible()
