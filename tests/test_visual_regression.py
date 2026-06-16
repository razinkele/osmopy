"""Visual-regression snapshots of the deterministic config page bodies + nav chrome.

Run explicitly (browser required):
    OSMOSE_UPDATE_SNAPSHOTS=1 .venv/bin/python -m pytest tests/test_visual_regression.py -m visual  # write baselines
    .venv/bin/python -m pytest tests/test_visual_regression.py -m visual                             # compare

Authoritative baselines come from the digest-pinned Playwright container via the
visual-update CI job; local runs (native browser) are advisory (font/AA drift). Guarded
out of normal collection by tests/conftest.py when playwright/PIL are absent.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._visual_support import assert_clip_snapshot, navigate_to, prepare_page

pytestmark = pytest.mark.visual

app = create_app_fixture("../app.py")


def test_visual_nav_chrome(page: Page, app: ShinyAppProc):
    # Bootstrap/shinyswatch-skinned nav rail -- the most sensitive catch for a theme/
    # Bootstrap-version regression (the page bodies are skinned mostly by osmose.css).
    # Use the unique #main_nav id: a bare ".nav-pills" matches 4 navsets in this app
    # (#main_nav, #cal_groups, #about_tabs, #help_tabs) -> Playwright strict-mode error.
    prepare_page(page, app.url)
    assert_clip_snapshot(page, "#main_nav", "nav_chrome")


def test_visual_setup_page(page: Page, app: ShinyAppProc):
    prepare_page(page, app.url)
    clip = navigate_to(page, "setup")
    assert_clip_snapshot(page, clip, "setup")
