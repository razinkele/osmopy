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

from tests._visual_compare import compare_images
from tests._visual_support import (
    _DEFAULT_MAX_PIXELS,
    _DEFAULT_MAX_RATIO,
    _DEFAULT_MEAN_THRESHOLD,
    _DEFAULT_THRESHOLD,
    _stable_screenshot,
    assert_clip_snapshot,
    navigate_to,
    prepare_page,
)

pytestmark = pytest.mark.visual

app = create_app_fixture("../app.py")


def test_visual_nav_chrome(page: Page, app: ShinyAppProc):
    # Bootstrap/shinyswatch-skinned nav rail -- the most sensitive catch for a theme/
    # Bootstrap-version regression (the page bodies are skinned mostly by osmose.css).
    # Use the unique #main_nav id: a bare ".nav-pills" matches 4 navsets in this app
    # (#main_nav, #cal_groups, #about_tabs, #help_tabs) -> Playwright strict-mode error.
    prepare_page(page, app.url)
    assert_clip_snapshot(page, "#main_nav", "nav_chrome")


@pytest.mark.parametrize("past_the_top", [55, 79])
def test_a_capture_does_not_depend_on_where_the_page_was_left_scrolled(
    page: Page, app: ShinyAppProc, past_the_top
):
    """The page background is fixed to the viewport, so the same clip looks different at another
    scroll position. The rail is taller than the viewport, and Playwright leaves a page alone
    when the viewport is already inside the element, so a scroll position left over from an
    earlier layout stays. Measured 2026-11-21 in CI: a capture 55 px past the rail's top differs
    from its baseline in 7,595 pixels (the fixed tint ends at row 955, not 900), with every nav
    item pixel-identical. No baseline is involved here: the two captures are compared to each
    other, so this holds in any browser.
    """
    prepare_page(page, app.url)
    rail = page.locator("#main_nav")
    normal, stable, _ = _stable_screenshot(page, rail, mask=None)
    assert stable
    page.evaluate(
        """(past) => {
            const top = document.querySelector('#main_nav').getBoundingClientRect().top;
            window.scrollTo({top: top + window.scrollY + past, behavior: 'instant'});
        }""",
        past_the_top,
    )
    assert page.evaluate("() => window.scrollY") > 0  # the page can scroll: the case is real
    after, stable, _ = _stable_screenshot(page, rail, mask=None)
    assert stable
    same, metrics, _ = compare_images(
        normal,
        after,
        threshold=_DEFAULT_THRESHOLD,
        max_ratio=_DEFAULT_MAX_RATIO,
        max_pixels=_DEFAULT_MAX_PIXELS,
        mean_threshold=_DEFAULT_MEAN_THRESHOLD,
    )
    assert same, metrics


def test_visual_setup_page(page: Page, app: ShinyAppProc):
    prepare_page(page, app.url)
    clip = navigate_to(page, "setup")
    assert_clip_snapshot(page, clip, "setup")


def test_visual_fishing_page(page: Page, app: ShinyAppProc):
    prepare_page(page, app.url)
    clip = navigate_to(page, "fishing")
    assert_clip_snapshot(page, clip, "fishing")


def test_visual_movement_page(page: Page, app: ShinyAppProc):
    prepare_page(page, app.url)
    clip = navigate_to(page, "movement")
    assert_clip_snapshot(page, clip, "movement")


def test_visual_advanced_page(page: Page, app: ShinyAppProc):
    prepare_page(page, app.url)
    clip = navigate_to(page, "advanced")
    # Advanced is the largest clip. Its "All Parameters" table renders the LOADED config's
    # params (~28 rows under `minimal`, not the full registry), so it under-exercises
    # multi-value/conditional rendering -- a known v1 fixture gap (see the runbook).
    assert_clip_snapshot(page, clip, "advanced")
