"""Playwright helpers for visual-regression snapshots.

Imports playwright (browser-only) -- guarded out of normal CI collection by
tests/conftest.py. The pure pixel diff lives in tests/_visual_compare.py.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests._e2e_support import dismiss_changelog_modal
from tests._visual_compare import compare_images

_CONNECT_TIMEOUT = 30_000  # cold container app-start can be slow
_TIMEOUT = 20_000
_BASELINE_DIR = Path(__file__).parent / "visual_baselines"
_OUTPUT_DIR = Path(__file__).parent / "visual_output"
_RUNBOOK = "tests/visual_baselines/README.md"

# Kill animation/focus/scrollbar nondeterminism inside the captured element.
#
# The notification-panel rule suppresses Shiny's transient toast (e.g. the
# "Loaded 'minimal' (N parameters)." toast fired by prepare_page). It renders
# position:fixed bottom-right -- overlapping the clip box of whichever page is
# captured -- and lives in the DOM for ~2s. With animations killed it shows as a
# fully-opaque, *stable* overlay, so _stable_screenshot can't tell it apart from
# settled content and bakes it into the capture, differing from the toast-free
# baseline (the movement-page flake). Hiding the panel makes every capture
# toast-free regardless of timing; persisted on the page, it also covers toasts
# raised by later navigations.
_DETERMINISM_CSS = (
    "*{transition:none!important;animation:none!important;caret-color:transparent!important}"
    "*:focus{outline:none!important}"
    "::-webkit-scrollbar{display:none!important}"
    "#shiny-notification-panel{display:none!important}"
)

_NAV_TO_CLIP = {
    "setup": "#split_setup",
    "fishing": "#split_fishing",
    "movement": "#split_movement",
    "advanced": "#split_advanced",
}

# Tuned (pre-merge) against the first container baselines; see the spec §4.
_DEFAULT_THRESHOLD = 4
_DEFAULT_MAX_RATIO = 0.002
_DEFAULT_MAX_PIXELS = 800
_DEFAULT_MEAN_THRESHOLD = 1.0


def _update_request() -> object | None:
    """None (compare), "all", or a set of page names (per-page update)."""
    v = os.environ.get("OSMOSE_UPDATE_SNAPSHOTS", "").strip()
    if not v:
        return None
    if v.lower() in ("1", "true", "yes", "all"):
        return "all"
    return {p.strip() for p in v.split(",") if p.strip()}


def _should_update(name: str) -> bool:
    req = _update_request()
    return req == "all" or (isinstance(req, set) and name in req)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v else default


def _neutralize(page: Page) -> None:
    page.mouse.move(0, 0)
    page.evaluate("() => document.activeElement && document.activeElement.blur()")
    expect(page.locator(".tooltip, .popover")).to_have_count(0)


def prepare_page(page: Page, app_url: str) -> None:
    """Load the app to a deterministic state with the `minimal` demo loaded.

    Pins theme=light before connect, dismisses the startup changelog modal, loads the
    `minimal` demo from the Domain page, and gates on the header config-stats (which read
    server-side state directly -- no client round-trip) before returning.
    """
    page.add_init_script("try { localStorage.setItem('osmose-theme', 'light'); } catch (e) {}")
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(app_url)
    page.wait_for_selector(".nav-pills", timeout=_CONNECT_TIMEOUT)
    dismiss_changelog_modal(page)

    # Load the deterministic `minimal` demo (controls live only on the Domain page).
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=_TIMEOUT)
    page.select_option("#load_example", "minimal")
    page.click("#btn_load_example")
    # config_header renders ".osm-config-stats" only once a config is loaded (app.py:547).
    expect(page.locator(".osm-config-stats")).to_contain_text("params", timeout=_TIMEOUT)

    page.add_style_tag(content=_DETERMINISM_CSS)
    page.evaluate("async () => { await document.fonts.ready; }")
    _neutralize(page)


def navigate_to(page: Page, nav_value: str) -> str:
    """Navigate to a config page by nav value; return its clip selector once content rendered."""
    clip = _NAV_TO_CLIP[nav_value]
    link = page.locator(f".nav-pills .nav-link[data-value='{nav_value}']")
    link.click()
    expect(link).to_have_class(re.compile(r"\bactive\b"), timeout=_TIMEOUT)
    page.wait_for_selector(clip, state="visible", timeout=_TIMEOUT)
    # Nav panels render lazily on shown.bs.tab (app.py:173) -- wait for inner content.
    page.wait_for_selector(f"{clip} .card", timeout=_TIMEOUT)
    _neutralize(page)
    return clip


def _stable_screenshot(page: Page, locator, *, mask, attempts: int = 6, settle_ms: int = 400):
    """Capture the clip until two consecutive shots match within the gate tolerances
    (i.e. the reactive page has SETTLED), or `attempts` is exhausted. Returns
    (shot, stable, last_metrics); the returned shot is the later, more-settled one.

    A one-shot double-capture is brittle: a still-rendering page (e.g. Setup's reactive
    species panels, slower in the CI container) differs a lot between two back-to-back
    captures. Retrying-until-stable converges once render settles, in any environment.
    The short fixed settle here is a bounded backoff in a condition loop — not a blind
    load wait.
    """
    shot = locator.screenshot(mask=mask or [], mask_color="#FF00FF")
    metrics: dict = {}
    for _ in range(attempts):
        page.wait_for_timeout(settle_ms)
        shot2 = locator.screenshot(mask=mask or [], mask_color="#FF00FF")
        stable, metrics, _ = compare_images(
            shot,
            shot2,
            threshold=_DEFAULT_THRESHOLD,
            max_ratio=_DEFAULT_MAX_RATIO,
            max_pixels=_DEFAULT_MAX_PIXELS,
            mean_threshold=_DEFAULT_MEAN_THRESHOLD,
        )
        shot = shot2
        if stable:
            return shot, True, metrics
    return shot, False, metrics


def assert_clip_snapshot(page: Page, clip_selector: str, name: str, *, mask=None) -> None:
    """Screenshot the (settled) clip element and compare to the committed baseline.

    Both modes wait for the clip to be pixel-stable (see _stable_screenshot) before using
    the shot. Update mode (OSMOSE_UPDATE_SNAPSHOTS) refuses to write if it never settles.
    Missing baseline in compare mode -> pytest.skip (operationally distinct from a
    regression). On failure writes <name>.actual.png + <name>.diff.png to tests/visual_output/.
    """
    locator = page.locator(clip_selector)
    baseline = _BASELINE_DIR / f"{name}.png"

    if _should_update(name):
        shot, stable, metrics = _stable_screenshot(page, locator, mask=mask)
        if not stable:
            raise AssertionError(
                f"Capture for {name!r} never settled within tolerances ({metrics}); "
                f"refusing to write baseline."
            )
        _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(shot)
        return

    if not baseline.exists():
        pytest.skip(f"No baseline for {name!r}; run the visual-update job (see {_RUNBOOK}).")

    shot, _, _ = _stable_screenshot(page, locator, mask=mask)
    passed, metrics, diff_png = compare_images(
        baseline.read_bytes(),
        shot,
        threshold=_DEFAULT_THRESHOLD,
        max_ratio=_env_float("OSMOSE_VISUAL_MAX_RATIO", _DEFAULT_MAX_RATIO),
        max_pixels=int(_env_float("OSMOSE_VISUAL_MAX_PIXELS", _DEFAULT_MAX_PIXELS)),
        mean_threshold=_env_float("OSMOSE_VISUAL_MEAN_THRESHOLD", _DEFAULT_MEAN_THRESHOLD),
    )
    if not passed:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (_OUTPUT_DIR / f"{name}.actual.png").write_bytes(shot)
        (_OUTPUT_DIR / f"{name}.diff.png").write_bytes(diff_png)
        raise AssertionError(
            f"Visual snapshot {name!r} differs: {metrics}. "
            f"See tests/visual_output/{name}.diff.png and the runbook {_RUNBOOK}."
        )
