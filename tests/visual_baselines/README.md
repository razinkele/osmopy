# Visual-regression baselines

Committed PNG baselines for `tests/test_visual_regression.py` (`-m visual`). The
authoritative baselines are generated in the pinned Playwright **container** (consistent
fonts/antialiasing); the `visual-gate` CI job is the sole pass/fail authority.

## Updating baselines (after an intended UI change)
1. Push your branch.
2. GitHub → Actions → **Visual** → **Run workflow** → pick your branch. Set `pages` to
   `all` (default) or a comma list (e.g. `fishing`) to refresh only specific pages — this
   avoids re-blessing an accidental regression on a page you did not intend to change.
3. Download the `visual-baselines` artifact, unzip it.
4. **Open and visually inspect each PNG** (do NOT blind-commit — `git diff` shows nothing
   useful for binaries). An unexpected page in the changed set is a stop sign.
5. Copy the intended PNGs into `tests/visual_baselines/` and commit them.

## Local runs are advisory only
`pytest -m visual` locally uses your native browser; font/AA differs from the container,
so expect noise. Trust the CI `visual-gate`, not local pass/fail.

## Adding a page
Add it to `_NAV_TO_CLIP` in `tests/_visual_support.py`, add a `test_visual_<page>`
mirroring the others, then regenerate baselines via the steps above.

## Bumping Playwright
Three things move together: the `playwright==X` pin in `pyproject.toml [viztest]`, the
`expected` version in the `visual.yml` version-assert step, and the **image digest** in
both `visual.yml` jobs (resolve via `docker manifest inspect mcr.microsoft.com/playwright/python:vX-noble`).
Then regenerate all baselines (new browser ⇒ new AA).

## Determinism safeguards
`_DETERMINISM_CSS` in `tests/_visual_support.py` is injected into the page before
capture to remove non-deterministic paint: transitions/animations, focus outlines,
the caret, scrollbars, and **Shiny notification toasts** (`#shiny-notification-panel`).
The toast suppression matters because `prepare_page` raises a "Loaded 'minimal' (N
parameters)." toast that lives ~2s; with animations killed it shows as a fully-opaque,
*stable* overlay that `_stable_screenshot` would otherwise bake into the capture
(this caused an intermittent movement-page failure). If a new transient overlay starts
flaking a baseline, suppress it here rather than re-blessing it into the PNG.

Every capture also starts from the top of the page (`_capture`). The page background is
fixed to the viewport, so a clip looks different at another scroll position, and
Playwright does not re-scroll a clip taller than the viewport (the nav rail) once the
viewport lies inside it. A scroll position left over from an earlier layout therefore
stayed: on 2026-11-21 one CI run in seven captured the rail 55 px past its top, and
`nav_chrome` differed in 7,595 pixels of background while every nav item was identical.
A diff that is one horizontal band of background is this, not a theme change;
`test_a_capture_does_not_depend_on_where_the_page_was_left_scrolled` guards it.

## Recovery
Baselines are plain committed PNGs: `git checkout <good-sha> -- tests/visual_baselines/`
reverts a bad update, then re-run `visual-update` for the page you actually meant to change.
