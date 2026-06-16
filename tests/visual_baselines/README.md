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

## Recovery
Baselines are plain committed PNGs: `git checkout <good-sha> -- tests/visual_baselines/`
reverts a bad update, then re-run `visual-update` for the page you actually meant to change.
