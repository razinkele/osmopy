# Full Baltic E2E Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) — this is ONE tightly-coupled Playwright test built by running it against the live app and adjusting; it is not cleanly splittable across fresh subagents. Steps use checkbox (`- [ ]`) syntax.

**Goal:** One committed e2e that loads Baltic, runs the Python engine (1 yr) with live movement, and asserts the key Results graphical outputs (biomass chart, diet heatmap, spatial map) render with real data.

**Architecture:** A single `tests/test_e2e_baltic.py` (matches the `test_e2e_*.py` conftest collect-ignore glob; `@pytest.mark.e2e`), using `create_app_fixture` + `dismiss_changelog_modal`, mirroring `tests/test_e2e_live_movement.py`. Built incrementally and RUN locally (chromium installed) after each phase.

**Tech Stack:** pytest, pytest-playwright (`[viztest]`), shiny.pytest `create_app_fixture`.

**Spec:** `docs/superpowers/specs/2026-06-17-e2e-baltic-full-design.md`

**Confirmed selectors (read from the UI):**
- Nav links: `.nav-pills .nav-link[data-value='grid'|'run'|'results'|'spatial_results']`
- Load: `#load_example`, `#btn_load_example`; Engine: `#engineBtnPython`; Overrides: `#py_param_overrides`
- Live: `#live_movement_view`, `#btn_run`, `#run_status`, `#live_movement_status`, `#live_movement_mode`, `#live_map`
- Results: biomass `#results_chart` (Time Series card, always visible); diet `#diet_chart` under the **"Diet Composition"** tab (`navset_card_tab`); diet empty-state title text `"Diet Composition (no prey data)"` (`results.py:198`)
- Spatial: pill `data-value='spatial_results'` (server-shown when spatial output exists); **"Flat View"** tab → `#spatial_flat_chart`; species selector `#spatial_species`
- `output_widget(...)` renders a plotly graph div with class `.js-plotly-plot`; a non-empty plot has `.js-plotly-plot .trace` (or, for a heatmap, `.heatmaplayer`) nodes.

---

## Task 1: Skeleton — load Baltic, Python run + live movement to completion

**Files:**
- Create: `tests/test_e2e_baltic.py`

- [ ] **Step 1: Write the test skeleton**

```python
"""Full Baltic end-to-end: Python run (1 yr) + live movement + Results outputs.

Run explicitly:  .venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider
"""

from __future__ import annotations

import pathlib

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = pathlib.Path(__file__).resolve().parent.parent
_LOAD_TIMEOUT = 20_000
_RUN_TIMEOUT = 120_000  # 1-yr Baltic Python run + spatial I/O


def test_baltic_full_run_and_outputs(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)

    # 1. Load Baltic (Domain/Grid page).
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=_LOAD_TIMEOUT)
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)

    # 2. Run page: Python engine (Baltic is Python-only), short run + spatial output on.
    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#engineBtnPython").click()
    overrides = page.locator("#py_param_overrides")
    expect(overrides).to_be_visible(timeout=_LOAD_TIMEOUT)
    overrides.fill("simulation.time.nyear=1\noutput.spatial.enabled=true")

    # 3. Live movement + run.
    page.locator("#live_movement_view").click()
    page.locator("#btn_run").click()
    expect(page.locator("#run_status")).not_to_contain_text("Validation failed", timeout=_LOAD_TIMEOUT)
    expect(page.locator("#live_movement_status")).to_contain_text("running", timeout=_RUN_TIMEOUT)
    expect(page.locator("#live_movement_status")).to_contain_text("done", timeout=_RUN_TIMEOUT)
    page.locator("#live_movement_mode").get_by_text("Dots").click()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider`
Expected: PASS. If `#live_movement_status` never reaches "done" within 120s, the Baltic 1-yr run is slower than expected — raise `_RUN_TIMEOUT` and note actual duration; do NOT remove the assertion. If "Validation failed" appears, capture `#run_status` text (a real config/override problem) and stop.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_baltic.py
git commit -m "test(e2e): Baltic full-run skeleton — load + Python run + live movement"
```

---

## Task 2: Assert Results biomass + diet heatmap (the regression guard)

**Files:**
- Modify: `tests/test_e2e_baltic.py`

- [ ] **Step 1: Append the Results assertions** (inside the same test, after the live-movement block)

```python
    # 4. Results: biomass time-series + diet heatmap (real data, not empty-state).
    page.locator(".nav-pills .nav-link[data-value='results']").click()

    # Biomass chart (Time Series card, always visible). A non-empty plotly plot has traces.
    expect(page.locator("#results_chart .js-plotly-plot")).to_be_visible(timeout=_RUN_TIMEOUT)
    assert page.locator("#results_chart .js-plotly-plot .trace").count() > 0, "biomass chart has no traces"

    # Diet Composition heatmap — click its tab, assert it rendered and is NOT the
    # "(no prey data)" empty-state (the bug a real Baltic run uniquely surfaced).
    page.get_by_role("tab", name="Diet Composition").click()
    diet = page.locator("#diet_chart")
    expect(diet.locator(".js-plotly-plot")).to_be_visible(timeout=_LOAD_TIMEOUT)
    assert diet.locator(".js-plotly-plot .heatmaplayer").count() > 0, "diet heatmap has no cells"
    assert "no prey data" not in diet.inner_text().lower(), "diet heatmap is the empty-state"
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider`
Expected: PASS. If `.trace`/`.heatmaplayer` selectors don't match the actual plotly DOM, inspect via a temporary `page.locator("#results_chart").inner_html()` print and adjust the content selector (the INTENT is "plot has rendered data") — keep the "no prey data" guard. If the diet tab name differs, use the exact `nav_panel` title "Diet Composition".

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_baltic.py
git commit -m "test(e2e): assert Baltic biomass chart + diet heatmap render (regression guard)"
```

---

## Task 3: Assert spatial map + screenshot + finalize

**Files:**
- Modify: `tests/test_e2e_baltic.py`

- [ ] **Step 1: Append the spatial assertion + screenshot**

```python
    # 5. Spatial output (enabled via the override). The Spatial Results pill is shown by
    # the server once spatial output is available; select a species and use the Flat View
    # plotly heatmap (more deterministic than the deck.gl canvas).
    spatial_pill = page.locator(".nav-pills .nav-link[data-value='spatial_results']")
    expect(spatial_pill).to_be_visible(timeout=_RUN_TIMEOUT)
    spatial_pill.click()
    # Pick a concrete species (the default may be "All species"); options come from the file.
    page.wait_for_selector("#spatial_species", timeout=_LOAD_TIMEOUT)
    page.get_by_role("tab", name="Flat View").click()
    expect(page.locator("#spatial_flat_chart .js-plotly-plot")).to_be_visible(timeout=_RUN_TIMEOUT)
    assert page.locator("#spatial_flat_chart .js-plotly-plot .heatmaplayer").count() > 0, "spatial flat heatmap empty"

    _REPO.joinpath("screenshots").mkdir(exist_ok=True)
    page.screenshot(path=str(_REPO / "screenshots" / "baltic_full_e2e.png"), full_page=True)
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider`
Expected: PASS.

**FALLBACK (spec risk):** if the Spatial pill never appears or the Flat heatmap stays empty (Baltic-with-override doesn't yield renderable spatial output), REMOVE the spatial block and instead assert an always-emitted graphical output: on the Results page, the biomass chart already passed; additionally assert the diet heatmap (Task 2) — and add a `# NOTE: spatial output not renderable for Baltic 1-yr run; covered biomass+diet` comment. Document the substitution in the commit message and tell the controller. Do NOT leave a flaky spatial assertion.

- [ ] **Step 3: Inspect the screenshot**

Read `screenshots/baltic_full_e2e.png` and confirm it shows real rendered outputs (not error states).

- [ ] **Step 4: Finalize checks**

Run:
- `.venv/bin/ruff check tests/test_e2e_baltic.py` → clean
- `.venv/bin/python -c "import fnmatch; assert fnmatch.fnmatch('test_e2e_baltic.py','test_e2e_*.py')"` → no error (conftest will collect-ignore it in CI where playwright is absent)
- `.venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider` once more → PASS (stability)

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_baltic.py screenshots/baltic_full_e2e.png
git commit -m "test(e2e): assert Baltic spatial map renders + screenshot artifact"
```

---

## Notes

- **Screenshots:** `screenshots/*.png` are committed by precedent (other e2e tests do, e.g. `live_movement_e2e.png`); confirm not gitignored (`git check-ignore screenshots/baltic_full_e2e.png` → no match).
- **Runtime:** this is one e2e (~1–2 min). It runs in CI's dedicated e2e leg (`[viztest]`), not the unit `test` job. It will NOT slow the unit suite.
- **No new deps.** Reuses the existing e2e harness.
- **Why inline, not subagent-driven:** the three phases are the same growing test, verified by re-running the live app; a fresh subagent per phase would re-pay the ~2-min run and lacks the iterative DOM-inspection context. Execute inline.
