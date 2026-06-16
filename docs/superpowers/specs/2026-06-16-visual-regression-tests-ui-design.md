# Visual Regression Tests for the UI — Design Spec

**Date:** 2026-06-16
**Status:** Approved (brainstormed)
**Topic:** Playwright screenshot-based visual regression tests for the deterministic config pages.

## 1. Goal & Scope

Catch *unintended* visual changes — layout, CSS, spacing, theme, control rendering — on the
deterministic configuration pages of the Shiny app. This is especially valuable immediately after
the shiny 1.6.3 / shinyswatch 0.11 theme swap, where a Bootstrap-version change could silently
shift rendering.

**v1 covers** (the static, form-driven pages + one modal):

| Snapshot | Nav value | Selector to await |
| --- | --- | --- |
| Species / Setup | `setup` | `#split_setup` |
| Fishing | `fishing` | the page's root card |
| Movement | `movement` | the page's root card |
| Advanced | `advanced` | the page's root card |
| About modal | (header link) | `#aboutModal` |

**Explicitly out of scope (v1):** Domain/grid, Forcing, Run, Results, Spatial Results, Calibration,
Sensitivity, Scenarios, Map Viewer — these render Plotly charts and/or deck.gl (WebGL) canvases
whose pixels are nondeterministic across environments. (See §8.)

## 2. Architecture

- **New pytest marker `visual`** — separate from `e2e`. The default `addopts` changes from
  `-m 'not e2e'` to `-m 'not e2e and not visual'` so the normal suite skips both.
- **`tests/test_visual_regression.py`** — one test per page/modal — reuses the existing e2e harness:
  `shiny.pytest.create_app_fixture("../app.py")` + the pytest-playwright `page` fixture (sync API).
- **`tests/_visual_support.py`** — in-repo helper (matches the `_e2e_support.py` convention):
  - `prepare_page(page, app, nav_value)` — establishes a deterministic state and navigates (§3).
  - `assert_page_snapshot(page, name, *, mask=None, threshold=..., max_ratio=...)` — screenshots,
    compares to the committed baseline, writes artifacts on mismatch, honors update mode (§4).
  - `compare_images(baseline_png, actual_png, *, threshold, max_ratio) -> (passed, diff_ratio, diff_png)`
    — the **pure**, browser-free comparison core (numpy + Pillow), unit-tested in the normal suite (§7).
- **Baselines:** `tests/visual_baselines/<name>.png` — committed, **generated in CI** (§5).
- **Failure artifacts:** `tests/visual_output/<name>.actual.png` + `<name>.diff.png` — gitignored.

### Component responsibilities

- `compare_images` — knows nothing about Playwright or pages. Given two PNG byte strings, returns a
  pass/fail, the differing-pixel ratio, and a diff-highlight PNG. Independently testable.
- `assert_page_snapshot` — orchestrates screenshot → `compare_images` → artifact/baseline I/O →
  pytest assertion. Reads `OSMOSE_UPDATE_SNAPSHOTS` to switch into baseline-write mode.
- `prepare_page` — the determinism front-end; the only place that knows the app's load sequence.
- `test_visual_regression.py` — declares *which* pages/masks; no comparison logic.

## 3. Determinism Strategy

The make-or-break part. `prepare_page` performs, in order:

1. **Fixed viewport** 1280×900 (`page.set_viewport_size`).
2. `page.goto(app.url)`, wait for `.nav-pills`, then **dismiss the changelog modal** via
   `_e2e_support.dismiss_changelog_modal` (the once-per-version startup modal overlays the header).
3. **Load the bundled `minimal` demo** once per session (a session-scoped fixture), via the existing
   `load_example` select + `btn_load_example` flow on the Domain page. This gives every form page a
   known, small, deterministic config to render (without a loaded config the species panels/forms
   are empty or default-variable, making snapshots meaningless or flaky).
4. **Inject CSS to kill animation nondeterminism:**
   `* { transition: none !important; animation: none !important; caret-color: transparent !important; }`
   (eliminates fade/spinner/caret-blink frames).
5. Wait for `page.wait_for_load_state("networkidle")` and `document.fonts.ready` (late font loads
   shift text layout).
6. Navigate to the target `nav_value` and wait for its content selector.

`assert_page_snapshot` takes a **full-page** screenshot with `mask=[locator, ...]` — Playwright
natively paints masked regions a solid color — for any residual dynamic element (e.g. a live
validation count, a version string). The chosen pages are static forms, so masks should be minimal;
the `mask` parameter exists so a page with one dynamic widget can still be snapshotted.

## 4. Comparison & Tolerance

`compare_images(baseline_png, actual_png, *, threshold, max_ratio)`:

1. Decode both PNGs (Pillow) to equal-shape RGB numpy arrays.
2. **Dimension mismatch → immediate fail** — a size/layout change is exactly what we want to catch;
   the diff PNG in this case is the actual image (cannot overlay differently-sized arrays).
3. Per-pixel max-channel absolute difference; count pixels where that exceeds `threshold`.
4. **Fail if `differing_pixels / total_pixels > max_ratio`.**
5. Return `(passed, diff_ratio, diff_png)` where `diff_png` highlights the differing pixels in red.

**Defaults:** `threshold = 8` (0–255 scale; absorbs sub-pixel antialiasing), `max_ratio = 0.001`
(0.1% of pixels). Both overridable per call. Defaults are tuned conservatively and may be adjusted
once the first container-generated baselines exist.

`assert_page_snapshot` behavior:
- **Update mode** (`OSMOSE_UPDATE_SNAPSHOTS=1`): write the screenshot to the baseline path; pass.
- **Normal mode:** if no baseline exists → fail with a clear "run update mode" message. Else compare;
  on fail, write `<name>.actual.png` + `<name>.diff.png` to `tests/visual_output/` and assert with a
  message naming the diff ratio and artifact paths.

## 5. Baseline & CI Workflow — `.github/workflows/visual.yml`

Baselines are authoritative only when generated in the pinned Playwright container (consistent
fonts/AA). Since Docker is unavailable on the dev box, **CI is the baseline generator.**

- **`visual-gate` job** — runs in `mcr.microsoft.com/playwright/python:vX.Y.Z-jammy` (the tag matched
  to the `playwright` python pin in `[viztest]`). Installs `.[viztest]`, runs `pytest -m visual`,
  uploads `tests/visual_output/` as an artifact on failure.
  **Opt-in trigger:** `pull_request` filtered to UI-affecting paths (`ui/**`, `www/**`, `app.py`) +
  manual `workflow_dispatch`. It does not burden unrelated PRs.
- **`visual-update` job** — `workflow_dispatch` only. Same container, runs with
  `OSMOSE_UPDATE_SNAPSHOTS=1`, then commits the regenerated `tests/visual_baselines/*.png` back to the
  triggering branch (a clear `chore(visual): update baselines [skip ci]`-style bot commit). This is how
  baselines are created and refreshed without local Docker.
- **Local (advisory):** `pytest -m visual` against the committed baselines using the dev's native
  chromium gives fast feedback, but font/AA drift means it is **not** authoritative — the container
  gate is the source of truth. Documented as advisory.

The Playwright container tag and the `playwright` python version must be kept in lockstep (a comment
in both `pyproject.toml` and the workflow notes this).

## 6. Dependencies & Guards

- **New optional extra `[viztest]`** in `pyproject.toml`:
  `["playwright", "pytest-playwright>=0.5", "pillow>=10"]` — with the exact `playwright` version pinned
  in the plan (e.g. `>=1.49,<2`) to match the chosen container tag (§5 lockstep). Kept **out of `[dev]`** so the normal
  CI legs stay playwright-free (the existing design choice). The container ships browsers; local devs
  run `playwright install chromium` once.
- **Collection guard** in `tests/conftest.py`: skip `test_visual_regression.py` at collection when
  `playwright` *or* `PIL` is unavailable (mirrors the existing e2e `find_spec("playwright")` guard).
  Keeps the normal CI test legs' collection clean.
- `numpy` is already a core dependency; `pillow` is already present transitively but is declared in the
  extra for safety.
- **`.gitignore`:** add `tests/visual_output/`.
- **`pyproject.toml`:** register the `visual` marker; update `addopts` to `-m 'not e2e and not visual'`.

## 7. Testing the Test Infrastructure (TDD)

`compare_images` is pure and gets real unit tests in the **normal suite** (guarded only by
numpy/Pillow availability — no browser, runs in standard CI):

- Identical images → `passed=True`, `diff_ratio == 0`.
- Sub-`threshold` antialiasing-level noise → `passed=True`.
- Dimension mismatch → `passed=False`.
- An injected changed block exceeding `max_ratio` → `passed=False`, and the diff PNG highlights the
  changed region (assert non-trivial red-pixel count in the changed area).
- `max_ratio` boundary: a change just under the ratio passes; just over fails.

The browser-driven `-m visual` snapshots are the integration layer on top; their correctness is
validated by the CI container producing stable baselines and the gate re-comparing.

## 8. Non-Goals (v1)

- No chart/map/Run/Results/Calibration/Sensitivity/Spatial/Scenarios snapshots (Plotly + deck.gl
  WebGL pixel nondeterminism).
- Not a mandatory gate on *every* PR — opt-in via path filter and manual dispatch.
- Chromium only; a single fixed viewport (no cross-browser or responsive matrix).
- Local runs are advisory, not authoritative (font/AA drift vs the container).
- No automatic baseline approval UI — baseline refresh is a deliberate `workflow_dispatch`.

## 9. Open Implementation Notes (resolved at plan time)

- Exact root-card selectors for Fishing/Movement/Advanced to be read from their `*_ui()` functions
  when writing the tests (each wraps content in a `ui.card`/`ui.div` whose stable selector the plan
  pins).
- The `minimal` demo's rendered species count drives the Species-page snapshot; the baseline captures
  whatever `minimal` produces (deterministic by construction).
- Pin the concrete `playwright` version and the matching `mcr.microsoft.com/playwright/python` tag in
  the plan (lockstep requirement from §5).
