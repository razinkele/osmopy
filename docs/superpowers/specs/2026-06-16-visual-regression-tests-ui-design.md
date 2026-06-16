# Visual Regression Tests for the UI — Design Spec

**Date:** 2026-06-16
**Status:** Approved (brainstormed; converged through 3 in-loop review rounds across 4 angles)
**Topic:** Playwright screenshot-based visual regression tests for the deterministic config pages.

## 1. Goal & Scope

Catch *unintended* visual changes — layout, CSS, spacing, theme, control rendering — on the
deterministic configuration **page bodies** of the Shiny app. Especially valuable right after the
shiny 1.6.3 / shinyswatch 0.11 theme swap, where a Bootstrap-version change could silently shift
rendering.

**v1 covers** the four static, form-driven page bodies, each clipped to its `osm-split-layout`
root container (verified selectors):

| Snapshot | Nav value | Clip selector |
| --- | --- | --- |
| Species / Setup | `setup` | `#split_setup` |
| Fishing | `fishing` | `#split_fishing` |
| Movement | `movement` | `#split_movement` |
| Advanced | `advanced` | `#split_advanced` |

**Why element-clipped, not full-page:** clipping to `#split_<page>` structurally excludes the app
**header** (which carries live, localStorage-restored, async chrome — `#config_header` param-count +
dirty flag, the engine-mode toggle, the version badge, the theme toggle) *and* the Setup
**validation panel** (`#config_validation`, a sibling rendered *above* `#split_setup` in
`setup.py:29`, which shows live error/warning counts). This removes the dominant nondeterminism
sources by construction rather than by masking. (See §3.)

**Explicitly dropped from v1:** the **About modal** — it renders live README + CHANGELOG markdown
(dated, versioned release notes), so its content rots on every release; it is a text-doc target, not
a CSS/layout target, and would false-positive on each version bump. Also out: Domain/grid, Forcing,
Run, Results, Spatial Results, Calibration, Sensitivity, Scenarios, Map Viewer (Plotly + deck.gl
WebGL pixel nondeterminism). See §8.

## 2. Architecture

- **New pytest marker `visual`** — separate from `e2e`. `addopts` changes from
  `-m 'not e2e' --dist loadfile` to **`-m 'not e2e and not visual' --dist loadfile`** (the
  `--dist loadfile` flag is preserved). The `visual` marker is appended to the existing `markers` list.
- **`tests/test_visual_regression.py`** — one test per page — reuses the e2e harness
  (`shiny.pytest.create_app_fixture("../app.py")` + the pytest-playwright `page` fixture, sync API).
- **`tests/_visual_support.py`** — in-repo helper (matches the `_e2e_support.py` convention):
  - `prepare_page(page, app, nav_value)` — establishes a deterministic state and navigates (§3).
    Called **per test** (no session-scoped config persistence is possible — see below).
  - `assert_clip_snapshot(page, clip_selector, name, *, mask=None, threshold=4, max_ratio=<tuned>)` —
    element-clipped screenshot → compare → artifact/baseline I/O → assert; honors update mode (§4).
  - `compare_images(baseline_png, actual_png, *, threshold, max_ratio) -> (passed, diff_ratio, diff_png)`
    — the **pure**, browser-free comparison core (numpy + Pillow), unit-tested in the normal suite (§7).
- **Baselines:** `tests/visual_baselines/<name>.png` — committed, **generated in CI** (§5).
- **Failure artifacts:** `tests/visual_output/<name>.actual.png` + `<name>.diff.png` — gitignored.

**Per-test loading (corrects a round-1 BLOCKER).** Each Playwright `page`/`context` is
function-scoped and opens a *new* Shiny websocket session; `state.config` (`ui/state.py:36`) is
per-session, so a "load once per session" fixture would persist nothing. `prepare_page` therefore
loads the demo config on **every** invocation. The app *process* is shared (module-scoped
`create_app_fixture`); the server-side reactive state is not.

## 3. Determinism Strategy (the make-or-break part)

`prepare_page(page, app, nav_value)` performs, in order:

1. **Pin theme to light before connect:** add an init script setting
   `localStorage['osmose-theme']='light'` (the app restores theme from this key async on
   `shiny:connected`, `app.py:102,109`; an unpinned theme can flip ~100% of pixels).
2. **Fixed viewport** 1280×900 (`page.set_viewport_size`).
3. `page.goto(app.url)`; `wait_for_selector(".nav-pills")`; **dismiss the changelog modal** via
   `_e2e_support.dismiss_changelog_modal`.
4. **Load the `minimal` demo** (deterministic, small — `list_demos()` includes `"minimal"`): the
   `load_example` select + `btn_load_example` button live only on the **Domain** page
   (`grid.py:105,112`). So: navigate to Domain, select `minimal`, click Load, then **wait for load
   completion by gating on a DEPENDENT RENDERED element.** `handle_load_example` is a *synchronous*
   `@reactive.effect` (`grid.py:808`) that sets `state.config` / `load_trigger`; do NOT gate on the
   `state.busy` loading overlay — it can clear before the dependent `output_ui`s flush (a race).
   Instead wait until the header `config_header` reflects `minimal`'s species/param counts — it reads
   `state.config` directly (server-side, no client round-trip), so it is the cleanest settled signal;
   prefer it over `species_panels`, which depends on the `input.n_species` client update and can render
   stale-then-fresh. `minimal`'s config_name has no timestamp
   and species order is `sp{i}`-indexed — deterministic (verified).
5. **Inject determinism CSS:**
   `*{transition:none!important;animation:none!important;caret-color:transparent!important}`
   **plus** `*:focus{outline:none!important}` (kills focus rings) and a scrollbar-hiding rule
   (`::-webkit-scrollbar{display:none}`) so an overflow scrollbar can't shift right-edge layout.
6. **Neutralize hover/focus state:** `page.mouse.move(0,0)`,
   `page.evaluate("document.activeElement && document.activeElement.blur()")`, and assert no
   `.tooltip, .popover` is visible (a residual nav-click hover/popover must not paint over the form).
7. **Await fonts as a promise:** `page.evaluate("async () => { await document.fonts.ready; }")`
   (late font loads shift text). **Do NOT use `networkidle`** — Shiny's persistent websocket keeps the
   network non-idle, so it fires late/never/inconsistently; no existing e2e test uses it. Instead wait
   on a concrete element being visible (see step 8) — never a fixed sleep.
8. Navigate to `nav_value`; `wait_for_selector("#split_<page>")`, then wait for a **known inner control**
   of that panel to have rendered (e.g. a specific field input, or a non-zero `.card` count inside
   `#split_<page>`) — nav panels render lazily on `shown.bs.tab` (`app.py:173`), so the wrapper
   existing does not guarantee its server-rendered content is present.

`assert_clip_snapshot` then screenshots **the clip element** (`locator(clip_selector).screenshot()`),
not the full page. For any residual in-body dynamic element it passes `mask=[locator,...]` with an
explicit pinned **`mask_color="#FF00FF"`** (so the masked color is deterministic and baked into the
baseline regardless of future Playwright default changes).

## 4. Comparison & Tolerance

`compare_images(baseline_png, actual_png, *, threshold, max_ratio)`:

1. Decode both PNGs (Pillow) to RGB numpy arrays.
2. **Dimension mismatch → immediate fail.** With element-clipping + the deterministic `minimal`
   config, the clip size is stable, so a size change is a real regression (not benign reflow). The
   diff PNG in this case is the actual image (cannot overlay differently-sized arrays).
3. Per-pixel max-channel absolute difference; count pixels exceeding `threshold`.
4. **Fail if `differing_pixels / total_pixels > max_ratio`.**
5. Return `(passed, diff_ratio, diff_png)` where `diff_png` highlights differing pixels in **red**
   (`#FF0000`, distinct from the `#FF00FF` mask color).

**Defaults:** `threshold = 4`, `max_ratio` tuned empirically against the first container baselines.
**A required plan task** is to capture the first baselines in the container, then tune `threshold` /
`max_ratio` so the suite is green on a no-op change yet flags a deliberate injected shift. Rationale:
a pure 1–2 px text translation produces mostly sub-threshold antialiased edge pixels; too-high
`threshold` + too-high `max_ratio` would miss it, while too-low values flake on AA. **The plan MUST
decide during tuning** between a ratio-only gate and an additional **absolute differing-pixel floor**
(the latter guards glyph-level regressions on small clips that a percentage ratio would hide). Tune
against the **`#split_advanced` clip specifically** — it is the largest / most content-dense (a ~100-row
registry param table, `advanced.py:236`) and is the worst case for both AA noise and ratio calibration.
(Tuning is a hard task, not optional.)

`assert_clip_snapshot` behavior:
- **Update mode** (`OSMOSE_UPDATE_SNAPSHOTS` truthy — `"1"`/`"true"`): write the screenshot to the
  baseline path; pass.
- **Normal mode:** missing baseline → fail with a "run update mode" message. Else compare; on fail,
  write `<name>.actual.png` + `<name>.diff.png` to `tests/visual_output/` and assert with a message
  naming the diff ratio and artifact paths.

**Local vs container.** Local runs use the dev's native chromium; font/AA rasterization differs from
the `mcr.microsoft.com/playwright/python` image, so per-glyph sub-pixel diffs will be widespread.
Local `-m visual` is therefore **gross-layout/dimension advisory only** — it is honest about this and
either uses an env-gated looser `max_ratio` or is documented as "expect text-edge noise; trust the
container gate for pixel pass/fail." The container gate (§5) is the sole authority.

## 5. Baseline & CI Workflow — `.github/workflows/visual.yml`

Baselines are authoritative only when generated in the pinned Playwright container (consistent
fonts/AA). Docker is unavailable on the dev box, so **CI generates them.**

- **`visual-gate` job** — runs in `mcr.microsoft.com/playwright/python:<tag>` (tag matched to the
  image's bundled Playwright; see lockstep below). Steps: `actions/checkout`,
  `git config --global --add safe.directory "$GITHUB_WORKSPACE"` (container runs as root vs checkout
  UID → "dubious ownership" otherwise), `pip install -e ".[viztest]"`, `pytest -m visual`, and
  `actions/upload-artifact` of `tests/visual_output/` on failure.
  **Opt-in trigger:** `pull_request` filtered to UI-affecting paths (`ui/**`, `www/**`, `app.py`) +
  manual `workflow_dispatch`.
  **NON-REQUIRED / advisory:** this job MUST NOT be added to branch-protection required checks — a
  path-filtered job reports *no status* on PRs that don't touch those paths, which would deadlock
  merges ("Expected — waiting for status"). Documented as advisory-only.
- **`visual-update` job** — `workflow_dispatch` only. Same container + `safe.directory` +
  `pip install -e ".[viztest]"`, then `OSMOSE_UPDATE_SNAPSHOTS=1 pytest -m visual`, then
  **`actions/upload-artifact` of the regenerated `tests/visual_baselines/*.png`**. The developer
  downloads the artifact and commits the PNGs locally (no Docker needed to *generate*; commit keeps
  authorship/signing intact). **This is the chosen default** because it needs no elevated token and
  is immune to fork-PR token restrictions. Set an explicit `retention-days` on the baseline artifact so
  it doesn't expire before the dev downloads it, and dispatch `visual-update` **from the PR's head
  branch** (optional `inputs.ref`) so baselines regenerate against that branch, not `master`.
  - *Documented alternative (opt-in):* have `visual-update` commit the baselines back to the branch
    directly. That requires `permissions: contents: write` on the job, only works on **same-repo**
    branches (fork PRs get a read-only token regardless), and should carry a `[skip ci]`-style bot
    commit. Recorded as a tradeoff; the artifact path is preferred for safety/simplicity.

**Lockstep (robust, not comment-only).** `pip` re-resolves an *unpinned* `playwright` to the latest on
PyPI, which can **upgrade past the image's bundled browser revisions** (→ "Executable doesn't exist" at
runtime). So `[viztest]` **pins `playwright==<image-version>`** matched to the chosen container tag, and
a small CI step asserts the installed `playwright` version equals the image's bundled version (fail
loudly on drift). The pin prevents the upgrade; the assert guards tag/pin coherence — both are code, not
a comment. (Version-assert alone, with an unpinned wheel, would only turn a silent break into a red gate;
it must be a pin.)

## 6. Dependencies & Guards

- **New optional extra `[viztest]`** in `pyproject.toml`:
  `["playwright==<image-version>", "pytest-playwright>=0.5", "pillow>=10"]` — `playwright` **pinned** to
  the container image's bundled version (§5 lockstep; prevents pip upgrading past the image's browsers).
  Kept **out of `[dev]`** so the normal CI legs (`lint`/`type-check`/`test`/`docker`, which install
  `.[dev]`) stay playwright-free.
- **Collection guard** in `tests/conftest.py`: the existing guard sets
  `collect_ignore_glob = ["test_e2e_*.py"]` only inside `if find_spec("playwright") is None`. Refactor to
  **initialize `collect_ignore_glob = []` first**, then append `"test_e2e_*.py"` when playwright is
  absent and append `"test_visual_regression.py"` when `find_spec("playwright") is None or
  find_spec("PIL") is None` (import name is `PIL`, not `pillow`). Initializing first avoids a `NameError`
  in the playwright-present-but-`PIL`-absent case; the two guards are distinct conditions and the visual
  append must not clobber the e2e one.
- **type-check legs:** safe by directory exclusion — `pyrightconfig.json` sets `exclude: ["tests"]`
  (pyright analyzes only `osmose`/`ui`), so `test_visual_regression.py` / `_visual_support.py` importing
  `playwright`/`PIL` at module top cause no type-check error (the same reason the `test_e2e_*` playwright
  imports are green today). Keep these files under `tests/`.
- `numpy` is already a core dependency; `pillow` (present, used by Plotly/kaleido stack) is declared in
  the extra for safety.
- **`.gitignore`:** add `tests/visual_output/` (not currently covered). **Do not** gitignore
  `tests/visual_baselines/` (intentionally committed); avoid any blanket `tests/visual_*` glob.

## 7. Testing the Test Infrastructure (TDD)

`compare_images` is pure and gets real unit tests in the **normal suite** (guarded only by numpy/Pillow
— no browser, runs in standard CI):

- Identical images → `passed=True`, `diff_ratio == 0`.
- Sub-`threshold` antialiasing-level noise → `passed=True`.
- Dimension mismatch → `passed=False`.
- An injected changed block exceeding `max_ratio` → `passed=False`, and the diff PNG highlights the
  changed region (assert a non-trivial red-pixel count there).
- `max_ratio` boundary: a change just under the ratio passes; just over fails.

The browser-driven `-m visual` snapshots are the integration layer; their stability is validated by
the container producing repeatable baselines and the gate re-comparing.

## 8. Non-Goals (v1)

- No chart/map/Run/Results/Calibration/Sensitivity/Spatial/Scenarios/About-modal snapshots
  (Plotly + deck.gl WebGL pixel nondeterminism; About modal is doc-text-driven).
- Not a mandatory gate on every PR — opt-in via path filter + manual dispatch; **non-required** check.
- Chromium only; single fixed 1280×900 viewport (no cross-browser or responsive matrix).
- Local runs advisory, not authoritative (font/AA drift vs the container).
- No automatic baseline approval — baseline refresh is a deliberate `workflow_dispatch` + manual commit.

## 9. Resolved Implementation Notes

- Clip selectors are pinned (verified): `#split_setup`, `#split_fishing`, `#split_movement`,
  `#split_advanced` — each an `osm-split-layout` `ui.div` wrapper. (`#split_advanced` wraps a Filters
  card + an "All Parameters" render-reactive card; deterministic under the loaded `minimal` config.)
- The concrete `playwright` version is whatever the chosen container image ships; the plan pins the
  image tag and adds the §5 version-assert step. (Installed locally today: playwright 1.58.0,
  pytest-playwright 0.7.2 — the image tag should match the wheel the team standardizes on.)
- `minimal` demo load is deterministic (no timestamp in config_name; `sp{i}`-ordered species).
- Masking remains available via `assert_clip_snapshot(mask=[...])` with pinned `mask_color="#FF00FF"`
  for any in-body dynamic element discovered during baseline capture, though element-clipping already
  excludes the header and validation panel.
