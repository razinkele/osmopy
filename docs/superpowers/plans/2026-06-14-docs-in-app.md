# Docs-in-app Implementation Plan (Sub-project 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freshen README/CHANGELOG (cut v0.13.0), render them in an enhanced About modal, and show a once-per-version startup "what's new" changelog modal.

**Architecture:** A pure `osmose/docs_content.py` loader (`read_doc` + `latest_changelog_entry`) read at UI-build time; the About modal gains Overview/README/Changelog tabs; a new `changelog_modal()` is auto-shown client-side (on `shiny:connected`, polling for Bootstrap) once per `__version__` via `localStorage`.

**Tech Stack:** Python 3.12, Shiny for Python (`ui.markdown`, `ui.navset_pill`, static Bootstrap modals), client-side JS, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-06-14-docs-in-app-design.md`

---

## File Structure

- **Modify** `osmose/__version__.py` — bump to 0.13.0 (Task 1)
- **Modify** `CHANGELOG.md` — cut [0.13.0] (Task 1); feature entry under fresh [Unreleased] (Task 5)
- **Modify** `README.md` — freshen status/feature rows (Task 1)
- **Create** `tests/test_docs_freshening.py` (Task 1)
- **Create** `osmose/docs_content.py` (Task 2)
- **Create** `tests/test_docs_content.py` (Task 2)
- **Modify** `ui/components/help_modal.py` — enhanced `about_modal` + new `changelog_modal` (Task 3)
- **Modify** `app.py` — `import json`, place `changelog_modal()`, version + startup scripts (Task 4)
- **Modify** `tests/test_app_structure.py` — modal/app wiring asserts (Tasks 3 & 4)
- **Create** `tests/test_e2e_docs.py` (Task 5)

Per-task gate: `.venv/bin/ruff check osmose/ ui/ tests/`, `.venv/bin/ruff format osmose/ ui/ tests/`, `.venv/bin/pyright <files touched>`.

**Task order matters:** Task 1 (doc freshening) must land first — Tasks 3/4 render the 0.13.0 content and several tests assert it.

---

### Task 1: Doc freshening (version + CHANGELOG cut + README)

**Files:**
- Modify: `osmose/__version__.py`, `CHANGELOG.md`, `README.md`
- Create: `tests/test_docs_freshening.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_docs_freshening.py`:

```python
"""Doc-freshening assertions: version bump + CHANGELOG 0.13.0 cut + README currency."""

from __future__ import annotations

import pathlib

_REPO = pathlib.Path(__file__).resolve().parent.parent


def test_version_is_0_13_0():
    from osmose import __version__

    assert __version__ == "0.13.0"


def test_changelog_has_dated_0_13_0_section():
    cl = (_REPO / "CHANGELOG.md").read_text()
    assert "## [0.13.0] - 2026-06-14" in cl
    # A fresh empty [Unreleased] remains above it.
    assert "## [Unreleased]" in cl
    assert cl.index("## [Unreleased]") < cl.index("## [0.13.0]")


def test_readme_names_sensitivity_page():
    rm = (_REPO / "README.md").read_text()
    assert "Sensitivity" in rm
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_docs_freshening.py -q`
Expected: FAIL (version is 0.12.0; no `## [0.13.0]`; no capital "Sensitivity" in README).

- [ ] **Step 3: Bump the version**

In `osmose/__version__.py`, change:

```python
__version__ = "0.12.0"
```

to:

```python
__version__ = "0.13.0"
```

- [ ] **Step 4: Cut the 0.13.0 changelog section**

In `CHANGELOG.md`, replace the single occurrence of:

```markdown
## [Unreleased]

### Added
```

with:

```markdown
## [Unreleased]

## [0.13.0] - 2026-06-14

### Added
```

This leaves a fresh empty `[Unreleased]` and moves everything currently under it beneath the dated `[0.13.0]` heading.

- [ ] **Step 5: Freshen the README status rows**

In `README.md`, make these three edits.

(a) Subtitle test count — replace:

```markdown
<sub>**Python 3.12** · **NumPy + Numba** · **Shiny for Python** · **2510 tests** · **ruff clean** · **MIT**</sub>
```

with:

```markdown
<sub>**Python 3.12** · **NumPy + Numba** · **Shiny for Python** · **3250+ tests** · **ruff clean** · **MIT**</sub>
```

(b) Shiny UI status row — replace:

```markdown
| Shiny UI | 10-tab end-to-end UI (Setup · Grid · Forcing · Fishing · Movement · Run · Results · Calibration · Scenarios · Advanced). |
```

with:

```markdown
| Shiny UI | End-to-end UI: Setup · Grid · Forcing · Fishing · Movement · Run · Results (with Scenario Diff & Config Diff) · Spatial Results · Diagnostics · Calibration · **Sensitivity** · Scenarios · Advanced · Map Viewer. Live-during-run movement map; in-app About (README/Changelog) + a per-version startup "what's new" modal. |
```

(c) Tests status row — replace:

```markdown
| Tests | 2510 passed, 15 skipped, 41 deselected. Pyright clean on `osmose/` and `ui/`. |
```

with:

```markdown
| Tests | 3250+ passed. Pyright clean on `osmose/` and `ui/`. |
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_docs_freshening.py -q`
Expected: PASS (3 passed).

- [ ] **Step 7: Guard against a stale version assertion elsewhere**

Run: `.venv/bin/python -m pytest -k version -q`
Expected: PASS. If any pre-existing test hardcodes `"0.12.0"`, update it to `"0.13.0"` (the bump is intentional).

- [ ] **Step 8: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright osmose/__version__.py tests/test_docs_freshening.py` → 0 errors.

- [ ] **Step 9: Commit**

```bash
git add osmose/__version__.py CHANGELOG.md README.md tests/test_docs_freshening.py
git commit -m "docs: freshen README + cut CHANGELOG v0.13.0; bump __version__"
```

---

### Task 2: `osmose/docs_content.py` loader

**Files:**
- Create: `osmose/docs_content.py`
- Create: `tests/test_docs_content.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_docs_content.py`:

```python
"""Unit tests for osmose.docs_content (doc loader + changelog parser)."""

from __future__ import annotations

import pytest

from osmose.docs_content import latest_changelog_entry, read_doc

_SAMPLE = """# Changelog

## [Unreleased]

## [0.13.0] - 2026-06-14

### Added
- Thing one
- Thing two

## [0.12.0] - 2026-05-08

### Added
- Old thing
"""


def test_read_doc_readme_has_marker():
    assert "OSMOSE" in read_doc("readme")


def test_read_doc_changelog_has_marker():
    text = read_doc("changelog")
    assert "Changelog" in text and "## [0.13.0]" in text


def test_read_doc_unknown_kind_raises():
    with pytest.raises(ValueError):
        read_doc("bogus")


def test_read_doc_missing_file_fallback(tmp_path):
    assert read_doc("readme", root=tmp_path) == "_Documentation unavailable._"


def test_latest_entry_parses_first_dated_section():
    e = latest_changelog_entry(_SAMPLE)
    assert e["version"] == "0.13.0"
    assert e["date"] == "2026-06-14"
    assert "Thing one" in e["body"] and "Thing two" in e["body"]
    assert "Old thing" not in e["body"]  # stops at the next ## section
    assert not e["body"].startswith("## [")  # heading line excluded


def test_latest_entry_date_optional():
    e = latest_changelog_entry("## [0.13.0]\n\n### Added\n- x\n")
    assert e["version"] == "0.13.0" and e["date"] is None


def test_latest_entry_unreleased_only_fallback():
    e = latest_changelog_entry("# Changelog\n\n## [Unreleased]\n- foo\n")
    assert e["version"] is None and "foo" in e["body"]


def test_latest_entry_empty_fallback():
    e = latest_changelog_entry("# Changelog\n\nnothing here\n")
    assert e["body"] == "No release notes yet."
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_docs_content.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.docs_content'`.

- [ ] **Step 3: Implement `osmose/docs_content.py`**

```python
"""Read and parse the project's README/CHANGELOG for in-app rendering.

Pure module (no UI import). Reads the maintained docs from the repo root by a fixed
whitelist and parses the latest changelog entry for the startup "what's new" modal.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # osmose/docs_content.py -> repo root
_DOC_FILES = {"readme": "README.md", "changelog": "CHANGELOG.md"}
_FALLBACK = "_Documentation unavailable._"

# First dated version heading "## [x.y.z] - YYYY-MM-DD" (date optional); skips [Unreleased].
_VERSION_HEADING = re.compile(
    r"^##\s*\[(?!Unreleased\])([^\]]+)\]\s*(?:-\s*(\S.*?))?\s*$", re.MULTILINE
)


def read_doc(kind: str, *, root: Path | None = None) -> str:
    """Return a whitelisted doc file's text ("readme"/"changelog").

    Returns a fallback string if the file is missing/unreadable; raises ValueError
    for an unknown kind.
    """
    try:
        filename = _DOC_FILES[kind]
    except KeyError:
        raise ValueError(f"Unknown doc kind: {kind!r}") from None
    path = (root or _PROJECT_ROOT) / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK


def _section_body(text: str, start: int) -> str:
    """Text from ``start`` up to the next '## ' heading, stripped."""
    nxt = re.search(r"^##\s", text[start:], re.MULTILINE)
    end = start + nxt.start() if nxt else len(text)
    return text[start:end].strip()


def latest_changelog_entry(text: str) -> dict:
    """Parse the first dated version section of a changelog (pure, never raises).

    Returns {"version": str|None, "date": str|None, "body": str}. version/date come
    from the first "## [x.y.z] - date" heading (date None if absent); body = lines
    AFTER that heading up to the next "## " heading, stripped (heading excluded).
    Falls back to the [Unreleased] body, else "No release notes yet.".
    """
    m = _VERSION_HEADING.search(text)
    if m is None:
        um = re.search(r"^##\s*\[Unreleased\]\s*$", text, re.MULTILINE)
        if um is not None:
            body = _section_body(text, um.end())
            if body:
                return {"version": None, "date": None, "body": body}
        return {"version": None, "date": None, "body": "No release notes yet."}
    version = m.group(1).strip()
    date = m.group(2).strip() if m.group(2) else None
    return {"version": version, "date": date, "body": _section_body(text, m.end())}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_docs_content.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright osmose/docs_content.py tests/test_docs_content.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add osmose/docs_content.py tests/test_docs_content.py
git commit -m "feat(docs): osmose/docs_content loader + changelog parser"
```

---

### Task 3: Enhanced About modal + startup changelog modal

**Files:**
- Modify: `ui/components/help_modal.py`
- Modify: `tests/test_app_structure.py`

- [ ] **Step 1: Write the failing structure tests**

In `tests/test_app_structure.py`, append:

```python
def test_about_modal_renders_doc_tabs():
    from ui.components.help_modal import about_modal

    html = str(about_modal())
    assert "Overview" in html and "README" in html and "Changelog" in html
    # Renders real CHANGELOG content (build-time read), not the old hardcoded block.
    assert "## [0.13.0]" in html or "0.13.0" in html
    # The stale hardcoded-changelog marker is gone (was a list item "Initial release").
    assert "Initial release" not in html


def test_changelog_modal_present():
    from ui.components.help_modal import changelog_modal

    html = str(changelog_modal())
    assert "changelogModal" in html
    assert "What's new" in html
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_about_modal_renders_doc_tabs tests/test_app_structure.py::test_changelog_modal_present -q`
Expected: FAIL (`changelog_modal` doesn't exist; About modal still has the hardcoded block / "Initial release").

- [ ] **Step 3: Add the docs_content import**

In `ui/components/help_modal.py`, after `from osmose import __version__` (line 9), add:

```python
from osmose.docs_content import latest_changelog_entry, read_doc
```

- [ ] **Step 4: Rewrite `about_modal`**

Replace the entire `about_modal()` function (lines 41-92) with:

```python
def about_modal():
    """About OSMOSE modal — curated Overview + rendered README / Changelog tabs."""
    overview = ui.TagList(
        ui.div(
            ui.tags.span(f"v{__version__}", class_="osmose-version-badge"),
            ui.tags.span("Python Interface", class_="osmose-version-label"),
            class_="osmose-about-version",
        ),
        ui.markdown(
            """
**OSMOSE** (Object-oriented Simulator of Marine Ecosystems) is an individual-based
model for exploring marine ecosystem dynamics. This Python interface provides
configuration, execution, calibration, and visualization of OSMOSE simulations.

---

### Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Shiny for Python |
| Plotting | Plotly |
| Calibration | pymoo (NSGA-II) |
| Sensitivity | SALib (Sobol) |
| GP Surrogate | scikit-learn |
| Data | xarray, pandas |
| Simulation | Java (OSMOSE engine) |

---

Released under the **MIT License**. [View on GitHub](https://github.com/razinkele/osmopy)
"""
        ),
    )
    body = ui.navset_pill(
        ui.nav_panel("Overview", overview),
        ui.nav_panel("README", ui.markdown(read_doc("readme"))),
        ui.nav_panel("Changelog", ui.markdown(read_doc("changelog"))),
        id="about_tabs",
    )
    return _bs_modal("aboutModal", "About OSMOSE", body, size="xl")
```

- [ ] **Step 5: Add `changelog_modal`**

In `ui/components/help_modal.py`, add immediately after `about_modal()`:

```python
def changelog_modal():
    """Startup 'What's new' modal — the latest changelog section."""
    entry = latest_changelog_entry(read_doc("changelog"))
    body = ui.TagList(
        ui.div(
            ui.tags.span(f"v{__version__}", class_="osmose-version-badge"),
            ui.tags.span("What's new", class_="osmose-version-label"),
            class_="osmose-about-version",
        ),
        ui.markdown(entry["body"]),
        ui.markdown("_Full history under **About → Changelog**._"),
    )
    return _bs_modal("changelogModal", f"What's new in v{__version__}", body, size="lg")
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py -q`
Expected: PASS (all, including the two new ones).

- [ ] **Step 7: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright ui/components/help_modal.py tests/test_app_structure.py` → 0 errors.

- [ ] **Step 8: Commit**

```bash
git add ui/components/help_modal.py tests/test_app_structure.py
git commit -m "feat(ui): About modal renders README/Changelog + startup changelog modal"
```

---

### Task 4: app.py wiring (placement + scripts)

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app_structure.py`

- [ ] **Step 1: Write the failing structure test**

In `tests/test_app_structure.py`, append:

```python
def test_startup_changelog_wired_in_app():
    from app import app_ui

    html = str(app_ui)
    assert "changelogModal" in html
    assert "window.OSMOSE_VERSION" in html
    assert "osmose_seen_changelog_version" in html
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_startup_changelog_wired_in_app -q`
Expected: FAIL (none of the three strings present yet).

- [ ] **Step 3: Add `import json` and the modal import**

In `app.py`, add `import json` near the top (after `from pathlib import Path`, line 3). Update the help_modal import (line 10) from:

```python
from ui.components.help_modal import about_modal, help_modal
```

to:

```python
from ui.components.help_modal import about_modal, changelog_modal, help_modal
```

- [ ] **Step 4: Place the startup modal in the layout**

In `app.py`, in the modals block (lines 293-295), change:

```python
        # ── Modals (static HTML, triggered client-side) ─────────────
        about_modal(),
        help_modal(),
```

to:

```python
        # ── Modals (static HTML, triggered client-side) ─────────────
        about_modal(),
        help_modal(),
        changelog_modal(),
```

- [ ] **Step 5: Add the version + startup scripts**

In `app.py`, immediately before `theme=THEME,` (after the existing `ui.tags.script("""...""")` IIFE block that ends ~line 476), add two scripts:

```python
    ui.tags.script(f"window.OSMOSE_VERSION = {json.dumps(__version__)};"),
    ui.tags.script("""
    (function() {
        var KEY = "osmose_seen_changelog_version";
        function showIfNew() {
            // Bootstrap's bundle may load after shiny:connected — poll until ready
            // (mirrors the deck.gl fallback pattern elsewhere in this file).
            if (typeof bootstrap === 'undefined') { setTimeout(showIfNew, 150); return; }
            var el = document.getElementById('changelogModal');
            if (!el) return;
            var seen = null;
            try { seen = localStorage.getItem(KEY); } catch (e) {}
            if (seen !== window.OSMOSE_VERSION) {
                bootstrap.Modal.getOrCreateInstance(el).show();
            }
            el.addEventListener('hidden.bs.modal', function() {
                try { localStorage.setItem(KEY, window.OSMOSE_VERSION); } catch (e) {}
            });
        }
        document.addEventListener('shiny:connected', showIfNew);
    })();
    """),
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py -q`
Expected: PASS (all).

- [ ] **Step 7: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright app.py tests/test_app_structure.py` → 0 errors.

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_app_structure.py
git commit -m "feat(ui): wire startup changelog modal (shiny:connected, once per version)"
```

---

### Task 5: e2e + CHANGELOG entry

**Files:**
- Create: `tests/test_e2e_docs.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the e2e test**

Create `tests/test_e2e_docs.py`:

```python
"""End-to-end test for the in-app docs (startup changelog modal + About tabs).

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_docs.py -v -m e2e
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000


def test_startup_modal_shows_once_then_suppressed(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Fresh context (empty localStorage) → the what's-new modal auto-shows.
    expect(page.locator("#changelogModal")).to_be_visible(timeout=_LOAD_TIMEOUT)

    # Dismiss it (Esc closes a Bootstrap modal) and confirm the guard recorded the version.
    page.keyboard.press("Escape")
    expect(page.locator("#changelogModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)
    seen = page.evaluate("() => localStorage.getItem('osmose_seen_changelog_version')")
    assert seen and seen != ""

    # Reload (same context preserves localStorage) → modal does NOT reappear.
    page.reload()
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    expect(page.locator("#changelogModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)


def test_about_modal_changelog_tab(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Dismiss the startup modal first (it covers the header).
    if page.locator("#changelogModal").is_visible():
        page.keyboard.press("Escape")
        expect(page.locator("#changelogModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)

    # Open About from the header, switch to the Changelog tab, assert rendered content.
    page.get_by_role("link", name="About").click()
    expect(page.locator("#aboutModal")).to_be_visible(timeout=_LOAD_TIMEOUT)
    page.locator("#aboutModal").get_by_role("tab", name="Changelog").click()
    expect(page.locator("#aboutModal")).to_contain_text("0.13.0", timeout=_LOAD_TIMEOUT)
```

- [ ] **Step 2: Run the e2e**

Run: `.venv/bin/python -m pytest tests/test_e2e_docs.py -v -m e2e`
Expected: PASS (2 passed). (`screenshots/` already exists if a screenshot is added; none required here.)

- [ ] **Step 3: Add the feature's CHANGELOG entry (under the fresh [Unreleased])**

In `CHANGELOG.md`, under the now-empty `## [Unreleased]` (created in Task 1), add an `### Added` block:

```markdown
## [Unreleased]

### Added

- **ui (docs):** the project docs are now surfaced in the app — the **About** modal renders
  `README.md` and `CHANGELOG.md` in tabs (replacing a stale hardcoded changelog), and a dismissable
  startup "What's new" modal shows the latest release notes once per version (client-side, keyed on
  `__version__` via `localStorage`). New pure `osmose/docs_content.py` loader/parser.
```

- [ ] **Step 4: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright tests/test_e2e_docs.py` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_docs.py CHANGELOG.md
git commit -m "test(docs): e2e startup modal + About tabs; CHANGELOG entry"
```

---

## Final verification (after all tasks)

- [ ] Full non-e2e suite: `.venv/bin/python -m pytest -m 'not e2e' -n auto -q`
- [ ] e2e: `.venv/bin/python -m pytest tests/test_e2e_docs.py -v -m e2e`
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` and `.venv/bin/ruff format --check osmose/ ui/ tests/` clean
- [ ] `.venv/bin/pyright` clean on all touched files
- [ ] Final whole-implementation review before finishing the branch.

## Self-Review (plan author)

- **Spec coverage:** doc freshening + v0.13.0 cut (Task 1) ↔ spec unit 1; `docs_content` loader/parser (Task 2) ↔ unit 2; About modal tabs (Task 3) ↔ unit 3; `changelog_modal` + `shiny:connected`/poll startup script + `import json` (Tasks 3/4) ↔ unit 4; data-flow + error handling realized across Tasks 2/3/4; tests (Tasks 1-5) ↔ spec Testing 1/2/3; CHANGELOG feature entry under fresh [Unreleased] (Task 5) ↔ spec note. No spec requirement without a task.
- **Type/contract consistency:** `read_doc(kind, *, root=None) -> str` and `latest_changelog_entry(text) -> {version,date,body}` identical between definition (Task 2), tests (Task 2), and call sites (Task 3); `changelog_modal`/`about_modal` names match between help_modal.py (Task 3) and the app.py import/placement (Task 4); the localStorage key `osmose_seen_changelog_version` and `window.OSMOSE_VERSION` are identical across the startup script (Task 4) and the e2e (Task 5); `latest_changelog_entry` body excludes the heading line (Task 2 test asserts `not startswith("## [")`).
- **Ordering:** Task 1 first (version/CHANGELOG content the later tests assert); the `import json` need is called out in Task 4.
- **No placeholders:** every code/edit step shows concrete old/new content; commands have expected output.
