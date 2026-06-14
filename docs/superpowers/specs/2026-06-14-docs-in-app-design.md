# Docs in the app — design (Sub-project 1)

**Date:** 2026-06-14
**Status:** Approved (design phase)
**Feature:** Surface the project docs inside the Shiny app — freshen `README.md`/`CHANGELOG.md`
(cutting v0.13.0), add a dismissable **startup modal** showing the latest changelog (once per
version), and make the **About** modal render `README.md`/`CHANGELOG.md` from the files instead of a
stale hardcoded changelog.

This is sub-project 1 of a two-part request; the **feedback system** (UI form + persistence + backend
API) is sub-project 2 and is out of scope here.

## Motivation

The header **About** modal (`ui/components/help_modal.py:about_modal`) embeds a **hardcoded changelog
stuck at "v0.1.0 — Initial release"** — wildly stale (the app is at 0.12.0 with much shipped since).
Users have no in-app way to see what changed, and `README`/`CHANGELOG` (the real, maintained docs)
are invisible in the running app. This feature makes the maintained docs the single source the UI
renders, and proactively shows release notes once per version.

## Decisions (locked during brainstorming)

1. **About surface = enhanced modal** (not a new nav page): keep the header "About" link → a
   `navset_pill` modal with **Overview / README / Changelog** tabs. (Rationale: reuses the modal +
   header-menu infra; the left sidebar already has 16 nav entries.)
2. **Startup modal = once per version**: a client-side `localStorage` "last-seen version" compared to
   `__version__`; shown only when they differ; recorded on dismiss. (Rationale: matches the existing
   `?`-key/Bootstrap client-side pattern; not naggy; per-browser.)
3. **Versioning = cut v0.13.0**: bump `osmose/__version__.py`, move `CHANGELOG.md`'s `[Unreleased]`
   into a dated `## [0.13.0] - 2026-06-14`, leave a fresh empty `[Unreleased]`, freshen `README.md`.
   **In-repo edits only** — no git tag, no `scripts/release.py`, no publish.

## Reuse (do not rebuild)

- `ui/components/help_modal.py` — `_bs_modal(modal_id, title, body, *, size)` builds a static
  Bootstrap-5 modal (client-side `data-bs-toggle`); `about_modal()` (the one we rewrite) and
  `help_modal()` (untouched) both use it. `ui.markdown(text)` renders GitHub-flavored markdown
  (tables/headers/links — the current About modal already uses tables) and **can render file
  contents** (read file → `ui.markdown(text)`).
- `app.py` — header "About"/"Help" links at `app.py:218-229` (`data-bs-toggle="modal"` +
  `data-bs-target="#aboutModal"`); modals placed in the layout at `app.py:293-295`; an existing
  client-side script (an IIFE with a `keydown` listener, ending `app.py:475`) shows that
  `bootstrap.Modal.getOrCreateInstance(el).show()` is the established show pattern (`app.py:468-472`).
- `osmose/__version__.py` (`__version__ = "0.12.0"`) — the version source, already shown as a badge
  (`help_modal.py:45`, `app.py:188`).
- `www/osmose.css` — `.osmose-about-version` / `.osmose-version-badge` / `.osmose-version-label`
  badge classes (reused by both modals).

## Architecture — four units

### 1. Doc freshening (content only, no code)

- `osmose/__version__.py`: `0.12.0` → `0.13.0`.
- `CHANGELOG.md`: move the **entire current `## [Unreleased]` block** (its `### Added` bullets —
  live-movement viz, scenario diff, config-diff panel, Pareto picker, cell-series, sensitivity
  explorer, usage guide, etc.) under a new `## [0.13.0] - 2026-06-14` heading; leave a fresh empty
  `## [Unreleased]` above it. (The docs-in-app feature's *own* CHANGELOG entry then lands under that
  fresh `[Unreleased]` — see Testing/Note.)
- `README.md`: freshen the status/feature list to name the new UI surfaces (Scenario Diff,
  Sensitivity Explorer, config-diff panel, live-during-run movement, spatial cell-series). Light
  accuracy edit, not a rewrite.

### 2. `osmose/docs_content.py` (pure loader — no UI import, unit-tested)

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # osmose/docs_content.py -> repo root
_DOC_FILES = {"readme": "README.md", "changelog": "CHANGELOG.md"}

def read_doc(kind: str, *, root: Path | None = None) -> str:
    """Read README.md/CHANGELOG.md by whitelist key. Returns a fallback string if the
    file is missing/unreadable; raises ValueError for an unknown kind."""

def latest_changelog_entry(text: str) -> dict:
    """Parse the first dated '## [x.y.z] - date' section of a changelog. Returns
    {"version": str|None, "date": str|None, "body": str}. If no dated section exists
    (only [Unreleased]), falls back to the [Unreleased] body, else a 'No release notes
    yet.' message. Never raises."""
```

- `read_doc`: maps `kind` via `_DOC_FILES` (whitelist — no arbitrary paths); reads
  `(root or _PROJECT_ROOT)/<file>`; on `OSError` returns `"_Documentation unavailable._"`. Unknown
  `kind` → `ValueError`. The `root=` param exists so the fallback path is unit-testable.
- `latest_changelog_entry`: scan for the first dated version heading with
  `^##\s*\[(?!Unreleased\])([^\]]+)\]\s*(?:-\s*(\S.*?))?\s*$` (group 1 = version, group 2 = optional
  date; `date is None` when no `- date` suffix). **`body` = the lines strictly AFTER that heading line
  up to (but excluding) the next `^## ` heading, with leading/trailing whitespace stripped — the
  version heading line itself is NOT included** (the modal title already shows the version, so
  including it would double-print). Tolerant of an empty/missing dated section (fallback as
  documented). Pure (operates on the passed text).

### 3. Enhanced About modal (`ui/components/help_modal.py:about_modal`)

Replace the hardcoded body with a `ui.navset_pill` of three `nav_panel`s:
- **Overview** — keep the curated intro + the Tech-Stack table + the version badge
  (`.osmose-about-version`); **drop** the stale hardcoded "### Changelog" block.
- **README** — `ui.markdown(read_doc("readme"))`.
- **Changelog** — `ui.markdown(read_doc("changelog"))`.

Still wrapped by `_bs_modal("aboutModal", "About OSMOSE", body, size="xl")` (bumped `lg`→`xl` for the
longer content; already `modal-dialog-scrollable`). The header "About" link is unchanged. Docs are
read at **UI-build time** (app startup), matching the static-modal pattern; a restart/deploy picks up
new content.

### 4. Startup changelog modal (`ui/components/help_modal.py:changelog_modal` + script in `app.py`)

- `changelog_modal()` — a static `_bs_modal("changelogModal", f"What's new in v{__version__}", body,
  size="lg")` whose body is the version badge + `ui.markdown(latest_changelog_entry(read_doc("changelog"))["body"])`
  + a small "Full changelog → About" hint. Placed in the `app.py` layout next to `about_modal()` /
  `help_modal()` (`app.py:293-295`).
- **Version injection + auto-show script** in `app.py` (alongside the existing IIFE, before
  `theme=THEME`; **add `import json` to `app.py`** — it is not currently imported):
  - `ui.tags.script(f"window.OSMOSE_VERSION = {json.dumps(__version__)};")` (json-quoted, safe).
  - an auto-show script that **binds to `shiny:connected`** (the established pattern at `app.py:429`
    for bootstrap-dependent work) — **NOT** bare `DOMContentLoaded`, because Bootstrap's bundle is not
    reliably loaded at `DOMContentLoaded` (the deck.gl fallback at `app.py:301-309` documents exactly
    this race, and `app.py:408,469` both pre-check `typeof bootstrap === "undefined"`). On
    `shiny:connected` (with a short poll/retry guarding `typeof bootstrap !== "undefined"`, mirroring
    `app.py:301-309`), read `localStorage.getItem("osmose_seen_changelog_version")` and — if it
    `!== window.OSMOSE_VERSION` — call
    `bootstrap.Modal.getOrCreateInstance(document.getElementById("changelogModal")).show()`. Separately
    bind `changelogModal.addEventListener("hidden.bs.modal", ...)` →
    `localStorage.setItem("osmose_seen_changelog_version", window.OSMOSE_VERSION)`. Guards a missing
    element and wraps `localStorage` access in `try/catch` (private-mode → modal shows each load,
    acceptable). Recording on `hidden.bs.modal` means **any** dismissal marks the version seen.

`help_modal()` (User Guide) is untouched.

## Data flow

```
osmose/__version__.py ──► window.OSMOSE_VERSION (injected <script>)
CHANGELOG.md ─► read_doc("changelog") ─► latest_changelog_entry().body ─► ui.markdown ─► #changelogModal body   (build time)
README.md   ─► read_doc("readme")    ────────────────────────────────► ui.markdown ─► About "README" tab        (build time)
CHANGELOG.md ─► read_doc("changelog") ───────────────────────────────► ui.markdown ─► About "Changelog" tab     (build time)

shiny:connected (poll for bootstrap):  localStorage["osmose_seen_changelog_version"] !== OSMOSE_VERSION  →  show #changelogModal
modal hidden:                          localStorage["osmose_seen_changelog_version"] =  OSMOSE_VERSION
header "About" link  →  #aboutModal (Overview / README / Changelog tabs)
?-key (existing)     →  #helpModal (User Guide, unchanged)
```

No server round-trip; no new reactives; no engine/config dependency.

## Error handling

- `read_doc`: missing/unreadable file → `"_Documentation unavailable._"` (never raises); unknown
  `kind` → `ValueError` (internal callers only pass `"readme"`/`"changelog"`).
- `latest_changelog_entry`: no dated section → `[Unreleased]` body, else `"No release notes yet."`;
  never raises.
- Startup script: runs on `shiny:connected` + polls for `bootstrap` before `.show()` (so it never
  races the Bootstrap bundle and silently no-shows); guards a missing `#changelogModal`; wraps
  `localStorage` access in `try/catch` (private-mode degrade = modal shows each load).
- `ui.markdown` on the 1278-line CHANGELOG renders inside a scrollable tab/modal body — acceptable.

## Testing

1. **Unit `tests/test_docs_content.py`** (pure):
   - `read_doc("readme")` non-empty + contains a known marker (e.g. `"OSMOSE"`); `read_doc("changelog")`
     contains `"Changelog"`/`"## [0.13.0]"`.
   - `read_doc("bogus")` raises `ValueError`.
   - `read_doc("readme", root=tmp_path)` (empty dir) → the `"_Documentation unavailable._"` fallback.
   - `latest_changelog_entry(sample)` with a dated `## [0.13.0] - 2026-06-14` section → returns
     `version=="0.13.0"`, `date=="2026-06-14"`, and a `body` containing that section's bullets and
     **not** the next section's; assert `body` does **not** start with `"## ["` (heading excluded).
   - a dated heading with **no** `- date` suffix (`## [0.13.0]`) → `date is None`.
   - `latest_changelog_entry("## [Unreleased]\n- foo\n")` (no dated section) → falls back to the
     Unreleased body (no crash).
2. **Structure tests** (`tests/test_app_structure.py`):
   - `str(about_modal())` contains `"README"`, `"Changelog"`, `"Overview"` and a known README/CHANGELOG
     marker; **no longer contains `"Initial release"`** (the stale hardcoded-changelog marker — present
     in the current modal, removed by the rewrite). Do NOT use `"v0.1.0 — Initial release"`: `ui.markdown`
     wraps the version in `<strong>`, so that contiguous substring never exists and the guard would be
     vacuously true.
   - `str(app_ui)` contains `changelogModal`, `window.OSMOSE_VERSION`, and
     `osmose_seen_changelog_version`.
   - `osmose.__version__.__version__ == "0.13.0"`; `CHANGELOG.md` contains `"## [0.13.0]"`.
   - `README.md` contains a new-feature marker (e.g. `"Sensitivity"`).
   - (These three presuppose **unit 1 (Doc freshening) is committed first** — current repo is at
     `0.12.0` with no `## [0.13.0]` and no README "Sensitivity" marker, so order matters.)
3. **e2e `tests/test_e2e_docs.py`**:
   - Fresh context (empty localStorage) → `#changelogModal` becomes visible automatically on load.
     Dismiss it; assert the guard actually ran by checking
     `page.evaluate("localStorage.getItem('osmose_seen_changelog_version')") == <version>`; then
     `page.reload()`, wait for the page to settle (`page.wait_for_selector(".nav-pills")`, mirroring
     `test_e2e_scenario_diff.py:105`), and assert `#changelogModal` is **not** visible — so the test
     proves suppression rather than racing it.
   - Click the header "About" link → `#aboutModal` visible → click the "Changelog" tab → it contains a
     known marker (assert the plain `"0.13.0"`, not `"[0.13.0]"`, to avoid any markdown
     bracket/link ambiguity).

**Gates:** every task runs `ruff check`, `ruff format --check`, and `pyright`. **CHANGELOG note:** this
feature's own entry lands under the *fresh* `## [Unreleased]` created by unit 1 (since 0.13.0 captures
only what shipped before this feature).

## Out of scope (YAGNI)

- Auto-generating the changelog from commits; a richer "what's new" diff beyond the latest section.
- Live doc reload without restart (a server `@render.ui` re-read).
- In-doc table-of-contents / anchor navigation.
- Git tag / PyPI publish / `scripts/release.py` run (version bump is in-repo only).
- The **feedback system** (sub-project 2: UI form + persistence + backend API).
