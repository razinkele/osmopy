---
name: feedback-e2e-changelog-modal-dismissal
description: "Playwright e2e tests must dismiss the startup \"What's new\" changelog modal before nav clicks; the default suite hides this class of break"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c43bb8b2-9fc9-4f4c-a030-02009958769b
---

The docs-in-app once-per-version **"What's new" startup modal** (`#changelogModal`, shipped 2026-06-14) auto-shows on every **fresh Playwright browser context** (empty `localStorage` → version mismatch → shows on `shiny:connected`). It overlays the header and **intercepts pointer events**, so any e2e test that clicks a nav pill / header button / `#btn_load_example` right after load fails with "...#changelogModal intercepts pointer events ... retrying click".

**How to apply:** in any e2e test that performs a click/fill shortly after load, call `dismiss_changelog_modal(page)` (in `tests/_e2e_support.py`) right after `page.wait_for_selector(".nav-pills", ...)` and before the first interaction. It clicks the close button `[data-bs-dismiss='modal']` — **NOT Escape**, which needs modal keyboard focus that is intermittently absent after navigation (same lesson as [[project_docs_in_app]] / the docs+feedback e2e). Pure-visibility tests (only `expect(...).to_be_visible()`/`to_have_class`/`text_content()`) do NOT need it — `to_be_visible` ignores obstruction. Do NOT globally suppress the modal via a localStorage init script: `test_e2e_docs.py::test_startup_modal_shows_once_then_suppressed` deliberately asserts the modal SHOWS — dismiss it, don't suppress it.

**Why this stayed hidden (the real lesson):** the default suite runs `-m 'not e2e'` (pyproject `addopts`), so **neither CI nor the full-suite run ever executes e2e** — a UI change that adds a global overlay can silently break the entire e2e suite and nothing flags it. The docs-in-app modal broke **2 outright + ~6 latent** e2e failures across `test_e2e_{live_movement,csv_map_display,grid_maps,grid_overlay,map_viewer,reactive,scenario_diff,sensitivity_explorer}.py`; the prior features' "e2e N/N green" only held because they shipped BEFORE the modal merged that same day. Found 2026-06-15 only because a manual "run the live-movement smoke test" request executed `-m e2e`.

**Durable rule:** after shipping any always-present overlay / global modal / header change, RUN THE E2E SUITE EXPLICITLY (`.venv/bin/python -m pytest tests/test_e2e_*.py -m e2e -n 0`) — the green non-e2e suite proves nothing about e2e. Fixed + shipped to origin/master `4e740cb` 2026-06-15 (shared `dismiss_changelog_modal` helper; 50 e2e passed). Relates to [[feedback_in_loop_review_pattern]] (always run e2e at the controller level).
