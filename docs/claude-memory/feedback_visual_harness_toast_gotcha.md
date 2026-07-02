---
name: feedback-visual-harness-toast-gotcha
description: "Visual-gate flake on a config page = a transient Shiny notification toast baked into the snapshot; suppress overlays in _DETERMINISM_CSS, don't re-bless the baseline"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

When the `visual-gate` job flakes on a config page with a **small, localized, non-recolor diff** (`diff_ratio` ~0.5–1%, `diff_pixels` a few thousand, **`mean_delta` ≈ 0.25** — near-zero, so NOT a uniform Bootstrap recolor), suspect a **transient overlay baked into the snapshot**, not a real UI change. The canonical culprit is a **Shiny notification toast**.

**Root cause (diagnosed + fixed 2026-06-16, PR #66, master `b9179af`).** `prepare_page` (`tests/_visual_support.py`) loads the `minimal` demo, which raises a *"Loaded 'minimal' (N parameters)."* toast in `#shiny-notification-panel` → `.shiny-notification`. It renders `position:fixed` bottom-right (overlapping the bottom-right of EVERY clip box, e.g. `#split_movement`) and lives ~2s in the DOM, then the node is **removed** (not faded). Because `_DETERMINISM_CSS` kills animations, the toast shows as a **fully-opaque, STABLE** overlay — so `_stable_screenshot`'s two-consecutive-match loop can't tell it apart from settled content and captures it. The committed baselines were taken after the toast expired, so any run whose capture lands inside the ~2s window diffs. This surfaced as an intermittent `test_visual_movement_page` failure (it's a later nav click, so timing in the CI container raced the toast).

**Fix = suppress the overlay in the harness, NOT a baseline update.** Added `#shiny-notification-panel{display:none!important}` to `_DETERMINISM_CSS`. The style tag is injected once (in `prepare_page`, after load) and persists across the app's SPA navigations, so every page and any future toast is hidden regardless of timing. Baselines were already toast-free → **no regeneration needed**. The runbook (`tests/visual_baselines/README.md`) now has a "Determinism safeguards" section saying: when a new transient overlay flakes a baseline, suppress it here rather than re-blessing it into the PNG.

**Why / How to apply (durable):**
- **Read the diff metrics first.** Low `mean_delta` + small localized `diff_pixels` = transient/overlay/AA, not a redesign. A high `mean_delta` would be the real-recolor case the mean-delta gate was added to catch. See [[feedback-in-loop-review-pattern]] for the discipline of reading the actual artifact (download the `visual-diffs` artifact: `gh api repos/razinkele/osmopy/actions/artifacts/<id>/zip` → inspect `<name>.diff.png`).
- **`_DETERMINISM_CSS` killing animations is a double-edged sword:** it removes fade nondeterminism but also freezes a timed overlay into a stable, capturable state. Any timed/auto-dismissing UI (toasts, transient banners) must be **hidden by selector**, not relied on to fade out.
- **`visual-gate` is path-filtered** (`ui/**`, `www/**`, `app.py`) and NON-REQUIRED. A `tests/**`-only PR (like the fix itself) does **not** auto-run it → verify via **`gh workflow run "Visual" --ref <branch> -f pages=all`** (workflow_dispatch runs `visual-gate` + `visual-update` in the pinned container against your branch). That dispatch is the authoritative proof; local `pytest -m visual` is native-browser and advisory only (font/AA drift). Full visual-regression context: the big "UI visual-regression tests" entry in MEMORY.md.
