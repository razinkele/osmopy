---
name: project-visual-regression-tests
description: "Opt-in Playwright screenshot gate (4 config pages + nav chrome), NON-REQUIRED, container-baselined; shipped 2026-06-16 (PR #64, 82428a2)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**UI visual-regression tests — FULLY SHIPPED + OPERATIONAL 2026-06-16** (PR #64 rebase-merged, master then `82428a2`; CI 6/6 green; built subagent-driven from a spec+plan that went through MANY in-loop review rounds across 8 angles).

Opt-in Playwright screenshot gate for 4 config page bodies (`#split_{setup,fishing,movement,advanced}`) + nav chrome (`#main_nav`). Pure `tests/_visual_compare.py compare_images` = 3 OR-ed gates: pixel-ratio + absolute-pixel-floor + **mean-delta** (mean-delta is ESSENTIAL — per-pixel threshold is BLIND to a uniform Bootstrap-style recolor). `tests/_visual_support.py` harness (per-test `minimal` load gated on `#config_header`, theme-pin, determinism CSS, element-clip, skip-on-absent-baseline, **retry-until-stable** capture, per-page updates). `.github/workflows/visual.yml` = opt-in (path-filter `ui/**`/`www/**`/`app.py` + dispatch), **NON-REQUIRED**, DIGEST-pinned `mcr.microsoft.com/playwright/python@sha256:678457c4...` (=v1.58.0-noble; tag is mutable→determinism hole). `playwright==1.58.0` in `[viztest]` (out of `[dev]`); pillow ALSO in `[dev]` so compare_images unit tests run in normal CI. **5 authoritative container baselines committed + inspected; visual-gate validated GREEN on a cross-run no-op compare.** Update baselines via **Visual→visual-update** dispatch → artifact → inspect → commit (runbook `tests/visual_baselines/README.md`).

**▶▶ 4 durable lessons (all caught by RUNNING, not by ~5 static review rounds across 8 angles):** (1) `.nav-pills` matched 4 navsets (strict-mode) → unique `#main_nav`; (2) byte-exact self-consistency too strict + a still-settling page (Setup species_panels, slower in container) → **retry-until-stable** capture (converges within gate tolerances, env-agnostic); (3) ini `addopts` carries `--dist loadfile` (xdist, in `[dev]` not `[viztest]`) → CI `pytest` errored "unrecognized --dist" → workflow runs `pytest tests/test_visual_regression.py -m visual -o addopts=""` (target the FILE + drop addopts, matching the proven-local cmd); (4) bare `pytest -m visual` collects the WHOLE suite → imports `test_feedback_api` (needs httpx, `[dev]`-only) → collection error → target the file. **Meta-lesson: make the CI invocation EXACTLY match the verified-local command.** `workflow_dispatch` needs the workflow on the default branch (GitHub) → baselines can't be generated pre-merge; skip-on-absent keeps the gate green until then. Detail in `docs/superpowers/*/2026-06-16-visual-regression-tests-ui*`.

Related flake gotcha: [[feedback-visual-harness-toast-gotcha]] (transient Shiny toast frozen-stable by the determinism CSS's animation-kill → suppressed via `#shiny-notification-panel{display:none}`; PR #66).
