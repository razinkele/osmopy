---
name: project-2026-06-14-features
description: "Three features shipped 2026-06-14: feedback system (ab6e6ed), docs-in-app v0.13.0 (044db54), sensitivity explorer (3eb4873). Key gotcha: Starlette route must insert(0,...) before Shiny's catch-all Mount"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Feedback system (sub-project 2 of 2) — SHIPPED 2026-06-14** (`ab6e6ed`). Header "Feedback" modal → JSONL `data/feedback/feedback.jsonl` (server-side, NO public POST) + token-gated read-only `GET /api/feedback` (403 unless `OSMOSE_FEEDBACK_TOKEN`). **▶▶ BLOCKER gotcha: mount a Starlette route via `app.starlette_app.routes.insert(0, Route(...))` — `add_route` appends AFTER Shiny's catch-all `Mount("/")` → silent 404.** Also: byte-compare the token (non-ASCII → TypeError/DoS); env-override store path for test isolation. Detail in `docs/superpowers/*/2026-06-14-feedback-system*` + `.remember/remember.md`.

**Docs-in-app (sub-project 1 of 2) — SHIPPED 2026-06-14** (`044db54`). About modal (`navset_pill` rendering real README/CHANGELOG via `ui.markdown(read_doc(...))`) + once-per-version "What's new" startup modal; cut in-repo **v0.13.0**. New pure `osmose/docs_content.py`. Gotchas: startup-modal JS binds `shiny:connected`+polls `bootstrap` (not DOMContentLoaded); pyright on app.py needs `--pythonpath`. **▶▶ LESSON: "verbatim" is a means not the end — a passing correct test trumps verbatim; ALWAYS run e2e at the controller level.** Detail in `docs/superpowers/*/2026-06-14-docs-in-app*`.

**Parameter Sensitivity Explorer — SHIPPED 2026-06-14** (`3eb4873`). Top-level "Sensitivity" page loads persisted Sobol results → ranked S1/ST tornado + table + CSV exports. Backend pre-existed (`SensitivityAnalyzer`); the gap was persistence (`osmose/calibration/sobol_io.py`) + browse surface (`ui/pages/sensitivity_explorer.py`). Gotchas: `rank_rows` dispatches 1-D/2-D on `int(n_objectives)>1` (not on objective_names presence); validate raw ts before `sobol_` prefix; NaN sinks to bottom. Detail in `docs/superpowers/*/2026-06-14-sensitivity-explorer*`.
