---
name: User profile + collaboration preferences
description: Arturas's working environment, development workflow preferences, and audience constraints. Frame suggestions accordingly.
type: user
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
**User:** Arturas Razinkovas-Baziukas, arturas.razinkovas-baziukas@ku.lt, Klaipėda University (Lithuania).

**Environment:**

- Linux Mint (Ubuntu-compatible).
- Server Python: `/opt/micromamba/envs/shiny/bin/python3` (3.13); dev venv: `.venv/bin/python` (3.12).
- Cannot access localhost dev server remotely — demos and live reviews happen via `https://laguna.ku.lt/osmose/`.

**Workflow preferences:**

- Prefers the `superpowers:subagent-driven-development` workflow for executing plans (splits work across fresh subagent sessions with review checkpoints).
- Favors write-plan-before-execute for anything non-trivial.
- Single bundled PR > many small PRs for refactors in the same area (confirmed 2026-04-18 calibration-sensitivity thread).
- Pushes to origin only after explicit user authorization, not automatically.

**UI / product preferences:**

- Chose shinyswatch `superhero` theme (dark blue-grey, orange accents).
- Target audience: non-technical stakeholders (e.g., fisheries scientists, policy). Server-deployed UI, not a developer console — avoid jargon in visible strings, keep the onboarding path obvious.
- Favors ambient / maritime theming (Nautical Observatory CSS overlay) — aesthetic matters, not just functional.

**How to apply:**

- When drafting user-facing copy, imagine a fisheries scientist not a developer.
- When proposing refactors, lean toward one bundled change not a series.
- When offering to push, pause and ask.
- Explain architecture in terms the user (trained in modeling, not software) will recognize — avoid unexplained Shiny/reactive jargon when a plain description works.
