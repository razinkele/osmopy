---
name: project-scenario-diff-polish-shipped
description: "Scenario-diff polish shipped 2026-06-22 — shared config-diff component + Scenarios compare modal; merged 3624c11, prod-deployed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9b0fdf28-14de-4aef-af1e-c62c175d61a7
---

# Scenario-Diff Polish — shipped 2026-06-22

Merge `3624c11` on master (5 commits, `--no-ff`), **pushed + prod-deployed + verified** (clone advanced `70c9499`→`3624c11`, NRestarts=0, `:8838` & `/osmose/` 200, no OTel crash, log clean).

**What:** DRY'd the two config-diff tables into one shared pure component and upgraded the Scenarios-page compare from a crude inline card to a modal.
- **New `ui/components/config_diff.py`** (pure): `classify_config_diffs(list[dict[str,str|None]])` (added=value_a None / removed=value_b None / else changed; sort changed→added→removed then alpha) + `render_config_diff_table` (count line + badged/scrollable Key/A/B/Change table; None→`—`, `""`→empty cell). Single source of truth.
- **`ui/pages/scenario_diff.py`** (run-diff page) now delegates to it — deleted its private `_classify_config_diffs`/`_CHANGE_ORDER` + the two style imports. Edge branches stay caller-owned (wording differs per surface).
- **`ui/pages/scenarios.py`**: inline "Compare Scenarios" card → **"Compare Scenarios" button + modal** (`size="l"`, `easy_close=True`, **reset-on-open** so no stale flash). Pure module-level `_resolve_compare_state(a,b,compare)` returns a tagged union `("none"|"same"|"identical"|"error", None)` / `("diffs", rows)`; `compare_results` switches on the tag (`payload is not None` narrows for pyright). Deleted the dead `update_compare_choices` effect + the now-orphaned `STYLE_DIFF_ROW` constant (`ui/styles.py`). Adapter is the **literal comprehension** `[{"key":d.key,"value_a":d.value_a,"value_b":d.value_b} ...]` (NOT `dataclasses.asdict`), matching `history.py` run-diff dict shape; guarded `compare()` in try/except (it was unguarded — deleted-scenario path).

**Process (full superpowers chain):** brainstorm → spec (2 review rounds) → plan (1 round, execution-verified) → subagent-driven execution (4 TDD tasks, fresh implementer + task review each) → final whole-branch review READY. 55 targeted tests + 1 e2e pass, ruff+pyright clean.

**Gotchas surfaced:** (1) the compare feature **already existed** as a crude inline card — recon-before-design caught it (designing from the backlog label alone would've built a 3rd table). (2) Task-1 implementer over-engineered with `TypedDict`/`cast` + falsely claimed lint-clean → task review caught the F401 → fix restored the brief's plain-dict code. (3) e2e: bare `.modal` matched hidden static modals (changelog/help) that also contain `<table>` → must scope to **`.modal.show`** (Bootstrap's visible-modal class). (4) `canonicalize_config` adds `osmose.version` to BOTH scenarios symmetrically → 1-key seed still diffs to exactly 1 key (e2e badge-count==1 holds). (5) modal-hosted `@reactive.event`/`@render.ui` work because Shiny binds by id on DOM insertion — proven by the existing wizard modal.

Spec `docs/superpowers/specs/2026-06-22-scenario-diff-polish-design.md`; plan `docs/superpowers/plans/2026-06-22-scenario-diff-polish.md`. Reuses run-diff infra [[project_scenario_diff_view]].
