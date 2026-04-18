# Post-Session Roadmap — Closing Remaining Fronts

> **STATUS-COMPLETE (2026-04-19):** All six active fronts shipped on master. Front 6 (Ev-OSMOSE MVP + economics) turned out to already be ~95% implemented at roadmap time (Phase 1 MVP merge `5b0e0aa` and the Phase 2 Genetics/Economics Core commits predated v0.9.0); the STATUS-COMPLETE pass on 2026-04-19 added the one missing wire-up (`write_economic_outputs` dispatch from `simulate(output_dir=...)`) and applied banners to the three Front-6 plans at `docs/superpowers/plans/2026-04-06-ev-osmose-{economic-mvp,economics-core,genetics-core}-plan.md`. See commit log for per-front commits; shipped commits summarized below.

> **For agentic workers:** This is a **roadmap**, not a task-by-task plan. Each numbered front links to a concrete plan (existing or to-be-written). Work them in order; the ordering reflects dependency and session-constraint reality.

**Goal:** Close every open front surfaced by the 2026-04-18 session audit. Six fronts, ordered by (1) session-gating, (2) dependency unblocking, (3) effort, (4) novelty.

**Shipped (2026-04-18 → 2026-04-19):**
- **Front 1** — ICES MCP Baltic validation: snapshots + validator + regression fence shipped across commits `a7d65a5`, `2be016f`, `0435fde`, `590240e`, `4031138`.
- **Front 2** — Baltic Java fishery config reformat: plan `docs/superpowers/plans/2026-04-19-baltic-java-fishery-reformat-plan.md` written + executed; fishery name `_`-strip to match Java 4.3.3's `FishingGear.java:107` normalization; lint fence `tests/test_baltic_java_compat.py` (commit `98f478e`).
- **Front 3** — Calibration UI Phase 3: plan `docs/superpowers/plans/2026-04-19-calibration-ui-phase3-plan.md`; three UI surfaces (red-banner PreflightEvalError, n_workers slider, Pareto/weighted-sum toggle) across commits `c8b2e32`, `95e858d`, `84e5ab7`.
- **Front 4** — SP-4 Output System parity: spec `docs/superpowers/specs/2026-04-19-sp4-output-system-design.md`; plan `docs/superpowers/plans/2026-04-19-sp4-output-system-plan.md`; capability commits `5e96916` (5.5 diet), `d5947f0` (5.6 NetCDF distributions + mortality), `901b3be` (5.4 spatial); CHANGELOG + STATUS-COMPLETE `445f354`.

Test baseline: 2461 → 2484 passing (+23 new tests across Fronts 1-4). Ruff clean.

Next step per line 84 below: cut v0.9.0 release via `scripts/release.py minor` (bumps `osmose/__version__.py`, regenerates CHANGELOG, creates tag `v0.9.0`).

---


**Principles baked into the order:**

- **Session-gated work first** — the ICES MCP plan can only execute in a session started from `osmose-python/` with `.mcp.json` loaded. Don't waste an MCP-loaded session on offline work.
- **Unblocking before polish** — Baltic Java fishery reformatting is short and unblocks Java parity runs against the Baltic, which in turn tightens SP-4 validation.
- **Write-before-execute for gaps** — SP-4 and UI Phase 3 need plan documents before execution; write the plan, take a beat, then execute in a subsequent session (subagent-driven-development).
- **Net-new initiatives last** — Ev-OSMOSE MVP is a separate product thread, not cleanup.

---

## Front 1 — Execute ICES MCP Baltic validation **(START HERE on restart)**

- **Plan:** `docs/superpowers/plans/2026-04-18-ices-mcp-baltic-validation-plan.md`
- **Status:** Plan loop-reviewed (7 iterations, 0 blockers remaining). Ready to execute.
- **Session requirement:** Claude Code must be launched from `/home/razinka/osmose/osmose-python/` so `.mcp.json` loads the `ices` MCP server.
- **Effort:** ~90 min (8 tasks, most are small fetch-and-compare).
- **First action in the fresh session:**
  ```
  # Verify MCP is live:
  list_stocks(year=2023, area_filter="27\\.2[0-9]")
  # Expected: cod.27.24-32, her.27.25-2932, spr.27.22-32, fle.27.24-32 and others.
  ```
  Then follow the plan tasks 1 → 8.
- **Why first:** only task blocked on a specific session configuration; frees up subsequent sessions for offline work.

## Front 2 — Baltic Java fishery config reformatting

- **Plan:** not yet written. Short and contained.
- **Status:** Open. `.remember/remember.md` calls this out as needed for Java parity runs on Baltic.
- **Effort:** ~2-4 hours. Reformat `data/baltic/baltic_param-fishing.csv` + `data/baltic/fishery-catchability.csv` into the matrix format Java 4.3.3 accepts (different separator/layout convention from the Python engine's representation).
- **First action:** `Read data/baltic/baltic_param-fishing.csv` + `Read osmose-master/src/.../FishingMortality.java` for the expected Java format; write `docs/superpowers/plans/2026-04-19-baltic-java-fishery-reformat-plan.md` (3-5 tasks). Then execute in same session.
- **Why second:** short, mechanical, unblocks Java cross-checks for SP-4 output parity validation.

## Front 3 — Write + execute Calibration UI Phase 3

- **Plan:** not yet written. Scope documented in `2026-04-15-calibration-ui-phase2-plan.md` STATUS-COMPLETE banner's "known gap" note.
- **Status:** Open. Engine features exist but no UI surface.
- **Effort:** ~1 day (plan + execute). ~4-6 Shiny UI tasks.
- **Scope:** expose (a) `PreflightEvalError` surfacing as a distinct red-banner error state in the preflight modal, (b) `n_workers` dropdown/slider in the preflight settings, (c) Pareto vs weighted-sum toggle for the surrogate optimum view + `weights` input row.
- **First action:** `superpowers:brainstorming` on Phase 3 scope (ask user about desired UX for Pareto front vs weighted toggle — two valid designs), then write plan to `docs/superpowers/plans/2026-04-XX-calibration-ui-phase3-plan.md`, then execute.
- **Why third:** modest scope, high user-visibility value, consolidates calibration thread.

## Front 4 — Write + execute SP-4 output system

- **Plan:** not yet written. Scope documented in `docs/parity-roadmap.md` Phase 5.
- **Status:** Open. Largest remaining parity sub-project per MEMORY.md.
- **Effort:** ~3-5 days (plan + execute).
- **Scope:**
  - 5.1 Output recording frequency (`output.recording.frequency`)
  - 5.2 Yield/catches output (per-species, per-fishery)
  - 5.3 Size/age distribution outputs
- **First action:** Gap analysis — diff `osmose-master/src/.../OutputManager.java` class list against `osmose/engine/outputs/` (or wherever current Python outputs live). Identify which Java output classes have no Python counterpart. Write plan to `docs/superpowers/plans/2026-04-XX-sp4-output-system-plan.md`.
- **Why fourth:** biggest effort + best tackled after Java parity runs work (Front 2) so cross-validation is feasible.

## Front 5 — Write Calibration UI Phase 3 execution checkpoint

Covered under Front 3 — keeping this line here so the numbering stays aligned with my session summary.

## Front 6 — Ev-OSMOSE + Economic MVP (separate thread)

- **Plan:** `docs/superpowers/plans/2026-04-06-ev-osmose-economic-mvp-plan.md` (+ companion economics-core and genetics-core plans).
- **Status:** Open. Plans exist from 2026-04-06.
- **Effort:** Large (1-2 weeks). Separate product initiative, not cleanup.
- **Dependency:** none technically — plans are self-contained — but would consume bandwidth otherwise dedicated to fronts 1-4.
- **Recommendation:** defer until fronts 1-4 close, or until explicit product priority shifts.

---

## Expected end state after fronts 1-4

- Calibration validated against ICES SAG (fronts 1).
- Baltic usable on both Python and Java engines (front 2).
- Calibration UI surfaces every library feature (front 3).
- Output parity complete; engine phases 1-9 + outputs = full Java 4.3.3 equivalence except Ev-OSMOSE genetics (front 4).
- Ev-OSMOSE MVP (front 6) is the only open initiative — a product decision, not a cleanup task.

## Version bump trigger

After front 4 ships, the output system is a feature-complete milestone; cut v0.9.0 or v1.0.0 with a release commit per `scripts/release.py minor`.

---

## Session handoff checklist

When opening the next session:

1. `cd /home/razinka/osmose/osmose-python` (MCP discovery requires this CWD).
2. Verify `.venv/bin/python -m pytest -q` → **2432 passed**, ruff clean.
3. Read `MEMORY.md` for latest state (should point here).
4. Start **Front 1** unless user redirects.
