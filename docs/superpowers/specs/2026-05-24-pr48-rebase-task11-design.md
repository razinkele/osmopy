# PR #48 (Ev-OSMOSE FIE-on-cod) — Rebase + Task 11 Resolution: Design

> Date: 2026-05-24
> Status: Design approved; ready for implementation-plan drafting
> PR: razinkele/osmopy#48 (`worktree-ev-osmose-fie-cod` → `master`)
> Target base: `origin/master` @ `3e58091` (post-PR-#49)

## Goal

Get PR #48 to merge-ready state on top of current `origin/master`, with Task 11
either passing on its own merits or formally deferred to a follow-up plan with
documented evidence.

PR #48 carries the Ev-OSMOSE FIE-on-cod genetics plumbing (Tasks 1–9), the FIE
demo script (Task 10), the FIE-direction regression test (Task 11, currently an
intentional fail), the tutorial (Task 12), plus four pre-existing engine bug
fixes surfaced during integration. It diverged before PR #49 (the CI unblocker)
landed and now needs rebasing onto a master that reformatted 91 files and drove
pyright to zero.

## Current state (verified 2026-05-24)

- **Divergence:** PR branch is 18 commits ahead of master, master is 8 commits
  ahead of the PR's base. Both sides touch ~98 files.
- **Master-side delta** (the 8 commits, all from PR #49): shiny_deckgl pin fix,
  `ruff format` on 91 drifted files (`6a7e941` — the dominant conflict source),
  Dockerfile `git` install, conftest e2e-skip, pyright 28→0 (`2bc6305`), pyright
  CI-mirror 57→silenced (`eda4273`), numba in `[dev]`.
- **PR #48 CI** currently red on lint, type-check, test (3.12), docker — measured
  against the stale base; rebasing onto the known-clean master should reset most.
- **Task 11** reports `drop_pct ≈ 0.06%` at year 50 vs a 2% threshold. Root cause
  is baltic_ev boom-bust dynamics (cod biomass: 16 M t y41 → 40 k t y44 →
  39 M t y49); the year-50 evaluation can land in a low-adult trough where
  size-selective fishing has too few adults to drive a selection differential.
  The genetics plumbing is verified independently by the Task 1–9 unit tests.

## Non-goals

- **PR #49's unmasked follow-ups** are out of scope:
  - **A** — 5 failures, optional deps missing from `[dev]` (3× `test_copernicus_ltl_mask`,
    2× `test_copernicus_mcp_env`).
  - **B** — 1 failure, `test_tutorial_3species` fixture scope timeout.
  - **C** — 3 failures, `test_engine_parity` baseline drift (research-depth; do not
    regenerate baselines speculatively).
  These are triaged in memory `project_pr49_ci_unblocker_shipped` and tracked
  separately.
- **Re-calibrating the baltic_ev boom-bust** is not in this plan. If Phase 2's
  nyear=100 run fails to break the noise floor, that calibration sprint becomes
  its own follow-up plan.
- **Splitting PR #48 into smaller PRs** (the PR body offers this) is deferred to
  keep this plan bounded.

## Approach: three formal phases, single PR forward

| Phase | Scope | Exit gate |
|---|---|---|
| **1a** | Mechanical rebase only — resolve conflicts, no semantic edits | `git rebase` completes; PR-introduced code compiles; diff vs pre-rebase HEAD shows only format-normalized changes |
| **1b** | CI repair for PR-introduced code | lint + type-check + docker green; test (3.12) red set == exactly {Task 11} ∪ {follow-ups A/B/C} |
| **2** | Task 11 resolution via nyear=100 | Conditional ladder (see below) |

Phases run sequentially with go/no-go gates. Phase 2 may terminate the plan early
(in its `<1%` branch) by shipping Phases 1a+1b alone.

## Phase 1a — Mechanical rebase

**Workspace:** the existing `.claude/worktrees/` worktree for
`worktree-ev-osmose-fie-cod`, leaving `master` untouched. Confirm clean state
first; abort if untracked work is present.

**Safety:** tag pre-rebase HEAD locally as `pr48-pre-rebase-backup` before
starting.

**Mechanic:** `git rebase origin/master` over the 18 commits (not `git merge` —
preserve the author's three logical commit groups for reviewers).

**Conflict resolution:**
1. Ruff-format collisions with `6a7e941`: take PR content, then `ruff format <file>`
   to apply master's formatting on top.
2. `git checkout --theirs <file>` **only** for files PR #48 owns entirely (new
   genetics modules, new tests, baltic_ev fixture data).
3. Real content conflicts in `osmose/engine/processes/reproduction.py`,
   `osmose/engine/genetics/inheritance.py`, `tests/conftest.py`, `pyproject.toml`:
   resolve by hand, preserving the four engine bug fixes.

**Validation (no test runs in 1a):**
- `git status` clean
- `ruff check osmose/ ui/ tests/` passes
- `ruff format --check osmose/ ui/ tests/` passes
- `.venv/bin/python -c "import osmose; import osmose.engine; import osmose.engine.genetics"` succeeds (the PR's new genetics package is at `osmose/engine/genetics/`: `expression.py`, `genotype.py`, `inheritance.py`, `trait.py`)
- `git diff pr48-pre-rebase-backup..HEAD -- <PR-only files>` shows only
  format-normalized, not semantic, changes

**Checkpoint:** `git push --force-with-lease` to `worktree-ev-osmose-fie-cod`.

**Estimate:** 1–3 h depending on real content conflicts beyond format-only.

## Phase 1b — CI repair

Reproduce each CI job locally, classify every failure as PR-introduced vs
pre-existing-on-master, fix only the former.

1. **lint** — `ruff check osmose/ ui/ tests/`. Likely green from 1a; fix
   PR-introduced lint only.
2. **type-check** — must run `pyright --pythonpath <ci-mirror>/bin/python` against
   a clean `[dev]`-only venv (per `feedback_ci_pyright_reproduction`: local pyright
   silently resolves a richer sibling venv and under-reports). The PR's new genetics
   modules never went through master's pyright sweep, so fresh errors are expected
   here. This is the long pole.
3. **docker** — `docker build` per CI Dockerfile. Inherits master's `git`-install
   fix via rebase; verify, don't assume.
4. **test (3.12)** — `.venv/bin/python -m pytest`. Classify the red:
   - **Expected red (leave):** `test_high_f_drives_lower_cod_imax_than_low_f`
     (Task 11), follow-ups A (5), B (1), C (3).
   - **Must-fix red:** anything else. PR claimed 2891 pass pre-rebase, so any *new*
     failure is a rebase regression → root-cause in 1a's conflict resolution.

**Exit gate:** lint + type-check + docker green; test (3.12) red set == exactly
{Task 11} ∪ {follow-ups A/B/C}. Document the classified red set as a PR comment.

**Verify live:** `gh pr checks 48 --watch` to confirm CI matches local
classification (rebuild CI trust explicitly per PR #49 memo; don't trust
local-only signal).

**Estimate:** 1–4 h, dominated by pyright CI-mirror reproduction and any rebase
regressions surfacing in the sweep.

## Phase 2 — Task 11 resolution (nyear=100 ladder)

Run `test_high_f_drives_lower_cod_imax_than_low_f` at `nyear=100` (edit the test's
`cfg["simulation.time.nyear"]`), 3 seeds × 2 arms. Branch on `drop_pct`:

| Result | Action |
|---|---|
| **≥ 2%** | Commit the nyear=100 change. Task 11 passes. Update tutorial caveat #7 to document the 100y result. PR merge-ready. **Plan complete.** |
| **1–2%** | Near the 2σ–6σ band per the plan-author's noise-floor math. Bump to 5 seeds (drift SD ∝ 1/√n_seeds), re-run. If ≥2% → commit with documented seed-count justification. Else → escalate to `<1%` branch. |
| **< 1%** | Signal structurally absent, not noisy. **Stop.** Leave Task 11 as today's intentional skip. Write a follow-up plan for the damping sprint (surface the 2026-05-22 4-phase design from `recent.md` first). Ship Phases 1a+1b alone: FIE plumbing + 4 engine fixes merge; Task 11 stays documented-limitation. |

**Why the ladder:** each branch has a pre-committed action, so "nyear=100" cannot
silently become weeks of fixture-tuning. Maximum cost is one 100y sweep plus
optionally one 5-seed re-run.

**Parallelism / wall-clock:** the test runs seeds in a serial for-loop. At ~3 h
per 100y run × 6 runs ≈ 18 h serial. Decision: **run serially via background**
(no new harness code; aligns with the bounded-runtime discipline in
`feedback_de_bounded_runtime`) rather than adding a `ProcessPoolExecutor`.
**Empirically verify per-100y-run cost** with one single-seed timing run before
committing to the 18 h estimate. Seeds are independent → an interrupted run
re-runs only the missing seed dirs.

**Rationale for nyear=100 as the primary lever:** the plan author explicitly
states "escalate to nyear=100 (~16 generations) BEFORE relaxing the threshold."
The noise-floor math predicts 3–10σ separation at 100y if the physics works; the
current 0.06% at 50y is no signal at all, consistent with cycle-phase masking
that should phase-average over 12–16 boom-bust cycles. nyear=100 is the cheapest
path to a definitive signal-or-no-signal verdict.

## Success criteria

- PR #48 CI: lint/type-check/docker green; test red set == {follow-ups A/B/C}
  (Task 11 green OR documented-skip per ladder).
- Four engine bug fixes preserved; 99/99 bioen+parity tests still pass.
- No semantic drift in PR-introduced code vs pre-rebase HEAD (other than the
  Task 11 nyear change).

## Risks + fallbacks

- *Rebase corrupts a later commit* → `git reset --hard pr48-pre-rebase-backup`,
  restart 1a.
- *Pyright surfaces many new errors on genetics code* → fix incrementally; if a fix
  needs design judgment, pause and consult rather than blanket `# type: ignore`.
- *nyear=100 still <1%* → not a plan failure; it is the designed `<1%` branch.
  Ship plumbing, defer calibration.
- *18 h run interrupted* → seeds are independent; re-run only missing seed dirs.

## Open item to verify during implementation

- Confirm the CI-mirror venv path for the pyright reproduction (per
  `feedback_ci_pyright_reproduction`).
- The `osmose/engine/processes/reproduction.py` and
  `osmose/engine/genetics/inheritance.py` paths carry the four engine bug fixes;
  these are the highest-value files to verify after rebase conflict resolution.
