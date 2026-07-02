---
name: Deep-review remediation plan shipped (9-iter convergence)
description: Five-phase remediation plan for 7-agent deep-review findings; landed on origin/master 2026-05-05 after 9-round in-loop reviewer convergence. Lists C/H IDs for navigation; flags the local-vs-origin divergence find that surfaced during merge.
type: project
originSessionId: d2b7f4a5-d107-4042-a473-f491e81f4df1
---
# Deep-review remediation plan

Lives at `docs/plans/2026-05-05-deep-review-remediation-plan.md` on
origin/master. Squashed commit `3851b9e`; merge commit on master `ebe8343`.

## What the plan covers (issue ID → location)

- **Phase 1 (1–2 d, Critical/High):** C1 movement schema/engine key
  inversion · C2 `output.bioen.sizeInf.enabled` casing · C3 path-traversal
  at `ui/pages/results.py:334, 586` · C4 cancellation + lost-error path ·
  C5 master-RED on `[baltic]` (background species)
- **Phase 2 (1 d):** H1 schema gaps from
  `data/examples/*_all-parameters.csv` (43 keys) and `data/baltic` (11) ·
  H4 add `examples`+`minimal` to validation parametrize · H2 doc sync
  (CLAUDE.md says 221, actual 223 via `sum(len(g) for g in
  osmose.schema.ALL_FIELDS)`)
- **Phase 3 (1 d):** H10 reproduction bounds at `config.py:490, 492`
  (NOT 469-470 — r1 was wrong) and `_load_spawning_seasons` at
  `config.py:919` · M2 feeding-stage boundary (verify Java's
  `FishStage` op first; current `side='right'` already gives `>=`
  semantics)
- **Phase 4 (1 d, optional, baseline-gated):** scratch-buffer hoisting,
  vectorisation, JIT memoisation
- **Phase 5 (2–3 d, splittable):** NaN propagation suite, JIT
  determinism, UI helper consolidation, path safety, etc.

## Execution prerequisite (load-bearing)

Plan's "Execution prerequisite — REBASE FIRST" section requires a
working branch rebased onto current `origin/master` before C5 work
starts. Verification: `ls data/baltic/baltic_param-background.csv`
returns the file. If absent, rebase didn't bring in the baltic
background-species CSVs and the `[baltic]` parametrize will go falsely
green.

## Loop-convergence pattern that worked

Round-trip cycle: dispatch 1–2 reviewer agents (rotated through
`feature-dev:code-reviewer`, `general-purpose` structural,
`superpowers:code-reviewer` fresh-eyes) → patch verified findings only
→ commit as r{n} with explicit revision-log entry → repeat. **Loop
terminates** when fresh-eyes reviewer reports no new substantive
issues. Took 9 rounds for this plan; per-round latency ~1–2 minutes
agent dispatch + ~2–5 min patch authoring. Total ~30–45 min wall-clock
for a 546-line plan.

Key blind-spot pattern observed: **two reviewers can converge on a
false "ready" signal if they share the same blind spot**. The
fresh-eyes superpowers reviewer in iter 5 caught 3 issues that the
prior 4 rounds had all missed (worktree-vs-master state, existing
`except` block in `_run_python_engine`, off-by-21 line refs in H10).
Conclusion: **rotate reviewer subagent types across iterations** —
don't dispatch the same agent type twice in a row when the loop seems
to be converging.

## Local-vs-origin divergence (caught at merge time)

Until 2026-05-05, `origin/master` was at `d4eebe1` (v0.10.0 release-
marker squash from 2026-04-21) while local master was at `cf5cb8e`
with **49 commits** that had never been pushed (B-H wiring, calibration
speedup, DE bounded-runtime, baltic background CSVs, etc.). After
landing PR #2 and merging origin into local with conflict resolution
in favour of local (`__version__.py` → `0.11.0`; `CHANGELOG.md` → both
sections), the merge commit `ebe8343` brought origin/master up to date.

**Implication for future plan authoring**: claims like "master is at
`<sha>`" are only meaningful if `<sha>` actually points to a public
ref. Verify with `git ls-remote origin master` before pinning a hash
into a plan's revision log or acceptance gate.
