---
name: In-loop reviewer convergence pattern
description: Rotate reviewer subagent types across iterations to avoid shared-blind-spot false convergence; verify file:line claims with the actual code each round; bound the loop with "two consecutive iterations report no new findings" plus a fresh-eyes final pass.
type: feedback
originSessionId: d2b7f4a5-d107-4042-a473-f491e81f4df1
---
When iterating with reviewer agents to converge a plan or design doc:

**Rule.** Rotate reviewer subagent types across iterations. Two
consecutive rounds with the same agent type can both miss the same
class of issue (shared blind spot). Pull in a different reviewer
(e.g., `superpowers:code-reviewer` after several `feature-dev:code-
reviewer` rounds) before declaring the loop converged.

**Why.** In the deep-review plan loop on 2026-05-05, iters 1–4 used
the same two reviewer types in parallel. Both converged to "ready"
at iter 4. Iter 5 dispatched a fresh-eyes
`superpowers:code-reviewer` and immediately surfaced 3 issues the
prior rounds had missed: branch-vs-master state inconsistency, an
existing-`except`-block confusion, and 21-line file:line drift. The
final loop took 9 rounds, not 4.

**How to apply.**

1. Default reviewers per round: 1 claim-verifier (`feature-dev:code-
   reviewer` or `pr-review-toolkit:code-reviewer`) + 1 structural
   (`general-purpose` with prompt focused on plan-level concerns).
2. Every 3rd round, swap one slot to a fresh-eyes reviewer that hasn't
   seen the doc before — `superpowers:code-reviewer` is good for this.
3. Stopping rule: two consecutive rounds report no new substantive
   findings AND at least one of those rounds used a reviewer type that
   wasn't in the previous N rounds.
4. Verify every concrete claim with Read/Grep/Bash, not just by
   reading the plan. r1 in the deep-review session had multiple
   claims that simply weren't true (wrong schema count, inverted
   `searchsorted` reasoning) that only died once a reviewer ran the
   verification commands.
