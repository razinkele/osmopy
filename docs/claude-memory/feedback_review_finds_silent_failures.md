---
name: Parallel review catches silent-failure-mode bugs that green tests miss
description: After shipping a non-trivial module with green tests, dispatch a code-reviewer agent to look for failure-mode coverage gaps. Tests verify happy paths; review surfaces the unhappy-path silent failures (silent data loss, infinite resume loops, success=True regardless of stop reason). Demonstrated 2026-04-29.
type: feedback
originSessionId: af3b28b2-0438-47e9-8b63-2b06b1debe34
---
**Rule:** When a non-trivial module (≥150 lines, novel logic) ships with green tests, dispatch a `feature-dev:code-reviewer` agent before relying on it for production work.

**Why:** On 2026-04-29 a parallel review of 3 just-shipped, fully-tested modules (osmose/calibration/cmaes_runner.py, surrogate_de.py, scripts/sensitivity_phase12.py) found 13 high-confidence findings — including 3 critical bugs that green test suites had not caught:
- Sensitivity script silently truncated existing y_csv on re-launch without `--resume` (would destroy 14h+ of work).
- Sensitivity script's `--resume` skipped NaN'd indices, producing an infinite loop of failed-resume attempts.
- CMA-ES runner's `success` flag was hardcoded to True regardless of whether cma stopped due to convergence (tolfun) or budget exhaustion (maxiter / pathological state).

These are textbook "looks-fine-on-the-happy-path" bugs. The tests verified the paths the author thought of. Review surfaced the paths the author didn't.

**How to apply:**
- After shipping a new module that handles failures, resume-from-disk, multi-process state, or external library wrappers, parallel-dispatch a code-reviewer with explicit "report only high/medium-confidence correctness, edge-case, and integration findings — skip style nits."
- Brief the reviewer with: what the module does, who calls it, specific concerns the author wants a second opinion on (this turns vague review into targeted review).
- When the reviewer finds something the author didn't think about, that's the win — don't waste review budget on style.
- Run review BEFORE the first production use, not after. The 2026-04-29 reviews caught the silent-truncation bug before any 14h sensitivity job had been launched on it.

**Counter-rule:** Don't over-rotate to review for trivial modules (<100 lines, pure functions, well-typed signatures). Review pays off proportional to the failure-mode surface area — measure that before spending an agent slot.
