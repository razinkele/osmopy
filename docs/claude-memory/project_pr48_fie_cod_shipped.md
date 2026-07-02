---
name: project-pr48-fie-cod-shipped
description: PR
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d1bf098-a2f0-40bf-8956-8ca282eb7799
---

PR #48 "Ev-OSMOSE FIE-on-cod: genetics plumbing + engine fixes + baltic_ev fixture" **MERGED to master 2026-05-27** as squash commit `29b5206` (local master fast-forwarded to match). CI fully green on the final commit (`a633186`). Supersedes the prior handoff that listed PR #48 as open with 5 unresolved CodeRabbit threads.

**Why:** closes the Ev-OSMOSE genetics/FIE feature line — genetics pipeline threaded through the Python engine, calibrated baltic_ev bioen fixture, paired high-F/low-F FIE demo, strict-validation support for `evolution.trait.<name>.*` keys.

**How to apply:** treat PR #48 as shipped; don't re-open the thread work below.

## CodeRabbit thread dispositions (don't re-litigate)
The 5 final unresolved threads were worked through 2026-05-27:
- **FIXED** — strict `validation.strict.enabled=error` rejected all baltic_ev genetics keys. Root cause: trait keys are indexed by a trait *name* (`imax`), but the validator's only wildcard `{idx}`→`\d+` required digits. Fix added a `{name}`→`\w+` placeholder in `config_validation.py` + registered the 6 trait shapes and `osmose.configuration.bioen/genetics` in `_SUPPLEMENTARY_ALLOWLIST`. +2 regression tests.
- **FIXED** — `run_fie_demo.py` `--seeds`/`--n-years` now positive-int-validated at argparse (was crashing in `pd.concat([])`).
- **REJECTED (whitefish manifest)** — baltic_ev has no whitefish; sp6 is **smelt** (swap was 2026-04-17). The only "whitefish" string under `data/baltic_ev/` is a comment. Adding it would inject a phantom species. See [[project-baltic-whitefish-to-smelt]].
- **REJECTED (hard 2% FIE gate in `test_fie_demo_direction.py`)** — kept deliberately. Test is preflight-gated (skips until cod-viability sentinel `.preflight_wired` exists), and the docstring's prescribed response to <2% is escalate-nyear, not relax. The 2% floor is ~6σ over multi-seed drift SD.
- **REJECTED (`gonad_ssb == 0.0` in `simulate.py`)** — correct as-is. It's a sum of non-negative gonad weights, so `==0.0` ⟺ all-zero (no FP cancellation possible). `atol=1e-12` would be a no-op at gonad scale and would perturb the seeding-fallback trigger — the documented root cause of this branch's env-dependent parity sensitivity.

## Leftover cleanup (not done — destructive, awaiting user go-ahead)
- Merged branch `worktree-ev-osmose-fie-cod` (remote + local + worktree at `.claude/worktrees/ev-osmose-fie-cod`) can be removed.
- 3 stashes on that branch, incl. `stash@{0}` "superseded CI-based parity skip", can be dropped.
