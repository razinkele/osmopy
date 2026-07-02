---
name: PYTHONPATH override required to benchmark a worktree branch
description: When measuring perf in an osmose-python worktree, set PYTHONPATH explicitly — `cd .worktrees/<branch>` alone is not enough because there is no .venv inside worktrees
type: feedback
originSessionId: d2b7f4a5-d107-4042-a473-f491e81f4df1
---
When running `scripts/benchmark_engine.py` (or any other script) against code in a `.worktrees/<branch>/` checkout, **set `PYTHONPATH` to the worktree root explicitly**. Worktrees do not have their own `.venv`; they share the main repo's interpreter, which by default imports from the main `master` checkout's `osmose/` package, not from the worktree's modified files.

**Why:** I shipped A1 (accessibility vectorisation) and got an apparent 88 ms regression measuring 4.960 s vs 4.872 s baseline. Cause: I had run `cd .worktrees/a1-accessibility-vectorise && .venv/bin/python ...` which failed with "No such file or directory" (no .venv there), then re-ran with `/home/razinka/osmose/osmose-python/.venv/bin/python scripts/benchmark_engine.py ...` from outside the worktree — that imported the **master** copy of `accessibility.py`, not A1's changes. The "regression" was just normal master-vs-master noise. The correct pattern caught a 17.7 % speedup.

**How to apply:**
```
PYTHONPATH=/home/razinka/osmose/osmose-python/.worktrees/<branch> \
  /home/razinka/osmose/osmose-python/.venv/bin/python \
  /home/razinka/osmose/osmose-python/scripts/benchmark_engine.py \
  --config eec_full --years 5 --repeats 7 --output /tmp/<branch>.json
```

If a perf measurement looks suspicious (regression where you expected a win, or vice versa), **first verify which copy of the code is actually loaded** before re-tuning the implementation. The microbench (`/tmp/a1_profile.py` pattern) is also a useful sanity check — if microbench shows a per-call speedup but the full-engine bench doesn't, the most likely cause is module-import path, not implementation cost.
