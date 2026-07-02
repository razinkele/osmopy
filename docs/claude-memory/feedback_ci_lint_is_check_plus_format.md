---
name: feedback_ci_lint_is_check_plus_format
description: "The CI \"lint\" job runs both ruff check AND ruff format --check on osmose/ ui/ tests/ — not `ruff check .`"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d0685b94-c147-4192-b2a2-983e55a7f8d6
---

The OSMOSE CI "lint" check (`.github/workflows/ci.yml`) runs TWO ruff commands:
`ruff check osmose/ ui/ tests/` and `ruff format --check osmose/ ui/ tests/`.
A red "lint" status can be the *format* check failing while `ruff check` is fully
green ("All checks passed!").

**Why:** On PR #50 I initially mis-diagnosed the lint failure as a `ruff check .`
parse error on tutorial-work CSVs (because running `ruff check .` locally walks the
whole tree, including `docs/tutorials/tutorial-work/` data files and `.venv`). That
theory was wrong — CI never scans `.`, only the three source dirs. The real failure
was `ruff format --check` flagging 3 files (config.py, reproduction.py,
test_engine_stock_recruitment.py) where hand-wrapped argument lists fit within 100
chars and ruff collapses them onto one line.

**How to apply:** To reproduce a lint failure, run the EXACT CI command
(`ruff check osmose/ ui/ tests/` then `ruff format --check osmose/ ui/ tests/`),
and run it against the PR BRANCH (checked out in `.claude/worktrees/<branch>/`),
not master. Don't run `ruff check .` — it scans data + .venv and produces false
parse errors that aren't what CI sees. ruff version drift (local 0.15.2 vs CI
0.15.15) did not matter here; formatting was identical. Fix is just
`ruff format <files>`. See [[feedback_pythonpath_worktree_benchmark]] for the
related "operate on the worktree, not master" lesson.
