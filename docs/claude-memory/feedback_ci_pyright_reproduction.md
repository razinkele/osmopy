---
name: ci-pyright-reproduction
description: "Reproducing CI pyright behavior locally requires --pythonpath against a clean [dev]-only venv; default discovery silently masks errors by picking up whichever sibling venv has more packages"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2e9ad84e-0447-461c-ae8e-36bcbf8152b6
---

When CI's `pyright` step fails but `.venv/bin/pyright` locally reports 0 errors with the same pyright version, pandas version, and Python version — pyright is implicitly resolving against a sibling venv with more packages installed (e.g. `numba` from `[numba]` extras). It silently masks `reportMissingImports` and shadows nearby stub differences.

**Why:** Without `--pythonpath`, pyright's default interpreter discovery walks env vars and parent directories. If a different venv has been used in the working tree recently, its `site-packages` can shadow the venv whose `bin/pyright` you ran.

**How to apply:**

```bash
# Build a venv that matches CI exactly
/usr/bin/python3.12 -m venv /tmp/ci_mirror
/tmp/ci_mirror/bin/pip install -e ".[dev]"   # the SAME line as .github/workflows/ci.yml

# Run pyright with explicit pythonpath — this is the critical flag
/tmp/ci_mirror/bin/pyright --pythonpath /tmp/ci_mirror/bin/python
```

The mirror venv reproduces CI's error count exactly. Without `--pythonpath`, you can get 0 errors locally and 57 on CI from the same source tree, same pyright binary.

**Incident (2026-05-22):** PR #49 fix("pyright errors 28 -> 0") looked clean locally; CI reported 57 errors. Root cause was my main `.venv` had numba + 80+ extra packages installed, so `from numba import njit` resolved locally but failed on CI's `[dev]`-only install. Three numba imports + five pandas-3.0 stub strictness sites became invisible. See commit eda4273 in PR #49 for the fix.

**Related:** [[check-call-path-before-perf-gate]] is the analogous "verify the runtime environment matches assumptions before trusting measurements."
