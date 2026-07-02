---
name: feedback-ci-clean-venv-reproduction
description: "CI \"Run tests with coverage\" red often = an undeclared transitive dep that .venv has but a clean [dev] install lacks; reproduce in a FRESH venv, not .venv"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c43bb8b2-9fc9-4f4c-a030-02009958769b
---

When a CI **test** job is red on a clean `pip install -e ".[dev]"` but the suite passes locally, suspect an **undeclared transitive dependency that the dev `.venv` happens to have but a clean `[dev]` install does not**. `.venv` accumulates extras over time (e.g. `httpx`, `fastmcp`, `copernicusmarine`, `python-dotenv` pulled in by something installed manually), so running the exact CI command in `.venv` gives a FALSE GREEN.

**Why:** Found 2026-06-15 — CI went red 2026-06-14 with two distinct undeclared-dep failures, both invisible in `.venv`:
1. `tests/test_copernicus_mcp_env.py` exec'd `server.py` (imports `copernicusmarine`/`fastmcp`, not in `[dev]`); it had guarded only `importorskip("dotenv")`, and once `dotenv` arrived transitively (required-by `fastmcp`/`pydantic-settings`) the test ran and hit missing `fastmcp` → error. Fix = `importorskip("copernicusmarine")`+`importorskip("fastmcp")` (skip heavy MCP deps, matching `test_copernicus_ltl_mask.py`). `6fa0769`.
2. `tests/test_feedback_api.py` used `starlette.testclient.TestClient`, which requires `httpx`; starlette ships with shiny but does NOT pull httpx → collection error. Fix = **declare** `httpx>=0.27` in `[dev]` (lightweight; keep the security-endpoint tests running, don't skip them). `003336c`.

**Treatment rule:** heavy/optional deps (copernicusmarine, fastmcp) → `importorskip`-guard the tests; lightweight deps whose tests you WANT in CI (httpx) → declare in `[dev]`.

**How to apply (durable):** to verify a CI test-red fix, build a throwaway venv and reproduce CI EXACTLY — `python3.12 -m venv /tmp/x && /tmp/x/bin/pip install -e ".[dev]"` then `pytest --collect-only -q -m "not e2e"` (catches collection/import errors in seconds) and the full CI command `pytest -n auto --cov=osmose --cov-fail-under=90`. Do NOT trust a `.venv` run for clean-install failures. I pushed a `.venv`-validated fix once and CI stayed red on a SECOND undeclared dep — the clean-venv run would have shown both at once. The deploy micromamba env (no fastmcp) is also a decent no-extras proxy for the import-skip cases. Coverage was never the issue (93.68% > 90%); the errors were collection-time import failures. Relates to [[feedback_ci_pyright_reproduction]] (same theme: a sibling env masks the real CI condition).
