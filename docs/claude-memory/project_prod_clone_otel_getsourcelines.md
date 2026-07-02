---
name: project-prod-clone-otel-getsourcelines
description: "Prod served from a symlink to the live dev tree → editing app.py under the running process made Shiny 1.6.3 OTel inspect.getsourcelines raise TokenError, crashing server() → all handlers dead ('Load does nothing'). Fixed: prod runs from a git clone + guard. PR #67 b141bfa"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Symptom (deployed app, 2026-06-16/17):** clicking Load → Baltic did NOTHING (no toast). Not Baltic-specific — *every* interactive handler was dead for new sessions. Reproduced locally? NO — local load is ~500ms clean; the bug was purely the deployed long-running process.

**Root cause (from `journalctl -u osmose-shiny.service`):** an uncaught exception aborts `server()` at session start on every connection:
```
File ".../app.py", line 531, in server      # a JS comment INSIDE a ui.tags.script("""…""") string
  shiny/session/_session.py set_renderer → extract_source_ref(renderer_func)
  shiny/otel/_attributes.py → inspect.getsourcelines(...)
  tokenize.TokenError: unterminated string literal (detected at line 1)
```
Prod served from a **symlink `/srv/shiny-server/osmose` → the live dev working tree**. The uvicorn process imported `app.py` at startup (renderer funcs carry `co_firstlineno` from then), but Shiny 1.6.3's **per-session OTel `extract_source_ref`** calls `inspect.getsourcelines`, which reads the *current* on-disk file. Editing the dev tree later that day (PR #65 added JS to `app.py`, shifting line numbers) made the recorded line numbers drift → `getsourcelines` tokenized a fragment starting mid-JS-string → `tokenize.TokenError`. Shiny wraps that call in `except (TypeError, OSError, ValueError)` — which does NOT catch `TokenError` (not a subclass) — so it propagates and aborts `server()` mid-registration → handlers defined after the crash point (incl. `grid_server`'s `handle_load_example`) never register.

**Ruled out:** disabling OTel via `SHINY_OTEL_COLLECT` does NOT help — `set_renderer` calls `extract_source_ref` UNCONDITIONALLY (not gated by collect level / `is_otel_tracing_enabled`). Verified in the installed shiny source.

**Fix (PR #67, master `b141bfa`, 2026-06-17) — two layers:**
1. **`deploy.sh` serves prod from a dedicated git clone, not the dev tree.** Clone once from the public repo → `fetch` + `checkout DEPLOY_REF` (default `origin/master`) each deploy → restart. The clone (29 MB tracked, incl. bundled example configs; working tree is 2.9 GB of mostly-untracked artifacts, so clone ≫ rsync-with-excludes) only changes on an explicit deploy. The `osmose-java` JAR is NOT in git → provisioned from the deploy host. `OSMOSE_REPO_URL`/`OSMOSE_DEPLOY_REF` override. This removes the trigger: dev edits can never again mutate the running process's source.
2. **`app.py` `_harden_shiny_otel_source_ref()`** wraps `shiny.session._session.extract_source_ref` to return `{}` on ANY exception (applied at import, before any `server()` runs; `setattr` not direct-assign to satisfy pyright; idempotent). Future source/line drift degrades to empty OTel attrs instead of crashing.

**VERIFIED WORKING IN PROD (2026-06-17):** prod serves from `/srv/shiny-server/osmose-src` (symlink target), currently `@ 9ed8a7f` (b141bfa #67 + 9ed8a7f #68 health-check poll); fresh service process, HTTP 200, clean startup. Baltic loads on the live app (`https://laguna.ku.lt/osmose/`): Load → Baltic → toast "Loaded 'baltic' (601 parameters)", "8 species • 601 params", ~500ms, NO `TokenError`/traceback in `journalctl`. The original "Load does nothing" symptom is resolved and protected against recurrence. 3 regression tests guard it (`tests/test_otel_source_ref_guard.py`).

**▶▶ DURABLE LESSONS:**
- A long-running Python web process must NOT read its source from a file being edited under it. `inspect.getsourcelines` (used by Shiny OTel, tracebacks, some reloaders) re-reads disk with import-time line numbers → drift → `TokenError`/wrong source. Serve prod from an immutable-between-deploys checkout.
- `tokenize.TokenError` is NOT a subclass of `ValueError`/`OSError` — code that "guards" `getsourcelines` with those will still crash on it.
- A "deploy = symlink to dev tree" shortcut is a latent footgun the moment dev and prod share a host. The git-tracked tree (29 MB) vs working tree (2.9 GB) gap made a clone the clean decoupling.
- Health check: cold start takes several seconds (numba/schema import) → a single immediate curl probe reports a spurious `000`; poll until 200 (PR #68).

Extends [[project-deployment-restart-gotcha]] (same theme: stale/mismatched in-memory code on the long-running prod service). Clean-venv masking: local `.venv` cached a deckgl `/tmp` include that a clean CI env lacked, so a module-scope `import app` in the new test broke CI collection — defer `import app` into test bodies ([[feedback_ci_clean_venv_reproduction]]).
