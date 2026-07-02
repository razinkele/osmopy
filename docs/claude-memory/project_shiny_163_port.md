---
name: project-shiny-163-port
description: "shiny 1.6.3 version-promotion (not a UI rewrite) + prod rollout; the real win = cma 4.x fixing CMA-ES under numpy 2; shipped 2026-06-16 (PR #62, d8eadad)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**shiny 1.6.3 port — SHIPPED + ROLLED OUT 2026-06-16** (PR #62 rebase-merged, master `d8eadad`, branch deleted, local synced; all 6 CI legs green incl. docker + 3.12/3.13). A version-promotion, NOT a UI rewrite (app already 1.6-compatible).

Shipped: `shiny>=1.6.3,<1.7`, `shinyswatch>=0.11` (Bootstrap 5.3.8 match → precompiled theme, NO libsass), `shinywidgets>=0.7`, `cma>=4.0`, `shiny_deckgl @v1.9.2`; version-aware `deploy.sh` w/ floor check.

**Prod rollout done:** shared micromamba `shiny` env upgraded shinyswatch 0.9.0→0.11.0 + cma 3.2.2→4.4.4 (the real win — cma ≤3.3.0 was SILENTLY BREAKING CMA-ES under numpy 2 via removed np.Inf) via `sudo -u shiny` pip + `systemctl restart` (env is shiny-owned, NOT razinka-writable, needs elevation; shiny itself was already 1.6.3 there). Validation: clean py3.12 venv unit 3291 passed @ -n2 (cov 93.68%), e2e 50/50 under 1.6.3, ruff/pyright(3.12+3.13) clean.

**▶▶ 2 durable findings:** (1) deckgl v1.9.2 imports `layer_legend_widget` into its namespace but omits it from `__all__` → pyright `reportPrivateImportUsage` on the hasattr-guarded access in `ui/pages/grid_helpers.py` → suppress at call site (runtime fine); (2) at `-n auto` (14 local workers) 1-2 engine tests flake with a numba JIT compile error (`config.reload_config()`) under cold-compile resource contention — PRE-EXISTING (identical numba 0.65.1/numpy 2.4.6 on .venv), NOT the port; green at CI's 2 workers.

Gotchas: [[feedback_ci_clean_venv_reproduction]], [[project-deployment-restart-gotcha]].
