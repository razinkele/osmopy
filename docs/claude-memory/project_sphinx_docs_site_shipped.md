---
name: project-sphinx-docs-site-shipped
description: "2026-06-21 Sphinx + GitHub Pages API-reference docs site — shipped, merged, deployed live"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Sphinx + GitHub Pages API reference site — SHIPPED 2026-06-21**, PR #94 (squash `1f3a37e`), all CI green, **live at https://razinkele.github.io/osmopy/** (verified: landing 200, `api/_autosummary/osmose.html` 200, `usage-guide.html` 200, private `osmose.engine._netcdf` 404 = correctly excluded).

**What it is:** a Sphinx + `myst-parser` docs site for the `osmose/` library — recursive auto-generated API reference + the existing narrative guides (usage-guide + both tutorials), built warnings-as-errors, auto-deployed to GitHub Pages on `master`. First docs site in the repo (none existed before).

**Files:** `docs/conf.py` (furo theme, recursive `autosummary`, `myst_parser` w/ `colon_fence` + `myst_heading_anchors=3`, `intersphinx` w/ `intersphinx_timeout=10`, `include_patterns` WHITELIST, `autodoc_typehints="description"`, data-only/import-safe); `docs/_templates/autosummary/module.rst` (public-filtered recursive template — iterates `modules`/`functions`/`classes`/`attributes`/`exceptions`, never `all_*`); `docs/api/index.rst` (`:recursive:` root over `osmose`); `docs/index.md` (landing + toctrees, root doc); `.github/workflows/docs.yml` (build `-W` on PR; build+deploy Pages on master; least-priv perms, major-pinned actions, deploy-job concurrency); `tests/test_docs_build.py` (fast guard: walk-import all osmose + exec conf.py data globals). `docs` extra = `sphinx>=8, myst-parser>=4, furo>=2024.1`. Plan/spec under `docs/superpowers/{specs,plans}/2026-06-21-sphinx-api-reference*`.

**Process:** brainstorm→spec→plan, TWO multi-agent in-loop workflow reviews (spec: 17 confirmed findings; plan: 10) w/ adversarial verification, then subagent-driven execution (9 tasks) + final holistic review. Reviews empirically reproduced builds (Sphinx 9.1.0) and caught real bugs before execution.

**DURABLE GOTCHAS (learned, verified):**
- **GitHub Pages must be enabled or the first master deploy 404s** — `actions/deploy-pages@v4` fails: "Failed to create deployment (status: 404) ... Ensure GitHub Pages has been enabled". Fix: enable once via `gh api -X POST /repos/razinkele/osmopy/pages -f build_type=workflow` (or Settings→Pages→Source="GitHub Actions"), then `gh run rerun <id> --failed`. The build job is unaffected; only deploy needs it.
- **autosummary writes `.rst` STUBS to the SOURCE dir** (`docs/api/_autosummary/`); rendered `.html` lands ONLY in the build dir (`docs/_build/html/api/_autosummary/`). Don't check for `.html` in the source tree (a plan-review bug — the check always-failed + triggered a needless `include_patterns` fallback).
- **`-W` builds must run from scratch** (`rm -rf docs/_build docs/api/_autosummary` before each) — Sphinx's incremental cache silently skips an unchanged-but-still-broken module → local false-green that then reds the fresh-checkout CI build.
- **MyST under `-W`:** relative `.md` links to out-of-srcdir (`../README.md`) or whitelist-EXCLUDED files emit `myst.xref_missing` (fatal); rewrite to absolute GitHub blob URLs. `nitpicky=False` does NOT suppress `myst.xref_missing` NOR intersphinx fetch-failure warnings.
- **Same-document heading anchors:** MyST's auto heading-anchor id ≠ docutils `make_id`. "## 6. Choose an engine & reproduce results" → docutils id `choose-an-engine-reproduce-results` (digit dropped, `&`+spaces→one dash). The original `#6-...--...` link resolved only via MyST *tolerant* matching; the reliable fix is an explicit `(choose-an-engine-reproduce-results)=` target above the heading (the "exact slug" link alone did NOT resolve under strict `-W`).
- **docutils "Undefined substitution referenced"** (fatal ERROR): bar-delimited tokens in docstrings (e.g. `|% change|`, `|pct_delta|` in `osmose/analysis.py`) parse as `|substitution|` refs → escape pipes or double-backtick the phrase.
- `docs/conf.py` is NOT in CI's ruff scope (`osmose/ ui/ tests/` only); keep it data-only so the `[dev]`-leg guard test can `exec` it without the `[docs]` extra installed.
- **Whole change is docs/docstring-only → ZERO prod Shiny-app runtime effect** — no `deploy.sh` needed; the only "deploy" is the Pages site. osmose/ edits were 5 files, docstring text only; full suite stayed 3538-pass.

Related: [[feedback-subagent-driven-gates-gotchas]], [[feedback-in-loop-review-pattern]].
