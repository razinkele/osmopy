---
name: project-deep-review-remediation-2026-06
description: 2026-06-20 whole-codebase deep review — findings + remediation plan; .env credential ROTATION OWED
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

Multi-agent deep review of the whole OSMOSE codebase, 2026-06-20 (7 subsystems, every finding adversarially verified refute-by-default; two complementary workflow runs — server-side rate-limiting thrashed which verifier batch failed each run, so coverage came from the union). **No confirmed CRITICAL active bug** — engine sound on all bundled-config/exercised paths (parity holds). Remediation plan: `docs/superpowers/plans/2026-06-20-deep-review-remediation.md` (branch `docs/deep-review-remediation`).

## ▶▶ OWED: rotate leaked credentials (USER action, provider login)
`.env` held the burned CMEMS + ICES password (literal redacted — see `tests/test_mcp_config_hygiene.py` for the burned string it scans for), world-readable AND in git history (find via `git log -S` on that burned literal, 3 commits). `tests/test_mcp_config_hygiene.py` already treats it as burned but it was still the LIVE value. **Done:** `chmod 600 .env` (2026-06-20). **Still owed:** rotate both passwords at Copernicus Marine + ICES; move secrets to a 0600 systemd EnvironmentFile (not a repo-root file).

## ▶▶ DEEP-REVIEW V2 (2026-06-20, post-features): all actionable PRs SHIPPED #90–#93
Second adversarial whole-codebase workflow (retry-resilient verify, 0 unverified) after Map Builder + Scenario Wizard + the 5 v1 remediation PRs. Found real defects in THIS session's merges:
- **#90 `de6537d` SECURITY**: burned password (`<redacted>`) was plaintext in 2 tracked plan docs (incl. the remediation doc) — hygiene test only scanned `.mcp.json`. Redacted both + added whole-tracked-tree `git grep` scan (literal now only in the 2 detector tests). **Rotation STILL OWED** (public git history). Also removed dead ICES_* from `.env`; `.env` is 600.
- **#91 `f3846ac` PR-A maps**: polygon mask-edit wrote paint value not -99 (REAL regression in the Map Builder); `lonlat_to_cell` neg-offset edge; `MapGrid.blank` mask-shape guard.
- **#92 `701d026` PR-B scenario-store**: `list_scenarios`/`import_all`/`load` deserialization hardening (data.get, known-fields filter, per-entry skip), reject `.` name, guarded `handle_load`, clear `config_dir` on scenario load/wizard.
- **#93 `c0728e7` PR-C**: results.py CSV skip += OSError; runner timeout → status='failed'; PR-86 mortality warnings deduped (process-global set, flooded calibration).
- **LATENT-ONLY left (no bundled config hits; deferred):** extend the PR-1 `_warn_unsupported_mortality_features` guard to fishing selectivity types 2/3 (Gaussian/log-normal silently knife-edged in the interleaved loop); wizard demo-tempdir atexit-only leak; assorted telemetry drift (RunRecord.duration_sec=0 etc.). See `/tmp/.../w0ve6ni0v.output` for the full v2 report.

## ▶▶ REMEDIATION STATUS (2026-06-20): all 5 PRs SHIPPED; only credential rotation owed
- P0 `.env` chmod 600 DONE; **rotation still owed (USER, providers)**.
- PR-1 `f5b0fcf` (#86): engine warns on parsed-but-unapplied mortality features (catch fishing, by-class fishing/additional) — REJECT not wire (no parity config exercises them).
- PR-2 `643aac9` (#87): one calibration failure policy — bad candidate → inf on all backends (was: thread/serial crashed, process absorbed).
- PR-3 `286d3758` (#88): config/schema — `validate_value` string coercion; reader 4.4.0 read gate uses `_numeric_version` (suffix-tolerant); `_scale_rate_value` sentinel pass-through.
- PR-4 `ac5bdea` (#84) + `922dd59` (#85): scenario-backup `with_suffix` data-loss; Java run-cancel status; empty-CSV reader skip.
- PR-5 `dc65979` (#89): Dockerfile `COPY www/` + `ENV OSMOSE_DATA_DIR`; CI docker run-smoke-test; `demo.py` OSMOSE_DATA_DIR + loud warning vs silent stub.
- **Only remaining item: numba pure-Python fallback CI coverage** (item 4, deferred — orthogonal test-infra, low value). All fixes had failing-first regression tests; each merged green incl. CodeRabbit.

## Top latent findings (verified; not on bundled configs → parity unaffected)
- **Engine — parsed-but-unwired config features (silent wrong results):** catch-based fishing `mortality.fishing.catches.spN` ignored by the production mortality loop (working impl exists in `fishing.py`); per-age/size additional mortality `byDt.byAge/bySize` not applied (the parallel fishing-by-class IS wired — inconsistent). `osmose/engine/processes/mortality.py`. → PR-1: wire-or-reject.
- **4.4.0 migration (DEFERRED jar-swap path) latent traps:** read-side version helper is suffix-intolerant — `reader.py:93-99` imports `demo._version_tuple`, so `4.4.0-SNAPSHOT`→`(0,)`→larval divide-back skipped→~24× mis-scale. Also `_scale_rate_value` crashes on `null` sentinels. **Fold into the jar-swap resume plan Phase 0** ([[project-config-key-migration-440]]). Use `aliases._numeric_version`.
- **Calibration:** objective exceptions abort the DEFAULT thread/serial NSGA-II backend (process backend absorbs them) — `problem.py`; centralize one failure policy → `[inf]*n_obj`.
- **`OsmoseField.validate_value` crashes on string values for FLOAT/INT fields** (`schema/base.py:127-131`) — coerce like `validator.py:validate_field`.
- **IO data-loss — PR-4 ALL SHIPPED:** ~~`scenarios.py` `with_suffix('.bak')` dotted-name `rmtree` data loss~~ (#84 `ac5bdea`); ~~Java run-cancel misreported as "Failed"~~ + ~~per-output getters crash on empty CSV~~ (#85 `922dd59`, branch was `fix/pr4-io-robustness`). All three with failing-first regression tests.
- **Packaging (prod unaffected — systemd serves a source clone):** Docker image missing `COPY www/` → crash-loops; docker CI is build-only (can't catch runtime); `demo.py` resolves `data/` as a package sibling → silent stub configs in wheel/Docker installs; numba pure-Python fallbacks coverage-omitted + never run in CI.

## Cross-cutting theme
Inconsistent failure semantics across SIBLING paths — one degrades gracefully while its twin crashes/misreports (calibration thread vs process; `read_csv` vs per-output getters; Python-engine vs Java cancel; `handle_save`/`fork` vs `handle_load`). Prefer a shared per-subsystem failure-policy helper. Plus parsed-but-unwired config features = silent-divergence traps.

Quick wins (≤1 file, high value): ~~scenario-backup `with_suffix` fix (data loss)~~ **DONE PR #84 `ac5bdea` 2026-06-20** (append `.bak` not `with_suffix`; regression test); `validate_value` string coercion; 4.4.0 read-side version-helper swap; Docker `COPY www/`.
