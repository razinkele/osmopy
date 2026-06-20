# Deep-Review Remediation Plan (2026-06-20)

Source: a multi-agent deep review of the whole codebase (7 subsystems, every finding adversarially verified refute-by-default; two complementary runs gave full subsystem coverage). **No confirmed CRITICAL active-correctness bug** — the engine is sound on all exercised/bundled-config paths. The actionable risks cluster as: a live secret exposure, parsed-but-unwired config features that silently produce wrong results, calibration robustness, the deferred 4.4.0 migration's latent traps, and packaging/CI gaps. This plan groups the fixes into prioritized, independently-shippable PRs with concrete file:line targets and verification.

**Cross-cutting theme to keep in mind:** the recurring defect is *inconsistent failure semantics across sibling code paths* — one path degrades gracefully while its twin crashes or misreports (calibration thread vs process backend; `read_csv` vs the per-output getters; Python-engine vs Java run cancellation; `handle_save`/`fork` vs `handle_load`). Prefer a shared per-subsystem failure-policy helper over one-off patches.

---

## P0 — Security (do first; partly done)

- [x] **`chmod 600 .env`** — done 2026-06-20 (was world-readable `-rw-rw-r--`).
- [ ] **Rotate the leaked credentials** (USER ACTION — needs provider login). `Razinka@2026` (CMEMS + ICES) was committed historically (`git log -S 'Razinka@2026'` shows 3 commits) and is still the live value; `tests/test_mcp_config_hygiene.py` already treats it as burned. Change both passwords at Copernicus Marine and ICES.
- [ ] **Move secrets out of a repo-root file for good.** Keep them only in a `0600` systemd `EnvironmentFile` / deployment secret store; have `load_dotenv` fall back to env vars. Add a deploy-time check that `.env` (if present) is mode `600`.

---

## PR-1 — Engine: wire or reject the silently-ignored config features (highest correctness value)

These are **parsed but never applied** in the production interleaved mortality/bioen loop → wrong results with no error for user-authored configs (bundled EEC/BoB/Baltic don't use them, which is why parity still holds).

- [ ] **Catch-based fishing** (`mortality.fishing.catches.spN`). `osmose/engine/processes/mortality.py` (~`_precompute_effective_rates`, lines ~175-267 / 656-791). A working implementation already exists in `fishing.py:_catch_based_fishing` with passing unit tests. **Fix:** wire it into `_precompute_effective_rates` (convert target catch → per-subdt removal proportional to fishable biomass), OR — minimum bar — raise/warn in config validation when `mortality.fishing.catches.spN` is set but unsupported by the active loop.
- [ ] **Per-age/size additional mortality** (`mortality.additional.byDt.byAge/bySize`). `mortality.py` lines ~656-693 (numba path) and ~124-172 (`_apply_additional_for_school`, non-numba fallback). The parallel `fishing_rate_by_dt_by_class` *is* wired — this is an inconsistent omission. **Fix:** add a by-class branch in BOTH paths, setting `eff_additional` per school via `ts.class_of(age_dt)` / `ts.get_by_class(step, class)`, mirroring `natural.additional_mortality` (lines ~44-60).
- **Verify:** a config that sets each feature produces the expected reduced biomass (new unit tests); existing 14/14 EEC + 8/8 BoB parity unchanged; `tests/test_engine_config_validation.py` stays warning-free.
- **Tests:** `tests/test_*mortality*`, a new targeted test per feature.

## PR-2 — Calibration robustness: one failure policy for all backends

- [ ] **Objective exceptions abort the default thread/serial NSGA-II backend** while the process backend absorbs them (`osmose/calibration/problem.py:224,232,388-389`). **Fix:** centralize in `_evaluate_candidate` — wrap objective eval in `try/except (_python_engine_errors)`, `log.warning`, return `[inf]*n_obj` (matching `_worker_eval`'s contract) so all three backends share one policy; a single pathological candidate no longer kills a multi-hour run.
- [ ] **`_validate_overrides` / numeric coercion** is fragile (`int(v)` shim crashes on `"3.0"`). Fold into the `validate_value` fix in PR-3 (coerce in the schema layer, pass raw strings).
- **Verify:** a deliberately-throwing objective yields `inf` candidates and the run completes on all backends (thread/serial/process).

## PR-3 — Config/schema hardening (also de-risks the deferred 4.4.0 jar swap)

- [ ] **`OsmoseField.validate_value` crashes on string values for FLOAT/INT fields** (`osmose/schema/base.py:127-131` — does `<`/`>` on a str). **Fix:** coerce inside `validate_value` (`try float(value)`; on failure append an "expected number" error), matching `osmose/config/validator.py:validate_field`. Then simplify `calibration/problem.py:_validate_overrides` to pass raw strings and drop its `int(v)` shim.
- [ ] **4.4.0 read-side migration uses a suffix-intolerant version helper** (`osmose/config/reader.py:93-99` imports `osmose.demo._version_tuple`). `_version_tuple('4.4.0-SNAPSHOT')` → `(0,)` → the larval-rate divide-back is skipped → larval mortality mis-scaled ~24×. **Fix:** use the suffix-tolerant `osmose.config.aliases._numeric_version` (or strip `-SNAPSHOT`/`+build` before comparing) so read and write gate identically. **Latent until jar-swap cutover — but this is exactly the deferred path; fold into the jar-swap resume plan's Phase 0.**
- [ ] **Larval-rate value migration crashes on sentinel values** (`osmose/config/aliases.py:61-63`, `_scale_rate_value`). **Fix:** skip empty / `{null,none,na,nan}` components verbatim, per `validator.py:31`'s unset convention. (MINOR; same deferred path.)
- **Verify:** round-trip tests for `4.4.0`, `4.4.0-SNAPSHOT`; `validate_value` on `"abc"`/`"3.0"`/`"null"` behaves; existing 3457-test suite green. Cross-link: [[project-config-key-migration-440]] / the jar-swap resume plan.

## PR-4 — IO / results robustness (sibling-path consistency)

- [ ] **Cancelled Java run misreported** as "Failed (exit -15)" (`osmose/runner.py`, `cancel()` ~236-243 sets no status; `run()` always returns `status='ok'`). **Fix:** add a `self._cancelled` flag set in `cancel()`, reset at `run()` start; after `wait()`, return `RunResult(returncode=-1, status='cancelled', ...)` per the docstring contract — matching the Python-engine path (`run.py:451-458`).
- [ ] **Scenario-save backup path collision → data loss** (`osmose/scenarios.py:83`). `target.with_suffix('.bak')` truncates at the last dot (`v1.2` and a real `v1.bak` both → `v1.bak`), then the unconditional `shutil.rmtree` (line 85) destroys the unrelated dir. **Fix:** use `target.parent / (target.name + '.bak')` or a uuid-named sibling temp dir. (Reproduced as real single-save data loss.)
- [ ] **Per-output getters crash on empty/partial CSV** (`osmose/results.py`, `_read_output_csv` ~98-101) while `read_csv` (~387-390) skips gracefully. **Fix:** centralize `try/except (pd.errors.ParserError, EmptyDataError)` + `log.warning` + continue in `_read_output_csv`, so `_read_species_output`/`_read_2d_output`/`size_spectrum` return valid files instead of aborting `biomass()`/`abundance()`. Reachable by the live-during-run streaming + interrupted runs.
- **Verify:** cancel a Java run → `status='cancelled'`; save scenarios `v1.2` and `v1` without cross-deletion; an empty CSV in the output dir doesn't abort the readers.

## PR-5 — Packaging / CI (prod unaffected; protects Docker + wheel installs)

- [ ] **Docker image ships broken** — missing `www/` so `app_ui`'s `ui.include_css(www/*.css)` crash-loops on startup (`Dockerfile`). **Fix:** add `COPY www/ www/` before the chown step.
- [ ] **Build-only docker CI gate can't detect runtime breakage** (`.github/workflows/ci.yml:83-88`). **Fix:** extend beyond `docker build` to `docker run` + poll the HEALTHCHECK + `curl -f /osmose.css`, so a non-functional image can't ship green.
- [ ] **`demo.py` silently degrades to stub configs in non-editable installs** (`osmose/demo.py:110-121`) — `Path(__file__).parent.parent / 'data'` only works under the editable dev install. **Fix:** load `data/` via `importlib.resources` (declare as package data) or resolve from `OSMOSE_DATA_DIR`; replace the silent `if data_dir.exists()` stub fallback with a clear log/raise so a missing bundle fails loudly.
- [ ] **Numba pure-Python fallbacks are coverage-omitted and never run in CI** (`pyproject.toml` coverage omit). **Fix:** a small CI leg / parametrized fixture running a representative engine smoke test with `_HAS_NUMBA` patched `False` (most clearly `movement.py`'s map-movement else-branch); narrow the coverage omit to `@njit` bodies via `# pragma: no cover`.
- **Verify:** `docker run` serves `/osmose/` 200 + styled CSS in CI; a wheel/sdist install produces real (not stub) demo configs; the numba-off smoke test passes. **Prod is unaffected** (systemd serves a source clone).

## P-MINOR — cleanup batch (one PR, low risk)

- [ ] `MultiPhaseCalibrator` forwards no fixed params (`multiphase.py:63-65,72-110`) — wire `fixed_params` into the objective OR delete the false docstring + dead plumbing (no production callers).
- [ ] Spatial temp/O2 negative-cell-index wrap for unlocated schools (`simulate.py:332,365-366`) — mask `cell_x>=0 & cell_y>=0`, substitute a domain-mean/15 °C fallback; add a non-constant-O2 branch or raise. (Dead until NetCDF spatial forcing wired.)
- [ ] Eval-cache key omits per-candidate seed (`problem.py:504-517`) — include `seed`/`run_id` in `_cache_key` or gate caching off for the Python engine (dormant; no caller sets `enable_cache=True`).
- [ ] Map rejection-sampling failure: Python path raises, numba path degrades to out-of-domain (`movement.py:92` vs `413-416`) — unify the policy (warn + out-of-domain in both, matching Java `School.out()`).
- [ ] `check_species_consistency` ignores `nbackground` and assumes contiguous resource indices (`validator.py:130-151`) — scan actual `species.name.spN` keys (as `background.py:154-159` does) + include `simulation.nbackground`.
- [ ] Larval migration `'null'` sentinel crash — covered in PR-3.
- [ ] Scenario/handler edge paths fail hard instead of skip-and-continue (`ui/pages/scenarios.py:366` load handler; `scenarios.py:204` `import_all` missing `name`; `list_scenarios` should filter `.bak`/tmp dirs).

---

## Suggested sequencing

1. **P0 rotation** (you) — the only live exposure.
2. **PR-1** (engine wire-or-reject) — highest correctness value; pure-core + tested.
3. **PR-4** (IO robustness) — data-loss fix (scenario backup) + cancel/empty-CSV; small, high-value.
4. **PR-3** (config/schema) — bundle with the jar-swap resume since it de-risks the cutover; otherwise do the `validate_value` fix standalone (it also unblocks PR-2's coercion cleanup).
5. **PR-2** (calibration failure policy) — independent.
6. **PR-5** (packaging/CI) — needed before any Docker/wheel distribution; not urgent for the systemd prod.
7. **P-MINOR** — opportunistic.

Quick wins (≤ ~1 file, high value): scenario-backup `with_suffix` fix (data loss), `validate_value` string coercion, the 4.4.0 read-side version helper swap, Docker `COPY www/`.

Verification gate for every PR: `.venv/bin/python -m pytest -q -m "not e2e and not visual" -n auto` (3457 baseline) + `ruff check`/`format --check` + `pyright` + `import app`.
