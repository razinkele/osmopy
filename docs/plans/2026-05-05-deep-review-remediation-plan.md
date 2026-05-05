# Deep Review Remediation Plan

> Created: 2026-05-05
> Branch: `claude/deep-app-review-xvuga`
> Scope: All issues identified by the 7-agent deep review (UI/Shiny, schema,
> config integrity, engine parity, performance, science plausibility, test
> coverage)

This plan is organised as five execution phases, each independently shippable.
Each issue is given a **stable ID** (matches the deep-review report — `C` =
critical, `H` = high, `M` = medium, `L` = low) and is paired with a concrete
patch sketch, acceptance criteria, and test plan. Phases are ordered so that
the cheapest, highest-leverage fixes ship first.

Estimated total: ~5–7 engineer-days to land all phases. Phases 1–2 alone
remove the user-visible silent-failure modes and unblock strict schema
validation.

---

## Phase 1 — Critical correctness & security (target: 1 day)

Fixes silent-failure modes the user can hit today and the path-traversal
weakness in the results page.

### C1 — Movement schema keys are inverted vs engine

**Symptom.** UI generates `movement.map{idx}.species` but engine reads
`movement.species.mapN`. UI writes are silently ignored.

**Files.**
- `osmose/schema/movement.py:35-84` (rewrite key patterns)
- `osmose/engine/movement_maps.py:129, 149, 153` (canonical reader)
- `tests/test_schema_all.py` (add: every schema key must appear in the engine
  config-validation allowlist OR `_SUPPLEMENTARY_ALLOWLIST`)

**Plan.**
1. Inspect `movement_maps.py` and enumerate the actual keys the engine reads:
   `movement.species.mapN`, `movement.file.mapN`, `movement.season.mapN`,
   `movement.initialage.mapN`, `movement.lastage.mapN`,
   `movement.year.min.mapN`, `movement.year.max.mapN`.
2. Rewrite the `OsmoseField` entries in `osmose/schema/movement.py` to use the
   correct `key_pattern` form. The placeholder is `{idx}` after the property
   token, e.g. `movement.species.map{idx}`.
3. Update `_INDEX_SUFFIXES` in `osmose/engine/config_validation.py` if `mapN`
   is not already covered (it should be — verify).
4. Migration check: scan `data/eec`, `data/baltic`, `data/eec_full`,
   `data/examples` for any current uses of the new pattern; round-trip them
   through reader/writer to confirm no breakage.

**Acceptance.**
- Round-trip script (`scripts/check_config_roundtrip.py`) passes on all five
  fixtures with zero unknown-key warnings for `movement.*`.
- New test: `tests/test_schema_engine_key_parity.py` — for every
  `OsmoseField`, resolve `key_pattern` with `idx=0` and assert the engine
  validation allowlist accepts it.

### C2 — `output.bioen.sizeInf.enabled` casing bug

**Symptom.** UI toggle for "size at infinity" bioen output is a no-op.

**Files.**
- `osmose/schema/output.py:166`

**Plan.**
1. Change `key_pattern="output.bioen.sizeInf.enabled"` → `output.bioen.sizeinf.enabled`.
2. Audit the rest of `_OUTPUT_ENABLE_FLAGS` for any other camelCase strays
   (`grep -E "[A-Z]" osmose/schema/output.py`).
3. Verify against `osmose/engine/config.py:820` and the surrounding
   `output.bioen.*` reader block.

**Acceptance.**
- Setting the toggle in the UI produces a NetCDF/CSV with the size-at-infinity
  output column populated.
- Test in `tests/test_schema_all.py`: assert all `OsmoseField.key_pattern`
  values match `^[a-z0-9._{}]+$` (lowercase only, no camelCase).

### C3 — Path traversal in results page

**Symptom.** `_load_results`, `download_results_csv`, `output_dir_status`
only check for the `..` substring; absolute paths like `/etc` slip through.

**Files.**
- `ui/pages/results.py:228, 333, 520, 542, 585`

**Plan.**
1. Extract the `comparison_chart` (line 521) validation into a single
   private helper `_safe_output_dir(raw: str) -> Path | None` in
   `ui/pages/results.py`:
   ```python
   def _safe_output_dir(raw: str) -> Path | None:
       p = Path(raw).resolve()
       cwd = Path.cwd().resolve()
       if not p.is_relative_to(cwd):
           return None
       if not p.is_dir():
           return None
       return p
   ```
2. Replace every ad-hoc `..`-substring check with a call to this helper.
3. Use `Path.resolve(strict=False)` then `is_relative_to(cwd)` — covers both
   absolute paths and traversal attempts.
4. Audit `ui/pages/scenarios.py` and any other page that takes a path input
   (forcing.py, advanced.py); apply the same helper if needed.

**Acceptance.**
- New test `tests/test_results_page_path_safety.py`:
  - `/etc` → rejected
  - `/tmp/foo` (outside cwd) → rejected
  - `../../etc/passwd` → rejected
  - `output/run123` (inside cwd) → accepted

### C4 — Run race + lost engine errors + non-cancellable Python engine

**Symptom.**
1. Cancel button is a silent no-op when running the Python engine.
2. On engine raise, the partial output dir lingers and `state.run_result` /
   `state.output_dir` are not invalidated; auto-load fires on bad data.
3. Live config can be edited mid-run with no visual lock.

**Files.**
- `ui/pages/run.py:243-253, 354-365, 481-485`
- `ui/state.py` (add cancellation token)
- `osmose/engine/simulate.py` (accept a cooperative-cancellation callback)

**Plan.**
1. Add a `threading.Event` to `AppState`: `state.run_cancel_token`. Reset
   on each run start.
2. Thread it through `PythonEngine.run(...)` and into the simulation loop.
   In `simulate.py`'s outer `for step in range(n_steps):` loop, add
   `if cancel_token is not None and cancel_token.is_set(): raise
   SimulationCancelled()`. Define `SimulationCancelled` in
   `osmose/engine/__init__.py`.
3. Wire `btn_cancel.click` → `state.run_cancel_token.set()` in `run.py`.
4. Wrap `_run_python_engine` in `try/except SimulationCancelled` and a broad
   `except Exception`; in both error branches:
   - Set `state.run_result = RunResult(status="failed", message=...)`.
   - Set `state.output_dir = None`.
   - Bump `state.run_dirty` so dependent reactives re-fire.
5. Audit `_handle_result` and the `Results` page's `_auto_load_results` to
   short-circuit when `state.run_result.status != "ok"`.
6. Optional UX: disable form inputs while `state.busy != ""` via CSS
   `pointer-events: none` overlay (already partly there with `osm-disabled`).

**Acceptance.**
- Manual: hit Cancel mid-run on Python engine — sim aborts within ~1 step,
  `Run` page shows "Cancelled", `Results` page does not auto-load.
- Test: `tests/test_run_cancellation.py` — fake engine that runs for 100
  steps; trigger cancel at step 10; assert `RunResult.status == "cancelled"`
  and no auto-load occurs.

---

## Phase 2 — Schema correctness & coverage (target: 1 day)

Eliminates silent UI-engine drift on lesser-used keys; unblocks strict
validation.

### H1 — Schema coverage gaps (~12+ undeclared engine-read keys)

**Plan.** For each missing key, add an `OsmoseField` in the appropriate
schema module. Default values must match `config.py` defaults (verify via
`grep`).

| Key | Schema module | Engine read site |
|---|---|---|
| `output.step0.include` | `output.py` | `config.py:774` |
| `population.seeding.year.max` | `simulation.py` (or new `population.py`) | `config.py:480` |
| `reproduction.normalisation.enabled` | `reproduction.py` (new file) | `config.py:883` |
| `mortality.fishing.spatial.distrib.file.sp{idx}` | `fishing.py` | `config.py:1427` |
| `mortality.additional.spatial.distrib.file.sp{idx}` | `fishing.py` | `config.py:963` |
| `predation.predprey.stage.structure.sp{idx}` (per-species) | `predation.py:35` (extend) | `config.py:530` |
| `fisheries.selectivity.a50.fsh{idx}`, `fisheries.selectivity.slope.fsh{idx}` | `fishing.py` | `config.py:291,300` |
| `fisheries.movement.file.map0`, `fisheries.seasonality.fsh{idx}` | `fishing.py` | `config.py:377,1419` |
| `movement.map.strict.coverage` | `movement.py` | `config.py:1634` |
| `mpa.percentage.mpa{idx}` | `fishing.py:101-122` (extend) | `config.py:837` |
| Economic block: `simulation.economic.enabled`, `simulation.economic.memory.decay`, `simulation.economic.rationality`, `economic.fleet.number`, `economic.fleet.*` | `economics.py` | (multiple) |

**Acceptance.**
- `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs`
  passes with zero unknown-key warnings on all five fixtures (`eec`, `baltic`,
  `eec_full`, `examples`, `minimal`).

### H4 — Validation test does not exercise `examples`

**Plan.**
1. Add `"examples"` to the `parametrize` list at
   `tests/test_engine_config_validation.py:206-220`.
2. Add the `rsc` index suffix to `_INDEX_SUFFIXES` in
   `osmose/engine/config_validation.py:25-32`:
   ```python
   ("rsc", re.compile(r"^rsc\d+$")),
   ```
3. Add the missing keys exposed by step 1 to `_SUPPLEMENTARY_ALLOWLIST`:
   - `osmose.configuration.ltl`
   - `ltl.*.rsc{idx}` (6 patterns)
   - `species.conversion2tons.sp{idx}`

### H2 — CLAUDE.md claims 221 schema params; actual is 154

**Plan.**
1. After Phase 2's schema additions, recount fields:
   `python -c "from osmose.schema import REGISTRY; print(len(REGISTRY))"`.
2. Update `CLAUDE.md` under "Architecture" with the correct count.

### Schema field-quality fixes (mostly L/M warnings from review)

**Files.** `osmose/schema/movement.py:8-15`, `output.py:88-93`,
`simulation.py:79-85, 150-156`, `predation.py:6-11`, `bioenergetics.py:113-247`,
`ltl.py:30-44`, `fishing.py:53-60`.

**Plan.**
- `movement.distribution.method.sp{idx}`: change default from `"maps"` to
  `"random"` (engine default).
- `output.distrib.bysize.incr`: `default=10.0` not `10`.
- `simulation.restart.file`: default `""`, mark `required=False`.
- `validation.strict.enabled`: shorten description to ≤120 chars.
- `predation.accessibility.file`: `required=False`.
- Bioenergetics fields: add `min_val=0`, units (`g`, `cm`, `°C`, etc.) where
  obvious.
- `ltl.species.size.min/max.sp{idx}`: add `min_val=0`, sensible defaults.
- `fisheries.selectivity.type.fsh{idx}`: keep ENUM but use semantic labels
  (`"sigmoid"`, `"gaussian"`, `"lognormal"`, `"knife-edge"`) mapped to ints
  via a writer helper.

---

## Phase 3 — Engine correctness, science bounds, parity drift (target: 1 day)

Tightens parameter validation and closes the second-order parity drifts.

### H10 / Science — Reproduction parameter bounds

**Files.** `osmose/engine/config.py:469-470`,
`osmose/engine/processes/reproduction.py:48`.

**Plan.**
1. In `config.py`, after loading `species.sexratio.spX` and
   `species.relativefecundity.spX`, validate:
   ```python
   if not 0.0 <= sex_ratio <= 1.0:
       raise ValueError(f"sex_ratio for sp{idx} must be in [0,1], got {sex_ratio}")
   if relative_fecundity <= 0:
       raise ValueError(f"relative_fecundity for sp{idx} must be > 0")
   ```
2. In `reproduction.py:48`, after loading `spawning_season[sp, :]`, assert
   `np.isclose(season.sum(), 1.0, atol=0.01)` and emit a warning (not raise)
   if violated — many configs may have legacy non-normalised vectors.

### Science M4 — Gompertz `linf` zero-default

**Files.** `osmose/engine/config.py:1655` plus the positivity guard at
`config.py:1338-1345`.

**Plan.** Extend the positivity guard to also check, for each species using
Gompertz growth, that `gompertz.linf > 0` and `gompertz.k > 0`.

### Science M5 — Unbounded accessibility / spatial multipliers

**Files.** `osmose/engine/processes/predation.py:213-214`,
`osmose/engine/processes/natural.py:84-89`.

**Plan.**
1. After loading the accessibility matrix in `config.py`, warn (don't fail —
   biological validity is a curve fit) if any entry > 1.0:
   ```python
   if (acc_matrix > 1.0).any():
       warnings.warn("predation accessibility coefficients > 1.0; biomass conservation may be violated")
   ```
2. In `natural.py:84-89`, clamp `n_dead = min(n_dead, abundance)` to prevent
   negative-abundance sentinel values when `spatial_factor > 1`.

### Engine parity M2 — Feeding-stage boundary

**Files.** `osmose/engine/processes/feeding_stage.py:73`.

**Plan.**
1. Change `np.searchsorted(thresholds, value, side="right")` to
   `side="left"` to match Java's `value >= threshold` semantics.
2. Re-run parity tests (`tests/test_parity_*`) with relaxed atol on
   `*_by_size` outputs; confirm no aggregate biomass drift.
3. If any of the 22 parity tests regress beyond tolerance, document and
   revert (boundary epsilon is biologically negligible; aggregate parity
   matters more).

### Engine parity M1 — Distribution averaging

**Files.** `osmose/engine/simulate.py:1000-1003`.

**Plan.**
1. In the distribution-output collector, change the "use last step in window"
   path to a running mean: maintain a `cumulative` array and `count`, divide
   at write time.
2. This affects `*_by_age`, `*_by_size` outputs only when
   `output.recordfrequency.ndt > 1`.
3. Update `tests/test_engine_outputs.py` to cover `ndt = 4` averaging.

### Engine parity M3 — Size-bin off-by-one

**Files.** `osmose/engine/simulate.py:740-746`.

**Plan.**
1. Compute `n_bins = int((max - min) / incr)` (floor) to match Java
   `SizeOutput`; verify no terminal-bin extra.
2. Test against EEC NetCDF reference output bin count.

### Engine parity — RNG reproducibility documentation

**Files.** `osmose/engine/rng.py` (top-of-file docstring), `CLAUDE.md`.

**Plan.** Document that `fixed=True` gives Python-side reproducibility only;
PCG64 ≠ Java MT19937, so byte-equivalent cross-engine outputs are impossible.

---

## Phase 4 — Performance (target: 1 day, optional)

All optional but high-leverage. Expected aggregate: 10–20% wall-time
reduction on Bay of Biscay / EEC.

### H5 — Hoist scratch buffers in `_apply_predation_numba`

**Files.** `osmose/engine/processes/mortality.py:846-848`.

**Plan.**
1. Pre-allocate `prey_type`, `prey_id`, `prey_eligible` arrays sized
   `max_n_local + n_resources` once in `_mortality_all_cells_numba` (caller).
2. Pass them as additional args into `_apply_predation_numba`. In the
   parallel kernel, allocate one set per `prange` iteration (acceptable
   since prange threads work on disjoint cells).
3. Benchmark: `pytest tests/test_engine_predation.py --benchmark-only`
   before and after; expect 5–15% improvement.

### H6 — Per-step full-state copies

**Files.** `osmose/engine/processes/mortality.py:1700-1957`.

**Plan.**
1. Add `_mort_scratch` dict on `ctx` populated lazily on first call:
   `n_dead_scratch`, `pred_success_rate_scratch`, `preyed_biomass_scratch`,
   `abundance_scratch`, `trophic_level_scratch` — all `np.empty(n_schools)`,
   resized only when `n_schools` grows.
2. Replace `state.n_dead.copy()` → `np.copyto(scratch, state.n_dead);
   scratch_view = scratch[:n]`.
3. Verify no scratch buffers are aliased between concurrent kernel calls
   (mortality is single-threaded at the Python level).

### H7 — Vectorise `biomass_by_cell` and fleet revenue

**Files.** `osmose/engine/simulate.py:1187-1194, 1231-1254`.

**Plan.**
1. `biomass_by_cell`: replace the Python loop with
   `np.add.at(biomass_by_cell, (sp, cy, cx), state.biomass)` after masking
   valid `(sp, cy, cx)`.
2. Fleet revenue: pre-bucket vessels by `(fleet, cy, cx)` once per step
   into a dict-of-arrays; iterate the bucket index per school instead of
   building three boolean masks per (school × fleet).

### M14 — Per-cell `cause_orders` allocation + 4-cause shuffle

**Files.** `osmose/engine/processes/mortality.py:1220-1226, 1385-1391`.

**Plan.**
1. Hoist `cause_orders = np.empty((max_n_local, 4), dtype=np.int64)` to a
   single ctx-level scratch buffer.
2. Replace `np.random.shuffle(causes)` with an inlined Fisher-Yates of 4
   ints (3 random integer draws, 3 conditional swaps).

### M11 — Memoise `PythonEngine` construction

**Files.** `ui/state.py`, `ui/pages/run.py:253`.

**Plan.** Lazy attribute on `AppState`: `state.python_engine` constructed
once, reused across runs. Avoids paying the Numba JIT cost per click.

---

## Phase 5 — Test coverage, UI cleanup, polish (target: 1–2 days)

### H8 — NaN/Inf propagation suite

**Files.** New `tests/test_numerical_propagation.py`.

**Plan.** For each of `predation`, `mortality`, `fishing`, `reproduction`,
`starvation`, `accessibility`: build a minimal `EngineConfig` + `State`,
inject a NaN into one input array, run one timestep, assert either:
- the NaN is caught and clamped (preferred), OR
- a clear `ValueError` is raised at validation time.

### H9 — Parallel-vs-sequential JIT parity

**Files.** New `tests/test_jit_determinism.py`.

**Plan.**
1. Set `NUMBA_NUM_THREADS=1` and run the standard mortality kernel.
2. Set `NUMBA_NUM_THREADS=4` and re-run with the same seeds.
3. Assert `np.allclose(state_seq, state_par, atol=1e-12)` on all output
   arrays. Catches any latent race in `prange` blocks.

### M7 — Lifespan-boundary cohort removal test

**Files.** New `tests/test_engine_aging_boundary.py`.

**Plan.** Build a cohort exactly at `age_dt = lifespan_dt - 1`, advance one
step, assert school is removed (abundance set to 0 / school compacted out).

### M8 — Runner failure modes

**Files.** Extend `tests/test_runner.py`.

**Plan.** Add four tests using `_ScriptRunner` with mock scripts:
1. Script writes 100 lines of CSV then segfaults — runner reports `failed`,
   not `ok`.
2. Script writes only to stderr — runner captures stderr, returns failure.
3. Ensemble of N=4 with replicate 2 failing mid-flight — runner aggregates
   correctly: 3 successes + 1 failure.
4. Cancel + verify no zombie children via `psutil.Process().children()`.

### M9 — Broaden MCP credential test

**Files.** `tests/test_copernicus_mcp_env.py`,
`tests/test_mcp_config_hygiene.py`.

**Plan.**
1. Generalise the literal-string scan to a regex-based credential sniffer:
   - High-entropy strings >20 chars in `mcp_servers/**/*.py` (excluding
     known-safe constants list).
   - Common credential names: `password`, `secret`, `token`, `api_key`,
     `credential` followed by `=` and a string literal.
2. Apply to all `mcp_servers/**/*.py`, not just copernicus.

### L — `test_path_escape_blocked` brittleness

**Files.** `tests/test_config_reader_errors.py:63`.

**Plan.** Replace `assert "root" not in str(result)` with
`assert "etc/passwd" not in str(result)`.

### UI consolidation (M10, M11, M13)

**Plan.**
- **M10**: Remove `state.loading`. Standardise on `state.busy: str`
  (empty string = idle). Audit every page for usage and migrate.
- **M13**: Add `param_form.input_id_for_field(field, idx) -> str` helper.
  Use it from both `render_field` and `render_species_table`. Update
  `sync_inputs` to call the same helper.

### H3 — `_inject_random_movement_ncell` mutation

**Files.** `ui/pages/run.py:77-106`.

**Plan.** Refactor to return a *new* dict rather than mutate. Caller does
`config = _inject_random_movement_ncell(config)`. This eliminates the
"reactive value silently mutates" surprise.

### H12 — `;`-array UI feedback

**Files.** `ui/components/param_form.py:96-124`, `ui/state.py:140-148`.

**Plan.**
1. Detect `";" in default_value`; if so, render the field as a *disabled*
   input with a tooltip "Multi-value field — edit via the Advanced tab".
2. Add a banner on the Advanced tab listing all multi-value fields with
   inline editing support.
3. Document the limitation in `CLAUDE.md` under "Gotchas".

### H11 — Eager imports in `app.py`

**Files.** `app.py:5-39`.

**Plan.**
1. Move `cleanup_old_temp_dirs()` from module top-level into the `server`
   function (runs once per session, not per import).
2. Lazy-import per-page modules under `server()` if startup time is a
   concern (~50–100 ms savings).

### M12 — `config_header` scans full dict per keystroke

**Files.** `app.py:483`.

**Plan.** Cache the dict-length count via a `@reactive.calc`-isolated
helper that depends only on `state.config_dirty` (a counter bumped by
`sync_inputs` once per debounce cycle), not on the dict contents.

### Misc Low/Nit cleanup

- `ui/charts.py` (18 lines) → fold into `ui/theme.py`.
- `genetics.py`/`economic.py`/`diagnostics.py` placeholder pages → factor
  into a single `placeholder_page(name, message)` helper.
- Add type hints on `*_server(state, input, output, session)` functions.
- Audit lines >100 chars: `ruff check --select E501 ui/ app.py` after
  `ruff format`.
- `state.py:34`: change `Path("data/scenarios")` → `Path(__file__).parent.parent / "data" / "scenarios"`.

---

## Acceptance / Done definition

The plan is fully landed when:

1. **All 7 reviewers re-run on the resulting branch find no Critical or
   High issues** (re-dispatch the same agents from this session).
2. **`.venv/bin/python -m pytest`** passes 100% with the new tests.
3. **Round-trip script** (`scripts/check_config_roundtrip.py`) passes on
   all five fixtures (`eec`, `baltic`, `eec_full`, `examples`, `minimal`)
   with zero unknown-key warnings.
4. **Java parity tests** (14/14 EEC, 8/8 BoB) still pass at original
   tolerances after Phase 3 changes (or tolerances widened by ≤2× with
   documented justification for `*_by_size` outputs).
5. **Manual UI smoke**: cancel a Python-engine run mid-flight; verify
   results page does not auto-load stale data; verify schema toggles for
   `output.bioen.sizeinf.enabled` and movement maps reach the engine.
6. **`CLAUDE.md` updated** with: corrected schema field count, RNG
   reproducibility note, multi-value-field gotcha.

---

## Sequencing notes

- **Phase 1 + Phase 2 are independent** and can be done in either order
  (or in parallel by two engineers).
- **Phase 3** depends on Phase 2 (some new schema fields need engine
  validation tightened simultaneously).
- **Phase 4** is independent and can ship anytime; defer if no
  performance complaints.
- **Phase 5** depends on all earlier phases (tests for the new behaviour).

## Out of scope

- Architectural rework of `state.config` from `dict[str, str]` to
  `dict[str, reactive.Value[str]]` (M9 in original review). Big refactor;
  defer to a dedicated design doc.
- Replacing `loading_overlay` with a richer per-field disable mechanism.
- Internationalisation of help text.
- Adding non-Java-OSMOSE features (genetics is already partly there;
  DSVM economics already partly there).

## Risk register

| Risk | Mitigation |
|---|---|
| Phase 3 feeding-stage boundary fix breaks parity tolerances | Land behind a config flag `engine.feeding_stage.boundary = "java" \| "numpy"`, default `"java"` after parity tests confirm; revert if regressions |
| Schema additions in Phase 2 break existing user configs that omit the keys | All new fields ship with `required=False` and engine-matching defaults; round-trip test on bundled fixtures gates the merge |
| Numba parallel determinism test (H9) flakes on different thread counts | Run with `NUMBA_NUM_THREADS` pinned in CI; mark test as `@pytest.mark.xfail` if Numba RNG semantics differ between thread counts (would be a separate engine bug) |
| Cancellation path (C4) leaves orphaned NetCDF/CSV file handles | Wrap output writers in `try/finally` close; verify with `lsof` in the test |
