# Deep Review Remediation Plan

> Created: 2026-05-05
> Branch: `claude/deep-app-review-xvuga`
> Scope: All issues identified by the 7-agent deep review (UI/Shiny, schema,
> config integrity, engine parity, performance, science plausibility, test
> coverage)
>
> **Revision log:**
> - 2026-05-05 r1 — initial draft (commit `59be0a0`).
> - 2026-05-05 r2 — verified against master `cf5cb8e`. Corrections applied:
>   - **H2 retracted** (claimed schema count 154; actual is 223 via
>     `sum(len(g) for g in osmose.schema.ALL_FIELDS)`; CLAUDE.md's "221" is
>     within ±2). Replaced with a CLAUDE.md sync task only.
>   - **H1 rebuilt from ground truth** by running
>     `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs`
>     on master and capturing the actual unknown-key warnings per fixture
>     (eec/eec_full/minimal clean; baltic = 11 background-species keys;
>     examples = 43 ltl/rsc + species.conversion2tons keys). The original
>     speculative table was replaced with this empirical list.
>   - **C5 added (NEW)**: master ships RED on
>     `test_from_dict_warn_mode_clean_on_example_configs[baltic]` —
>     CLAUDE.md asserts this test "must stay warning-free", but it isn't.
>     Promoted from "schema gap" to a critical regression in its own row.
>   - **C3 narrowed** to the two remaining unguarded sites (lines 334 and
>     586). Lines 521 and 543 already do `is_relative_to(Path.cwd())`; line
>     228 is a non-privileged display-only path-check call. Original "5
>     bug sites" claim was overcounted.
>   - **M2 reasoning flipped.** `searchsorted(arr, v, side='right')` already
>     gives `count(arr ≤ v)` semantics — i.e., `value >= threshold`. The
>     original plan proposed `'left'` to fix `>=`-semantics; that's
>     backwards. Re-scoped as "verify against Java's `FishStage` source
>     before changing anything; if Java uses strict `>`, then `'left'`;
>     otherwise current code is correct".
>   - **C4 caveated** as user-facing-UI-only — calibration / headless users
>     never trigger the cancel button, so this is not a blocking-correctness
>     issue for the simulation engine.
>   - **Phase 4 baseline gate** added — perf claims must cite a baseline
>     benchmark run before merging.
>   - **Acceptance #1 made actionable** by naming the agents and dispatch
>     command instead of "all 7 reviewers re-run".
>   - **Out-of-scope cross-ref** to "M9" deduplicated; renamed to a
>     descriptive label.
> - 2026-05-05 r3 — second-pass review by claim-verification + structural
>   reviewers. Corrections applied:
>   - **C3** function-name labels fixed: line 586 is `download_results_csv`,
>     not `output_dir_status` (that's at line 227 and is out-of-scope).
>   - **C4** `RunResult` patch sketch corrected: r2 wrote
>     `RunResult(status="failed", message=...)` but the dataclass at
>     `osmose/runner.py:43` only has `returncode/output_dir/stdout/stderr`.
>     r3 explicitly adds an "extend `RunResult` with `status` + `message`"
>     step and shows correct constructor calls.
>   - **Acceptance #1** dispatch instructions reworded — agent
>     `subagent_type` strings are listed with a note to verify presence
>     in the session before dispatch (rather than presented as
>     guaranteed-executable code).
>   - **Sequencing notes** updated — Phase 2 acceptance now explicitly
>     depends on C5 (Phase 1) landing first.
>   - **Phase 1 estimate** revised from "1 day" to "1–2 days" to reflect
>     C5's addition.
>   - **Phase 5 estimate** revised from "1–2 days" to "2–3 days" with
>     5a/5b split option.
>   - **C3 acceptance** extended with symlink test cases.
>   - **M2 parity gate** tightened with numeric-baseline requirement.
>   - **H10 pre-change sweep** added to avoid red-master post-merge.
>   - **Schema field-quality acceptance** added (no acceptance criterion
>     in r1/r2).
>   - **Risk register** extended with 4 new rows: allowlist-typo masking,
>     symlink behaviour, legacy fixture bound failures, RunResult
>     constructor ripple.

This plan is organised as five execution phases, each independently shippable.
Each issue is given a **stable ID** (matches the deep-review report — `C` =
critical, `H` = high, `M` = medium, `L` = low) and is paired with a concrete
patch sketch, acceptance criteria, and test plan. Phases are ordered so that
the cheapest, highest-leverage fixes ship first.

Estimated total: ~5–7 engineer-days to land all phases. Phases 1–2 alone
remove the user-visible silent-failure modes and unblock strict schema
validation.

---

## Phase 1 — Critical correctness & security (target: 1–2 days)

Fixes silent-failure modes the user can hit today and the path-traversal
weakness in the results page.

> **Estimate revised in r3.** r1/r2 said "1 day". With C5 added and C4 still
> in scope, realistic budget is **1.5–2 engineer-days**: ~3 hours for C1
> (movement schema rewrite + parity test), ~30 min C2, ~2 h C3 (helper +
> 4 sites + symlink test), ~3 h C4 (RunResult extension + cancellation
> plumbing through simulate.py + new test), ~2 h C5 (schema or allowlist
> additions + test re-green). If C4 is deferred per its scope-note, drop to
> ~1 day.

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

### C3 — Path traversal in results page (narrowed in r2)

**Symptom.** `_load_results` and `download_results_csv` reject only the
`..` substring; absolute paths like `/etc` slip through.

**Verified scope (r3 — function names corrected after r2 review).**
- `results.py:521` (inside `comparison_chart` reactive starting near
  line 510) and `results.py:543` (inside `config_diff_table` reactive
  starting near line 540) **already** combine the `..` check with
  `out_dir.is_absolute() and not out_dir.is_relative_to(Path.cwd())` —
  no fix needed for these two.
- `results.py:334` (inside `_load_results` at line 332) — **`..` check
  only**, real bug.
- `results.py:586` (inside `download_results_csv` download handler at
  line ~582) — **`..` check only**, real bug. (r2 mistakenly named this
  function `output_dir_status`; that function is at line 227 and is
  out of scope.)
- `results.py:227` (`output_dir_status`) only does `p.is_dir()` and
  `glob` for on-screen feedback, no privileged file access. Out of scope.

**Files.**
- `ui/pages/results.py:334, 586` (apply helper)
- `ui/pages/results.py:521, 543` (refactor to call the same helper for
  consistency, no behaviour change)

**Plan.**
1. Extract the existing `is_relative_to`-based check (already present at
   lines 521/543) into a private helper in `ui/pages/results.py`:
   ```python
   def _safe_output_dir(raw: str) -> Path | None:
       try:
           p = Path(raw).resolve(strict=False)
       except OSError:
           return None
       cwd = Path.cwd().resolve()
       if p != cwd and not p.is_relative_to(cwd):
           return None
       if not p.is_dir():
           return None
       return p
   ```
2. Apply the helper at lines 334 and 586 — those are the actual unguarded
   sites.
3. Refactor 521 and 543 to call the same helper for consistency (no new
   behaviour, just dedup).
4. Audit `ui/pages/scenarios.py`, `forcing.py`, `advanced.py` for similar
   user-input-path patterns; apply the helper where they read sensitive
   files (skip display-only sites like `output_dir_status`).

**Acceptance.**
- New test `tests/test_results_page_path_safety.py`:
  - `/etc` → rejected
  - `/tmp/foo` (outside cwd) → rejected
  - `../../etc/passwd` → rejected
  - `output/run123` (inside cwd) → accepted
  - **Symlink case (added r3):** symlink at `output/symlink-out` pointing
    to a directory inside cwd → accepted; symlink pointing to `/etc` →
    rejected. (Calibration / scenario forks may legitimately symlink
    output dirs; the helper must not break that workflow while still
    catching escape-via-symlink.)

### C4 — Run race + lost engine errors + non-cancellable Python engine

> **Scope note (r2):** All three symptoms are **interactive-UI only**.
> Calibration and headless `python -m osmose.engine.simulate` users do
> not run through the Cancel button or `_handle_result`, so this is a UX
> regression rather than a simulation-correctness bug. Land if interactive
> users surface complaints; defer otherwise. Symptom (2) — silent
> auto-load of stale results — is the highest-impact piece and could be
> fixed standalone (state-invalidation only) without the cancellation
> plumbing.

**Symptom.**
1. Cancel button is a silent no-op when running the Python engine.
2. On engine raise, the partial output dir lingers and `state.run_result` /
   `state.output_dir` are not invalidated; auto-load fires on bad data.
3. Live config can be edited mid-run with no visual lock.

**Files.**
- `ui/pages/run.py:243-253, 354-365, 481-485`
- `ui/state.py` (add cancellation token)
- `osmose/runner.py` (extend `RunResult` with status + message)
- `osmose/engine/simulate.py` (accept a cooperative-cancellation callback)

**Plan.**
1. **Extend `RunResult`** at `osmose/runner.py:40-47` to carry a
   `status: Literal["ok", "failed", "cancelled"]` field and an optional
   `message: str = ""` field. Default `status="ok"` so existing callers
   that construct `RunResult(returncode=..., output_dir=..., stdout=...,
   stderr=...)` keep working. (r2 patch sketch tried to write
   `RunResult(status="failed", message=...)` directly; `RunResult` only
   has `returncode/output_dir/stdout/stderr`. r3 corrects this.)
2. Add a `threading.Event` to `AppState`: `state.run_cancel_token`. Reset
   on each run start.
3. Thread it through `PythonEngine.run(...)` and into the simulation loop.
   In `simulate.py`'s outer `for step in range(n_steps):` loop, add
   `if cancel_token is not None and cancel_token.is_set(): raise
   SimulationCancelled()`. Define `SimulationCancelled` in
   `osmose/engine/__init__.py`.
4. Wire `btn_cancel.click` → `state.run_cancel_token.set()` in `run.py`.
5. Wrap `_run_python_engine` in `try/except SimulationCancelled` and a broad
   `except Exception`; in both error branches:
   - On cancel: `state.run_result = RunResult(returncode=-1,
     output_dir=Path(""), stdout="", stderr="", status="cancelled",
     message="user cancelled")`.
   - On exception: `state.run_result = RunResult(returncode=1,
     output_dir=Path(""), stdout="", stderr=str(exc), status="failed",
     message=str(exc))`.
   - Set `state.output_dir = None`.
   - Bump `state.run_dirty` so dependent reactives re-fire.
6. Audit `_handle_result` and the `Results` page's `_auto_load_results` to
   short-circuit when `state.run_result.status != "ok"`.
7. Optional UX: disable form inputs while `state.busy != ""` via CSS
   `pointer-events: none` overlay (already partly there with `osm-disabled`).

**Acceptance.**
- Manual: hit Cancel mid-run on Python engine — sim aborts within ~1 step,
  `Run` page shows "Cancelled", `Results` page does not auto-load.
- Test: `tests/test_run_cancellation.py` — fake engine that runs for 100
  steps; trigger cancel at step 10; assert `RunResult.status == "cancelled"`
  and no auto-load occurs.
- Test: existing runner tests still pass without modification (default
  `status="ok"` keeps the old constructor calls working).

### C5 — Master is RED on `test_from_dict_warn_mode_clean_on_example_configs[baltic]` (NEW in r2)

**Symptom.** CLAUDE.md asserts:
> "Integration test: `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs[*]` must stay warning-free."

But on master `cf5cb8e` the `[baltic]` parametrization fails — 11 unknown-key warnings:

| Key | Source | Engine read site (current) |
|---|---|---|
| `osmose.configuration.background` | top-level switch | `engine/config.py` background-species loader |
| `species.nclass.sp{14,15}` | seal/cormorant | background-species block |
| `species.trophic.level.sp{14,15}` | seal/cormorant | background-species block |
| `species.length.sp{14,15}` | seal/cormorant | background-species block |
| `species.size.proportion.sp{14,15}` | seal/cormorant | background-species block |
| `species.age.sp{14,15}` | seal/cormorant | background-species block |

These were added when seal/cormorant background species were wired into
baltic (per memory: 2026-04-25 "Engine reproduction.py fix") without
extending the schema or `_SUPPLEMENTARY_ALLOWLIST`. CI either doesn't
run this test or has been red for several commits.

**Why this is C-priority** — we can't claim Phase 2 acceptance ("zero
unknown-key warnings on all five fixtures") if master is already red on
one fixture. Fix this *before* enumerating new schema gaps.

**Files.**
- `osmose/engine/config_validation.py:_SUPPLEMENTARY_ALLOWLIST` (or new
  `osmose/schema/background.py` with `OsmoseField` entries)
- `osmose/engine/config_validation.py:_INDEX_SUFFIXES` (verify `sp` covers
  background-species indices >= n_focal)

**Plan.**
1. Decide whether background species belong in the schema (preferred —
   makes the keys discoverable to UI form generators) or only in
   `_SUPPLEMENTARY_ALLOWLIST` (shorter path; matches existing pattern for
   reader-injected metadata). I recommend adding a
   `osmose/schema/background.py` with the 5 indexed fields plus
   `osmose.configuration.background` global, then re-running the test.
2. Run `.venv/bin/python -m pytest tests/test_engine_config_validation.py`
   and confirm `[baltic]` passes.
3. Add a CI gate so this can't regress silently again — already covered by
   the existing parametrize, but verify it actually runs in CI.

**Acceptance.**
- `tests/test_engine_config_validation.py` — 100% green on master with
  no skips on baltic.

---

## Phase 2 — Schema correctness & coverage (target: 1 day)

Eliminates silent UI-engine drift on lesser-used keys; unblocks strict
validation.

### H1 — Schema coverage gaps (rebuilt from ground truth in r2)

**Method.** Instead of speculating, ran each fixture through
`EngineConfig.from_dict(cfg)` with `validation.strict.enabled=warn` and
captured every "Unknown config key" warning. Script saved at
`/tmp/list_unknown_keys.py` for reproducibility.

**Per-fixture results on master `cf5cb8e`:**

| Fixture | Unknown-key warnings |
|---|---|
| `eec` | 0 (clean) |
| `eec_full` | 0 (clean) |
| `minimal` | 0 (clean) |
| `baltic` | 11 — covered by **C5** (above) |
| `examples` | 43 — covered by this item (H1) |

**`examples` — 43 warnings, three groups:**

1. `osmose.configuration.ltl` — top-level switch. 1 warning.
2. `species.conversion2tons.sp{8..13}` — 6 warnings. Engine reads at
   `osmose/engine/config.py` (existing `species.*` block); plain schema
   field add.
3. LTL resource block — `ltl.{name,tl,size.min,size.max,accessibility2fish,conversion2tons}.rsc{0..5}`
   — 36 warnings (6 fields × 6 indices).

**Plan.**
1. Add `rsc` to `_INDEX_SUFFIXES` in
   `osmose/engine/config_validation.py:25-32`:
   ```python
   ("rsc", re.compile(r"^rsc\d+$")),
   ```
2. Add to `_SUPPLEMENTARY_ALLOWLIST` (or a new `osmose/schema/ltl.py`
   extension):
   - `osmose.configuration.ltl`
   - `species.conversion2tons.sp{idx}`
   - `ltl.name.rsc{idx}`, `ltl.tl.rsc{idx}`, `ltl.size.min.rsc{idx}`,
     `ltl.size.max.rsc{idx}`, `ltl.accessibility2fish.rsc{idx}`,
     `ltl.conversion2tons.rsc{idx}`
3. Add `examples` and `minimal` to the parametrize list (also in **H4**).

**What dropped from r1.** The r1 table listed ~13 keys that the deep-review
agents *thought* were undeclared (`output.step0.include`,
`population.seeding.year.max`, the mortality/fishing/MPA spatial fields,
the economic block, etc.). Re-running validation shows those keys are
already in the allowlist — either via the schema or
`_SUPPLEMENTARY_ALLOWLIST` already added before this session. They are
**not** Phase 2 work.

**Caveat.** Validation only catches keys present in fixture CSVs. Keys
read by the engine but absent from any shipped fixture won't surface here.
Address those (if any are found) under **H2**.

**Acceptance.**
- All five `parametrize` entries (`eec`, `baltic`, `eec_full`,
  `examples`, `minimal`) pass with zero unknown-key warnings on master
  after C5 + H1 + H4 land.

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

### H2 — CLAUDE.md schema-param count sync (retracted "154", revised in r2)

**r1 retraction.** r1 claimed "actual is 154". Verified on master `cf5cb8e`:

```
$ .venv/bin/python -c "from osmose.schema import ALL_FIELDS; print(sum(len(g) for g in ALL_FIELDS))"
223
```

`osmose.schema` exposes `ALL_FIELDS` as a list of per-module groups
(`MOVEMENT_FIELDS`, `OUTPUT_FIELDS`, etc.); `len(ALL_FIELDS) == 10` is the
group count, not the field count. The r1 reviewer almost certainly
imported the wrong symbol (e.g., `len(ALL_FIELDS)` or a single-module
import).

**Reality.** CLAUDE.md says "221", actual is **223**. The discrepancy is
two fields, well within "documentation drift" rather than "documentation
bug". Demote H2 from a bug to a sync task.

**Plan.**
1. After Phase 2 schema additions land, recount with
   `sum(len(g) for g in osmose.schema.ALL_FIELDS)`.
2. Update CLAUDE.md's "Architecture" block to the new total.

**Acceptance.** CLAUDE.md count matches the live recount (±0).

### Schema field-quality fixes (mostly L/M warnings from review)

> **Acceptance (added r3).** After the per-field changes below land:
> 1. `tests/test_engine_config_validation.py` — full parametrize set still
>    green (no new unknown-key warnings introduced).
> 2. Round-trip via `scripts/check_config_roundtrip.py` on all five
>    fixtures — every changed field round-trips identically (no value
>    munging from the new defaults).
> 3. New defaults must match the engine-side defaults verified against
>    `osmose/engine/config.py` (`grep` for the same `cfg.get(key,
>    "DEFAULT")` and assert the schema default string-equals it).


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

> **Pre-change sweep (added r3).** Before tightening these bounds, run
> `grep -rE 'species\.(sexratio|relativefecundity)\.sp' data/*/`*.csv` and
> inspect every value across all five fixtures. If any fixture has
> `sexratio` outside `[0, 1]` or `relativefecundity ≤ 0`, soften the
> raise to a warning (with the offending fixture path) instead of
> failing master post-merge.

**Plan.**
1. **Run the sweep first.** If clean, proceed with hard bounds. If not,
   downgrade to warnings and document the legacy values.
2. In `config.py`, after loading `species.sexratio.spX` and
   `species.relativefecundity.spX`, validate:
   ```python
   if not 0.0 <= sex_ratio <= 1.0:
       raise ValueError(f"sex_ratio for sp{idx} must be in [0,1], got {sex_ratio}")
   if relative_fecundity <= 0:
       raise ValueError(f"relative_fecundity for sp{idx} must be > 0")
   ```
3. In `reproduction.py:48`, after loading `spawning_season[sp, :]`, assert
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

### Engine parity M2 — Feeding-stage boundary (rationale flipped in r2)

**Files.** `osmose/engine/processes/feeding_stage.py:73`.

**r2 correction.** The r1 plan said `side='right'` is wrong and `side='left'`
matches Java's `value >= threshold` semantics. **That is backwards.**

For an ascending threshold array `arr`:
- `np.searchsorted(arr, v, side='right')` returns `count(arr ≤ v)` — i.e.,
  the count of thresholds ≤ value, which IS the `value >= threshold`
  count. Example: `arr=[1,2,3], v=2 → 2` (thresholds 1 and 2 both satisfy
  `value >= threshold`).
- `np.searchsorted(arr, v, side='left')` returns `count(arr < v)` — i.e.,
  strict `>`. Example: same input → `1`.

The current code (`side='right'`) is consistent with the comment "Count
thresholds exceeded (>= comparison)" and matches `value >= threshold`
semantics.

**Verification still owed.** What's not verified is whether **Java's
actual `FishStage` implementation uses `>=` or `>`**. Two outcomes:

1. **Java uses `>=` (most likely)** — current Python code is correct, M2
   is a non-issue, drop it.
2. **Java uses strict `>`** — change to `side='left'` matches Java; the
   "comment says `>=`" is the bug.

**Plan.**
1. Read the Java `FishStage` source on the OSMOSE GitHub repo
   (`osmose-model/osmose`) to determine the actual comparison operator.
2. **If Java is `>=`**: close M2 as "verified; current code matches".
   Update the comment if anything is unclear.
3. **If Java is `>`**: switch to `side='left'`. Then capture the
   pre-change drift on `*_by_size` outputs vs Java reference NetCDF as a
   numeric baseline (`max_abs_diff`, `mean_abs_diff` per size bin); the
   change must show post-change drift ≤ pre-change drift on **every
   size-bin metric** before merging. Aggregate-biomass drift must remain
   inside the existing parity tolerance (no widening).
4. If parity drift increases either way (or is unchanged within
   noise), document and revert; boundary epsilon is biologically
   negligible relative to aggregate biomass.

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

All optional but high-leverage.

> **Baseline gate (added in r2).** The r1 plan claimed "expected aggregate:
> 10–20% wall-time reduction" without a baseline. Before merging any Phase
> 4 commit, capture a current-master benchmark (`scripts/optimizer_bench.py`
> or a single `simulate.run` over `data/eec` for 50 years × 24 dt × 3
> seeds) and report the per-change delta in the PR description. Drop any
> sub-task whose measured improvement is < 2 % (noise floor on a 28-core
> box).

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

## Phase 5 — Test coverage, UI cleanup, polish (target: 2–3 days, splittable)

> **Estimate revised in r3.** This phase contains 13 distinct items
> (H8/H9/M7/M8/M9 tests + L test fix + UI consolidation M10/M13 + H3/H12
> /H11/M12 + a 5-bullet "Misc Low/Nit" list). Even at 1 hour per item the
> total is realistically 2 engineer-days; UI consolidation alone (M10
> across every page) is half a day.
>
> If sequencing matters, consider splitting:
> - **Phase 5a (tests-only, ~1 day):** H8, H9, M7, M8, M9, L
> - **Phase 5b (UI/polish, ~1–2 days):** M10, M13, H3, H12, H11, M12,
>   misc nits

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

1. **Re-dispatch the deep-review agent suite on the resulting branch and
   find no remaining Critical or High issues.** From a fresh Claude Code
   session in this repo, dispatch via the Agent tool. Verify each
   `subagent_type` is present in the session's agent list before
   dispatching (the four below are present in this author's
   environment as of 2026-05-05; if any are missing, install the
   corresponding plugin or substitute an equivalent agent):
   - `feature-dev:code-reviewer` — bug / logic / convention adherence
   - `pr-review-toolkit:silent-failure-hunter` — error-handling / fallback
   - `pr-review-toolkit:type-design-analyzer` — type / dataclass review
   - `superpowers:code-reviewer` — plan-vs-implementation cross-check

   Each agent's prompt must include: branch under review
   (`claude/deep-app-review-xvuga` post-merge of all phases), severity
   filter (Critical and High only), and the report-format request used in
   this session's review iterations (numbered findings with file:line +
   evidence + severity).

   Iteration ends when all four agents return zero new C/H findings on a
   single round (note: a finding is "new" only if it points to code or
   docs touched by the remediation; pre-existing unrelated issues do not
   block this acceptance).
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

- **Phase 2 acceptance depends on C5 (Phase 1) landing first** — H1's
  "zero unknown-key warnings on all five fixtures" is unreachable while
  master's `[baltic]` parametrization is red. Other than that single
  cross-phase coupling, Phase 1 and Phase 2 are independent and can be
  parallelised by two engineers (engineer A: C1 + C2 + C3 + C4; engineer
  B: C5 + Phase 2).
- **Phase 3** depends on Phase 2 (some new schema fields need engine
  validation tightened simultaneously).
- **Phase 4** is independent and can ship anytime; defer if no
  performance complaints. Each Phase-4 sub-task gates on its own baseline
  benchmark (see "Baseline gate" at top of Phase 4).
- **Phase 5** depends on all earlier phases (tests for the new behaviour).
  If split into 5a / 5b, Phase 5a (tests) can land in parallel with
  Phase 4 once Phase 3 is in.

## Out of scope

- **Reactive-config refactor** — moving `state.config` from
  `dict[str, str]` to `dict[str, reactive.Value[str]]`. (Original review
  labelled this M9 internally — note that the review's M9 ID conflicts
  with the M9 listed in **Phase 5** above; in this plan, the reactive
  refactor is *not* numbered, only listed here as out-of-scope.) Big
  refactor; defer to a dedicated design doc.
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
| **C5/H1 allowlist additions silently mask real config typos** (added r3) | Prefer schema fields over `_SUPPLEMENTARY_ALLOWLIST` entries. For every key added to the allowlist, also add a positive engine-read assertion in `tests/test_engine_config.py` confirming the engine actually consumes it. If the engine doesn't read a key, it shouldn't be in the allowlist. |
| **`_safe_output_dir` (C3) breaks symlinked output dirs** (added r3) | Acceptance test must include the symlink case (symlink → inside-cwd directory accepted; symlink → `/etc` rejected). Calibration / scenario forks may legitimately symlink; the helper resolves with `Path.resolve(strict=False)` which follows symlinks — confirm that's the desired behaviour with the test. |
| **H10 reproduction bound fails on legacy fixture data** (added r3) | Run the pre-change sweep over `data/*/`*.csv` for `species.sexratio.*` and `species.relativefecundity.*`; if any value is outside the new bound, soften the raise to a warning (with offending fixture path) before merge. Hard-failing master because of legacy data is worse than the bug. |
| **Phase 1 sequencing — extending `RunResult` (C4) ripples through callers** (added r3) | Default the new `status` field to `"ok"` and `message` to `""` so existing `RunResult(returncode=..., output_dir=..., stdout=..., stderr=...)` constructors keep working. Audit `osmose/runner.py:182,192` and any test fakes; run the full test suite before pushing. |
