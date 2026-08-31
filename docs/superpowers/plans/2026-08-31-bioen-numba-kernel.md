# Bioen-Aware Numba Mortality Kernel — Implementation Plan (rev. 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the batched Numba mortality kernels the five bioenergetics behaviours that currently exist only in the pure-Python per-cell path, and re-enable Numba dispatch under bioen, so bioen runs cost minutes instead of tens of hours.

**Architecture:** The pure-Python per-cell path (`_apply_*_for_school`, `_mortality_in_cell` with Numba forced off) stays exactly as it is and is the reference — it is what was reviewed against Java 4.3.3. This plan adds a bioen branch to the Numba entry points and pins them to that reference with deterministic, bit-exact, per-cell equivalence tests that run against **the kernels production actually executes**.

**Tech Stack:** Python 3.12, NumPy, Numba (`njit`, `prange`), pytest. Everything runs with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-30-baltic-c3-bioen-stage1-design.md` — this work reverses that spec's decision 14 (a bioen Numba kernel was declared a non-goal). Parent plan: `docs/superpowers/plans/2026-08-30-baltic-c3-bioen-stage1.md`; **its Task 4b is superseded by this document and must be struck in Task 0 below.**

**Revision note (rev. 2, 2026-08-31):** rewritten after a 29-agent adversarial review. Six things the first draft got wrong are corrected here and called out where they bite: the reference path dispatches *into* the kernel unless Numba is forced off (so the draft's harness compared the kernel with itself); `_pre_generate_cell_rng` emits four permutations while the reference draws five; the survivor rescale was specified for two of five death sites; `n_subdt` was missing from the starvation spec; `_apply_predation_numba` has 41 positional callers; and the tests that pin today's behaviour assert the *opposite* of what this plan does.

## Why this exists

| measurement (this machine, warm cache) | result | status |
|---|---|---|
| production Baltic, bioen OFF, Numba ON, 4 yr | 3.9 s (0.99 s/yr) | measured |
| `baltic_ev`, bioen ON, pure-Python path, 4 yr, ~12 000 schools | 442 s (110 s/yr) | measured |
| same-population ratio | **152×** | measured, two different configs |
| production Baltic, bioen OFF, Numba **disabled**, 4 yr | >13 min before I stopped it | **one-sided bound**, not a measurement |
| school growth, bioen ON vs OFF | identical, ~3 000/yr | measured — **not** a bioen defect |
| single 50-yr bioen run | ~19 h | **extrapolation** from quadratic fit (cost ∝ schools ∝ years) |
| parent plan's Task 12 A/B, 10 bioen runs, serial | ~190 h | extrapolation |
| same A/B with arms in parallel processes (1 thread each, 28 cores) | ~19–24 h | extrapolation; **this is the fallback, see below** |

Two honest qualifications the review forced. First, the Numba-disabled row is a lower bound from a
killed run on a *different* config, so "152×" is the cross-config ratio, not a controlled
measurement — the conclusion it supports (the gap is JIT-versus-interpreter, not bioen-specific)
survives, but the number should not be quoted as precise. Second, **the A/B is not strictly
infeasible without this work**: the runs are independent and the repo already has a spawn-pool
pattern for exactly this (`[[uq-parallel-threading]]`: one thread per worker). Running the ten
bioen runs as parallel processes brings the A/B to roughly one day. This plan is therefore a
large, permanent optimisation — it also makes Task 13's smoke test, Stage 2, and every future
bioen run practical — rather than the only way forward. If its gate cannot be met, the fallback is
parallel arms at full 5 seeds × 50 yr, **not** a reduced design (ledger ruling R7 keeps the
5-seed/50-yr shape; a 30-year horizon would also break the final-decade convention every Baltic
result in this repo uses).

## THE CONSTRAINT THAT SHAPES THIS PLAN

**The Numba and Python paths do not share an RNG stream.** `_mortality_all_cells_numba`
(`mortality.py:1332`) calls `np.random.seed(rng_seed)` and uses Numba's MT19937 inline; the Python
path consumes a caller-supplied PCG64 `Generator`. Whole-run outputs are therefore **not**
comparable bit-for-bit, and no step may demand it.

**And the reference dispatches into the kernel.** `_mortality_in_cell` computes
`use_full_numba = _HAS_NUMBA and inst_abd is not None and rsc_size_min is not None and eff_starv
is not None and not config.bioen_enabled` (`mortality.py:1745-1751`) and, when true, calls
`_mortality_in_cell_numba`. Once Task 4 removes the `bioen` term, a harness that just calls
`_mortality_in_cell` twice compares the kernel with itself and passes vacuously. **Every
"Python arm" in this plan must set `M._HAS_NUMBA = False` for the duration of the call.** That is
also what makes the harness simple: with `_HAS_NUMBA` toggled, calling the *same* function twice
on deep copies with a freshly seeded `np.random.default_rng(seed)` feeds both arms identical
`seq_*` and cause-order draws, with no RNG replay machinery at all.

**Do not build on `_pre_generate_cell_rng`.** It emits four permutations and a `(total, 4)` cause
buffer over `[0,1,2,3]`; the reference draws **five** (`seq_pred`, `seq_starv`, `seq_fish`,
`seq_nat`, `seq_for` — `mortality.py:1735-1739`) and under bioen needs a five-cause order. The
first draft called it "the tool for this"; it is not. Either extend it (drawing five
unconditionally and taking its cause list from `_get_mortality_causes`) or bypass it.

## Global Constraints

- `.venv/bin/python -m pytest`; `ruff check` + `ruff format`; line length 100.
- Branch `c3-bioen-stage1`. Per-path `git add` only — **never `git add -A`**: the tree carries the
  user's unrelated uncommitted work (`osmose/runner.py`, `osmose/cli.py`,
  `osmose/engine/movement_maps.py`, `.mcp.json`, `mcp_servers/copernicus/server.py`,
  `tests/test_runner.py`, `tests/test_engine_map_movement.py`,
  `tests/test_hpc_container_touchups.py`).
- Commit trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01KSP4ExqHQmyMWf8KsfmZU1`
- **`tests/test_engine_parity.py` (17) must pass after every task** — it pins the bioen-OFF Numba
  path against committed fixed-seed baselines.
- **The Python reference implementation is not to be modified** — `_apply_*_for_school`, `_kill`,
  `_consume`, and `_mortality_in_cell`'s Python branch. The one permitted exception is Task 4's
  dispatch flip, which touches the *gates* at `mortality.py:2101` and `:1750`, not the arithmetic.
  If a task appears to need any other change there, stop and report.
- Never two engine jobs at once. Do not run the full suite (`test_run_fie_demo_short_smoke`
  currently exceeds 38 min); deselect it explicitly if running anything broad.

## The five behaviours to port

| # | Behaviour | Python reference | Java |
|---|---|---|---|
| 1 | Per-fish cap: `max_eatable = cap_fish[p] * inst_abd[p]` | `_apply_predation_for_school` | `BioenPredationMortality.java:140-145` |
| 2 | Survivor rescale of `preyed_biomass` AND `e_net` **at every death, all five causes** | `_consume`/`_kill`, `mortality.py:100-131` | `School.incrementNdead` — no switch on cause |
| 3 | Bioen starvation in the interleaved loop, previous step's `e_net`, gonad buffer | `_apply_starvation_for_school` + `bioen_starvation_substep` | `BioenStarvationMortality`, `Species.java:224-226` |
| 4 | Five causes in the shuffle (FORAGING) with its own `seq_for` | `_get_mortality_causes`, `mortality.py:1735-1739` | `MortalityProcess.java:506-517` |
| 5 | Trophic level divides by the RAW preyed total; the budget uses the rescaled one | `raw_preyed` | `AbstractSchool.preyedBiomass` vs `School.ingestion` |

**Behaviour 2 is the one the first draft got wrong and is the easiest to under-implement.** The
rescale is cause-agnostic in Java and in the port: `_apply_additional_for_school` (`:250`),
`_apply_fishing_for_school` (`:357`) and `_apply_foraging_for_school` (`:421`) all end in
`_kill`/`_consume`. Port it as ONE shared inline helper called from every death site, carrying all
four properties of `_consume` (`:107-127`): the denominator is the **pre-death** `inst_abd`; guard
`before > 0.0` (else 0/0 → NaN); clamp with `max(inst_abd[idx], 0.0)`; skip when
`is_background[idx]`. For FISHING, record the `n_dead[:,3]`/`n_dead[:,6]` split first and rescale
**once** with the full death count (Java's two `incrementNdead` calls telescope to
`(I−nDead)/I` — task-4-report §2).

## File structure

| File | Responsibility |
|---|---|
| `osmose/engine/processes/mortality.py` | `_apply_predation_numba` (`:918`), `_apply_single_cause` (`:1146`), `_mortality_in_cell_numba` (`:1188`), `_mortality_all_cells_numba` (`:1332`), `_mortality_all_cells_parallel` (`:1501`), `_pre_generate_cell_rng` (`:670`), dispatches (`:1750`, `:2101`) |
| `tests/test_engine_bioen_numba_kernel.py` (create) | the equivalence gates |
| `tests/_bioen_overlay.py` (create) | the shared `BIOEN_OVERLAY` fixture config (see Task 0) |
| `scripts/bench_bioen_kernel.py` (create) | before/after cost, committed so the number is reproducible |

**Duplication hazard:** `_mortality_in_cell_numba` and the two batch kernels each carry their own
inlined copy of the interleaved loop. Every behavioural change must land in **all three**, and
**production runs `_mortality_all_cells_parallel`** (`mortality.py:2117`) — not the per-cell
kernel. Gates that only pin `_mortality_in_cell_numba` pin code production never executes.

---

### Task 0: Preconditions — supersede Task 4b, define the overlay, fix `_pre_generate_cell_rng`

**Files:** `docs/superpowers/plans/2026-08-30-baltic-c3-bioen-stage1.md` (mark Task 4b superseded),
`tests/_bioen_overlay.py` (create), `osmose/engine/processes/mortality.py` (`_pre_generate_cell_rng`),
`tests/test_engine_mortality_causes.py` (its `_pre_generate_cell_rng` expectations).

- [ ] **Step 1:** In the parent plan, replace Task 4b's body with a one-line pointer to this
  document, so two plans cannot both be executed against the same filenames.

- [ ] **Step 2:** Create `tests/_bioen_overlay.py` exporting `BIOEN_OVERLAY: dict[str, str]` and a
  helper `apply_overlay(cfg, n_species, background_indices)`. Content — copied from the parent
  plan's Task 4b and **not** to be weakened:
  `module.bioenergetics.enabled=true`, `simulation.bioen.phit.enabled=true`,
  `simulation.bioen.fo2.enabled=false`, `temperature.value=7.0`; per focal species
  `species.maturity.m0 = species.maturity.size`, `m1=0`, `r=0.2`, `eta=1`, `beta=0.8`,
  `assimilation=0.7`, `mobilized.tp=10`, `e.mobi=0.65`, `e.d=1.5`, `maint.e.maint=0.65`, and
  **`c_m` chosen so maintenance is a material fraction of intake** — `data/baltic_ev`'s value makes
  it ~1e-8, which switches starvation off entirely and would make behaviour 3 untestable
  (`.superpowers/sdd/2026-08-30-baltic-c3-bioen-stage1/task-6-carried-items.md`, item A). Use
  Task 3's `_three_schools()` ratio (`e_maint/e_gross` ≈ 0.8) as the target and assert it in a test.
  For background species: `species.beta.sp{i}` explicitly, and `predation.ingestion.rate.max.sp{i}`
  in **per-time-step** units (ledger ruling R1 — Java's `getMaxPredationRate` early-returns for
  background predators without the `/nStepYear` the focal branch applies).

- [ ] **Step 3:** Fix `_pre_generate_cell_rng` to draw **five** permutations unconditionally and to
  build its cause list from `_get_mortality_causes(config)` rather than a literal `[0,1,2,3]`, so
  the FORAGING code comes from the enum. Update its docstring (it currently claims to be the
  reference for buffers it does not produce) and its tests.

- [ ] **Step 4:** Run `tests/test_engine_mortality_causes.py`, `tests/test_engine_parity.py`. Commit.

---

### Task 1: The equivalence harness, targeting the kernels production runs

**Files:** `tests/test_engine_bioen_numba_kernel.py` (create).

**Interfaces produced (used by Tasks 2–4):**
```python
def run_cell_both_paths(state, config, *, seed, n_subdt=10, resources=None, ...)
    -> tuple[SchoolState, SchoolState]     # (python_arm, numba_arm), deep copies
def run_batch_both_paths(state, config, *, seed, parallel: bool, ...)
    -> tuple[SchoolState, SchoolState]     # per-cell kernel vs the batch kernel production runs
```
`COMPARED` must cover every field either path writes:
`("abundance", "n_dead", "preyed_biomass", "raw_preyed", "pred_success_rate", "e_net",
"gonad_weight", "trophic_level")`, plus `inst_abd` and, when resources are present,
`resources.biomass` and the diet matrix. The first draft's tuple included `abundance` (which the
cell path does not write) and omitted `raw_preyed`, resource depletion and the TL accumulator.

- [ ] **Step 1: Write the harness.** The Python arm sets `M._HAS_NUMBA = False` for the duration of
  its `_mortality_in_cell` call and restores it after (see "the reference dispatches into the
  kernel" above — without this the test is vacuous). Both arms get a freshly seeded
  `np.random.default_rng(seed)`, so their `seq_*` and cause orders coincide with no replay
  machinery. **Do not monkeypatch a `Generator`** — `np.random.Generator` is a C extension type
  and attribute assignment raises.

- [ ] **Step 2: `run_batch_both_paths`.** The batch kernels generate RNG internally from
  `rng_seed`, so drive the comparison the other way: seed legacy `np.random` in Python with the
  same seed, draw the five permutations and the cause orders exactly as the kernel does, feed those
  to `_mortality_in_cell_numba`, and compare against a batch-kernel call on the same state. Use
  **at least two non-empty cells** so `prange` actually iterates more than once, and run the
  parallel kernel with `NUMBA_NUM_THREADS >= 2`.

- [ ] **Step 3: Prove both harnesses bite.** Sabotage one line of `_apply_single_cause` (e.g.
  `n_dead[idx, 2] += dead * 1.001`), confirm each harness FAILS, restore, and paste the failure
  output into the report. A harness that has never failed is not a harness.

- [ ] **Step 4:** Bioen-OFF equivalence must pass for both harnesses before any behaviour change.
  **Known risk to check here, not later:** exact equality may be unattainable for FISHING because
  `_precompute_effective_rates` composes the rate in a different multiplication order than the
  per-school path. If that shows up, report it with the ULP magnitude and propose the narrowest
  possible exception (a single documented field, or reordering the kernel to match) — **do not**
  loosen the whole comparison to a tolerance.

- [ ] **Step 5:** Commit.

---

### Task 2: All five behaviours in the Numba code paths

Merged from the first draft's Tasks 2 and 3: the review showed Task 2's bioen test could not run
before Task 3's cause-widening landed, so they are one unit of work.

**Files:** `osmose/engine/processes/mortality.py`; `tests/test_engine_bioen_numba_kernel.py`;
`tests/test_engine_bioen_mortality_parity.py` (positional-call updates, see Step 1).

- [ ] **Step 1: Deal with the 41-argument problem first.** `_apply_predation_numba` has 41
  parameters and is called positionally from `_mortality_in_cell_numba` and both batch kernels, and
  is called directly by tests. Adding trailing parameters breaks every positional caller —
  including ~24 currently-green tests. Enumerate the callers
  (`grep -n "_apply_predation_numba(" osmose/engine/processes/mortality.py tests/`), update them
  all in this step, and add `tests/test_engine_bioen_mortality_parity.py` to this task's gate list.
  Numba requires stable types: pass **real arrays always** (zero-filled when bioen is off), never
  `None`, and gate every use on the `bioen` flag.

- [ ] **Step 2: Write the failing tests.** Fixtures must make each behaviour observable, or the
  gate is green on absent code:
  - the cap must **bind** (otherwise behaviour 1 is untested);
  - a prey school must **die during the sub-step** (behaviour 2 via predation);
  - **`additional_mortality_rate > 0` AND `fishing_rate > 0` with a non-zero discard rate**, on a
    school that has already eaten — this is the case the first draft could not see, and the
    natural fixture (`tests/test_engine_bioen_mortality_parity.py:205-243`) zeroes both rates;
  - starvation with `n_subdt >= 2` (the deficit is `|e_net|/n_subdt`, and `n_subdt` was missing
    from the first draft's parameter list entirely), one school whose gonad covers the deficit and
    one whose does not, one school at exactly `ageDt == firstFeedingAgeDt` (strict boundary), and
    one where `deficit/weight > inst_abd` so the deliberate no-clamp behaviour is pinned
    (`_consume`'s docstring records that Java's factor can go negative there);
  - `k_for > 0` so FORAGING is not inert, with a positive witness
    `n_dead[:, int(MortalityCause.FORAGING)].sum() > 0` on both arms;
  - one background school (the rescale must skip it).

- [ ] **Step 3: Run, see them fail** for real divergence, not import errors.

- [ ] **Step 4: Implement**, mirroring the reference exactly:
  - **Behaviour 1** in `_apply_predation_numba`: branch on `bioen` for
    `max_eatable = cap_fish[p_idx] * inst_abd_p`.
  - **Behaviour 2** as ONE shared inline rescale, called from **all five** death sites, with the
    four `_consume` properties spelled out above. FISHING records its split first, then rescales
    once with the full count.
  - **Behaviour 3** in `_apply_single_cause`: inline `bioen_starvation_substep` — including its
    zero-weight and non-negative-`e_net` guards, Java's flush-before-credit ordering, and the
    strict `ageDt > firstFeedingAgeDt` eligibility. Needs `n_subdt`, `gonad_weight`, `weight`,
    `eta_by_species`, `e_net`, `preyed_biomass`, `species_id`, `age_dt`, `first_feeding_age_dt`,
    `is_background`, and the `bioen` flag.
  - **Behaviour 4**: five causes and a fifth school sequence `seq_for` under bioen; **bioen-OFF
    keeps exactly four causes and its current RNG consumption** — that is what protects
    `test_engine_parity.py`. Mirror `_apply_foraging_for_school` fully, including its
    genetic-mode variant (`k1_for`/`k2_for`/`imax_trait`) and both early returns
    (background, pre-first-feeding); if the genetic branch is impractical in the kernel, fall back
    to Python when `imax_trait is not None` and say so loudly rather than silently dropping it.
  - **Behaviour 5**: accumulate `raw_preyed` alongside `preyed_biomass`; TL divides by raw.
  - Apply all of it to **`_mortality_in_cell_numba`, `_mortality_all_cells_numba` AND
    `_mortality_all_cells_parallel`.** In the parallel kernel, confirm the new per-school writes
    (`e_net`, `gonad_weight`, `raw_preyed`) stay within one cell's index set so `prange` remains
    race-free, exactly as the existing per-cell writes do.

- [ ] **Step 5: Cross-kernel agreement.** `run_batch_both_paths` for both batch kernels, ≥2 cells,
  ≥2 threads. This is the only check that the three inlined copies received the same edits.

- [ ] **Step 6:** Run the new file, `test_engine_bioen_mortality_parity.py`,
  `test_engine_mortality_causes.py`, `test_engine_parity.py`. Commit.

---

### Task 3: Flip dispatch, fix the tests that encode the old rule, measure

**Files:** `osmose/engine/processes/mortality.py` (`:1750`, `:2101`);
`tests/test_engine_bioen_mortality_parity.py`; `tests/test_engine_bioen_starvation_rate_suppressed.py`;
`scripts/bench_bioen_kernel.py` (create).

- [ ] **Step 1: Invert the tests that pin the old behaviour — they are expected failures, not
  surprises.** Task 4's own revert probe A ("drop `and not config.bioen_enabled`") produced **9
  failures**; that is the experiment this step performs deliberately.
  - `test_mortality_never_enters_batched_numba_under_bioen` asserts the batch kernels are NEVER
    reached under bioen. **Invert it**: assert they ARE reached. Without inversion nothing pins the
    dispatch at all.
  - The two parameterised tests at `tests/test_engine_bioen_mortality_parity.py:392,401` have
    expectations tied to the old RNG stream; re-derive or re-select their seeds, and assert their
    guard clauses (`raw_eaten > 0`, `n_dead[...] > 0`) so they cannot pass vacuously on a stream
    where nothing happens.
  - `eff_starv[:] = 0.0` in `_precompute_effective_rates` was defence-in-depth while the kernels
    were bypassed; after the flip it is **the only guard** against double-counted starvation. Add
    an explicit assertion that it is still applied under bioen.

- [ ] **Step 2: Flip both dispatch gates** — `mortality.py:2101` (outer) and the `use_full_numba`
  term at `:1750` (inner). Keep them consistent; a mismatch silently splits behaviour between the
  two entry points.

- [ ] **Step 3: Measure.** `scripts/bench_bioen_kernel.py` runs `data/baltic_ev` bioen for 4 years
  with the kernel and with `_HAS_NUMBA=False`, printing s/simulated-year and the ratio, and runs
  the bioen-OFF config as the reference point. Record all three.
  **Stop rule, stated as a conjunction** (the first draft's "under 10× means object-mode fallback"
  named an impossible cause — `njit` has no object-mode fallback): stop and report if the speed-up
  is **< 10×** OR any `NumbaWarning`/`NumbaPerformanceWarning` is emitted during the run. The
  realistic ceiling is not ~100×: mortality is not 100% of step time, so by Amdahl the expected
  whole-run gain is ~50–80×. Criterion: **≥ 10× required, ~50× expected**, with the measured number
  recorded either way.

- [ ] **Step 4: Full gates.** `tests/test_engine_parity.py` (17); the whole new kernel file;
  `test_engine_bioen_mortality_parity.py`; `test_engine_bioen_budget_parity.py`;
  `test_engine_bioen_reproduction_parity.py`; `test_engine_bioen_starvation_rate_suppressed.py`;
  `test_engine_mortality_causes.py`; and
  `PYTHONPATH=. .venv/bin/python scripts/c3_gate_a_reference.py --check` → `IDENTICAL`.
  Then re-run the bioen suites that change execution path under the flip
  (`test_engine_bioen_activation.py`, `test_bioen_orchestration.py`,
  `test_genetics_bioen_integration.py`) and report any change in their pass/fail/xfail status.

- [ ] **Step 5:** Commit.

---

### Task 4: Whole-run sanity, honestly labelled

- [ ] **Step 1:** A distributional check, not a gate: run the `BIOEN_OVERLAY` config for 3 years on
  ≥ 5 seeds with the kernel and ≥ 5 seeds with `_HAS_NUMBA=False`, and compare the two **sets** of
  final-year biomass per species with a two-sample test (Mann-Whitney or KS), reporting the
  p-values. The first draft's "Numba mean within 2 SD of the Python mean" compares a mean against a
  single-run spread and is both insensitive and flaky. State in the docstring that the per-cell and
  cross-kernel tests are the real gates and this is a smoke check. Mark it `@pytest.mark.slow` or
  keep it out of the default selection — it costs minutes.

- [ ] **Step 2:** Update the ledger with the measured speed-up, and record whether the parent
  plan's Task 12 A/B is now a serial overnight job (expected) or still needs parallel arms.

- [ ] **Step 3:** Commit.

## If the gate cannot be met

Stop and report; **do not** relax an equivalence test to a tolerance. A kernel that is
approximately the reviewed path makes every downstream result unattributable. The fallback is to
keep bioen on the Python path and run the parent plan's A/B with **arms in parallel processes at
the full 5 seeds × 50 yr** (~19–24 h wall clock, one thread per worker per
`[[uq-parallel-threading]]`), which preserves the experimental design ruling R7 fixed.

## Success criteria

1. Per-cell AND batch-kernel equivalence to the Python reference is exact for bioen ON and OFF, and
   both harnesses are proven able to fail.
2. All three Numba entry points carry the same bioen behaviour, verified against each other with
   ≥ 2 cells and ≥ 2 threads.
3. `tests/test_engine_parity.py` and Gate A unchanged and passing; every test that encoded the old
   dispatch rule updated deliberately, none left silently red.
4. Measured speed-up ≥ 10× (≈ 50× expected), recorded, with no Numba performance warnings.
5. The parent plan's Task 12 A/B is feasible as a single overnight run.
