# Bioen-Aware Numba Mortality Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the batched Numba mortality kernels the five bioenergetics behaviours that currently exist only in the pure-Python per-cell path, and re-enable Numba dispatch under bioen, so a 50-year bioen run costs minutes instead of ~12–18 hours.

**Architecture:** The pure-Python per-cell path (`_apply_*_for_school`, `_mortality_in_cell`) stays exactly as it is and remains the reference implementation — it is what was reviewed against Java 4.3.3. This plan adds a bioen branch to the three Numba entry points that mirror it, and pins the new code to the reference with a *deterministic, bit-exact, per-cell* equivalence test.

**Tech Stack:** Python 3.12, NumPy, Numba (`njit`, `prange`), pytest. Run everything with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-30-baltic-c3-bioen-stage1-design.md` (this work reverses that spec's decision 14, which declared a bioen Numba kernel a non-goal; see "Why" below). Parent plan: `docs/superpowers/plans/2026-08-30-baltic-c3-bioen-stage1.md`, whose Task 4b this plan replaces and expands.

## Why this exists (measured, not assumed)

| measurement (this machine, warm cache) | result |
|---|---|
| production Baltic, bioen OFF, Numba ON, 4 yr | 3.9 s (0.99 s/yr) |
| `baltic_ev`, bioen ON, pure-Python path, 4 yr, ~12 000 schools | 442 s (110 s/yr) |
| same-population ratio | **152×** |
| production Baltic, bioen OFF, Numba **disabled**, 4 yr | >13 min (stopped; >200×) |
| school growth, bioen ON vs bioen OFF | identical, ~3 000/yr — **not** a bioen defect |
| extrapolated single 50-yr bioen run | ~12–18 h |
| the parent plan's Task 12 A/B (10 bioen runs) | **~120 h — not runnable** |

The third row is decisive: with Numba off, the *non-bioen* path is just as slow, so bioen adds no
overhead of its own. The entire gap is JIT-versus-interpreter, incurred because Task 4 correctly
stopped dispatching bioen to the batched kernels (they ignore the bioen ingestion cap — which is
why that cap had been a silent no-op).

## THE CONSTRAINT THAT SHAPES THIS ENTIRE PLAN

**The Numba path and the Python path do not share an RNG stream and never have.**
`_mortality_all_cells_numba` calls `np.random.seed(rng_seed)` and then Numba's
`np.random.permutation` / `np.random.shuffle` (legacy MT19937, compiled), while the Python path
consumes a NumPy `Generator` (PCG64) supplied by the caller. Whole-run outputs from the two paths
are therefore **not comparable bit-for-bit**, and any plan step that demands it is impossible.

What *is* achievable, and what this plan uses as its gate:

* `_mortality_in_cell_numba` (`mortality.py:1188`) takes `cause_orders`, `seq_pred`, `seq_starv`,
  `seq_fish`, `seq_nat` **as arguments**. `_mortality_in_cell` (Python, `:1660`) generates the
  same quantities from its `rng`. Feed BOTH the same pre-generated buffers and their outputs must
  agree exactly. That is a true bit-identity gate at the per-cell level, and it is where all the
  bioen logic lives.
* At whole-run level, use a *statistical* check across seeds (the two paths are two samples of the
  same process), never an equality assertion.

`_pre_generate_cell_rng` (`:670`) already exists and is documented as "a tested reference
implementation" that produces exactly these buffers — it is the tool for this.

## Global Constraints

- `.venv/bin/python -m pytest`; `.venv/bin/ruff check` + `ruff format`; line length 100.
- Branch `c3-bioen-stage1`. Commit per task, per-path `git add` only — **never `git add -A`**: the
  tree carries the user's unrelated uncommitted work (`osmose/runner.py`, `osmose/cli.py`,
  `osmose/engine/movement_maps.py`, `.mcp.json`, `mcp_servers/copernicus/server.py`,
  `tests/test_runner.py`, `tests/test_engine_map_movement.py`,
  `tests/test_hpc_container_touchups.py`).
- Commit trailers, on their own lines:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01KSP4ExqHQmyMWf8KsfmZU1`
- **`tests/test_engine_parity.py` (17 tests) must pass after every task.** It pins the bioen-OFF
  Numba path against committed fixed-seed baselines; this plan must not perturb it.
- **The Python per-cell path is not to be modified.** It is the oracle. If a task appears to need a
  change there, stop and report instead.
- Never run two engine jobs at once on this machine.
- Do not run the full suite: `tests/test_run_fie_demo.py::test_run_fie_demo_short_smoke` exceeds
  38 minutes on the current (pre-fix) code. Deselect it explicitly if running anything broad.

## The five behaviours to port (each already implemented in Python by Task 4)

| # | Behaviour | Python reference | Java source |
|---|---|---|---|
| 1 | Per-fish ingestion cap: `max_eatable = cap_fish[p] * inst_abd[p]` | `_apply_predation_for_school`, `mortality.py:~470` | `BioenPredationMortality.java:140-145` |
| 2 | Survivor scaling of `preyed_biomass` and `e_net` at every death, skipping background schools | `_consume` / `_kill`, `mortality.py:~100-125` | `School.java:371-402` |
| 3 | Bioen starvation inside the interleaved loop, on the PREVIOUS step's `e_net`, gonad buffer, strict `ageDt > firstFeedingAgeDt` | `_apply_starvation_for_school` bioen branch + `bioen_starvation_substep` | `BioenStarvationMortality.computeStarvation`, `Species.java:224-226` |
| 4 | Five causes in the shuffle (FORAGING added) | `_get_mortality_causes` | `MortalityProcess.java:506-517` |
| 5 | Trophic level divides by the RAW preyed total; the budget consumes the survivor-scaled one | `raw_preyed` in `_apply_predation_for_school` | `AbstractSchool.preyedBiomass` vs `School.ingestion` |

## File structure

| File | Responsibility |
|---|---|
| `osmose/engine/processes/mortality.py` (modify) | `_apply_predation_numba` (`:918`), `_apply_single_cause` (`:1146`), `_mortality_in_cell_numba` (`:1188`), `_mortality_all_cells_numba` (`:1332`), `_mortality_all_cells_parallel` (`:1501`), dispatch (`:2101`) |
| `tests/test_engine_bioen_numba_kernel.py` (create) | the per-cell equivalence gate + the statistical whole-run check |
| `scripts/bench_bioen_kernel.py` (create) | before/after cost measurement, committed so the number is reproducible |

**Note the duplication hazard:** `_mortality_in_cell_numba` and the two batch kernels each carry
their own copy of the interleaved loop (the batch kernels inline it for speed). Every behavioural
change must land in **all three**. Task 3 of this plan exists solely to make that explicit and
verified rather than hoped for.

---

### Task 1: Deterministic per-cell equivalence harness (no behaviour change yet)

This task builds the gate before anything it gates. It must FAIL to detect nothing at the end —
i.e. it must demonstrably catch a deliberately broken kernel.

**Files:**
- Create: `tests/test_engine_bioen_numba_kernel.py`

**Interfaces:**
- Produces: `run_cell_both_paths(state, config, ...) -> tuple[SchoolState, SchoolState]` — a test
  helper that runs one cell through `_mortality_in_cell` (Python) and `_mortality_in_cell_numba`
  with the SAME pre-generated RNG buffers, returning both resulting states for comparison.
  Tasks 2–4 reuse it.

- [ ] **Step 1: Write the harness and a bioen-OFF equivalence test**

Start with bioen OFF, where the two paths are already supposed to agree — this proves the harness
itself is sound before any new code depends on it.

```python
# tests/test_engine_bioen_numba_kernel.py
"""The Numba mortality kernels must reproduce the pure-Python per-cell path EXACTLY.

The Python path is the reference: Task 4's review verified it against Java 4.3.3 (survivor
scaling, the cap form, starvation ordering, the raw/scaled preyed split). Whole-run outputs of the
two paths are NOT comparable — the batch kernel seeds Numba's MT19937 inline while the Python path
consumes a PCG64 Generator — so equivalence is asserted per cell, with both paths fed the same
pre-generated sequences.
"""
from __future__ import annotations

import numpy as np
import pytest

import osmose.engine.processes.mortality as M

pytestmark = pytest.mark.skipif(not M._HAS_NUMBA, reason="kernel tests require numba")

COMPARED = ("abundance", "n_dead", "preyed_biomass", "pred_success_rate", "e_net", "gonad_weight")


def run_cell_both_paths(state, config, *, seed=0, n_subdt=None, resources=None, grid_nx=10):
    """Run ONE cell through both implementations with identical RNG buffers.

    Returns (python_state, numba_state) as deep copies, plus the shared inst_abd arrays, so the
    caller can compare any field. The RNG buffers come from `_pre_generate_cell_rng`, which both
    paths then consume identically.
    """
    ...  # see Step 2 for the concrete body


def test_harness_detects_a_deliberately_broken_kernel(monkeypatch, bioen_off_cell):
    """The gate must be able to fail. If this passes with a sabotaged kernel, the gate is vacuous."""
    ...
```

- [ ] **Step 2: Implement `run_cell_both_paths` concretely**

It must: build `inst_abd` from `state.abundance`; call `M._pre_generate_cell_rng(rng, boundaries,
n_cells=1)` to get `seq_bufs` and `cause_orders_buf`; deep-copy the state twice; call
`M._mortality_in_cell(...)` on copy A with a `Generator` **whose draws are stubbed by the same
buffers** (simplest: monkeypatch `rng.permutation`/`rng.shuffle` to replay `seq_bufs` /
`cause_orders_buf` in order), and `M._mortality_in_cell_numba(...)` on copy B with the buffers
passed directly. Assert nothing here — return both.

If replaying the Generator proves awkward, the alternative is to compare `_mortality_in_cell_numba`
against a small pure-Python reimplementation of the same loop that consumes the buffers directly —
but prefer the real Python path; the point is to pin the kernel to *the reviewed code*.

- [ ] **Step 3: Prove the harness bites**

Temporarily sabotage one line of `_apply_single_cause` (e.g. `n_dead[idx, 2] += dead * 1.001`),
run the bioen-OFF equivalence test, confirm it FAILS, then restore. Record the failure output in
the report. A harness that has never failed is not a harness.

- [ ] **Step 4: Run and commit**

`.venv/bin/python -m pytest tests/test_engine_bioen_numba_kernel.py -q` → passes;
`tests/test_engine_parity.py` → 17 passed.

```bash
git add tests/test_engine_bioen_numba_kernel.py
git commit -m "test(engine): deterministic per-cell equivalence harness for the Numba mortality kernels"
```

---

### Task 2: Behaviours 1, 2 and 5 in the Numba predation kernel

**Files:**
- Modify: `osmose/engine/processes/mortality.py` — `_apply_predation_numba` (`:918`)
- Modify: `tests/test_engine_bioen_numba_kernel.py`

**Interfaces:**
- `_apply_predation_numba` gains four trailing parameters: `cap_fish` (float64[:], per-fish tonnes
  per sub-step; ignored when `bioen` is False), `raw_preyed` (float64[:]), `e_net` (float64[:]),
  `is_background` (bool[:]), and `bioen` (bool). Numba requires consistent types across calls, so
  pass real arrays (never `None`) — use zero-length or zero-filled arrays for the bioen-OFF case
  and gate every use on the `bioen` flag.

- [ ] **Step 1: Write the failing test** — a bioen cell with a predator whose cap binds, asserting
  Python/Numba equality on all of `COMPARED` plus `raw_preyed`. Build the fixture so that: the cap
  actually limits intake (otherwise behaviour 1 is untested), at least one prey school dies during
  the sub-step (otherwise behaviour 2 is untested), and one school is background (otherwise the
  skip-background half of behaviour 2 is untested).

- [ ] **Step 2: Run it, see it fail** with a real divergence (not an import error).

- [ ] **Step 3: Implement.** In `_apply_predation_numba`, replace
  `max_eatable = biomass_p * ingestion_rate[sp_pred] / (n_dt_per_year * n_subdt)` with a branch on
  `bioen` selecting `cap_fish[p_idx] * inst_abd_p`; at the prey-death site
  (`n_dead[q_idx, 0] += n_dead_prey; inst_abd[q_idx] -= n_dead_prey`) add the survivor rescale of
  `preyed_biomass[q_idx]` and `e_net[q_idx]` when `bioen and not is_background[q_idx]`; accumulate
  the predator's eaten total into `raw_preyed[p_idx]` as well as `preyed_biomass[p_idx]`.
  **Mirror `_apply_predation_for_school`'s ordering exactly** — the rescale happens on the same
  side of the accumulation as in the Python code.

- [ ] **Step 4: Run** the new test, the bioen-OFF equivalence test, and `test_engine_parity.py`.

- [ ] **Step 5: Commit** (`perf(engine): bioen cap, survivor scaling and raw-preyed in the Numba predation kernel`).

---

### Task 3: Behaviours 3 and 4 — bioen starvation and the five-cause shuffle, in all three kernels

**Files:**
- Modify: `osmose/engine/processes/mortality.py` — `_apply_single_cause` (`:1146`),
  `_mortality_in_cell_numba` (`:1188`), `_mortality_all_cells_numba` (`:1332`),
  `_mortality_all_cells_parallel` (`:1501`), `_pre_generate_cell_rng` (`:670`)
- Modify: `tests/test_engine_bioen_numba_kernel.py`

- [ ] **Step 1: Write the failing tests.** Two: (a) a bioen cell where starvation fires — build it
  with `e_net < 0` and a gonad that covers the deficit for one school and not another, so both
  branches of `bioen_starvation_substep` are exercised, and a school at exactly
  `ageDt == firstFeedingAgeDt` to pin the strict boundary; (b) a cell where FORAGING is in the
  cause set (set `k_for > 0` so it is not inert) and the per-cell result still matches Python.

- [ ] **Step 2: Run, see them fail.**

- [ ] **Step 3: Implement.**
  - `_apply_single_cause` gains the bioen starvation branch: when `bioen and cause == 1`, inline
    the scalar form of `bioen_starvation_substep` (deficit from the PREVIOUS step's `e_net[idx]`,
    gonad buffer with Java's flush-before-credit ordering, `ndead = deficit / weight`), then the
    same survivor rescale as behaviour 2. It needs `gonad_weight`, `weight`, `eta_by_species`,
    `e_net`, `species_id`, `first_feeding_age_dt`, `age_dt`, `is_background`.
  - Add a FORAGING branch (cause 5) computing `M = k_for[sp] / (ndt * n_subdt)` and the standard
    `dead = abd * (1 - exp(-M))`, mirroring `_apply_foraging_for_school`.
  - Widen the cause buffers from 4 to 5 under bioen in **all three** kernels and in
    `_pre_generate_cell_rng`. Keep the bioen-OFF path at exactly 4 causes and its existing RNG
    consumption — this is what protects `test_engine_parity.py`.
  - **Add the same code to `_mortality_all_cells_numba` and `_mortality_all_cells_parallel`.**
    Their loops are inlined copies; a change in one and not the others is the most likely defect
    in this plan.

- [ ] **Step 4: Verify all three kernels are consistent.** Add a test that runs the same cell
  through `_mortality_in_cell_numba` and through a one-cell call of each batch kernel, asserting
  identical results. Without this, the two batch kernels are untested for bioen.

- [ ] **Step 5: Run** the whole new file + `test_engine_parity.py` + `test_engine_bioen_mortality_parity.py`.

- [ ] **Step 6: Commit** (`perf(engine): bioen starvation and five-cause shuffle in all three Numba kernels`).

---

### Task 4: Re-enable dispatch, measure, and gate

**Files:**
- Modify: `osmose/engine/processes/mortality.py` — dispatch at `:2101`
- Create: `scripts/bench_bioen_kernel.py`
- Modify: `tests/test_engine_bioen_numba_kernel.py`

- [ ] **Step 1: Write the whole-run statistical check.** The two paths cannot be compared by
  equality (different RNG streams). Assert instead that final-year biomass per species from the
  Numba path lies within the across-seed spread of the Python path over ≥ 5 seeds — e.g. each
  species' Numba mean within 2 standard deviations of the Python mean, on a 3-year run of the
  minimal bioen overlay. State in the docstring that this is a distributional check and that the
  per-cell tests are the real gate.

- [ ] **Step 2: Flip the dispatch.** Remove `and not config.bioen_enabled` from `mortality.py:2101`
  and pass the bioen arrays through. Keep `_mortality_in_cell`'s own `use_full_numba` gate
  consistent.

- [ ] **Step 3: Measure.** `scripts/bench_bioen_kernel.py` runs `data/baltic_ev` bioen for 4 years
  with the kernel and with `_HAS_NUMBA=False`, printing s/simulated-year for both and the ratio.
  Record the numbers in the report. **Target: bioen-with-kernel within ~2× of the bioen-OFF Numba
  path at equal school count** (i.e. roughly 1–3 s/yr, versus the 110 s/yr measured today). If the
  speed-up is under 10×, stop and report — something is falling back to the object-mode path.

- [ ] **Step 4: Full gates.** `tests/test_engine_parity.py` (17), the whole new kernel file,
  `test_engine_bioen_mortality_parity.py`, `test_engine_bioen_budget_parity.py`,
  `test_engine_bioen_reproduction_parity.py`, and
  `PYTHONPATH=. .venv/bin/python scripts/c3_gate_a_reference.py --check` → must print `IDENTICAL`.

- [ ] **Step 5: Commit** (`perf(engine): dispatch bioen to the Numba kernels -- <N>x faster, pinned to the Python reference`).

## If the gate cannot be met

If the per-cell equivalence cannot be made exact — for instance if a bioen behaviour turns out to
depend on Python-side state the kernel cannot see — **stop and report; do not relax the test to a
tolerance.** A kernel that is approximately the reviewed path makes every downstream result
unattributable. The documented fallback is to keep bioen on the Python path and re-scope the
parent plan's A/B to 3 seeds × 30 years (~24 h), recording the reduced statistical power.

## Success criteria

1. Per-cell equivalence between `_mortality_in_cell_numba` and `_mortality_in_cell` is exact, for
   bioen ON and OFF, and the harness is proven able to fail.
2. All three Numba entry points carry the same bioen behaviour, verified against each other.
3. `tests/test_engine_parity.py` and Gate A both unchanged and passing.
4. Measured bioen speed-up ≥ 10×, target ~100×, with the number recorded.
5. The parent plan's Task 12 A/B is feasible in a single overnight run.
