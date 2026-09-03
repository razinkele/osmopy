# Bioen-aware Numba mortality kernel — final review record

**Branch:** `c3-bioen-stage1` · **Range reviewed:** `8546627..d9ff38e` (15 commits)
**Verdict:** APPROVED WITH FINDINGS — 0 Critical, 0 Important, 3 Minor
**Date:** 2026-09-03

This is the durable record of a four-task plan executed with per-task review plus a final
whole-branch pass. The working ledger it distils is git-ignored scratch and will not survive; the
evidence below is the part worth keeping. Written for someone who needs to trust — or re-open —
this work later, not as a process log.

## What changed and why

Bioenergetics ("bioen") behaviours existed only in the pure-Python per-cell mortality path, so
enabling bioen forced the whole simulation onto the interpreter. This branch taught the batched
Numba kernels five bioen behaviours and flipped the dispatch gates, so bioen now runs compiled.

**Measured speed-up: 149×** (implementer) and **157.8×** (reviewer, independent re-run) on a 4-year
window; ~80–85× on a third config. The ratio is not horizon-free — per-step cost tracks school
count, and the same benchmark gives **81.9× at 1 year against 149× at 4 years**. The interpreted
path degrades faster than the compiled one as the population builds, so a 50-year run should exceed
149×, not fall short of it. Quote the ratio with its horizon.

**The Python per-cell path is the ORACLE.** It is what was verified against Java OSMOSE 4.3.3, and
every equivalence test on this branch asserts "the kernel matches the reference". That makes the
oracle's integrity load-bearing: if it moved, those tests would be circular.

## The five ported behaviours

| # | Behaviour | Java reference |
|---|---|---|
| 1 | Per-fish ingestion cap `max_eatable = cap_fish[p] * inst_abd[p]` | `BioenPredationMortality.java:140-145` |
| 2 | Survivor rescale of `preyed_biomass` AND `e_net` at **every** death, all five causes | `School.incrementNdead` — no switch on cause |
| 3 | Bioen starvation in the interleaved loop, previous step's `e_net`, gonad buffer | `BioenStarvationMortality`, `Species.java:224-226` |
| 4 | Five causes in the shuffle (FORAGING) with its own `seq_for` | `MortalityProcess.java:506-517` |
| 5 | Trophic level divides by the RAW preyed total; the budget uses the rescaled one | `AbstractSchool.preyedBiomass` vs `School.ingestion` |

Behaviour 2 is the one most easily under-implemented — it is cause-agnostic in Java and must fire at
all five in-step death sites. An early draft of the plan specified it for two of five.

## Evidence

Ranked by how much it would cost to reconstruct.

**The oracle is intact.** All seven standalone reference functions — `_kill`, `_consume`, and the
five `_apply_*_for_school` — are byte-identical across all 15 commits, verified by AST-extracting
each function from `git show` at both ends of the range rather than from any report. Scope limit,
stated so it is not over-read: this does not cover `_mortality_in_cell`'s Python branch, which lives
inside a function Tasks 2 and 3 legitimately modified. That branch rests on per-task review — a
line-range diff plus six sabotage/restore cycles.

**Bioen-OFF is bit-for-bit unchanged.** `tests/test_engine_parity.py` passes 17/17, and **no parity
baseline file appears in any of the 15 commits** — so those results are against pre-branch committed
fixtures, not regenerated ones. That is the strongest form the claim can take.

**Three composed-system revert probes**, sabotaging the *seams* between tasks. Each per-task
reviewer could only sabotage inside its own slice; these could not have been run earlier.
`mortality.py` was md5-verified back to `26b01705225b4948c64da2f90ea03f72` after each.

| probe | sabotage | result |
|---|---|---|
| A | Hoist the fifth permutation out of `if bioen:` in `_mortality_all_cells_parallel` | parity **6 failed, 11 passed** |
| B | Parallel-kernel-only bioen drift (`raw_preyed += 1e-30` at the FORAGING site) | exactly `test_batch_kernel_matches_the_per_cell_kernel_under_bioen[True]` reddens |
| C | Drop `_consume_numba`'s `is_background` guard | **8 failed, 42 passed** |

Probe A is the important one: the bioen-OFF invariant is protected by a **live gate**, not by
inspection. Every prior check established that by reading code. Probe B confirms that parametrising
the bioen batch tests over `parallel=[False, True]` is load-bearing — it is the only mechanism that
can catch a divergence between the three inlined copies of the interleaved loop, and production
dispatches to the parallel one. Probe C shows behaviour 2's background carve-out is pinned by five
independent fixtures, not one lucky test.

**No `prange` race.** Every array write reachable from a parallel iteration was enumerated with its
index provenance; all indices resolve through `cell_indices = sorted_indices[start:end]` slices that
are disjoint by construction, and `q_idx == p_idx` is explicitly skipped so a predator cannot
rescale its own `preyed_biomass` mid-phase. The three new bioen writes inherit that disjointness.

**Parallelism survived the edits:** fresh compile with a redirected `NUMBA_CACHE_DIR` lists `prange`
as parallel loop #0, reports "Parallel structure is already optimal", and emits **0 Numba warnings**.

## The recurring defect, and the seventh instance

Seven times on this branch a gate reported green over code it never executed: a parity suite blind
to a refactor; a fixture whose `c_m` silently disabled the behaviour under test; a code path inert
in every config in the repo; an xfail indistinguishable from an unimplemented feature; a revert
probe that reddened nothing because the sabotage sat in a function both arms shared; tautological
value witnesses on output accumulators; and finally —

**A differential test whose control arm was never checked.** The whole-run smoke check forced its
control onto the Python path with `_HAS_NUMBA = False` but asserted nothing about dispatch. A broken
toggle would make both arms run the kernel, the result sets become identical, and Mann-Whitney
return **p = 1.0 for every species** — the test passing *maximally*, with its most reassuring
possible output. Closed with a two-directional witness (kernel arm `calls > 0`, control arm
`calls == 0`) and proven, not asserted: with the toggle skipped, the kernel was entered 240× and the
new assertion caught it.

The general lesson, worth more than the instances: **most broken tests degrade toward false failure
— noisy and self-announcing. These degrade toward false success, silently.** For any differential
test, assert that the two arms actually differ in the way you arranged, in both directions.

## Findings and dispositions

| # | Finding | Disposition |
|---|---|---|
| F1 | No bioen fixture has a school with `age_dt` strictly below `first_feeding_age_dt`, so the FORAGING pre-first-feeding exemption is exercised by nothing. Proven: dropping the exemption leaves the kernel file at 50 passed and the parent bioen files at 51. | Fixed on-branch. Code was correct; the gate was blind. |
| F2 | `eff_starv[:] = 0.0` documented in two places as "the ONLY guard" against double-counted starvation. It is dead — all seven exit paths return first; replacing it with `pass` leaves 95 passed. | Wording fixed; line kept as defence in depth. The real guard is the `return` at `mortality.py:1391`. |
| F3 | `Co-Authored-By` is not a machine-readable trailer on five docs-only commits — a blank line separates it from `Claude-Session`, ending git's trailer block. | Not fixed. Rebasing would invalidate the SHAs the recovery ledger names, for cosmetic attribution on docs-only commits. |

F1 was latent, not live: `forage.k_for` is unset in every config in the repo, so FORAGING is inert
today. It stops being inert when a fit sets `k_for`, at which point every post-spawning step has egg
schools at `age_dt = 0 < ffa = 1` and the kernel would apply FORAGING mortality to eggs that the
reference and Java's `ForagingMortality` spare (reference `0.0` deaths against a sabotaged kernel's
`43696.99`).

The F1 fix is a **self-contained one-school fixture** (`_pre_feeding_forage_fixture`), not a school
added to the shared `bioen_gate_fixture()`. The reasoning is worth keeping, because the tempting
edit is the wrong one: the shared fixture has 51 uses, and adding a school there shifts RNG
permutation widths per cell — but the deeper problem is that every existing prey school sits inside
some predator's eligible size window, so PREDATION's `abd <= 0` guard could mask the age exemption
and make the test's reddening **seed-dependent**. A single school alone in its own cell removes that
confound. The test also carries an `assert config.bioen_enabled` (so it cannot pass vacuously if
bioen keys ever drop out of the fixture) and `kernel_calls` pins (so the comparison cannot silently
degrade into kernel-vs-kernel or reference-vs-reference) — a blind test for a blind gate would have
been the eighth instance of the defect above.

Proven, not asserted. Reproducing the reviewer's exact sabotage: at unit level `eff_foraging[0]`
moved `0.0` → `0.0125`; at integration level the kernel arm's `n_dead[0, FORAGING]` moved `0.0` →
`222210.79` while the reference arm stayed `0.0` on its own independent guard. Restored clean.

**Also closed by measurement:** the composition-order hazard between `_precompute_effective_rates`
and the per-school path had been characterised bioen-OFF only, with no one checking whether the
survivor rescale amplifies it. Measured across all four seasonality × spatial combinations under
bioen: it **damps**. Worst case `n_dead` 1.64e-14; `e_net`/`preyed_biomass`/`raw_preyed` ≤ 4.2e-16;
`gonad` exact — two orders inside the 1e-12 bound the branch set for itself.

## Carried forward — read this before extending bioen

**The kernel test file is not a sufficient gate set on its own.** Removing the `if en >= 0.0: return`
guard from the kernel's bioen starvation branch — the branch a *healthy* population takes on nearly
every visit, and one the gate fixture cannot exercise because all eight of its schools have
`e_net < 0` — leaves `tests/test_engine_bioen_numba_kernel.py` at 50 passed. It is caught only by
`tests/test_engine_bioen_mortality_parity.py::test_no_starvation_under_bioen_when_enet_is_positive`,
a test the dispatch flip silently converted from a Python-path test into a kernel gate. **Keep the
whole bioen suite in the required list**, not just the kernel file.

**An open Java-parity question, pre-existing in the oracle.** Java's `School.incrementNdead` is
cause-agnostic, but `out_mortality` (`osmose/engine/processes/natural.py:184-208`) is a sixth death
site outside the five and applies no rescale. `preyed_biomass` resets every step so that half is a
no-op — but **`e_net` does not reset**, so out-of-domain schools may carry a larger energy budget
into the next step than Java's would. Unmeasured, and note that Java's OUT branch has been
*inferred* from `incrementNdead` being cause-agnostic, not traced — read the Java source before
treating this as a bug. A Java cross-engine gate is the only harness that can see it directly.

**A gonad-credit coverage gap.** Bioen starvation's credit branch (deficit covered from the gonad
reserve) is pinned per-cell by synthetic fixtures but has never run end-to-end, because
`data/baltic_ev` never builds a material reserve within the few simulated years any test runs. The
first fitted overlay that sustains an energy surplus is where to check it — and to say so either
way, since an unmentioned gap reads as a covered one.

## Reproducing the headline numbers

- Speed-up: `scripts/bench_bioen_kernel.py` (three arms; self-enforces a fresh `NUMBA_CACHE_DIR` and
  records `cache_mode`, because a warm cache silently flatters the compiled arm).
- Bioen-OFF baseline: `scripts/c3_gate_a_reference.py` against
  `docs/diagnostics/c3_gate_a_master_baseline.json` (5 seeds × 50 yr, production Baltic). Must print
  `IDENTICAL`.
- The whole-run distributional smoke check is opt-in via `OSMOSE_BIOEN_WHOLE_RUN_SMOKE=1` and needs
  `-s` to show its p-values. At 5 seeds per arm the exact two-sided Mann-Whitney floor is
  **2/252 ≈ 0.0079**, so `p <= 0.008` is exactly equivalent to complete separation and nothing else
  can satisfy it. That threshold is tied to the seed count — re-derive it if either seed tuple
  changes. The check rules out a gross divergence and nothing finer; the real guarantees are the
  bit-exact per-cell and cross-kernel gates.
