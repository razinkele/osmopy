# C# engine, Phase 0 spikes and gate G0 (2026-11-18)

Plan: `docs/superpowers/specs/2026-11-17-csharp-engine-implementation-plan.md`, Phase 0.
Spec: `docs/superpowers/specs/2026-11-17-csharp-engine-design.md`.
**G0 is the owner's decision.** This document gives the measurements and a recommendation.

## Read first: which threshold applies

When the owner was asked (Q2, 2026-11-17), the question named the kernel spike's gate as
"at least 1.15x faster". **That was a misquote.**
- **The plan's G0 for the kernel spike:** go at 1.0x Numba or better, stop below 0.8x after one tuning pass, and the owner decides in between.
- **1.15x is the separate end-to-end gate:** Baltic, 10 yr, one thread, checked when Phase 2 is declared done.

The owner agreed to "performance gates are go/no-go" after reading "1.15x". If the result below
falls **between 0.8x and 1.15x**, the two readings give different answers, and the owner should
say which applies.

## 1. Bridge spike (task 0.2, PR #292): 7/7

| # | Check | Measured |
|---|---|---|
| 1 | pythonnet 3.2.1 hosts .NET 10 through clr_loader, `DOTNET_ROOT` unset | .NET 10.0.15, root found at `~/.dotnet` (on the CI runner: `PATH`) |
| 2 | NumPy buffer passed as an `Int64` address, modified in place | same buffer, values changed |
| 3 | GIL released during a call | 1,889 Python ticks during a 2 s C# call |
| 4 | Control-block cancel | stopped one iteration after the write |
| 5 | .NET exception reaches Python | `InvalidOperationException`, with message and stack |
| 6 | Two threads call at once | 0.50 s wall for two 500 ms calls |
| 7 | `import osmose.engine_cs` does not load the CLR | neither `clr` nor `pythonnet` in `sys.modules` |

Each check is mutation-tested. Taking `Py.GIL()` inside the call turns checks 3 and 6 red.

## 2. Kernel correctness (task 0.6, checks 1 to 4)

**Check 1, libm.** `Math.Exp(-x)` gives the same bits as Numba's `exp(-x)` on all 2,000,063
arguments: the EEC step's own rates plus 2,000,000 log-uniform draws in [1e-8, 20]. Both call
glibc. So whole-cell comparisons run at `atol = 0`, not at tier L. *(This holds on this machine.
Spec Q7 keeps cross-machine bit-reproducibility out of scope.)*

**Checks 2 to 4, the oracle.** The oracle is the Numba per-cell kernel `_mortality_in_cell_numba`, given the same recorded draws.

| Fixture | Located schools | Max cell | Variants exact (of 6) | Distinct digests |
|---|---:|---:|---:|---:|
| EEC step 5, zero rates | 899 | 9 | 6 | 1 |
| EEC step 5, real rates | 899 | 9 | 6 | 1 |
| Baltic step 129 (May, spawning), zero rates | 16,876 | 133 | 6 | 1 |
| Baltic step 129, real rates | 16,876 | 133 | 6 | 1 |
| Baltic step 141 (November), zero rates | 18,059 | 85 | 6 | 1 |
| Baltic step 141, real rates | 18,059 | 85 | 6 | 1 |

The variants are {regrouped, indexed} layout x {full scan, Vector256 mask, #279 candidate lists}.
The probe can see failure: six kernel mutations, from egg release to an empty mask, all turn it red.

**What the Baltic fixtures exercise.**
- Production config, year 5. In step 129: 10,473 prey schools killed, 15,902 predators fed, and deaths from starvation (3,138 schools), additional mortality (14,022) and fishing (9,390).
- 2,464 background schools; stage accessibility; 390 egg schools in step 129, none in step 141.
- Occupancy close to the measured profile:
  - step 129: median 25 schools per cell, 43 cells over 64 (multi-word mask rows), 53.5% of pairs in cells over 50, against 57.5% measured by perf item 3;
  - step 141: median 29.
- **Not exercised:** discards. No Baltic school has a discard fraction above 0. The EEC fixtures do not have one either.

**The tier-L branch was forced once.** With a deliberate 1-ulp error in every `exp` (`Math.BitIncrement`):
- about 14,000 abundances differ, so the instrument sees one-ulp effects;
- the worst scaled error is 8.8e-14 and the worst per-element relative error is 8.2e-11 (starvation deaths, amplified by 1/D as CLAUDE.md describes).

Both are more than four orders of magnitude inside the 1e-9 tier-L bound, so a cross-machine libm difference would not fail the test spuriously.

## 3. Kernel speed (check 5): one Baltic step, one thread, idle machine

**Protocol.**
- Interleaved in one process, with the order rotated each round; 9 rounds; medians.
- 12 warm-up calls per C# arm, for tiered JIT and PGO.
- **The gate baseline is production's kernel:** `_mortality_all_cells_parallel` at `NUMBA_NUM_THREADS=1`. `mortality()` defaults to `parallel=True` and `simulate` never overrides it.
- Both sides time the same work: grouping by cell (Python's argsort, or C#'s counting-sort regroup), the prefilter build, and every sub-step with its own draws.
- Each arm's eating (total predation deaths, preyed biomass) is checked to agree.

Three independent invocations (step 129 twice), 9 rounds each:

| Arm | Step 129 median (s) | Gate / arm | Step 141 median (s) | Gate / arm | Step 129 again (s) | Gate / arm |
|---|---:|---:|---:|---:|---:|---:|
| numba-parallel (gate, production) | 0.0936 | 1.000 | 0.0996 | 1.000 | 0.0951 | 1.000 |
| numba-serial | 0.1183 | 0.791 | 0.1237 | 0.805 | 0.1187 | 0.801 |
| C# regrouped + mask | 0.0431 | **2.172** | 0.0421 | **2.366** | 0.0426 | **2.234** |
| C# regrouped + lists | 0.0407 | **2.297** | 0.0435 | **2.290** | 0.0409 | **2.326** |
| C# indexed + mask | 0.0432 | **2.167** | 0.0434 | **2.293** | 0.0443 | **2.148** |
| C# indexed + lists | 0.0427 | **2.190** | 0.0455 | **2.188** | 0.0423 | **2.248** |

- **Per-round spread:** every round of every C# arm is above 2.07× the gate. The narrowest per-round range is 2.07–2.26 (indexed + lists, step 129).
- **The work agrees.**
  - Step 129: predation deaths 2.4203e10 (C#) against 2.4210e10 (Numba), preyed biomass 488,065 t against 488,090 t.
  - Step 141: 8.869e9 against 8.861e9 deaths, 611,963 t against 611,985 t.
  - The differences (below 0.1%) are the two engines' different random streams. All four C# variants report identical totals, as determinism across layout and prefilter requires.
- **Best variant:** regrouped + lists on step 129, 2.30× and then 2.33× in the confirming run; regrouped + mask on step 141, 2.37×. The four variants are within about 10% of each other and their order changes between steps.
- **Side observation:** Numba's serial batch kernel is about 20% slower than its parallel one at one thread.
- **Warm-up matters.** On the small EEC step-5 fixture (899 schools, cells of at most 9), an early smoke with a single warm-up call measured C# at 0.85×. After the 12-call warm-up, one round measured 1.25×: .NET's tiered JIT and PGO need several calls before the hot loop is optimised. This is one round on a tiny fixture, so it is context, not a result.

## 4. Recommendation: GO

- **The kernel gate is met by a wide margin.** Every C# variant runs a production Baltic mortality step 2.15–2.37× faster than production's Numba kernel at one thread. The bridge passes 7/7. That clears the plan's G0 (go at 1.0×) and also the misquoted 1.15×, so the threshold question at the top does not change the decision.
- **Correctness came first.** The same kernel equals the Numba oracle bit for bit on six fixtures (EEC and two Baltic steps, zero and real rates) in all six variants.
- **Keep the spec's design (regrouped + mask).** Keep candidate lists as the drop-in the spec already allows. The variants are within 10% of each other with no stable winner, so decide in task 1.9 on whole-run measurements, not on this kernel number.

**What this number is not.**
- **It is the kernel, at one thread, on this machine.** Mortality is about 78% of a one-thread Baltic run. If everything else ran exactly as fast as Python's, a 2.2× kernel would give about 1 / (0.78 / 2.2 + 0.22) ≈ 1.7× end to end.
- **The 1.15× end-to-end gate is Phase 2's.** It needs the rest of the step loop ported first.
- **Not measured:** multi-thread scaling, and the discard path (no fixture has a discard fraction above 0).
- **Where the 2.2× comes from is not decomposed.** Candidates: cheaper draws (one Lehmer code per slot instead of a shuffle), the cell-outer sub-step loop, and code generation. The near-equal regrouped and indexed timings suggest memory layout matters less than the spec expected at this occupancy.
- **Bit-equality with the oracle rests on glibc's `exp`.** Both engines call it on this machine. Another libm drops the comparison to tier L (section 2), which the forced 1-ulp run shows the tests tolerate.

**If the owner says go:** Phase 1 (spec section 7.2), starting with tasks 1.1 (typed parameters and capabilities) and 1.2 (the bundle builder for EEC and Bay of Biscay).
