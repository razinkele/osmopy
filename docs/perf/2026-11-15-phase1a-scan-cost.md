# Phase 1a scan: its in-situ cost on the Baltic, 2026-11-15

This is the first measurement for item 6 of `docs/osmopy-performance-optimization-plan.md`: the
share of one-thread mortality-kernel time spent in Phase 1a, the size-window scan over school prey
in `_apply_predation_numba`. Item 3 (`docs/perf/2026-11-14-baltic-cell-occupancy.md`) found
Baltic cells full and 88% of scanned prey failing the size test. It could not say what that costs
in time.

**Result:**
- **The whole Phase 1a block costs about 6.7 s of 26.3 s.** That is 25% of one-thread mortality
  time and about 20% of a one-thread 10-yr Baltic run.
- **Of that, about 4.4 s is the per-pair testing.** The rest, about 2.3 s, is the work done for
  eligible prey: the accessibility lookup, the prey-biomass load and the buffer writes.
- **A prefilter's ceiling is smaller than the 20%.** It can only remove the testing of pairs that
  fail, roughly 3.8–4.4 s, or 11–13% of a one-thread run. That is before paying for the
  prefilter's own bookkeeping, and before the bit-exactness constraints below.

## Method: repeat the scan in place

**The edit.** In a detached scratch worktree, Phase 1a of `_apply_predation_numba` is repeated:
- **Full repeat:** the scan loop and its `total_available`/`n_prey` reset, wrapped in
  `for _rep in range(N):`. The repeats recompute the same buffers from unchanged inputs.
- **Test-only repeat:** an extra pass that runs only the per-pair tests: skip self, empty and
  zero-length prey, then the size window. It does none of the eligible-prey work. Its result
  feeds a branch that is never taken, so the compiler cannot drop the loop.

**Why outputs stay the same.** Either way the run must be unchanged, and it is: the biomass hash is
identical in every run below. The added `mortality()` time is therefore the repeated work's cost
inside the real kernel, with its memory access and code generation.

**The repeat is linear.** Two extra full passes added 13.77 s, 6.89 s each, matching the single
extra pass's 6.55–6.81 s. Warm-cache and code-generation effects on the repeat are therefore small. (A repeat
could err in either direction: a warm cache would understate the first pass, and a restructured
loop nest could overstate it.)

**Setup:** warm 10-yr Baltic `run_in_memory`, one Numba thread (the calibration worker regime since
item 1), seed 42, master `dd196435` (engine code unchanged since). Arms interleaved, each in its own
interpreter.

| arm | control `mortality()` | probed `mortality()` | added |
|---|---:|---:|---:|
| full ×2, pair 1 | 26.10 s | 32.91 s | 6.81 s |
| full ×2, pair 2 | 26.34 s | 32.98 s | 6.64 s |
| full ×2, pair 3 (committed script) | 26.53 s | 33.08 s | 6.55 s |
| full ×3 | 26.21 s | 39.98 s | 13.77 s, so 6.89 s per pass |
| test-only, one extra pass | 26.30 s | 30.62 s | 4.32 s |

For the last two rows, the control ran once before and once after the probed arms (26.21 s and
26.30 s). Each probed arm is set against the nearer control; against the mean of 26.26 s, the
test-only pass adds 4.37 s.

**How the result is assembled:**
- **Phase 1a as a whole:** 6.67 s, the mean of the three ×2 pairs. That is 25.3% of mortality time
  and about 20% of the 33.1 s run.
- **Per-pair testing:** 4.3–4.4 s, from the test-only pass. It includes the tests of the 11.8% of
  pairs that pass, so the rejected pairs' share is roughly 3.8–4.4 s.
- **Eligible-prey work:** the remainder, about 2.3 s.

**A standalone replica undercounts by half.** Before these measurements, a Numba replica of Phase
1a's control flow, run on each step's end-of-step state through per-cell index lists, gave 3.2 s
(12.2%).
- **Its loop count is close to item 3's:** 1.25 billion pair tests over the run against 1.33
  billion. The replica skips egg and empty predators.
- **What it leaves out:** the accessibility lookup and the prey-biomass load for eligible pairs,
  and it compiles in isolation.

The in-situ repeats are the measurement to use.

## What it means for item 6

**Ceilings.** Removing Phase 1a entirely would save about 20% of a one-thread run. A prefilter can
remove at most the rejected pairs' testing, about 11–13%.

**A length prefilter is the obvious candidate.** Length is fixed during the mortality step, so a
per-cell index sorted by length could be built once per cell per step. To stay
performance-preserving rather than statistically equivalent, it must keep:
- the same eligible set: the abundance test still runs per visit, because abundances fall within
  the step;
- the original visit order, because `total_available` is a running sum and Phase 2 walks prey in
  buffer order;
- `n_local` and the per-cell random stream, unchanged.

**Why that is hard.** Restoring the original order costs a sort of the eligible subset per predator
visit, or a mask scan over the cell. Either could eat much of an 11–13% ceiling at a median of 26
schools per cell. September argued, without measuring, that sorting loses to a flat scan at EEC's
median of 15. Unmeasured here.

**The next number:** a prototype's cost per predator visit, measured the same way, before any
change to the kernel is designed in earnest.

## Reproduce

`scripts/baltic_phase1a_cost.py`:
- `apply-double <scratch worktree> [--passes N] [--test-only]` applies the probe;
- `time <repo>` times one arm;
- interleave the arms.

The script refuses this repository's own checkout, any worktree on a branch, and any worktree with
uncommitted changes. See its docstring.
