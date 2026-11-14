# Phase 1a scan: its in-situ cost on the Baltic, 2026-11-15

This is the first measurement for item 6 of `docs/osmopy-performance-optimization-plan.md`: the
share of one-thread mortality-kernel time spent in Phase 1a, the size-window scan over school prey
in `_apply_predation_numba`. Item 3 (`docs/perf/2026-11-14-baltic-cell-occupancy.md`) found
Baltic cells full and 88% of scanned prey failing the size test. It could not say what that costs
in time.

**Result: the scan costs about 6.7 s of 26.3 s.** That is about 25% of one-thread mortality time,
and about 20% of a one-thread 10-yr Baltic run. Item 6 has a real target. Whether a faster scan
can preserve the bit-exact baselines is a separate question (below).

## Method: run the scan twice, in place

**The edit.** In a scratch worktree, Phase 1a (the scan loop together with its `total_available`
and `n_prey` reset) is wrapped in `for _rep in range(2):`.
- **Why outputs stay the same.** The second pass recomputes the same buffers from unchanged inputs.
  So the run must be identical, and it is: the biomass hash matches in all six runs.
- **What the extra time measures.** The extra `mortality()` time is the scan's cost in place, with
  the real kernel's memory access, accessibility lookups and code generation.
- **Why it is, if anything, a lower bound.** The repeat runs on a warm cache, so it may understate
  the cold first pass.

**Setup:** warm 10-yr Baltic `run_in_memory`, one Numba thread (the calibration worker regime since
item 1), seed 42, master `dd196435`. Arms interleaved, each in its own interpreter.

| pair | control `mortality()` | doubled `mortality()` | added |
|---|---:|---:|---:|
| 1 | 26.10 s | 32.91 s | 6.81 s |
| 2 | 26.34 s | 32.98 s | 6.64 s |
| 3, through the committed `scripts/baltic_phase1a_cost.py` | 26.53 s | 33.08 s | 6.55 s |

- **Mean:** 6.67 s added, against a control mean of 26.32 s, so 25.3% of mortality time.
- **Share of the run:** the control's wall time averaged 33.11 s, so about 20% of it.

**A standalone replica undercounts by half.** Before this measurement, a Numba replica of Phase
1a's control flow, run on each step's end-of-step state through per-cell index lists, gave 3.2 s
(12.2%).
- **Its loop count is close to item 3's:** 1.25 billion pair tests over the run against 1.33
  billion. The replica skips egg and empty predators.
- **What it leaves out:** the accessibility lookup and the prey-biomass load for eligible pairs,
  and it compiles in isolation.

The in-situ doubling is the measurement to use.

## What it means for item 6

**Upper bound.** Removing the scan's cost entirely would save about 20% of a one-thread run.

**A length prefilter is the obvious candidate,** and length is fixed during the mortality step, so
a per-cell index sorted by length could be built once per cell per step. It would still have to
keep, to stay performance-preserving rather than statistically equivalent:
- the same eligible set: the abundance test still runs per visit, because abundances fall within
  the step;
- the original visit order, because `total_available` is a running sum and Phase 2 walks prey in
  buffer order;
- `n_local` and the per-cell random stream, unchanged.

**Why that is hard.** Restoring the original order costs a sort of the eligible subset per predator
visit, or a mask scan over the cell. Either could eat the saving at a median of 26 schools per
cell. September found exactly that at EEC's smaller cells. Unmeasured here.

**The next number:** a prototype's cost per visit, measured the same way, before any change to the
kernel is designed in earnest.

## Reproduce

`scripts/baltic_phase1a_cost.py`: `apply-double <scratch worktree>`, then `time` for each arm,
interleaved. It refuses to edit this repository's own checkout or any worktree with uncommitted
changes. See its docstring.
