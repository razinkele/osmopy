---
name: Verify the perf-target code path is actually exercised by the benchmark fixture
description: A perf change in a function gated on a config flag won't measurably help if the benchmark fixture has the flag disabled — even if the code change is sound and ships value for other users
type: feedback
originSessionId: d2b7f4a5-d107-4042-a473-f491e81f4df1
---
When measuring a perf change on a benchmark fixture, **verify the target code path is actually executed** before trusting the measurement. A change to a function gated on a config flag that's `false` on the benchmark fixture cannot affect runtime on that fixture. Any apparent improvement is run-to-run noise or machine-state drift, not the change.

**Why:** P2 (PR #41) replaced `np.add.at` with `np.bincount` in `_collect_spatial_outputs` (`simulate.py:867-880`). I measured a 3.1 % wall-time improvement on eec_full 5-yr (2.973 s → 2.881 s) and shipped on that gate. But `_collect_spatial_outputs` is wrapped at `simulate.py:905-908` in `if config.output_spatial_enabled and grid is not None:` — and **eec_full has `output.spatial.enabled=false`**. The function is never called on the benchmark. The 92 ms delta was machine-state drift.

The change is still **bit-exact and ships real value** for users with `output.spatial.enabled=true`, so no harm done — but the gate was decorative on this fixture, and the cumulative-target table over-counted.

**How to apply:**
1. Before measuring, grep the call path of the function being changed for `if config.X` or `if some_flag:` gating. If gating is present, check the relevant config keys in the benchmark fixture's CSVs.
2. If the gate is false on the fixture, **either** (a) re-run on a fixture where the gate is true, **or** (b) flip the gate temporarily to measure (then revert before commit), **or** (c) ship the change as bit-exact-with-no-measurement and document explicitly that the 2 % gate was not exercised on the available fixtures.
3. cProfile output is the easiest way to spot this: if the target function doesn't appear at all in the top-N tottime list, it likely isn't being called. cProfile silence on a supposed hot path is a red flag.

**Concrete check for OSMOSE:**
- `output.spatial.enabled` (eec_full + baltic = false)
- `output.spatial.ltl.enabled` (eec_full + baltic = false)
- `ctx.fleet_state is not None` — economics module gating (DSVM call at simulate.py:1236)

When proposing a perf change in an output-collection or economics path, sanity-check these flags first.
