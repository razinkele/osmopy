---
name: project-movement-diet-perf-2026-06
description: "2026-06-24 engine perf — movement-mask vectorization + diet bincount = −12.6% (the \"closed\" surface had two missed hotspots)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

Two parity-preserving Python-side perf wins, merged to master 2026-06-24 (`c415c19`,
--no-ff). **eec_full 2.834s → 2.478s = −12.6%** (same-session baseline, drift ruled out per
the arc's trap). Both BIT-EXACT: 14/14 EEC `atol=0` + 8/8 BoB Java-parity unchanged + focused
guards (`tests/test_perf_movement_diet_vectorization.py`).

## What was done
- **`movement()`** (the dominant win): the per-timestep `np.array([config.movement_method[s]
  == "random"/"maps" for s in sp])` list comprehensions — an O(n_schools) Python loop with a
  dict lookup per school EVERY step — replaced by fancy-indexing precomputed per-species bool
  masks. Added `EngineConfig.movement_is_random` / `movement_is_maps` as `@cached_property`
  (EngineConfig is a non-frozen `@dataclass`, no `__slots__` → cached_property works; not a
  dataclass field so no `__eq__`/`__repr__` impact; calibration process-pool rebuilds config
  fresh per worker so no pickle issue). `movement()` does `config.movement_is_random[sp]`. A1/A2
  template. ~63× on that piece in isolation; far bigger in-engine than the micro-bench showed.
- **`aggregate_diet_by_species`** (output.py): `np.add.at(result, ids, diet)` → per-column
  `np.bincount(ids, weights=diet[:,c], minlength=n_pred)`. bincount accumulates per group in
  the SAME input order as add.at → bit-identical (verified 0.0 diff), ~3.7× faster. P2 template.
  Output-gated (diet tracking) so helps full-output runs, not calibration. Only the one call
  site simulate.py:1490 is hot (the sibling `aggregate_diet_all_predators` is not per-step).

## KEY durable fact (updates the "perf surface is closed" claim)
The post-v0.12.0 perf arc declared the surface "effectively closed" at the 2% gate, but it had
NOT profiled `movement()`'s own body or the live `aggregate_diet` add.at site — both were
unoptimized and together ~13% of wall. **The standing "re-profile before any new perf plan"
rule PAID OFF here** — re-profiling current master found real wins the arc missed. Lesson:
"closed" meant "the items we looked at are sub-gate," not "no headroom anywhere."

## Method that worked
Re-profile (cProfile, warm JIT) → confirm each hotspot's caller-gating + isolate it →
micro-measure speed AND bit-exact parity BEFORE implementing (don't ship a non-win) → TDD with
parity guards pinning output vs the old impl → run the atol=0/Java-parity suites → same-session
before/after benchmark (re-baseline master NOW to avoid the drift trap) → in-loop review
(reviewer empirically confirmed bincount summation order + pickle path). Related: closed perf
arc `docs/perf/2026-05-08-perf-arc-overview.md`; [[project-native-predation-kernel-spike]]
(the kernel — the other ~35%, boundary-bound).
