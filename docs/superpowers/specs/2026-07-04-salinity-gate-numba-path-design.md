# Salinity gate on the Numba movement path — design

**Date:** 2026-07-04
**Status:** approved design, pre-implementation. Follow-up #1 to the merged salinity-gate prototype.
**Author:** brainstormed with the user
**Related:** `docs/superpowers/specs/2026-07-04-salinity-gated-cod-occupancy-design.md` (the prototype whose Python-path behavior this must match exactly); `osmose/engine/processes/movement.py` (`_map_move_batch_numba`, `_map_move_school`, `movement()`, `_movement_salinity_weight`).

## 1. Motivation

The merged salinity-gate prototype applies only on the **Python movement fallback**. Production runs (`_HAS_NUMBA=True`, `flat_map_data` populated) take the JIT batch kernel `_map_move_batch_numba`, where the gate currently does nothing — enabling it emits a `RuntimeWarning` and is a no-op. This change ports the *exact same gate behavior* into the Numba kernel so the gate applies on real Baltic runs, and removes the now-redundant warning.

## 2. Behavior target (fixed — must mirror the Python path)

For each **gated** school (gate enabled AND the school's species is in the gate mask):
- **Step 3a (placement):** rejection-sample with per-cell weight `wmap = current_map · sal_w`, using `wmax = max(wmap)` as the acceptance normalizer → occupancy ∝ `wmap`.
- **Step 3b (random walk):** select among accessible cells (`wmap > 0`) **weighted by `wmap`**, not uniform.
- **All-zero guard:** if the school's map has no positive-finite `wmap` cell, fall back to the ungated path (original `current_map`, cached `max_p`) so cod is never annihilated.

For **ungated** schools (gate off, or species not in the mask, or all-zero-guard fallback): the code path is **byte-for-byte the original kernel** — same expressions, same `np.random` calls, same RNG consumption — so `gate_active=False` reproduces today's Numba output bit-for-bit.

**Cross-path note:** the Numba kernel uses numba's `np.random` (seeded from the caller's Generator), a *different* RNG stream from the Python path's `Generator`. So Numba-vs-Python outputs are NOT bit-identical (they never were — separate code paths). The parity guarantee is *per-path*: Numba-gate-off == today's Numba output. Numba-vs-Python are compared *statistically* (same ∝-weight occupancy distribution), not bit-for-bit.

## 3. Goals and non-goals

**Goals**
- Gate applies on the Numba batch path with behavior mirroring the Python path.
- Inert-by-default and bit-identical-when-off on the Numba path.
- Remove the redundant `RuntimeWarning` and its test.

**Non-goals**
- Real CMEMS `so` forcing and the full Baltic run (still follow-ups; this change makes the gate *work* on the hot path, using the same synthetic/constant field the prototype supports).
- Per-distinct-map-per-step normalizer precompute (a perf optimization). This spec computes `wmax` / the 3b weight-sum **per gated school** (O(n_cells) each), matching the Python path; optimize later if profiling warrants.
- Any change to the Python path or the gate's config/schema/loader.

## 4. Threading (reuse the Python seam)

In `movement()`'s **Numba branch** (`if _HAS_NUMBA and flat_map_data is not None:`), compute the weight grid once with the existing seam and build stable-typed args:

```python
            sal_w = _movement_salinity_weight(config, grid, step)
            if sal_w is not None:
                gate_active = True
                sal_w_arr = sal_w                                  # (ny, nx) float64
                gate_species = config.salinity_gate_species        # (n_sp,) bool, not None when on
            else:
                gate_active = False
                sal_w_arr = np.zeros((1, 1), dtype=np.float64)     # dummy; never indexed when off
                gate_species = np.zeros(config.n_species, dtype=np.bool_)
```

Pass `sal_w_arr, gate_active, gate_species` as three new trailing args to the `_map_move_batch_numba(...)` call (after the existing `new_out`). The dummy `(1,1)` array keeps the njit signature's arg *types* stable (numba caches on types, not shapes); it is never indexed because `gate_active=False` makes every school ungated.

Delete the `RuntimeWarning` block (currently the first statement inside this branch) — the gate is no longer a no-op here.

## 5. Kernel changes (`_map_move_batch_numba`)

Add three trailing parameters to the signature: `sal_w, gate_active, gate_species` (after `out_is_out`).

Inside the per-school loop, after `current_map`/`max_p` are resolved (after the null-map checks), compute the per-school gate state + normalizer:

```python
            gated = gate_active and gate_species[sp]
            use_gate = gated
            wmax = 0.0
            if gated:
                for jj in range(ny):
                    for ii in range(nx):
                        v = current_map[jj, ii] * sal_w[jj, ii]
                        if not np.isnan(v) and v > wmax:
                            wmax = v
                if wmax <= 0.0:
                    use_gate = False   # all-zero guard: behave exactly ungated
```

**Step 3a (placement)** — branch on `use_gate`, keeping the ungated branch identical to today:

```python
            if not same_map[k] or cell_x[idx] < 0:
                placed = False
                for _ in range(10_000):
                    flat_idx = np.random.randint(0, n_cells)
                    j = flat_idx // nx
                    i = flat_idx % nx
                    if use_gate:
                        proba = current_map[j, i] * sal_w[j, i]
                        if proba > 0 and not np.isnan(proba):
                            if proba >= np.random.random() * wmax:
                                out_cx[idx] = i
                                out_cy[idx] = j
                                out_is_out[idx] = False
                                placed = True
                                break
                    else:
                        proba = current_map[j, i]
                        if proba > 0 and not np.isnan(proba):
                            if max_p == 0.0 or proba >= np.random.random() * max_p:
                                out_cx[idx] = i
                                out_cy[idx] = j
                                out_is_out[idx] = False
                                placed = True
                                break
                if not placed:
                    out_cx[idx] = -1
                    out_cy[idx] = -1
                    out_is_out[idx] = True
                continue
```

**Step 3b (random walk)** — count accessible cells (and, if gated, accumulate total weight `W`), then select:

```python
            cx_k = cell_x[idx]
            cy_k = cell_y[idx]
            wr = walk_range[sp]
            y_lo = max(0, cy_k - wr)
            y_hi = min(ny, cy_k + wr + 1)
            x_lo = max(0, cx_k - wr)
            x_hi = min(nx, cx_k + wr + 1)

            n_accessible = 0
            W = 0.0
            for yi in range(y_lo, y_hi):
                for xi in range(x_lo, x_hi):
                    if ocean_mask[yi, xi] and current_map[yi, xi] > 0 and not np.isnan(current_map[yi, xi]):
                        if use_gate:
                            wv = current_map[yi, xi] * sal_w[yi, xi]
                            if wv > 0 and not np.isnan(wv):
                                n_accessible += 1
                                W += wv
                        else:
                            n_accessible += 1

            if n_accessible == 0:
                out_cx[idx] = cx_k
                out_cy[idx] = cy_k
                out_is_out[idx] = False
                continue

            if use_gate:
                r = np.random.random() * W
                acc = 0.0
                sel_x = cx_k          # safety default (overwritten below)
                sel_y = cy_k
                found = False
                for yi in range(y_lo, y_hi):
                    for xi in range(x_lo, x_hi):
                        if ocean_mask[yi, xi] and current_map[yi, xi] > 0 and not np.isnan(current_map[yi, xi]):
                            wv = current_map[yi, xi] * sal_w[yi, xi]
                            if wv > 0 and not np.isnan(wv):
                                acc += wv
                                sel_x = xi   # track last valid cell as float-rounding fallback
                                sel_y = yi
                                if acc >= r:
                                    found = True
                                    break
                    if found:
                        break
                out_cx[idx] = sel_x
                out_cy[idx] = sel_y
                out_is_out[idx] = False
            else:
                target = np.random.randint(0, n_accessible)
                count = 0
                for yi in range(y_lo, y_hi):
                    for xi in range(x_lo, x_hi):
                        if ocean_mask[yi, xi] and current_map[yi, xi] > 0 and not np.isnan(current_map[yi, xi]):
                            if count == target:
                                out_cx[idx] = xi
                                out_cy[idx] = yi
                                out_is_out[idx] = False
                            count += 1
                            if count > target:
                                break
                    if count > target:
                        break
```

The ungated 3b branch is the verbatim original (`randint(0, n_accessible)` + walk-to-target). The gated branch's `sel_x/sel_y` default + last-valid tracking guarantees a selection even under floating-point rounding (`r = random()·W < W`, so `acc` reaches `W ≥ r` at the final accessible cell).

## 6. Testing

- **Numba kernel graded behavior (direct):** call `_map_move_batch_numba` directly with a constructed flattened single presence map + a 3-band synthetic `sal_w` (weights 0 / 0.5 / 1.0 by column, as in the Python-path test), `gate_active=True`, `gate_species=[True]`. Over many schools/draws assert: excluded columns get 0 placements; high band ≈ 2× the mid band — for **both** a fresh-placement batch (`same_map=False`) and a random-walk batch (`same_map=True`, located schools). (`_HAS_NUMBA` must be true; skip with a clear reason if numba is unavailable in the env.)
- **Numba-vs-Python agreement (statistical):** run the identical gated config through both paths (Python via `flat_map_data=None`, Numba via populated `flat_map_data`) and assert the per-column occupancy *distribution* matches within tolerance — same ∝-weight shape, not bit-identical RNG.
- **Bit-identical-when-off (Numba path):** an engine run with the gate off is bit-identical to master on the Numba path (the existing `test_gate_off_is_bit_identical` already exercises the Numba path since eec_full uses it; confirm/keep it).
- **Warning removed:** delete the prototype's Numba-no-op warning test (`test_...numba...` that asserted the `RuntimeWarning`); the warning no longer exists.
- **Regression:** `tests/test_engine_map_movement.py` stays green (ungated Numba path unchanged).

## 7. Deliverables

- `osmose/engine/processes/movement.py` — kernel signature + gated 3a/3b logic; caller threading in the Numba branch; delete the `RuntimeWarning` block.
- `tests/test_salinity_gate.py` — Numba graded tests + Numba-vs-Python agreement test; remove the warning test.
