# Salinity Gate on the Numba Movement Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the salinity-gated cod-occupancy gate into the production Numba movement batch kernel (`_map_move_batch_numba`), so the gate — currently Python-path only — applies on real runs, mirroring the merged Python-path behavior exactly.

**Architecture:** Add three trailing params (`sal_w, gate_active, gate_species`) to the njit kernel and per-school gated 3a/3b/guard logic; the ungated branch stays byte-for-byte the original (bit-identical when off). `movement()`'s Numba branch computes the weight grid via the existing `_movement_salinity_weight` seam and passes it in; the now-redundant `RuntimeWarning` is deleted. Numba and Python paths use different RNG streams, so they're validated *statistically* (same ∝-weight occupancy distribution), not bit-for-bit.

**Tech Stack:** Python 3.12, NumPy, Numba (`@njit(cache=True)`, nopython), pytest.

## Global Constraints

- Branch: `salinity-gate-numba-path` (already created).
- Run everything with `.venv/bin/python` (system `python` may not exist).
- Line length 100; lint `.venv/bin/ruff check osmose/ tests/` + `format --check` on touched files.
- **Inert-by-default & bit-identical-when-off (Numba path):** when `gate_active=False` (gate off, species not in mask, or all-zero-guard fallback), the kernel executes the *exact original statements* with the *same* `np.random` call sequence — no extra/reordered RNG draws.
- **Behavior mirrors the merged Python path** (`_map_move_school`) exactly: gated 3a rejection-sampling with `nanmax(wmap)` normalizer; gated 3b weighted selection ∝ `wmap` via cumulative-weight draw; all-zero guard falls back to ungated (original `current_map` + cached `max_p`).
- **Finite checks:** gated code uses `not np.isnan(...)` (mirrors Python `nanmax`/isnan). `inf` is unreachable given `current_map ∈ presence/[0,1]` and `sal_w = clip(...,0,1)`.
- **Numba-vs-Python:** compared statistically (N=4000, `atol=0.05` on per-column occupancy fractions), never bit-for-bit.
- Spike scope unchanged: real CMEMS `so` + full Baltic run remain out of scope.

---

## File Structure

- `osmose/engine/processes/movement.py` — kernel signature + gated 3a/3b/guard logic (`_map_move_batch_numba`); caller threading in `movement()`'s Numba branch; delete the `RuntimeWarning` block.
- `tests/test_movement_numba.py` — extend `_call_numba` with 3 optional off-default kwargs (keeps the 5 existing direct-kernel tests green; shared builder for new tests).
- `tests/test_salinity_gate.py` — new Numba kernel tests (graded 3a/3b, all-zero guard, gated stranding, numba-vs-python agreement); remove `test_gate_enabled_warns_on_numba_path`.

---

## Task 1: Extend `_call_numba` + Numba kernel gate + caller threading

**Files:**
- Modify: `tests/test_movement_numba.py` (`_call_numba`, ~line 147)
- Modify: `osmose/engine/processes/movement.py` (`_map_move_batch_numba` signature + body ~line 405-509; `movement()` Numba branch ~line 303-343)
- Test: `tests/test_salinity_gate.py`

**Interfaces:**
- Consumes: `_movement_salinity_weight(config, grid, step) -> NDArray | None` (already in movement.py); `salinity_weight` (salinity_gate.py).
- Produces: `_map_move_batch_numba(..., out_is_out, sal_w, gate_active, gate_species)` — three new trailing njit params. `gate_active=False` ⇒ bit-identical to today.

- [ ] **Step 1: Extend the `_call_numba` helper (keeps existing tests green, gives new tests a builder)**

In `tests/test_movement_numba.py`, change `_call_numba` to accept three optional off-default kwargs and pass them to the kernel:

```python
def _call_numba(
    seed,
    school_indices,
    map_idx,
    same_map,
    cx,
    cy,
    sp_ids,
    all_maps,
    all_max_proba,
    all_is_null,
    sp_offsets,
    ocean_mask,
    walk_range,
    ny,
    nx,
    sal_w=None,
    gate_active=False,
    gate_species=None,
):
    """Helper to call _map_move_batch_numba with in-place output arrays."""
    out_cx = cx.copy()
    out_cy = cy.copy()
    out_is_out = np.zeros(len(cx), dtype=np.bool_)
    if sal_w is None:
        sal_w = np.zeros((1, 1), dtype=np.float64)
    if gate_species is None:
        gate_species = np.zeros(1, dtype=np.bool_)  # never indexed when gate_active=False
    _map_move_batch_numba(
        seed,
        school_indices,
        map_idx,
        same_map,
        out_cx,
        out_cy,
        sp_ids,
        all_maps,
        all_max_proba,
        all_is_null,
        sp_offsets,
        ocean_mask,
        walk_range,
        ny,
        nx,
        out_cx,
        out_cy,
        out_is_out,
        sal_w,
        gate_active,
        gate_species,
    )
    return out_cx, out_cy, out_is_out
```

- [ ] **Step 2: Run the existing numba tests to confirm they FAIL (signature mismatch, expected)**

Run: `.venv/bin/python -m pytest tests/test_movement_numba.py -q`
Expected: FAIL — the kernel doesn't yet accept the 3 new args (`_map_move_batch_numba` gets 21 args, signature has 18). This is the RED that Step 4 turns green.

- [ ] **Step 3: Write the four kernel-behavior tests (they will fail until the kernel logic exists)**

Append to `tests/test_salinity_gate.py`:

```python
import numpy as np
import pytest

numba = pytest.importorskip("numba", reason="numba unavailable")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_numba import _call_numba  # noqa: E402


def _three_band_sal_w(ny=5, nx=6):
    # cols 0-1 = 0.0 (weight 0), cols 2-3 = 0.5, cols 4-5 = 1.0
    w = np.zeros((ny, nx), dtype=np.float64)
    w[:, 2:4] = 0.5
    w[:, 4:6] = 1.0
    return w


def _batch_placement(sal_w, n=4000, seed=0, same_map=False, cx0=-1, cy0=-1, walk=9):
    ny, nx = sal_w.shape
    all_maps = np.ones((1, ny, nx), dtype=np.float64)   # presence map (max_proba 0.0)
    out_cx, out_cy, is_out = _call_numba(
        seed,
        np.arange(n, dtype=np.int32),          # school_indices
        np.zeros(n, dtype=np.int32),           # map_idx -> map 0
        np.full(n, same_map, dtype=np.bool_),  # same_map
        np.full(n, cx0, dtype=np.int32),       # cx
        np.full(n, cy0, dtype=np.int32),       # cy
        np.zeros(n, dtype=np.int32),           # sp_ids -> species 0
        all_maps,
        np.array([0.0]),                       # all_max_proba (presence)
        np.array([False]),                     # all_is_null
        np.array([0], dtype=np.int32),         # sp_offsets
        np.ones((ny, nx), dtype=np.bool_),     # ocean_mask
        np.array([walk], dtype=np.int32),      # walk_range
        ny,
        nx,
        sal_w=sal_w,
        gate_active=True,
        gate_species=np.array([True]),
    )
    return out_cx, out_cy, is_out


def test_numba_gated_placement_graded():
    out_cx, _, is_out = _batch_placement(_three_band_sal_w(), same_map=False)
    assert not is_out.any()
    cols = np.bincount(out_cx, minlength=6)
    assert cols[0] == 0 and cols[1] == 0
    high = cols[4] + cols[5]
    mid = cols[2] + cols[3]
    assert mid > 0 and high > 0
    assert high / mid == pytest.approx(2.0, rel=0.15)


def test_numba_gated_random_walk_graded():
    # Located schools on the same map, started at col 3 (mid band), walk spans cols 2-5.
    out_cx, _, is_out = _batch_placement(
        _three_band_sal_w(), same_map=True, cx0=3, cy0=2, walk=3
    )
    assert not is_out.any()
    cols = np.bincount(out_cx, minlength=6)
    assert cols[0] == 0 and cols[1] == 0
    high = cols[4] + cols[5]
    mid = cols[2] + cols[3]
    assert mid > 0 and high > 0
    assert high / mid == pytest.approx(2.0, rel=0.2)


def test_numba_gated_all_zero_guard_places_not_annihilated():
    # sal_w all zero over the whole map -> wmax<=0 -> fall back to ungated placement.
    ny, nx = 5, 6
    sal_w = np.zeros((ny, nx), dtype=np.float64)
    out_cx, out_cy, is_out = _batch_placement(sal_w, n=200, same_map=False)
    assert not is_out.any()                    # cod is placed, never annihilated


def test_numba_gated_local_stranding_stays_in_place():
    # Gated located school whose walk window is all-zero-weight but map has weight elsewhere.
    ny, nx = 5, 8
    sal_w = np.zeros((ny, nx), dtype=np.float64)
    sal_w[:, 6:8] = 1.0                        # weight only far from the school
    # school located at (cx=1, cy=2), walk_range=1 -> window cols 0-2 all weight 0
    out_cx, out_cy, is_out = _batch_placement(
        sal_w, n=50, same_map=True, cx0=1, cy0=2, walk=1
    )
    assert not is_out.any()
    assert np.all(out_cx == 1) and np.all(out_cy == 2)   # stays in place
```

- [ ] **Step 4: Run the new tests to confirm they FAIL**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -k "numba_gated" -v`
Expected: FAIL — `_map_move_batch_numba` still has the old 18-arg signature (TypeError on the 21-arg call), OR (after Step 5's signature change) wrong results because the gate logic isn't implemented. Either way RED.

- [ ] **Step 5: Implement the kernel gate**

In `osmose/engine/processes/movement.py`, change `_map_move_batch_numba`'s signature to add three trailing params after `out_is_out`:

```python
        out_cx,
        out_cy,
        out_is_out,
        sal_w,
        gate_active,
        gate_species,
    ):
```

Then, in the per-school loop, immediately after `max_p = all_max_proba[global_map_idx]` (and before `if not same_map[k] or cell_x[idx] < 0:`), insert the gate block:

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

Replace the Step 3a placement block with the `use_gate`-branched version (ungated branch is the verbatim original):

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

Replace the Step 3b random-walk block (the `cx_k = cell_x[idx]` block through the end of the loop body) with:

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
                sel_x = cx_k
                sel_y = cy_k
                found = False
                for yi in range(y_lo, y_hi):
                    for xi in range(x_lo, x_hi):
                        if ocean_mask[yi, xi] and current_map[yi, xi] > 0 and not np.isnan(current_map[yi, xi]):
                            wv = current_map[yi, xi] * sal_w[yi, xi]
                            if wv > 0 and not np.isnan(wv):
                                acc += wv
                                sel_x = xi
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

- [ ] **Step 6: Thread the weight grid in `movement()`'s Numba branch + delete the warning**

In `osmose/engine/processes/movement.py`, in the `if _HAS_NUMBA and flat_map_data is not None:` branch: (a) delete the `RuntimeWarning` block (the `if config.salinity_gate_enabled: ... warnings.warn(...)` block that is currently the first thing in this branch); (b) before the `_map_move_batch_numba(...)` call, compute the gate args:

```python
            sal_w = _movement_salinity_weight(config, grid, step)
            if sal_w is not None:
                sal_w_arr = sal_w
                gate_active = True
                gate_species_arr = config.salinity_gate_species
            else:
                sal_w_arr = np.zeros((1, 1), dtype=np.float64)
                gate_active = False
                gate_species_arr = np.zeros(config.n_species, dtype=np.bool_)
```

Then append the three args to the `_map_move_batch_numba(...)` call, after `new_out` (the current last argument):

```python
                new_out,
                sal_w_arr,
                gate_active,
                gate_species_arr,
            )
```

- [ ] **Step 7: Run the kernel-behavior + existing numba + movement suites**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -k "numba_gated" tests/test_movement_numba.py tests/test_engine_map_movement.py -q`
Expected: PASS — the 4 gated tests pass, the 5 existing `test_movement_numba` tests pass (off-defaults keep them bit-identical), and the movement regression stays green.

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_movement_numba.py tests/test_salinity_gate.py
git commit -m "feat: apply salinity gate on the Numba movement batch path"
```

---

## Task 2: Numba-vs-Python agreement + warning-test cleanup

**Files:**
- Test: `tests/test_salinity_gate.py`

**Interfaces:**
- Consumes: the gated `_map_move_batch_numba` (Task 1) and the Python-path `_map_move_school` (already merged).

- [ ] **Step 1: Write the numba-vs-python statistical agreement test**

Append to `tests/test_salinity_gate.py`. Reuse the existing `_uniform_map_set` + `_draw_columns` helpers (Python-path draws) already in this file and the `_batch_placement` helper (Task 1) for the Numba draws:

```python
def test_numba_python_agreement_statistical():
    sal_w = _three_band_sal_w()
    # Numba path: per-column occupancy fractions.
    ncx, _, _ = _batch_placement(sal_w, n=4000, seed=0, same_map=False)
    frac_numba = np.bincount(ncx, minlength=6) / 4000.0
    # Python path: same fixture via the existing _draw_columns helper.
    pcols = _draw_columns(sal_w, n=4000)          # from the Python-path tests in this file
    frac_python = pcols / pcols.sum()
    np.testing.assert_allclose(frac_numba, frac_python, atol=0.05)
```

If `_draw_columns` in this file uses a `(5, 6)` grid with a different band layout, adjust `_three_band_sal_w`/the call so both paths use the *same* grid and bands; the point is identical inputs, compared as fractions. Read the existing `_draw_columns` / `_uniform_map_set` first and match their grid dims.

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py::test_numba_python_agreement_statistical -v`
Expected: PASS — both paths yield the same ∝-weight column distribution within `atol=0.05` (cols 0–1 ≈ 0, high ≈ 2× mid).

- [ ] **Step 3: Remove the now-obsolete warning test**

The Numba path no longer emits the no-op `RuntimeWarning` (deleted in Task 1). Delete `test_gate_enabled_warns_on_numba_path` from `tests/test_salinity_gate.py` entirely.

- [ ] **Step 4: Confirm the bit-identical-off test still passes**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py::test_gate_off_is_bit_identical -v`
Expected: PASS — the gate-off engine run (real Numba path on eec_full) is unchanged. If it now FAILS, the ungated kernel path is not byte-for-byte the original — revisit Task 1 Step 5.

- [ ] **Step 5: Commit**

```bash
git add tests/test_salinity_gate.py
git commit -m "test: numba-vs-python agreement + drop obsolete numba no-op warning test"
```

---

## Task 3: Full-suite gate + lint

**Files:** none new (verification).

- [ ] **Step 1: Lint and format**

Run: `.venv/bin/ruff check osmose/ tests/` and `.venv/bin/ruff format --check osmose/ tests/`
Expected: clean on touched files. Fix findings only on the touched files (do not reformat unrelated pre-existing files; note any already-unformatted ones, e.g. a known `tests/test_config_migration_440.py`).

- [ ] **Step 2: Run the feature + related suites**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py tests/test_movement_numba.py tests/test_engine_map_movement.py tests/test_engine_config_validation.py -q`
Expected: all PASS.

- [ ] **Step 3: Confirm inert-by-default across bundled configs**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -q`
Expected: PASS.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint + final verification for numba salinity gate"
```

---

## Self-Review Notes (author, against the spec)

- **Spec §4 threading:** Task 1 Step 6 (compute `sal_w` via `_movement_salinity_weight`, dummy `(1,1)` + all-False `(n_species,)` mask when off, append 3 args, delete warning).
- **Spec §5 kernel (3a nanmax / 3b cumulative-weight / all-zero guard / ungated byte-identical):** Task 1 Step 5, code transcribed verbatim from the spec.
- **Spec §6 tests:** graded-3a/3b + all-zero-guard + gated-stranding (Task 1 Step 3); numba-vs-python agreement + warning-test removal + bit-identical-off (Task 2); `_call_numba` extension keeping 5 direct-kernel tests green (Task 1 Step 1).
- **Spec §7 deliverables:** all three files covered (movement.py, test_movement_numba.py, test_salinity_gate.py).
- **Known integration point flagged, not a placeholder:** Task 2 Step 1 names that the implementer must read the existing `_draw_columns`/`_uniform_map_set` in `tests/test_salinity_gate.py` and match grid dims so both paths use identical inputs — a reuse instruction, not undefined behavior.
- **Insertion points are line-approximate:** the plan says "after `max_p = all_max_proba[global_map_idx]`" and "after `new_out`" rather than pinning line numbers, since the implementer edits the live file; anchors are exact code strings.
