# Native Predation-Kernel Feasibility Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throwaway harness that measures whether a hand-written C port of `_apply_predation_numba` beats Numba's LLVM codegen on the leaf math (boundary-free), with verified parity, producing a go/no-go artifact.

**Architecture:** A standalone harness under `scripts/spikes/native_predation/`. It (1) captures the real inputs to the production cell-loop kernel from a live `eec_full` run by monkeypatching the dispatched njit function, (2) reconstructs exact leaf-call arguments for cells at p10/p50/p95 of the call-weighted `n_local` distribution, (3) ports the leaf to C compiled via `cffi`, (4) verifies C-vs-Numba parity to f64 op-order rounding, and (5) benchmarks boundary-free math throughput (an `@njit` driver looping the real leaf vs a C loop), emitting a `docs/perf/` artifact with a pre-registered 1.3× gate.

**Tech Stack:** Python 3.12, NumPy, Numba 0.65 (existing), `cffi` 2.0 + `gcc` 13.3 (both already present — no new dependency), pytest for the two harness self-tests.

## Global Constraints

- **No new system/runtime dependency** — `cffi` + `gcc` only; Rust is not available and must not be introduced.
- **Spike is throwaway** — everything lives under `scripts/spikes/native_predation/`; nothing is imported by `osmose/`, added to the engine, the suite (except the two self-tests under the spike's own `tests/`), or CI.
- **Source of truth for the algorithm:** `osmose/engine/processes/mortality.py:881-1053` (`_apply_predation_numba` body). The C port is a line-faithful transcription; **reduction order must match** (school prey in `cell_indices` order, then resource prey in `r` order; apportionment in insertion order) or parity fails.
- **Provenance guards are mandatory and run first** (worktree `__file__`, `_HAS_NUMBA is True`, captured flag config) — two documented benchmark traps (`docs/perf/2026-05-08-perf-arc-overview.md:103-104`).
- **Production kernel is `_mortality_all_cells_parallel`** (`parallel=True` default, mortality.py:1806/1985) — capture must patch *that* function, not the serial one.
- **Run everything with `PYTHONPATH` pinned to this worktree** so the imported `osmose` is the worktree's, not an installed copy.
- **Parity bar:** ≤ 1e-12 max relative difference on every mutated array (`inst_abd`, `rsc_biomass`, `n_dead`, `preyed_biomass`, `pred_success_rate`, and gated `diet_matrix`/`tl_weighted_sum`). Larger = correctness failure, not a tunable.
- **Gate metric:** call-count-weighted C-vs-Numba boundary-free math-throughput ratio on the **portable `-O3`** build (not `-march=native`). PASS (≥1.3×) authorizes only a follow-on integration spike, nothing more.
- Spec: `docs/superpowers/specs/2026-06-24-native-predation-kernel-spike-design.md`.

---

### Task 1: Harness scaffold + provenance guards

**Files:**
- Create: `scripts/spikes/native_predation/__init__.py` (empty)
- Create: `scripts/spikes/native_predation/provenance.py`
- Create: `scripts/spikes/native_predation/README.md`
- Test: `scripts/spikes/native_predation/tests/test_provenance.py`
- Create: `scripts/spikes/native_predation/tests/__init__.py` (empty)

**Interfaces:**
- Produces: `assert_provenance(worktree_root: Path) -> dict` — runs the three guards; returns `{"mortality_file": str, "has_numba": True, "numba_version": str}`; raises `RuntimeError` on any guard failure. `capture_flag_config(ctx) -> dict[str, bool]` — reads `diet_enabled`, `tl_tracking`, `use_stage_access`, `has_access` from a `SimulationContext`/predation-param object and returns them as plain bools.

- [ ] **Step 1: Write the failing test**

```python
# scripts/spikes/native_predation/tests/test_provenance.py
from pathlib import Path

import pytest

from scripts.spikes.native_predation.provenance import assert_provenance

WORKTREE = Path(__file__).resolve().parents[4]  # .../native_predation/tests -> repo root


def test_assert_provenance_passes_in_worktree():
    info = assert_provenance(WORKTREE)
    assert info["has_numba"] is True
    assert str(WORKTREE) in info["mortality_file"]
    assert info["numba_version"]  # non-empty


def test_assert_provenance_rejects_wrong_root():
    with pytest.raises(RuntimeError, match="not under worktree"):
        assert_provenance(Path("/nonexistent/elsewhere"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/native_predation/tests/test_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (provenance.py absent).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/spikes/native_predation/provenance.py
"""Provenance & call-path guards. Run FIRST, fail loudly.

Guards the two documented benchmark traps (perf-arc-overview.md:103-104):
importing the wrong osmose, and timing the non-numba dead-code path.
"""
from __future__ import annotations

from pathlib import Path


def assert_provenance(worktree_root: Path) -> dict:
    import numba

    from osmose.engine.processes import mortality

    mfile = Path(mortality.__file__).resolve()
    root = Path(worktree_root).resolve()
    if root not in mfile.parents and root != mfile.parent.parent.parent.parent:
        # mortality.py lives at <root>/osmose/engine/processes/mortality.py
        if str(root) not in str(mfile):
            raise RuntimeError(
                f"mortality.py resolved to {mfile}, not under worktree {root}. "
                "Set PYTHONPATH to the worktree before running the spike."
            )
    if not getattr(mortality, "_HAS_NUMBA", False):
        raise RuntimeError(
            "mortality._HAS_NUMBA is False — the per-cell Python path is dead "
            "code in production; timing it measures the wrong kernel."
        )
    return {
        "mortality_file": str(mfile),
        "has_numba": True,
        "numba_version": numba.__version__,
    }


def capture_flag_config(diet_enabled: bool, tl_tracking: bool,
                        use_stage_access: bool, has_access: bool) -> dict[str, bool]:
    return {
        "diet_enabled": bool(diet_enabled),
        "tl_tracking": bool(tl_tracking),
        "use_stage_access": bool(use_stage_access),
        "has_access": bool(has_access),
    }
```

Also write `README.md` (one paragraph: throwaway spike, how to run `run_spike.py`, pointer to the spec) and the two empty `__init__.py` files.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/native_predation/tests/test_provenance.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/native_predation/
git commit -m "spike(perf): native-predation harness scaffold + provenance guards"
```

---

### Task 2: Capture cell-loop pre-state from a live eec_full run

**Files:**
- Create: `scripts/spikes/native_predation/capture.py`
- Test: (manual smoke — produces a fixture; no unit test, validated by Task 3's loader)

**Interfaces:**
- Consumes: `assert_provenance`, `capture_flag_config` (Task 1).
- Produces: `capture_cellloop(config_path: Path, capture_call_index: int, out_dir: Path) -> Path` — runs `eec_full`, monkeypatches `mortality._mortality_all_cells_parallel` to deep-copy all positional args on the `capture_call_index`-th invocation (0-based, after warmup), then restores. Writes `cellloop.npz` (every captured array, keyed by the kernel's parameter name) + `meta.json` (`{"provenance": ..., "flags": ..., "arg_order": [param names in positional order], "n_resources": int, "n_cells": int}`). Returns the `.npz` path. The captured arrays are the **pre-state** (deep-copied before the real kernel runs).

**Background the implementer needs:** `mortality()` (mortality.py:1798) dispatches at mortality.py:1985 via `_batch_fn = _mortality_all_cells_parallel if parallel else _mortality_all_cells_numba`, reading the module global by name → patching `mortality._mortality_all_cells_parallel` is picked up at call time. The kernel's positional parameter order is mortality.py:1242-1286 for the serial kernel; the **parallel** kernel `_mortality_all_cells_parallel` (mortality.py:1411) takes the **same parameters plus a leading `rng_seed`** — read its signature to get the exact order, and record that order in `meta.json["arg_order"]`.

- [ ] **Step 1: Implement capture**

```python
# scripts/spikes/native_predation/capture.py
"""Capture the production cell-loop kernel's pre-state from a live eec_full run.

The leaf is called from inside @njit and cannot be intercepted directly, but
its args are (almost all) the arrays the cell-loop kernel receives — which IS
patchable because mortality() dispatches to it by module-global name.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid as G
from osmose.engine.processes import mortality as M
from osmose.engine.simulate import simulate

from .provenance import assert_provenance


def capture_cellloop(config_path: Path, capture_call_index: int, out_dir: Path,
                     worktree_root: Path) -> Path:
    info = assert_provenance(worktree_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_fn = M._mortality_all_cells_parallel
    params = list(inspect.signature(real_fn.py_func).parameters)  # njit -> .py_func
    state = {"n": 0, "captured": None}

    def wrapper(*args):
        if state["n"] == capture_call_index and state["captured"] is None:
            state["captured"] = {
                name: (np.copy(a) if isinstance(a, np.ndarray) else a)
                for name, a in zip(params, args)
            }
        state["n"] += 1
        return real_fn(*args)

    M._mortality_all_cells_parallel = wrapper
    try:
        reader = OsmoseConfigReader()
        raw = reader.read(config_path)
        raw["simulation.time.nyear"] = "2"  # enough to pass warmup and reach capture_call_index
        # Calibration workload: diet + meanTL OFF (spec §4.0) — calibration is the
        # stated motivation and diet aggregation is output-gated/skipped there, so the
        # gate must measure the leaf as calibration exercises it. The captured flags are
        # recorded in meta.json regardless; a second default-on capture is optional.
        raw["output.diet.composition.enabled"] = "false"
        raw["output.meantl.enabled"] = "false"
        cfg = EngineConfig.from_dict(raw)
        grid = G.from_netcdf(config_path.parent / raw["grid.netcdf.file"],
                             mask_var=raw.get("grid.var.mask", "mask"))
        simulate(cfg, grid, np.random.default_rng(42))
    finally:
        M._mortality_all_cells_parallel = real_fn

    cap = state["captured"]
    if cap is None:
        raise RuntimeError(f"capture_call_index={capture_call_index} never reached "
                           f"(only {state['n']} cell-loop calls)")

    arrays = {k: v for k, v in cap.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in cap.items() if not isinstance(v, np.ndarray)}
    npz_path = out_dir / "cellloop.npz"
    np.savez(npz_path, **arrays)
    meta = {
        "provenance": info,
        "arg_order": params,
        "scalars": {k: (int(v) if isinstance(v, (int, np.integer)) else
                        bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                    for k, v in scalars.items()},
        "n_resources": int(scalars.get("n_resources", arrays["rsc_biomass"].shape[0])),
        "n_cells": int(len(arrays["boundaries"]) - 1),
        "flags": {
            "diet_enabled": bool(scalars.get("diet_enabled", False)),
            "tl_tracking": bool(scalars.get("tl_tracking", False)),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return npz_path


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[3]
    cfg = root / "data" / "eec_full" / "eec_all-parameters.csv"
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    p = capture_cellloop(cfg, idx, root / "scripts/spikes/native_predation/_fixtures", root)
    print("captured ->", p)
```

- [ ] **Step 2: Run the smoke capture**

Run: `PYTHONPATH=. .venv/bin/python -m scripts.spikes.native_predation.capture 200`
Expected: prints `captured -> .../_fixtures/cellloop.npz`; `_fixtures/meta.json` exists and `arg_order` lists the parallel kernel's params with `rng_seed` first; `flags.diet_enabled`/`tl_tracking` recorded.

- [ ] **Step 3: Verify the fixture is non-trivial**

Run: `PYTHONPATH=. .venv/bin/python -c "import numpy as np; d=np.load('scripts/spikes/native_predation/_fixtures/cellloop.npz'); print('schools', d['inst_abd'].shape, 'live', (d['inst_abd']>0).sum(), 'cells', len(d['boundaries'])-1)"`
Expected: non-zero live schools and >1 cell.

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/native_predation/capture.py
git commit -m "spike(perf): capture cell-loop pre-state via monkeypatched parallel kernel"
```

---

### Task 3: Leaf-arg reconstruction + cell selection (p10/p50/p95)

**Files:**
- Create: `scripts/spikes/native_predation/leaf_args.py`
- Test: `scripts/spikes/native_predation/tests/test_leaf_args.py`

**Interfaces:**
- Consumes: the `cellloop.npz` + `meta.json` from Task 2.
- Produces:
  - `load_capture(npz_path: Path) -> tuple[dict, dict]` → `(arrays, meta)`.
  - `select_cells(arrays) -> dict[str, int]` → cell indices at the call-weighted `n_local` distribution: keys `"p10"`, `"p50"`, `"p95"`, `"small"` (smallest non-empty cell). Weighting: each non-empty cell contributes weight = its `n_local`; percentiles taken over the per-call `n_local` values (i.e. each cell's `n_local` repeated `n_local` times).
  - `build_leaf_args(arrays, meta, cell: int) -> tuple[list, int]` → `(args, p_idx)` where `args` is the exact positional argument list for `_apply_predation_numba` (order per its signature, mortality.py:828-870), built on **fresh copies** of every mutated array, with freshly-allocated scratch (`np.empty(n_local + n_resources, ...)`), `cell_id=cell`, and `p_idx` the first school in the cell that will not early-return (feeding-age, `inst_abd>0`, `max_eatable>0`). Raises `ValueError` if no live predator exists in the cell.

**Leaf positional arg order (verbatim from mortality.py:828-870):** `p_idx, cell_indices, inst_abd, n_dead, species_id, length, weight, age_dt, first_feeding_age_dt, feeding_stage, pred_success_rate, preyed_biomass, trophic_level, size_ratio_min, size_ratio_max, ingestion_rate, fr_shape, fr_halfsat, n_dt_per_year, n_subdt, access_matrix, has_access, use_stage_access, prey_access_idx, pred_access_idx, rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl, rsc_access_rows, n_resources, n_species, cell_id, tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled, prey_type_buf, prey_id_buf, prey_eligible_buf, egg_retained`.

- [ ] **Step 1: Write the failing test**

```python
# scripts/spikes/native_predation/tests/test_leaf_args.py
from pathlib import Path

import numpy as np
import pytest

from scripts.spikes.native_predation.leaf_args import (
    build_leaf_args, load_capture, select_cells,
)

FIX = Path(__file__).resolve().parents[1] / "_fixtures" / "cellloop.npz"

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="run capture.py first")


def test_select_cells_returns_four_valid_indices():
    arrays, meta = load_capture(FIX)
    sel = select_cells(arrays)
    assert set(sel) == {"p10", "p50", "p95", "small"}
    n_cells = len(arrays["boundaries"]) - 1
    for c in sel.values():
        assert 0 <= c < n_cells


def test_build_leaf_args_isolates_one_call_and_does_not_mutate_capture():
    arrays, meta = load_capture(FIX)
    sel = select_cells(arrays)
    before = np.copy(arrays["inst_abd"])
    args, p_idx = build_leaf_args(arrays, meta, sel["p50"])
    assert len(args) == 41  # full leaf signature
    # building args must not touch the captured arrays (fresh copies)
    assert np.array_equal(arrays["inst_abd"], before)
    # p_idx is a real live predator
    assert arrays["inst_abd"][p_idx] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/native_predation/tests/test_leaf_args.py -v`
Expected: FAIL with `ImportError` (leaf_args.py absent).

- [ ] **Step 3: Implement**

```python
# scripts/spikes/native_predation/leaf_args.py
"""Reconstruct exact _apply_predation_numba args from a captured cell-loop pre-state."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEAF_ARG_ORDER = [
    "p_idx", "cell_indices", "inst_abd", "n_dead", "species_id", "length", "weight",
    "age_dt", "first_feeding_age_dt", "feeding_stage", "pred_success_rate",
    "preyed_biomass", "trophic_level", "size_ratio_min", "size_ratio_max",
    "ingestion_rate", "fr_shape", "fr_halfsat", "n_dt_per_year", "n_subdt",
    "access_matrix", "has_access", "use_stage_access", "prey_access_idx",
    "pred_access_idx", "rsc_biomass", "rsc_size_min", "rsc_size_max", "rsc_tl",
    "rsc_access_rows", "n_resources", "n_species", "cell_id", "tl_weighted_sum",
    "tl_tracking", "diet_matrix", "diet_enabled", "prey_type_buf", "prey_id_buf",
    "prey_eligible_buf", "egg_retained",
]
MUTATED = ["inst_abd", "n_dead", "pred_success_rate", "preyed_biomass",
           "rsc_biomass", "tl_weighted_sum", "diet_matrix"]


def load_capture(npz_path: Path):
    arrays = dict(np.load(npz_path))
    meta = json.loads((Path(npz_path).parent / "meta.json").read_text())
    return arrays, meta


def _n_local(arrays):
    b = arrays["boundaries"]
    return (b[1:] - b[:-1]).astype(np.int64)


def select_cells(arrays) -> dict[str, int]:
    nl = _n_local(arrays)
    nonempty = np.where(nl > 0)[0]
    if nonempty.size == 0:
        raise ValueError("no non-empty cells in capture")
    # call-weighted distribution: repeat each cell's n_local that many times
    weighted = np.repeat(nl[nonempty], nl[nonempty])
    p10, p50, p95 = np.percentile(weighted, [10, 50, 95])
    def nearest(target):
        return int(nonempty[np.argmin(np.abs(nl[nonempty] - target))])
    return {
        "p10": nearest(p10),
        "p50": nearest(p50),
        "p95": nearest(p95),
        "small": int(nonempty[np.argmin(nl[nonempty])]),
    }


def _scalar(meta, name, default):
    return meta.get("scalars", {}).get(name, default)


def build_leaf_args(arrays, meta, cell: int):
    b = arrays["boundaries"]
    start, end = int(b[cell]), int(b[cell + 1])
    if end <= start:
        raise ValueError(f"cell {cell} is empty")
    cell_indices = np.asarray(arrays["sorted_indices"][start:end], dtype=np.int32)
    n_local = end - start
    n_resources = int(_scalar(meta, "n_resources", arrays["rsc_biomass"].shape[0]))
    n_dt = int(_scalar(meta, "n_dt_per_year", arrays["fr_shape"].shape[0]))  # fallback
    n_subdt = int(_scalar(meta, "n_subdt", 1))

    inst_abd = arrays["inst_abd"]
    age_dt = arrays["age_dt"]
    ffa = arrays["first_feeding_age_dt"]
    weight = arrays["weight"]
    species_id = arrays["species_id"]
    ingestion = arrays["ingestion_rate"]

    p_idx = -1
    for q in cell_indices:
        q = int(q)
        if age_dt[q] < ffa[q] or inst_abd[q] <= 0:
            continue
        biomass = inst_abd[q] * weight[q]
        max_eatable = biomass * ingestion[species_id[q]] / (n_dt * n_subdt)
        if max_eatable > 0:
            p_idx = q
            break
    if p_idx < 0:
        raise ValueError(f"cell {cell} has no live feeding predator")

    fresh = {k: np.copy(arrays[k]) for k in MUTATED}
    scratch_n = n_local + n_resources
    built = {
        "p_idx": np.int32(p_idx),
        "cell_indices": cell_indices,
        "cell_id": np.int32(cell),
        "n_resources": np.int32(n_resources),
        "n_species": np.int32(_scalar(meta, "n_species", int(species_id.max()) + 1)),
        "n_dt_per_year": np.int32(n_dt),
        "n_subdt": np.int32(n_subdt),
        "has_access": bool(_scalar(meta, "has_access", True)),
        "use_stage_access": bool(_scalar(meta, "use_stage_access", False)),
        "tl_tracking": bool(meta["flags"]["tl_tracking"]),
        "diet_enabled": bool(meta["flags"]["diet_enabled"]),
        "prey_type_buf": np.empty(scratch_n, dtype=np.int32),
        "prey_id_buf": np.empty(scratch_n, dtype=np.int32),
        "prey_eligible_buf": np.empty(scratch_n, dtype=np.float64),
    }
    args = []
    for name in LEAF_ARG_ORDER:
        if name in built:
            args.append(built[name])
        elif name in fresh:
            args.append(fresh[name])
        else:
            args.append(arrays[name])
    return args, p_idx
```

> **Note for the implementer:** `n_dt_per_year`, `n_species`, `n_subdt`, `has_access`, `use_stage_access` come from `meta["scalars"]` captured in Task 2 — confirm these scalar names exist in `meta.json["arg_order"]`/`scalars`; if the parallel kernel passes any of them under a different name, align the keys here. The `n_dt` fallback in the code is a guard, not the intended source.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/native_predation/tests/test_leaf_args.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/native_predation/leaf_args.py scripts/spikes/native_predation/tests/test_leaf_args.py
git commit -m "spike(perf): leaf-arg reconstruction + p10/p50/p95 cell selection"
```

---

### Task 4: C port of the leaf + cffi build (portable & native)

**Files:**
- Create: `scripts/spikes/native_predation/kernel.c`
- Create: `scripts/spikes/native_predation/build_ffi.py`

**Interfaces:**
- Produces two importable compiled modules: `_leaf_portable` (`-O3`) and `_leaf_native` (`-O3 -march=native`), each exposing:
  - `void apply_predation(const idx_t* args..., int n_iter)` — runs the leaf `n_iter` times, **re-initialising the mutated arrays from caller-provided pristine copies each iteration, inside C, outside any timed region the caller controls** (the caller passes both the working buffers and pristine snapshots + does the reset via `reset_mutated`). For simplicity expose two C entry points: `apply_predation_once(...)` (one call, for parity) and `apply_predation_bench(..., int n_iter, /* pristine snapshot ptrs */)` (loops n_iter, resetting mutated buffers from the snapshots each iteration before the call).
  - `noop(...)` — same ~40-arg signature, empty body, for the boundary-cost probe.
- 2D arrays are passed as flat row-major pointers + explicit shapes: `access_matrix` (`acc_nrow`,`acc_ncol`), `rsc_biomass` (`n_resources`,`n_cells`), `n_dead` (`n_schools`,`n_causes`), `diet_matrix` (`n_pred`,`diet_ncol`).

**Background:** transcribe `osmose/engine/processes/mortality.py:881-1053` exactly. Preserve `total_available` accumulation order (schools in `cell_indices` order, then resources in `r` order) and the apportionment loop order. Use `double` throughout (NumPy f64). Indices are `int32` (`int`). The functional-response branch (lines 999-1010) and the conservation clamp must be copied verbatim.

- [ ] **Step 1: Write the C kernel**

```c
/* scripts/spikes/native_predation/kernel.c
 * Faithful C transcription of _apply_predation_numba (mortality.py:881-1053).
 * Reduction order MUST match the Python source or parity (<=1e-12) fails.
 */
#include <stddef.h>

typedef int i32;

/* Core single-predator predation. All 2D arrays are flat row-major. */
static void leaf(
    i32 p_idx, const i32* cell_indices, i32 n_local,
    double* inst_abd, double* n_dead, int n_causes,
    const i32* species_id, const double* length, const double* weight,
    const i32* age_dt, const i32* first_feeding_age_dt, const i32* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max, int srm_ncol,
    const double* ingestion_rate, const i32* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix, i32 acc_nrow, i32 acc_ncol,
    int has_access, int use_stage_access,
    const i32* prey_access_idx, const i32* pred_access_idx,
    double* rsc_biomass, i32 n_cells, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const i32* rsc_access_rows,
    i32 n_resources, i32 n_species, i32 cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, i32 diet_ncol, int diet_enabled,
    i32* prey_type_buf, i32* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained)
{
    if (age_dt[p_idx] < first_feeding_age_dt[p_idx]) return;
    double abd_p = inst_abd[p_idx];
    if (abd_p <= 0) return;

    i32 sp_pred = species_id[p_idx];
    double pred_len = length[p_idx];
    i32 stage = feeding_stage[p_idx];
    double r_min = size_ratio_min[sp_pred * srm_ncol + stage];
    double r_max = size_ratio_max[sp_pred * srm_ncol + stage];

    double biomass_p = abd_p * weight[p_idx];
    double max_eatable = biomass_p * ingestion_rate[sp_pred] / (n_dt_per_year * n_subdt);
    if (max_eatable <= 0) return;

    double total_available = 0.0;
    i32 n_prey = 0;

    /* 1a: school prey (cell_indices order) */
    for (i32 q_pos = 0; q_pos < n_local; q_pos++) {
        i32 q_idx = cell_indices[q_pos];
        if (q_idx == p_idx) continue;
        double abd_q = inst_abd[q_idx] - egg_retained[q_idx];
        if (abd_q < 0.0) abd_q = 0.0;
        if (abd_q <= 0) continue;
        double prey_len = length[q_idx];
        if (prey_len <= 0) continue;
        double ratio = pred_len / prey_len;
        if (ratio < r_min || ratio >= r_max) continue;

        double access_coeff = 1.0;
        if (has_access) {
            if (use_stage_access) {
                i32 p_acc = pred_access_idx[p_idx], q_acc = prey_access_idx[q_idx];
                if (p_acc >= 0 && q_acc >= 0) {
                    if (q_acc < acc_nrow && p_acc < acc_ncol)
                        access_coeff = access_matrix[q_acc * acc_ncol + p_acc];
                    if (access_coeff <= 0) continue;
                }
            } else {
                i32 sp_prey = species_id[q_idx];
                if (sp_pred < acc_nrow && sp_prey < acc_ncol) {
                    access_coeff = access_matrix[sp_pred * acc_ncol + sp_prey];
                    if (access_coeff <= 0) continue;
                }
            }
        }
        double prey_bio = abd_q * weight[q_idx];
        if (prey_bio <= 0) continue;
        double eligible = prey_bio * access_coeff;
        prey_type_buf[n_prey] = 0;
        prey_id_buf[n_prey] = q_idx;
        prey_eligible_buf[n_prey] = eligible;
        total_available += eligible;
        n_prey++;
    }

    /* 1b: resource prey (r order) */
    for (i32 r = 0; r < n_resources; r++) {
        double rsc_bio = rsc_biomass[r * n_cells + cell_id];
        if (rsc_bio <= 0) continue;
        if (r_min <= 0 || r_max <= 0) continue;
        double prey_size_min = pred_len / r_max;
        double prey_size_max = pred_len / r_min;
        double rsmn = rsc_size_min[r], rsmx = rsc_size_max[r];
        double overlap_min = rsmn > prey_size_min ? rsmn : prey_size_min;
        double overlap_max = rsmx < prey_size_max ? rsmx : prey_size_max;
        if (overlap_max <= overlap_min) continue;
        double rsc_range = rsmx - rsmn;
        if (rsc_range <= 0) continue;
        double percent_resource = (overlap_max - overlap_min) / rsc_range;

        double access_coeff = 1.0;
        if (use_stage_access) {
            i32 rsc_row = rsc_access_rows[r], p_acc = pred_access_idx[p_idx];
            if (rsc_row >= 0 && p_acc >= 0) {
                if (rsc_row < acc_nrow && p_acc < acc_ncol) {
                    access_coeff = access_matrix[rsc_row * acc_ncol + p_acc];
                    if (access_coeff <= 0) continue;
                }
            }
        } else if (has_access) {
            i32 rsc_sp_idx = n_species + r;
            if (sp_pred < acc_nrow && rsc_sp_idx < acc_ncol) {
                access_coeff = access_matrix[sp_pred * acc_ncol + rsc_sp_idx];
                if (access_coeff <= 0) continue;
            }
        }
        double eligible_bio = rsc_bio * percent_resource * access_coeff;
        prey_type_buf[n_prey] = 1;
        prey_id_buf[n_prey] = r;
        prey_eligible_buf[n_prey] = eligible_bio;
        total_available += eligible_bio;
        n_prey++;
    }

    if (total_available <= 0) return;

    /* Phase 2: functional response */
    double eaten_total;
    if (fr_shape[sp_pred] == 1) {
        eaten_total = total_available < max_eatable ? total_available : max_eatable;
    } else {
        double rr = total_available / max_eatable;
        double k_fr = fr_halfsat[sp_pred];
        double g_form;
        if (fr_shape[sp_pred] == 2) g_form = rr / (rr + k_fr);
        else g_form = (rr * rr) / (rr * rr + k_fr * k_fr);
        double cap = rr < 1.0 ? rr : 1.0;
        double g = g_form < cap ? g_form : cap;
        eaten_total = max_eatable * g;
    }

    for (i32 k = 0; k < n_prey; k++) {
        double share = prey_eligible_buf[k] / total_available;
        double eaten_from_prey = eaten_total * share;
        if (prey_type_buf[k] == 0) {
            i32 q_idx = prey_id_buf[k];
            if (weight[q_idx] > 0) {
                double n_dead_prey = eaten_from_prey / weight[q_idx];
                n_dead[q_idx * n_causes + 0] += n_dead_prey;
                inst_abd[q_idx] -= n_dead_prey;
            }
            if (tl_tracking) {
                double prey_tl = trophic_level[q_idx];
                if (prey_tl <= 0) prey_tl = 1.0;
                tl_weighted_sum[p_idx] += prey_tl * eaten_from_prey;
            }
            if (diet_enabled) {
                i32 prey_sp = species_id[q_idx];
                if (p_idx < diet_ncol /*rows*/ && prey_sp < diet_ncol)
                    diet_matrix[p_idx * diet_ncol + prey_sp] += eaten_from_prey;
            }
        } else {
            i32 r_idx = prey_id_buf[k];
            double cur = rsc_biomass[r_idx * n_cells + cell_id] - eaten_from_prey;
            rsc_biomass[r_idx * n_cells + cell_id] = cur > 0.0 ? cur : 0.0;
            if (tl_tracking) {
                double r_tl = rsc_tl[r_idx];
                if (r_tl <= 0) r_tl = 1.0;
                tl_weighted_sum[p_idx] += r_tl * eaten_from_prey;
            }
            if (diet_enabled) {
                i32 rsc_col = n_species + r_idx;
                if (rsc_col < diet_ncol)
                    diet_matrix[p_idx * diet_ncol + rsc_col] += eaten_from_prey;
            }
        }
    }

    double success = eaten_total / max_eatable;
    if (success > 1.0) success = 1.0;
    pred_success_rate[p_idx] += success / n_subdt;
    preyed_biomass[p_idx] += eaten_total;
}
```

> **Diet-matrix bounds note:** the Python checks `p_idx < diet_matrix.shape[0]` and `col < diet_matrix.shape[1]` separately. Pass `diet_nrow` too and use it for the `p_idx` bound; the snippet above collapses both to `diet_ncol` for brevity — **fix this to use the real row/col bounds** when wiring `build_ffi.py` (Task 4 Step 2), since `p_idx` indexes rows.

Add the public wrappers (`apply_predation_once`, `apply_predation_bench` with pristine-snapshot reset, `noop`) below `leaf` — `apply_predation_bench` memcpy's the mutated arrays from pristine snapshots each of `n_iter` iterations before calling `leaf`.

- [ ] **Step 2: Write the cffi builder**

```python
# scripts/spikes/native_predation/build_ffi.py
"""Compile kernel.c into two cffi modules: portable (-O3) and native (-march=native)."""
from __future__ import annotations

from pathlib import Path

from cffi import FFI

HERE = Path(__file__).resolve().parent
CDEF = """
void apply_predation_once(/* full flat signature — copy from kernel.c wrappers */);
void apply_predation_bench(/* full flat signature + int n_iter + pristine ptrs */);
void noop(/* same ~40-arg signature */);
"""


def build(variant: str) -> str:
    ffi = FFI()
    ffi.cdef(CDEF)
    flags = ["-O3"] if variant == "portable" else ["-O3", "-march=native"]
    ffi.set_source(
        f"scripts.spikes.native_predation._leaf_{variant}",
        (HERE / "kernel.c").read_text(),
        extra_compile_args=flags,
    )
    return ffi.compile(tmpdir=str(HERE))


if __name__ == "__main__":
    for v in ("portable", "native"):
        print(v, "->", build(v))
```

> The `CDEF` placeholders MUST be filled with the exact flat C signatures of the three public wrappers from `kernel.c`. This is the one spot where the signature is written twice — keep them identical. Verify by compiling.

- [ ] **Step 3: Build and smoke-test**

Run: `PYTHONPATH=. .venv/bin/python -m scripts.spikes.native_predation.build_ffi`
Expected: prints two `.so` paths, no compile errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/native_predation/kernel.c scripts/spikes/native_predation/build_ffi.py
git commit -m "spike(perf): C leaf transcription + cffi build (portable & native)"
```

---

### Task 5: Parity gate — C vs Numba oracle, ≤1e-12

**Files:**
- Create: `scripts/spikes/native_predation/parity.py`
- Test: `scripts/spikes/native_predation/tests/test_parity.py`

**Interfaces:**
- Consumes: `build_leaf_args` (Task 3), the compiled `_leaf_portable` (Task 4), the real `_apply_predation_numba` (Task target source).
- Produces: `parity_for_cell(arrays, meta, cell) -> dict[str, float]` — builds two independent fresh arg sets for the same cell, runs the real Numba leaf on one (the oracle) and the C `apply_predation_once` on the other, returns `{array_name: max_rel_diff}` over the mutated set. `assert_parity(report, bar=1e-12)` raises if any entry exceeds `bar`.

- [ ] **Step 1: Write the failing test**

```python
# scripts/spikes/native_predation/tests/test_parity.py
from pathlib import Path

import pytest

from scripts.spikes.native_predation.leaf_args import load_capture, select_cells
from scripts.spikes.native_predation.parity import assert_parity, parity_for_cell

FIX = Path(__file__).resolve().parents[1] / "_fixtures" / "cellloop.npz"
pytestmark = pytest.mark.skipif(not FIX.exists(), reason="run capture.py + build_ffi.py first")


def test_c_matches_numba_to_op_order_rounding():
    arrays, meta = load_capture(FIX)
    sel = select_cells(arrays)
    for key in ("small", "p10", "p50", "p95"):
        report = parity_for_cell(arrays, meta, sel[key])
        assert_parity(report, bar=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/native_predation/tests/test_parity.py -v`
Expected: FAIL — `ImportError` (parity.py absent). (After implementing, a *content* failure here means a transcription bug in `kernel.c` — fix the C, not the bar.)

- [ ] **Step 3: Implement parity**

`parity.py` builds the leaf args twice (Numba set, C set) from the same cell via `build_leaf_args`; calls `mortality._apply_predation_numba(*numba_args)`; unpacks the C set into the flat pointer call via `ffi.cast`/`.ctypes.data`; compares each mutated array with `max(|a-b| / (|b| + 1e-300))`. Include a helper that maps the NumPy arg list to the flat C arglist (passing `.ctypes.data_as` pointers + shape ints, matching `kernel.c`'s signature).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/native_predation/tests/test_parity.py -v`
Expected: PASS. If it fails on magnitude, debug `kernel.c` reduction order / bounds (esp. the diet-matrix row/col bound noted in Task 4).

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/native_predation/parity.py scripts/spikes/native_predation/tests/test_parity.py
git commit -m "spike(perf): C-vs-Numba parity gate (<=1e-12 op-order rounding)"
```

---

### Task 6: Boundary-free benchmark + boundary-cost probes

**Files:**
- Create: `scripts/spikes/native_predation/numba_driver.py`
- Create: `scripts/spikes/native_predation/bench.py`

**Interfaces:**
- Produces:
  - `numba_driver.make_driver()` → an `@njit` function `driver(*leaf_args, n_iter, *pristine_snapshots)` that, for `n_iter` iterations, resets the mutated arrays from the pristine snapshots (in-njit, untimed by construction — see below) then calls `_apply_predation_numba(*leaf_args)`. Because both reset+call sit inside the njit driver and we time the whole driver across a fixed `n_iter`, **subtract a reset-only baseline** (a second njit driver that resets but skips the leaf) so the reported time is leaf-only and the reset cost is removed identically on both Numba and C sides.
  - `bench.bench_cell(arrays, meta, cell, variant, n_iter, n_samples) -> dict` → returns `{"numba_med", "numba_iqr", "c_med", "c_iqr", "ratio", "n_local"}` using interleaved A/B samples (Mann-Whitney optional). Time = (driver_total − reset_only_total) / n_iter per side.
  - `bench.boundary_probe(...)` → per-call ns for cffi `noop` and an empty-cell Numba leaf call.
  - `bench.run_all(arrays, meta, sel, variant) -> dict` → per-cell results + call-count-weighted ratio.

**Reset-cost handling (the load-bearing detail):** time `T_full = driver_with_leaf(n_iter)` and `T_reset = driver_reset_only(n_iter)`; leaf-only per-call = `(T_full − T_reset)/n_iter`. Do the *identical* subtraction for the C side (`apply_predation_bench` vs a C `reset_only_bench`). This makes the ratio reflect leaf math, immune to memset asymmetry, per the spec §4.2.

- [ ] **Step 1: Implement `numba_driver.py`** — an `@njit` driver looping the real `_apply_predation_numba` over fixed args with per-iteration reset from snapshots, plus a reset-only twin. (The leaf is already `@njit` and called from `@njit` bodies at 1330/1511, so the driver compiles with no `objmode`.)

- [ ] **Step 2: Implement `bench.py`** — warm both drivers (one untimed call each), then collect `n_samples` interleaved A/B timings via `time.perf_counter_ns`, report median + IQR, compute the subtracted leaf-only per-call time each side, the ratio, and the boundary probes.

- [ ] **Step 3: Sanity-run the benchmark on the p50 cell**

Run: `PYTHONPATH=. .venv/bin/python -c "from scripts.spikes.native_predation import bench, leaf_args; a,m=leaf_args.load_capture(leaf_args.Path('scripts/spikes/native_predation/_fixtures/cellloop.npz')); sel=leaf_args.select_cells(a); print(bench.bench_cell(a,m,sel['p50'],'portable',2000,15))"`
Expected: prints a dict with positive `numba_med`/`c_med`, a finite `ratio`, and the cell's `n_local`.

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/native_predation/numba_driver.py scripts/spikes/native_predation/bench.py
git commit -m "spike(perf): boundary-free throughput bench + boundary-cost probes"
```

---

### Task 7: Orchestrator + artifact write-up + verdict

**Files:**
- Create: `scripts/spikes/native_predation/run_spike.py`
- Create: `docs/perf/2026-06-24-native-predation-kernel-spike.md` (generated/filled)

**Interfaces:**
- Consumes: every prior module.
- Produces: `run_spike.py` — end-to-end: provenance → (capture if no fixture) → select cells → build both variants → parity (hard-fail if >1e-12) → bench p10/p50/p95(+small) on both build variants → boundary probes → write the artifact markdown with all numbers and the verdict.

- [ ] **Step 1: Implement `run_spike.py`** orchestrating the pipeline; it must (a) call `assert_provenance` first and abort on failure, (b) run parity before bench and refuse to report a ratio if parity fails, (c) compute the call-weighted ratio on the **portable** build for the gate, (d) emit the artifact.

- [ ] **Step 2: Run the full spike**

Run: `PYTHONPATH=. .venv/bin/python -m scripts.spikes.native_predation.run_spike`
Expected: completes; writes `docs/perf/2026-06-24-native-predation-kernel-spike.md`; prints the portable call-weighted ratio and the PASS/STOP verdict against 1.3×.

- [ ] **Step 3: Fill the artifact** with the required sections (spec §5): provenance assertions; `n_local` histogram + the four cell characteristics; per-cell + call-weighted median+IQR and ratio for **both** portable and native builds; boundary-cost probe numbers; per-output max abs/rel parity diffs; and a go/no-go recommendation that explicitly states a PASS authorizes **only** the integration spike (port the *parallel* cell-loop + RNG), never the full port.

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/native_predation/run_spike.py docs/perf/2026-06-24-native-predation-kernel-spike.md
git commit -m "spike(perf): orchestrator + native-predation-kernel spike artifact + verdict"
```

---

## Notes for the executor

- **If parity (Task 5) cannot reach ≤1e-12**, the bug is in `kernel.c` (reduction order, a bound, an `int`/`double` mismatch), never the bar. The most likely culprits: the diet-matrix row vs col bound (Task 4 note), 2D flat-indexing strides, or accumulation order.
- **If the portable call-weighted ratio is < 1.3×**, that is a legitimate **STOP** — write the artifact honestly (it's an artifact-of-record like the K-arc not-shipping docs) and recommend dropping the native-kernel path. The spike is designed to fail cheaply.
- **Do not** wire any of this into `osmose/`, the engine, the main test suite, or CI. The only tests are the three under `scripts/spikes/native_predation/tests/`, run manually.
- Keep the harness (don't delete) so the artifact's numbers are reproducible.
