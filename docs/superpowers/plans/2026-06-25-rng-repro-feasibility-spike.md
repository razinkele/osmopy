# Stage-0 RNG-Reproduction Feasibility Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove (or disprove) that hand-written C can reproduce, bit-for-bit, the per-cell RNG sequence the production mortality cell-loop draws (`seed → 4× permutation(n) → per-school shuffle`), and measure its generation speed — the disqualifier for a bit-exact native cell-loop port.

**Architecture:** A throwaway harness under `scripts/spikes/rng_repro/`. An `@njit` oracle mirrors the cell-loop's exact RNG usage (ground truth). A C reimplementation of NumPy-legacy MT19937 (init-by-array seeding + `rk_interval` masked-rejection + Fisher-Yates `permutation`/`shuffle`), compiled via `cffi`, must produce bit-identical sequences across an `n × seed` grid. A speed probe times C vs Numba RNG-gen. An orchestrator emits a `docs/perf/` go/no-go artifact.

**Tech Stack:** Python 3.12, NumPy, Numba 0.65 (oracle), `cffi` 2.0 + `gcc` 13.3 (already present — no new dependency), pytest.

## Global Constraints

- **No new system/runtime dependency** — `cffi` + `gcc` only; no Rust.
- **Throwaway harness** — everything under `scripts/spikes/rng_repro/`; nothing imported by `osmose/`, no engine edits, not in CI or the main suite (only the spike's own `tests/`, run manually).
- **Reproduction target = NumPy-legacy MT19937** — exploration (spec §2) proved Numba `np.random` ≡ CPython legacy `RandomState` bit-identically (incl. the int64 per-cell seed, which reduces as `seed & 0xFFFFFFFF`). The C targets that documented algorithm.
- **Parity bar = BIT-EXACT** — C sequences must equal the oracle exactly (`np.array_equal`). No "close enough"; this spike tests whether bit-exact is achievable.
- **The kernel's exact RNG (mortality.py:1479–1497), which the oracle and C must mirror:** per cell — `np.random.seed(rng_seed + cell*7919)`; then `seq_pred = permutation(n)`, `seq_starv = permutation(n)`, `seq_fish = permutation(n)`, `seq_nat = permutation(n)` (in that order); then `causes = [0,1,2,3]` created ONCE, and for each of `n` schools `np.random.shuffle(causes)` **in place** (carrying the prior permutation — NOT reset each iteration), recording `cause_orders[i] = causes`.
- **Run everything with `PYTHONPATH=.`** from the worktree root so the imported `osmose` is the worktree's; the provenance guard enforces this. The worktree has no committed `.venv` — symlink the main one (`ln -s /home/razinka/osmose/osmose-python/.venv .venv`) so `.venv/bin/python` resolves.
- **cffi `.so` placement gotcha (from the leaf spike):** `ffi.set_source` MUST use a NON-dotted module name (e.g. `"_rng_portable"`) with `tmpdir=HERE`, or the `.so` is written to a nested path and is not importable.
- Spec: `docs/superpowers/specs/2026-06-25-rng-repro-feasibility-spike-design.md`.

---

### Task 1: Scaffold + the @njit oracle

**Files:**
- Create: `scripts/spikes/rng_repro/__init__.py` (empty)
- Create: `scripts/spikes/rng_repro/tests/__init__.py` (empty)
- Create: `scripts/spikes/rng_repro/oracle.py`
- Create: `scripts/spikes/rng_repro/README.md` (one paragraph: throwaway spike, how to run `run_spike.py`, pointer to the spec)
- Test: `scripts/spikes/rng_repro/tests/test_oracle.py`

**Interfaces:**
- Produces: `oracle_cell_rng(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning `(seq_pred, seq_starv, seq_fish, seq_nat, cause_orders)` — the first four are int32 length-`n`; `cause_orders` is int32 `(n, 4)`. Mirrors mortality.py:1479–1497 exactly. Also `cpython_reference(seed, n)` returning the same tuple computed with CPython `np.random.RandomState` (the documented oracle, for the 0a cross-check).

- [ ] **Step 1: Write the failing test**

```python
# scripts/spikes/rng_repro/tests/test_oracle.py
import numpy as np

from scripts.spikes.rng_repro.oracle import cpython_reference, oracle_cell_rng


def test_oracle_matches_cpython_legacy_randomstate():
    # The premise (spec 0a): the @njit oracle == CPython legacy RandomState, bit-identical.
    # Seeds MUST span up to the large value the Task 3 parity grid uses (2**62+...): the
    # parity gate compares C-vs-oracle and ASSUMES oracle==RandomState there, so the premise
    # must be proven at that seed too — else a real Numba-vs-NumPy gap (a STOP/beta-signal)
    # would be misread as a C bug.
    for seed in (0, 1, 7919, 12345, 2**40 + 7919 * 3, 2**62 + 12345 + 990 * 7919):
        for n in (1, 2, 4, 12, 24, 33, 100):
            got = oracle_cell_rng(seed, n)
            ref = cpython_reference(seed, n)
            for g, r in zip(got, ref):
                assert np.array_equal(g, r), f"oracle != RandomState at seed={seed} n={n}"


def test_oracle_shuffle_carries_over_not_reset():
    # cause_orders rows must come from shuffling-in-place (carry-over), not re-shuffling
    # a fresh [0,1,2,3] each row. With carry-over, consecutive rows are (almost surely)
    # different permutations; this pins the in-place semantics.
    _, _, _, _, orders = oracle_cell_rng(12345, 20)
    assert orders.shape == (20, 4)
    assert {tuple(r) for r in orders}.issubset({(a, b, c, d) for a in range(4)
            for b in range(4) for c in range(4) for d in range(4)
            if len({a, b, c, d}) == 4})  # every row is a permutation of 0..3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/rng_repro/tests/test_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError` (oracle.py absent).

- [ ] **Step 3: Write the oracle**

```python
# scripts/spikes/rng_repro/oracle.py
"""Ground-truth RNG oracle: mirrors the cell-loop's per-cell draws (mortality.py:1479-1497).

The @njit oracle is what production actually draws; the CPython RandomState reference is
the documented NumPy-legacy MT19937 algorithm the C port targets (spec 0a proved they agree).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _njit_cell_rng(seed, n):
    np.random.seed(seed)
    seq_pred = np.random.permutation(n).astype(np.int32)
    seq_starv = np.random.permutation(n).astype(np.int32)
    seq_fish = np.random.permutation(n).astype(np.int32)
    seq_nat = np.random.permutation(n).astype(np.int32)
    causes = np.array([0, 1, 2, 3], dtype=np.int32)  # created ONCE
    cause_orders = np.empty((n, 4), dtype=np.int32)
    for i in range(n):
        np.random.shuffle(causes)  # in place — carries over from the previous row
        cause_orders[i, 0] = causes[0]
        cause_orders[i, 1] = causes[1]
        cause_orders[i, 2] = causes[2]
        cause_orders[i, 3] = causes[3]
    return seq_pred, seq_starv, seq_fish, seq_nat, cause_orders


def oracle_cell_rng(seed: int, n: int):
    return _njit_cell_rng(np.int64(seed), int(n))


def cpython_reference(seed: int, n: int):
    rs = np.random.RandomState(np.uint32(seed & 0xFFFFFFFF))  # legacy MT19937, uint32 seed
    a = rs.permutation(n).astype(np.int32)
    b = rs.permutation(n).astype(np.int32)
    c = rs.permutation(n).astype(np.int32)
    d = rs.permutation(n).astype(np.int32)
    causes = np.array([0, 1, 2, 3], dtype=np.int32)
    orders = np.empty((n, 4), dtype=np.int32)
    for i in range(n):
        rs.shuffle(causes)
        orders[i] = causes
    return a, b, c, d, orders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/rng_repro/tests/test_oracle.py -v`
Expected: PASS (2 passed).

> If `test_oracle_matches_cpython_legacy_randomstate` FAILS, the premise is wrong for some
> (seed, n) — STOP and report: it means Numba and NumPy-legacy diverge there, which changes
> the whole spike (the C target would be Numba-specific, not documented).

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/rng_repro/
git commit -m "spike(rng): scaffold + njit cell-RNG oracle (mirrors mortality.py:1479-1497)"
```

---

### Task 2: C NumPy-legacy MT19937 + cffi build

**Files:**
- Create: `scripts/spikes/rng_repro/mt19937.c`
- Create: `scripts/spikes/rng_repro/build_ffi.py`

**Interfaces:**
- Produces: an importable compiled module `scripts.spikes.rng_repro._rng_portable` (and `_rng_native`) whose `.lib` exposes
  `void cell_rng(int64_t seed, int n, int32_t* out_pred, int32_t* out_starv, int32_t* out_fish, int32_t* out_nat, int32_t* out_orders)`.
  `out_pred/starv/fish/nat` are length `n`; `out_orders` is length `n*4` (row-major `(n,4)`). It seeds, fills the four permutation buffers (arange→Fisher-Yates), then runs `n` in-place shuffles of a carried-over `causes[4]`.

- [ ] **Step 1: Write `mt19937.c`** (faithful NumPy-legacy MT19937)

```c
/* scripts/spikes/rng_repro/mt19937.c
 * NumPy-legacy MT19937 + init_genrand scalar seeding + rk_interval masked-rejection
 * bounded integers + Fisher-Yates permutation/shuffle. Targets CPython legacy
 * RandomState (== Numba np.random, per spec 0a). Parity gate (Task 3) validates.
 */
#include <stdint.h>

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

typedef struct { uint32_t mt[N]; int mti; } mt_state;

/* NumPy-legacy RandomState(scalar) and Numba np.random.seed(scalar) seed the MT state via
 * the SINGLE-INTEGER path init_genrand(seed) — NOT init_by_array. (Verified empirically:
 * RandomState(0)'s first uint32 == init_genrand(0) == 2357136044, whereas init_by_array({0})
 * gives a different stream.) cell_rng therefore seeds with init_genrand. */
static void init_genrand(mt_state* s, uint32_t seed) {
    s->mt[0] = seed;
    for (int i = 1; i < N; i++)
        s->mt[i] = (uint32_t)(1812433253UL * (s->mt[i-1] ^ (s->mt[i-1] >> 30)) + (uint32_t)i);
    s->mti = N;
}

static uint32_t genrand_uint32(mt_state* s) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, MATRIX_A};
    if (s->mti >= N) {
        int kk;
        for (kk = 0; kk < N - M; kk++) {
            y = (s->mt[kk] & UPPER_MASK) | (s->mt[kk+1] & LOWER_MASK);
            s->mt[kk] = s->mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < N - 1; kk++) {
            y = (s->mt[kk] & UPPER_MASK) | (s->mt[kk+1] & LOWER_MASK);
            s->mt[kk] = s->mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (s->mt[N-1] & UPPER_MASK) | (s->mt[0] & LOWER_MASK);
        s->mt[N-1] = s->mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        s->mti = 0;
    }
    y = s->mt[s->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

/* NumPy-legacy rk_interval: returns a uniform integer in [0, max] via masked rejection. */
static uint32_t rk_interval(uint32_t max, mt_state* s) {
    if (max == 0) return 0;
    uint32_t mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16;
    uint32_t value;
    while ((value = (genrand_uint32(s) & mask)) > max) { }
    return value;
}

/* NumPy-legacy Fisher-Yates: for i = n-1 down to 1, j = rk_interval(i), swap(arr[i], arr[j]). */
static void fisher_yates(int32_t* arr, int n, mt_state* s) {
    for (int i = n - 1; i > 0; i--) {
        uint32_t j = rk_interval((uint32_t)i, s);
        int32_t t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

void cell_rng(int64_t seed, int n, int32_t* out_pred, int32_t* out_starv,
              int32_t* out_fish, int32_t* out_nat, int32_t* out_orders) {
    mt_state s;
    uint32_t key = (uint32_t)(seed & 0xFFFFFFFF);  /* uint32 reduction (spec 0a) */
    init_genrand(&s, key);                          /* scalar seeding (matches RandomState/Numba) */

    int32_t* outs[4] = {out_pred, out_starv, out_fish, out_nat};
    for (int p = 0; p < 4; p++) {
        for (int i = 0; i < n; i++) outs[p][i] = i;   /* arange(n) */
        fisher_yates(outs[p], n, &s);                 /* permutation */
    }
    int32_t causes[4] = {0, 1, 2, 3};                 /* created ONCE, carried over */
    for (int i = 0; i < n; i++) {
        fisher_yates(causes, 4, &s);                  /* shuffle in place */
        out_orders[i*4 + 0] = causes[0];
        out_orders[i*4 + 1] = causes[1];
        out_orders[i*4 + 2] = causes[2];
        out_orders[i*4 + 3] = causes[3];
    }
}
```

- [ ] **Step 2: Write `build_ffi.py`** (NON-dotted module name — leaf-spike gotcha)

```python
# scripts/spikes/rng_repro/build_ffi.py
"""Compile mt19937.c into cffi modules: portable (-O3) and native (-O3 -march=native)."""
from __future__ import annotations

from pathlib import Path

from cffi import FFI

HERE = Path(__file__).resolve().parent
CDEF = """
void cell_rng(int64_t seed, int n, int32_t* out_pred, int32_t* out_starv,
              int32_t* out_fish, int32_t* out_nat, int32_t* out_orders);
"""


def build(variant: str) -> str:
    ffi = FFI()
    ffi.cdef(CDEF)
    flags = ["-O3"] if variant == "portable" else ["-O3", "-march=native"]
    ffi.set_source(f"_rng_{variant}", '#include <stdint.h>\n'
                   + (HERE / "mt19937.c").read_text(), extra_compile_args=flags)
    return ffi.compile(tmpdir=str(HERE))


if __name__ == "__main__":
    for v in ("portable", "native"):
        print(v, "->", build(v))
```

> Note: `set_source` is given a NON-dotted name `_rng_{variant}` so the `.so` lands directly
> in `scripts/spikes/rng_repro/`. Import it as `from scripts.spikes.rng_repro import _rng_portable`.

- [ ] **Step 3: Build + smoke (compile AND import — the leaf spike's miss)**

Run: `PYTHONPATH=. .venv/bin/python -m scripts.spikes.rng_repro.build_ffi`
Then: `PYTHONPATH=. .venv/bin/python -c "from scripts.spikes.rng_repro import _rng_portable as R; print([s for s in dir(R.lib) if not s.startswith('_')])"`
Expected: build prints two `.so` paths under `scripts/spikes/rng_repro/`; the import prints `['cell_rng']`.

- [ ] **Step 4: Create `scripts/spikes/rng_repro/.gitignore`** (build artifacts are not committed)

```
*.so
*.o
_rng_*.c
__pycache__/
```

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/rng_repro/mt19937.c scripts/spikes/rng_repro/build_ffi.py scripts/spikes/rng_repro/.gitignore
git commit -m "spike(rng): C NumPy-legacy MT19937 (seed+permutation+shuffle) + cffi build"
```

---

### Task 3: Parity gate — C vs oracle, bit-identical across the grid

**Files:**
- Create: `scripts/spikes/rng_repro/parity.py`
- Test: `scripts/spikes/rng_repro/tests/test_parity.py`

**Interfaces:**
- Consumes: `oracle_cell_rng` (Task 1), the compiled `_rng_portable` (Task 2).
- Produces:
  - `c_cell_rng(seed: int, n: int, lib, ffi) -> tuple[np.ndarray, ...]` — marshals 5 contiguous int32 out-buffers, calls `lib.cell_rng`, returns `(seq_pred, seq_starv, seq_fish, seq_nat, cause_orders(n,4))`.
  - `GRID_N = [1, 2, 3, 4, 5, 8, 12, 24, 33, 50, 100]` and `GRID_SEEDS` (see below).
  - `compare_cell(seed, n, lib, ffi) -> dict[str, bool]` — per-array `np.array_equal` C-vs-oracle.

**Grid rationale:** `GRID_N` spans 1..100, a superset of the real eec_full cell range (1..~33), and a bit-exact RNG is n-agnostic, so this is stricter coverage than 3 percentiles. `GRID_SEEDS` must include the real per-cell form and the int64 wrap edge:
`GRID_SEEDS = [0, 1, 7919, 12345] + [rs + c*7919 for (rs, c) in [(0,0),(0,1),(987654321,5),(2**62 + 12345, 990)]]` — the last exercises a large `rng_seed` near 2^62 (production uses `rng_seed = int(rng.integers(0, 2**63))`; `rng_seed + cell*7919` stays within int64 but is large), so the uint32 reduction `seed & 0xFFFFFFFF` and the high-bit handling are tested, not assumed.

- [ ] **Step 1: Write the failing test**

```python
# scripts/spikes/rng_repro/tests/test_parity.py
import numpy as np
import pytest

from scripts.spikes.rng_repro.parity import GRID_N, GRID_SEEDS, compare_cell

try:
    from scripts.spikes.rng_repro import _rng_portable as R
    _HAVE_SO = True
except ImportError:
    _HAVE_SO = False

pytestmark = pytest.mark.skipif(not _HAVE_SO, reason="run build_ffi.py first")


def test_c_matches_numba_oracle_bit_exact_across_grid():
    for seed in GRID_SEEDS:
        for n in GRID_N:
            report = compare_cell(int(seed), n, R.lib, R.ffi)
            assert all(report.values()), f"C != oracle at seed={seed} n={n}: {report}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/rng_repro/tests/test_parity.py -v`
Expected: FAIL with `ImportError` (parity.py absent).

- [ ] **Step 3: Implement `parity.py`**

```python
# scripts/spikes/rng_repro/parity.py
"""Bit-exact parity: C cell_rng vs the @njit oracle, across an n x seed grid."""
from __future__ import annotations

import numpy as np

from .oracle import oracle_cell_rng

GRID_N = [1, 2, 3, 4, 5, 8, 12, 24, 33, 50, 100]
GRID_SEEDS = [0, 1, 7919, 12345] + [
    rs + c * 7919 for (rs, c) in [(0, 0), (0, 1), (987654321, 5), (2**62 + 12345, 990)]
]


def c_cell_rng(seed: int, n: int, lib, ffi):
    a = np.empty(n, dtype=np.int32)
    b = np.empty(n, dtype=np.int32)
    c = np.empty(n, dtype=np.int32)
    d = np.empty(n, dtype=np.int32)
    orders = np.empty(n * 4, dtype=np.int32)
    cast = lambda arr: ffi.cast("int32_t *", arr.ctypes.data)  # noqa: E731
    # seed is int64 in the C signature; pass the Python int directly (cffi coerces).
    lib.cell_rng(int(seed), n, cast(a), cast(b), cast(c), cast(d), cast(orders))
    return a, b, c, d, orders.reshape(n, 4)


def compare_cell(seed: int, n: int, lib, ffi) -> dict[str, bool]:
    got = c_cell_rng(seed, n, lib, ffi)
    ref = oracle_cell_rng(seed, n)
    names = ("seq_pred", "seq_starv", "seq_fish", "seq_nat", "cause_orders")
    return {name: bool(np.array_equal(g, r)) for name, g, r in zip(names, got, ref)}
```

> **If parity fails on content** (not import): the C has a transcription bug — debug by finding
> the FIRST diverging draw. Likely culprits in priority order: (1) wrong SEEDING routine —
> must be `init_genrand(seed)` (the scalar path RandomState/Numba use), NOT `init_by_array`;
> (2) `causes` reset each loop instead of carried over; (3) Fisher-Yates loop direction or
> `rk_interval(i)` vs `(i+1)` bound (legacy uses inclusive `[0, i]`); (4) the `seed & 0xFFFFFFFF`
> reduction for the large-seed edge; (5) mask computation in `rk_interval`. NEVER loosen the bar.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest scripts/spikes/rng_repro/tests/test_parity.py -v`
Expected: PASS. A PASS here is the spike's core feasibility result.

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/rng_repro/parity.py scripts/spikes/rng_repro/tests/test_parity.py
git commit -m "spike(rng): bit-exact parity gate (C vs njit oracle across n x seed grid)"
```

---

### Task 4: Speed probe — C vs Numba RNG-gen

**Files:**
- Create: `scripts/spikes/rng_repro/speed.py`
- Modify: `scripts/spikes/rng_repro/mt19937.c` (add a `cell_rng_bench` loop fn), `scripts/spikes/rng_repro/build_ffi.py` (add `cell_rng_bench` to the `CDEF`)

**Interfaces:**
- Consumes: `_njit_cell_rng` (Task 1, the `@njit` oracle — call it directly so timing excludes Python wrapping), `_rng_portable` (Task 2).
- Produces: `bench_rng(n: int, n_iter: int, n_samples: int, lib, ffi) -> dict` returning `{"numba_med_ns", "numba_iqr_ns", "c_med_ns", "c_iqr_ns", "ratio", "n"}` — **boundary-free** per-cell RNG-gen time per side: the full per-cell draw is looped `n_iter` times inside native code on both sides (one `@njit` dispatch vs one C/cffi call), so neither side pays a per-cell Python→C boundary. Interleaved A/B sampling, median + IQR. `ratio = numba_med / c_med` (>1 = C faster).

- [ ] **Step 1: Add `cell_rng_bench` to `mt19937.c`** (boundary-free C loop — reseeds each iter so every iteration does identical work)

```c
/* append to mt19937.c, after cell_rng */
void cell_rng_bench(int64_t seed, int n, int n_iter, int32_t* out_pred, int32_t* out_starv,
                    int32_t* out_fish, int32_t* out_nat, int32_t* out_orders) {
    for (int it = 0; it < n_iter; it++)
        cell_rng(seed, n, out_pred, out_starv, out_fish, out_nat, out_orders);
}
```

- [ ] **Step 2: Add its declaration to the `CDEF` in `build_ffi.py` and rebuild**

Append to `CDEF` (must match the C signature exactly):
```c
void cell_rng_bench(int64_t seed, int n, int n_iter, int32_t* out_pred, int32_t* out_starv,
                    int32_t* out_fish, int32_t* out_nat, int32_t* out_orders);
```
Run: `PYTHONPATH=. .venv/bin/python -m scripts.spikes.rng_repro.build_ffi` (rebuilds both variants).

- [ ] **Step 3: Implement `speed.py`**

```python
# scripts/spikes/rng_repro/speed.py
"""Boundary-free per-cell RNG-gen timing: C cell_rng_bench vs an @njit driver."""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from .oracle import _njit_cell_rng


@njit(cache=True)
def _numba_bench(seed, n, n_iter):
    acc = np.int64(0)
    for _ in range(n_iter):
        a, b, c, d, orders = _njit_cell_rng(seed, n)
        acc += a[0] + b[0] + c[0] + d[0] + orders[0, 0]  # defeat dead-code elimination
    return acc


def bench_rng(n: int, n_iter: int, n_samples: int, lib, ffi) -> dict:
    seed = np.int64(12345)
    out = [np.empty(n if k < 4 else n * 4, dtype=np.int32) for k in range(5)]
    cast = lambda a: ffi.cast("int32_t *", a.ctypes.data)  # noqa: E731
    cargs = (int(seed), n, int(n_iter), cast(out[0]), cast(out[1]),
             cast(out[2]), cast(out[3]), cast(out[4]))

    _numba_bench(seed, n, 1)             # warm JIT
    lib.cell_rng_bench(*cargs)           # warm C

    numba_ns, c_ns = [], []
    for _ in range(n_samples):           # interleaved A/B to cancel drift
        t = time.perf_counter_ns(); _numba_bench(seed, n, n_iter)
        numba_ns.append((time.perf_counter_ns() - t) / n_iter)
        t = time.perf_counter_ns(); lib.cell_rng_bench(*cargs)
        c_ns.append((time.perf_counter_ns() - t) / n_iter)

    def med_iqr(xs):
        s = sorted(xs)
        med = s[len(s) // 2]
        iqr = s[int(len(s) * 0.75)] - s[int(len(s) * 0.25)]
        return med, iqr

    nm, ni = med_iqr(numba_ns)
    cm, ci = med_iqr(c_ns)
    return {"numba_med_ns": nm, "numba_iqr_ns": ni, "c_med_ns": cm,
            "c_iqr_ns": ci, "ratio": (nm / cm if cm else float("inf")), "n": n}
```

- [ ] **Step 4: Sanity-run at p50/p95-representative n**

Run: `PYTHONPATH=. .venv/bin/python -c "from scripts.spikes.rng_repro import speed, _rng_portable as R; print(speed.bench_rng(12, 50000, 20, R.lib, R.ffi)); print(speed.bench_rng(33, 50000, 20, R.lib, R.ffi))"`
Expected: dicts with positive `numba_med_ns`/`c_med_ns` and a finite `ratio`.

- [ ] **Step 5: Commit**

```bash
git add scripts/spikes/rng_repro/speed.py scripts/spikes/rng_repro/mt19937.c scripts/spikes/rng_repro/build_ffi.py
git commit -m "spike(rng): C-vs-Numba per-cell RNG-gen speed probe"
```

---

### Task 5: Orchestrator + artifact + verdict

**Files:**
- Create: `scripts/spikes/rng_repro/run_spike.py`
- Create: `docs/perf/2026-06-25-rng-repro-feasibility-spike.md` (generated/filled)

**Interfaces:**
- Consumes: every prior module + `scripts.spikes.native_predation.provenance.assert_provenance` (reuse the leaf-spike guard).

- [ ] **Step 1: Implement `run_spike.py`** — end-to-end: (a) derive the worktree root as
  `worktree_root = Path(__file__).resolve().parents[3]` (run_spike.py is at
  `scripts/spikes/rng_repro/run_spike.py` → parents[0]=`rng_repro`, [1]=`spikes`, [2]=`scripts`,
  [3]=worktree root) and call `assert_provenance(worktree_root)` FIRST (import it via
  `from scripts.spikes.native_predation.provenance import assert_provenance`), aborting on
  failure; (b) build both variants if absent, import; (c) run `compare_cell` across the full
  `GRID_N × GRID_SEEDS` grid, collect every result, HARD-FAIL the verdict if any mismatch
  (record the first-diverging `(seed, n, array, index)`); (d) run `bench_rng` at n∈{12,33} on
  the portable build; (e) write the artifact + print the PASS/STOP verdict.

- [ ] **Step 2: Run the full spike**

Run: `PYTHONPATH=. .venv/bin/python -m scripts.spikes.rng_repro.run_spike`
Expected: completes; writes `docs/perf/2026-06-25-rng-repro-feasibility-spike.md`; prints PASS (bit-identical across the grid) or STOP (with the first-diverging (seed, n, array, index)).

- [ ] **Step 3: Fill the artifact** (spec §6) with the REAL results: the 0a premise; the full parity grid (every (seed, n) → match, or the first divergence + its classification); the C-vs-Numba RNG-gen median+IQR + ratio at n∈{12,33} for the portable build (note `-march=native` is not the gate); and a go/no-go that states explicitly a PASS authorises only *designing* Stage 1 (the cell-loop port: `_apply_single_cause` + leaf integration + prange/OpenMP + end-to-end measurement), never building it blind — and re-states the permanent-second-implementation maintenance caveat.

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/rng_repro/run_spike.py docs/perf/2026-06-25-rng-repro-feasibility-spike.md
git commit -m "spike(rng): orchestrator + RNG-reproduction feasibility artifact + verdict"
```

---

## Notes for the executor

- **If parity (Task 3) cannot reach bit-identical**, the bug is in `mt19937.c` (see the Task 3 culprit-priority list), never the bar or the oracle. The oracle is validated against CPython RandomState in Task 1.
- **If the C-RNG-gen is a large fraction of the per-cell cost**, note it as a Stage-1 risk in the artifact (it would erode the integration win) — but it does NOT fail this spike; the parity result is the gate.
- **Do not** wire any of this into `osmose/`, the engine, the main suite, or CI. Only the tests under `scripts/spikes/rng_repro/tests/`, run manually.
- Keep the harness (don't delete) so the artifact's numbers are reproducible.
- Build artifacts (`*.so`, `*.o`, `_rng_*.c`) are NOT committed; add a `scripts/spikes/rng_repro/.gitignore` (`*.so`, `*.o`, `_rng_*.c`, `__pycache__/`) in Task 2.
