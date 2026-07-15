# Single-run Numba Thread Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a single Python-engine run auto-select affinity-capped physical cores for Numba (instead of all logical/HT cores), recovering ~1.5–2× wall-time bit-exactly, without touching the calibration regime.

**Architecture:** A new pure-stdlib module `osmose/engine/thread_policy.py` resolves a sensible single-run thread count (physical cores within the CPU-affinity budget, with a degenerate-topology fallback) and applies it via `numba.set_num_threads`. It is wired only into explicit single-run entry points — the UI run thread and the benchmark script — never into the shared `simulate()`/`PythonEngine.run` core, so calibration workers (which call the core directly) keep the unrestricted default.

**Tech Stack:** Python 3.12 stdlib (`os.sched_getaffinity`, `/sys` topology), numba (`set_num_threads`, an optional extra imported lazily), pytest.

## Global Constraints

- Python floor `>=3.12`; **no new declared dependency** — the resolver is pure stdlib; `numba` is imported lazily and its absence must be a no-op (engine has a pure-Python fallback). Do NOT add `psutil` (it is undeclared; relying on it reds CI in a clean venv).
- Degenerate-topology guard constant is exactly `_MAX_PLAUSIBLE_SMT = 4`.
- `resolve_engine_threads(requested)`: `requested >= 1` → `max(1, min(requested, logical_budget()))`; `requested < 1` or `None` → `physical_budget()`.
- `apply_single_run_threads(requested=None)` returns `0` when numba is absent and **never raises**.
- The cap goes ONLY at single-run call sites — never in `simulate()`, `PythonEngine.run`, or `run_in_memory`. No calibration file is modified.
- UI `py_threads` label must read exactly `"Threads (Numba; 0 = auto — physical cores)"`.
- Output must stay **bit-identical** across thread counts (the mortality prange is race-free); tests assert exact equality via `np.array_equal`, not `assert_allclose`.
- Spec: `docs/superpowers/specs/2026-07-15-single-run-thread-policy-design.md`.

---

## File Structure

- **Create** `osmose/engine/thread_policy.py` — the resolver: `logical_budget`, `_core_key`, `physical_budget`, `resolve_engine_threads`, `apply_single_run_threads`.
- **Create** `tests/test_thread_policy.py` — unit tests for the resolver (pure, mocked topology) + `apply_single_run_threads` behavior + the same-process thread-local + forkserver-boundary isolation tests + the engine determinism test.
- **Modify** `ui/pages/run.py` — replace the inline numba cap in `_python_engine_thread` (lines 320–328) with a call to `apply_single_run_threads`; add a module-level import; change the `py_threads` label (line 237).
- **Modify** `scripts/benchmark_engine.py` — apply the single-run policy once in `main()` before the timed runs.
- **Modify** `tests/helpers.py` — add one picklable worker function (`numba_thread_count()`) used by the forkserver-boundary test.

---

## Task 1: `thread_policy.py` resolver module + unit tests

**Files:**
- Create: `osmose/engine/thread_policy.py`
- Test: `tests/test_thread_policy.py`

**Interfaces:**
- Consumes: nothing (stdlib + lazy numba).
- Produces:
  - `logical_budget() -> int`
  - `_core_key(cpu: int) -> tuple[str, str] | None`
  - `physical_budget() -> int`
  - `resolve_engine_threads(requested: int | None) -> int`
  - `apply_single_run_threads(requested: int | None = None) -> int`
  - module constant `_MAX_PLAUSIBLE_SMT = 4`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_thread_policy.py`:

```python
"""Unit tests for the single-run Numba thread policy resolver."""
from __future__ import annotations

import os
import sys

import pytest

from osmose.engine import thread_policy as tp


@pytest.fixture
def restore_numba_threads():
    """Save/restore Numba's active thread count so tests don't leak into each other."""
    numba = pytest.importorskip("numba")
    saved = numba.get_num_threads()
    yield
    numba.set_num_threads(saved)


def _fake_topology(monkeypatch, affinity, core_of):
    """Make logical/physical budgets deterministic: `affinity` = set of logical cpu ids,
    `core_of` = dict cpu_id -> (pkg, core) or None (unreadable)."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(affinity))
    monkeypatch.setattr(tp, "_core_key", lambda cpu: core_of.get(cpu))


def test_ht_box_auto_is_physical(monkeypatch):
    # 28 logical, cpu N and N+14 share a physical core -> 14 physical.
    aff = set(range(28))
    core_of = {c: ("0", str(c % 14)) for c in aff}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.logical_budget() == 28
    assert tp.physical_budget() == 14
    assert tp.resolve_engine_threads(None) == 14
    assert tp.resolve_engine_threads(0) == 14


def test_no_ht_box_auto_is_all_cores(monkeypatch):
    aff = set(range(8))
    core_of = {c: ("0", str(c)) for c in aff}  # 8 distinct cores
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.physical_budget() == 8
    assert tp.resolve_engine_threads(None) == 8  # no-op: physical == logical


def test_container_quota_respected(monkeypatch):
    aff = {0, 1, 2, 3}
    core_of = {0: ("0", "0"), 1: ("0", "1"), 2: ("0", "2"), 3: ("0", "3")}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.logical_budget() == 4
    assert tp.physical_budget() == 4  # min(4 physical, 4 budget)


def test_sys_read_failure_falls_back_to_logical(monkeypatch):
    aff = set(range(28))
    _fake_topology(monkeypatch, aff, {c: None for c in aff})  # every /sys read fails
    assert tp.physical_budget() == 28  # fallback = today's behavior, NOT 1


def test_explicit_request_honored_and_capped(monkeypatch):
    aff = set(range(28))
    core_of = {c: ("0", str(c % 14)) for c in aff}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.resolve_engine_threads(20) == 20     # honored
    assert tp.resolve_engine_threads(100) == 28    # capped to logical budget
    assert tp.resolve_engine_threads(1) == 1


def test_degenerate_topology_falls_back(monkeypatch):
    # /sys reads succeed but claim all 28 cpus are one physical core (SMT width 28 > 4).
    aff = set(range(28))
    core_of = {c: ("0", "0") for c in aff}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.physical_budget() == 28  # distrusted -> logical budget, NOT 1


def test_per_cpu_tolerance_skips_unreadable(monkeypatch):
    # One cpu's /sys entry is unreadable; the rest map to 14 physical cores.
    aff = set(range(28))
    core_of = {c: ("0", str(c % 14)) for c in aff}
    core_of[7] = None
    _fake_topology(monkeypatch, aff, core_of)
    # cpu 7 skipped; remaining 27 cpus still cover all 14 core ids -> 14 physical.
    assert tp.physical_budget() == 14


def test_apply_returns_zero_when_numba_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)  # `import numba` -> ImportError
    assert tp.apply_single_run_threads(4) == 0


def test_apply_sets_and_returns_resolved(monkeypatch, restore_numba_threads):
    numba = pytest.importorskip("numba")
    monkeypatch.setattr(tp, "resolve_engine_threads", lambda requested: 3)
    n = tp.apply_single_run_threads(999)
    assert n == 3
    assert numba.get_num_threads() == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_thread_policy.py -q`
Expected: collection/import error — `ModuleNotFoundError: No module named 'osmose.engine.thread_policy'`.

- [ ] **Step 3: Implement the module**

Create `osmose/engine/thread_policy.py`:

```python
"""Single-run Numba thread policy.

A single Python-engine run's mortality kernel is a Numba ``prange`` cell-loop
(``@njit(parallel=True)``). Its output is bit-identical at every thread count, but
it does NOT scale past ~physical cores: on a hyperthreaded box, using all logical
cores is ~1.5-2x SLOWER than the physical-core optimum (per-timestep fork/join +
hyperthread contention + memory bandwidth). This module resolves a sensible thread
count for a SINGLE run and applies it.

Scope: single-run entry points ONLY (UI run thread, benchmark, headless single-run
scripts). NOT calibration — its nested ``ProcessPoolExecutor`` workers call the
engine directly and want the unrestricted default; keep this out of the shared
engine core so they are unaffected.
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

_MAX_PLAUSIBLE_SMT = 4  # threads per physical core; guards degenerate /sys data


def logical_budget() -> int:
    """Logical CPUs the process may actually use (cgroup/taskset-aware)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # non-Linux / unavailable
        n = os.cpu_count() or 1
    return max(1, n)


def _core_key(cpu: int) -> tuple[str, str] | None:
    """`(physical_package_id, core_id)` for a logical CPU from /sys, or None."""
    base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
    try:
        with open(f"{base}/physical_package_id") as f:
            pkg = f.read().strip()
        with open(f"{base}/core_id") as f:
            core = f.read().strip()
    except (OSError, ValueError):
        return None
    return (pkg, core)


def physical_budget() -> int:
    """Distinct physical cores within the affinity set, via /sys topology.

    Falls back to ``logical_budget()`` if the topology cannot be read OR looks
    degenerate — never worse than today's all-logical default.
    """
    budget = logical_budget()
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return budget
    cores: set[tuple[str, str]] = set()
    seen = 0
    for cpu in allowed:
        key = _core_key(cpu)
        if key is None:
            continue  # skip this CPU; don't forfeit the whole scan
        cores.add(key)
        seen += 1
    n_phys = len(cores)
    if n_phys < 1 or seen < 1:
        return budget
    # Degenerate-topology guard: an implied SMT width far above any real CPU
    # (e.g. a virtualized host reporting every CPU as one core) would make auto
    # single-threaded — a regression. Distrust it, fall back.
    if seen // n_phys > _MAX_PLAUSIBLE_SMT:
        return budget
    return min(n_phys, budget)


def resolve_engine_threads(requested: int | None) -> int:
    """Resolve a single-run Numba thread count.

    ``requested >= 1`` -> honor it, capped to the logical budget.
    ``requested < 1`` / ``None`` (auto) -> physical-core budget.
    """
    if requested is not None and requested >= 1:
        return max(1, min(requested, logical_budget()))
    return physical_budget()


def apply_single_run_threads(requested: int | None = None) -> int:
    """Set Numba's thread count for a single run; log it; return the count.

    No-op (returns 0) if numba is absent. Never raises — a bad detection must not
    block a run.
    """
    try:
        import numba  # type: ignore[import-untyped]  # optional extra
    except Exception:  # noqa: BLE001
        return 0
    try:
        n = resolve_engine_threads(requested)
        numba.set_num_threads(n)
        _log.info(
            "engine threads: using %d (requested=%r, logical=%d, physical=%d)",
            n, requested, logical_budget(), physical_budget(),
        )
        return n
    except Exception:  # noqa: BLE001
        _log.warning(
            "could not apply single-run thread policy; using Numba default",
            exc_info=True,
        )
        return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_thread_policy.py -q`
Expected: all tests in the file PASS (9 passed).

- [ ] **Step 5: Lint + type-check the new module**

Run: `.venv/bin/python -m ruff check osmose/engine/thread_policy.py tests/test_thread_policy.py && .venv/bin/python -m ruff format --check osmose/engine/thread_policy.py tests/test_thread_policy.py`
Expected: no errors. (If format check fails, run `.venv/bin/python -m ruff format osmose/engine/thread_policy.py tests/test_thread_policy.py`.)

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/thread_policy.py tests/test_thread_policy.py
git commit -m "feat(engine): single-run Numba thread resolver (affinity-capped physical cores)"
```

---

## Task 2: Wire the single-run entry points (UI + benchmark)

**Files:**
- Modify: `ui/pages/run.py` (`_python_engine_thread` body ~320–328; module imports; `py_threads` label ~237)
- Modify: `scripts/benchmark_engine.py` (`main()`)
- Test: `tests/test_thread_policy.py` (append wiring tests)

**Interfaces:**
- Consumes: `apply_single_run_threads` from Task 1.
- Produces: no new public symbols.

- [ ] **Step 1: Write the failing wiring tests**

Append to `tests/test_thread_policy.py`:

```python
def test_python_engine_thread_applies_policy(monkeypatch):
    """_python_engine_thread must pass the raw py_threads value to the policy
    and post a 'done' outcome — without running a real engine."""
    import queue

    from ui.pages import run as run_mod

    recorded = {}
    monkeypatch.setattr(
        run_mod, "apply_single_run_threads",
        lambda requested: recorded.setdefault("requested", requested) or 14,
    )

    class _FakeEngine:
        def run(self, *a, **k):
            return "RESULT"

    monkeypatch.setattr(run_mod, "PythonEngine", _FakeEngine)
    done_q: queue.Queue = queue.Queue()
    run_mod._python_engine_thread("cfg", "out", None, None, done_q, n_threads=7)

    assert recorded["requested"] == 7
    kind, result, msg = done_q.get_nowait()
    assert kind == "done"
    assert result == "RESULT"


def test_benchmark_main_applies_policy(monkeypatch):
    """benchmark_engine.main() must call apply_single_run_threads without running
    the engine (run_benchmark is stubbed)."""
    import scripts.benchmark_engine as be

    called = {}
    monkeypatch.setattr(be, "apply_single_run_threads", lambda requested=None: called.setdefault("hit", True) or 12)
    monkeypatch.setattr(
        be, "run_benchmark",
        lambda *a, **k: {"elapsed_s": 0.1, "final_biomass": {"sp0": 1.0}},
    )
    monkeypatch.setattr(sys, "argv", ["benchmark_engine.py", "--config", "minimal", "--years", "1", "--repeats", "1"])
    be.main()
    assert called.get("hit") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_thread_policy.py::test_python_engine_thread_applies_policy tests/test_thread_policy.py::test_benchmark_main_applies_policy -q`
Expected: FAIL — `AttributeError: ... has no attribute 'apply_single_run_threads'` (the name is not yet imported into `ui.pages.run` / `scripts.benchmark_engine`).

- [ ] **Step 3: Wire `ui/pages/run.py`**

Add a module-level import near the other `osmose` imports at the top of `ui/pages/run.py`:

```python
from osmose.engine.thread_policy import apply_single_run_threads
```

Replace the current block in `_python_engine_thread` (lines 320–328):

```python
    try:
        import numba  # type: ignore[import-untyped]  # optional extra; engine has a pure-Python fallback

        cap = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
        numba.set_num_threads(
            min(n_threads, cap) if n_threads >= 1 else cap
        )  # n<1 = auto/all cores
    except Exception:  # noqa: BLE001 — never block a run on numba absence/bad count
        _log.warning("could not apply py_threads; using Numba default", exc_info=True)
```

with:

```python
    # Cap single-run threads to ~physical cores (0/<1 = auto). Never raises;
    # no-op if numba is absent. See osmose/engine/thread_policy.py.
    apply_single_run_threads(n_threads)
```

Change the `py_threads` input label (line 237) from:

```python
                        "Threads (Numba; 0 = auto/all cores)",
```

to:

```python
                        "Threads (Numba; 0 = auto — physical cores)",
```

- [ ] **Step 4: Wire `scripts/benchmark_engine.py`**

Add the import at the top of `scripts/benchmark_engine.py` (after the existing imports):

```python
from osmose.engine.thread_policy import apply_single_run_threads
```

In `main()`, immediately after the `print()` following the "Benchmarking ..." header line (before the `timings = []` loop), add:

```python
    n_threads = apply_single_run_threads()
    print(f"  Numba threads: {n_threads or 'default (numba absent)'}\n")
```

- [ ] **Step 5: Run the wiring tests + confirm no import cycle**

Run: `.venv/bin/python -m pytest tests/test_thread_policy.py -q && .venv/bin/python -c "import ui.pages.run, scripts.benchmark_engine"`
Expected: all tests PASS; the import line prints nothing and exits 0 (no circular-import error).

- [ ] **Step 6: Lint + type-check touched files**

Run: `.venv/bin/python -m ruff check ui/pages/run.py scripts/benchmark_engine.py tests/test_thread_policy.py && .venv/bin/python -m ruff format --check ui/pages/run.py scripts/benchmark_engine.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add ui/pages/run.py scripts/benchmark_engine.py tests/test_thread_policy.py
git commit -m "feat(run): apply single-run thread policy in UI run + benchmark"
```

---

## Task 3: Safety guarantee tests — bit-identical determinism + calibration isolation

**Files:**
- Modify: `tests/helpers.py` (add `numba_thread_count()`)
- Test: `tests/test_thread_policy.py` (append determinism + isolation tests)

**Interfaces:**
- Consumes: `apply_single_run_threads` (Task 1); the engine `simulate()`.
- Produces: `tests.helpers.numba_thread_count() -> int` (module-level, picklable for forkserver).

- [ ] **Step 1: Add the picklable worker helper**

Append to `tests/helpers.py`:

```python
def numba_thread_count(_=None) -> int:
    """Return Numba's active thread count. Module-level + picklable so a
    forkserver/spawn worker can run it (used by the thread-policy isolation test)."""
    import numba

    return numba.get_num_threads()
```

- [ ] **Step 2: Write the failing determinism + isolation tests**

Append to `tests/test_thread_policy.py`:

```python
import numpy as np


def _run_minimal(n_years: int, seed: int, threads: int):
    """Run the `minimal` fixture at a fixed Numba thread count; return outputs."""
    import numba

    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "minimal", "osm_all-parameters.csv"
    )
    reader = OsmoseConfigReader()
    raw = reader.read(cfg_path)
    raw["simulation.time.nyear"] = str(n_years)
    cfg = EngineConfig.from_dict(raw)
    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            os.path.join(os.path.dirname(cfg_path), grid_file),
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        grid = Grid.from_dimensions(
            ny=int(raw.get("grid.nline", "1")), nx=int(raw.get("grid.ncolumn", "1"))
        )
    numba.set_num_threads(threads)
    return simulate(cfg, grid, np.random.default_rng(seed))


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores to compare thread counts")
def test_mortality_bit_identical_across_thread_counts(restore_numba_threads):
    """The mortality prange is race-free: 1 thread vs many threads must be
    EXACTLY equal (np.array_equal, not allclose). Two hardcoded, different counts
    so this cannot degenerate into comparing a run against itself."""
    pytest.importorskip("numba")
    hi = os.cpu_count()
    out1 = _run_minimal(2, 123, 1)
    outN = _run_minimal(2, 123, hi)
    b1 = np.asarray(out1[-1].biomass, dtype=np.float64)
    bN = np.asarray(outN[-1].biomass, dtype=np.float64)
    assert np.array_equal(b1, bN), f"biomass differs between 1 and {hi} threads"


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores to observe a cap")
def test_cap_is_thread_local_same_process(restore_numba_threads):
    """Same-process (thread-backed) calibration paths are unaffected: a main-thread
    cap is invisible to a sibling thread (numba.set_num_threads is thread-local)."""
    import threading

    numba = pytest.importorskip("numba")
    default = numba.config.NUMBA_NUM_THREADS
    if default < 2:
        pytest.skip("default thread count is 1; cannot distinguish a cap")
    tp.apply_single_run_threads(1)
    assert numba.get_num_threads() == 1
    seen = []
    t = threading.Thread(target=lambda: seen.append(numba.get_num_threads()))
    t.start()
    t.join()
    assert seen[0] == default, "single-run cap leaked into a sibling thread"


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores to observe a cap")
def test_cap_does_not_leak_into_forkserver_worker(restore_numba_threads):
    """Calibration's ProcessPoolExecutor(forkserver) workers must NOT inherit a
    single-run cap. Cap the main process to 1, then a forkserver worker must still
    see the unrestricted default."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    numba = pytest.importorskip("numba")
    default = numba.config.NUMBA_NUM_THREADS
    if default < 2:
        pytest.skip("default thread count is 1; cannot distinguish a cap")

    from tests.helpers import numba_thread_count

    tp.apply_single_run_threads(1)
    assert numba.get_num_threads() == 1
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        worker_threads = ex.submit(numba_thread_count, None).result(timeout=120)
    assert worker_threads == default, "single-run cap leaked into a forkserver worker"
```

- [ ] **Step 3: Run tests to verify they fail (for the right reason)**

Run: `.venv/bin/python -m pytest tests/test_thread_policy.py::test_cap_does_not_leak_into_forkserver_worker -q`
Expected: FAIL with `ImportError: cannot import name 'numba_thread_count' from 'tests.helpers'` — until Step 1's helper exists. (If Step 1 was already applied, this test PASSES; the determinism test may already pass because the engine is unchanged — that is fine, it is a regression guard. Confirm all three at least COLLECT and run.)

- [ ] **Step 4: Run the full thread-policy test file**

Run: `.venv/bin/python -m pytest tests/test_thread_policy.py -q`
Expected: all tests PASS. If `test_cap_does_not_leak_into_forkserver_worker` errors on worker import, verify `tests/__init__.py` exists (it does) and that the helper is module-level in `tests/helpers.py`.

- [ ] **Step 5: Run the broader engine suite (no regressions)**

Run: `.venv/bin/python -m pytest tests/test_thread_policy.py tests/ -q -k "thread or determinism or mortality" 2>&1 | tail -20`
Expected: green; the pre-existing `test_mortality_deterministic_across_thread_counts` (if present) still passes alongside the new stricter test.

- [ ] **Step 6: Lint**

Run: `.venv/bin/python -m ruff check tests/test_thread_policy.py tests/helpers.py && .venv/bin/python -m ruff format --check tests/test_thread_policy.py tests/helpers.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add tests/test_thread_policy.py tests/helpers.py
git commit -m "test(engine): bit-identical determinism + calibration-isolation guards for thread policy"
```

---

## Verification (whole-branch, after all tasks)

- [ ] Full suite: `.venv/bin/python -m pytest tests/ -q -n auto` — green (or only the documented CI-fragile-emergent skips).
- [ ] Lint/format/type: `.venv/bin/python -m ruff check osmose/ ui/ tests/ scripts/ && .venv/bin/python -m ruff format --check osmose/ ui/ tests/` and the project pyright config — clean.
- [ ] Clean-venv import check: `osmose/engine/thread_policy.py` imports only stdlib at module scope (no `psutil`, no top-level `numba`).
- [ ] Manual perf sanity (optional, on this box): `.venv/bin/python scripts/benchmark_engine.py --config eec_full --years 3 --repeats 5` prints `Numba threads: 14` and a median near ~1.16 s (vs the ~1.78 s all-logical default), biomass unchanged.

---

## Self-Review

**1. Spec coverage:**
- §1 resolver (`logical_budget`/`physical_budget`+floor+tolerance/`resolve_engine_threads`/`apply_single_run_threads`) → Task 1. ✅
- §2a UI wiring + label → Task 2. ✅ §2b benchmark wiring → Task 2. ✅ §2c "helper as mechanism, no specific batch script wired" → satisfied (only UI + benchmark wired; helper is public). ✅
- §3 calibration untouched → no calibration file in any task; guarded by Task 3 thread-local + forkserver-boundary tests. ✅
- Testing items 1 (cases i–viii) → Task 1; item 2 (determinism, exact, two hardcoded counts, skip<2) → Task 3; item 3 (real forkserver boundary) → Task 3; item 4 (apply behavior) → Task 1; item 5 (no regression, UI honors explicit value) → Task 2 wiring test + Verification suite. ✅
- Edge cases (no-HT, container quota, unreadable/degenerate topology, numba absent, explicit>budget) → Task 1 unit tests. ✅

**2. Placeholder scan:** No TBD/TODO/"handle errors" placeholders; every code step shows complete code and exact commands. ✅

**3. Type consistency:** `resolve_engine_threads(requested: int | None) -> int`, `apply_single_run_threads(requested: int | None = None) -> int`, `_core_key(cpu) -> tuple[str,str] | None`, `physical_budget()/logical_budget() -> int`, `tests.helpers.numba_thread_count(_=None) -> int` — names/signatures consistent across Tasks 1–3 and match the spec. UI passes the raw `n_threads` int (0/negative → auto, handled by `resolve_engine_threads`), consistent with `resolve`'s contract. ✅
