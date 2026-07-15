# Single-run Numba thread policy — design

**Date:** 2026-07-15
**Status:** Draft (brainstorming → spec)
**Branch:** `perf/single-run-thread-policy`

## Motivation

The mortality cell-loop is the engine's dominant cost, and the standing perf backlog
listed "parallelize the cell-loop (~35% lever)" / a native C+OpenMP port as the biggest
remaining prize. A 2026-07-15 re-profile (per the "re-profile before any perf plan" rule)
found both framings are wrong:

- **The cell-loop is *already* parallel.** `_mortality_all_cells_parallel`
  (`osmose/engine/processes/mortality.py:1410`, `@njit(cache=True, parallel=True)`,
  `prange` over cells) is the production path (`mortality()` default `parallel=True` at
  `mortality.py:1806`; dispatched at `mortality.py:1985`). Cells are disjoint index slices
  with a per-cell deterministic seed (`rng_seed + cell*7919`), so output is **bit-identical
  at every thread count** (verified 1→28 threads on eec_full *and* baltic — race-free,
  embarrassingly parallel).
- **Production runs it at the pessimal thread count.** A thread-scaling sweep of whole-engine
  wall-time (box: Intel i9-10940X, **14 physical cores × 2 HT = 28 logical**, 1 NUMA node)
  shows scaling peaks at ≈ physical-core count and then *regresses* when all logical/HT cores
  are used:

  | config | 1 thread | ~12–14 threads (peak) | 28 threads (prod default) |
  |---|---|---|---|
  | eec_full 3yr (median-of-7) | 3.081 s | **1.155 s (2.67×)** | 1.779 s (1.73×) |
  | baltic 1yr (median-of-5) | 0.713 s | **0.250 s (2.85×)** | 0.412 s (1.73×) |

  The UI's `py_threads` input (`ui/pages/run.py:235`) defaults to `value=0` = "auto/all
  cores", which `_python_engine_thread` (`ui/pages/run.py:323–326`) turns into
  `numba.set_num_threads(NUMBA_NUM_THREADS)` = **28 (all logical)**. So every interactive
  "Start Run" — and every headless single run using the raw default — is ~1.5–2× slower than
  optimal.

Cause: a fork/join per timestep over ~1–2k uneven cells. Past the physical-core count,
thread-team coordination + hyperthread cache/port contention + memory bandwidth swamp the
compute saved. This is the same oversubscription pathology already documented for DE
calibration workers (24→16 default; `docs/claude-memory/feedback_de_workers_default.md`).

Full investigation record: `docs/perf/2026-07-15-mortality-thread-oversubscription.md`
(to be written alongside this work) and memory `project-mortality-thread-oversubscription`.

## Goal

For a **single** Python-engine run, auto-select the physical-core count (capped by the
process's CPU-affinity budget) instead of all logical/HT cores — recovering the measured
~1.5–2× wall-time, **bit-exact** — while leaving the calibration regime (nested parallelism)
untouched. A user's explicit thread choice is still honored.

## Non-goals (YAGNI)

- **No env-var override knob.** The auto policy is affinity-capped physical cores; the
  existing `py_threads` UI input remains the interactive override, and headless callers pass
  an explicit count in code. (An `OSMOSE_ENGINE_THREADS` env var was considered and dropped.)
- **No change to calibration's thread/worker policy.** Calibration runs 16 concurrent
  `ProcessPoolExecutor` workers (`osmose/calibration/problem.py`); a prior measurement
  (`docs/claude-memory/project_v010_calibration_python_engine.md`) showed capping inner Numba
  threads there *slows* it. It stays unrestricted.
- **No native C/OpenMP port** of the cell-loop. It would hit the identical fork/join +
  bandwidth + HT walls; it is not the bottleneck. Do not re-open the prior spike's Stage 1.
- **No change** to the Numba kernel, `simulate()`'s signature, the Java engine/CLI runner
  (`osmose/runner.py` — a JVM subprocess, not a Numba path), or config format.

## Design

### 1. `osmose/engine/thread_policy.py` — pure-stdlib resolver (no new dependency)

`psutil` is **not** a declared dependency (`pyproject.toml` has zero references) — relying on
it would red CI in a clean venv (known gotcha `feedback_ci_clean_venv_reproduction`). The
resolver uses only the standard library and is Linux-first (prod, dev, and the HPC Apptainer
image are all Linux), with a safe fallback everywhere else.

```python
import os


def logical_budget() -> int:
    """Logical CPUs the process may actually use (cgroup/taskset-aware)."""
    try:
        n = len(os.sched_getaffinity(0))  # respects affinity/cgroup pinning
    except (AttributeError, OSError):     # non-Linux / unavailable
        n = os.cpu_count() or 1
    return max(1, n)


_MAX_PLAUSIBLE_SMT = 4  # threads per physical core; guards degenerate /sys data


def physical_budget() -> int:
    """Distinct physical cores within the affinity set, via /sys topology.

    Returns logical_budget() unchanged if the topology cannot be read OR looks
    degenerate — so this is never worse than today's all-logical default.
    """
    budget = logical_budget()
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):     # non-Linux / unavailable
        return budget
    cores = set()
    seen = 0
    for cpu in allowed:
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        try:
            with open(f"{base}/physical_package_id") as f:
                pkg = f.read().strip()
            with open(f"{base}/core_id") as f:
                core = f.read().strip()
        except (OSError, ValueError):
            continue                      # skip this CPU; don't forfeit the whole scan
        cores.add((pkg, core))
        seen += 1
    n_phys = len(cores)
    if n_phys < 1 or seen < 1:
        return budget
    # Degenerate-topology guard: an implied SMT width far above any real CPU
    # (e.g. a virtualized host reporting every CPU as one core -> n_phys=1)
    # would make auto single-threaded — a regression. Distrust it, fall back.
    if seen // n_phys > _MAX_PLAUSIBLE_SMT:
        return budget
    return min(n_phys, budget)


def resolve_engine_threads(requested: int | None) -> int:
    """Resolve a single-run Numba thread count.

    requested >= 1 -> honor it, capped to the logical budget (respect explicit intent).
    requested < 1 / None (auto) -> physical-core budget (the measured sweet spot).
    """
    if requested is not None and requested >= 1:
        return max(1, min(requested, logical_budget()))
    return physical_budget()


def apply_single_run_threads(requested: int | None = None) -> int:
    """Set Numba's thread count for a single run; log the choice; return it.

    No-op (returns 0) if numba is absent — the engine has a pure-Python fallback.
    Never raises: a bad detection must not block a run.
    """
    try:
        import numba  # optional extra
    except Exception:
        return 0
    try:
        n = resolve_engine_threads(requested)
        numba.set_num_threads(n)
        _log.info("engine threads: using %d (requested=%r, logical=%d, physical=%d)",
                  n, requested, logical_budget(), physical_budget())
        return n
    except Exception:
        _log.warning("could not apply single-run thread policy; using Numba default",
                     exc_info=True)
        return 0
```

Rationale for `min(n_phys, budget)`: in a container limited to *k* logical CPUs (`--cpus k`),
`sched_getaffinity` returns *k* entries; distinct physical cores among them is ≤ *k*, so we
never over-subscribe the quota. On a non-HT box, physical == logical, so auto is a no-op
(never a regression). On read failure **or degenerate topology**, we return the logical budget
= today's behavior. The `_MAX_PLAUSIBLE_SMT` guard specifically defends the HPC/cloud/Apptainer
targets this design cares about, where a virtualized host can report a misleading `/sys`
topology; without it, junk data claiming one physical core would silently force single-threaded
runs. Verified on this box: `logical_budget()=28`, `physical_budget()=14` (each `cpuN`/`cpuN+14`
pair shares one `(pkg, core)`), matching the measured sweet spot. Accepted edge (low impact): a
container pinned to exactly the two HT siblings of one core (affinity `{0,14}`) yields
`n_phys=1` → auto uses 1 thread; at that scale fork/join cost is negligible and the result is
still within quota, so this is left as-is rather than special-cased.

### 2. Wiring — explicit opt-in at single-run entry points

**a. UI single-run (`ui/pages/run.py`) — the production path.** Replace the inline block at
`_python_engine_thread` (`run.py:323–326`):

```python
# before:
cap = numba.config.NUMBA_NUM_THREADS
numba.set_num_threads(min(n_threads, cap) if n_threads >= 1 else cap)
# after:
from osmose.engine.thread_policy import apply_single_run_threads
apply_single_run_threads(n_threads if n_threads >= 1 else None)
```

Update the `py_threads` input label (`run.py:237`) from `"Threads (Numba; 0 = auto/all
cores)"` to `"Threads (Numba; 0 = auto — physical cores)"`. The user override still works
(`input.py_threads()` read at `run.py:948`); only what `0` means changes.

**b. Benchmark (`scripts/benchmark_engine.py`).** Call `apply_single_run_threads()` once
before the timed runs, so perf measurement reflects the production single-run policy.

**c. Headless / HPC batch — helper as the mechanism, not a specific wired script.** The
Apptainer batch pattern is `apptainer exec ... python scripts/<name>.py` (`apptainer/osmose.def`).
Not every documented batch entry is a single run: `scripts/compute_model_reference_points.py`
drives the **fmsy sweep**, whose `ProcessPoolExecutor(spawn)` workers already self-pin to
`set_num_threads(1)` (`osmose/validation/fmsy_sweep.py:245`) — wiring the parent there would be a
no-op and would
contradict §3's fmsy exclusion. So this change does **not** claim any specific batch script as
"wired." Instead, `apply_single_run_threads()` is the public one-liner that a *genuine*
single-run headless script (one that itself calls `simulate()` / `PythonEngine().run_in_memory`,
e.g. `scripts/native_440_parity.py`, `scripts/baltic_rv_hindcast.py`) calls at startup. The
benchmark wiring in **(b)** is the one concrete headless call site this change adds; broader
per-script adoption is left to those scripts (out of scope here).

### 3. Calibration & fmsy stay unrestricted — by construction and by Numba semantics

The cap is placed **only at explicit single-run call sites**, never inside the shared engine
core (`simulate()` / `PythonEngine.run` / `run_in_memory`). This is the load-bearing decision:
calibration workers call the engine **directly** — `run_in_memory` (`problem.py:441`,
`larva_recal.py:118`) or `run` (`preflight.py:723–726`) — inside their workers, so a cap inside
the shared core (`run` / `run_in_memory` / `simulate`) *would* throttle every calibration worker
to physical cores — which a prior measurement
(`project_v010_calibration_python_engine`) showed hurts calibration's nested-parallelism
regime. Keeping the cap out of the shared core avoids that entirely.

**No cross-context leak is possible — verified, so no calibration-side change is made:**

- `numba.set_num_threads()` is **thread-local** in the installed Numba (0.65.1; a deliberate
  Numba design since ≥0.47). A single-run cap set on the UI/main thread is invisible to any
  other thread — so the same-process thread-backed calibration paths (`ThreadPoolExecutor`,
  `parallel_backend="thread"`) see the unrestricted default regardless.
- Calibration's process pool uses `mp_context="forkserver"` (`problem.py:349`) and the fmsy
  sweep uses `mp_context="spawn"` (`fmsy_sweep.py:342`) — **not** the bare `fork` default.
  Their workers start from a fresh/exec'd interpreter image and additionally self-select their
  own thread count (fmsy pins `set_num_threads(1)`, `fmsy_sweep.py:245`), so they never inherit
  a live parent thread-count mutation.

An earlier draft proposed resetting Numba threads in the calibration worker initializer
(`problem.py::_worker_init`, `:71`) as fork-inheritance defense. That is **dropped**: the
premise (`fork` default + process-global thread state) is false on both counts above, so the
reset would guard a non-existent leak. YAGNI — this change touches no calibration file.

## Edge cases

- **No hyperthreading:** physical == logical → auto is a no-op.
- **Container CPU quota (`--cpus k`):** affinity budget = *k* → `min(physical, k) ≤ k`, quota
  respected.
- **Topology unreadable / non-Linux:** `physical_budget()` returns the logical budget =
  today's behavior.
- **Degenerate `/sys` topology (virtualized/cloud host):** if the implied SMT width exceeds
  `_MAX_PLAUSIBLE_SMT`, distrust the reading and fall back to the logical budget — prevents a
  bogus "1 physical core" from forcing single-threaded runs.
- **Numba absent:** `apply_single_run_threads` returns 0, run proceeds on the pure-Python
  fallback.
- **Explicit user value > budget:** capped to the logical budget (never oversubscribe beyond
  what the OS grants).

## Testing strategy

1. **`resolve_engine_threads` / budget unit tests** (pure, no engine): monkeypatch
   `os.sched_getaffinity` and the `/sys` reads to simulate
   (i) HT box (28 logical / 14 physical) → auto = 14;
   (ii) no-HT box (8/8) → auto = 8;
   (iii) container (affinity = {0,1,2,3}) → auto ≤ 4;
   (iv) `/sys` read raises → falls back to logical budget;
   (v) explicit `requested=20` on a 28-logical box → 20; `requested=100` → 28;
   (vi) `requested=0/None` → physical budget;
   (vii) **degenerate topology** — all CPUs report one `(pkg, core)` (implied SMT width >
   `_MAX_PLAUSIBLE_SMT`) → falls back to logical budget (NOT 1);
   (viii) one CPU's `/sys` entry unreadable, the rest fine → that CPU skipped, others counted
   (per-CPU tolerance, not all-or-nothing).
2. **Engine determinism integration test:** run a small config (`data/minimal`, ≤2 yr) at
   **two hardcoded, explicitly different** thread counts — `1` and `os.cpu_count()` (skip/xfail
   if `os.cpu_count() < 2`) — and assert **exact** equality of final biomass via
   `np.array_equal` (not `assert_allclose`). Hardcoding both counts avoids the trap where, on a
   small CI runner, `resolve_engine_threads` and the raw default coincide and the test
   degenerates into comparing a run against itself. (This complements the existing
   `test_mortality_deterministic_across_thread_counts`, which uses `allclose`; the new test
   enforces the stronger bit-identical bar the design relies on.) This is a controlled
   bit-identical assertion on fixed input — NOT a golden-value/rel-change emergent test, so it
   does not fall into the known CI-fragile-emergent-Baltic trap ([[feedback-ci-fragile-emergent-tests]]).
3. **Calibration-isolation guard test (real process boundary):** call
   `apply_single_run_threads(...)` (or `numba.set_num_threads(2)`) in the test's main process,
   then launch a small `ProcessPoolExecutor` built with the **production** `mp_context`
   (`forkserver`) + the real worker initializer, and assert a task *running inside a spawned
   worker* observes the unrestricted `NUMBA_NUM_THREADS` default — i.e. the single-run cap does
   not cross into calibration workers. This falsifies the actual isolation claim (a boundary
   crossing), unlike an in-process check of the initializer, and requires no NSGA-II run.
4. **`apply_single_run_threads` behavior:** returns 0 when numba import fails (monkeypatch);
   returns the resolved count and calls `set_num_threads` when present (spy).
5. **No regression:** existing engine/parity tests stay green; the UI run path still honors an
   explicit `py_threads` value.

## Verification

- Unit + integration tests green; `ruff` + type-check clean; clean-venv import check (no
  psutil).
- Manual: on this 28-logical box, an eec_full single run via the UI (or benchmark) drops from
  ~1.78 s to ~1.16 s (3 yr), bit-identical biomass, with a log line reporting `engine
  threads: using 14`.
- Re-run the thread-scaling sweep to confirm the auto default lands in the flat optimum, not
  the 28-thread regression.

## Rollback

Additive and revertible: one new stdlib-only module (`thread_policy.py`); two call-site edits
(UI `_python_engine_thread`, `scripts/benchmark_engine.py`) — the UI one replaces an existing
thread-set line — plus the `py_threads` label copy change; plus tests. **No calibration file is
touched**, and no kernel, `simulate()`, or config change. Reverting restores the all-logical
default.
