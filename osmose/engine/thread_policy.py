"""Numba thread policy for single runs and for engine worker pools.

A Python-engine run's mortality kernel is a Numba ``prange`` cell-loop
(``@njit(parallel=True)``). Its output is bit-identical at every thread count, but
it does NOT scale past ~physical cores: on a hyperthreaded box, using all logical
cores is ~1.5-2x SLOWER than the physical-core optimum (per-timestep fork/join +
hyperthread contention + memory bandwidth).

Two policies:

- **Single run** (``apply_single_run_threads``): UI run thread, benchmark, headless
  single-run scripts use the physical-core budget.
- **Worker pools** (``apply_worker_threads``, ``worker_threads_for_fork``): each of W
  engine worker processes uses ``max(1, physical cores // W)`` threads. Measured
  2026-11-14 on a 14-physical-core box (docs/perf/2026-11-14-worker-thread-sweep.md):
  8 workers at Numba's unrestricted default (28 threads each) gave 124 evals/h, while
  14 workers at 1 thread gave 492 evals/h (3.96x, interleaved repeats). Unrestricted
  workers oversubscribe the cores. This reverses the 2026-07-15 decision to leave
  calibration workers unrestricted, which rested on a thread-backend measurement.
  ``OSMOSE_WORKER_THREADS`` overrides the rule: a positive integer forces that count,
  ``unrestricted`` restores Numba's default.

The calls are thread-local runtime settings (``numba.set_num_threads``), never
inherited environment variables, so a single-run cap cannot leak into a worker pool.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator

_log = logging.getLogger(__name__)

_MAX_PLAUSIBLE_SMT = 4  # threads per physical core; guards degenerate /sys data

WORKER_THREADS_ENV = "OSMOSE_WORKER_THREADS"


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
    except Exception:
        return 0
    try:
        n = resolve_engine_threads(requested)
        numba.set_num_threads(n)
        _log.info(
            "engine threads: using %d (requested=%r, logical=%d, physical=%d)",
            n,
            requested,
            logical_budget(),
            physical_budget(),
        )
        return n
    except Exception:
        _log.warning(
            "could not apply single-run thread policy; using Numba default",
            exc_info=True,
        )
        return 0


def worker_thread_count(n_workers: int) -> int | None:
    """Numba threads for each of ``n_workers`` engine worker processes.

    The split rule ``max(1, physical_budget() // n_workers)``, unless
    ``OSMOSE_WORKER_THREADS`` is set: a positive integer forces that count (capped to the
    logical budget), ``unrestricted`` returns None (leave Numba's default). An unparseable
    value is logged and ignored.
    """
    raw = os.environ.get(WORKER_THREADS_ENV, "").strip().lower()
    if raw in ("unrestricted", "none"):
        return None
    if raw:
        try:
            forced = int(raw)
        except ValueError:
            forced = 0
        if forced >= 1:
            return min(forced, logical_budget())
        _log.warning(
            "ignoring %s=%r (want a positive integer or 'unrestricted')", WORKER_THREADS_ENV, raw
        )
    return max(1, physical_budget() // max(1, int(n_workers)))


def _limit_threads(n: int) -> int:
    """Set Numba's (thread-local) count, and BLAS/OpenMP pools when threadpoolctl is present."""
    import numba  # type: ignore[import-untyped]  # optional extra

    n = max(1, min(n, getattr(numba.config, "NUMBA_NUM_THREADS", n)))
    numba.set_num_threads(n)
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]  # optional

        threadpool_limits(limits=n)
    except Exception:  # optional; the engine's hot loops are Numba
        pass
    return n


def apply_worker_threads(n_workers: int) -> int:
    """Apply the worker rule in THIS process (a pool initializer); return the count applied.

    Returns 0 when the rule says leave Numba's default, when numba is absent, or on any
    error. Never raises: a bad detection must not break a pool.
    """
    n = worker_thread_count(n_workers)
    if n is None:
        return 0
    try:
        import numba  # type: ignore[import-untyped]  # noqa: F401
    except Exception:
        return 0
    try:
        return _limit_threads(n)
    except Exception:
        _log.warning("could not apply the worker thread policy; using Numba default", exc_info=True)
        return 0


@contextlib.contextmanager
def worker_threads_for_fork(n_workers: int) -> Iterator[int]:
    """Set this thread's Numba count to the worker rule while fork-started workers are created.

    A forked child inherits the forking thread's Numba thread count, so a pool forked inside
    the block (e.g. scipy's ``differential_evolution(workers=W)``) runs at the rule. Numba's
    count and any BLAS/OpenMP limit are restored on exit, so serial work in the parent
    afterwards keeps its threads. Yields the count applied (0 if left unchanged).
    """
    n = worker_thread_count(n_workers)
    try:
        import numba  # type: ignore[import-untyped]
    except Exception:
        yield 0
        return
    if n is None:
        yield 0
        return
    previous = numba.get_num_threads()
    n = max(1, min(n, getattr(numba.config, "NUMBA_NUM_THREADS", n)))
    numba.set_num_threads(n)
    blas: contextlib.AbstractContextManager = contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]  # optional

        blas = threadpool_limits(limits=n)  # applies now; restores on exit
    except Exception:
        pass
    try:
        with blas:
            yield n
    finally:
        numba.set_num_threads(previous)
