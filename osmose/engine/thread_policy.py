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
            n,
            requested,
            logical_budget(),
            physical_budget(),
        )
        return n
    except Exception:  # noqa: BLE001
        _log.warning(
            "could not apply single-run thread policy; using Numba default",
            exc_info=True,
        )
        return 0
