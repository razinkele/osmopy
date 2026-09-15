"""Support helpers for running the suite under pytest-xdist.

Kept separate from conftest.py so the logic is unit-testable without importing
the conftest machinery.
"""

from pathlib import Path


def worker_numba_cache_dir(worker: str | None, base: Path) -> Path | None:
    """Per-worker numba cache dir, or None for a serial run.

    The engine kernels compile with ``@njit(cache=True)`` into a shared
    ``__pycache__``. Under xdist, workers would race to write the same cache
    files on a cold cache. Giving each worker its own ``NUMBA_CACHE_DIR``
    removes the race at the cost of a one-time per-worker compile.

    ``worker`` is the value of ``PYTEST_XDIST_WORKER`` (e.g. ``"gw0"``), or
    None/empty when running serially (no isolation needed).
    """
    if not worker:
        return None
    return base / f"numba_cache_{worker}"


#: Threads each xdist worker may use. Chosen (PR follow-up to #148) as the balance
#: between contention and coverage: on the 28-cpu dev box it takes 14 workers x 28
#: threads = 392 runnable threads down to 112 (~4x oversubscribed instead of ~14x),
#: while leaving the 1-vs-8 comparison in
#: test_thread_policy.py::test_mortality_bit_identical_across_thread_counts a real
#: multi-threaded check. Must stay >= 4: tests/test_jit_determinism.py calls
#: set_num_threads(min(4, cpu_count)) and numba RAISES on n > max.
NUMBA_WORKER_THREAD_CAP = 8


def worker_numba_thread_cap(worker: str | None, available: int) -> str | None:
    """Per-worker ``NUMBA_NUM_THREADS`` value, or None for a serial run.

    Numba sizes its thread pool from the LOGICAL cpu count, while ``-n auto``
    spawns one worker per PHYSICAL cpu — so every worker independently believes it
    owns the whole machine. On the dev box that is 14 x 28 = 392 runnable threads
    on 28 cpus, and it is what pushed a ~15s Java subprocess past a 120s
    wall-clock budget (see tests/test_engine_java_comparison.py). Capping bounds
    the contention; ``.github/workflows/ci.yml`` documents the same hazard and
    prescribes a bounded ``-n`` as the manual remedy.

    This does NOT change any test's numbers: engine output is bit-identical across
    thread counts, asserted on two different fixtures by
    ``test_thread_policy.py::test_mortality_bit_identical_across_thread_counts``
    and ``test_jit_determinism.py::test_mortality_deterministic_across_thread_counts``.

    ``available`` is the caller's view of usable cpus — pass the affinity set size
    where the platform has one, since numba derives its own maximum the same way
    and honours taskset/cgroup pinning (the HPC/Apptainer target). The result is
    ``min(cap, available)`` so this can only ever LOWER the count: a flat cap would
    make a 4-cpu CI runner worse, not better.

    Returned as ``str`` because it goes straight into ``os.environ``, and it must
    be set BEFORE numba is first imported — numba reads the variable once, at
    import. (Contrast ``numba.set_num_threads``, which works post-import but is
    per-process runtime state.)
    """
    if not worker:
        return None
    return str(min(NUMBA_WORKER_THREAD_CAP, max(1, available)))
