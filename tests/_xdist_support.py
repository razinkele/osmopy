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
