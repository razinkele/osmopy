"""Guard tests for pytest-xdist numba-cache isolation.

The engine kernels use ``@njit(cache=True)``, writing compiled artifacts into a
shared ``__pycache__``. Under xdist every worker would race to compile+write the
same files on a cold cache. ``worker_numba_cache_dir`` gives each worker its own
cache dir; conftest wires it into ``NUMBA_CACHE_DIR`` before any kernel compiles.
"""

from pathlib import Path

from tests._xdist_support import worker_numba_cache_dir


def test_serial_run_returns_none():
    """No PYTEST_XDIST_WORKER (serial run) -> no isolation needed."""
    assert worker_numba_cache_dir(None, Path("/tmp/base")) is None


def test_empty_worker_returns_none():
    """Empty/blank worker id is treated as serial."""
    assert worker_numba_cache_dir("", Path("/tmp/base")) is None


def test_worker_gets_dedicated_subdir():
    """A worker id yields a base-rooted, worker-named cache dir."""
    result = worker_numba_cache_dir("gw0", Path("/tmp/base"))
    assert result == Path("/tmp/base") / "numba_cache_gw0"


def test_distinct_workers_get_distinct_dirs():
    """Two workers never share a cache dir (the whole point)."""
    base = Path("/tmp/base")
    assert worker_numba_cache_dir("gw0", base) != worker_numba_cache_dir("gw1", base)
