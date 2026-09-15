"""Guard tests for pytest-xdist numba-cache isolation.

The engine kernels use ``@njit(cache=True)``, writing compiled artifacts into a
shared ``__pycache__``. Under xdist every worker would race to compile+write the
same files on a cold cache. ``worker_numba_cache_dir`` gives each worker its own
cache dir; conftest wires it into ``NUMBA_CACHE_DIR`` before any kernel compiles.
"""

from pathlib import Path

from tests._xdist_support import worker_numba_cache_dir, worker_numba_thread_cap


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


# ---------------------------------------------------------------------------
# Per-worker Numba thread cap
# ---------------------------------------------------------------------------
#
# Each xdist worker's Numba defaults to one thread per LOGICAL cpu, so `-n auto`
# (one worker per PHYSICAL cpu) oversubscribes badly: on the dev box 14 workers x
# 28 threads = 392 runnable threads on 28 cpus. That is what pushed a ~15s Java
# subprocess past a 120s wall-clock budget (PR #148). Capping each worker bounds
# the contention. Engine results do NOT depend on the thread count -- see
# test_thread_policy.py::test_mortality_bit_identical_across_thread_counts.


def test_thread_cap_serial_run_returns_none():
    """Serial runs have no cross-worker contention, so they keep the full width."""
    assert worker_numba_thread_cap(None, 28) is None


def test_thread_cap_empty_worker_returns_none():
    """Empty/blank worker id is treated as serial, matching the cache-dir helper."""
    assert worker_numba_thread_cap("", 28) is None


def test_thread_cap_applies_under_xdist():
    """A many-core box is capped to 8 -- the chosen contention/coverage balance."""
    assert worker_numba_thread_cap("gw0", 28) == "8"


def test_thread_cap_never_exceeds_available():
    """The cap may only LOWER the count, never raise it.

    A flat "8" would make a 4-core CI runner WORSE (4 workers x 8 threads on 4
    cpus). numba also treats this value as its max, so inflating it past the real
    availability is meaningless at best.
    """
    assert worker_numba_thread_cap("gw0", 4) == "4"
    assert worker_numba_thread_cap("gw0", 1) == "1"


def test_thread_cap_floor_respects_jit_determinism():
    """The cap must stay >= 4 on any box that HAS 4 cpus.

    tests/test_jit_determinism.py calls set_num_threads(min(4, cpu_count)), and
    numba RAISES on set_num_threads(n > max) rather than skipping. A cap below 4
    on a >=4-core box would turn that test red instead of leaving it meaningful.
    """
    assert int(worker_numba_thread_cap("gw0", 28)) >= 4
    assert int(worker_numba_thread_cap("gw0", 8)) >= 4
    assert int(worker_numba_thread_cap("gw0", 4)) >= 4


def test_thread_cap_returns_str_for_environ():
    """The value goes straight into os.environ, which rejects non-str."""
    assert isinstance(worker_numba_thread_cap("gw0", 28), str)
