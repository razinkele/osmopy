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
    assert tp.resolve_engine_threads(20) == 20  # honored
    assert tp.resolve_engine_threads(100) == 28  # capped to logical budget
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
