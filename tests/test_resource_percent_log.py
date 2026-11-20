"""OSMOSE 4.4.0's log form of the accessible fraction of a resource (#137).

A predator eats a resource in proportion to the share of the resource's size range that falls
in the predator's prey window. Java computes that share in linear size (the legacy form) or, since
4.4.0, in log size: ``ResourceSpecies.computePercentLog``, selected by
``simulation.resources.computePercent.legacy = false``. When the key is absent Java keeps the
linear form (``ResourceSpecies`` constructor, v4.4.1 lines 144-155), so nothing changes for a
config that does not ask.
"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest

import osmose.engine.processes.mortality as mort
from osmose.engine.config import EngineConfig
from osmose.engine.processes.resource_percent import resource_percent
from tests._tutorial_config import build_config

KEY = "simulation.resources.computepercent.legacy"  # the reader lowercases keys


def _tutorial(tmp_path, extra: dict[str, str]) -> dict[str, str]:
    return dict(build_config(tmp_path, n_year=1)) | extra


class TestTheKey:
    def test_absent_or_true_is_the_linear_form_and_false_the_log_form(self, tmp_path):
        base = _tutorial(tmp_path, {})
        base.pop(KEY, None)
        assert EngineConfig.from_dict(base).resource_percent_log is False
        assert EngineConfig.from_dict(base | {KEY: "true"}).resource_percent_log is False
        assert EngineConfig.from_dict(base | {KEY: "TRUE"}).resource_percent_log is False
        assert EngineConfig.from_dict(base | {KEY: "false"}).resource_percent_log is True

    def test_the_log_form_refuses_a_resource_whose_smallest_size_is_not_positive(self, tmp_path):
        # log(0) makes the denominator infinite and the share 0: Java then never feeds on the
        # resource, silently. Refused here, naming the resource.
        from osmose.engine.grid import Grid
        from osmose.engine.resources import ResourceState

        raw = _tutorial(tmp_path, {KEY: "false"})
        names = [k for k in raw if k.startswith("species.size.min.sp")]
        assert names, "the tutorial config has resources with a size range"
        raw[names[0]] = "0"
        with pytest.raises(ValueError, match="computepercent.legacy"):
            ResourceState(config=raw, grid=Grid.from_dimensions(ny=2, nx=2))
        raw[KEY] = "true"
        ResourceState(config=raw, grid=Grid.from_dimensions(ny=2, nx=2)).close()


class TestTheShare:
    # A resource from 0.02 to 2 cm; prey windows that cover part of it, all of it, or one end.
    CASES = ((0.05, 0.5), (0.02, 2.0), (0.02, 0.2), (1.0, 2.0), (0.021, 0.022))

    def test_the_linear_form_is_the_engines_formula_bit_for_bit(self):
        for lo, hi in self.CASES:
            want = (hi - lo) / (2.0 - 0.02)
            assert resource_percent(lo, hi, 0.02, 2.0, False) == want

    def test_the_log_form_is_javas_compute_percent_log(self):
        for lo, hi in self.CASES:
            want = (math.log(hi) - math.log(lo)) / (math.log(2.0) - math.log(0.02))
            assert resource_percent(lo, hi, 0.02, 2.0, True) == want
        # The two forms agree on the whole range and differ in between: a window from 0.02 to
        # 0.2 cm is a tenth of the linear range and half of the log range.
        assert resource_percent(0.02, 2.0, 0.02, 2.0, True) == 1.0
        assert resource_percent(0.02, 0.2, 0.02, 2.0, False) == pytest.approx(0.0909, abs=1e-4)
        assert resource_percent(0.02, 0.2, 0.02, 2.0, True) == pytest.approx(0.5)

    def test_the_kernels_use_the_same_function(self):
        if not mort._HAS_NUMBA:
            pytest.skip("Numba is not installed")
        for lo, hi in self.CASES:
            for log_form in (False, True):
                assert mort._resource_percent_nb(lo, hi, 0.02, 2.0, log_form) == resource_percent(
                    lo, hi, 0.02, 2.0, log_form
                )


def _run(raw: dict[str, str]) -> list:
    import osmose.engine.simulate as sim
    from osmose.engine import PythonEngine

    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 3)
    return sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)


@pytest.mark.parametrize("numba", [True, False], ids=["kernels", "python-path"])
def test_a_run_changes_only_when_the_log_form_is_asked_for(tmp_path, numba):
    """Both mortality paths: the batched kernels (production) and the path used without Numba.
    Absent and ``true`` are the same run, bit for bit; ``false`` is a different one."""
    if numba and not mort._HAS_NUMBA:
        pytest.skip("Numba is not installed")
    base = _tutorial(tmp_path, {})
    base.pop(KEY, None)
    with mock.patch.object(mort, "_HAS_NUMBA", numba):
        absent = _run(base)
        legacy = _run(base | {KEY: "true"})
        log = _run(base | {KEY: "false"})
    for a, b in zip(absent, legacy, strict=True):
        assert np.array_equal(a.biomass, b.biomass)
    assert any(not np.array_equal(a.biomass, b.biomass) for a, b in zip(absent, log, strict=True))


def test_the_csharp_engine_refuses_the_log_form_and_nothing_else_changes(tmp_path):
    """The C# engine does not implement the log form yet: it refuses a config that asks for it,
    by name, and is not affected by the key otherwise."""
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    base = _tutorial(tmp_path, {})
    base.pop(KEY, None)
    plain = csharp_engine_block_reason(base) or ""
    assert "resource_percent_log" not in plain
    assert "resource_percent_log" not in (csharp_engine_block_reason(base | {KEY: "true"}) or "")
    assert "resource_percent_log" in (csharp_engine_block_reason(base | {KEY: "false"}) or "")
