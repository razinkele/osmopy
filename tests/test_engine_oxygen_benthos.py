"""O2->benthos K coupling: factor math, gating, config validation (spec Phase 2a)."""

from __future__ import annotations

import numpy as np
from osmose.engine.grid import Grid
from osmose.engine.physical_data import PhysicalData
from osmose.engine.processes.oxygen_function import f_o2_hill
from osmose.engine.resources import ResourceState


def _cfg(enabled="true"):
    return {
        "simulation.nresource": "1",
        "simulation.time.ndtperyear": "24",
        "ltl.name.rsc0": "Benthos",
        "ltl.size.min.rsc0": "0.5",
        "ltl.size.max.rsc0": "10.0",
        "ltl.tl.rsc0": "2.5",
        "ltl.accessibility2fish.rsc0": "0.8",
        "ltl.biomass.total.rsc0": "1000.0",
        "ltl.oxygen.benthos.enabled": enabled,
        "ltl.oxygen.benthos.c50": "60.0",
        "ltl.oxygen.benthos.n": "3.0",
        "ltl.oxygen.benthos.rsc": "Benthos",
    }


def _oxygen(ny=2, nx=2, values=(0.0, 60.0, 90.0, 300.0)):
    data = np.array(values, dtype=np.float64).reshape(1, ny, nx)
    return PhysicalData(data=data, constant=None, nsteps_year=1)


def test_hill_response_shape():
    # Normalized: ~1 when oxygenated (no artifact cut), 0.5 at c50, collapsing under hypoxia.
    o2 = np.array([0.0, 30.0, 60.0, 90.0, 300.0])
    f = f_o2_hill(o2, 60.0, 3.0)
    assert f[0] == 0.0
    np.testing.assert_allclose(f[2], 0.5, rtol=1e-12)
    assert f[1] < 0.15 and 0.7 < f[3] < 0.85 and f[4] > 0.98


def test_factor_applied_to_benthos_k():
    grid = Grid.from_dimensions(ny=2, nx=2)
    rs = ResourceState(config=_cfg(), grid=grid, oxygen=_oxygen())
    rs.update(step=0)
    base = 1000.0 / 4 * 0.8  # uniform per-cell K without oxygen
    expected = base * f_o2_hill(np.array([0.0, 60.0, 90.0, 300.0]), 60.0, 3.0)
    np.testing.assert_allclose(rs.biomass[0], expected, rtol=1e-12)
    # anoxic cell -> zero benthos; oxygenated cell -> essentially unreduced
    assert rs.biomass[0][0] == 0.0
    assert rs.biomass[0][3] > 0.98 * base


def test_disabled_or_no_oxygen_is_identity():
    grid = Grid.from_dimensions(ny=2, nx=2)
    base = 1000.0 / 4 * 0.8
    for rs in (
        ResourceState(config=_cfg(enabled="false"), grid=grid, oxygen=_oxygen()),
        ResourceState(config=_cfg(), grid=grid, oxygen=None),
    ):
        rs.update(step=0)
        np.testing.assert_allclose(rs.biomass[0], np.full(4, base), rtol=1e-12)


def test_named_resource_only():
    grid = Grid.from_dimensions(ny=2, nx=2)
    cfg = _cfg()
    cfg["ltl.oxygen.benthos.rsc"] = "SomethingElse"
    rs = ResourceState(config=cfg, grid=grid, oxygen=_oxygen())
    rs.update(step=0)
    np.testing.assert_allclose(rs.biomass[0], np.full(4, 1000.0 / 4 * 0.8), rtol=1e-12)


def test_new_keys_validate_clean():
    # validate() has NO default for `mode` — omitting it raises TypeError (review finding).
    # The new ltl.oxygen.benthos.* keys must be clean via AST capture of resources.py's literal
    # cfg.get reads, NOT via an allowlist entry (which would break the frozen-snapshot guard).
    from osmose.engine.config_validation import validate

    issues = validate(_cfg(), mode="warn")
    assert not [i for i in issues if "ltl.oxygen" in getattr(i, "key", "")]
