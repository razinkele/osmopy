"""O2->benthos K coupling: factor math, gating, config validation (spec Phase 2a)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from osmose.engine.grid import Grid
from osmose.engine.physical_data import PhysicalData
from osmose.engine.processes.oxygen_function import f_o2_hill
from osmose.engine.resources import ResourceState
from osmose.engine.simulate import _load_oxygen_data


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


def test_oxygen_row_constant_mode():
    # _oxygen_row's constant-mode branch: PhysicalData.from_constant has no spatial grid, so
    # get_scalar() must be broadcast to every cell rather than going through get_grid/regrid.
    grid = Grid.from_dimensions(ny=2, nx=2)
    oxygen = PhysicalData.from_constant(90.0)
    rs = ResourceState(config=_cfg(), grid=grid, oxygen=oxygen)
    rs.update(step=0)
    expected_factor = f_o2_hill(np.full(4, 90.0), 60.0, 3.0)
    np.testing.assert_allclose(rs.oxygen_factor_last, expected_factor, rtol=1e-12)
    base = 1000.0 / 4 * 0.8
    np.testing.assert_allclose(rs.biomass[0], base * expected_factor, rtol=1e-12)


def test_oxygen_row_regrid_path_asymmetric_grid():
    # _oxygen_row's regrid branch: oxygen frame shape (4, 6) != model grid shape (2, 3), so
    # _regrid_to_model's nearest-neighbor index mapping is exercised. Values are per-cell
    # distinguishable (0..23, no repeats) and the grid is asymmetric (ny != nx) so a transposed
    # axis order anywhere in the pipeline would pick different cells and fail the assertion.
    grid = Grid.from_dimensions(ny=2, nx=3)
    oxy_ny, oxy_nx = 4, 6
    frame = np.arange(oxy_ny * oxy_nx, dtype=np.float64).reshape(oxy_ny, oxy_nx)
    oxygen = PhysicalData(data=frame[np.newaxis, :, :], constant=None, nsteps_year=1)

    cfg = _cfg()
    cfg["ltl.biomass.total.rsc0"] = "600.0"  # 6 ocean cells -> 100/cell before access & oxygen
    rs = ResourceState(config=cfg, grid=grid, oxygen=oxygen)
    rs.update(step=0)

    # Same nearest-neighbor index math as _regrid_to_model: rows=[0,2], cols=[0,2,4].
    picked = frame[np.ix_([0, 2], [0, 2, 4])].flatten()
    expected_factor = f_o2_hill(picked, 60.0, 3.0)
    np.testing.assert_allclose(rs.oxygen_factor_last, expected_factor, rtol=1e-12)

    base = 600.0 / 6 * 0.8
    np.testing.assert_allclose(rs.biomass[0], base * expected_factor, rtol=1e-12)


def _write_oxygen_nc(path, n_frames, ny=2, nx=2, value=90.0):
    data = np.full((n_frames, ny, nx), value, dtype=np.float64)
    ds = xr.Dataset({"o2b": (["time", "y", "x"], data)})
    ds.to_netcdf(path)
    ds.close()


def test_frame_count_guard_raises_on_mismatch(tmp_path):
    # PhysicalData._nsteps_year is decorative (get_value/get_grid index step % data.shape[0]),
    # so the guard must compare the LOADED array's actual frame count against
    # simulation.time.ndtperyear — a config-declared oxygen.nsteps.year=24 must not mask a file
    # that actually has 12 frames.
    nc_path = tmp_path / "oxygen_12frames.nc"
    _write_oxygen_nc(nc_path, n_frames=12)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "oxygen.nsteps.year": "24",  # deliberately-wrong declared value, must not save it
        "simulation.time.ndtperyear": "24",
    }
    with pytest.raises(ValueError, match=r"12 frame.*ndtperyear=24"):
        _load_oxygen_data(raw_config, None)


def test_frame_count_guard_passes_on_match(tmp_path):
    nc_path = tmp_path / "oxygen_24frames.nc"
    _write_oxygen_nc(nc_path, n_frames=24)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
    }
    result = _load_oxygen_data(raw_config, None)
    assert result is not None
    assert not result.is_constant


def _write_oxygen_nc_4d(path, n_frames, n_layers=2, ny=2, nx=2, value=90.0):
    data = np.full((n_frames, n_layers, ny, nx), value, dtype=np.float64)
    ds = xr.Dataset({"o2b": (["time", "z", "y", "x"], data)})
    ds.to_netcdf(path)
    ds.close()


def test_zlayer_guard_raises_on_4d_file_with_nonzero_zlayer(tmp_path):
    # Task 7 review Finding 1: Java's OxygenFunction resolves a per-species depth layer
    # for oxygen exactly as TempFunction does for temperature (OxygenFunction.java:130-133,
    # same PhysicalData.getValue(int, Cell) mechanism as TempFunction.java:162) -- oxygen
    # forcing CAN be 4-D (time, z, y, x) in Java. This port doesn't route
    # species.zlayer.sp{idx} into the oxygen branch of _bioen_step, so a 4-D oxygen file
    # combined with a nonzero zlayer must raise rather than silently read layer 0 and drop
    # the configured depth with no signal -- when that oxygen branch is actually reachable
    # (bioenergetics + its fo2 limitation both enabled; fo2 defaults true so it's implicit).
    nc_path = tmp_path / "oxygen_4d.nc"
    _write_oxygen_nc_4d(nc_path, n_frames=24, n_layers=2)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "module.bioenergetics.enabled": "true",
        "species.zlayer.sp0": "0",
        "species.zlayer.sp1": "1",  # nonzero -- must trip the guard
    }
    with pytest.raises(ValueError, match=r"depth axis.*species\.zlayer\.sp1"):
        _load_oxygen_data(raw_config, None)


def test_zlayer_guard_passes_when_all_zlayer_zero_or_unset(tmp_path):
    # sp0 explicitly zero, sp1 unset entirely (schema default is 0 too) -- neither should
    # trip the guard, proving it isn't a blanket ban on 4-D oxygen files. Bioenergetics is
    # enabled (fo2 defaults true) so this test actually enters the guarded block instead
    # of short-circuiting on the bioen/fo2 gate -- it is the zero/unset handling that must
    # carry this test, not the gate.
    nc_path = tmp_path / "oxygen_4d_ok.nc"
    _write_oxygen_nc_4d(nc_path, n_frames=24, n_layers=2)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "module.bioenergetics.enabled": "true",
        "species.zlayer.sp0": "0",
    }
    result = _load_oxygen_data(raw_config, None)
    assert result is not None
    assert result.n_layers == 2


def test_zlayer_guard_passes_on_3d_file_even_with_nonzero_zlayer(tmp_path):
    # A 3-D oxygen file has no depth axis to be wrongly ignored -- zlayer is moot here,
    # so the guard must not fire regardless of what zlayer is configured. (n_layers == 1
    # short-circuits before the bioen/fo2 gate is even consulted, so bioen is enabled here
    # only for realism, not because it matters to this assertion.)
    nc_path = tmp_path / "oxygen_3d.nc"
    _write_oxygen_nc(nc_path, n_frames=24)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "module.bioenergetics.enabled": "true",
        "species.zlayer.sp0": "1",
    }
    result = _load_oxygen_data(raw_config, None)
    assert result is not None
    assert result.n_layers == 1


def test_zlayer_guard_passes_on_single_layer_4d_file(tmp_path):
    # (time, 1, y, x) is technically 4-D but has only one layer, so there's no second
    # layer to be wrongly ignored. The guard tests n_layers > 1, not raw ndim == 4. (Also
    # short-circuits before the bioen/fo2 gate; bioen enabled here only for realism.)
    nc_path = tmp_path / "oxygen_4d_single_layer.nc"
    _write_oxygen_nc_4d(nc_path, n_frames=24, n_layers=1)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "module.bioenergetics.enabled": "true",
        "species.zlayer.sp0": "1",
    }
    result = _load_oxygen_data(raw_config, None)
    assert result is not None
    assert result.n_layers == 1


def test_zlayer_guard_passes_when_fo2_disabled_even_with_nonzero_zlayer(tmp_path):
    # Fix round 2, Problem 1: bioenergetics enabled with a nonzero zlayer driving
    # TEMPERATURE depth sampling (simulate.py:513) is a legitimate config even when the
    # oxygen forcing happens to be multi-layer, as long as simulation.bioen.fo2.enabled is
    # false -- _bioen_step's oxygen branch (`if config.bioen_fo2_enabled and o2_data is
    # not None`) never executes for any species in that case, so there is nothing for the
    # unwired oxygen zlayer routing to silently get wrong. The round-1 guard was
    # unconditional on this and rejected it; this must now load cleanly.
    nc_path = tmp_path / "oxygen_4d_fo2_off.nc"
    _write_oxygen_nc_4d(nc_path, n_frames=24, n_layers=2)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "module.bioenergetics.enabled": "true",
        "simulation.bioen.fo2.enabled": "false",
        "species.zlayer.sp0": "1",  # nonzero, but only drives temperature sampling here
    }
    result = _load_oxygen_data(raw_config, None)
    assert result is not None
    assert result.n_layers == 2


def test_zlayer_guard_passes_when_bioenergetics_module_disabled(tmp_path):
    # Fix round 2, Problem 1: the production shape of the legitimate config -- bottom-O2
    # -> benthos-K coupling (resources.py:323, ResourceState.update) is the ONLY consumer
    # of a multi-layer oxygen file when module.bioenergetics.enabled=false. That coupling
    # calls o2.get_grid(step) at layer 0, has no species dimension, and never consults
    # zlayer at all, so a nonzero zlayer here is inert with respect to oxygen and must not
    # be rejected.
    nc_path = tmp_path / "oxygen_4d_bioen_off.nc"
    _write_oxygen_nc_4d(nc_path, n_frames=24, n_layers=2)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "species.zlayer.sp0": "1",  # nonzero, but bioenergetics module is off entirely
    }
    result = _load_oxygen_data(raw_config, None)
    assert result is not None
    assert result.n_layers == 2


def test_zlayer_guard_treats_blank_value_as_unset(tmp_path):
    # Fix round 2, Problem 3: the config reader's separator-stripping can legitimately
    # produce an empty-string value for a stray "species.zlayer.spN;" line
    # (value.rstrip(";,:\t =") -> ""). That is "unset", matching the schema default of 0
    # -- not a suspicious, unparseable value to raise on. float("") raising and being
    # caught would previously (fix round 1) have been treated as "unparseable -- stay
    # loud", tripping the guard with a message asserting the user "set a nonzero depth
    # layer", which was false for a blank value.
    nc_path = tmp_path / "oxygen_4d_blank.nc"
    _write_oxygen_nc_4d(nc_path, n_frames=24, n_layers=2)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "module.bioenergetics.enabled": "true",
        "species.zlayer.sp0": "",  # blank -- must not trip the guard
    }
    result = _load_oxygen_data(raw_config, None)
    assert result is not None
    assert result.n_layers == 2


def test_zlayer_guard_still_raises_on_genuinely_unparseable_value(tmp_path):
    # Companion to the blank-value test above: a non-blank, genuinely unparseable value
    # (a typo, not a reader artifact) must still trip the guard -- "stay loud" survives
    # for the case it actually protects against.
    nc_path = tmp_path / "oxygen_4d_garbage.nc"
    _write_oxygen_nc_4d(nc_path, n_frames=24, n_layers=2)
    raw_config = {
        "oxygen.filename": str(nc_path),
        "oxygen.varname": "o2b",
        "simulation.time.ndtperyear": "24",
        "module.bioenergetics.enabled": "true",
        "species.zlayer.sp0": "abc",
    }
    with pytest.raises(ValueError, match=r"depth axis.*species\.zlayer\.sp0"):
        _load_oxygen_data(raw_config, None)
