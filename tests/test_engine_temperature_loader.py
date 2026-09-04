"""Tests for temperature forcing: 4-D depth layers, the loader's Java precedence
(temperature.value > temperature.filename > None), the frame-count guard, per-species
`zlayer` routing inside `_bioen_step`, and gridded `f_o2`.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from osmose.engine.physical_data import PhysicalData
from osmose.engine.simulate import _load_temperature_data


def _write(tmp_path, arr, name="temperature", dims=None):
    dims = dims or (
        ["time", "layer", "latitude", "longitude"]
        if arr.ndim == 4
        else ["time", "latitude", "longitude"]
    )
    p = tmp_path / "t.nc"
    xr.Dataset({name: (dims, arr.astype(np.float32))}).to_netcdf(p)
    return p


def test_from_netcdf_4d_layers_and_3d_compat(tmp_path):
    a4 = np.arange(24 * 2 * 3 * 4, dtype=float).reshape(24, 2, 3, 4)
    pd4 = PhysicalData.from_netcdf(_write(tmp_path, a4), varname="temperature", nsteps_year=24)
    assert pd4.n_layers == 2
    np.testing.assert_array_equal(pd4.get_grid(25, layer=1), a4[1, 1])  # step % 24, layer 1
    assert pd4.get_value(0, 2, 3, layer=0) == a4[0, 0, 2, 3]
    with pytest.raises(IndexError):
        pd4.get_grid(0, layer=2)
    a3 = np.ones((24, 3, 4))
    pd3 = PhysicalData.from_netcdf(
        _write(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, a3),
        varname="temperature",
        nsteps_year=24,
    )
    assert (
        pd3.n_layers == 1
        and pd3.get_grid(5).shape == (3, 4)
        and pd3.get_grid(5, layer=0).shape
        == (
            3,
            4,
        )
    )


def test_loader_java_precedence_value_then_file_then_none(tmp_path):
    p = _write(tmp_path, np.full((24, 2, 3, 4), 7.0))
    raw = {
        "temperature.filename": str(p),
        "temperature.varname": "temperature",
        "temperature.nsteps.year": "24",
        "simulation.time.ndtperyear": "24",
    }
    assert _load_temperature_data(raw, None).n_layers == 2
    raw["temperature.value"] = "5.5"
    assert _load_temperature_data(raw, None).is_constant  # .value wins (Java PhysicalData.init)
    assert _load_temperature_data({"simulation.time.ndtperyear": "24"}, None) is None


def test_loader_frame_mismatch_and_factor_offset(tmp_path):
    p = _write(tmp_path, np.full((12, 2, 3, 4), 7.0))
    raw = {
        "temperature.filename": str(p),
        "temperature.varname": "temperature",
        "temperature.nsteps.year": "12",
        "simulation.time.ndtperyear": "24",
    }
    with pytest.raises(ValueError, match="24"):
        _load_temperature_data(raw, None)
    p2 = _write(
        tmp_path / "c" if (tmp_path / "c").mkdir() is None else tmp_path, np.full((24, 3, 4), 7.0)
    )
    raw2 = {
        "temperature.filename": str(p2),
        "temperature.varname": "temperature",
        "temperature.nsteps.year": "24",
        "simulation.time.ndtperyear": "24",
        "temperature.offset": "2.0",
    }
    assert _load_temperature_data(raw2, None).get_value(0, 0, 0) == 9.0


def test_bioen_without_temperature_source_raises():
    from tests.test_bioen_orchestration import _make_bioen_config_dict
    from osmose.engine import PythonEngine

    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    cfg.pop("temperature.value", None)
    cfg["simulation.time.nyear"] = "1"
    with pytest.raises(ValueError, match="temperature"):
        PythonEngine().run_in_memory(cfg, seed=0)


def test_bioen_step_reads_assigned_layer_and_skips_out_schools():
    from tests.test_bioen_orchestration import _make_bioen_config_dict, _make_school_state
    from osmose.engine.config import EngineConfig
    from osmose.engine.simulate import _bioen_step

    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    cfg["species.zlayer.sp1"] = "1"
    config = EngineConfig.from_dict(cfg)
    st = _make_school_state(n_schools=6, n_species=2)
    st = st.replace(
        cell_x=np.array([1, 1, 2, 2, 0, 0], dtype=np.int32),
        cell_y=np.array([1, 1, 2, 2, 0, 0], dtype=np.int32),
        is_out=np.array([False, False, False, False, True, True]),
    )
    grid = np.zeros((24, 2, 5, 5))
    grid[:, 0] = 4.0
    grid[:, 1] = 12.0
    td = PhysicalData(data=grid, constant=None, nsteps_year=24)
    cap: dict = {}
    _bioen_step(st, config, td, step=0, debug_capture=cap)
    t = cap["temp_c"]
    assert (
        t[0] == 4.0 and t[1] == 12.0 and t[2] == 4.0 and t[3] == 12.0
    )  # sp0 -> layer 0, sp1 -> layer 1
    assert np.isnan(t[4]) and np.isnan(t[5])  # out schools: no temperature, no budget


def test_gridded_o2_reaches_f_o2():
    from tests.test_bioen_orchestration import _make_bioen_config_dict, _make_school_state
    from osmose.engine.config import EngineConfig
    from osmose.engine.processes.oxygen_function import f_o2
    from osmose.engine.simulate import _bioen_step

    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    cfg["simulation.bioen.fo2.enabled"] = "true"
    config = EngineConfig.from_dict(cfg)
    st = _make_school_state(n_schools=4, n_species=2).replace(
        preyed_biomass=np.full(4, 1e-3),
        cell_x=np.zeros(4, dtype=np.int32),
        cell_y=np.zeros(4, dtype=np.int32),
    )
    o2 = PhysicalData(data=np.full((24, 5, 5), 3.0), constant=None, nsteps_year=24)
    td = PhysicalData.from_constant(10.0)
    cap: dict = {}
    _bioen_step(st, config, td, step=0, o2_data=o2, debug_capture=cap)
    expected = f_o2(np.array(3.0), config.bioen_o2_c1[0], config.bioen_o2_c2[0])
    assert cap["f_o2"][0] == pytest.approx(float(expected))
