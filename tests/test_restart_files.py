"""Restart files in Java 4.4.1's format (osmose/engine/restart.py; plan 2026-11-26).

Write: the schedule Java uses (after spinup, every frequency steps, and always at the end;
``simulation.restart.enabled`` absent means on). Read: ``simulation.restart.file`` resumes after
the file's step; ``population.initialization.file`` starts at step 0. The file layout is the
contract with Java and R (variable names, dtypes, units, the ``step`` attribute).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.engine.restart import read_snapshot, snapshot_due, write_snapshot

ROOT = Path(__file__).resolve().parents[1]
BALTIC = ROOT / "data/baltic/baltic_all-parameters.csv"


def _baltic(years: int, **extra) -> dict:
    cfg = dict(OsmoseConfigReader().read(BALTIC))
    cfg["simulation.time.nyear"] = str(years)
    cfg.update({k: str(v) for k, v in extra.items()})
    return cfg


def test_schedule_is_javas():
    # spinup 48, every 24 steps -> after steps 47, 71, 95 ...
    due = [s for s in range(120) if snapshot_due(s, 48, 24)]
    assert due == [47, 71, 95, 119]
    assert not any(snapshot_due(s, 0, 0) for s in range(50))  # no frequency: none mid-run


def test_settings_follow_java_defaults():
    c = EngineConfig.from_dict(_baltic(1, **{"simulation.restart.enabled": ""}))
    assert c.restart_write  # absent -> write (Java warns and writes)
    c = EngineConfig.from_dict(_baltic(1))
    assert not c.restart_write  # Baltic sets it false
    c = EngineConfig.from_dict(
        _baltic(
            1,
            **{
                "simulation.restart.spinup.nyear": "3",
                "simulation.restart.recordfrequency.ndt": "24",
            },
        )
    )
    assert (c.restart_spinup, c.restart_frequency) == (3 * 24, 24)


@pytest.fixture(scope="module")
def two_year_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("a")
    PythonEngine().run(_baltic(2, **{"simulation.restart.enabled": "true"}), out, seed=1)
    return out


def test_a_run_writes_its_end_snapshot_in_javas_layout(two_year_run):
    files = sorted((two_year_run / "restart").glob("*"))
    assert [f.name for f in files] == ["baltic_snapshot_step47.nc.0"]
    ds = xr.open_dataset(files[0])
    assert ds.attrs["step"] == "47"
    assert list(ds.data_vars) == [
        "species",
        "x",
        "y",
        "abundance",
        "age",
        "length",
        "weight",
        "trophiclevel",
    ]
    assert ds["species"].dtype == np.int32 and ds["abundance"].dtype == np.float64
    assert ds["age"].dtype == np.float32 and ds["weight"].attrs["units"] == "g"
    assert ds.sizes["nschool"] > 1000


def test_read_back_reproduces_the_table(two_year_run, tmp_path):
    cfg = EngineConfig.from_dict(_baltic(4))
    f = next((two_year_run / "restart").glob("*"))
    st, step = read_snapshot(f, cfg)
    assert step == 47 and len(st) > 1000
    again = tmp_path / "again.nc.0"
    write_snapshot(st, step, again, cfg)
    a, b = xr.open_dataset(f), xr.open_dataset(again)
    for v in a.data_vars:
        assert a[v].values.tobytes() == b[v].values.tobytes(), v  # write -> read -> write is exact


def test_a_restart_resumes_after_the_files_step(two_year_run, tmp_path):
    f = next((two_year_run / "restart").glob("*"))
    PythonEngine().run(_baltic(4, **{"simulation.restart.file": str(f)}), tmp_path, seed=1)
    bio = pd.read_csv(next(tmp_path.glob("*_biomass_Simu0.csv")), skiprows=1)
    assert bio["Time"].round(3).tolist() == [2.958, 3.958]  # years 2 and 3 only
    assert bio.drop(columns="Time").to_numpy().sum() > 0


def test_an_initialisation_file_starts_at_step_zero(two_year_run, tmp_path):
    f = next((two_year_run / "restart").glob("*"))
    res = PythonEngine().run_in_memory(
        _baltic(2, **{"population.initialization.file": str(f)}), seed=1
    )
    times = sorted(set(res.biomass()["Time"]))
    assert len(times) == 2 and min(times) < 1.0  # years 0 and 1


def test_the_ranked_file_name_is_accepted(two_year_run, tmp_path):
    f = next((two_year_run / "restart").glob("*"))
    plain = str(f)[: -len(".0")]  # Java looks for <name>.<rank> as well as <name>
    res = PythonEngine().run_in_memory(_baltic(3, **{"simulation.restart.file": plain}), seed=1)
    assert min(res.biomass()["Time"]) > 2.0


def test_a_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="restart file not found"):
        PythonEngine().run_in_memory(
            _baltic(2, **{"simulation.restart.file": str(tmp_path / "nope.nc")}), seed=1
        )


@pytest.mark.parametrize(
    ("var", "value", "message"),
    [
        ("x", 10_000.0, "outside the grid"),
        ("y", -5.0, "outside the grid"),
        ("x", 1.5, "not whole numbers"),
        ("abundance", -1.0, "abundance negative"),
        ("weight", np.nan, "weight negative or not finite"),
        ("age", 999.0, "beyond the lifespan"),
    ],
)
def test_a_corrupt_file_is_refused(two_year_run, tmp_path, var, value, message):
    """The cells index the kernels' arrays and the numbers feed them: refuse, never trust."""
    f = next((two_year_run / "restart").glob("*"))
    ds = xr.open_dataset(f).load()
    vals = ds[var].values.copy()
    vals[0] = value
    ds[var] = (ds[var].dims, vals, ds[var].attrs)
    bad = tmp_path / "bad.nc.0"
    ds.to_netcdf(bad)
    with pytest.raises(ValueError, match=message):
        PythonEngine().run_in_memory(_baltic(3, **{"simulation.restart.file": str(bad)}), seed=1)


def test_an_unlocated_school_is_accepted_and_marked_out(two_year_run, tmp_path):
    f = next((two_year_run / "restart").glob("*"))
    ds = xr.open_dataset(f).load()
    x, y = ds["x"].values.copy(), ds["y"].values.copy()
    x[0] = y[0] = -1.0
    ds["x"], ds["y"] = (("nschool",), x, ds["x"].attrs), (("nschool",), y, ds["y"].attrs)
    g = tmp_path / "out.nc.0"
    ds.to_netcdf(g)
    st, _ = read_snapshot(g, EngineConfig.from_dict(_baltic(3)))
    assert st.is_out[0] and not st.is_out[1:].any()


def test_a_traversing_path_is_refused():
    with pytest.raises(ValueError, match="contains '..'"):
        EngineConfig.from_dict(_baltic(3, **{"simulation.restart.file": "../../etc/x.nc"}))
