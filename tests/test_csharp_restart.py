"""Restart files on the C# engine (osmose.engine.restart; plan 2026-11-26, R4).

The CLR runs in a child process (tests/_csharp_e2e_child.py). Checked: a C# run writes Java's
snapshot layout; the C# engine resumes from a snapshot at the step after it; and it loads a
starting population exactly as the Python engine does (the initial record, which no draw has
touched, is byte-identical). Needs the built engine; skips without it unless
OSMOSE_REQUIRE_CSHARP=1.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from tests.test_csharp_engine_e2e import _child
from tests.test_csharp_pipeline import engine_ok  # noqa: F401  (a fixture)

ROOT = Path(__file__).resolve().parents[1]


def _eec(years: int, **extra) -> dict:
    raw = dict(OsmoseConfigReader().read(ROOT / "data/eec_full/eec_all-parameters.csv"))
    raw["simulation.time.nyear"] = str(years)
    raw.update({k: str(v) for k, v in extra.items()})
    return raw


@pytest.fixture(scope="module")
def py_snapshot(tmp_path_factory):
    out = tmp_path_factory.mktemp("py")
    PythonEngine().run(_eec(1, **{"simulation.restart.enabled": "true"}), out, seed=3)
    return next((out / "restart").glob("*.nc.0"))


def test_a_csharp_run_writes_javas_snapshot(engine_ok, tmp_path):  # noqa: F811
    out = tmp_path / "cs"
    _child("files", _eec(1, **{"simulation.restart.enabled": "true"}), out)
    files = sorted((out / "restart").glob("*"))
    assert [f.name for f in files] == ["eec_snapshot_step23.nc.0"]
    ds = xr.open_dataset(files[0])
    assert ds.attrs["step"] == "23" and ds.sizes["nschool"] > 1000
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
    assert ds["species"].dtype == np.int32 and ds["weight"].attrs["units"] == "g"


def test_restart_disabled_writes_nothing(engine_ok, tmp_path):  # noqa: F811
    out = tmp_path / "cs"
    _child("files", _eec(1, **{"simulation.restart.enabled": "false"}), out)
    assert not list(out.rglob("*snapshot*"))


def test_csharp_resumes_after_the_snapshots_step(engine_ok, py_snapshot, tmp_path):  # noqa: F811
    out = tmp_path / "cs"
    _child("files", _eec(3, **{"simulation.restart.file": str(py_snapshot)}), out)
    bio = pd.read_csv(next(out.glob("*_biomass_Simu0.csv")), skiprows=1)
    assert bio["Time"].round(3).tolist() == [1.958, 2.958]  # years 1 and 2 only
    assert bio.drop(columns="Time").to_numpy().sum() > 0


def test_csharp_loads_the_population_exactly_as_python(engine_ok, py_snapshot, tmp_path):  # noqa: F811
    """population.initialization.file starts at step 0 from the file: the initial record (no
    step, no draw) must be byte-identical between the engines."""
    raw = _eec(
        1, **{"population.initialization.file": str(py_snapshot), "output.step0.include": "true"}
    )
    _child("outputs", raw, tmp_path / "cs.pkl", "3")
    cs = pickle.loads((tmp_path / "cs.pkl").read_bytes())["records"][0]
    from osmose.engine.simulate import simulate

    cfg, grid, rng, mv, mo = PythonEngine()._prepare_run(dict(raw), 3)
    py = simulate(cfg, grid, rng, movement_rngs=mv, mortality_rngs=mo)[0]
    assert cs.step == py.step == -1
    assert np.asarray(py.biomass).sum() > 0
    assert np.asarray(cs.biomass).tobytes() == np.asarray(py.biomass).tobytes()
    assert np.asarray(cs.abundance).tobytes() == np.asarray(py.abundance).tobytes()
