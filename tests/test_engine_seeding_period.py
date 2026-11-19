"""The seeding period when ``population.seeding.year.max`` is not set (issue #334).

Java seeds every species for the same period, the longest lifespan of any species
(``ReproductionProcess.init``, the same in the 4.3.3-era tree and at v4.4.1):

    yearMaxSeeding = 0;
    if (!isNull("population.seeding.year.max")) { yearMaxSeeding = getInt(...) * nStepYear; }
    else { for (i ...) yearMaxSeeding = Math.max(yearMaxSeeding, getSpecies(i).getLifespanDt()); }

The Python engine used each species' own lifespan. The two agree on Baltic and on EEC, where
no species has an empty spawning stock between its own lifespan and the longest one. They
disagree on Bay of Biscay and on ``data/minimal`` from the fifth year on, where short-lived
species live on seeding alone.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig

DATA = Path(__file__).resolve().parents[1] / "data"
KEY = "population.seeding.year.max"


def _minimal(tmp_path: Path, **over: str) -> dict:
    """``data/minimal`` through the real reader: Anchovy lives 4 years, Hake 12; 12 steps a
    year. Neither ever spawns for real in this config, so both live on seeding alone."""
    dst = tmp_path / "minimal"
    if not dst.exists():
        shutil.copytree(DATA / "minimal", dst)
    raw = dict(OsmoseConfigReader().read(str(dst / "osm_all-parameters.csv")))
    assert KEY not in raw
    return raw | {k.replace("__", "."): v for k, v in over.items()}


def _biomass(raw: dict, years: int) -> np.ndarray:
    raw = raw | {"simulation.time.nyear": str(years)}
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 7)
    records = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    return np.array([np.asarray(r.biomass)[: config.n_species] for r in records])


def test_without_the_key_every_species_is_seeded_for_the_longest_lifespan(tmp_path):
    config = EngineConfig.from_dict(_minimal(tmp_path))
    assert config.species_names == ["Anchovy", "Hake"]
    assert (config.lifespan_dt[:2] // config.n_dt_per_year).tolist() == [4, 12]
    assert config.seeding_max_step[:2].tolist() == [144, 144]  # 12 years, for both


def test_the_key_still_sets_one_period_for_every_species(tmp_path):
    config = EngineConfig.from_dict(_minimal(tmp_path, population__seeding__year__max="3"))
    assert config.seeding_max_step[:2].tolist() == [36, 36]


@pytest.mark.parametrize(
    ("name", "path", "years"),
    [
        ("baltic", "baltic/baltic_all-parameters.csv", 20),
        ("eec_full", "eec_full/eec_all-parameters.csv", 25),
        ("bay_of_biscay", "examples/osm_all-parameters.csv", 15),
    ],
)
def test_the_shipped_configs_that_leave_the_key_unset(name, path, years):
    raw = dict(OsmoseConfigReader().read(str(DATA / path)))
    assert KEY not in raw, f"{name} now sets the key: this case no longer tests the default"
    config = EngineConfig.from_dict(raw)
    n = config.n_species
    assert set(config.seeding_max_step[:n].tolist()) == {years * config.n_dt_per_year}
    assert int(config.lifespan_dt[:n].max()) == years * config.n_dt_per_year


def test_a_short_lived_species_is_seeded_past_its_own_lifespan(tmp_path):
    """Six years of ``data/minimal``. Anchovy's own lifespan ends after year 4; Java goes on
    seeding it until year 12. The run with the key unset must be the run with the key at 12,
    number for number, and not the run with the key at 4 (the old rule for Anchovy)."""
    raw = _minimal(tmp_path)
    default = _biomass(raw, 6)
    longest = _biomass(raw | {KEY: "12"}, 6)
    own = _biomass(raw | {KEY: "4"}, 6)
    assert default.tobytes() == longest.tobytes()
    # The control: the period matters here, so the equality above is not empty.
    assert default[:4].tobytes() == own[:4].tobytes()  # the first four years are the same
    assert default[-1, 0] > own[-1, 0] > 0  # year 6: more Anchovy when it is still seeded
