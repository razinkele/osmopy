"""output.start.year (Java OutputManager): outputs only from that year on; the UI refuses a run
that would write nothing (owner decision 2026-11-25); keys Java 4.4.1 never reads are inert."""

from __future__ import annotations

from pathlib import Path

import pytest

import osmose.engine.config_validation as cv
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parents[1]
BOB = ROOT / "data/examples/osm_all-parameters.csv"


def _bob(nyear: int, start: str | None = None) -> dict:
    cfg = dict(OsmoseConfigReader().read(BOB))
    cfg["simulation.time.nyear"] = str(nyear)
    if start is not None:
        cfg["output.start.year"] = start
    return cfg


def test_records_before_the_start_year_are_dropped():
    res = PythonEngine().run_in_memory(_bob(3, "1"), seed=1)
    times = sorted(set(res.biomass()["Time"]))  # BoB records every step (24 a year)
    assert len(times) == 2 * 24 and min(times) >= 1.0  # years 1 and 2 kept, year 0 dropped
    full = PythonEngine().run_in_memory(_bob(3, "0"), seed=1)
    assert len(set(full.biomass()["Time"])) == 3 * 24


def test_a_run_shorter_than_the_start_year_writes_nothing_and_warns(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="osmose"):
        PythonEngine().run(_bob(2), tmp_path, seed=1)  # BoB starts at 20
    assert not list(tmp_path.rglob("*_biomass_Simu0.csv"))
    assert any("output.start.year is 20" in r.getMessage() for r in caplog.records)


def test_the_ui_refuses_such_a_run_and_honours_the_overrides():
    from ui.pages.run import start_year_block_reason

    cfg = _bob(50)
    assert start_year_block_reason(cfg, "") is None  # 50 years > 20
    msg = start_year_block_reason(cfg, "simulation.time.nyear=5")
    assert msg and "output.start.year is 20" in msg and "5 years" in msg
    assert start_year_block_reason(cfg, "simulation.time.nyear=5\noutput.start.year=0") is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("population.initialization.method.sp0", "biomass"),
        ("population.initialization.method", "spectrum"),
        ("predation.accessibility.stage.threshold.sp3", "1.5"),
        ("species.conversion2tons.sp9", "2.0"),
    ],
)
def test_keys_java_441_never_reads_are_inert(key, value):
    """Deprecated by Java 4.x's Releases and read by nothing in 4.4.1, whatever their value."""
    assert cv.warn_unread_java_only_keys({key: value}) == []


def test_bay_of_biscay_lists_nothing_any_more():
    cfg = OsmoseConfigReader().read(BOB)
    assert cv.warn_unread_java_only_keys(cfg) == []
