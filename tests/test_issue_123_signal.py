"""#123 follow-up (2026-11-24): the Java-only-keys warning lists only keys that can matter, and
the two latent parity gaps it used to hide are closed.

- Inert values are not listed: a switch that is off, labels/class names/ncpu, a single replicate,
  and the identity fishing-period triple (one period at 0, multiplier 1 = constant fishing).
  Benguela went from 81 listed keys to 7.
- output.cutoff.enabled = false is honoured (Java applies cutoff ages only when it is true).
- predation.accessibility.stage.structure = size raises: the engine reads stage thresholds as ages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import config_validation as cv

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _fresh_dedup():
    cv._WARNED_JAVA_ONLY_KEYS.clear()
    yield
    cv._WARNED_JAVA_ONLY_KEYS.clear()


def test_off_switches_and_labels_are_not_listed_but_on_switches_are():
    cfg = {
        "output.mortality.enabled": "FALSE",  # off: inert
        "output.biomass.bytl.enabled": "true",  # on: an output this engine won't write
        "fisheries.name.fsh0": "trawl",  # label
        "grid.java.classname": "fr.ird.osmose.grid.NcGrid",
        "simulation.ncpu": "8",
        "simulation.nsimulation": "1",
    }
    assert cv.warn_unread_java_only_keys(cfg) == ["output.biomass.bytl.enabled"]


def test_more_than_one_replicate_is_listed():
    assert cv.warn_unread_java_only_keys({"simulation.nsimulation": "3"}) == [
        "simulation.nsimulation"
    ]


def test_identity_fishing_periods_are_inert_real_seasonality_is_not():
    identity = {
        "fisheries.period.number.fsh0": "1",
        "fisheries.period.start.fsh0": "0",
        "fisheries.rate.byperiod.fsh0": "1",
    }
    assert cv.warn_unread_java_only_keys(identity) == []
    seasonal = {**identity, "fisheries.period.number.fsh0": "2"}
    assert cv.warn_unread_java_only_keys(seasonal) == sorted(identity)


def test_benguela_lists_only_the_keys_that_can_matter():
    cfg = OsmoseConfigReader().read(ROOT / "data/benguela/benguela_all-parameters.csv")
    listed = cv.warn_unread_java_only_keys(cfg)
    assert len(cv.java_only_keys_set(cfg)) > 60  # the raw set is still complete
    assert listed == [
        "output.file.prefix",
        "output.fisheries.enabled",
        "output.start.year",  # 60 on a 15-yr run: Java would write nothing
        "simulation.restart.recordfrequency.ndt",
        "simulation.restart.spinup.nyear",
    ]


def test_values_matching_this_engine_are_not_listed():
    cfg = {
        "output.biomass.enabled": "true",  # always written here
        "output.abundance.enabled": "true",
        "output.yield.biomass.enabled": "true",
        "output.mortality.enabled": "true",
        "output.diet.composition.enabled": "true",
        "output.diet.pressure.enabled": "true",  # written with diet output
        "output.start.year": "0",
        "output.dir.path": "output",
    }
    assert cv.warn_unread_java_only_keys(cfg) == []
    assert cv.warn_unread_java_only_keys({"output.start.year": "5"}) == ["output.start.year"]
    # predator pressure without diet output on is NOT produced -> listed
    assert cv.warn_unread_java_only_keys({"output.diet.pressure.enabled": "true"}) == [
        "output.diet.pressure.enabled"
    ]


def test_legacy_grid_keys_inert_only_with_a_netcdf_grid():
    corners = {"grid.upleft.lat": "66", "grid.lowright.lon": "30", "grid.mask.file": "m.csv"}
    assert cv.warn_unread_java_only_keys({**corners, "grid.netcdf.file": "g.nc"}) == []
    assert cv.warn_unread_java_only_keys(corners) == sorted(corners)


def test_baltic_lists_exactly_the_unimplemented_features():
    cfg = OsmoseConfigReader().read(ROOT / "data/baltic/baltic_all-parameters.csv")
    listed = cv.warn_unread_java_only_keys(cfg)
    assert len(listed) == 28
    assert not [k for k in listed if k.startswith("grid.")]
    assert "output.biomass.enabled" not in listed


def _cutoff(cfg_extra: dict[str, str]):
    from osmose.engine.config import _parse_output_flags

    cfg = {"output.cutoff.age.sp0": "0.5", "output.cutoff.age.sp1": "1.0", **cfg_extra}
    return _parse_output_flags(cfg, 2, 0)["output_cutoff_age"]


def test_cutoff_enabled_false_is_honoured():
    assert _cutoff({"output.cutoff.enabled": "false"}) is None  # Java: no cutoff
    assert list(_cutoff({"output.cutoff.enabled": "true"})) == [0.5, 1.0]
    assert list(_cutoff({})) == [0.5, 1.0]  # absent: unchanged behaviour


@pytest.mark.parametrize("value", ["age", "AGE", " age "])
def test_stage_structure_age_is_accepted(value):
    from osmose.engine.config import _check_accessibility_stage_structure

    _check_accessibility_stage_structure({"predation.accessibility.stage.structure": value})
    _check_accessibility_stage_structure({})


def test_stage_structure_size_or_invalid_raises():
    from osmose.engine.config import _check_accessibility_stage_structure

    with pytest.raises(ValueError, match="size is not supported"):
        _check_accessibility_stage_structure({"predation.accessibility.stage.structure": "size"})
    with pytest.raises(ValueError, match="must be 'age' or 'size'"):
        _check_accessibility_stage_structure({"predation.accessibility.stage.structure": "length"})


def test_honoured_keys_left_the_java_only_list():
    assert "output.cutoff.enabled" not in cv._ALLOWLIST_JAVA_ONLY
    assert "predation.accessibility.stage.structure" not in cv._ALLOWLIST_JAVA_ONLY
    assert "output.cutoff.enabled" in cv._SUPPLEMENTARY_ALLOWLIST
