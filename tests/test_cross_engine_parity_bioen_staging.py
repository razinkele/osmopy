"""Task 9 (C3 Gate B) unit tests for the Java-4.3.3 bioen key staging and the non-degeneracy
precondition helper added to scripts/cross_engine_parity_440.py.

Loaded by path (the script lives under scripts/, not a package) rather than imported normally.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xeng", ROOT / "scripts" / "cross_engine_parity_440.py")
xeng = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(xeng)


def test_inject_java_bioen_keys_appends_bioen_imax_for_every_predator(tmp_path):
    master = tmp_path / "osm_all-parameters.csv"
    master.write_text("predation.ingestion.rate.max.sp0 ; 3.5\nspecies.type.sp15 ; background\n")
    raw = {
        "module.bioenergetics.enabled": "true",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp15": "2.5",
        "species.type.sp15": "background",
        "predation.larval.ingestion.rate.increase.ratio.sp0": "1.0",
        "predation.c.bioen.sp0": "0.0",
        "simulation.nspecies": "1",
    }
    n = xeng.inject_java_bioen_keys(master, raw)
    text = master.read_text()
    assert "predation.ingestion.rate.max.bioen.sp0 ; 3.5" in text
    assert "predation.ingestion.rate.max.bioen.sp15 ; 2.5" in text
    assert (
        "predation.coef.ingestion.rate.max.larvae.bioen.sp15 ; 1.0" in text
        and "predation.c.bioen.sp15 ; 0.0" in text
    )
    assert "predation.ingestion.rate.max.sp0 ; 3.5" in text  # legacy standard key kept (Java reads both)
    assert n >= 4


def test_inject_is_noop_without_bioen(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("a ; 1\n")
    assert xeng.inject_java_bioen_keys(master, {"module.bioenergetics.enabled": "false"}) == 0
    assert master.read_text() == "a ; 1\n"


def test_inject_raises_on_missing_imax_for_a_predator(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("x ; 1\n")
    raw = {"module.bioenergetics.enabled": "true", "simulation.nspecies": "2"}
    import pytest

    with pytest.raises(KeyError):
        xeng.inject_java_bioen_keys(master, raw)


def test_inject_resource_nsteps_year_adds_global_fallback_for_file_forced_resource(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("species.file.sp8 ; some.nc\n")
    raw = {"species.file.sp8": "some.nc", "simulation.time.ndtperyear": "24"}
    n = xeng.inject_java_resource_nsteps_year(master, raw)
    assert n == 1
    assert "species.biomass.nsteps.year ; 24" in master.read_text()


def test_inject_resource_nsteps_year_is_noop_when_already_present(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("x ; 1\n")
    raw = {
        "species.file.sp8": "some.nc",
        "simulation.time.ndtperyear": "24",
        "species.biomass.nsteps.year": "24",
    }
    assert xeng.inject_java_resource_nsteps_year(master, raw) == 0
    assert master.read_text() == "x ; 1\n"


def test_inject_resource_nsteps_year_is_noop_without_file_forced_resources(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("x ; 1\n")
    raw = {"simulation.time.ndtperyear": "24"}
    assert xeng.inject_java_resource_nsteps_year(master, raw) == 0
    assert master.read_text() == "x ; 1\n"


def test_nondegenerate_flags_species_collapsed_in_too_many_reps():
    import numpy as np

    ens = {"biomass": {"A": np.array([500.0] * 16), "B": np.array([500.0] * 13 + [0.5] * 3)}}
    nd = xeng.nondegenerate(ens, "biomass", n=16, floor=1.0, frac=0.9)
    assert nd == {"A": True, "B": False}


def test_nondegenerate_flags_missing_reps_as_nan_collapsed():
    import numpy as np

    # A species array shorter than n (some reps never reported it) still fails on size==n,
    # and NaN entries never count as "ok" regardless of magnitude.
    ens = {"biomass": {"A": np.array([500.0, np.nan, 500.0])}}
    nd = xeng.nondegenerate(ens, "biomass", n=4, floor=1.0, frac=0.9)
    assert nd == {"A": False}
