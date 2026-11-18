"""``scripts/csharp/families_equivalence.py``: how one run's records become the per-species
numbers the ensemble compares (plan task 2.8). The reduction is the same for both engines, so a
mistake in it would hide or invent a difference for both alike; it is pinned here on records
small enough to read. No engine runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from osmose.engine.simulate import StepOutput

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "families_equivalence", ROOT / "scripts" / "csharp" / "families_equivalence.py"
)
assert spec is not None and spec.loader is not None
fam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fam)

# Two focal species and one background predator; two resources; 24 steps a year.
CONFIG = SimpleNamespace(
    n_species=2, n_background=1, n_dt_per_year=24, all_species_names=["cod", "sprat", "seal"]
)


def _record(step: int, scale: float, *, tl: dict | None = None, full: bool = True) -> StepOutput:
    """A record whose every value is ``scale`` times a fixed pattern."""
    zeros = np.zeros(3)
    if not full:
        return StepOutput(step=step, biomass=zeros, abundance=zeros, mortality_by_cause=np.zeros((2, 8)))  # fmt: skip
    return StepOutput(
        step=step,
        biomass=zeros,
        abundance=zeros,
        mortality_by_cause=np.zeros((2, 8)),
        ssb=scale * np.array([10.0, 20.0]),
        yield_n=scale * np.array([1.0, 2.0]),
        abundance_by_stage=scale * np.array([[100.0, 10.0, 1.0], [200.0, 20.0, 2.0]]),
        abundance_by_age={0: scale * np.array([5.0, 3.0, 2.0]), 1: scale * np.array([7.0, 1.0])},
        biomass_by_age={0: scale * np.array([0.5, 30.0, 20.0]), 1: scale * np.array([0.25, 1.5])},
        # [predator: cod, sprat, seal] x [prey: cod, sprat, resource 0, resource 1]
        diet_by_species=scale
        * np.array([[1.0, 2.0, 4.0, 8.0], [0.0, 0.0, 16.0, 32.0], [64.0, 128.0, 0.0, 0.0]]),
        mean_size={0: scale * 30.0, 1: scale * 10.0},
        mean_tl={0: 4.0, 1: 3.0} if tl is None else tl,
    )


def test_the_spin_up_and_the_initial_record_are_dropped_and_the_rest_averaged():
    records = [
        _record(-1, 1000.0),  # the initial record
        _record(23, 1000.0),  # year 1: spin-up
        _record(47, 1.0),  # year 2
        _record(71, 3.0),  # year 3
    ]
    out = fam.reduce_records(records, CONFIG, spinup_years=1)
    assert out["ssb"] == {"cod": 20.0, "sprat": 40.0}  # the mean of scales 1 and 3 is 2
    assert out["yield_n"] == {"cod": 2.0, "sprat": 4.0}
    assert out["eggs"] == {"cod": 200.0, "sprat": 400.0}
    assert out["juveniles"] == {"cod": 20.0, "sprat": 40.0}
    assert out["adults"] == {"cod": 2.0, "sprat": 4.0}
    assert out["age0"] == {"cod": 10.0, "sprat": 14.0}  # the first age class
    assert out["age1plus"] == {"cod": 10.0, "sprat": 2.0}  # every older class, added
    assert out["biomass_age0"] == {"cod": 1.0, "sprat": 0.5}
    assert out["biomass_age1plus"] == {"cod": 100.0, "sprat": 3.0}
    assert out["mean_size"] == {"cod": 60.0, "sprat": 20.0}
    with pytest.raises(ValueError, match="no record after the 5-year spin-up"):
        fam.reduce_records(records, CONFIG, spinup_years=5)


def test_the_diet_is_split_into_fish_and_resources_and_background_predators_have_rows():
    out = fam.reduce_records([_record(47, 1.0)], CONFIG, spinup_years=1)
    assert out["eaten_fish"] == {"cod": 3.0, "sprat": 0.0, "seal": 192.0}
    assert out["eaten_resources"] == {"cod": 12.0, "sprat": 48.0, "seal": 0.0}
    # Only the diet has background rows: every other metric is per focal species.
    assert all("seal" not in out[m] for m in out if not m.startswith("eaten_"))


def test_a_species_absent_from_a_mean_is_averaged_over_the_records_that_have_it():
    records = [
        _record(47, 1.0, tl={0: 4.0, 1: 3.0}),
        _record(71, 1.0, tl={0: 5.0}),  # sprat has no fed school in this record
        _record(95, 1.0, tl={0: 6.0}),
    ]
    out = fam.reduce_records(records, CONFIG, spinup_years=1)
    assert out["mean_tl"] == {"cod": 5.0, "sprat": 3.0}  # sprat: its one record, not 3 / 3
    never = fam.reduce_records([_record(47, 1.0, tl={0: 4.0})], CONFIG, spinup_years=1)
    assert never["mean_tl"] == {"cod": 4.0}  # absent everywhere: left out, not 0


def test_a_family_the_run_did_not_produce_is_left_out():
    out = fam.reduce_records([_record(47, 1.0, full=False)], CONFIG, spinup_years=1)
    assert out == {}


def test_the_margins_cover_every_metric_once():
    assert set(fam.MARGINS) == set(fam.EXTENSIVE) | set(fam.INTENSIVE)
    assert not set(fam.EXTENSIVE) & set(fam.INTENSIVE)
    assert all(fam.MARGINS[m] == pytest.approx(np.log10(1.25)) for m in fam.EXTENSIVE)
    assert all(fam.MARGINS[m] == pytest.approx(np.log10(1.05)) for m in fam.INTENSIVE)
    assert fam.SEEDS == {"python": 1000, "python-control": 5000, "csharp": 9000}  # spec 10.2
