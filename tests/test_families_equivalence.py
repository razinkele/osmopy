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


def _record(
    step: int, scale: float, *, tl: dict | None = None, full: bool = True, window: int = 1
) -> StepOutput:
    """A record whose every value is ``scale`` times a fixed pattern."""
    zeros = np.zeros(3)
    if not full:
        return StepOutput(step=step, biomass=zeros, abundance=zeros, mortality_by_cause=np.zeros((2, 8)))  # fmt: skip
    return StepOutput(
        step=step,
        window_steps=window,
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


def test_a_window_that_starts_inside_the_spin_up_is_dropped():
    """A record is a window ending at its step. One that ends after the spin-up but starts
    inside it would average spin-up steps into the post-spin-up numbers."""
    records = [
        _record(35, 1000.0, window=24),  # steps 12..35: straddles the end of year 1
        _record(47, 1.0, window=24),  # steps 24..47: the whole of year 2
        _record(71, 3.0, window=24),
    ]
    out = fam.reduce_records(records, CONFIG, spinup_years=1)
    assert out["ssb"] == {"cod": 20.0, "sprat": 40.0}  # scales 1 and 3 only
    with pytest.raises(ValueError, match="no record after the 1-year spin-up"):
        fam.reduce_records(records[:1], CONFIG, spinup_years=1)
    # With no spin-up the initial record (step -1) is still dropped, whatever its window says.
    initial = [_record(-1, 1000.0, window=0), _record(23, 1.0, window=24)]
    assert fam.reduce_records(initial, CONFIG, spinup_years=0)["ssb"] == {
        "cod": 10.0,
        "sprat": 20.0,
    }


def _arms(metrics=("ssb", "mean_tl"), species=("cod", "sprat"), n: int = 8) -> dict:
    """Three arms drawn from one distribution, a tenth of a percent of noise."""
    arms = {}
    for k, name in enumerate(fam.ARMS):
        rng = np.random.default_rng(100 + k)
        arms[name] = {
            m: {sp: 100.0 * 10 ** rng.normal(0.0, 0.0005, n) for sp in species} for m in metrics
        }
    return arms


def test_three_arms_of_one_distribution_pass_and_report_nothing_unseen():
    result = fam.judge(_arms(), splits=200)
    assert fam.passed(result)
    assert result["asymmetric"] == {} and result["one_sided"] == [] and result["dropped"] == []
    assert result["constant"] == [] and result["invalid"] == [] and result["floored"] == []
    assert set(result["margins"]) == {"ssb", "mean_tl"}
    assert len(result["treatment"]) == 4


def test_a_cell_that_is_one_number_everywhere_is_counted_and_listed():
    """A predator that eats no fish reads 0 in every replicate of every arm. The rule counts the
    cell as equivalent; it is listed, so the count is not read as more than it is. A cell that
    is constant in each arm but NOT the same number across arms is an ordinary cell."""
    arms = _arms()
    for name in fam.ARMS:
        arms[name]["mean_tl"]["sprat"] = np.full(8, 2.0)
    result = fam.judge(arms, splits=200)
    assert result["constant"] == ["mean_tl/sprat"]
    assert ("mean_tl", "sprat") in result["treatment"] and fam.passed(result)
    arms["csharp"]["mean_tl"]["sprat"] = np.full(8, 3.0)
    result = fam.judge(arms, splits=200)
    assert result["constant"] == []
    assert not result["treatment"][("mean_tl", "sprat")].equivalent
    assert not fam.passed(result)


def test_a_family_one_engine_omits_fails_the_run():
    """Left out silently, a run in which the C# engine never writes an output would pass on the
    cells that remain."""
    arms = _arms()
    del arms["csharp"]["ssb"]
    result = fam.judge(arms, splits=200)
    assert result["asymmetric"] == {"ssb": ["csharp"]}
    assert result["verdict"].passed, "the rule itself passes on what is left: that is the trap"
    assert not fam.passed(result)
    arms = _arms()
    del arms["python"]["mean_tl"], arms["python-control"]["mean_tl"]
    result = fam.judge(arms, splits=200)
    assert result["asymmetric"] == {"mean_tl": ["python", "python-control"]}
    assert not fam.passed(result)


@pytest.mark.parametrize("how", ["one replicate", "every replicate"])
def test_a_cell_only_one_engine_has_fails_the_run(how):
    arms = _arms()
    if how == "one replicate":
        arms["csharp"]["mean_tl"]["sprat"][3] = np.nan
    else:
        del arms["csharp"]["mean_tl"]["sprat"]
    result = fam.judge(arms, splits=200)
    assert result["one_sided"] == ["mean_tl/sprat"] and result["dropped"] == []
    assert ("mean_tl", "sprat") not in result["treatment"]
    assert result["verdict"].passed and not fam.passed(result)
    # The other way round: the C# engine has a cell neither Python arm has.
    arms = _arms()
    del arms["python"]["mean_tl"]["sprat"], arms["python-control"]["mean_tl"]["sprat"]
    result = fam.judge(arms, splits=200)
    assert result["one_sided"] == ["mean_tl/sprat"] and not fam.passed(result)


@pytest.mark.parametrize("level", [100.0, 0.3])
def test_an_output_one_engine_never_filled_fails_the_run(level):
    """An output the engine never wrote comes back as zeros, finite and complete. At ``level``
    0.3 the rule alone cannot see it: its log scale reads every value under 1 as 1, on both
    sides, and the cell would read as no difference at all."""
    arms = _arms()
    for name in fam.ARMS:
        arms[name]["ssb"]["cod"] = arms[name]["ssb"]["cod"] * (level / 100.0)
    arms["csharp"]["ssb"]["cod"] = np.zeros(8)
    result = fam.judge(arms, splits=200)
    assert result["one_sided"] == ["ssb/cod"] and not fam.passed(result)
    arms = _arms()  # the other way round: the Python engine reads zero, the C# engine does not
    for name in ("python", "python-control"):
        arms[name]["ssb"]["cod"] = np.zeros(8)
    arms["csharp"]["ssb"]["cod"] = arms["csharp"]["ssb"]["cod"] * (level / 100.0)
    result = fam.judge(arms, splits=200)
    assert result["one_sided"] == ["ssb/cod"] and not fam.passed(result)


def test_values_under_the_rules_floor_are_listed():
    arms = _arms()
    for name in fam.ARMS:
        arms[name]["ssb"]["cod"] = arms[name]["ssb"]["cod"] * 0.003
    result = fam.judge(arms, splits=200)
    assert result["floored"] == ["ssb/cod"] and result["one_sided"] == []
    assert fam.passed(result)  # reported, not failed: the rule's floor is the owner's
    assert fam.judge(_arms(), splits=200)["floored"] == []


def test_a_cell_the_python_engine_itself_lacks_is_reported_and_does_not_fail():
    """A species absent from a mean in some replicate of a PYTHON arm cannot be tested, and that
    is not an engine difference."""
    arms = _arms()
    arms["python-control"]["mean_tl"]["sprat"][0] = np.nan
    result = fam.judge(arms, splits=200)
    assert result["dropped"] == ["mean_tl/sprat"] and result["one_sided"] == []
    assert fam.passed(result)
    arms = _arms()  # absent from all three: nothing to compare, nothing to fail
    for name in fam.ARMS:
        arms[name]["mean_tl"]["sprat"][1] = np.nan
    result = fam.judge(arms, splits=200)
    assert result["dropped"] == ["mean_tl/sprat"] and result["one_sided"] == []
    assert fam.passed(result)


def test_a_gap_in_an_output_that_always_exists_fails_the_run():
    """SSB, the yields, the life stages and the diet exist for every species in every record.
    A missing or non-finite value there, in ANY arm, is bad evidence, not a cell to leave out:
    left out, a Python arm with one NaN would hide a C# arm of zeros in the same cell."""
    arms = _arms()
    arms["python-control"]["ssb"]["cod"][0] = np.nan
    result = fam.judge(arms, splits=200)
    assert result["invalid"] == ["ssb/cod"] and result["dropped"] == []
    assert result["verdict"].passed and not fam.passed(result)
    arms["csharp"]["ssb"]["cod"] = np.zeros(8)  # the case the gap would have hidden
    result = fam.judge(arms, splits=200)
    assert result["invalid"] == ["ssb/cod"] and not fam.passed(result)
    arms = _arms()
    arms["csharp"]["ssb"]["sprat"][2] = np.inf
    assert fam.judge(arms, splits=200)["invalid"] == ["ssb/sprat"]
    arms = _arms()
    del arms["python"]["ssb"]["sprat"]
    result = fam.judge(arms, splits=200)
    assert result["invalid"] == ["ssb/sprat"] and not fam.passed(result)
    # The same gap in a weighted mean is only reported (the test above this one).
    assert set(fam.EXTENSIVE) & {"mean_size", "mean_tl"} == set()


def test_a_family_with_no_testable_cell_does_not_stop_the_others():
    """With every cell of the first family dropped, the rule used to be handed an empty family
    and stopped (it sizes the arms from the first cell). Two weighted means, the only outputs a
    species may be absent from: ``mean_size`` comes first in the margins."""
    arms = _arms(metrics=("mean_size", "mean_tl"))
    for sp in ("cod", "sprat"):
        arms["python-control"]["mean_size"][sp][0] = np.nan
    result = fam.judge(arms, splits=200)
    assert sorted(result["dropped"]) == ["mean_size/cod", "mean_size/sprat"]
    assert set(result["margins"]) == {"mean_tl"}
    assert fam.passed(result)
    for sp in ("cod", "sprat"):
        arms["python-control"]["mean_tl"][sp][0] = np.nan
    result = fam.judge(arms, splits=200)
    assert result["verdict"] is None and result["margins"] == {}
    assert not fam.passed(result), "nothing testable is not a pass"
    fam.report(result)  # prints the dropped cells and a FAIL, and does not raise
