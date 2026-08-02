"""The diet / predator-pressure PREY axis is schools-then-resources (#146).

Regression guard for a silent mislabelling: the axis used to be
``config.all_species_names`` (focal + BACKGROUND species), while the predation kernel
writes resources at column ``n_species + r_idx``. On the Baltic config that put Diatoms
and Dinoflagellates in the GreySeal/Cormorant slots and pushed the remaining four
resource groups past the end of the array, where a bounds check dropped them without
error. The visible symptom was smelt — a planktivore — reporting 94.3% of its diet as
GreySeal, which it cannot eat, and 0% resources, which is nearly all of what it eats.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.engine.output import diet_prey_names

BALTIC_DIR = Path(__file__).resolve().parents[1] / "data" / "baltic"

_LTL = {
    "Diatoms",
    "Dinoflagellates",
    "Microzooplankton",
    "Mesozooplankton",
    "Macrozooplankton",
    "Benthos",
}


def test_prey_axis_is_schools_then_resources_never_background():
    """Background species are not prey — the kernel types prey as school or resource only."""
    cfg = EngineConfig.from_dict(
        dict(OsmoseConfigReader().read(str(BALTIC_DIR / "baltic_all-parameters.csv")))
    )
    # resource_names is published by simulate(); before a run it is empty, so the axis is
    # just the schools. The invariant that matters holds either way: never background.
    names = diet_prey_names(cfg)
    assert names[: cfg.n_species] == cfg.species_names
    assert names[cfg.n_species :] == cfg.resource_names
    for bg in set(cfg.all_species_names) - set(cfg.species_names):
        assert bg not in names, f"background species {bg!r} must not appear on the prey axis"


@pytest.fixture(scope="module")
def baltic_diet():
    work = Path(tempfile.mkdtemp())
    target = work / "baltic"
    shutil.copytree(BALTIC_DIR, target)
    cfg = OsmoseConfigReader().read(str(target / "baltic_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "2"
    return PythonEngine().run_in_memory(config=cfg, seed=42).diet_matrix()


def _prey_shares(diet, predator):
    prefix = f"{predator}_"
    vals = {c[len(prefix) :]: float(diet[c].mean()) for c in diet.columns if c.startswith(prefix)}
    total = sum(vals.values())
    return vals, total


def test_resource_prey_reach_the_diet_matrix(baltic_diet):
    """Every resource group must be addressable, and the plankton ones actually fed.

    The old axis was two columns too short for the resource block, so Microzoo/Mesozoo/
    Macrozoo/Benthos were silently discarded. Asserting the columns merely EXIST is not
    enough — they existed under wrong names before. Assert they carry biomass.
    """
    vals, total = _prey_shares(baltic_diet, "smelt")
    assert total > 0, "smelt ate nothing; the fixture is not exercising predation"
    assert _LTL <= set(vals), f"missing resource prey columns: {_LTL - set(vals)}"

    resource_share = sum(v for k, v in vals.items() if k in _LTL) / total
    assert resource_share > 0.5, (
        f"smelt is a planktivore but only {100 * resource_share:.1f}% of its recorded diet is "
        f"resource-derived. Shares: { {k: round(100 * v / total, 1) for k, v in vals.items() if v > 0} }"
    )


def test_no_predator_eats_a_background_species(baltic_diet):
    """The original symptom: smelt reported 94.3% GreySeal."""
    cfg = EngineConfig.from_dict(
        dict(OsmoseConfigReader().read(str(BALTIC_DIR / "baltic_all-parameters.csv")))
    )
    background = set(cfg.all_species_names) - set(cfg.species_names)
    assert background, "fixture assumption: the Baltic config has background species"
    for predator in cfg.species_names:
        vals, _ = _prey_shares(baltic_diet, predator)
        for bg in background:
            assert bg not in vals, (
                f"{predator} has a {bg!r} prey column — background species are not prey"
            )


def test_predator_pressure_shares_the_same_prey_axis(baltic_diet):
    """predatorPressure must use diet_prey_names too — it did not, and dietMatrix cover missed it.

    e121c6d fixed three of the four prey-axis sites and left
    ``_build_predator_pressure_dataframe`` on ``all_species_names``. The full suite stayed green
    because nothing asserted on absolute consumption, and the defect only surfaced as a
    biologically impossible Q/B: herring consuming 0.12x its body weight per year while feeding
    61% on Mesozooplankton.

    Asserting the row COUNT is the cheap structural guard; the Q/B sanity check below is the one
    that would actually have caught it.
    """
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp())
    target = work / "baltic"
    shutil.copytree(BALTIC_DIR, target)
    cfg_dict = OsmoseConfigReader().read(str(target / "baltic_all-parameters.csv"))
    cfg_dict["simulation.time.nyear"] = "2"
    res = PythonEngine().run_in_memory(config=cfg_dict, seed=42)

    pp = res._csv_cache["predatorPressure"]
    first_t = pp["Time"].iloc[0]
    prey_rows = set(pp[pp["Time"] == first_t]["Prey"])
    assert _LTL <= prey_rows, f"predatorPressure is missing resource prey rows: {_LTL - prey_rows}"

    cfg = EngineConfig.from_dict(
        dict(OsmoseConfigReader().read(str(BALTIC_DIR / "baltic_all-parameters.csv")))
    )
    for bg in set(cfg.all_species_names) - set(cfg.species_names):
        assert bg not in prey_rows, (
            f"background species {bg!r} must not be a predatorPressure prey row"
        )


def test_planktivore_consumption_is_biologically_possible(baltic_diet):
    """A fish must eat on the order of its own body weight per year, at minimum.

    This is the assertion that catches a dropped prey block regardless of WHICH site drops it:
    undercounting consumption drives Q/B toward zero. Herring read 0.12 under the bug and 4.59
    once every resource row was restored.
    """
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp())
    target = work / "baltic"
    shutil.copytree(BALTIC_DIR, target)
    cfg_dict = OsmoseConfigReader().read(str(target / "baltic_all-parameters.csv"))
    cfg_dict["simulation.time.nyear"] = "3"
    res = PythonEngine().run_in_memory(config=cfg_dict, seed=42)

    pp = res._csv_cache["predatorPressure"]
    bio = res.biomass()
    n_dt = 24
    late_t = pp["Time"].max()
    late_pp = pp[pp["Time"] == late_t]
    late_b = bio[bio["Time"] == late_t]

    for planktivore in ("herring", "sprat"):
        q = float(late_pp[planktivore].sum()) * n_dt
        b = float(late_b[planktivore].iloc[0])
        assert b > 0, f"{planktivore} has no biomass; fixture is not exercising the food web"
        assert q / b > 1.0, (
            f"{planktivore} Q/B = {q / b:.2f} per year — below its own body weight, which no fish "
            f"can sustain. A prey block is being dropped from predatorPressure (#146)."
        )
