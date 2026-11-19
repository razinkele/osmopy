"""The diet / predator-pressure PREY axis: focal species, background species, resources.

That is Java's ``DietOutput``. Two earlier layouts each lost something, and both are guarded
here: #146 (the first paragraph below) and #322 (an eaten background school booked in the
column of the resource with the same index, because there were no background columns).

The original note, for #146:

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


def test_prey_axis_is_focal_then_background_then_resources():
    """A background school is a school, and a predator large enough eats it: it has a column."""
    cfg = EngineConfig.from_dict(
        dict(OsmoseConfigReader().read(str(BALTIC_DIR / "baltic_all-parameters.csv")))
    )
    # resource_names is published by simulate(); before a run it is empty, so the axis is
    # just the species. The order holds either way.
    names = diet_prey_names(cfg)
    nf, nb = cfg.n_species, cfg.n_background
    assert nb == 2
    assert names[:nf] == cfg.species_names
    assert (
        names[nf : nf + nb]
        == list(cfg.all_species_names[nf : nf + nb])
        == ["GreySeal", "Cormorant"]
    )
    assert names[nf + nb :] == cfg.resource_names
    cfg.resource_names = ["Diatoms", "Benthos"]
    assert diet_prey_names(cfg) == [
        *cfg.species_names,
        "GreySeal",
        "Cormorant",
        "Diatoms",
        "Benthos",
    ]


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
    """The original symptom: smelt reported 94.3% GreySeal. The background species have prey
    columns now (#322), and on Baltic as shipped they must stay empty: nothing there is large
    enough to eat a seal or a cormorant. A resource's biomass under their label would show."""
    cfg = EngineConfig.from_dict(
        dict(OsmoseConfigReader().read(str(BALTIC_DIR / "baltic_all-parameters.csv")))
    )
    background = set(cfg.all_species_names) - set(cfg.species_names)
    assert background, "fixture assumption: the Baltic config has background species"
    for predator in cfg.species_names:
        vals, _ = _prey_shares(baltic_diet, predator)
        for bg in background:
            assert bg in vals, f"{predator} has no {bg!r} prey column"
            assert vals[bg] == 0.0, f"{predator} is recorded eating {bg!r}"


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
    # The same prey axis as the diet matrix: the background species have rows too (#322), and
    # they are empty on Baltic as shipped.
    for bg in set(cfg.all_species_names) - set(cfg.species_names):
        assert bg in prey_rows, f"background species {bg!r} has no predatorPressure prey row"
        eaten = pp[pp["Prey"] == bg].drop(columns=["Time"]).select_dtypes("number")
        assert eaten.shape[1] >= cfg.n_species, "the predator columns are numeric"
        assert float(eaten.to_numpy(dtype=float).sum()) == 0.0, f"something eats {bg!r}"


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


_FORCED_STEP = 40  # late enough for fish that eat 8 cm prey, early enough to be quick


class _Captured(Exception):
    """Ends the run once the step under test has been captured."""


def _step_with_eaten_background_schools(tmp_path, *, numba: bool) -> dict:
    """One mortality step of the Baltic-derived tutorial config in which fish eat background
    schools: every third one is shrunk to 8 cm as the step starts. ``numba`` selects the
    batched kernels (production) or the per-school Python path (what runs without Numba).
    Returns the kernel's per-school diet rows and what was eaten of each school."""
    from unittest import mock

    import numpy as np

    import osmose.engine.processes.mortality as mort
    import osmose.engine.simulate as sim
    from osmose.engine.state import MortalityCause
    from tests._tutorial_config import build_config

    raw = dict(build_config(tmp_path, n_year=_FORCED_STEP // 24 + 1))
    raw["output.diet.composition.enabled"] = "true"
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    got: dict = {}
    real = sim._mortality

    def spy(state, resources, config_, rng_, grid_, step=0, species_rngs=None, ctx=None):
        if step != _FORCED_STEP:
            return real(
                state,
                resources,
                config_,
                rng_,
                grid_,
                step=step,
                species_rngs=species_rngs,
                ctx=ctx,
            )
        background = np.flatnonzero(state.is_background)
        state.length[background[::3]] = 8.0
        species, weight = state.species_id.copy(), state.weight.copy()
        rsc_before = resources.biomass.copy()
        with mock.patch.object(mort, "_HAS_NUMBA", numba):
            after = real(
                state,
                resources,
                config_,
                rng_,
                grid_,
                step=step,
                species_rngs=species_rngs,
                ctx=ctx,
            )
        got.update(
            config=config_,
            rows=ctx.diet_matrix[: ctx._diet_active_rows].copy(),
            species=species,
            eaten=after.n_dead[:, MortalityCause.PREDATION] * weight,
            preyed=after.preyed_biomass.copy(),
            rsc_eaten=(rsc_before - resources.biomass).sum(axis=1),
        )
        raise _Captured

    with mock.patch.object(sim, "_mortality", spy), pytest.raises(_Captured):
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    return got


@pytest.mark.parametrize("numba", [True, False], ids=["batched-kernels", "python-path"])
def test_an_eaten_background_school_is_booked_in_its_own_column(tmp_path, numba):
    """Issue #322. A background school is a school: a predator large enough eats it. The kernel
    books a school prey at its species id, which for a background school is ``n_species + k``.
    With no background columns that was the column of resource ``k``, so the diet showed the
    predator eating a plankton group it had not eaten and showed the background prey nowhere.

    Nothing eats a seal or a cormorant on Baltic as shipped, so the case is forced (see the
    helper). Both mortality paths write the matrix, each with its own code: the batched Numba
    kernels, and the per-school Python loop that runs when Numba is absent."""
    import numpy as np

    import osmose.engine.processes.mortality as mort

    if numba and not mort._HAS_NUMBA:
        pytest.skip("the batched kernels need Numba; the Python path is the other case")
    got = _step_with_eaten_background_schools(tmp_path, numba=numba)
    config, rows = got["config"], got["rows"]
    nf, nb, n_rsc = config.n_species, config.n_background, len(config.resource_names)
    assert nb == 2 and n_rsc > nb
    assert rows.shape[1] == nf + nb + n_rsc == len(diet_prey_names(config))

    by_prey = np.bincount(got["species"], weights=got["eaten"], minlength=nf + nb)
    assert by_prey[nf:].sum() > 0, "no background school was eaten: the case was not forced"
    # Every species column, the background ones included, holds what was eaten of that species
    # and nothing else.
    assert np.allclose(rows[:, : nf + nb].sum(axis=0), by_prey, rtol=1e-9, atol=0.0)
    # The resources are in the last columns, each in its own: what a column holds is what that
    # resource lost.
    resources = rows[:, nf + nb :].sum(axis=0)
    assert (resources > 0).sum() >= 2
    assert np.allclose(resources, got["rsc_eaten"], rtol=1e-9, atol=1e-12)
    # A predator's row sums to what it ate.
    n = len(rows)
    assert np.allclose(rows.sum(axis=1), got["preyed"][:n], rtol=1e-9, atol=1e-12)
