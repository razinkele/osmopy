"""Plan task 3.1: depletable resources (feature ``depletable_ltl``) on the C# engine.

By default a resource is reset to its carrying capacity K every step, so grazing has no
memory. With ``ltl.depletable.enabled`` the biomass left after predation is carried over and
regrown toward K (``osmose.engine.resources.logistic_regrow``):

    B = max(B_carried, floor * K);  B_new = min(K, B + rate * B * (1 - B / K));  K <= 0 -> 0

Three things must hold for the C# engine to do the same, and each has its own test:

1. the update regrows what it is handed as the Python engine does, bit for bit (this file,
   on the biomass a real run carried into the step and on made-up edge cases);
2. the kernel's eating is written to the buffer the next update reads (the whole-step test
   in tests/test_csharp_pipeline.py compares the biomass AFTER mortality);
3. nothing between two steps resets that buffer (a run without fish, below: its resources
   are then a pure sequence of updates, the same on both engines).

The runs happen in probe subprocesses and need the built engine.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from osmose.engine.resources import ResourceState
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import csharp_engine_block_reason, required_features
from tests._csharp_resources import oxygen_coupled, simulation_resources
from tests.test_csharp_prep import capture_state, run_probe, write_state

STEPS = {"eec_full": 30, "baltic": 129}
RATES = (0.3, 1.0, 2.5, 5.0)  # per step; cycled over the resources


def depletable(raw: dict) -> dict:
    """The keys that make every resource of ``raw`` depletable, each with its own rate."""
    resources = sorted(
        int(k.rsplit("sp", 1)[1])
        for k, v in raw.items()
        if k.startswith("species.type.sp") and v.strip().lower() == "resource"
    )
    keys = {"ltl.depletable.enabled": "true", "ltl.depletable.floor": "0.05"}
    keys |= {
        f"species.regrowth.rate.sp{fi}": str(RATES[i % len(RATES)])
        for i, fi in enumerate(resources)
    }
    return keys


@pytest.fixture(scope="module")
def carried(tmp_path_factory) -> dict:
    """Per demo: a real run with depletable resources, stopped at one step. ``carried`` is the
    biomass the run brought INTO that step's update (what the last step's predation left),
    ``live`` what the update made of it."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine_cs import csharp_engine_available

    ok, why = csharp_engine_available()
    if not ok:
        import os

        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    out = {}
    for demo, step in STEPS.items():
        d = tmp_path_factory.mktemp(demo)
        plain = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, d / "keys")["config_file"])))
        got: dict = {}
        real_update = ResourceState.update

        def spy(self, at, got=got, step=step, real_update=real_update):
            if at == step:
                got["carried"] = self.biomass.copy()
            return real_update(self, at)

        with mock.patch.object(ResourceState, "update", spy):
            config, grid, raw, st, live = capture_state(demo, step, d, overrides=depletable(plain))
        build_bundle(config, raw, grid, seed=11).save(d / "model.bin")
        out[demo] = {
            "config": config, "grid": grid, "raw": raw, "state": st, "live": live,
            "carried": got["carried"], "step": step, "dir": d,
        }  # fmt: skip
    return out


def _update(case: dict, biomass: np.ndarray, step: int, name: str) -> np.ndarray:
    """The C# update of ``biomass`` (``[resource, cell]``) at ``step``."""
    d = case["dir"]
    plain, state = d / f"{name}.plain.bin", d / f"{name}.state.bin"
    write_state(case["state"], plain)
    # A loaded bundle is a memory map of its file: it must be saved to ANOTHER file, or the
    # arrays are read from the file being overwritten.
    b = Bundle.load(plain)
    b.set_array("state.resources", np.ascontiguousarray(biomass, dtype=np.float64))
    b.save(state)
    res = run_probe(d / "model.bin", state, step, d / f"{name}.prep.bin")
    return np.array(res.get("prep.resources"))[: biomass.shape[0]]


def _python_update(case: dict, biomass: np.ndarray, step: int) -> np.ndarray:
    rs = simulation_resources(case["config"], case["grid"])
    assert rs.depletable
    rs.biomass[:] = biomass
    rs.update(step)
    out = rs.biomass.copy()
    rs.close()
    return out


@pytest.mark.parametrize("demo", sorted(STEPS))
def test_the_config_needs_the_feature_and_the_bundle_carries_it(carried, demo):
    case = carried[demo]
    assert "depletable_ltl" in required_features(case["config"], case["raw"])
    assert csharp_engine_block_reason(case["raw"]) is None
    b = Bundle.load(case["dir"] / "model.bin")
    n = len(case["live"])
    assert b.get("resources.depletable.floor") == 0.05
    assert b.get("resources.depletable.regrowth_rate").tolist() == [
        RATES[i % len(RATES)] for i in range(n)
    ]


@pytest.mark.parametrize("demo", sorted(STEPS))
def test_the_update_regrows_what_a_real_run_carried_into_the_step(carried, demo):
    """The biomass a real run brought into the step, and what its update made of it. On
    Baltic the benthos is depletable AND scaled by bottom oxygen: K is scaled first, then the
    regrowth is toward that K."""
    case = carried[demo]
    got = _update(case, case["carried"], case["step"], "real")
    assert got.tobytes() == case["live"].tobytes()
    # The case is not empty: predation left less than the update gives back, somewhere, and
    # what the update gives is not the plain reset to K either.
    assert (case["carried"] < case["live"]).any()
    # On Baltic, negative biomass is in the comparison too: the run itself holds some at this
    # step (issue #337), so the two engines are compared on it, sign included. The EEC step
    # happens to hold none; the made-up case below has it for both.
    if demo == "baltic":
        assert (case["carried"] < 0).any() or (case["live"] < 0).any()
    rs = simulation_resources(case["config"], case["grid"])
    assert (oxygen_coupled(case["config"], rs) is not None) == (demo == "baltic")
    rs.depletable = False
    rs.update(case["step"])
    assert rs.biomass.tobytes() != case["live"].tobytes()
    rs.close()


@pytest.mark.parametrize("demo", sorted(STEPS))
def test_the_update_at_the_edges(carried, demo):
    """Made-up biomass, cell by cell a different fraction of K: nothing (regrowth starts from
    the floor), under the floor, in between, exactly K, and above K. Cells with K = 0 stay
    empty whatever they are handed.

    Above K the logistic step goes DOWN, and with a rate over about 1.4 it lands below zero
    from 1.7 K: the function caps its result at K and does not bound it below. That is the
    Python engine's arithmetic (issue #337, which shows it happening on the shipped baltic_a2
    demo every year), and the C# engine must reproduce it, sign included, until it changes
    for both."""
    case = carried[demo]
    rs = simulation_resources(case["config"], case["grid"])
    rs.depletable = False
    rs.update(case["step"])
    k = rs.biomass.copy()
    rs.close()
    fractions = np.array([0.0, 0.01, 0.05, 0.2, 0.9, 1.0, 1.7])
    handed = k * fractions[np.arange(k.shape[1]) % len(fractions)]
    handed[k <= 0] = 3.0  # biomass where there is no capacity: must come out 0
    want = _python_update(case, handed, case["step"])
    got = _update(case, handed, case["step"], "edges")
    assert got.tobytes() == want.tobytes()
    assert (k <= 0).any() and (want[k <= 0] == 0).all()
    assert (want[k > 0] <= k[k > 0]).all()
    assert (want < 0).any() and (want > 0).any()  # both signs are in the case (issue #337)


def test_the_first_step_starts_from_the_floor_not_from_k(carried):
    """A run starts with empty resources, so its first update regrows from ``floor * K``, not
    from K. The C# buffer starts empty too, and the first update must be the same number."""
    case = carried["eec_full"]
    empty = np.zeros_like(case["live"])
    want = _python_update(case, empty, 0)
    got = _update(case, empty, 0, "first")
    assert got.tobytes() == want.tobytes()
    rs = simulation_resources(case["config"], case["grid"])
    rs.depletable = False
    rs.update(0)
    k = rs.biomass.copy()
    rs.close()
    some = k > 0
    assert (want[some] < k[some]).any()  # below capacity at the start


# ---------------------------------------------------------------- 2. one whole step


def test_a_whole_step_eats_from_the_buffer_the_next_update_reads(carried, tmp_path):
    """One step of a real run, from the biomass it carried in to the state it leaves: the C#
    step is handed that biomass, regrows it, lets the schools eat, and must leave the same
    schools AND the same resources as the Python engine. A kernel that ate from a copy would
    leave its buffer as the update made it; here predation demonstrably took something."""
    from tests.test_csharp_mortality_step import _write_draws
    from tests.test_csharp_pipeline import (
        FAMILIES,
        STEP_CASES,
        TIER_L,
        _capture_step,
        _probe,
        _worst,
        _write_states,
    )
    from tests.test_csharp_pipeline import _raw as pipeline_raw

    step = STEP_CASES["eec_full"]
    plain = pipeline_raw("eec_full", tmp_path / "keys", 1)
    raw = pipeline_raw("eec_full", tmp_path, step // 24 + 1, depletable(plain))
    seen: dict = {}
    real_update = ResourceState.update

    def spy(self, at):
        if at == step:
            seen["carried"] = self.biomass.copy()
        return real_update(self, at)

    with mock.patch.object(ResourceState, "update", spy):
        got = _capture_step(raw, step)
    assert "depletable_ltl" in required_features(got["config"], got["raw"])
    build_bundle(got["config"], got["raw"], got["grid"], seed=11).save(tmp_path / "model.bin")
    _write_states(
        [got["entry"]], tmp_path / "entry.bin", extra={"state.resources": seen["carried"]}
    )
    _write_draws(got["record"], tmp_path / "draws.bin")
    res = _probe(
        "Step", tmp_path / "model.bin", tmp_path / "entry.bin", tmp_path / "draws.bin", step,
        tmp_path / "out.bin",
    )  # fmt: skip
    final, out = got["final"], got["output"]
    assert len(res.get("s.species_id")) == len(final)
    for name in ("species_id", "age_dt", "cell_x", "cell_y", "is_egg", "is_out"):
        want = np.asarray(getattr(final, name)).astype(res.get(f"s.{name}").dtype)
        assert np.array_equal(res.get(f"s.{name}"), want), name
    worst = {}
    for name in ("abundance", "length", "weight", "trophic_level", "starvation_rate", "n_dead"):
        worst[name] = _worst(res.get(f"s.{name}"), getattr(final, name))
    for cs_name, py_name in FAMILIES:
        worst[f"out.{cs_name}"] = _worst(res.get(f"out.{cs_name}")[0], getattr(out, py_name))
    n_rsc = len(seen["carried"])
    left = np.array(res.get("s.resources"))[:n_rsc]
    worst["resources"] = _worst(left, got["rsc_after"])
    print(worst)
    assert all(v <= TIER_L for v in worst.values()), worst
    # Predation took from the resources in this step, and the update before it was a
    # regrowth of what was carried in, not a reset: both are part of what was compared.
    eaten = got["rsc_entry"] - got["rsc_after"]
    assert (eaten > 0).sum() > 100 and (eaten >= 0).all()
    assert (seen["carried"] < got["rsc_entry"]).any()


# ---------------------------------------------------------------- 3. between the steps


def test_a_run_without_fish_is_a_pure_sequence_of_updates(carried, tmp_path):
    """With no fish nothing eats, so the resources of a run are the updates applied one after
    another to the same buffer: 48 of them here. The C# run must end where the Python
    function ends. A buffer cleared, or reset, anywhere between two steps would end
    elsewhere: one update from an empty buffer is the control."""
    from osmose.engine import PythonEngine
    from tests.test_csharp_pipeline import _probe
    from tests.test_csharp_pipeline import _raw as pipeline_raw

    plain = pipeline_raw("eec_full", tmp_path / "keys", 2)
    n_species = int(plain["simulation.nspecies"])
    raw = plain | depletable(plain)
    raw |= {f"population.seeding.biomass.sp{i}": "0" for i in range(n_species)}
    config, grid, *_ = PythonEngine()._prepare_run(raw, 3)
    build_bundle(config, raw, grid, seed=3).save(tmp_path / "model.bin")
    res = _probe("Run", tmp_path / "model.bin", tmp_path / "run.out")
    assert res.get("out.summary")[0] == 0  # completed
    assert np.asarray(res.get("out.abundance")).sum() == 0  # no fish, so nothing ate

    rs = simulation_resources(config, grid)
    for step in range(int(config.n_steps)):
        rs.update(step)
    want = rs.biomass.copy()
    rs.close()
    assert int(config.n_steps) == 48
    assert np.array(res.get("out.resources"))[: len(want)].tobytes() == want.tobytes()

    once = simulation_resources(config, grid)
    once.update(int(config.n_steps) - 1)
    assert once.biomass.tobytes() != want.tobytes()
    once.close()
