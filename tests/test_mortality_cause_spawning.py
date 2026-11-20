"""OSMOSE 4.4.x's ninth mortality cause, SPAWNING (#137).

Java 4.4.1 (``MortalityCause``) has nine causes, the last ``SPAWNING(8)``, and every mortality
output loops over all of them, so its files carry a ninth stage triple (its header names only
eight; GitHub #141). The engine carries the cause too. No process books it yet, so the column is
0; the C# engine, which books eight causes, gets a zero ninth column from the bridge.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osmose.engine.state import MortalityCause, SchoolState


def test_the_ninth_cause_is_spawning():
    assert MortalityCause.SPAWNING == 8 and len(MortalityCause) == 9
    assert list(MortalityCause)[-2:] == [MortalityCause.AGING, MortalityCause.SPAWNING]


def test_a_new_state_carries_nine_causes():
    assert SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32)).n_dead.shape == (
        3,
        9,
    )


def test_the_csharp_bridge_pads_a_zero_spawning_column():
    from osmose.engine_cs.engine import _step_outputs

    n, nf = 2, 3
    mortality = np.arange(n * nf * 8, dtype=np.float64).reshape(n, nf, 8) + 1.0
    staged = np.arange(n * nf * 8 * 3, dtype=np.float64).reshape(n, nf, 8, 3) + 1.0
    buffers = SimpleNamespace(
        record_step=np.array([0, 1], dtype=np.int32),
        window_steps=np.ones(n, dtype=np.int32),
        biomass=np.zeros((n, nf)),
        abundance=np.zeros((n, nf)),
        mortality=mortality,
        yield_=np.zeros((n, nf)),
        families={
            "mortality_by_cause_stage": staged,
            "mortality_rate_by_cause_stage": staged * 2,
        },
        age_bin_offsets=None,
    )
    out = _step_outputs(buffers, n)
    for i, o in enumerate(out):
        assert o.mortality_by_cause.shape == (nf, 9)
        assert np.array_equal(o.mortality_by_cause[:, :8], mortality[i])
        assert (o.mortality_by_cause[:, 8] == 0).all()
        for got, src in (
            (o.mortality_by_cause_stage, staged[i]),
            (o.mortality_rate_by_cause_stage, staged[i] * 2),
        ):
            assert got.shape == (nf, 9, 3)
            assert np.array_equal(got[:, :8], src) and (got[:, 8] == 0).all()


def test_a_run_writes_a_zero_spawning_column(tmp_path):
    import osmose.engine.simulate as sim
    from osmose.engine import PythonEngine
    from osmose.engine.output import _build_mortality_dataframes

    cfg = {
        "simulation.time.ndtperyear": "4",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "mortality.subdt": "10",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Fish0",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.maturity.size.sp0": "12.0",
        "mortality.additional.rate.sp0": "0.4",
        "population.seeding.biomass.sp0": "1000",
        "_osmose.config.dir": str(tmp_path),
    }
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, 3)
    outputs = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    df = next(iter(_build_mortality_dataframes(outputs, config).values()))
    spawning = df["Spawning"]
    assert (spawning.to_numpy() == 0).all()
    assert (df["Additional"].to_numpy() > 0).any()  # the other causes still book deaths
