"""OSMOSE 4.4.x's ninth mortality cause, SPAWNING (#137).

Java 4.4.1 (``MortalityCause``) has nine causes, the last ``SPAWNING(8)``, and every mortality
output loops over all of them, so its files carry a ninth stage triple (its header names only
eight; GitHub #141). The engine carries the cause too. No process books it yet, so the column is
0; the C# engine carries it as well (nine-wide buffers), always zero.
"""

from __future__ import annotations

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


def test_the_csharp_result_buffers_carry_the_ninth_cause():
    """The C# engine books eight causes and refuses semelparous species; its buffers (and its
    RunResults.MortalityCauses) are nine wide, so the ninth column stays zero."""
    from osmose.engine_cs.build import N_MORTALITY_CAUSES

    assert N_MORTALITY_CAUSES == len(MortalityCause)
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "csharp/src/Osmose.Engine/Step/Outputs.cs"
    ).read_text()
    assert f"public const int MortalityCauses = {len(MortalityCause)};" in src


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
