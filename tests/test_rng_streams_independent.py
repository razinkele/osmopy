"""Adjacent replicate seeds must not share a random stream (perf plan section 7).

``PythonEngine._prepare_run`` builds three kinds of generator: the main ``default_rng(seed)``,
the per-species movement generators from ``seed`` and the per-species mortality generators from
``seed + 1``. Replicate and UQ tasks use consecutive seeds, so if the mortality generators were
consumed, replicate s's mortality stream would be replicate s+1's movement and main stream, and
adjacent replicates would be correlated.

Settled 2026-11-17 by tracing and by this test: no production path consumes either per-species
set. Mortality's batched kernels seed their per-cell MT19937 from the main generator, and movement
(maps and random) draws from it too, so each replicate has one stream of its own and consecutive
seeds are independent. One consequence is that ``movement.randomseed.fixed`` and
``stochastic.mortality.randomseed.fixed`` have no effect on the Python engine's results.

If this test fails, something now draws from a per-species generator. Before accepting that, fix
the seeding at the task level (space or spawn replicate seeds in the caller; see section 7 of
docs/osmopy-performance-optimization-plan.md). Changing ``_prepare_run`` moves every fixed-seed
baseline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.simulate import simulate


class _Trap:
    def __init__(self, label: str):
        self.label = label

    def __getattr__(self, name):
        raise AssertionError(
            f"{self.label} was consumed (.{name}): adjacent replicate seeds may now share a stream"
        )


@pytest.mark.parametrize("demo", ["bay_of_biscay", "eec_full"])  # random, maps movement
def test_per_species_generators_are_never_consumed(demo, tmp_path):
    info = osmose_demo(demo, tmp_path)
    cfg = dict(OsmoseConfigReader().read(Path(info["config_file"])))
    cfg["simulation.time.nyear"] = "1"
    config, grid, rng, _mv, _mo = PythonEngine()._prepare_run(cfg, 7)
    traps = {
        "movement_rngs": [_Trap("movement_rngs")] * config.n_species,
        "mortality_rngs": [_Trap("mortality_rngs")] * config.n_species,
    }
    simulate(config, grid, rng, **traps)
