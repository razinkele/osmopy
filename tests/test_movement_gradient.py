"""OSMOSE 4.4.x gradient movement (#137).

Java (``GradientDistribution``, v4.4.1): a school with no map this step goes out of the domain.
Otherwise it moves to a cell drawn from the accessible cells (ocean, map value above 0) with
weight ``coef / N + (1 - coef) * value / sum(values)``, ``coef`` being
``movement.random.walk.coef.sp#``. An unlocated school may go anywhere accessible. A located one
searches neighbourhoods of ``movement.base.search.radius.sp# + e`` cells,
``e`` in (0, 1, 2, 3, 5, 10), but Java's loop leaves on the first EMPTY neighbourhood, against
its own comment. So in the jar a school with an accessible cell within the base radius chooses
among every accessible cell within base + 10, and one without chooses among every accessible
cell of the grid. The port does what the jar does.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.movement import (
    gradient_candidates,
    gradient_draw,
    gradient_weights,
)

OCEAN = np.ones((30, 30), dtype=bool)


def _map(cells: dict[tuple[int, int], float], shape=(30, 30)) -> np.ndarray:
    m = np.zeros(shape)
    for (y, x), v in cells.items():
        m[y, x] = v
    return m


class TestTheCandidates:
    def test_an_unlocated_school_may_go_to_any_accessible_cell(self):
        land = OCEAN.copy()
        land[5, 5] = False
        m = _map({(1, 1): 0.5, (20, 25): 2.0, (5, 5): 9.0, (7, 7): np.nan})
        assert sorted(gradient_candidates(-1, -1, m, land, 1)) == [(1, 1), (20, 25)]

    def test_with_a_cell_in_the_base_radius_it_searches_ten_cells_further(self):
        # (10, 11) is within radius 1 of the school at x=10, y=10; (10, 21) is 11 cells away,
        # inside base + 10; (10, 22) is 12 cells away, outside it.
        m = _map({(10, 11): 1.0, (10, 21): 1.0, (10, 22): 1.0, (29, 0): 1.0})
        assert sorted(gradient_candidates(10, 10, m, OCEAN, 1)) == [(10, 11), (10, 21)]

    def test_without_a_cell_in_the_base_radius_it_goes_global(self):
        # Nothing within radius 1; (10, 13) is within radius 1 + 2, but Java leaves its loop on
        # the empty first neighbourhood and takes every accessible cell of the grid.
        m = _map({(10, 13): 1.0, (29, 29): 1.0})
        assert sorted(gradient_candidates(10, 10, m, OCEAN, 1)) == [(10, 13), (29, 29)]

    def test_the_neighbourhood_is_a_clipped_square_centre_included(self):
        # Radius 0 from the corner: the cell itself is accessible, so the search runs out to
        # 10 cells, clipped at the edge: (10, 10) is in, (11, 11) and (0, 11) are not.
        m = _map({(0, 0): 1.0, (10, 10): 1.0, (11, 11): 1.0, (0, 11): 1.0}, shape=(12, 12))
        cells = gradient_candidates(0, 0, m, np.ones((12, 12), dtype=bool), 0)
        assert sorted(cells) == [(0, 0), (10, 10)]

    def test_the_global_list_follows_javas_ocean_cell_order(self):
        # AbstractGrid: j from ny-1 down (outer), i from nx-1 down (inner).
        m = _map({(0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0, (0, 0): 1.0})
        assert gradient_candidates(-1, -1, m, OCEAN, 1) == [(1, 1), (1, 0), (0, 1), (0, 0)]

    def test_javas_order_x_outer_y_inner(self):
        m = _map({(0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0})
        assert gradient_candidates(0, 0, m, OCEAN, 1) == [(1, 0), (0, 1), (1, 1)]


class TestTheDraw:
    def test_weights_mix_uniform_and_proportional(self):
        w = gradient_weights(np.array([1.0, 3.0]), 0.25)
        assert w == pytest.approx([0.25 / 2 + 0.75 * 0.25, 0.25 / 2 + 0.75 * 0.75])

    def test_coef_one_is_uniform_and_zero_is_proportional(self):
        assert gradient_weights(np.array([1.0, 3.0]), 1.0) == pytest.approx([0.5, 0.5])
        assert gradient_weights(np.array([1.0, 3.0]), 0.0) == pytest.approx([0.25, 0.75])

    def test_the_draw_takes_the_first_cumulative_weight_at_or_above(self):
        w = np.array([0.2, 0.3, 0.5])
        assert [gradient_draw(w, u) for u in (0.0, 0.2, 0.2000001, 0.5, 0.51, 0.999)] == [
            0,
            0,
            1,
            1,
            2,
            2,
        ]


def _config(tmp_path, **extra: str) -> dict[str, str]:
    grid = np.zeros((6, 6))
    grid[1:3, 1:3] = 1.0
    grid[4, 4] = 5.0
    path = tmp_path / "map0.csv"
    path.write_text("\n".join(";".join(str(v) for v in row) for row in grid) + "\n")
    cfg = {
        "simulation.time.ndtperyear": "4",
        "simulation.time.nyear": "2",
        "simulation.nspecies": "1",
        "mortality.subdt": "10",
        "grid.nlon": "6",
        "grid.nlat": "6",
        "simulation.nschool.sp0": "10",
        "species.name.sp0": "Fish0",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "1000",
        "movement.distribution.method.sp0": "gradient",
        "movement.base.search.radius.sp0": "1",
        "movement.random.walk.coef.sp0": "0.3",
        "movement.species.map0": "Fish0",
        "movement.initialage.map0": "0",
        "movement.lastage.map0": "5",
        "movement.file.map0": str(path),
        "_osmose.config.dir": str(tmp_path),
    }
    return cfg | extra


class TestTheKeys:
    def test_they_are_carried(self, tmp_path):
        c = EngineConfig.from_dict(_config(tmp_path))
        assert c.movement_is_gradient[0]
        assert c.movement_gradient_radius[0] == 1 and c.movement_gradient_coef[0] == 0.3  # type: ignore[index]

    def test_the_method_ignores_case_as_java(self, tmp_path):
        c = EngineConfig.from_dict(
            _config(tmp_path, **{"movement.distribution.method.sp0": "Gradient"})
        )
        assert c.movement_is_gradient[0]

    def test_without_gradient_nothing_is_set(self, tmp_path):
        c = EngineConfig.from_dict(
            _config(tmp_path, **{"movement.distribution.method.sp0": "maps"})
        )
        assert c.movement_gradient_radius is None and not c.movement_is_gradient.any()

    def test_the_radius_is_required(self, tmp_path):
        cfg = _config(tmp_path)
        del cfg["movement.base.search.radius.sp0"]
        with pytest.raises(ValueError, match="movement.base.search.radius.sp0"):
            EngineConfig.from_dict(cfg)

    @pytest.mark.parametrize("coef", ["1.5", "-0.1"])
    def test_a_coef_outside_zero_one_is_refused(self, tmp_path, coef):
        with pytest.raises(ValueError, match="random.walk.coef.sp0 in \\[0, 1\\]"):
            EngineConfig.from_dict(_config(tmp_path, **{"movement.random.walk.coef.sp0": coef}))


def _run(cfg: dict[str, str], seed: int = 5):
    import osmose.engine.simulate as sim
    from osmose.engine import PythonEngine

    seen: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real = sim._movement

    def spy(state, *a, **k):
        out = real(state, *a, **k)
        seen.append((out.cell_x.copy(), out.cell_y.copy(), out.is_out.copy()))
        return out

    sim._movement = spy
    try:
        config, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, seed)
        records = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    finally:
        sim._movement = real
    return records, seen


def test_a_whole_run_keeps_every_school_on_an_accessible_cell(tmp_path):
    records, seen = _run(_config(tmp_path))
    # CSV row 0 is the northernmost row, grid row ny - 1 (movement_maps flips on load).
    accessible = {(5 - y, x) for y, x in [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4)]}
    assert seen and all(np.isfinite(r.biomass).all() for r in records)
    visited = set()
    for cx, cy, out in seen:
        for x, y, o in zip(cx, cy, out, strict=True):
            if not o:
                assert (int(y), int(x)) in accessible
                visited.add((int(y), int(x)))
    assert visited == accessible  # the heavy cell and the patch are both used


def test_a_species_without_a_map_this_step_goes_out(tmp_path):
    _, seen = _run(_config(tmp_path, **{"movement.lastage.map0": "0.25"}))
    assert any(out.any() for _, _, out in seen)


def test_a_school_comes_back_from_out_of_the_domain(tmp_path):
    # Map 0 covers steps 0, 1 and 3 of each year; map 1 is null at step 2: out, then back.
    cfg = _config(tmp_path) | {
        "movement.steps.map0": "0;1;3",
        "movement.species.map1": "Fish0",
        "movement.initialage.map1": "0",
        "movement.lastage.map1": "5",
        "movement.steps.map1": "2",
        "movement.file.map1": "null",
    }
    _, seen = _run(cfg)
    accessible = {(5 - y, x) for y, x in [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4)]}
    out_at_2 = seen[2][2]
    assert out_at_2.any()
    cx, cy, out = seen[3]
    back = out_at_2[: len(out)] & ~out[: len(out_at_2)]
    assert back.any()
    for i in np.flatnonzero(back):
        assert (int(cy[i]), int(cx[i])) in accessible


def test_with_fixed_seeds_it_draws_from_the_species_stream_only(tmp_path):
    from osmose.engine import PythonEngine
    from osmose.engine.movement_maps import MovementMapSet
    from osmose.engine.processes.movement import movement
    from osmose.engine.state import SchoolState

    cfg = _config(tmp_path) | {"simulation.fixed.seed.enabled": "true"}
    config, grid, *_ = PythonEngine()._prepare_run(cfg, 1)
    assert config.movement_seed_fixed
    maps = {
        0: MovementMapSet(
            config=cfg,
            species_name="Fish0",
            n_dt_per_year=4,
            n_years=2,
            lifespan_dt=int(config.lifespan_dt[0]),
            ny=grid.ny,
            nx=grid.nx,
            config_dir=str(tmp_path),
        )
    }
    state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32)).replace(
        abundance=np.full(3, 10.0), age_dt=np.full(3, 4, dtype=np.int32)
    )
    global_rng, species_rng = np.random.default_rng(1), np.random.default_rng(2)
    before_global = global_rng.bit_generator.state
    before_species = species_rng.bit_generator.state
    movement(state, grid, config, 1, global_rng, map_sets=maps, species_rngs=[species_rng])
    assert global_rng.bit_generator.state == before_global
    assert species_rng.bit_generator.state != before_species


def test_the_salinity_gate_is_refused_with_gradient(tmp_path):
    cfg = _config(tmp_path) | {
        "movement.salinity.gate.enabled": "true",
        "movement.salinity.gate.species.enabled.sp0": "true",
        "movement.salinity.field.constant": "7.0",
    }
    with pytest.raises(ValueError, match="salinity gate"):
        EngineConfig.from_dict(cfg)


def test_the_csharp_engine_refuses_it_and_only_it(tmp_path):
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    assert "movement_gradient" in (csharp_engine_block_reason(_config(tmp_path)) or "")
    maps = _config(tmp_path, **{"movement.distribution.method.sp0": "maps"})
    for k in ("movement.base.search.radius.sp0", "movement.random.walk.coef.sp0"):
        del maps[k]
    assert "movement_gradient" not in (csharp_engine_block_reason(maps) or "")
