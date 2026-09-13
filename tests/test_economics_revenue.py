"""Coverage for `accumulate_fleet_revenue` (the H7 vectorisation).

Before this file, nothing exercised the revenue-accumulation arithmetic at all:
`test_economics_output.py` assigns `fs.vessel_revenue[:] = 500.0` to test the
writer, and the bit-exact parity baselines
(`tests/baselines/parity_baseline_*.npz`) are recorded from configs with
economics DISABLED, so `test_engine_parity.py` never reaches this code either.
H7 rewrote it from a nested (school x fleet) loop into bucketed `np.add.at`
calls, which is exactly the kind of change a missing test lets through.

`_reference_accumulate` below is a direct transcription of the pre-H7 loop.
Each test asserts the vectorised implementation agrees with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.economics.fleet import FleetConfig, create_fleet_state
from osmose.engine.simulate import accumulate_fleet_revenue
from osmose.engine.state import MortalityCause, SchoolState

NY, NX = 3, 4


def _fleet(name: str, n_vessels: int, port_y: int, port_x: int, targets, prices) -> FleetConfig:
    return FleetConfig(
        name=name,
        n_vessels=n_vessels,
        home_port_y=port_y,
        home_port_x=port_x,
        gear_type="trawl",
        max_days_at_sea=200,
        fuel_cost_per_cell=1.0,
        base_operating_cost=10.0,
        stock_elasticity=np.ones(len(prices), dtype=np.float64),
        target_species=list(targets),
        price_per_tonne=np.asarray(prices, dtype=np.float64),
    )


def _state(rows) -> SchoolState:
    """rows: list of (species_id, cell_y, cell_x, n_dead_fishing, weight)."""
    st = SchoolState.create(len(rows), species_id=np.array([r[0] for r in rows], dtype=np.int32))
    st.cell_y[:] = [r[1] for r in rows]
    st.cell_x[:] = [r[2] for r in rows]
    st.n_dead[:, int(MortalityCause.FISHING)] = [r[3] for r in rows]
    st.weight[:] = [r[4] for r in rows]
    return st


def _reference_accumulate(fleet_state, state) -> np.ndarray:
    """The pre-H7 implementation, transcribed verbatim as the oracle."""
    n_fleets = len(fleet_state.fleets)
    ny_f = fleet_state.catch_memory.shape[1]
    nx_f = fleet_state.catch_memory.shape[2]
    realized = np.zeros((n_fleets, ny_f, nx_f), dtype=np.float64)
    fishing_cause = int(MortalityCause.FISHING)

    for i in range(len(state)):
        fishing_dead = state.n_dead[i, fishing_cause]
        if fishing_dead <= 0:
            continue
        sp = int(state.species_id[i])
        cy, cx = int(state.cell_y[i]), int(state.cell_x[i])
        if not (0 <= cy < ny_f and 0 <= cx < nx_f):
            continue
        catch_biomass = fishing_dead * state.weight[i]
        for fi, fleet_cfg in enumerate(fleet_state.fleets):
            if sp in fleet_cfg.target_species:
                vessel_mask = (
                    (fleet_state.vessel_fleet == fi)
                    & (fleet_state.vessel_cell_y == cy)
                    & (fleet_state.vessel_cell_x == cx)
                )
                n_in_cell = int(vessel_mask.sum())
                if n_in_cell > 0:
                    rev_per_vessel = catch_biomass * fleet_cfg.price_per_tonne[sp] / n_in_cell
                    fleet_state.vessel_revenue[vessel_mask] += rev_per_vessel
                realized[fi, cy, cx] += catch_biomass
    return realized


def _both(fleets, rows, place_vessels=None):
    """Run reference and vectorised paths on identical inputs; return both results."""
    out = []
    for impl in (_reference_accumulate, accumulate_fleet_revenue):
        fs = create_fleet_state(fleets, NY, NX)
        if place_vessels is not None:
            place_vessels(fs)
        realized = impl(fs, _state(rows))
        out.append((realized, fs.vessel_revenue.copy()))
    return out


def test_matches_reference_single_fleet_single_cell():
    fleets = [_fleet("A", 2, 1, 1, targets=[0], prices=[100.0, 0.0])]
    rows = [(0, 1, 1, 3.0, 2.0)]  # catch = 6 t -> 600 revenue, split over 2 vessels
    (r_ref, v_ref), (r_new, v_new) = _both(fleets, rows)
    np.testing.assert_allclose(r_new, r_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v_new, v_ref, rtol=0, atol=1e-12)
    assert r_new[0, 1, 1] == pytest.approx(6.0)
    np.testing.assert_allclose(v_new, [300.0, 300.0])


def test_matches_reference_multi_fleet_multi_species_multi_cell():
    fleets = [
        _fleet("A", 3, 0, 0, targets=[0, 2], prices=[100.0, 50.0, 20.0]),
        _fleet("B", 2, 2, 3, targets=[1, 2], prices=[10.0, 500.0, 7.0]),
    ]
    rows = [
        (0, 0, 0, 1.0, 1.0),
        (1, 2, 3, 2.0, 3.0),
        (2, 0, 0, 0.5, 4.0),
        (2, 2, 3, 1.5, 2.0),
        (1, 0, 0, 9.0, 9.0),  # fleet B targets sp1 but has no vessel in (0,0)
    ]
    (r_ref, v_ref), (r_new, v_new) = _both(fleets, rows)
    np.testing.assert_allclose(r_new, r_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v_new, v_ref, rtol=0, atol=1e-12)


def test_catch_without_a_vessel_present_is_realized_but_unpaid():
    """Documents the pre-existing asymmetry H7 preserves."""
    fleets = [_fleet("A", 2, 0, 0, targets=[0], prices=[100.0])]
    rows = [(0, 2, 2, 4.0, 1.0)]  # caught far from the fleet's only occupied cell
    (r_ref, v_ref), (r_new, v_new) = _both(fleets, rows)
    np.testing.assert_allclose(r_new, r_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v_new, v_ref, rtol=0, atol=1e-12)
    assert r_new[0, 2, 2] == pytest.approx(4.0)
    assert v_new.sum() == 0.0


def test_non_target_species_earns_nothing():
    fleets = [_fleet("A", 1, 0, 0, targets=[1], prices=[100.0, 100.0])]
    rows = [(0, 0, 0, 5.0, 1.0)]  # sp0 is not targeted
    (r_ref, v_ref), (r_new, v_new) = _both(fleets, rows)
    np.testing.assert_allclose(r_new, r_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v_new, v_ref, rtol=0, atol=1e-12)
    assert r_new.sum() == 0.0
    assert v_new.sum() == 0.0


def test_out_of_grid_and_zero_catch_schools_are_skipped():
    fleets = [_fleet("A", 1, 0, 0, targets=[0], prices=[100.0])]
    # Each of the four rejection paths gets its own row. Mutation-tested: with
    # only a negative cell_y here, dropping the `cell_x >= 0` guard from the
    # implementation went undetected, so both negative axes are covered.
    rows = [
        (0, -1, 0, 5.0, 1.0),  # negative cell_y
        (0, 0, -1, 5.0, 1.0),  # negative cell_x
        (0, NY + 5, 0, 5.0, 1.0),  # cell_y past the grid
        (0, 0, NX + 5, 5.0, 1.0),  # cell_x past the grid
        (0, 0, 0, 0.0, 1.0),  # zero fishing deaths
        (0, 0, 0, 2.0, 1.0),  # the only contributing row
    ]
    (r_ref, v_ref), (r_new, v_new) = _both(fleets, rows)
    np.testing.assert_allclose(r_new, r_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v_new, v_ref, rtol=0, atol=1e-12)
    assert r_new.sum() == pytest.approx(2.0)
    assert v_new.sum() == pytest.approx(200.0)


def test_vessels_spread_across_cells_split_only_their_own_cell():
    fleets = [_fleet("A", 4, 0, 0, targets=[0], prices=[100.0])]

    def spread(fs):
        # two vessels in (1,1), one in (2,2), one left at the home port (0,0)
        fs.vessel_cell_y[:] = [1, 1, 2, 0]
        fs.vessel_cell_x[:] = [1, 1, 2, 0]

    rows = [(0, 1, 1, 1.0, 1.0), (0, 2, 2, 3.0, 1.0)]
    (r_ref, v_ref), (r_new, v_new) = _both(fleets, rows, place_vessels=spread)
    np.testing.assert_allclose(r_new, r_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v_new, v_ref, rtol=0, atol=1e-12)
    # (1,1): 100 split two ways; (2,2): 300 to one vessel; home port earns nothing.
    np.testing.assert_allclose(v_new, [50.0, 50.0, 300.0, 0.0])


def test_empty_catch_returns_zeros_and_leaves_revenue_untouched():
    fleets = [_fleet("A", 2, 0, 0, targets=[0], prices=[100.0])]
    fs = create_fleet_state(fleets, NY, NX)
    fs.vessel_revenue[:] = 7.5
    realized = accumulate_fleet_revenue(fs, _state([(0, 0, 0, 0.0, 1.0)]))
    assert realized.shape == (1, NY, NX)
    assert realized.sum() == 0.0
    np.testing.assert_allclose(fs.vessel_revenue, [7.5, 7.5])


def test_revenue_accumulates_across_repeated_steps():
    fleets = [_fleet("A", 1, 0, 0, targets=[0], prices=[100.0])]
    fs = create_fleet_state(fleets, NY, NX)
    st = _state([(0, 0, 0, 1.0, 1.0)])
    for _ in range(3):
        accumulate_fleet_revenue(fs, st)
    assert fs.vessel_revenue[0] == pytest.approx(300.0)
