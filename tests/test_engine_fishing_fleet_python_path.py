"""Fleet-effort scaling must apply on the pure-Python fishing fallback too, not
only the Numba path. Deep review 2026-06-22 (HIGH-2, _HAS_NUMBA=False-scoped)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osmose.engine.processes.mortality import _fleet_effort_factor


def _fleet_state(target_species, effort):
    # effort_map shape (n_fleets, ny, nx); one fleet, 2x2 grid.
    emap = np.zeros((1, 2, 2), dtype=np.float64)
    emap[0] = effort
    return SimpleNamespace(
        fleets=[SimpleNamespace(target_species=set(target_species))],
        effort_map=emap,
    )


def test_factor_none_when_no_fleet_state():
    assert _fleet_effort_factor(0, 0, 0, None) == 1.0


def test_factor_one_for_non_targeted_species():
    fs = _fleet_state(target_species=[1], effort=np.full((2, 2), 3.0))
    assert _fleet_effort_factor(0, 0, 0, fs) == 1.0  # sp 0 not targeted


def test_factor_sums_effort_for_targeted_species_in_cell():
    fs = _fleet_state(target_species=[0], effort=np.array([[2.0, 0.0], [0.0, 5.0]]))
    assert _fleet_effort_factor(0, 0, 0, fs) == 2.0
    assert _fleet_effort_factor(0, 1, 1, fs) == 5.0


def test_factor_zero_when_targeted_cell_out_of_bounds():
    fs = _fleet_state(target_species=[0], effort=np.full((2, 2), 3.0))
    assert _fleet_effort_factor(0, 9, 9, fs) == 0.0


def test_apply_fishing_for_school_scales_by_fleet_effort(monkeypatch):
    import osmose.engine.processes.mortality as m
    from osmose.engine.config import EngineConfig
    from osmose.engine.state import MortalityCause, SchoolState
    from tests.test_engine_mortality_loop import _make_config

    monkeypatch.setattr(m, "_HAS_NUMBA", False)
    cfg_dict = _make_config()
    cfg_dict["mortality.fishing.rate.sp0"] = "0.5"
    cfg = EngineConfig.from_dict(cfg_dict)

    def _run(fleet_state):
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state.abundance[0] = 1000.0
        state.age_dt[0] = 50
        state.length[0] = 30.0
        inst = state.abundance.copy()
        m._apply_fishing_for_school(
            0, state, cfg, n_subdt=1, inst_abd=inst, step=0, fleet_state=fleet_state
        )
        return (
            state.n_dead[0, int(MortalityCause.FISHING)]
            + state.n_dead[0, int(MortalityCause.DISCARDS)]
        )

    base = _run(None)
    scaled = _run(_fleet_state(target_species=[0], effort=np.full((2, 2), 2.0)))
    assert base > 0
    assert scaled > base  # ~2x effort -> more fishing deaths
