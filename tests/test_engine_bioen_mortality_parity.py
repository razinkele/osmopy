"""Gate G (spec §4): ingestion cap form, survivor scaling, interleaved starvation, dispatch.

Four bioen mortality defects are pinned here, each transcribed from Java 4.3.3:

* ``BioenPredationMortality.getMaxPredationRate`` + the ``maxBiomassToPredate`` lines —
  the cap is per FISH, in tonnes, and is multiplied by the *instantaneous* abundance at
  every predator visit (not applied post-hoc to a per-school total).
* ``School.setNdead``/``incrementNdead`` — every death rescales the school's accumulated
  ingestion and its stored ``e_net`` by the survivor fraction.
* ``BioenStarvationMortality.computeStarvation`` — runs INSIDE the interleaved sub-step
  loop, on the previous step's ``e_net`` (Java step order mortality -> EnergyBudget ->
  reproduction), with strict ``ageDt > firstFeedingAgeDt`` eligibility.
* ``mortality()`` must not reach the batched Numba kernels under bioen (spec decision 14).
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes import mortality as M
from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap
from osmose.engine.processes.bioen_starvation import bioen_starvation_substep
from osmose.engine.simulate import SimulationContext
from osmose.engine.state import MortalityCause, SchoolState

N_CAUSES = len(MortalityCause)


# ---------------------------------------------------------------------------
# 1. Per-fish ingestion cap (BioenPredationMortality)
# ---------------------------------------------------------------------------


def test_per_fish_cap_matches_java_bioen_predation_mortality():
    # Java: Imax_eff = (Imax + (coef-1)*c_rate)/ndt ; cap_school = Imax_eff * (w*1e6)^beta
    #       / subdt * N * 1e-6
    weight = np.array([1e-3, 1e-3])
    species = np.array([0, 0], dtype=np.int32)
    age = np.array([10, 0], dtype=np.int32)  # second is larval (0 < larvae_thres_dt = 1)
    cap = per_fish_ingestion_cap(
        weight,
        species,
        age,
        i_max_all=np.array([3.5]),
        beta=np.array([0.8]),
        larvae_thres_dt=np.array([1], dtype=np.int32),
        theta=np.array([2.0]),
        c_rate=np.array([1.0]),
        n_species=1,
        n_dt_per_year=24,
        n_subdt=10,
    )
    adult = (3.5 / 24) * (1e-3 * 1e6) ** 0.8 / 10 * 1e-6
    larval = ((3.5 + (2.0 - 1.0) * 1.0) / 24) * (1e-3 * 1e6) ** 0.8 / 10 * 1e-6
    assert cap[0] == pytest.approx(adult, rel=1e-12)
    assert cap[1] == pytest.approx(larval, rel=1e-12)


def test_background_predator_uses_its_own_entry_without_the_ndt_division():
    """Java's ``getMaxPredationRate`` early-returns for ``speciesIndex >= nSpecies``.

    ``BioenPredationMortality.java:246-248`` returns ``predationRateBioen[speciesIndex]``
    *before* the ``/ getNStepYear()`` that the focal branch applies. Background Imax is
    therefore consumed as a PER-TIME-STEP rate while focal Imax is per year. Replicated
    verbatim; see the module docstring of ``bioen_predation.py`` for the overlay hazard.
    """
    weight = np.array([1e-2])
    species = np.array([3], dtype=np.int32)
    age = np.array([10], dtype=np.int32)
    cap = per_fish_ingestion_cap(
        weight,
        species,
        age,
        i_max_all=np.array([1.0, 1.0, 1.0, 9.0]),
        beta=np.array([0.8, 0.8, 0.8]),
        larvae_thres_dt=np.ones(3, dtype=np.int32),
        theta=np.ones(3),
        c_rate=np.zeros(3),
        n_species=3,
        n_dt_per_year=24,
        n_subdt=1,
    )
    assert cap[0] == pytest.approx(9.0 * (1e4) ** 0.8 * 1e-6, rel=1e-12)


def test_larval_coefficient_only_applies_below_the_threshold():
    """The larval branch is gated on ``ageDt < larvaeThresDt`` (Java, strict)."""
    weight = np.full(3, 1e-3)
    species = np.zeros(3, dtype=np.int32)
    age = np.array([1, 2, 3], dtype=np.int32)
    cap = per_fish_ingestion_cap(
        weight,
        species,
        age,
        i_max_all=np.array([3.0]),
        beta=np.array([0.8]),
        larvae_thres_dt=np.array([2], dtype=np.int32),
        theta=np.array([4.0]),
        c_rate=np.array([1.0]),
        n_species=1,
        n_dt_per_year=24,
        n_subdt=1,
    )
    w_term = (1e-3 * 1e6) ** 0.8 * 1e-6
    assert cap[0] == pytest.approx((3.0 + 3.0) / 24 * w_term, rel=1e-12)  # ageDt 1 < 2
    assert cap[1] == pytest.approx(3.0 / 24 * w_term, rel=1e-12)  # ageDt 2, not larval
    assert cap[2] == pytest.approx(3.0 / 24 * w_term, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Survivor scaling (School.setNdead / incrementNdead)
# ---------------------------------------------------------------------------


def test_kill_scales_ingestion_and_enet_by_survivor_fraction_only_under_bioen():
    st = SchoolState.create(1)
    st = st.replace(preyed_biomass=np.array([4.0]), e_net=np.array([-2.0]))
    inst = np.array([100.0])
    M._kill(st, 0, M._PREDATION, 50.0, inst, bioen=True)
    assert inst[0] == 50.0 and st.n_dead[0, M._PREDATION] == 50.0
    assert st.preyed_biomass[0] == 2.0 and st.e_net[0] == -1.0

    st2 = SchoolState.create(1).replace(preyed_biomass=np.array([4.0]), e_net=np.array([-2.0]))
    inst2 = np.array([100.0])
    M._kill(st2, 0, M._PREDATION, 50.0, inst2, bioen=False)
    assert inst2[0] == 50.0 and st2.n_dead[0, M._PREDATION] == 50.0
    assert st2.preyed_biomass[0] == 4.0 and st2.e_net[0] == -2.0


def test_kill_does_not_scale_background_schools():
    """Java's ``AbstractSchool.incrementNdead`` (background) has no ingestion/e_net."""
    st = SchoolState.create(1).replace(
        preyed_biomass=np.array([4.0]),
        e_net=np.array([-2.0]),
        is_background=np.array([True]),
    )
    inst = np.array([100.0])
    M._kill(st, 0, M._PREDATION, 50.0, inst, bioen=True)
    assert inst[0] == 50.0
    assert st.preyed_biomass[0] == 4.0 and st.e_net[0] == -2.0


# ---------------------------------------------------------------------------
# 3. Interleaved starvation (BioenStarvationMortality.computeStarvation)
# ---------------------------------------------------------------------------


def test_starvation_substep_matches_java_branches():
    # sufficient gonad: pays eta*deficit, repays e_net, no deaths
    n_dead, gonad, e_net = bioen_starvation_substep(
        e_net=-10.0, gonad_weight=5.0, weight=1e-3, eta=1.0, n_subdt=2
    )
    assert n_dead == 0.0 and gonad == 0.0 and e_net == -5.0  # deficit per subdt = 5; gonad 5 >= 5
    # insufficient: gonad flushed, zero repayment (Java flush-then-credit), deaths = deficit/w
    n_dead, gonad, e_net = bioen_starvation_substep(
        e_net=-10.0, gonad_weight=1.0, weight=1e-3, eta=1.0, n_subdt=2
    )
    assert gonad == 0.0 and e_net == -10.0 and n_dead == pytest.approx(5.0 / 1e-3)
    # positive e_net: nothing happens
    assert bioen_starvation_substep(1.0, 3.0, 1e-3, 1.0, 2) == (0.0, 3.0, 1.0)


def test_starvation_substep_eta_scales_the_gonad_price():
    """Java compares ``gonadWeight >= eta * eNetSubDt`` and pays ``eta * eNetSubDt``."""
    n_dead, gonad, e_net = bioen_starvation_substep(
        e_net=-4.0, gonad_weight=3.0, weight=1e-3, eta=1.5, n_subdt=2
    )
    # deficit = 2.0, eta*deficit = 3.0, gonad 3.0 >= 3.0 -> sufficient
    assert n_dead == 0.0
    assert gonad == pytest.approx(0.0)
    assert e_net == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# 4. Cause set
# ---------------------------------------------------------------------------


def test_causes_include_starvation_and_foraging_under_bioen():
    from types import SimpleNamespace

    assert M._get_mortality_causes(SimpleNamespace(bioen_enabled=True)) == [
        M._PREDATION,
        M._STARVATION,
        M._ADDITIONAL,
        M._FISHING,
        M._FORAGING,
    ]
    assert M._get_mortality_causes(SimpleNamespace(bioen_enabled=False)) == [
        M._PREDATION,
        M._STARVATION,
        M._ADDITIONAL,
        M._FISHING,
    ]


# ---------------------------------------------------------------------------
# 5. Dispatch + end-to-end wiring through mortality()
# ---------------------------------------------------------------------------


def _bioen_config(**overrides: str) -> EngineConfig:
    """A one-species bioen config with every competing mortality switched off.

    Only STARVATION and PREDATION can kill here, which makes the interleaved-loop
    assertions below independent of the shuffled cause order.
    """
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nresource": "0",
        "mortality.subdt": "1",
        "simulation.bioen.enabled": "true",
        "simulation.bioen.phit.enabled": "false",
        "simulation.bioen.fo2.enabled": "false",
        "mortality.fishing.recruitment.age.sp0": "0",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "Fish",
        "species.linf.sp0": "60.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.egg.weight.sp0": "0.0005",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "10",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.first.feeding.age.sp0": "0.0417",  # 1 dt at ndt = 24
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.predprey.sizeratio.min.sp0": "2.0",
        "predation.predprey.sizeratio.max.sp0": "20.0",
        "mortality.natural.rate.sp0": "0.0",
        "mortality.fishing.rate.sp0": "0.0",
        "species.beta.sp0": "0.8",
        "species.zlayer.sp0": "0",
        "species.bioen.assimilation.sp0": "0.7",
        "species.bioen.maint.energy.c_m.sp0": "0.001",
        "species.maturity.eta.sp0": "1.0",
        "species.maturity.r.sp0": "0.5",
        "species.maturity.m0.sp0": "30.0",
        "species.maturity.m1.sp0": "0.0",
        "species.bioen.mobilized.e.mobi.sp0": "0.65",
        "species.bioen.mobilized.e.d.sp0": "1.5",
        "species.bioen.mobilized.tp.sp0": "20.0",
        "species.bioen.maint.e.maint.sp0": "0.65",
        "species.bioen.forage.k_for.sp0": "0.0",
        "predation.larval.ingestion.rate.increase.ratio.sp0": "1.0",
        "predation.c.bioen.sp0": "0.0",
        "species.larvae.growth.threshold.age.sp0": "0.0417",
    }
    cfg.update(overrides)
    return EngineConfig.from_dict(cfg)


def _one_school(**overrides) -> SchoolState:
    st = SchoolState.create(1)
    base = dict(
        abundance=np.array([1e5]),
        weight=np.array([1e-3]),
        length=np.array([40.0]),
        length_start=np.array([40.0]),
        age_dt=np.array([50], dtype=np.int32),
        first_feeding_age_dt=np.array([1], dtype=np.int32),
        trophic_level=np.array([3.0]),
    )
    base.update(overrides)
    return st.replace(**base)


def test_mortality_never_enters_batched_numba_under_bioen(monkeypatch):
    """The batched kernels must not be reached when bioen is on (spec §0 'Numba dispatch')."""
    config = _bioen_config()
    grid = Grid.from_dimensions(ny=3, nx=3)
    state = _one_school()
    calls = {"batched": 0}

    def _boom(*a, **k):
        calls["batched"] += 1
        raise AssertionError("batched Numba kernel reached under bioen")

    monkeypatch.setattr(M, "_HAS_NUMBA", True, raising=False)
    monkeypatch.setattr(M, "_mortality_all_cells_numba", _boom, raising=False)
    monkeypatch.setattr(M, "_mortality_all_cells_parallel", _boom, raising=False)
    M.mortality(state, None, config, np.random.default_rng(0), grid, step=3)
    assert calls["batched"] == 0


def test_bioen_starvation_fires_inside_the_interleaved_loop():
    """End-to-end: the previous step's e_net kills fish, flushes gonad and repays e_net.

    Hand-derived (Java ``computeStarvation``, subdt = 1):
      deficit  = |e_net| / subdt = 2.0 t/school
      gonad (0.5) < eta*deficit (2.0)  -> insufficient branch
      gonad -> 0, e_net unchanged by the repayment, nDead = 2.0 / 1e-3 = 2000
    then ``School.incrementNdead`` scales e_net by the survivor fraction
      f = (1e5 - 2000) / 1e5 = 0.98  ->  e_net = -2.0 * 0.98 = -1.96
    """
    config = _bioen_config()
    grid = Grid.from_dimensions(ny=3, nx=3)
    state = _one_school(e_net=np.array([-2.0]), gonad_weight=np.array([0.5]))

    out = M.mortality(state, None, config, np.random.default_rng(7), grid, step=3)

    assert out.n_dead[0, MortalityCause.STARVATION] == pytest.approx(2000.0, rel=1e-12)
    assert out.gonad_weight[0] == 0.0
    assert out.e_net[0] == pytest.approx(-1.96, rel=1e-12)
    assert out.abundance[0] == pytest.approx(98000.0, rel=1e-12)


def test_bioen_starvation_skips_schools_at_first_feeding_age():
    """Java ``isStarvationEnabledBioen``: ``ageDt > firstFeedingAgeDt`` (strict)."""
    config = _bioen_config()
    grid = Grid.from_dimensions(ny=3, nx=3)
    state = _one_school(
        e_net=np.array([-2.0]),
        gonad_weight=np.array([0.0]),
        age_dt=np.array([1], dtype=np.int32),
        first_feeding_age_dt=np.array([1], dtype=np.int32),
    )
    out = M.mortality(state, None, config, np.random.default_rng(7), grid, step=3)
    assert out.n_dead[0, MortalityCause.STARVATION] == 0.0
    assert out.e_net[0] == -2.0


def test_no_starvation_under_bioen_when_enet_is_positive():
    config = _bioen_config()
    grid = Grid.from_dimensions(ny=3, nx=3)
    state = _one_school(e_net=np.array([3.0]), gonad_weight=np.array([0.5]))
    out = M.mortality(state, None, config, np.random.default_rng(7), grid, step=3)
    assert out.n_dead[0, MortalityCause.STARVATION] == 0.0
    assert out.gonad_weight[0] == 0.5
    assert out.e_net[0] == 3.0


def _predator_prey_state() -> SchoolState:
    """Predator (school 0) with a starvation deficit, plus a prey school in the same cell."""
    st = SchoolState.create(2)
    return st.replace(
        abundance=np.array([1e5, 1e7]),
        weight=np.array([1e-3, 1e-6]),
        length=np.array([40.0, 4.0]),
        length_start=np.array([40.0, 4.0]),
        age_dt=np.array([50, 50], dtype=np.int32),
        first_feeding_age_dt=np.array([1, 1], dtype=np.int32),
        trophic_level=np.array([3.0, 2.0]),
        e_net=np.array([-2.0, 0.0]),
        gonad_weight=np.array([0.0, 0.0]),
    )


def test_predation_uses_the_bioen_cap_in_the_loop():
    """max_eatable = cap_fish[p] * instantaneous abundance, not biomass * ingestion_rate."""
    config = _bioen_config()
    grid = Grid.from_dimensions(ny=3, nx=3)
    state = _predator_prey_state().replace(e_net=np.zeros(2))
    out = M.mortality(state, None, config, np.random.default_rng(3), grid, step=3)

    # Java: Imax_eff * (w*1e6)^beta / subdt * N * 1e-6, with N the instantaneous abundance
    # (1e5 here — nothing kills the predator in this arm).
    expected = (3.5 / 24) * (1e-3 * 1e6) ** 0.8 / 1 * 1e5 * 1e-6
    assert out.preyed_biomass[0] == pytest.approx(expected, rel=1e-12)
    # The standard (bioen-off) cap would be ~7x larger — proves the form changed.
    standard = 1e5 * 1e-3 * 3.5 / 24
    assert out.preyed_biomass[0] < 0.5 * standard


def _run_predator_prey(seed: int):
    """Two sub-steps, so at least one death is guaranteed to follow at least one meal
    whatever the shuffled cause order does inside a sub-step.

    Returns ``(out_state, raw_eaten_by_predator)``; ``ctx.diet_matrix`` accumulates the RAW
    eaten biomass and is never rescaled, so it is an independent witness of the unscaled
    total.
    """
    config = _bioen_config(**{"mortality.subdt": "2"})
    grid = Grid.from_dimensions(ny=3, nx=3)
    state = _predator_prey_state()
    ctx = SimulationContext(
        diet_tracking_enabled=True,
        diet_matrix=np.zeros((2, config.n_species + 1), dtype=np.float64),
    )
    out = M.mortality(state, None, config, np.random.default_rng(seed), grid, step=3, ctx=ctx)
    return out, float(ctx.diet_matrix[0].sum())


@pytest.mark.parametrize("seed", [0, 11, 12345])
def test_survivor_scaling_reduces_ingestion_below_the_raw_eaten_total(seed):
    """The predator loses fish to starvation, so its ingestion is rescaled (Java setNdead)."""
    out, raw_eaten = _run_predator_prey(seed)
    assert raw_eaten > 0.0
    assert out.n_dead[0, MortalityCause.STARVATION] > 0.0
    assert 0.0 < out.preyed_biomass[0] < raw_eaten


@pytest.mark.parametrize("seed", [0, 11, 12345])
def test_trophic_level_uses_the_raw_preyed_total_under_bioen(seed):
    """Java's TL reads the never-rescaled ``preyedBiomass`` (MortalityProcess:396-401).

    Every prey here has TL 2.0, so the Java answer is exactly 3.0. Dividing the weighted
    sum by the survivor-scaled ingestion instead would push it above 3.
    """
    out, raw_eaten = _run_predator_prey(seed)
    # Guard: the assertion below is only meaningful while the rescaling actually bit.
    assert out.preyed_biomass[0] < raw_eaten
    assert out.trophic_level[0] == pytest.approx(3.0, rel=1e-12)
