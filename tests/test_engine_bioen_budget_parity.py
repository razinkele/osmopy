"""Gate G (spec §4): energy budget transcribed from EnergyBudget.java, hand-computed.

Java runs the whole bioen budget in TONNES PER SCHOOL and divides by the school's
instantaneous abundance only at the per-fish weight increments (``getDw``/``getDg``).
These tests pin that framework:

* ``getMaintenance``  -> ``c_m * (w*1e6)^beta * arrhenius / nStepYear * N * 1e-6`` (t/school)
* ``getEgross``       -> ``ingestion * assimilation * phi_T * f_O2``               (t/school)
* ``computeEnetFaced``-> per-fish, per-``g^beta``, annualised running mean
* ``getRho``          -> ``r / (eta * enet_faced) * (w*1e6)^(1-beta)`` clamped to [0, 1]
* ``getDw``/``getDg`` -> ``(1-rho) * max(Enet, 0) / N`` and ``rho * max(Enet, 0) / N``
"""

import numpy as np
import pytest

from osmose.engine.processes.energy_budget import (
    compute_energy_budget,
    energy_terms,
    update_enet_faced,
)
from osmose.engine.processes.temp_function import arrhenius

K = dict(
    assimilation=0.7,
    c_m=1.0e12,
    beta=0.8,
    eta=1.0,
    r=0.5,
    m0=30.0,
    m1=0.0,
    e_maint_energy=0.65,
    n_dt_per_year=24,
)


def _three_schools():
    # school 0: N=1e3, school 1: N=1e6 (same per-fish weight), school 2: immature small fish
    weight = np.array([1e-3, 1e-3, 1e-5])  # t/fish (1 kg, 1 kg, 10 g)
    abundance = np.array([1e3, 1e6, 1e6])
    ingestion = np.array([0.05, 50.0, 0.1])  # t/school this step (1e3x apart for 0 vs 1)
    length = np.array([50.0, 50.0, 10.0])
    age_dt = np.array([120, 120, 12], dtype=np.int32)
    gonad = np.zeros(3)
    return weight, abundance, ingestion, length, age_dt, gonad


def test_maintenance_is_per_school_tonnes():
    weight, abundance, ingestion, *_ = _three_schools()
    temp = 10.0
    e_gross, e_maint, e_net = energy_terms(
        ingestion,
        weight,
        abundance,
        temp,
        K["assimilation"],
        K["c_m"],
        K["beta"],
        K["e_maint_energy"],
        1.0,
        1.0,
        24,
    )
    w_g = weight * 1e6
    expected = K["c_m"] * w_g ** K["beta"] * arrhenius(np.array(temp), 0.65) / 24 * abundance * 1e-6
    np.testing.assert_allclose(e_maint, expected, rtol=1e-12)
    # 1e3x abundance at equal per-fish weight -> 1e3x maintenance (per-school framework)
    assert e_maint[1] / e_maint[0] == pytest.approx(1e3, rel=1e-12)
    np.testing.assert_allclose(e_gross, ingestion * 0.7, rtol=1e-12)
    np.testing.assert_allclose(e_net, ingestion * 0.7 - e_maint, rtol=1e-12)


def test_egross_scales_with_phi_t_and_f_o2():
    """Java getEgross multiplies ingestion by assimilation, phi_T and f_O2."""
    weight, abundance, ingestion, *_ = _three_schools()
    phi = np.array([0.5, 1.0, 0.25])
    fo2 = np.array([1.0, 0.4, 0.8])
    e_gross, _, _ = energy_terms(
        ingestion, weight, abundance, 10.0, 0.7, K["c_m"], 0.8, 0.65, phi, fo2, 24
    )
    np.testing.assert_allclose(e_gross, ingestion * 0.7 * phi * fo2, rtol=1e-12)


def test_maintenance_rises_with_temperature():
    """Arrhenius: warmer water costs more maintenance, so E_net falls."""
    weight, abundance, ingestion, *_ = _three_schools()
    args = (weight, abundance)
    _, m_cold, net_cold = energy_terms(
        ingestion, *args, 5.0, 0.7, K["c_m"], 0.8, 0.65, 1.0, 1.0, 24
    )
    _, m_warm, net_warm = energy_terms(
        ingestion, *args, 25.0, 0.7, K["c_m"], 0.8, 0.65, 1.0, 1.0, 24
    )
    assert np.all(m_warm > m_cold)
    assert np.all(net_warm < net_cold)


def test_dw_is_per_fish_and_independent_of_abundance_at_equal_intake_per_fish():
    weight, abundance, ingestion, length, age_dt, gonad = _three_schools()
    temp = 10.0
    e_gross, e_maint, e_net = energy_terms(
        ingestion, weight, abundance, temp, 0.7, K["c_m"], 0.8, 0.65, 1.0, 1.0, 24
    )
    # Guard against the assertions below going quiet if K is ever retuned so that
    # E_net <= 0 (dw and dg would then be trivially 0 for every school).
    assert e_net[0] > 0
    faced = update_enet_faced(
        np.zeros(3),
        e_net,
        abundance,
        weight,
        age_dt,
        np.ones(3, dtype=np.int32),
        larvae_thres_dt=1,
        larval_coef=1.0,
        beta=0.8,
        n_dt_per_year=24,
    )
    dw, dg, e_net2, e_gross2, e_maint2, rho = compute_energy_budget(
        ingestion,
        weight,
        abundance,
        gonad,
        age_dt,
        length,
        temp,
        0.7,
        K["c_m"],
        0.8,
        1.0,
        0.5,
        30.0,
        0.0,
        0.65,
        1.0,
        1.0,
        24,
        faced,
    )
    # compute_energy_budget re-derives the same terms internally
    np.testing.assert_allclose(e_net2, e_net, rtol=1e-12)
    np.testing.assert_allclose(e_gross2, e_gross, rtol=1e-12)
    np.testing.assert_allclose(e_maint2, e_maint, rtol=1e-12)
    # schools 0 and 1 have identical per-fish intake (0.05/1e3 == 50/1e6) -> identical dw, dg
    assert dw[0] == pytest.approx(dw[1], rel=1e-12) and dg[0] == pytest.approx(dg[1], rel=1e-12)
    assert dw[0] + dg[0] > 0  # non-trivial: the split below is not 0 == 0
    np.testing.assert_allclose(dw, (1 - rho) * np.maximum(e_net, 0) / abundance, rtol=1e-12)
    np.testing.assert_allclose(dg, rho * np.maximum(e_net, 0) / abundance, rtol=1e-12)
    # dw + dg exhausts the positive net energy, per fish
    np.testing.assert_allclose(dw + dg, np.maximum(e_net, 0) / abundance, rtol=1e-12)
    assert rho[2] == 0.0  # immature (10 cm < m0=30)


def test_interior_rho_splits_net_energy_between_soma_and_gonad():
    """With a large enet_faced, rho lands strictly inside (0, 1): both dw and dg are positive."""
    weight, abundance, ingestion, length, age_dt, gonad = _three_schools()
    faced = np.full(3, 1e4)  # rho_raw = 0.5/1e4 * (1e3)^0.2 ~ 2e-4
    dw, dg, e_net, _, _, rho = compute_energy_budget(
        ingestion,
        weight,
        abundance,
        gonad,
        age_dt,
        length,
        10.0,
        0.7,
        K["c_m"],
        0.8,
        1.0,
        0.5,
        30.0,
        0.0,
        0.65,
        1.0,
        1.0,
        24,
        faced,
    )
    assert 0.0 < rho[0] < 1.0
    assert dw[0] > 0 and dg[0] > 0
    assert dg[0] / dw[0] == pytest.approx(rho[0] / (1.0 - rho[0]), rel=1e-12)
    np.testing.assert_allclose(dw[0] + dg[0], e_net[0] / abundance[0], rtol=1e-12)


def test_negative_e_net_gives_no_increment():
    """Java getDw/getDg: nothing is added when E_net <= 0 (starvation handles the deficit)."""
    weight, abundance, _, length, age_dt, gonad = _three_schools()
    starved = np.zeros(3)  # no ingestion at all -> E_net = -E_maint < 0
    dw, dg, e_net, *_ = compute_energy_budget(
        starved,
        weight,
        abundance,
        gonad,
        age_dt,
        length,
        10.0,
        0.7,
        K["c_m"],
        0.8,
        1.0,
        0.5,
        30.0,
        0.0,
        0.65,
        1.0,
        1.0,
        24,
        np.ones(3),
    )
    assert np.all(e_net < 0)
    np.testing.assert_array_equal(dw, np.zeros(3))
    np.testing.assert_array_equal(dg, np.zeros(3))


def test_maturity_follows_lmrn_line():
    """Java getMaturation: mature iff length >= m0 + m1 * age_years; rho is 0 below it."""
    weight = np.full(3, 1e-3)
    abundance = np.full(3, 1e3)
    ingestion = np.full(3, 0.05)
    age_dt = np.array([48, 48, 48], dtype=np.int32)  # 2 years at 24 dt/yr
    length = np.array([8.9, 9.0, 20.0])  # l_mat = m0 + m1*2 = 5 + 2*2 = 9 cm
    *_, rho = compute_energy_budget(
        ingestion,
        weight,
        abundance,
        np.zeros(3),
        age_dt,
        length,
        10.0,
        0.7,
        K["c_m"],
        0.8,
        1.0,
        0.5,
        5.0,
        2.0,
        0.65,
        1.0,
        1.0,
        24,
        np.full(3, 1e4),
    )
    assert rho[0] == 0.0  # just below the LMRN line -> immature
    assert rho[1] > 0.0  # exactly on the line -> mature (Java uses >=)
    assert rho[2] > 0.0


def test_enet_faced_matches_java_compute_enet_faced():
    weight = np.array([1e-3, 1e-3, 1e-3])
    abundance = np.array([1e3, 1e3, 1e3])
    e_net = np.array([0.05, 0.05, 0.05])
    age_dt = np.array([1, 3, 50], dtype=np.int32)
    ff = np.ones(3, dtype=np.int32)
    prev = np.array([9.0, 9.0, 9.0])
    faced = update_enet_faced(
        prev,
        e_net,
        abundance,
        weight,
        age_dt,
        ff,
        larvae_thres_dt=5,
        larval_coef=2.0,
        beta=0.8,
        n_dt_per_year=24,
    )
    per_fish = 0.05 * 24 / 1e3 * 1e6 / (1e-3 * 1e6) ** 0.8
    assert faced[0] == pytest.approx(per_fish / 2.0)  # ageDt == firstFeeding: /coef, no averaging
    assert faced[1] == pytest.approx((per_fish / 2.0 + 9.0 * 3) / 4)  # larval: /coef, weighted
    assert faced[2] == pytest.approx((per_fish + 9.0 * 50) / 51)  # adult: no coef
    # Java: output = 0 for a school that has not reached first feeding age.
    faced0 = update_enet_faced(
        prev, e_net, abundance, weight, np.zeros(3, dtype=np.int32), ff, 5, 2.0, 0.8, 24
    )
    np.testing.assert_array_equal(faced0, np.zeros(3))


def test_enet_faced_larval_threshold_boundary():
    """ageDt == larvaeThresDt takes the adult branch (Java's < in the larval test)."""
    weight = np.full(2, 1e-3)
    abundance = np.full(2, 1e3)
    e_net = np.full(2, 0.05)
    age_dt = np.array([4, 5], dtype=np.int32)  # thres = 5 -> 4 larval, 5 adult
    prev = np.full(2, 9.0)
    faced = update_enet_faced(
        prev, e_net, abundance, weight, age_dt, np.ones(2, dtype=np.int32), 5, 2.0, 0.8, 24
    )
    per_fish = 0.05 * 24 / 1e3 * 1e6 / (1e-3 * 1e6) ** 0.8
    assert faced[0] == pytest.approx((per_fish / 2.0 + 9.0 * 4) / 5)
    assert faced[1] == pytest.approx((per_fish + 9.0 * 5) / 6)


def test_enet_faced_dead_school_keeps_previous_value():
    """Java loops getAliveSchools(): a school with no fish left is never touched."""
    weight = np.full(2, 1e-3)
    abundance = np.array([0.0, 1e3])
    e_net = np.array([0.0, 0.05])
    prev = np.array([7.0, 9.0])
    faced = update_enet_faced(
        prev,
        e_net,
        abundance,
        weight,
        np.array([50, 50], dtype=np.int32),
        np.ones(2, dtype=np.int32),
        1,
        1.0,
        0.8,
        24,
    )
    assert faced[0] == 7.0
    assert np.isfinite(faced[1]) and faced[1] != 9.0


def test_enet_faced_running_mean_converges():
    """Repeated adult updates with a constant per-fish E_net converge on that value."""
    weight = np.array([1e-3])
    abundance = np.array([1e3])
    e_net = np.array([0.05])
    per_fish = 0.05 * 24 / 1e3 * 1e6 / (1e-3 * 1e6) ** 0.8
    faced = np.array([0.0])
    for age in range(1, 601):
        faced = update_enet_faced(
            faced,
            e_net,
            abundance,
            weight,
            np.array([age], dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            0,
            1.0,
            0.8,
            24,
        )
    assert faced[0] == pytest.approx(per_fish, rel=1e-2)


def test_rho_guard_matches_java_clamp_semantics():
    weight, abundance, ingestion, length, age_dt, gonad = _three_schools()
    temp = 10.0
    faced = np.array([0.0, -1.0, 1e-9])  # zero -> +inf -> 1 ; negative -> 0 ; tiny positive -> 1
    *_, rho = compute_energy_budget(
        ingestion,
        weight,
        abundance,
        gonad,
        age_dt,
        length,
        temp,
        0.7,
        K["c_m"],
        0.8,
        1.0,
        0.5,
        30.0,
        0.0,
        0.65,
        1.0,
        1.0,
        24,
        faced,
    )
    assert rho[0] == 1.0 and rho[1] == 0.0
    assert rho[2] == 0.0  # school 2 immature regardless


def test_rho_zero_when_r_zero_and_faced_zero():
    """r == 0 with enet_faced == 0 is Java's 0/0 -> NaN; we map it to 0 (r=0 means no gonads)."""
    weight, abundance, ingestion, length, age_dt, gonad = _three_schools()
    *_, rho = compute_energy_budget(
        ingestion,
        weight,
        abundance,
        gonad,
        age_dt,
        length,
        10.0,
        0.7,
        K["c_m"],
        0.8,
        1.0,
        0.0,  # r = 0
        30.0,
        0.0,
        0.65,
        1.0,
        1.0,
        24,
        np.zeros(3),
    )
    np.testing.assert_array_equal(rho, np.zeros(3))
    assert np.all(np.isfinite(rho))


def test_zero_abundance_school_gets_no_increment():
    weight = np.array([1e-3])
    abundance = np.array([0.0])
    ingestion = np.array([1.0])  # positive E_net: only the alive guard keeps dw finite
    dw, dg, e_net, *_ = compute_energy_budget(
        ingestion,
        weight,
        abundance,
        np.zeros(1),
        np.array([120], dtype=np.int32),
        np.array([50.0]),
        10.0,
        0.7,
        0.0,
        0.8,
        1.0,
        0.5,
        30.0,
        0.0,
        0.65,
        1.0,
        1.0,
        24,
        np.array([1.0]),
    )
    assert e_net[0] > 0
    assert dw[0] == 0.0 and dg[0] == 0.0 and np.isfinite(dw[0]) and np.isfinite(dg[0])


def test_budget_is_vectorized_over_schools():
    weight, abundance, ingestion, length, age_dt, gonad = _three_schools()
    faced = np.full(3, 1e4)
    dw, dg, e_net, e_gross, e_maint, rho = compute_energy_budget(
        ingestion,
        weight,
        abundance,
        gonad,
        age_dt,
        length,
        np.array([5.0, 10.0, 15.0]),
        0.7,
        K["c_m"],
        0.8,
        1.0,
        0.5,
        30.0,
        0.0,
        0.65,
        1.0,
        1.0,
        24,
        faced,
    )
    for arr in (dw, dg, e_net, e_gross, e_maint, rho):
        assert arr.shape == (3,)
        assert np.all(np.isfinite(arr))
