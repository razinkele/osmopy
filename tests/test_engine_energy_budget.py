"""Tests for energy_budget: bioenergetic energy allocation."""
import numpy as np
import pytest

from osmose.engine.processes.energy_budget import compute_energy_budget, update_e_net_avg


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _default_params() -> dict:
    return dict(
        assimilation=0.7,
        c_m=0.001,
        beta=0.8,
        eta=1.5,
        r=0.5,
        m0=5.0,
        m1=2.0,
        e_maint_energy=0.65,
        phi_t=1.0,
        f_o2=1.0,
        n_dt_per_year=24,
    )


class TestComputeEnergyBudget:
    def test_positive_e_net_gives_growth(self):
        """Positive net energy produces positive weight increments."""
        params = _default_params()
        ingestion = np.array([0.01])  # tonnes — large enough to beat maintenance
        weight = np.array([0.001])
        gonad_weight = np.array([0.0])
        age_dt = np.array([48])  # 2 years (mature for m0=5, m1=2 → l_mat=9cm)
        length = np.array([15.0])  # above l_mature
        # Use a large e_net_avg so rho is < 1 and both soma + gonad get positive increments
        e_net_avg = np.array([100.0])

        dw, dg, e_net = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=15.0, e_net_avg=e_net_avg, **params
        )
        assert e_net[0] > 0
        assert dw[0] > 0
        assert dg[0] > 0  # mature fish → gonad allocation

    def test_negative_e_net_no_growth(self):
        """Negative net energy produces zero weight increments."""
        params = _default_params()
        ingestion = np.array([0.0])  # no food
        weight = np.array([0.01])   # heavy enough for maintenance to dominate
        gonad_weight = np.array([0.0])
        age_dt = np.array([24])
        length = np.array([10.0])
        e_net_avg = np.array([0.0])

        dw, dg, e_net = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=15.0, e_net_avg=e_net_avg, **params
        )
        assert e_net[0] < 0
        assert dw[0] == pytest.approx(0.0)
        assert dg[0] == pytest.approx(0.0)

    def test_arrhenius_maintenance_increases_with_temperature(self):
        """Higher temperature → higher maintenance cost → lower E_net."""
        params = _default_params()
        ingestion = np.array([0.005])
        weight = np.array([0.001])
        gonad_weight = np.array([0.0])
        age_dt = np.array([12])
        length = np.array([8.0])
        e_net_avg = np.array([0.1])

        _, _, e_net_cold = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=5.0, e_net_avg=e_net_avg, **params
        )
        _, _, e_net_warm = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=25.0, e_net_avg=e_net_avg, **params
        )
        assert e_net_cold[0] > e_net_warm[0]

    def test_rho_zero_for_immature(self):
        """Immature fish (length < l_mature) get rho=0: all growth goes to soma."""
        params = _default_params()
        ingestion = np.array([0.01])
        weight = np.array([0.001])
        gonad_weight = np.array([0.0])
        age_dt = np.array([12])  # 0.5 years → l_mature = 5 + 2*0.5 = 6 cm
        length = np.array([4.0])  # below l_mature → immature
        e_net_avg = np.array([0.05])

        dw, dg, e_net = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=15.0, e_net_avg=e_net_avg, **params
        )
        assert e_net[0] > 0
        assert dg[0] == pytest.approx(0.0)
        assert dw[0] > 0

    def test_rho_positive_for_mature(self):
        """Mature fish allocate positive fraction to gonads."""
        params = _default_params()
        ingestion = np.array([0.02])
        weight = np.array([0.005])
        gonad_weight = np.array([0.0])
        age_dt = np.array([96])  # 4 years → l_mature = 5 + 2*4 = 13 cm
        length = np.array([20.0])  # well above l_mature
        # Large e_net_avg to keep rho < 1 so both soma and gonad get non-trivial share
        e_net_avg = np.array([1000.0])

        dw, dg, e_net = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=15.0, e_net_avg=e_net_avg, **params
        )
        assert dg[0] > 0
        assert dw[0] > 0
        # Somatic + gonad increment should equal total positive net energy
        total_increment_g = (dw[0] + dg[0]) * 1e6
        e_pos = max(e_net[0], 0.0)
        assert total_increment_g == pytest.approx(e_pos, rel=1e-6)

    def test_units_tonnes(self):
        """Weight increments are returned in tonnes (< ingestion in tonnes)."""
        params = _default_params()
        ingestion = np.array([0.001])
        weight = np.array([0.001])
        gonad_weight = np.array([0.0])
        age_dt = np.array([24])
        length = np.array([15.0])
        e_net_avg = np.array([0.1])

        dw, dg, _ = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=15.0, e_net_avg=e_net_avg, **params
        )
        # Increments should be small fractions of a tonne, not huge values
        assert abs(dw[0]) < 0.1
        assert abs(dg[0]) < 0.1

    def test_vectorized_schools(self):
        """Multiple schools handled correctly."""
        params = _default_params()
        ingestion = np.array([0.01, 0.0, 0.02])
        weight = np.array([0.001, 0.001, 0.005])
        gonad_weight = np.zeros(3)
        age_dt = np.array([12, 12, 96])
        length = np.array([4.0, 4.0, 20.0])
        e_net_avg = np.array([0.05, 0.05, 0.1])

        dw, dg, e_net = compute_energy_budget(
            ingestion, weight, gonad_weight, age_dt, length,
            temp_c=15.0, e_net_avg=e_net_avg, **params
        )
        assert dw.shape == (3,)
        assert e_net[1] < 0   # no ingestion → negative
        assert dw[1] == pytest.approx(0.0)
        assert dg[1] == pytest.approx(0.0)


class TestUpdateENetAvg:
    def test_first_timestep_becomes_e_net(self):
        """At first feeding step, average equals current e_net."""
        e_net_avg = np.array([0.0])
        e_net = np.array([0.5])
        weight = np.array([0.001])
        age_dt = np.array([5])
        first_feeding_age_dt = np.array([5])

        new_avg = update_e_net_avg(e_net_avg, e_net, weight, age_dt, first_feeding_age_dt, 24)
        assert new_avg[0] == pytest.approx(0.5)

    def test_pre_feeding_unchanged(self):
        """Schools younger than first_feeding_age are not updated."""
        e_net_avg = np.array([0.0])
        e_net = np.array([0.8])
        weight = np.array([0.001])
        age_dt = np.array([2])
        first_feeding_age_dt = np.array([5])

        new_avg = update_e_net_avg(e_net_avg, e_net, weight, age_dt, first_feeding_age_dt, 24)
        assert new_avg[0] == pytest.approx(0.0)

    def test_running_average_converges(self):
        """Repeated updates with constant e_net converge to that value."""
        e_net_avg = np.array([0.0])
        e_net_val = 0.4
        first_feeding_age_dt = np.array([0])
        n_steps = 1000

        avg = e_net_avg.copy()
        for step in range(1, n_steps + 1):
            avg = update_e_net_avg(
                avg, np.array([e_net_val]),
                np.array([0.001]),
                np.array([step], dtype=np.int32),
                first_feeding_age_dt, 24
            )
        assert avg[0] == pytest.approx(e_net_val, rel=1e-2)
