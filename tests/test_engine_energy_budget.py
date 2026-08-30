"""Tests for energy_budget: bioenergetic energy allocation.

Rewritten for Java's per-school framework (see
``tests/test_engine_bioen_budget_parity.py`` for the hand-computed Java gate):
``ingestion``/``e_gross``/``e_maint``/``e_net`` are tonnes PER SCHOOL, ``weight``
is tonnes PER FISH, and ``compute_energy_budget`` returns ``dw``/``dg`` in tonnes
PER FISH. ``enet_faced`` (stored in ``state.e_net_avg``) is per fish, per
``g^beta``, annualised.
"""

import numpy as np
import pytest

from osmose.engine.processes.energy_budget import (
    compute_energy_budget,
    energy_terms,
    update_enet_faced,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

# c_m is ~1e12 because arrhenius(T, 0.65) ~ 1e-12 at 5–25 degC; the product is O(1).
_ABUNDANCE = 1e4


def _default_params() -> dict:
    return dict(
        assimilation=0.7,
        c_m=1.0e12,
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


def _budget(ingestion, weight, age_dt, length, enet_faced, *, abundance=None, temp_c=15.0, **over):
    """Call compute_energy_budget with the shared defaults, in per-school units."""
    params = _default_params()
    params.update(over)
    n = len(ingestion)
    abundance = np.full(n, _ABUNDANCE) if abundance is None else abundance
    return compute_energy_budget(
        ingestion,
        weight,
        abundance,
        np.zeros(n),
        age_dt,
        length,
        temp_c,
        params["assimilation"],
        params["c_m"],
        params["beta"],
        params["eta"],
        params["r"],
        params["m0"],
        params["m1"],
        params["e_maint_energy"],
        params["phi_t"],
        params["f_o2"],
        params["n_dt_per_year"],
        enet_faced,
    )


class TestComputeEnergyBudget:
    def test_positive_e_net_gives_growth(self):
        """Positive net energy produces positive weight increments."""
        # Large enet_faced keeps rho well below 1 so soma AND gonad both grow.
        dw, dg, e_net, *_ = _budget(
            ingestion=np.array([10.0]),  # t/school, enough to beat maintenance
            weight=np.array([0.001]),  # t/fish
            age_dt=np.array([48], dtype=np.int32),  # 2 yr -> l_mat = 5 + 2*2 = 9 cm
            length=np.array([15.0]),  # above l_mature
            enet_faced=np.array([1e4]),
        )
        assert e_net[0] > 0
        assert dw[0] > 0
        assert dg[0] > 0  # mature fish -> gonad allocation

    def test_negative_e_net_no_growth(self):
        """Negative net energy produces zero weight increments."""
        dw, dg, e_net, *_ = _budget(
            ingestion=np.array([0.0]),  # no food
            weight=np.array([0.01]),  # heavy enough for maintenance to dominate
            age_dt=np.array([24], dtype=np.int32),
            length=np.array([10.0]),
            enet_faced=np.array([1.0]),
        )
        assert e_net[0] < 0
        assert dw[0] == pytest.approx(0.0)
        assert dg[0] == pytest.approx(0.0)

    def test_arrhenius_maintenance_increases_with_temperature(self):
        """Higher temperature -> higher maintenance cost -> lower E_net."""
        kwargs = dict(
            ingestion=np.array([5.0]),
            weight=np.array([0.001]),
            age_dt=np.array([12], dtype=np.int32),
            length=np.array([8.0]),
            enet_faced=np.array([1e4]),
        )
        _, _, e_net_cold, _, e_maint_cold, _ = _budget(temp_c=5.0, **kwargs)
        _, _, e_net_warm, _, e_maint_warm, _ = _budget(temp_c=25.0, **kwargs)
        assert e_maint_warm[0] > e_maint_cold[0]
        assert e_net_cold[0] > e_net_warm[0]

    def test_maintenance_scales_with_abundance(self):
        """Java getMaintenance multiplies by instantaneous abundance: E_maint is per school."""
        weight = np.array([0.001, 0.001])
        abundance = np.array([1e4, 1e7])
        _, e_maint, _ = energy_terms(
            np.zeros(2), weight, abundance, 15.0, 0.7, 1.0e12, 0.8, 0.65, 1.0, 1.0, 24
        )
        assert e_maint[1] / e_maint[0] == pytest.approx(1e3, rel=1e-12)

    def test_rho_zero_for_immature(self):
        """Immature fish (length < l_mature) get rho=0: all growth goes to soma."""
        dw, dg, e_net, *_ = _budget(
            ingestion=np.array([10.0]),
            weight=np.array([0.001]),
            age_dt=np.array([12], dtype=np.int32),  # 0.5 yr -> l_mature = 5 + 2*0.5 = 6 cm
            length=np.array([4.0]),  # below l_mature -> immature
            enet_faced=np.array([1e4]),
        )
        assert e_net[0] > 0
        assert dg[0] == pytest.approx(0.0)
        assert dw[0] > 0

    def test_rho_positive_for_mature(self):
        """Mature fish allocate a positive fraction to gonads; dw + dg exhausts E_net/N."""
        abundance = np.array([_ABUNDANCE])
        dw, dg, e_net, *_ = _budget(
            ingestion=np.array([50.0]),
            weight=np.array([0.005]),
            age_dt=np.array([96], dtype=np.int32),  # 4 yr -> l_mature = 5 + 2*4 = 13 cm
            length=np.array([20.0]),  # well above l_mature
            enet_faced=np.array([1e4]),  # keeps rho < 1
            abundance=abundance,
        )
        assert dg[0] > 0
        assert dw[0] > 0
        # Somatic + gonad increment equals the positive net energy, PER FISH.
        e_pos = max(e_net[0], 0.0)
        assert dw[0] + dg[0] == pytest.approx(e_pos / abundance[0], rel=1e-12)

    def test_units_tonnes_per_fish(self):
        """dw/dg are per-fish tonnes: a 1e3x bigger school at the same per-fish intake
        gets the same increment, and both are far below the per-school ingestion."""
        weight = np.array([0.001, 0.001])
        abundance = np.array([1e4, 1e7])
        ingestion = np.array([1.0, 1000.0])  # identical intake per fish
        dw, dg, *_ = _budget(
            ingestion=ingestion,
            weight=weight,
            age_dt=np.array([24, 24], dtype=np.int32),
            length=np.array([15.0, 15.0]),
            enet_faced=np.array([1e4, 1e4]),
            abundance=abundance,
        )
        assert dw[0] == pytest.approx(dw[1], rel=1e-12)
        assert dg[0] == pytest.approx(dg[1], rel=1e-12)
        # Increments are small fractions of a tonne, not huge values
        assert abs(dw[0]) < 0.1
        assert abs(dg[0]) < 0.1

    def test_vectorized_schools(self):
        """Multiple schools handled correctly."""
        dw, dg, e_net, *_ = _budget(
            ingestion=np.array([10.0, 0.0, 50.0]),
            weight=np.array([0.001, 0.001, 0.005]),
            age_dt=np.array([12, 12, 96], dtype=np.int32),
            length=np.array([4.0, 4.0, 20.0]),
            enet_faced=np.array([1e4, 1e4, 1e4]),
        )
        assert dw.shape == (3,)
        assert e_net[1] < 0  # no ingestion -> negative
        assert dw[1] == pytest.approx(0.0)
        assert dg[1] == pytest.approx(0.0)


class TestUpdateEnetFaced:
    """Java EnergyBudget.computeEnetFaced (state.e_net_avg holds Java's enet_faced)."""

    @staticmethod
    def _per_fish(e_net, abundance, weight, beta=0.8, n_dt=24):
        return e_net * n_dt / abundance * 1e6 / (weight * 1e6) ** beta

    def test_first_feeding_step_replaces_the_average(self):
        """At ageDt == firstFeedingAgeDt there is no averaging: the value is replaced."""
        prev = np.array([3.0])
        faced = update_enet_faced(
            prev,
            np.array([0.5]),
            np.array([1e4]),
            np.array([0.001]),
            np.array([5], dtype=np.int32),
            np.array([5], dtype=np.int32),
            larvae_thres_dt=1,
            larval_coef=1.0,
            beta=0.8,
            n_dt_per_year=24,
        )
        assert faced[0] == pytest.approx(self._per_fish(0.5, 1e4, 0.001))

    def test_pre_feeding_is_zero(self):
        """Java sets output = 0 for a school younger than first feeding age."""
        faced = update_enet_faced(
            np.array([3.0]),
            np.array([0.8]),
            np.array([1e4]),
            np.array([0.001]),
            np.array([2], dtype=np.int32),
            np.array([5], dtype=np.int32),
            larvae_thres_dt=1,
            larval_coef=1.0,
            beta=0.8,
            n_dt_per_year=24,
        )
        assert faced[0] == 0.0

    def test_larval_branch_divides_by_the_larval_coefficient(self):
        """Below larvaeThresDt, E_net is divided by larvaePredationRateBioen."""
        args = (
            np.array([0.0]),
            np.array([0.4]),
            np.array([1e4]),
            np.array([0.001]),
            np.array([3], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
        larval = update_enet_faced(*args, 5, 2.0, 0.8, 24)
        adult = update_enet_faced(*args, 1, 2.0, 0.8, 24)
        assert larval[0] == pytest.approx(adult[0] / 2.0)

    def test_running_average_converges(self):
        """Repeated updates with a constant E_net converge on the per-fish value."""
        e_net_val = 0.4
        abundance = np.array([1e4])
        weight = np.array([0.001])
        faced = np.array([0.0])
        for step in range(1, 1001):
            faced = update_enet_faced(
                faced,
                np.array([e_net_val]),
                abundance,
                weight,
                np.array([step], dtype=np.int32),
                np.array([0], dtype=np.int32),
                larvae_thres_dt=0,
                larval_coef=1.0,
                beta=0.8,
                n_dt_per_year=24,
            )
        assert faced[0] == pytest.approx(self._per_fish(e_net_val, 1e4, 0.001), rel=1e-2)
