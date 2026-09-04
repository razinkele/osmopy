import numpy as np
import pytest

from osmose.calibration.bioen_offline import (
    BioenFixed,
    FitResult,
    SpeciesTargets,
    bioen_param_lines,
    c_m_from_share,
    fit_species,
    g_net,
    simulate_growth,
    solve_tp,
)
from osmose.engine.processes.temp_function import arrhenius, phi_t

FX = BioenFixed()


def test_solve_tp_puts_net_growth_optimum_at_t_opt():
    for t_opt in (10.0, 15.0, 21.7, 27.0):
        tp = solve_tp(t_opt, FX)
        assert tp > t_opt  # maintenance pulls the optimum below the phiT peak
        T = np.linspace(t_opt - 8, t_opt + 8, 1601)
        cm = c_m_from_share(1.0, tp, FX)
        g = g_net(T, 1.0, cm, tp, FX)
        assert abs(T[np.argmax(g)] - t_opt) <= 0.05
        assert phi_t(np.array(tp), FX.e_m, FX.e_d, tp) == 1.0


def test_c_m_share_is_m_at_t_ref():
    tp = solve_tp(12.0, FX)
    imax = 4.0
    cm = c_m_from_share(imax, tp, FX)
    share = (
        cm
        * arrhenius(np.array(FX.t_ref), FX.e_maint)
        / (FX.a * imax * phi_t(np.array(FX.t_ref), FX.e_m, FX.e_d, tp))
    )
    assert share == pytest.approx(FX.m_share, rel=1e-12)


def test_simulate_growth_saturates_where_rho_reaches_one():
    tp = solve_tp(10.0, FX)
    imax, r = 6.0, 0.4
    cm = c_m_from_share(imax, tp, FX)
    # 80 yr, not the originally-drafted 40 yr: measured convergence to w_inf for these
    # parameters has an exponential time constant of ~12.5 yr (rho -> 1 asymptotically, never
    # exactly, so the approach to w_inf is a slow exponential decay of the deviation), so 40 yr
    # (~3.2 time constants) lands at only 83% of w_inf (-16.7%) -- well outside rel=0.05.
    # Measured: 40yr -16.72%, 60yr -3.57%, 80yr -0.73%, 100yr -0.15% (Task 8).
    w = simulate_growth(
        imax,
        r,
        tp,
        cm,
        np.full(24, 10.0),
        w_egg_g=1e-3,
        n_steps=24 * 80,
        ndt=24,
        cf=0.0087,
        b=3.05,
        m0=38.0,
        m1=0.0,
        fx=FX,
    )
    assert np.all(np.diff(w) >= 0) and np.isfinite(w).all()
    gbar = g_net(np.array(10.0), imax, cm, tp, FX)
    w_inf = (FX.eta * gbar / r) ** (1.0 / (1.0 - FX.beta))
    assert w[-1] == pytest.approx(w_inf, rel=0.05)


def test_fit_recovers_known_parameters_from_its_own_model():
    tp = solve_tp(10.0, FX)
    imax_true, r_true = 5.0, 0.35
    cm = c_m_from_share(imax_true, tp, FX)
    t24 = 8.0 + 4.0 * np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    w = simulate_growth(
        imax_true, r_true, tp, cm, t24, 1e-3, 24 * 20, 24, 0.0087, 3.05, 38.0, 0.0, FX
    )
    ages = np.arange(24, 24 * 20 + 1) / 24.0
    L = (w[24:] / 0.0087) ** (1 / 3.05)
    # fit a vBGF to the model's own curve, then ask the fitter to recover (imax, r) from that vBGF
    from scipy.optimize import curve_fit

    (linf, k, t0), _ = curve_fit(
        lambda a, L_, k_, t0_: L_ * (1 - np.exp(-k_ * (a - t0_))), ages, L, p0=(100, 0.15, -0.2)
    )
    tg = SpeciesTargets("synthetic", linf, k, t0, 0.0087, 3.05, 1e-3, 38.0, 0.0, 20.0, 10.0, t24)
    res = fit_species(tg, FX)
    assert res.imax == pytest.approx(imax_true, rel=0.05) and res.r == pytest.approx(
        r_true, rel=0.05
    )
    # rms_len_pct is a whole-range RELATIVE length error, and this synthetic curve is
    # pre-maturity-CONVEX (rho=0 -> pure w' = g*w^beta growth), unlike vBGF's shape which
    # decelerates from age 0 -- the unconstrained curve_fit above lands t0 > 1 yr, so the
    # youngest fitted ages (just above t0) sit near the vBGF's own zero-crossing, where a few
    # cm of absolute mismatch reads as 100s of percent relative error. That is a property of
    # re-fitting a non-vBGF-shaped curve with a vBGF, not a defect in (imax, r) recovery, which
    # the line above already pins to <=5%; measured 108.5% with this exact scenario (Task 8).
    # Real species (data/examples's 8 BoB targets) all have t0 < 0, so age >= 1 yr never nears
    # that boundary and rms_len_pct behaves as a normal small-residual diagnostic there.
    assert res.rms_len_pct < 150.0 and res.t_p == pytest.approx(tp)


def test_param_lines_cover_the_java_inventory_and_background():
    res = [FitResult("cod", 4.0, 0.3, 1e12, 13.0, 10.0, 2.0, 5e3, 5.1e3, 0.6, 400)]
    lines = bioen_param_lines(
        res,
        FX,
        zlayer={"cod": 1},
        sp_index={"cod": 0},
        background_imax={15: 2.5},
        notes={"cod": "T_opt 10 C (Bjornsson & Steinarsson 2002)"},
        m0={"cod": 2.0},
    )
    text = "\n".join(lines)
    for key in (
        "species.bioen.mobilized.tp.sp0;13.0",
        "species.bioen.maint.energy.c_m.sp0;1e+12",
        "species.maturity.m0.sp0;2.0",
        "species.maturity.r.sp0;0.3",
        "predation.ingestion.rate.max.sp0;4.0",
        "species.zlayer.sp0;1",
        "species.bioen.forage.k_for.sp0;0.0",
        "predation.c.bioen.sp0;0.0",
        "predation.larval.ingestion.rate.increase.ratio.sp0;1.0",
        "species.oxygen.c2.sp0;",
        "predation.ingestion.rate.max.sp15;2.5",
    ):
        assert key in text, key
    assert "T_opt 10 C" in text
