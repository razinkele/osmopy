import importlib.util
from dataclasses import replace
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]


def _load_fit_module():
    """Load scripts/fit_baltic_bioen_params.py by path (same pattern as the Gate-B harness
    tests, tests/test_cross_engine_parity_bioen_staging.py -- the script lives under
    scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "fit_baltic_bioen_params", ROOT / "scripts" / "fit_baltic_bioen_params.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    assert res.t_p == pytest.approx(tp)

    # res.rms_len_pct is NOT asserted here: it measures the fit against the intermediate
    # vBGF re-fit target, and this synthetic curve is pre-maturity-CONVEX (rho=0 -> pure
    # w' = g*w^beta growth), unlike vBGF's shape which decelerates from age 0 -- the
    # unconstrained curve_fit above lands t0 > 1 yr, so the youngest fitted ages (just above
    # t0) sit near the vBGF's own zero-crossing, where a few cm of absolute mismatch reads as
    # 100s of percent relative error (measured 108.5% with this exact scenario, Task 8). That
    # is a property of re-fitting a non-vBGF-shaped curve with a vBGF, not a defect in
    # fit_species -- real species (data/examples's 8 BoB targets) all have t0 < 0, so
    # age >= 1 yr never nears that boundary and rms_len_pct is a normal diagnostic there.
    #
    # A more meaningful self-consistency check: does the recovered (imax, r, c_m, t_p)
    # reproduce the ORIGINAL simulated curve `w` itself (not the lossy vBGF proxy)? Past the
    # transient (age >= 3 yr, clear of the same t0-adjacent region above), it should -- this
    # is the assertion that actually pins fit quality and the length_all > 0.0 filter.
    w_recovered = simulate_growth(
        res.imax, res.r, res.t_p, res.c_m, t24, 1e-3, 24 * 20, 24, 0.0087, 3.05, 38.0, 0.0, FX
    )
    idx_all = np.arange(24, 24 * 20 + 1)
    ages_all = idx_all / 24.0
    mask = ages_all >= 3.0
    len_recovered = (w_recovered[idx_all][mask] / 0.0087) ** (1 / 3.05)
    len_true = (w[idx_all][mask] / 0.0087) ** (1 / 3.05)
    rms_vs_true = np.sqrt(np.mean(((len_recovered - len_true) / len_true) ** 2)) * 100
    assert rms_vs_true < 2.0  # measured 0.34% (Task 8)


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


def test_param_lines_writes_background_beta_matching_the_default():
    """species.beta.sp{background} must be authored (progress-log ruling R1, hard requirement
    for Task 11): Java's BackgroundSpecies.java has no default and errors on absence."""
    res = [FitResult("cod", 4.0, 0.3, 1e12, 13.0, 10.0, 2.0, 5e3, 5.1e3, 0.6, 400)]
    kwargs = dict(
        zlayer={"cod": 1},
        sp_index={"cod": 0},
        background_imax={15: 2.5},
        notes={"cod": "x"},
        m0={"cod": 2.0},
    )
    text_default = "\n".join(bioen_param_lines(res, FX, **kwargs))
    # default beta must equal per_fish_ingestion_cap's hardcoded background exponent (0.8) --
    # if these two numbers drift apart the "cap equals standard cap at w_mean" property that
    # background_imax was solved for silently stops holding.
    assert "species.beta.sp15;0.8" in text_default

    text_custom = "\n".join(bioen_param_lines(res, FX, background_beta={15: 0.75}, **kwargs))
    assert "species.beta.sp15;0.75" in text_custom
    assert "species.beta.sp15;0.8" not in text_custom


def test_habitat_t24_uses_engine_map_loader_and_layer(tmp_path):
    import xarray as xr

    fit = _load_fit_module()
    temp = np.zeros((24, 2, 3, 3))
    temp[:, 0] = 5.0
    temp[:, 1] = np.arange(24)[:, None, None]
    nc = tmp_path / "t.nc"
    xr.Dataset(
        {"temperature": (["time", "layer", "latitude", "longitude"], temp.astype(np.float32))}
    ).to_netcdf(nc)
    m = tmp_path / "map.csv"
    m.write_text("0;0;0\n0;1;0\n0;0;0\n")  # one habitat cell; orientation handled by _load_csv_grid
    t24 = fit.habitat_t24(nc, layer=1, map_files=[m], ny=3, nx=3)
    np.testing.assert_allclose(t24, np.arange(24))
    assert fit.habitat_t24(nc, layer=0, map_files=[m], ny=3, nx=3).mean() == 5.0


def test_habitat_t24_raises_on_frame_count_mismatch(tmp_path):
    """PhysicalData.get_grid indexes step % <loaded frame count>, not any declared nsteps.year
    metadata -- a 12-frame file fed 24 engine steps would silently repeat instead of erroring
    (CLAUDE.md frame-count gotcha). habitat_t24 must catch this itself."""
    import xarray as xr

    fit = _load_fit_module()
    temp = np.full((12, 1, 3, 3), 5.0, dtype=np.float32)
    nc = tmp_path / "t12.nc"
    xr.Dataset({"temperature": (["time", "layer", "latitude", "longitude"], temp)}).to_netcdf(nc)
    m = tmp_path / "map.csv"
    m.write_text("0;0;0\n0;1;0\n0;0;0\n")
    with pytest.raises(ValueError, match="frame"):
        fit.habitat_t24(nc, layer=0, map_files=[m], ny=3, nx=3)


def test_habitat_t24_raises_when_habitat_entirely_land_masked(tmp_path):
    import xarray as xr

    fit = _load_fit_module()
    temp = np.full((24, 1, 3, 3), np.nan, dtype=np.float32)
    nc = tmp_path / "tland.nc"
    xr.Dataset({"temperature": (["time", "layer", "latitude", "longitude"], temp)}).to_netcdf(nc)
    m = tmp_path / "map.csv"
    m.write_text("0;0;0\n0;1;0\n0;0;0\n")
    with pytest.raises(ValueError, match="land-masked"):
        fit.habitat_t24(nc, layer=0, map_files=[m], ny=3, nx=3)


def test_background_imax_matches_standard_cap_exactly_at_w_mean():
    """Closed-form pin for ruling R1: the bioen per-fish cap at w_mean must equal the
    standard (non-bioen) per-fish cap at w_mean. Also demonstrates the 24x hazard directly:
    the brief's own step-2 prose formula (no /n_dt_per_year) overshoots by exactly
    n_dt_per_year."""
    from osmose.engine.background import BackgroundSpeciesInfo

    fit = _load_fit_module()

    class _EC:
        n_dt_per_year = 24

    b = BackgroundSpeciesInfo(
        name="GreySeal",
        species_index=0,
        file_index=15,
        n_class=3,
        lengths=[50.0, 100.0, 150.0],
        trophic_levels=[4.0, 4.0, 4.0],
        ages_dt=[0, 24, 48],
        condition_factor=0.01,
        allometric_power=3.0,
        size_ratio_min=[0.0],
        size_ratio_max=[1.0],
        ingestion_rate=13.0,
        multiplier=1.0,
        offset=0.0,
        forcing_nsteps_year=12,
        proportions=[0.3, 0.4, 0.3],
    )
    beta = 0.8
    imax = fit.background_imax(_EC(), b, beta=beta)
    w_mean = np.mean(
        [b.condition_factor * length**b.allometric_power * 1e-6 for length in b.lengths]
    )

    lhs = imax * (w_mean * 1e6) ** beta * 1e-6  # bioen per-fish, per-substep cap (n_subdt cancels)
    rhs = w_mean * b.ingestion_rate / _EC.n_dt_per_year  # standard per-fish, per-substep cap
    assert lhs == pytest.approx(rhs, rel=1e-12)

    # The 24x hazard, made concrete: without the /n_dt_per_year division this closed form
    # requires, Imax would be exactly n_dt_per_year times too large.
    imax_without_ndt_division = b.ingestion_rate * (w_mean * 1e6) ** (1.0 - beta)
    assert imax_without_ndt_division == pytest.approx(imax * _EC.n_dt_per_year, rel=1e-12)


def test_build_overlay_is_flat_and_carries_every_bioen_key(tmp_path):
    fit = _load_fit_module()
    csv = tmp_path / "baltic_param-bioen.csv"
    csv.write_text("# c\nspecies.bioen.mobilized.tp.sp0;12.5\nspecies.maturity.r.sp0;0.3\n")
    ov = fit.build_overlay(csv, tmp_path / "temp.nc")
    assert ov["module.bioenergetics.enabled"] == "true"
    assert ov["simulation.bioen.phit.enabled"] == "true"
    assert ov["simulation.bioen.fo2.enabled"] == "false"
    assert ov["species.bioen.mobilized.tp.sp0"] == "12.5"
    assert ov["species.maturity.r.sp0"] == "0.3"
    assert ov["temperature.filename"] == str((tmp_path / "temp.nc").resolve())
    assert ov["temperature.varname"] == "temperature"
    assert ov["temperature.nsteps.year"] == "24"
    assert not any(k.startswith("osmose.configuration.") for k in ov)
    assert "temperature.value" not in ov


def test_species_t_opt_zlayer_note_cover_the_same_nine_species():
    """Stale-fixture-sweep pin: the three per-species dicts must agree on their key set, or a
    future species add/rename to data/baltic silently drops a species from one of them."""
    fit = _load_fit_module()
    assert set(fit.SPECIES_T_OPT) == set(fit.SPECIES_ZLAYER) == set(fit.SPECIES_NOTE)
    assert len(fit.SPECIES_T_OPT) == 9
    assert {"cod_west", "cod_east"} <= set(fit.SPECIES_T_OPT)


def test_assert_baltic_pins_fires_on_each_violation():
    """The production `--baltic` run happened to pass every pin on the first try, so without
    this test every branch of _assert_baltic_pins would be a gate that was never shown to be
    able to fail -- this branch's own named recurring defect. Constructs a passing FitResult
    (t_p actually solved for t_opt=10.0, so phi_t/argmax hold) and breaks one field at a time."""
    fit = _load_fit_module()

    tp = solve_tp(10.0, FX)
    ok = FitResult("cod", 4.0, 0.3, 1e12, tp, 10.0, 2.0, 5e3, 5.1e3, 0.6, 400)
    fit._assert_baltic_pins(ok, FX)  # the good case must not raise

    for bad, pattern in (
        (replace(ok, rms_len_pct=20.0), "RMS"),
        (replace(ok, imax=0.0), "Imax"),
        (replace(ok, r=0.0), "r = "),
        (replace(ok, t_opt=25.0), "argmax"),  # t_p no longer solves for the (moved) t_opt
    ):
        with pytest.raises(AssertionError, match=pattern):
            fit._assert_baltic_pins(bad, FX)

    # phi_t(t_p) == 1.0 is NOT independently mutable this way: phi_t(x, ..., peak=x) == 1.0
    # is a tautological identity of phi_t's own formula (any x satisfies it when evaluated at
    # its own declared peak -- same fact test_solve_tp_puts_net_growth_optimum_at_t_opt relies
    # on), so perturbing FitResult.t_p alone cannot make that branch fire; it falls through to
    # the argmax check instead (verified: raises "argmax", not "phi_t"). This matches the
    # spec's own note that "Gate F pins the argmax, not phi_t(T_p) = 1 alone" -- the phi_t
    # pin here is a defensive invariant on phi_t's implementation, not on this function's
    # inputs, and is exercised by phi_t's own tests
    # (tests/test_bioen_offline_fit.py::test_solve_tp_puts_net_growth_optimum_at_t_opt), not
    # here.
    with pytest.raises(AssertionError, match="argmax"):
        fit._assert_baltic_pins(replace(ok, t_p=tp + 5.0), FX)


def _demo_target():
    """A small, self-contained SpeciesTargets -- no config, no engine, no network."""
    import numpy as np

    from osmose.calibration.bioen_offline import SpeciesTargets

    return SpeciesTargets(
        name="demo",
        linf=110.0,
        k=0.15,
        t0=-0.20,
        cf=0.0087,
        b=3.05,
        egg_weight_g=0.001,
        m0=38.0,
        m1=0.0,
        lifespan_years=20.0,
        t_opt=10.0,
        t24=np.full(24, 8.69),
    )


def test_simulate_growth_default_is_bit_identical_without_boost():
    """The juvenile-boost parameters must be a no-op by default.

    The committed C3 overlay (`data/baltic/scenarios/c3_bioen/`) was produced by this function,
    and the published A/B describes those exact numbers. If the defaults ever stopped being a
    no-op the artifact would silently cease to be reproducible from its own generator, so this
    is pinned bit-for-bit rather than approximately.
    """
    import numpy as np

    from osmose.calibration.bioen_offline import (
        BioenFixed,
        c_m_from_share,
        simulate_growth,
        solve_tp,
    )

    fx = BioenFixed()
    tg = _demo_target()
    t_p = solve_tp(tg.t_opt, fx)
    args = (
        13.9,
        1.16,
        t_p,
        c_m_from_share(13.9, t_p, fx),
        tg.t24,
        tg.egg_weight_g,
        24 * 8,
        24,
        tg.cf,
        tg.b,
        tg.m0,
        tg.m1,
        fx,
    )

    base = simulate_growth(*args)
    # explicit defaults, and a multiplier with a zero-width window: both must change nothing
    np.testing.assert_array_equal(base, simulate_growth(*args, 1.0, 0))
    np.testing.assert_array_equal(base, simulate_growth(*args, 5.0, 0))
    # and the boost must actually DO something when its window is open, or the test above is
    # vacuous -- a no-op that is no-op in every configuration proves nothing
    boosted = simulate_growth(*args, 5.0, 24)
    assert boosted[24 * 8] > base[24 * 8] * 1.05


def test_juvenile_boost_fixes_age_one_without_costing_rms():
    """The boost lets the fit BEND the curve, not just slide it.

    Without it the forward model has one allometry `imax*w^beta` for every age, so improving
    year one necessarily distorts the adult curve (measured: age-weighting alone moved cod_west
    to 0.77 at age 1 but pushed RMS 8.33% -> 18.49%). With it, both improve together.
    """
    import numpy as np

    from osmose.calibration.bioen_offline import BioenFixed, fit_species, simulate_growth

    fx = BioenFixed()
    tg = _demo_target()

    def age1_ratio(res):
        w = simulate_growth(
            res.imax,
            res.r,
            res.t_p,
            res.c_m,
            tg.t24,
            tg.egg_weight_g,
            24 * 8,
            24,
            tg.cf,
            tg.b,
            tg.m0,
            tg.m1,
            fx,
            res.juvenile_boost,
            res.boost_thres_dt,
        )
        target = tg.linf * (1.0 - np.exp(-tg.k * (1.0 - tg.t0)))
        return ((w[24] / tg.cf) ** (1.0 / tg.b)) / target

    plain = fit_species(tg, fx)
    boosted = fit_species(tg, fx, juvenile_boost=True)

    assert plain.juvenile_boost == 1.0 and plain.boost_thres_dt == 0
    assert boosted.juvenile_boost > 1.0 and boosted.boost_thres_dt == 24
    # year one goes from badly short to on target ...
    assert age1_ratio(plain) < 0.75
    assert 0.9 < age1_ratio(boosted) < 1.25
    # ... and the overall fit gets BETTER at the same time, not worse
    assert boosted.rms_len_pct < plain.rms_len_pct


def test_juvenile_boost_maps_onto_the_engine_cap_form():
    """`j` must be expressible in the engine's own `imax + (theta-1)*c_rate` form."""
    from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap
    import numpy as np

    imax, j = 7.46, 2.88
    # the documented mapping: c_rate := imax, theta := j
    i_max_all = np.array([imax])
    cap_juv = per_fish_ingestion_cap(
        np.array([1e-5]),
        np.array([0], dtype=np.int32),
        np.array([5], dtype=np.int32),
        i_max_all,
        np.array([0.8]),
        np.array([24], dtype=np.int32),
        np.array([j]),
        np.array([imax]),
        1,
        24,
        1,
    )
    cap_adult = per_fish_ingestion_cap(
        np.array([1e-5]),
        np.array([0], dtype=np.int32),
        np.array([30], dtype=np.int32),
        i_max_all,
        np.array([0.8]),
        np.array([24], dtype=np.int32),
        np.array([j]),
        np.array([imax]),
        1,
        24,
        1,
    )
    assert float(cap_juv[0] / cap_adult[0]) == pytest.approx(j, rel=1e-12)


def test_display_path_handles_a_directory_outside_the_repo():
    """`run_baltic(out_dir=...)` must not crash for a path outside the repo.

    The completion message used `Path.relative_to(ROOT)` unconditionally, which raises
    `ValueError` for an external directory -- AFTER the files were written. So a fully
    successful fit exited non-zero, and it did so on the one workflow that matters for not
    clobbering the committed overlay: regenerate into a scratch dir, diff against the live one.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_fitmod", root / "scripts" / "fit_baltic_bioen_params.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    inside = root / "data" / "baltic" / "scenarios" / "c3_bioen" / "x.csv"
    assert mod._display_path(inside) == "data/baltic/scenarios/c3_bioen/x.csv"

    outside = Path("/tmp/regen_scratch/x.csv")
    assert mod._display_path(outside) == "/tmp/regen_scratch/x.csv"
