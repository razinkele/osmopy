import math
import os

import numba
import pandas as pd
import pytest

from osmose.calibration import larva_recal
from osmose.calibration.larva_recal import (
    RECAL_RATES,
    SP1_STOCKS,
    StockRecal,
    e_clip_first_guess,
    resolved_d0,
    solve_larva_rate,
    solve_per_stock,
    sp1_on_config,
    species_index,
    stock_means,
    stock_means_from_biomass,
    with_determinism,
)
from osmose.config import OsmoseConfigReader

GRID = [0.0, 5.0, 10.0, 15.0]


def test_solve_bisects_to_interior_root():
    # mean decreases with rate: mean(d) = 100 - 4d; baseline 70 -> root at d=7.5.
    # grid f = mean-baseline = [30, 10, -10, -30] -> exactly one crossing in (5, 10).
    r = solve_larva_rate(70.0, lambda d: 100.0 - 4.0 * d, grid_points=GRID, tol=1e-4)
    assert r.feasible and r.converged
    assert r.rate is not None and abs(r.rate - 7.5) < 0.05
    assert abs(r.mean_on - 70.0) / 70.0 <= 1e-4


def test_solve_near_zero_shortcircuit_returns_grid_point():
    # grid includes the exact root 7.5 -> short-circuit, iters=0.
    r = solve_larva_rate(70.0, lambda d: 100.0 - 4.0 * d, grid_points=[0.0, 7.5, 15.0], tol=0.02)
    assert r.feasible and r.converged and r.iters == 0
    assert r.rate == 7.5


def test_solve_d0_already_within_tol_means_no_recalibration():
    # SP1 barely moved the mean: mean(d0=15)=40 == baseline 40 -> near-zero hit at the last
    # grid point -> rate == d0, no recalibration. (mean(0)=100, mean(7.5)=70 are far off.)
    r = solve_larva_rate(40.0, lambda d: 100.0 - 4.0 * d, grid_points=[0.0, 7.5, 15.0], tol=0.02)
    assert r.feasible and r.converged and r.rate == 15.0


def test_solve_infeasible_zero_crossings():
    # every grid mean is far below baseline -> baseline unreachable -> feasible=False.
    r = solve_larva_rate(200.0, lambda d: 50.0 - d, grid_points=GRID, tol=0.02)
    assert not r.feasible and r.rate is None and "0 sign change" in r.message


def test_solve_infeasible_multiple_crossings():
    # f = 30*cos(d), baseline 100: grid values give two sign changes -> ambiguous.
    r = solve_larva_rate(
        100.0, lambda d: 100.0 + 30.0 * math.cos(d), grid_points=[1.0, 2.5, 4.0, 5.5], tol=0.02
    )
    assert not r.feasible and r.rate is None and "2 sign change" in r.message


def test_solve_max_iter_not_converged_reports_best():
    # baseline 71 -> root at d=7.25 (NON-dyadic, so bisection midpoints never hit it exactly);
    # impossibly tight tol + tiny max_iter -> feasible bracket but converged=False.
    r = solve_larva_rate(71.0, lambda d: 100.0 - 4.0 * d, grid_points=GRID, tol=1e-12, max_iter=2)
    assert r.feasible and not r.converged and r.rate is not None
    assert 5.0 < r.rate < 10.0


# ---------------------------------------------------------------------------
# Per-stock sweep (Gauss-Seidel over coupled stocks; engine injected as run_means_on)
# ---------------------------------------------------------------------------


def _coupled_model(rates: dict[str, float | None]) -> dict[str, float]:
    """Two stocks, each linear in its own rate with a weak cross-term. d0 = 10 for both.

    west(dw, de) = 100 - 4*dw + 0.5*(10 - de)   east(dw, de) = 1000 - 40*de + 2*(10 - dw)
    A `None` rate means "key omitted" -> the stock runs at its d0.
    """
    dw = 10.0 if rates.get("cod_west") is None else float(rates["cod_west"])  # type: ignore[arg-type]
    de = 10.0 if rates.get("cod_east") is None else float(rates["cod_east"])  # type: ignore[arg-type]
    return {
        "cod_west": 100.0 - 4.0 * dw + 0.5 * (10.0 - de),
        "cod_east": 1000.0 - 40.0 * de + 2.0 * (10.0 - dw),
    }


def test_solve_per_stock_converges_both_stocks_within_tol():
    # Baselines sit at the uncoupled roots dw=7.5, de=7.5 (plus their small cross-terms).
    baselines = {"cod_west": 70.0, "cod_east": 700.0}
    grids = {"cod_west": [0.0, 5.0, 10.0], "cod_east": [0.0, 5.0, 10.0]}
    r = solve_per_stock(
        baselines, _coupled_model, order=("cod_east", "cod_west"), grids=grids, tol=0.01
    )
    assert r.converged
    assert set(r.rates) == {"cod_west", "cod_east"}
    for name, base in baselines.items():
        assert abs(r.means[name] - base) / base <= 0.01, name
    assert r.sweeps >= 1


def test_solve_per_stock_reports_infeasible_stock_and_omits_its_rate():
    # cod_east's baseline is unreachable (above every grid mean) -> infeasible for that stock;
    # cod_west still solves. converged is False because one stock never reached tol.
    baselines = {"cod_west": 70.0, "cod_east": 5000.0}
    grids = {"cod_west": [0.0, 5.0, 10.0], "cod_east": [0.0, 5.0, 10.0]}
    r = solve_per_stock(
        baselines, _coupled_model, order=("cod_east", "cod_west"), grids=grids, tol=0.01
    )
    assert not r.converged
    assert r.rates["cod_east"] is None
    assert not r.per_stock["cod_east"].feasible
    assert r.rates["cod_west"] is not None
    assert abs(r.means["cod_west"] - 70.0) / 70.0 <= 0.01


def test_solve_per_stock_reuses_identical_evaluations():
    # The joint check after a sweep is the same rate set the last solve ended on; a
    # deterministic evaluator must not be called twice for one rate set.
    calls: list[tuple] = []

    def counted(rates):
        calls.append(tuple(sorted(rates.items())))
        return _coupled_model(rates)

    solve_per_stock(
        {"cod_west": 70.0, "cod_east": 700.0},
        counted,
        order=("cod_east", "cod_west"),
        grids={"cod_west": [0.0, 5.0, 10.0], "cod_east": [0.0, 5.0, 10.0]},
        tol=0.01,
    )
    assert len(calls) == len(set(calls))


SP_FIELD = "data/baltic/forcing/baltic_rv_field.nc"
SPAWN_WEST = "data/baltic/maps/cod_west_spawning.csv"
SPAWN_EAST = "data/baltic/maps/cod_east_spawning.csv"


def test_e_clip_first_guess_bounds():
    for spawn in (SPAWN_WEST, SPAWN_EAST):
        d1, e_clip = e_clip_first_guess(SP_FIELD, spawn, d0=15.0)
        assert 0.0 < e_clip < 1.0  # some but not all viable
        assert 0.0 <= d1 <= 15.0  # a valid rate inside the bracket
        # d1 = d0 + ln(e_clip); ln(e_clip) < 0 so d1 < d0
        assert d1 < 15.0
        assert abs(d1 - max(0.0, 15.0 + math.log(e_clip))) < 1e-9


# ---------------------------------------------------------------------------
# Config helpers: name -> index, resolved d0, overlay assembly
# ---------------------------------------------------------------------------

DET_KEYS = ("movement.randomseed.fixed", "stochastic.mortality.randomseed.fixed")


def _rate_key(i: int) -> str:
    return f"mortality.additional.larva.rate.sp{i}"


def _two_stock_base(d0_west: float = 10.0, d0_east: float = 10.0) -> dict[str, str]:
    """A config fragment with the two cod stocks at non-adjacent indices plus a bystander."""
    return {
        "species.name.sp0": "cod_west",
        "species.name.sp1": "herring",
        "species.name.sp8": "cod_east",
        _rate_key(0): repr(d0_west),
        _rate_key(1): "4.0",
        _rate_key(8): repr(d0_east),
    }


def test_species_index_resolves_by_name_and_none_when_absent():
    base = _two_stock_base()
    assert species_index(base, "cod_west") == 0
    assert species_index(base, "cod_east") == 8
    assert species_index(base, "cod") is None


def test_resolved_d0_reads_live_larva_rate_for_stock():
    base = _two_stock_base(d0_west=10.5, d0_east=9.25)
    assert resolved_d0(base, "cod_west") == 10.5
    assert resolved_d0(base, "cod_east") == 9.25


def test_resolved_d0_raises_for_absent_stock():
    with pytest.raises(KeyError):
        resolved_d0(_two_stock_base(), "cod")


def test_sp1_stocks_are_the_two_cod_stocks():
    assert SP1_STOCKS == ("cod_west", "cod_east")


def test_with_determinism_sets_both_keys_without_mutating():
    base = {"a": "1"}
    out = with_determinism(base)
    assert all(out[k] == "true" for k in DET_KEYS)
    assert "a" in out and base == {"a": "1"}  # original untouched


def test_sp1_on_config_enables_every_sp1_stock_by_name():
    cfg = sp1_on_config(_two_stock_base(), SP_FIELD, larva_rates=None)
    assert cfg["reproduction.rv.spatial.enabled"] == "true"
    assert cfg["reproduction.rv.spatial.field.file"] == SP_FIELD
    assert cfg["reproduction.rv.spatial.species.enabled.sp0"] == "true"
    assert cfg["reproduction.rv.spatial.species.enabled.sp8"] == "true"
    assert "reproduction.rv.spatial.species.enabled.sp1" not in cfg  # herring untouched
    assert all(cfg[k] == "true" for k in DET_KEYS)


def test_sp1_on_config_none_omits_all_rate_keys_and_keeps_base_d0():
    base = _two_stock_base(d0_west=10.0, d0_east=9.0)
    cfg = sp1_on_config(base, SP_FIELD, larva_rates=None)
    assert cfg[_rate_key(0)] == "10.0" and cfg[_rate_key(8)] == "9.0"  # infeasible path


def test_sp1_on_config_explicit_mapping_sets_rate_keys_per_stock():
    base = _two_stock_base()
    cfg = sp1_on_config(base, SP_FIELD, larva_rates={"cod_west": 8.0, "cod_east": None})
    assert float(cfg[_rate_key(0)]) == 8.0
    assert cfg[_rate_key(8)] == base[_rate_key(8)]  # None -> this stock's d0 stands
    assert cfg[_rate_key(1)] == "4.0"  # bystander untouched


def test_sp1_on_config_default_reads_module_rates(monkeypatch):
    base = _two_stock_base(d0_west=10.0, d0_east=10.0)
    monkeypatch.setattr(
        larva_recal,
        "RECAL_RATES",
        {"cod_west": StockRecal(rate=8.0, d0=10.0), "cod_east": StockRecal(rate=9.5, d0=10.0)},
    )
    cfg = sp1_on_config(base, SP_FIELD)
    assert float(cfg[_rate_key(0)]) == 8.0
    assert float(cfg[_rate_key(8)]) == 9.5
    monkeypatch.setattr(larva_recal, "RECAL_RATES", {})
    cfg2 = sp1_on_config(base, SP_FIELD)
    assert cfg2[_rate_key(0)] == "10.0" and cfg2[_rate_key(8)] == "10.0"


def test_sp1_on_config_raises_when_live_d0_drifted_from_solved(monkeypatch):
    # The 2026-07 trap: the rate was solved against d0=15, the baseline was later
    # recalibrated to ~10.16, and the frozen constant silently became an INCREASE.
    base = _two_stock_base(d0_west=10.15686139, d0_east=10.15686139)
    monkeypatch.setattr(larva_recal, "RECAL_RATES", {"cod_west": StockRecal(rate=14.655, d0=15.0)})
    with pytest.raises(ValueError, match="cod_west.*15\\.0.*10\\.157"):
        sp1_on_config(base, SP_FIELD)


def test_sp1_on_config_default_skips_stocks_absent_from_config(monkeypatch):
    # A solved entry for a stock the config does not declare is ignored, not an error.
    base = {"species.name.sp0": "cod", _rate_key(0): "15.0"}
    monkeypatch.setattr(larva_recal, "RECAL_RATES", {"cod_west": StockRecal(rate=8.0, d0=10.0)})
    cfg = sp1_on_config(base, SP_FIELD)
    assert cfg[_rate_key(0)] == "15.0"
    assert "reproduction.rv.spatial.species.enabled.sp0" not in cfg


# ---------------------------------------------------------------------------
# Means: pure window helper on a biomass frame, then the engine-backed wrapper
# ---------------------------------------------------------------------------


def _frame(**cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_stock_means_from_biomass_windows_years_3_to_14_finite_positive():
    years = list(range(20))
    west = [float(y) for y in years]  # mean over [3:15] = mean(3..14) = 8.5
    east = [100.0] * 20
    east[5] = float("nan")  # dropped
    east[7] = 0.0  # dropped (>0 filter)
    m = stock_means_from_biomass(_frame(cod_west=west, cod_east=east))
    assert m["cod_west"] == pytest.approx(8.5)
    assert m["cod_east"] == pytest.approx(100.0)


def test_stock_means_from_biomass_windows_by_time_column_when_recorded_per_step():
    # 24 records per year: rows with 3 <= Time < 15 are the window, not row index [3:15].
    n = 24 * 20
    time = [i / 24 for i in range(n)]
    west = [1.0 if 3.0 <= t < 15.0 else 1000.0 for t in time]
    m = stock_means_from_biomass(_frame(Time=time, cod_west=west, species=["all"] * n))
    assert m == {"cod_west": pytest.approx(1.0)}  # Time/species columns are not stocks


def test_mean_cod_from_biomass_sums_the_two_stocks_and_falls_back_to_aggregate():
    split = _frame(cod_west=[1.0] * 20, cod_east=[2.0] * 20)
    assert larva_recal.mean_cod_from_biomass(split) == pytest.approx(3.0)
    agg = _frame(cod=[5.0] * 20)
    assert larva_recal.mean_cod_from_biomass(agg) == pytest.approx(5.0)


def test_stock_overshoot_from_biomass_is_window_max_over_mean():
    west = [1.0] * 20
    west[10] = 13.0  # inside the window: mean over 3..14 = (11*1 + 13)/12 = 2.0, max 13
    west[0] = 1000.0  # outside the window, ignored
    o = larva_recal.stock_overshoot_from_biomass(_frame(cod_west=west))
    assert o["cod_west"] == pytest.approx(13.0 / 2.0)


MINIMAL = "data/minimal/osm_all-parameters.csv"


def test_stock_means_runs_engine_and_returns_every_focal_column(numba_warmup):
    cfg = dict(OsmoseConfigReader().read(MINIMAL))
    cfg["simulation.time.nyear"] = "5"
    m = stock_means(with_determinism(cfg))
    assert set(m) == {"Anchovy", "Hake"}
    assert all(math.isfinite(v) and v > 0 for v in m.values())


BALTIC = "data/baltic/baltic_all-parameters.csv"


def _baltic_15yr():
    cfg = dict(OsmoseConfigReader().read(BALTIC))
    cfg["simulation.time.nyear"] = "15"
    return cfg


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="RECAL_RATES are solved on the maintainer's host; the 15-yr Baltic sim is not "
    "bit-reproducible across dependency/hardware environments, so the frozen rates only hit "
    "mean-neutrality where they were solved. The fast solver unit tests cover the mechanism.",
)
def test_sp1b_mean_neutral_drift_guard():
    active = {n: e for n, e in RECAL_RATES.items() if e.rate is not None}
    if not active:
        pytest.skip("SP1b: no solved per-stock rate (see docs/diagnostics/sp1b_recalibration.md)")
    numba.set_num_threads(1)  # runtime determinism pin (config keys added by the helpers)
    base = _baltic_15yr()
    off = stock_means(with_determinism(base))
    on = stock_means(sp1_on_config(base, SP_FIELD))  # default -> RECAL_RATES, d0-checked
    for name in active:
        assert name in off, f"{name} is not a focal species of the config"
        assert abs(on[name] - off[name]) / off[name] <= 0.02, name
