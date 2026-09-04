"""Offline Java-form bioenergetics growth model for parameter fitting (C3 spec §3.4).

Per fish, grams, one 24-step year cycle. Same equations as EnergyBudget.java / the parity-fixed
engine with N = 1 and ingestion at the cap (food-unlimited). Used to (a) solve T_p so that the
net-growth optimum equals a cited growth optimum, (b) fit (Imax, r) to a config's own vBGF curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from osmose.engine.processes.temp_function import arrhenius, phi_t


@dataclass(frozen=True)
class BioenFixed:
    a: float = 0.7
    beta: float = 0.8
    eta: float = 1.0
    e_m: float = 0.65
    e_d: float = 1.5
    e_maint: float = 0.65
    m_share: float = 0.3  # maintenance / maximal ingestion at t_ref (Bernreuther et al. 2012, 16 C)
    t_ref: float = 16.0
    larval_coef: float = 1.0
    larvae_thres_dt: int = 1
    first_feeding_dt: int = 1


@dataclass
class SpeciesTargets:
    name: str
    linf: float
    k: float
    t0: float
    cf: float
    b: float
    egg_weight_g: float
    m0: float
    m1: float
    lifespan_years: float
    t_opt: float
    t24: NDArray[np.float64]


@dataclass
class FitResult:
    name: str
    imax: float
    r: float
    c_m: float
    t_p: float
    t_opt: float
    rms_len_pct: float
    w_inf_fit_g: float
    w_inf_vb_g: float
    larval_ratio_half_year: float
    n_points: int


def c_m_from_share(imax: float, t_p: float, fx: BioenFixed) -> float:
    tr = np.array(fx.t_ref)
    return (
        fx.m_share
        * fx.a
        * imax
        * float(phi_t(tr, fx.e_m, fx.e_d, t_p))
        / float(arrhenius(tr, fx.e_maint))
    )


def g_net(T, imax: float, c_m: float, t_p: float, fx: BioenFixed):
    T = np.asarray(T, dtype=np.float64)
    return fx.a * phi_t(T, fx.e_m, fx.e_d, t_p) * imax - c_m * arrhenius(T, fx.e_maint)


def _dg_dT_at(t_opt: float, t_p: float, fx: BioenFixed, h: float = 1e-3) -> float:
    cm = c_m_from_share(1.0, t_p, fx)
    return float(g_net(t_opt + h, 1.0, cm, t_p, fx) - g_net(t_opt - h, 1.0, cm, t_p, fx)) / (2 * h)


def solve_tp(t_opt: float, fx: BioenFixed) -> float:
    """T_p such that argmax_T g_net(T) == t_opt (Imax cancels; depends on m, e_*, t_ref only)."""
    lo, hi = t_opt, t_opt + 25.0
    assert _dg_dT_at(t_opt, lo, fx) < 0.0, (
        "at t_p == t_opt the slope must be negative (maintenance)"
    )
    return float(brentq(lambda tp: _dg_dT_at(t_opt, tp, fx), lo, hi, xtol=1e-6))


def vbgf_weight(age_years, linf, k, t0, cf, b):
    length = linf * (1.0 - np.exp(-k * (np.asarray(age_years, dtype=np.float64) - t0)))
    return cf * np.power(np.maximum(length, 1e-9), b)


def simulate_growth(imax, r, t_p, c_m, t24, w_egg_g, n_steps, ndt, cf, b, m0, m1, fx: BioenFixed):
    """Per-fish weight (g) at step index 0..n_steps; step index == age_dt. Java budget with N = 1."""
    t24 = np.asarray(t24, dtype=np.float64)
    w = np.empty(n_steps + 1)
    w[0] = w_egg_g
    faced = 0.0
    for age_dt in range(1, n_steps + 1):
        wg = w[age_dt - 1]
        T = t24[(age_dt - 1) % ndt]
        wb = wg**fx.beta
        if age_dt < fx.first_feeding_dt:
            w[age_dt] = wg
            continue
        e_gross = fx.a * float(phi_t(np.array(T), fx.e_m, fx.e_d, t_p)) * imax * wb / ndt
        e_maint = c_m * float(arrhenius(np.array(T), fx.e_maint)) * wb / ndt
        e_net = e_gross - e_maint
        per_fish = e_net * ndt / wb
        if age_dt == fx.first_feeding_dt:
            faced = per_fish / fx.larval_coef
        elif age_dt < fx.larvae_thres_dt:
            faced = (per_fish / fx.larval_coef + faced * age_dt) / (age_dt + 1)
        else:
            faced = (per_fish + faced * age_dt) / (age_dt + 1)
        length = (wg / cf) ** (1.0 / b)
        mature = length >= m0 + m1 * (age_dt / ndt)
        if mature:
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = r * wg ** (1.0 - fx.beta) / (fx.eta * faced)
            rho = 1.0 if np.isnan(rho) else float(np.clip(rho, 0.0, 1.0))
        else:
            rho = 0.0
        w[age_dt] = wg + (1.0 - rho) * max(e_net, 0.0)
    return w


def fit_species(tg: SpeciesTargets, fx: BioenFixed, ndt: int = 24) -> FitResult:
    t_p = solve_tp(tg.t_opt, fx)
    n_steps = int(round(tg.lifespan_years * ndt))
    idx_all = np.arange(ndt, n_steps + 1)  # ages >= 1 yr (spec 3.4: larval phase not fitted)
    ages_all = idx_all / ndt
    # A vBGF fitted to a growth curve that is convex (not vBGF-shaped) at the youngest fitted
    # ages can land its own t0 above those ages, predicting a negative length there.
    # `vbgf_weight`'s `np.maximum(L, 1e-9)` clamp would turn that into a target weight of
    # ~cf*1e-9**b (~1e-30), which no (imax, r) can match in log-space -- the optimizer would
    # then trade away the whole fit chasing a fabricated near-zero point. Drop those ages
    # instead of clamping through them.
    length_all = tg.linf * (1.0 - np.exp(-tg.k * (ages_all - tg.t0)))
    valid = length_all > 0.0
    idx = idx_all[valid]
    ages = ages_all[valid]
    target_w = vbgf_weight(ages, tg.linf, tg.k, tg.t0, tg.cf, tg.b)

    def resid(x):
        imax, r = np.exp(x)
        w = simulate_growth(
            imax,
            r,
            t_p,
            c_m_from_share(imax, t_p, fx),
            tg.t24,
            tg.egg_weight_g,
            n_steps,
            ndt,
            tg.cf,
            tg.b,
            tg.m0,
            tg.m1,
            fx,
        )
        return np.log(np.maximum(w[idx], 1e-12)) - np.log(target_w)

    # `loss="soft_l1"` (scipy's default f_scale=1.0): the youngest valid points can still sit
    # close enough to the vBGF's own zero-crossing (small but not clamped) to dominate a plain
    # sum-of-squares in log-space out of proportion to the information they carry -- a robust
    # loss down-weights that handful of outlier residuals instead of letting them drag the
    # whole fit toward them, with no effect on a well-conditioned fit (residuals all small).
    sol = least_squares(
        resid,
        x0=np.log([5.0, 0.3]),
        bounds=(np.log([1e-3, 1e-4]), np.log([1e3, 1e2])),
        loss="soft_l1",
    )
    imax, r = np.exp(sol.x)
    c_m = c_m_from_share(imax, t_p, fx)
    w = simulate_growth(
        imax, r, t_p, c_m, tg.t24, tg.egg_weight_g, n_steps, ndt, tg.cf, tg.b, tg.m0, tg.m1, fx
    )
    len_model = (w[idx] / tg.cf) ** (1 / tg.b)
    len_target = (target_w / tg.cf) ** (1 / tg.b)
    rms = float(np.sqrt(np.mean(((len_model - len_target) / len_target) ** 2)) * 100)
    gbar = float(np.mean(g_net(tg.t24, imax, c_m, t_p, fx)))
    w_inf_fit = (fx.eta * gbar / r) ** (1.0 / (1.0 - fx.beta))
    half = ndt // 2
    # Same guard as above: at t0 >= 0.5 yr the vBGF predicts a non-positive length at 0.5 yr,
    # and raising a negative base to a non-integer power (**tg.b) is undefined -- report the
    # ratio as nan explicitly rather than let numpy warn and produce a meaningless negative-base
    # power. Callers (bioen_param_lines) must not silently ship a nan; see Task 8 report.
    half_len = tg.linf * (1.0 - np.exp(-tg.k * (0.5 - tg.t0)))
    w_half_target = tg.cf * half_len**tg.b if half_len > 0.0 else np.nan
    return FitResult(
        tg.name,
        float(imax),
        float(r),
        float(c_m),
        t_p,
        tg.t_opt,
        rms,
        float(w_inf_fit),
        float(tg.cf * tg.linf**tg.b),
        float(w[half] / w_half_target),
        int(idx.size),
    )


def _fmt(x: float) -> str:
    """Round-trip-exact float formatting for the config writer (Gate F pin).

    ``repr()`` for "normal"-magnitude values (matches what the reader/writer round-trip
    elsewhere in the repo). Fitted ``c_m`` can land at either end of a wide dynamic range
    (Task 3: ``e_maint = c_m*(w*1e6)^beta*Arr(T)/ndt * N * 1e-6``, and ``w`` spans tonnes
    down to ~1e-6) -- ``repr()`` alone would write those as a wall of zeros instead of
    scientific notation, so very large/small magnitudes go through
    ``np.format_float_scientific(unique=True)`` instead, which is still exact round-trip.
    """
    x = float(x)
    ax = abs(x)
    if x != 0.0 and (ax >= 1e5 or ax < 1e-3):
        return np.format_float_scientific(x, unique=True, trim="-", exp_digits=0)
    return repr(x)


def bioen_param_lines(results, fx: BioenFixed, zlayer, sp_index, background_imax, notes, m0):
    """Full Java 4.3.3 bioen key inventory (canonical 4.4.0 spellings), one block per species.

    m0: {species name: maturity length (cm)} -- the config's species.maturity.size values."""
    out = [
        "# Generated by scripts/fit_baltic_bioen_params.py -- DO NOT EDIT BY HAND (C3 spec 3.4)",
        f"# fixed: a={fx.a} beta={fx.beta} eta={fx.eta} e_M={fx.e_m} e_D={fx.e_d} e_maint={fx.e_maint} eV "
        f"(engine defaults); m={fx.m_share} at {fx.t_ref} C (Bernreuther et al. 2012)",
    ]
    for res in results:
        i = sp_index[res.name]
        out += [
            f"# --- {res.name} (sp{i}): {notes.get(res.name, '')}; growth optimum {res.t_opt} C -> engine T_p "
            f"{res.t_p:.4f} C; fit RMS length {res.rms_len_pct:.1f}% (ages >= 1 yr); W_inf fit "
            f"{res.w_inf_fit_g:.0f} g vs vBGF {res.w_inf_vb_g:.0f} g; larval ratio at 0.5 yr "
            f"{res.larval_ratio_half_year:.2f}",
            f"species.beta.sp{i};{_fmt(fx.beta)}",
            f"species.zlayer.sp{i};{zlayer[res.name]}",
            f"species.bioen.assimilation.sp{i};{_fmt(fx.a)}",
            f"species.bioen.maint.energy.c_m.sp{i};{_fmt(res.c_m)}",
            f"species.bioen.maint.e.maint.sp{i};{_fmt(fx.e_maint)}",
            f"species.bioen.mobilized.e.mobi.sp{i};{_fmt(fx.e_m)}",
            f"species.bioen.mobilized.e.d.sp{i};{_fmt(fx.e_d)}",
            f"species.bioen.mobilized.tp.sp{i};{_fmt(res.t_p)}",
            f"species.maturity.eta.sp{i};{_fmt(fx.eta)}",
            f"species.maturity.r.sp{i};{_fmt(res.r)}",
            f"species.maturity.m0.sp{i};{_fmt(m0[res.name])}",
            f"species.maturity.m1.sp{i};0.0",
            f"predation.ingestion.rate.max.sp{i};{_fmt(res.imax)}",
            f"predation.larval.ingestion.rate.increase.ratio.sp{i};{_fmt(fx.larval_coef)}",
            f"predation.c.bioen.sp{i};0.0",
            f"species.oxygen.c1.sp{i};1.0",
            f"species.oxygen.c2.sp{i};60.0",
            f"species.bioen.forage.k_for.sp{i};0.0",
            f"species.bioen.forage.k1_for.sp{i};0.0",
            f"species.bioen.forage.k2_for.sp{i};0.0",
        ]
    for b_idx, imax in sorted(background_imax.items()):
        out += [
            f"# background predator sp{b_idx}: Imax in bioen units = standard rate * mean-weight factor",
            f"predation.ingestion.rate.max.sp{b_idx};{_fmt(imax)}",
            f"predation.larval.ingestion.rate.increase.ratio.sp{b_idx};1.0",
            f"predation.c.bioen.sp{b_idx};0.0",
        ]
    return out
