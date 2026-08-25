import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "fit_codwest_thermal_sr",
    Path(__file__).resolve().parent.parent / "scripts" / "fit_codwest_thermal_sr.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _synthetic(beta1, n=29, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n)
    temp = 16.0 + rng.normal(0, 1.0, n)
    ssb = np.exp(rng.normal(9.0, 0.3, n))
    b0, b3 = -1.0, 1e-4
    ln_r = -b0 + beta1 * temp + np.log(ssb) - np.log1p(b3 * ssb) + rng.normal(0, 0.1, n)
    return np.exp(ln_r), ssb, temp, years


def test_fit_recovers_known_beta():
    r, ssb, temp, _ = _synthetic(beta1=-0.4)
    fit = m.fit_bh_exp(r, ssb, temp)
    assert abs(fit["beta1"] - (-0.4)) < 0.05
    assert fit["p"] < 0.01


def test_paired_data_applies_age1_lag():
    recs = [
        {"year": "2000", "ssb": "100.0", "recruitment": "555.0"},
        {"year": "2001", "ssb": "110.0", "recruitment": "666.0"},
    ]
    r, ssb, t = m.paired_data(recs, {2000: 15.0}, range(2000, 2001))
    assert r[0] == 666.0 and ssb[0] == 100.0 and t[0] == 15.0  # R_{y+1} <- SSB_y, T_y


def test_detrend_kills_trend_only_signal():
    rng = np.random.default_rng(1)
    n = 29
    years = np.arange(1993, 1993 + n)
    temp = 14.0 + 0.05 * (years - 1993) + rng.normal(0, 0.15, n)  # strong trend
    ssb = np.exp(rng.normal(9.0, 0.3, n))
    ln_r = np.log(ssb) - 0.02 * (years - 1993) + rng.normal(0, 0.1, n)  # trend, NOT T-driven
    fit = m.fit_bh_exp(np.exp(ln_r), ssb, temp)
    fit_d = m.fit_bh_exp(np.exp(ln_r), ssb, m.detrended(temp, years))
    v = m.verdict(fit, fit_d)
    assert isinstance(v["enabled"], bool)  # runs end-to-end; and on this fixture:
    assert abs(fit_d["beta1"]) < abs(fit["beta1"])  # detrending shrinks the spurious signal
