#!/usr/bin/env python
"""F1 historical-fishing hindcast (spec 2026-08-23, Stage 1 of B1). Two arms x 5
seeds x 50 yr on the certified Baltic config: A = constant F, B = by-year ICES F
(4 stocks). Sim-year 19 = 1993. Scores herring+sprat (pass/fail, decision 7
margins); cod_west/cod_east/flounder reported-only. NOT a CI gate (emergent).

Instrument check (blocking for herring/sprat/cod_east): arm B realized
yield-per-biomass must rank-correlate with the imposed factor pattern.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data/baltic/reference/ices_snapshots"
YEARS = list(range(1993, 2024))
SPINUP = 19
N_YEAR = 50
SEEDS = (42, 123, 7, 999, 2024)
FORCED = (0, 1, 2, 8)
SPECIES = {0: "cod_west", 1: "herring", 2: "sprat", 3: "flounder", 8: "cod_east"}
SCORED = ("herring", "sprat")
BLOCKING_INSTRUMENT = ("herring", "sprat", "cod_east")
DECADES = ((1993, 2002), (2003, 2012), (2013, 2023))
HERRING_STOCKS = ["her.27.25-2932", "her.27.28", "her.27.3031", "her.27.20-24"]
OBS_STOCK = {"cod_west": "cod.27.22-24", "sprat": "spr.27.22-32",
             "flounder": "fle.27.2223", "cod_east": "cod.27.24-32"}


def annualize(x, n_year: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == n_year:
        return x
    if len(x) % n_year == 0:
        return x.reshape(n_year, -1).mean(axis=1)
    raise ValueError(f"series of {len(x)} not divisible into {n_year} years")


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() <= 2 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def decadal_trend_signs(values, years) -> list[int]:
    v, y = np.asarray(values, float), np.asarray(years)
    out = []
    for lo, hi in DECADES:
        mask = (y >= lo) & (y <= hi) & ~np.isnan(v)
        slope = np.polyfit(y[mask], v[mask], 1)[0]
        out.append(1 if slope > 0 else -1)
    return out


def skill_verdict(dr_per_seed) -> dict:
    dr = np.asarray(dr_per_seed, float)
    mean, sd = float(np.nanmean(dr)), float(np.nanstd(dr, ddof=1))
    return {"mean_dr": mean, "sd_dr": sd, "passes": bool(mean >= 0.10 and mean > 2 * sd)}


def _ssb_series(snap_dir: Path, stock_key: str, years) -> np.ndarray:
    recs = json.loads((snap_dir / f"{stock_key}.assessment.json").read_text())
    by_year = {int(r["year"]): float(r["ssb"]) for r in recs if r.get("ssb") not in ("", None)}
    return np.array([by_year.get(y, np.nan) for y in years], dtype=float)


def observed_stock_z(snap_dir: Path, stock_key: str, years) -> np.ndarray:
    return zscore(_ssb_series(snap_dir, stock_key, years))


def observed_herring_z(snap_dir: Path, years) -> np.ndarray:
    """Decision 6: fixed-weight mean of per-stock z-scores; weights = mean catch
    share over the window."""
    zs, weights = [], []
    for key in HERRING_STOCKS:
        recs = json.loads((snap_dir / f"{key}.assessment.json").read_text())
        catches = {int(r["year"]): float(r["catches"])
                   for r in recs if r.get("catches") not in ("", None)}
        w = np.nanmean([catches.get(y, np.nan) for y in years])
        zs.append(observed_stock_z(snap_dir, key, years))
        weights.append(0.0 if np.isnan(w) else w)
    z, w = np.stack(zs), np.asarray(weights, float)
    wsum = np.where(np.isnan(z), 0.0, w[:, None]).sum(axis=0)
    return np.nansum(z * w[:, None], axis=0) / np.where(wsum == 0, np.nan, wsum)


def _spearman(a, b) -> float:
    from scipy.stats import rankdata  # scipy is in the venv (SALib dependency)

    return pearson(rankdata(a), rankdata(b))


def instrument_check(factors, yields, biomass) -> float:
    """Rank corr between the imposed factor pattern and realized yield-per-biomass
    over 1993-2023. Wrong-mapping / silent-no-op canary."""
    ypb = np.asarray(yields, float) / np.maximum(np.asarray(biomass, float), 1e-9)
    return _spearman(np.asarray(factors, float), ypb)


def load_factors(sp_idx: int) -> np.ndarray:
    arr = np.loadtxt(ROOT / f"data/baltic/reference/f_byyear_sp{sp_idx}.csv")
    return arr[SPINUP:] / arr[0]  # scaled rows / base F = factor series


def arm_overrides(mode: str) -> dict:
    base = {"simulation.time.nyear": str(N_YEAR), "output.ssb.enabled": "true"}
    if mode == "fhist":
        for i in FORCED:
            base[f"mortality.fishing.rate.byyear.file.sp{i}"] = str(
                ROOT / f"data/baltic/reference/f_byyear_sp{i}.csv"
            )
    return base


def run_hindcast(seeds=SEEDS) -> dict:
    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    tmp = Path(tempfile.mkdtemp())
    base_cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))

    # Spec §3 startup assertion: spin-up rows must equal the live base F bit-exactly,
    # so the two arms share the 1974-1992 pre-period.
    for i in FORCED:
        arr = np.loadtxt(ROOT / f"data/baltic/reference/f_byyear_sp{i}.csv")
        base_f = float(base_cfg[f"fisheries.rate.base.fsh{i}"])
        assert (arr[:SPINUP] == base_f).all(), (
            f"sp{i}: spin-up rows != base F {base_f}; regenerate the CSVs "
            "(scripts/build_baltic_f_byyear.py) after any recalibration"
        )

    ssb: dict[str, dict[str, list[np.ndarray]]] = {a: {} for a in ("A", "B")}
    ypb_inputs: dict[str, list] = {}
    for seed in seeds:
        for arm, mode in (("A", "base"), ("B", "fhist")):
            raw = {**base_cfg, **arm_overrides(mode)}
            res = PythonEngine().run_in_memory(raw, seed=seed)
            ssb_df, yld_df, bio_df = res.ssb(), res.yield_biomass(), res.biomass()
            for name in SPECIES.values():
                series = annualize(ssb_df[name].to_numpy(dtype=float), N_YEAR)[SPINUP:]
                ssb[arm].setdefault(name, []).append(series)
                if arm == "B":
                    # spec §3: yield-per-BIOMASS (not SSB) — selectivity admits
                    # pre-mature fish, so the denominators differ.
                    yld = annualize(yld_df[name].to_numpy(dtype=float), N_YEAR)[SPINUP:]
                    bio = annualize(bio_df[name].to_numpy(dtype=float), N_YEAR)[SPINUP:]
                    ypb_inputs.setdefault(name, []).append((yld, bio))

    obs = {"herring": observed_herring_z(SNAP, YEARS)}
    for name, key in OBS_STOCK.items():
        obs[name] = observed_stock_z(SNAP, key, YEARS)

    report: dict = {"stocks": {}, "instrument": {}}
    for sp_idx in FORCED:
        name = SPECIES[sp_idx]
        factors = load_factors(sp_idx)
        rhos = [instrument_check(factors, y, b) for y, b in ypb_inputs[name]]
        report["instrument"][name] = {
            "rho_per_seed": rhos,
            "blocking": name in BLOCKING_INSTRUMENT,
        }
    for name in SPECIES.values():
        a_runs, b_runs = np.stack(ssb["A"][name]), np.stack(ssb["B"][name])
        dr = [pearson(zscore(b), obs[name]) - pearson(zscore(a), obs[name])
              for a, b in zip(a_runs, b_runs)]
        report["stocks"][name] = {
            "scored": name in SCORED,
            "trend_model_B": decadal_trend_signs(b_runs.mean(axis=0), YEARS),
            "trend_observed": decadal_trend_signs(obs[name], YEARS),
            "skill": skill_verdict(dr),
            "r_A_mean": pearson(zscore(a_runs.mean(axis=0)), obs[name]),
            "r_B_mean": pearson(zscore(b_runs.mean(axis=0)), obs[name]),
            "ssb_A_mean": a_runs.mean(axis=0).tolist(),
            "ssb_B_mean": b_runs.mean(axis=0).tolist(),
            "obs_z": np.asarray(obs[name], float).tolist(),
        }
    return report


REPORT_PATH = Path("/tmp/f1_hindcast_report.json")

if __name__ == "__main__":
    out = run_hindcast()
    REPORT_PATH.write_text(json.dumps(out, indent=2, default=float))
    print(f"report written to {REPORT_PATH}")
