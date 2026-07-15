#!/usr/bin/env python
"""SPIKE (throwaway): does driving the Baltic model with the historical annual F trajectory
make modeled SSB track ICES SSB better than constant F? Skill-delta framing, same as the RV
hindcast (PR #109), but F-forced (a DIFFERENT, untested lever -- RV held F constant).

Two arms over Y0..Y0+N-1:
  null  = current config (constant base F for every species)
  fdriven = cod(sp0)+sprat(sp2) F set to a per-year RELATIVE trajectory:
            F_model[t] = base_F * ICES_F[year] / mean(ICES_F[2018..2022])
            (relative -> keeps the model in its calibrated regime; isolates trajectory SHAPE).
Metric per species: corr(SSB_fdriven, ices_SSB) - corr(SSB_null, ices_SSB) over a post-spinup
window. Correlation is scale-invariant, so cod's index-scale ICES SSB is fine alongside sprat's
tonnes. GO if skill delta clearly positive for >=1 species; NO-GO/reframe if RV-style ~0.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data/baltic/reference/ices_snapshots"
Y0, N_YEAR = 1993, 30  # 1993-2022
REF = range(2018, 2023)  # scaling reference window (model calibrated here)
SEEDS = (0, 1, 2, 3, 4)

# model species index -> (ICES stock, base F). cod=sp0, sprat=sp2.
SPECIES = {
    "cod": {"sp": 0, "stock": "cod.27.24-32", "base_f": 0.08},
    "sprat": {"sp": 2, "stock": "spr.27.22-32", "base_f": 0.32},
}


def _series(stock: str, field: str) -> dict[int, float]:
    d = json.load(open(SNAP / f"{stock}.assessment.json"))
    out = {}
    for r in d:
        v = r.get(field)
        try:
            out[int(r["year"])] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def build_byyear_f(info: dict) -> np.ndarray | None:
    f = _series(info["stock"], "f")
    ref_vals = [f[y] for y in REF if y in f]
    if not ref_vals:
        return None
    ref_mean = float(np.mean(ref_vals))
    years = [Y0 + t for t in range(N_YEAR)]
    # fall back to base F (ratio 1.0) for any missing year
    return np.array([info["base_f"] * (f.get(y, ref_mean) / ref_mean) for y in years])


def ices_ssb_window(stock: str) -> np.ndarray:
    s = _series(stock, "ssb")
    return np.array([s.get(Y0 + t, np.nan) for t in range(N_YEAR)])


def run(fdriven: bool, seed: int, byyear: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    tmp = Path(tempfile.mkdtemp())
    base = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    raw = {**base, "simulation.time.nyear": str(N_YEAR), "output.ssb.enabled": "true"}
    if fdriven:
        for sp_idx, arr in byyear.items():
            f_csv = tmp / f"f_byyear_sp{sp_idx}.csv"
            np.savetxt(f_csv, arr)
            raw[f"mortality.fishing.rate.byyear.file.sp{sp_idx}"] = str(f_csv)
    res = PythonEngine().run_in_memory(raw, seed=seed).ssb()
    return {name: res[name].to_numpy(dtype=float) for name in SPECIES}


def corr(m: np.ndarray, o: np.ndarray, win: slice) -> float:
    m, o = np.asarray(m, float)[win], np.asarray(o, float)[win]
    ok = np.isfinite(m) & np.isfinite(o)
    if ok.sum() <= 2 or np.std(m[ok]) == 0 or np.std(o[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(m[ok], o[ok])[0, 1])


if __name__ == "__main__":
    byyear = {info["sp"]: build_byyear_f(info) for info in SPECIES.values()}
    print("=== scaled byyear-F (relative trajectory, model base preserved) ===")
    for name, info in SPECIES.items():
        arr = byyear[info["sp"]]
        print(
            f"{name} (sp{info['sp']}, base={info['base_f']}): "
            f"min={arr.min():.3f} max={arr.max():.3f} "
            f"[{Y0}]={arr[0]:.3f} [2020]={arr[2020 - Y0]:.3f}"
        )
    obs = {name: ices_ssb_window(info["stock"]) for name, info in SPECIES.items()}

    null_runs, fdrv_runs = {n: [] for n in SPECIES}, {n: [] for n in SPECIES}
    for seed in SEEDS:
        n = run(False, seed, byyear)
        f = run(True, seed, byyear)
        for name in SPECIES:
            null_runs[name].append(n[name])
            fdrv_runs[name].append(f[name])
        print(f"  seed {seed} done")

    print("\n=== RESULTS (skill delta = corr(fdriven,ices) - corr(null,ices)) ===")
    for win_name, win in [("yr4-29", slice(4, 30)), ("yr4-15", slice(4, 16))]:
        print(f"\n-- window {win_name} --")
        for name in SPECIES:
            o = obs[name]
            cn = [corr(m, o, win) for m in null_runs[name]]
            cf = [corr(m, o, win) for m in fdrv_runs[name]]
            deltas = np.array(cf) - np.array(cn)
            print(
                f"{name:6s}: corr_null={np.nanmean(cn):+.3f}  corr_fdriven={np.nanmean(cf):+.3f}"
                f"  skill_delta={np.nanmean(deltas):+.3f} +/-{np.nanstd(deltas):.3f}"
            )

    # dump mean trajectories for eyeballing
    print("\n=== mean SSB trajectories (for eyeball) ===")
    for name in SPECIES:
        mn = np.nanmean(np.stack(null_runs[name]), axis=0)
        mf = np.nanmean(np.stack(fdrv_runs[name]), axis=0)
        print(f"{name} ices  :", np.round(obs[name], 2))
        print(f"{name} null  :", np.round(mn, 1))
        print(f"{name} fdrivn:", np.round(mf, 1))
