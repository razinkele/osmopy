#!/usr/bin/env python
"""Phase 3: A/B reproductive-volume hindcast. Three arms over 1993-2021 (nyear=29):
off (no RV mechanism), clim (stationary climatology), inter (real interannual) — the two
enabled arms share one forced RV_ref so the A/B isolates temporal structure. Scores modeled
cod SSB vs observed ICES SSB (skill delta). NOT a CI gate (emergent)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTER = ROOT / "data/baltic_rv/baltic_rv_field_interannual.nc"
CLIM = ROOT / "data/baltic/forcing/baltic_rv_field.nc"
N_YEAR = 29  # 1993-2021
WINDOW = slice(6, 16)  # usable window sim-yr 6-15 (~1999-2008); intrinsic collapse dominates later


def arm_overrides(mode: str, rv_ref: float, inter_path: str, clim_path: str) -> dict:
    # output.ssb.enabled in the SHARED base so ALL arms (incl. "off") emit a real maturity-based
    # SSB — otherwise the "off" arm's in-memory results have no "SSB" entry and .ssb() raises.
    base = {"simulation.time.nyear": str(N_YEAR), "output.ssb.enabled": "true"}
    if mode == "off":
        return {**base, "reproduction.rv.spatial.enabled": "false"}
    path = inter_path if mode == "inter" else clim_path
    return {
        **base,
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": path,
        "reproduction.rv.spatial.field.varname": "reproductive_volume",
        "reproduction.rv.spatial.ref": str(rv_ref),  # SHARED across arms
        "reproduction.rv.spatial.species.enabled.sp0": "true",
    }


def skill_delta(model_a, model_b, observed) -> float:
    """corr(B, obs) - corr(A, obs) over the overlap (window applied by caller). A zero-variance
    (collapsed) arm has no correlation signal -> nan; callers aggregate with nanmean/nanstd."""
    o = np.asarray(observed, float)

    def c(m):
        m = np.asarray(m, float)
        n = min(len(m), len(o))
        if n <= 2 or np.std(m[:n]) == 0 or np.std(o[:n]) == 0:
            return float("nan")
        with np.errstate(invalid="ignore"):
            return float(np.corrcoef(m[:n], o[:n])[0, 1])

    return c(model_b) - c(model_a)


def _rv_ref_of(path: Path) -> float:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return float(ds["reproductive_volume"].attrs["RV_ref"])


def _cod_ssb(raw: dict, seed: int) -> np.ndarray:
    # Real maturity-based spawning-stock biomass (length>=maturity_size AND age>=maturity_age),
    # matched to observed ICES ssb_t — enabled via output.ssb.enabled in arm_overrides' base.
    from osmose.engine import PythonEngine

    b = PythonEngine().run_in_memory(raw, seed=seed).ssb()
    return b["cod"].to_numpy(dtype=float)


def run_hindcast(seeds=(0, 1, 2, 3, 4)) -> dict:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo

    rv_ref = _rv_ref_of(INTER)
    tmp = Path(tempfile.mkdtemp())
    base = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    obs = pd.read_csv(ROOT / "docs/diagnostics/ices_cod_2732_observed.csv", comment="#")[
        "ssb_t"
    ].to_numpy()
    obs_win = obs[WINDOW]
    series = {m: [] for m in ("off", "clim", "inter")}
    for seed in seeds:
        for m in series:
            raw = {**base, **arm_overrides(m, rv_ref, str(INTER), str(CLIM))}
            series[m].append(_cod_ssb(raw, seed))
    means = {m: np.mean(np.stack(v), axis=0) for m, v in series.items()}
    deltas = [
        skill_delta(
            np.stack(series["clim"])[i][WINDOW],
            np.stack(series["inter"])[i][WINDOW],
            obs_win,
        )
        for i in range(len(seeds))
    ]
    return {"means": means, "skill_delta_per_seed": deltas, "rv_ref": rv_ref, "obs": obs}


if __name__ == "__main__":
    sys.exit(0 if run_hindcast() else 0)
