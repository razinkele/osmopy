#!/usr/bin/env python
"""Share of the ICES-age-1 sprat cohort that is mature at spawning time (2026-11-13).

Measured on the state the reproduction step sees, at every sprat spawning step (season weight
> 0). ICES age = calendar year now - calendar year of birth (birth step = step - age_dt), which
matches the assessment's 1 January birthday because the model year starts in January. The share
is by numbers, weighted by the spawning-season weight, over the final decade. Also records sprat's
regulated egg output (after the stock-recruitment curve), and writes the run's outputs so the
biomass, SSB and the ICES validator can be read from them.

Usage (from the repo root, so `osmose` resolves to this checkout):
    PYTHONPATH=. .venv/bin/python scripts/sprat_age1_maturity_share.py \\
        --config data/baltic/baltic_all-parameters.csv --seed 42 --years 50 --out <dir>

Criteria and results: docs/diagnostics/sprat_maturity_age_prestated_2026-11-13.md,
docs/baltic_sprat_maturity_age_2026-11-13.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import osmose.engine.processes.reproduction as repro
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults
from osmose.validation.ices import compare_outputs_to_ices, load_snapshot

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "data" / "baltic" / "reference" / "ices_snapshots"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--species", default="sprat")
    ap.add_argument("--out", type=Path, required=True, help="directory for outputs + summary")
    args = ap.parse_args()

    cfg = dict(OsmoseConfigReader().read(str(args.config)))
    cfg["simulation.time.nyear"] = str(args.years)
    names = {
        v: int(k.rsplit("sp", 1)[1]) for k, v in cfg.items() if k.startswith("species.name.sp")
    }
    sp = names[args.species]

    rows: list[dict] = []
    eggs: dict[int, float] = {}
    orig_repro, orig_reg = repro.reproduction, repro.regulate_recruitment

    def _reg(n_eggs_linear, ssb, seeded, config, step):
        out = orig_reg(n_eggs_linear, ssb, seeded, config, step)
        eggs[step] = float(out[sp])
        return out

    def _repro(state, config, step, rng, **kw):
        ndt = config.n_dt_per_year
        season = float(config.spawning_season[sp, step % config.spawning_season.shape[1]])
        if season > 0 and len(state):
            m = (state.species_id == sp) & (state.abundance > 0) & ~state.is_egg
            age, n, length = state.age_dt[m], state.abundance[m], state.length[m]
            ices_age = step // ndt - np.floor_divide(step - age, ndt)
            mature = (length >= config.maturity_size[sp]) & (age >= config.maturity_age_dt[sp])
            a1, a2p = ices_age == 1, ices_age >= 2
            rows.append(
                {
                    "step": step,
                    "season": season,
                    "a1": float(n[a1].sum()),
                    "a1_mature": float(n[a1 & mature].sum()),
                    "a2p": float(n[a2p].sum()),
                    "a2p_mature": float(n[a2p & mature].sum()),
                }
            )
        return orig_repro(state, config, step, rng, **kw)

    repro.reproduction, repro.regulate_recruitment = _repro, _reg
    try:
        PythonEngine().run(cfg, args.out / "output", seed=args.seed)
    finally:
        repro.reproduction, repro.regulate_recruitment = orig_repro, orig_reg

    ndt = 24
    first = args.years - 10
    tail = [r for r in rows if r["step"] // ndt >= first]
    w = np.array([r["season"] for r in tail])

    def _share(num: str, den: str) -> np.ndarray:
        return np.array([r[num] / r[den] if r[den] > 0 else np.nan for r in tail])

    a1, a2p = _share("a1_mature", "a1"), _share("a2p_mature", "a2p")
    per_step: dict[int, list[float]] = {}
    for r, s in zip(tail, a1, strict=True):
        per_step.setdefault(r["step"] % ndt, []).append(float(s))
    egg_years = [sum(v for k, v in eggs.items() if k // ndt == y) for y in range(first, args.years)]

    res = OsmoseResults(args.out / "output", prefix="osm", strict=False)
    bio, ssb = res.biomass(), res.ssb()
    late_bio, late_ssb = bio[bio["Time"] >= first], ssb[ssb["Time"] >= first]
    species = [c for c in late_ssb.columns if c not in ("Time", "species")]
    comparisons = compare_outputs_to_ices(res, load_snapshot(SNAPSHOTS), window_years=5)
    summary = {
        "config": str(args.config),
        "seed": args.seed,
        "years": args.years,
        "maturity_age_key": cfg.get(f"species.maturity.age.sp{sp}"),
        "a1_share_season_weighted": float(np.nansum(w * a1) / w[~np.isnan(a1)].sum()),
        "a1_share_plain_mean": float(np.nanmean(a1)),
        "a1_share_per_step": {k: float(np.nanmean(v)) for k, v in sorted(per_step.items())},
        "a2p_share_season_weighted": float(np.nansum(w * a2p) / w[~np.isnan(a2p)].sum()),
        "eggs_per_year_final_decade": float(np.mean(egg_years)),
        "biomass_final_decade": {c: float(late_bio[c].mean()) for c in species},
        "ssb_final_decade": {c: float(late_ssb[c].mean()) for c in species},
        "validator": {
            c.species: {
                "ssb": c.model_mean_tonnes,
                "envelope": [c.ices_min_tonnes, c.ices_max_tonnes],
                "in_range": c.in_range,
                "magnitude": c.magnitude_factor,
            }
            for c in comparisons
            if c.in_range is not None
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("seed", "a1_share_season_weighted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
