"""A/B diagnostic: Baltic cod boom/bust overshoot with the recruitment ceiling
off vs on. This is the go/no-go signal for the ceiling lever (mirrors the RV-gate
diagnostic). See docs/superpowers/specs/2026-07-03-baltic-recruitment-ceiling-design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def overshoot_ratio(biomass_series: np.ndarray, late_frac: float = 1.0 / 3.0) -> float:
    """Boom/bust overshoot = peak biomass / late-window mean biomass."""
    b = np.asarray(biomass_series, dtype=np.float64)
    n = len(b)
    late = b[int(n * (1.0 - late_frac)) :]
    late_mean = float(np.mean(late))
    if late_mean <= 0:
        return float("inf")
    return float(np.max(b)) / late_mean


def run_ab(config_path: str, cod_index: int, cod_name: str = "cod", out_ceiling=None) -> dict:
    """Run Baltic with the ceiling off, derive the ceiling, run with it on for cod,
    and return the cod overshoot ratio in both cases.

    Biomass series come from OsmoseResults.biomass() -- a wide frame keyed by
    species NAME (column `cod_name`); the enable key uses the species INDEX.
    """
    import tempfile

    import derive_recruitment_ceiling as derive
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig

    base = OsmoseConfigReader().read(config_path)

    # 1. OFF run -- cod biomass series (wide frame, species-name column).
    off_series = PythonEngine().run_in_memory(dict(base), seed=0).biomass()[cod_name].to_numpy()

    # 2. Derive the ceiling from an F=0 run (run() forwards step_observer).
    out_ceiling = out_ceiling or str(Path(config_path).parent / "baltic_recruitment_ceiling.csv")
    zero = derive.zero_fishing(dict(base))
    ecfg = EngineConfig.from_dict(zero)
    n_dt = ecfg.n_dt_per_year
    n_cols = ecfg.spawning_season.shape[1] if ecfg.spawning_season is not None else n_dt
    rec = derive.RecruitmentRecorder(ecfg.n_species)
    with tempfile.TemporaryDirectory() as td:
        PythonEngine().run(zero, Path(td), seed=0, step_observer=rec)

    for w in derive.check_stationarity(rec.records, n_cols, ecfg.n_species, n_dt, 1.0 / 3.0):
        print("WARNING:", w)
    total_steps = max(s for s, _ in rec.records) + 1
    for w in derive.seeding_overlap_warnings(ecfg.seeding_max_step, total_steps, n_dt, 1.0 / 3.0):
        print("WARNING:", w)

    ceiling = derive.late_window_ceiling(rec.records, n_cols, ecfg.n_species, n_dt, 1.0 / 3.0)
    derive.write_ceiling_csv(ceiling, out_ceiling)
    print(f"Derived cod (sp{cod_index}) ceiling per season: {ceiling[:, cod_index].tolist()}")

    # 3. ON run -- enable the ceiling for cod only.
    on_cfg = dict(base)
    on_cfg["reproduction.recruitment.ceiling.enabled"] = "true"
    on_cfg["reproduction.recruitment.ceiling.series.file"] = Path(out_ceiling).name
    on_cfg[f"reproduction.recruitment.ceiling.species.enabled.sp{cod_index}"] = "true"
    on_series = PythonEngine().run_in_memory(on_cfg, seed=0).biomass()[cod_name].to_numpy()

    return {"off": overshoot_ratio(off_series), "on": overshoot_ratio(on_series)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic cod recruitment-ceiling A/B diagnostic.")
    ap.add_argument("--config", default="data/baltic/baltic_all-parameters.csv")
    ap.add_argument("--cod-index", type=int, required=True)
    ap.add_argument("--cod-name", default="cod")
    args = ap.parse_args(argv)
    res = run_ab(args.config, args.cod_index, args.cod_name)
    print(f"cod overshoot ratio  OFF={res['off']:.3f}  ON={res['on']:.3f}")
    if res["on"] < res["off"]:
        print("GO: ceiling damps the boom/bust overshoot.")
    else:
        print("NO-GO: ceiling does not damp overshoot (rule out, like the RV gate).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
