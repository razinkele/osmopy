"""CLI: compute model-internal fishery reference points (Fmsy/Bmsy/Blim) by a yield-vs-F sweep."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.validation.fmsy_sweep import compute_model_reference_points


def write_model_sidecar(refs, out_path: Path, meta: dict) -> None:
    """Write model reference points to a JSON sidecar.

    Parameters
    ----------
    refs : dict
        Mapping of species name to :class:`ModelReferencePoint`.
    out_path : Path
        Output JSON path.
    meta : dict
        Metadata dict (grid, replicates, etc.) to include as "_meta" key.
    """
    payload = {"_meta": meta}
    for sp, rp in refs.items():
        payload[sp] = {
            "fmsy": rp.fmsy,
            "bmsy": rp.bmsy,
            "b0": rp.b0,
            "blim": rp.blim,
            "fmsy_at_boundary": rp.fmsy_at_boundary,
            "multi_peak": rp.multi_peak,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the master parameters file")
    ap.add_argument("--grid", type=float, nargs="*", default=None, help="absolute F grid")
    ap.add_argument("--n-years", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg_path = Path(a.config)
    base = dict(OsmoseConfigReader().read(str(cfg_path)))  # injects _osmose.config.dir
    grid = np.asarray(a.grid, dtype=float) if a.grid else None
    n_grid = len(grid) if grid is not None else 7
    from osmose.engine.config import EngineConfig

    n_sp = EngineConfig.from_dict(dict(base)).n_species
    n_years = a.n_years or max(int(base.get("simulation.time.nyear", "30")), 30)
    print(
        f"Sweep: {n_sp} species x {n_grid} F x {a.replicates} reps = "
        f"{n_sp * n_grid * a.replicates} runs of {n_years} yr each (this is offline; expect "
        f"tens of minutes to hours)."
    )
    t0 = time.time()
    refs = compute_model_reference_points(
        base, grid=grid, n_years=a.n_years, replicates=a.replicates, max_workers=a.workers
    )
    ecosystem = cfg_path.parent.name
    out = (
        Path(a.out)
        if a.out
        else (Path("data") / ecosystem / "reference" / "fisheries_model_reference_points.json")
    )
    meta = {
        "grid": (grid.tolist() if grid is not None else None),
        "n_years": n_years,
        "replicates": a.replicates,
        "window_years": 10,
        "f_basis": "realized_exploited_stage",
    }
    write_model_sidecar(refs, out, meta)
    print(f"Wrote {out} ({len(refs)} species) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
