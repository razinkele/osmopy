"""SP-A Phase 0 diagnostic — identify the Baltic collapse driver.

Runs the bundled Baltic config 50 yr x 3 seeds and reports, per seed: the year each focal species
first drops below 0.1 x its ICES-lower bound, and for the earliest-collapsing species the mean share
of each mortality cause (predation / starvation / additional / fishing) in the 5 steps before its
collapse. Output feeds the finding note docs/baltic_stability_diagnostic_2026-07-01.md.

    PYTHONPATH=. .venv/bin/python scripts/baltic_stability_diagnostic.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults

FOCAL = ["cod", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt", "stickleback"]
LOWER = {
    "cod": 60000, "herring": 800000, "sprat": 800000, "flounder": 20000,
    "perch": 8000, "pikeperch": 4000, "smelt": 20000, "stickleback": 50000,
}


def _first_below_year(values: np.ndarray, floor: float) -> int | None:
    below = np.where(values < floor)[0]
    return int(below[0]) if len(below) else None


def _mortality_shares(results: OsmoseResults, sp: str, collapse_year: int) -> dict | None:
    """Mean share of each mortality cause over the ~5 years before collapse, or None if unavailable."""
    try:
        m = results.mortality(sp)  # (cause, stage) MultiIndex columns, indexed by time
    except Exception as exc:  # noqa: BLE001 — diagnostic must not crash on output-shape drift
        print(f"    (mortality({sp}) unavailable: {exc})")
        return None
    try:
        # Sum over stage level -> per-cause series; average the last 5 rows before collapse.
        by_cause = m.groupby(level=0, axis=1).sum() if hasattr(m.columns, "levels") else m
        lo = max(0, collapse_year - 5)
        window = by_cause.iloc[lo:collapse_year] if collapse_year else by_cause.iloc[-5:]
        means = window.mean(axis=0)
        tot = float(means.sum()) or 1.0
        return {str(c): round(float(means[c]) / tot, 3) for c in means.index}
    except Exception as exc:  # noqa: BLE001
        print(f"    (mortality decomposition failed for {sp}: {exc})")
        return None


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg["simulation.time.nyear"] = "50"

    for seed in (42, 123, 7):
        outdir = tmp / f"out{seed}"
        outdir.mkdir()
        PythonEngine().run(cfg, output_dir=outdir, seed=seed)
        r = OsmoseResults(outdir)
        bio = r.biomass()  # wide: Time + per-species columns
        print(f"\n--- seed {seed}: first year below 0.1*ICES-lower ---")
        first = {}
        for sp in FOCAL:
            if sp not in bio.columns:
                print(f"  {sp:12s}: (no column)")
                continue
            yr = _first_below_year(np.asarray(bio[sp].values, float), 0.1 * LOWER[sp])
            first[sp] = yr
            print(f"  {sp:12s}: {'yr ' + str(yr) if yr is not None else 'persists'}")
        # earliest collapser -> mortality decomposition
        collapsers = {sp: y for sp, y in first.items() if y is not None}
        if collapsers:
            keystone = min(collapsers, key=collapsers.get)
            print(f"  keystone (first to collapse): {keystone} at yr {collapsers[keystone]}")
            shares = _mortality_shares(r, keystone, collapsers[keystone])
            if shares:
                print(f"  {keystone} mortality cause-share (5 yr pre-collapse): {shares}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
