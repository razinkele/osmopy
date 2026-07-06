"""Phase 3: cross-engine ICES consistency (magnitude_factor, NOT in_range — uncalibrated configs).
Gate: the three engines agree within Delta = log10(3) on each mapped species' magnitude_factor.
Reuses the persisted Java OsmoseResults from Task 9; recomputes the Python arm directly."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults
from osmose.validation.ices import compare_outputs_to_ices, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
DELTA = np.log10(3)


def _factors(results, snapshot) -> dict[str, float]:
    cmp = compare_outputs_to_ices(
        results, snapshot, window_years=5
    )  # -> list[SpeciesBiomassComparison]
    return {r.species: r.magnitude_factor for r in cmp if r.magnitude_factor is not None}


def run(config: Path, snap_dir: Path, persisted: Path, prefix: str, years: int = 10) -> None:
    snapshot = load_snapshot(str(snap_dir))
    raw = dict(OsmoseConfigReader().read(str(config)))
    raw["simulation.time.nyear"] = str(years)  # >= spinup + window_years (=5)
    py = _factors(PythonEngine().run_in_memory(raw, seed=1000), snapshot)
    j441 = _factors(
        OsmoseResults(persisted / f"{prefix}_4.4.1", prefix=prefix, strict=False), snapshot
    )
    j433 = _factors(
        OsmoseResults(persisted / f"{prefix}_4.3.3", prefix=prefix, strict=False), snapshot
    )
    common = sorted(set(py) & set(j441) & set(j433))
    # Fail LOUDLY if nothing mapped, rather than printing a vacuous PASS (review finding).
    assert common, (
        f"Phase 3 compared ZERO species — check snapshot species-name case and that "
        f"model_biomass_window_mean read the persisted results (py={len(py)}, "
        f"441={len(j441)}, 433={len(j433)})."
    )
    fails = []
    print(f"{'species':<16}{'Python':>10}{'4.3.3':>10}{'4.4.1':>10}{'agree<=D':>10}")
    for sp in common:
        # Defensive floor only: without it, a species genuinely ~0 in all
        # three engines gives log10(0) = -inf in every arm, so
        # (-inf) - (-inf) = nan fails `<= DELTA` and mis-flags a true
        # 0==0==0 agreement as a disagreement. 1e-9 is far below any real
        # biomass value, so it does not mask genuine divergences.
        vals = np.log10(np.clip([py[sp], j433[sp], j441[sp]], 1e-9, None))
        agree = (vals.max() - vals.min()) <= DELTA
        if not agree:
            fails.append(sp)
        print(
            f"{sp:<16}{py[sp]:>10.2f}{j433[sp]:>10.2f}{j441[sp]:>10.2f}{'Y' if agree else 'N':>10}"
        )
    print(
        f"\nGATE ({len(common)} species, agree within {10**DELTA:.1f}x): {'PASS' if not fails else 'REVIEW: ' + ', '.join(fails)}"
    )


if __name__ == "__main__":
    persisted = Path(
        "/tmp/claude-1000/-home-razinka-osmose/f7b91731-5bf2-427b-aaab-4e339882ae8b/scratchpad/phase3_results"
    )
    run(
        ROOT / "data" / "examples" / "osm_all-parameters.csv",
        ROOT / "data" / "examples" / "reference" / "ices_snapshots",
        persisted,
        prefix="biscay",
    )
