"""Derive the unfished-level recruitment ceiling from an F=0 reference run.

Runs a config with fishing mortality zeroed, records per-step per-species
recruitment (fresh natural eggs) through the engine's step_observer hook, and
writes a per-season sidecar CSV (season_idx,ceiling_sp0,...) that the engine
loads via reproduction.recruitment.ceiling.series.file. See
docs/superpowers/specs/2026-07-03-baltic-recruitment-ceiling-design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def zero_fishing(cfg: dict) -> dict:
    """Copy of cfg with both fishing modes disabled (rate-based + v4 fisheries)."""
    out = dict(cfg)
    out["module.multispecies.fisheries.enabled"] = "false"
    out["simulation.fishing.mortality.enabled"] = "false"
    return out


def per_species_recruitment(state, n_species: int) -> np.ndarray:
    """Sum abundance of fresh eggs (age_dt==0, is_egg) by species.

    Note: SchoolState has no per-school "seeded vs. natural" marker (the
    seeding bootstrap in reproduction() is tracked only as a local per-species
    flag inside that function, not attached to the resulting schools), so this
    cannot distinguish seeded eggs from naturally-spawned ones at the state
    level. The derivation CLI relies instead on the late-window averaging
    (main() only reads the last `late_frac` of years) to avoid the early
    seeding-bootstrap period, matching the plan's architecture note ("fresh
    eggs are the age_dt == 0 schools -- no engine change needed").
    """
    out = np.zeros(n_species, dtype=np.float64)
    fresh = (state.age_dt == 0) & state.is_egg
    if fresh.any():
        np.add.at(out, state.species_id[fresh], state.abundance[fresh])
    return out


class RecruitmentRecorder:
    """step_observer that records (step, per-species recruitment) each step."""

    def __init__(self, n_species: int):
        self.n_species = n_species
        self.records: list[tuple[int, np.ndarray]] = []

    def __call__(self, step, state, grid, config, map_sets):
        self.records.append((step, per_species_recruitment(state, self.n_species)))


def late_window_ceiling(records, n_cols: int, n_species: int, n_dt: int, frac: float) -> np.ndarray:
    """Mean recruitment over the last `frac` of model years, grouped by step % n_cols."""
    if not records:
        raise ValueError("No recruitment records; cannot derive a ceiling.")
    max_step = max(s for s, _ in records)
    n_years = (max_step + 1) / n_dt
    start_year = n_years * (1.0 - frac)
    start_step = int(start_year * n_dt)
    sums = np.zeros((n_cols, n_species), dtype=np.float64)
    counts = np.zeros(n_cols, dtype=np.int64)
    for step, rec in records:
        if step < start_step:
            continue
        col = step % n_cols
        sums[col] += rec
        counts[col] += 1
    if np.any(counts == 0):
        raise ValueError(
            f"Late window covers only {counts.tolist()} steps per season; "
            f"increase run length or late-frac."
        )
    return sums / counts[:, None]


def seeding_overlap_warnings(seeding_max_step, total_steps, n_dt, frac):
    """Warn per species whose seeding-eligible window overlaps the late window.

    Seeded eggs are indistinguishable from natural eggs here (no from_seeding
    field on schools), so a species that collapses to SSB=0 inside the late
    averaging window would have its ceiling contaminated by the configured
    seeding biomass rather than a natural equilibrium. Non-fatal: returns a
    list of warning strings (empty = no overlap).
    """
    n_years = total_steps / n_dt
    start_step = int(n_years * (1.0 - frac) * n_dt)
    warnings = []
    for sp, smax in enumerate(seeding_max_step):
        if int(smax) > start_step:
            warnings.append(
                f"sp{sp}: seeding-eligible until step {int(smax)} overlaps the late "
                f"averaging window (starts step {start_step}); if this species "
                f"collapses to SSB=0 in that window its derived ceiling may be "
                f"contaminated by seeding biomass. Increase simulation.time.nyear "
                f"or --late-frac."
            )
    return warnings


def check_stationarity(records, n_cols, n_species, n_dt, frac, tol=0.25) -> list[str]:
    """Warn if the last-window per-season mean differs from the preceding window
    by more than `tol` (relative). Returns a list of warning strings (empty = ok)."""
    late = late_window_ceiling(records, n_cols, n_species, n_dt, frac)
    # Preceding window of the same width, immediately before the late window.
    max_step = max(s for s, _ in records)
    n_years = (max_step + 1) / n_dt
    lo = int(n_years * (1.0 - 2 * frac) * n_dt)
    hi = int(n_years * (1.0 - frac) * n_dt)
    sums = np.zeros((n_cols, n_species))
    counts = np.zeros(n_cols, dtype=np.int64)
    for step, rec in records:
        if lo <= step < hi:
            sums[step % n_cols] += rec
            counts[step % n_cols] += 1
    warnings: list[str] = []
    if np.any(counts == 0):
        return ["Preceding window empty; cannot assess stationarity (run longer)."]
    prev = sums / counts[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(late - prev) / np.where(late > 0, late, np.nan)
    bad = np.nanmax(rel) if np.isfinite(np.nanmax(rel)) else 0.0
    if bad > tol:
        warnings.append(
            f"Unfished run may not be stationary: max per-season drift "
            f"{bad:.0%} > {tol:.0%}. The derived ceiling may be unreliable."
        )
    return warnings


def write_ceiling_csv(ceiling: np.ndarray, path: Path) -> Path:
    path = Path(path)
    n_cols, n_sp = ceiling.shape
    header = "season_idx," + ",".join(f"ceiling_sp{i}" for i in range(n_sp))
    lines = [header]
    for s in range(n_cols):
        lines.append(str(s) + "," + ",".join(f"{ceiling[s, i]:.6f}" for i in range(n_sp)))
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Derive the unfished recruitment ceiling.")
    ap.add_argument("--config", required=True, help="Path to the all-parameters config CSV.")
    ap.add_argument("--out", required=True, help="Output sidecar CSV path.")
    ap.add_argument("--late-frac", type=float, default=1.0 / 3.0)
    args = ap.parse_args(argv)

    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig

    reader = OsmoseConfigReader()
    cfg = zero_fishing(reader.read(Path(args.config)))
    engine_cfg = EngineConfig.from_dict(cfg)
    n_sp = engine_cfg.n_species
    n_dt = engine_cfg.n_dt_per_year
    n_cols = engine_cfg.spawning_season.shape[1] if engine_cfg.spawning_season is not None else n_dt

    # run() (not run_in_memory) forwards step_observer; it needs an output_dir,
    # so send disk outputs to a throwaway temp dir -- we only want the recorder.
    import tempfile

    recorder = RecruitmentRecorder(n_sp)
    with tempfile.TemporaryDirectory() as td:
        PythonEngine().run(cfg, Path(td), seed=0, step_observer=recorder)

    for w in check_stationarity(recorder.records, n_cols, n_sp, n_dt, args.late_frac):
        print("WARNING:", w)
    _total_steps = max(s for s, _ in recorder.records) + 1
    for w in seeding_overlap_warnings(
        engine_cfg.seeding_max_step, _total_steps, n_dt, args.late_frac
    ):
        print("WARNING:", w)
    ceiling = late_window_ceiling(recorder.records, n_cols, n_sp, n_dt, args.late_frac)
    out = write_ceiling_csv(ceiling, args.out)
    print(f"Wrote ceiling ({ceiling.shape[0]} seasons x {ceiling.shape[1]} species) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
