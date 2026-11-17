"""Capture one mortality step for the C# engine's kernel tests (plan task 0.4).

Runs the Python engine to step ``--step``, snapshots ``mortality()``'s inputs there, then
replays the step's interleaved loop with recorded draws through the per-cell Numba oracle
``_mortality_in_cell_numba``:
- egg release before each sub-step, as ``mortality()`` does;
- per sub-step and cell, four permutations and one cause order per slot, drawn here and stored.

The result is one bundle file (``osmose/engine_cs/bundle.py`` format) holding the inputs, the
draws, the oracle's outputs and a libm check (Numba's ``exp`` on the step's own rates plus a
log-uniform sample), so the C# kernel can be compared at tier E/L with no Python process.
Per-school arrays are stored in cell order (Python's mergesort by cell), so a cell's schools
are the contiguous range ``cell.start[c]:cell.start[c + 1]`` and local positions match the
oracle's ``cell_indices`` positions.

Bioenergetics must be off (the spike's scope). ``--zero-rates`` zeroes the starvation,
additional and fishing rates, which makes the loop free of ``exp``: the tier-E fixture.

This mirrors the setup in ``mortality()`` (osmose/engine/processes/mortality.py). If that setup
changes, re-read this script against it; ``fixture.mortality_sha`` records which version of the
module generated the file.

    .venv/bin/python scripts/csharp/make_fixtures.py --demo eec_full --step 30 --out DIR
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's osmose

import numpy as np

import osmose.engine.processes.mortality as mort
import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine_cs.bundle import Bundle

FIXTURE_CONTRACT = 1
N_LIBM_SAMPLE = 2_000_000


class _Captured(Exception):
    pass


def load_config(demo: str | None, config_path: str | None, workdir: Path) -> dict[str, str]:
    if demo:
        config_path = osmose_demo(demo, workdir)["config_file"]
    assert config_path is not None
    return dict(OsmoseConfigReader().read(Path(config_path)))


def capture(cfg: dict[str, str], seed: int, step: int) -> dict:
    """Run to ``step`` and return deep copies of ``mortality()``'s inputs there."""
    engine = PythonEngine()
    config, grid, rng, movement_rngs, mortality_rngs = engine._prepare_run(cfg, seed)
    if config.bioen_enabled:
        raise SystemExit("bioenergetics is on; the kernel spike covers bioen-off configs only")
    snap: dict = {}
    real = sim._mortality

    def spy(state, resources, config_, rng_, grid_, step=0, species_rngs=None, ctx=None):
        if step == target:
            snap.update(
                state=copy.deepcopy(state),
                resources=copy.deepcopy(resources),
                config=config_,
                grid=grid_,
                step=step,
                prey_density_scale=None if ctx is None else ctx.prey_density_scale,
                fleet_state=None if ctx is None else copy.deepcopy(ctx.fleet_state),
            )
            raise _Captured
        return real(
            state, resources, config_, rng_, grid_, step=step, species_rngs=species_rngs, ctx=ctx
        )

    target = step
    sim._mortality = spy
    try:
        sim.simulate(config, grid, rng, movement_rngs=movement_rngs, mortality_rngs=mortality_rngs)
    except _Captured:
        pass
    finally:
        sim._mortality = real
    if not snap:
        raise SystemExit(f"the run ended before step {step}")
    return snap


def build(snap: dict, zero_rates: bool, draw_seed: int) -> Bundle:
    config, grid, step = snap["config"], snap["grid"], snap["step"]
    state, resources = snap["state"], snap["resources"]
    n_subdt = config.mortality_subdt

    # --- mortality() setup, in its order ---
    state = mort.larva_mortality(state, config, step=step)
    egg_retained0 = np.where(state.is_egg, state.abundance, 0.0)
    state = state.replace(egg_retained=egg_retained0, n_dead=np.zeros_like(state.n_dead))
    state = state.replace(feeding_stage=mort.compute_feeding_stages(state, config))
    if config.stage_accessibility is not None:
        sa = config.stage_accessibility
        args = (state.species_id, state.age_dt, config.n_dt_per_year, config.all_species_names)
        prey_idx = sa.compute_school_indices(*args, role="prey")
        pred_idx = sa.compute_school_indices(*args, role="pred")
        access, has_access, use_stage = sa.raw_matrix, True, True
    else:
        prey_idx = np.zeros(len(state), dtype=np.int32)
        pred_idx = np.zeros(len(state), dtype=np.int32)
        has_access = config.accessibility_matrix is not None
        access = config.accessibility_matrix if has_access else mort._DUMMY_ACCESS
        use_stage = False
    if snap["prey_density_scale"] is not None and has_access:
        raise SystemExit(
            "dynamic accessibility (prey-density scaling) is outside the spike's scope"
        )

    cell_ids = state.cell_y * grid.nx + state.cell_x
    valid_idx = np.where((state.cell_x >= 0) & (state.cell_y >= 0))[0].astype(np.int32)
    order = np.argsort(cell_ids[valid_idx], kind="mergesort")
    sorted_idx = valid_idx[order]
    n_cells = grid.ny * grid.nx
    boundaries = np.searchsorted(cell_ids[valid_idx][order], np.arange(n_cells + 1))

    rsc_sm, rsc_sx, rsc_tl, rsc_ar, n_rsc = mort._precompute_resource_arrays(config, resources)
    eff_s, eff_a, eff_f, f_disc = mort._precompute_effective_rates(
        state, config, n_subdt, step, fleet_state=snap["fleet_state"]
    )
    if zero_rates:
        eff_s, eff_a, eff_f = np.zeros_like(eff_s), np.zeros_like(eff_a), np.zeros_like(eff_f)

    # --- draws, then the oracle, sub-step outer and cell inner, as mortality() runs them ---
    n = len(sorted_idx)
    rng = np.random.default_rng(draw_seed)
    perms = {k: np.zeros((n_subdt, n), dtype=np.int32) for k in ("pred", "starv", "fish", "nat")}
    orders = np.zeros((n_subdt, n, 4), dtype=np.int32)
    inst_abd = state.abundance.copy()
    n_dead = np.zeros_like(state.n_dead)
    psr = state.pred_success_rate.copy()
    preyed = state.preyed_biomass.copy()
    tl_ws = np.zeros(len(state))
    rsc_bio = resources.biomass.copy() if resources is not None else mort._DUMMY_RSC_2D.copy()
    egg_ret = egg_retained0.copy()
    zeros = np.zeros(len(state))
    for sub in range(n_subdt):
        release = np.where(state.is_egg & (egg_ret > 0), state.abundance / n_subdt, 0.0)
        egg_ret = np.maximum(0, egg_ret - release)
        for cell in range(n_cells):
            s, e = int(boundaries[cell]), int(boundaries[cell + 1])
            if e <= s:
                continue
            n_local = e - s
            for k in ("pred", "starv", "fish", "nat"):
                perms[k][sub, s:e] = rng.permutation(n_local)
            for i in range(n_local):
                orders[sub, s + i] = rng.permutation(4)
            mort._mortality_in_cell_numba(
                sorted_idx[s:e],
                perms["pred"][sub, s:e],
                perms["starv"][sub, s:e],
                perms["fish"][sub, s:e],
                perms["nat"][sub, s:e],
                perms["pred"][sub, s:e],  # seq_for: unused with a 4-wide order
                orders[sub, s:e],
                inst_abd, n_dead, eff_s, eff_a, eff_f, f_disc,
                state.species_id, state.length, state.weight, state.age_dt,
                state.first_feeding_age_dt, state.feeding_stage, psr, preyed, state.trophic_level,
                config.size_ratio_min, config.size_ratio_max, config.ingestion_rate,
                config.fr_shape, config.fr_halfsat, config.n_dt_per_year, n_subdt,
                access, has_access, use_stage, prey_idx, pred_idx,
                rsc_bio, rsc_sm, rsc_sx, rsc_tl, rsc_ar, n_rsc, config.n_species, cell,
                tl_ws, True, mort._DUMMY_DIET, False, egg_ret,
                False, mort._DUMMY_RSC_1D, mort._DUMMY_RSC_1D, zeros.copy(), zeros.copy(),
                zeros, state.is_background, zeros,
            )  # fmt: skip

    b = Bundle(FIXTURE_CONTRACT)
    sha = hashlib.sha256(Path(mort.__file__).read_bytes()).hexdigest()
    for name, value in {
        "fixture.step": step, "fixture.n_subdt": n_subdt, "fixture.n_dt_per_year": config.n_dt_per_year,
        "fixture.n_species": config.n_species, "fixture.n_cells": n_cells, "fixture.n_resources": n_rsc,
        "fixture.has_access": int(has_access), "fixture.use_stage_access": int(use_stage),
        "fixture.zero_rates": int(zero_rates), "fixture.mortality_sha": int(sha[:15], 16),
        "fixture.n_state": len(state), "fixture.bioen": int(config.bioen_enabled),
    }.items():  # fmt: skip
        b.set_int(name, int(value))

    def school(name, arr, dtype):
        b.set_array(
            f"school.{name}", np.ascontiguousarray(np.asarray(arr)[sorted_idx], dtype=dtype)
        )

    for name, arr, dt in [
        ("species_id", state.species_id, np.int32), ("length", state.length, np.float64),
        ("weight", state.weight, np.float64), ("age_dt", state.age_dt, np.int32),
        ("first_feeding_age_dt", state.first_feeding_age_dt, np.int32),
        ("feeding_stage", state.feeding_stage, np.int32), ("abundance", state.abundance, np.float64),
        ("egg_retained", egg_retained0, np.float64), ("is_egg", state.is_egg, np.uint8),
        ("trophic_level", state.trophic_level, np.float64), ("pred_access_idx", pred_idx, np.int32),
        ("prey_access_idx", prey_idx, np.int32), ("rate_starvation", eff_s, np.float64),
        ("rate_additional", eff_a, np.float64), ("rate_fishing", eff_f, np.float64),
        ("discard_fraction", f_disc, np.float64), ("is_background", state.is_background, np.uint8),
        ("pred_success_rate", state.pred_success_rate, np.float64),
        ("preyed_biomass", state.preyed_biomass, np.float64),
    ]:  # fmt: skip
        school(name, arr, dt)
    b.set_array("cell.start", boundaries.astype(np.int64))
    b.set_array("school.python_index", sorted_idx.astype(np.int32))  # state index of each school
    b.set_array(
        "species.size_ratio_min", np.ascontiguousarray(config.size_ratio_min, dtype=np.float64)
    )
    b.set_array(
        "species.size_ratio_max", np.ascontiguousarray(config.size_ratio_max, dtype=np.float64)
    )
    b.set_array("species.ingestion_rate", np.asarray(config.ingestion_rate, dtype=np.float64))
    b.set_array("species.fr_shape", np.asarray(config.fr_shape, dtype=np.int32))
    b.set_array("species.fr_halfsat", np.asarray(config.fr_halfsat, dtype=np.float64))
    b.set_array("access.matrix", np.ascontiguousarray(access, dtype=np.float64))
    b.set_array(
        "resources.biomass",
        np.ascontiguousarray(
            resources.biomass if resources is not None else rsc_bio, dtype=np.float64
        ),
    )
    b.set_array("resources.size_min", np.asarray(rsc_sm, dtype=np.float64).reshape(-1))
    b.set_array("resources.size_max", np.asarray(rsc_sx, dtype=np.float64).reshape(-1))
    b.set_array("resources.trophic_level", np.asarray(rsc_tl, dtype=np.float64).reshape(-1))
    b.set_array("resources.access_rows", np.asarray(rsc_ar, dtype=np.int32).reshape(-1))
    b.set_array("draws.perm_predation", perms["pred"])
    b.set_array("draws.perm_starvation", perms["starv"])
    b.set_array("draws.perm_fishing", perms["fish"])
    b.set_array("draws.perm_additional", perms["nat"])
    b.set_array("draws.cause_orders", orders.astype(np.uint8))
    school_out = {
        "abundance": inst_abd, "pred_success_rate": psr, "preyed_biomass": preyed,
        "tl_weighted_sum": tl_ws, "egg_retained": egg_ret,
    }  # fmt: skip
    for name, arr in school_out.items():
        b.set_array(f"expected.{name}", np.ascontiguousarray(arr[sorted_idx]))
    b.set_array("expected.n_dead", np.ascontiguousarray(n_dead[sorted_idx]))
    b.set_array("expected.resources_biomass", np.ascontiguousarray(rsc_bio, dtype=np.float64))

    rates = np.concatenate([eff_s, eff_a, eff_f])
    sample = np.exp(np.random.default_rng(7).uniform(np.log(1e-8), np.log(20.0), N_LIBM_SAMPLE))
    args = np.concatenate([np.unique(rates[rates > 0]), sample])
    b.set_array("libm.args", args)
    b.set_array("libm.exp_neg", _numba_exp_neg(args))
    return b


def _numba_exp_neg(x: np.ndarray) -> np.ndarray:
    from numba import njit

    @njit(cache=False)
    def kernel(v):
        out = np.empty_like(v)
        for i in range(v.size):
            out[i] = np.exp(-v[i])
        return out

    return kernel(x)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", help="a demo name from osmose.demo (eec_full, baltic, ...)")
    src.add_argument("--config", help="path to a master config file")
    p.add_argument("--step", type=int, required=True, help="capture the mortality of this step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--draw-seed", type=int, default=20261117)
    p.add_argument("--zero-rates", action="store_true", help="tier-E fixture: no exp anywhere")
    p.add_argument("--out", required=True, help="output directory")
    a = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(a.demo, a.config, Path(tmp))
        ndt = int(cfg.get("simulation.time.ndtperyear", "24"))
        cfg["simulation.time.nyear"] = str(a.step // ndt + 1)
        snap = capture(cfg, a.seed, a.step)
        bundle = build(snap, a.zero_rates, a.draw_seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = a.demo or Path(a.config).stem
    path = out / f"{stem}_step{a.step}{'_zero' if a.zero_rates else ''}.bin"
    bundle.save(path)
    n = len(bundle.get("school.length"))  # type: ignore[arg-type]
    print(f"wrote {path} ({n} located schools, {path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
