"""Plan task 2.4: the salinity movement gate (feature salinity_gate) against production.

Baltic's own gate (cod, on its bottom-salinity climatology) at ``_movement`` entry of a real run.
Production is ``_map_move_batch_numba`` (its ``py_func``) driven by SCRIPTED draws, as in
tests/test_csharp_movement.py; the C# side is ``MovementProbe.Gate`` (probe subprocess).

Tier E:
- the bundle's weight table, frame map, species mask and per-(map, frame) maximum equal the
  engine's ``_movement_salinity_weight`` and the kernel's own maximum rule;
- each gated school's branch (out, place, weighted walk, plain walk, stranded) equals production's;
- a gated walker's candidates are the plain candidates with a positive product, and their total
  is the kernel's running sum;
- for scripted uniforms, C#'s weighted choice is the cell production's kernel picks.

Tier S (chi-square, each with a control arm that must reject): the gated walk follows the
weights, and gated placement follows ``map * weight``.

The all-zero guard: with a field that excludes every cell, production and C# both move exactly
as with the gate off. The runs happen in child processes and need the built engine.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chisquare

from osmose.engine import PythonEngine
from osmose.engine.physical_data import PhysicalData
from osmose.engine.processes import movement
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import required_features
from tests.test_csharp_movement import (
    CASES,
    _capture,
    _counts,
    _kernel_args,
    _production_map_branch,
    _raw,
    _replicated,
    _run_probe,
    _ShimNumpy,
    _state_of,
    _write_state,
)

ROOT = Path(__file__).resolve().parents[1]
UNIFORMS = (0.0, 1e-9, 0.2, 0.5, 0.8, 0.999999)
N_SAMPLE = 60


def _gate_probe(model: Path, state: Path, step: int, out: Path) -> Bundle:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.MovementProbe.Gate(sys.argv[2], sys.argv[3], int(sys.argv[4]), "
        "sys.argv[5], sys.argv[6])"
    )
    us = ",".join(repr(u) for u in UNIFORMS)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(model), str(state), str(step), us, str(out)],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-4000:]
    return Bundle.load(out)


@pytest.fixture(scope="module")
def gated(tmp_path_factory) -> dict:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    d = tmp_path_factory.mktemp("baltic_gate")
    got = _capture(_raw("baltic", d), CASES["baltic"])
    assert "salinity_gate" in required_features(got["config"], got["raw"])
    bundle = build_bundle(got["config"], got["raw"], got["grid"], seed=11)
    bundle.save(d / "model.bin")
    _write_state(d / "state.bin", *_state_of(got["state"]))
    schools, cur, same, gate_args = _kernel_args(got, gate=True)
    got.update(
        dir=d, bundle=bundle, schools=schools, cur=cur, same=same, gate_args=gate_args,
        gate=_gate_probe(d / "model.bin", d / "state.bin", got["step"], d / "gate.bin"),
        k_of={int(i): k for k, i in enumerate(schools)},
    )  # fmt: skip
    return got


def _weight(got: dict) -> np.ndarray:
    b, step = got["bundle"], got["step"]
    return b.get("movement.salinity.weight")[b.get("movement.salinity.frame_by_step")[step]]


def _map_of(got: dict, i: int) -> np.ndarray:
    """The flattened current map of school ``i`` (its species' offset plus its map index)."""
    k = got["k_of"][i]
    g = got["flat"][3][got["state"].species_id[i]] + got["cur"][k]
    return np.asarray(got["flat"][0][g]).ravel()


def _cells(res: Bundle, name: str, i: int) -> np.ndarray:
    off = res.get(f"{name}.offsets")
    return np.array(res.get(f"{name}.cells")[off[i] : off[i + 1]])


# ------------------------------------------------------------------ the bundle (tier E)


def test_the_gate_tables_are_the_engines(gated):
    config, grid, b = gated["config"], gated["grid"], gated["bundle"]
    weight = b.get("movement.salinity.weight")
    n_frames, n_steps = weight.shape[0], int(config.n_steps)
    assert n_frames == 24 and weight.shape[1] == grid.ny * grid.nx
    for f in range(n_frames):
        want = movement._movement_salinity_weight(config, grid, f).ravel()
        assert weight[f].tobytes() == np.ascontiguousarray(want).tobytes(), f
    assert b.get("movement.salinity.frame_by_step").tolist() == [s % 24 for s in range(n_steps)]
    n = config.n_species
    assert b.get("movement.salinity.species").tolist() == [
        int(v) for v in config.salinity_gate_species[:n]
    ]
    assert b.get("movement.salinity.species").sum() == 2  # the two cod stocks
    # The kernel's maximum rule, cell by cell, for every map at three frames.
    maps, wmax = np.asarray(gated["flat"][0]), b.get("movement.salinity.wmax")
    assert wmax.shape == (len(maps), n_frames)
    for g in range(len(maps)):
        for f in (0, 9, 23):
            best = 0.0
            for v in maps[g].ravel() * weight[f]:
                if not np.isnan(v) and v > best:
                    best = v
            assert wmax[g, f] == best, (g, f)
    assert np.isnan(weight).any() and (weight[~np.isnan(weight)] < 1).any()  # land, and a gradient


# ------------------------------------------------------------------ branches and sets (tier E)


def _gated_schools(got: dict) -> np.ndarray:
    st, species = got["state"], got["config"].salinity_gate_species
    return np.array([int(i) for i in got["schools"] if species[st.species_id[i]]])


def test_gated_branches_equal_production(gated):
    branch = gated["gate"].get("gate.branch")
    idx = _gated_schools(gated)
    rng = np.random.default_rng(1)
    sample = rng.choice(idx, size=min(300, len(idx)), replace=False)
    seen: dict[tuple[int, str], int] = {}
    for i in sample:
        k = gated["k_of"][int(i)]
        prod = _production_map_branch(
            gated, k, int(i), gated["cur"], gated["same"], gated["gate_args"]
        )[0]
        n_cand = len(_cells(gated["gate"], "gate.cand", int(i)))
        want = {
            0: {"out"},
            1: {"place"},
            2: {"place"},
            3: {"walk", "stay"},  # a gated species on a map with no positive product: ungated
            4: {"weighted"} if n_cand else {"stay"},
        }[int(branch[i])]
        assert prod in want, (int(i), int(branch[i]), prod, n_cand)
        seen[(int(branch[i]), prod)] = seen.get((int(branch[i]), prod), 0) + 1
    assert seen.get((4, "weighted"), 0) > 20 and seen.get((2, "place"), 0) > 0, seen
    ungated = np.setdiff1d(gated["schools"], idx)
    assert not np.isin(branch[ungated], (2, 4)).any()  # other species are never gated


def test_gated_candidates_and_their_total_follow_the_kernel(gated):
    """Candidates: the plain walk candidates (ocean, map placeable) whose product is placeable
    too. Total: the products added in row-major order, as the kernel's W."""
    d, res = gated["dir"], gated["gate"]
    plain = _run_probe(d / "model.bin", d / "state.bin", gated["step"], d / "run.bin")
    w = _weight(gated)
    walkers = np.flatnonzero(res.get("gate.branch") == 4)
    assert len(walkers) > 100
    dropped = 0
    for i in walkers[:: max(1, len(walkers) // 400)]:
        m = _map_of(gated, int(i))
        base = _cells(plain, "move.cand", int(i))  # MapWalkCandidates: the ungated set
        product = m[base] * w[base]
        keep = base[(product > 0) & ~np.isnan(product)]
        got = _cells(res, "gate.cand", int(i))
        assert got.tolist() == keep.tolist(), int(i)
        dropped += len(base) - len(keep)
        total = 0.0
        for v in m[keep] * w[keep]:
            total += v
        assert res.get("gate.total")[i] == total, int(i)
    assert dropped > 0  # the gate removes cells somewhere, or the comparison shows nothing


class _UniformRandom:
    """``np.random`` for the gated walk: every uniform is ``u``; an index draw is an error here."""

    def __init__(self, u: float):
        self.u = u

    def seed(self, _):
        pass

    def randint(self, low, high):
        raise AssertionError("a gated walker drew an index")

    def random(self):
        return self.u


def _production_weighted_destination(got: dict, i: int, u: float) -> int:
    config, grid, st = got["config"], got["grid"], got["state"]
    k = got["k_of"][i]
    ox, oy, oo = st.cell_x.copy(), st.cell_y.copy(), np.zeros(len(st), dtype=np.bool_)
    old = movement.np
    movement.np = _ShimNumpy(_UniformRandom(u))
    try:
        movement._map_move_batch_numba.py_func(
            0, np.array([i], dtype=np.int32), got["cur"][k : k + 1], got["same"][k : k + 1],
            st.cell_x.copy(), st.cell_y.copy(), st.species_id, *got["flat"], grid.ocean_mask,
            config.random_walk_range.astype(np.int32), grid.ny, grid.nx, ox, oy, oo,
            *got["gate_args"],
        )  # fmt: skip
    finally:
        movement.np = old
    assert not oo[i]
    return int(oy[i]) * grid.nx + int(ox[i])


def test_the_weighted_choice_is_productions_for_scripted_uniforms(gated):
    res = gated["gate"]
    n_cand = np.diff(res.get("gate.cand.offsets"))
    walkers = np.flatnonzero((res.get("gate.branch") == 4) & (n_cand > 1))
    assert len(walkers) > N_SAMPLE
    pick = np.random.default_rng(2).choice(walkers, size=N_SAMPLE, replace=False)
    choice = res.get("gate.choice")
    distinct = 0
    for i in pick:
        cells = [_production_weighted_destination(gated, int(i), u) for u in UNIFORMS]
        assert choice[i].tolist() == cells, int(i)
        distinct += len(set(cells)) > 1
    assert distinct > N_SAMPLE // 2  # different uniforms really reach different cells


# ------------------------------------------------------------------ frequencies (tier S)


@pytest.fixture(scope="module")
def gradient(gated, tmp_path_factory) -> dict:
    """The same state under a synthetic field: salinity rising west to east from below s_low to
    above s_high, so the weights go from 0 to 1 across the grid and differ within a walk's reach.
    Baltic's own field is nearly flat where its cod are at the captured step (weights 0.77 to 1),
    too even for a control arm to reject a uniform walk."""
    cfg, grid = copy.copy(gated["config"]), gated["grid"]
    lo, hi = cfg.salinity_gate_s_low, cfg.salinity_gate_s_high
    field = np.broadcast_to(np.linspace(lo - 1.0, hi + 1.0, grid.nx), (grid.ny, grid.nx)).copy()
    cfg.salinity_field = PhysicalData(data=field[np.newaxis], constant=None, nsteps_year=1)
    d = tmp_path_factory.mktemp("gradient")
    bundle = build_bundle(cfg, gated["raw"], grid, seed=11)
    bundle.save(d / "model.bin")
    res = _gate_probe(d / "model.bin", gated["dir"] / "state.bin", gated["step"], d / "gate.bin")
    weight = np.array(bundle.get("movement.salinity.weight")[0])
    assert weight.min() == 0.0 and weight.max() == 1.0
    return {"dir": d, "gate": res, "weight": weight}


def _pooled(obs: np.ndarray, expected: np.ndarray, floor: float = 10.0):
    """Chi-square inputs with the cells of small expected count pooled into one bin."""
    small = expected < floor
    if small.sum() < 2:
        return obs, expected
    return (
        np.append(obs[~small], obs[small].sum()),
        np.append(expected[~small], expected[small].sum()),
    )


def test_the_gated_walk_follows_the_weights(gated, gradient, tmp_path):
    res, st, nx, w = gradient["gate"], gated["state"], gated["grid"].nx, gradient["weight"]
    walkers = np.flatnonzero(res.get("gate.branch") == 4)

    def products(i):
        cells = _cells(res, "gate.cand", int(i))
        return cells, _map_of(gated, int(i))[cells] * w[cells]

    def spread(i):  # how uneven the candidates' weights are
        p = products(i)[1]
        return (p.max() / p.min()) if len(p) >= 4 else 0.0

    i = int(max(walkers, key=spread))
    cells, p = products(i)
    assert p.max() / p.min() > 2.0
    path = tmp_path / "walk.bin"
    _replicated(
        path, int(st.species_id[i]), int(st.age_dt[i]), 20_000, (int(st.cell_x[i]), int(st.cell_y[i]))
    )  # fmt: skip
    out = _run_probe(gradient["dir"] / "model.bin", path, gated["step"], tmp_path / "walk_out.bin")
    obs = _counts(out, nx, cells)  # asserts every school landed on a candidate
    assert chisquare(obs, p / p.sum() * obs.sum()).pvalue > 1e-4, obs
    assert chisquare(obs).pvalue < 1e-6  # control: a uniform walk is rejected


def test_gated_placement_follows_map_times_weight(gated, gradient, tmp_path):
    res, st, w = gradient["gate"], gated["state"], gradient["weight"]
    i = int(np.flatnonzero(np.isin(res.get("gate.branch"), (2, 4)))[0])  # any gated school
    m = _map_of(gated, i)
    product = m * w
    cells = np.flatnonzero((product > 0) & ~np.isnan(product))
    assert w[cells].max() / w[cells].min() > 2.0  # the weights matter on this map
    path = tmp_path / "place.bin"
    _replicated(path, int(st.species_id[i]), int(st.age_dt[i]), 40_000)  # unlocated: placed
    out = _run_probe(gradient["dir"] / "model.bin", path, gated["step"], tmp_path / "place_out.bin")
    assert not out.get("move.is_out").any()
    obs = _counts(out, gated["grid"].nx, cells)  # asserts every school landed on such a cell
    expected = product[cells] / product[cells].sum() * obs.sum()
    assert chisquare(*_pooled(obs, expected)).pvalue > 1e-4
    plain = m[cells] / m[cells].sum() * obs.sum()  # control: the map alone, ungated
    assert chisquare(*_pooled(obs, plain)).pvalue < 1e-6


# ------------------------------------------------------------------ the all-zero guard


def test_a_field_that_excludes_every_cell_moves_schools_as_with_the_gate_off(gated, tmp_path):
    """Weight 0 everywhere (salinity below s_low): no map has a positive product, production
    falls back to the ungated walk, and the C# moves equal the gate-off run's, cell for cell."""
    cfg = copy.copy(gated["config"])
    cfg.salinity_field = PhysicalData.from_constant(cfg.salinity_gate_s_low - 1.0)
    excluded = build_bundle(cfg, gated["raw"], gated["grid"], seed=11)
    assert not excluded.get("movement.salinity.wmax").any()
    excluded.save(tmp_path / "excluded.bin")
    state = gated["dir"] / "state.bin"
    res = _gate_probe(tmp_path / "excluded.bin", state, gated["step"], tmp_path / "g.bin")
    assert not np.isin(res.get("gate.branch"), (2, 4)).any()  # nothing is gated
    cfg_off = copy.copy(gated["config"])  # the same config with the gate switched off
    cfg_off.salinity_gate_enabled = False
    build_bundle(cfg_off, gated["raw"], gated["grid"], seed=11).save(tmp_path / "off.bin")
    a = _run_probe(tmp_path / "excluded.bin", state, gated["step"], tmp_path / "a.bin")
    b = _run_probe(tmp_path / "off.bin", state, gated["step"], tmp_path / "b.bin")
    for name in ("move.cell_x", "move.cell_y", "move.is_out"):
        assert np.array(a.get(name)).tobytes() == np.array(b.get(name)).tobytes(), name
    # Production takes the plain walk too: an index draw, not a uniform.
    old_field = gated["config"].salinity_field
    gated["config"].salinity_field = cfg.salinity_field
    try:
        _, cur, same, gate_args = _kernel_args(gated, gate=True)
        i = int(np.flatnonzero(gated["gate"].get("gate.branch") == 4)[0])
        k = gated["k_of"][i]
        assert _production_map_branch(gated, k, i, cur, same, gate_args)[0] in ("walk", "stay")
    finally:
        gated["config"].salinity_field = old_field


# ------------------------------------------------------------------ end to end


def _eec_bundle(d: Path, gate: bool, threads: int = 1) -> tuple[Bundle, list[str]]:
    """EEC with minimal outputs; with ``gate``, every species gated on a synthetic salinity file
    (24 frames on the model grid, values straddling the gate's 3 to 6 window)."""
    import xarray as xr

    raw = _raw("eec_full", d) | {"output.minimal.enabled": "true", "simulation.time.nyear": "2"}
    if gate:
        nc = d / "salinity.nc"
        if not nc.exists():
            data = np.random.default_rng(5).uniform(2.0, 7.0, (24, 22, 45))
            ds = xr.Dataset({"so": (["time", "y", "x"], data)})
            ds.to_netcdf(nc)
            ds.close()
        raw |= {
            "movement.salinity.gate.enabled": "true",
            "movement.salinity.field.file": str(nc),
            "movement.salinity.field.varname": "so",
        } | {f"movement.salinity.gate.species.enabled.sp{sp}": "true" for sp in range(14)}
    config, grid, *_ = PythonEngine()._prepare_run(raw, 0)
    return build_bundle(config, raw, grid, seed=6, threads=threads), required_features(config, raw)


def _run(b: Bundle, d: Path, name: str) -> bytes:
    b.save(d / f"{name}.bin")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.PipelineProbe.Run(sys.argv[2], sys.argv[3])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(d / f"{name}.bin"), str(d / f"{name}.out")],
        capture_output=True, text=True, timeout=1200, cwd=ROOT,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = Bundle.load(d / f"{name}.out")
    assert out.get("out.summary")[0] == 0  # completed
    names = ("out.biomass", "out.abundance", "out.mortality", "out.yield")
    return b"".join(np.ascontiguousarray(out.get(n)).tobytes() for n in names)


def test_gated_runs_are_deterministic_and_differ_from_ungated_ones(tmp_path):
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    bundle, features = _eec_bundle(tmp_path, gate=True)
    assert features == ["core", "salinity_gate"]
    assert bundle.get("movement.salinity.weight").shape == (24, 22 * 45)
    runs = {t: _run(_eec_bundle(tmp_path, True, t)[0], tmp_path, f"on{t}") for t in (1, 2, 14)}
    assert runs[1] == runs[2] == runs[14]
    plain, plain_features = _eec_bundle(tmp_path, gate=False)
    assert plain_features == ["core"]
    assert not [n for n in plain.names() if n.startswith("movement.salinity.")]
    assert _run(plain, tmp_path, "off") != runs[1]
