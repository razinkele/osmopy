"""Plan task 1.6: C# movement against the Python engine's production movement.

States are captured at ``_movement`` entry of a real run (the spy also hands over production's
``map_sets`` and flattened maps). ``MovementProbe`` moves them in C# (probe subprocess; the CLR
never enters pytest).

Tier E, against production code driven by SCRIPTED draws:
- the map index and same-map flag equal ``_precompute_map_indices``;
- each map school's branch (out, place, walk, stranded) and, for walkers, the candidate set
  equal what ``_map_move_batch_numba`` (its ``py_func``, with ``np.random`` scripted) does;
- random walkers' candidate sets equal ``random_walk``'s (``rng.integers`` scripted).
The scripted index ``k = 0..n-1`` makes production enumerate its own candidates; sets are
compared, not orders (the two Python walks enumerate in different orders).

Tier S: placement and walk frequencies (chi-square, each with a control arm that must reject).

Synthetic cases cover what no shipped config shows: a proportional map (every shipped map is
presence/absence), a positive land cell (placement ignores the ocean mask), a map with no
placeable cell (10,000 failures), an ``ncell`` patch (a school placed this step does not walk),
pre-set ``is_out`` flags, and the same-map guards. Baltic is compared with its salinity gate off
here; the gate (task 2.4) has its own file, tests/test_csharp_salinity_gate.py, and a must-differ
arm here shows production's gated walk is weighted.

The test needs the built engine. It skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chisquare

import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.processes import movement
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import required_features

ROOT = Path(__file__).resolve().parents[1]
CASES = {"eec_full": 30, "bay_of_biscay": 9, "baltic": 129}  # BoB step 9: 160 unlocated new eggs
GATE_OFF = {"movement.salinity.gate.enabled": "false"}
SEED = 20261118
N_SAMPLE = 300  # walkers whose full candidate set is enumerated (every walker's COUNT is checked)


class _Stop(Exception):
    pass


def _raw(demo: str, d: Path, extra: dict | None = None) -> dict:
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, d)["config_file"])))
    return raw | (extra or {})


def _capture(raw: dict, step: int) -> dict:
    """Production's state, map sets and flattened maps at ``_movement`` entry of ``step``."""
    raw = dict(raw)
    raw["output.minimal.enabled"] = "true"
    raw["simulation.time.nyear"] = str(step // 24 + 1)
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    got: dict = {}
    real = sim._movement
    sig = inspect.signature(real)

    def spy(*args, **kwargs):
        a = sig.bind(*args, **kwargs).arguments
        if a["step"] == step:
            got.update(
                state=copy.deepcopy(a["state"]),
                map_sets=a.get("map_sets"),
                flat=a.get("flat_map_data"),
            )
            raise _Stop
        return real(*args, **kwargs)

    sim._movement = spy
    try:
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    except _Stop:
        pass
    finally:
        sim._movement = real
    got.update(config=config, grid=grid, raw=raw, step=step)
    return got


def _write_state(path: Path, species, age, cx, cy, is_out) -> None:
    b = Bundle(1)
    for name, arr, dt in [
        ("species_id", species, np.int32), ("age_dt", age, np.int32), ("cell_x", cx, np.int32),
        ("cell_y", cy, np.int32), ("is_out", is_out, np.uint8),
    ]:  # fmt: skip
        b.set_array(f"state.{name}", np.ascontiguousarray(arr, dtype=dt))
    b.save(path)


def _state_of(st, is_out=None):
    return (st.species_id, st.age_dt, st.cell_x, st.cell_y, st.is_out if is_out is None else is_out)


def _run_probe(
    model: Path, state: Path, step: int, out: Path, seed: int = SEED, regroup: bool = False
) -> Bundle:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.MovementProbe.Run(sys.argv[2], sys.argv[3], int(sys.argv[4]), "
        "int(sys.argv[5]), sys.argv[6] == '1', sys.argv[7])"
    )
    args = [
        sys.executable,
        "-c",
        code,
        str(ROOT),
        str(model),
        str(state),
        str(step),
        str(seed),
        "1" if regroup else "0",
        str(out),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return Bundle.load(out)


def _candidates(res: Bundle, i: int) -> set[tuple[int, int]]:
    off, cells = res.get("move.cand.offsets"), res.get("move.cand.cells")
    return (
        set()
        if off[i + 1] == off[i]
        else {(int(c % NX[0]), int(c // NX[0])) for c in cells[off[i] : off[i + 1]]}
    )


NX = [0]  # grid width for _candidates, set per demo before use


# ------------------------------------------------------------------------- production oracles


class _PlacementBranch(Exception):
    pass


class _WeightedWalk(Exception):
    pass


class _ScriptedRandom:
    """``np.random`` for the map kernel: a placement draw (over the whole grid) stops the run, a
    walk draw returns the scripted index and records production's candidate count, a uniform
    (only the salinity-gated weighted walk draws one outside placement) stops the run."""

    def __init__(self, n_cells: int, k: int):
        self.n_cells, self.k, self.highs = n_cells, k, []

    def seed(self, _):
        pass

    def randint(self, low, high):
        if high == self.n_cells:
            raise _PlacementBranch
        self.highs.append(int(high))
        return min(self.k, int(high) - 1)

    def random(self):
        raise _WeightedWalk


class _ShimNumpy:
    def __init__(self, random):
        self.random = random

    def __getattr__(self, name):
        return getattr(np, name)


def _kernel_args(got: dict, gate: bool):
    config, grid, st = got["config"], got["grid"], got["state"]
    uses_maps = config.movement_is_maps[st.species_id]
    cur, same = movement._precompute_map_indices(
        st.species_id, st.age_dt, uses_maps, got["map_sets"], got["step"]
    )
    if gate:
        sal_w = movement._movement_salinity_weight(config, grid, got["step"])
        assert sal_w is not None
        gate_args = (sal_w, True, config.salinity_gate_species)
    else:
        gate_args = (np.zeros((1, 1)), False, np.zeros(config.n_species, dtype=np.bool_))
    return np.flatnonzero(uses_maps).astype(np.int32), cur, same, gate_args


def _production_map_branch(got: dict, k: int, school: int, cur, same, gate_args, script_k: int = 0):
    """One map school through ``_map_move_batch_numba.py_func`` with scripted draws:
    (branch, production candidate count, destination)."""
    config, grid, st = got["config"], got["grid"], got["state"]
    script = _ScriptedRandom(grid.ny * grid.nx, script_k)
    ox, oy, oo = st.cell_x.copy(), st.cell_y.copy(), np.zeros(len(st), dtype=np.bool_)
    old = movement.np
    movement.np = _ShimNumpy(script)
    try:
        movement._map_move_batch_numba.py_func(
            0, np.array([school], dtype=np.int32), cur[k : k + 1], same[k : k + 1],
            st.cell_x.copy(), st.cell_y.copy(), st.species_id, *got["flat"], grid.ocean_mask,
            config.random_walk_range.astype(np.int32), grid.ny, grid.nx, ox, oy, oo, *gate_args,
        )  # fmt: skip
    except _PlacementBranch:
        return "place", 0, None
    except _WeightedWalk:
        return "weighted", 0, None
    finally:
        movement.np = old
    if not script.highs:
        return ("out" if oo[school] else "stay"), 0, None
    return "walk", script.highs[0], (int(ox[school]), int(oy[school]))


class _DealRng:
    """``rng`` for ``random_walk``: always deal the j-th candidate (clamped)."""

    def __init__(self, j: int):
        self.j = j

    def integers(self, low, high, size=None):
        return np.minimum(self.j, np.asarray(high) - 1)


def _production_random_candidates(got: dict) -> dict[int, set[tuple[int, int]]]:
    config, grid, st = got["config"], got["grid"], got["state"]
    uses_random = config.movement_is_random[st.species_id]
    walk = np.where(uses_random & (st.cell_x >= 0), config.random_walk_range[st.species_id], 0)
    walkers = np.flatnonzero((walk > 0) & (st.cell_x >= 0))
    sets: dict[int, set[tuple[int, int]]] = {int(i): set() for i in walkers}
    r_max = int(walk.max())
    for j in range((2 * r_max + 1) ** 2):
        moved = movement.random_walk(st, grid, walk.astype(np.int32), _DealRng(j))
        for i in walkers:
            sets[int(i)].add((int(moved.cell_x[i]), int(moved.cell_y[i])))
    return sets


# ------------------------------------------------------------------------------------ fixture


@pytest.fixture(scope="module")
def moved(tmp_path_factory) -> dict:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    out = {}
    for demo, step in CASES.items():
        d = tmp_path_factory.mktemp(demo)
        got = _capture(_raw(demo, d, GATE_OFF if demo == "baltic" else None), step)
        build_bundle(got["config"], got["raw"], got["grid"], seed=11).save(d / "model.bin")
        _write_state(d / "state.bin", *_state_of(got["state"]))
        got["res"] = _run_probe(d / "model.bin", d / "state.bin", step, d / "out.bin")
        got["dir"] = d
        out[demo] = got
    return out


# ---------------------------------------------------------------------------------- tier E


@pytest.mark.parametrize("demo", ["eec_full", "baltic"])
def test_map_index_and_same_map_equal_the_engine(moved, demo):
    got = moved[demo]
    schools, cur, same, _ = _kernel_args(got, gate=False)
    res = got["res"]
    assert np.array_equal(res.get("move.map_index")[schools], cur)
    assert np.array_equal(res.get("move.same_map")[schools].astype(bool), same)
    assert same.any() and (~same).any()


def test_same_map_guards_at_step_0_age_0_and_past_the_table(moved, tmp_path):
    """Located schools at step 0, age 0, and ages past the index table: the guards no captured
    state exercises in full."""
    got = moved["eec_full"]
    st = copy.deepcopy(got["state"])
    maps = np.flatnonzero(got["config"].movement_is_maps[st.species_id])
    st.age_dt[maps[0::3]] = 0
    st.age_dt[maps[1::3]] = 10_000  # far past every species' table
    edge = maps[2::3]  # the FIRST age past the table: off-by-one bounds read the next table
    st.age_dt[edge] = [got["map_sets"][int(sp)].index_maps.shape[0] for sp in st.species_id[edge]]
    _write_state(tmp_path / "state.bin", *_state_of(st))
    for step in (0, got["step"]):
        res = _run_probe(
            got["dir"] / "model.bin", tmp_path / "state.bin", step, tmp_path / f"out{step}.bin"
        )
        uses_maps = got["config"].movement_is_maps[st.species_id]
        cur, same = movement._precompute_map_indices(
            st.species_id, st.age_dt, uses_maps, got["map_sets"], step
        )
        assert np.array_equal(res.get("move.map_index")[maps], cur), step
        assert np.array_equal(res.get("move.same_map")[maps].astype(bool), same), step
        assert (cur[1::3] == -1).all() and (cur[2::3] == -1).all() and not same[0::3].any()
        if step == 0:
            assert not same.any()
        far = np.concatenate([maps[1::3], maps[2::3]])  # past the table: out of the domain
        assert (res.get("move.is_out")[far] == 1).all() and (
            res.get("move.cell_x")[far] == -1
        ).all()


@pytest.mark.parametrize("demo", ["eec_full", "baltic"])
def test_map_branches_and_walk_candidates_equal_production(moved, demo):
    got = moved[demo]
    NX[0] = got["grid"].nx
    res, st = got["res"], got["state"]
    schools, cur, same, gate_args = _kernel_args(got, gate=False)
    n_cand = np.diff(res.get("move.cand.offsets"))
    cx, cy, out = res.get("move.cell_x"), res.get("move.cell_y"), res.get("move.is_out")
    stack = got["flat"][0]
    offsets = got["flat"][3]
    branches: dict[str, list[int]] = {"out": [], "stay": [], "place": [], "walk": []}
    walkers = []
    for k, i in enumerate(schools):
        branch, n, _ = _production_map_branch(got, k, int(i), cur, same, gate_args)
        branches[branch].append(int(i))
        if branch == "out":
            assert out[i] == 1 and cx[i] == -1, i
        elif branch == "stay":
            assert (
                n_cand[i] == 0 and out[i] == 0 and (cx[i], cy[i]) == (st.cell_x[i], st.cell_y[i])
            ), i
        elif branch == "place":
            g = offsets[st.species_id[i]] + cur[k]
            assert out[i] == 0 and n_cand[i] == 0 and stack[g, cy[i], cx[i]] > 0, i
        else:
            assert n_cand[i] == n and out[i] == 0, (i, n, n_cand[i])
            walkers.append((k, int(i), n))
    for k, i, n in walkers[:: max(1, len(walkers) // N_SAMPLE)]:
        want = {
            _production_map_branch(got, k, i, cur, same, gate_args, script_k=j)[2] for j in range(n)
        }
        assert _candidates(res, i) == want, i
        assert (int(cx[i]), int(cy[i])) in want
    assert branches["walk"] and branches["place"]
    if demo == "eec_full":
        assert branches["out"]  # EEC's null maps


def test_random_walk_candidates_equal_production(moved):
    got = moved["bay_of_biscay"]
    NX[0] = got["grid"].nx
    res, st = got["res"], got["state"]
    want = _production_random_candidates(got)
    assert len(want) > 100
    for i, cells in want.items():
        assert _candidates(res, i) == cells, i
        assert (int(res.get("move.cell_x")[i]), int(res.get("move.cell_y")[i])) in cells
    unlocated = np.flatnonzero(st.cell_x < 0)
    assert unlocated.size  # new eggs: placed on the domain (the whole ocean here)
    assert (res.get("move.cell_x")[unlocated] >= 0).all()


@pytest.mark.parametrize("demo", sorted(CASES))
def test_no_school_lands_on_land_on_shipped_configs(moved, demo):
    got = moved[demo]
    cx, cy = got["res"].get("move.cell_x"), got["res"].get("move.cell_y")
    located = cx >= 0
    assert located.sum() > 100 and got["grid"].ocean_mask[cy[located], cx[located]].all()


@pytest.mark.parametrize("demo", ["eec_full", "bay_of_biscay"])
def test_moves_are_keyed_by_school_id_not_store_order(moved, demo, tmp_path):
    got = moved[demo]
    again = _run_probe(
        got["dir"] / "model.bin",
        got["dir"] / "state.bin",
        got["step"],
        tmp_path / "r.bin",
        regroup=True,
    )
    for name in ("move.cell_x", "move.cell_y", "move.is_out"):
        assert np.array_equal(again.get(name), got["res"].get(name)), name
    other = _run_probe(
        got["dir"] / "model.bin",
        got["dir"] / "state.bin",
        got["step"],
        tmp_path / "s.bin",
        seed=SEED + 1,
    )
    assert not np.array_equal(other.get("move.cell_x"), got["res"].get("move.cell_x"))


def test_is_out_is_written_for_map_schools_only(moved, tmp_path):
    """At ``_movement`` entry the flag is always false, so pre-set it: map schools that are placed,
    walk or are stranded must clear it (production's kernel writes False); random-method schools
    must keep it (production's random branch never writes it)."""
    for demo in ("eec_full", "bay_of_biscay"):
        got = moved[demo]
        st = got["state"]
        _write_state(
            tmp_path / f"{demo}.bin", *_state_of(st, is_out=np.ones(len(st), dtype=np.uint8))
        )
        res = _run_probe(
            got["dir"] / "model.bin",
            tmp_path / f"{demo}.bin",
            got["step"],
            tmp_path / f"{demo}_o.bin",
        )
        flag, base = res.get("move.is_out"), got["res"].get("move.is_out")
        if demo == "eec_full":
            assert np.array_equal(flag, base) and (flag == 0).any()  # same as with flags clear
        else:
            assert (flag == 1).all()


# -------------------------------------------------------------------------- tier S + synthetic


def _replicated(path: Path, sp: int, age: int, n: int, cell=(-1, -1)) -> None:
    _write_state(
        path, np.full(n, sp), np.full(n, age), np.full(n, cell[0]), np.full(n, cell[1]), np.zeros(n)
    )


def _counts(res: Bundle, nx: int, cells: np.ndarray) -> np.ndarray:
    landed = res.get("move.cell_y") * nx + res.get("move.cell_x")
    assert np.isin(landed, cells).all()
    return np.array([(landed == c).sum() for c in cells])


def _live_map_school(got: dict, sp: int, step: int) -> tuple[int, int]:
    """(age, global map) of a non-null map of species ``sp`` at ``step``."""
    ms = got["map_sets"][sp]
    off = got["flat"][3][sp]
    for age in range(ms.index_maps.shape[0]):
        k = ms.index_maps[age, step]
        if k >= 0 and ms.maps[k] is not None:
            return age, int(off + k)
    raise AssertionError(sp)


def test_walk_choice_is_uniform_over_candidates(moved, tmp_path):
    for demo in ("eec_full", "bay_of_biscay"):
        got = moved[demo]
        res, st, nx = got["res"], got["state"], got["grid"].nx
        n_cand = np.diff(res.get("move.cand.offsets"))
        i = int(np.argmax(n_cand))  # the walker with the most candidates
        off = res.get("move.cand.offsets")
        cells = res.get("move.cand.cells")[off[i] : off[i + 1]]
        path = tmp_path / f"{demo}_w.bin"
        _replicated(
            path,
            int(st.species_id[i]),
            int(st.age_dt[i]),
            20_000,
            (int(st.cell_x[i]), int(st.cell_y[i])),
        )
        obs = _counts(
            _run_probe(got["dir"] / "model.bin", path, got["step"], tmp_path / f"{demo}_wo.bin"),
            nx,
            cells,
        )
        assert chisquare(obs).pvalue > 1e-4, (demo, obs)
        skew = np.ones(len(cells))
        skew[0] = 2.0  # control: one candidate twice as likely must be rejected
        assert chisquare(obs, skew / skew.sum() * obs.sum()).pvalue < 1e-6, demo


def test_presence_map_placement_is_uniform_over_placeable_cells(moved, tmp_path):
    got = moved["eec_full"]
    nx, step = got["grid"].nx, got["step"]
    sp = 0
    age, g = _live_map_school(got, sp, step)
    stack = got["flat"][0][g]
    assert got["flat"][1][g] == 0.0  # presence/absence
    cells = np.flatnonzero((stack > 0) & ~np.isnan(stack))
    _replicated(tmp_path / "p.bin", sp, age, 20_000)
    obs = _counts(
        _run_probe(got["dir"] / "model.bin", tmp_path / "p.bin", step, tmp_path / "po.bin"),
        nx,
        cells,
    )
    assert chisquare(obs).pvalue > 1e-4
    ramp = np.linspace(1.0, 3.0, len(cells))  # control: a proportional law must be rejected
    assert chisquare(obs, ramp / ramp.sum() * obs.sum()).pvalue < 1e-6


def test_proportional_map_placement_follows_the_map(tmp_path):
    """Every shipped map is presence/absence (max_proba 0). Rewrite EEC's map 0 with values in
    (0, 1) on its presence cells, through the config files, so MovementMapSet itself classes it
    proportional; placement frequency must then follow the values."""
    if not csharp_engine_available()[0]:
        pytest.skip("C# engine unavailable")
    raw = _raw("eec_full", tmp_path)
    map_file = Path(raw["_osmose.config.dir"]) / raw["movement.file.map0"]
    rows = [line.split(";") for line in map_file.read_text().split("\n") if line.strip()]
    cols = [j for r in rows for j, v in enumerate(r) if v.strip() == "1"]
    lo, hi = min(cols), max(cols)  # stretch 0.05..0.95 over the presence columns

    def value(j: int) -> str:
        return f"{0.05 + 0.9 * (j - lo) / max(1, hi - lo):.6f}"

    new = [";".join(value(j) if v.strip() == "1" else v for j, v in enumerate(r)) for r in rows]
    map_file.write_text("\n".join(new) + "\n")
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    b = build_bundle(config, raw, grid, seed=11)
    b.save(tmp_path / "model.bin")
    sp = config.species_names.index(raw["movement.species.map0"])
    step = 30
    shape, idx_off = (
        b.get("movement.maps.index.shape")[sp],
        b.get("movement.maps.index.offsets")[sp],
    )
    table = b.get("movement.maps.index")[idx_off : idx_off + shape[0] * shape[1]].reshape(shape)
    age = 24
    g = int(b.get("movement.maps.species_offset")[sp] + table[age, step])
    max_p = b.get("movement.maps.max_proba")[g]
    stack = b.get("movement.maps.stack")[g]
    assert 0 < max_p < 1  # really proportional, classed so by MovementMapSet
    cells = np.flatnonzero((stack > 0) & ~np.isnan(stack))
    values = stack.ravel()[cells]
    assert values.min() < 0.2 and values.max() > 0.8
    _replicated(tmp_path / "p.bin", sp, age, 40_000)
    obs = _counts(
        _run_probe(tmp_path / "model.bin", tmp_path / "p.bin", step, tmp_path / "po.bin"),
        grid.nx,
        cells,
    )
    assert chisquare(obs, values / values.sum() * obs.sum()).pvalue > 1e-4
    assert chisquare(obs).pvalue < 1e-6  # control: uniform over the placeable cells is rejected


def _with_map(got: dict, g: int, new_map: np.ndarray, path: Path) -> None:
    """The captured model bundle with stacked map ``g`` replaced (a bundle never overwrites an
    entry, so every entry is copied into a fresh one)."""
    src = Bundle.load(got["dir"] / "model.bin")
    stack = np.array(src.get("movement.maps.stack"))
    stack[g] = new_map
    b = Bundle(src.contract_version)
    for name in src.names():
        v = stack if name == "movement.maps.stack" else src.get(name)
        if isinstance(v, np.ndarray):
            b.set_array(name, np.array(v))
        elif isinstance(v, float):
            b.set_float(name, v)
        else:
            b.set_int(name, int(v))
    b.save(path)


def test_placement_ignores_the_ocean_mask(moved, tmp_path):
    """Production accepts any placeable map cell, land or not. One positive cell, on land."""
    got = moved["eec_full"]
    grid, step = got["grid"], got["step"]
    age, g = _live_map_school(got, 0, step)
    land_y, land_x = np.argwhere(~grid.ocean_mask)[0]
    m = np.zeros((grid.ny, grid.nx))
    m[land_y, land_x] = 1.0
    _with_map(got, g, m, tmp_path / "model.bin")
    _replicated(tmp_path / "s.bin", 0, age, 50)
    res = _run_probe(tmp_path / "model.bin", tmp_path / "s.bin", step, tmp_path / "o.bin")
    assert (res.get("move.cell_x") == land_x).all() and (res.get("move.cell_y") == land_y).all()
    assert (res.get("move.is_out") == 0).all()
    # Production does the same with its own kernel and real draws: school 0, made an unlocated
    # species-0 school on the same map (its local index passed in), must land on that land cell.
    flat = list(got["flat"])
    flat[0] = flat[0].copy()
    flat[0][g] = m
    st = got["state"]
    sp_ids, cx, cy = st.species_id.copy(), st.cell_x.copy(), st.cell_y.copy()
    sp_ids[0], cx[0], cy[0] = 0, -1, -1
    ox, oy, oo = cx.copy(), cy.copy(), np.zeros(len(st), dtype=np.bool_)
    movement._map_move_batch_numba.py_func(
        1, np.array([0], np.int32), np.array([g - flat[3][0]], np.int32), np.array([False]),
        cx, cy, sp_ids, *flat, grid.ocean_mask,
        got["config"].random_walk_range.astype(np.int32), grid.ny, grid.nx, ox, oy, oo,
        np.zeros((1, 1)), False, np.zeros(got["config"].n_species, dtype=np.bool_),
    )  # fmt: skip
    assert (ox[0], oy[0], oo[0]) == (land_x, land_y, False)


def test_ten_thousand_failures_send_the_school_out(moved, tmp_path):
    got = moved["eec_full"]
    grid, step = got["grid"], got["step"]
    age, g = _live_map_school(got, 0, step)
    _with_map(
        got, g, np.zeros((grid.ny, grid.nx)), tmp_path / "model.bin"
    )  # not null, never placeable
    _replicated(tmp_path / "s.bin", 0, age, 5)
    res = _run_probe(tmp_path / "model.bin", tmp_path / "s.bin", step, tmp_path / "o.bin")
    assert (res.get("move.is_out") == 1).all() and (res.get("move.cell_x") == -1).all()


def test_random_placement_stays_in_its_patch_and_does_not_walk(tmp_path):
    """An ncell patch (no shipped config has one): placed schools must all be inside the patch,
    uniformly; with a walk range of at least 1, a walk after placing would leave it."""
    if not csharp_engine_available()[0]:
        pytest.skip("C# engine unavailable")
    raw = _raw("bay_of_biscay", tmp_path, {"movement.distribution.ncell.sp0": "20"})
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    assert config.random_walk_range[0] >= 1
    b = build_bundle(config, raw, grid, seed=11)
    b.save(tmp_path / "model.bin")
    off = b.get("movement.random.domain.offsets")
    patch = b.get("movement.random.domain.cells")[off[0] : off[1]]
    assert len(patch) == 20
    cells = patch[:, 1] * grid.nx + patch[:, 0]
    _replicated(tmp_path / "s.bin", 0, 30, 8_000)
    obs = _counts(
        _run_probe(tmp_path / "model.bin", tmp_path / "s.bin", 5, tmp_path / "o.bin"),
        grid.nx,
        cells,
    )
    assert chisquare(obs).pvalue > 1e-4


def test_salinity_gate_changes_the_walk(tmp_path):
    """Must-differ arm for the Baltic comparisons above, which run with the gate off: with the
    gate on, production's walk for gated species is WEIGHTED (it draws a uniform, not an index).
    The gate itself (task 2.4) is tested in tests/test_csharp_salinity_gate.py."""
    if not csharp_engine_available()[0]:
        pytest.skip("C# engine unavailable")
    got = _capture(_raw("baltic", tmp_path), CASES["baltic"])
    assert "salinity_gate" in required_features(got["config"], got["raw"])
    schools, cur, same, gate_args = _kernel_args(got, gate=True)
    gated = got["config"].salinity_gate_species
    weighted = 0
    for k, i in enumerate(schools):
        if gated[got["state"].species_id[i]] and same[k] and got["state"].cell_x[i] >= 0:
            weighted += (
                _production_map_branch(got, k, int(i), cur, same, gate_args)[0] == "weighted"
            )
            if weighted:
                break
    assert weighted


def test_null_map_sends_located_schools_out(moved, tmp_path):
    """A null map sends a school out even when it is located and its map is unchanged. A stored
    null map is all zeros, so skipping the null test would leave such a school walking with no
    candidate, still in the domain. No real trajectory is located on a null map, so place them;
    and age the null-map schools forward (age + d at step + d) until their null map persists."""
    base = moved["eec_full"]
    grid = base["grid"]
    oy, ox = np.argwhere(grid.ocean_mask)[0]
    null, off = base["flat"][2], base["flat"][3]
    for d in range(1, 25):
        got = copy.copy(base)
        b = base["state"]
        st = b.replace(age_dt=b.age_dt + d, cell_x=b.cell_x.copy(), cell_y=b.cell_y.copy())
        got["state"], got["step"] = st, base["step"] + d
        schools, cur, same, gate_args = _kernel_args(got, gate=False)
        on_null = [
            int(i) for k, i in enumerate(schools)
            if cur[k] >= 0 and null[off[st.species_id[i]] + cur[k]] and same[k]
        ]  # fmt: skip
        if on_null:
            break
    else:
        raise AssertionError("no school stays on a null map")
    st.cell_x[on_null], st.cell_y[on_null] = ox, oy
    k_of = {int(i): k for k, i in enumerate(schools)}
    for i in on_null:
        assert _production_map_branch(got, k_of[i], i, cur, same, gate_args)[0] == "out", i
    _write_state(tmp_path / "s.bin", *_state_of(st))
    res = _run_probe(base["dir"] / "model.bin", tmp_path / "s.bin", got["step"], tmp_path / "o.bin")
    assert (res.get("move.is_out")[on_null] == 1).all() and (
        res.get("move.cell_x")[on_null] == -1
    ).all()


def test_map_walk_excludes_land_where_the_map_is_positive(moved, tmp_path):
    """Shipped maps hold -99 on land, so land is never placeable and the walk's ocean test is
    invisible. Make the land around one coastal walker positive on its map: production's walk
    still excludes it, and so must C#."""
    got = copy.copy(moved["eec_full"])
    grid, st = got["grid"], got["state"]
    NX[0] = grid.nx
    schools, cur, same, gate_args = _kernel_args(got, gate=False)
    off, r = got["flat"][3], got["config"].random_walk_range
    for k, i in enumerate(schools):
        sp, x, y = int(st.species_id[i]), int(st.cell_x[i]), int(st.cell_y[i])
        if not same[k] or x < 0 or cur[k] < 0:
            continue
        ys = slice(max(0, y - r[sp]), y + r[sp] + 1)
        xs = slice(max(0, x - r[sp]), x + r[sp] + 1)
        if (~grid.ocean_mask[ys, xs]).any():
            break
    else:
        raise AssertionError("no coastal walker")
    g = int(off[sp] + cur[k])
    m = got["flat"][0][g].copy()
    land = np.zeros_like(grid.ocean_mask)
    land[ys, xs] = ~grid.ocean_mask[ys, xs]
    m[land] = 1.0
    flat = list(got["flat"])
    flat[0] = flat[0].copy()
    flat[0][g] = m
    got["flat"] = tuple(flat)
    branch, n, _ = _production_map_branch(got, k, int(i), cur, same, gate_args)
    assert branch == "walk"
    want = {
        _production_map_branch(got, k, int(i), cur, same, gate_args, script_k=j)[2]
        for j in range(n)
    }
    _with_map(moved["eec_full"], g, m, tmp_path / "model.bin")
    res = _run_probe(
        tmp_path / "model.bin", got["dir"] / "state.bin", got["step"], tmp_path / "o.bin"
    )
    assert _candidates(res, int(i)) == want
    assert not any(land[cy, cx] for cx, cy in want)  # positive land cells, all excluded
