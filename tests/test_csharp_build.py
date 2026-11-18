"""Plan task 1.2a: the bundle's header and static tables equal what the Python engine reads.

Each table is checked against the Python engine's OWN function, not just against the config
array it came from. A reference lookup written over the bundle arrays (the algorithm the C#
engine implements) must reproduce:
- ``AccessibilityMatrix.compute_school_indices`` for accessibility;
- ``compute_feeding_stages`` for feeding stages.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.state import SchoolState
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import STAGE_METRICS, build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import load_catalog, required_features

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    out = {}
    for demo in ("eec_full", "bay_of_biscay", "baltic"):
        info = osmose_demo(demo, tmp_path_factory.mktemp(demo))
        raw = dict(OsmoseConfigReader().read(Path(info["config_file"])))
        raw["output.minimal.enabled"] = "true"
        raw["simulation.time.nyear"] = "2"
        config, grid, *_ = PythonEngine()._prepare_run(raw, 7)
        out[demo] = (config, raw, grid, build_bundle(config, raw, grid, seed=7))
    return out


# ------------------------------------------------------------------ reference lookups (the C# algorithm)


def lookup_access(b: Bundle, role: str, species: np.ndarray, age_dt: np.ndarray) -> np.ndarray:
    off = b.get(f"access.{role}.offsets")
    thr = b.get(f"access.{role}.thresholds")
    idx = b.get(f"access.{role}.index")
    ndt = b.get("time.n_dt_per_year")
    out = np.full(species.size, -1, dtype=np.int32)
    for i, (sp, a) in enumerate(zip(species, age_dt, strict=True)):
        lo, hi = off[sp], off[sp + 1]
        if hi == lo:  # absent from the CSV: unresolved, the -1 "full access" rule
            continue
        age_years = float(a) / ndt
        k = int(np.searchsorted(thr[lo:hi], age_years, side="right"))
        out[i] = idx[lo + min(k, hi - lo - 1)]
    return out


def lookup_stage(b: Bundle, state: SchoolState) -> np.ndarray:
    off = b.get("predation.stage.offsets")
    thr = b.get("predation.stage.thresholds")
    metric = b.get("predation.stage.metric")
    ndt = b.get("time.n_dt_per_year")
    out = np.zeros(len(state), dtype=np.int32)
    for i in range(len(state)):
        sp = state.species_id[i]
        lo, hi = off[sp], off[sp + 1]
        m = metric[sp]
        # The contract's codes, pinned here as literals (the C# engine hard-codes them too):
        # importing the builder's STAGE_METRICS would let a wrong code pass on both sides.
        v = {
            0: state.length[i],  # size
            1: state.age_dt[i] / ndt,  # age, years
            2: state.weight[i] * 1e6,  # weight, grams
            3: state.trophic_level[i],  # tl
        }[m]
        out[i] = int(np.searchsorted(thr[lo:hi], v, side="right"))
    return out


def random_schools(config, n: int, seed: int) -> SchoolState:
    rng = np.random.default_rng(seed)
    n_total = config.n_species + config.n_background
    sp = rng.integers(0, n_total, n).astype(np.int32)
    st = SchoolState.create(n, species_id=sp)
    lifespan = np.asarray(config.lifespan_dt)
    return st.replace(
        age_dt=rng.integers(0, np.maximum(lifespan[sp], 1)).astype(np.int32),
        length=rng.uniform(0.05, 1.2, n) * np.asarray(config.linf)[sp],
        weight=rng.uniform(1e-9, 1e-2, n),
        trophic_level=rng.uniform(1.0, 5.0, n),
    )


# ------------------------------------------------------------------ tests


def test_bundle_round_trips_through_the_file_format(runs, tmp_path):
    b = runs["eec_full"][3]
    b.save(tmp_path / "eec.bin")
    loaded = Bundle.load(tmp_path / "eec.bin")
    assert loaded.names() == b.names()
    for name in b.names():
        a, c = np.asarray(loaded.get(name)), np.asarray(b.get(name))
        assert a.dtype == c.dtype and a.shape == c.shape, name
        assert a.tobytes() == c.tobytes(), name  # bit-exact, NaN payloads included


@pytest.mark.parametrize("demo", ["eec_full", "bay_of_biscay", "baltic"])
def test_contract_features_are_the_capability_detectors_answer(runs, demo):
    config, raw, _, b = runs[demo]
    catalog = load_catalog()
    assert b.get("contract.features").tolist() == [
        catalog[n].id for n in required_features(config, raw)
    ]
    assert b.get("contract.features")[0] == 0  # core first


@pytest.mark.parametrize("demo", ["eec_full", "baltic"])
@pytest.mark.parametrize("role", ["prey", "pred"])
def test_access_tables_reproduce_compute_school_indices(runs, demo, role):
    config, _, _, b = runs[demo]
    st = random_schools(config, 4000, seed=1)
    want = config.stage_accessibility.compute_school_indices(
        st.species_id, st.age_dt, config.n_dt_per_year, config.all_species_names, role=role
    )
    got = lookup_access(b, role, st.species_id, st.age_dt)
    assert np.array_equal(got, want)
    assert (want >= 0).any()  # the instrument exercises real rows


def test_a_species_absent_from_the_csv_gets_an_empty_range_not_a_zero_row(runs):
    config, raw, grid, _ = runs["eec_full"]
    cfg = copy.copy(config)
    sa = copy.deepcopy(config.stage_accessibility)
    del sa._stages_by_role["prey"][3]  # species 3 now has no prey row: unresolved
    cfg.stage_accessibility = sa
    b = build_bundle(cfg, raw, grid, seed=7)
    off = b.get("access.prey.offsets")
    assert off[3] == off[4]  # empty range
    st = random_schools(cfg, 2000, seed=2)
    want = sa.compute_school_indices(
        st.species_id, st.age_dt, cfg.n_dt_per_year, cfg.all_species_names, role="prey"
    )
    got = lookup_access(b, "prey", st.species_id, st.age_dt)
    assert np.array_equal(got, want)
    assert (got[st.species_id == 3] == -1).all() and (st.species_id == 3).any()


@pytest.mark.parametrize("demo", ["eec_full", "baltic"])
def test_feeding_stage_tables_reproduce_compute_feeding_stages(runs, demo):
    config, _, _, b = runs[demo]
    st = random_schools(config, 4000, seed=3)
    assert np.array_equal(lookup_stage(b, st), compute_feeding_stages(st, config))


def test_stage_metric_codes_are_the_contract():
    assert STAGE_METRICS == {"size": 0, "age": 1, "weight": 2, "tl": 3}


def test_feeding_stages_see_a_second_stage_in_eec(runs):
    config, _, _, b = runs["eec_full"]
    st = random_schools(config, 4000, seed=4)
    assert compute_feeding_stages(st, config).max() >= 1  # the instrument can see a stage change
    assert (np.diff(b.get("predation.stage.offsets")) > 0).sum() >= 1


@pytest.mark.parametrize("demo", ["eec_full", "bay_of_biscay", "baltic"])
def test_mortality_tables_are_the_rates_the_engine_applies(runs, demo):
    config, _, _, b = runs[demo]
    n_total = config.n_species + config.n_background
    add = b.get("mortality.additional.rate_by_step")
    larva = b.get("mortality.larva.rate_by_step")
    assert add.shape == (config.n_steps, n_total)
    assert larva.shape == (config.n_steps, config.n_species)
    assert np.array_equal(add, np.tile(config.additional_mortality_rate, (config.n_steps, 1)))
    assert np.array_equal(
        larva,
        np.tile(np.asarray(config.larva_mortality_rate)[: config.n_species], (config.n_steps, 1)),
    )
    assert np.array_equal(b.get("mortality.out.rate"), config.out_mortality_rate)
    assert np.array_equal(b.get("mortality.starvation.rate_max"), config.starvation_rate_max)


@pytest.mark.parametrize("demo", ["eec_full", "bay_of_biscay", "baltic"])
def test_per_species_tables_have_n_total_rows_and_the_engine_values(runs, demo):
    config, _, grid, b = runs[demo]
    n_total = config.n_species + config.n_background
    for name in ("linf", "k", "t0", "egg_size", "condition_factor", "allometric_power", "lmax"):
        got = b.get(f"growth.{name}")
        assert got.shape == (n_total,), name
        assert np.array_equal(got, np.asarray(getattr(config, name)), equal_nan=True), name
    assert b.get("grid.ocean_mask").shape == (grid.ny, grid.nx)
    assert b.get("time.n_subdt") == config.mortality_subdt
    assert b.get("species.n_schools").shape == (config.n_species,)


def test_the_csharp_engine_accepts_the_header_and_refuses_only_core(runs, tmp_path):
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    path = tmp_path / "eec.bin"
    runs["eec_full"][3].save(path)
    code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "e = bridge.load(); c = e.SimulationConfig.Load(sys.argv[2])\n"
        "try:\n    e.EngineParameters.From(c); print(json.dumps(['accepted']))\n"
        "except Exception as exc:\n    print(json.dumps([type(exc).__name__, str(exc).splitlines()[0]]))\n"
        "finally:\n    c.Dispose()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    kind, message = json.loads(proc.stdout.strip().splitlines()[-1])
    assert kind == "UnsupportedFeatureException", (
        message
    )  # header valid; refusal is the feature check
    assert "core (not implemented yet; phase 1)" in message


# ------------------------------------------------------------------ 1.2b: fishing and reproduction


def school_table(config, grid, n: int, seed: int) -> SchoolState:
    """Random schools with everything _precompute_effective_rates reads."""
    rng = np.random.default_rng(seed)
    st = random_schools(config, n, seed)
    n_focal = config.n_species
    return st.replace(
        starvation_rate=np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0, 2.0, n)),
        first_feeding_age_dt=rng.integers(0, 3, n).astype(np.int32),
        is_background=st.species_id >= n_focal,
        cell_x=rng.integers(-1, grid.nx, n).astype(np.int32),
        cell_y=rng.integers(-1, grid.ny, n).astype(np.int32),
        abundance=rng.uniform(0, 1e6, n),
    )


def reference_rates(b: Bundle, st: SchoolState, step: int):
    """The C# engine's per-school rates, computed from the bundle tables only."""
    ndt, nsub = int(b.get("time.n_dt_per_year")), int(b.get("time.n_subdt"))
    denom = ndt * nsub
    sp = st.species_id
    n_focal = int(b.get("species.n_focal"))
    exempt = st.is_background | (st.age_dt < st.first_feeding_age_dt)

    def zero_exempt(a):
        a = a.copy()
        a[exempt] = 0.0
        a[a < 0] = 0.0
        return a

    eff_s = zero_exempt(st.starvation_rate / denom)
    add = zero_exempt(b.get("mortality.additional.rate_by_step")[step][sp])
    eff_a = np.nan_to_num(add / denom, nan=0.0, posinf=0.0, neginf=0.0)
    eff_f = np.zeros(len(st))
    disc = np.zeros(len(st))
    if b.get("fishing.enabled"):
        f = b.get("fishing.rate_by_step")[step][sp]
        sel = np.ones(len(st))
        spatial = np.ones(len(st))
        stype, a50, l50 = (b.get(f"fishing.selectivity.{k}") for k in ("type", "a50", "l50"))
        index, maps = b.get("fishing.map_index"), b.get("fishing.maps")
        for i in range(len(st)):
            s = sp[i]
            if s >= n_focal:
                continue
            if stype[s] == 0:
                sel[i] = 0.0 if (st.age_dt[i] / ndt) < a50[s] else 1.0
            elif stype[s] != 1:  # logistic is a refused-until-implemented feature
                sel[i] = 0.0 if (l50[s] > 0 and st.length[i] < l50[s]) else 1.0
            if index[s] >= 0:
                cy, cx = st.cell_y[i], st.cell_x[i]
                m = maps[index[s]]
                v = m[cy, cx] if (0 <= cy < m.shape[0] and 0 <= cx < m.shape[1]) else 0.0
                spatial[i] = 0.0 if (np.isnan(v) or v <= 0) else v
        if "fishing.season_by_step" in b.names():
            eff_f = f * sel * spatial * 1.0 * b.get("fishing.season_by_step")[step][sp] / nsub
        else:
            eff_f = f * sel * spatial * 1.0 / denom
        eff_f = np.nan_to_num(zero_exempt(eff_f), nan=0.0, posinf=0.0, neginf=0.0)
        if "fishing.discard_rate" in b.names():
            disc = np.where(eff_f > 0, b.get("fishing.discard_rate")[sp], 0.0)
    return eff_s, eff_a, eff_f, disc


@pytest.mark.parametrize("demo", ["eec_full", "bay_of_biscay", "baltic"])
def test_per_school_rates_from_the_bundle_equal_precompute_effective_rates(runs, demo):
    from osmose.engine.processes.mortality import _precompute_effective_rates

    config, _, grid, b = runs[demo]
    st = school_table(config, grid, 3000, seed=11)
    nsub = config.mortality_subdt
    for step in (0, 5, 23, 24, 31):
        want = _precompute_effective_rates(st, config, nsub, step)
        got = reference_rates(b, st, step)
        for name, w, g in zip(
            ("starvation", "additional", "fishing", "discard"), want, got, strict=True
        ):
            assert np.array_equal(w, g), f"{demo} step {step}: {name} differs"
    assert (want[2] > 0).any()  # the instrument sees fishing


def test_eec_fishing_uses_seasonality_maps_and_the_age_knife_edge(runs):
    _, _, _, b = runs["eec_full"]
    assert "fishing.season_by_step" in b.names()
    assert (b.get("fishing.map_index") >= 0).all()
    assert (b.get("fishing.selectivity.type") == 0).all()
    _, _, _, bob = runs["bay_of_biscay"]
    assert "fishing.season_by_step" not in bob.names()  # the other branch of the denominator


def reference_eggs(b: Bundle, st: SchoolState, step: int) -> np.ndarray:
    n_focal = int(b.get("species.n_focal"))
    sp = st.species_id
    mature = (
        (st.length >= b.get("reproduction.maturity_size")[sp])
        & (st.age_dt >= b.get("reproduction.maturity_age_dt")[sp])
        & (st.abundance > 0)
    )
    ssb = np.zeros(n_focal)
    np.add.at(ssb, sp[mature], st.abundance[mature] * st.weight[mature])
    for s in range(n_focal):
        if ssb[s] == 0.0 and step < b.get("reproduction.seeding_max_step")[s]:
            ssb[s] = b.get("reproduction.seeding_biomass")[s]
    return (
        b.get("reproduction.sex_ratio") * b.get("reproduction.relative_fecundity") * ssb
        * b.get("reproduction.season_by_step")[step] * 1_000_000.0
    )  # fmt: skip


@pytest.mark.parametrize("demo", ["eec_full", "bay_of_biscay"])
def test_egg_production_from_the_bundle_equals_reproduction(runs, demo):
    from osmose.engine.processes.reproduction import reproduction

    config, _, grid, b = runs[demo]
    n_focal = config.n_species
    for step in (0, 7, 30):
        st = school_table(config, grid, 2000, seed=20 + step)
        st = st.replace(
            species_id=st.species_id % n_focal, is_background=np.zeros(len(st), dtype=bool)
        )
        n_eggs = reference_eggs(b, st, step)
        after = reproduction(st, config, step, np.random.default_rng(0), grid.ny, grid.nx)
        new = (
            after.species_id[len(st) :],
            after.abundance[len(st) :],
            after.length[len(st) :],
            after.weight[len(st) :],
        )
        for s in range(n_focal):
            mask = new[0] == s
            if n_eggs[s] <= 0:
                assert not mask.any()
                continue
            n_new = int(config.n_schools[s]) if n_eggs[s] >= config.n_schools[s] else 1
            assert mask.sum() == n_new
            assert np.all(new[1][mask] == n_eggs[s] / n_new)
            assert np.all(new[2][mask] == b.get("reproduction.egg_length")[s])
            assert np.all(new[3][mask] == b.get("reproduction.egg_weight")[s])


def test_discards_reach_the_rates_when_a_config_sets_them(runs):
    """No shipped demo has a discard rate above 0, so give EEC some."""
    from osmose.engine.processes.mortality import _precompute_effective_rates

    config, raw, grid, _ = runs["eec_full"]
    cfg = copy.copy(config)
    cfg.fishing_discard_rate = np.linspace(0.0, 0.6, config.n_species)
    b = build_bundle(cfg, raw, grid, seed=7)
    st = school_table(cfg, grid, 3000, seed=12)
    want = _precompute_effective_rates(st, cfg, cfg.mortality_subdt, 9)
    got = reference_rates(b, st, 9)
    assert np.array_equal(want[3], got[3]) and (want[3] > 0).any()


def test_each_species_gets_its_own_fishing_map(runs):
    """EEC's 14 maps are identical, so a mixed-up map index would pass on them. Give each
    species a different map."""
    from osmose.engine.processes.mortality import _precompute_effective_rates

    config, raw, grid, _ = runs["eec_full"]
    cfg = copy.copy(config)
    rng = np.random.default_rng(5)
    cfg.fishing_spatial_maps = [
        rng.uniform(-0.2, 2.0, (grid.ny, grid.nx)) for _ in range(config.n_species)
    ]
    b = build_bundle(cfg, raw, grid, seed=7)
    st = school_table(cfg, grid, 3000, seed=13)
    want = _precompute_effective_rates(st, cfg, cfg.mortality_subdt, 4)
    assert np.array_equal(want[2], reference_rates(b, st, 4)[2]) and (want[2] > 0).any()


# ------------------------------------------------------------------ 1.2c: resources


@pytest.mark.parametrize("demo", ["eec_full", "bay_of_biscay", "baltic"])
def test_resource_capacity_from_the_bundle_equals_resource_state_update(runs, demo):
    from osmose.engine.resources import ResourceState

    config, _, grid, b = runs[demo]
    rs = ResourceState(config=config.raw_config, grid=grid)
    n = int(b.get("resources.n"))
    assert n == rs.n_resources and n > 0
    frames, fbs, abs_ = (
        b.get(f"resources.{k}") for k in ("forcing", "frame_by_step", "accessibility_by_step")
    )
    for step in (0, 1, 11, 23, 24, 40):
        rs.update(step)
        for r in range(n):
            got = frames[r, fbs[step, r]] * abs_[step, r]  # the C# engine's K
            assert got.tobytes() == rs.biomass[r].tobytes(), f"{demo} step {step} resource {r}"
    rs.close()
    assert (frames > 0).any()


def test_resource_access_rows_are_the_kernels(runs):
    from osmose.engine.processes.mortality import _precompute_resource_arrays
    from osmose.engine.resources import ResourceState

    config, _, grid, b = runs["eec_full"]
    rs = ResourceState(config=config.raw_config, grid=grid)
    *_, rows, _n = _precompute_resource_arrays(config, rs)
    assert np.array_equal(b.get("resources.access_rows"), rows) and (rows >= 0).any()
    rs.close()


def test_resource_multipliers_reach_the_capacity(runs):
    """Every shipped resource has multiplier 1.0, so give EEC's resources other ones."""
    from osmose.engine.resources import ResourceState

    config, raw, grid, _ = runs["eec_full"]
    rsc_keys = [
        k
        for k, v in raw.items()
        if k.startswith("species.type.sp") and v.strip().lower() == "resource"
    ]
    assert rsc_keys
    cfg = copy.copy(config)
    cfg.raw_config = dict(config.raw_config) | {
        k.replace("species.type.", "species.multiplier."): str(1.5 + i)
        for i, k in enumerate(rsc_keys)
    }
    b = build_bundle(cfg, raw, grid, seed=7)
    rs = ResourceState(config=cfg.raw_config, grid=grid)
    frames, fbs, abs_ = (
        b.get(f"resources.{k}") for k in ("forcing", "frame_by_step", "accessibility_by_step")
    )
    for step in (0, 13):
        rs.update(step)
        for r in range(rs.n_resources):
            assert (frames[r, fbs[step, r]] * abs_[step, r]).tobytes() == rs.biomass[r].tobytes()
    assert {s.multiplier for s in rs.species} != {1.0}
    rs.close()


# ------------------------------------------------------------------ 1.2c: movement


def engine_map_sets(config, grid) -> dict:
    """map_sets exactly as simulate() builds them."""
    from osmose.engine.movement_maps import MovementMapSet

    return {
        sp: MovementMapSet(
            config=config.raw_config, species_name=config.species_names[sp],
            n_dt_per_year=config.n_dt_per_year, n_years=config.n_year,
            lifespan_dt=int(config.lifespan_dt[sp]), ny=grid.ny, nx=grid.nx,
            config_dir=config.raw_config.get("_osmose.config.dir", ""),
            strict=config.movement_strict_coverage,
        )
        for sp in range(config.n_species)
        if config.movement_method[sp] == "maps"
    }  # fmt: skip


def reference_map_index(b: Bundle, species: np.ndarray, age: np.ndarray, step: int):
    off, shapes, flat = (
        b.get(k)
        for k in ("movement.maps.index.offsets", "movement.maps.index.shape", "movement.maps.index")
    )

    def at(sp, a, s):
        n_ages, n_steps = shapes[sp]
        if not (0 <= a < n_ages and 0 <= s < n_steps):
            return -1
        return int(flat[off[sp] + a * n_steps + s])

    cur = np.array([at(sp, a, step) for sp, a in zip(species, age, strict=True)], dtype=np.int32)
    prev = np.array(
        [at(sp, a - 1, step - 1) for sp, a in zip(species, age, strict=True)], dtype=np.int32
    )
    return cur, (cur == prev) & (age > 0) & (step > 0)


def test_map_indices_from_the_bundle_equal_precompute_map_indices(runs):
    from osmose.engine.processes.movement import _precompute_map_indices

    config, _, grid, b = runs["eec_full"]
    map_sets = engine_map_sets(config, grid)
    rng = np.random.default_rng(31)
    n = 3000
    sp = np.asarray(sorted(map_sets))[rng.integers(0, len(map_sets), n)].astype(np.int32)
    age = rng.integers(-1, np.asarray(config.lifespan_dt)[sp] + 2).astype(np.int32)
    uses = np.ones(n, dtype=bool)
    seen_change = False
    for step in (0, 1, 23, 24, 25, config.n_steps - 1, config.n_steps):
        want = _precompute_map_indices(sp, age, uses, map_sets, step)
        got = reference_map_index(b, sp, age, step)
        assert np.array_equal(want[0], got[0]), f"step {step}: current index"
        assert np.array_equal(want[1], got[1]), f"step {step}: same-map flag"
        seen_change |= bool((~want[1] & (want[0] >= 0)).any())
    assert seen_change  # the instrument sees map changes


def test_map_stack_holds_each_species_maps(runs):
    config, _, grid, b = runs["eec_full"]
    map_sets = engine_map_sets(config, grid)
    stack, null, offset = (
        b.get(k)
        for k in ("movement.maps.stack", "movement.maps.is_null", "movement.maps.species_offset")
    )
    for sp, ms in map_sets.items():
        for k, m in enumerate(ms.maps):
            g = offset[sp] + k
            if m is None:
                assert null[g] == 1 and not stack[g].any()
            else:
                assert (
                    null[g] == 0 and stack[g].tobytes() == np.asarray(m, dtype=np.float64).tobytes()
                )
    assert null.any()  # EEC's null maps (out of domain) are carried


def test_bay_of_biscay_random_domains_are_the_whole_ocean(runs):
    config, _, grid, b = runs["bay_of_biscay"]
    off, cells = b.get("movement.random.domain.offsets"), b.get("movement.random.domain.cells")
    ys, xs = np.nonzero(grid.ocean_mask)
    whole = np.column_stack((xs, ys))
    for sp in range(config.n_species):
        assert config.movement_method[sp] == "random"
        assert np.array_equal(cells[off[sp] : off[sp + 1]], whole)


def test_a_configured_ncell_gives_a_connected_patch_of_that_size(runs):
    config, raw, grid, _ = runs["bay_of_biscay"]
    cfg = copy.copy(config)
    cfg.random_distribution_ncell = np.array([25] + [0] * (config.n_species - 1), dtype=np.int32)
    b1 = build_bundle(cfg, raw, grid, seed=7)
    off, cells = b1.get("movement.random.domain.offsets"), b1.get("movement.random.domain.cells")
    patch = cells[off[0] : off[1]]
    assert len(patch) == 25 and all(grid.ocean_mask[y, x] for x, y in patch)
    for i in range(1, len(patch)):  # BFS order: every cell touches an earlier one
        assert any(max(abs(patch[i][0] - q[0]), abs(patch[i][1] - q[1])) <= 1 for q in patch[:i])
    again = build_bundle(cfg, raw, grid, seed=7).get("movement.random.domain.cells")
    assert np.array_equal(again, cells)  # deterministic in the run seed


# ------------------------------------------------------------------ result buffers


@pytest.mark.parametrize(
    ("n_steps", "freq", "step0", "want"),
    [
        (24, 24, False, 1),
        (50, 24, False, 3),
        (50, 24, True, 4),
        (24, 10, False, 3),
        (1, 5, True, 2),
    ],
)
def test_record_count_formula(n_steps, freq, step0, want):
    from osmose.engine_cs.build import n_records

    assert n_records(n_steps, freq, step0) == want


def test_record_count_matches_a_real_python_run_with_a_partial_window(runs):
    from osmose.engine.simulate import simulate
    from osmose.engine_cs.build import n_records

    _, raw, _, _ = runs["bay_of_biscay"]
    cfg = raw | {"simulation.time.nyear": "1", "output.recordfrequency.ndt": "10"}
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, 3)
    outputs = simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    assert (
        len(outputs)
        == n_records(config.n_steps, config.output_record_frequency, config.output_step0_include)
        == 3
    )


def test_result_buffers_are_registered_by_address(runs):
    from osmose.engine_cs.build import allocate_results

    config, raw, grid, _ = runs["eec_full"]
    b = build_bundle(config, raw, grid, seed=7)
    buf = allocate_results(b, config)
    # The bundle holds the SAME memory the engine will write into (no copy).
    assert b.get("result.biomass").ctypes.data == buf.biomass.ctypes.data
    assert b.get("control.block").ctypes.data == buf.control.ctypes.data
    assert buf.mortality.shape[1:] == (config.n_species, 8)
