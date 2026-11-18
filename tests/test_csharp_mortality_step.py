"""Plan task 1.5: one whole C# mortality step equals Python's ``mortality()`` with the same draws.

A school state is captured from a real Python run at mortality entry. The REAL ``mortality()``
then runs on it with the batched kernel replaced by a replay of the per-cell oracle
(``_mortality_in_cell_numba``) on recorded draws. The larva pre-pass, egg retention, the rate
tables, the commit, out-of-domain mortality, the next starvation rate and the trophic level all
stay Python's own code. ``MortalityProbe`` runs the C# step on the same state and draws, in a
probe subprocess (the CLR never enters pytest).

There are two references (spec 6.1):
- **NumPy reference**, ``mortality()`` as it ships. The larva and out-of-domain steps call NumPy's
  ``exp``, which can differ from the C library's in the last unit, and the whole step is coupled.
  So this is ONE comparison at tier L, 1e-9 relative, with the worst error recorded.
- **libm reference**, the same run with ``natural.np.exp`` replaced by ``math.exp``, the C
  library's ``exp`` that .NET's ``Math.Exp`` also calls. The C# step must equal this reference
  BIT FOR BIT, which proves every remaining difference against the NumPy reference is ``exp``.

The test needs the built engine. It skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import osmose.engine.processes.mortality as mort
import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.processes import natural
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from tests._tutorial_config import build_config

ROOT = Path(__file__).resolve().parents[1]
CASES = {"eec_full": 30, "tutorial": 129}
DRAW_SEED = 20261118
PERMS = ("pred", "starv", "fish", "nat")
CAUSES = (
    "predation",
    "starvation",
    "additional",
    "fishing",
    "out",
    "foraging",
    "discards",
    "aging",
)
FIELDS = ("abundance", "n_dead", "starvation_rate", "trophic_level", "pred_success_rate", "preyed_biomass", "egg_retained")  # fmt: skip
TIER_L = 1e-9


class _Stop(Exception):
    pass


class _LibmNumpy:
    """``numpy``, except that ``exp`` is the C library's (``math.exp``), as .NET's ``Math.Exp`` is."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def exp(x):
        a = np.asarray(x, dtype=np.float64)
        flat = np.fromiter((math.exp(v) for v in a.ravel()), dtype=np.float64, count=a.size)
        return flat.reshape(a.shape)


def _replay_kernel(record: list):
    """A stand-in for the batched kernel: per cell, draw four permutations and one cause order per
    slot from a seeded stream, record them, and run the per-cell oracle. Arguments are bound by
    NAME from the two signatures, so a reordered kernel signature fails loudly, not silently."""
    oracle = mort._mortality_in_cell_numba
    o_names = list(inspect.signature(oracle.py_func).parameters)
    b_names = list(inspect.signature(mort._mortality_all_cells_parallel.py_func).parameters)
    rng = np.random.default_rng(DRAW_SEED)

    def replay(*args):
        a = dict(zip(b_names, args, strict=True))
        sorted_idx, bounds = a["sorted_indices"], a["boundaries"]
        n = len(sorted_idx)
        perms = {k: np.zeros(n, dtype=np.int32) for k in PERMS}
        orders = np.zeros((n, 4), dtype=np.int32)
        for cell in range(int(a["n_cells"])):
            s, e = int(bounds[cell]), int(bounds[cell + 1])
            if e <= s:
                continue
            for k in PERMS:
                perms[k][s:e] = rng.permutation(e - s)
            for i in range(s, e):
                orders[i] = rng.permutation(4)
            a.update(
                cell_indices=sorted_idx[s:e], seq_pred=perms["pred"][s:e], seq_starv=perms["starv"][s:e],
                seq_fish=perms["fish"][s:e], seq_nat=perms["nat"][s:e], seq_for=perms["pred"][s:e],
                cause_orders=orders[s:e], cell_id=cell,
            )  # fmt: skip
            oracle(*[a[k] for k in o_names])
        record.append((perms, orders))

    return replay


def _reference(state, resources, config, grid, step, ctx, rsc0, libm: bool):
    """The real ``mortality()`` on a copy of ``state``, with the replayed kernel."""
    record: list = []
    resources.biomass[...] = rsc0
    names = ("_mortality_all_cells_parallel", "_mortality_all_cells_numba")
    saved = {name: getattr(mort, name) for name in names}
    replay = _replay_kernel(record)
    old_np = natural.np
    for name in names:
        setattr(mort, name, replay)
    if libm:
        natural.np = _LibmNumpy()
    try:
        out = mort.mortality(
            copy.deepcopy(state),
            resources,
            config,
            np.random.default_rng(0),
            grid,
            step=step,
            ctx=ctx,
        )
    finally:
        for name, fn in saved.items():
            setattr(mort, name, fn)
        natural.np = old_np
    assert len(record) == config.mortality_subdt  # the replay really ran, once per sub-step
    return out, record, resources.biomass.copy()


def _disagreeing(rate: float) -> float:
    """The first value just above ``rate`` whose ``1 - exp(-v)`` NumPy and the C library round
    differently (they disagree on about 6% of arguments)."""
    cand = rate * (1.0 + np.arange(1, 20001) * 1e-9)
    numpy_exp = np.exp(-cand)
    libm_exp = np.array([math.exp(-v) for v in cand])
    hit = np.flatnonzero(numpy_exp != libm_exp)
    assert hit.size, rate
    return float(cand[hit[0]])


def _make_exp_visible(config) -> None:
    """Every shipped larva and out-of-domain rate happens to be an argument where NumPy's ``exp``
    and the C library's agree, which would make the two references identical and the libm test
    uninformative. Move each focal larva rate (by at most 2e-5 relative) to one where they differ."""
    assert config.larva_mortality_by_dt is None or all(
        t is None for t in config.larva_mortality_by_dt
    )
    rates = config.larva_mortality_rate
    for sp in range(config.n_species):
        if rates[sp] > 0:
            rates[sp] = _disagreeing(float(rates[sp]))


def _eec_synthetic(config, state) -> None:
    """EEC at the capture step: larva rates where NumPy and libm disagree; a critical success
    rate of 0 for the species of the first egg school (eggs never feed, so the next starvation
    rate takes the ``csr > 0 ? csr : 1`` guard, which no shipped species reaches); and a discard
    fraction, which no shipped config sets (the DISCARDS counter would otherwise always be 0)."""
    _make_exp_visible(config)
    assert not config.fishing_discard_rate.any()  # shipped: all zero
    config.fishing_discard_rate[: config.n_species] = 0.3
    sp = int(state.species_id[state.is_egg.astype(bool)][0])
    assert config.starvation_rate_max[sp] > 0
    config.critical_success_rate[sp] = 0.0


def _tutorial_synthetic(config, state) -> None:
    """Tutorial config at the capture step: give the background species a starvation maximum and
    a critical success rate. Every shipped background species has a zero maximum, so the zeroing
    of background starvation rates would otherwise change nothing."""
    for b in np.unique(state.species_id[state.is_background.astype(bool)]):
        config.starvation_rate_max[b] = 2.0
        config.critical_success_rate[b] = 1.0


SYNTHETIC = {"eec_full": _eec_synthetic, "tutorial": _tutorial_synthetic}


def _capture(raw: dict, step: int, synthetic) -> dict:
    raw = dict(raw)
    raw["output.minimal.enabled"] = "true"
    raw["simulation.time.nyear"] = str(step // 24 + 1)
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    got: dict = {}
    real = sim._mortality

    def spy(state, resources, config_, rng_, grid_, step=0, species_rngs=None, ctx=None):
        if step == target:
            st = state.compact()
            assert len(st) == len(state)  # nothing dead at entry: the C# regroup drops nothing
            assert not st.n_dead.any() and not st.preyed_biomass.any()
            rsc0 = resources.biomass.copy()
            synthetic(config_, st)  # at this step only: the trajectory up to it is the shipped one
            got["numpy"] = _reference(st, resources, config_, grid_, step, ctx, rsc0, libm=False)
            got["libm"] = _reference(st, resources, config_, grid_, step, ctx, rsc0, libm=True)
            got["state"], got["rsc0"] = st, rsc0
            raise _Stop
        return real(
            state, resources, config_, rng_, grid_, step=step, species_rngs=species_rngs, ctx=ctx
        )

    target = step
    sim._mortality = spy
    try:
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    except _Stop:
        pass
    finally:
        sim._mortality = real
    got.update(config=config, grid=grid, raw=raw, step=step)
    return got


def _write_state(st, path: Path) -> None:
    b = Bundle(1)
    for name, dt in [
        ("species_id", np.int32), ("age_dt", np.int32), ("length", np.float64), ("weight", np.float64),
        ("abundance", np.float64), ("trophic_level", np.float64), ("cell_x", np.int32),
        ("cell_y", np.int32), ("is_background", np.uint8), ("is_egg", np.uint8), ("is_out", np.uint8),
        ("first_feeding_age_dt", np.int32), ("starvation_rate", np.float64),
        ("pred_success_rate", np.float64), ("preyed_biomass", np.float64),
    ]:  # fmt: skip
        b.set_array(f"state.{name}", np.ascontiguousarray(getattr(st, name), dtype=dt))
    b.save(path)


def _write_draws(record: list, path: Path) -> None:
    b = Bundle(1)
    for key, name in zip(PERMS, ("predation", "starvation", "fishing", "additional"), strict=True):
        b.set_array(f"draws.perm_{name}", np.stack([r[0][key] for r in record]).astype(np.int32))
    b.set_array("draws.cause_orders", np.stack([r[1] for r in record]).astype(np.uint8))
    b.save(path)


def _run_probe(
    model: Path, state: Path, draws: Path, step: int, out: Path
) -> subprocess.CompletedProcess:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.MortalityProbe.Run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])"
    )
    args = [
        sys.executable,
        "-c",
        code,
        str(ROOT),
        str(model),
        str(state),
        str(draws),
        str(step),
        str(out),
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=ROOT)


@pytest.fixture(scope="module")
def stepped(tmp_path_factory) -> dict:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    out = {}
    for demo, step in CASES.items():
        d = tmp_path_factory.mktemp(demo)
        if demo == "tutorial":
            # The bottom-oxygen benthos coupling is phase 2 (feature oxygen_benthos): the engine
            # refuses a config that enables it, so the step is compared without it.
            raw = dict(build_config(d, n_year=1)) | {"ltl.oxygen.benthos.enabled": "false"}
        else:
            raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, d)["config_file"])))
        got = _capture(raw, step, SYNTHETIC[demo])
        (_, rec_np, _), (_, rec_libm, _) = got["numpy"], got["libm"]
        for (p1, o1), (p2, o2) in zip(rec_np, rec_libm, strict=True):  # one draw set for both
            assert all(np.array_equal(p1[k], p2[k]) for k in PERMS) and np.array_equal(o1, o2)
        build_bundle(got["config"], got["raw"], got["grid"], seed=11, checks=True).save(
            d / "model.bin"
        )
        _write_state(got["state"], d / "state.bin")
        _write_draws(rec_libm, d / "draws.bin")
        proc = _run_probe(d / "model.bin", d / "state.bin", d / "draws.bin", step, d / "out.bin")
        assert proc.returncode == 0, proc.stderr[-4000:]
        got["cs"] = Bundle.load(d / "out.bin")
        got["dir"] = d
        out[demo] = got
    return out


def _pairs(got: dict, ref: str):
    """(name, C# array, reference array) for every compared output."""
    state_out, _, rsc_out = got[ref]
    cs = got["cs"]
    for name in FIELDS:
        yield (
            name,
            np.asarray(cs.get(f"mort.{name}")),
            np.asarray(getattr(state_out, name), dtype=np.float64),
        )
    n_rsc = rsc_out.shape[0]
    yield "resources", np.asarray(cs.get("mort.resources"))[:n_rsc], rsc_out


@pytest.mark.parametrize("demo", sorted(CASES))
def test_step_equals_the_libm_reference_bit_for_bit(stepped, demo):
    for name, got, want in _pairs(stepped[demo], "libm"):
        assert got.shape == want.shape, name
        bad = np.flatnonzero(got.ravel().view(np.uint64) != want.ravel().view(np.uint64))
        assert bad.size == 0, (
            f"{name}: {bad.size} values differ, first at {bad[0]}: {got.ravel()[bad[0]]!r} vs {want.ravel()[bad[0]]!r}"
        )


@pytest.mark.parametrize("demo", sorted(CASES))
def test_step_matches_the_numpy_reference_at_tier_l(stepped, demo, record_property):
    worst = {}
    start = stepped[demo]["state"].abundance
    for name, got, want in _pairs(stepped[demo], "numpy"):
        diff = np.abs(got - want)
        both_nan = np.isnan(got) & np.isnan(want)  # land cells of the resource fields
        # The retained egg count is what is left after n_subdt releases of abundance / n_subdt: a
        # cancellation residue, exactly 0 on one side and a few ulps on the other. Relative to
        # itself it is meaningless; its scale is the abundance being cancelled.
        scale = start if name == "egg_retained" else np.abs(want)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where((diff == 0) | both_nan, 0.0, diff / scale)
        worst[name] = float(rel.max()) if rel.size else 0.0
    record_property("worst_relative_error", worst)
    print(f"{demo} worst relative error vs the NumPy reference: {worst}")
    assert all(v <= TIER_L for v in worst.values()), worst
    if demo == "eec_full":  # its larva rates sit where NumPy and libm disagree: the gap is visible
        assert worst["n_dead"] > 0 and worst["abundance"] > 0, worst


@pytest.mark.parametrize("demo", sorted(CASES))
def test_death_accounting_holds_for_every_school(stepped, demo):
    got = stepped[demo]
    start = got["state"].abundance
    removed = start - got["cs"].get("mort.abundance")
    counted = got["cs"].get("mort.n_dead").sum(axis=1)
    assert (np.abs(removed - counted) <= TIER_L * start).all()


@pytest.mark.parametrize("demo", sorted(CASES))
def test_the_captured_step_exercises_every_path(stepped, demo):
    """Guards the comparison against a capture that has nothing in it: eggs (larva pre-pass), every
    loop cause, plankton eaten and, on EEC, out-of-domain schools; on the tutorial config, the
    background predators feed."""
    got = stepped[demo]
    st, n_dead = got["state"], got["cs"].get("mort.n_dead")
    by_cause = dict(zip(CAUSES, n_dead.sum(axis=0), strict=True))
    egg = st.is_egg.astype(bool)
    assert egg.any() and (n_dead[egg, CAUSES.index("additional")] > 0).all()
    for cause in ("predation", "starvation", "additional", "fishing"):
        assert by_cause[cause] > 0, cause
    if demo == "eec_full":
        assert by_cause["discards"] > 0
    rsc_after = np.asarray(got["cs"].get("mort.resources"))[: got["rsc0"].shape[0]]
    assert (rsc_after < got["rsc0"]).any()
    if demo == "eec_full":
        assert st.is_out.any() and by_cause["out"] > 0
    else:
        bkg = st.is_background.astype(bool)
        assert (got["cs"].get("mort.preyed_biomass")[bkg] > 0).any()


def test_run_checks_switches_the_death_accounting_check(stepped, tmp_path):
    """Correct code never violates the invariant, so poison one school (NaN abundance, which no
    counter can account for) and run the same step with ``run.checks`` on and off: on must raise
    the invariant violation, off must run to the end."""
    got = stepped["eec_full"]
    d, step = got["dir"], got["step"]
    st = copy.deepcopy(got["state"])
    st.abundance[0] = np.nan
    _write_state(st, tmp_path / "state.bin")
    off = build_bundle(got["config"], got["raw"], got["grid"], seed=11, checks=False)
    off.save(tmp_path / "model_off.bin")
    on = _run_probe(
        d / "model.bin", tmp_path / "state.bin", d / "draws.bin", step, tmp_path / "on.bin"
    )
    assert (
        on.returncode != 0
        and "InvariantViolationException" in on.stderr
        and "death accounting" in on.stderr
    )
    ran = _run_probe(
        tmp_path / "model_off.bin",
        tmp_path / "state.bin",
        d / "draws.bin",
        step,
        tmp_path / "off.bin",
    )
    assert ran.returncode == 0, ran.stderr[-2000:]
    assert np.isnan(Bundle.load(tmp_path / "off.bin").get("mort.abundance")[0])
