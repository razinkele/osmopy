"""The catch-derived families on the C# engine equal the Python engine's, bit for bit.

Yield by Java size class (Java SizeIndicators/yieldDistribBySize), the catch-weighted mean size
and mean TL as numerator/denominator (meanSizeCatch / meanTLCatch), and mortality rates by cause
and size class (MortalitySpeciesOutput on sizeDistrib). Twelve perturbed copies of a real
post-reproduction state, with fishing deaths forced onto a share of the schools, go through the
C# "Windows" probe and ``simulate``'s own recording rule. Needs the built engine; skips without it
unless OSMOSE_REQUIRE_CSHARP=1.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from osmose.engine.state import MortalityCause
from osmose.engine_cs.build import (
    FAMILY_MORTALITY_BY_SIZE,
    FAMILY_SIZE_CATCH,
    FAMILY_TL_CATCH,
    FAMILY_YIELD_BY_SIZE,
    build_bundle,
    output_families,
)
from tests.test_csharp_outputs_families import ALL
from tests.test_csharp_pipeline import (
    _post_reproduction_state,
    _probe,
    _python_records,
    _raw,
    _write_states,
    engine_ok,  # noqa: F401  (a fixture)
)

CATCH = ALL | {
    "output.yield.biomass.bysize.enabled": "true",
    "output.size.catch.enabled": "true",
    "output.tl.catch.enabled": "true",
    "output.mortality.perspecies.bysize.enabled": "true",
    "output.distrib.bysize.min": "2",
    "output.distrib.bysize.max": "40",
    "output.distrib.bysize.incr": "4",
}
NAMES = (
    "yield_by_size",
    "size_catch_num",
    "size_catch_den",
    "tl_catch_num",
    "tl_catch_den",
    "mortality_rate_by_cause_size",
)
F = int(MortalityCause.FISHING)


@pytest.fixture(scope="module")
def catch_states(tmp_path_factory):
    out = {}
    for demo, step in (("tutorial", 129), ("eec_full", 30)):
        d = tmp_path_factory.mktemp(demo)
        config, grid, raw, st = _post_reproduction_state(_raw(demo, d, step // 24 + 1, CATCH), step)
        rng = np.random.default_rng(13)
        sz_t = config.output_size_thresholds.astype(np.float64)
        states = []
        for _ in range(12):
            s = copy.deepcopy(st)
            s.abundance[:] *= rng.uniform(0.5, 1.5, len(s))
            s.n_dead[:] *= rng.uniform(0.5, 1.5, s.n_dead.shape)
            focal = np.flatnonzero(s.species_id < config.n_species)
            caught = focal[0::3]
            s.n_dead[caught, F] = s.abundance[caught] * rng.uniform(0.01, 0.2, len(caught))
            s.length[focal[1::9]] = sz_t[2]  # exactly on a threshold
            s.length[focal[2::9]] = sz_t[0] - 0.5  # below the first class
            s.length[focal[4::9]] = sz_t[-1] + 50.0  # beyond the last
            states.append(s.replace(biomass=s.abundance * s.weight))
        out[demo] = (config, grid, raw, states)
    return out


def _row(record, name, nf):
    v = getattr(record, name)
    assert v is not None, name
    if isinstance(v, dict):
        return np.stack([v[sp] for sp in range(nf)])
    return np.asarray(v, dtype=np.float64)


def _same(a, b) -> bool:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    live = ~np.isnan(a)
    return a[live].tobytes() == b[live].tobytes()


def _near(a, b) -> bool:
    """Same shape, same NaN/inf cells, the same zero cells, finite values within 1e-13."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    if not np.array_equal(np.isinf(a), np.isinf(b)) or not np.array_equal(a == 0, b == 0):
        return False
    fin = np.isfinite(a)
    return bool(np.allclose(a[fin], b[fin], rtol=1e-13, atol=0.0))


@pytest.mark.parametrize("demo", ["tutorial", "eec_full"])
@pytest.mark.parametrize(("freq", "step0"), [(5, False), (1, False), (12, True)])
def test_catch_families_equal_python_bit_for_bit(
    engine_ok,  # noqa: F811
    catch_states,
    demo,
    freq,
    step0,
    tmp_path,
):
    config, grid, raw, states = catch_states[demo]
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    fam = output_families(cfg)
    for bit in (FAMILY_YIELD_BY_SIZE, FAMILY_SIZE_CATCH, FAMILY_TL_CATCH, FAMILY_MORTALITY_BY_SIZE):
        assert fam & bit
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states, tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    want = _python_records(states, cfg, freq, step0)
    nf = cfg.n_species
    for name in NAMES:
        got = res.get(f"out.{name}")
        assert got.shape[0] == len(want), name
        for i, r in enumerate(want):
            w = _row(r, name, nf)
            if name == "mortality_rate_by_cause_size":
                # -ln(N_end/N_start): NumPy's SIMD log and the CLR's libm log disagree in the last
                # ULP on a few arguments (measured: 1 cell of 972), the np.exp/libm class in
                # CLAUDE.md. Everything feeding the log is compared exactly; the log to 1e-13.
                assert _near(got[i], w), (name, i)
            else:
                assert _same(got[i], w), (name, i)


def test_the_catch_families_are_not_trivial(catch_states):
    """Guards the byte test against a vacuous fixture: catch in several size classes including
    both edge classes, several species caught, and nonzero fishing rates by size."""
    from osmose.engine.simulate import _collect_outputs

    config, _, _, states = catch_states["eec_full"]
    out = _collect_outputs(states[0], config, 0)
    nf = config.n_species
    y = _row(out, "yield_by_size", nf)
    assert (y.sum(axis=0) > 0).sum() >= 3 and y[:, 0].sum() > 0 and y[:, -1].sum() > 0
    assert (out.size_catch_den > 0).sum() >= 3 and (out.tl_catch_den > 0).sum() >= 3
    rates = out.mortality_rate_by_cause_size
    assert (rates[:, F, :] > 0).sum() >= 3 and np.isfinite(rates[:, F, :]).any()
