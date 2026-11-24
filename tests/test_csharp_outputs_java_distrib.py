"""The Java-rule distribution families on the C# engine equal the Python engine's, bit for bit.

Biomass by trophic-level class, and the numerator/denominator of mean size by age and mean TL by
size (osmose/engine/output_distrib.py; Java DistribOutput / WeightedDistribOutput). Twelve perturbed
copies of a real post-reproduction state go through the C# "Windows" probe and through
``simulate``'s own recording rule; every record of every family must be byte-identical. The states
are edited so values sit exactly on a threshold, below the first and beyond the last, so the class
rule is exercised, not just the bulk. Needs the built engine; skips without it unless
OSMOSE_REQUIRE_CSHARP=1.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from osmose.engine_cs.build import (
    FAMILY_BIOMASS_BY_TL,
    FAMILY_MEANSIZE_BY_AGE,
    FAMILY_MEANTL_BY_SIZE,
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

JAVA = ALL | {
    "output.biomass.bytl.enabled": "true",
    "output.meansize.byage.enabled": "true",
    "output.meantl.bysize.enabled": "true",
    # coarse explicit classes so edits can land on, below and beyond them
    "output.distrib.bytl.min": "2",
    "output.distrib.bytl.max": "4",
    "output.distrib.bytl.incr": "0.5",
    "output.distrib.byage.min": "1",
    "output.distrib.byage.max": "4",
    "output.distrib.byage.incr": "1",
}
NAMES = (
    "biomass_by_tl",
    "meansize_byage_num",
    "meansize_byage_den",
    "meantl_bysize_num",
    "meantl_bysize_den",
)


@pytest.fixture(scope="module")
def java_states(tmp_path_factory):
    out = {}
    for demo, step in (("tutorial", 129), ("eec_full", 30)):
        d = tmp_path_factory.mktemp(demo)
        config, grid, raw, st = _post_reproduction_state(_raw(demo, d, step // 24 + 1, JAVA), step)
        rng = np.random.default_rng(11)
        states = []
        tl_t = config.output_tl_thresholds.astype(np.float64)
        sz_t = config.output_size_thresholds.astype(np.float64)
        for _ in range(12):
            s = copy.deepcopy(st)
            s.abundance[:] *= rng.uniform(0.5, 1.5, len(s))
            focal = np.flatnonzero(s.species_id < config.n_species)
            s.trophic_level[focal[0::6]] = tl_t[1]  # exactly on a threshold
            s.trophic_level[focal[1::6]] = tl_t[0] - 0.3  # below the first class
            s.trophic_level[focal[2::6]] = tl_t[-1] + 2.0  # beyond the last
            s.length[focal[3::6]] = sz_t[2]  # a length exactly on a threshold
            states.append(s.replace(biomass=s.abundance * s.weight))
        out[demo] = (config, grid, raw, states)
    return out


def _row(record, name, nf):
    v = getattr(record, name)
    assert v is not None, name
    return np.stack([v[sp] for sp in range(nf)])


@pytest.mark.parametrize("demo", ["tutorial", "eec_full"])
@pytest.mark.parametrize(("freq", "step0"), [(5, False), (1, False), (12, True)])
def test_java_rule_families_equal_python_bit_for_bit(
    engine_ok,  # noqa: F811
    java_states,
    demo,
    freq,
    step0,
    tmp_path,
):
    config, grid, raw, states = java_states[demo]
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    fam = output_families(cfg)
    assert (
        fam & FAMILY_BIOMASS_BY_TL and fam & FAMILY_MEANSIZE_BY_AGE and fam & FAMILY_MEANTL_BY_SIZE
    )
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
            assert got[i].shape == w.shape and got[i].tobytes() == w.tobytes(), (name, i)


def test_the_families_are_not_trivial(java_states):
    """Guards the byte test: several classes populated, the edge classes reached (three C#
    checks in this project were once vacuous by fixture)."""
    from osmose.engine.simulate import _collect_outputs

    config, _, _, states = java_states["eec_full"]
    out = _collect_outputs(states[0], config, 0)
    nf = config.n_species
    tl = _row(out, "biomass_by_tl", nf)
    assert (tl.sum(axis=0) > 0).sum() >= 3  # several TL classes populated
    assert tl[:, 0].sum() > 0 and tl[:, -1].sum() > 0  # below-min and beyond-max classes reached
    assert (_row(out, "meantl_bysize_den", nf).sum(axis=0) > 0).sum() >= 3
    assert (_row(out, "meansize_byage_den", nf).sum(axis=0) > 0).sum() >= 2
