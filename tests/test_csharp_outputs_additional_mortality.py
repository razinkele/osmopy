"""Additional mortality by size (Java Indicators/additionalMortality(N)DistribBySize) on the C#
engine equals the Python engine's, bit for bit, through the C# "Windows" probe.

DistribOutput's defaults: window MEAN, alive schools only (abundance > 0 and age <= lifespan - 1),
Java size classes. The states carry additional deaths, schools with no survivor and schools past
their lifespan, so the alive filter is exercised. Needs the built engine; skips without it unless
OSMOSE_REQUIRE_CSHARP=1.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from osmose.engine.state import MortalityCause
from osmose.engine_cs.build import build_bundle
from tests.test_csharp_outputs_families import ALL
from tests.test_csharp_pipeline import (
    _post_reproduction_state,
    _probe,
    _python_records,
    _raw,
    _write_states,
    engine_ok,  # noqa: F401  (a fixture)
)

ON = ALL | {
    "output.mortality.additional.bysize.enabled": "true",
    "output.mortality.additionaln.bysize.enabled": "true",
}
NAMES = ("additional_mortality_bysize", "additional_n_bysize")
ADD = int(MortalityCause.ADDITIONAL)


@pytest.fixture(scope="module")
def states(tmp_path_factory):
    d = tmp_path_factory.mktemp("eec")
    config, grid, raw, st = _post_reproduction_state(_raw("eec_full", d, 2, ON), 30)
    rng = np.random.default_rng(17)
    out = []
    for _ in range(12):
        s = copy.deepcopy(st)
        s.abundance[:] *= rng.uniform(0.5, 1.5, len(s))
        focal = np.flatnonzero(s.species_id < config.n_species)
        s.n_dead[focal, ADD] = s.abundance[focal] * rng.uniform(0.0, 0.2, len(focal))
        s.abundance[focal[0::17]] = 0.0  # no survivor: not alive, excluded
        s.age_dt[focal[1::19]] = config.lifespan_dt[s.species_id[focal[1::19]]]  # past lifespan
        out.append(s.replace(biomass=s.abundance * s.weight))
    return config, grid, raw, out


@pytest.mark.parametrize(("freq", "step0"), [(5, False), (1, False), (12, True)])
def test_additional_mortality_by_size_equals_python_bit_for_bit(
    engine_ok,  # noqa: F811
    states,
    freq,
    step0,
    tmp_path,
):
    config, grid, raw, sts = states
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(sts, tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    want = _python_records(sts, cfg, freq, step0)
    nf = cfg.n_species
    for name in NAMES:
        got = res.get(f"out.{name}")
        for i, r in enumerate(want):
            w = np.stack([getattr(r, name)[sp] for sp in range(nf)])
            assert got[i].shape == w.shape and got[i].tobytes() == w.tobytes(), (name, i)


def test_the_alive_filter_and_the_classes_are_exercised(states):
    from osmose.engine.simulate import _collect_outputs

    config, _, _, sts = states
    s = sts[0]
    out = _collect_outputs(s, config, 0)
    nf = config.n_species
    n = np.stack([out.additional_n_bysize[sp] for sp in range(nf)])
    assert (n.sum(axis=0) > 0).sum() >= 3  # several size classes
    focal = s.species_id < nf
    excluded = focal & (
        (s.abundance <= 0) | (s.age_dt > config.lifespan_dt[np.clip(s.species_id, 0, nf - 1)] - 1)
    )
    assert excluded.any() and s.n_dead[excluded, ADD].sum() > 0  # the filter removes real deaths
    assert n.sum() < s.n_dead[focal, ADD].sum()
