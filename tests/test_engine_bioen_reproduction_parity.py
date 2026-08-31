"""Gate G (spec §4): Java `BioenReproductionProcess.run()` parity for the bioen
reproduction path, plus proof that factoring the recruitment-regulation block out of
`reproduction()` left the STANDARD path bit-identical.

Java reference (verified at source, 4.3.3,
`process/bioen/BioenReproductionProcess.java:89-175`)::

    if (indexTimeSimu < yearSeading && SSB[cpt] == 0.) {        // no MATURE school
        SSB[cpt] = seedingBiomass[cpt];
        nEgg = sexRatio * beta * season * SSB * 1e6;
    } else {
        for (School school : schoolset) {
            if (!school.isMature()) continue;
            float wEgg = school.getGonadWeight() * (float) season;
            if (wEgg <= 0) continue;
            school.incrementGonadWeight(-wEgg);                 // partial, NOT a flush
            nEgg += wEgg * sexRatio / species.getEggWeight() * 1e6
                    * school.getInstantaneousAbundance();       // scales with N
        }
    }
    create_reproduction_schools(cpt, nEgg, ...)                 // nSchool UNLOCATED schools

and `Species.getEggSize()` (`Species.java:327`) returns `computeLength(eggWeight)` under
bioen, so egg schools are created at the length implied by the egg WEIGHT, not at
`species.egg.size`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.bioen_reproduction import bioen_egg_release
from osmose.engine.processes.reproduction import create_egg_schools, regulate_recruitment

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
REF_NPZ = BASELINE_DIR / "reproduction_reference_seed3.npz"
REF_JSON = BASELINE_DIR / "reproduction_reference_seed3.json"


def _synthetic_config(n_species: int = 2) -> EngineConfig:
    from tests.test_bioen_orchestration import _make_bioen_config_dict

    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=n_species).items()}
    return EngineConfig.from_dict(cfg)


# --------------------------------------------------------------------------- #
# 1. Java egg-release formula
# --------------------------------------------------------------------------- #


def test_egg_release_matches_java_formula():
    """wEgg = gonad*season per fish; nEgg = wEgg*sexRatio/eggWeight*N."""
    gonad = np.array([2e-6, 2e-6, 0.0])
    n = np.array([1e6, 1e3, 1e6])
    mature = np.array([True, True, True])
    n_eggs, w_egg = bioen_egg_release(
        gonad, n, mature, season=0.25, sex_ratio=0.5, egg_weight_t=1e-9
    )
    # wEgg = 2e-6 * 0.25 = 5e-7 t/fish ; nEgg = 5e-7 * 0.5 / 1e-9 * N
    np.testing.assert_allclose(w_egg, [5e-7, 5e-7, 0.0])
    np.testing.assert_allclose(n_eggs, [5e-7 * 0.5 / 1e-9 * 1e6, 5e-7 * 0.5 / 1e-9 * 1e3, 0.0])


def test_egg_release_scales_with_abundance():
    """v1 computed eggs per SCHOOL with no `* N`; two schools differing only in
    abundance produced identical egg counts. Java multiplies by
    `school.getInstantaneousAbundance()`."""
    gonad = np.array([2e-6, 2e-6])
    n = np.array([1e6, 1e3])
    mature = np.array([True, True])
    n_eggs, _ = bioen_egg_release(gonad, n, mature, 0.25, 0.5, 1e-9)
    assert n_eggs[0] / n_eggs[1] == pytest.approx(1e3)


def test_egg_release_immature_releases_nothing():
    gonad = np.array([2e-6, 2e-6, 0.0])
    n = np.array([1e6, 1e3, 1e6])
    n2, w2 = bioen_egg_release(gonad, n, np.array([False, True, True]), 0.25, 0.5, 1e-9)
    assert n2[0] == 0.0 and w2[0] == 0.0


def test_egg_release_is_a_partial_gonad_decrement_not_a_flush():
    """Java removes `gonad*season`, leaving `gonad*(1-season)`. v1 zeroed the gonad
    every step it was positive, which starved `create_offspring_genotypes` of parents."""
    gonad = np.array([4e-6])
    _, w_egg = bioen_egg_release(gonad, np.array([1e5]), np.array([True]), 0.2, 0.5, 1e-9)
    assert w_egg[0] == pytest.approx(4e-6 * 0.2)
    assert (gonad - w_egg)[0] == pytest.approx(4e-6 * 0.8)
    assert (gonad - w_egg)[0] > 0.0


def test_egg_release_season_scales_output_linearly():
    gonad = np.array([2e-6])
    n = np.array([1e6])
    mature = np.array([True])
    lo, _ = bioen_egg_release(gonad, n, mature, 0.1, 0.5, 1e-9)
    hi, _ = bioen_egg_release(gonad, n, mature, 0.4, 0.5, 1e-9)
    assert hi[0] / lo[0] == pytest.approx(4.0)


def test_egg_release_zero_season_releases_nothing():
    """`wEgg <= 0 -> continue`: outside the spawning window nothing is released and
    the gonad is untouched."""
    gonad = np.array([2e-6])
    n_eggs, w_egg = bioen_egg_release(gonad, np.array([1e6]), np.array([True]), 0.0, 0.5, 1e-9)
    assert n_eggs[0] == 0.0 and w_egg[0] == 0.0


# --------------------------------------------------------------------------- #
# 2. The shared regulation helper
# --------------------------------------------------------------------------- #


def test_regulate_recruitment_is_identity_when_no_regulation_configured():
    """The Gate-B (Bay of Biscay) config sets none of these keys, so the helper must be
    the identity there and Java parity is preserved on the bioen path."""
    config = _synthetic_config()
    assert config.recruitment_type[:2] == ["none", "none"]
    assert config.rv_gate_factor_by_index is None
    assert config.recruitment_ceiling_by_season is None
    assert config.thermal_gate_factor_by_index is None
    assert config.depensation_gate_enabled is None

    lin = np.array([100.0, 5.0])
    ssb = np.array([10.0, 0.0])
    seeded = np.array([False, True])
    out = regulate_recruitment(lin, ssb, seeded, config, step=3)
    np.testing.assert_array_equal(out, lin)


GATE_B_CONFIG = Path(__file__).resolve().parents[1] / "data" / "examples" / "osm_all-parameters.csv"


@pytest.mark.skipif(not GATE_B_CONFIG.exists(), reason="no Gate-B (Bay of Biscay) config")
def test_regulate_recruitment_is_identity_on_the_real_gate_b_config():
    """Gate B must stay parity-pure: on the ACTUAL Bay of Biscay config the shared
    regulation helper has to be the identity, or the bioen path would diverge from Java
    exactly where cross-engine parity is measured. Asserted on the real config rather than
    on the synthetic one, because a future edit to `data/examples` is what would break it.
    """
    from osmose.config import OsmoseConfigReader

    config = EngineConfig.from_dict(OsmoseConfigReader().read(GATE_B_CONFIG))
    assert set(config.recruitment_type) == {"none"}
    assert config.seeding_mode == "stock_recruitment"
    assert config.rv_gate_factor_by_index is None
    assert config.recruitment_ceiling_by_season is None
    assert config.thermal_gate_factor_by_index is None
    assert config.depensation_gate_enabled is None

    n = config.n_species
    lin = np.arange(1.0, n + 1.0) * 1e6
    ssb = np.arange(1.0, n + 1.0) * 100.0
    for seeded in (np.zeros(n, dtype=np.bool_), np.ones(n, dtype=np.bool_)):
        out = regulate_recruitment(lin, ssb, seeded, config, step=7)
        np.testing.assert_array_equal(out, lin)


def test_regulate_recruitment_applies_the_sr_curve():
    """With a curve configured the helper is NOT the identity — proving the previous
    test measures inertness rather than a no-op helper."""
    from tests.test_bioen_orchestration import _make_bioen_config_dict

    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    cfg["stock.recruitment.type.sp0"] = "beverton_holt"
    cfg["stock.recruitment.ssbhalf.sp0"] = "10.0"
    config = EngineConfig.from_dict(cfg)

    lin = np.array([100.0, 5.0])
    ssb = np.array([10.0, 0.0])
    seeded = np.array([False, False])
    out = regulate_recruitment(lin, ssb, seeded, config, step=3)
    assert out[0] == pytest.approx(100.0 / 2.0)  # ssb == ssbhalf -> halved
    assert out[1] == pytest.approx(5.0)  # type "none" -> untouched


# --------------------------------------------------------------------------- #
# 3. Egg-school creation
# --------------------------------------------------------------------------- #


def test_create_egg_schools_makes_n_schools_unlocated_with_given_length():
    config = _synthetic_config()
    n_eggs = np.array([1e6, 0.0])
    seeded = np.array([False, False])
    schools = create_egg_schools(n_eggs, seeded, config, egg_length=np.array([0.49, 0.49]))
    assert len(schools) == 1
    s = schools[0]
    assert len(s) == int(config.n_schools[0])
    assert np.all(s.cell_x == -1) and np.all(s.cell_y == -1)
    assert np.all(s.is_egg)
    assert s.abundance.sum() == pytest.approx(1e6)
    assert np.all(s.length == 0.49)


def test_create_egg_schools_egg_weight_ignores_the_length_override():
    """Java's bioen egg school gets weight = `species.egg.weight` and length =
    `computeLength(eggWeight)`. The length override must NOT feed back into the weight."""
    config = _synthetic_config()
    n_eggs = np.array([1e6, 0.0])
    seeded = np.array([False, False])
    expected_w = (
        config.condition_factor[0] * config.egg_size[0] ** config.allometric_power[0] * 1e-6
    )
    schools = create_egg_schools(n_eggs, seeded, config, egg_length=np.array([12.3, 12.3]))
    assert np.all(schools[0].weight == pytest.approx(expected_w))
    assert np.all(schools[0].length == 12.3)


def test_create_egg_schools_default_length_is_config_egg_size():
    config = _synthetic_config()
    schools = create_egg_schools(np.array([1e6, 0.0]), np.array([False, False]), config)
    assert np.all(schools[0].length == config.egg_size[0])


def test_create_egg_schools_fewer_eggs_than_schools_makes_one_school():
    """Java `create_reproduction_schools`: `nEgg < nSchool` -> a single school."""
    config = _synthetic_config()
    schools = create_egg_schools(np.array([3.0, 0.0]), np.array([False, False]), config)
    assert len(schools) == 1 and len(schools[0]) == 1
    assert schools[0].abundance[0] == pytest.approx(3.0)


def test_create_egg_schools_tags_seeded_origin():
    config = _synthetic_config()
    schools = create_egg_schools(np.array([1e6, 1e6]), np.array([True, False]), config)
    assert np.all(schools[0].from_seeding)
    assert not np.any(schools[1].from_seeding)


# --------------------------------------------------------------------------- #
# 4. Gate A at unit level: the standard path is bit-identical after the extraction
# --------------------------------------------------------------------------- #


def _load_reference():
    if not REF_NPZ.exists() or not REF_JSON.exists():
        pytest.skip(
            "reproduction reference not generated (gitignored; see task-5-report.md §Step 2)"
        )
    z = np.load(REF_NPZ)  # plain arrays only; never allow_pickle (security policy)
    meta = json.loads(REF_JSON.read_text())
    return z, meta


@pytest.mark.parametrize("case", ["none", "a", "b", "c", "d", "e"])
def test_standard_reproduction_bit_identical_after_extraction(case):
    """`reproduction()` output must equal the PRE-extraction reference, array-for-array.

    Six cases, each enabling exactly one regulation block so a botched move of any
    single block fails exactly one case:
      none  Shepherd SR only
      a     + population.seeding.mode=linear (the `np.where(seeded, linear, sr)` branch)
      b     + depensation gate
      c     + RV gate
      d     + recruitment ceiling (parameterised to BIND)
      e     + thermal gate
    """
    from dataclasses import fields

    from osmose.engine.processes.reproduction import reproduction
    from osmose.engine.state import SchoolState

    z, meta = _load_reference()
    if case not in meta["cases"]:
        pytest.skip(f"reference predates case {case!r}; regenerate")

    config = EngineConfig.from_dict(dict(meta["cases"][case]))
    state_fields = set(meta["state_fields"])
    kwargs = {
        f.name: (z[f"state_{f.name}"] if f.name in state_fields else None)
        for f in fields(SchoolState)
    }
    state = SchoolState(**kwargs)

    out = reproduction(
        state,
        config,
        int(meta["step"]),
        np.random.default_rng(int(meta["seed"])),
        grid_ny=10,
        grid_nx=10,
    )
    for k in meta["out_keys"]:
        np.testing.assert_array_equal(
            getattr(out, k), z[f"out_{case}_{k}"], err_msg=f"case {case}: {k} diverged"
        )


def test_reference_cases_are_mutually_distinct():
    """Guards the test above: if every case produced the same arrays, five of the six
    parametrisations would prove nothing about the gate blocks."""
    z, meta = _load_reference()
    sums = {
        c: float(z[f"out_{c}_abundance"].sum()) for c in meta["cases"] if f"out_{c}_abundance" in z
    }
    assert len(set(sums.values())) == len(sums), f"cases are not distinguishable: {sums}"
