"""The evolving imax trait acts in mortality, as in Java (2026-11-24).

Until then ``state.imax_trait`` was assigned nowhere and phenotypes were expressed after
mortality, so the only trait the shipped ``baltic_ev`` demo evolves changed nothing. Java applies
it in two places, both inside mortality:

- the bioen ingestion cap, ``BioenPredationMortality.getMaxPredationRate``: a ``School`` with an
  ``imax`` trait uses it in place of the species rate (larval term and /nstep still applied);
- genetic foraging mortality, ``ForagingMortality.getRate``: ``k1*exp(k2*(imax - I_max))/nstep``.

Also covers the target validation (an unread target is an error) and the bioen-off warning.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap


def _cap(imax_school=None):
    # two focal schools of species 0, one background school (species id 1 >= n_species)
    return per_fish_ingestion_cap(
        weight=np.array([0.01, 0.01, 0.01]),
        species_id=np.array([0, 0, 1], dtype=np.int32),
        age_dt=np.array([50, 50, 50], dtype=np.int32),
        i_max_all=np.array([3.5, 7.0]),
        beta=np.array([0.8]),
        larvae_thres_dt=np.array([1], dtype=np.int32),
        theta=np.array([1.0]),
        c_rate=np.array([0.0]),
        n_species=1,
        n_dt_per_year=24,
        n_subdt=10,
        imax_school=imax_school,
    )


def test_ingestion_cap_uses_per_school_imax_for_focal_schools():
    base = _cap()
    traited = _cap(np.array([7.0, np.nan, 99.0]))
    assert traited[0] == pytest.approx(2.0 * base[0])  # imax 7 vs species 3.5: cap doubles
    assert traited[1] == base[1]  # NaN = no trait -> species value
    assert traited[2] == base[2]  # background schools never take the trait (not a School)


def test_ingestion_cap_without_trait_is_bit_identical():
    assert np.array_equal(_cap(None), _cap(np.full(3, np.nan)))


def test_genetic_foraging_uses_trait_where_present_constant_elsewhere():
    from osmose.engine.processes.mortality import _precompute_foraging_rates

    config = SimpleNamespace(
        n_species=1,
        n_dt_per_year=24,
        bioen_enabled=True,
        foraging_k1_for=np.array([2.4]),
        foraging_k2_for=np.array([0.5]),
        foraging_I_max=np.array([3.5]),
        bioen_k_for=np.array([1.2]),
    )

    class _W(SimpleNamespace):
        def __len__(self):
            return 2

    w = _W(species_id=np.array([0, 0], dtype=np.int32), imax_trait=np.array([4.5, np.nan]))
    import osmose.engine.processes.mortality as mort

    orig = mort._zero_exempt
    mort._zero_exempt = lambda rates, state: None  # isolate the rate formula
    try:
        rates = _precompute_foraging_rates(w, config, n_subdt=1)
    finally:
        mort._zero_exempt = orig
    assert rates[0] == pytest.approx(2.4 * np.exp(0.5 * (4.5 - 3.5)) / 24)  # genetic form
    assert rates[1] == pytest.approx(1.2 / 24)  # no trait -> constant k_for


def _registry_state(target: str):
    from osmose.engine.genetics.genotype import GeneticState

    trait = SimpleNamespace(target_param=target, species_mean=np.array([3.5]))
    gs = SimpleNamespace(
        registry=SimpleNamespace(traits={"imax": trait}),
        alleles={"imax": np.array([[[0.25, 0.25]], [[0.0, -0.5]]])},
        env_noise={"imax": np.array([0.0, 0.1])},
    )
    _ = GeneticState  # the duck-typed stand-in mirrors its attributes
    return gs


def test_imax_phenotype_expressed_for_the_imax_target_only():
    from osmose.engine.simulate import _imax_phenotype

    sp = np.array([0, 0], dtype=np.int32)
    got = _imax_phenotype(_registry_state("bioen_i_max"), sp)
    assert got == pytest.approx([4.0, 3.1])  # mean 3.5 + alleles + env noise
    assert _imax_phenotype(_registry_state("bioen_r"), sp) is None
    assert _imax_phenotype(None, sp) is None


def _trait_cfg(target: str, bioen: bool) -> dict[str, str]:
    return {
        "evolution.trait.imax.target": target,
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.02",
        "module.bioenergetics.enabled": "true" if bioen else "false",
    }


def test_unread_trait_target_is_an_error():
    from osmose.engine.config import _validate_trait_declarations

    with pytest.raises(ValueError, match="not a trait the engine reads"):
        _validate_trait_declarations(_trait_cfg("species_linf", True), 1, True)
    with pytest.raises(ValueError, match="not a trait the engine reads"):
        _validate_trait_declarations(_trait_cfg("bioen_imax", True), 1, True)  # typo


def test_supported_targets_pass_and_bioen_off_warns():
    from osmose.engine.config import GENETIC_TRAIT_TARGETS, _validate_trait_declarations

    for t in GENETIC_TRAIT_TARGETS:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_trait_declarations(_trait_cfg(t, True), 1, True)
    with pytest.warns(UserWarning, match="will not affect the simulation"):
        _validate_trait_declarations(_trait_cfg("bioen_i_max", False), 1, False)


def test_shipped_baltic_ev_config_validates_cleanly():
    from pathlib import Path

    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig

    root = Path(__file__).resolve().parent.parent / "data" / "baltic_ev"
    master = next(root.glob("*all-parameters*.csv"))
    cfg = OsmoseConfigReader().read(master)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*will not affect the simulation.*")
        EngineConfig.from_dict(cfg)
