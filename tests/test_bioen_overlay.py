"""Pin ``tests/_bioen_overlay.py``'s ``C_M`` to a MATERIAL maintenance fraction.

``BIOEN_OVERLAY``'s whole reason to exist (see its module docstring, and
``.superpowers/sdd/2026-08-30-baltic-c3-bioen-stage1/task-6-carried-items.md`` item A) is
that a low ``c_m`` makes ``E_net == E_gross`` at every abundance -- bioen starvation
becomes exactly zero and structurally untestable. This file is the guard against that:
if ``C_M`` is ever retuned back down (e.g. copied from ``data/baltic_ev``'s production
value, ~0.001), these tests fail loudly instead of every downstream bioen-Numba-kernel
test silently exercising a starvation-free budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine.config import EngineConfig
from osmose.engine.processes.energy_budget import energy_terms
from tests._bioen_overlay import C_M, apply_overlay
from tests.test_engine_bioen_budget_parity import _three_schools


def test_c_m_reaches_the_pinned_ratio_at_reference_temperature():
    """At T=10 degC (the ``_three_schools`` convention), C_M=1e12 reproduces the exact
    ratio quoted as provenance in task-6-carried-items.md item A: 0.81277.../10.20798...
    """
    weight, abundance, ingestion, *_ = _three_schools()
    e_gross, e_maint, _ = energy_terms(
        ingestion, weight, abundance, 10.0, 0.7, C_M, 0.8, 0.65, 1.0, 1.0, 24
    )
    ratio = e_maint / e_gross
    # Schools 0 and 1: same per-fish weight, adult-scale fish.
    np.testing.assert_allclose(ratio[0], 0.8127739316201607, rtol=1e-9)
    np.testing.assert_allclose(ratio[1], 0.8127739316201607, rtol=1e-9)
    # School 2: small juvenile, ingestion far outpaced by maintenance -- deep starvation.
    assert ratio[2] > 5.0


def test_c_m_is_material_not_just_at_the_reference_temperature():
    """A retune that keeps the T=10 ratio near 0.8 but happens to zero it out at other
    temperatures would still defeat behaviour 3 (bioen starvation) on every real run,
    since BIOEN_OVERLAY itself runs at T=7.0. Guard the actual operating point too.
    """
    weight, abundance, ingestion, *_ = _three_schools()
    e_gross, e_maint, _ = energy_terms(
        ingestion, weight, abundance, 7.0, 0.7, C_M, 0.8, 0.65, 1.0, 1.0, 24
    )
    ratio = e_maint / e_gross
    # Lower than the T=10 ratio (Arrhenius: maintenance falls as temperature drops), but
    # still a material fraction of intake -- not the ~1e-8 the un-fixed c_m produced.
    assert 0.3 < ratio[0] < 0.95
    np.testing.assert_allclose(ratio[0], 0.6111071362185285, rtol=1e-9)


def test_apply_overlay_c_m_matches_module_constant(tmp_path: Path):
    """The ratio pinned above must be reachable through the actual config path -- not
    just the bare constant -- so a change to ``apply_overlay``'s key spelling can't
    silently stop writing ``c_m`` at all while ``C_M`` itself stays correct.
    """
    demo = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    apply_overlay(cfg, n_species=9, background_indices=[15, 16])

    # EVERY focal index, not just sp0: an off-by-one in apply_overlay's `range(n_species)`
    # would leave the LAST species at the engine's c_m default of 0.0 -- i.e. permanently
    # starvation-free -- with an sp0-only assertion still green.
    for i in range(9):
        assert float(cfg[f"species.bioen.maint.energy.c_m.sp{i}"]) == C_M, (
            f"sp{i} did not receive c_m"
        )
        assert cfg[f"species.bioen.assimilation.sp{i}"] == "0.7"
        assert cfg[f"species.maturity.eta.sp{i}"] == "1"
    assert float(cfg["temperature.value"]) == pytest.approx(7.0)

    weight, abundance, ingestion, *_ = _three_schools()
    beta = float(cfg["species.beta.sp0"])
    assimilation = float(cfg["species.bioen.assimilation.sp0"])
    e_maint_energy = float(cfg["species.bioen.maint.e.maint.sp0"])
    temp = float(cfg["temperature.value"])
    c_m = float(cfg["species.bioen.maint.energy.c_m.sp0"])

    e_gross, e_maint, _ = energy_terms(
        ingestion, weight, abundance, temp, assimilation, c_m, beta, e_maint_energy, 1.0, 1.0, 24
    )
    ratio = e_maint / e_gross
    assert 0.3 < ratio[0] < 0.95


def test_apply_overlay_survives_engine_config_parsing(tmp_path: Path):
    """Read the overlay back through ``EngineConfig.from_dict``, not out of the dict.

    Every other test here re-reads the same dict ``apply_overlay`` just wrote, so a
    MISSPELLED key would round-trip perfectly while the engine silently fell back to its
    default (``c_m`` defaults to 0.0 -- starvation-free -- and ``eta``/``e.maint`` to
    values that are not the overlay's). Only the parser can tell a wrong key name from a
    right one.
    """
    demo = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    apply_overlay(cfg, n_species=9, background_indices=[15, 16])

    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_enabled

    n_focal = 9
    np.testing.assert_array_equal(ec.bioen_c_m[:n_focal], np.full(n_focal, C_M))
    np.testing.assert_array_equal(ec.bioen_assimilation[:n_focal], np.full(n_focal, 0.7))
    np.testing.assert_array_equal(ec.bioen_eta[:n_focal], np.ones(n_focal))
    np.testing.assert_array_equal(ec.bioen_e_maint[:n_focal], np.full(n_focal, 0.65))
    np.testing.assert_array_equal(ec.bioen_tp[:n_focal], np.full(n_focal, 10.0))
    np.testing.assert_array_equal(ec.bioen_e_mobi[:n_focal], np.full(n_focal, 0.65))
    np.testing.assert_array_equal(ec.bioen_e_d[:n_focal], np.full(n_focal, 1.5))
    np.testing.assert_array_equal(ec.bioen_r[:n_focal], np.full(n_focal, 0.2))
    np.testing.assert_array_equal(ec.bioen_m1[:n_focal], np.zeros(n_focal))
    assert np.all(ec.bioen_m0[:n_focal] > 0.0), "m0 must be copied from maturity.size"
    # Task 0 recorded this as a known gap: BIOEN_OVERLAY does NOT set k_for, so FORAGING
    # is inert. Pinned here so plan Task 2 discovers it from a failing expectation rather
    # than from a silently-zero `n_dead[:, FORAGING]` witness.
    np.testing.assert_array_equal(ec.bioen_k_for[:n_focal], np.zeros(n_focal))


def test_apply_overlay_copies_maturity_size_into_m0(tmp_path: Path):
    demo = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    original_size = cfg["species.maturity.size.sp0"]
    apply_overlay(cfg, n_species=9, background_indices=[15, 16])
    assert cfg["species.maturity.m0.sp0"] == original_size
    assert cfg["species.maturity.m1.sp0"] == "0"


def test_apply_overlay_converts_background_ingestion_to_per_timestep(tmp_path: Path):
    """Ledger ruling R1: Java's ``getMaxPredationRate`` early-returns for background
    predators WITHOUT the ``/nStepYear`` the focal branch applies, so background
    ``predation.ingestion.rate.max`` must be authored already per-time-step.
    """
    demo = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    annual_sp15 = float(cfg["predation.ingestion.rate.max.sp15"])
    annual_sp16 = float(cfg["predation.ingestion.rate.max.sp16"])
    ndt = float(cfg["simulation.time.ndtperyear"])

    apply_overlay(cfg, n_species=9, background_indices=[15, 16])

    assert float(cfg["predation.ingestion.rate.max.sp15"]) == pytest.approx(annual_sp15 / ndt)
    assert float(cfg["predation.ingestion.rate.max.sp16"]) == pytest.approx(annual_sp16 / ndt)
    assert cfg["species.beta.sp15"] == "0.8"
    assert cfg["species.beta.sp16"] == "0.8"


def test_apply_overlay_raises_on_missing_background_ingestion_key():
    """A missing background ingestion key must fail loudly, not silently default to 0
    (which would disable that background predator's cap entirely and unmoor the two
    paths from disagreeing loudly to agreeing quietly on the wrong number)."""
    cfg = {
        "species.maturity.size.sp0": "38.0",
        "simulation.time.ndtperyear": "24",
        "simulation.nspecies": "1",
        "species.type.sp15": "background",
    }
    with pytest.raises(KeyError):
        apply_overlay(cfg, n_species=1, background_indices=[15])


def test_apply_overlay_raises_on_n_species_mismatch(tmp_path: Path):
    """A caller passing the wrong n_species (e.g. copying baltic_ev's 8 for baltic's 9)
    would silently leave one focal species at c_m's engine default (0.0) -- starvation-free
    -- rather than erroring; ``simulation.nspecies`` in the config is the ground truth.
    """
    demo = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    assert cfg["simulation.nspecies"] == "9"
    with pytest.raises(ValueError, match="n_species"):
        apply_overlay(cfg, n_species=8, background_indices=[15, 16])


def test_apply_overlay_raises_on_background_index_below_n_species(tmp_path: Path):
    demo = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    with pytest.raises(ValueError, match="background"):
        apply_overlay(cfg, n_species=9, background_indices=[8, 16])  # 8 is focal, not background


def test_apply_overlay_raises_on_background_index_not_typed_background(tmp_path: Path):
    demo = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    with pytest.raises(ValueError, match="species.type"):
        apply_overlay(cfg, n_species=9, background_indices=[15, 99])  # sp99 doesn't exist
