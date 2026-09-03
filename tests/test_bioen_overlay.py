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

    assert float(cfg["species.bioen.maint.energy.c_m.sp0"]) == C_M
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
