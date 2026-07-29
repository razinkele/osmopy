import numpy as np
import pytest
from osmose.engine.config import _load_depensation_gate
from osmose.engine.processes.depensation_gate import depensation_factor
from osmose.results import total_cod


def test_half_at_s50():
    # A(S50) = 0.5 exactly, any theta
    f = depensation_factor(
        np.array([50_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert f[0] == pytest.approx(0.5)


def test_approaches_zero_at_low_ssb():
    f = depensation_factor(
        np.array([1_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert 0.0 < f[0] < 1e-4


def test_zero_at_ssb_zero():
    f = depensation_factor(np.array([0.0]), np.array([50_000.0]), np.array([4.0]), np.array([True]))
    assert f[0] == 0.0


def test_approaches_one_at_high_ssb():
    f = depensation_factor(
        np.array([5_000_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert f[0] == pytest.approx(1.0, abs=1e-3)


def test_disabled_is_one():
    f = depensation_factor(
        np.array([1_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([False])
    )
    assert f[0] == 1.0


def test_theta_one_boundary():
    # theta=1: A = SSB/(S50+SSB); at S50 still 0.5, still <1 at low SSB
    f = depensation_factor(
        np.array([50_000.0]), np.array([50_000.0]), np.array([1.0]), np.array([True])
    )
    assert f[0] == pytest.approx(0.5)


def test_multi_species_isolation():
    # only enabled species differ from 1.0
    ssb = np.array([1_000.0, 1_000.0])
    s50 = np.array([50_000.0, 50_000.0])
    theta = np.array([4.0, 4.0])
    enabled = np.array([True, False])
    f = depensation_factor(ssb, s50, theta, enabled)
    assert f[0] < 1e-4
    assert f[1] == 1.0


# --- Task 2: config loader ---


def _cfg(**over):
    base = {
        "reproduction.depensation.gate.enabled": "true",
        "reproduction.depensation.gate.species.enabled.sp0": "true",
        "reproduction.depensation.gate.s50.sp0": "60000",
        "reproduction.depensation.gate.theta.sp0": "4.0",
    }
    base.update(over)
    return base


def test_loader_off_returns_triple_of_none():
    assert _load_depensation_gate({}, 2) == (None, None, None)


def test_loader_parses_enabled_species():
    enabled, s50, theta = _load_depensation_gate(_cfg(), 2)
    assert list(enabled) == [True, False]
    assert s50[0] == 60000.0
    assert theta[0] == 4.0


def test_loader_failfast_theta_below_one():
    with pytest.raises(ValueError):
        _load_depensation_gate(_cfg(**{"reproduction.depensation.gate.theta.sp0": "0.5"}), 2)


def test_loader_failfast_s50_nonpositive():
    with pytest.raises(ValueError):
        _load_depensation_gate(_cfg(**{"reproduction.depensation.gate.s50.sp0": "0"}), 2)


def test_loader_failfast_global_on_no_species():
    with pytest.raises(ValueError):
        _load_depensation_gate({"reproduction.depensation.gate.enabled": "true"}, 2)


# --- Task 3: wiring into reproduction() ---

import os  # noqa: E402


def _repro_cfg_dict():
    # Single-species config that produces a large per-step n_eggs at step 0
    # (mirrors tests/test_recruitment_ceiling.py::_repro_cfg_dict).
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "800",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "50000",
    }


def _eggs_produced(new_state, sp=0):
    fresh = (new_state.age_dt == 0) & new_state.is_egg & (new_state.species_id == sp)
    return float(new_state.abundance[fresh].sum())


def test_reproduction_depensation_skips_seeded_step():
    # Empty state -> SSB is seeded from population.seeding.biomass; a strong Allee
    # gate (S50 = 100x the seeding biomass -> A(SSB) ~ 1e-8) must NOT be applied
    # on the seeded bootstrap step, mirroring
    # tests/test_recruitment_ceiling.py::test_reproduction_ceiling_skips_seeded_step.
    from osmose.engine.config import EngineConfig
    from osmose.engine.processes.reproduction import reproduction
    from osmose.engine.state import SchoolState

    base_cfg = EngineConfig.from_dict(_repro_cfg_dict())
    empty = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
    baseline_eggs = _eggs_produced(
        reproduction(empty, base_cfg, step=0, rng=np.random.default_rng(0))
    )
    assert baseline_eggs > 0  # sanity: seeded bootstrap produces eggs

    gated_dict = dict(_repro_cfg_dict())
    gated_dict.update(
        {
            "reproduction.depensation.gate.enabled": "true",
            "reproduction.depensation.gate.species.enabled.sp0": "true",
            # S50 is 100x the seeding biomass: if the gate were (incorrectly)
            # applied on the seeded step, A(SSB) ~ 1e-8 and eggs would collapse
            # to near zero instead of matching the unclipped baseline.
            "reproduction.depensation.gate.s50.sp0": "5000000",
            "reproduction.depensation.gate.theta.sp0": "4.0",
        }
    )
    gated_cfg = EngineConfig.from_dict(gated_dict)
    empty2 = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
    gated_eggs = _eggs_produced(
        reproduction(empty2, gated_cfg, step=0, rng=np.random.default_rng(0))
    )

    assert gated_eggs == pytest.approx(baseline_eggs)  # gate skipped on seeded step


def _run_cod_ssb(overrides, seed=0, n_year=8):
    import tempfile
    from pathlib import Path

    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    tmp = Path(tempfile.mkdtemp())
    base = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    raw = {**base, "simulation.time.nyear": str(n_year), "output.ssb.enabled": "true", **overrides}
    return total_cod(PythonEngine().run_in_memory(raw, seed=seed).ssb())


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="real-engine Baltic, core-count-sensitive"
)
def test_gate_off_is_bit_identical_to_baseline():
    base = _run_cod_ssb({})
    off = _run_cod_ssb({"reproduction.depensation.gate.enabled": "false"})
    np.testing.assert_array_equal(base, off)


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="real-engine Baltic, core-count-sensitive"
)
def test_gate_on_changes_cod_recruitment():
    base = _run_cod_ssb({})
    on = _run_cod_ssb(
        {
            "reproduction.depensation.gate.enabled": "true",
            "reproduction.depensation.gate.species.enabled.sp0": "true",
            "reproduction.depensation.gate.s50.sp0": "200000",
            "reproduction.depensation.gate.theta.sp0": "4.0",
        }
    )
    assert not np.array_equal(base, on)  # a strong Allee at high S50 must move cod


# --- Task 4: Java-engine guard ---

from osmose.runner import java_engine_block_reason  # noqa: E402


def test_java_engine_blocked_for_depensation_gate():
    reason = java_engine_block_reason(
        {"reproduction.depensation.gate.enabled": "true"}, jar_version="4.4.1"
    )
    assert reason is not None and "depensation" in reason.lower()


def test_java_engine_not_blocked_when_gate_off():
    reason = java_engine_block_reason(
        {"reproduction.depensation.gate.enabled": "false"}, jar_version="4.4.1"
    )
    # off => the gate itself does not block (other guards may still apply for other configs)
    assert reason is None or "depensation" not in (reason or "").lower()
