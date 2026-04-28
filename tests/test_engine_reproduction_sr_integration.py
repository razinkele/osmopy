"""Integration smoke: B-H caps cumulative egg production over a multi-step run."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import SchoolState


def _base_cfg() -> dict[str, str]:
    """Single-species synthetic config sized to produce non-trivial SSB quickly."""
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "2",
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


def _seed_state(n_schools: int = 5) -> SchoolState:
    """Seed mature schools that will spawn at every step."""
    s = SchoolState.create(
        n_schools=n_schools, species_id=np.zeros(n_schools, dtype=np.int32)
    )
    return s.replace(
        abundance=np.full(n_schools, 10_000.0, dtype=np.float64),
        length=np.full(n_schools, 20.0, dtype=np.float64),
        weight=np.full(n_schools, 0.006 * 20.0**3, dtype=np.float64),  # 48.0
        biomass=np.full(n_schools, 480_000.0, dtype=np.float64),
        age_dt=np.full(n_schools, 24, dtype=np.int32),
    )


def _run_steps(cfg: EngineConfig, n_steps: int) -> float:
    """Run reproduction n_steps times, return total eggs produced across the run."""
    state = _seed_state()
    rng = np.random.default_rng(42)
    total_eggs = 0.0
    for step in range(n_steps):
        n_before = len(state)
        state = reproduction(state, cfg, step=step, rng=rng)
        new_egg_mask = np.zeros(len(state), dtype=np.bool_)
        new_egg_mask[n_before:] = state.is_egg[n_before:]
        total_eggs += float(state.abundance[new_egg_mask].sum())
    return total_eggs


def test_bh_produces_strictly_fewer_eggs_than_linear():
    """Over 12 reproduction steps, B-H caps cumulative eggs vs linear baseline."""
    cfg_lin = EngineConfig.from_dict(_base_cfg())
    eggs_lin = _run_steps(cfg_lin, n_steps=12)

    cfg_bh = EngineConfig.from_dict({
        **_base_cfg(),
        "stock.recruitment.type.sp0": "beverton_holt",
        "stock.recruitment.ssbhalf.sp0": "1000.0",  # ssb_half << per-step SSB
    })
    eggs_bh = _run_steps(cfg_bh, n_steps=12)

    assert eggs_bh < eggs_lin, (
        f"B-H must cap eggs vs linear: linear={eggs_lin:.3e}, bh={eggs_bh:.3e}"
    )
    # ssb_half << SSB → B-H multiplier ≈ ssb_half / SSB → near-zero ratio.
    assert eggs_bh / eggs_lin < 0.1, (
        f"B-H cap should be aggressive at low ssb_half: ratio={eggs_bh/eggs_lin:.4f}"
    )


def test_ricker_collapses_eggs_at_high_ssb():
    """Ricker with ssb_half << SSB drives eggs to near-zero (over-compensation)."""
    cfg_ricker = EngineConfig.from_dict({
        **_base_cfg(),
        "stock.recruitment.type.sp0": "ricker",
        "stock.recruitment.ssbhalf.sp0": "100.0",  # SSB / ssb_half ≈ 24000 → exp(-24000) ≈ 0
    })
    eggs_ricker = _run_steps(cfg_ricker, n_steps=6)
    assert eggs_ricker < 1.0, (
        f"Ricker should collapse to ~0 at SSB/h=24000: got {eggs_ricker}"
    )
