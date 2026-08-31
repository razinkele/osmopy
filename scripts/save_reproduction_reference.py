"""Capture the PRE-EXTRACTION output of `processes.reproduction.reproduction()`.

Feeds `tests/test_engine_bioen_reproduction_parity.py
::test_standard_reproduction_bit_identical_after_extraction`, which proves that factoring
`regulate_recruitment` / `create_egg_schools` / `merge_new_schools` out of `reproduction()`
left the STANDARD (non-bioen) path bit-identical.

!! MUST BE RUN ON A PRE-EXTRACTION TREE !!  Running it on the current tree would record
the post-extraction output and compare the new code against itself, making the test
vacuous. The guard below refuses to run once `regulate_recruitment` exists; to regenerate,
check out a commit before the extraction (39c43a2 or earlier on `c3-bioen-stage1`), run
this, then return to the branch tip. Outputs are gitignored, so a fresh clone simply skips
the test until someone does that.

Saves plain arrays (np.savez, no pickle, never `allow_pickle`) plus a JSON sidecar with
the per-case config dicts, the non-None state field names and the step.

Six cases, so that all four post-SR gate blocks are covered and not just the SR curve.
Each gate is enabled ALONE, so a botched move of any single block fails exactly one case:
  none  shepherd SR only (the identity reference)
  a     shepherd SR + seeding_mode=linear (the seeded species takes the `np.where` branch)
  b     + depensation gate      (config keys only)
  c     + RV gate               (tiny generated CSV)
  d     + recruitment ceiling   (tiny generated CSV, parameterised to BIND)
  e     + thermal gate          (tiny generated CSV)
The gates are skipped for seeded species by design, so every gate is enabled on sp0 (which
has mature schools); sp1 is always the seeded species.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import MortalityCause, SchoolState

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "tests" / "baselines"
FILES = BASE / "reproduction_reference_files"
STEP = 5
SEED = 3


def base_config() -> dict[str, str]:
    """`minimal_config` from tests/test_engine_bioen_integration.py, bioen OFF."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "5",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "20",
        "simulation.nschool.sp1": "15",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        "species.linf.sp0": "15.0",
        "species.linf.sp1": "25.0",
        "species.k.sp0": "0.4",
        "species.k.sp1": "0.3",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.15",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.008",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.1",
        "species.lifespan.sp0": "3",
        "species.lifespan.sp1": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.0",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
        # --- reproduction: make the SR branch live ---
        "stock.recruitment.type.sp0": "shepherd",
        "stock.recruitment.type.sp1": "shepherd",
        "stock.recruitment.ssbhalf.sp0": "50",
        "stock.recruitment.ssbhalf.sp1": "50",
        "stock.recruitment.shape.sp0": "1.2",
        "stock.recruitment.shape.sp1": "0.8",
        # sp1 has no mature school -> SSB == 0 -> the seeding branch fires
        "species.maturity.size.sp0": "8.0",
        "species.maturity.size.sp1": "30.0",
        "species.maturity.age.sp0": "0.5",
        "species.maturity.age.sp1": "0.5",
        "population.seeding.biomass.sp0": "100.0",
        "population.seeding.biomass.sp1": "200.0",
        "population.seeding.year.max": "3",
        "species.sexratio.sp0": "0.55",
        "species.sexratio.sp1": "0.45",
        "species.relativefecundity.sp0": "400.0",
        "species.relativefecundity.sp1": "600.0",
    }


def case_a() -> dict[str, str]:
    cfg = base_config()
    cfg["population.seeding.mode"] = "linear"
    return cfg


def case_b() -> dict[str, str]:
    cfg = base_config()
    cfg["reproduction.depensation.gate.enabled"] = "true"
    cfg["reproduction.depensation.gate.species.enabled.sp0"] = "true"
    cfg["reproduction.depensation.gate.s50.sp0"] = "40.0"
    cfg["reproduction.depensation.gate.theta.sp0"] = "2.0"
    cfg["reproduction.depensation.gate.species.enabled.sp1"] = "true"
    cfg["reproduction.depensation.gate.s50.sp1"] = "120.0"
    cfg["reproduction.depensation.gate.theta.sp1"] = "1.5"
    return cfg


def write_gate_files() -> None:
    FILES.mkdir(parents=True, exist_ok=True)
    rv = ["year,spawning_rv"]
    for y in range(10):
        rv.append(f"{2000 + y},{0.10 + 0.02 * y:.6f}")
    (FILES / "rv_series.csv").write_text("\n".join(rv) + "\n")

    th = ["year,temp_sp0,temp_sp1"]
    for y in range(10):
        th.append(f"{2000 + y},{14.0 + 0.3 * y:.4f},{17.0 + 0.4 * y:.4f}")
    (FILES / "thermal_series.csv").write_text("\n".join(th) + "\n")

    # Deliberately low so the cap BINDS on sp0 (linear eggs are ~2.7e8 at step 5).
    ce = ["season_idx,ceiling_sp0,ceiling_sp1"]
    for s in range(24):
        ce.append(f"{s},{5.0e7 + 1.0e6 * s:.1f},{5.0e8 + 1.0e7 * s:.1f}")
    (FILES / "ceiling_series.csv").write_text("\n".join(ce) + "\n")


def case_c() -> dict[str, str]:
    cfg = base_config()
    cfg["_osmose.config.dir"] = str(FILES)
    cfg["reproduction.rv.gate.enabled"] = "true"
    cfg["reproduction.rv.gate.series.file"] = "rv_series.csv"
    cfg["reproduction.rv.gate.species.enabled.sp0"] = "true"
    cfg["reproduction.rv.gate.mode"] = "mean_preserving"
    cfg["reproduction.rv.gate.start.year"] = "2001"
    return cfg


def case_d() -> dict[str, str]:
    cfg = base_config()
    cfg["_osmose.config.dir"] = str(FILES)
    cfg["reproduction.recruitment.ceiling.enabled"] = "true"
    cfg["reproduction.recruitment.ceiling.series.file"] = "ceiling_series.csv"
    cfg["reproduction.recruitment.ceiling.species.enabled.sp0"] = "true"
    return cfg


def case_e() -> dict[str, str]:
    cfg = base_config()
    cfg["_osmose.config.dir"] = str(FILES)
    cfg["reproduction.thermal.gate.enabled"] = "true"
    cfg["reproduction.thermal.gate.series.file"] = "thermal_series.csv"
    cfg["reproduction.thermal.gate.species.enabled.sp0"] = "true"
    cfg["reproduction.thermal.gate.t50.sp0"] = "18.5"
    cfg["reproduction.thermal.gate.slope.sp0"] = "1.5"
    return cfg


def case_none() -> dict[str, str]:
    """No gate at all - the identity reference every gate case is compared against."""
    return base_config()


def make_state(n: int = 30) -> SchoolState:
    """30 schools: sp0 gets a mature/immature mix, sp1 is entirely immature.

    Built through SchoolState.create()+replace() so every optional field except
    imax_trait is populated (from_seeding in particular — a raw SchoolState(...)
    leaves it None and reproduction()'s batch merge then refuses to concatenate).
    """
    rng = np.random.default_rng(11)
    species_id = np.array([i % 2 for i in range(n)], dtype=np.int32)
    st = SchoolState.create(n_schools=n, species_id=species_id)
    # sp0: alternating long (mature, >= 8 cm) and short; sp1: all short.
    length = np.where(
        (species_id == 0) & (np.arange(n) % 4 == 0),
        rng.uniform(9.0, 20.0, size=n),
        rng.uniform(1.0, 6.0, size=n),
    )
    weight = 0.006 * length**3 * 1e-6
    age_dt = np.array([12 + 3 * (i % 7) for i in range(n)], dtype=np.int32)
    return st.replace(
        abundance=rng.uniform(1e3, 1e6, size=n),
        biomass=weight * rng.uniform(1e3, 1e6, size=n),
        length=length,
        length_start=length.copy(),
        weight=weight,
        age_dt=age_dt,
        trophic_level=np.full(n, 3.0),
        cell_x=rng.integers(0, 10, size=n).astype(np.int32),
        cell_y=rng.integers(0, 10, size=n).astype(np.int32),
        pred_success_rate=np.full(n, 0.5),
        preyed_biomass=rng.uniform(0.0, 1e-5, size=n),
        gonad_weight=rng.uniform(0.0, 1e-6, size=n),
        n_dead=np.zeros((n, len(MortalityCause)), dtype=np.float64),
        first_feeding_age_dt=np.full(n, 1, dtype=np.int32),
    )


OUT_KEYS = ("abundance", "weight", "length", "age_dt", "species_id", "biomass", "is_egg")


def main() -> None:
    from dataclasses import fields

    import osmose.engine.processes.reproduction as repro_mod

    if hasattr(repro_mod, "regulate_recruitment"):
        raise SystemExit(
            "Refusing to run: osmose.engine.processes.reproduction already exposes "
            "regulate_recruitment, so this tree is POST-extraction. Recording the reference "
            "here would compare the new code against itself and make "
            "test_standard_reproduction_bit_identical_after_extraction vacuous. Check out a "
            "pre-extraction commit (39c43a2 or earlier on c3-bioen-stage1) and re-run."
        )

    write_gate_files()
    BASE.mkdir(parents=True, exist_ok=True)

    cases = {
        "none": case_none(),
        "a": case_a(),
        "b": case_b(),
        "c": case_c(),
        "d": case_d(),
        "e": case_e(),
    }
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {"step": STEP, "seed": SEED, "cases": {}}

    state = make_state()
    state_fields = [f.name for f in fields(state) if getattr(state, f.name) is not None]
    for name in state_fields:
        arrays[f"state_{name}"] = np.asarray(getattr(state, name))

    for tag, cfg in cases.items():
        config = EngineConfig.from_dict(dict(cfg))
        out = reproduction(state, config, STEP, np.random.default_rng(SEED), grid_ny=10, grid_nx=10)
        for k in OUT_KEYS:
            arrays[f"out_{tag}_{k}"] = np.asarray(getattr(out, k))
        meta["cases"][tag] = cfg  # type: ignore[index]
        print(
            f"case {tag}: n_out={len(out)} "
            f"eggs_sp0={out.abundance[len(state) :][out.species_id[len(state) :] == 0].sum():.6g} "
            f"eggs_sp1={out.abundance[len(state) :][out.species_id[len(state) :] == 1].sum():.6g}"
        )

    meta["state_fields"] = state_fields
    meta["out_keys"] = list(OUT_KEYS)

    np.savez(BASE / "reproduction_reference_seed3.npz", **arrays)
    (BASE / "reproduction_reference_seed3.json").write_text(json.dumps(meta, indent=1))
    print("wrote", BASE / "reproduction_reference_seed3.npz")
    print("state_fields:", state_fields)


if __name__ == "__main__":
    main()
