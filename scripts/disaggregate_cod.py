#!/usr/bin/env python
"""Disaggregate the aggregated Baltic cod stock into cod_west (sp0) + cod_east (sp8).

Phase 1, Task 2 of docs/superpowers/plans/2026-07-24-baltic-cod-disaggregation-phase1.md.

Steps (idempotency-guarded — aborts if already disaggregated):
  1. append_focal_species(): shift LTL/background sp>=8 up one, bump nspecies 8->9,
     freeing sp8 for a new focal species.
  2. Rename species.name.sp0  cod -> cod_west, split its seeding biomass.
  3. Write cod_east's full scalar parameter set at sp8.
  4. Create the cod_east summer-shifted spawning seasonality file.

Deferred to later tasks (this only needs to PARSE-load): movement maps (Task 3),
predation + fishery-catchability matrices (Task 4), ICES targets (Task 5),
elevated M + RV recruitment gate reassignment + fishing fsh8 + SR recalibration
(Tasks 6-7).

Eastern Baltic cod (cod.27.24-32) science encoded below — impaired condition, NOT
a heritable slow-growth trait (fidelity review): Linf/K equal western, the eastern
signal is in condition, maturity, egg buoyancy, lifespan, plus (later) M and RV.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reindex_species = _load("reindex_species")
apply_calibration = _load("apply_calibration")
set_key = apply_calibration.set_key

DEFAULT_CONFIG_DIR = _REPO / "data" / "baltic"

# cod_east (sp8) scalar params, mapped file -> {key: value}. Values that differ
# from cod_west carry the eastern-stock rationale inline.
COD_EAST_PARAMS: dict[str, dict[str, str]] = {
    "baltic_param-species.csv": {
        "species.type.sp8": "focal",
        "species.name.sp8": "cod_east",
        "species.lifespan.sp8": "15",  # truncated age structure (western 20)
        "species.linf.sp8": "110.0",  # SAME as western — condition, not slow growth
        "species.k.sp8": "0.15",  # same
        "species.t0.sp8": "-0.20",  # same
        "species.vonbertalanffy.threshold.age.sp8": "0.5",
        # ~22% below western 0.00870 — documented eastern condition decline
        "species.length2weight.condition.factor.sp8": "0.00680",
        "species.length2weight.allometric.power.sp8": "3.050",
        "species.egg.size.sp8": "0.17",  # larger/more buoyant (western 0.15) — low-salinity spawning
        "species.egg.weight.sp8": "0.0012",  # larger egg (western 0.001)
        "species.maturity.size.sp8": "22.0",  # early maturation at small size (western 38.0)
        "species.relativefecundity.sp8": "400",  # lower, condition-linked (western 500)
        "species.sexratio.sp8": "0.5",
    },
    "baltic_param-additional-mortality.csv": {
        # copy western; Task 6 elevates M (the collapse needs doubled M + RV failure)
        "mortality.additional.rate.sp8": "1.2545949046281932",
        "mortality.additional.larva.rate.sp8": "243.7646732526793",
    },
    "baltic_param-starvation.csv": {
        "mortality.starvation.rate.max.sp8": "0.3",
    },
    "baltic_param-out-mortality.csv": {
        "mortality.out.rate.sp8": "0",
    },
    "baltic_param-predation.csv": {
        "predation.efficiency.critical.sp8": "0.57",
        "predation.ingestion.rate.max.sp8": "3.5",
        "predation.predprey.sizeratio.max.sp8": "50",
        "predation.predprey.sizeratio.min.sp8": "3.5",
        "predation.predprey.stage.threshold.sp8": "null",
    },
    "baltic_param-reproduction.csv": {
        "reproduction.season.file.sp8": "reproduction/reproduction-seasonality-sp8.csv",
        "stock.recruitment.type.sp8": "shepherd",
        "stock.recruitment.ssbhalf.sp8": "60000.0",  # placeholder; Task 7 calibrates
        "stock.recruitment.shape.sp8": "1.5",  # placeholder; Task 7 calibrates
    },
    "baltic_param-init-pop.csv": {
        "population.seeding.biomass.sp8": "100000",  # eastern historically the larger stock
    },
    "baltic_param-movement.csv": {
        "movement.distribution.method.sp8": "maps",  # maps added in Task 3
        "movement.randomwalk.range.sp8": "2",
        "movement.salinity.gate.species.enabled.sp8": "true",
    },
    "baltic_param-output.csv": {
        "output.cutoff.age.sp8": "0.5",
        "output.diet.stage.threshold.sp8": "10;30",
    },
    "baltic_all-parameters.csv": {
        "simulation.nschool.sp8": "50",
    },
}


def _cod_east_seasonality() -> str:
    """cod_west spring curve shifted +4 timesteps (~2 months) → eastern summer peak
    (Jul-Aug deep-basin spawning). Sums to 1.0."""
    western = [
        0.021739, 0.043478, 0.065217, 0.086957, 0.130435, 0.152174,
        0.152174, 0.130435, 0.086957, 0.065217, 0.043478, 0.021739,
    ]
    n = 24
    shift = 4  # ~2 months later than western
    curve = [0.0] * n
    start = 4 + shift  # western block begins at ts 4
    for i, v in enumerate(western):
        curve[start + i] = v
    lines = ['"Time (year)";"Cod_east"']
    for i in range(n):
        lines.append(f'"{i / n:.9f}";"{curve[i]:.6f}"')
    return "\n".join(lines) + "\n"


def disaggregate(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> None:
    config_dir = Path(config_dir)

    # Idempotency guard
    nspecies = reindex_species._read_int_key(config_dir, "simulation.nspecies")
    if nspecies != 8:
        raise SystemExit(
            f"expected simulation.nspecies=8 (undisaggregated); found {nspecies}. Aborting."
        )

    # Step 1: shift LTL/background up, bump nspecies -> 9, free sp8
    shifts = reindex_species.append_focal_species(config_dir)
    print(f"reindexed: shifts={shifts}, nspecies 8->9")

    # Step 2: rename cod -> cod_west, split its seeding biomass
    species_csv = config_dir / "baltic_param-species.csv"
    set_key(species_csv, "species.name.sp0", "cod_west")
    initpop_csv = config_dir / "baltic_param-init-pop.csv"
    set_key(initpop_csv, "population.seeding.biomass.sp0", "50000")  # western smaller stock
    print("renamed cod -> cod_west; split seeding biomass (west 50 kt, east 100 kt)")

    # Step 3: write cod_east scalar params
    for filename, params in COD_EAST_PARAMS.items():
        path = config_dir / filename
        for key, value in params.items():
            set_key(path, key, value)
    print(f"wrote cod_east (sp8) scalar params across {len(COD_EAST_PARAMS)} files")

    # Step 4: cod_east spawning seasonality (summer-shifted)
    season_path = config_dir / "reproduction" / "reproduction-seasonality-sp8.csv"
    season_path.write_text(_cod_east_seasonality(), encoding="utf-8")
    print(f"created {season_path.relative_to(config_dir)}")


if __name__ == "__main__":
    disaggregate()
    print("cod disaggregation (Task 2) complete.")
