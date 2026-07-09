"""Smoke: build a cod-dominated and a clupeid-dominated STANDING stock at t=0 via the
warm-start primitive, on the real Baltic config. Demonstrates the two initial conditions
the definitive bistability/invasion test needs (which egg-only, single-cod-axis ICs could
not construct). No full simulation — just the t=0 population build."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/razinka/osmopy")
sys.path.insert(0, str(ROOT))

from osmose.config.reader import OsmoseConfigReader  # noqa: E402
from osmose.engine import PythonEngine  # noqa: E402
from osmose.engine.config import EngineConfig  # noqa: E402
from osmose.engine.initialization import build_initial_population  # noqa: E402

NAMES = ["cod", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt", "stickleback"]

base = OsmoseConfigReader().read(str(ROOT / "data" / "baltic" / "baltic_all-parameters.csv"))
base["module.population.initialisation.enabled"] = "true"
grid = PythonEngine()._resolve_grid(base)

for label, overrides in (
    ("cod-dominated", {"population.seeding.biomass.sp0": "300000"}),
    (
        "clupeid-dominated",
        {
            "population.seeding.biomass.sp0": "1000",
            "population.seeding.biomass.sp1": "1500000",
            "population.seeding.biomass.sp2": "1500000",
        },
    ),
):
    c = dict(base)
    c.update(overrides)
    ec = EngineConfig.from_dict(c)
    st = build_initial_population(ec, grid, np.random.default_rng(0))
    print(
        f"\n=== {label}: {len(st)} schools; egg-seeding disabled (seeding_max_step[cod]={int(ec.seeding_max_step[0])}) ==="
    )
    for sp in range(min(8, ec.n_species)):
        m = st.species_id == sp
        if m.any():
            bm = float((st.abundance[m] * st.weight[m]).sum())
            ages = sorted({int(a) for a in (st.age_dt[m] // ec.n_dt_per_year)})
            print(
                f"  sp{sp} {NAMES[sp]:11s}: {bm:>13,.0f} t  age-classes(y)={ages[:6]}"
                f"{'...' if len(ages) > 6 else ''}"
            )
