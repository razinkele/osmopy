"""Percid-removals feasibility gate (Task 4) — the cheap go/no-go before the 4-8h
re-calibration. Runs three 40-yr contrasts to ATTRIBUTE the percid response to
each lever (so Lever B can't mask a skipped Lever A, per the deep review):

  both           : percid F fixed (0.40/0.50) + cormorant maxed (mult 3.0, ingest 80)
  F-only         : percid F fixed              + cormorant OFF   (mult 0)
  cormorant-only : percid F baseline (~0.03)   + cormorant maxed

GO if perch/smelt move toward their envelopes AND each moving lever is causally
responsible (the F-only and cormorant-only contrasts each move perch a non-trivial
share) AND the well-assessed stocks (cod/herring/sprat/flounder/stickleback) stay
in-envelope. Single seed 42 — a SCREEN; the authoritative check is the Task-5
50-yr x 5-seed certification.
"""

import warnings
from pathlib import Path


warnings.filterwarnings("ignore")
from osmose.config import OsmoseConfigReader  # noqa: E402
from osmose.engine import PythonEngine  # noqa: E402
from osmose.results import OsmoseResults  # noqa: E402

ENVELOPE = {
    "cod": (60000, 250000), "herring": (800000, 3000000), "sprat": (800000, 2500000),
    "flounder": (20000, 100000), "perch": (8000, 50000), "pikeperch": (4000, 25000),
    "smelt": (20000, 120000), "stickleback": (50000, 500000),
}
WELL_ASSESSED = ["cod", "herring", "sprat", "flounder", "stickleback"]
OUT = Path("/tmp/claude-1000/-home-razinka-osmopy/d89da751-bed8-4745-b75d-c26886735ab3/scratchpad")

base = dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))
base["simulation.time.nyear"] = "40"
BASE_PERCH_F, BASE_PP_F = "0.02940885412976362", "0.009543135539394263"

CONTRASTS = {
    "both": {"species.biomass.multiplier.sp15": "3.0", "predation.ingestion.rate.max.sp15": "80.0"},
    "F-only": {"predation.ingestion.rate.max.sp15": "0.0"},  # cormorant present, eats nothing
    "cormorant-only": {
        "species.biomass.multiplier.sp15": "3.0", "predation.ingestion.rate.max.sp15": "80.0",
        "fisheries.rate.base.fsh4": BASE_PERCH_F, "fisheries.rate.base.fsh5": BASE_PP_F,
    },
}


def run(name, over):
    cfg = dict(base)
    cfg.update(over)
    r = PythonEngine().run(cfg, output_dir=OUT / f"gate_{name}", seed=42)
    if r.returncode != 0:
        return None
    res = OsmoseResults(OUT / f"gate_{name}", strict=False)
    bio = res.biomass().select_dtypes("number")
    res.close()
    return bio.iloc[-10:].mean()


results = {name: run(name, over) for name, over in CONTRASTS.items()}

print(f"\n{'species':12s} {'both':>12s} {'F-only':>12s} {'corm-only':>12s} {'envelope':>18s}")
for sp, (lo, hi) in ENVELOPE.items():
    vals = {k: (v.get(sp, float('nan')) if v is not None else float('nan')) for k, v in results.items()}
    print(f"{sp:12s} {vals['both']:12.0f} {vals['F-only']:12.0f} {vals['cormorant-only']:12.0f} "
          f"{f'[{lo},{hi}]':>18s}")

b = results["both"]
if b is not None:
    perch_in = ENVELOPE["perch"][0] <= b["perch"] <= ENVELOPE["perch"][1] * 2
    wa_ok = all(ENVELOPE[s][0] <= b.get(s, 0) <= ENVELOPE[s][1] for s in WELL_ASSESSED)
    # attribution: does perch move under EACH lever alone?
    f_moves = results["F-only"] is not None and results["F-only"]["perch"] < b["perch"] * 3
    c_moves = results["cormorant-only"] is not None and results["cormorant-only"]["perch"] < b["perch"] * 3
    print(f"\nperch toward envelope (<=2x upper): {perch_in}")
    print(f"well-assessed stocks all in-envelope: {wa_ok}")
    print(f"perch responds to F-only: {f_moves} | to cormorant-only: {c_moves}")
    print("\nGATE:", "GO" if (perch_in and wa_ok) else "review — see per-species table + attribution")
