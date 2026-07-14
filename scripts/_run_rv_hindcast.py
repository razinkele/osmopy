import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/razinka/osmopy/scripts")
from baltic_rv_hindcast import ROOT, WINDOW, run_hindcast

r = run_hindcast()  # default seeds (0,1,2,3,4)
d = np.array(r["skill_delta_per_seed"], float)
n_ok = int(np.isfinite(d).sum())
print("rv_ref:", round(float(r["rv_ref"]), 4))
print("skill_delta per seed:", [round(float(x), 3) for x in d])
print(
    "mean skill delta (clim->inter):",
    round(float(np.nanmean(d)), 4),
    "+/-",
    round(float(np.nanstd(d)), 4),
    f"(non-nan seeds: {n_ok}/{len(d)})",
)
obs = pd.read_csv(ROOT / "docs/diagnostics/ices_cod_2732_observed.csv", comment="#")
years = obs["year"].to_numpy()[WINDOW]
print("window sim-yr6-15 =", int(years[0]), "-", int(years[-1]))
print("observed ssb_t   (window):", [round(float(x)) for x in obs["ssb_t"].to_numpy()[WINDOW]])
for m, s in r["means"].items():
    print(f"{m:5s} cod SSB (window):", [round(float(x)) for x in s[WINDOW]])
print("DONE")
