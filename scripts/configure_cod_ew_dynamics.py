#!/usr/bin/env python
"""Configure cod_east/cod_west recruitment, mortality and fishing (Phase 1 Task 6).

  1. RV recruitment gate -> cod_east ONLY (sp8), mode raw_cap (ref 250): the
     eastern collapse must emerge from recruitment failure in low reproductive-
     volume years, not just variability (RV-gate plan Task-3 go/no-go). cod_west
     (sp0) gets standard Shepherd recruitment, gate disabled.
  2. Elevated additional mortality on cod_east (~doubled): the eastern collapse
     needs elevated M AND recruitment failure together (fidelity review).
  3. Fishing: separate trawlcod_east fishery (fsh8) for cod_east at a low rate
     (2019 eastern moratorium), cod_west keeps trawlcod (fsh0). Expands the
     name-labeled catchability matrix and bumps simulation.nfisheries 8 -> 9.

Starting values only — Task 7's DE calibration tunes ssbhalf/shape/M/F.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
CONFIG_DIR = _HERE.parent / "data" / "baltic"

spec = importlib.util.spec_from_file_location("apply_calibration", _HERE / "apply_calibration.py")
apply_calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_calibration)
set_key = apply_calibration.set_key


def configure_recruitment_and_mortality() -> None:
    repro = CONFIG_DIR / "baltic_param-reproduction.csv"
    set_key(repro, "reproduction.rv.gate.mode", "raw_cap")
    set_key(repro, "reproduction.rv.gate.ref", "250")  # historical-high RV; caps low-RV years
    set_key(repro, "reproduction.rv.gate.species.enabled.sp0", "false")  # cod_west: standard SR
    set_key(repro, "reproduction.rv.gate.species.enabled.sp8", "true")  # cod_east: RV-gated
    set_key(repro, "stock.recruitment.ssbhalf.sp0", "15000.0")  # cod_west western Bpa (was cod 120k)

    addmort = CONFIG_DIR / "baltic_param-additional-mortality.csv"
    set_key(addmort, "mortality.additional.rate.sp8", "2.5")  # ~doubled M (hypoxia, seals, parasites)


FSH8 = {
    "fisheries.name.fsh8": "trawlcod_east",
    "fisheries.selectivity.type.fsh8": "0",
    "fisheries.selectivity.a50.fsh8": "2.0",
    "fisheries.rate.base.fsh8": "0.01",  # low — 2019 eastern moratorium; calibratable
    "fisheries.period.number.fsh8": "1",
    "fisheries.rate.byperiod.fsh8": "1",
    "fisheries.period.start.fsh8": "0",
    "fisheries.seasonality.fsh8": ";".join(["0.04167"] * 24),
}


def configure_fishing() -> None:
    fishing = CONFIG_DIR / "baltic_param-fishing.csv"
    set_key(fishing, "simulation.nfisheries", "9")
    for key, val in FSH8.items():
        set_key(fishing, key, val)

    # Expand the name-labeled catchability matrix: rename cod->cod_west row, add
    # cod_east row + trawlcod_east column (col 8 -> fsh8).
    catch = CONFIG_DIR / "fishery-catchability.csv"
    df = pd.read_csv(catch, index_col=0)
    df = df.rename(index={"cod": "cod_west"})
    df["trawlcod_east"] = 0
    df.loc["cod_east"] = 0
    df.loc["cod_east", "trawlcod_east"] = 1
    # reorder rows so cod_east sits at position 8 (matches its sp index)
    order = ["cod_west", "herring", "sprat", "flounder", "perch", "pikeperch",
             "smelt", "stickleback", "cod_east"]
    df = df.reindex(order)
    df.to_csv(catch)
    print("expanded catchability matrix + fsh8 (trawlcod_east); nfisheries 8->9")


if __name__ == "__main__":
    configure_recruitment_and_mortality()
    configure_fishing()
    print("cod E/W dynamics (Task 6 config) complete.")
