"""Percid missing-removals (2026-07-28): realistic percid fishing F (Lever A) +
tunable cormorant predation (Lever B). Fail-first tests; the config edits in the
plan's Tasks 2-3 turn these green. On master's disaggregated 9-species config
cormorant = sp16 (it is sp15 on the aggregate 646a36d branch)."""

import sys

import pandas as pd

from osmose.config import OsmoseConfigReader

CFG = "data/baltic/baltic_all-parameters.csv"


def _cfg():
    return dict(OsmoseConfigReader().read(CFG))


# ---- Task 2: realistic percid fishing F (Lever A) ----


def test_percid_fishing_F_is_elevated_to_coastal_levels():
    cfg = _cfg()
    assert float(cfg["fisheries.rate.base.fsh4"]) >= 0.3  # perch, coastal + recreational
    assert float(cfg["fisheries.rate.base.fsh5"]) >= 0.3  # pikeperch


# ---- Task 3: cormorant predation (Lever B) ----


def test_cormorant_is_a_predator_column_shaped_toward_percids():
    df = pd.read_csv("data/baltic/predation-accessibility.csv", sep=";", index_col=0)
    assert "Cormorant" in df.columns  # predator column added
    # cormorant preys harder on perch than on herring (shaped toward percids)
    assert df.loc["perch", "Cormorant"] > df.loc["herring", "Cormorant"]


def test_cormorant_biomass_multiplier_present():
    cfg = _cfg()
    assert "species.biomass.multiplier.sp16" in cfg  # cormorant = sp16 on this base


def test_phase13_free_set_fixes_percid_F_and_frees_cormorant():
    # The plan's most load-bearing invariant: percid F is FIXED (not in the DE free
    # set) and the cormorant levers ARE free.
    sys.path.insert(0, "scripts")
    import calibrate_baltic as cb

    keys, bounds, x0 = cb.get_phase13_shepherd_params()
    assert "fisheries.rate.base.fsh4" not in keys  # perch F fixed, not optimised
    assert "fisheries.rate.base.fsh5" not in keys  # pikeperch F fixed
    assert "species.biomass.multiplier.sp16" in keys  # cormorant levers present
    assert "predation.ingestion.rate.max.sp16" in keys
    assert len(keys) == len(bounds) == len(x0)


def test_apply_calibration_routes_background_predation_keys():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("ac", "scripts/apply_calibration.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    d = Path("data/baltic")
    assert ac._file_for("species.biomass.multiplier.sp16", d).name == "baltic_param-background.csv"
    assert ac._file_for("predation.ingestion.rate.max.sp16", d).name == "baltic_param-background.csv"
