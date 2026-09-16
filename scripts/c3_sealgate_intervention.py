"""SEALGATE: is GreySeal the thing that kills cod_west between 10 and 20 cm?

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. Under the C3 bioen overlay with
`seeding.year.max = 1`, cod_west goes extinct, real SSB is exactly 0.0 t, and its occupied size bins
stop at [15,20) cm against the 38 cm it needs to mature. Growth is NOT the cause -- measured intake
is 90-100% of the allometric cap at every size, `m_share` 0.13-0.20 (below the fit's 0.30 target),
and specific growth 13.8% of body weight per step in the top occupied bin. It is ATTRITION: fish
grow well and disappear.

An earlier intervention appeared to rule out predation by zeroing cod_west's PREY row at every age
and finding the ceiling unmoved. **That arm did not remove predation**, and the reason is a defect:

  * GreySeal (sp15) is declared a background predator (`baltic_param-background.csv:16-22`,
    `predation.ingestion.rate.max.sp15 = 13.0`, size classes 110 and 170 cm, prey size ratio 3-12)
    but has **no predator COLUMN** in `predation-accessibility.csv` -- the header runs
    `cod_west ... Benthos, Cormorant` and stops.
  * `AccessibilityMatrix.resolve_name("GreySeal")` therefore returns `None`, and
    `compute_school_indices` leaves `pred_access_idx == -1` for every seal school.
  * The production kernel (`mortality.py:1172-1180`) defaults `access_coeff = 1.0` and only
    overwrites it inside `if p_acc >= 0 and q_acc >= 0:`. A `-1` skips that block ENTIRELY --
    including the `access_coeff <= 0: continue` test.

So **a `-1` does not mean "inaccessible"; it means full accessibility survives.** GreySeal has been
eating cod_west at coefficient **1.0**, twenty times the 0.05 every listed predator is capped at,
and zeroing cod_west's prey row cannot touch it because there is no column to zero. The seal's prey
window -- 9.2-36.7 cm for a 110 cm animal and 14.2-56.7 cm for a 170 cm one -- covers the whole
10-20 cm band where cod_west vanishes, and continues past 38 cm.

The intervention. Add the missing GreySeal COLUMN and set exactly one cell. Nothing else changes:
not growth, not bioenergetics, not reproduction, not any other predator, not cod_west's own prey
row, not what cod_west eats (that is its own column, untouched).

Arms (8 yr, `seeding.year.max = 1`, seed 42 -- identical to every published arm):

    baseline   bioen OFF, production matrix          -- the control that sustains
    bioen      bioen ON, production matrix           -- the published collapsing arm
    sham       + GreySeal column, 1.0 for EVERY prey -- MUST reproduce `bioen` exactly
    treat      + GreySeal column, 1.0 for every prey EXCEPT cod_west = 0.0

`sham` is the load-bearing control. Writing 1.0 everywhere is supposed to be a no-op, because that
is precisely what the `-1` path already produces. If `sham` does NOT reproduce `bioen`, then the
`-1`-means-full-accessibility reading is wrong and nothing else here may be read. Note the resource
rows (Diatoms..Benthos) get 1.0 too: the seal currently eats them at an implicit 1.0 as well, so
writing 0 or omitting them would silently starve it and confound the contrast.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Primary instrument is cod_west's largest OCCUPIED size bin plus real SSB (run-total SSB minus the
year-1 seeding contribution, `24 x population.seeding.biomass.sp0`).

WITH A MAGNITUDE FLOOR, added because the previous pre-registration lacked one and would have fired
on a biologically dead remnant: "recovery" requires real SSB > 1% of the bioen-off control's real
SSB (129 265.5 t -> floor 1 292.7 t), OR the largest occupied size bin reaching the 38 cm maturity
length. A few tonnes of spawners is not a rescue.

  (1) sham reproduces bioen AND treat clears the floor
        -> SEAL CONFIRMED: GreySeal, at an accessibility it was never meant to have, is what
           removes cod_west before maturity. The C3 collapse is then a CONFIG/ENGINE DEFECT, not
           a bioenergetics result.
  (2) sham reproduces bioen AND treat does not clear the floor
        -> SEAL REFUTED as the binding killer. Predation by the seal is real but not sufficient;
           the search continues, with the seal now correctly accounted for.
  (3) sham does NOT reproduce bioen
        -> INCONCLUSIVE, and the finding is that the -1 path does not behave as traced. Everything
           above must be re-derived before any biology is read.

Engagement checks:
  E1 `sham` and `treat` differ from each other in the loaded matrix at [cod_west, GreySeal];
  E2 GreySeal resolves to a real predator column in both (it must NOT still be -1);
  E3 baseline sustains all nine species.

======================================================================================================

Run: .venv/bin/python scripts/c3_sealgate_intervention.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import osmose.engine.processes.reproduction as repro_mod  # noqa: E402
from osmose.config import OsmoseConfigReader  # noqa: E402
from osmose.demo import osmose_demo  # noqa: E402
from osmose.engine import PythonEngine  # noqa: E402
from osmose.engine.config import EngineConfig  # noqa: E402

N_YEAR = 8
SEED = 42
SUBJECT = "cod_west"
SUBJECT_SP = 0
SEAL = "GreySeal"
MATURITY_CM = 38.0
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIG_REGULATE = repro_mod.regulate_recruitment


def write_with_seal_column(src: Path, dst: Path, cod_west_value: float) -> None:
    """Append a GreySeal predator COLUMN: 1.0 for every prey row, `cod_west_value` for cod_west.

    1.0 everywhere reproduces the current `-1` behaviour exactly (the kernel's default
    `access_coeff`), so the sham arm is a true no-op. Resource rows get 1.0 as well -- the seal
    eats them at an implicit 1.0 today, and zeroing them would starve it and confound the contrast.
    """
    rows = list(csv.reader(src.open(), delimiter=";"))
    out = [rows[0] + [SEAL]]
    for row in rows[1:]:
        if not row:
            continue
        val = cod_west_value if row[0].strip() == SUBJECT else 1.0
        out.append(row + [repr(float(val))])
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def seal_cell(cfg_dict: dict) -> tuple[float, bool]:
    """Read [cod_west prey, GreySeal predator] back out of a CONSTRUCTED EngineConfig."""
    try:
        cfg = EngineConfig.from_dict(dict(cfg_dict))
    except Exception as exc:
        print(f"    (E1/E2 config build failed: {exc})")
        return float("nan"), False
    acc = getattr(cfg, "stage_accessibility", None)
    if acc is None or not hasattr(acc, "prey_lookup"):
        return float("nan"), False
    seal_name = acc.resolve_name(SEAL)
    prey = acc.prey_lookup.get(SUBJECT) or []
    pred = acc.pred_lookup.get(seal_name or "") or []
    if not prey or not pred:
        return float("nan"), False
    return float(acc.raw_matrix[prey[0].matrix_index, pred[0].matrix_index]), True


def run_arm(raw, cfg_dir, overlay, seal_value, focal):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg["population.seeding.year.max"] = "1"
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    if seal_value is not None:
        name = f"acc-seal{int(seal_value * 100):03d}.csv"
        write_with_seal_column(cfg_dir / "predation-accessibility.csv", cfg_dir / name, seal_value)
        cfg["predation.accessibility.file"] = name

    n_sp = int(raw["simulation.nspecies"])
    ssb = np.zeros(n_sp)
    seeded = np.zeros(n_sp)

    def wrapper(n_eggs_linear, ssb_in, seeded_this_step, config, step):
        ssb[:] += np.asarray(ssb_in, dtype=np.float64)
        seeded[:] += np.asarray(seeded_this_step, dtype=bool).astype(np.float64)
        return _ORIG_REGULATE(n_eggs_linear, ssb_in, seeded_this_step, config, step)

    repro_mod.regulate_recruitment = wrapper
    try:
        res = PythonEngine().run_in_memory(cfg, seed=SEED)
    finally:
        repro_mod.regulate_recruitment = _ORIG_REGULATE

    bio, abd = res.biomass(), res.abundance()
    max_len = float("nan")
    try:
        bs = res.abundance_by_size(SUBJECT)
        sb = pd.to_numeric(bs["bin"], errors="coerce")
        sv = pd.to_numeric(bs["value"], errors="coerce")
        occ = sb[(sv > 0) & sb.notna()]
        if len(occ):
            max_len = float(occ.max())
    except Exception as exc:
        print(f"    (abundance_by_size unavailable: {exc})")
    return {
        "final": {s: (float(bio[s].iloc[-1]), float(abd[s].iloc[-1])) for s in focal},
        "ssb": ssb.copy(),
        "seeded": seeded.copy(),
        "max_len": max_len,
        "cfg": cfg,
    }


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_seal_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    cfg_dir = cf.parent
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)]
    focal = names[:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}
    seed_b = float(raw[f"population.seeding.biomass.sp{SUBJECT_SP}"])

    arms = (
        ("baseline", None, None),
        ("bioen", overlay, None),
        ("sham", overlay, 1.0),
        ("treat", overlay, 0.0),
    )
    out = {}
    for tag, ov, sv in arms:
        print(f"  running {tag} ...", flush=True)
        out[tag] = run_arm(raw, cfg_dir, ov, sv, focal)

    print(f"\n{'=' * 96}\nE1/E2 — [cod_west prey, GreySeal predator], read back from EngineConfig")
    print("=" * 96)
    cells = {}
    for tag, *_ in arms:
        val, resolved = seal_cell(out[tag]["cfg"])
        cells[tag] = (val, resolved)
        print(f"  {tag:<10} cell = {val!r:<12} GreySeal resolves as a predator column: {resolved}")
    e2 = cells["sham"][1] and cells["treat"][1]
    e1 = cells["sham"][0] != cells["treat"][0]

    print(f"\n{'=' * 96}\nFINAL STATE (year {N_YEAR}, seed {SEED})\n{'=' * 96}")
    print(f"{'species':<13}" + "".join(f"{t:>16}" for t, *_ in arms))
    for s in focal:
        line = f"{s:<13}"
        for tag, *_ in arms:
            bm, ab = out[tag]["final"][s]
            line += f"{bm:>12.1f}{'  X' if ab <= 0 else '   '}"
        print(line)

    print(f"\n{'=' * 96}\n{SUBJECT} — primary instruments\n{'=' * 96}")
    print(f"{'arm':<10}{'SSB total':>15}{'seeded':>14}{'REAL SSB':>14}{'max size bin cm':>18}")
    real = {}
    for tag, *_ in arms:
        tot = out[tag]["ssb"][SUBJECT_SP]
        sd = out[tag]["seeded"][SUBJECT_SP] * seed_b
        real[tag] = tot - sd
        print(f"{tag:<10}{tot:>15.1f}{sd:>14.1f}{real[tag]:>14.1f}{out[tag]['max_len']:>18.2f}")

    floor = 0.01 * real["baseline"]
    print(
        f"\n  magnitude floor = 1% of baseline real SSB = {floor:.1f} t "
        f"(or max size bin >= {MATURITY_CM} cm)"
    )

    e3 = all(out["baseline"]["final"][s][1] > 0 for s in focal)
    sham_matches = (
        np.isclose(real["sham"], real["bioen"], rtol=1e-9, atol=1e-6)
        and np.isclose(out["sham"]["max_len"], out["bioen"]["max_len"], equal_nan=True)
        and np.isclose(
            out["sham"]["final"][SUBJECT][0], out["bioen"]["final"][SUBJECT][0], rtol=1e-9
        )
    )

    print(f"\n{'=' * 96}\nPRE-REGISTERED VERDICT\n{'=' * 96}")
    print(f"  E1 sham and treat differ at the cell : {e1}")
    print(f"  E2 GreySeal resolves as a column     : {e2}")
    print(f"  E3 baseline sustains all nine        : {e3}")
    print(f"  SHAM reproduces BIOEN                : {sham_matches}")
    if not (e1 and e2 and e3):
        print("\n  INCONCLUSIVE — an engagement check failed; the finding is the knob.")
        return 0
    if not sham_matches:
        print("\n  INCONCLUSIVE — the SHAM no-op did NOT reproduce bioen, so the")
        print("  '-1 means full accessibility' reading is wrong. Re-derive before reading biology.")
        return 0
    cleared = real["treat"] > floor or (
        np.isfinite(out["treat"]["max_len"]) and out["treat"]["max_len"] >= MATURITY_CM
    )
    if cleared:
        print(f"\n  SEAL CONFIRMED — with GreySeal unable to eat {SUBJECT}, real SSB goes")
        print(f"  {real['bioen']:.1f} -> {real['treat']:.1f} t and the max size bin goes")
        print(
            f"  {out['bioen']['max_len']:.0f} -> {out['treat']['max_len']:.0f} cm, clearing the floor."
        )
        print("  The C3 collapse is then a CONFIG/ENGINE DEFECT, not a bioenergetics result.")
    else:
        print(f"\n  SEAL REFUTED as the binding killer — treat real SSB {real['treat']:.1f} t and")
        print(f"  max size bin {out['treat']['max_len']:.0f} cm do not clear the floor. Seal")
        print("  predation is real but not sufficient; the search continues with it now")
        print("  correctly accounted for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
