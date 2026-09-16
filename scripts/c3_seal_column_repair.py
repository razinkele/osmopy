"""Give GreySeal a realistic accessibility column: how many collapsed Baltic stocks come back?

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. GreySeal (sp15) is a declared background
predator with **no predator COLUMN** in `predation-accessibility.csv`, so
`AccessibilityMatrix.resolve_name("GreySeal")` returns `None`, `pred_access_idx == -1`, and the
production kernel (`mortality.py:1172-1180`) leaves its default `access_coeff = 1.0` -- FULL
accessibility, twenty times the 0.05 that Cormorant and every listed fish predator are capped at.
SEALGATE (`scripts/c3_sealgate_intervention.py`, `0f4aedb`) proved this with a bit-identical sham
and showed that zeroing ONE cell takes cod_west's max occupied size bin from 15 cm to 75 cm, past
its 38 cm maturity length.

That test changed only cod_west's cell. flounder, perch and cod_east stayed extinct while exposed to
the identical defect, since the seal reaches every prey row at 1.0. This script repairs the COLUMN
for all prey and asks how much of the "bioenergetics collapse" was really this.

Arms (8 yr, `seeding.year.max = 1`, seed 42 -- identical to every published arm):

    baseline    bioen OFF, production matrix                  -- the control that sustains
    bioen       bioen ON, production matrix (seal at 1.0)     -- the published collapsing arm
    realistic   + GreySeal column = CORMORANT'S COLUMN        -- the repair
    sealfree    + GreySeal column = 0.0 for every prey        -- upper bound, seal removed

`realistic` uses Cormorant's column verbatim (cod_west 0.05, herring 0.15, sprat 0.15, flounder 0.1,
perch 0.6, pikeperch 0.4, smelt 0.25, stickleback 0.15, cod_east 0.05, and **0 for every resource
row**). It is a PROXY for a plausible top-predator column, not a calibration: it is simply the only
in-config example of what a background predator's accessibility is supposed to look like. Note the
resource zeros are themselves part of the repair -- today the seal eats Benthos and the plankton
groups at an implicit 1.0, which no one wrote down either.

`sealfree` brackets the answer from the other side: if `realistic` and `sealfree` agree, the
residual seal predation at realistic coefficients is not doing the work.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Primary instrument, per species: REAL SSB = run-total SSB minus the year-1 seeding contribution
(`24 x population.seeding.biomass.sp{i}`), the decomposition established in the recruitment test.

RECOVERY CRITERION, single and non-disjunctive -- the SEALGATE floor was an OR (`SSB > 1%` OR
`size >= 38 cm`) and fired on the size limb while SSB missed by four orders of magnitude, which made
it far weaker than it looked. Here:

    a stock RECOVERS iff its real SSB on the arm exceeds 1% of its OWN real SSB on `baseline`.

Max occupied size bin is reported alongside as SUPPORTING evidence and is NOT part of the criterion.

  (1) >= 2 of the four collapsed stocks (cod_west, cod_east, flounder, perch) recover on `realistic`
        -> THE COLLAPSE IS SUBSTANTIALLY THE MISSING COLUMN. C3's headline negative is then largely
           a config defect and the Stage-1 verdict needs re-running on a repaired matrix.
  (2) exactly 1 recovers
        -> PARTIAL. The defect is real and material but not the whole story.
  (3) 0 recover on `realistic` AND 0 on `sealfree`
        -> THE SEAL IS NOT WHAT COLLAPSES THE OTHER STOCKS. cod_west's 15->75 cm response stands as
           a size effect that did not translate into population recovery, and the search continues.
  (4) 0 on `realistic` but >= 1 on `sealfree`
        -> the repair is too weak rather than wrong; report the gap between the two.

Engagement checks:
  E1 GreySeal resolves as a predator column on `realistic` and `sealfree` (it must NOT still be -1);
  E2 the two arms differ in the loaded matrix;
  E3 baseline sustains all nine species.

======================================================================================================

Run: .venv/bin/python scripts/c3_seal_column_repair.py
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

import osmose.engine.processes.reproduction as repro_mod
from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig

N_YEAR = 8
SEED = 42
SEAL = "GreySeal"
TEMPLATE = "Cormorant"
COLLAPSED = ("cod_west", "cod_east", "flounder", "perch")
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIG_REGULATE = repro_mod.regulate_recruitment


def write_seal_column(src: Path, dst: Path, mode: str) -> dict[str, float]:
    """Append a GreySeal column. mode='template' copies Cormorant's; mode='zero' writes 0.0."""
    rows = list(csv.reader(src.open(), delimiter=";"))
    hdr = [h.strip() for h in rows[0]]
    tcol = hdr.index(TEMPLATE)
    out = [rows[0] + [SEAL]]
    used: dict[str, float] = {}
    for row in rows[1:]:
        if not row:
            continue
        val = 0.0 if mode == "zero" else float(row[tcol])
        used[row[0].strip()] = val
        out.append(row + [repr(val)])
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)
    return used


def seal_resolves(cfg_dict: dict) -> tuple[bool, float]:
    try:
        cfg = EngineConfig.from_dict(dict(cfg_dict))
    except Exception as exc:
        print(f"    (E1 config build failed: {exc})")
        return False, float("nan")
    acc = getattr(cfg, "stage_accessibility", None)
    if acc is None or not hasattr(acc, "prey_lookup"):
        return False, float("nan")
    name = acc.resolve_name(SEAL)
    pred = acc.pred_lookup.get(name or "") or []
    prey = acc.prey_lookup.get("cod_west") or []
    if not pred or not prey:
        return False, float("nan")
    return True, float(acc.raw_matrix[prey[0].matrix_index, pred[0].matrix_index])


def run_arm(raw, cfg_dir, overlay, mode, focal):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg["population.seeding.year.max"] = "1"
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    if mode is not None:
        name = f"acc-seal-{mode}.csv"
        write_seal_column(cfg_dir / "predation-accessibility.csv", cfg_dir / name, mode)
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
    max_len = {}
    for s in focal:
        v = float("nan")
        try:
            bs = res.abundance_by_size(s)
            sb = pd.to_numeric(bs["bin"], errors="coerce")
            sv = pd.to_numeric(bs["value"], errors="coerce")
            occ = sb[(sv > 0) & sb.notna()]
            if len(occ):
                v = float(occ.max())
        except Exception:
            pass
        max_len[s] = v
    return {
        "final": {s: (float(bio[s].iloc[-1]), float(abd[s].iloc[-1])) for s in focal},
        "ssb": ssb.copy(),
        "seeded": seeded.copy(),
        "max_len": max_len,
        "cfg": cfg,
    }


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_sealcol_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    cfg_dir = cf.parent
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)]
    focal = names[:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}
    seed_b = {s: float(raw[f"population.seeding.biomass.sp{i}"]) for i, s in enumerate(focal)}

    used = write_seal_column(
        ROOT / "data" / "baltic" / "predation-accessibility.csv", tmp / "_preview.csv", "template"
    )
    print("GreySeal column to be written (copied verbatim from Cormorant's):")
    print("  " + ", ".join(f"{k}={v:g}" for k, v in used.items() if v > 0) + "; resources = 0")

    arms = (
        ("baseline", None, None),
        ("bioen", overlay, None),
        ("realistic", overlay, "template"),
        ("sealfree", overlay, "zero"),
    )
    out = {}
    for tag, ov, mode in arms:
        print(f"\n  running {tag} ...", flush=True)
        out[tag] = run_arm(raw, cfg_dir, ov, mode, focal)

    print(f"\n{'=' * 100}\nE1/E2 — GreySeal as a predator column\n{'=' * 100}")
    res_ok = {}
    for tag, *_ in arms:
        ok, cell = seal_resolves(out[tag]["cfg"])
        res_ok[tag] = (ok, cell)
        print(f"  {tag:<11} resolves={ok}   [cod_west, GreySeal] = {cell!r}")
    e1 = res_ok["realistic"][0] and res_ok["sealfree"][0]
    e2 = res_ok["realistic"][1] != res_ok["sealfree"][1]
    e3 = all(out["baseline"]["final"][s][1] > 0 for s in focal)

    real = {}
    for tag, *_ in arms:
        real[tag] = {
            s: out[tag]["ssb"][i] - out[tag]["seeded"][i] * seed_b[s] for i, s in enumerate(focal)
        }

    print(f"\n{'=' * 100}\nREAL SSB by species (t) — primary instrument\n{'=' * 100}")
    print(f"{'species':<13}" + "".join(f"{t:>16}" for t, *_ in arms) + f"{'floor (1%)':>14}")
    for s in focal:
        floor = 0.01 * real["baseline"][s]
        line = f"{s:<13}" + "".join(f"{real[t][s]:>16.1f}" for t, *_ in arms) + f"{floor:>14.1f}"
        print(line)

    print(f"\n{'=' * 100}\nMax occupied size bin (cm) — SUPPORTING, not part of the criterion")
    print("=" * 100)
    print(f"{'species':<13}" + "".join(f"{t:>16}" for t, *_ in arms))
    for s in focal:
        print(f"{s:<13}" + "".join(f"{out[t]['max_len'][s]:>16.1f}" for t, *_ in arms))

    print(f"\n{'=' * 100}\nFINAL BIOMASS (t)\n{'=' * 100}")
    print(f"{'species':<13}" + "".join(f"{t:>16}" for t, *_ in arms))
    for s in focal:
        line = f"{s:<13}"
        for tag, *_ in arms:
            bm, ab = out[tag]["final"][s]
            line += f"{bm:>12.1f}{'  X' if ab <= 0 else '   '}"
        print(line)

    print(f"\n{'=' * 100}\nPRE-REGISTERED VERDICT\n{'=' * 100}")
    print(f"  E1 GreySeal resolves on both repair arms : {e1}")
    print(f"  E2 the two repair arms differ            : {e2}")
    print(f"  E3 baseline sustains all nine            : {e3}")
    if not (e1 and e2 and e3):
        print("\n  INCONCLUSIVE — an engagement check failed; the finding is the knob.")
        return 0

    def recovered(tag):
        return [s for s in COLLAPSED if real[tag][s] > 0.01 * real["baseline"][s]]

    rec_r, rec_s = recovered("realistic"), recovered("sealfree")
    print(f"\n  recovered on 'realistic' ({len(rec_r)}/4): {rec_r or 'none'}")
    print(f"  recovered on 'sealfree'  ({len(rec_s)}/4): {rec_s or 'none'}")
    if len(rec_r) >= 2:
        print("\n  THE COLLAPSE IS SUBSTANTIALLY THE MISSING COLUMN. C3's headline negative is")
        print(
            "  largely a CONFIG DEFECT; the Stage-1 verdict needs re-running on a repaired matrix."
        )
    elif len(rec_r) == 1:
        print("\n  PARTIAL — the defect is real and material but not the whole story.")
    elif not rec_s:
        print("\n  THE SEAL IS NOT WHAT COLLAPSES THE OTHER STOCKS. cod_west's 15->75 cm response")
        print("  stands as a size effect that did not translate into population recovery.")
    else:
        print("\n  THE REPAIR IS TOO WEAK RATHER THAN WRONG — 'sealfree' recovers stocks that")
        print("  'realistic' does not; the gap between them is the finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
