"""Does the C3 Stage-1 verdict survive repairing the GreySeal accessibility column?

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md`. Stage 1 closed BY CHARACTERIZATION: at the
certifying 50-yr scale with PRODUCTION seeding, four of five assessed stocks (cod_west, cod_east,
herring, flounder) read a final-decade mean of **exactly 0.0 t**, bit-identical across all five
seeds, and only sprat survived. Every one of the six harness gates passed, so it was not a wiring
failure.

It now appears to have been, in part, a CONFIG defect. GreySeal (sp15) is a declared background
predator with **no predator COLUMN** in `predation-accessibility.csv`, so `pred_access_idx == -1`
and the production kernel (`mortality.py:1172-1180`) leaves its default `access_coeff = 1.0` --
twenty times the 0.05 that Cormorant and every listed fish predator are capped at. Proven by a
bit-identical sham (`scripts/c3_sealgate_intervention.py`, `0f4aedb`).

Repairing the column at the 8-yr single-cohort stress (`scripts/c3_seal_column_repair.py`,
`36d7b12`) restored SIZE broadly -- both cods 15 -> 75 cm, flounder fully back to its baseline
40 cm -- but **0 of 4 stocks recovered on abundance**, SSB staying 3-4 orders of magnitude below
floor. That test was explicitly caveated: `seeding.year.max = 1` means exactly ONE cohort exists, so
population recovery needs its offspring to mature inside the remaining window, and cod_west matures
at ~2.6 yr. The SIZE result was within-cohort and robust; the ABUNDANCE result may have been
window-limited.

This script removes that caveat. **Production seeding** (no `seeding.year.max` override -- the
engine default is per-species `lifespan`, cod_west 20 yr) and the **certifying 50-yr horizon**, so
the final decade sits 21-46 years past every assessed stock's own seeding-window closure. This is
the scale at which Stage 1 drew its verdict, so the arms drop straight into that table.

Arms (50 yr, production seeding, seed 42):

    baseline   bioen OFF, production matrix                 -- the control
    bioen      bioen ON, production matrix (seal at 1.0)    -- must REPRODUCE the Stage-1 collapse
    repaired   bioen ON, GreySeal column = Cormorant's      -- the question

=========================== PRE-REGISTERED READING (written before the run) ===========================

Primary instrument: **final-decade (years 41-50) mean biomass**, per species -- the exact metric
Stage 1 used, so the numbers are comparable row for row. Final-decade mean SSB is reported alongside
as supporting evidence and is NOT part of the criterion.

Criterion, single and non-disjunctive (the SEALGATE floor was an OR and fired on its size limb while
SSB missed by four orders of magnitude):

    a stock RECOVERS iff its final-decade mean biomass on `repaired` exceeds 1% of its OWN
    final-decade mean biomass on `baseline`.

  (1) >= 2 of the four Stage-1 collapsed stocks (cod_west, cod_east, herring, flounder) recover
        -> THE STAGE-1 VERDICT DOES NOT SURVIVE THE REPAIR. C3's headline negative is substantially
           a config defect and must be re-run on a repaired matrix before it can be cited.
  (2) exactly 1 recovers
        -> PARTIAL. The verdict is weakened and the affected row must be footnoted.
  (3) 0 recover
        -> THE STAGE-1 VERDICT STANDS. The missing column is a real defect that governs size
           structure (already established) and does NOT change the collapse. C3's negative is then
           a genuine bioenergetics result and the remaining question is what caps abundance.

Engagement checks -- all must pass before any of (1)-(3) may be read:
  E1 the `bioen` arm REPRODUCES the published Stage-1 collapse: cod_west, cod_east, herring and
     flounder all at ~0 final-decade mean biomass. If it does not, this harness is not measuring
     what Stage 1 measured and nothing else here is comparable to it.
  E2 `baseline` sustains all nine species.
  E3 GreySeal resolves as a predator column on `repaired` and NOT on the other two.

======================================================================================================

Run: .venv/bin/python scripts/c3_repaired_matrix_restage.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import osmose.engine.processes.reproduction as repro_mod  # noqa: E402
from osmose.config import OsmoseConfigReader  # noqa: E402
from osmose.demo import osmose_demo  # noqa: E402
from osmose.engine import PythonEngine  # noqa: E402
from osmose.engine.config import EngineConfig  # noqa: E402

N_YEAR = 50
FINAL_DECADE = 10
SEED = 42
SEAL = "GreySeal"
TEMPLATE = "Cormorant"
STAGE1_COLLAPSED = ("cod_west", "cod_east", "herring", "flounder")
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIG_REGULATE = repro_mod.regulate_recruitment


def write_seal_column(src: Path, dst: Path) -> None:
    """Append a GreySeal predator column copied verbatim from Cormorant's."""
    rows = list(csv.reader(src.open(), delimiter=";"))
    hdr = [h.strip() for h in rows[0]]
    tcol = hdr.index(TEMPLATE)
    out = [rows[0] + [SEAL]]
    for row in rows[1:]:
        if row:
            out.append(row + [repr(float(row[tcol]))])
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def seal_resolves(cfg_dict: dict) -> bool:
    try:
        cfg = EngineConfig.from_dict(dict(cfg_dict))
    except Exception:
        return False
    acc = getattr(cfg, "stage_accessibility", None)
    if acc is None or not hasattr(acc, "pred_lookup"):
        return False
    return bool(acc.pred_lookup.get(acc.resolve_name(SEAL) or ""))


def run_arm(raw, cfg_dir, overlay, repair, focal):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg.pop("population.seeding.year.max", None)  # PRODUCTION seeding: engine default = lifespan
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    if repair:
        write_seal_column(cfg_dir / "predation-accessibility.csv", cfg_dir / "acc-repaired.csv")
        cfg["predation.accessibility.file"] = "acc-repaired.csv"

    n_sp = int(raw["simulation.nspecies"])
    ssb_steps: list[np.ndarray] = []

    def wrapper(n_eggs_linear, ssb_in, seeded_this_step, config, step):
        ssb_steps.append(np.asarray(ssb_in, dtype=np.float64).copy())
        return _ORIG_REGULATE(n_eggs_linear, ssb_in, seeded_this_step, config, step)

    repro_mod.regulate_recruitment = wrapper
    try:
        res = PythonEngine().run_in_memory(cfg, seed=SEED)
    finally:
        repro_mod.regulate_recruitment = _ORIG_REGULATE

    bio = res.biomass()
    tail = max(1, int(len(bio) * FINAL_DECADE / N_YEAR))
    ssb_arr = np.array(ssb_steps) if ssb_steps else np.zeros((1, n_sp))
    ssb_tail = max(1, int(len(ssb_arr) * FINAL_DECADE / N_YEAR))
    return {
        "bio_fd": {s: float(np.nanmean(bio[s].to_numpy()[-tail:])) for s in focal},
        "ssb_fd": {s: float(ssb_arr[-ssb_tail:, i].mean()) for i, s in enumerate(focal)},
        "cfg": cfg,
    }


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_restage_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    cfg_dir = cf.parent
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    focal = [raw[f"species.name.sp{i}"] for i in range(n_sp)][:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    print(f"{N_YEAR} yr, PRODUCTION seeding (no seeding.year.max override), seed {SEED}")
    print(f"metric: final-decade (last {FINAL_DECADE} yr) mean — the Stage-1 metric\n")

    arms = (("baseline", None, False), ("bioen", overlay, False), ("repaired", overlay, True))
    out = {}
    for tag, ov, rep in arms:
        print(f"  running {tag} ...", flush=True)
        out[tag] = run_arm(raw, cfg_dir, ov, rep, focal)

    print(f"\n{'=' * 92}\nFINAL-DECADE MEAN BIOMASS (t) — primary\n{'=' * 92}")
    print(f"{'species':<13}" + "".join(f"{t:>18}" for t, *_ in arms) + f"{'floor (1%)':>14}")
    for s in focal:
        floor = 0.01 * out["baseline"]["bio_fd"][s]
        print(
            f"{s:<13}"
            + "".join(f"{out[t]['bio_fd'][s]:>18.1f}" for t, *_ in arms)
            + f"{floor:>14.1f}"
        )

    print(f"\n{'=' * 92}\nFINAL-DECADE MEAN SSB (t) — supporting, not part of the criterion")
    print("=" * 92)
    print(f"{'species':<13}" + "".join(f"{t:>18}" for t, *_ in arms))
    for s in focal:
        print(f"{s:<13}" + "".join(f"{out[t]['ssb_fd'][s]:>18.1f}" for t, *_ in arms))

    e1 = all(
        out["bioen"]["bio_fd"][s] < 0.01 * out["baseline"]["bio_fd"][s] for s in STAGE1_COLLAPSED
    )
    e2 = all(out["baseline"]["bio_fd"][s] > 0 for s in focal)
    e3 = (
        seal_resolves(out["repaired"]["cfg"])
        and not seal_resolves(out["bioen"]["cfg"])
        and not seal_resolves(out["baseline"]["cfg"])
    )

    print(f"\n{'=' * 92}\nPRE-REGISTERED VERDICT\n{'=' * 92}")
    print(f"  E1 bioen reproduces the Stage-1 collapse : {e1}")
    print(f"  E2 baseline sustains all nine            : {e2}")
    print(f"  E3 GreySeal resolves only on 'repaired'  : {e3}")
    if not (e1 and e2 and e3):
        print("\n  INCONCLUSIVE — an engagement check failed. If E1 failed, this harness is not")
        print("  measuring what Stage 1 measured and nothing here is comparable to it.")
        return 0

    rec = [
        s
        for s in STAGE1_COLLAPSED
        if out["repaired"]["bio_fd"][s] > 0.01 * out["baseline"]["bio_fd"][s]
    ]
    print(f"\n  recovered on 'repaired' ({len(rec)}/4): {rec or 'none'}")
    if len(rec) >= 2:
        print("\n  THE STAGE-1 VERDICT DOES NOT SURVIVE THE REPAIR. C3's headline negative is")
        print("  substantially a CONFIG DEFECT and must be re-run on a repaired matrix before")
        print("  it can be cited.")
    elif len(rec) == 1:
        print(f"\n  PARTIAL — {rec[0]} recovers; the verdict is weakened and that row needs a")
        print("  footnote, but the headline survives.")
    else:
        print("\n  THE STAGE-1 VERDICT STANDS. The missing column is a real defect governing size")
        print("  structure and does NOT change the collapse. C3's negative is a genuine")
        print("  bioenergetics result; the open question is what caps ABUNDANCE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
