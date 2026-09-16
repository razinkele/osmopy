"""Does the C3 Stage-1 verdict survive repairing the GreySeal accessibility column? (5 seeds)

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md`. Stage 1 closed BY CHARACTERIZATION: at the
certifying 50-yr scale with PRODUCTION seeding, four of five assessed stocks (cod_west, cod_east,
herring, flounder) read a final-decade mean of **exactly 0.0 t**, bit-identical across all five
seeds, and only sprat survived. All six harness gates passed, so it was not a wiring failure.

It looked like it might be a CONFIG defect. GreySeal (sp15) is a declared background predator with
**no predator COLUMN** in `predation-accessibility.csv`, so `pred_access_idx == -1` and the
production kernel (`mortality.py:1172-1180`) leaves its default `access_coeff = 1.0` -- twenty times
the 0.05 that Cormorant and every listed fish predator are capped at. Proven by a bit-identical sham
(`scripts/c3_sealgate_intervention.py`, `0f4aedb`). Repairing it restores SIZE decisively: cod_west
and cod_east 15 -> 75 cm, flounder fully back to its baseline 40 cm.

A single-seed run of this harness (`7036b6e`, seed 42) found **0 of 4 recover** -- the defect governs
size structure and not the collapse. That was explicitly caveated: Stage 1's headline rested on the
collapse being bit-identical across FIVE seeds, and a NULL from one seed is weaker evidence than a
positive would have been. **This is the 5-seed version, at the same certifying configuration Stage 1
used, so the result can be held to the same standard as the claim it is testing.**

Seeds are Stage 1's own `(42, 123, 7, 999, 2024)`, read from `scripts/baltic_c3_bioen_ab.py:121` at
startup and asserted to match -- a copied constant that silently drifts would make the comparison
meaningless without anyone noticing.

Arms (50 yr, production seeding, 5 seeds = 15 engine runs):

    baseline   bioen OFF, production matrix                 -- the control
    bioen      bioen ON, production matrix (seal at 1.0)    -- must REPRODUCE the Stage-1 collapse
    repaired   bioen ON, GreySeal column = Cormorant's      -- the question

=========================== PRE-REGISTERED READING (written before the run) ===========================

Primary instrument: **final-decade (years 41-50) mean biomass**, per species per seed -- the exact
metric Stage 1 used. Aggregated ACROSS seeds by the mean, with per-seed values printed so a result
that hangs on one outlier seed is visible rather than hidden by the average.

Criterion, single and non-disjunctive:

    a stock RECOVERS iff its ACROSS-SEED MEAN final-decade biomass on `repaired` exceeds 1% of its
    ACROSS-SEED MEAN on `baseline`.

Reported alongside, not part of the criterion: the number of INDIVIDUAL seeds in which the stock
clears its own floor. A 5/5 and a 1/5 mean very different things at the same average.

  (1) >= 2 of the four Stage-1 collapsed stocks recover
        -> THE STAGE-1 VERDICT DOES NOT SURVIVE. C3's headline negative is substantially a config
           defect and must be re-run on a repaired matrix before it can be cited.
  (2) exactly 1 recovers
        -> PARTIAL. The verdict is weakened and that row needs a footnote.
  (3) 0 recover
        -> THE STAGE-1 VERDICT STANDS, now at the same 5-seed standard as the original claim.

Engagement checks -- all must pass before any of (1)-(3) may be read:
  E1 the `bioen` arm reproduces the published Stage-1 collapse ON EVERY SEED: cod_west, cod_east,
     herring and flounder each below 1% of baseline in all five. If not, this harness is not
     measuring what Stage 1 measured and nothing here is comparable to it.
  E2 `baseline` sustains all nine species on every seed.
  E3 GreySeal resolves as a predator column on `repaired` and NOT on the other two.
  E4 the seed constant matches `baltic_c3_bioen_ab.py` (drift guard, checked at startup).

======================================================================================================

Run: .venv/bin/python scripts/c3_repaired_matrix_restage.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import osmose.engine.processes.reproduction as repro_mod
from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig

N_YEAR = 50
FINAL_DECADE = 10
SEEDS = (42, 123, 7, 999, 2024)  # Stage 1's certifying set — baltic_c3_bioen_ab.py:121
SEAL = "GreySeal"
TEMPLATE = "Cormorant"
STAGE1_COLLAPSED = ("cod_west", "cod_east", "herring", "flounder")
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIG_REGULATE = repro_mod.regulate_recruitment


def assert_seeds_match_stage1() -> bool:
    """E4: the seed set is COPIED from the A/B script, so guard against silent drift.

    Parsed by regex rather than imported: importing that module executes its top level, and this
    check must not have side effects.
    """
    src = (ROOT / "scripts" / "baltic_c3_bioen_ab.py").read_text()
    m = re.search(r"^SEEDS\s*=\s*\(([^)]*)\)", src, re.MULTILINE)
    if not m:
        print("  E4 WARNING: could not find SEEDS in baltic_c3_bioen_ab.py")
        return False
    declared = tuple(int(x) for x in re.findall(r"\d+", m.group(1)))
    if declared != SEEDS:
        print(f"  E4 FAIL: Stage-1 SEEDS={declared} but this script uses {SEEDS}")
        return False
    print(f"  E4 seed set matches Stage 1: {SEEDS}")
    return True


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


def run_arm(raw, cfg_dir, overlay, repair, focal, seed):
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
        res = PythonEngine().run_in_memory(cfg, seed=seed)
    finally:
        repro_mod.regulate_recruitment = _ORIG_REGULATE

    bio = res.biomass()
    tail = max(1, int(len(bio) * FINAL_DECADE / N_YEAR))
    ssb_arr = np.array(ssb_steps) if ssb_steps else np.zeros((1, n_sp))
    stail = max(1, int(len(ssb_arr) * FINAL_DECADE / N_YEAR))
    return {
        "bio_fd": {s: float(np.nanmean(bio[s].to_numpy()[-tail:])) for s in focal},
        "ssb_fd": {s: float(ssb_arr[-stail:, i].mean()) for i, s in enumerate(focal)},
        "cfg": cfg,
    }


def main() -> int:
    warnings.simplefilter("ignore")
    print(f"{N_YEAR} yr, PRODUCTION seeding, {len(SEEDS)} seeds — the Stage-1 certifying scale")
    e4 = assert_seeds_match_stage1()

    tmp = Path(tempfile.mkdtemp(prefix="c3_restage5_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    cfg_dir = cf.parent
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    focal = [raw[f"species.name.sp{i}"] for i in range(n_sp)][:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    arms = (("baseline", None, False), ("bioen", overlay, False), ("repaired", overlay, True))
    out: dict[tuple[str, int], dict] = {}
    for seed in SEEDS:
        for tag, ov, rep in arms:
            print(f"  running {tag} seed={seed} ...", flush=True)
            out[(tag, seed)] = run_arm(raw, cfg_dir, ov, rep, focal, seed)

    def across(tag, s):
        return float(np.mean([out[(tag, sd)]["bio_fd"][s] for sd in SEEDS]))

    print(f"\n{'=' * 96}\nACROSS-SEED MEAN final-decade biomass (t) — primary\n{'=' * 96}")
    print(f"{'species':<13}" + "".join(f"{t:>18}" for t, *_ in arms) + f"{'floor (1%)':>14}")
    for s in focal:
        floor = 0.01 * across("baseline", s)
        print(f"{s:<13}" + "".join(f"{across(t, s):>18.1f}" for t, *_ in arms) + f"{floor:>14.1f}")

    print(f"\n{'=' * 96}\nPER-SEED final-decade biomass (t), the four Stage-1 collapsed stocks")
    print("=" * 96)
    for s in STAGE1_COLLAPSED:
        floor = 0.01 * across("baseline", s)
        print(f"\n  {s}   (floor = {floor:.1f} t)")
        print(f"    {'seed':<8}" + "".join(f"{t:>18}" for t, *_ in arms))
        for sd in SEEDS:
            print(f"    {sd:<8}" + "".join(f"{out[(t, sd)]['bio_fd'][s]:>18.1f}" for t, *_ in arms))

    e1 = all(
        out[("bioen", sd)]["bio_fd"][s] < 0.01 * across("baseline", s)
        for s in STAGE1_COLLAPSED
        for sd in SEEDS
    )
    e2 = all(out[("baseline", sd)]["bio_fd"][s] > 0 for s in focal for sd in SEEDS)
    e3 = (
        seal_resolves(out[("repaired", SEEDS[0])]["cfg"])
        and not seal_resolves(out[("bioen", SEEDS[0])]["cfg"])
        and not seal_resolves(out[("baseline", SEEDS[0])]["cfg"])
    )

    print(f"\n{'=' * 96}\nPRE-REGISTERED VERDICT\n{'=' * 96}")
    print(f"  E1 bioen reproduces the Stage-1 collapse on EVERY seed : {e1}")
    print(f"  E2 baseline sustains all nine on every seed            : {e2}")
    print(f"  E3 GreySeal resolves only on 'repaired'                : {e3}")
    print(f"  E4 seed set matches Stage 1                            : {e4}")
    if not (e1 and e2 and e3 and e4):
        print("\n  INCONCLUSIVE — an engagement check failed. If E1 failed, this harness is not")
        print("  measuring what Stage 1 measured and nothing here is comparable to it.")
        return 0

    rec = [s for s in STAGE1_COLLAPSED if across("repaired", s) > 0.01 * across("baseline", s)]
    print("\n  per-seed floor clearances on 'repaired' (supporting, not the criterion):")
    for s in STAGE1_COLLAPSED:
        floor = 0.01 * across("baseline", s)
        n = sum(1 for sd in SEEDS if out[("repaired", sd)]["bio_fd"][s] > floor)
        print(f"    {s:<12} {n}/{len(SEEDS)} seeds")

    print(f"\n  recovered on 'repaired' ({len(rec)}/4): {rec or 'none'}")
    if len(rec) >= 2:
        print("\n  THE STAGE-1 VERDICT DOES NOT SURVIVE THE REPAIR. C3's headline negative is")
        print("  substantially a CONFIG DEFECT and must be re-run on a repaired matrix.")
    elif len(rec) == 1:
        print(f"\n  PARTIAL — {rec[0]} recovers; the verdict is weakened and that row needs a")
        print("  footnote, but the headline survives.")
    else:
        print("\n  THE STAGE-1 VERDICT STANDS, now at the SAME 5-SEED STANDARD as the original")
        print("  claim. The missing column is a real defect governing size structure and does NOT")
        print("  change the collapse. C3's negative is a genuine bioenergetics result; the open")
        print("  question is what caps ABUNDANCE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
