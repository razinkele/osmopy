"""Test the C3 PREDATION hypothesis by intervention: make cod_west juveniles inedible.

Context. Under the C3 bioen overlay with `population.seeding.year.max = 1`, cod_west/cod_east/
flounder/perch go to zero and herring to a 1.2 t remnant, while a bioen-OFF control on the
identical config sustains all nine (`docs/baltic_c3_bioen_stage1_2026-09-05.md` §9). Two candidate
causes are already dead, each killed by intervention:

  GROWTH      -- the offline fit was 42-51% short at age 1. Fixing it completely changed NOTHING.
  RECRUITMENT -- ILL-POSED. cod_west/cod_east real SSB is EXACTLY 0.0 t across all 168 post-seeding
                 steps, so a 10x egg boost multiplied zero by ten (R = 1.00). Recruitment is
                 downstream of whatever removes the pre-maturity cohort.

What is left is Task 13's measured signature: predation, not starvation, drives the collapse.

Why predation is now mechanistically plausible despite low accessibility. cod_west has only FOUR
predators in `data/baltic/predation-accessibility.csv` -- itself (cannibalism), pikeperch, smelt and
the Cormorant background predator -- each at accessibility 0.05. That looks too weak to wipe a
cohort. But the bioen overlay also re-fits every species' max ingestion rate, and the two cap
formulas differ in shape, not just level:

    classic:  max_eatable = biomass_p * R / (ndt * n_subdt)                 (mortality.py:470)
    bioen:    max_eatable = N * (Imax/ndt) * w_g^beta / n_subdt * 1e-6      (bioen_predation.py)

so the ratio is `(Imax/R) * w_g^(-0.2)` at beta = 0.8. For cod_west's predators that is:

    pikeperch  14.6x (1 g) .. 5.8x (100 g) .. 3.7x (1 kg) .. 2.3x (10 kg)   [Imax 3.5 -> 51.02]
    smelt       4.6x .. 1.8x .. 1.2x .. 0.7x                                [Imax 4.0 -> 18.53]
    cod_west    4.0x .. 1.6x .. 1.0x .. 0.6x                                [Imax 3.5 -> 13.90]

Every predator of cod_west may eat MORE under bioen at every realistic size, and §9 measured
pikeperch's realized/cap at only 0.36 -- large unused headroom on a cap that is itself ~14x bigger.

The intervention. `predation.accessibility.stage.structure = age`, and the accessibility CSV
supports age-split labels (`"name < threshold"`, `osmose/engine/accessibility.py:_parse_label`),
with prey ROWS and predator COLUMNS parsed independently (header: "v Prey / Predator >"). So
cod_west's PREY row can be split into a juvenile stage and an adult stage, and the juvenile stage
scaled by a dose D, touching nothing else -- not growth, not bioenergetics, not reproduction, not
cod_west's own behaviour as a predator (that is the COLUMN, left alone).

The split is at age 2.63 yr, which is when cod_west reaches its maturity size: vBGF with
Linf = 110, K = 0.15, t0 = -0.20 gives L = 38 cm = `species.maturity.m0.sp0` at age 2.63. So the
suppressed stage is exactly "not yet able to spawn".

Window adequacy, settled BEFORE the run. Time to maturity is 2.63 yr against ~7 yr of life
available to the year-1 cohort in an 8-yr run, and the bioen-OFF control reaches non-zero SSB
(129,265 t) in this same window. The test is not dead on arrival.

Arms (8 yr, `seeding.year.max = 1`, seed 42 -- the exact stress used by the growth and recruitment
interventions, so all three are directly comparable):

    baseline     bioen OFF, unmodified matrix        -- the control that sustains
    bioen        committed overlay, unmodified       -- the arm that collapses
    acc50        juvenile accessibility x 0.50       -- dose
    acc10        juvenile accessibility x 0.10       -- dose
    acc00        juvenile accessibility x 0.00       -- FULL IMMUNITY, the maximal dose

Why this knob cannot fail to engage the way the recruitment knob did. There, the maximal dose was
still zero because it multiplied an SSB of zero. Here accessibility is the structural gate on
whether cod_west is edible at all: at D = 0 the matrix entry is 0 and no predator can take a
cod_west juvenile, independent of any population state. The dose ladder additionally gives a
graded response rather than a single binary.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Primary instrument is SSB, NOT biomass and NOT abundance -- SSB is the quantity proven to be
identically zero, and `biomass()`/`abundance()` exclude young-of-year while `*_by_age()` does not
(CLAUDE.md). Real SSB = run-total SSB minus the year-1 seeding contribution
(24 x `population.seeding.biomass.sp0`), the same decomposition that diagnosed the recruitment test.

  (1) acc00 gives cod_west real SSB > 0
        -> PREDATION CONFIRMED as the binding constraint on cod_west.
           Strengthened if the dose ladder is monotone (acc00 >= acc10 >= acc50 >= bioen).
  (2) acc00 gives real SSB still EXACTLY 0, AND the engagement checks below pass
        -> PREDATION REFUTED. All three named candidates are then dead and the cause is
           something not yet enumerated.
  (3) engagement checks fail
        -> INCONCLUSIVE. The finding is the knob, not the biology. Report as such, NOT as a null.

Engagement checks, all of which must pass for (1) or (2) to be read:
  E1 the loaded accessibility matrix differs between arms at cod_west's juvenile prey row, read
     back from a constructed EngineConfig (not merely written to the CSV);
  E2 baseline sustains all nine species (the control still works);
  E3 cod_west juvenile survivorship differs between `bioen` and `acc00` -- measured as standing
     abundance in age bins >= 1 yr, which excludes the egg-contaminated bin 0.
  E3 is the one that distinguishes a real null from a dead knob: if immunity does not change how
  many juveniles are alive, the intervention did not happen.

======================================================================================================

Run: .venv/bin/python scripts/c3_predation_intervention.py
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
SUBJECT = "cod_west"
SUBJECT_SP = 0
MATURITY_AGE = 2.63  # yr; vBGF Linf=110 K=0.15 t0=-0.20 reaches m0=38 cm here
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIGINAL_REGULATE = repro_mod.regulate_recruitment


def write_staged_matrix(src: Path, dst: Path, dose: float, split: bool = True) -> list[float]:
    """Split SUBJECT's PREY row into juvenile/adult stages, scaling the juvenile row by `dose`.

    Rows are prey and columns are predators (CSV header: "v Prey / Predator >"), and
    `accessibility.py:_parse_label` reads "name < threshold" as an age stage. The predator COLUMN
    for SUBJECT is deliberately untouched -- this changes what eats cod_west, never what cod_west
    eats.
    """
    rows = list(csv.reader(src.open(), delimiter=";"))
    out: list[list[str]] = []
    juvenile_values: list[float] = []
    for row in rows:
        if row and row[0].strip() == SUBJECT:
            original = [float(v) for v in row[1:]]
            juvenile_values = [v * dose for v in original]
            if split:
                out.append([f"{SUBJECT} < {MATURITY_AGE}"] + [repr(v) for v in juvenile_values])
                out.append([SUBJECT] + [repr(v) for v in original])
            else:
                # ALL-AGES dose: one unsplit row. Needed because the stage split is by AGE while
                # maturity is by LENGTH (38 cm) -- a fish that grows slowly becomes edible again at
                # MATURITY_AGE while still juvenile-sized, so the age-split arm is only a PARTIAL
                # dose. This is the unambiguous maximal dose: cod_west is inedible at every age.
                out.append([SUBJECT] + [repr(v) for v in juvenile_values])
        else:
            out.append(row)
    if not juvenile_values:
        raise RuntimeError(f"prey row {SUBJECT!r} not found in {src}")
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)
    return juvenile_values


def juvenile_row_in_effect(cfg_dict: dict) -> list[float]:
    """E1: read the juvenile prey row back out of a CONSTRUCTED EngineConfig, not the CSV."""
    # Use `stage_accessibility` (the AccessibilityMatrix object carrying prey_lookup), NOT
    # `accessibility_matrix` -- the latter is a bare ndarray, and `a or b` on an ndarray raises
    # "truth value of an array is ambiguous".
    try:
        cfg = EngineConfig.from_dict(dict(cfg_dict))
    except Exception as exc:
        print(f"    (E1 config build failed: {exc})")
        return []
    acc = getattr(cfg, "stage_accessibility", None)
    if acc is None or not hasattr(acc, "prey_lookup"):
        return []
    stages = acc.prey_lookup.get(SUBJECT) or []
    if not stages:
        return []
    juv = min(stages, key=lambda s: s.threshold)
    return [float(v) for v in acc.raw_matrix[juv.matrix_index]]


def run_arm(raw, cfg_dir, overlay, dose, focal, split=True):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg["population.seeding.year.max"] = "1"
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    if dose is not None:
        tag_s = "split" if split else "allages"
        name = f"predation-accessibility-{tag_s}{int(dose * 100):03d}.csv"
        write_staged_matrix(
            cfg_dir / "predation-accessibility.csv", cfg_dir / name, dose, split=split
        )
        cfg["predation.accessibility.file"] = name

    n_sp = int(raw["simulation.nspecies"])
    ssb = np.zeros(n_sp)
    seeded = np.zeros(n_sp)

    def wrapper(n_eggs_linear, ssb_in, seeded_this_step, config, step):
        ssb[:] += np.asarray(ssb_in, dtype=np.float64)
        seeded[:] += np.asarray(seeded_this_step, dtype=bool).astype(np.float64)
        return _ORIGINAL_REGULATE(n_eggs_linear, ssb_in, seeded_this_step, config, step)

    repro_mod.regulate_recruitment = wrapper
    try:
        res = PythonEngine().run_in_memory(cfg, seed=SEED)
    finally:
        repro_mod.regulate_recruitment = _ORIGINAL_REGULATE

    bio, abd = res.biomass(), res.abundance()
    # E3: juvenile standing stock, age bins >= 1 yr only -- bin 0 contains egg schools (CLAUDE.md).
    # E3 uses biomassByAge: abundanceByAge is NOT produced in in-memory mode (verified at runtime
    # 2026-09-13 -- the available set is abundance/abundanceBySize/biomassByAge/...). Age bins >= 1 yr
    # only; bin 0 carries egg schools (CLAUDE.md).
    # `biomass_by_age` returns LONG format -- columns (time, species, bin, value), per
    # `_read_2d_output`'s docstring -- not wide age columns. Verified at runtime 2026-09-13: the
    # wide reading made numpy string-concatenate the object columns. Age bins >= 1 yr only; bin 0
    # carries egg schools (CLAUDE.md).
    juv = float("nan")
    cols_seen: list[str] = []
    try:
        sub = res.biomass_by_age(SUBJECT)
        cols_seen = [str(c) for c in sub.columns]
        bins = pd.to_numeric(sub["bin"], errors="coerce")
        vals = pd.to_numeric(sub["value"], errors="coerce")
        sel = (bins >= 1.0) & (bins < 3.0)
        if bool(sel.any()):
            juv = float(np.nansum(vals[sel].to_numpy()))
    except Exception as exc:  # an instrument must never take the run down
        print(f"    (biomass_by_age unavailable: {exc})")

    # WINDOW CHECK (E4). The single biggest way this test can lie: if cod_west never grows near the
    # 38 cm maturity length inside the run, SSB = 0 means "too small to spawn", NOT "predation is
    # not the cause". Maturity is LENGTH-only here (species.maturity.age.sp0 absent) while the
    # accessibility stage split is by AGE, so the two boundaries need not coincide.
    #
    # Instrument is the LARGEST OCCUPIED SIZE BIN from abundanceBySize, not mean size: the question
    # is whether ANY cod_west reached 38 cm, and a mean is dragged down by the juveniles that
    # dominate the numbers. Long format (time, species, bin, value) per _read_2d_output.
    max_len = float("nan")
    try:
        bs = res.abundance_by_size(SUBJECT)
        sb = pd.to_numeric(bs["bin"], errors="coerce")
        sv = pd.to_numeric(bs["value"], errors="coerce")
        occupied = sb[(sv > 0) & sb.notna()]
        if len(occupied):
            max_len = float(occupied.max())
    except Exception as exc:
        print(f"    (abundance_by_size unavailable: {exc})")

    return {
        "final": {s: (float(bio[s].iloc[-1]), float(abd[s].iloc[-1])) for s in focal},
        "ssb": ssb.copy(),
        "seeded": seeded.copy(),
        "juv_1to3": juv,
        "max_len": max_len,
        "age_cols": cols_seen,
        "cfg": cfg,
    }


def _age_of(col):
    try:
        return float(str(col).split("-")[0].split("_")[-1])
    except Exception:
        return None


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_pred_"))
    demo = osmose_demo("baltic", tmp)
    cf_path = Path(demo["config_file"])
    cfg_dir = cf_path.parent
    raw = dict(OsmoseConfigReader().read(cf_path))
    n_sp = int(raw["simulation.nspecies"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)]
    focal = names[:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}
    seed_b = float(raw[f"population.seeding.biomass.sp{SUBJECT_SP}"])

    arms = (
        ("baseline", None, None),
        ("bioen", overlay, None),
        ("acc50", overlay, 0.50),
        ("acc10", overlay, 0.10),
        ("acc00", overlay, 0.00),
        ("accALL", overlay, 0.00, False),
    )
    out = {}
    for spec in arms:
        tag, ov, dose = spec[0], spec[1], spec[2]
        split = spec[3] if len(spec) > 3 else True
        print(f"  running {tag} ...", flush=True)
        out[tag] = run_arm(raw, cfg_dir, ov, dose, focal, split=split)

    print(
        f"\n{'=' * 92}\nE1 -- juvenile prey row of {SUBJECT}, READ BACK from a built EngineConfig"
    )
    print("=" * 92)
    rows = {}
    for tag, *_rest in arms:
        rows[tag] = juvenile_row_in_effect(out[tag]["cfg"])
        shown = rows[tag][:9] if rows[tag] else []
        print(f"  {tag:<10} {[f'{v:.4f}' for v in shown]}")
    distinct = {tuple(v) for v in rows.values() if v}
    e1 = len(distinct) > 1
    print(f"  E1 arms differ in the loaded matrix: {e1}")

    print(f"\n{'=' * 92}\nFINAL STATE (year {N_YEAR}, seed {SEED})\n{'=' * 92}")
    print(f"{'species':<13}" + "".join(f"{t:>17}" for t, *_r in arms))
    for s in focal:
        line = f"{s:<13}"
        for tag, *_rest in arms:
            bm, ab = out[tag]["final"][s]
            line += f"{bm:>13.1f}{'  X' if ab <= 0 else '   '}"
        print(line)
    print("X = zero abundance")

    print(f"\n{'=' * 92}\n{SUBJECT} SSB DECOMPOSITION (the primary instrument)\n{'=' * 92}")
    print(
        f"{'arm':<12}{'SSB total':>15}{'seeded part':>14}{'REAL SSB':>13}"
        f"{'juv bio 1-3yr':>15}{'max len cm':>12}"
    )
    real = {}
    for tag, *_rest in arms:
        tot = out[tag]["ssb"][SUBJECT_SP]
        sp = out[tag]["seeded"][SUBJECT_SP] * seed_b
        real[tag] = tot - sp
        print(
            f"{tag:<12}{tot:>15.1f}{sp:>14.1f}{real[tag]:>13.1f}"
            f"{out[tag]['juv_1to3']:>15.4g}{out[tag]['max_len']:>12.2f}"
        )
    print("  (maturity length m0 = 38.0 cm — max len must approach it for SSB to be reachable)")

    e2 = all(out["baseline"]["final"][s][1] > 0 for s in focal)
    jb, ja = out["bioen"]["juv_1to3"], out["acc00"]["juv_1to3"]
    e3 = bool(np.isfinite(jb) and np.isfinite(ja) and not np.isclose(jb, ja))

    print(f"\n{'=' * 92}\nPRE-REGISTERED VERDICT\n{'=' * 92}")
    print(f"  E1 matrix differs between arms          : {e1}")
    print(f"  E2 baseline sustains all nine            : {e2}")
    print(f"  E3 juvenile survivorship bioen != acc00  : {e3}   ({jb:.4g} -> {ja:.4g})")
    if not (e1 and e2 and e3):
        print("\n  INCONCLUSIVE -- an engagement check failed. The finding is the KNOB, not the")
        print("  biology. Do NOT read this as evidence against predation.")
        return 0

    monotone = real["acc00"] >= real["acc10"] >= real["acc50"] >= real["bioen"]
    # accALL is the DECISIVE arm: the age-split arms only protect cod_west to MATURITY_AGE, but
    # maturity is by LENGTH (38 cm), so a slow-growing fish becomes edible again while still
    # juvenile-sized. accALL zeroes the prey row at every age -- inedible, full stop.
    decisive = "accALL" if "accALL" in real else "acc00"
    if real[decisive] > 0:
        print(
            f"\n  PREDATION CONFIRMED -- {SUBJECT} real SSB {real['bioen']:.1f} -> "
            f"{real[decisive]:.1f} t on arm {decisive!r} (total predation immunity)."
        )
        print(f"  dose ladder monotone: {monotone}")
    else:
        # E4, the window check. Maturity here is LENGTH-only (species.maturity.age.sp0 absent), so
        # SSB = 0 has two very different explanations and they must not be conflated: the cohort was
        # eaten, or the cohort survived but never grew to 38 cm. Only the first is "predation
        # refuted"; the second means the test could not reach the question.
        reached = out[decisive]["max_len"]
        if not np.isfinite(reached):
            print("\n  INCONCLUSIVE (E4 instrument dark) -- the max-occupied-size-bin readout is")
            print("  NaN, so whether cod_west ever approached 38 cm is UNMEASURED. This is an")
            print("  instrument failure, not a window finding and not a null. Fix E4 and re-run.")
        elif reached < 0.9 * 38.0:
            print(
                f"\n  INCONCLUSIVE (window/growth-limited) -- at FULL juvenile immunity "
                f"{SUBJECT} still"
            )
            print(f"  only reaches {reached:.2f} cm against a maturity length of 38.0 cm, so it")
            print(
                "  could not have spawned regardless of predation. SSB = 0 here means 'too small",
            )
            print("  to mature', NOT 'predation is not the cause'. The in-engine growth shortfall")
            print("  is then the finding, and it is NOT the same claim as the refuted offline-fit")
            print("  growth account -- that one was about the FIT's curve, this is about realized")
            print("  in-engine growth under food limitation.")
        else:
            print(
                f"\n  PREDATION REFUTED -- {SUBJECT} real SSB is still exactly 0.0 t at FULL juvenile"
            )
            print(f"  immunity while reaching {reached:.2f} cm (>= 0.9 x m0), so it was big enough")
            print(
                "  to mature and still produced no spawning stock. Every engagement check passed."
            )
            print("  Growth, recruitment and predation are then all dead, and the cause is")
            print("  something not yet enumerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
