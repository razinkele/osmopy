"""Why does cod_west stall at 15-20 cm under bioen? Energy budget resolved BY SIZE CLASS.

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. Three interventions have run:

  GROWTH (the offline fit) -- REFUTED. Fixing the fit's curve completely changed nothing.
  RECRUITMENT              -- ILL-POSED. cod_west real SSB is EXACTLY 0.0 t; a 10x egg boost
                              multiplied zero by ten.
  PREDATION                -- REAL BUT NOT BINDING. A clean monotone 3.07x dose-response on
                              juvenile survivorship, yet with cod_west inedible to EVERY predator
                              at EVERY age it still tops out in the [15,20) cm size bin against the
                              38 cm it needs to mature. Bioen-off on the identical config: 110 cm.

So the binding constraint is realized in-engine growth: a ~7x shortfall in length that the offline
forward model cannot represent, because it assumes ingestion at 100% of cap with no competitor
drawing the same prey down. This script asks WHY, at the only resolution that can answer it.

Two rival explanations produce the SAME size distribution and must be separated:

  (A) STALL   -- fish reach 15-20 cm and stop growing. Specific growth dw/w -> 0 there while fish
                 are still present and alive.
  (B) ATTRITION -- fish keep growing fine but none survive long enough to exceed 15-20 cm. Specific
                 growth stays healthy in the top occupied bin; the bin is simply the survival edge.

The discriminator is specific growth IN the top occupied bin, with abundance present. A size
distribution alone cannot tell these apart -- which is exactly the trap this whole line of work
keeps hitting.

Instrument. `_bioen_step` (`osmose/engine/simulate.py:390`) returns a state carrying per-SCHOOL
`e_gross`/`e_maint`/`e_net`/`rho` (Java's tonnes-per-school framework), and `weight` is per-FISH, so
`new.weight - old.weight` is dw per fish directly. Wrapping it gives every quantity at once with no
engine edit. Recorded per length bin, abundance-weighted:

  ing/cap    realized ingestion / the allometric cap. Cap per fish per step is
             `(Imax/ndt) * w_g^beta * 1e-6` t -- the per-subdt cap times n_subdt. theta=1 and
             c_rate=0 in the committed overlay, so the larval branch is inert.
  m_share    e_maint / e_gross. The fit pins this at 0.30 at T_ref BY CONSTRUCTION (c_m is solved
             from it), and it is SIZE-INDEPENDENT in the fit because e_gross and e_maint both
             scale as w^beta. Any rise with size is therefore a pure engine/fit divergence.
  dw/w       specific growth per step -- the discriminator above.
  e_net/fish net energy per fish per step.

=========================== PRE-REGISTERED READING (written before the run) ===========================

  (1) m_share rises toward 1.0 with size AND dw/w -> 0 in the top occupied bins, with abundance
      present there
        -> STALL CONFIRMED: maintenance overtakes intake, a hard energetic ceiling. The sub-reading
           is then whether it is driven by ing/cap falling (food limitation) or by m_share rising
           at constant ing/cap (the c_m / beta calibration itself).
  (2) dw/w stays healthy (> 1% per step) in the top occupied bin
        -> ATTRITION, not a stall. The ceiling is a survival edge and the question becomes what
           kills them, NOT what stops their growth.
  (3) fewer than 3 occupied bins, or no abundance above 10 cm
        -> INCONCLUSIVE: too little of the size range is populated to resolve a trend.

Engagement checks:
  E1 the wrapper fires (rows recorded > 0) and covers more than one size bin;
  E2 cod_west abundance is present in the top occupied bin -- otherwise (1) and (2) are both
     unreadable and only the INCONCLUSIVE branch applies.

Run on the `accALL` predation-immune arm as well as plain `bioen`: if the ceiling is energetic it
must persist when nothing can eat cod_west, and that arm is the cleaner read because attrition by
predation is removed by construction.

======================================================================================================

Run: .venv/bin/python scripts/c3_growth_ceiling_diagnosis.py
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

import osmose.engine.simulate as sim
from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

N_YEAR = 8
SEED = 42
SUBJECT_SP = 0
SUBJECT = "cod_west"
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
BIN_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 38, 50, 70, 120], dtype=np.float64)
_ORIG_BIOEN_STEP = sim._bioen_step


def _accum():
    n = len(BIN_EDGES) - 1
    return {k: np.zeros(n) for k in ("abd", "ing", "cap", "egross", "emaint", "enet", "dw", "w")}


def _make_wrapper(acc, imax, beta, ndt):
    def wrapper(state, config, temp_data, step, o2_data=None, trait_overrides=None, dbg=None):
        pre_w = state.weight.copy()
        pre_abd = state.abundance.copy()
        pre_len = state.length.copy()
        pre_ing = state.preyed_biomass.copy()
        new = _ORIG_BIOEN_STEP(state, config, temp_data, step, o2_data, trait_overrides, dbg)

        m = (
            (state.species_id == SUBJECT_SP)
            & (pre_abd > 0)
            & (~state.is_out)
            & (~state.is_egg)
            & (pre_w > 0)
        )
        if not m.any():
            return new
        w_g = pre_w[m] * 1e6
        abd = pre_abd[m]
        cap = (imax / ndt) * np.power(w_g, beta) * 1e-6  # tonnes per fish per step
        idx = np.clip(np.digitize(pre_len[m], BIN_EDGES) - 1, 0, len(BIN_EDGES) - 2)
        dw = new.weight[m] - pre_w[m]
        contrib = {
            "abd": abd,
            "ing": pre_ing[m] / np.maximum(abd, 1e-30),
            "cap": cap,
            "egross": new.e_gross[m] / np.maximum(abd, 1e-30),
            "emaint": new.e_maint[m] / np.maximum(abd, 1e-30),
            "enet": new.e_net[m] / np.maximum(abd, 1e-30),
            "dw": dw,
            "w": pre_w[m],
        }
        for key, vals in contrib.items():
            weights = abd if key != "abd" else np.ones_like(abd)
            np.add.at(acc[key], idx, vals * weights)
        return new

    return wrapper


def write_all_ages_zero(src: Path, dst: Path) -> None:
    """cod_west inedible at every age: one unsplit prey row, all zeros."""
    rows = list(csv.reader(src.open(), delimiter=";"))
    out = []
    for row in rows:
        if row and row[0].strip() == SUBJECT:
            out.append([SUBJECT] + ["0.0"] * (len(row) - 1))
        else:
            out.append(row)
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def run_arm(raw, cfg_dir, overlay, immune, imax, beta, ndt):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg["population.seeding.year.max"] = "1"
    cfg.update(overlay)
    if immune:
        write_all_ages_zero(cfg_dir / "predation-accessibility.csv", cfg_dir / "acc-immune.csv")
        cfg["predation.accessibility.file"] = "acc-immune.csv"
    acc = _accum()
    sim._bioen_step = _make_wrapper(acc, imax, beta, ndt)
    try:
        res = PythonEngine().run_in_memory(cfg, seed=SEED)
    finally:
        sim._bioen_step = _ORIG_BIOEN_STEP
    return acc, res


def report(tag, acc):
    print(f"\n{'=' * 104}\n{tag} — cod_west energy budget by LENGTH CLASS (abundance-weighted)")
    print("=" * 104)
    hdr = (
        f"{'length cm':<12}{'abundance':>14}{'ing/cap':>10}{'m_share':>10}"
        f"{'dw/w per step':>16}{'e_net/fish t':>15}{'mean w g':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = 0
    top = None
    for i in range(len(BIN_EDGES) - 1):
        a = acc["abd"][i]
        if a <= 0:
            continue
        rows += 1
        top = i
        ing, cap = acc["ing"][i] / a, acc["cap"][i] / a
        eg, em = acc["egross"][i] / a, acc["emaint"][i] / a
        w, dw = acc["w"][i] / a, acc["dw"][i] / a
        lbl = f"{BIN_EDGES[i]:.0f}-{BIN_EDGES[i + 1]:.0f}"
        print(
            f"{lbl:<12}{a:>14.4g}{(ing / cap if cap > 0 else np.nan):>10.3f}"
            f"{(em / eg if eg > 0 else np.nan):>10.3f}{(dw / w if w > 0 else np.nan):>16.5f}"
            f"{acc['enet'][i] / a:>15.3e}{w * 1e6:>11.2f}"
        )
    return rows, top


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_ceiling_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    cfg_dir = cf.parent
    raw = dict(OsmoseConfigReader().read(cf))
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}
    imax = float(overlay[f"predation.ingestion.rate.max.sp{SUBJECT_SP}"])
    beta = float(overlay[f"species.beta.sp{SUBJECT_SP}"])
    ndt = int(raw["simulation.time.ndtperyear"])
    print(f"cod_west bioen Imax = {imax:.4f}, beta = {beta}, ndt = {ndt}")
    print("fit pins m_share = 0.30 at T_ref BY CONSTRUCTION, and it is SIZE-INDEPENDENT in the fit")
    print(
        "(e_gross and e_maint both scale as w^beta) — so any rise with size is engine/fit divergence."
    )

    results = {}
    for tag, immune in (("bioen", False), ("accALL (predation-immune)", True)):
        print(f"\n  running {tag} ...", flush=True)
        acc, _res = run_arm(raw, cfg_dir, overlay, immune, imax, beta, ndt)
        results[tag] = acc

    verdicts = {}
    for tag, acc in results.items():
        rows, top = report(tag, acc)
        verdicts[tag] = (rows, top, acc)

    print(f"\n{'=' * 104}\nPRE-REGISTERED VERDICT\n{'=' * 104}")
    for tag, (rows, top, acc) in verdicts.items():
        print(f"\n[{tag}]")
        if rows == 0 or top is None:
            print("  INCONCLUSIVE — the wrapper recorded nothing (E1 failed).")
            continue
        a = acc["abd"][top]
        w = acc["w"][top] / a
        sg = (acc["dw"][top] / a) / w if w > 0 else float("nan")
        eg, em = acc["egross"][top] / a, acc["emaint"][top] / a
        ms = em / eg if eg > 0 else float("nan")
        lbl = f"{BIN_EDGES[top]:.0f}-{BIN_EDGES[top + 1]:.0f} cm"
        print(f"  occupied bins: {rows}; top occupied bin {lbl}, abundance {a:.4g}")
        print(f"  in that bin: dw/w = {sg:.5f} per step, m_share = {ms:.3f}")
        if rows < 3 or BIN_EDGES[top + 1] <= 10:
            print("  INCONCLUSIVE — too little of the size range is populated to resolve a trend.")
        elif np.isfinite(sg) and sg > 0.01:
            print("  ATTRITION, NOT A STALL — growth is still healthy in the top occupied bin, so")
            print("  the ceiling is a SURVIVAL EDGE. The question becomes what kills them, not")
            print("  what stops their growth.")
        else:
            print("  STALL CONFIRMED — growth has gone to ~zero in the top occupied bin while fish")
            print(
                "  are present there. Maintenance has overtaken intake: a hard energetic ceiling."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
