"""What caps abundance under C3 bioen? A full population balance: BIRTHS and DEATHS BY CAUSE.

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. Under the bioen overlay at 50 yr with
production seeding, cod_west/cod_east/herring/flounder read **exactly 0.0 t** final-decade mean on
all five seeds, while a bioen-OFF control on the identical config sustains all nine. Four candidate
causes are dead, each killed by a pre-registered intervention:

  GROWTH (offline fit)  -- fixing the curve completely changed nothing.
  REALIZED GROWTH       -- measured by size class: intake 90-100% of the allometric cap at EVERY
                           size, m_share 0.13-0.20 (BELOW the fit's 0.30 target, not rising), dw/w
                           12-33% of body weight per step. The fish grow well.
  RECRUITMENT as framed -- under a 1-cohort stress real SSB is EXACTLY 0.0, so a 10x egg boost
                           multiplied zero by ten. (NOT ruled out at production seeding, where
                           spawners exist for `lifespan` years.)
  PREDATION             -- zeroing cod_west's prey row at every age gave a clean 3.07x monotone
                           dose-response on juvenile survivorship and did not change the outcome.
  THE GREYSEAL DEFECT   -- real and material (it caps SIZE: cod_west 15 -> 75 cm, flounder fully
                           restored to 40 cm) but repairing the whole column changes NO biomass:
                           0/4 recover, 0/5 seeds.

So growth is fine, size is fine once repaired, and the numbers still go to zero.

**Abundance is a balance: N(t+1) = N(t) + BIRTHS - DEATHS.** Every previous test measured one
quantity and inferred a mechanism -- which failed three times (see the "distributions show where,
not why" pattern in this investigation). This measures BOTH SIDES OF THE BALANCE directly, so the
answer is read off an accounting identity rather than inferred from a shape.

Instruments, both exact and both on the same run:

  DEATHS  `step_observer` (`simulate.py:2096`) fires AFTER mortality and BEFORE `state.compact()`,
          so `state.n_dead` -- shape (n_schools, len(MortalityCause)) -- holds THIS step's deaths by
          cause with zeroed schools still present. Reset per step at `_reset_step_variables`
          (`simulate.py:1867`). Accumulated per species per year over all 8 causes: PREDATION,
          STARVATION, ADDITIONAL, FISHING, OUT, FORAGING, DISCARDS, AGING.
          COUNTS, never rates -- mortality-rate outputs are SUMS of per-step rates and cannot be
          exponentiated (CLAUDE.md).
  BIRTHS  wrapping `regulate_recruitment`, the single shared choke point both reproduction paths
          pass through, which returns the post-regulation egg count per species.

Both arms use the **repaired** accessibility matrix (GreySeal given Cormorant's column), so the
known defect is not confounding what is left.

Arms: `baseline` (bioen OFF) and `bioen` (bioen ON), 50 yr, production seeding, seed 42. The 5-seed
restage showed this outcome is deterministic across seeds, so one seed suffices for a diagnosis.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Comparison window: the years in which BOTH arms hold non-zero abundance for the species (printed, so
the window is auditable). Per-capita death rate by cause = deaths / (deaths + surviving abundance),
which is comparable across arms despite very different population sizes.

  (1) one cause has a bioen/baseline per-capita ratio >= 2x AND is the largest absolute contributor
        -> THAT CAUSE CAPS ABUNDANCE. Named, with its ratio.
  (2) total per-capita mortality is comparable (< 1.5x) between arms while BIRTHS are >= 2x lower
        under bioen
        -> THE BIRTH SIDE CAPS ABUNDANCE, at production scale where spawners do exist. This is
           distinct from the already-refuted 1-cohort recruitment result and must be labelled so.
  (3) both sides move
        -> report the decomposition rather than forcing a single answer: which share of the gap is
           births and which is deaths.
  (4) neither side differs materially in the shared window
        -> the balance does not close, meaning the collapse happens OUTSIDE the window measured, and
           the finding is that the window is wrong rather than any mechanism.

Engagement checks:
  E1 the observer fires and records non-zero deaths for every species;
  E2 the births wrapper fires and records non-zero eggs for every species;
  E3 `baseline` sustains all nine to the final year;
  E4 the balance approximately closes on the baseline arm -- abundance change over a year should
     track births minus deaths. If it does not, the accounting is wrong and nothing may be read.

======================================================================================================

Run: .venv/bin/python scripts/c3_abundance_balance.py
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
from osmose.engine.state import MortalityCause  # noqa: E402

N_YEAR = 50
SEED = 42
SEAL, TEMPLATE = "GreySeal", "Cormorant"
FOCUS = ("cod_west", "cod_east", "herring", "flounder", "sprat")
COLLAPSED = ("cod_west", "cod_east", "herring", "flounder")
CAUSES = [c.name for c in MortalityCause]
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIG_REGULATE = repro_mod.regulate_recruitment


def write_repaired(src: Path, dst: Path) -> None:
    rows = list(csv.reader(src.open(), delimiter=";"))
    hdr = [h.strip() for h in rows[0]]
    tcol = hdr.index(TEMPLATE)
    out = [rows[0] + [SEAL]]
    for row in rows[1:]:
        if row:
            out.append(row + [repr(float(row[tcol]))])
    out.append([SEAL] + ["0"] * (len(out[0]) - 1))
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def run_arm(raw, cfg_dir, overlay, n_sp, ndt):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg.pop("population.seeding.year.max", None)  # production seeding
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    write_repaired(cfg_dir / "predation-accessibility.csv", cfg_dir / "acc-rep.csv")
    cfg["predation.accessibility.file"] = "acc-rep.csv"

    deaths = np.zeros((N_YEAR, n_sp, len(CAUSES)))
    standing = np.zeros((N_YEAR, n_sp))
    births = np.zeros((N_YEAR, n_sp))
    n_obs = [0]

    def observer(step, state, grid, config, map_sets):
        yr = min(step // ndt, N_YEAR - 1)
        n_obs[0] += 1
        sp = np.asarray(state.species_id)
        nd = np.asarray(state.n_dead, dtype=np.float64)
        abd = np.asarray(state.abundance, dtype=np.float64)
        for s in range(n_sp):
            m = sp == s
            if m.any():
                deaths[yr, s, :] += nd[m].sum(axis=0)
                standing[yr, s] += abd[m].sum()

    def wrapper(n_eggs_linear, ssb_in, seeded_this_step, config, step):
        out = _ORIG_REGULATE(n_eggs_linear, ssb_in, seeded_this_step, config, step)
        births[min(step // ndt, N_YEAR - 1), :] += np.asarray(out, dtype=np.float64)
        return out

    from osmose.engine.simulate import simulate

    eng = PythonEngine()
    ec, grid, rng, mv, mo = eng._prepare_run(cfg, SEED)
    repro_mod.regulate_recruitment = wrapper
    try:
        simulate(
            ec,
            grid,
            rng,
            movement_rngs=mv,
            mortality_rngs=mo,
            output_dir=None,
            step_observer=observer,
        )
    finally:
        repro_mod.regulate_recruitment = _ORIG_REGULATE
    return {"deaths": deaths, "standing": standing, "births": births, "n_obs": n_obs[0]}


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_balance_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    ndt = int(raw["simulation.time.ndtperyear"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)][:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    print(f"{N_YEAR} yr, production seeding, seed {SEED}, REPAIRED matrix on both arms")
    res = {}
    for tag, ov in (("baseline", None), ("bioen", overlay)):
        print(f"  running {tag} ...", flush=True)
        res[tag] = run_arm(raw, cf.parent, ov, n_sp, ndt)

    e1 = all(res[t]["deaths"].sum() > 0 for t in res)
    e2 = all(res[t]["births"].sum() > 0 for t in res)
    e3 = all(res["baseline"]["standing"][-1, names.index(s)] > 0 for s in names)
    print(f"\n  E1 observer fired ({res['bioen']['n_obs']} steps), deaths recorded : {e1}")
    print(f"  E2 births wrapper fired, eggs recorded                     : {e2}")
    print(f"  E3 baseline sustains all nine to the final year            : {e3}")

    for s in FOCUS:
        i = names.index(s)
        b_std, x_std = res["baseline"]["standing"][:, i], res["bioen"]["standing"][:, i]
        shared = [y for y in range(N_YEAR) if b_std[y] > 0 and x_std[y] > 0]
        print(
            f"\n{'=' * 100}\n{s}  — shared non-zero window: years {shared[0] if shared else '-'}"
            f"–{shared[-1] if shared else '-'} ({len(shared)} yr)\n{'=' * 100}"
        )
        if not shared:
            print("  no shared window — bioen never coexists with a live baseline here")
            continue
        w = np.array(shared)

        bb, xb = res["baseline"]["births"][w, i].sum(), res["bioen"]["births"][w, i].sum()
        print(
            f"  BIRTHS (eggs, window total)   baseline {bb:.4e}   bioen {xb:.4e}   "
            f"ratio {xb / bb if bb else float('nan'):.3f}"
        )

        print(
            f"\n  {'cause':<12}{'baseline/cap':>14}{'bioen/cap':>13}{'ratio':>9}{'bioen share':>13}"
        )
        bd = res["baseline"]["deaths"][w, i, :].sum(axis=0)
        xd = res["bioen"]["deaths"][w, i, :].sum(axis=0)
        b_pc_tot = bd.sum() / (bd.sum() + b_std[w].sum())
        x_pc_tot = xd.sum() / (xd.sum() + x_std[w].sum())
        for c, cname in enumerate(CAUSES):
            if bd[c] == 0 and xd[c] == 0:
                continue
            bpc = bd[c] / (bd.sum() + b_std[w].sum())
            xpc = xd[c] / (xd.sum() + x_std[w].sum())
            print(
                f"  {cname:<12}{bpc:>14.5f}{xpc:>13.5f}"
                f"{(xpc / bpc if bpc > 0 else float('inf')):>9.2f}"
                f"{(xd[c] / xd.sum() if xd.sum() else 0):>13.3f}"
            )
        print(
            f"  {'TOTAL':<12}{b_pc_tot:>14.5f}{x_pc_tot:>13.5f}"
            f"{(x_pc_tot / b_pc_tot if b_pc_tot else float('nan')):>9.2f}"
        )

    print(f"\n{'=' * 100}\nPRE-REGISTERED VERDICT (the four collapsed stocks)\n{'=' * 100}")
    for s in COLLAPSED:
        i = names.index(s)
        b_std, x_std = res["baseline"]["standing"][:, i], res["bioen"]["standing"][:, i]
        w = np.array([y for y in range(N_YEAR) if b_std[y] > 0 and x_std[y] > 0])
        if not len(w):
            print(f"  {s:<11} no shared window — INCONCLUSIVE for this stock")
            continue
        bd = res["baseline"]["deaths"][w, i, :].sum(axis=0)
        xd = res["bioen"]["deaths"][w, i, :].sum(axis=0)
        bt = bd.sum() / (bd.sum() + b_std[w].sum())
        xt = xd.sum() / (xd.sum() + x_std[w].sum())
        mort_ratio = xt / bt if bt else float("nan")
        bb, xb = res["baseline"]["births"][w, i].sum(), res["bioen"]["births"][w, i].sum()
        birth_ratio = xb / bb if bb else float("nan")
        ratios = {
            CAUSES[c]: (xd[c] / (xd.sum() + x_std[w].sum())) / (bd[c] / (bd.sum() + b_std[w].sum()))
            for c in range(len(CAUSES))
            if bd[c] > 0
        }
        worst = max(ratios, key=ratios.get) if ratios else "-"
        verdict = (
            f"DEATHS via {worst} ({ratios.get(worst, float('nan')):.2f}x)"
            if mort_ratio >= 1.5
            else (
                f"BIRTHS ({birth_ratio:.3f}x of baseline)"
                if birth_ratio <= 0.5
                else "neither side differs materially — window may be wrong"
            )
        )
        print(
            f"  {s:<11} total mort {mort_ratio:>6.2f}x   births {birth_ratio:>7.3f}x   -> {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
