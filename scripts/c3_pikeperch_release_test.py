"""Does pikeperch survive C3 bioen because its cod predators die first? An intervention.

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. pikeperch is not exempt from the maturation
bottleneck — its `frac_mature` ratio is 0.0888 and its SSB ratio 0.0340, *lower* than herring's
0.0356, which dies. What distinguishes it is the trajectory: zero mature fish until year 3, then
takeoff to 1.3e8 (yr4) / 3.7e9 (yr6) / 6.8e9 (yr8), with total abundance rebounding 22x from a
year-5 floor of 6.0e10 — climbing while every collapser decays. Both cod stocks are listed predators
of pikeperch (`cod_west 0.1`, `cod_east 0.05`) and both go extinct over years 3-5.

THE COMPETING EXPLANATIONS, and why timing cannot separate them:

  RELEASE  cod predation suppresses pikeperch; when the cods die it is released and builds.
  GROWTH   pikeperch simply has the largest m0 (40 cm) and takes ~3 yr to reach it, so onset at
           year 4 is when its first cohort crosses the threshold — and the cods dying then is
           coincidence.

Measured vBGF time-to-m0 for pikeperch is **2.97 yr**, and its observed onset moves from year 2
(baseline) to year 4 (bioen), tracking its growth delay (mean weight 0.564x). cod_west shifts the
same way, year 1 -> year 4. **So onset timing is consistent with BOTH stories and discriminates
neither.** What still needs explaining is why pikeperch BUILDS after onset while everything else
decays.

THE INTERVENTION. One surgical edit: in the accessibility matrix, zero the `cod_west` and `cod_east`
cells of pikeperch's PREY row. Nothing else changes — not pikeperch's own predator column (what it
eats), not the Cormorant or cannibalism cells, not any other species' row, not a single bioen
parameter. The GreySeal column is repaired on both arms so that defect cannot confound.

THE WINDOW IS THE POINT. Years 0-3 are the discriminating window, because the cods are ALIVE then in
the reference arm. If cod predation is materially suppressing pikeperch, removing it must show up
THERE. From year ~6 the cods are extinct on both arms, so the arms must CONVERGE — that convergence
is a built-in specificity check: an intervention that changed late-run behaviour would be doing
something other than what it claims.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Primary instrument: pikeperch `N_total` and `N_mature` per year, from `step_observer`, same maturity
predicate as `_bioen_reproduction`.

  (1) pikeperch N_total over years 0-3 is >= 1.5x higher on `nocod`
        -> PREDATION RELEASE IS REAL. Cod predation materially suppresses pikeperch, and its
           survival is at least partly a release effect.
  (2) years 0-3 differ by < 1.2x
        -> RELEASE REFUTED. Cod predation on pikeperch is immaterial, the year-4 takeoff is
           growth-timed, and the cod-extinction coincidence was exactly that. pikeperch's survival
           must then be explained by its pool size and 0.0137/yr drain rate alone.
  (3) between 1.2x and 1.5x
        -> WEAK/PARTIAL; report the number without claiming a mechanism.

Specificity check (not the verdict, but invalidating if it fails): years 8-19 must agree within 1.5x
between arms. Both have extinct cods there, so a persistent late difference would mean the edit
changed something beyond cod predation on pikeperch.

Engagement checks:
  E1 the two matrices differ at exactly the two intended cells and nowhere else (asserted on the
     written CSVs before either run);
  E2 the cods still collapse on BOTH arms — the intervention must not rescue them, or the
     comparison is between different worlds.

======================================================================================================

Run:  .venv/bin/python scripts/c3_pikeperch_release_test.py ref   <out.json>
      .venv/bin/python scripts/c3_pikeperch_release_test.py nocod <out.json>
      .venv/bin/python scripts/c3_pikeperch_release_test.py       (combine; expects the two files)
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

from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

N_YEAR = 20
SEED = 42
SEAL, TEMPLATE = "GreySeal", "Cormorant"
SUBJECT = "pikeperch"
COD = ("cod_west", "cod_east")
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
SCRATCH = Path(
    "/tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad"
)


def write_matrix(src: Path, dst: Path, drop_cod_on_subject: bool) -> None:
    """Repair the GreySeal column on both arms; optionally zero the cods' cells in SUBJECT's row."""
    rows = list(csv.reader(src.open(), delimiter=";"))
    hdr = [h.strip() for h in rows[0]]
    t = hdr.index(TEMPLATE)
    out = [rows[0] + [SEAL]]
    for r in rows[1:]:
        if not r:
            continue
        r = list(r) + [repr(float(r[t]))]
        if drop_cod_on_subject and r[0].strip() == SUBJECT:
            for c in COD:
                r[hdr.index(c)] = "0.0"
        out.append(r)
    out.append([SEAL] + ["0"] * (len(out[0]) - 1))
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def run_arm(raw, cfg_dir, overlay, drop, n_sp, ndt, m0_arr, m1_arr):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg.pop("population.seeding.year.max", None)
    cfg.update(overlay)
    name = f"acc-{'nocod' if drop else 'ref'}.csv"
    write_matrix(cfg_dir / "predation-accessibility.csv", cfg_dir / name, drop)
    cfg["predation.accessibility.file"] = name

    tot = np.zeros((N_YEAR, n_sp))
    mat = np.zeros((N_YEAR, n_sp))

    def observer(step, state, grid, config, map_sets):
        yr = min(step // ndt, N_YEAR - 1)
        sp = np.asarray(state.species_id)
        abd = np.asarray(state.abundance, dtype=np.float64)
        ok = (abd > 0) & (~np.asarray(state.is_egg)) & (sp < n_sp)
        if not ok.any():
            return
        spo, a = sp[ok], abd[ok]
        tot[yr] += np.bincount(spo, weights=a, minlength=n_sp)[:n_sp]
        thr = m0_arr[spo] + m1_arr[spo] * (np.asarray(state.age_dt)[ok] / ndt)
        m = np.asarray(state.length, dtype=np.float64)[ok] >= thr
        if m.any():
            mat[yr] += np.bincount(spo[m], weights=a[m], minlength=n_sp)[:n_sp]

    from osmose.engine.simulate import simulate

    ec, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, SEED)
    simulate(
        ec, grid, rng, movement_rngs=mv, mortality_rngs=mo, output_dir=None, step_observer=observer
    )
    return {"tot": tot, "mat": mat}


def main() -> int:
    warnings.simplefilter("ignore")
    arm = sys.argv[1] if len(sys.argv) > 1 else None
    tmp = Path(tempfile.mkdtemp(prefix="c3_release_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    ndt = int(raw["simulation.time.ndtperyear"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)][:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    from osmose.engine.config import EngineConfig

    probe = dict(raw)
    probe.update(overlay)
    ec = EngineConfig.from_dict(probe)
    m0_arr = np.asarray(getattr(ec, "bioen_m0", np.zeros(n_sp)), dtype=np.float64)
    m1_arr = np.asarray(getattr(ec, "bioen_m1", np.zeros(n_sp)), dtype=np.float64)

    if arm in ("ref", "nocod"):
        # E1: the two matrices must differ at EXACTLY the two intended cells.
        a, b = tmp / "_a.csv", tmp / "_b.csv"
        write_matrix(cf.parent / "predation-accessibility.csv", a, False)
        write_matrix(cf.parent / "predation-accessibility.csv", b, True)
        ra = [r for r in csv.reader(a.open(), delimiter=";")]
        rb = [r for r in csv.reader(b.open(), delimiter=";")]
        diffs = [
            (ra[i][0], ra[0][j])
            for i in range(len(ra))
            for j in range(len(ra[i]))
            if ra[i][j] != rb[i][j]
        ]
        print(f"  E1 cells differing between arms: {diffs}")
        assert len(diffs) == 2 and all(d[0] == SUBJECT for d in diffs), f"E1 FAILED: {diffs}"

        print(f"  running {arm} ...", flush=True)
        r = run_arm(raw, cf.parent, overlay, arm == "nocod", n_sp, ndt, m0_arr, m1_arr)
        Path(sys.argv[2]).write_text(json.dumps({k: v.tolist() for k, v in r.items()}))
        print(f"  wrote {sys.argv[2]}")
        return 0

    res = {}
    for tag in ("ref", "nocod"):
        pth = SCRATCH / f"_rel_{tag}.json"
        if not pth.exists():
            print(f"  MISSING {pth}")
            return 1
        res[tag] = {k: np.asarray(v) for k, v in json.loads(pth.read_text()).items()}

    i = names.index(SUBJECT)
    cw, ce = names.index("cod_west"), names.index("cod_east")
    print(f"\n{'=' * 92}\n{SUBJECT}: reference vs cod-predation removed\n{'=' * 92}")
    print(
        f"{'yr':>3}{'ref N_tot':>13}{'nocod N_tot':>14}{'ratio':>8}"
        f"{'ref N_mat':>13}{'nocod N_mat':>14}{'  cods alive (ref)':>20}"
    )
    for y in range(14):
        rt, nt = res["ref"]["tot"][y, i], res["nocod"]["tot"][y, i]
        rm, nm = res["ref"]["mat"][y, i], res["nocod"]["mat"][y, i]
        codn = res["ref"]["tot"][y, cw] + res["ref"]["tot"][y, ce]
        print(
            f"{y:>3}{rt:>13.3e}{nt:>14.3e}{(nt / rt if rt > 0 else float('nan')):>8.3f}"
            f"{rm:>13.3e}{nm:>14.3e}{codn:>20.3e}"
        )

    early = slice(0, 4)
    late = slice(8, N_YEAR)
    r_e = res["nocod"]["tot"][early, i].sum() / max(res["ref"]["tot"][early, i].sum(), 1e-30)
    r_l = res["nocod"]["tot"][late, i].sum() / max(res["ref"]["tot"][late, i].sum(), 1e-30)
    cods_die = all(res[t]["tot"][12, c] < 1.0 for t in ("ref", "nocod") for c in (cw, ce))

    print(f"\n{'=' * 92}\nPRE-REGISTERED VERDICT\n{'=' * 92}")
    print(f"  E2 cods still collapse on BOTH arms       : {cods_die}")
    print(f"  years 0-3 (cods ALIVE)  nocod/ref N_total : {r_e:.3f}")
    print(f"  years 8-19 (cods gone)  nocod/ref N_total : {r_l:.3f}  (specificity: want ~1)")
    if not cods_die:
        print("\n  INCONCLUSIVE — the intervention rescued a cod stock, so the arms are different")
        print("  worlds and the comparison is not clean.")
    elif not (1 / 1.5 <= r_l <= 1.5):
        print("\n  INCONCLUSIVE — the arms differ late, when both have extinct cods. The edit is")
        print("  doing something beyond cod predation on pikeperch.")
    elif r_e >= 1.5:
        print("\n  PREDATION RELEASE IS REAL — removing cod predation materially raises pikeperch")
        print("  while the cods are alive, so its survival is at least partly a release effect.")
    elif r_e < 1.2:
        print(
            "\n  RELEASE REFUTED — cod predation on pikeperch is immaterial. The year-4 takeoff is"
        )
        print(
            "  growth-timed (vBGF t(m0) = 2.97 yr) and the cod-extinction coincidence was exactly"
        )
        print("  that. pikeperch's survival rests on its pool size and 0.0137/yr drain rate.")
    else:
        print(
            f"\n  WEAK/PARTIAL — early ratio {r_e:.3f} sits between the bars; report, claim nothing."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
