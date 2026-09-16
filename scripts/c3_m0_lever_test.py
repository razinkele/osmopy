"""Is the maturity threshold really the binding constraint? Lower m0 and see if the stocks live.

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. The C3 collapse has been traced to a
MATURATION BOTTLENECK, every link measured rather than inferred:

  mortality IDENTICAL between arms (1.00-1.01x per-capita)
    -> births 16-24x down
    -> a SPAWNER deficit, not a fecundity one (eggs per unit SSB are 1.02-1.24x)
    -> the SSB deficit is `frac_mature`, with `N_total` largely preserved
    -> maturity is LENGTH-based and bioen size-at-age is 0.32-0.82x baseline, so few cross m0
    -> no spawners -> no eggs -> extinction, with mortality normal throughout

cod_west is the cleanest single row: over years 2-6 it holds **79% of baseline's fish and
essentially ZERO of them mature**. And pikeperch, long the standing exception, turned out to obey the
same rule -- mature stock = pool x mature fraction, and 1.39e12 x 0.0028 is still ~4e9 spawners
while cod_west's 3.46e8 x 0.0000 is zero.

That chain is correlational at its last link: it says few fish cross m0 and that SSB is therefore
zero, but it has never been tested by INTERVENTION. Five compelling correlations in this
investigation have already turned out not to be causal (the growth account, the 11-48% egg ratios,
the size ceiling, the 24x gonad-flush fingerprint, the pikeperch predation release). This is the
test that would kill the sixth.

THE INTERVENTION, and why m0 rather than growth. Lowering `species.maturity.m0.sp{i}` changes the
maturity THRESHOLD and nothing else: not growth, not the energy budget, not predation, not
fecundity, not mortality. Raising growth instead would have been confounded, because
`predation.ingestion.rate.max` is simultaneously the bioen `Imax` AND the predation cap (CLAUDE.md's
lossy-alias gotcha), so a growth intervention silently changes predation pressure too. m0 is the one
knob that isolates the claim.

A DOSE LADDER, not a binary: m0 x 1.00 (reference), x 0.50, x 0.25, applied to the four collapsing
stocks only (cod_west 38, cod_east 22, herring 18, flounder 22). Survivors are left untouched so
they serve as an internal control -- if they move, the edit is not as surgical as claimed.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Primary instrument: final-decade mean biomass per species (the Stage-1 metric), plus `frac_mature`
and mature abundance per year from `step_observer`.

RECOVERY, single and non-disjunctive: a stock recovers iff its final-decade mean biomass exceeds
**1% of its own final-decade mean on the bioen-OFF baseline**.

  (1) >= 2 of the four recover at m0 x 0.25, AND the response is monotone in the dose
        -> MATURATION BOTTLENECK CONFIRMED BY INTERVENTION. The last link of the chain is causal,
           and the lever for C3 Stage 2 is the growth trajectory against m0.
  (2) frac_mature rises with the dose but biomass does not recover
        -> THE THRESHOLD IS NOT SUFFICIENT. Crossing m0 is necessary but something downstream
           (egg survival, larval mortality, density dependence) still caps the population. That
           would be a genuinely new finding and must be reported as one, not as a null.
  (3) frac_mature does NOT rise with the dose
        -> INCONCLUSIVE and the finding is the knob: lowering the threshold failed to make more
           fish mature, so the instrument or the edit is wrong, not the biology.
  (4) 0 recover with frac_mature clearly rising
        -> MATURATION BOTTLENECK REFUTED as the binding constraint, and the sixth correlation in
           this investigation falls.

Engagement checks:
  E1 `bioen_m0` read back from a constructed EngineConfig differs per arm at exactly the four
     intended species, and is unchanged for the other five;
  E2 the untouched survivors (sprat, pikeperch, smelt, stickleback) move < 1.5x across arms --
     otherwise the edit is reaching past its stated scope;
  E3 `frac_mature` for the four actually rises with the dose (the knob engages at all).

======================================================================================================

Run:  .venv/bin/python scripts/c3_m0_lever_test.py <arm> <out.json>     arm in base|d100|d050|d025
      .venv/bin/python scripts/c3_m0_lever_test.py                      (combine)
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

N_YEAR = 25
TAIL = 8
SEED = 42
SEAL, TEMPLATE = "GreySeal", "Cormorant"
TARGETS = {"cod_west": 0, "cod_east": 8, "herring": 1, "flounder": 3}
SURVIVORS = ("sprat", "pikeperch", "smelt", "stickleback")
DOSES = {"base": None, "d100": 1.00, "d050": 0.50, "d025": 0.25}
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
SCRATCH = Path(
    "/tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad"
)


def repaired(src: Path, dst: Path) -> None:
    rows = list(csv.reader(src.open(), delimiter=";"))
    t = [h.strip() for h in rows[0]].index(TEMPLATE)
    out = [rows[0] + [SEAL]] + [r + [repr(float(r[t]))] for r in rows[1:] if r]
    out.append([SEAL] + ["0"] * (len(out[0]) - 1))
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def build_cfg(raw, cfg_dir, overlay, dose):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg.pop("population.seeding.year.max", None)
    if dose is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
        for i in TARGETS.values():
            k = f"species.maturity.m0.sp{i}"
            cfg[k] = repr(float(overlay[k]) * dose)
    repaired(cfg_dir / "predation-accessibility.csv", cfg_dir / "acc-rep.csv")
    cfg["predation.accessibility.file"] = "acc-rep.csv"
    return cfg


def run_arm(cfg, n_sp, ndt):
    tot = np.zeros((N_YEAR, n_sp))
    mat = np.zeros((N_YEAR, n_sp))
    from osmose.engine.config import EngineConfig

    ec_probe = EngineConfig.from_dict(dict(cfg))
    m0 = np.asarray(
        getattr(ec_probe, "bioen_m0", None)
        if getattr(ec_probe, "bioen_m0", None) is not None
        else np.zeros(n_sp),
        dtype=np.float64,
    )
    m1 = np.asarray(
        getattr(ec_probe, "bioen_m1", None)
        if getattr(ec_probe, "bioen_m1", None) is not None
        else np.zeros(n_sp),
        dtype=np.float64,
    )

    def observer(step, state, grid, config, map_sets):
        yr = min(step // ndt, N_YEAR - 1)
        sp = np.asarray(state.species_id)
        abd = np.asarray(state.abundance, dtype=np.float64)
        ok = (abd > 0) & (~np.asarray(state.is_egg)) & (sp < n_sp)
        if not ok.any():
            return
        spo, a = sp[ok], abd[ok]
        tot[yr] += np.bincount(spo, weights=a, minlength=n_sp)[:n_sp]
        thr = m0[spo] + m1[spo] * (np.asarray(state.age_dt)[ok] / ndt)
        mm = np.asarray(state.length, dtype=np.float64)[ok] >= thr
        if mm.any():
            mat[yr] += np.bincount(spo[mm], weights=a[mm], minlength=n_sp)[:n_sp]

    from osmose.engine.simulate import simulate
    from osmose.results import OsmoseResults

    ec, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, SEED)
    outs = simulate(
        ec, grid, rng, movement_rngs=mv, mortality_rngs=mo, output_dir=None, step_observer=observer
    )
    res = OsmoseResults.from_outputs(outs, ec, grid)
    bio = res.biomass()
    tail = max(1, int(len(bio) * TAIL / N_YEAR))
    # Index by species NAME and coerce: bio.columns carries a non-numeric column that nanmean
    # cannot consume, and silently including it aborted the run.
    import pandas as pd

    fd = {}
    for s in [c for c in bio.columns if str(c).strip().lower() not in ("time", "year", "step")]:
        v = pd.to_numeric(bio[s], errors="coerce").to_numpy(dtype=float)[-tail:]
        if v.size and np.isfinite(v).any():
            fd[str(s)] = float(np.nanmean(v))
    return {"tot": tot, "mat": mat, "fd": fd, "m0": m0}


def main() -> int:
    warnings.simplefilter("ignore")
    arm = sys.argv[1] if len(sys.argv) > 1 else None
    tmp = Path(tempfile.mkdtemp(prefix="c3_m0_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    ndt = int(raw["simulation.time.ndtperyear"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)][:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    if arm in DOSES:
        cfg = build_cfg(raw, cf.parent, overlay, DOSES[arm])
        print(f"  running {arm} (dose {DOSES[arm]}) ...", flush=True)
        r = run_arm(cfg, n_sp, ndt)
        Path(sys.argv[2]).write_text(
            json.dumps(
                {
                    "tot": r["tot"].tolist(),
                    "mat": r["mat"].tolist(),
                    "fd": r["fd"],
                    "m0": r["m0"].tolist(),
                }
            )
        )
        print("  m0 in effect: " + ", ".join(f"{s}={r['m0'][i]:.2f}" for s, i in TARGETS.items()))
        print(f"  wrote {sys.argv[2]}")
        return 0

    res = {}
    for tag in DOSES:
        pth = SCRATCH / f"_m0_{tag}.json"
        if not pth.exists():
            print(f"  MISSING {pth}")
            return 1
        d = json.loads(pth.read_text())
        res[tag] = {
            "tot": np.asarray(d["tot"]),
            "mat": np.asarray(d["mat"]),
            "fd": d["fd"],
            "m0": np.asarray(d["m0"]),
        }

    print(f"\n{'=' * 96}\nE1 — bioen_m0 in effect per arm\n{'=' * 96}")
    print(f"{'species':<12}" + "".join(f"{t:>10}" for t in DOSES))
    for s, i in TARGETS.items():
        print(f"{s:<12}" + "".join(f"{res[t]['m0'][i]:>10.2f}" for t in DOSES))
    print("untouched survivors:")
    for s in SURVIVORS:
        i = names.index(s)
        print(f"{s:<12}" + "".join(f"{res[t]['m0'][i]:>10.2f}" for t in DOSES))

    w = slice(N_YEAR - TAIL, N_YEAR)
    print(
        f"\n{'=' * 96}\nE3 — frac_mature (years {N_YEAR - TAIL}-{N_YEAR - 1}); does the knob engage?"
    )
    print("=" * 96)
    print(f"{'species':<12}" + "".join(f"{t:>12}" for t in DOSES))
    for s in list(TARGETS) + list(SURVIVORS):
        i = names.index(s)
        row = f"{s:<12}"
        for t in DOSES:
            tt = res[t]["tot"][w, i].sum()
            row += f"{(res[t]['mat'][w, i].sum() / tt if tt > 0 else 0.0):>12.4f}"
        print(row)

    print(f"\n{'=' * 96}\nFINAL-DECADE MEAN BIOMASS (t)\n{'=' * 96}")
    print(f"{'species':<12}" + "".join(f"{t:>14}" for t in DOSES) + f"{'floor 1%':>12}")
    rec = {}
    for s in list(TARGETS) + list(SURVIVORS):
        floor = 0.01 * res["base"]["fd"].get(s, 0.0)
        row = f"{s:<12}" + "".join(f"{res[t]['fd'].get(s, 0.0):>14.1f}" for t in DOSES)
        print(row + f"{floor:>12.1f}")
        if s in TARGETS:
            rec[s] = {t: res[t]["fd"].get(s, 0.0) > floor for t in ("d100", "d050", "d025")}

    e2 = all(
        max(res[t]["fd"].get(s, 0.0) for t in ("d100", "d050", "d025"))
        <= 1.5 * max(min(res[t]["fd"].get(s, 1e-30) for t in ("d100", "d050", "d025")), 1e-30)
        for s in SURVIVORS
    )

    def fm(t, i):
        tt = res[t]["tot"][w, i].sum()
        return res[t]["mat"][w, i].sum() / tt if tt > 0 else 0.0

    e3 = all(fm("d025", names.index(s)) > fm("d100", names.index(s)) for s in TARGETS)

    print(f"\n{'=' * 96}\nPRE-REGISTERED VERDICT\n{'=' * 96}")
    print(f"  E2 untouched survivors stable across arms : {e2}")
    print(f"  E3 frac_mature rises with the dose        : {e3}")
    n25 = sum(1 for s in TARGETS if rec[s]["d025"])
    mono = all(
        res["d025"]["fd"].get(s, 0.0)
        >= res["d050"]["fd"].get(s, 0.0)
        >= res["d100"]["fd"].get(s, 0.0)
        for s in TARGETS
    )
    print(f"  recovered at m0 x 0.25 : {n25}/4   monotone in dose: {mono}")
    if not e3:
        print(
            "\n  INCONCLUSIVE — lowering the threshold did not raise frac_mature, so the finding is"
        )
        print("  the knob, not the biology.")
    elif n25 >= 2:
        print(
            "\n  MATURATION BOTTLENECK CONFIRMED BY INTERVENTION — lowering m0 rescues the stocks."
        )
        print("  The last link of the chain is causal; the C3 Stage-2 lever is growth against m0.")
    elif n25 == 0:
        print("\n  THRESHOLD NOT SUFFICIENT — frac_mature rises but biomass does not recover, so")
        print("  crossing m0 is NECESSARY BUT NOT SUFFICIENT and something downstream still caps")
        print("  the population. That is a NEW finding, not a null.")
    else:
        print(f"\n  PARTIAL — {n25}/4 recover; report the split without forcing a single reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
