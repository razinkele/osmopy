"""Why is C3 bioen's SPAWNING STOCK low? An exact three-way decomposition of SSB.

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. The chain so far, each step measured rather
than inferred:

  * Abundance is capped on the BIRTH side: mortality is IDENTICAL between arms (1.00-1.01x
    per-capita) while births are 16-24x lower.
  * The birth deficit is a SPAWNER deficit, not a fecundity one: eggs per unit SSB are 1.02-1.24x
    baseline, and eggs track SSB almost exactly (cod_east SSB 0.095x -> eggs 0.101x; herring
    0.188x -> 0.192x; flounder 0.326x -> 0.377x).
  * A gonad-flush mechanism with a near-perfect 24x quantitative fingerprint was REFUTED: its
    `e_net < 0` trigger fires on only 0.04-0.11% of mature-school steps for the cods, and herring
    (0.1157) and sprat (0.1138) trigger identically while one collapses and the other survives.

So: far less spawning biomass, with normal per-spawner fecundity. **Why?**

SSB is a product, and the decomposition is an ALGEBRAIC IDENTITY rather than a hypothesis:

    SSB  =  N_total  x  (N_mature / N_total)  x  (SSB / N_mature)
         =  N_total  x   frac_mature         x   mean_weight_mature

Taking the bioen/baseline ratio of each factor gives an exact multiplicative split of the SSB ratio,
with each factor separately interpretable:

    N_total ratio        are there simply fewer fish at all?
    frac_mature ratio    do fewer of the fish that exist reach the 38 cm maturity length?
    mean_weight ratio    are the mature ones smaller? (cod_west reaches 75 cm under the repaired
                         matrix against 110 cm on baseline, so this term is live)

Because maturity here is LENGTH-based (`species.maturity.m0`, `species.maturity.age` absent), the
second and third factors are coupled: slower realized growth both delays maturity and lowers weight
at age. The decomposition says which one carries the deficit; it does not by itself say why.

Horizon 30 yr, past cod_west's own 20-yr seeding window, so its deficit is visible — at 20 yr
cod_west still read SSB 1.143x because seeding was still active. Production seeding throughout.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Per species, over the final third of the run (years 20-29), abundance-weighted from `step_observer`
using the SAME maturity predicate `_bioen_reproduction` uses for SSB
(`length >= bioen_m0 + bioen_m1 * age`, eggs excluded).

The identity must hold: `N_total_ratio x frac_mature_ratio x mean_w_ratio` must reproduce the
directly computed `SSB_ratio` to within 1%. **This is E1 and it is a hard gate** — if the product
does not reconstruct the whole, the accounting is wrong and no factor may be interpreted.

  (1) mean_w_ratio within [0.8, 1.25] and N_total_ratio ~ SSB_ratio
        -> FEWER FISH. The deficit is upstream in numbers, and the question becomes what removes
           them before maturity given that per-capita mortality is identical.
  (2) N_total_ratio within [0.8, 1.25] and mean_w_ratio ~ SSB_ratio
        -> SMALLER MATURE FISH. The deficit is weight-at-maturity, i.e. realized growth after all,
           but expressed through SSB rather than through survival.
  (3) frac_mature_ratio is the smallest factor
        -> A MATURATION BOTTLENECK: the fish exist and are not small, but fewer of them cross the
           length threshold. Given length-based maturity this points back at growth RATE rather than
           growth CEILING.
  (4) two or more factors contribute comparably
        -> report the split as measured; do not force a single cause.

sprat is carried throughout as the built-in control: it survives, so whichever factor separates it
from the four collapsers is the one that matters.

======================================================================================================

Run: .venv/bin/python scripts/c3_ssb_decomposition.py
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

# 20 yr, one ARM PER PROCESS: a 30-yr two-arm run exceeded the memory envelope (swap 87% full).
# cod_east/herring/flounder seed for 15/12/15 yr, so years 15-19 are already 4-8 yr post-seeding
# for them. cod_west seeds for 20 yr, so its window is still OPEN here and it is flagged, not
# silently averaged in.
N_YEAR = 20
TAIL_FROM = 15
SEED = 42
SEAL, TEMPLATE = "GreySeal", "Cormorant"
FOCUS = ("cod_west", "cod_east", "herring", "flounder", "sprat")
COLLAPSED = ("cod_west", "cod_east", "herring", "flounder")
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"


def write_repaired(src: Path, dst: Path) -> None:
    rows = list(csv.reader(src.open(), delimiter=";"))
    t = [h.strip() for h in rows[0]].index(TEMPLATE)
    out = [rows[0] + [SEAL]] + [r + [repr(float(r[t]))] for r in rows[1:] if r]
    out.append([SEAL] + ["0"] * (len(out[0]) - 1))
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def run_arm(raw, cfg_dir, overlay, n_sp, ndt, m0_arr, m1_arr):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg.pop("population.seeding.year.max", None)
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    write_repaired(cfg_dir / "predation-accessibility.csv", cfg_dir / "acc-rep.csv")
    cfg["predation.accessibility.file"] = "acc-rep.csv"

    # per YEAR x species: total abundance, mature abundance, mature biomass (= SSB), n steps
    tot = np.zeros((N_YEAR, n_sp))
    mat_n = np.zeros((N_YEAR, n_sp))
    mat_b = np.zeros((N_YEAR, n_sp))
    steps = np.zeros(N_YEAR)

    def observer(step, state, grid, config, map_sets):
        # One vectorised bincount pass — a per-species loop with full-length boolean temporaries
        # exhausted memory on an earlier harness.
        yr = min(step // ndt, N_YEAR - 1)
        steps[yr] += 1
        sp = np.asarray(state.species_id)
        abd = np.asarray(state.abundance, dtype=np.float64)
        ok = (abd > 0) & (~np.asarray(state.is_egg)) & (sp < n_sp)
        if not ok.any():
            return
        spo, a = sp[ok], abd[ok]
        w = np.asarray(state.weight, dtype=np.float64)[ok]
        tot[yr] += np.bincount(spo, weights=a, minlength=n_sp)[:n_sp]
        thr = m0_arr[spo] + m1_arr[spo] * (np.asarray(state.age_dt)[ok] / ndt)
        mat = np.asarray(state.length, dtype=np.float64)[ok] >= thr
        if not mat.any():
            return
        mat_n[yr] += np.bincount(spo[mat], weights=a[mat], minlength=n_sp)[:n_sp]
        mat_b[yr] += np.bincount(spo[mat], weights=a[mat] * w[mat], minlength=n_sp)[:n_sp]

    from osmose.engine.simulate import simulate

    ec, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, SEED)
    simulate(
        ec, grid, rng, movement_rngs=mv, mortality_rngs=mo, output_dir=None, step_observer=observer
    )
    return {"tot": tot, "mat_n": mat_n, "mat_b": mat_b, "steps": steps}


def main() -> int:
    warnings.simplefilter("ignore")
    arm_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    tmp = Path(tempfile.mkdtemp(prefix="c3_ssbdec_"))
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

    print(
        f"{N_YEAR} yr, production seeding, seed {SEED}, repaired matrix; "
        f"decomposing years {TAIL_FROM}-{N_YEAR - 1}"
    )
    if arm_arg in ("baseline", "bioen"):
        # ONE arm per process, results to JSON — halves peak memory.
        ov = None if arm_arg == "baseline" else overlay
        print(f"  running {arm_arg} ...", flush=True)
        r = run_arm(raw, cf.parent, ov, n_sp, ndt, m0_arr, m1_arr)
        Path(out_arg).write_text(json.dumps({k: v.tolist() for k, v in r.items()}))
        print(f"  wrote {out_arg}")
        return 0

    res = {}
    for tag, fn in (("baseline", "_base.json"), ("bioen", "_bioen.json")):
        pth = Path(sys.argv[0]).parent / fn
        if not pth.exists():
            print(f"  MISSING {pth} — run:  python {sys.argv[0]} {tag} {pth}")
            return 1
        res[tag] = {k: np.asarray(v) for k, v in json.loads(pth.read_text()).items()}

    w = slice(TAIL_FROM, N_YEAR)

    def facs(tag, i):
        r = res[tag]
        t, mn, mb = r["tot"][w, i].sum(), r["mat_n"][w, i].sum(), r["mat_b"][w, i].sum()
        return {
            "N_total": t,
            "frac_mature": mn / t if t > 0 else 0.0,
            "mean_w": mb / mn if mn > 0 else 0.0,
            "SSB": mb,
            "N_mature": mn,
        }

    print(
        f"\n{'=' * 104}\nSSB = N_total x frac_mature x mean_weight_mature  (bioen / baseline ratios)"
    )
    print("=" * 104)
    hdr = (
        f"{'species':<13}{'SSB ratio':>11}{'N_total':>10}{'frac_mat':>10}{'mean_w':>10}"
        f"{'product':>10}{'closes?':>9}   dominant factor"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for s in FOCUS:
        i = names.index(s)
        b, x = facs("baseline", i), facs("bioen", i)
        rt = {
            k: (x[k] / b[k] if b[k] > 0 else float("nan"))
            for k in ("N_total", "frac_mature", "mean_w", "SSB")
        }
        prod = rt["N_total"] * rt["frac_mature"] * rt["mean_w"]
        closes = abs(prod - rt["SSB"]) <= 0.01 * max(rt["SSB"], 1e-12)
        three = {k: rt[k] for k in ("N_total", "frac_mature", "mean_w")}
        dom = min(three, key=three.get)
        rows[s] = (rt, prod, closes, dom)
        print(
            f"{s:<13}{rt['SSB']:>11.4f}{rt['N_total']:>10.4f}{rt['frac_mature']:>10.4f}"
            f"{rt['mean_w']:>10.4f}{prod:>10.4f}{closes!s:>9}   {dom}"
        )

    print(
        f"\n{'=' * 104}\nABSOLUTE VALUES (years {TAIL_FROM}-{N_YEAR - 1}, step-summed)\n{'=' * 104}"
    )
    print(
        f"{'species':<13}{'arm':<11}{'N_total':>14}{'N_mature':>14}{'frac_mat':>10}"
        f"{'mean w (g)':>13}"
    )
    for s in FOCUS:
        i = names.index(s)
        for tag in ("baseline", "bioen"):
            f = facs(tag, i)
            print(
                f"{s if tag == 'baseline' else '':<13}{tag:<11}{f['N_total']:>14.4e}"
                f"{f['N_mature']:>14.4e}{f['frac_mature']:>10.4f}{f['mean_w'] * 1e6:>13.2f}"
            )
        print()

    print(f"{'=' * 104}\nPRE-REGISTERED VERDICT\n{'=' * 104}")
    e1 = all(rows[s][2] for s in FOCUS)
    print(f"  E1 the identity closes to within 1% for every species : {e1}")
    if not e1:
        print("\n  INCONCLUSIVE — the product does not reconstruct SSB, so the accounting is wrong")
        print("  and no factor may be interpreted.")
        return 0
    for s in COLLAPSED:
        rt, _prod, _c, dom = rows[s]
        parts = {k: rt[k] for k in ("N_total", "frac_mature", "mean_w")}
        near = [k for k, v in parts.items() if 0.8 <= v <= 1.25]
        if dom == "mean_w" and "N_total" in near:
            verdict = "SMALLER MATURE FISH — weight-at-maturity carries it"
        elif dom == "N_total" and "mean_w" in near:
            verdict = "FEWER FISH — the deficit is upstream in numbers"
        elif dom == "frac_mature":
            verdict = "MATURATION BOTTLENECK — they exist and are not small, but fewer cross 38 cm"
        else:
            verdict = f"SPLIT — {', '.join(f'{k} {v:.3f}' for k, v in parts.items())}"
        print(f"  {s:<11} SSB {rt['SSB']:.4f}x  -> {verdict}")
    rt_s = rows["sprat"][0]
    print(
        f"\n  sprat (SURVIVES) SSB {rt_s['SSB']:.4f}x: N_total {rt_s['N_total']:.3f}, "
        f"frac_mat {rt_s['frac_mature']:.3f}, mean_w {rt_s['mean_w']:.3f}"
    )
    print("  -> whichever factor separates sprat from the four collapsers is the one that matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
