"""Why does cod_east read 0.0 t at m0 x 0.25 while 67 % of it is "mature"?

THE OPEN QUESTION (`docs/baltic_c3_bioen_stage1_2026-09-05.md`, m0-lever section): the m0 dose
ladder drove cod_east's `frac_mature` to 0.67 at x 0.25, yet its biomass stayed at 0.0 t, and the
cods collapsed FASTER than at x 1.00. The doc reasons: "With eggs tracking SSB at 1.02-1.24x, more
spawners should mean more eggs. That they do not translate into biomass points at egg->recruit
survival, which no test here has yet isolated."

That inference has THREE unexamined premises, and reading the engine shows each can fail on its own.
This script measures all of them in one run rather than assuming any.

  H1  WEIGHT TAX.  SSB = N_total x frac_mature x mean_weight_mature. The same commit found that
      maturing diverts `E_net` from somatic growth to gonad (`rho`). 67 % of much LIGHTER fish is
      still ~0 SSB. Then the deficit is mean weight and egg->recruit is never reached.

  H2  EGG->RECRUIT (the doc's).  SSB really does rise, eggs follow, and the loss is between egg and
      age-1.

  H3  DISJOINT POPULATIONS -- the two numbers never described the same fish.  `frac_mature` is
      computed over EVERY non-egg school against `length >= bioen_m0`; at x 0.25 cod_east's bar is
      22.0 -> 5.5 cm. `biomass()` applies `output.cutoff.age` = 0.5 yr for every Baltic species
      (`_collect_biomass_abundance`, `simulate.py:1067-1072`) and so excludes young-of-year
      ENTIRELY. If the "mature" fish are age-0, "67 % mature yet 0.0 t" is not a paradox and there
      is no egg->recruit puzzle to explain. Same trap CLAUDE.md records for `by_age` bin 0.

  H4  SEEDING-CRUTCH DISPLACEMENT -- the lever switched off life-support.  `_bioen_reproduction`
      (`simulate.py:864-868`) substitutes phantom SSB when, and ONLY when, SSB is EXACTLY zero:

          if ssb[sp] == 0.0 and step < seeding_max_step[sp] and seeding_biomass[sp] > 0:
              ssb[sp] = float(config.seeding_biomass[sp])

      and eggs are strictly proportional to whatever `ssb` then holds. cod_east's
      `population.seeding.biomass.sp8` is 100 000 t and, with `population.seeding.year.max` popped
      (these scripts force production seeding), the window defaults to `lifespan * ndt` =
      15 yr of a 25 yr run (`config.py:537-542`). So at x 1.00, where SSB is exactly 0, cod_east
      was fed eggs from 100 000 t of phantom biomass for 15 years. At x 0.25 a handful of 5.5 cm
      fish make SSB > 0 but minuscule -- which DISABLES the fallback and drops egg supply to the
      tiny real SSB. Lowering m0 would then REDUCE recruitment for 15 years by turning off the
      crutch, predicting exactly the observed "collapses faster" WITHOUT any growth tax.

H4 matters most because it is the only one that explains "faster", and if it fires the m0 lever did
not cleanly test maturation at all -- the intervention confounded itself.

WHAT IS MEASURED, and why each quantity is here. Per year x species, from a `step_observer` plus a
wrapper around `regulate_recruitment` (the same two instruments `c3_ssb_decomposition.py` and
`c3_abundance_balance.py` already validated):

  tot, mat_n, mat_b       N_total, N_mature, SSB -- closes the identity (H1). Maturity is the bioen
                          conjunction `length >= bioen_m0 + bioen_m1*age`, eggs excluded, matching
                          `_bioen_reproduction:842-845` exactly, and weight is per-individual in
                          TONNES excluding gonad (`state.gonad_weight` is a separate array), which
                          is the same product the engine forms.
  mat_n_yoy, mat_b_yoy    the mature fish BELOW the 0.5 yr cutoff (H3).
  n_cut, b_cut            everything at or above it -- `age_dt/ndt >= cutoff`, the exact mask
                          `_collect_biomass_abundance` uses. This is what `biomass()` reports.
  n_age1                  non-egg abundance in [1,2) yr -- recruits, for the egg->recruit ratio (H2)
                          computed WITHOUT the egg-contaminated `by_age` bin 0 the repo warns about.
  deaths_yoy/_adult       `state.n_dead` by cause, split at the same cutoff. Without this the run
                          shows THAT age-0 fish vanish but not WHY -- the exact error of reading a
                          mechanism off a distribution.
  eggs, ssb_in, seeded    straight from the regulator's own arguments (H4): the eggs it returns, the
                          SSB it was handed, and `seeded_this_step` -- so whether the crutch fired
                          is READ, not inferred.

PRE-REGISTERED READINGS (fixed here BEFORE the run; the verdicts are independent and MAY co-fire --
H1 and H3 especially, since age-0 fish are both young and light). All ratios for cod_east.

  H4 FIRES  if in the seeding window `seeded(d100)` > 0 in years where `seeded(d025)` == 0
            AND eggs(d025)/eggs(d100) < 0.5 across those years.
            -> the lever disabled life-support; "collapses faster" is an artifact of the
               intervention, NOT the rho growth tax the doc attributed it to.
  H3 FIRES  if mat_b_yoy/mat_b >= 0.90 at d025 AND the d025 b_cut tail mean is < 1 % of base's.
            -> the 67 % and the 0.0 t count disjoint populations; the paradox dissolves.
  H1 FIRES  if mean_w_mature(d025)/mean_w_mature(base) <= 0.25 AND SSB(d025)/SSB(base) < 0.05
            while frac_mature >= 0.6.  -> the deficit is weight-at-maturity.
  H2 FIRES  if SSB(d025)/SSB(base) >= 0.20 AND eggs(d025)/eggs(base) >= 0.20
            AND (n_age1/eggs)(d025) <= 0.10 x baseline.  -> egg->recruit survival, as the doc says.
  NONE      -> report the three factors and the egg/recruit ratio and state plainly what is still
               not isolated. Do not narrate a mechanism the numbers do not carry.

FALSIFIERS -- if any fails, nothing below it is readable.
  E1 the observer fires in every year of every arm.
  E2 the regulator wrapper fires and records non-zero eggs.
  E3 INSTRUMENT VALIDITY, load-bearing: this script's own `b_cut` per-step mean reproduces
     `res.biomass()` over the tail to within 5 %. If it does not, `b_cut` is not the series the
     doc's "0.0 t" came from and no comparison against that number means anything.
  E4 `bioen_m0` read back from a constructed EngineConfig differs per arm at exactly the four
     targets and nowhere else (the lever engaged, and only where intended).
  E5 STOCK FLOOR: 67 % of what? If tot(d025) < 10 % of tot(d100) over the same years, the fraction
     is noise on a dying stock and NO hypothesis is being tested -- report "insufficient stock".

Run:  .venv/bin/python scripts/c3_cod_east_egg_pathway.py <arm> <out.json>   arm in base|d100|d050|d025
      .venv/bin/python scripts/c3_cod_east_egg_pathway.py                    (combine + verdicts)
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
# sys.path[0] is this script's directory, so `osmose` would otherwise resolve through the editable
# install to whatever branch the MAIN checkout is on -- the trap CLAUDE.md records.
sys.path.insert(0, str(ROOT))

import osmose.engine.processes.reproduction as repro_mod
from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.state import MortalityCause

# OSMOSE_C3_SMOKE_YEARS shortens the run to smoke-test the instrument (E1-E4 wiring) without
# paying for the full ladder. It is NOT a reporting mode: the pre-registered readings are all
# defined on the 25 yr horizon, and the 15 yr seeding window H4 turns on needs the full run.
N_YEAR = int(__import__("os").environ.get("OSMOSE_C3_SMOKE_YEARS", "25"))
TAIL = 8
SEED = 42
SEAL, TEMPLATE = "GreySeal", "Cormorant"
TARGETS = {"cod_west": 0, "cod_east": 8, "herring": 1, "flounder": 3}
DOSES = {"base": None, "d100": 1.00, "d050": 0.50, "d025": 0.25}
CAUSES = [c.name for c in MortalityCause]
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"

_ORIG_REGULATE = repro_mod.regulate_recruitment


def repaired(src: Path, dst: Path) -> None:
    """Add the missing GreySeal column (copied from Cormorant) + an all-zero prey row.

    Without this the seal eats every focal species at accessibility 1.0 via the `-1` path.
    """
    rows = list(csv.reader(src.open(), delimiter=";"))
    t = [h.strip() for h in rows[0]].index(TEMPLATE)
    out = [rows[0] + [SEAL]] + [r + [repr(float(r[t]))] for r in rows[1:] if r]
    out.append([SEAL] + ["0"] * (len(out[0]) - 1))
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def build_cfg(raw, cfg_dir, overlay, dose):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg.pop("population.seeding.year.max", None)  # production seeding
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
    from osmose.engine.config import EngineConfig

    ec_probe = EngineConfig.from_dict(dict(cfg))

    def _arr(name):
        v = getattr(ec_probe, name, None)
        return np.asarray(v if v is not None else np.zeros(n_sp), dtype=np.float64)

    m0, m1 = _arr("bioen_m0"), _arr("bioen_m1")
    cutoff = _arr("output_cutoff_age")[:n_sp].copy()
    if not np.isfinite(cutoff).any() or cutoff.max() <= 0:
        cutoff = np.full(n_sp, 0.5)

    z = lambda: np.zeros((N_YEAR, n_sp))
    acc = {
        k: z()
        for k in (
            "tot",
            "mat_n",
            "mat_b",
            "mat_n_yoy",
            "mat_b_yoy",
            "n_cut",
            "b_cut",
            "n_age1",
            "eggs",
            "ssb_in",
            "n_linear",
            "seeded",
        )
    }
    deaths = {
        "yoy": np.zeros((N_YEAR, n_sp, len(CAUSES))),
        "adult": np.zeros((N_YEAR, n_sp, len(CAUSES))),
    }
    steps = np.zeros(N_YEAR)
    n_calls = [0]

    def observer(step, state, grid, config, map_sets):
        # One vectorised bincount pass per quantity: a per-species loop with full-length boolean
        # temporaries exhausted memory on an earlier harness.
        yr = min(step // ndt, N_YEAR - 1)
        steps[yr] += 1
        sp = np.asarray(state.species_id)
        abd = np.asarray(state.abundance, dtype=np.float64)
        age_y = np.asarray(state.age_dt, dtype=np.float64) / ndt
        live = (abd > 0) & (sp < n_sp)
        if not live.any():
            return
        nonegg = live & (~np.asarray(state.is_egg))

        def bc(mask, w=None):
            if not mask.any():
                return np.zeros(n_sp)
            return np.bincount(sp[mask], weights=None if w is None else w[mask], minlength=n_sp)[
                :n_sp
            ]

        w = np.asarray(state.weight, dtype=np.float64)  # tonnes/fish, gonad EXCLUDED
        aw = abd * w
        acc["tot"][yr] += bc(nonegg, abd)

        thr = np.full(len(sp), np.inf)
        thr[live] = m0[sp[live]] + m1[sp[live]] * age_y[live]
        mat = nonegg & (np.asarray(state.length, dtype=np.float64) >= thr)
        acc["mat_n"][yr] += bc(mat, abd)
        acc["mat_b"][yr] += bc(mat, aw)

        # H3: the mature fish that `biomass()` cannot see, and the ones it can.
        below = np.zeros(len(sp), dtype=bool)
        below[live] = age_y[live] < cutoff[sp[live]]
        acc["mat_n_yoy"][yr] += bc(mat & below, abd)
        acc["mat_b_yoy"][yr] += bc(mat & below, aw)
        # E3: the EXACT mask of `_collect_biomass_abundance` -- `age_dt/ndt >= cutoff`, eggs
        # included in principle but excluded in fact since their age is 0.
        acc["n_cut"][yr] += bc(live & ~below, abd)
        acc["b_cut"][yr] += bc(live & ~below, aw)
        acc["n_age1"][yr] += bc(nonegg & (age_y >= 1.0) & (age_y < 2.0), abd)

        nd = np.asarray(state.n_dead, dtype=np.float64)
        if nd.size:
            for c in range(len(CAUSES)):
                deaths["yoy"][yr, :, c] += bc(live & below, nd[:, c])
                deaths["adult"][yr, :, c] += bc(live & ~below, nd[:, c])

    def wrapper(n_eggs_linear, ssb_in, seeded_this_step, config, step):
        out = _ORIG_REGULATE(n_eggs_linear, ssb_in, seeded_this_step, config, step)
        yr = min(step // ndt, N_YEAR - 1)
        n_calls[0] += 1
        acc["eggs"][yr] += np.asarray(out, dtype=np.float64)[:n_sp]
        acc["ssb_in"][yr] += np.asarray(ssb_in, dtype=np.float64)[:n_sp]
        acc["n_linear"][yr] += np.asarray(n_eggs_linear, dtype=np.float64)[:n_sp]
        acc["seeded"][yr] += np.asarray(seeded_this_step, dtype=np.float64)[:n_sp]
        return out

    from osmose.engine.simulate import simulate
    from osmose.results import OsmoseResults

    ec, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, SEED)
    repro_mod.regulate_recruitment = wrapper
    try:
        outs = simulate(
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

    # E3 reference: the published series, read exactly as the m0 lever test read it.
    import pandas as pd

    res = OsmoseResults.from_outputs(outs, ec, grid)
    bio = res.biomass()
    tl = max(1, int(len(bio) * TAIL / N_YEAR))
    fd = {}
    for s in [c for c in bio.columns if str(c).strip().lower() not in ("time", "year", "step")]:
        v = pd.to_numeric(bio[s], errors="coerce").to_numpy(dtype=float)[-tl:]
        if v.size and np.isfinite(v).any():
            fd[str(s)] = float(np.nanmean(v))

    out = {k: v.tolist() for k, v in acc.items()}
    out.update(
        deaths_yoy=deaths["yoy"].tolist(),
        deaths_adult=deaths["adult"].tolist(),
        steps=steps.tolist(),
        m0=m0.tolist(),
        cutoff=cutoff.tolist(),
        fd=fd,
        n_calls=n_calls[0],
    )
    return out


def main() -> int:
    warnings.simplefilter("ignore")
    arm = sys.argv[1] if len(sys.argv) > 1 else None
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    tmp = Path(tempfile.mkdtemp(prefix="c3_eggpath_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    ndt = int(raw["simulation.time.ndtperyear"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    if arm in DOSES:
        cfg = build_cfg(raw, cf.parent, overlay, DOSES[arm])
        r = run_arm(cfg, n_sp, ndt)
        r["names"] = names
        r["arm"] = arm
        Path(dst).write_text(json.dumps(r))
        print(f"{arm}: wrote {dst}  (regulator calls {r['n_calls']}, steps {sum(r['steps']):.0f})")
        return 0

    return combine(Path(dst) if dst else Path("."), names, ndt)


def _per_step(arm, key, i):
    """Step-summed accumulator -> per-step mean per year (steps differ only if a run is cut short)."""
    v = np.asarray(arm[key])[:, i]
    s = np.asarray(arm["steps"])
    return np.divide(v, s, out=np.zeros_like(v), where=s > 0)


def combine(where: Path, names, ndt) -> int:
    arms = {}
    for a in DOSES:
        p = where / f"eggpath_{a}.json" if where.is_dir() else where
        if p.exists():
            arms[a] = json.loads(p.read_text())
    if not arms:
        print(f"no arm json files found under {where}")
        return 2
    names = arms[next(iter(arms))]["names"]
    i = names.index("cod_east")
    tail = slice(N_YEAR - TAIL, N_YEAR)

    print(f"\n{'=' * 100}\nC3 cod_east: why 0.0 t at m0 x 0.25 while 67 % is 'mature'")
    print(
        f"arms present: {', '.join(arms)}   cod_east = sp{i}   cutoff = "
        f"{arms[next(iter(arms))]['cutoff'][i]} yr\n{'=' * 100}"
    )

    # ---- FALSIFIERS -------------------------------------------------------------------------
    print("\nFALSIFIERS")
    e1 = all(np.asarray(a["steps"]).min() > 0 for a in arms.values())
    e2 = all(a["n_calls"] > 0 and np.asarray(a["eggs"]).sum() > 0 for a in arms.values())
    print(f"  E1 observer fired in every year, every arm                  : {e1}")
    print(f"  E2 regulator wrapper fired, non-zero eggs                   : {e2}")
    e3ok = True
    for k, a in arms.items():
        mine = float(np.mean(_per_step(a, "b_cut", i)[tail]))
        pub = float(a["fd"].get("cod_east", float("nan")))
        rel = abs(mine - pub) / pub if pub else (0.0 if mine == 0 else float("inf"))
        ok = rel <= 0.05 or (mine < 1e-9 and pub < 1e-9)
        e3ok &= ok
        print(
            f"     E3 {k:<5} b_cut tail mean {mine:>14.4f} t  vs biomass() {pub:>14.4f} t"
            f"   rel {rel:>7.2%}  {'OK' if ok else 'MISMATCH'}"
        )
    print(f"  E3 instrument reproduces the published series (<=5 %)       : {e3ok}")
    m0s = {k: a["m0"][i] for k, a in arms.items()}
    print(f"  E4 cod_east bioen_m0 per arm                                : {m0s}")
    e5 = True
    if "d100" in arms and "d025" in arms:
        t100 = float(np.mean(_per_step(arms["d100"], "tot", i)[tail]))
        t025 = float(np.mean(_per_step(arms["d025"], "tot", i)[tail]))
        frac = t025 / t100 if t100 else float("inf")
        e5 = frac >= 0.10
        print(
            f"  E5 stock floor tot(d025)/tot(d100) over tail = {frac:>8.3f}     : "
            f"{'OK' if e5 else 'INSUFFICIENT STOCK — fraction is noise on a dying stock'}"
        )
    if not (e1 and e2 and e3ok):
        print("\n  A falsifier failed. Nothing below is readable.")
        return 1

    # ---- SSB identity -----------------------------------------------------------------------
    print(
        f"\n{'-' * 100}\nSSB = N_total x frac_mature x mean_w_mature   (per-step means over the "
        f"final {TAIL} yr)\n{'-' * 100}"
    )
    # CAVEAT, and it is not cosmetic: with bioen OFF `bioen_m0` is absent and defaults to 0.0
    # (`config.py:2509`), so the observer's `length >= m0` marks EVERY non-egg school mature and
    # `mat_n` collapses onto `tot`. The bioen-off engine's real maturity is `config.maturity_size`,
    # a DIFFERENT key the observer never reads. So base's `mean_w_mature` is a whole-population
    # mean, not a spawner mean, and dividing a spawner weight by it would manufacture a growth tax.
    # Two arm-comparable quantities are used instead:
    #   ssb_engine  the SSB the engine itself formed under ITS OWN maturity rule for that arm,
    #               read straight off the regulator's argument (phantom when `seeded` > 0).
    #   mean_w_cut  b_cut / n_cut -- mean weight per fish above the output cutoff, which needs no
    #               maturity definition at all and so is directly comparable across arms.
    base_degenerate = {k: a["m0"][i] == 0.0 for k, a in arms.items()}
    print(
        f"  {'arm':<6}{'N_total':>13}{'frac_mat':>10}{'mean_w (t)':>12}{'SSB_obs':>12}"
        f"{'SSB_eng':>12}{'mean_w_cut':>12}{'b_cut (t)':>13}{'biomass()':>13}"
    )
    ident = {}
    for k, a in arms.items():
        n = float(np.mean(_per_step(a, "tot", i)[tail]))
        mn = float(np.mean(_per_step(a, "mat_n", i)[tail]))
        mb = float(np.mean(_per_step(a, "mat_b", i)[tail]))
        bc = float(np.mean(_per_step(a, "b_cut", i)[tail]))
        nc = float(np.mean(_per_step(a, "n_cut", i)[tail]))
        se = float(np.mean(_per_step(a, "ssb_in", i)[tail]))
        fm = mn / n if n else 0.0
        mw = mb / mn if mn else 0.0
        mwc = bc / nc if nc else 0.0
        ident[k] = {"n": n, "fm": fm, "mw": mw, "ssb": mb, "bcut": bc, "ssb_eng": se, "mwc": mwc}
        flag = "  <- m0=0, 'mature' = all fish" if base_degenerate[k] else ""
        print(
            f"  {k:<6}{n:>13.4e}{fm:>10.4f}{mw:>12.4e}{mb:>12.2f}{se:>12.2f}{mwc:>12.4e}"
            f"{bc:>13.2f}{a['fd'].get('cod_east', float('nan')):>13.2f}{flag}"
        )

    # ---- H4: the seeding crutch -------------------------------------------------------------
    print(
        f"\n{'-' * 100}\nH4  seeding crutch: eggs are proportional to the SSB handed to the "
        f"regulator,\n    and that SSB is PHANTOM whenever real SSB is exactly 0 inside the "
        f"window\n{'-' * 100}"
    )
    print(
        f"  {'arm':<6}{'steps seeded':>14}{'yrs seeded':>12}{'eggs (total)':>15}"
        f"{'ssb_in (mean t)':>18}{'real SSB (t)':>15}"
    )
    h4 = {}
    for k, a in arms.items():
        sd = np.asarray(a["seeded"])[:, i]
        eggs = float(np.asarray(a["eggs"])[:, i].sum())
        si = float(np.mean(_per_step(a, "ssb_in", i)))
        rs = float(np.mean(_per_step(a, "mat_b", i)))
        h4[k] = {"seeded": sd, "eggs": eggs}
        print(
            f"  {k:<6}{sd.sum():>14.0f}{int((sd > 0).sum()):>12}{eggs:>15.4e}{si:>18.4f}{rs:>15.4f}"
        )
    if "d100" in arms and "d025" in arms:
        s100, s025 = h4["d100"]["seeded"], h4["d025"]["seeded"]
        yrs = np.where((s100 > 0) & (s025 == 0))[0]
        print(f"\n  years where d100 was crutched but d025 was NOT: {yrs.tolist()}")
        if yrs.size:
            e100 = float(np.asarray(arms["d100"]["eggs"])[yrs, i].sum())
            e025 = float(np.asarray(arms["d025"]["eggs"])[yrs, i].sum())
            r = e025 / e100 if e100 else float("inf")
            print(f"  eggs over those years:  d100 {e100:.4e}   d025 {e025:.4e}   ratio {r:.4f}")
            h4_fires = r < 0.5
        else:
            h4_fires = False
            print("  d025 was crutched in every year d100 was -> the lever did not displace it")
    else:
        h4_fires = False

    # ---- H3: are the 'mature' fish below the output cutoff? ---------------------------------
    print(
        f"\n{'-' * 100}\nH3  do the two numbers count the same fish? (mature biomass below the "
        f"0.5 yr cutoff)\n{'-' * 100}"
    )
    print(
        f"  {'arm':<6}{'mat_b (t)':>14}{'mat_b_yoy (t)':>16}{'yoy share':>12}"
        f"{'mat_n':>14}{'mat_n_yoy':>14}"
    )
    h3 = {}
    for k, a in arms.items():
        mb = float(np.mean(_per_step(a, "mat_b", i)[tail]))
        my = float(np.mean(_per_step(a, "mat_b_yoy", i)[tail]))
        mn = float(np.mean(_per_step(a, "mat_n", i)[tail]))
        mny = float(np.mean(_per_step(a, "mat_n_yoy", i)[tail]))
        sh = my / mb if mb else 0.0
        h3[k] = sh
        print(f"  {k:<6}{mb:>14.4f}{my:>16.4f}{sh:>12.4f}{mn:>14.4e}{mny:>14.4e}")

    # ---- egg -> recruit ---------------------------------------------------------------------
    print(
        f"\n{'-' * 100}\nH2  egg -> recruit: age-1 abundance per egg (eggs from year y-1)"
        f"\n{'-' * 100}"
    )
    print(f"  {'arm':<6}{'eggs/yr':>15}{'age-1 N':>15}{'recruits/egg':>15}")
    sr = {}
    for k, a in arms.items():
        eg = np.asarray(a["eggs"])[:, i]
        a1 = _per_step(a, "n_age1", i)
        w = slice(max(1, N_YEAR - TAIL), N_YEAR)
        e = float(np.mean(eg[w.start - 1 : w.stop - 1]))
        n1 = float(np.mean(a1[w]))
        sr[k] = n1 / e if e else 0.0
        print(f"  {k:<6}{e:>15.4e}{n1:>15.4e}{sr[k]:>15.4e}")

    # ---- deaths by cause, split at the cutoff ------------------------------------------------
    print(
        f"\n{'-' * 100}\ndeaths by cause over the final {TAIL} yr (young-of-year | at or above "
        f"cutoff)\n{'-' * 100}"
    )
    print(f"  {'arm':<6}{'stage':<7}" + "".join(f"{c[:9]:>13}" for c in CAUSES))
    for k, a in arms.items():
        for lab, key in (("yoy", "deaths_yoy"), ("adult", "deaths_adult")):
            d = np.asarray(a[key])[tail, i, :].sum(axis=0)
            print(f"  {k:<6}{lab:<7}" + "".join(f"{x:>13.3e}" for x in d))

    # ---- pre-registered verdicts -------------------------------------------------------------
    print(
        f"\n{'=' * 100}\nPRE-REGISTERED VERDICTS (fixed before the run; may co-fire)\n{'=' * 100}"
    )
    # E5 gates H1/H2/H3 because all three are read off the TAIL. When the stock floor fails the
    # tail holds ~1e-22 fish and every ratio through it is noise -- the pre-registration says so, so
    # the printout must say so too rather than reporting a FIRES a later reader would quote.
    b, d = ident.get("base"), ident.get("d025")
    if not e5:
        print(
            "  H1/H2/H3 are TAIL readings and E5 FAILED -> NOT READABLE. The tail holds ~1e-22 "
            "fish;\n     no hypothesis about weight, egg->recruit or the cutoff is under test "
            "there. Read the\n     live years instead (the egg->recruit table above uses yr1->yr2"
            ", where fish existed)."
        )
        if b and d:
            print(
                f"     for the record only, NOT a verdict: frac_mature(d025) = {d['fm']:.4f} over "
                f"N = {d['n']:.4e} fish -- this IS the doc's '67 % maturation'."
            )
    elif b and d:
        # SSB vs base uses the ENGINE's own per-arm SSB, and mean weight uses the definition-free
        # per-fish weight above the cutoff -- see the caveat above the identity table.
        ssb_r = d["ssb_eng"] / b["ssb_eng"] if b["ssb_eng"] else float("inf")
        mw_r = d["mwc"] / b["mwc"] if b["mwc"] else float("inf")
        h1 = mw_r <= 0.25 and ssb_r < 0.05 and d["fm"] >= 0.6
        print(
            f"  H1 weight tax        mean_w_cut ratio {mw_r:.4f} (<=0.25), SSB_eng ratio "
            f"{ssb_r:.4f} (<0.05), frac_mat {d['fm']:.4f} (>=0.6)  -> {'FIRES' if h1 else 'no'}"
        )
        if "d100" in ident:
            p = ident["d100"]
            print(
                f"     within-bioen (d025 vs d100, same maturity machinery): mean_w_mature "
                f"{d['mw'] / p['mw'] if p['mw'] else float('nan'):.4f}x, "
                f"mean_w_cut {d['mwc'] / p['mwc'] if p['mwc'] else float('nan'):.4f}x, "
                f"SSB_obs {d['ssb'] / p['ssb'] if p['ssb'] else float('nan'):.4f}x"
            )
        eg_r = np.asarray(arms["d025"]["eggs"])[:, i].sum() / max(
            np.asarray(arms["base"]["eggs"])[:, i].sum(), 1e-300
        )
        rec_r = sr.get("d025", 0.0) / sr["base"] if sr.get("base") else float("inf")
        h2 = ssb_r >= 0.20 and eg_r >= 0.20 and rec_r <= 0.10
        print(
            f"  H2 egg->recruit      SSB ratio {ssb_r:.4f} (>=0.20), eggs ratio {eg_r:.4f} "
            f"(>=0.20), recruits/egg ratio {rec_r:.4f} (<=0.10)  -> {'FIRES' if h2 else 'no'}"
        )
        h3f = h3.get("d025", 0.0) >= 0.90 and d["bcut"] < 0.01 * max(b["bcut"], 1e-300)
        print(
            f"  H3 disjoint pops     yoy share of mature biomass {h3.get('d025', 0):.4f} "
            f"(>=0.90)  -> {'FIRES' if h3f else 'no'}"
        )
    print(f"  H4 seeding crutch    -> {'FIRES' if h4_fires else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
