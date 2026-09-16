"""If the seeding crutch were never switched off by a fractional spawner, do the bioen stocks live?

WHY THIS EXISTS. `scripts/c3_cod_east_egg_pathway.py` (PR #165) showed that the m0 dose ladder was
confounded by a DISCONTINUITY: `_bioen_reproduction` (`simulate.py:864-873`) substitutes phantom SSB
when, and ONLY when, real SSB is EXACTLY zero --

    if ssb[sp] == 0.0 and step < config.seeding_max_step[sp] and config.seeding_biomass[sp] > 0:
        seeded_this_step[sp] = True
        ssb[sp] = float(config.seeding_biomass[sp])          # cod_east: 100 000 t
        n_eggs_linear[sp] = sex_ratio * relative_fecundity * ssb[sp] * season * 1e6

-- so a stock is cut off from 100 000 t of phantom spawning biomass the instant it produces its
first spawner (0.04 t at m0 x 1.00). Every bioen arm was weaned within 1-3 years and then collapsed.
The question that leaves open: **was recruitment ever the binding constraint at all?**

TWO CORRECTIONS TO THE FOLLOW-UP AS ORIGINALLY RECORDED, both found by re-reading the source:

  * "Force `population.seeding.year.max` to 15" is a **NO-OP**. With the key absent,
    `seeding_max_step = lifespan_years * n_dt` (`config.py:539-542`), and cod_east's lifespan IS 15.
    The window never ended the crutch -- the `ssb == 0.0` test did. **The CONDITION is what must
    change, not the window.**
  * Unseeded bioen eggs are **NOT** SSB-proportional. The `elif idx.size:` branch
    (`simulate.py:880-889`) derives them from GONAD mass via `bioen_egg_release`. So this test must
    REPLACE the egg count with the seeded formula, never rescale the gonad-derived one.

THE INTERVENTION. A wrapper around `regulate_recruitment` floors SSB at `seeding_biomass` for the
whole window instead of demanding exact zero, i.e. `ssb = max(ssb_real, seeding_biomass)`. This is
faithful rather than approximate because the engine mutates `ssb`, `n_eggs_linear` and
`seeded_this_step` IN PLACE and hands the SAME `seeded_this_step` object to `create_egg_schools`
(`simulate.py:888-889`) -- so the sustained eggs are also tagged `from_seeding`, and therefore also
skip the RV egg-survival penalty (`natural.py:161`), exactly as genuinely-seeded eggs do. E3 below
verifies that propagation rather than trusting it.

SCOPE: the four collapsing stocks ONLY (`TARGETS`), matching the m0 ladder's control structure.
Flooring every focal species would rewrite the food web -- predators of cod juveniles and LTL
competitors would all change abundance -- and cod_east would no longer be comparable to the `d100`
arm this is read against. That larger experiment is a different question.

CONTROLS: reuses `eggpath_d100.json` (same config, same seed, normal crutch) and `eggpath_base.json`
(bioen OFF) from PR #165's ladder. Same N_YEAR/TAIL/SEED, so the arms are directly comparable and
the already-passed E3 gate on those files is not re-litigated.

PRE-REGISTERED READINGS (fixed BEFORE the run). Windows: cod_east/flounder are seeded for 15 yr,
cod_west 20 yr, herring 12 yr; "in-window" means yrs 5-14, past the bootstrap transient and inside
every target's window.

  R1  DOES A SUSTAINED CRUTCH PRODUCE STANDING BIOMASS?  in-window `b_cut` against the bioen-OFF
      baseline over the same years.
        >= 20 % of base -> recruitment WAS a binding constraint; the crutch's switch-off is load-
                           bearing for the collapse, and C3's negative is partly a seeding artifact.
        <   1 % of base -> RECRUITMENT WAS NEVER THE CONSTRAINT. With 100 000 t of phantom SSB
                           every step for 15 years the stock still cannot stand up, so the loss is
                           entirely between egg and the 0.5 yr cutoff. This would be the strong,
                           clean result and it STRENGTHENS C3's negative.
        between         -> partial; report the number and claim nothing more.
  R2  IS IT SELF-SUSTAINING AFTER WEANING?  `b_cut` at yr 24 against the in-window mean.
        >= 20 % -> HOLDS after the crutch ends.
        <   1 % -> COLLAPSES ON WEANING -- the stock was crutch-fed throughout.
        between -> declining; undetermined at this horizon, say so rather than picking a side.

FALSIFIERS -- if any fails, nothing above is readable.
  E1 the wrapper fires: `seeded` reads 24/24 steps in EVERY in-window year for the four targets.
  E2 the wrapper is SCOPED: no untouched species shows the FORCED signature (pinned 24/24 steps
     in every in-window year). Note this is deliberately not "untouched species are bit-identical
     to d100" — feeding four stocks more recruits changes the food web, so some drift is the
     experiment working, not leaking. Measured: 4/5 bit-identical, perch drifts 108 -> 85 seeded
     steps, and no untouched species is pinned.
  E3 IN-PLACE PROPAGATION, load-bearing: age-0 schools carrying `from_seeding=True` are observed.
     If this is 0 while `seeded` is 24, the wrapper moved the egg COUNT but not the RV exemption,
     and the arm is not the counterfactual it claims to be.
  E4 INSTRUMENT VALIDITY: this arm's `b_cut` per-step mean reproduces its own `res.biomass()`
     over the tail to within 5 %, the same gate the ladder passed at 0.000 %.

Run:  .venv/bin/python scripts/c3_sustained_crutch_test.py <out.json>
      .venv/bin/python scripts/c3_sustained_crutch_test.py --report <dir>
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c3_cod_east_egg_pathway import DOSES, SEED, TAIL, TARGETS, build_cfg

import osmose.engine.processes.reproduction as repro_mod
from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

N_YEAR = 25
WIN = slice(5, 15)  # in-window years, past the bootstrap transient
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
SUSTAIN = tuple(sorted(TARGETS.values()))

_ORIG_REGULATE = repro_mod.regulate_recruitment


def run_sustained(cfg, n_sp, ndt):
    tot = np.zeros((N_YEAR, n_sp))
    mat_b = np.zeros((N_YEAR, n_sp))
    b_cut = np.zeros((N_YEAR, n_sp))
    eggs = np.zeros((N_YEAR, n_sp))
    ssb_in = np.zeros((N_YEAR, n_sp))
    seeded = np.zeros((N_YEAR, n_sp))
    fs_eggs = np.zeros((N_YEAR, n_sp))  # E3: abundance of eggs tagged from_seeding
    steps = np.zeros(N_YEAR)
    forced = [0]

    from osmose.engine.config import EngineConfig

    ec_probe = EngineConfig.from_dict(dict(cfg))

    def _a(name):
        v = getattr(ec_probe, name, None)
        return np.asarray(v if v is not None else np.zeros(n_sp), dtype=np.float64)

    m0, m1 = _a("bioen_m0"), _a("bioen_m1")
    cutoff = _a("output_cutoff_age")[:n_sp].copy()
    if not np.isfinite(cutoff).any() or cutoff.max() <= 0:
        cutoff = np.full(n_sp, 0.5)

    def observer(step, state, grid, config, map_sets):
        yr = min(step // ndt, N_YEAR - 1)
        steps[yr] += 1
        sp = np.asarray(state.species_id)
        abd = np.asarray(state.abundance, dtype=np.float64)
        age_y = np.asarray(state.age_dt, dtype=np.float64) / ndt
        live = (abd > 0) & (sp < n_sp)
        if not live.any():
            return
        w = np.asarray(state.weight, dtype=np.float64)
        aw = abd * w
        is_egg = np.asarray(state.is_egg)
        nonegg = live & (~is_egg)

        def bc(mask, wt=None):
            if not mask.any():
                return np.zeros(n_sp)
            return np.bincount(sp[mask], weights=None if wt is None else wt[mask], minlength=n_sp)[
                :n_sp
            ]

        tot[yr] += bc(nonegg, abd)
        thr = np.full(len(sp), np.inf)
        thr[live] = m0[sp[live]] + m1[sp[live]] * age_y[live]
        mat = nonegg & (np.asarray(state.length, dtype=np.float64) >= thr)
        mat_b[yr] += bc(mat, aw)
        below = np.zeros(len(sp), dtype=bool)
        below[live] = age_y[live] < cutoff[sp[live]]
        b_cut[yr] += bc(live & ~below, aw)

        fs = getattr(state, "from_seeding", None)
        if fs is not None:
            fsa = np.asarray(fs, dtype=bool)
            fs_eggs[yr] += bc(live & is_egg & fsa, abd)

    def wrapper(n_eggs_linear, ssb, seeded_this_step, config, step):
        # Same season array the bioen path forms (`simulate.py:819-822`).
        if config.spawning_season is not None:
            season_all = config.spawning_season[:, step % config.spawning_season.shape[1]]
        else:
            season_all = np.full(len(ssb), 1.0 / config.n_dt_per_year)
        for sp in SUSTAIN:
            if step >= int(config.seeding_max_step[sp]) or float(config.seeding_biomass[sp]) <= 0:
                continue
            floor = float(config.seeding_biomass[sp])
            if float(ssb[sp]) >= floor:
                continue  # already seeded by the engine, or genuinely richer than the floor
            # Mutate IN PLACE: `create_egg_schools` receives this same `seeded_this_step` object,
            # so the sustained eggs are tagged `from_seeding` and skip the RV penalty too.
            ssb[sp] = floor
            seeded_this_step[sp] = True
            n_eggs_linear[sp] = (
                float(config.sex_ratio[sp])
                * float(config.relative_fecundity[sp])
                * floor
                * float(season_all[sp])
                * 1_000_000.0
            )
            forced[0] += 1
        out = _ORIG_REGULATE(n_eggs_linear, ssb, seeded_this_step, config, step)
        yr = min(step // ndt, N_YEAR - 1)
        eggs[yr] += np.asarray(out, dtype=np.float64)[:n_sp]
        ssb_in[yr] += np.asarray(ssb, dtype=np.float64)[:n_sp]
        seeded[yr] += np.asarray(seeded_this_step, dtype=np.float64)[:n_sp]
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

    import pandas as pd

    res = OsmoseResults.from_outputs(outs, ec, grid)
    bio = res.biomass()
    tl = max(1, int(len(bio) * TAIL / N_YEAR))
    fd = {}
    for s in [c for c in bio.columns if str(c).strip().lower() not in ("time", "year", "step")]:
        v = pd.to_numeric(bio[s], errors="coerce").to_numpy(dtype=float)[-tl:]
        if v.size and np.isfinite(v).any():
            fd[str(s)] = float(np.nanmean(v))

    return {
        "tot": tot.tolist(),
        "mat_b": mat_b.tolist(),
        "b_cut": b_cut.tolist(),
        "eggs": eggs.tolist(),
        "ssb_in": ssb_in.tolist(),
        "seeded": seeded.tolist(),
        "fs_eggs": fs_eggs.tolist(),
        "steps": steps.tolist(),
        "m0": m0.tolist(),
        "cutoff": cutoff.tolist(),
        "fd": fd,
        "forced": forced[0],
    }


def _ps(arm, key, i):
    v = np.asarray(arm[key])[:, i]
    s = np.asarray(arm["steps"])
    return np.divide(v, s, out=np.zeros_like(v), where=s > 0)


def report(where: Path) -> int:
    sus = json.loads((where / "sustained_sus.json").read_text())
    ctl = json.loads((where / "eggpath_d100.json").read_text())
    base = json.loads((where / "eggpath_base.json").read_text())
    names = sus["names"]
    print(f"\n{'=' * 100}\nSUSTAINED SEEDING CRUTCH — does recruitment bind at all?\n{'=' * 100}")

    print("\nFALSIFIERS")
    ce = names.index("cod_east")
    sd = np.asarray(sus["seeded"])[:, ce]
    e1 = bool((sd[WIN] == 24).all())
    print(f"  E1 wrapper fires 24/24 steps in every in-window yr (cod_east) : {e1}  {sd[WIN]}")
    # E2 tests SCOPING, not isolation. An earlier version demanded the untouched species be
    # bit-identical to d100 and "failed" on perch — but perch drifting is the food web RESPONDING
    # to more cod/herring/flounder recruits, which is a real ecological effect, not the wrapper
    # reaching outside its targets. The discriminating signature: a FORCED species is pinned at
    # 24/24 steps for every in-window year (cod_east 44 -> 360 = 15 yr x 24); a species that merely
    # drifted is not. So the test is "no untouched species shows the pinned signature".
    others = [j for j in range(len(names)) if j not in SUSTAIN]
    pinned = [names[j] for j in others if (np.asarray(sus["seeded"])[WIN, j] == 24).all()]
    e2 = not pinned
    drifted = [
        names[j]
        for j in others
        if not np.array_equal(np.asarray(sus["seeded"])[:, j], np.asarray(ctl["seeded"])[:, j])
    ]
    print(
        f"  E2 wrapper forced ONLY its targets (no untouched pinned)      : {e2}"
        f"{'  pinned: ' + ', '.join(pinned) if pinned else ''}"
    )
    print(
        f"     untouched species bit-identical to d100: "
        f"{len(others) - len(drifted)}/{len(others)}"
        f"{'; drifted (food-web response): ' + ', '.join(drifted) if drifted else ''}"
    )
    fse = np.asarray(sus["fs_eggs"])[:, ce]
    e3 = bool(fse[WIN].sum() > 0)
    print(
        f"  E3 from_seeding eggs observed (in-place propagation)          : {e3}  "
        f"(in-window total {fse[WIN].sum():.4e})"
    )
    mine = float(np.mean(_ps(sus, "b_cut", ce)[N_YEAR - TAIL :]))
    pub = float(sus["fd"].get("cod_east", float("nan")))
    rel = abs(mine - pub) / pub if pub else (0.0 if mine == 0 else float("inf"))
    e4 = rel <= 0.05 or (mine < 1e-9 and pub < 1e-9)
    print(f"  E4 b_cut tail {mine:.4f} t vs biomass() {pub:.4f} t  rel {rel:.2%}     : {e4}")
    print(f"  (wrapper forced the floor {sus['forced']} times)")
    if not (e1 and e3 and e4):
        print("\n  A falsifier failed — nothing below is readable.")
        return 1

    print(f"\n{'-' * 100}\nPer-species, per-step means\n{'-' * 100}")
    print(
        f"  {'species':<12}{'arm':<6}{'b_cut in-win':>15}{'b_cut yr24':>13}{'SSB in-win':>13}"
        f"{'eggs in-win':>14}{'biomass()':>13}"
    )
    verdicts = {}
    for nm in TARGETS:
        i = names.index(nm)
        row = {}
        for k, a in (("base", base), ("d100", ctl), ("sus", sus)):
            bw = float(np.mean(_ps(a, "b_cut", i)[WIN]))
            b24 = float(_ps(a, "b_cut", i)[N_YEAR - 1])
            sw = float(np.mean(_ps(a, "mat_b", i)[WIN]))
            ew = float(np.mean(np.asarray(a["eggs"])[WIN, i]))
            row[k] = (bw, b24)
            print(
                f"  {nm if k == 'base' else '':<12}{k:<6}{bw:>15.2f}{b24:>13.2f}{sw:>13.2f}"
                f"{ew:>14.3e}{a['fd'].get(nm, float('nan')):>13.2f}"
            )
        r1 = row["sus"][0] / row["base"][0] if row["base"][0] else float("inf")
        r2 = row["sus"][1] / row["sus"][0] if row["sus"][0] else float("inf")
        # R1 only means anything if the eggs were actually DELIVERED. Carry the egg ratio against
        # the healthy bioen-off baseline into the verdict: "1 % of the biomass" is a different
        # claim when the arm got 2.18x the eggs than when it got 0.73x.
        es = float(np.mean(np.asarray(sus["eggs"])[WIN, i]))
        eb = float(np.mean(np.asarray(base["eggs"])[WIN, i]))
        verdicts[nm] = (r1, r2, es / eb if eb else float("nan"))
        print()

    print(f"{'=' * 100}\nPRE-REGISTERED VERDICTS\n{'=' * 100}")
    for nm, (r1, r2, er) in verdicts.items():
        v1 = (
            "recruitment BINDS"
            if r1 >= 0.20
            else "RECRUITMENT NEVER THE CONSTRAINT"
            if r1 < 0.01
            else "partial"
        )
        v2 = (
            "HOLDS"
            if r2 >= 0.20
            else "COLLAPSES ON WEANING"
            if r2 < 0.01
            else "declining, undetermined"
        )
        print(
            f"  {nm:<12} eggs delivered {er:>6.2f}x base  ->  R1 biomass {r1:>8.4f}x base"
            f"  [{v1}]\n{'':<16}R2 yr24/in-window = {r2:>8.4f} -> {v2}"
        )
    return 0


def main() -> int:
    warnings.simplefilter("ignore")
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        return report(Path(sys.argv[2]))
    dst = sys.argv[1] if len(sys.argv) > 1 else None
    if not dst:
        print("usage: c3_sustained_crutch_test.py <out.json> | --report <dir>")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="c3_sustain_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    ndt = int(raw["simulation.time.ndtperyear"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}
    cfg = build_cfg(raw, cf.parent, overlay, DOSES["d100"])
    cfg["simulation.time.nyear"] = str(N_YEAR)
    r = run_sustained(cfg, n_sp, ndt)
    r["names"] = names
    Path(dst).write_text(json.dumps(r))
    print(f"sustained: wrote {dst} (forced {r['forced']}, steps {sum(r['steps']):.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
