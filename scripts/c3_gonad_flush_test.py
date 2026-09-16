"""Is the C3 egg deficit caused by the bioen starvation substep FLUSHING the gonad to zero?

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9. The birth side caps abundance: at 50 yr
production seeding, mortality is IDENTICAL between arms (1.00-1.01x per-capita) while births are
16-24x lower for the four collapsed stocks and only 1.25x lower for sprat, which survives.

THE HYPOTHESIS. `osmose/engine/processes/mortality.py:1366-1391` (the production Numba path, and
identically `bioen_starvation.py`) does:

    deficit = abs(e_net[idx]) / n_subdt     # e_net is tonnes PER SCHOOL   (state.py:82-83)
    gonad   = gonad_weight[idx]             # gonad is tonnes PER FISH     (getDg = rho*E_net/N)
    if gonad >= eta * deficit:              # <-- compared DIRECTLY, off by a factor of N
        ...buffer, kill nobody...
    gonad_weight[idx] = 0.0                 # <-- otherwise the ENTIRE gonad is destroyed
    dead = deficit / w                      # (t/school)/(t/fish) = fish -- dimensionally CORRECT

The mismatch is self-documented at `bioen_starvation.py:58`: *"`E_net` (tonnes per school) divided
by `subdt` while the gonad is per FISH"*. Its two halves behave differently: the death toll's units
cancel, so it is small and correct; the buffer test's do not, so for any realistic school (1e4-1e10
fish) it fails and the gonad is annihilated. **A marginal energy deficit kills almost no fish and
destroys 100% of accumulated reproductive investment** — which is precisely why the abundance-balance
run saw STARVATION at 3.4-7.2x with a death share of 0.000-0.004 and total mortality at 1.00x.

THE QUANTITATIVE FINGERPRINT, verified independently of this script. Release is partial:
`released = gonad_weight * season` (`bioen_reproduction.py:42-45`), and every Baltic species'
`spawning_season` sums to **exactly 1.0** (checked; max entry 0.152-0.333). With no flushing, annual
release = annual accrual = `24 * d`. With a flush every step, the pool at release holds only that
step's accrual, so annual release = `d * sum(season) = d`. **Ceiling of the egg deficit =
n_dt_per_year / sum(season) = 24.00x.** Measured: flounder 23.8x, cod_west 20.0x, cod_east 16.4x,
herring 16.4x, sprat 1.25x. All four collapsers sit at or just under that ceiling.

But a number matching a prediction is an INFERENCE, and inferring a mechanism from a number has
failed three times in this investigation. This measures the mechanism DIRECTLY.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Instrument: `step_observer` (fires after `mortality()`, before `compact()`, so `state.gonad_weight`
and `state.e_net` hold this step's post-mortality values). Over MATURE, non-egg schools only —
`length >= bioen_m0 + bioen_m1 * age`, the same predicate `_bioen_reproduction` uses for SSB — the
abundance-weighted:

  (a) fraction with `e_net < 0`          — the flush TRIGGER rate
  (b) fraction with `gonad_weight == 0.0` EXACTLY  — the flush FINGERPRINT
  (c) mean gonad per fish                — the pool

**(b) is the discriminator, and it is binary.** Under correct buffering a mature school's gonad can
NEVER be exactly 0.0 after its first positive accrual step, because release retains at least
`1 - max(season)` = 66.7% of the pool. An exact zero can therefore only come from the flush branch.

  (1) (b) >= 0.5 for the collapsed stocks AND materially lower for sprat
        -> GONAD FLUSH CONFIRMED as the mechanism capping births. The C3 collapse is then driven by
           a per-fish/per-school unit mismatch in the starvation substep.
  (2) (b) ~ 0 for the collapsed stocks
        -> HYPOTHESIS DEAD. The 24x fingerprint is coincidence and the egg deficit is elsewhere.
  (3) (b) high for collapsers AND equally high for sprat
        -> the flush is real but NOT what separates survivors from collapsers; it cannot be the
           explanation on its own.

sprat is the built-in control: same run, same code path, predicted to differ sharply.

Also captured, settling the doc's open decomposition in the same run: `ssb` and the pre-regulation
`n_eggs_linear` from the `regulate_recruitment` wrapper, giving eggs per unit SSB. If SSB is
comparable between arms but eggs/SSB is 16-24x down, the deficit is eggs-PER-SPAWNER, which is what
a gonad flush produces; if SSB itself is down, it is fewer spawners and the flush is not sufficient.

Engagement checks:
  E1 the observer fires and finds mature schools for every focal species;
  E2 the baseline arm (bioen OFF) shows (b) ~ 0 — the flush is bioen-only by construction, so a
     non-zero baseline reading means the instrument is measuring something else.

======================================================================================================

Run: .venv/bin/python scripts/c3_gonad_flush_test.py
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

import osmose.engine.processes.reproduction as repro_mod
from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

N_YEAR = 20  # the flush is a PER-STEP property; 20 yr is ample and keeps the footprint small
SEED = 42
SEAL, TEMPLATE = "GreySeal", "Cormorant"
FOCUS = ("cod_west", "cod_east", "herring", "flounder", "sprat")
COLLAPSED = ("cod_west", "cod_east", "herring", "flounder")
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIG_REGULATE = repro_mod.regulate_recruitment


def write_repaired(src: Path, dst: Path) -> None:
    rows = list(csv.reader(src.open(), delimiter=";"))
    hdr = [h.strip() for h in rows[0]]
    t = hdr.index(TEMPLATE)
    out = [rows[0] + [SEAL]] + [r + [repr(float(r[t]))] for r in rows[1:] if r]
    out.append([SEAL] + ["0"] * (len(out[0]) - 1))
    with dst.open("w", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(out)


def run_arm(raw, cfg_dir, overlay, n_sp, ndt, ec_probe):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg.pop("population.seeding.year.max", None)
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    write_repaired(cfg_dir / "predation-accessibility.csv", cfg_dir / "acc-rep.csv")
    cfg["predation.accessibility.file"] = "acc-rep.csv"

    acc = {k: np.zeros(n_sp) for k in ("w", "neg", "zero", "gonad")}
    m0_arr = np.asarray(getattr(ec_probe, "bioen_m0", np.zeros(n_sp)), dtype=np.float64)
    m1_arr = np.asarray(getattr(ec_probe, "bioen_m1", np.zeros(n_sp)), dtype=np.float64)
    ssb_tot = np.zeros(n_sp)
    eggs_lin = np.zeros(n_sp)

    def observer(step, state, grid, config, map_sets):
        # ONE vectorised pass with bincount, not a 9-iteration loop building full-length boolean
        # temporaries per species per step — that shape is what made the 50-yr version die in warmup.
        sp = np.asarray(state.species_id)
        abd = np.asarray(state.abundance, dtype=np.float64)
        ok = (abd > 0) & (~np.asarray(state.is_egg)) & (sp < n_sp)
        if not ok.any():
            return
        spo = sp[ok]
        thr = m0_arr[spo] + m1_arr[spo] * (np.asarray(state.age_dt)[ok] / ndt)
        mat = np.asarray(state.length, dtype=np.float64)[ok] >= thr
        if not mat.any():
            return
        idx = spo[mat]
        a = abd[ok][mat]
        gon = np.asarray(state.gonad_weight, dtype=np.float64)[ok][mat]
        en = np.asarray(state.e_net, dtype=np.float64)[ok][mat]
        acc["w"] += np.bincount(idx, weights=a, minlength=n_sp)[:n_sp]
        acc["neg"] += np.bincount(idx, weights=a * (en < 0.0), minlength=n_sp)[:n_sp]
        acc["zero"] += np.bincount(idx, weights=a * (gon == 0.0), minlength=n_sp)[:n_sp]
        acc["gonad"] += np.bincount(idx, weights=a * gon, minlength=n_sp)[:n_sp]

    def wrapper(n_eggs_linear, ssb_in, seeded_this_step, config, step):
        ssb_tot[:] += np.asarray(ssb_in, dtype=np.float64)
        eggs_lin[:] += np.asarray(n_eggs_linear, dtype=np.float64)
        return _ORIG_REGULATE(n_eggs_linear, ssb_in, seeded_this_step, config, step)

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
    return {"acc": acc, "ssb": ssb_tot, "eggs_lin": eggs_lin}


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_flush_"))
    demo = osmose_demo("baltic", tmp)
    cf = Path(demo["config_file"])
    raw = dict(OsmoseConfigReader().read(cf))
    n_sp = int(raw["simulation.nspecies"])
    ndt = int(raw["simulation.time.ndtperyear"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)][:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    # Build the bioen EngineConfig ONCE to read bioen_m0/bioen_m1 — the same maturity predicate
    # `_bioen_reproduction` uses for SSB. Hoisted out of the observer so it is not rebuilt per step.
    from osmose.engine.config import EngineConfig

    probe_cfg = dict(raw)
    probe_cfg.update(overlay)
    ec_probe = EngineConfig.from_dict(probe_cfg)

    print(f"{N_YEAR} yr, production seeding, seed {SEED}, repaired matrix")
    res = {}
    for tag, ov in (("bioen", overlay), ("baseline", None)):
        print(f"  running {tag} ...", flush=True)
        res[tag] = run_arm(raw, cf.parent, ov, n_sp, ndt, ec_probe)

    print(f"\n{'=' * 96}\nMATURE-SCHOOL GONAD STATE (abundance-weighted over the whole run)")
    print("=" * 96)
    print(
        f"{'species':<13}{'arm':<11}{'frac e_net<0':>14}{'frac gonad==0':>16}{'mean gonad t/fish':>20}"
    )
    for s in FOCUS:
        i = names.index(s)
        for tag in ("bioen", "baseline"):
            a = res[tag]["acc"]
            w = a["w"][i]
            if w <= 0:
                print(f"{s if tag == 'bioen' else '':<13}{tag:<11}{'— no mature schools':>50}")
                continue
            print(
                f"{s if tag == 'bioen' else '':<13}{tag:<11}"
                f"{a['neg'][i] / w:>14.4f}{a['zero'][i] / w:>16.4f}{a['gonad'][i] / w:>20.3e}"
            )
        print()

    print(f"{'=' * 96}\nBIRTH DECOMPOSITION — spawners vs eggs-per-spawner\n{'=' * 96}")
    print(
        f"{'species':<13}{'SSB bioen/base':>16}{'eggsLin bioen/base':>20}{'eggs per SSB ratio':>20}"
    )
    for s in FOCUS:
        i = names.index(s)
        sb, sx = res["baseline"]["ssb"][i], res["bioen"]["ssb"][i]
        eb, ex = res["baseline"]["eggs_lin"][i], res["bioen"]["eggs_lin"][i]
        r_ssb = sx / sb if sb else float("nan")
        r_egg = ex / eb if eb else float("nan")
        per = (ex / sx) / (eb / sb) if (sx and sb and eb) else float("nan")
        print(f"{s:<13}{r_ssb:>16.3f}{r_egg:>20.3f}{per:>20.3f}")

    print(f"\n{'=' * 96}\nPRE-REGISTERED VERDICT\n{'=' * 96}")
    e1 = all(res["bioen"]["acc"]["w"][names.index(s)] > 0 for s in FOCUS)
    base_zero = max(
        res["baseline"]["acc"]["zero"][names.index(s)]
        / max(res["baseline"]["acc"]["w"][names.index(s)], 1e-30)
        for s in FOCUS
    )
    print(f"  E1 mature schools found for every focal species : {e1}")
    print(f"  E2 baseline frac(gonad==0) is ~0 (max {base_zero:.4f})   : {base_zero < 0.05}")

    zc = {
        s: res["bioen"]["acc"]["zero"][names.index(s)]
        / max(res["bioen"]["acc"]["w"][names.index(s)], 1e-30)
        for s in FOCUS
    }
    coll = [zc[s] for s in COLLAPSED]
    print("\n  frac(gonad==0) collapsers: " + ", ".join(f"{s} {zc[s]:.3f}" for s in COLLAPSED))
    print(f"  frac(gonad==0) sprat (control): {zc['sprat']:.3f}")
    if not e1:
        print("\n  INCONCLUSIVE — E1 failed.")
    elif min(coll) >= 0.5 and zc["sprat"] < min(coll):
        print("\n  GONAD FLUSH CONFIRMED — the per-fish/per-school unit mismatch in the starvation")
        print(
            "  substep destroys accumulated reproductive investment on every negative-E_net step,"
        )
        print("  capping births while killing almost no fish. That is what caps abundance.")
    elif max(coll) < 0.05:
        print("\n  HYPOTHESIS DEAD — mature gonads are not being zeroed. The 24x fingerprint is")
        print("  coincidence and the egg deficit is elsewhere.")
    elif zc["sprat"] >= min(coll):
        print("\n  FLUSH IS REAL BUT NOT THE SEPARATOR — sprat flushes as much as the collapsers,")
        print("  so it cannot explain the survivor/collapser split on its own.")
    else:
        print(
            "\n  PARTIAL — report the numbers; the flush is present but below the pre-registered bar."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
