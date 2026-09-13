"""Test the C3 RECRUITMENT hypothesis by intervention.

Context. `docs/baltic_c3_bioen_stage1_2026-09-05.md` §9 closed C3 by characterization: under the
bioen overlay, cod_west/cod_east/flounder/herring/perch collapse under a `seeding.year.max=1`
stress while a bioen-off control on the identical config sustains all nine. The GROWTH account of
that collapse was located and then REFUTED BY INTERVENTION the same day -- fixing the growth curve
completely changed nothing. The surviving candidate is RECRUITMENT: bioen produced 11-48% of
baseline egg output for every collapsing stock (scratch measurement, indicative only -- re-derived
here rather than reconciled against). It was recorded as explicitly UNESTABLISHED, with the
standing instruction: *do not treat recruitment as established until someone raises bioen egg
output and shows the stocks persist.* This script is that test.

The intervention. `regulate_recruitment` (`osmose/engine/processes/reproduction.py:84`) is the
single shared choke point for BOTH reproduction paths -- the classic path calls it at
`reproduction.py:344`, the bioen path at `simulate.py:885`. Wrapping it therefore serves as both
instrument and intervention, with no engine edit. On the bioen arms the wrapper replaces the
gonad-derived `n_eggs_linear` with the baseline path's OWN formula

    sex_ratio * relative_fecundity * SSB * season * 1e6 * K

for every species not seeded that step (the seeding bootstrap already uses this formula, so
substituting there would double-apply it). Everything downstream is untouched: the Shepherd curve,
cod_east's RV gate, the ceiling/thermal/depensation gates and egg-school creation all run exactly
as before. Growth, maintenance, starvation and the TPC are untouched. Only the number of eggs
changes.

Arms (all 8 yr, `population.seeding.year.max = 1`, seed 42 -- identical stress to the growth
refutation, so the two are directly comparable):

    baseline     bioen OFF                                  -- the control that sustains
    bioen        the committed C3 overlay                    -- the arm that collapses
    bioen_rec    bioen + baseline's recruitment formula, K=1 -- the MECHANISTIC arm
    bioen_rec10  bioen + the same formula at K=10            -- the FALSIFIER

Why the falsifier is the one that carries the verdict. Under bioen the fish are smaller and mature
later, so SSB is itself lower; feeding baseline's formula a bioen SSB still yields fewer eggs than
baseline produces. `bioen_rec` therefore cannot be relied on to close the egg gap, and a collapse
there would be a partially-engaged knob rather than a result. `bioen_rec10` deliberately overshoots
to remove that ambiguity.

=========================== PRE-REGISTERED READING (written before the run) ===========================

Let R = (realized post-regulation eggs on `bioen_rec10`) / (realized post-regulation eggs on
`bioen`), per species, summed over the run. For the five stocks that collapse on `bioen`
(cod_west, cod_east, herring, flounder, perch):

  (1) R >= 5 AND the stock persists (final abundance > 0) on `bioen_rec10`
        -> RECRUITMENT CONFIRMED as a cause for that stock.
  (2) R >= 5 AND the stock still collapses
        -> RECRUITMENT REFUTED for that stock. Same standing as the growth account.
  (3) R < 5
        -> INCONCLUSIVE for that stock. The knob did not deliver, so the finding is about the
           knob, not the biology, and must be reported as such -- NOT as a null.

Two conditions invalidate the whole run rather than any single species:
  - `baseline` must sustain all nine (the control still works), and
  - `bioen_rec` must differ from `bioen` in realized eggs for at least one species (the wrapper
    actually fires; a silent no-op would make the arms identical and mimic a null).

Reported per species per arm so case (3) can be diagnosed rather than guessed: SSB, pre-regulation
`n_eggs_linear`, post-regulation `n_eggs`, their ratio (how much the Shepherd curve / RV gate eat),
the gonad-derived count the bioen path would have produced, substitution and seeding event counts,
and whether the RV gate is enabled. cod_east is called out separately: its recruitment is partly
prescribed by the RV gate, so the intervention may be unable to move it regardless of arm, and a
cod_east null must not be read as "recruitment refuted".

======================================================================================================

Run: .venv/bin/python scripts/c3_recruitment_intervention.py
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

import osmose.engine.processes.reproduction as repro_mod  # noqa: E402
from osmose.config import OsmoseConfigReader  # noqa: E402
from osmose.demo import osmose_demo  # noqa: E402
from osmose.engine import PythonEngine  # noqa: E402

N_YEAR = 8
SEED = 42
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
_ORIGINAL = repro_mod.regulate_recruitment


def _make_recorder(n_sp: int) -> dict[str, np.ndarray]:
    z = lambda: np.zeros(n_sp, dtype=np.float64)  # noqa: E731
    return {
        "ssb": z(),
        "linear_used": z(),
        "linear_gonad": z(),
        "eggs": z(),
        "n_subst": z(),
        "n_seeded": z(),
        "n_calls": z(),
    }


def _make_wrapper(substitute: bool, k: float, rec: dict[str, np.ndarray]):
    """Wrap `regulate_recruitment`: record always, substitute only on the bioen intervention arms."""

    def wrapper(n_eggs_linear, ssb, seeded_this_step, config, step):
        n_sp = config.n_species
        incoming = np.asarray(n_eggs_linear, dtype=np.float64)
        used = incoming.copy()
        if substitute:
            if config.spawning_season is not None:
                season = config.spawning_season[:n_sp, step % config.spawning_season.shape[1]]
            else:
                season = np.full(n_sp, 1.0 / config.n_dt_per_year)
            classic = (
                np.asarray(config.sex_ratio[:n_sp], dtype=np.float64)
                * np.asarray(config.relative_fecundity[:n_sp], dtype=np.float64)
                * np.asarray(ssb, dtype=np.float64)
                * season
                * 1_000_000.0
                * k
            )
            # Never substitute on a seeded step: that branch already uses this exact formula.
            fire = ~np.asarray(seeded_this_step, dtype=bool)
            used = np.where(fire, classic, incoming)
            rec["n_subst"] += fire.astype(np.float64)
        out = _ORIGINAL(used, ssb, seeded_this_step, config, step)
        rec["ssb"] += np.asarray(ssb, dtype=np.float64)
        rec["linear_used"] += used
        rec["linear_gonad"] += incoming
        rec["eggs"] += np.asarray(out, dtype=np.float64)
        rec["n_seeded"] += np.asarray(seeded_this_step, dtype=bool).astype(np.float64)
        rec["n_calls"] += 1.0
        return out

    return wrapper


def run_arm(tag, raw, overlay, focal, substitute, k, n_sp):
    cfg = dict(raw)
    cfg["simulation.time.nyear"] = str(N_YEAR)
    cfg["population.seeding.year.max"] = "1"
    if overlay is None:
        cfg["module.bioenergetics.enabled"] = "false"
    else:
        cfg.update(overlay)
    rec = _make_recorder(n_sp)
    repro_mod.regulate_recruitment = _make_wrapper(substitute, k, rec)
    try:
        res = PythonEngine().run_in_memory(cfg, seed=SEED)
    finally:
        repro_mod.regulate_recruitment = _ORIGINAL
    bio, abd = res.biomass(), res.abundance()
    return {
        "final": {s: (float(bio[s].iloc[-1]), float(abd[s].iloc[-1])) for s in focal},
        "rec": {key: arr.copy() for key, arr in rec.items()},
    }


def main() -> int:
    warnings.simplefilter("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="c3_recruit_"))
    demo = osmose_demo("baltic", tmp)
    raw = dict(OsmoseConfigReader().read(Path(demo["config_file"])))
    n_sp = int(raw["simulation.nspecies"])
    names = [raw[f"species.name.sp{i}"] for i in range(n_sp)]
    focal = names[:9]
    overlay = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if k != "_meta"}

    arms = (
        ("baseline", None, False, 1.0),
        ("bioen", overlay, False, 1.0),
        ("bioen_rec", overlay, True, 1.0),
        ("bioen_rec10", overlay, True, 10.0),
    )
    out = {}
    for tag, ov, subst, k in arms:
        print(f"  running {tag} ...", flush=True)
        out[tag] = run_arm(tag, raw, ov, focal, subst, k, n_sp)

    print(f"\n{'=' * 96}\nFINAL STATE (year {N_YEAR}, seed {SEED}, seeding.year.max=1)\n{'=' * 96}")
    print(f"{'species':<13}" + "".join(f"{t:>19}" for t, *_ in arms))
    for i, s in enumerate(focal):
        row = f"{s:<13}"
        for tag, *_ in arms:
            bm, ab = out[tag]["final"][s]
            row += f"{bm:>15.1f}{'  X' if ab <= 0 else '   '}"
        print(row)
    print("X = zero abundance (extinct)")

    print(f"\n{'=' * 96}\nRECRUITMENT INSTRUMENT (run totals)\n{'=' * 96}")
    hdr = (
        f"{'species':<13}{'arm':<13}{'SSB':>12}{'linear':>13}{'eggs':>13}"
        f"{'egg/lin':>9}{'subst':>7}{'seed':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(focal):
        for tag, *_ in arms:
            r = out[tag]["rec"]
            lin, eggs = r["linear_used"][i], r["eggs"][i]
            ratio = eggs / lin if lin > 0 else float("nan")
            print(
                f"{s if tag == 'baseline' else '':<13}{tag:<13}{r['ssb'][i]:>12.1f}"
                f"{lin:>13.3e}{eggs:>13.3e}{ratio:>9.3f}"
                f"{int(r['n_subst'][i]):>7}{int(r['n_seeded'][i]):>6}"
            )
        print()

    print(f"{'=' * 96}\nPRE-REGISTERED VERDICT\n{'=' * 96}")
    base_ok = all(out["baseline"]["final"][s][1] > 0 for s in focal)
    fired = any(
        not np.isclose(out["bioen_rec"]["rec"]["eggs"][i], out["bioen"]["rec"]["eggs"][i])
        for i in range(len(focal))
    )
    print(
        f"validity: baseline sustains all nine = {base_ok};  wrapper fires (rec != bioen) = {fired}"
    )
    if not (base_ok and fired):
        print("RUN INVALID -- a validity condition failed; do not read the per-species verdicts.")

    print(f"\n{'species':<13}{'R (rec10/bioen eggs)':>22}{'persists on rec10':>20}   verdict")
    for i, s in enumerate(focal):
        e_b = out["bioen"]["rec"]["eggs"][i]
        e_10 = out["bioen_rec10"]["rec"]["eggs"][i]
        collapsed = out["bioen"]["final"][s][1] <= 0
        persists = out["bioen_rec10"]["final"][s][1] > 0
        r = e_10 / e_b if e_b > 0 else float("inf") if e_10 > 0 else float("nan")
        if not collapsed:
            verdict = "(survives on bioen -- not part of the test)"
        elif not (r >= 5.0):
            verdict = "INCONCLUSIVE -- knob delivered < 5x; finding is the knob, not the biology"
        elif persists:
            verdict = "RECRUITMENT CONFIRMED"
        else:
            verdict = "RECRUITMENT REFUTED"
        print(f"{s:<13}{r:>22.2f}{str(persists):>20}   {verdict}")

    print("\ncod_east caveat: its recruitment is partly prescribed by the RV gate, so the")
    print("intervention may be unable to move it. Read its row with that in mind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
