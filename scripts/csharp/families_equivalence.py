"""Ensemble equivalence of the FULL output families, C# engine against the Python engine.

``scripts/cross_engine_parity_440.py`` compares biomass, abundance, yield and mean weight, on
minimal outputs. This script does the same for the families the C# engine collects since task
2.6: the life-stage split, the distributions, SSB, yield in numbers, the two weighted means and
the diet. Same design (spec 10.2): two Python arms on disjoint seeds (the control is A against
B), one C# arm (the treatment is A against C#), the rule and its calibration from
``osmose.validation.engine_equivalence``.

Each replicate runs the config AS SHIPPED (every output switch it sets) and is reduced, per
species, to post-spin-up means of:

    ssb, yield_n                              tonnes, numbers            margin 1.25x
    eggs, juveniles, adults                   abundance by life stage    margin 1.25x
    age0, age1plus                            abundance by age class     margin 1.25x
    biomass_age0, biomass_age1plus            biomass by age class       margin 1.25x
    eaten_fish, eaten_resources               tonnes eaten, by predator  margin 1.25x
    mean_size, mean_tl                        cm, trophic level          margin 1.05x

A metric a config does not produce is left out. Background predators appear in the two diet
metrics only (they have diet rows and nothing else).

Usage: PYTHONPATH=. python scripts/csharp/families_equivalence.py [--config baltic] [--n 16]
       [--years 15] [--spinup-years 5] [--csharp-threads 1] [--json out.json] [--rejudge in.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = {
    "eec_full": ROOT / "data" / "eec_full" / "eec_all-parameters.csv",
    "baltic": ROOT / "data" / "baltic" / "baltic_all-parameters.csv",
    "examples": ROOT / "data" / "examples" / "osm_all-parameters.csv",
}
#: The same seed bases as the main ensemble (spec 10.2): two Python arms, one C# arm, disjoint.
SEEDS = {"python": 1000, "python-control": 5000, "csharp": 9000}
EXTENSIVE = (
    "ssb", "yield_n", "eggs", "juveniles", "adults", "age0", "age1plus",
    "biomass_age0", "biomass_age1plus", "eaten_fish", "eaten_resources",
)  # fmt: skip
INTENSIVE = ("mean_size", "mean_tl")
MARGINS = {m: float(np.log10(1.25)) for m in EXTENSIVE} | {
    m: float(np.log10(1.05)) for m in INTENSIVE
}


def reduce_records(records: list, config, spinup_years: int) -> dict[str, dict[str, float]]:
    """One run's records -> {metric: {species: post-spin-up mean}}.

    The spin-up is dropped by the record's window: a record is kept only if its whole window
    lies after the spin-up (the initial record, step -1, is always dropped). A species absent
    from a mean in some record (no qualifying school)
    is averaged over the records that have it; absent from all, it is left out of that metric.
    """
    nf, n_bkg = config.n_species, config.n_background
    names = list(config.all_species_names)[: nf + n_bkg]
    first = spinup_years * config.n_dt_per_year
    # A record is a window that ENDS at ``step``: keep it only if it also starts after the
    # spin-up, or spin-up steps would be averaged into the post-spin-up numbers.
    kept = [r for r in records if r.step >= first and r.step - r.window_steps + 1 >= first]
    if not kept:
        raise ValueError(f"no record after the {spinup_years}-year spin-up")
    out: dict[str, dict[str, float]] = {}

    def put(metric: str, values, species=names[:nf]) -> None:
        out[metric] = {sp: float(v) for sp, v in zip(species, values, strict=True)}

    def mean(rows) -> np.ndarray:
        return np.mean(np.stack(rows), axis=0)

    if kept[0].ssb is not None:
        put("ssb", mean([r.ssb for r in kept]))
    if kept[0].yield_n is not None:
        put("yield_n", mean([r.yield_n for r in kept]))
    if kept[0].abundance_by_stage is not None:
        stages = mean([r.abundance_by_stage for r in kept])
        for i, metric in enumerate(("eggs", "juveniles", "adults")):
            put(metric, stages[:, i])
    if kept[0].abundance_by_age is not None:
        put("age0", [np.mean([r.abundance_by_age[sp][0] for r in kept]) for sp in range(nf)])
        put("age1plus", [np.mean([r.abundance_by_age[sp][1:].sum() for r in kept]) for sp in range(nf)])  # fmt: skip
    if kept[0].biomass_by_age is not None:
        put("biomass_age0", [np.mean([r.biomass_by_age[sp][0] for r in kept]) for sp in range(nf)])
        put("biomass_age1plus", [np.mean([r.biomass_by_age[sp][1:].sum() for r in kept]) for sp in range(nf)])  # fmt: skip
    if kept[0].diet_by_species is not None:
        diet = mean(
            [r.diet_by_species for r in kept]
        )  # [predator (focal, background), prey column]
        put("eaten_fish", diet[:, :nf].sum(axis=1), names)
        put("eaten_resources", diet[:, nf:].sum(axis=1), names)
    for metric in INTENSIVE:
        if getattr(kept[0], metric) is None:
            continue
        values = {}
        for sp in range(nf):
            present = [getattr(r, metric)[sp] for r in kept if sp in getattr(r, metric)]
            if present:
                values[names[sp]] = float(np.mean(present))
        out[metric] = values
    return out


def run_arm(name: str, config_path: Path, years: int, n: int, spinup: int, threads: int) -> dict:
    """-> {metric: {species: array(n)}} for one arm, the config's own outputs on."""
    from osmose.config.reader import OsmoseConfigReader

    raw = dict(OsmoseConfigReader().read(config_path))
    raw["simulation.time.nyear"] = str(years)
    reps = []
    for s in range(n):
        seed = SEEDS[name] + s
        if name == "csharp":
            from osmose.engine_cs import CSharpEngine

            config, _grid, records = CSharpEngine(threads=threads)._execute(raw, seed, None)
        else:
            import osmose.engine.simulate as sim
            from osmose.engine import PythonEngine

            config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, seed)
            records = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
        reps.append(reduce_records(records, config, spinup))
    arm: dict[str, dict[str, np.ndarray]] = {}
    for metric in MARGINS:
        species = sorted({sp for r in reps for sp in r.get(metric, {})})
        if species:
            arm[metric] = {
                sp: np.array([r.get(metric, {}).get(sp, np.nan) for r in reps]) for sp in species
            }
    return arm


ARMS = ("python", "python-control", "csharp")


def judge(arms: dict, splits: int) -> dict:
    """Apply the rule to every cell all three arms have in every replicate, and record what the
    rule could not see:

    - ``asymmetric``: a family some arms produced and others did not. A failure: left out
      silently, a run in which one engine omits an output altogether would pass.
    - ``one_sided``: a cell both Python arms have in every replicate and the C# arm does not
      (or the reverse); or a cell that is zero throughout on one engine and not on the other,
      which is what an output the engine never filled looks like. A failure, for the same
      reason. The zero test does not lean on the rule: its log scale reads every value under
      ``COLLAPSE`` as ``COLLAPSE``, so zeros against values under 1 would read as no difference.
    - ``floored``: a tested cell with values between 0 and ``COLLAPSE``. The rule clamps them,
      so it sees less than there is; listed.
    - ``dropped``: a cell a Python arm itself lacks in some replicate (a species absent from a
      mean). Not testable, and reported.
    - ``constant``: a tested cell that is one same number in every replicate of every arm (a
      predator that eats no fish). It counts as equivalent and carries no information; it is
      listed so that the count of equivalent cells is not read as more than it is.
    """
    from osmose.validation.engine_equivalence import COLLAPSE, acceptance, calibrate, compare

    asymmetric = {
        metric: [name for name in ARMS if metric not in arms[name]]
        for metric in MARGINS
        if 0 < sum(metric in arms[name] for name in ARMS) < len(ARMS)
    }
    tested: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in ARMS}
    margins: dict[str, float] = {}
    dropped: list[str] = []
    one_sided: list[str] = []
    constant: list[str] = []
    floored: list[str] = []
    for metric, margin in MARGINS.items():
        if not all(metric in arms[name] for name in ARMS):
            continue
        keep = []
        for sp in sorted(set().union(*(arms[name][metric] for name in ARMS))):
            py, control, cs = (
                sp in arms[name][metric] and bool(np.isfinite(arms[name][metric][sp]).all())
                for name in ARMS
            )
            if py and control and cs:
                filled = [bool(np.any(np.asarray(arms[name][metric][sp]) != 0)) for name in ARMS]
                if filled[0] == filled[1] != filled[2]:  # zero throughout on one engine only
                    one_sided.append(f"{metric}/{sp}")
                else:
                    keep.append(sp)
            elif py == control and py != cs:  # Python agrees with itself; C# differs
                one_sided.append(f"{metric}/{sp}")
            else:
                dropped.append(f"{metric}/{sp}")
        if keep:  # a family with no testable cell is left out: the rule needs a cell to size from
            margins[metric] = margin
            for name in ARMS:
                tested[name][metric] = {sp: arms[name][metric][sp] for sp in keep}
            for sp in keep:
                values = np.concatenate([np.asarray(arms[name][metric][sp]) for name in ARMS])
                if np.all(values == values[0]):
                    constant.append(f"{metric}/{sp}")
                if np.any((values > 0) & (values < COLLAPSE)):
                    floored.append(f"{metric}/{sp}")
    result: dict = {
        "margins": margins, "dropped": dropped, "one_sided": one_sided, "asymmetric": asymmetric,
        "constant": constant, "floored": floored, "verdict": None,
    }  # fmt: skip
    if not margins:
        return result
    a, b, c = (tested[name] for name in ARMS)
    chosen, rates, attribution = calibrate(a, b, n_splits=splits, margins=margins)
    control, treatment = compare(a, b, margins), compare(a, c, margins)
    return result | {
        "chosen": chosen, "rates": rates, "attribution": attribution, "control": control,
        "treatment": treatment, "other": compare(b, c, margins),
        "verdict": acceptance(control, treatment, margins, chosen),
    }  # fmt: skip


def passed(result: dict) -> bool:
    """The rule's verdict, and nothing the rule could not see: at least one testable cell, no
    family only some arms produced, no cell only one engine has."""
    verdict = result["verdict"]
    return (
        verdict is not None
        and verdict.passed
        and not result["asymmetric"]
        and not result["one_sided"]
    )


def report(result: dict) -> None:
    for metric, missing in result["asymmetric"].items():
        print(f"FAMILY NOT IN EVERY ARM: {metric} is missing from {', '.join(missing)}")
    if result["one_sided"]:
        print(f"CELLS ONLY ONE ENGINE HAS: {', '.join(result['one_sided'])}")
    if result["dropped"]:
        print(f"not testable (a Python arm lacks the value): {', '.join(result['dropped'])}")
    if result["floored"]:
        print(
            "values under the rule's floor are read as the floor, so these cells are seen only "
            f"in part: {', '.join(result['floored'])}"
        )
    if result["verdict"] is None:
        print("\nVERDICT: FAIL (no cell is testable)")
        return
    print(
        "\n[calibration] false-fail rate of the rule for identical engines (Python A+B re-split):"
    )
    for rung, rate in result["rates"].items():
        mark = "  <- frozen" if rung == result["chosen"] else ""
        print(f"  slack {rung.eq_slack}, ratio {rung.max_ratio}, variance {rung.var_factor}: {rate:.3f}{mark}")  # fmt: skip
    print(f"  at the frozen rung, splits failing each criterion: {result['attribution']}")
    control, treatment = result["control"], result["treatment"]
    print(f"\n{'metric':16s} {'species':14s} {'d(P-C#)':>8s} {'ci90':>7s} {'eq':>3s} {'d(P-P)':>8s} {'eq':>3s}")  # fmt: skip
    for key in sorted(treatment):
        t, c = treatment[key], control[key]
        print(
            f"{key[0]:16s} {key[1]:14s} {t.d:8.4f} {t.ci:7.4f} {'Y' if t.equivalent else 'n':>3s} "
            f"{c.d:8.4f} {'Y' if c.equivalent else 'n':>3s}"
        )
    v = result["verdict"]
    print(
        f"\nequivalent cells: C# {v.n_equivalent}, control {v.n_equivalent_control} of "
        f"{len(treatment)}; largest |d|: C# {v.max_d:.4f}, control {v.max_d_control:.4f}"
    )
    if result["constant"]:
        print(
            f"of those, {len(result['constant'])} are one same number in every replicate of "
            f"every arm and carry no information: {', '.join(result['constant'])}"
        )
    print(f"VERDICT: {'PASS' if passed(result) else 'FAIL'}")
    for failure in v.failures:
        print(f"  {failure}")
    if v.passed and not passed(result):
        print("  the rule passes on the cells it saw, but an engine lacks outputs the other has")
    by_metric: dict[str, list[float]] = {}
    for (metric, _sp), cell in treatment.items():
        by_metric.setdefault(metric, []).append(abs(cell.d))
    print("\nlargest |d| by family (C# against Python A; control in brackets):")
    for metric, ds in by_metric.items():
        cs = [abs(cell.d) for (m, _sp), cell in control.items() if m == metric]
        print(f"  {metric:16s} {max(ds):.4f}  [{max(cs):.4f}]  over {len(ds)} species")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--config", default="baltic", help=f"a fixture ({', '.join(FIXTURES)}) or a path"
    )
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--spinup-years", type=int, default=5)
    ap.add_argument("--csharp-threads", type=int, default=1)
    ap.add_argument("--calibration-splits", type=int, default=2000)
    ap.add_argument("--json", type=Path, default=None, help="keep every arm's raw replicates")
    ap.add_argument("--rejudge", type=Path, default=None, help="apply the rule to saved replicates")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    if args.rejudge is not None:
        saved = json.loads(args.rejudge.read_text())
        arms = {
            name: {m: {sp: np.asarray(v, dtype=float) for sp, v in by.items()} for m, by in data.items()}
            for name, data in saved["replicates"].items()
        }  # fmt: skip
        print(f"[rejudge] {args.rejudge}: {saved['n']} replicates per arm, {saved['config']}")
    else:
        config = (FIXTURES[args.config] if args.config in FIXTURES else Path(args.config)).resolve()
        if not config.is_file():
            ap.error(f"--config: {args.config!r} is neither a fixture nor a file")
        arms = {}
        for name in ARMS:
            t = time.time()
            arms[name] = run_arm(name, config, args.years, args.n, args.spinup_years, args.csharp_threads)  # fmt: skip
            print(f"[arm] {name}: {args.n} replicates in {time.time() - t:.0f} s", flush=True)
        if args.json:
            args.json.write_text(json.dumps({
                "config": str(config), "years": args.years, "spinup": args.spinup_years, "n": args.n,
                "replicates": {
                    name: {m: {sp: v.tolist() for sp, v in by.items()} for m, by in arm.items()}
                    for name, arm in arms.items()
                },
            }))  # fmt: skip
    result = judge(arms, args.calibration_splits)
    report(result)
    print(f"({time.time() - t0:.0f} s)")
    sys.exit(0 if passed(result) else 1)


if __name__ == "__main__":
    main()
