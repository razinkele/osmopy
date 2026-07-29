#!/usr/bin/env python
"""Evaluate a calibrated Baltic parameter set against ICES biomass envelopes.

Runs ONE full simulation with a given parameter JSON (plus the per-phase
fixed-config the calibrator injects) and reports, per species, the last-10-year
mean biomass against the ``biomass_targets.csv`` ICES envelope: in-range /
overshoot / undershoot and a magnitude factor.

Used to compare the Beverton-Holt (phase 12) baseline against the Shepherd
(phase 13) result on equal footing — same simulation length, same seed — which
the bare objective number cannot do when the two runs were calibrated at
different sim lengths.

Example
-------
    .venv/bin/python scripts/evaluate_calibration_vs_ices.py \
        --params data/baltic/calibration_results/phase12_results.json \
        --mode bh --years 40 --seed 42

    .venv/bin/python scripts/evaluate_calibration_vs_ices.py \
        --params data/baltic/calibration_results/phase13_results.json \
        --mode shepherd --years 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_baltic import (
    BALTIC_CONFIG,
    SPECIES_NAMES,
    load_targets,
    run_simulation,
)
from validate_baltic_vs_ices_sag import (  # reuse snapshot loaders (dependency-free leaf)
    WINDOW_YEARS,
    _load_assessment,
    _load_manifest,
    _load_reference_points,
    _series_by_year,
)

RECRUITMENT_ASSESSED = ("cod", "herring", "sprat", "flounder")


def _species_recruitment_age(species: str) -> str | None:
    """Common ICES recruitment_age (as a string) across a species' mapped stocks, or None if
    the species has no mapped stocks, a stock lacks the age, or the stocks disagree."""
    stocks = _load_manifest()["model_species_to_ices_stocks"].get(species, [])
    if not stocks:
        return None
    ages = set()
    for st in stocks:
        a = _load_reference_points(st).get("recruitment_age")
        if a in (None, ""):
            return None
        ages.add(str(a))
    return ages.pop() if len(ages) == 1 else None


def _ices_recruitment_geomean(species: str) -> tuple[float, float, float] | None:
    """(geomean, min, max) of the per-year SUMMED ICES recruitment across a species' mapped
    stocks over WINDOW_YEARS, keeping only years all stocks report R. None if no clean numeric R.

    Summability is an inferred assumption: the snapshot records SSB units but not recruitment
    units; the mapped stocks' recruitments are all absolute counts on a self-consistent scale.
    """
    if _species_recruitment_age(species) is None:
        return None
    stocks = _load_manifest()["model_species_to_ices_stocks"][species]
    series = [_series_by_year(_load_assessment(st), "recruitment") for st in stocks]
    per_year = [sum(s[y] for s in series) for y in WINDOW_YEARS if all(y in s for s in series)]
    if not per_year:
        return None
    arr = np.asarray(per_year, dtype=float)
    geomean = float(np.exp(np.mean(np.log(arr))))
    return geomean, float(arr.min()), float(arr.max())


def _recruitment_verdict(model_R: float, ices_geomean: float) -> tuple[float, str]:
    """(ratio, verdict). OK if 1/3 <= ratio <= 3 (order-of-magnitude), else FLAG."""
    ratio = model_R / ices_geomean if ices_geomean > 0 else float("inf")
    verdict = "OK" if (1.0 / 3.0) <= ratio <= 3.0 else "FLAG"
    return ratio, verdict


# The 4 FR-calibrated predators (phase 14): cod_west sp0, pikeperch sp5, GreySeal
# sp15 (runtime slot 9), Cormorant sp16 (runtime slot 10). All type-III.
# The background pair shifted up by one when cod_east was appended as focal sp8; keep in
# sync with calibrate_baltic.get_phase14_params (guarded by
# tests/test_baltic_species_index_layout.py).
FR_PREDATOR_SP = (0, 5, 15, 16)
_FR_KEY_SHAPE = "predation.functional.response.shape.sp{i}"
_FR_KEY_HALFSAT = "predation.functional.response.halfsat.sp{i}"


def _apply_mode(base_config: dict[str, str], mode: str, params: dict | None = None) -> None:
    """Inject the fixed-config that the calibrator's ``main()`` applies per phase.

    ``bh``          — base config unchanged (B-H types already live in
                      baltic_param-reproduction.csv for sp0/3/4/5).
    ``shepherd``    — every species switched to Shepherd SR; cod sp0 ssb_half
                      pinned at the Bpa (120 kt), matching ``phase == "13"`` setup.
    ``shepherd-fr`` — everything ``shepherd`` does PLUS injects the phase-14
                      predator functional response: the 4 FR predators
                      (sp0/5/14/15) get ``shape=type3`` and a ``halfsat`` K read
                      from the evaluated params JSON. A missing halfsat key
                      defaults to K=1.0 with a printed note.
    """
    if mode == "bh":
        return
    if mode in ("shepherd", "shepherd-fr"):
        for sp_idx in range(len(SPECIES_NAMES)):
            base_config[f"stock.recruitment.type.sp{sp_idx}"] = "shepherd"
        base_config["stock.recruitment.ssbhalf.sp0"] = "120000"
        if mode == "shepherd-fr":
            params = params or {}
            for i in FR_PREDATOR_SP:
                base_config[_FR_KEY_SHAPE.format(i=i)] = "type3"
                k_key = _FR_KEY_HALFSAT.format(i=i)
                if k_key in params:
                    base_config[k_key] = str(params[k_key])
                else:
                    base_config[k_key] = "1.0"
                    print(
                        f"NOTE: {k_key} absent from params JSON; "
                        f"defaulting FR halfsat sp{i} to K=1.0"
                    )
        return
    raise ValueError(f"unknown mode {mode!r}; expected 'bh', 'shepherd' or 'shepherd-fr'")


def _format_recruitment_section(rows: list[dict]) -> str:
    """Pure formatter for the recruitment table (never runs the engine)."""
    lines = ["\nRecruitment (model vs ICES R geomean, 2018-2022)"]
    lines.append(
        f"  {'species':10s} {'age':>3s} {'model_R':>14s} "
        f"{'ICES_geomean [min-max]':>38s} {'ratio':>7s}  verdict"
    )
    for r in rows:
        if r.get("ices_geomean") is None:
            lines.append(
                f"  {r['species']:10s} {'—':>3s} {'—':>14s} {'—':>38s} {'—':>7s}  {r['reason']}"
            )
        else:
            ref = f"{r['ices_geomean']:,.0f} [{r['ices_min']:,.0f}-{r['ices_max']:,.0f}]"
            model = f"{r['model_R']:,.0f}" if r["model_R"] is not None else "—"
            ratio = f"{r['ratio']:.2f}x" if r["ratio"] is not None else "—"
            verdict = r["verdict"] or "—"
            note = "  (age-0: model reads ~0.4-0.6x low)" if r["age"] == "0" else ""
            lines.append(
                f"  {r['species']:10s} {r['age']:>3s} {model:>14s} {ref:>38s} "
                f"{ratio:>7s}  {verdict}{note}"
            )
    return "\n".join(lines)


def evaluate(params_path: Path, mode: str, n_years: int, seed: int) -> dict:
    from osmose.config.reader import OsmoseConfigReader

    with open(params_path) as f:
        params = json.load(f)["parameters"]

    reader = OsmoseConfigReader()
    base_config = reader.read(BALTIC_CONFIG)
    _apply_mode(base_config, mode, params)

    # The JSON stores real-space (de-log10'd) values; apply directly as overrides.
    # Every params key is re-applied here as a raw (string-cased) override for all
    # modes — harmless because the value equals what _apply_mode already set, and any
    # FR halfsat keys present in the JSON are raw K (not log10).
    overrides = {k.lower(): str(v) for k, v in params.items()}

    base_config["output.abundance.byage.enabled"] = "true"
    recruitment_ages = {
        sp: age for sp in RECRUITMENT_ASSESSED if (age := _species_recruitment_age(sp)) is not None
    }
    stats = run_simulation(
        base_config, overrides, n_years=n_years, seed=seed, recruitment_ages=recruitment_ages
    )
    if not stats:
        raise RuntimeError("simulation failed (run_simulation returned {})")

    # Biomass-only comparison (mean simulated biomass vs. envelope) — exclude
    # catch-type rows so a species' catch band doesn't shadow its biomass row
    # in this species-keyed dict (last-wins on duplicate species).
    targets = {
        t.species: t
        for t in load_targets()
        if getattr(t, "reference_point_type", "biomass") != "catch"
    }
    rows = []
    in_range = 0
    for sp in SPECIES_NAMES:
        mean = stats.get(f"{sp}_mean", 0.0)
        t = targets[sp]
        if mean <= 0:
            status, factor = "EXTINCT", 0.0
        elif mean < t.lower:
            status, factor = "under", mean / t.target
        elif mean > t.upper:
            status, factor = "OVER", mean / t.target
        else:
            status, factor = "in-range", mean / t.target
            in_range += 1
        rows.append(
            {
                "species": sp,
                "mean_tonnes": mean,
                "lower": t.lower,
                "upper": t.upper,
                "target": t.target,
                "weight": t.weight,
                "status": status,
                "factor_vs_target": factor,
                "cv": stats.get(f"{sp}_cv", 0.0),
            }
        )
    recruitment = []
    for sp in RECRUITMENT_ASSESSED:
        age = _species_recruitment_age(sp)
        geo = _ices_recruitment_geomean(sp) if age is not None else None
        if age is None or geo is None:
            reason = (
                "no clean ICES R (eastern index + age mismatch 0 vs 1)"
                if sp == "cod"
                else "no clean ICES R (none reported)"
            )
            recruitment.append(
                {
                    "species": sp,
                    "age": None,
                    "model_R": None,
                    "ices_geomean": None,
                    "ices_min": None,
                    "ices_max": None,
                    "ratio": None,
                    "verdict": None,
                    "reason": reason,
                }
            )
            continue
        geomean, gmin, gmax = geo
        model_R = stats.get(f"{sp}_recruitment_mean")
        if model_R is None:
            ratio, verdict = None, None
        else:
            ratio, verdict = _recruitment_verdict(model_R, geomean)
        recruitment.append(
            {
                "species": sp,
                "age": age,
                "model_R": model_R,
                "ices_geomean": geomean,
                "ices_min": gmin,
                "ices_max": gmax,
                "ratio": ratio,
                "verdict": verdict,
                "reason": None,
            }
        )

    return {
        "params_path": str(params_path),
        "mode": mode,
        "n_years": n_years,
        "seed": seed,
        "in_range_count": in_range,
        "species": rows,
        "recruitment": recruitment,
    }


def _print_report(result: dict) -> None:
    print(
        f"\n=== {result['mode'].upper()} | {Path(result['params_path']).name} "
        f"| {result['n_years']}y seed={result['seed']} ==="
    )
    hdr = f"{'species':12s} {'mean (t)':>14s} {'range (t)':>22s} {'status':>9s} {'×target':>8s} {'CV':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in result["species"]:
        rng = f"{r['lower']:,.0f}–{r['upper']:,.0f}"
        print(
            f"{r['species']:12s} {r['mean_tonnes']:>14,.0f} {rng:>22s} "
            f"{r['status']:>9s} {r['factor_vs_target']:>8.2f} {r['cv']:>5.2f}"
        )
    print(f"\nIN ICES RANGE: {result['in_range_count']} / {len(result['species'])}")
    print(_format_recruitment_section(result.get("recruitment", [])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, required=True, help="calibration result JSON")
    ap.add_argument("--mode", choices=["bh", "shepherd", "shepherd-fr"], required=True)
    ap.add_argument("--years", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, default=None, help="optional output JSON path")
    args = ap.parse_args()

    result = evaluate(args.params, args.mode, args.years, args.seed)
    _print_report(result)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
