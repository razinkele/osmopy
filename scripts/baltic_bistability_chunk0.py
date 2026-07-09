"""Chunk 0 de-risk harness (v3). See docs/superpowers/plans/2026-07-08-baltic-chunk0-bistability-derisk-v3.md.

One shared classify_state turns biomass into an ICES-band state for BOTH experiments.
Near-zero is a STABLE 'collapsed' band (checked before the stationarity gate, so
run_simulation's cv=10.0 zero-mean sentinel cannot hide an extinction). Verdicts are
count-based over band transitions; _failed seeds are excluded; per-seed states are
aggregated to a consensus band or an explicit 'seed-split'. Unit-tested with fake runners;
real Baltic runs are CLI-only (Python engine, minutes each).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagnostics"
_DEFAULT_SCALES = [0.03, 0.1, 0.3, 0.5, 1.0]
_DEFAULT_SEEDS = [0, 1, 2]
_PLANKTON_GROUPS = (8, 10, 11, 12)
_SEEDING_WINDOW_Y = 4
_NONBANDS = ("failed", "undetermined")
_ENABLE_KEY = "module.population.initialisation.enabled"  # canonical warm-start flag
_VALID_BANDS = ("collapsed", "low", "in_range", "overshoot")  # determinate bands


def is_stationary(cv: float, trend: float, cv_max: float = 0.30, trend_max: float = 0.05) -> bool:
    return cv <= cv_max and trend <= trend_max


def classify_state(mean, cv, trend, target, lower, upper, collapse_frac: float = 0.05) -> str:
    """ICES-band state. Near-zero is a stable 'collapsed' (checked FIRST so the cv=10.0
    zero-mean sentinel from run_simulation cannot mask an extinction)."""
    if mean < collapse_frac * target:
        return "collapsed"
    if not is_stationary(cv, trend):
        return "undetermined"
    if mean < lower:
        return "low"
    if mean > upper:
        return "overshoot"
    return "in_range"


def bistability_gap(a: float, b: float) -> float:
    return abs(a - b) / (max(a, b) + 1.0)


def basins_differ(rich_state, poor_state, gap, gap_thresh: float = 0.5) -> bool:
    _bands = ("collapsed", "low", "in_range", "overshoot")
    if rich_state not in _bands or poor_state not in _bands:
        return False
    if rich_state == poor_state == "collapsed":
        return False
    if rich_state == poor_state == "overshoot":
        return False
    if rich_state != poor_state:
        return True
    return gap >= gap_thresh


def aggregate_states(states) -> str:
    """Consensus band if ALL valid (non-failed, non-undetermined) seeds agree; 'seed-split'
    if valid seeds disagree; 'undetermined' if no seed is valid."""
    valid = [s for s in states if s not in _NONBANDS]
    if not valid:
        return "undetermined"
    return valid[0] if len(set(valid)) == 1 else "seed-split"


def species_states(stats: dict, targets) -> dict:
    """Per-species ICES-band state from one successful run (cv/trend gated)."""
    out = {}
    for t in targets:
        mean = stats.get(f"{t.species}_mean")
        if mean is None:
            out[t.species] = "undetermined"
            continue
        cv = float(stats.get(f"{t.species}_cv", 10.0))
        trend = float(stats.get(f"{t.species}_trend", 1.0))
        out[t.species] = classify_state(
            float(mean), cv, trend, float(t.target), float(t.lower), float(t.upper)
        )
    return out


def accessibility_transition(
    base_states: dict,
    low_states: dict,
    targets,
    weight_threshold: float = 1.0,
    collapse_veto_weight: float = 0.3,
) -> dict:
    """Band transitions baseline->lowered. Improvement counters gate only well-assessed stocks
    (weight >= weight_threshold); the absolute collapse veto (collapsed_lowered) fires when ANY
    stock with weight >= collapse_veto_weight ends 'collapsed' in the lowered arm."""
    bands = ("collapsed", "low", "in_range", "overshoot")
    high = {"in_range", "overshoot"}
    below = {"low", "collapsed"}
    c = {
        "in_range_base": 0,
        "in_range_low": 0,
        "overshoot_base": 0,
        "overshoot_low": 0,
        "new_undershoot": 0,
        "undetermined": 0,
        "gated_species": 0,
        "collapsed_lowered": 0,
    }
    for t in targets:
        w = float(getattr(t, "weight", 1.0))
        low = low_states.get(t.species, "undetermined")
        if w >= collapse_veto_weight and low == "collapsed":
            c["collapsed_lowered"] += 1
        if w < weight_threshold:
            continue
        c["gated_species"] += 1
        b = base_states.get(t.species, "undetermined")
        if b not in bands or low not in bands:
            c["undetermined"] += 1
            continue
        c["in_range_base"] += b == "in_range"
        c["in_range_low"] += low == "in_range"
        c["overshoot_base"] += b == "overshoot"
        c["overshoot_low"] += low == "overshoot"
        if b in high and low in below:
            c["new_undershoot"] += 1
    return c


def accessibility_verdict(t: dict) -> tuple[bool, str]:
    if t["undetermined"] > 0:
        return False, (
            f"PROVISIONAL — {t['undetermined']} species non-stationary/absent in one arm; "
            f"verdict withheld (raise --years or --seeds)."
        )
    if t.get("collapsed_lowered", 0) > 0:
        return False, (
            f"WEB STILL BROKEN — {t['collapsed_lowered']} weight-relevant stock(s) are collapsed "
            f"in the lowered arm. Accessibility may relieve overshoot but does not fix the "
            f"collapse; NOT a green light for A1 on its own."
        )
    if t["new_undershoot"] > 0:
        return False, (
            f"COLLAPSES the web, not relaxes it — {t['new_undershoot']} well-assessed species "
            f"dropped from in-range/overshoot to below-lower. NOT evidence for A1."
        )
    relaxed = t["in_range_low"] > t["in_range_base"] and t["overshoot_low"] <= t["overshoot_base"]
    if relaxed:
        return True, (
            f"Relaxes over-production toward ICES bands (in-range {t['in_range_base']} -> "
            f"{t['in_range_low']}; overshooters {t['overshoot_base']} -> {t['overshoot_low']}; "
            f"no weight-relevant stock collapsed). A1 is a real lever."
        )
    return False, (
        f"No clean relaxation (in-range {t['in_range_base']} -> {t['in_range_low']}; "
        f"overshooters {t['overshoot_base']} -> {t['overshoot_low']}). Reconsider A1."
    )


def larva_scale_override(scale: float, base_rates: dict) -> dict:
    return {f"mortality.additional.larva.rate.sp{i}": str(r * scale) for i, r in base_rates.items()}


def accessibility_override(value: float, resource_indices=_PLANKTON_GROUPS) -> dict:
    return {f"species.accessibility2fish.sp{i}": str(value) for i in resource_indices}


def cod_rich_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    # Vary ONLY the cod SEEDING BIOMASS; population.seeding.year.max is a GLOBAL key that
    # truncates the seeding window for ALL species to `window` years in both arms.
    return {"population.seeding.biomass.sp0": "300000", "population.seeding.year.max": str(window)}


def cod_poor_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    return {"population.seeding.biomass.sp0": "1000", "population.seeding.year.max": str(window)}


def warmstart_override(enabled: bool) -> dict:
    """Merge-in override that turns the warm-start standing-stock init ON (canonical flag).
    Empty when off, so an egg-only run's overrides stay byte-identical."""
    return {_ENABLE_KEY: "true"} if enabled else {}


def cod_dominated_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    """Cod-dominated standing-stock IC: cod at the ICES upper band, clupeids suppressed."""
    return {
        "population.seeding.biomass.sp0": "250000",  # cod, ICES upper
        "population.seeding.biomass.sp1": "800000",  # herring, lower
        "population.seeding.biomass.sp2": "600000",  # sprat, suppressed
        "population.seeding.year.max": str(window),
    }


def clupeid_dominated_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    """Clupeid-dominated (sprat-dominated) standing-stock IC: cod a remnant/invader,
    herring + sprat at target/upper — the real post-1990 Baltic regime."""
    return {
        "population.seeding.biomass.sp0": "1000",  # cod, remnant/invader
        "population.seeding.biomass.sp1": "1500000",  # herring, target
        "population.seeding.biomass.sp2": "2500000",  # sprat, upper
        "population.seeding.year.max": str(window),
    }


def safe_run(runner, config, overrides, n_years, seed) -> dict:
    """Model call; `_failed` sentinel (distinct from a real cod_mean==0) on crash or empty output."""
    try:
        stats = runner(config, overrides, n_years, seed)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not abort the whole grid
        return {"_failed": True, "_error": repr(exc)}
    if not stats or "cod_mean" not in stats:
        return {"_failed": True, "_error": "empty or partial stats (no cod_mean)"}
    return stats


def _cod_state(stats: dict, bands: dict) -> tuple[str, float]:
    if stats.get("_failed"):
        return "failed", 0.0
    mean = float(stats.get("cod_mean", 0.0))
    st = classify_state(
        mean,
        float(stats.get("cod_cv", 10.0)),
        float(stats.get("cod_trend", 1.0)),
        bands["target"],
        bands["lower"],
        bands["upper"],
    )
    return st, mean


def _median_valid(states, means) -> float:
    vals = [m for s, m in zip(states, means) if s not in _NONBANDS]
    return statistics.median(vals) if vals else 0.0


def clupeid_axis(runs, clupeid_targets) -> tuple[float, bool]:
    """Clupeid regime signal for ONE arm: median summed herring+sprat biomass over non-failed
    seeds, plus a validity flag. Valid iff BOTH stocks aggregate to a determinate band across
    seeds (stationary + seed-consensus). Summing sidesteps banding two stocks with different
    ICES ranges; validity gating mirrors the cod-axis stationarity discipline."""
    bands = {t.species: [] for t in clupeid_targets}
    sums = []
    for st in runs:
        if st.get("_failed"):
            continue
        total = 0.0
        for t in clupeid_targets:
            mean = float(st.get(f"{t.species}_mean", 0.0))
            total += mean
            bands[t.species].append(
                classify_state(
                    mean,
                    float(st.get(f"{t.species}_cv", 10.0)),
                    float(st.get(f"{t.species}_trend", 1.0)),
                    float(t.target),
                    float(t.lower),
                    float(t.upper),
                )
            )
        sums.append(total)
    if not sums:
        return 0.0, False
    valid = all(aggregate_states(bands[t.species]) in _VALID_BANDS for t in clupeid_targets)
    return statistics.median(sums), valid


def _partial(points: list) -> dict:
    return {
        "points": points,
        "bistable": None,
        "bistable_scales": [],
        "seed_split_scales": [],
        "undetermined_scales": [],
        "establishment_fraction": None,
        "trustworthy": None,
        "verdict": "incomplete",
        "complete": False,
    }


def run_bistability_point(
    scale, base_config, base_rates, cod_bands, seeds, *, runner, n_years
) -> dict:
    driver = larva_scale_override(scale, base_rates)
    rich_states, poor_states, rich_means, poor_means = [], [], [], []
    for seed in seeds:
        r = safe_run(runner, base_config, {**driver, **cod_rich_seeding()}, n_years, seed)
        p = safe_run(runner, base_config, {**driver, **cod_poor_seeding()}, n_years, seed)
        rs, rm = _cod_state(r, cod_bands)
        ps, pm = _cod_state(p, cod_bands)
        rich_states.append(rs)
        poor_states.append(ps)
        rich_means.append(rm)
        poor_means.append(pm)
    rich_agg = aggregate_states(
        rich_states
    )  # consensus band (all valid seeds agree) or 'seed-split'
    poor_agg = aggregate_states(poor_states)
    gap = bistability_gap(
        _median_valid(rich_states, rich_means), _median_valid(poor_states, poor_means)
    )
    established = rich_agg in ("low", "in_range", "overshoot")
    if rich_agg == "seed-split" or poor_agg == "seed-split":
        outcome = "seed-split"
    elif rich_agg == "undetermined" or poor_agg == "undetermined":
        outcome = "undetermined"
    elif basins_differ(rich_agg, poor_agg, gap):
        outcome = "bistable"
    else:
        outcome = "same-basin"
    return {
        "scale": scale,
        "rich_state": rich_agg,
        "poor_state": poor_agg,
        "per_seed_rich": rich_states,
        "per_seed_poor": poor_states,
        "rich_cod_median": _median_valid(rich_states, rich_means),
        "poor_cod_median": _median_valid(poor_states, poor_means),
        "gap": gap,
        "established": established,
        "outcome": outcome,
        "bistable": outcome == "bistable",
    }


def run_bistability_sweep(
    scales, base_config, base_rates, cod_bands, seeds, *, runner, n_years, on_point=None
) -> dict:
    points = []
    for s in scales:
        pt = run_bistability_point(
            s, base_config, base_rates, cod_bands, seeds, runner=runner, n_years=n_years
        )
        points.append(pt)
        if on_point is not None:
            on_point(_partial(points))
    bistable = [p["scale"] for p in points if p["outcome"] == "bistable"]
    seed_split = [p["scale"] for p in points if p["outcome"] == "seed-split"]
    undet = [p["scale"] for p in points if p["outcome"] == "undetermined"]
    est_frac = sum(p["established"] for p in points) / len(points) if points else 0.0
    trustworthy = est_frac >= 0.5
    if not trustworthy:
        split_note = (
            f" (a basin split WAS seen at scale(s) {bistable} — treat as tentative)"
            if bistable
            else ""
        )
        verdict = (
            f"INSTRUMENT-LIMITED — cod-rich established a non-collapsed stock at only "
            f"{est_frac:.0%} of scales, so egg-seeding (not the biology) may set the outcome"
            f"{split_note}. No MONOSTABLE conclusion; a definitive test needs the warm-start "
            f"primitive (Task 7)."
        )
    elif bistable:
        verdict = (
            f"BISTABLE (conservative) — different cod basins from the two ICs at larva-scale(s) "
            f"{bistable}. Egg-only ICs + Beverton-Holt bias this test toward MONOSTABLE, so a "
            f"positive result is strong. Confirm with a warm-start standing IC (Task 7)."
        )
    elif seed_split:
        verdict = (
            f"AMBIGUOUS — seed-split (per-seed basin disagreement) at scale(s) {seed_split}; "
            f"near a tipping point. Re-run with more --seeds before concluding."
        )
    else:
        verdict = (
            f"MONOSTABLE by this CONSERVATIVE method — no basin split at any established scale "
            f"(undetermined: {undet}). Cannot rule out bistability (egg-only ICs, and the "
            f"single-cod-axis ICs omit the sprat-dominated start); add the warm-start primitive "
            f"(Task 7) for a definitive test, or proceed to Chunks C & A2 to CREATE a self-locking "
            f"bistability. Read the rich/poor cod response curve."
        )
    return {
        "points": points,
        "bistable": bool(bistable) and trustworthy,
        "bistable_scales": bistable,
        "seed_split_scales": seed_split,
        "undetermined_scales": undet,
        "establishment_fraction": est_frac,
        "trustworthy": trustworthy,
        "verdict": verdict,
        "complete": True,
    }


def run_accessibility_ab(
    base_config, targets, seeds, *, runner, n_years, low_value: float = 0.1
) -> dict:
    per_seed = []
    n_failed = 0
    for seed in seeds:
        b = safe_run(runner, base_config, {}, n_years, seed)
        low = safe_run(runner, base_config, accessibility_override(low_value), n_years, seed)
        if b.get("_failed") or low.get("_failed"):
            n_failed += 1
            continue
        per_seed.append((species_states(b, targets), species_states(low, targets)))
    if not per_seed:
        return {
            "relaxed": False,
            "n_failed": n_failed,
            "low_value": low_value,
            "verdict": (
                f"INSTRUMENT-FAILED — all {len(seeds)} seeds crashed or returned empty "
                f"output; no accessibility verdict (instrument failure, not an ecological signal)."
            ),
        }
    base_agg = {t.species: aggregate_states([ps[0][t.species] for ps in per_seed]) for t in targets}
    low_agg = {t.species: aggregate_states([ps[1][t.species] for ps in per_seed]) for t in targets}
    transition = accessibility_transition(base_agg, low_agg, targets)
    relaxed, verdict = accessibility_verdict(transition)
    return {
        "baseline_states": base_agg,
        "lowered_states": low_agg,
        "transition": transition,
        "low_value": low_value,
        "n_failed": n_failed,
        "relaxed": relaxed,
        "verdict": verdict,
    }


def read_base_config() -> dict:
    from calibrate_baltic import BALTIC_CONFIG
    from osmose.config.reader import OsmoseConfigReader

    return OsmoseConfigReader().read(str(BALTIC_CONFIG))


def read_base_larva_rates(base_config: dict, n_focal: int = 8) -> dict:
    rates = {}
    for i in range(n_focal):
        key = f"mortality.additional.larva.rate.sp{i}"
        if key in base_config:
            rates[i] = float(base_config[key])  # post-4.4.1 migration => per-dt (~15 for cod)
    return rates


def read_cod_bands(targets) -> dict:
    t = next(x for x in targets if x.species == "cod")
    return {"target": float(t.target), "lower": float(t.lower), "upper": float(t.upper)}


def _default_runner(config, overrides, n_years, seed):
    from calibrate_baltic import run_simulation

    return run_simulation(config, overrides, n_years=n_years, seed=seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic Chunk 0 de-risk experiments (v3)")
    ap.add_argument(
        "--experiment", choices=["bistability", "accessibility", "both"], default="both"
    )
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS)
    ap.add_argument("--scales", type=float, nargs="+", default=_DEFAULT_SCALES)
    ap.add_argument("--low-accessibility", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    from calibrate_baltic import load_targets

    seeds = [args.seeds[0]] if args.smoke else args.seeds
    scales = [1.0, 0.1] if args.smoke else args.scales
    years = 3 if args.smoke else args.years

    base_config = read_base_config()
    base_rates = read_base_larva_rates(base_config)
    targets = load_targets()
    cod_bands = read_cod_bands(targets)
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"base larva rates (post-migration, per-dt): {base_rates}")

    if args.experiment in ("bistability", "both"):
        out_path = _DIAG_DIR / "baltic_chunk0_bistability.json"
        result = run_bistability_sweep(
            scales,
            base_config,
            base_rates,
            cod_bands,
            seeds,
            runner=_default_runner,
            n_years=years,
            on_point=lambda payload: out_path.write_text(json.dumps(payload, indent=2)),
        )
        print("\n=== BISTABILITY (conservative) ===")
        print(
            f"establishment fraction (cod-rich reaches a non-collapsed stock): {result['establishment_fraction']:.0%}"
        )
        for p in result["points"]:
            print(
                f"  larva x{p['scale']:<5} rich={p['rich_state']:<11} poor={p['poor_state']:<11} "
                f"gap={p['gap']:.3f} rich_seeds={p['per_seed_rich']} -> {p['outcome']}"
            )
        print(f"\nVERDICT: {result['verdict']}")
        out_path.write_text(json.dumps(result, indent=2))

    if args.experiment in ("accessibility", "both"):
        result = run_accessibility_ab(
            base_config,
            targets,
            seeds,
            runner=_default_runner,
            n_years=years,
            low_value=args.low_accessibility,
        )
        print("\n=== ACCESSIBILITY A/B ===")
        print(f"  n_failed_seeds={result.get('n_failed')}")
        print(f"  baseline_states={result.get('baseline_states')}")
        print(f"  lowered_states={result.get('lowered_states')}")
        print(f"\nVERDICT: {result['verdict']}")
        (_DIAG_DIR / "baltic_chunk0_accessibility_ab.json").write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
