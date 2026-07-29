#!/usr/bin/env python
"""FR-on vs FR-off realized-predation process diagnostic (PR-B / phase 14).

For the 4 phase-14 FR predators — cod_west sp0, pikeperch sp5, GreySeal sp15
(runtime slot 9), Cormorant sp16 (runtime slot 10) — this script measures their
REALIZED predation mortality on each focal prey under two engine variants:

  FR-OFF : functional-response shape ``type1`` (linear, no refuge) on all 4.
  FR-ON  : functional-response shape ``type3`` + a half-saturation K on all 4.

Realized predation mortality of predator p on prey q over a window is

    M_pq  ≈  (Σ biomass of q eaten by p over the window)
             / (mean biomass of q over the window)          [per year]

aggregated from the engine's per-step diet matrix via
``aggregate_diet_all_predators`` (which, unlike ``aggregate_diet_by_species``,
keeps the background-predator rows 8/9).

It runs both variants across several seeds and reports, per (predator, prey),
the FR-off mortality, FR-on mortality, and the delta (FR-on − FR-off) as
``mean ± std across seeds``. Pairs whose ``|mean delta| > 2·std`` exceed the
seed noise band and are flagged — this is the falsifiable basis for PR-B's
verdict: does a type-III refuge measurably reduce a predator's realized
predation on its prey?

The frozen base is the phase-13 Shepherd config (same injection as the eval
script's ``--mode shepherd-fr``): all 8 species on Shepherd SR, cod ssb_half
pinned at the Bpa, the 24 mortality/fishing params + 16 SR params frozen from
the params JSON.

Example
-------
    PYTHONPATH=. .venv/bin/python scripts/fr_process_diagnostic.py \
        --params data/baltic/calibration_results/phase14_results.json \
        --seeds 3 --years 40 --window 10 --halfsat-from-params

Smoke (tiny — machinery only, not a real refuge measurement):
    PYTHONPATH=. .venv/bin/python scripts/fr_process_diagnostic.py \
        --params data/baltic/calibration_results/phase13_results.json \
        --seeds 1 --years 3 --window 1 --k 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import numpy as np

# scripts/ is on sys.path[0] when run as `python scripts/fr_process_diagnostic.py`,
# so the sibling calibrate_baltic + evaluate modules import directly (mirrors how
# evaluate_calibration_vs_ices.py imports calibrate_baltic).
from calibrate_baltic import BALTIC_CONFIG, SPECIES_NAMES
from evaluate_calibration_vs_ices import FR_PREDATOR_SP, _apply_mode

# Diet matrix width for the diagnostic: 8 focal + 2 background + 6 resources.
# Production hardwires width = n_species + n_background (=10), dropping the
# pure-resource columns; we widen so the matrix is complete. We only normalise
# against the 8 FOCAL prey (which have biomass in results.biomass()).
_DIET_WIDTH = 16
_N_BACKGROUND = 2  # GreySeal, Cormorant
_N_FOCAL = len(SPECIES_NAMES)
_N_TOTAL = _N_FOCAL + _N_BACKGROUND

# Config-space FR species -> runtime diet-row slot. Focal species map to their own index;
# background species land above the focal block at n_focal + background_index, so both the
# config index and the slot shift when a focal species is added.
_BACKGROUND_SP = tuple(FR_PREDATOR_SP[-_N_BACKGROUND:])  # (GreySeal, Cormorant)
_SP_TO_SLOT = {sp: sp for sp in FR_PREDATOR_SP if sp < _N_FOCAL}
_SP_TO_SLOT.update({sp: _N_FOCAL + i for i, sp in enumerate(_BACKGROUND_SP)})
PREDATOR_SLOTS = {sp: _SP_TO_SLOT[sp] for sp in FR_PREDATOR_SP}
_SP_NAME = {0: "cod_west", 5: "pikeperch", _BACKGROUND_SP[0]: "GreySeal", _BACKGROUND_SP[1]: "Cormorant"}
PREDATOR_LABEL = {_SP_TO_SLOT[sp]: f"{_SP_NAME[sp]}(sp{sp})" for sp in FR_PREDATOR_SP}


def realized_mortality(
    diet_eaten: np.ndarray,
    prey_biomass: np.ndarray,
    n_years: float,
) -> np.ndarray:
    """Realized predation mortality of a predator on each prey (pure function).

    M_q = (total biomass of prey q eaten over the window)
          / (mean standing biomass of prey q over the window) / n_years

    Args:
        diet_eaten: shape (n_prey,) — total biomass of each prey eaten by the
            predator, summed over the window.
        prey_biomass: shape (n_prey,) — mean standing biomass of each prey over
            the same window.
        n_years: number of years in the window (to express the rate per year).

    Returns:
        shape (n_prey,) per-year realized predation mortality. Prey with
        non-positive biomass yield 0.0 (no defined rate).
    """
    diet_eaten = np.asarray(diet_eaten, dtype=float)
    prey_biomass = np.asarray(prey_biomass, dtype=float)
    out = np.zeros_like(diet_eaten, dtype=float)
    n_years = max(float(n_years), 1e-9)
    valid = prey_biomass > 0
    out[valid] = diet_eaten[valid] / prey_biomass[valid] / n_years
    return out


def resolve_halfsat(params: dict, predators: tuple[int, ...], default_k: float) -> dict[int, float]:
    """Resolve per-predator half-saturation K values from a params dict.

    For each predator species index ``sp`` in ``predators``, look up
    ``predation.functional.response.halfsat.sp{sp}`` in ``params``; fall back to
    ``default_k`` when the key is absent.

    Args:
        params: flat dict of config-key -> value (the ``"parameters"`` sub-dict
            from a phase-14 results JSON, or any dict with the same shape).
        predators: tuple of config-space species indices (e.g. ``(0, 5, 14, 15)``).
        default_k: fallback K for predators whose key is absent from ``params``.

    Returns:
        dict mapping each sp index -> float K.
    """
    result: dict[int, float] = {}
    for sp in predators:
        key = f"predation.functional.response.halfsat.sp{sp}"
        result[sp] = float(params[key]) if key in params else default_k
    return result


def _build_base_config(params: dict, *, fr_on: bool, k: float) -> dict[str, str]:
    """Construct a frozen phase-13 Shepherd config, then inject the FR variant.

    FR-OFF: the 4 predators get shape=type1 (no halfsat).
    FR-ON : the 4 predators get shape=type3 + per-predator halfsat resolved via
            ``resolve_halfsat`` (reads individual K from params when available,
            falls back to ``k`` for absent keys).
    """
    from osmose.config.reader import OsmoseConfigReader

    reader = OsmoseConfigReader()
    cfg = reader.read(BALTIC_CONFIG)

    # shepherd-fr sets shepherd SR on all 8 species + ssb_half pin + (for FR-ON)
    # type3 + halfsat from params. We reuse it for the ON variant, then override
    # the shape/halfsat for the OFF variant.
    _apply_mode(cfg, "shepherd-fr", params)

    # Apply the frozen params (SR + mortality + fishing) as real-space overrides.
    for key, val in params.items():
        cfg[key.lower()] = str(val)

    # Re-assert the FR variant AFTER the param overrides (params JSON may carry
    # halfsat keys that would otherwise leak into the FR-OFF variant).
    # For FR-ON: use per-predator K resolved from params (with --k as fallback)
    # so that calibrated values are not clobbered by a uniform write.
    per_predator_k = resolve_halfsat(params, FR_PREDATOR_SP, default_k=k)
    for sp in FR_PREDATOR_SP:
        shape_key = f"predation.functional.response.shape.sp{sp}"
        hs_key = f"predation.functional.response.halfsat.sp{sp}"
        if fr_on:
            cfg[shape_key] = "type3"
            cfg[hs_key] = str(per_predator_k[sp])
        else:
            cfg[shape_key] = "type1"
            cfg.pop(hs_key, None)

    return cfg


def _run_with_diet(cfg: dict[str, str], n_years: int, seed: int):
    """Run one sim with width-16 diet tracking; return (window_diet, biomass_df).

    Reuses the width-16 monkeypatch pattern from
    tests/test_engine_functional_response.py::_run_baltic_short_with_diet:
      - widen ``enable_diet_tracking`` so resource columns survive,
      - capture each step's RAW wide diet matrix + species_id, then truncate the
        aggregated result back to the production width so the run completes.

    Returns:
        window_diet: (n_total, n_prey_cols) per-predator-species diet summed over
            ALL captured steps (caller selects the window via n_years/window).
        results: OsmoseResults (for biomass()).
        n_steps: number of captured diet steps.
    """
    import osmose.engine.output as _output
    import osmose.engine.processes.predation as _pred
    from osmose.engine import PythonEngine
    from osmose.engine.output import aggregate_diet_all_predators

    cfg = dict(cfg)
    cfg["simulation.time.nyear"] = str(n_years)
    cfg["output.diet.composition.enabled"] = "true"

    real_enable = _pred.enable_diet_tracking

    def _wide_enable(n_schools, n_species, ctx=None):
        return real_enable(n_schools, _DIET_WIDTH, ctx=ctx)

    real_agg = _output.aggregate_diet_by_species
    step_aggs: list[np.ndarray] = []

    def _capturing_agg(diet_matrix, species_id, n_pred_species):
        dm = np.asarray(diet_matrix)
        sid = np.asarray(species_id)
        # Per-step per-predator-species diet INCLUDING background slots 8/9.
        step_aggs.append(aggregate_diet_all_predators(dm, sid, n_total=_N_TOTAL))
        result = real_agg(diet_matrix, species_id, n_pred_species)
        prod_width = n_pred_species + 2  # n_species + n_background
        if result.shape[1] > prod_width:
            return result[:, :prod_width]
        return result

    with mock.patch.object(_pred, "enable_diet_tracking", _wide_enable):
        with mock.patch.object(_output, "aggregate_diet_by_species", _capturing_agg):
            results = PythonEngine().run_in_memory(cfg, seed=seed)

    if not step_aggs:
        raise RuntimeError("diet aggregation hook was never invoked (no diet captured)")
    return step_aggs, results


def _mortality_for_seed(params, *, fr_on, k, n_years, window, seed):
    """Run one variant for one seed; return (n_pred, n_focal_prey) mortality matrix."""
    cfg = _cfg_cache(params, fr_on, k)
    step_aggs, results = _run_with_diet(cfg, n_years, seed)

    bio = results.biomass()
    results.close()

    # Window = last `window` years of biomass + the matching diet steps.
    window = min(window, n_years)
    bio_window = bio.iloc[-window:] if len(bio) > window else bio
    prey_biomass = np.array(
        [float(bio_window[sp].mean()) if sp in bio_window.columns else 0.0 for sp in SPECIES_NAMES]
    )

    # Diet steps are recorded once per timestep; take the last window-fraction.
    n_steps = len(step_aggs)
    steps_per_year = max(n_steps // max(n_years, 1), 1)
    window_steps = min(window * steps_per_year, n_steps)
    window_diet = np.sum(step_aggs[-window_steps:], axis=0)  # (n_total, n_prey_cols)

    # Realized mortality of each FR predator on each FOCAL prey (cols 0..7).
    mort = np.zeros((len(FR_PREDATOR_SP), _N_FOCAL))
    for pi, sp in enumerate(FR_PREDATOR_SP):
        slot = PREDATOR_SLOTS[sp]
        eaten = window_diet[slot, :_N_FOCAL]
        mort[pi] = realized_mortality(eaten, prey_biomass, n_years=window)
    return mort


# Tiny per-call config cache so FR-on and FR-off don't both re-read + re-inject
# the (identical) base every seed. Keyed by (id(params), fr_on, k).
_CFG_CACHE: dict = {}


def _cfg_cache(params: dict, fr_on: bool, k: float) -> dict[str, str]:
    key = (id(params), fr_on, round(float(k), 9))
    cfg = _CFG_CACHE.get(key)
    if cfg is None:
        cfg = _build_base_config(params, fr_on=fr_on, k=k)
        _CFG_CACHE[key] = cfg
    return dict(cfg)


def run_diagnostic(
    params: dict, *, k: float, n_years: int, window: int, seeds: int, base_seed: int = 42
) -> dict:
    """Run the FR-off/on diagnostic across seeds and assemble the summary dict."""
    n_pred = len(FR_PREDATOR_SP)
    off = np.zeros((seeds, n_pred, _N_FOCAL))
    on = np.zeros((seeds, n_pred, _N_FOCAL))

    for s in range(seeds):
        seed = base_seed + s
        print(f"  seed {seed}: FR-OFF run ...", flush=True)
        off[s] = _mortality_for_seed(
            params, fr_on=False, k=k, n_years=n_years, window=window, seed=seed
        )
        print(f"  seed {seed}: FR-ON  run ...", flush=True)
        on[s] = _mortality_for_seed(
            params, fr_on=True, k=k, n_years=n_years, window=window, seed=seed
        )

    delta = on - off  # (seeds, n_pred, n_focal)
    off_mean, off_std = off.mean(0), off.std(0)
    on_mean, on_std = on.mean(0), on.std(0)
    d_mean, d_std = delta.mean(0), delta.std(0)

    pairs = []
    for pi, sp in enumerate(FR_PREDATOR_SP):
        slot = PREDATOR_SLOTS[sp]
        for qi, prey in enumerate(SPECIES_NAMES):
            exceeds = abs(d_mean[pi, qi]) > 2.0 * d_std[pi, qi]
            pairs.append(
                {
                    "predator_sp": sp,
                    "predator_slot": slot,
                    "predator": PREDATOR_LABEL[slot],
                    "prey": prey,
                    "fr_off_mort_mean": float(off_mean[pi, qi]),
                    "fr_off_mort_std": float(off_std[pi, qi]),
                    "fr_on_mort_mean": float(on_mean[pi, qi]),
                    "fr_on_mort_std": float(on_std[pi, qi]),
                    "delta_mean": float(d_mean[pi, qi]),
                    "delta_std": float(d_std[pi, qi]),
                    "exceeds_noise": bool(exceeds),
                }
            )
    return {
        "k_fallback": k,  # default used only for predators absent from params
        "halfsat_per_predator": {
            f"sp{sp}": v for sp, v in resolve_halfsat(params, FR_PREDATOR_SP, k).items()
        },
        "n_years": n_years,
        "window": window,
        "seeds": seeds,
        "base_seed": base_seed,
        "predator_slots": PREDATOR_SLOTS,
        "pairs": pairs,
    }


def _print_table(summary: dict) -> None:
    k_str = ", ".join(f"{sp}={v:.3f}" for sp, v in summary["halfsat_per_predator"].items())
    print(
        f"\n=== FR process diagnostic | K[{k_str}] | "
        f"{summary['n_years']}y window={summary['window']}y "
        f"seeds={summary['seeds']} (base {summary['base_seed']}) ==="
    )
    hdr = (
        f"{'predator':16s} {'prey':12s} {'FR-off M':>12s} {'FR-on M':>12s} "
        f"{'delta±std':>20s} {'>2σ':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))
    flagged = 0
    for p in summary["pairs"]:
        flag = "***" if p["exceeds_noise"] else ""
        if p["exceeds_noise"]:
            flagged += 1
        delta_str = f"{p['delta_mean']:+.4f}±{p['delta_std']:.4f}"
        print(
            f"{p['predator']:16s} {p['prey']:12s} "
            f"{p['fr_off_mort_mean']:>12.4f} {p['fr_on_mort_mean']:>12.4f} "
            f"{delta_str:>20s} {flag:>4s}"
        )
    print(
        f"\nPairs exceeding seed-noise band (|mean delta| > 2·std): {flagged} / "
        f"{len(summary['pairs'])}"
    )
    print("(*** = FR effect on realized predation mortality exceeds seed noise)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, required=True, help="phase-14 or phase-13 params JSON")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--years", type=int, default=40)
    ap.add_argument("--window", type=int, default=10, help="trailing window of years")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument(
        "--halfsat-from-params",
        action="store_true",
        help="read FR halfsat K per predator from the params JSON (phase-14)",
    )
    ap.add_argument("--k", type=float, default=1.0, help="uniform halfsat K when not from params")
    ap.add_argument("--json", type=Path, default=None, help="optional output JSON path")
    args = ap.parse_args()

    with open(args.params) as f:
        params = json.load(f)["parameters"]

    # K resolution: --halfsat-from-params reads per-predator halfsat from the
    # JSON (falling back to --k for any absent predator); otherwise uniform --k.
    if args.halfsat_from_params:
        for sp in FR_PREDATOR_SP:
            key = f"predation.functional.response.halfsat.sp{sp}"
            if key not in params:
                print(f"NOTE: {key} absent from params JSON; using --k={args.k} for sp{sp}")
    # `k` is only the FALLBACK for predators absent from params; run_diagnostic /
    # resolve_halfsat read each predator's actual K from params when present and report
    # the per-predator values in the summary.
    k = args.k

    if args.seeds == 1:
        print(
            "WARNING: seeds=1 → per-pair std is 0; the >2σ noise-band flag is meaningless "
            "(every nonzero delta flags). Use --seeds 3+ for a real noise band."
        )

    print(
        f"Running FR diagnostic: {args.seeds} seed(s) x 2 variants "
        f"x {args.years}y (window {args.window}y), "
        f"K={'per-predator from params' if args.halfsat_from_params else k}"
    )
    summary = run_diagnostic(
        params,
        k=k,
        n_years=args.years,
        window=args.window,
        seeds=args.seeds,
        base_seed=args.base_seed,
    )
    summary["params_path"] = str(args.params)
    summary["halfsat_from_params"] = bool(args.halfsat_from_params)
    _print_table(summary)

    out = args.json or Path("data/baltic/calibration_results/fr_process_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
