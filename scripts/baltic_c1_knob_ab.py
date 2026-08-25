#!/usr/bin/env python
"""C1 thermal-knob A/B harness (spec 2026-08-25, Task 5 of the C1 plan).

Four arms x 5 seeds x 50 yr on the certified Baltic config: off (gate absent),
knob0 (gate on, dT=0 -- the identity/bit-exactness control), knob2 (dT=+2 C),
knob4 (dT=+4 C). Constant-T forcing series are generated at run time (not
read from CMEMS): spin-up 1974-1992 held at tref in every arm; the historical
block 1993-2023 is held at tref+dT for the knob arms.

SETTLED CONSTANTS (Task 4, cod_west fit): the pre-registered fit returned
enabled=False for cod_west (p=0.887, detrended sign flip) -- the knob is
HERRING-ONLY. sp0 (cod_west) stays UNFORCED in every arm; only sp1 (herring)
carries reproduction.thermal.gate.* overrides. TREF is the CSV's full-
precision bottom-T Q4 mean (9.670314810741907), not the README's rounded
9.67 -- a rounded tref would break the exact-zero exponent the bit-identity
arm (knob0, dT=0) rests on (docs/baltic_c1_codwest_fit_2026-08-25.md).

Pre-registered, NOT tunable: the arm set (off/knob0/knob2/knob4), the seed
set, the bit-identity criterion (off == knob0 array_equal), and the
monotonicity criterion (knob0 > knob2 > knob4, final-decade mean, 5-seed
means). This script is NOT a CI gate (emergent, ~70 min full run -- Task 6).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 123, 7, 999, 2024)
N_YEAR = 50
SPINUP = 19
FIRST_YEAR = 1974
ARMS_DT = {"knob0": 0.0, "knob2": 2.0, "knob4": 4.0}

# Herring-only knob (Task 4 verdict): cod_west (sp0) stays unforced.
ENABLED = (1,)
BETAS = {1: -0.51}
TREFS = {1: 9.670314810741907}

REPORT_PATH = Path("/tmp/c1_knob_report.json")


def write_arm_series(path, trefs: dict[int, float], dT: float) -> None:
    """Write a constant-T forcing CSV: spin-up rows at tref, historical rows at
    tref+dT, shared spin-up across arms (spec §4a's identity control depends on
    every arm's spin-up block being identical)."""
    species = sorted(trefs)
    lines = ["year," + ",".join(f"temp_sp{sp}" for sp in species)]
    for i in range(N_YEAR):
        year = FIRST_YEAR + i
        vals = [trefs[sp] if i < SPINUP else trefs[sp] + dT for sp in species]
        lines.append(",".join([str(year)] + [str(v) for v in vals]))
    Path(path).write_text("\n".join(lines) + "\n")


def expected_factors(
    beta: float, dT: float, n_year: int = N_YEAR, spinup: int = SPINUP
) -> np.ndarray:
    """The loader-level determinism check's ground truth: 1.0 for every spin-up
    year, exp(beta*dT) for every historical year (raw exponential response,
    C1 spec decision 8 -- no renormalisation)."""
    factors = np.ones(n_year, dtype=np.float64)
    factors[spinup:] = np.exp(beta * dT)
    return factors


def arm_overrides(
    mode: str,
    series_path: str,
    trefs: dict[int, float],
    betas: dict[int, float],
    enabled: tuple[int, ...],
) -> dict[str, str]:
    """Config overrides for one arm. mode="off" leaves the thermal gate keys
    out entirely (gate stays disabled by the production config's default);
    any other mode turns the gate on for exactly the species in `enabled`."""
    base: dict[str, str] = {"simulation.time.nyear": str(N_YEAR)}
    if mode == "off":
        return base
    base["reproduction.thermal.gate.enabled"] = "true"
    base["reproduction.thermal.gate.series.file"] = str(series_path)
    base["reproduction.thermal.gate.response"] = "exponential"
    for sp in enabled:
        base[f"reproduction.thermal.gate.species.enabled.sp{sp}"] = "true"
        base[f"reproduction.thermal.gate.beta.sp{sp}"] = str(betas[sp])
        base[f"reproduction.thermal.gate.tref.sp{sp}"] = str(trefs[sp])
    return base


def _annualize(x: np.ndarray, n_year: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == n_year:
        return x
    if len(x) % n_year == 0:
        return x.reshape(n_year, -1).mean(axis=1)
    raise ValueError(f"series of {len(x)} not divisible into {n_year} years")


def run_ab(seeds=SEEDS) -> dict:
    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine.config import _load_thermal_gate

    tmp = Path(tempfile.mkdtemp())
    base_cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    n_sp = int(base_cfg["simulation.nspecies"])
    n_dt = int(base_cfg["simulation.time.ndtperyear"])

    series_dir = Path(tempfile.mkdtemp())
    series_paths = {}
    for arm, dT in ARMS_DT.items():
        p = series_dir / f"{arm}.csv"
        write_arm_series(p, TREFS, dT)
        series_paths[arm] = p

    # Assembled per-arm configs (shared across seeds -- only `seed` varies the run).
    arm_cfgs = {"off": {**base_cfg, **arm_overrides("off", "", {}, {}, ENABLED)}}
    for arm, dT in ARMS_DT.items():
        arm_cfgs[arm] = {
            **base_cfg,
            **arm_overrides(arm, str(series_paths[arm]), TREFS, BETAS, ENABLED),
        }

    # Instrument (spec §4c): loader-level determinism check, before any run.
    instrument: dict[str, dict[str, bool]] = {}
    for arm, dT in ARMS_DT.items():
        factor, enabled_mask, _offset = _load_thermal_gate(arm_cfgs[arm], n_sp, n_dt, N_YEAR)
        instrument[arm] = {}
        for sp in ENABLED:
            assert enabled_mask is not None and enabled_mask[sp], (
                f"{arm}: sp{sp} not marked enabled by _load_thermal_gate"
            )
            exp_f = expected_factors(BETAS[sp], dT)
            ok = bool(np.array_equal(factor[:, sp], exp_f))
            assert ok, f"{arm}: sp{sp} factor column != expected_factors(beta, dT) exactly"
            instrument[arm][f"sp{sp}"] = ok

    # Per-seed runs, all four arms.
    all_species: list[str] | None = None
    raw_series: dict[str, dict[str, list[np.ndarray]]] = {a: {} for a in arm_cfgs}
    for seed in seeds:
        for arm, cfg in arm_cfgs.items():
            bio_df = PythonEngine().run_in_memory(cfg, seed=seed).biomass()
            if all_species is None:
                all_species = [c for c in bio_df.columns if c not in ("Time", "species")]
            for name in all_species:
                raw_series[arm].setdefault(name, []).append(bio_df[name].to_numpy(dtype=float))

    assert all_species is not None

    # Identity (spec §4a): off vs knob0 must be bit-identical, every seed, every species.
    identity_violations = []
    for i, seed in enumerate(seeds):
        for name in all_species:
            bio_off = raw_series["off"][name][i]
            bio_knob0 = raw_series["knob0"][name][i]
            if not np.array_equal(bio_off, bio_knob0):
                identity_violations.append({"seed": seed, "species": name})

    # Monotonicity (spec §4b) + elasticity (spec §4d): final-decade means (per
    # spec, the annualized final 10 of 50 years), enabled species only.
    monotonicity: dict[str, dict] = {}
    elasticity: dict[str, dict] = {}
    sp_names = {sp: base_cfg.get(f"species.name.sp{sp}", f"sp{sp}") for sp in ENABLED}
    for sp in ENABLED:
        name = sp_names[sp]
        decade_means = {}
        for arm in ("knob0", "knob2", "knob4"):
            per_seed = [
                float(_annualize(series, N_YEAR)[-10:].mean()) for series in raw_series[arm][name]
            ]
            decade_means[arm] = float(np.mean(per_seed))
        strictly_decreasing = decade_means["knob0"] > decade_means["knob2"] > decade_means["knob4"]
        monotonicity[name] = {**decade_means, "pass": bool(strictly_decreasing)}

        elasticity[name] = {}
        for arm, dT in (("knob2", ARMS_DT["knob2"]), ("knob4", ARMS_DT["knob4"])):
            realized = (
                decade_means[arm] / decade_means["knob0"] if decade_means["knob0"] else float("nan")
            )
            expected = float(np.exp(BETAS[sp] * dT))
            elasticity[name][arm] = {
                "realized_ratio": realized,
                "expected_ratio": expected,
                "elasticity": realized / expected if expected else float("nan"),
            }

    report = {
        "seeds": list(seeds),
        "enabled_species": {f"sp{sp}": sp_names[sp] for sp in ENABLED},
        "betas": {f"sp{sp}": BETAS[sp] for sp in ENABLED},
        "trefs": {f"sp{sp}": TREFS[sp] for sp in ENABLED},
        "instrument": instrument,
        "identity": {
            "violations": identity_violations,
            "pass": len(identity_violations) == 0,
        },
        "monotonicity": monotonicity,
        "elasticity": elasticity,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=float)
    return report


if __name__ == "__main__":
    out = run_ab()
    print(f"report written to {REPORT_PATH}")
    print(f"identity pass: {out['identity']['pass']}")
    for name, m in out["monotonicity"].items():
        print(f"{name}: monotonicity pass={m['pass']}")
