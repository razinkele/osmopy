#!/usr/bin/env python
"""B2 literature-delta scenario harness (spec 2026-08-29, Task 3 of the B2 plan).

Six arms x 5 house seeds x 50 yr on the certified Baltic config:
  baseline    -- production config, unmodified (no B2 machinery engaged).
  zero        -- all B2 machinery engaged (thermal knob + O2 offset file), zero deltas.
                 Must be bit-identical to baseline per seed (spec §4a) -- the identity
                 control for the whole scenario-application pipeline.
  rcp45_bsap, rcp45_ref, rcp85_bsap, rcp85_ref
              -- the four literature-delta arms of `data/baltic/scenarios/b2_literature_
                 deltas.json` (Meier et al. 2022, Tables 7/10; rcp45_ref is a *sourced
                 zero* on the O2 axis -- temperature-only by the source table, not a
                 degenerate case).

Drivers: dT via the C1 thermal knob (herring-only, constant-T series at tref+dT -- see
`scripts/baltic_c1_knob_ab.py`); dO2 as a wet-cell-only additive offset on the production
O2 NetCDF (`scripts/build_baltic_b2_forcing.py`, reused here for both artifact generation
and its wet-mask/zero-check helpers).

BLOCKING checks (spec §4), run in this order -- any failure is a wiring bug, stop, no
interpretation of the run that follows:
  1. builder zero-check: the ACTUAL zero-arm O2 artifact this run's engine runs and
     downstream checks use (`artifacts["zero"]["o2_nc"]`, written by the same
     `write_arm_dir` call as every other arm here -- not a separate, freshly-written copy
     under its own throwaway directory) is value-identical (NaN-aware) to the production
     input (controller review MINOR 2: pointed at the real artifact, not a proxy for it).
  2. §4(b) O2 load-through, STRENGTHENED to a three-way assert (controller review,
     IMPORTANT): per arm with an O2 artifact, the ENGINE's own loader
     (`osmose.engine.simulate._load_oxygen_data`, directly importable and callable on a
     raw config dict -- confirmed by reading its signature, so the load-through route is
     the primary one, not the NetCDF-reread fallback) is called on the assembled config,
     and its held array, the array on disk, AND the array independently recomputed from
     the untouched production field via the builder's own `offset_o2` must all agree
     (`load_through_ok`). A plain engine-held == on-disk check (the original §4(b) design)
     cannot detect a silent no-op offset write -- a written file byte-identical to
     production despite a nonzero delta passes engine==disk trivially, and would also
     pass §4(c)'s ordering check trivially in the flat region of the Hill curve (see
     `hill_ordering_ok`'s docstring) -- so the recomputed-expected third term is the
     actual detector, not a redundant belt-and-braces addition.
  3. §4(c) Hill ordering: per arm with a dO2, `f_o2_hill` over the arm's O2 field vs the
     baseline field obeys the delta's sign, wet cells only (`hill_ordering_ok`).
  4. §4(d) knob-factor instrument: the loader's actual float path -- exp(beta *
     (float(str(tref+dT)) - tref)) -- not the naive exp(beta*dT); `expected_knob_factor`
     pins the review-found 3-ULP gap at the non-dyadic dT=2.9 (RCP8.5).
  5. engine runs (all arms x all seeds).
  6. §4(a) zero-arm bit-identity to baseline, per seed, every species.

REPORTED (no pass/fail, spec §4): herring's decline per arm; the within-RCP load contrast
(BSAP vs REF at identical dT) for cod_east and flounder, printed beside the predicted
effective-K change (from the builder) and the +-1.9% cod_east seed-noise floor
(`docs/baltic_hypoxia_benthos_ab_2026-08-09.md`); the full §4 label list (SST-for-bottom-T
proxy, summer-only+SLR-variant O2 applied year-round, uniform-offset spatial blindness +
floor asymmetry, LTL-at-baseline, reference-period overstatement, cod_east's RV-narrative
conditioning).

This module's three pure helpers (`expected_knob_factor`, `arm_overlays`,
`hill_ordering_ok`) are covered by `tests/test_baltic_b2_harness_helpers.py`. `run_b2`
itself is NOT invoked by that test suite or by this task -- a full 6-arm x 5-seed x 50yr
run is ~2.5-3h (spec decision 7) and is Task 4's deliverable, not Task 3's. Its wiring was
instead validated manually during development, without any engine simulation: (1) a tiny
smoke run (nyear=2, one seed, arm "zero" only) proved `PythonEngine().run_in_memory(cfg,
seed).biomass()` accepts the assembled config and returns the exact species-name columns
this module's REPORTED section hardcodes; (2) BLOCKING 1-4 were separately run against the
REAL production files, at the real N_YEAR=50 (not nyear=2's truncated window, which never
leaves the knob series' spin-up block), for all five non-baseline arms -- including the
non-dyadic RCP8.5 dT=2.9 case, which reproduced the exact 3-ULP-affected factor value
`expected_knob_factor` predicts, and the rcp85_ref floor-interaction case (1344 wet
cell-frames floored under its -8.9 offset), which still passed Hill ordering. See
task-3-report.md for the full transcript.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

from osmose.engine.processes.oxygen_function import f_o2_hill

_HERE = Path(__file__).resolve().parent

# C1's knob constants (herring-only thermal knob) -- reused verbatim via the established
# scripts/ importlib-from-path idiom (see scripts/build_baltic_b2_forcing.py, which imports
# the same module the same way for its series writer).
_c1_spec = importlib.util.spec_from_file_location(
    "baltic_c1_knob_ab", _HERE / "baltic_c1_knob_ab.py"
)
_c1 = importlib.util.module_from_spec(_c1_spec)
_c1_spec.loader.exec_module(_c1)
TREFS = _c1.TREFS
BETAS = _c1.BETAS
ENABLED = _c1.ENABLED
N_YEAR = _c1.N_YEAR
SPINUP = _c1.SPINUP

# Task 2's builder (artifact generation + wet-mask/zero-check helpers), same idiom.
_b2_spec = importlib.util.spec_from_file_location(
    "build_baltic_b2_forcing", _HERE / "build_baltic_b2_forcing.py"
)
_b2 = importlib.util.module_from_spec(_b2_spec)
_b2_spec.loader.exec_module(_b2)

SEEDS = (42, 123, 7, 999, 2024)
DEFAULT_SPEC_PATH = _b2.DEFAULT_SPEC_PATH
DEFAULT_GRID_PATH = _b2.DEFAULT_GRID_PATH
DEFAULT_O2_PATH = _b2.DEFAULT_O2_PATH

SCENARIO_ARM_NAMES = ("rcp45_bsap", "rcp45_ref", "rcp85_bsap", "rcp85_ref")
ARMS = ("baseline", "zero", *SCENARIO_ARM_NAMES)
# Single source of truth (controller review MINOR 1): the builder module owns this dict --
# its own zero-delta self-check uses the identical object.
ZERO_ARM_DEF = _b2.ZERO_ARM_DEF

REPORT_PATH = Path("/tmp/b2_scenario_report.json")

# ±1.9% cod_east across-seed spread at baseline O2 (design doc "The O2 axis..." section /
# docs/baltic_hypoxia_benthos_ab_2026-08-09.md) -- the noise floor the O2 axis's ecological
# reads must be printed against.
COD_EAST_SEED_NOISE_FLOOR = 0.019

# spec §4's REPORTED label list, restated verbatim (design doc §4, REPORTED bullet list).
REPORT_LABELS = [
    "SST-for-bottom-T proxy: annual SST applied to a Q4 bottom-T knob; deep warming runs "
    "higher than SST in ventilated basins, so the herring decline is likely UNDERSTATED.",
    "Summer-only + SLR-variant delta-O2 (Meier2022 Table 10) applied year-round.",
    "Uniform-offset spatial blindness + floor asymmetry: the additive O2 offset is spatially "
    "uniform and floor-clipped on the negative side but uncapped on the positive side.",
    "LTL-at-baseline: the BSAP cells are a partial-load world -- the load cut's O2 benefit "
    "enters but its plankton/nutrient pathways do not; the omitted pathways plausibly "
    "oppose the included one.",
    "Reference-period overstatement (spec decision 5): literature deltas are end-century "
    "vs 1976-2005 but applied raw on a present-day baseline (O2: 2024 analysis; "
    "tref: 1993-2021 mean) -- overstates end-century forcing by the realized "
    "1976-2005->present component.",
    "cod_east's trajectory is partly prescribed by the RV narrative series (gate factor "
    "0.32-0.87 across the scored decade) -- its scenario deltas are conditioned on that "
    "prescription.",
]


def expected_knob_factor(beta: float, tref: float, dT: float) -> float:
    """spec §4(d): the loader's ACTUAL float path for the exponential thermal-gate
    response -- exp(beta * (float(str(tref + dT)) - tref)) -- not the naive exp(beta*dT).

    The knob series is written to CSV as `str(tref + dT)` (`baltic_c1_knob_ab.
    write_arm_series`) and read back through `pandas.read_csv(..., float_precision=
    "round_trip")` (`osmose.engine.config._load_thermal_gate`), which parses the decimal
    string to the nearest float64 -- equivalent to Python's own `float(str(...))`. Because
    `(tref + dT) - tref != dT` in general float64 arithmetic (textbook cancellation), this
    differs from `exp(beta*dT)` by a few ULP at non-dyadic dT (pinned at dT=2.9, RCP8.5,
    by tests/test_baltic_b2_harness_helpers.py) even though C1's own dT values (2.0, 4.0)
    happened to round-trip losslessly -- dyadic luck, not a general property.
    """
    temp = float(str(tref + dT))
    return float(np.exp(beta * (temp - tref)))


def arm_overlays(
    arm_name: str,
    artifacts: dict,
    trefs: dict[int, float],
    betas: dict[int, float],
) -> dict[str, str]:
    """Config overlay for one B2 arm (spec Design §3).

    `baseline` carries no B2 machinery at all -- the production config as-is. Every other
    arm turns on the C1 thermal knob (herring-only; full-precision tref strings, the exact
    value the zero arm's bit-identity to baseline rests on) and, only when the arm carries
    an O2 artifact (`artifacts["o2_nc"]` is not None), overlays `oxygen.filename` at that
    arm's generated file (already an absolute path per the builder's own contract --
    `write_arm_dir` resolves `out_dir` before writing, so this never depends on the
    caller's cwd). The key is left ABSENT (not set to None) when there is no O2 artifact,
    so a dict-merge onto the base config never shadows its production `oxygen.filename`.
    """
    if arm_name == "baseline":
        return {}

    overlays: dict[str, str] = {
        "reproduction.thermal.gate.enabled": "true",
        "reproduction.thermal.gate.series.file": str(artifacts["series_csv"]),
        "reproduction.thermal.gate.response": "exponential",
    }
    for sp in ENABLED:
        overlays[f"reproduction.thermal.gate.species.enabled.sp{sp}"] = "true"
        overlays[f"reproduction.thermal.gate.beta.sp{sp}"] = str(betas[sp])
        overlays[f"reproduction.thermal.gate.tref.sp{sp}"] = str(trefs[sp])

    if artifacts.get("o2_nc") is not None:
        overlays["oxygen.filename"] = str(artifacts["o2_nc"])

    return overlays


def hill_ordering_ok(
    arm_o2: np.ndarray,
    base_o2: np.ndarray,
    wet: np.ndarray,
    delta_sign: int,
    c50: float = 60.0,
    n: float = 3.0,
) -> bool:
    """spec §4(c): deterministic Hill ordering, per (wet cell, frame).

    `f_o2_hill` is monotonically non-decreasing in O2 for O2 >= 0, so a correctly-wired
    positive-delta arm must have `f_o2_hill(arm) >= f_o2_hill(base)` at every wet cell,
    a negative-delta arm `<=`, and a zero-delta arm exactly `==` -- any violation means the
    wrong field was loaded (wiring bug), not an ecological surprise. Land/non-wet cells are
    excluded from the comparison entirely (never merely tolerated as NaN). `c50`/`n`
    default to the production coupling's values (`ltl.oxygen.benthos.c50`/`.n`,
    `data/baltic/baltic_param-oxygen.csv`).

    NOT a sufficient detector on its own (controller review, IMPORTANT): a silent no-op
    offset write (arm field == base field despite a nonzero delta) still satisfies `>=` /
    `<=` by construction, since equality trivially satisfies either non-strict inequality --
    and in the Hill curve's flat region (O2 far from c50) even a genuinely-applied but small
    offset can leave `arm_hill == base_hill` to float precision. `load_through_ok`'s
    recomputed-expected three-way check is the actual detector for that failure mode; this
    function alone cannot distinguish "correctly zero-changed" from "silently never wrote
    the offset".
    """
    arm_o2 = np.asarray(arm_o2, dtype=np.float64)
    base_o2 = np.asarray(base_o2, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    wet3 = np.broadcast_to(wet, arm_o2.shape)

    arm_hill = f_o2_hill(arm_o2[wet3], c50, n)
    base_hill = f_o2_hill(base_o2[wet3], c50, n)

    if delta_sign > 0:
        return bool(np.all(arm_hill >= base_hill))
    if delta_sign < 0:
        return bool(np.all(arm_hill <= base_hill))
    return bool(np.array_equal(arm_hill, base_hill))


def load_through_ok(
    engine_o2: np.ndarray,
    disk_o2: np.ndarray,
    production_o2: np.ndarray,
    wet: np.ndarray,
    delta: float,
) -> bool:
    """spec §4(b), STRENGTHENED to a three-way assert (controller review, IMPORTANT --
    an ADDED wiring assert, no pre-registered criterion changed).

    The original two-way check (engine-loaded array == array on disk) only catches a
    *loader* bug -- it cannot catch a *writer* bug where the offset was silently never
    applied: a written file that is byte-identical to the untouched production field
    despite a nonzero delta passes engine==disk trivially (the engine faithfully loads
    whatever is on disk, correct or not), and would also pass `hill_ordering_ok` trivially
    wherever `>=`/`<=` is satisfied by exact equality (see that function's docstring). The
    actual detector is the third term: independently recompute the EXPECTED offset field
    from the untouched production array via the builder's own `offset_o2` (single source of
    truth for the offset math, imported from `build_baltic_b2_forcing`) and require
    engine-loaded == on-disk == expected.

    `equal_nan=True` throughout: the real production O2 field is confirmed NaN-free (land
    cells are 0.0, verified in Task 2), but this function is also exercised directly against
    NaN-bearing synthetic fixtures (land cells) in the test suite, so it can't assume that.
    """
    engine_o2 = np.asarray(engine_o2, dtype=np.float64)
    disk_o2 = np.asarray(disk_o2, dtype=np.float64)
    production_o2 = np.asarray(production_o2, dtype=np.float64)
    expected = _b2.offset_o2(production_o2, wet, delta)
    return bool(
        np.array_equal(engine_o2, disk_o2, equal_nan=True)
        and np.array_equal(disk_o2, expected, equal_nan=True)
    )


def _delta_sign(dO2: dict | None) -> int:
    if dO2 is None:
        return 0
    v = float(dO2["value_mmol_m3"])
    if v > 0.0:
        return 1
    if v < 0.0:
        return -1
    return 0


def _expected_knob_factor_array(beta: float, tref: float, dT: float) -> np.ndarray:
    """Per-year expected factor array over the knob series' full window (spin-up years at
    1.0, historical years at `expected_knob_factor`) -- the ground truth §4(d) checks the
    loader's `factor[:, sp]` column against, exactly (`np.array_equal`)."""
    factors = np.ones(N_YEAR, dtype=np.float64)
    factors[SPINUP:] = expected_knob_factor(beta, tref, dT)
    return factors


def run_b2(seeds=SEEDS) -> dict:
    """Run all six B2 arms across `seeds`, blocking checks first (spec §4, in the fixed
    order documented in this module's docstring), engine runs second, results JSON last.

    NOT invoked by the test suite (see module docstring) -- a full call is ~2.5-3h.
    """
    import xarray as xr

    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine.config import _load_thermal_gate
    from osmose.engine.simulate import _load_oxygen_data

    tmp = Path(tempfile.mkdtemp())
    base_cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    base_cfg["simulation.time.nyear"] = str(N_YEAR)
    n_sp = int(base_cfg["simulation.nspecies"])
    n_dt = int(base_cfg["simulation.time.ndtperyear"])

    out_root = Path(tempfile.mkdtemp(prefix="b2_harness_"))

    # Assemble the non-baseline arm definitions and write their forcing artifacts. This
    # must happen before BLOCKING 1 below so BLOCKING 1 can point at the REAL zero-arm
    # artifact this run's engine runs and downstream checks go on to use, rather than a
    # separate, freshly-written copy (controller review MINOR 2).
    delta_spec = json.loads(DEFAULT_SPEC_PATH.read_text())
    spec_arms = {a["name"]: a for a in delta_spec["arms"]}
    arm_defs = {"zero": ZERO_ARM_DEF, **{n: spec_arms[n] for n in SCENARIO_ARM_NAMES}}

    artifacts: dict[str, dict] = {}
    for name, arm_def in arm_defs.items():
        artifacts[name] = _b2.write_arm_dir(
            arm_def, out_root / name, DEFAULT_O2_PATH, DEFAULT_GRID_PATH, TREFS, BETAS
        )

    wet = _b2.load_wet_mask(DEFAULT_GRID_PATH)
    with xr.open_dataset(DEFAULT_O2_PATH) as ds:
        base_o2 = ds[_b2._single_data_var(ds)].values.astype(np.float64)

    # --- BLOCKING 1: builder zero-check, pointed at the actual zero-arm artifact ---
    with xr.open_dataset(artifacts["zero"]["o2_nc"]) as ds_zero:
        zero_vals = ds_zero[_b2._single_data_var(ds_zero)].values.astype(np.float64)
    zero_check_ok = bool(np.array_equal(base_o2, zero_vals, equal_nan=True))
    if not zero_check_ok:
        raise AssertionError(
            "B2 harness BLOCKED (§Design 2 builder zero-check): the zero arm's own O2 file "
            f"({artifacts['zero']['o2_nc']}) -- the actual file this run's engine runs and "
            "downstream checks will use -- diverges from the production input it was "
            "supposed to copy value-identically. Wiring bug -- no further checks or runs "
            "attempted."
        )

    # --- BLOCKING 2: §4(b) O2 load-through, STRENGTHENED to a three-way assert (controller
    # review, IMPORTANT) -- see load_through_ok's docstring for why engine==disk alone
    # cannot catch a silent no-op offset write. ---
    load_through: dict[str, bool] = {}
    for name, arm_def in arm_defs.items():
        o2_nc = artifacts[name]["o2_nc"]
        if o2_nc is None:
            continue
        cfg = {**base_cfg, "oxygen.filename": str(o2_nc)}
        loaded = _load_oxygen_data(cfg, None)
        if loaded is None:
            raise AssertionError(
                f"B2 harness BLOCKED (§4b, arm={name}): _load_oxygen_data returned None "
                "for an arm with a resolvable oxygen.filename -- the silent-fallback trap."
            )
        with xr.open_dataset(o2_nc) as ds2:
            written = ds2[_b2._single_data_var(ds2)].values.astype(np.float64)
        delta = float(arm_def["dO2"]["value_mmol_m3"])
        ok = load_through_ok(loaded._data, written, base_o2, wet, delta)
        load_through[name] = ok
        if not ok:
            raise AssertionError(
                f"B2 harness BLOCKED (§4b, arm={name}): three-way load-through check failed "
                "(engine-loaded == on-disk == recomputed-expected via offset_o2). Wiring "
                "bug -- possibly a silent no-op offset write, not just a loader mismatch."
            )

    # --- BLOCKING 3: §4(c) Hill ordering, per arm with a dO2 ---
    hill_ordering: dict[str, bool] = {}
    for name, arm_def in arm_defs.items():
        dO2 = arm_def.get("dO2")
        if dO2 is None:
            continue
        with xr.open_dataset(artifacts[name]["o2_nc"]) as ds2:
            arm_o2 = ds2[_b2._single_data_var(ds2)].values.astype(np.float64)
        ok = hill_ordering_ok(arm_o2, base_o2, wet, _delta_sign(dO2))
        hill_ordering[name] = ok
        if not ok:
            raise AssertionError(
                f"B2 harness BLOCKED (§4c, arm={name}): Hill ordering violates the dO2 sign. "
                "Wiring bug (guaranteed by monotonicity)."
            )

    # --- BLOCKING 4: §4(d) knob-factor instrument, per arm ---
    knob_instrument: dict[str, dict[str, bool]] = {}
    all_arm_cfgs: dict[str, dict[str, str]] = {"baseline": dict(base_cfg)}
    for name, arm_def in arm_defs.items():
        cfg = {**base_cfg, **arm_overlays(name, artifacts[name], TREFS, BETAS)}
        all_arm_cfgs[name] = cfg

        dT = float(arm_def["dT_C"])
        factor, enabled_mask, _offset = _load_thermal_gate(cfg, n_sp, n_dt, N_YEAR)
        knob_instrument[name] = {}
        for sp in ENABLED:
            if enabled_mask is None or not enabled_mask[sp]:
                raise AssertionError(
                    f"B2 harness BLOCKED (§4d, arm={name}): sp{sp} not marked enabled by "
                    "_load_thermal_gate."
                )
            exp_arr = _expected_knob_factor_array(BETAS[sp], TREFS[sp], dT)
            ok = bool(np.array_equal(factor[:, sp], exp_arr))
            knob_instrument[name][f"sp{sp}"] = ok
            if not ok:
                raise AssertionError(
                    f"B2 harness BLOCKED (§4d, arm={name}): sp{sp} loaded factor column != "
                    "expected_knob_factor's float-path value, exactly. Wiring bug."
                )

    # --- engine runs (all arms x all seeds) ---
    sp_names = {sp: base_cfg.get(f"species.name.sp{sp}", f"sp{sp}") for sp in range(n_sp)}
    all_species: list[str] | None = None
    raw_series: dict[str, dict[str, list[np.ndarray]]] = {a: {} for a in ARMS}
    for seed in seeds:
        for arm in ARMS:
            bio_df = PythonEngine().run_in_memory(all_arm_cfgs[arm], seed=seed).biomass()
            if all_species is None:
                all_species = [c for c in bio_df.columns if c not in ("Time", "species")]
            for name in all_species:
                raw_series[arm].setdefault(name, []).append(bio_df[name].to_numpy(dtype=float))
    assert all_species is not None

    # --- BLOCKING 5: §4(a) zero-arm bit-identity to baseline, per seed ---
    identity_violations = []
    for i, seed in enumerate(seeds):
        for name in all_species:
            bio_baseline = raw_series["baseline"][name][i]
            bio_zero = raw_series["zero"][name][i]
            if not np.array_equal(bio_baseline, bio_zero):
                identity_violations.append({"seed": seed, "species": name})
    if identity_violations:
        raise AssertionError(
            "B2 harness BLOCKED (§4a): zero arm is NOT bit-identical to baseline -- "
            f"{len(identity_violations)} (seed, species) violations, e.g. "
            f"{identity_violations[0]}. Wiring bug."
        )

    # --- REPORTED (no pass/fail): herring decline, within-RCP load contrast ---
    def _decade_mean(arm: str, name: str) -> float:
        per_seed = [
            float(_c1._annualize(series, N_YEAR)[-10:].mean()) for series in raw_series[arm][name]
        ]
        return float(np.mean(per_seed))

    herring_name = sp_names[1]
    baseline_herring = _decade_mean("baseline", herring_name)
    herring_decline = {
        arm: (_decade_mean(arm, herring_name) / baseline_herring - 1.0)
        if baseline_herring
        else None
        for arm in ARMS
        if arm != "baseline"
    }

    load_contrast = {}
    for rcp, bsap_arm, ref_arm in (
        ("RCP4.5", "rcp45_bsap", "rcp45_ref"),
        ("RCP8.5", "rcp85_bsap", "rcp85_ref"),
    ):
        load_contrast[rcp] = {}
        for sp_key in ("cod_east", "flounder"):
            bsap_mean = _decade_mean(bsap_arm, sp_key)
            ref_mean = _decade_mean(ref_arm, sp_key)
            load_contrast[rcp][sp_key] = (bsap_mean / ref_mean - 1.0) if ref_mean else None

    report = {
        "seeds": list(seeds),
        "arms": list(ARMS),
        "instrument": {
            "load_through": load_through,
            "hill_ordering": hill_ordering,
            "knob_factor": knob_instrument,
        },
        "identity": {"pass": len(identity_violations) == 0, "violations": identity_violations},
        "predicted_dK": {name: artifacts[name]["predicted_dK"] for name in arm_defs},
        "reported": {
            "herring_decline_vs_baseline": herring_decline,
            "within_rcp_load_contrast_bsap_vs_ref": load_contrast,
            "cod_east_seed_noise_floor": COD_EAST_SEED_NOISE_FLOOR,
        },
        "labels": REPORT_LABELS,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=float)
    return report


if __name__ == "__main__":
    out = run_b2()
    print(f"report written to {REPORT_PATH}")
    print(f"identity pass: {out['identity']['pass']}")
