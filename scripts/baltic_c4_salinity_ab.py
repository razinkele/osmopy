#!/usr/bin/env python
"""C4 salinity-gate sensitivity harness (spec 2026-08-30, Task 3 of the C4 plan).

Five arms x 5 house seeds x 50 yr on the certified Baltic config:
  baseline  -- production config, unmodified (no C4 machinery engaged). The salinity gate is
               already live in production (both cod stocks, ramp 3-6 PSU, since July --
               `baltic_param-movement.csv:220-233`) -- baseline exercises exactly that.
  zero      -- points `movement.salinity.field.file` at the builder's zero-delta artifact
               (all C4 machinery engaged, dS=0). Must be bit-identical to baseline per seed
               (spec decision 4, gate order item 7) -- the identity control for the whole
               overlay-application pipeline.
  ds_m1, ds_m2, ds_m3
            -- the three ΔS arms of `data/baltic/scenarios/c4_salinity_sensitivity.json`
               (-1, -2, -3 PSU uniform additive offsets on the production bottom-salinity
               climatology).

Driver: dS via a single overlay key, `movement.salinity.field.file`, pointed at the arm's
absolute-path forcing artifact written by `scripts/build_baltic_c4_forcing.py::write_arm_dir`
(Task 2). Nothing else is overlaid -- the gate's enable flag, species mask, and ramp bounds
are already live in the production config and are left untouched (`arm_overlays`).

BLOCKING gates (spec decision 4), run in this fixed order -- any failure is a wiring bug,
stop, no interpretation of the run that follows:
  1. builder zero-check: the ACTUAL zero-arm salinity artifact this run's engine runs and
     downstream checks use (`artifacts["zero"]["sal_nc"]`, written by the same
     `write_arm_dir` call as every other arm here) is value-identical (NaN-aware) to the
     production input.
  2. per-arm `.constant`-absence assert (`assert_no_salinity_constant`): the salinity loader
     (`_load_salinity_gate`, osmose/engine/config.py) prefers
     `movement.salinity.field.constant` over `.file` whenever both are present -- a stray
     `.constant` key (there is none in production today) would silently discard the arm's
     ONLY lever without any error.
  3. per-arm frame-count==24 assert on the written field (`assert_arm_frame_count`): the
     salinity loader has NO frame validation of its own -- `PhysicalData.get_grid`/
     `get_value` index `step % <loaded array's frame count>`, a silent misalignment trap
     (CLAUDE.md), not a runtime error.
  4. three-way load-through (`salinity_load_through_ok`): the engine's OWN loader
     (`osmose.engine.config._load_salinity_gate`, directly importable and callable on a raw
     config dict + species count -- confirmed by reading its signature, so this is the
     PRIMARY route, not a from_dict/EngineConfig fallback) is called on the assembled
     config; its held field (`PhysicalData._data`, per the task-2 review), the array on
     disk, AND the array independently recomputed from the untouched production field via
     the builder's own `offset_salinity` must all agree. A plain engine==disk check cannot
     detect a silent no-op offset write -- a written file byte-identical to production
     despite a nonzero dS passes engine==disk trivially, and can also pass ramp ordering
     trivially wherever equality satisfies the non-strict inequality -- so the
     recomputed-expected third term is the actual detector.
  5. ramp ordering (`ramp_ordering_ok`): per arm with a dS, the production ramp
     (`_c4.ramp_w`, a thin wrapper on `salinity_weight`) obeys the dS sign at every wet
     cell -- monotonicity guarantees this absent a wiring bug (dS<0 -> w' <= w, dS==0 ->
     w'==w); wet cells only.
  6. engine runs (all arms x all seeds).
  7. zero-arm bit-identity to baseline, per seed, every species.

REPORTED (no pass/fail, spec decision 5): per-arm per-species final-decade means for ALL
NINE Baltic species (cod_west, herring, sprat, flounder, perch, pikeperch, smelt,
stickleback, cod_east) + across-seed spreads; the builder's instruments (i)-(iv) per arm
(TV distance, prey-overlap shift incl. the juvenile stage, excluded fraction, mean-Δw) and
any all-zero (map, frame) events; the spec decision 6 label list.

This module's five pure helpers (`arm_overlays`, `assert_no_salinity_constant`,
`assert_arm_frame_count`, `ramp_ordering_ok`, `salinity_load_through_ok`) are covered by
`tests/test_baltic_c4_harness_helpers.py`. `run_c4` itself is NOT invoked by that test suite
or by this task -- a full 5-arm x 5-seed x 50yr run is Task 4's deliverable, not Task 3's.
Its wiring was instead validated manually during development, without a full run: (1) a tiny
smoke run (nyear=2, one seed, arm "ds_m1" only) proved `PythonEngine().run_in_memory(cfg,
seed).biomass()` accepts the assembled config (an absolute-path, non-production salinity
NetCDF overlaid onto an otherwise-untouched production config) and returns a superset of the
nine Baltic species columns (plus GreySeal/Cormorant background groups -- 11 total,
discovered by this smoke run, not assumed) that `run_c4`'s REPORTED section reads
dynamically and restricts to the nine named `species.name.sp{idx}` entries, never hardcoded
and never left at the wider bio_df column set; (2) BLOCKING gates 1-5 were separately
exercised against the REAL production files for all four non-baseline arms, including the
-3 PSU exclusion-regime arm (all passed; zero all-zero-frame events at any arm, including
-3 PSU -- the whole-frame all-zero guard never trips at any lever tested here. Whether the
per-cell `excluded_fraction` instrument itself reads non-vacuous at -3 PSU is a Task 4
reading, not measured by this sanity pass -- the spec's own vacuity criterion applies).
See task-3-report.md for the full transcript.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

# Task 2's builder (artifact generation + wet-mask/offset/ramp helpers), imported via the
# established scripts/ importlib-from-path idiom (see scripts/baltic_b2_scenario_ab.py,
# which imports scripts/build_baltic_b2_forcing.py the same way).
_c4_spec = importlib.util.spec_from_file_location(
    "build_baltic_c4_forcing", _HERE / "build_baltic_c4_forcing.py"
)
_c4 = importlib.util.module_from_spec(_c4_spec)
_c4_spec.loader.exec_module(_c4)

SEEDS = (42, 123, 7, 999, 2024)
N_YEAR = 50

DEFAULT_SPEC_PATH = _c4.DEFAULT_SPEC_PATH
DEFAULT_GRID_PATH = _c4.DEFAULT_GRID_PATH
DEFAULT_SAL_PATH = _c4.DEFAULT_SAL_PATH

SCENARIO_ARM_NAMES = ("ds_m1", "ds_m2", "ds_m3")
ARMS = ("baseline", "zero", *SCENARIO_ARM_NAMES)
# Single source of truth (B2 precedent): the builder module owns this dict -- its own zero
# self-check uses the identical object.
ZERO_ARM_DEF = _c4.ZERO_ARM_DEF

REPORT_PATH = Path("/tmp/c4_salinity_report.json")

# spec decision 6's label list, restated verbatim (all restated in the eventual results doc).
REPORT_LABELS = [
    "Not a projection: no ensemble generation supplies a citable mean freshening delta "
    "(Meier et al. 2022 Table 8: BalticAPP -0.06, ECOSUPPORT -0.15, CLIMSEA ~=0 g/kg SSS; "
    "only 2006-era extremes reached -45%). The dS levers are chosen, not cited.",
    "RV confound: this is an occupancy-pathway-only instrument; cod_east recruitment is "
    "RV-prescribed (gate factor 0.32-0.87 across the scored decade), so its response here "
    "is conditioned on that prescription, not free-running.",
    "Single-source climatology: the bottom-salinity field's provenance is CMEMS PHY, "
    "deepest-valid level (see the file's own attrs) -- one source, not an ensemble.",
    "Fixed production ramp 3-6 PSU: the ramp bounds are the live production values, not "
    "retuned by this experiment (non-goal).",
    "Uniform-offset spatial blindness: dS is a spatially uniform additive offset -- it does "
    "not represent any real spatial pattern of change.",
    "cod_west = saturated null control: cod_west's gate is a no-op in production (mean "
    "w=1.0000 on all three maps, every frame) and stays ~0 at dS=-1/-2 -- this is "
    "effectively a cod_east experiment.",
    "The all-zero/un-gate guard status is reported per arm (`all_zero_events`): the "
    "engine's all-zero guard silently reverts a species to UNGATED movement for any (map, "
    "frame) where map*w sums to zero -- a wiring hazard the builder turns visible, not a "
    "harness-fixed bug.",
    "Java gap: Java silently ignores movement.salinity.* -- no Java cross-check exists for "
    "this experiment (joins the C1 thermal item, both waiting on the user-dirty runner.py).",
    "The gate conserves total occupancy -- it redistributes and excludes, it never removes "
    "fish. mean_dw is a wiring check only, never a stock-response metric on its own.",
]


def arm_overlays(arm_name: str, artifacts: dict) -> dict[str, str]:
    """Config overlay for one C4 arm (spec Design §3 / binding facts): overlays ONLY
    `movement.salinity.field.file` at the arm's absolute path -- nothing else. The gate's
    enable flag, species mask, and ramp bounds are already live in the production config and
    are left untouched.

    `baseline` carries no override at all -- the production config as-is, no C4 machinery
    engaged. Every other arm (including `zero`) points the salinity field at its own written
    artifact (`artifacts["sal_nc"]`, already an absolute path per the builder's own contract
    -- `write_arm_dir` resolves `out_dir` before writing, so this never depends on the
    caller's cwd).
    """
    if arm_name == "baseline":
        return {}
    return {"movement.salinity.field.file": str(artifacts["sal_nc"])}


def assert_no_salinity_constant(cfg: dict[str, str], arm_name: str) -> None:
    """spec decision 4, gate order item 2: `movement.salinity.field.constant` must be absent
    (or empty) from an arm's assembled config.

    `_load_salinity_gate` (osmose/engine/config.py) prefers `.constant` over `.file` when
    both resolve to non-empty strings -- there is no `.constant` key in production today
    (`baltic_param-movement.csv`), but a stray one (inherited from an overlay bug, a merged
    scenario file, etc.) would silently discard the arm's ONLY lever, the `.file` overlay,
    without any error. Raises ValueError -- never a silent pass.
    """
    if cfg.get("movement.salinity.field.constant", ""):
        raise ValueError(
            f"C4 harness BLOCKED (gate 2, arm={arm_name}): movement.salinity.field.constant "
            "is present in the assembled config -- the loader prefers .constant over .file, "
            "so the arm's overlay would be silently discarded. Wiring bug."
        )


def assert_arm_frame_count(field: np.ndarray, arm_name: str) -> None:
    """spec decision 4, gate order item 3: per-arm frame-count==24 assert on the written
    salinity field. The engine's salinity loader (`_load_salinity_gate` ->
    `PhysicalData.from_netcdf`) has NO frame validation of its own -- a mismatched frame
    count would silently misalign the month-to-step mapping via `step % <frames>` wrap
    (CLAUDE.md's frame-count trap), not raise. The harness re-asserts independently of the
    builder's own write-time check (`_c4._require_24_frames`, the single source of truth
    this delegates to rather than reimplementing).
    """
    _c4._require_24_frames(int(np.asarray(field).shape[0]), f"arm={arm_name} (harness re-check)")


def ramp_ordering_ok(sal_arm: np.ndarray, sal_base: np.ndarray, wet: np.ndarray, dS: float) -> bool:
    """spec decision 4, gate order item 5: deterministic ramp ordering, per wet cell.

    `ramp_w` (`_c4.ramp_w`, a thin wrapper on the engine's own `salinity_weight`) is
    monotonically non-decreasing in salinity, so a correctly-wired negative-dS arm must have
    `ramp_w(arm) <= ramp_w(base)` at every wet cell, a positive-dS arm `>=`, and a zero-dS
    arm exactly `==` -- any violation means the wrong field was loaded (wiring bug), not an
    ecological surprise. Land/non-wet cells are excluded from the comparison entirely (never
    merely tolerated as NaN). Every arm in this spec has dS <= 0; the `>=` branch is kept for
    symmetry with B2's `hill_ordering_ok` precedent, not exercised by `run_c4`.

    NOT a sufficient detector on its own (B2 precedent): a silent no-op offset write (arm
    field == base field despite dS != 0) still satisfies `<=` / `>=` by construction, since
    equality trivially satisfies either non-strict inequality. `salinity_load_through_ok`'s
    recomputed-expected three-way check is the actual detector for that failure mode.
    """
    sal_arm = np.asarray(sal_arm, dtype=np.float64)
    sal_base = np.asarray(sal_base, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    wet3 = np.broadcast_to(wet, sal_arm.shape)

    w_arm = _c4.ramp_w(sal_arm[wet3])
    w_base = _c4.ramp_w(sal_base[wet3])

    if dS < 0:
        return bool(np.all(w_arm <= w_base))
    if dS > 0:
        return bool(np.all(w_arm >= w_base))
    return bool(np.array_equal(w_arm, w_base))


def salinity_load_through_ok(
    engine_sal: np.ndarray,
    disk_sal: np.ndarray,
    production_sal: np.ndarray,
    wet: np.ndarray,
    dS: float,
) -> bool:
    """spec decision 4, gate order item 4: three-way load-through per arm (B2's
    `load_through_ok` precedent, adapted for salinity's NaN-land convention).

    The original two-way check (engine-loaded array == array on disk) only catches a
    *loader* bug -- it cannot catch a *writer* bug where the offset was silently never
    applied: a written file byte-identical to the untouched production field despite a
    nonzero dS passes engine==disk trivially, and would also pass `ramp_ordering_ok`
    trivially wherever `<=`/`>=` is satisfied by exact equality. The actual detector is the
    third term: independently recompute the EXPECTED offset field from the untouched
    production array via the builder's own `offset_salinity` (single source of truth for the
    offset math, imported from `build_baltic_c4_forcing`) and require engine-loaded ==
    on-disk == expected.

    `equal_nan=True` throughout: the real production salinity field's land convention is NaN
    (the opposite of B2's O2 file), and this function is also exercised directly against
    NaN-bearing synthetic fixtures in the test suite.
    """
    engine_sal = np.asarray(engine_sal, dtype=np.float64)
    disk_sal = np.asarray(disk_sal, dtype=np.float64)
    production_sal = np.asarray(production_sal, dtype=np.float64)
    expected = _c4.offset_salinity(production_sal, wet, dS)
    return bool(
        np.array_equal(engine_sal, disk_sal, equal_nan=True)
        and np.array_equal(disk_sal, expected, equal_nan=True)
    )


def _annualize(x: np.ndarray, n_year: int) -> np.ndarray:
    """Per-step biomass series -> per-year series (mean within each year's steps)."""
    x = np.asarray(x, dtype=float)
    if len(x) == n_year:
        return x
    if len(x) % n_year == 0:
        return x.reshape(n_year, -1).mean(axis=1)
    raise ValueError(f"series of {len(x)} not divisible into {n_year} years")


def run_c4(seeds=SEEDS) -> dict:
    """Run all five C4 arms across `seeds`, blocking checks first (spec decision 4, in the
    fixed order documented in this module's docstring), engine runs second, results JSON
    last.

    NOT invoked by the test suite (see module docstring) -- a full call is Task 4's
    deliverable.
    """
    import xarray as xr

    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine.config import _load_salinity_gate

    tmp = Path(tempfile.mkdtemp())
    base_cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    base_cfg["simulation.time.nyear"] = str(N_YEAR)
    n_sp = int(base_cfg["simulation.nspecies"])

    out_root = Path(tempfile.mkdtemp(prefix="c4_harness_"))

    # Assemble the non-baseline arm definitions and write their forcing artifacts. This must
    # happen before gate 1 below so gate 1 can point at the REAL zero-arm artifact this run's
    # engine runs and downstream checks go on to use (B2 precedent).
    delta_spec = json.loads(DEFAULT_SPEC_PATH.read_text())
    spec_arms = {a["name"]: a for a in delta_spec["arms"]}
    arm_defs = {"zero": ZERO_ARM_DEF, **{n: spec_arms[n] for n in SCENARIO_ARM_NAMES}}

    artifacts: dict[str, dict] = {}
    for name, arm_def in arm_defs.items():
        artifacts[name] = _c4.write_arm_dir(
            arm_def, out_root / name, DEFAULT_SAL_PATH, DEFAULT_GRID_PATH
        )

    with xr.open_dataset(DEFAULT_SAL_PATH) as ds:
        base_sal = ds[_c4._single_data_var(ds)].values.astype(np.float64)
    wet = _c4.load_wet_mask(base_sal[0], DEFAULT_GRID_PATH)

    # --- gate 1: builder zero-check, pointed at the actual zero-arm artifact ---
    with xr.open_dataset(artifacts["zero"]["sal_nc"]) as ds_zero:
        zero_vals = ds_zero[_c4._single_data_var(ds_zero)].values.astype(np.float64)
    zero_check_ok = bool(np.array_equal(base_sal, zero_vals, equal_nan=True))
    if not zero_check_ok:
        raise AssertionError(
            "C4 harness BLOCKED (gate 1, builder zero-check): the zero arm's own salinity "
            f"file ({artifacts['zero']['sal_nc']}) -- the actual file this run's engine runs "
            "and downstream checks will use -- diverges from the production input it was "
            "supposed to copy value-identically. Wiring bug -- no further checks or runs "
            "attempted."
        )

    all_arm_cfgs: dict[str, dict[str, str]] = {"baseline": dict(base_cfg)}
    for name, arm_def in arm_defs.items():
        all_arm_cfgs[name] = {**base_cfg, **arm_overlays(name, artifacts[name])}

    # --- gate 2: per-arm .constant-absence assert. arm_overlays (spec binding facts) never
    # emits .constant itself, so every call below inspects the SAME inherited value from
    # base_cfg -- in practice this asserts the production config is clean, which is the
    # actual hazard (a stray .constant key inherited from base would silently discard every
    # arm's .file lever at once, not just one arm's). Looped per-arm anyway because that is
    # what the spec's gate order literally names, and because it is what would catch a
    # future per-arm overlay that DID start setting .constant. ---
    for name, cfg in all_arm_cfgs.items():
        assert_no_salinity_constant(cfg, name)

    # --- gate 3: per-arm frame-count==24 assert on the written field ---
    disk_arrays: dict[str, np.ndarray] = {}
    for name in arm_defs:
        with xr.open_dataset(artifacts[name]["sal_nc"]) as ds2:
            disk_arrays[name] = ds2[_c4._single_data_var(ds2)].values.astype(np.float64)
        assert_arm_frame_count(disk_arrays[name], name)

    # --- gate 4: three-way load-through, per arm ---
    load_through: dict[str, bool] = {}
    for name, arm_def in arm_defs.items():
        _enabled, _mask, _s_low, _s_high, field = _load_salinity_gate(all_arm_cfgs[name], n_sp)
        if field is None or field._data is None:
            raise AssertionError(
                f"C4 harness BLOCKED (gate 4, arm={name}): _load_salinity_gate returned no "
                "spatial field for an arm with a resolvable movement.salinity.field.file -- "
                "the silent-fallback trap."
            )
        dS = float(arm_def["dS_PSU"])
        ok = salinity_load_through_ok(field._data, disk_arrays[name], base_sal, wet, dS)
        load_through[name] = ok
        if not ok:
            raise AssertionError(
                f"C4 harness BLOCKED (gate 4, arm={name}): three-way load-through check "
                "failed (engine-loaded == on-disk == recomputed-expected via "
                "offset_salinity). Wiring bug -- possibly a silent no-op offset write, not "
                "just a loader mismatch."
            )

    # --- gate 5: ramp ordering, per arm ---
    ramp_ordering: dict[str, bool] = {}
    for name, arm_def in arm_defs.items():
        dS = float(arm_def["dS_PSU"])
        ok = ramp_ordering_ok(disk_arrays[name], base_sal, wet, dS)
        ramp_ordering[name] = ok
        if not ok:
            raise AssertionError(
                f"C4 harness BLOCKED (gate 5, arm={name}): ramp ordering violates the dS "
                "sign. Wiring bug (guaranteed by monotonicity)."
            )

    # --- engine runs (all arms x all seeds) ---
    # biomass() returns the 9 focal fish species PLUS background predator groups
    # (GreySeal, Cormorant) as columns -- 11 total, verified against a real run during
    # development (task-3-report.md). `all_species` (every column) drives the zero-identity
    # gate, deliberately as broad as possible (a background-group divergence is as much a
    # wiring bug as a fish-species one); `report_species` (exactly the 9 named
    # `species.name.sp{0..n_sp-1}` entries) restricts the REPORTED final-decade section to
    # the nine Baltic species the spec asks for, never the wider bio_df column set.
    report_species = [base_cfg.get(f"species.name.sp{sp}", f"sp{sp}") for sp in range(n_sp)]
    all_species: list[str] | None = None
    raw_series: dict[str, dict[str, list[np.ndarray]]] = {a: {} for a in ARMS}
    for seed in seeds:
        for arm in ARMS:
            bio_df = PythonEngine().run_in_memory(all_arm_cfgs[arm], seed=seed).biomass()
            if all_species is None:
                all_species = [c for c in bio_df.columns if c not in ("Time", "species")]
                missing = set(report_species) - set(all_species)
                if missing:
                    raise AssertionError(
                        f"C4 harness BLOCKED (report species check): {missing} not found "
                        f"among biomass() columns {all_species} -- species.name.sp{{idx}} "
                        "mismatch."
                    )
            for name in all_species:
                raw_series[arm].setdefault(name, []).append(bio_df[name].to_numpy(dtype=float))
    assert all_species is not None

    # --- gate 6/7: zero-arm bit-identity to baseline, per seed (every biomass() column) ---
    identity_violations = []
    for i, seed in enumerate(seeds):
        for name in all_species:
            bio_baseline = raw_series["baseline"][name][i]
            bio_zero = raw_series["zero"][name][i]
            if not np.array_equal(bio_baseline, bio_zero):
                identity_violations.append({"seed": seed, "species": name})
    if identity_violations:
        raise AssertionError(
            "C4 harness BLOCKED (gate 7): zero arm is NOT bit-identical to baseline -- "
            f"{len(identity_violations)} (seed, species) violations, e.g. "
            f"{identity_violations[0]}. Wiring bug."
        )

    # --- REPORTED (no pass/fail): final-decade means + seed spreads, all nine species ---
    final_decade: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        final_decade[arm] = {}
        for name in report_species:
            per_seed = [
                float(_annualize(series, N_YEAR)[-10:].mean()) for series in raw_series[arm][name]
            ]
            final_decade[arm][name] = {
                "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed)),
                "per_seed": per_seed,
            }

    report = {
        "seeds": list(seeds),
        "arms": list(ARMS),
        "n_year": N_YEAR,
        "gates": {
            "zero_check": zero_check_ok,
            "load_through": load_through,
            "ramp_ordering": ramp_ordering,
        },
        "identity": {"pass": len(identity_violations) == 0, "violations": identity_violations},
        "final_decade_means": final_decade,
        "instruments": {name: artifacts[name]["instruments"] for name in arm_defs},
        "all_zero_events": {name: artifacts[name]["all_zero_events"] for name in arm_defs},
        "labels": REPORT_LABELS,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=float)
    return report


if __name__ == "__main__":
    out = run_c4()
    print(f"report written to {REPORT_PATH}")
    print(f"identity pass: {out['identity']['pass']}")
