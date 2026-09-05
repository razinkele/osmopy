#!/usr/bin/env python
"""C3 bioenergetics A/B harness (spec 2026-08-30-baltic-c3-bioen-stage1-design.md, Task 12).

Three arms x 5 house seeds x 50 yr on the certified Baltic config, bioenergetics OFF vs ON:
  baseline      -- production config, unmodified. No bioen machinery engaged.
  bioen         -- production config + the flat overlay at
                   `data/baltic/scenarios/c3_bioen/c3_bioen_arm.json` (Task 11's fit):
                   `module.bioenergetics.enabled=true`, the 9-species bioen parameter set, and
                   the two-layer temperature forcing
                   (`data/baltic/forcing/baltic_temperature_2layer_climatology.nc`, Task 10).
  bioen_plus2C  -- the bioen arm's config with `temperature.offset=2.0` added (the loader's own
                   factor/offset path, `_load_temperature_data`) -- reported only, not gated on
                   the decision rule.

`data/baltic/` production files are untouched by this overlay (spec decision 1) -- the arm is
built by merging the overlay dict in memory (`arm_config`), never by appending to
`baltic_all-parameters.csv`.

Cost (task-12-carried-items.md, superseding the spec's stale 3-6h serial estimate): the
Numba-kernel work (Tasks 4-7 of this plan) measured 149x/157.8x speed-ups on a 4-yr window and
~80-85x on a third config. The bioen-OFF 50yr x 5-seed baseline was itself measured at ~9-19 min
(Task 3's Gate A). The two bioen arms at 50 yr are NOT separately measured -- they rest on a
structural argument (one compiled kernel specialisation serves bioen-on and bioen-off at equal
horizon) plus a measurement that a materially-starving bioen population runs cheaper, not
dearer, than the config Task 4 timed. Point estimate: 27-57 min total. The load-bearing claim is
the margin, not the point estimate: even at 3x on the two bioen arms, the full run stays under
three hours serial -- comfortably below the threshold where a parallel harness would pay for
itself. Per-step cost tracks school count and the speed-up ratio itself grows with horizon
(81.9x at 1yr vs 149x at 4yr), so both directions of surprise are plausible; the margin is sized
to absorb them. Ruling R7 fixes 5 seeds x 50 yr -- not reduced here, and not run here: this
module BUILDS the harness (Task 12); the full run is Task 14's deliverable.

Pattern: `scripts/baltic_c4_salinity_ab.py` (gates first, engine runs second, committed JSON
last). Gate A is reused verbatim from `scripts/c3_gate_a_reference.py` (Task 3's committed
fixture, `docs/diagnostics/c3_gate_a_master_baseline.json`) rather than reimplemented.

BLOCKING gates (spec Sec.4), run in order before any engine run -- any failure is a wiring or
parity bug, stop, no interpretation of the run that follows:
  Gate A  -- baseline arm `array_equal` to the committed master fixture, all 5 seeds, every
             `biomass()` column (`c3_gate_a_reference.check_against_fixture`). Only meaningful
             at the fixture's own horizon (50 yr, 5 seeds); at any other `--years`/`--seeds`
             this harness marks it not_applicable rather than attempt a shape-mismatched check.
  Gate C  -- temperature load-through, three-way, per layer: engine-held array
             (`_load_temperature_data` on the arm config) == file on disk == builder
             recomputation from the CMEMS cache (skippable via `--no-recompute`: the cache read
             takes minutes), plus the wet-cell finite/range check. For `bioen_plus2C`: engine ==
             disk + offset, and, in float64, `engine_arm - engine_base == 2.0` exactly on wet
             cells (`assert_plus2_exact`).
  Gate D  -- structural and parameter asserts: temperature forcing frames==24, layers==2
             (`gate_d_frames_layers`); `config.bioen_zlayer` equals the spec Sec.2.4 assignment
             (`fit_baltic_bioen_params.SPECIES_ZLAYER`); no `temperature.value` in any bioen arm
             config (Java/engine precedence would shadow the file); no overlay applied to a
             bioen-off config (`arm_config`'s own guard); every `EngineConfig.bioen_*`/maturity
             field equals the fit script's emitted value per species (`gate_d_structure`).
  Gate E  -- zlayer wiring, engine-side: for one step and one seed, the per-species temperature
             array `_bioen_step` consumes (via its `debug_capture` hook) equals the assigned
             depth layer sampled at those schools' cells for every species; `is_out` schools are
             absent from it (NaN) (`gate_e_zlayer`).
  Gate F  -- thermal instrument: `phi_t(T_p) == 1.0` exactly per species; argmax of the offline
             `g_net(T)` equals the cited optimum +/-0.1 C; field phiT on the loaded temperature
             grid, restricted to each species' own habitat footprint, in (0, 1]; the
             `bioen_plus2C` arm's per-species habitat-mean `g_net` moves in the direction of
             sign(t_opt - T-bar) (`gate_f_thermal` / `gate_f_direction`).

REPORTED (no pass/fail, spec Sec.4): final-decade mean biomass per species and arm (5-seed mean,
spread), ratio to the certified final-decade means (`CERTIFIED_MEANS`,
docs/baltic_certification_2026-08-14.md) and to `ENVELOPE`
(`scripts/baltic_stability_certify.py`), persistence; realized ration f and e-bar/g-hat per
species (decision 7); length-at-age (`length_at_age`), paired baseline vs bioen, RMS % over ages
>= 1 yr; realized annual ingestion per species vs the Imax inflation factor (decision 17);
seeding diagnostics where available; `bioen_plus2C` minus `bioen` deltas; the spec's REPORT_LABELS
verbatim; the pre-registered decision rule (spec Sec.4), evaluated and printed as
`STAGE 2: WARRANTED` or `CLOSE BY CHARACTERIZATION (failed: ...)`.

This module's helpers (`arm_config`, `assert_plus2_exact`, `gate_c_load_through`,
`gate_d_frames_layers`, `gate_d_structure`, `gate_e_zlayer`, `gate_f_direction`,
`gate_f_thermal`, `length_from_age_bins`, `length_at_age`) are covered by
`tests/test_baltic_c3_harness.py`, each exercised on both a synthetic pass and a synthetic
violation. `run_c3` itself is NOT invoked by that suite or by this task -- a full 3-arm x
5-seed x 50yr run is Task 14's deliverable. Its wiring was instead validated manually during
development at a tiny horizon (see task-12-report.md for the full transcript): every BLOCKING
gate above was exercised against the REAL production files and the REAL fitted overlay, and a
smoke engine run (small `--years`, one seed) proved `run_c3`'s REPORTED section executes
end-to-end on the actual `biomass()` / `abundance_by_age()` / `biomass_by_age()` /
`meanEnetFaced` / `ingestion` shapes this module reads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Established scripts/ importlib-from-path idiom (see scripts/baltic_c4_salinity_ab.py).
_gate_a = _load_module("c3_gate_a_reference", "scripts/c3_gate_a_reference.py")
_fit = _load_module("fit_baltic_bioen_params", "scripts/fit_baltic_bioen_params.py")
_certify = _load_module("baltic_stability_certify", "scripts/baltic_stability_certify.py")
_temp_builder = _load_module(
    "build_baltic_temperature_forcing", "scripts/build_baltic_temperature_forcing.py"
)

from osmose.calibration.bioen_offline import BioenFixed, g_net as _g_net  # noqa: E402
from osmose.engine.processes.temp_function import phi_t as _phi_t  # noqa: E402

SEEDS = (42, 123, 7, 999, 2024)
N_YEAR = 50
ARMS = ("baseline", "bioen", "bioen_plus2C")

# The 5 stocks the decision rule (spec Sec.4) is scored against -- pikeperch/smelt/perch/
# stickleback are indicative-tier (not ICES-assessed, docs/baltic_certification_2026-08-14.md)
# and are reported but not part of the verdict.
ASSESSED_STOCKS = ("cod_west", "cod_east", "herring", "sprat", "flounder")

OVERLAY_PATH = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"
DEFAULT_GRID_PATH = ROOT / "data" / "baltic" / "baltic_grid.nc"
DEFAULT_TEMP_NC = ROOT / "data" / "baltic" / "forcing" / "baltic_temperature_2layer_climatology.nc"
DEFAULT_CACHE_DIR = ROOT / "data" / "cmems_cache" / "cmems_downloads"

REPORT_PATH = ROOT / "docs" / "diagnostics" / "baltic_c3_bioen_report.json"

# Harness-level output-instrumentation flags, applied to EVERY arm's config in `run_c3` --
# deliberately NOT inside `arm_config`, which must return exactly `dict(base_cfg)` for the
# baseline arm (see test_arm_config_overlay_only_on_bioen_arms). Verified bit-identical to a
# flags-off baseline run on a 2yr/1seed smoke check before being relied on here
# (task-12-report.md) -- `_collect_distributions` (simulate.py) only appends to StepOutput
# fields the standard biomass()/abundance() aggregation never reads, but that was checked, not
# assumed, per this branch's repeated "gate green over untested code" defect.
HARNESS_OUTPUT_FLAGS = {
    "output.abundance.byage.enabled": "true",
    "output.biomass.byage.enabled": "true",
    "output.bioen.ingest.enabled": "true",
}

# BioenFixed().m_share -- decision 7's "m" (maintenance share of Imax at the 16C anchor).
MAINTENANCE_SHARE = BioenFixed().m_share

# docs/baltic_certification_2026-08-14.md: midpoint of the 5-seed final-decade-mean range per
# species. Kept literal (B2/C4 precedent) so this stays byte-comparable across reruns of THIS
# harness; the certification doc itself is the source of truth for the underlying run.
CERTIFIED_MEANS = {
    "cod_west": 12874.506889773746,
    "cod_east": 65209.155398683084,
    "herring": 2547745.6387030575,
    "sprat": 1024567.4953097925,
    "flounder": 32937.1886317007,
    "perch": 43701.4643774093,
    "pikeperch": 1417534.8462227543,
    "smelt": 683302.9579076422,
    "stickleback": 81025.35735381991,
}

ENVELOPE = _certify.ENVELOPE

# spec Sec.4 REPORTED labels, restated verbatim.
REPORT_LABELS = [
    "Single optimum per species (cod's is size-dependent, Bjornsson & Steinarsson 2002).",
    "Herring optimum (15 C) is PROVISIONAL -- no herring growth optimum was retrieved in three "
    "literature searches.",
    "Secondary-source optima for flounder (19 C, via Kusakabe et al. 2016 quoting Fonds et al. "
    "1992) and smelt (15 C, via Krause 2008 quoting Vinni et al. 2004).",
    "Maintenance share m anchored on juvenile herring trials at 16 C (Bernreuther et al. 2012), "
    "transplanted to every species.",
    "No upper thermal limit at e_D = 1.5 -- phi_t(T) never turns back down at high T in this "
    "parameterisation.",
    "Perch and pikeperch are lagoon species fitted against the open-coast surface field -- "
    "phiT peaks at 0.7-0.8 in their actual lagoon habitat, inflating the fitted Imax.",
    "Ingestion is capped at Imax*w^beta BEFORE phiT (Java form) -- consumption inflation for "
    "cold-habitat species, decision 17.",
    "Food-unlimited offline fit vs a food-limited engine -- the in-engine A/B measures the "
    "emergent departure from the fitted curve, not a re-run of the fit.",
    "Larval phase (age < 1 yr) is unfitted -- decision 10, reported not fitted.",
    "Two-layer temperature is a proxy (surface nan-mean of 5 CMEMS depth levels; bottom = "
    "CMEMS bottomT), a climatology (1993-2021 monthly means, not a hindcast), and fo2 is off "
    "in Stage 1 (decision 19).",
    "Reproduction under bioen keeps the certified Python-side stock-recruitment regulation "
    "(decision 5) -- this A/B changes growth structure, not recruitment structure.",
]


# --------------------------------------------------------------------------------------- #
# arm_config
# --------------------------------------------------------------------------------------- #


def arm_config(base_cfg: dict[str, str], arm: str, overlay: dict[str, str]) -> dict[str, str]:
    """Config for one C3 arm (spec Sec.3.5): `baseline` returns `base_cfg` untouched (no bioen
    machinery engaged at all); `bioen` merges the overlay; `bioen_plus2C` merges the overlay
    plus `temperature.offset=2.0` (the loader's own factor/offset path,
    `osmose.engine.simulate._load_temperature_data`).

    Refuses (`AssertionError`, unconditionally, even for the `baseline` arm) to build ANY arm
    from an `overlay` whose own `module.bioenergetics.enabled` is not `"true"` -- the overlay
    argument is supposed to BE the C3 bioen arm; a bioen-off overlay reaching this function at
    all is a caller wiring bug (a stale/emptied overlay dict, e.g.), not a legitimate baseline
    request -- `arm="baseline"` never even applies the overlay, but the guard still fires,
    because building any arm from a broken overlay argument should never silently succeed.
    """
    if overlay.get("module.bioenergetics.enabled", "").lower() != "true":
        raise AssertionError(
            f"C3 harness BLOCKED (arm_config, arm={arm}): overlay module.bioenergetics.enabled "
            f"is {overlay.get('module.bioenergetics.enabled')!r}, not 'true' -- refusing to "
            "build any arm, including baseline, from a bioen-off overlay. The whole point of "
            "this overlay is turning bioenergetics on; passing a bioen-off one here is a "
            "caller wiring bug, not a valid baseline request."
        )
    if arm == "baseline":
        return dict(base_cfg)
    if arm == "bioen":
        return {**base_cfg, **overlay}
    if arm == "bioen_plus2C":
        return {**base_cfg, **overlay, "temperature.offset": "2.0"}
    raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")


# --------------------------------------------------------------------------------------- #
# Gate C -- temperature load-through
# --------------------------------------------------------------------------------------- #


def assert_plus2_exact(arm: np.ndarray, base: np.ndarray, wet: np.ndarray) -> None:
    """Gate C (+2C exactness): `arm - base == 2.0` exactly, in float64, on wet cells.

    The two arrays must already be the ENGINE-HELD arrays (post `_load_temperature_data`,
    factor/offset already applied), not raw disk reads -- see `gate_c_load_through`, which
    forms these from the same underlying float32 file widened to float64 (exact) plus an
    offset of 2.0 (exactly representable, no further rounding for values in the Baltic's
    physical range). Raises on ANY wet-cell mismatch, however small -- this is not a
    tolerance check, it is proof the loader applied its own offset arithmetic and nothing else.
    """
    arm = np.asarray(arm, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    wet_b = np.broadcast_to(wet, arm.shape)
    diff = arm[wet_b] - base[wet_b]
    if not np.all(diff == 2.0):
        n_bad = int(np.sum(diff != 2.0))
        raise AssertionError(
            f"Gate C BLOCKED (+2C exactness): {n_bad}/{diff.size} wet-cell(s) have "
            "engine_arm - engine_base != 2.0 exactly in float64 -- offset arithmetic wiring bug."
        )


def gate_c_load_through(
    arm_cfg: dict[str, str],
    base_nc: Path,
    builder_recompute: np.ndarray | None,
    wet: np.ndarray,
) -> None:
    """Gate C: engine-held temperature array == on-disk forcing file == builder recomputation
    from the CMEMS cache (NaN-aware, `array_equal`), plus the wet-cell finite/range check.

    `builder_recompute`: the (24,2,ny,nx) float64 array from
    `build_baltic_temperature_forcing.build(...)`, computed ONCE by the caller (the cache read
    takes minutes) and shared across both bioen arms -- pass `None` to skip this leg with a
    printed note (`--no-recompute`); a skip is never silent.

    For `bioen_plus2C` (a `temperature.offset` key in `arm_cfg`): the on-disk comparison is
    `disk + offset` (not disk itself), and `assert_plus2_exact` additionally checks the engine
    array against a freshly-loaded offset=0 reading of the SAME config (the loader's own
    factor/offset code path, run twice).
    """
    import xarray as xr

    from osmose.engine.simulate import _load_temperature_data

    engine_data = _load_temperature_data(arm_cfg, None)
    if engine_data is None or engine_data._data is None:
        raise AssertionError(
            "Gate C BLOCKED: _load_temperature_data returned no spatial field for a bioen arm "
            "config with a resolvable temperature.filename -- the silent-fallback trap."
        )
    engine_arr = np.asarray(engine_data._data, dtype=np.float64)

    varname = arm_cfg.get("temperature.varname", "temperature")
    with xr.open_dataset(base_nc) as ds:
        disk_arr = ds[varname].values.astype(np.float64)

    offset = float(arm_cfg.get("temperature.offset", "0.0"))
    if offset == 0.0:
        if not np.array_equal(engine_arr, disk_arr, equal_nan=True):
            raise AssertionError(
                "Gate C BLOCKED: engine-held temperature != on-disk forcing file (offset=0 arm)."
            )
    else:
        expected = disk_arr + offset
        if not np.array_equal(engine_arr, expected, equal_nan=True):
            raise AssertionError(
                f"Gate C BLOCKED (offset={offset} arm): engine-held temperature != "
                "on-disk file + offset."
            )
        base_only_cfg = {k: v for k, v in arm_cfg.items() if k != "temperature.offset"}
        base_engine_data = _load_temperature_data(base_only_cfg, None)
        if base_engine_data is None or base_engine_data._data is None:
            raise AssertionError(
                "Gate C BLOCKED: could not re-load the offset=0 baseline reading for the "
                "+2C exactness check."
            )
        assert_plus2_exact(engine_arr, np.asarray(base_engine_data._data, dtype=np.float64), wet)

    if builder_recompute is None:
        print(
            "Gate C: skipping builder-recomputation-vs-disk check (--no-recompute) -- "
            "on-disk forcing file NOT independently verified against the CMEMS cache this run."
        )
    else:
        if not np.array_equal(
            disk_arr, np.asarray(builder_recompute, dtype=np.float64), equal_nan=True
        ):
            raise AssertionError(
                "Gate C BLOCKED: on-disk forcing file != builder recomputation from the CMEMS "
                "cache -- the shipped file no longer matches scripts/"
                "build_baltic_temperature_forcing.py's own pipeline."
            )

    wet_b = np.broadcast_to(wet, engine_arr.shape)
    wet_vals = engine_arr[wet_b]
    if not np.isfinite(wet_vals).all():
        raise AssertionError("Gate C BLOCKED: non-finite temperature on a wet cell.")
    lo, hi = -2.0 + offset, 30.0 + offset
    if not (float(wet_vals.min()) >= lo and float(wet_vals.max()) <= hi):
        raise AssertionError(
            f"Gate C BLOCKED: wet-cell temperature outside [{lo}, {hi}] C "
            "(builder's own physical-range pin, shifted by this arm's offset)."
        )


# --------------------------------------------------------------------------------------- #
# Gate D -- structural and parameter asserts
# --------------------------------------------------------------------------------------- #


def gate_d_frames_layers(temp_nc: Path, varname: str = "temperature") -> None:
    """Gate D: the two-layer temperature forcing has exactly 24 frames and 2 layers.

    No conditional skip: unlike `gate_c_load_through`'s explicit, printed
    `--no-recompute` skip, this check always opens the file. `PhysicalData.get_grid` indexes
    `step % <loaded frame count>`, not any declared nsteps.year metadata (CLAUDE.md), so a
    silently-skipped shape check here would be exactly the "gate never executed" defect this
    branch keeps re-finding.
    """
    import xarray as xr

    with xr.open_dataset(temp_nc) as ds:
        arr = ds[varname].values
    if arr.ndim != 4:
        raise AssertionError(
            f"Gate D BLOCKED: {temp_nc}:{varname} is not 4-D (time,layer,y,x); got shape "
            f"{arr.shape} -- the two-layer contract requires a depth axis."
        )
    n_frames, n_layers = arr.shape[0], arr.shape[1]
    if n_frames != 24:
        raise AssertionError(f"Gate D BLOCKED: {temp_nc} has {n_frames} frame(s), expected 24.")
    if n_layers != 2:
        raise AssertionError(f"Gate D BLOCKED: {temp_nc} has {n_layers} layer(s), expected 2.")


# CSV/overlay key prefix -> EngineConfig attribute (only focal-length, species-indexed bioen
# and maturity fields -- osmose/engine/config.py:2489-2530). Background predators (sp15/16)
# authored only species.beta + the Imax family into arrays sized n_species, not
# n_species+n_background; `gate_d_structure` skips any sp >= engine_cfg.n_species rather than
# index past the end of those arrays.
_BIOEN_FIELD_MAP = {
    "species.bioen.mobilized.tp.sp": "bioen_tp",
    "species.bioen.mobilized.e.d.sp": "bioen_e_d",
    "species.bioen.mobilized.e.mobi.sp": "bioen_e_mobi",
    "species.bioen.maint.e.maint.sp": "bioen_e_maint",
    "species.bioen.maint.energy.c_m.sp": "bioen_c_m",
    "species.bioen.assimilation.sp": "bioen_assimilation",
    "species.beta.sp": "bioen_beta",
    "species.maturity.eta.sp": "bioen_eta",
    "species.maturity.r.sp": "bioen_r",
    "species.maturity.m0.sp": "bioen_m0",
    "species.maturity.m1.sp": "bioen_m1",
    "species.zlayer.sp": "bioen_zlayer",
    "predation.ingestion.rate.max.sp": "bioen_i_max",
    "predation.larval.ingestion.rate.increase.ratio.sp": "bioen_theta",
    "predation.c.bioen.sp": "bioen_c_rate",
    "species.oxygen.c1.sp": "bioen_o2_c1",
    "species.oxygen.c2.sp": "bioen_o2_c2",
    "species.bioen.forage.k_for.sp": "bioen_k_for",
}


def gate_d_structure(
    arm_cfg: dict[str, str],
    engine_cfg,
    fit_csv_values: dict[str, str],
    expected_zlayer: dict[int, int] | None = None,
) -> None:
    """Gate D (parameter half): no `temperature.value` in the arm config (Java/engine
    precedence tries `.value` before `.filename`, so a stray constant would silently shadow the
    forcing file); `engine_cfg.bioen_zlayer` matches `expected_zlayer` (spec Sec.2.4's
    per-species assignment); every `EngineConfig.bioen_*`/maturity field the fit script emitted
    equals the parsed engine value, per species (`float(csv) == float(engine)` exactly -- the
    CSV holds `repr()`/`np.format_float_scientific` round-trip-exact floats, so this is not a
    tolerance comparison).

    `gate_d_frames_layers` (frames==24, layers==2) is a separate, unconditional function --
    NOT called from here, so this function stays usable against a synthetic `arm_cfg` that
    names no real file (see tests/test_baltic_c3_harness.py).
    """
    if arm_cfg.get("temperature.value", ""):
        raise AssertionError(
            "Gate D BLOCKED: temperature.value is set on a bioen arm config -- the engine's "
            "loader precedence tries temperature.value BEFORE temperature.filename "
            "(_load_temperature_data), so a stray constant here would silently shadow the "
            "two-layer forcing file."
        )

    if expected_zlayer is not None:
        got_zlayer = np.asarray(engine_cfg.bioen_zlayer)
        for sp, exp in expected_zlayer.items():
            got = int(got_zlayer[sp])
            if got != int(exp):
                raise AssertionError(
                    f"Gate D BLOCKED: EngineConfig.bioen_zlayer[{sp}] = {got} != expected "
                    f"{int(exp)} (fit_baltic_bioen_params.SPECIES_ZLAYER)."
                )

    n_sp = int(engine_cfg.n_species)
    for key, csv_val in fit_csv_values.items():
        for prefix, attr in _BIOEN_FIELD_MAP.items():
            if not key.startswith(prefix):
                continue
            sp_str = key[len(prefix) :]
            if not sp_str.isdigit():
                continue
            sp = int(sp_str)
            if sp >= n_sp:
                break  # background predator index -- not in this focal-length array
            arr = getattr(engine_cfg, attr, None)
            if arr is None:
                break
            got = float(arr[sp])
            want = float(csv_val)
            if got != want:
                raise AssertionError(
                    f"Gate D BLOCKED: EngineConfig.{attr}[{sp}] = {got!r} != CSV {key} = "
                    f"{want!r} -- case-mismatch or unresolved-include trap."
                )
            break


# --------------------------------------------------------------------------------------- #
# Gate E -- zlayer wiring, engine-side
# --------------------------------------------------------------------------------------- #


def gate_e_zlayer(arm_cfg: dict[str, str], seed: int = 0, step: int = 12) -> dict:
    """Gate E: `_bioen_step`'s per-school temperature array (`debug_capture["temp_c"]`) equals
    each species' assigned depth layer sampled at its schools' cells, for a synthetic batch of
    schools covering BOTH zlayer groups plus at least one `is_out` school per species.

    Every precondition is asserted non-vacuous before the check it guards (this branch has
    repeatedly shipped gates that passed only because the state they inspected was empty):
    at least one `is_out` school exists; both zlayer groups (0, 1) are represented among
    in-domain schools; every in-domain, in-range school gets a FINITE temp_c (not merely equal
    to something). Schools are placed on a cell that is wet in the grid mask AND finite in
    every temperature layer at `step` -- `fit_baltic_bioen_params.habitat_t24`'s own land-mask
    warning proves grid-wet-but-forcing-NaN cells exist, so a naive random-wet placement would
    make the finiteness assert fail for reasons unrelated to zlayer wiring.
    """
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import _bioen_step, _load_temperature_data
    from osmose.engine.state import SchoolState

    engine_cfg = EngineConfig.from_dict(arm_cfg)
    temp_data = _load_temperature_data(arm_cfg, None)
    if temp_data is None or temp_data._data is None:
        raise AssertionError("Gate E BLOCKED: no temperature field for a bioen arm config.")

    grid = Grid.from_netcdf(DEFAULT_GRID_PATH)
    zlayer = np.asarray(engine_cfg.bioen_zlayer)
    n_sp = int(engine_cfg.n_species)

    finite_all_layers = np.ones(grid.ocean_mask.shape, dtype=bool)
    for layer in range(temp_data.n_layers):
        finite_all_layers &= np.isfinite(temp_data.get_grid(step, layer=layer))
    candidates = np.argwhere(grid.ocean_mask & finite_all_layers)
    if len(candidates) == 0:
        raise AssertionError(
            "Gate E BLOCKED: no wet cell is finite in every temperature layer at this step -- "
            "cannot build a non-vacuous synthetic state."
        )
    cy, cx = candidates[len(candidates) // 2]

    species_id = np.repeat(np.arange(n_sp, dtype=np.int32), 2)  # one in-domain + one is_out each
    is_out = np.tile(np.array([False, True]), n_sp)
    n = len(species_id)
    state = SchoolState.create(n, species_id=species_id)
    state = state.replace(
        cell_y=np.where(is_out, -1, int(cy)).astype(np.int32),
        cell_x=np.where(is_out, -1, int(cx)).astype(np.int32),
        is_out=is_out,
        weight=np.full(n, 1.0),
        abundance=np.full(n, 1000.0),
    )

    in_domain_species = set(species_id[~is_out].tolist())
    zlayer_groups_present = {int(zlayer[sp]) for sp in in_domain_species}
    if zlayer_groups_present != {0, 1}:
        raise AssertionError(
            f"Gate E test-construction bug: in-domain synthetic schools cover zlayer group(s) "
            f"{zlayer_groups_present}, need both {{0, 1}} -- SPECIES_ZLAYER assignment or "
            "n_species changed; this gate would not be exercising both depth layers."
        )
    if not is_out.any():
        raise AssertionError("Gate E test-construction bug: no is_out school in the batch.")

    debug: dict = {}
    _bioen_step(state, engine_cfg, temp_data, step, o2_data=None, debug_capture=debug)

    temp_c = debug["temp_c"]
    got_is_out = debug["is_out"]
    got_species = debug["species_id"]

    if not got_is_out.any():
        raise AssertionError(
            "Gate E BLOCKED: debug_capture reports zero is_out schools -- state construction "
            "lost is_out, or _bioen_step's own bookkeeping dropped it."
        )
    if not np.isnan(temp_c[got_is_out]).all():
        raise AssertionError(
            f"Gate E BLOCKED: is_out school(s) have non-NaN temp_c={temp_c[got_is_out]} -- "
            "out-of-domain schools must be excluded from the thermal lookup (spec decision 18)."
        )

    in_mask = ~got_is_out
    if not np.isfinite(temp_c[in_mask]).all():
        raise AssertionError(
            "Gate E BLOCKED: in-domain school(s) have a non-finite temp_c -- the placement "
            "cell is not finite in some species' assigned layer."
        )

    checked = 0
    for sp in range(n_sp):
        mask = in_mask & (got_species == sp)
        if not mask.any():
            continue
        layer = int(zlayer[sp])
        expected = float(temp_data.get_grid(step, layer=layer)[cy, cx])
        got = temp_c[mask]
        if not np.all(got == expected):
            raise AssertionError(
                f"Gate E BLOCKED: sp{sp} (zlayer={layer}) temp_c={got.tolist()} != "
                f"field[step={step},layer={layer},y={cy},x={cx}]={expected} -- zlayer wiring bug."
            )
        checked += 1
    if checked != n_sp:
        raise AssertionError(
            f"Gate E test-construction bug: only {checked}/{n_sp} species had an in-domain "
            "school reach the check -- not every species was exercised."
        )
    return {"cell": (int(cy), int(cx)), "step": step, "n_species_checked": checked}


# --------------------------------------------------------------------------------------- #
# Gate F -- thermal instrument
# --------------------------------------------------------------------------------------- #


def gate_f_direction(
    t_bar: dict[str, float],
    t_opt: dict[str, float],
    g_base: dict[str, float],
    g_plus2: dict[str, float],
) -> dict[str, bool]:
    """Gate F (direction half): per species, `g_plus2 > g_base` iff `t_opt > t_bar` -- moving
    2C toward the cited optimum must raise the habitat-mean net growth, moving away must lower
    it. Deterministic given `g_net`'s shape (single interior maximum at T_p, monotone
    Arrhenius maintenance), so any violation is a wiring bug, not an ecological surprise (spec
    Gate F). Assumes T-bar+2 has not overshot t_opt in the wrong direction, true for every
    fitted Baltic species (T-bar sits 1.7-5.3 C below t_opt, README.md).
    """
    out: dict[str, bool] = {}
    for name in t_bar:
        want_increase = t_opt[name] > t_bar[name]
        got_increase = g_plus2[name] > g_base[name]
        if got_increase != want_increase:
            raise AssertionError(
                f"Gate F BLOCKED (direction): {name}: t_opt={t_opt[name]} t_bar={t_bar[name]} "
                f"g_base={g_base[name]} g_plus2={g_plus2[name]} -- direction mismatch, "
                "wiring bug."
            )
        out[name] = True
    return out


def gate_f_thermal(engine_cfg, temp_field, habitat_masks: dict[str, np.ndarray]) -> dict:
    """Gate F: `phi_t(T_p) == 1.0` exactly per species; argmax of the offline `g_net(T)` equals
    the cited optimum +/- 0.1 C; field phiT, restricted to each species' own habitat footprint
    (wet + finite cells only -- `phi_t(NaN)` is NaN, not an error, so an unrestricted check
    would silently pass on land), lies in (0, 1]; the habitat-mean `g_net` moves in the
    direction of sign(t_opt - T-bar) under +2C (`gate_f_direction`).

    `temp_field`: the loaded `PhysicalData` (or a raw `(24,2,ny,nx)` array) for the arm.
    `habitat_masks`: `{species_name: (ny,nx) bool}`, unioned across life stages, matching
    `fit_baltic_bioen_params.habitat_t24`'s own footprint convention.

    Returns the per-species diagnostics (t_bar, t_opt, T_p, phi_t at T_p, argmax, g_base,
    g_plus2) for the REPORTED section.
    """
    fx = BioenFixed()
    data = temp_field._data if hasattr(temp_field, "_data") else np.asarray(temp_field)
    n_layers = data.shape[1] if data.ndim == 4 else 1

    out: dict[str, dict] = {}
    t_bar_d: dict[str, float] = {}
    t_opt_d: dict[str, float] = {}
    g_base_d: dict[str, float] = {}
    g_plus2_d: dict[str, float] = {}

    for sp in range(int(engine_cfg.n_species)):
        name = engine_cfg.species_names[sp]
        if name not in _fit.SPECIES_T_OPT:
            continue  # not a C3-fitted species (shouldn't happen on the production Baltic set)
        tp = float(engine_cfg.bioen_tp[sp])
        imax = float(engine_cfg.bioen_i_max[sp])
        c_m = float(engine_cfg.bioen_c_m[sp])
        e_mobi = float(engine_cfg.bioen_e_mobi[sp])
        e_d = float(engine_cfg.bioen_e_d[sp])
        t_opt = _fit.SPECIES_T_OPT[name]

        val = float(_phi_t(np.array([tp]), e_mobi, e_d, tp)[0])
        if val != 1.0:
            raise AssertionError(f"Gate F BLOCKED: phi_t(T_p) for {name} = {val!r} != 1.0 exactly.")

        t_grid = np.linspace(t_opt - 15.0, t_opt + 15.0, 30001)
        g_curve = _g_net(t_grid, imax, c_m, tp, fx)
        argmax_t = float(t_grid[int(np.argmax(g_curve))])
        if abs(argmax_t - t_opt) > 0.1:
            raise AssertionError(
                f"Gate F BLOCKED: argmax g_net for {name} = {argmax_t:.4f} C, cited optimum "
                f"{t_opt} C, |diff| = {abs(argmax_t - t_opt):.4f} > 0.1 C."
            )

        mask = habitat_masks[name]
        layer = int(engine_cfg.bioen_zlayer[sp]) if n_layers > 1 else 0
        frames = data[:, layer] if data.ndim == 4 else data  # (24, ny, nx)
        cell_vals = frames[:, mask]  # (24, n_habitat_cells)
        finite = np.isfinite(cell_vals)
        if not finite.any():
            raise AssertionError(
                f"Gate F BLOCKED: no finite habitat cell for {name} at layer {layer} across "
                "any of the 24 frames."
            )
        phi_field = _phi_t(cell_vals[finite], e_mobi, e_d, tp)
        if not (np.all(phi_field > 0.0) and np.all(phi_field <= 1.0)):
            raise AssertionError(
                f"Gate F BLOCKED: field phi_t for {name} leaves (0, 1] -- "
                f"min={float(phi_field.min())} max={float(phi_field.max())}."
            )

        t_bar = float(cell_vals[finite].mean())
        g_base = float(_g_net(np.array([t_bar]), imax, c_m, tp, fx)[0])
        g_plus2 = float(_g_net(np.array([t_bar + 2.0]), imax, c_m, tp, fx)[0])

        t_bar_d[name] = t_bar
        t_opt_d[name] = t_opt
        g_base_d[name] = g_base
        g_plus2_d[name] = g_plus2
        out[name] = {
            "t_bar": t_bar,
            "t_opt": t_opt,
            "t_p": tp,
            "phi_t_at_tp": val,
            "argmax_g_net": argmax_t,
            "g_base": g_base,
            "g_plus2": g_plus2,
        }

    gate_f_direction(t_bar_d, t_opt_d, g_base_d, g_plus2_d)
    return out


# --------------------------------------------------------------------------------------- #
# Length-at-age instrument
# --------------------------------------------------------------------------------------- #


def length_from_age_bins(
    abundance_df: pd.DataFrame,
    biomass_df: pd.DataFrame,
    cf: float,
    b: float,
    species: str,
) -> dict[int, float]:
    """Per-age-bin length (cm) from paired `abundance_by_age()`/`biomass_by_age()` frames.

    `OsmoseResults.abundance_by_age()`/`biomass_by_age()` (in-memory mode) return LONG form:
    columns `time, species, bin, value`, `bin` a STRING age-bin index -- confirmed against a
    real 2-yr smoke run (task-12-report.md), not the wide `Time, age, <species>` layout an
    earlier draft of this instrument assumed. Callers pass frames already restricted to the
    time window of interest (a single final year, or several final-decade years to be averaged
    -- this function groups by `bin` and takes the mean `value` within each, so either
    convention works transparently).

    Mean weight per fish (g) = biomass (tonnes) * 1e6 / abundance (count) -- the same
    tonnes<->grams convention `osmose.engine.config`'s `w_mean = cf * L**b * 1e-6` uses
    (confirmed on a real cod_west run: bins with weight saturating at the config's own
    `cf * Linf**b` reproduce `species.linf.sp0 = 110` cm exactly, task-12-report.md). Bins with
    non-positive abundance are dropped (no fish to average a weight over -- bin 0 in
    particular is egg-dominated, CLAUDE.md's by-age cutoff caveat). `length = (weight_g / cf)
    ** (1/b)`, the inverse of the vBGF weight-length relation `bioen_offline.vbgf_weight` uses.
    """
    a = abundance_df[abundance_df["species"] == species].groupby("bin")["value"].mean()
    w = biomass_df[biomass_df["species"] == species].groupby("bin")["value"].mean()
    out: dict[int, float] = {}
    for bin_str in a.index:
        abundance = float(a[bin_str])
        if abundance <= 0 or bin_str not in w.index:
            continue
        weight_g = float(w[bin_str]) * 1e6 / abundance
        out[int(bin_str)] = (weight_g / cf) ** (1.0 / b)
    return out


def length_at_age(results, config) -> dict[str, np.ndarray]:
    """Per-species length-at-age array (index = age bin, cm), final-year `abundance_by_age()`/
    `biomass_by_age()` via `length_from_age_bins`. `config`: an `EngineConfig` (species'
    `condition_factor`/`allometric_power` are focal-indexed the same as `species_names`).

    Requires `output.abundance.byage.enabled`/`output.biomass.byage.enabled` on the run that
    produced `results` (`HARNESS_OUTPUT_FLAGS`) -- raises the same `FileNotFoundError`
    `OsmoseResults` does if they were off.
    """
    ab_all = results.abundance_by_age()
    bb_all = results.biomass_by_age()
    t_final = ab_all["time"].max()
    ab_f = ab_all[ab_all["time"] == t_final]
    bb_f = bb_all[bb_all["time"] == t_final]

    out: dict[str, np.ndarray] = {}
    for sp in range(int(config.n_species)):
        name = config.species_names[sp]
        cf = float(config.condition_factor[sp])
        b = float(config.allometric_power[sp])
        per_bin = length_from_age_bins(ab_f, bb_f, cf, b, name)
        if not per_bin:
            out[name] = np.zeros(0, dtype=np.float64)
            continue
        arr = np.full(max(per_bin) + 1, np.nan, dtype=np.float64)
        for k, v in per_bin.items():
            arr[k] = v
        out[name] = arr
    return out


# --------------------------------------------------------------------------------------- #
# run_c3
# --------------------------------------------------------------------------------------- #


def _annualize(x: np.ndarray, n_year: int) -> np.ndarray:
    """Per-step series -> per-year series (mean within each year's steps)."""
    x = np.asarray(x, dtype=float)
    if len(x) == n_year:
        return x
    if len(x) % n_year == 0:
        return x.reshape(n_year, -1).mean(axis=1)
    raise ValueError(f"series of {len(x)} not divisible into {n_year} years")


def _final_window_mean(res, output_type: str, name: str, window_years: float = 10.0) -> float:
    """Mean of a per-species bioen output over the final `window_years` of a run.

    In-memory per-species outputs (`_build_dataframes_from_outputs` -> `_read_species_output`)
    keep columns `["Time", <output_type>, "species"]` -- capital "Time", and the value column
    named after the output type itself -- NOT the `time`/`value` 2D-output convention
    `length_from_age_bins` reads (confirmed against a real run, task-12-report.md; the two
    in-memory output families use different column contracts and neither is the "wide
    Time,age,<species>" layout the brief's original length-at-age stub assumed either).
    """
    df = res._read_species_output(output_type, name)
    t_final = df["Time"].max()
    window = df[df["Time"] > t_final - window_years]
    return float(window[output_type].mean())


def _habitat_mask(
    raw_cfg: dict[str, str], name: str, config_dir: Path, ny: int, nx: int
) -> np.ndarray:
    """Boolean (ny, nx) habitat footprint for one species, unioned across life-stage maps --
    same convention as `fit_baltic_bioen_params.habitat_t24` (reuses its map-file resolver)."""
    from osmose.engine.movement_maps import _load_csv_grid

    mask = np.zeros((ny, nx), dtype=bool)
    for mf in _fit._species_map_files(raw_cfg, name, config_dir):
        grid = _load_csv_grid(Path(mf), ny, nx)
        mask |= np.nan_to_num(grid, nan=0.0) > 0
    return mask


def evaluate_decision_rule(
    final_decade: dict[str, dict[str, dict]],
    ration: dict[str, dict],
) -> dict:
    """The pre-registered decision rule (spec Sec.4), extracted from `run_c3` so it can be
    exercised without an engine run (task-12-review.md finding 1).

    Every criterion is three-way: pass / fail / undetermined. A NaN input (`bi_mean`,
    `b_mean`, or `e_over_g`) makes that species' criterion UNDETERMINED, never a silent pass
    (the pre-fix bug: `nan < threshold` is `False`, so a NaN `e_over_g` read as "criterion (ii)
    satisfied") and never a silent fail either -- collapsing undetermined into failure would be
    a different lie. Thresholds and algebra are unchanged from the original inline block; only
    the NaN handling and the three-way status are new.
    """
    failed: list[str] = []
    undetermined: list[str] = []
    criteria: dict[str, dict[str, str]] = {name: {} for name in ASSESSED_STOCKS}

    for name in ASSESSED_STOCKS:
        b_mean = final_decade["baseline"][name]["mean"]
        bi_mean = final_decade["bioen"][name]["mean"]
        if not (np.isfinite(b_mean) and np.isfinite(bi_mean)):
            criteria[name]["i_no_structural_collapse"] = "undetermined"
            undetermined.append(
                f"(i) no-structural-collapse: {name} bioen or baseline final-decade mean is NaN"
            )
        elif bi_mean < 0.10 * b_mean:
            criteria[name]["i_no_structural_collapse"] = "fail"
            failed.append(
                f"(i) no-structural-collapse: {name} bioen/baseline = {bi_mean / b_mean:.3f} < 0.10"
            )
        else:
            criteria[name]["i_no_structural_collapse"] = "pass"

        e_over_g = ration[name]["e_over_g"]
        if not np.isfinite(e_over_g):
            criteria[name]["ii_ebar_ghat"] = "undetermined"
            undetermined.append(f"(ii) e-bar/g-hat: {name} e_over_g is NaN (g_hat == 0)")
        elif e_over_g < 0.6:
            criteria[name]["ii_ebar_ghat"] = "fail"
            failed.append(f"(ii) e-bar/g-hat: {name} = {e_over_g:.3f} < 0.6")
        else:
            criteria[name]["ii_ebar_ghat"] = "pass"

    within_factor_2 = 0
    within_factor_2_undetermined = 0
    for name in ASSESSED_STOCKS:
        bi_mean = final_decade["bioen"][name]["mean"]
        cert = CERTIFIED_MEANS[name]
        ratio = bi_mean / cert
        if not np.isfinite(ratio):
            criteria[name]["iii_bounded_displacement"] = "undetermined"
            undetermined.append(f"(iii) bounded-displacement: {name} bioen/certified ratio is NaN")
            within_factor_2_undetermined += 1
            continue
        if not (0.2 <= ratio <= 5.0):
            criteria[name]["iii_bounded_displacement"] = "fail"
            failed.append(
                f"(iii) bounded-displacement: {name} bioen/certified = {ratio:.3f} "
                "outside [0.2, 5.0]"
            )
        else:
            criteria[name]["iii_bounded_displacement"] = "pass"
        if 0.5 <= ratio <= 2.0:
            within_factor_2 += 1

    if within_factor_2 < 3:
        if within_factor_2 + within_factor_2_undetermined < 3:
            failed.append(
                f"(iii) bounded-displacement: only {within_factor_2}/5 assessed stocks within "
                "a factor of 2 of their certified mean (need >= 3)"
            )
        else:
            undetermined.append(
                f"(iii) bounded-displacement: only {within_factor_2}/5 assessed stocks "
                f"confirmed within a factor of 2 of their certified mean, "
                f"{within_factor_2_undetermined} undetermined -- cannot confirm or rule out "
                "the >= 3 threshold"
            )

    if failed and undetermined:
        verdict = f"CLOSE BY CHARACTERIZATION (failed: {failed}; undetermined: {undetermined})"
    elif failed:
        verdict = f"CLOSE BY CHARACTERIZATION (failed: {failed})"
    elif undetermined:
        verdict = f"UNDETERMINED (could not evaluate: {undetermined})"
    else:
        verdict = "STAGE 2: WARRANTED"

    return {
        "failed": failed,
        "undetermined": undetermined,
        "criteria": criteria,
        "verdict": verdict,
    }


def run_c3(
    seeds=SEEDS,
    n_year: int = N_YEAR,
    no_recompute: bool = False,
) -> dict:
    """Run all three C3 arms across `seeds`, BLOCKING gates first (in the fixed order
    documented in this module's docstring), engine runs second, results JSON last.

    NOT invoked by the test suite (see module docstring) -- a full call at `seeds=SEEDS,
    n_year=N_YEAR` is Task 14's deliverable, not Task 12's.
    """

    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import _load_temperature_data

    warnings.simplefilter("ignore")

    tmp = Path(tempfile.mkdtemp(prefix="c3_bioen_harness_"))
    demo_info = osmose_demo("baltic", tmp)
    config_file = Path(demo_info["config_file"])
    config_dir = config_file.parent
    raw_base = dict(OsmoseConfigReader().read(str(config_file)))
    raw_base["simulation.time.nyear"] = str(n_year)
    n_sp = int(raw_base["simulation.nspecies"])

    overlay = json.loads(OVERLAY_PATH.read_text())
    overlay = {k: v for k, v in overlay.items() if k != "_meta"}

    all_arm_cfgs: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        cfg = arm_config(raw_base, arm, overlay)
        all_arm_cfgs[arm] = {**cfg, **HARNESS_OUTPUT_FLAGS}

    base_config = EngineConfig.from_dict(raw_base)  # bioen-off: gives species/growth structure
    sp_index = {raw_base[f"species.name.sp{i}"]: i for i in range(n_sp)}
    expected_zlayer = {sp_index[name]: zl for name, zl in _fit.SPECIES_ZLAYER.items()}

    ny = int(raw_base["grid.nlat"])
    nx = int(raw_base["grid.nlon"])
    wet = Grid.from_netcdf(DEFAULT_GRID_PATH).ocean_mask

    targets, sp_index2, t24_by_name = _fit._species_targets_from_baltic(
        raw_base, base_config, config_dir, DEFAULT_TEMP_NC
    )
    assert sp_index2 == sp_index
    habitat_masks = {name: _habitat_mask(raw_base, name, config_dir, ny, nx) for name in sp_index}

    gates: dict = {}

    # --- Gate D (frames/layers half, unconditional) ---
    gate_d_frames_layers(DEFAULT_TEMP_NC)
    gates["gate_d_frames_layers"] = "PASS"

    # --- Gate C: builder recomputation, computed ONCE and shared by both bioen arms ---
    builder_recompute = None
    if not no_recompute:
        thetao_files = sorted(DEFAULT_CACHE_DIR.glob("baltic_phy_monthly_reanalysis_thetao_*.nc"))
        bottomt_files = sorted(DEFAULT_CACHE_DIR.glob("baltic_phy_monthly_reanalysis_bottomT_*.nc"))
        so_files = sorted(DEFAULT_CACHE_DIR.glob("baltic_phy_monthly_reanalysis_so_*.nc"))
        if thetao_files and bottomt_files and so_files:
            ds_recomp = _temp_builder.build(
                thetao_files, bottomt_files, so_files[0], DEFAULT_GRID_PATH
            )
            builder_recompute = ds_recomp["temperature"].values.astype(np.float64)
        else:
            print(
                "Gate C: CMEMS cache not found under "
                f"{DEFAULT_CACHE_DIR} -- treating as --no-recompute."
            )

    engine_cfgs: dict[str, EngineConfig] = {}
    for arm in ("bioen", "bioen_plus2C"):
        cfg = all_arm_cfgs[arm]
        gate_c_load_through(cfg, DEFAULT_TEMP_NC, builder_recompute, wet)
        engine_cfgs[arm] = EngineConfig.from_dict(cfg)
        gate_d_structure(cfg, engine_cfgs[arm], overlay, expected_zlayer=expected_zlayer)
    gates["gate_c"] = "PASS"
    gates["gate_d_structure"] = "PASS"

    # --- Gate E: one step, one seed, the bioen arm ---
    gate_e_result = gate_e_zlayer(all_arm_cfgs["bioen"], seed=seeds[0], step=12)
    gates["gate_e"] = gate_e_result

    # --- Gate F: thermal instrument, the bioen arm's engine config + temperature field ---
    bioen_temp_data = _load_temperature_data(all_arm_cfgs["bioen"], None)
    gate_f_result = gate_f_thermal(engine_cfgs["bioen"], bioen_temp_data, habitat_masks)
    gates["gate_f"] = gate_f_result

    # --- Gate A: only meaningful at the fixture's own horizon, for seeds the fixture covers.
    # A subset of the committed 5 seeds is fine (e.g. a 1-seed smoke run at n_year=50) --
    # `check_against_fixture` is called per-seed below and only needs each requested seed to
    # be one of the fixture's own, not the full 5-seed set requested at once. ---
    fixture = _gate_a.load_gate_a_fixture()
    gate_a_applicable = n_year == fixture["n_year"] and set(seeds) <= set(fixture["seeds"])
    gates["gate_a"] = "PENDING" if gate_a_applicable else "not_applicable_years_or_seeds_mismatch"
    if not gate_a_applicable:
        print(
            f"Gate A: not applicable at n_year={n_year}, seeds={seeds} "
            f"(fixture is n_year={fixture['n_year']}, seeds={fixture['seeds']}) -- skipped, "
            "not silently attempted."
        )

    # --- engine runs (all arms x all seeds) ---
    report_species = [raw_base[f"species.name.sp{sp}"] for sp in range(n_sp)]
    results_by_arm_seed: dict[str, dict] = {a: {} for a in ARMS}
    raw_series: dict[str, dict[str, list[np.ndarray]]] = {a: {} for a in ARMS}
    all_species: list[str] | None = None
    for seed in seeds:
        for arm in ARMS:
            res = PythonEngine().run_in_memory(all_arm_cfgs[arm], seed=seed)
            results_by_arm_seed[arm][seed] = res
            bio_df = res.biomass()
            if all_species is None:
                all_species = [c for c in bio_df.columns if c not in ("Time", "species")]
                missing = set(report_species) - set(all_species)
                if missing:
                    raise AssertionError(
                        f"C3 harness BLOCKED (report species check): {missing} not found among "
                        f"biomass() columns {all_species}."
                    )
            for name in all_species:
                raw_series[arm].setdefault(name, []).append(bio_df[name].to_numpy(dtype=float))
    assert all_species is not None

    if gate_a_applicable:
        bad = {
            seed: _gate_a.check_against_fixture(
                fixture, seed, results_by_arm_seed["baseline"][seed].biomass()
            )
            for seed in seeds
        }
        ok = all(not v for v in bad.values())
        gates["gate_a"] = "PASS" if ok else {"FAIL": bad}
        if not ok:
            raise AssertionError(
                f"C3 harness BLOCKED (Gate A): baseline arm diverges from the committed master "
                f"fixture {fixture['engine_commit']} -- {bad}. This is a bioen-OFF regression, "
                "not an A/B result."
            )

    # --- REPORTED: final-decade means/spreads, all nine species, all three arms ---
    final_decade: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        final_decade[arm] = {}
        for name in report_species:
            per_seed = [
                float(_annualize(series, n_year)[-10:].mean()) for series in raw_series[arm][name]
            ]
            per_seed_min = [
                float(_annualize(series, n_year)[-10:].min()) for series in raw_series[arm][name]
            ]
            final_decade[arm][name] = {
                "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed)),
                "per_seed": per_seed,
                "final_decade_min": float(np.min(per_seed_min)),
            }
            if name in CERTIFIED_MEANS:
                final_decade[arm][name]["ratio_to_certified"] = (
                    final_decade[arm][name]["mean"] / CERTIFIED_MEANS[name]
                )
            if name in ENVELOPE:
                lo, hi = ENVELOPE[name]
                final_decade[arm][name]["ratio_to_envelope_mid"] = final_decade[arm][name][
                    "mean"
                ] / ((lo + hi) / 2.0)
                final_decade[arm][name]["in_envelope"] = lo <= final_decade[arm][name]["mean"] <= hi

    # --- realized ration / e-bar/g-hat (decision 7) ---
    ration: dict[str, dict] = {}
    for name in sp_index:
        sp = sp_index[name]
        seed_vals = [
            _final_window_mean(results_by_arm_seed["bioen"][seed], "meanEnetFaced", name)
            for seed in seeds
        ]
        e_bar = float(np.mean(seed_vals))
        g_hat = float(
            np.mean(
                _g_net(
                    t24_by_name[name],
                    engine_cfgs["bioen"].bioen_i_max[sp],
                    engine_cfgs["bioen"].bioen_c_m[sp],
                    engine_cfgs["bioen"].bioen_tp[sp],
                    BioenFixed(),
                )
            )
        )
        e_over_g = e_bar / g_hat if g_hat != 0 else float("nan")
        f_realized = 1.0 - (1.0 - e_over_g) * (1.0 - MAINTENANCE_SHARE)
        ration[name] = {
            "e_bar_meanEnetFaced": e_bar,
            "g_hat_fitted": g_hat,
            "e_over_g": e_over_g,
            "realized_ration_f": f_realized,
        }

    # --- realized annual ingestion (decision 17) ---
    ingestion: dict[str, dict] = {}
    for name in sp_index:
        seed_means = {}
        for arm in ("baseline", "bioen"):
            vals = []
            for seed in seeds:
                res = results_by_arm_seed[arm][seed]
                try:
                    vals.append(_final_window_mean(res, "ingestion", name))
                except FileNotFoundError:
                    vals = None
                    break
            seed_means[arm] = float(np.mean(vals)) if vals else None
        ingestion[name] = seed_means

    # --- length-at-age, baseline vs bioen, RMS % over ages >= 1 yr ---
    length_at_age_result: dict[str, dict] = {}
    for name in sp_index:
        rms_per_seed = []
        for seed in seeds:
            la_base = length_from_age_bins(
                results_by_arm_seed["baseline"][seed].abundance_by_age(name),
                results_by_arm_seed["baseline"][seed].biomass_by_age(name),
                float(base_config.condition_factor[sp_index[name]]),
                float(base_config.allometric_power[sp_index[name]]),
                name,
            )
            la_bioen = length_from_age_bins(
                results_by_arm_seed["bioen"][seed].abundance_by_age(name),
                results_by_arm_seed["bioen"][seed].biomass_by_age(name),
                float(base_config.condition_factor[sp_index[name]]),
                float(base_config.allometric_power[sp_index[name]]),
                name,
            )
            shared_bins = sorted(set(la_base) & set(la_bioen) - {0})
            if not shared_bins:
                continue
            pct_diff = [
                (la_bioen[k] - la_base[k]) / la_base[k] * 100.0
                for k in shared_bins
                if la_base[k] != 0
            ]
            if pct_diff:
                rms_per_seed.append(float(np.sqrt(np.mean(np.square(pct_diff)))))
        length_at_age_result[name] = {
            "rms_pct_ages_ge_1": float(np.mean(rms_per_seed)) if rms_per_seed else None,
            "n_seeds": len(rms_per_seed),
        }

    # --- seeding diagnostics (SSB not available in-memory -- skip with a note, brief's
    # documented fallback, confirmed 2026-09-05: OsmoseResults.ssb() raises FileNotFoundError
    # in-memory mode) ---
    seeding_note = (
        "results.ssb() raises FileNotFoundError in in-memory mode (no SSB output family "
        "built by _build_dataframes_from_outputs) -- seeding/first-spawning-step diagnostic "
        "skipped per the brief's documented fallback."
    )

    # --- bioen_plus2C minus bioen deltas ---
    plus2_deltas: dict[str, dict] = {}
    for name in report_species:
        plus2_deltas[name] = {
            "delta_mean": final_decade["bioen_plus2C"][name]["mean"]
            - final_decade["bioen"][name]["mean"],
            "g_net_shift": gate_f_result.get(name, {}).get("g_plus2", float("nan"))
            - gate_f_result.get(name, {}).get("g_base", float("nan")),
        }

    # --- decision rule (spec Sec.4) ---
    decision = evaluate_decision_rule(final_decade, ration)
    failed = decision["failed"]
    verdict = decision["verdict"]

    report = {
        "seeds": list(seeds),
        "arms": list(ARMS),
        "n_year": n_year,
        "certifying": sorted(seeds) == sorted(SEEDS) and n_year == N_YEAR,
        "assessed_stocks": list(ASSESSED_STOCKS),
        "gates": gates,
        "final_decade_means": final_decade,
        "realized_ration": ration,
        "maintenance_share_m": MAINTENANCE_SHARE,
        "realized_ingestion": ingestion,
        "length_at_age": length_at_age_result,
        "seeding_diagnostics_note": seeding_note,
        "bioen_plus2C_minus_bioen": plus2_deltas,
        "labels": REPORT_LABELS,
        "decision_rule_failed": failed,
        "decision_rule_undetermined": decision["undetermined"],
        "decision_rule_criteria": decision["criteria"],
        "verdict": verdict,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=float)
    return report


def main(argv: list[str] | None = None) -> int:
    global REPORT_PATH

    ap = argparse.ArgumentParser(description="C3 bioenergetics A/B harness.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--years", type=int, default=N_YEAR)
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    certifying = sorted(args.seeds) == sorted(SEEDS) and args.years == N_YEAR
    if args.out is None:
        if not certifying:
            ap.error(
                "--out is required for a non-certifying run (seeds/years differ from the "
                f"certifying configuration SEEDS={SEEDS}, N_YEAR={N_YEAR}) -- pass --out "
                f"explicitly to avoid silently overwriting the committed deliverable at "
                f"{REPORT_PATH}"
            )
        args.out = REPORT_PATH

    REPORT_PATH = args.out

    out = run_c3(seeds=tuple(args.seeds), n_year=args.years, no_recompute=args.no_recompute)
    print(f"report written to {REPORT_PATH}")
    print(f"verdict: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
