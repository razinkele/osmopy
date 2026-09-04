#!/usr/bin/env python
"""Offline Java-form bioenergetics parameter-fit CLI (C3 spec §3.4, Task 8/11).

``--gate-b [OUT_DIR]``: fit ``osmose.calibration.bioen_offline`` against the 8 Bay-of-Biscay
demo species in ``data/examples/`` and write a runnable, bioen-on overlay at ``OUT_DIR``
(default ``data/examples_bioen/``) -- the Gate-B config Task 9's cross-engine parity run reads.
This is a SYNTHETIC fit for a demo config, not a validated species calibration: every species
gets the same ``t_opt = 15.0`` / isothermal ``t24`` (matching the config's own
``temperature.value``), because ``data/examples`` carries no cited growth-optimum literature to
fit against -- that is the whole point of Gate B (bioen-on, runnable, non-degenerate), not a
biological claim about Bay-of-Biscay species.

``--baltic``: fits the production Baltic 9-species set (Task 11) from cited growth optima
(spec §1) and the production config's own vBGF curves, and writes a FLAT overlay -- CSV +
``c3_bioen_arm.json`` + README -- under ``data/baltic/scenarios/c3_bioen/``. This is an ARM
for Task 12's A/B harness, not a production change: nothing under ``data/baltic/`` other than
the new ``scenarios/c3_bioen/`` directory is written or modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from osmose.calibration.bioen_offline import (
    BioenFixed,
    FitResult,
    SpeciesTargets,
    bioen_param_lines,
    c_m_from_share,
    fit_species,
    g_net,
)
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.background import BackgroundSpeciesInfo, parse_background_species
from osmose.engine.config import EngineConfig
from osmose.engine.movement_maps import _load_csv_grid
from osmose.engine.path_resolution import resolve_data_path
from osmose.engine.physical_data import PhysicalData
from osmose.engine.processes.temp_function import phi_t

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

# RMS-length pin (brief step 5): not a hard gate for Gate B -- it needs a runnable,
# non-degenerate bioen-on config, not a perfect fit, so a species over this prints a warning
# and is still written. Under --baltic (production overlay) it is a HARD pin: see
# _assert_baltic_pins.
RMS_PIN_PCT = 15.0


def _species_targets_from_examples(
    raw: dict[str, str], n_sp: int
) -> tuple[list[SpeciesTargets], dict[str, int], dict[str, int], dict[str, float]]:
    """Build one ``SpeciesTargets`` per focal species from the read master config.

    ``t_opt``/``t24``: isothermal 15 C for every species (see module docstring) -- matches the
    Gate-B config's own ``temperature.value;15.0``. ``m1``: not present in ``data/examples`` for
    any species; Java default 0.0 (flat maturity threshold, no age dependence).
    """
    targets: list[SpeciesTargets] = []
    sp_index: dict[str, int] = {}
    zlayer: dict[str, int] = {}
    m0: dict[str, float] = {}
    for i in range(n_sp):
        name = raw[f"species.name.sp{i}"]
        linf = float(raw[f"species.linf.sp{i}"])
        k = float(raw[f"species.k.sp{i}"])
        t0 = float(raw[f"species.t0.sp{i}"])
        lifespan = float(raw[f"species.lifespan.sp{i}"])
        cf = float(raw[f"species.length2weight.condition.factor.sp{i}"])
        b = float(raw[f"species.length2weight.allometric.power.sp{i}"])
        egg_size = float(raw.get(f"species.egg.size.sp{i}", "0.0"))
        egg_weight_key = f"species.egg.weight.sp{i}"
        egg_weight_g = float(raw[egg_weight_key]) if egg_weight_key in raw else cf * egg_size**b
        m0_val = float(raw[f"species.maturity.size.sp{i}"])

        targets.append(
            SpeciesTargets(
                name=name,
                linf=linf,
                k=k,
                t0=t0,
                cf=cf,
                b=b,
                egg_weight_g=egg_weight_g,
                m0=m0_val,
                m1=0.0,
                lifespan_years=lifespan,
                t_opt=15.0,
                t24=np.full(24, 15.0),
            )
        )
        sp_index[name] = i
        zlayer[name] = 0
        m0[name] = m0_val
    return targets, sp_index, zlayer, m0


def _print_fit_table(results: list[FitResult]) -> None:
    header = (
        f"{'species':<14}{'imax':>8}{'r':>8}{'c_m':>14}{'t_p':>8}"
        f"{'rms%':>8}{'w_inf_fit':>12}{'w_inf_vb':>12}{'larv0.5y':>10}{'n_pts':>7}  pin"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        flag = "OK" if res.rms_len_pct <= RMS_PIN_PCT else f"FAIL (>{RMS_PIN_PCT:g}%)"
        print(
            f"{res.name:<14}{res.imax:>8.3f}{res.r:>8.3f}{res.c_m:>14.4g}{res.t_p:>8.3f}"
            f"{res.rms_len_pct:>8.2f}{res.w_inf_fit_g:>12.1f}{res.w_inf_vb_g:>12.1f}"
            f"{res.larval_ratio_half_year:>10.3f}{res.n_points:>7d}  {flag}"
        )


def run_gate_b(out_dir: Path) -> list[FitResult]:
    examples_dir = ROOT / "data" / "examples"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(examples_dir, out_dir)

    master = out_dir / "osm_all-parameters.csv"
    reader = OsmoseConfigReader()
    raw = reader.read(master)
    n_sp = int(raw["simulation.nspecies"])

    fx = BioenFixed()
    targets, sp_index, zlayer, m0 = _species_targets_from_examples(raw, n_sp)
    notes = {
        tg.name: (
            "T_opt=15.0C isothermal placeholder (Gate-B synthetic fit -- data/examples carries "
            "no cited thermal optimum for this species; matches the config's own "
            "temperature.value)"
        )
        for tg in targets
    }

    results = [fit_species(tg, fx) for tg in targets]

    bioen_lines = bioen_param_lines(
        results, fx, zlayer=zlayer, sp_index=sp_index, background_imax={}, notes=notes, m0=m0
    )
    bioen_path = out_dir / "osm_param-bioen.csv"
    bioen_path.write_text("\n".join(bioen_lines) + "\n")

    append_lines = [
        "osmose.configuration.bioen;osm_param-bioen.csv",
        "module.bioenergetics.enabled;true",
        "simulation.bioen.phit.enabled;true",
        "simulation.bioen.fo2.enabled;false",
        "temperature.value;15.0",
        "oxygen.value;300.0",
    ]
    with master.open("a") as fh:
        fh.write("\n" + "\n".join(append_lines) + "\n")

    _print_fit_table(results)
    n_fail = sum(1 for res in results if res.rms_len_pct > RMS_PIN_PCT)
    print(
        f"\n{len(results)} species fitted, {len(results) - n_fail} within the {RMS_PIN_PCT:g}% RMS pin."
    )
    print(
        f"Wrote {bioen_path.relative_to(ROOT)} and appended bioen keys to {master.relative_to(ROOT)}."
    )
    return results


# --------------------------------------------------------------------------------------------
# --baltic: production Baltic 9-species bioen parameter set (Task 11)
# --------------------------------------------------------------------------------------------

# Ruling R1 (progress log, 2026-08-30): background predators take their bioen Imax with NO
# n_dt_per_year division (Java's early return in BioenPredationMortality skips it) -- this is
# also the beta that per_fish_ingestion_cap hardcodes for every background predator
# (osmose/engine/processes/bioen_predation.py). background_imax()'s exponent and the authored
# species.beta.sp{background} value must both equal this constant, or the "bioen cap equals
# the standard cap at w_mean" property this Imax is solved for breaks.
BACKGROUND_BETA = 0.8

# Cited growth (or physiological) optimum per species, °C (C3 spec §1, verified via scite
# 2026-08-30). T_p (the engine parameter) is SOLVED per species so that the net-growth
# optimum equals this value (spec decision 16) -- see SPECIES_NOTE for the per-species label
# and citation.
SPECIES_T_OPT = {
    "cod_west": 10.0,
    "cod_east": 10.0,
    "herring": 15.0,
    "sprat": 18.0,
    "flounder": 19.0,
    "perch": 25.0,
    "pikeperch": 27.0,
    "smelt": 15.0,
    "stickleback": 21.7,
}

# species.zlayer.sp{i}: 0 = surface, 1 = bottom (C3 spec decision 4, user judgment call).
# cod (both stocks) and flounder are demersal -> bottom; the rest -> surface. Perch and
# pikeperch are coastal/lagoon species living in the warm shallow layer; smelt is pelagic.
SPECIES_ZLAYER = {
    "cod_west": 1,
    "cod_east": 1,
    "flounder": 1,
    "herring": 0,
    "sprat": 0,
    "stickleback": 0,
    "smelt": 0,
    "perch": 0,
    "pikeperch": 0,
}

# Per-species label from C3 spec §1, verbatim where the spec uses one of the four canonical
# words (PROVISIONAL / SECONDARY / CONSUMPTION PROXY / SIZE COMPROMISE), plus the citation.
SPECIES_NOTE = {
    "cod_west": (
        "growth optimum 10 C, a SIZE COMPROMISE across the 100-1000 g range that carries most "
        "Baltic cod biomass (Bjornsson & Steinarsson 2002, doi:10.1139/f02-028: optimum falls "
        "from 14.3 C at 50 g to 5.9 C at 5000 g)"
    ),
    "cod_east": (
        "growth optimum 10 C, a SIZE COMPROMISE across the 100-1000 g range that carries most "
        "Baltic cod biomass (Bjornsson & Steinarsson 2002, doi:10.1139/f02-028: optimum falls "
        "from 14.3 C at 50 g to 5.9 C at 5000 g)"
    ),
    "herring": (
        "growth optimum 15 C is PROVISIONAL -- no herring growth optimum was retrieved in "
        "three literature searches; the maintenance-share trials (Bernreuther et al. 2012, "
        "doi:10.1111/jai.12045) ran at 16 C, which is not a growth optimum"
    ),
    "sprat": (
        "growth optimum 18 C is a CONSUMPTION PROXY, not a growth optimum -- gastric "
        "evacuation rate peaks there (Bernreuther et al. 2009, "
        "doi:10.1111/j.1095-8649.2009.02353.x)"
    ),
    "flounder": (
        "growth optimum 19 C is a SECONDARY quotation -- Kusakabe et al. 2016 "
        "(doi:10.1007/s12562-016-1053-1) quoting Fonds et al. 1992"
    ),
    "perch": (
        "growth optimum 25 C (Hokanson 1977, doi:10.1139/f77-217); the 40x50 open-coast "
        "surface field runs ~5 C below the lagoon habitat this species actually occupies -- "
        "phiT peaks at 0.7-0.8 there and the fit inflates Imax accordingly"
    ),
    "pikeperch": (
        "growth optimum 27 C (Hokanson 1977, doi:10.1139/f77-217); the 40x50 open-coast "
        "surface field runs ~5 C below the lagoon habitat this species actually occupies -- "
        "phiT peaks at 0.7-0.8 there and the fit inflates Imax accordingly"
    ),
    "smelt": (
        "growth optimum 15 C is SECONDARY and a preference, not a growth optimum -- via "
        "Krause 2008 quoting Vinni et al. 2004, doi:10.1111/j.0022-1112.2004.00323.x"
    ),
    "stickleback": (
        "growth optimum 21.7 C (Lefebure et al. 2011, doi:10.1111/j.1095-8649.2011.03121.x)"
    ),
}


def habitat_t24(
    temp_nc: Path, layer: int, map_files: list[Path], ny: int, nx: int, ndt: int = 24
) -> NDArray[np.float64]:
    """Habitat-mean temperature per engine step, for one species/layer.

    Loads the field the way the engine does (``PhysicalData.from_netcdf``) and the habitat
    footprint the way movement does (``_load_csv_grid`` -- the CSVs are stored upside-down,
    the C4 trap; this reuses the engine's own flip rather than re-deriving it). A cell counts
    as habitat if its value is > 0 in ANY of the species' map files (juvenile + adult +
    spawning stages unioned, since the species occupies all of them across its life).

    Raises if the loaded file's frame count isn't ``ndt``: ``PhysicalData.get_grid`` indexes
    ``step % <loaded frame count>``, not any declared nsteps-year metadata, so a mismatch
    would silently misalign the month-to-step mapping partway through the year instead of
    erroring (CLAUDE.md frame-count gotcha). Also raises if a step has zero
    finite-temperature habitat cells -- every habitat cell land-masked there means the
    layer/map pairing is wrong for that species, not a legitimate data gap.
    """
    from osmose.engine._netcdf import open_dataset_safe

    ds = open_dataset_safe(Path(temp_nc))
    n_frames = ds["temperature"].shape[0]
    if n_frames != ndt:
        raise ValueError(
            f"{temp_nc}: temperature has {n_frames} frame(s), expected {ndt} -- "
            "PhysicalData.get_grid indexes step % <frame count>, not any declared "
            "nsteps.year metadata, so a mismatch would silently misalign the month-to-step "
            "mapping instead of erroring"
        )

    habitat = np.zeros((ny, nx), dtype=bool)
    for mf in map_files:
        grid = _load_csv_grid(Path(mf), ny, nx)
        habitat |= np.nan_to_num(grid, nan=0.0) > 0
    n_habitat = int(habitat.sum())
    if n_habitat == 0:
        raise ValueError(
            f"no habitat cells (value > 0) found across {len(map_files)} map file(s): {map_files}"
        )

    pd_ = PhysicalData.from_netcdf(temp_nc, varname="temperature", nsteps_year=ndt)
    frames = np.stack([pd_.get_grid(s, layer) for s in range(ndt)])  # (ndt, ny, nx)
    finite_every_step = np.all(np.isfinite(frames), axis=0)
    n_land = int((habitat & ~finite_every_step).sum())
    if n_land:
        logger.warning(
            "habitat_t24(layer=%d): %d/%d movement-map habitat cell(s) are land-masked "
            "(NaN) in the temperature field for at least one step -- dropped from the "
            "habitat mean via nanmean",
            layer,
            n_land,
            n_habitat,
        )

    with warnings.catch_warnings():
        # A step with zero finite habitat cells makes np.nanmean warn ("Mean of empty
        # slice") and return NaN by design -- that NaN is exactly what the check below
        # turns into a clear raise, so the warning is expected noise, not a bug to fix here.
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        t24 = np.array([np.nanmean(frames[s][habitat]) for s in range(ndt)])
    if np.isnan(t24).any():
        bad = np.where(np.isnan(t24))[0].tolist()
        raise ValueError(
            f"habitat_t24(layer={layer}): no finite temperature within the habitat footprint "
            f"at step(s) {bad} -- every habitat cell is land-masked there"
        )
    return t24


def background_imax(config: EngineConfig, b: BackgroundSpeciesInfo, beta: float = 0.8) -> float:
    """Per-time-step bioen Imax for one background predator (ruling R1).

    Java's bioen predation cap (``BioenPredationMortality``) early-returns for background
    predators BEFORE the ``/n_dt_per_year`` division the focal branch and the standard
    (non-bioen) path both apply (see ``osmose/engine/processes/bioen_predation.py``'s module
    docstring). ``config.bioen_i_max_all``'s background half is filled straight from
    ``BackgroundSpeciesInfo.ingestion_rate`` -- the SAME per-year config value the standard
    (bioen-off) cap divides by ``n_dt_per_year`` at its use site. So a background Imax
    authored on the focal per-year convention would overshoot per-step consumption by
    ``n_dt_per_year`` (24x) the moment bioen turns on. Pre-dividing here is what keeps the
    two paths physically equivalent.

    ``w_mean``: the abundance-unweighted mean per-fish weight (t), ``cf * L**b * 1e-6``,
    over the species' size classes. The returned Imax makes the bioen per-fish, per-substep
    cap (``imax * (w*1e6)**beta * 1e-6 / n_subdt``) equal the standard per-fish, per-substep
    cap (``w * ingestion_rate / (n_dt_per_year * n_subdt)``) EXACTLY at ``w = w_mean``; away
    from ``w_mean`` the two diverge by ``(w_class / w_mean) ** (beta - 1)`` since the
    standard cap is linear in weight and the bioen cap goes as ``weight**beta`` (reported by
    the caller, not computed here).
    """
    w_mean = float(
        np.mean([b.condition_factor * (length**b.allometric_power) * 1e-6 for length in b.lengths])
    )
    return (b.ingestion_rate / config.n_dt_per_year) * (w_mean * 1e6) ** (1.0 - beta)


def build_overlay(csv_path: Path, temp_nc: Path) -> dict[str, str]:
    """Flat overlay dict: every key in the bioen CSV, plus the bioen/temperature switches.

    Deliberately excludes ``osmose.configuration.bioen`` -- spec review C6: that key would
    point Java's include mechanism at the CSV, but a dict overlay merged by
    ``run_in_memory`` never resolves an ``osmose.configuration.*`` include, so the CSV's own
    keys are flattened directly into the overlay instead (which this does). Also excludes
    ``temperature.value``: the engine's loader precedence (spec decision 6) tries ``.value``
    before the file, so a stray scalar here would silently shadow the forcing file.
    """
    overlay = dict(OsmoseConfigReader().read_file(Path(csv_path)))
    overlay.update(
        {
            "module.bioenergetics.enabled": "true",
            "simulation.bioen.phit.enabled": "true",
            "simulation.bioen.fo2.enabled": "false",
            "temperature.filename": str(Path(temp_nc).resolve()),
            "temperature.varname": "temperature",
            "temperature.nsteps.year": "24",
        }
    )
    return overlay


def _species_map_files(raw: dict[str, str], name: str, config_dir: Path) -> list[Path]:
    """Every movement map file assigned to species ``name`` (union across all life stages)."""
    files: list[Path] = []
    for key, val in raw.items():
        m = re.fullmatch(r"movement\.species\.map(\d+)", key)
        if not m or val.strip() != name:
            continue
        file_val = raw.get(f"movement.file.map{m.group(1)}", "").strip()
        if not file_val or file_val.lower() in ("null", "none"):
            continue  # authored null map (intentionally moves schools out) -- not habitat
        resolved = resolve_data_path(file_val, config_dir=str(config_dir))
        if resolved is None:
            raise FileNotFoundError(
                f"movement.file.map{m.group(1)} for {name!r} could not be resolved: {file_val!r}"
            )
        files.append(resolved)
    if not files:
        raise ValueError(f"no movement map files found for species {name!r}")
    return sorted(set(files))


def _species_targets_from_baltic(
    raw: dict[str, str], config: EngineConfig, config_dir: Path, temp_nc: Path
) -> tuple[list[SpeciesTargets], dict[str, int], dict[str, NDArray[np.float64]]]:
    """Build one ``SpeciesTargets`` per Baltic focal species from the production config."""
    names = {raw[f"species.name.sp{i}"] for i in range(config.n_species)}
    missing = set(SPECIES_T_OPT) - names
    extra = names - set(SPECIES_T_OPT)
    if missing or extra:
        raise ValueError(
            f"Baltic focal species set changed: missing {sorted(missing)}, unexpected "
            f"{sorted(extra)} -- SPECIES_T_OPT/SPECIES_ZLAYER/SPECIES_NOTE need updating "
            "(see the stale-fixture-sweep skill)"
        )

    ny = int(raw["grid.nlat"])
    nx = int(raw["grid.nlon"])

    targets: list[SpeciesTargets] = []
    sp_index: dict[str, int] = {}
    t24_by_name: dict[str, NDArray[np.float64]] = {}
    for i in range(config.n_species):
        name = raw[f"species.name.sp{i}"]
        linf = float(raw[f"species.linf.sp{i}"])
        k = float(raw[f"species.k.sp{i}"])
        t0 = float(raw[f"species.t0.sp{i}"])
        cf = float(raw[f"species.length2weight.condition.factor.sp{i}"])
        b = float(raw[f"species.length2weight.allometric.power.sp{i}"])
        lifespan = float(raw[f"species.lifespan.sp{i}"])
        m0_val = float(config.maturity_size[i])

        override = config.egg_weight_override
        if override is not None and not np.isnan(override[i]):
            egg_weight_g = float(override[i]) * 1e6  # config.egg_weight_override is in TONNES
        else:
            egg_size = float(raw.get(f"species.egg.size.sp{i}", "0.0"))
            egg_weight_g = cf * egg_size**b

        maps = _species_map_files(raw, name, config_dir)
        t24 = habitat_t24(temp_nc, SPECIES_ZLAYER[name], maps, ny=ny, nx=nx)
        t24_by_name[name] = t24

        targets.append(
            SpeciesTargets(
                name=name,
                linf=linf,
                k=k,
                t0=t0,
                cf=cf,
                b=b,
                egg_weight_g=egg_weight_g,
                m0=m0_val,
                m1=0.0,
                lifespan_years=lifespan,
                t_opt=SPECIES_T_OPT[name],
                t24=t24,
            )
        )
        sp_index[name] = i
    return targets, sp_index, t24_by_name


def _assert_baltic_pins(res: FitResult, fx: BioenFixed, rms_pin: float = RMS_PIN_PCT) -> None:
    """Hard sanity pins for the production overlay (brief step 3): raise, never warn."""
    T = np.linspace(res.t_opt - 8, res.t_opt + 8, 1601)
    cm = c_m_from_share(1.0, res.t_p, fx)
    g = g_net(T, 1.0, cm, res.t_p, fx)
    argmax_t = float(T[np.argmax(g)])
    phi_at_tp = float(phi_t(np.array(res.t_p), fx.e_m, fx.e_d, res.t_p))
    if phi_at_tp != 1.0:
        raise AssertionError(f"{res.name}: phi_t(t_p) = {phi_at_tp} != 1.0")
    if abs(argmax_t - res.t_opt) > 0.1:
        raise AssertionError(
            f"{res.name}: argmax g_net = {argmax_t:.3f} C, cited optimum {res.t_opt} C "
            "(tolerance 0.1 C)"
        )
    if not res.imax > 0:
        raise AssertionError(f"{res.name}: Imax = {res.imax} <= 0")
    if not res.r > 0:
        raise AssertionError(f"{res.name}: r = {res.r} <= 0")
    if res.rms_len_pct > rms_pin:
        raise AssertionError(
            f"{res.name}: RMS length error {res.rms_len_pct:.2f}% > {rms_pin:g}% pin (ages >= 1 yr)"
        )


def _print_baltic_table(
    results: list[FitResult], fx: BioenFixed, t24_by_name: dict[str, NDArray[np.float64]]
) -> None:
    header = (
        f"{'species':<12}{'t_opt':>7}{'t_p':>9}{'imax':>8}{'r':>8}{'c_m':>13}"
        f"{'rms%':>7}{'w_inf_fit':>11}{'w_inf_vb':>11}{'larv0.5y':>9}"
        f"{'phiT(Tbar)':>11}{'inflate':>9}  pin"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        tbar = float(t24_by_name[res.name].mean())
        phi_tbar = float(phi_t(np.array(tbar), fx.e_m, fx.e_d, res.t_p))
        inflation = 1.0 / (phi_tbar * (1.0 - fx.m_share))
        print(
            f"{res.name:<12}{res.t_opt:>7.2f}{res.t_p:>9.4f}{res.imax:>8.3f}{res.r:>8.3f}"
            f"{res.c_m:>13.4g}{res.rms_len_pct:>7.2f}{res.w_inf_fit_g:>11.1f}"
            f"{res.w_inf_vb_g:>11.1f}{res.larval_ratio_half_year:>9.3f}"
            f"{phi_tbar:>11.4f}{inflation:>9.3f}  OK"
        )


def _background_ratio_bounds(b_sp: BackgroundSpeciesInfo, beta: float) -> tuple[float, float]:
    """Min/max of the bioen-vs-standard cap ratio across a background species' size classes.

    The two caps agree exactly at w_mean (see background_imax); this reports how far they
    diverge at the smallest and largest size class, ``(w_class/w_mean)**(beta-1)``.
    """
    w_classes = np.array(
        [b_sp.condition_factor * (length**b_sp.allometric_power) for length in b_sp.lengths]
    )
    w_mean = float(w_classes.mean())
    ratios = (w_classes / w_mean) ** (beta - 1.0)
    return float(ratios.min()), float(ratios.max())


def _print_background_table(
    background_list: list[BackgroundSpeciesInfo],
    bg_imax: dict[int, float],
    n_dt: int,
    beta: float,
) -> None:
    print("\nbackground predators (Imax already per-time-step, ruling R1):")
    for b_sp in background_list:
        imax = bg_imax[b_sp.file_index]
        lo, hi = _background_ratio_bounds(b_sp, beta)
        print(
            f"  sp{b_sp.file_index} {b_sp.name}: config ingestion_rate={b_sp.ingestion_rate:g}/yr "
            f"-> bioen Imax={imax:.6g} (= rate/{n_dt} at w_mean); per-class cap ratio to "
            f"standard [{lo:.3f}, {hi:.3f}] across {len(b_sp.lengths)} size classes "
            "(1.0 only exactly at w_mean)"
        )


# The generator's own direct-authoring source: what actually decides the content this
# function's caller writes to c3_bioen_arm.json. Kept in sync with the imports at the top of
# this file. Deliberately NOT the full import closure (e.g. not
# osmose/engine/movement_maps.py, which habitat_t24 uses via _load_csv_grid and which does
# feed the fit): widening this to "everything imported" would flag +dirty any time the user
# has unrelated in-progress edits anywhere upstream in the repo (routinely true in this
# project's own working tree), defeating the point of a clean-vs-dirty signal. So a bare SHA
# here means "these two files match HEAD," not "a fresh checkout of HEAD reproduces this file
# bit-for-bit" -- narrower than full reproducibility, but still closes the specific failure
# R42 named (the generator itself being uncommitted).
_GENERATOR_SOURCES = ("scripts/fit_baltic_bioen_params.py", "osmose/calibration/bioen_offline.py")


def _git_commit_sha() -> str:
    """HEAD's short SHA, for ``c3_bioen_arm.json``'s ``_meta.commit`` provenance field.

    Appends ``+dirty`` whenever ``_GENERATOR_SOURCES`` has uncommitted changes relative to
    HEAD at generation time -- staged, unstaged, OR untracked (``git status --porcelain``,
    not ``git diff``: a diff against HEAD is silent about a brand-new untracked file, which
    would reintroduce exactly the failure this guards against). Otherwise this field can
    silently record a commit that predates the code that actually produced the artifact
    (review R42, 2026-09-05: an earlier run recorded ``cf028a1``, a commit at which
    ``--baltic`` did not exist, because the generator itself was uncommitted when it ran).
    Committing the generator before regenerating the artifact is what keeps this field bare
    and reproducible; running it against a dirty generator is still allowed, but the field
    then says so instead of implying a clean checkout can reproduce the file.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        sha = result.stdout.strip()
    except Exception:
        return "unknown"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *_GENERATOR_SOURCES],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        clean = status.stdout.strip() == ""
    except Exception:
        clean = False  # can't verify -- be conservative and flag it, not silently trust HEAD
    return sha if clean else f"{sha}+dirty"


def _write_readme(
    out_dir: Path,
    results: list[FitResult],
    background_list: list[BackgroundSpeciesInfo],
    bg_imax: dict[int, float],
    t24_by_name: dict[str, NDArray[np.float64]],
) -> None:
    fx = BioenFixed()
    lines = [
        "# C3 bioenergetics parameter set -- Baltic (`--baltic`)",
        "",
        "**This is an ARM, not production.** `baltic_param-bioen.csv` and "
        "`c3_bioen_arm.json` are a flat overlay merged in memory by Task 12's A/B harness "
        "(`{**base_cfg, **overlay}`); they are never appended to "
        "`data/baltic/baltic_all-parameters.csv` or any other production config file, and "
        "`data/baltic/` is otherwise untouched (C3 spec decision 1). Applying the overlay "
        "turns on `module.bioenergetics.enabled` and the two-layer temperature forcing "
        "(`data/baltic/forcing/baltic_temperature_2layer_climatology.nc`, Task 10) for a "
        "bioen A/B run.",
        "",
        "Generated by `scripts/fit_baltic_bioen_params.py --baltic` (C3 spec "
        "`docs/superpowers/specs/2026-08-30-baltic-c3-bioen-stage1-design.md` Sec.1/3.4).",
        "",
        "## Two temperatures per species",
        "",
        "`t_opt` is the CITED growth (or physiological) optimum (spec Sec.1). `T_p` is the "
        "engine parameter SOLVED so that the net-growth optimum `argmax_T g_net(T)` equals "
        "`t_opt` -- T_p sits above t_opt because maintenance (a bare Arrhenius, no peak) "
        "pulls the net-growth optimum below the mobilized-energy peak that T_p marks. "
        "T-bar is the species' own habitat-mean temperature (its own layer and movement-map "
        "footprint, `habitat_t24().mean()`).",
        "",
        "| species | t_opt (C) | T_p (C) | T-bar (C) | phiT(T-bar) | inflation "
        "1/(phiT*(1-m)) | Imax | r | RMS len % | label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for res in results:
        tbar = float(t24_by_name[res.name].mean())
        phi_tbar = float(phi_t(np.array(tbar), fx.e_m, fx.e_d, res.t_p))
        inflation = 1.0 / (phi_tbar * (1.0 - fx.m_share))
        note = SPECIES_NOTE.get(res.name, "")
        lines.append(
            f"| {res.name} | {res.t_opt:g} | {res.t_p:.3f} | {tbar:.2f} | {phi_tbar:.3f} | "
            f"{inflation:.2f} | {res.imax:.3f} | {res.r:.3f} | {res.rms_len_pct:.2f} | {note} |"
        )
    lines += [
        "",
        "## What the RMS pin does and does not check",
        "",
        "The RMS <= 15% pin (`RMS len %` column above) measures how well the fitted `(Imax, r)` "
        "reproduce the config's own vBGF weight-at-age curve, and it is sensitive to `K` "
        "(`species.k.spN`, the config's own growth-rate parameter) -- a sustained error there "
        "walks the residual toward the pin (a -50% K perturbation moved cod_west's RMS from "
        "8.3% to 13.6% in a review sensitivity sweep, 2026-09-05). It is flat-to-inverted "
        "against the other two literature/config inputs this overlay ships: an 8 C error in "
        "`t_opt` moved that same RMS by <0.3 points, and a 30% error in `Linf` "
        "(`species.linf.spN`) *improved* it -- `Imax`/`r` are free parameters chosen by the "
        "fit specifically to minimize this residual, so they absorb almost any `t_opt`/`Linf` "
        "rescaling of the target curve. `t_opt` comes from cited literature (`SPECIES_T_OPT`, "
        "spec Sec.1 -- see each species' label above); `Linf` and `K` come from the production "
        "Baltic config's own already-calibrated vBGF curve, not from this fit. **Passing the "
        "RMS pin corroborates `K` and the growth-curve shape it implies; it is not evidence "
        "that `t_opt` or `Linf` are scientifically correct** -- the separate "
        "`phi_t(t_p)==1.0` / argmax-within-0.1 C pins only check that `T_p` was solved "
        "correctly FROM `t_opt`, not that the cited `t_opt` itself is right.",
        "",
        "## Background predators (GreySeal sp15, Cormorant sp16)",
        "",
        "Ruling R1 (progress log, 2026-08-30): Java's bioen predation cap skips the "
        "`/n_dt_per_year` division for background predators that both the standard "
        "(non-bioen) path and the FOCAL half of the bioen cap apply. `background_imax()` "
        "pre-divides by `n_dt_per_year` so the bioen cap equals the standard cap *exactly at "
        "the class-mean weight*; per size class the two diverge by "
        "`(w_class/w_mean)**(beta-1)` since the standard cap is linear in weight and the "
        "bioen cap goes as `weight**beta` (bounds in the table below).",
        "",
        "Only `species.beta` and the Imax family "
        "(`predation.ingestion.rate.max`, `...larval.ingestion.rate.increase.ratio`, "
        "`predation.c.bioen`) are authored for background species. "
        "`species.bioen.{assimilation,maint.*,mobilized.*}`, `species.maturity.*` and "
        "`species.bioen.forage.*` are deliberately NOT authored: those arrays are "
        "focal-length (`n_species`, not `n_species+n_background`) in "
        "`osmose/engine/config.py`, so the Python engine cannot read them at a background "
        "index -- inventing a T_p or a maturity curve for a grey seal would be a fabricated "
        "number with no code path behind it. Consequence: **this overlay is not "
        "Java-loadable as authored.** Java's Gate-B key inventory (spec Sec.4) wants the "
        "full bioen block for every predator index, focal and background; a future Java "
        "cross-check of this Baltic overlay would need those additional background keys "
        "added first.",
        "",
        "| sp | name | config ingestion_rate (/yr) | bioen Imax (per-timestep) | beta | "
        "per-class cap ratio to standard [min, max] |",
        "|---|---|---:|---:|---:|---|",
    ]
    for b_sp in background_list:
        imax = bg_imax[b_sp.file_index]
        lo, hi = _background_ratio_bounds(b_sp, BACKGROUND_BETA)
        lines.append(
            f"| {b_sp.file_index} | {b_sp.name} | {b_sp.ingestion_rate:g} | {imax:.6g} | "
            f"{BACKGROUND_BETA:g} | [{lo:.3f}, {hi:.3f}] |"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def run_baltic(out_dir: Path | None = None) -> list[FitResult]:
    """Fit the production Baltic 9-species bioen parameter set (C3 spec Sec.1/3.4, Task 11).

    The fit met the RMS <= 15% pin on all nine species on its default >= 1 yr window (worst:
    pikeperch at 10.71%); there is currently no automatic fallback if a future refit fails it.
    An earlier ``widen`` escape hatch (re-fit a failing species from a later minimum age) was
    removed (review R40, 2026-09-05): it called ``fit_species(tg, fx, min_age_years=...)``, and
    ``fit_species`` has never accepted that keyword, so the path would raise ``TypeError`` on
    first use. If a future fit genuinely needs window-widening, implement
    ``min_age_years`` on ``fit_species`` itself (its ``idx_all = np.arange(ndt, n_steps + 1)``
    would start from ``round(min_age_years * ndt)`` instead) and add a test that exercises it,
    against a real failing case -- do not restore this parameter without that.
    """
    import tempfile

    from osmose.demo import osmose_demo

    out_dir = out_dir or (ROOT / "data" / "baltic" / "scenarios" / "c3_bioen")
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_nc = ROOT / "data" / "baltic" / "forcing" / "baltic_temperature_2layer_climatology.nc"
    if not temp_nc.exists():
        raise FileNotFoundError(
            f"{temp_nc} not found -- run scripts/build_baltic_temperature_forcing.py first "
            "(Task 10)"
        )

    with tempfile.TemporaryDirectory(prefix="c3_bioen_baltic_") as tmp:
        demo_info = osmose_demo("baltic", Path(tmp))
        config_file = Path(demo_info["config_file"])
        config_dir = config_file.parent
        raw = OsmoseConfigReader().read(config_file)
        config = EngineConfig.from_dict(raw)

        targets, sp_index, t24_by_name = _species_targets_from_baltic(
            raw, config, config_dir, temp_nc
        )

        fx = BioenFixed()
        results = [fit_species(tg, fx) for tg in targets]
        for res in results:
            _assert_baltic_pins(res, fx)

        zlayer = {name: SPECIES_ZLAYER[name] for name in sp_index}
        m0 = {tg.name: tg.m0 for tg in targets}

        background_list = parse_background_species(raw, config.n_species, config.n_dt_per_year)
        bg_imax: dict[int, float] = {}
        bg_beta: dict[int, float] = {}
        for b_sp in background_list:
            bg_imax[b_sp.file_index] = background_imax(config, b_sp, beta=BACKGROUND_BETA)
            bg_beta[b_sp.file_index] = BACKGROUND_BETA

        _print_baltic_table(results, fx, t24_by_name)
        _print_background_table(background_list, bg_imax, config.n_dt_per_year, BACKGROUND_BETA)

        bioen_lines = bioen_param_lines(
            results,
            fx,
            zlayer=zlayer,
            sp_index=sp_index,
            background_imax=bg_imax,
            notes=SPECIES_NOTE,
            m0=m0,
            background_beta=bg_beta,
        )
        bioen_path = out_dir / "baltic_param-bioen.csv"
        bioen_path.write_text("\n".join(bioen_lines) + "\n")

        overlay = build_overlay(bioen_path, temp_nc)
        overlay_with_meta: dict = dict(overlay)
        overlay_with_meta["_meta"] = {
            "spec": (
                "docs/superpowers/specs/2026-08-30-baltic-c3-bioen-stage1-design.md Sec.1/3.4"
            ),
            "generated_by": "scripts/fit_baltic_bioen_params.py --baltic",
            "commit": _git_commit_sha(),
        }
        arm_path = out_dir / "c3_bioen_arm.json"
        arm_path.write_text(json.dumps(overlay_with_meta, indent=2, sort_keys=True) + "\n")

        _write_readme(out_dir, results, background_list, bg_imax, t24_by_name)

        n_fail = sum(1 for res in results if res.rms_len_pct > RMS_PIN_PCT)
        print(
            f"\n{len(results)} species fitted, {len(results) - n_fail} within the "
            f"{RMS_PIN_PCT:g}% RMS pin (hard pin under --baltic: a failure raises)."
        )
        print(
            f"Wrote {bioen_path.relative_to(ROOT)}, {arm_path.relative_to(ROOT)}, and "
            f"{(out_dir / 'README.md').relative_to(ROOT)}."
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gate-b",
        nargs="?",
        const="data/examples_bioen",
        default=None,
        metavar="OUT_DIR",
        help="Fit the Bay-of-Biscay demo config (Task 8). Default OUT_DIR: data/examples_bioen",
    )
    parser.add_argument(
        "--baltic",
        action="store_true",
        help=(
            "Fit the production Baltic 9-species config (Task 11); writes an overlay ARM "
            "under data/baltic/scenarios/c3_bioen/, not a production config change."
        ),
    )
    args = parser.parse_args(argv)

    if args.baltic:
        run_baltic()
        return
    if args.gate_b is not None:
        out_dir = Path(args.gate_b)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        run_gate_b(out_dir)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
