#!/usr/bin/env python
"""Baltic cod reproductive-volume vs model-overshoot diagnostic.

Tests the hypothesis (docs/baltic-fish-lifecycle.md:386-406) that the OSMOSE
Baltic model's cod overshoot is driven by the *missing* reproductive-volume
recruitment gate: in reality eastern-Baltic-cod recruitment is capped by the
volume of deep-basin water that is simultaneously saline enough (>= ~11 PSU,
for eggs to stay neutrally buoyant) and oxygenated enough (>= ~2 mL/L, for
eggs to survive). The model has no salinity/oxygen state, so its cod cannot
respond to that cap.

What it does:
  1. Runs (or reads) the OSMOSE Baltic model and extracts the cod biomass
     trajectory, characterising its instability (CV, boom/bust ratio, trend).
  2. Reads CMEMS bottom salinity (`so`, PHY product) and bottom oxygen
     (`o2b`, or the deepest valid level of `o2`, BGC product), regrids to the
     40x50 Baltic grid, restricts to the deep-basin cod-spawning cells, and
     computes an annual reproductive-volume fraction = share of deep-basin
     cells meeting BOTH thresholds.
  3. Confirms from the config whether the model actually has any salinity/
     oxygen forcing (the mechanistic reason an overlay can/can't show coupling).
  4. If both series span multiple aligned years, overlays them and reports the
     correlation. Otherwise it reports what each source shows and states
     precisely which data are missing to complete the interannual test.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/baltic_rv_overshoot_diagnostic.py --run-model
  PYTHONPATH=. .venv/bin/python scripts/baltic_rv_overshoot_diagnostic.py \
      --cmems-phy data/cmems_cache/.../baltic_phy_..._so_...nc \
      --cmems-bgc data/cmems_cache/.../baltic_bgc_..._o2b_...nc \
      --run-model --years 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
BALTIC_CFG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
GRID_NC = ROOT / "data" / "baltic" / "baltic_grid.nc"
COD_SPAWN = ROOT / "data" / "baltic" / "maps" / "cod_spawning.csv"
CMEMS_DIR = ROOT / "data" / "cmems_cache" / "cmems_downloads"
MODEL_OUT = ROOT / "data" / "baltic" / "output"
DIAG_DIR = ROOT / "docs" / "diagnostics"

# 1 mL/L dissolved O2 ~= 44.66 umol/L == 44.66 mmol/m3 (CMEMS o2/o2b units).
ML_L_TO_MMOL_M3 = 44.66

# Eastern-Baltic-cod spawning window (Mar-Aug) — RV during these months gates recruitment.
SPAWNING_MONTHS = (3, 4, 5, 6, 7, 8)

log = logging.getLogger("rv_diag")


# --------------------------------------------------------------------------- #
# Grid / mask helpers
# --------------------------------------------------------------------------- #
def build_grid():
    """GridSpec for the Baltic config (40x50, lat 66->54 N, lon 10->30 E)."""
    from osmose.config import OsmoseConfigReader
    from osmose.maps.builder import GridSpec

    cfg = OsmoseConfigReader().read(BALTIC_CFG)
    return GridSpec.from_config(cfg)


def load_deep_basin_mask() -> np.ndarray:
    """(nlat, nlon) bool mask of cod deep-basin spawning cells (cod_spawning > 0).

    OSMOSE map CSVs are stored SOUTH-first (row 0 = southernmost, 54 N); the grid
    NetCDF and target_coords()/regrid() are NORTH-first (row 0 = 66 N). We flipud
    the CSV so the mask rows align with the north-first regridded CMEMS fields —
    without this the mask lands in the fresh Gulf of Bothnia instead of the
    Bornholm/Gdansk/Gotland cod basins.
    """
    grid = np.flipud(np.genfromtxt(COD_SPAWN, delimiter=";"))
    return grid > 0


def bottom_slice(da: xr.DataArray) -> np.ndarray:
    """Deepest valid level of a (time, depth, lat, lon) field -> (time, lat, lon).

    NaN (land / below-seafloor) is preserved so it is never mistaken for a real
    zero. Cells with no valid level at any depth stay NaN.
    """
    v = da.values.astype(np.float64)  # (t, d, lat, lon)
    _, nd, _, _ = v.shape
    valid = np.isfinite(v)
    depth_rank = np.where(valid, np.arange(nd)[None, :, None, None], -1)
    bottom_i = depth_rank.max(axis=1)  # (t, lat, lon); -1 => all-NaN column
    gather = np.clip(bottom_i, 0, nd - 1)[:, None, :, :]
    bot = np.take_along_axis(v, gather, axis=1)[:, 0, :, :]
    bot[bottom_i < 0] = np.nan
    return bot


def regrid_bottom(ds: xr.Dataset, var: str, grid) -> np.ndarray | None:
    """Extract var's bottom field and nearest-neighbour regrid to (time, nlat, nlon).

    Returns None if the variable is absent. NaN is preserved end-to-end.
    """
    from osmose.forcing.grid import get_coords, regrid

    if var not in ds:
        return None
    da = ds[var]
    if "depth" in da.dims:
        src = bottom_slice(da)  # (t, lat, lon), already deepest valid
    else:
        src = da.values.astype(np.float64)
        if src.ndim == 2:
            src = src[None, :, :]
    src_lat, src_lon = get_coords(ds)
    return regrid(src, src_lat, src_lon, grid)  # NaN copied through nearest-idx


# --------------------------------------------------------------------------- #
# Reproductive volume from CMEMS
# --------------------------------------------------------------------------- #
def _load_bottom_series(
    files: list[Path], var_candidates: list[str], grid
) -> tuple[np.ndarray | None, np.ndarray | None, float, bool]:
    """Concatenate the bottom field of `var_candidates[0]` (or fallback) across files.

    Files are processed in chronological order (sorted by name), each reduced to
    its (time, nlat, nlon) bottom field and stacked along time. Returns
    (series, times, src_max_depth_m, used_first_candidate).
    """
    chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    src_max_depth = np.nan
    used_first = False
    for path in sorted(files):
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            arr = var = None
            for cand in var_candidates:
                arr = regrid_bottom(ds, cand, grid)
                if arr is not None:
                    var = cand
                    break
            if arr is None:
                continue
            used_first = used_first or (var == var_candidates[0])
            chunks.append(arr)
            if "time" in ds.coords:
                time_chunks.append(ds["time"].values)
            else:
                time_chunks.append(np.arange(arr.shape[0]))
            if "depth" in ds.coords:
                src_max_depth = float(
                    np.nanmax([src_max_depth, float(np.nanmax(ds["depth"].values))])
                )
    if not chunks:
        return None, None, src_max_depth, used_first
    order = np.argsort(np.concatenate(time_chunks))
    series = np.concatenate(chunks, axis=0)[order]
    times = np.concatenate(time_chunks)[order]
    return series, times, src_max_depth, used_first


def reproductive_volume(
    sal_files: list[Path],
    oxy_files: list[Path],
    grid,
    deep_mask: np.ndarray,
    sal_thresh: float,
    o2_thresh_mll: float,
) -> dict:
    """Per-timestep fraction of deep-basin cells meeting the RV thresholds.

    Accepts one-or-many CMEMS files per variable (e.g. yearly reanalysis files),
    concatenated into a single chronological series. If salinity is unavailable,
    computes an OXYGEN-ONLY proxy (optimistic upper bound on RV, since it drops
    the >=11 PSU constraint).
    """
    o2_thresh = o2_thresh_mll * ML_L_TO_MMOL_M3
    n_deep = int(deep_mask.sum())

    sal, sal_times, _, _ = _load_bottom_series(sal_files, ["so"], grid)
    oxy, oxy_times, src_max_depth, used_o2b = _load_bottom_series(oxy_files, ["o2b", "o2"], grid)
    times = oxy_times if oxy is not None else sal_times

    if sal is None and oxy is None:
        return {"available": False, "n_deep": n_deep}

    # If both are present but differ in length (mismatched file sets), truncate to
    # the shorter so the elementwise threshold test stays aligned.
    if sal is not None and oxy is not None and sal.shape[0] != oxy.shape[0]:
        n = min(sal.shape[0], oxy.shape[0])
        sal, oxy = sal[:n], oxy[:n]
        times = times[:n] if times is not None else None

    # Data-adequacy check: if the O2 source is depth-capped well above the deep
    # basins AND every valid bottom value is far above threshold, the file is
    # blind to the deep hypoxia that actually gates cod-egg survival.
    o2_bottom_mean = o2_bottom_min = np.nan
    blind_to_hypoxia = False
    if oxy is not None:
        deep_vals = oxy[:, deep_mask]
        o2_bottom_mean = float(np.nanmean(deep_vals))
        o2_bottom_min = float(np.nanmin(deep_vals))
        # "shallow bottom" if the field never reaches the deep-basin sill (~60 m)
        shallow = not used_o2b and np.isfinite(src_max_depth) and src_max_depth < 60.0
        blind_to_hypoxia = shallow and o2_bottom_min > o2_thresh

    both = sal is not None and oxy is not None
    ref = sal if sal is not None else oxy
    nt = ref.shape[0]
    frac = np.full(nt, np.nan)
    for t in range(nt):
        ok = np.ones((grid.nlat, grid.nlon), dtype=bool)
        if sal is not None:
            ok &= np.nan_to_num(sal[t], nan=-1.0) >= sal_thresh
        if oxy is not None:
            ok &= np.nan_to_num(oxy[t], nan=-1.0) >= o2_thresh
        frac[t] = float(ok[deep_mask].mean()) if n_deep else np.nan

    return {
        "available": True,
        "both_criteria": both,
        "salinity_used": sal is not None,
        "oxygen_used": oxy is not None,
        "n_deep": n_deep,
        "times": times,
        "fraction": frac,
        "sal_thresh": sal_thresh,
        "o2_thresh_mmol_m3": o2_thresh,
        "src_max_depth_m": src_max_depth,
        "o2_bottom_mean": o2_bottom_mean,
        "o2_bottom_min": o2_bottom_min,
        "blind_to_hypoxia": blind_to_hypoxia,
    }


# --------------------------------------------------------------------------- #
# Model cod biomass
# --------------------------------------------------------------------------- #
def cod_biomass_series(run_model: bool, years: int | None) -> dict:
    """Return {'time': years[], 'biomass': tons[]} for cod, running the model if asked."""
    from osmose.config import OsmoseConfigReader
    from osmose.results import OsmoseResults

    if run_model:
        from osmose.engine import PythonEngine

        cfg = OsmoseConfigReader().read(BALTIC_CFG)
        if years is not None:
            cfg["simulation.time.nyear"] = str(years)
        log.info("Running Baltic on the Python engine (nyear=%s) ...", cfg["simulation.time.nyear"])
        results = PythonEngine().run_in_memory(cfg, seed=0)
    else:
        results = OsmoseResults(MODEL_OUT, prefix="baltic", strict=False)

    # biomass() returns a WIDE frame: a time column ("Time"/"time") + one column
    # per species (+ a "species" tag column). Select the cod column directly.
    df = results.biomass()
    if df is None or len(df) == 0 or "cod" not in df.columns:
        return {"available": False}
    time_col = "Time" if "Time" in df.columns else ("time" if "time" in df.columns else None)
    df = df.sort_values(time_col) if time_col else df
    time = df[time_col].to_numpy(dtype=float) if time_col else np.arange(len(df), dtype=float)
    return {
        "available": True,
        "time": time,
        "biomass": df["cod"].to_numpy(dtype=float),
    }


def model_forcing_audit() -> dict:
    """Confirm whether the config wires any salinity/oxygen forcing at all."""
    from osmose.config import OsmoseConfigReader

    cfg = OsmoseConfigReader().read(BALTIC_CFG)
    keys = list(cfg.keys())
    oxy_keys = [k for k in keys if k.startswith("oxygen.") or ".oxygen." in k]
    sal_keys = [k for k in keys if "salinity" in k]
    fo2 = cfg.get("simulation.bioen.fo2.enabled", "false")
    return {
        "oxygen_keys": oxy_keys,
        "salinity_keys": sal_keys,
        "bioen_fo2_enabled": str(fo2).strip().lower() == "true",
    }


def characterise_instability(
    t: np.ndarray, b: np.ndarray, window: tuple[int, int] | None = None
) -> dict:
    """Summary stats describing how unstable the trajectory is.

    window=(lo, hi) restricts to model years [lo, hi] (inclusive) before the
    stats, so the spin-up transient can be excluded.
    """
    t = np.asarray(t, dtype=float)
    b = np.asarray(b, dtype=float)
    if window is not None:
        sel = (t >= window[0]) & (t <= window[1])
        b = b[sel]
    finite = b[np.isfinite(b) & (b > 0)]
    if finite.size == 0:
        return {"empty": True}
    mean = float(finite.mean())
    cv = float(finite.std() / mean) if mean else float("nan")
    boom_bust = float(finite.max() / finite.min()) if finite.min() > 0 else float("inf")
    # trend over the last third of the run (is it still moving?)
    tail = b[max(0, len(b) - max(3, len(b) // 3)) :]
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0]) if tail.size >= 2 else float("nan")
    return {
        "empty": False,
        "mean": mean,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "cv": cv,
        "boom_bust_ratio": boom_bust,
        "tail_slope_per_step": slope,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_plot(model: dict, rv: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping PNG (CSV still written)")
        return False

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    if model.get("available"):
        axes[0].plot(model["time"], model["biomass"], color="#1f6f8b")
    axes[0].set_title("OSMOSE Baltic — cod biomass (model)")
    axes[0].set_xlabel("model year")
    axes[0].set_ylabel("biomass (t)")

    xlabel = "CMEMS timestep (month)"
    if rv.get("available"):
        yrs, spawn = annual_rv(rv.get("times"), rv["fraction"], months=SPAWNING_MONTHS)
        if yrs is not None:
            axes[1].plot(yrs, spawn, marker="o", color="#c1440e")
            axes[1].axhline(np.nanmean(spawn), ls="--", lw=0.8, color="grey")
            xlabel = "year (spawning-season Mar-Aug mean)"
        else:
            axes[1].plot(
                np.arange(rv["fraction"].size), rv["fraction"], marker="o", color="#c1440e"
            )
        crit = "S>=%.0f PSU & O2>=%.0f mmol/m3" % (rv["sal_thresh"], rv["o2_thresh_mmol_m3"])
        if not rv["both_criteria"]:
            crit += "  (OXYGEN-ONLY proxy — no salinity)"
        axes[1].set_title("Deep-basin reproductive-volume fraction  [%s]" % crit)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("fraction of deep-basin cells")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def write_csv(rv: dict, path: Path) -> None:
    if not rv.get("available"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    times = rv.get("times")
    lines = ["timestep,time,rv_fraction"]
    for i, f in enumerate(rv["fraction"]):
        tlabel = str(times[i]) if times is not None and i < len(times) else ""
        lines.append("%d,%s,%.6f" % (i, tlabel, f))
    path.write_text("\n".join(lines) + "\n")


def report(model: dict, rv: dict, forcing: dict) -> None:
    print("\n" + "=" * 74)
    print("BALTIC COD  —  reproductive-volume vs model-overshoot diagnostic")
    print("=" * 74)

    print("\n[1] MODEL FORCING AUDIT (does the config couple cod to salinity/O2?)")
    print("    oxygen forcing keys : %s" % (forcing["oxygen_keys"] or "NONE"))
    print("    salinity forcing keys: %s" % (forcing["salinity_keys"] or "NONE"))
    print("    bioen fO2 enabled    : %s" % forcing["bioen_fo2_enabled"])
    coupled = bool(
        forcing["oxygen_keys"] or forcing["salinity_keys"] or forcing["bioen_fo2_enabled"]
    )
    print(
        "    => Model cod recruitment is %s to reproductive volume."
        % ("COUPLED" if coupled else "NOT coupled (no environmental brake)")
    )

    print("\n[2] MODEL COD INSTABILITY")
    if not model.get("available"):
        print("    No cod biomass available (empty output). Re-run with --run-model.")
    else:
        stats = characterise_instability(model["time"], model["biomass"])
        if stats["empty"]:
            print("    Cod biomass all-zero.")
        else:
            print(
                "    mean=%.3g t   min=%.3g   max=%.3g"
                % (stats["mean"], stats["min"], stats["max"])
            )
            print(
                "    CV=%.2f   boom/bust(max/min)=%.1fx   tail slope=%.3g t/step"
                % (stats["cv"], stats["boom_bust_ratio"], stats["tail_slope_per_step"])
            )

    print("\n[3] REAL-WORLD REPRODUCTIVE VOLUME (CMEMS, deep-basin cells)")
    if not rv.get("available"):
        print("    No CMEMS salinity/oxygen field found — cannot compute RV.")
    else:
        f = rv["fraction"]
        print(
            "    deep-basin cells: %d   criteria: %s"
            % (
                rv["n_deep"],
                "salinity+oxygen" if rv["both_criteria"] else "OXYGEN-ONLY (proxy, optimistic)",
            )
        )
        print(
            "    RV fraction  mean=%.3f  min=%.3f  max=%.3f  (n=%d timesteps)"
            % (np.nanmean(f), np.nanmin(f), np.nanmax(f), f.size)
        )
        yrs, spawn = annual_rv(rv.get("times"), f, months=SPAWNING_MONTHS)
        if yrs is not None:
            order = np.argsort(spawn)
            lo = ", ".join("%d(%.2f)" % (yrs[i], spawn[i]) for i in order[:5])
            hi = ", ".join("%d(%.2f)" % (yrs[i], spawn[i]) for i in order[::-1][:5])
            print(
                "    spawning-season (Mar-Aug) RV over %d-%d: mean=%.3f  range %.3f-%.3f"
                % (yrs[0], yrs[-1], np.nanmean(spawn), np.nanmin(spawn), np.nanmax(spawn))
            )
            print("      lowest  RV years: %s" % lo)
            print("      highest RV years: %s  (major-inflow pulses)" % hi)
        if np.isfinite(rv.get("o2_bottom_mean", np.nan)):
            print(
                "    source bottom O2 over deep cells: mean=%.0f  min=%.0f mmol/m3"
                "  (threshold=%.0f)  source max depth=%.0f m"
                % (
                    rv["o2_bottom_mean"],
                    rv["o2_bottom_min"],
                    rv["o2_thresh_mmol_m3"],
                    rv["src_max_depth_m"],
                )
            )
        if rv.get("blind_to_hypoxia"):
            print(
                "    !! DATA INADEQUATE: O2 source is depth-capped at %.0f m and every"
                " deep-cell" % rv["src_max_depth_m"]
            )
            print("       bottom value exceeds the threshold, so this file CANNOT see the sub-sill")
            print(
                "       hypoxia (Bornholm/Gdansk/Gotland anoxia sits below ~80 m) that"
                " gates cod eggs."
            )
            print("       The RV fraction above is therefore NOT a usable RV estimate.")

    print("\n[4] VERDICT")
    _verdict(model, rv, coupled)
    print("=" * 74 + "\n")


def _verdict(model: dict, rv: dict, coupled: bool) -> None:
    rv_ok = rv.get("available", False)
    multiyear_rv = rv_ok and rv["fraction"].size >= 24  # >1 yr of monthly data
    both = rv_ok and rv.get("both_criteria", False)

    if not coupled:
        print("    * CONFIRMED: model has NO salinity/oxygen coupling, so cod recruitment")
        print("      has no environmental cap. Any overshoot is internally generated —")
        print("      consistent with the missing reproductive-volume gate being the")
        print("      binding constraint.")
    if rv_ok and np.nanmean(rv["fraction"]) < 0.5:
        kind = "" if both else " (oxygen-only proxy — TRUE fraction is lower)"
        print(
            "    * Real deep basins are substantially RV-limited: only %.0f%% of cod"
            " spawning cells%s meet the survival thresholds."
            % (100 * np.nanmean(rv["fraction"]), kind)
        )

    if multiyear_rv and both:
        yrs, spawn = annual_rv(rv.get("times"), rv["fraction"], months=SPAWNING_MONTHS)
        print(
            "    * Interannual RV series BUILT (%d-%d, full-depth salinity+oxygen)."
            % (yrs[0], yrs[-1])
        )
        print(
            "      Real spawning RV is chronically low (mean %.0f%%) and pulses with major"
            % (100 * np.nanmean(spawn))
        )
        print("      Baltic inflows — the exact negative feedback the model lacks.")
        print(
            "\n    NEXT STEP — the data blocker is now cleared; what remains is to BUILD the gate:"
        )
        print(
            "      - Feed this RV series as forcing and multiply cod B-H recruitment by RV/RV_ref"
        )
        print(
            "        (clip [0,1]) in osmose/engine/processes/reproduction.py, then re-run"
            " and compare"
        )
        print("        cod stability with vs without the gate (docs/diagnostics/).")
        print(
            "      - A literal model-year<->calendar-year correlation still needs the model driven"
        )
        print("        by real-year forcing (i.e. the gate itself) — this series enables that.")
        return

    print(
        "\n    To COMPLETE the interannual overlay (model-year <-> real-year"
        " correlation), the following data are still required:"
    )
    if not both:
        print("      - CMEMS PHY bottom SALINITY `so` (for the >=11 PSU criterion).")
    if not multiyear_rv:
        print(
            "      - A MULTI-DECADE CMEMS reanalysis (phy+bgc), not a single forecast"
            " year, to build an interannual RV series."
        )
    if not coupled:
        print(
            "      - An RV-driven egg-survival term in the model (forcing/ + reproduction)"
            " so the model can actually RESPOND to the series above."
        )


# --------------------------------------------------------------------------- #
def _auto_find_all(patterns: list[str]) -> list[Path]:
    """All files matching the first pattern that hits (sorted chronologically)."""
    if not CMEMS_DIR.exists():
        return []
    for pat in patterns:
        hits = sorted(CMEMS_DIR.glob(pat))
        if hits:
            return hits
    return []


def build_rv_gate_series(rv: dict, out_path: Path) -> Path:
    """Write per-year spawning-season RV (year,spawning_rv) for the engine gate.

    Requires the full salinity+oxygen RV (not the oxygen-only proxy) and a
    calendar time axis spanning >= 2 years. Raises rather than emitting a
    degenerate/optimistic file.
    """
    if not rv.get("available") or not rv.get("both_criteria"):
        raise ValueError("RV gate series requires both criteria (salinity + oxygen).")
    yrs, spawn = annual_rv(rv.get("times"), rv["fraction"], months=SPAWNING_MONTHS)
    if yrs is None or spawn is None:
        raise ValueError("RV series needs a calendar time axis spanning >= 2 years.")
    if np.any(~np.isfinite(spawn)):
        raise ValueError("RV series has NaN spawning-season year(s); cannot emit gate series.")
    lines = ["year,spawning_rv"] + ["%d,%.6f" % (int(y), v) for y, v in zip(yrs, spawn)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def annual_rv(times, fraction: np.ndarray, months: tuple[int, ...] | None = None):
    """Aggregate a monthly RV series to per-calendar-year means.

    `months` (1-12) restricts to a season, e.g. (3,4,5,6,7,8) for the cod
    spawning window. Returns (years[], annual_mean_rv[]) or (None, None) if the
    time axis is not calendar dates.
    """
    if times is None:
        return None, None
    try:
        yrs = np.array([int(str(t)[:4]) for t in times])
        mos = np.array([int(str(t)[5:7]) for t in times])
    except (ValueError, TypeError):
        return None, None
    if yrs.min() == yrs.max():  # single year -> nothing to aggregate
        return None, None
    sel = np.isin(mos, months) if months else np.ones(len(mos), bool)
    uy = np.unique(yrs)
    out = np.array([np.nanmean(fraction[(yrs == y) & sel]) for y in uy])
    return uy, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmems-phy", type=Path, default=None, help="CMEMS PHY file with `so`")
    ap.add_argument("--cmems-bgc", type=Path, default=None, help="CMEMS BGC file with `o2b`/`o2`")
    ap.add_argument(
        "--run-model", action="store_true", help="run the Python engine for cod biomass"
    )
    ap.add_argument("--years", type=int, default=None, help="override simulation.time.nyear")
    ap.add_argument("--sal-threshold", type=float, default=11.0, help="min salinity (PSU)")
    ap.add_argument("--o2-threshold", type=float, default=2.0, help="min oxygen (mL/L)")
    ap.add_argument("--out", type=Path, default=DIAG_DIR, help="output dir for plot/csv")
    ap.add_argument(
        "--emit-gate-series",
        type=Path,
        default=None,
        help="write the per-year RV gate series CSV for the engine and exit",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    phy = [args.cmems_phy] if args.cmems_phy else _auto_find_all(["*phy*so*.nc", "*phy*.nc"])
    bgc = (
        [args.cmems_bgc]
        if args.cmems_bgc
        else _auto_find_all(["*bgc*o2b*.nc", "*bgc*o2*.nc", "*bgc*.nc"])
    )
    log.info("CMEMS PHY (salinity): %d file(s)", len(phy))
    log.info("CMEMS BGC (oxygen)  : %d file(s)", len(bgc))

    grid = build_grid()
    deep_mask = load_deep_basin_mask()
    log.info("deep-basin cod-spawning cells: %d of %d", int(deep_mask.sum()), deep_mask.size)

    forcing = model_forcing_audit()
    model = cod_biomass_series(args.run_model, args.years)
    rv = reproductive_volume(phy, bgc, grid, deep_mask, args.sal_threshold, args.o2_threshold)

    if args.emit_gate_series is not None:
        path = build_rv_gate_series(rv, args.emit_gate_series)
        log.info("wrote gate series %s", path)
        return 0

    write_csv(rv, args.out / "baltic_rv_fraction.csv")
    if write_plot(model, rv, args.out / "baltic_rv_overshoot.png"):
        log.info("wrote %s", args.out / "baltic_rv_overshoot.png")

    report(model, rv, forcing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
