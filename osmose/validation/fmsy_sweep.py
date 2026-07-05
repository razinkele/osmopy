"""Model-internal fishery reference points via a per-species yield-vs-F sweep."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.validation import fisheries as fis

_DEFAULT_GRID = np.linspace(0.0, 2.0, 7)

# Force outputs required for the sweep. Note: output.ssb.enabled gates SSB in-memory
# (required for results.ssb()); output.yield.biomass.enabled is inert for in-memory runs
# (yield is collected unconditionally), but kept defensively for a future disk-write path.
_FORCE_OUTPUTS = {"output.ssb.enabled": "true", "output.yield.biomass.enabled": "true"}


class SharedFisheryError(RuntimeError):
    """Raised when a species' fishery lands >1 species (per-species sweep is ambiguous)."""


def _fisheries_enabled(cfg: dict) -> bool:
    return (
        cfg.get("module.multispecies.fisheries.enabled", "false").lower() == "true"
        and int(cfg.get("simulation.nfisheries", "0")) > 0
    )


def _species_to_fishery(cfg: dict) -> dict[str, int]:
    """Return species_name.lower() -> fishery column index (first catchability column > 0).

    Resolve the catchability file the same way the engine does: the reader injects
    the config dir under ``_osmose.config.dir``, and the engine resolves the relative
    path via :func:`osmose.engine.path_resolution.resolve_data_path`.
    """
    from osmose.engine.path_resolution import resolve_data_path

    catch_rel = cfg.get("fisheries.catchability.file")
    if not catch_rel:
        raise FileNotFoundError("fisheries.catchability.file not set")
    config_dir = cfg.get("_osmose.config.dir", "")
    catch_path = resolve_data_path(catch_rel, config_dir=config_dir)  # -> Path | None
    if catch_path is None:
        raise FileNotFoundError(f"catchability file not resolvable: {catch_rel!r}")
    df = pd.read_csv(catch_path, index_col=0)
    out: dict[str, int] = {}
    for r in range(len(df)):
        name = str(df.index[r]).strip().lower()
        for c in range(len(df.columns)):
            if float(df.iloc[r, c]) > 0:
                out[name] = c
                break
    return out


def fishing_override(
    base_config: dict, config: EngineConfig, species_idx: int
) -> tuple[str, float]:
    """Return ``(override_key, baseline_value)`` for the active fishing knob of species *i*.

    Parameters
    ----------
    base_config:
        Raw config dict as returned by :class:`~osmose.config.reader.OsmoseConfigReader`.
    config:
        Parsed :class:`~osmose.engine.config.EngineConfig` (built from the same dict).
    species_idx:
        Zero-based species index.

    Returns
    -------
    override_key:
        The config key that controls the fishing rate for this species:
        ``fisheries.rate.base.fsh{j}`` (v4 fisheries mode) or
        ``mortality.fishing.rate.sp{i}`` (legacy mode).
    baseline_value:
        The current value of ``config.fishing_rate[species_idx]``.

    Raises
    ------
    SharedFisheryError
        If the species' fishery is shared by more than one species (sweep is ambiguous).
    ValueError
        If the species is not found in the catchability matrix.
    """
    sp_name = config.species_names[species_idx].strip().lower()
    if _fisheries_enabled(base_config):
        s2f = _species_to_fishery(base_config)
        fsh = s2f.get(sp_name)
        if fsh is None:
            raise ValueError(f"species {sp_name!r} maps to no fishery")
        sharing = [n for n, j in s2f.items() if j == fsh]
        if len(sharing) > 1:
            raise SharedFisheryError(f"fishery {fsh} lands {len(sharing)} species: {sharing}")
        key = f"fisheries.rate.base.fsh{fsh}"
    else:
        key = f"mortality.fishing.rate.sp{species_idx}"
    return key, float(config.fishing_rate[species_idx])


@dataclass
class SweepPoint:
    """One point from a per-species yield-vs-F sweep.

    Parameters
    ----------
    species
        Species identifier (name).
    f_nominal
        Fishing rate input to the model (may differ from realized if the model doesn't converge).
    f_realized
        Actual steady-state fishing rate achieved.
    yield_eq
        Equilibrium yield at this F.
    ssb_eq
        Equilibrium spawning-stock biomass at this F.
    not_converged
        Flag if the equilibrium run did not converge (lower confidence).
    """

    species: str
    f_nominal: float
    f_realized: float
    yield_eq: float
    ssb_eq: float
    not_converged: bool = False


@dataclass
class ModelReferencePoint:
    """Fishery reference points (Fmsy, Bmsy, B0, Blim) for one species.

    Parameters
    ----------
    species
        Species identifier.
    fmsy
        Fishing rate at maximum sustainable yield (None if no valid peak).
    bmsy
        Spawning-stock biomass at Fmsy (None if no valid peak).
    b0
        Spawning-stock biomass at F=0 (unfished baseline).
    blim
        Limit reference point (0.2 * B0, or None if B0 <= 0).
    fmsy_at_boundary
        True if the yield peak is at the last grid point (grid extension recommended).
    multi_peak
        True if the yield curve has >1 interior local maximum (Fmsy ambiguous).
    caveats
        List of cautions or explanations (e.g., "multi-peak yield curve").
    curve
        The original SweepPoints (for debugging / CLI output).
    """

    species: str
    fmsy: float | None
    bmsy: float | None
    b0: float | None
    blim: float | None
    fmsy_at_boundary: bool = False
    multi_peak: bool = False
    caveats: list[str] = field(default_factory=list)
    curve: list = field(default_factory=list)


def _count_interior_peaks(y: list[float]) -> int:
    """Count interior local maxima (peaks not at the boundaries)."""
    return sum(1 for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1])


def equilibrium_mean(df: pd.DataFrame, sp: str, window_years: int) -> tuple[float, bool]:
    """Trailing-window mean of a wide Time+species frame.

    Returns ``(value, not_converged)`` where ``not_converged`` is True if the
    last window differs from the prior window by more than 5 % (relative).

    Parameters
    ----------
    df:
        Wide DataFrame with a ``Time`` column and species-named columns.
    sp:
        Species name (must be a column in *df*).
    window_years:
        Number of trailing years to average.
    """
    if sp not in df.columns or "Time" not in df.columns:
        return 0.0, True
    by_year = fis.annual_by_year(df[sp].to_numpy(), df["Time"].to_numpy(), how="mean")
    years = sorted(by_year)
    vals = [by_year[y] for y in years]
    if len(vals) < 2 * window_years:
        return (float(np.mean(vals)) if vals else 0.0), True
    last = float(np.mean(vals[-window_years:]))
    prior = float(np.mean(vals[-2 * window_years : -window_years]))
    not_conv = abs(last - prior) > 0.05 * (abs(prior) + 1e-9)
    return last, bool(not_conv)


def realized_exploited_f(results, sp: str, window_years: int) -> float:
    """Realized annual fishing mortality F, mean over the trailing window.

    Reads the FLAT in-memory ``results.mortality(sp)`` frame — columns
    ``[Time, Predation, Starvation, Additional, Fishing, Out, Foraging,
    Discards, Aging, species]`` — NOT the ``(cause, stage)`` MultiIndex used
    by the on-disk CSV reader in ``osmose.validation.fisheries``.  We sum the
    flat ``Fishing`` column per absolute year, then average the trailing
    ``window_years``.

    Parameters
    ----------
    results:
        In-memory ``OsmoseResults`` object.
    sp:
        Species name.
    window_years:
        Trailing-window length (years) for the mean.
    """
    try:
        df = results.mortality(sp)
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return 0.0
    if "Fishing" not in df.columns or "Time" not in df.columns:
        return 0.0
    by_year = fis.annual_by_year(df["Fishing"].to_numpy(), df["Time"].to_numpy(), how="sum")
    years = sorted(by_year)[-window_years:]
    return float(np.mean([by_year[y] for y in years])) if years else 0.0


def _run_one(args: tuple) -> tuple[float, float, float, float, bool]:
    """Worker function: run one (species, F-value, replicate) simulation.

    Accepts a single tuple so it is picklable for ``ProcessPoolExecutor.map``.
    """
    base_config, override_key, f_val, seed, sp_name, window_years = args
    try:
        import numba  # type: ignore[import-untyped]

        numba.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    cfg = dict(base_config)
    cfg.update(_FORCE_OUTPUTS)
    cfg[override_key] = str(f_val)
    res = PythonEngine().run_in_memory(cfg, seed=seed)
    y, _ = equilibrium_mean(res.yield_biomass(), sp_name, window_years)
    b, nc = equilibrium_mean(res.ssb(), sp_name, window_years)
    fr = realized_exploited_f(res, sp_name, window_years)
    return (f_val, fr, y, b, nc)


def run_yield_f_sweep(
    base_config: dict[str, str],
    config: EngineConfig,
    species_list: list[tuple[int, str]],
    *,
    grid: np.ndarray,
    n_years: int,
    replicates: int,
    window_years: int,
    max_workers: int | None,
    seed0: int = 0,
) -> dict[str, list[SweepPoint]]:
    """Run a per-species yield-vs-F sweep over *grid* and return curves.

    Parameters
    ----------
    base_config:
        Raw config dict (string keys + string values).
    config:
        Parsed ``EngineConfig`` built from *base_config*.
    species_list:
        ``[(species_idx, species_name), ...]``.
    grid:
        Array of nominal F values to evaluate.
    n_years:
        Simulation length (years) for each sweep point.
    replicates:
        Number of stochastic replicates per (species, F) combination.
    window_years:
        Trailing window for equilibrium means.
    max_workers:
        Worker count for ``ProcessPoolExecutor``; ``<=1`` runs serially
        in-process (monkeypatch-friendly for tests).
    seed0:
        Base RNG seed; replicate *r* uses ``seed0 + r``.

    Returns
    -------
    dict[str, list[SweepPoint]]
        One list of ``SweepPoint`` per species (F-sorted, averaged over
        replicates).
    """
    base = dict(base_config)
    base["simulation.time.nyear"] = str(n_years)

    tasks: list[tuple] = []
    meta: list[tuple[str, float]] = []

    for sp_idx, sp_name in species_list:
        try:
            key, _ = fishing_override(base, config, sp_idx)
        except (SharedFisheryError, ValueError, FileNotFoundError) as exc:
            import warnings

            warnings.warn(
                f"Skipping species {sp_name!r} (index {sp_idx}): {exc}",
                stacklevel=2,
            )
            continue

        # no-op-trap guard: assert the override key actually moves fishing_rate[sp_idx]
        probe = dict(base)
        probe[key] = str(float(config.fishing_rate[sp_idx]) + 1.0)
        if EngineConfig.from_dict(probe).fishing_rate[sp_idx] == config.fishing_rate[sp_idx]:
            raise RuntimeError(
                f"override key {key!r} does not move fishing_rate[{sp_idx}] "
                "(no-op-trap guard failed)"
            )

        for f_val in grid:
            for r in range(replicates):
                tasks.append((base, key, float(f_val), seed0 + r, sp_name, window_years))
                meta.append((sp_name, float(f_val)))

    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    if workers <= 1:
        # Serial in-process: monkeypatch-friendly, no pickling overhead
        raw_results = [_run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # ex.map yields results in task-submission order
            raw_results = list(ex.map(_run_one, tasks))

    # Group by (species, f_nominal) → average over replicates
    curves: dict[str, dict[float, list[tuple[float, float, float, bool]]]] = {}
    for (sp_name, f_val), (fv, fr, y, b, nc) in zip(meta, raw_results):
        curves.setdefault(sp_name, {}).setdefault(f_val, []).append((fr, y, b, nc))

    out: dict[str, list[SweepPoint]] = {}
    for sp_name, byf in curves.items():
        pts: list[SweepPoint] = []
        for f_nominal, reps in sorted(byf.items()):
            fr_mean = float(np.mean([r[0] for r in reps]))
            y_mean = float(np.mean([r[1] for r in reps]))
            b_mean = float(np.mean([r[2] for r in reps]))
            any_nc = any(r[3] for r in reps)
            pts.append(SweepPoint(sp_name, f_nominal, fr_mean, y_mean, b_mean, any_nc))
        out[sp_name] = pts
    return out


def compute_model_reference_points(
    base_config: dict[str, str],
    *,
    grid: np.ndarray | None = None,
    n_years: int | None = None,
    replicates: int = 3,
    window_years: int = 10,
    max_workers: int | None = None,
) -> dict[str, ModelReferencePoint]:
    """Compute model-internal fishery reference points via a yield-vs-F sweep.

    Runs the Python engine for each (species, F) combination over *grid*,
    averages equilibrium yield/SSB over *replicates*, and derives Fmsy/Bmsy/B0/Blim
    via :func:`derive_reference_points`.

    Parameters
    ----------
    base_config:
        Raw OSMOSE config dict (string keys + string values).
    grid:
        Array of nominal F values to sweep; defaults to
        ``np.linspace(0.0, 2.0, 7)``.
    n_years:
        Simulation length per sweep run; defaults to
        ``max(config.n_year, 30)``.
    replicates:
        Stochastic replicates per (species, F) point (default 3).
    window_years:
        Trailing-window length for equilibrium averaging (default 10).
    max_workers:
        ``ProcessPoolExecutor`` worker count; ``None`` uses
        ``os.cpu_count()``.  Pass ``1`` or ``max_workers=1`` to run
        serially in-process (useful for tests / monkeypatching).

    Returns
    -------
    dict[str, ModelReferencePoint]
        Keyed by species name.
    """
    config = EngineConfig.from_dict(dict(base_config))
    f_grid = _DEFAULT_GRID if grid is None else np.asarray(grid, dtype=float)
    # EngineConfig field is n_year (singular), NOT n_years
    effective_n_years = max(config.n_year, 30) if n_years is None else n_years
    species_list = list(enumerate(config.species_names))
    curves = run_yield_f_sweep(
        base_config,
        config,
        species_list,
        grid=f_grid,
        n_years=effective_n_years,
        replicates=replicates,
        window_years=window_years,
        max_workers=max_workers,
    )
    refs = derive_reference_points(curves)
    for sp, rp in refs.items():
        rp.curve = curves.get(sp, [])  # attach raw curve for CLI/debugging
    return refs


def derive_reference_points(curves: dict[str, list[SweepPoint]]) -> dict[str, ModelReferencePoint]:
    """Derive Fmsy, Bmsy, B0, and Blim from yield-vs-F sweep curves.

    For each species, identifies the global yield maximum and extracts reference points.
    Detects boundary peaks, multi-peak curves, and non-positive B0.

    Parameters
    ----------
    curves
        Dict mapping species name to list of SweepPoints (one curve per species).

    Returns
    -------
    dict[str, ModelReferencePoint]
        Reference points for each species.
    """
    out: dict[str, ModelReferencePoint] = {}
    for sp, pts in curves.items():
        pts = sorted(pts, key=lambda p: p.f_nominal)
        caveats: list[str] = []
        b0 = next((p.ssb_eq for p in pts if p.f_nominal == 0.0), None)
        blim = 0.2 * b0 if (b0 is not None and b0 > 0) else None
        if b0 is not None and b0 <= 0:
            caveats.append("B0 <= 0; no Blim")
        ys = [p.yield_eq for p in pts]
        imax = max(range(len(ys)), key=lambda i: ys[i]) if ys else None
        multi_peak = _count_interior_peaks(ys) > 1
        if multi_peak:
            caveats.append("multi-peak yield curve; Fmsy ambiguous")
        rp = ModelReferencePoint(
            sp, None, None, b0, blim, multi_peak=multi_peak, caveats=caveats, curve=pts
        )
        if imax is None or ys[imax] <= 0:
            caveats.append("no positive-yield F; no Fmsy")
        elif imax == 0:
            caveats.append("yield maximal at F=0 (over-fished at baseline); no valid Fmsy")
        elif imax == len(ys) - 1:
            rp.fmsy = pts[imax].f_realized
            rp.bmsy = pts[imax].ssb_eq
            rp.fmsy_at_boundary = True
            caveats.append("Fmsy at the last grid F (boundary); extend the grid")
        else:
            rp.fmsy = pts[imax].f_realized
            rp.bmsy = pts[imax].ssb_eq
            if pts[imax].not_converged:
                caveats.append("Fmsy grid point not converged; lower confidence")
        out[sp] = rp
    return out
