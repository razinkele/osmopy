"""Model-internal fishery reference points via a per-species yield-vs-F sweep."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from osmose.engine.config import EngineConfig


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
