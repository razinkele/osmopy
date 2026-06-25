"""Indicative stock status: per-year SSB/Bmsy and exploited-stage F/Fmsy → Kobe quadrants."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from osmose.validation import fisheries as fis
from osmose.validation.fisheries_reference import ReferencePoint

_EXPLOITABLE = ("Pre-recruits", "Recruits")  # Eggs excluded


@dataclass
class StockStatus:
    species: str
    years: list[int]
    b_over_bmsy: list[float | None]
    f_over_fmsy: list[float | None]
    b_ref_label: str
    latest_quadrant: str | None = None
    takeaway: str | None = None
    caveats: list[str] = field(default_factory=list)


def _quadrant(b: float | None, f: float | None) -> str | None:
    if b is None or f is None:
        return None
    if b >= 1 and f <= 1:
        return "green"
    if b < 1 and f > 1:
        return "red"
    return "yellow" if b >= 1 else "orange"


def _exploited_f_by_year(results, species: str, caveats: list[str]) -> dict[int, float] | None:
    """{absolute_year: annual F} on the exploited stage = the fished stage (Eggs excluded)
    with the largest total annual F. Years come from the mortalityRate Time column."""
    from osmose.validation.fisheries import _FISHED_TOL, _mortality_path, read_mortality

    try:
        df = read_mortality(_mortality_path(results.output_dir, results.prefix, species))
    except (FileNotFoundError, KeyError, ValueError, AttributeError) as e:
        print(f"WARN: no mortalityRate for {species!r}: {e}", file=sys.stderr)
        return None
    time = df.iloc[:, 0]  # first column = Time (fractional sim-year)
    per_stage = {
        s: fis.annual_by_year(df[("F", s)].to_numpy(), time.to_numpy(), how="sum")
        for s in _EXPLOITABLE
        if ("F", s) in df.columns
    }
    fished = {s: d for s, d in per_stage.items() if sum(d.values()) > _FISHED_TOL}
    if not fished:
        return None
    stage = max(fished, key=lambda s: sum(fished[s].values()))
    if len(fished) > 1:
        caveats.append(f"F measured on '{stage}'; other fished stages present")
    return fished[stage]


def compute_stock_status(
    results,
    refs: dict[str, ReferencePoint],
    config,
    *,
    species_list: list[str] | None = None,
    _f_override: dict[str, dict[int, float]] | None = None,
) -> list[StockStatus]:
    """Compute indicative stock status per species.

    Parameters
    ----------
    results:
        Object with a ``.ssb(species)`` method returning a DataFrame with
        ``Time`` and per-species columns.
    refs:
        Per-species reference points (Bmsy, Fmsy).
    config:
        Config object (currently unused; reserved for future cadence detection).
    species_list:
        Subset of species to process.  Defaults to all keys in *refs*.
    _f_override:
        Testing hook: ``{species: {year: annual_F}}``.  When provided for a
        species, skips the mortalityRate CSV read.

    Returns
    -------
    list[StockStatus]
        One entry per species in *species_list*.
    """
    species_list = species_list or list(refs)
    out: list[StockStatus] = []
    for sp in species_list:
        rp: ReferencePoint = refs.get(sp, ReferencePoint(species=sp))
        caveats = list(rp.caveats)

        # F per absolute year (dict {year: annual F})
        if _f_override and sp in _f_override:
            f_map: dict[int, float] = dict(_f_override[sp])
        else:
            f_map = _exploited_f_by_year(results, sp, caveats) or {}

        # SSB per absolute year — annual MEAN of the saved rows in each year (cadence-correct)
        b_map: dict[int, float] = {}
        try:
            sdf = results.ssb(sp)
            if sp in sdf.columns:
                b_map = fis.annual_by_year(sdf[sp].to_numpy(), sdf["Time"].to_numpy(), how="mean")
        except (FileNotFoundError, KeyError, ValueError):
            caveats.append("SSB unavailable (enable output.ssb.enabled)")

        years = sorted(set(f_map) | set(b_map))
        b_ratio: list[float | None] = []
        f_ratio: list[float | None] = []
        for y in years:
            b = b_map.get(y)
            f = f_map.get(y)
            b_ratio.append(b / rp.bmsy if (rp.has_b_axis and b is not None) else None)
            f_ratio.append(f / rp.fmsy if (rp.has_f_axis and f is not None) else None)

        if not rp.has_b_axis:
            caveats.append("No Bmsy supplied — B-axis unavailable")
        if not rp.has_f_axis:
            caveats.append("No Fmsy — F-axis unavailable")

        # Walk backwards in time to find the latest year with a computable quadrant
        quad = None
        takeaway = None
        for i in range(len(years) - 1, -1, -1):
            quad = _quadrant(b_ratio[i], f_ratio[i])
            if quad is not None:
                br = b_ratio[i]
                fr = f_ratio[i]
                takeaway = (
                    f"Indicative: F {'above' if fr > 1 else 'at/below'} Fmsy and "
                    f"SSB {'below' if br < 1 else 'at/above'} your Bmsy"
                )
                break

        out.append(
            StockStatus(
                species=sp,
                years=years,
                b_over_bmsy=b_ratio,
                f_over_fmsy=f_ratio,
                b_ref_label=rp.b_ref_label,
                latest_quadrant=quad,
                takeaway=takeaway,
                caveats=caveats,
            )
        )
    return out
