"""Resolve per-species fisheries reference points: user-supplied Bmsy + ICES-auto-filled Fmsy.

Design
------
- ReferencePoint is the central value object: one per species per run.
- load_reference_points merges two layers in order (later wins):
  1. ICES snapshot auto-fill: Fmsy from the primary (largest msy_btrigger) tonnes-unit stock.
  2. User overrides from fisheries_reference_points.json in the config dir.
- save_reference_points persists ONLY user-owned fields (bmsy always; fmsy ONLY when
  source is "user" or "mixed").  ICES auto-filled Fmsy is never written — re-derive on load
  so stale auto-fill cannot propagate across snapshot updates.
- ecosystem_of is a simple basename resolver used by the UI layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from osmose.validation.ices import load_snapshot

_SIDECAR = "fisheries_reference_points.json"


@dataclass
class ReferencePoint:
    """Per-species fisheries reference points.

    Attributes
    ----------
    species:
        Model species name (e.g. "herring").
    fmsy:
        Maximum sustainable yield fishing mortality rate.  Set from the
        primary ICES tonnes-unit stock, then overridden by any user-supplied
        value.  None when not available.
    bmsy:
        Maximum sustainable yield biomass in tonnes.  User-supplied only;
        ICES snapshots do not provide this directly.
    fmsy_stock:
        ICES stock key from which *Fmsy* was drawn (e.g. "her.27.3031").
    fmsy_year:
        Advice year of the ICES stock assessment that provided *Fmsy*.
    b_ref_kind:
        One of "bmsy_user" (user supplied) or "none" (unavailable).
    source:
        Provenance string: "ices:<stock>@<year>" (auto-filled), "user"
        (fully user-supplied), "mixed" (ICES stock + user fmsy override),
        or "none" (no data).
    caveats:
        Human-readable warnings (e.g. multi-stock species).
    """

    species: str
    fmsy: float | None = None
    bmsy: float | None = None
    fmsy_stock: str | None = None
    fmsy_year: int | None = None
    b_ref_kind: str = "none"  # "bmsy_user" | "none"
    source: str = "none"  # "ices:<stock>@<year>" | "user" | "mixed" | "none"
    caveats: list[str] = field(default_factory=list)

    @property
    def has_f_axis(self) -> bool:
        """True when a positive Fmsy reference point is available."""
        return self.fmsy is not None and self.fmsy > 0

    @property
    def has_b_axis(self) -> bool:
        """True when a positive Bmsy reference point is available."""
        return self.bmsy is not None and self.bmsy > 0

    @property
    def b_ref_label(self) -> str:
        """Human-readable label for the B-axis reference kind."""
        return "Bmsy [user]"


def ecosystem_of(config_dir: Path | None) -> str:
    """Return the ecosystem name from a config/data directory path.

    The ecosystem name is the immediate directory basename (e.g. "baltic",
    "eec_full").  Returns "unknown" for a None path.
    """
    return config_dir.name if config_dir is not None else "unknown"


def _to_float(v: object) -> float | None:
    """Coerce *v* to float; return None on failure or when *v* is None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _autofill_fmsy(species: str, snapshot: object, rp: ReferencePoint) -> None:
    """Populate *rp* with Fmsy from the primary tonnes-unit ICES stock.

    Primary stock selection
    -----------------------
    Among all tonnes-unit stocks linked to *species* that have a non-null
    *fmsy*:

    1. Pick the one with the **largest** msy_btrigger (proxy for stock
       size / importance).
    2. On tie, pick the one with the **latest** advice_year.

    Adds a caveat when more than one tonnes-unit stock is linked (multi-stock
    species).
    """
    manifest = snapshot.manifest
    stocks: list[str] = manifest.get("model_species_to_ices_stocks", {}).get(species, [])
    units: dict[str, str] = manifest.get("units_by_stock", {})
    years: dict[str, int] = manifest.get("advice_year_by_stock", {})
    ref_pts: dict[str, dict] = snapshot.reference_points

    tonnes_stocks = [s for s in stocks if units.get(s) == "tonnes"]
    candidates = [(s, _to_float(ref_pts.get(s, {}).get("fmsy"))) for s in tonnes_stocks]
    candidates = [(s, f) for s, f in candidates if f is not None]
    if not candidates:
        return

    def _sort_key(s: str) -> tuple[float, int]:
        bt = _to_float(ref_pts.get(s, {}).get("msy_btrigger")) or 0.0
        return (bt, years.get(s, 0))

    primary = max((s for s, _ in candidates), key=_sort_key)
    rp.fmsy = _to_float(ref_pts.get(primary, {}).get("fmsy"))
    rp.fmsy_stock = primary
    rp.fmsy_year = years.get(primary)
    rp.source = f"ices:{primary}@{rp.fmsy_year}"

    if len(tonnes_stocks) > 1:
        rp.caveats.append(
            f"Fmsy from primary stock {primary}; species maps to {len(tonnes_stocks)} tonnes stocks"
        )


def load_reference_points(
    ref_dir: Path,
    species_list: list[str],
    *,
    ices_snapshot_dir: Path | None = None,
) -> tuple[dict[str, ReferencePoint], list[str]]:
    """Load per-species reference points, merging ICES auto-fill with user overrides.

    Parameters
    ----------
    ref_dir:
        Directory that may contain fisheries_reference_points.json with
        user-supplied overrides.  A non-existent directory is tolerated
        (treated as no overrides).
    species_list:
        Model species names to resolve.
    ices_snapshot_dir:
        Optional path to an ICES SAG snapshot directory (must contain
        index.json).  When provided and the directory exists, Fmsy is
        auto-filled from the primary tonnes-unit stock per species.

    Returns
    -------
    refs:
        dict[species -> ReferencePoint] with one entry per species in
        *species_list*.
    unmatched:
        Keys present in the user sidecar JSON that have no corresponding
        species in *species_list* (useful for surfacing stale / mis-spelled
        overrides).
    """
    # Load user overrides
    user_data: dict[str, dict] = {}
    sidecar = Path(ref_dir) / _SIDECAR
    if sidecar.exists():
        user_data = json.loads(sidecar.read_text())

    unmatched = [k for k in user_data if k not in species_list]

    # Load ICES snapshot once (may be None)
    snapshot = None
    if ices_snapshot_dir is not None and Path(ices_snapshot_dir).exists():
        snapshot = load_snapshot(Path(ices_snapshot_dir))

    refs: dict[str, ReferencePoint] = {}
    for sp in species_list:
        rp = ReferencePoint(species=sp)

        # Layer 1: ICES auto-fill (Fmsy only)
        if snapshot is not None:
            _autofill_fmsy(sp, snapshot, rp)

        # Layer 2: user overrides (win over ICES auto-fill)
        u = user_data.get(sp, {})
        user_fmsy = _to_float(u.get("fmsy"))
        user_bmsy = _to_float(u.get("bmsy"))

        if user_fmsy is not None:
            rp.fmsy = user_fmsy
            rp.source = "user" if rp.fmsy_stock is None else "mixed"

        if user_bmsy is not None:
            rp.bmsy = user_bmsy
            rp.b_ref_kind = "bmsy_user"

        refs[sp] = rp

    return refs, unmatched


def save_reference_points(ref_dir: Path, refs: dict[str, ReferencePoint]) -> None:
    """Persist user-owned reference-point fields to the sidecar JSON.

    Only user-owned fields are written:

    - bmsy always (when set).
    - fmsy ONLY when source is "user" or "mixed" (i.e. the user
      explicitly supplied it).  ICES-auto-filled Fmsy (source starting
      with "ices:") is **never** persisted — it is re-derived from the
      live snapshot on each load.

    Species with no user-owned fields produce no entry in the JSON.

    Parameters
    ----------
    ref_dir:
        Output directory; created if it does not exist.
    refs:
        Mapping from species name to :class:.
    """
    Path(ref_dir).mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict] = {}
    for sp, rp in refs.items():
        entry: dict = {}
        if rp.bmsy is not None:
            entry["bmsy"] = rp.bmsy
        if rp.fmsy is not None and rp.source in ("user", "mixed"):
            entry["fmsy"] = rp.fmsy
        if entry:
            payload[sp] = entry
    (Path(ref_dir) / _SIDECAR).write_text(json.dumps(payload, indent=2))
