"""Reconstruct exact _apply_predation_numba args from a captured cell-loop pre-state."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEAF_ARG_ORDER = [
    "p_idx", "cell_indices", "inst_abd", "n_dead", "species_id", "length", "weight",
    "age_dt", "first_feeding_age_dt", "feeding_stage", "pred_success_rate",
    "preyed_biomass", "trophic_level", "size_ratio_min", "size_ratio_max",
    "ingestion_rate", "fr_shape", "fr_halfsat", "n_dt_per_year", "n_subdt",
    "access_matrix", "has_access", "use_stage_access", "prey_access_idx",
    "pred_access_idx", "rsc_biomass", "rsc_size_min", "rsc_size_max", "rsc_tl",
    "rsc_access_rows", "n_resources", "n_species", "cell_id", "tl_weighted_sum",
    "tl_tracking", "diet_matrix", "diet_enabled", "prey_type_buf", "prey_id_buf",
    "prey_eligible_buf", "egg_retained",
    # Bioen tail added by the bioen-Numba-kernel plan (Task 2 Step 1). APPENDED, so the
    # MUTATED indices below and the by-name lookups in parity.py / bench.py are unmoved.
    # This spike captures a bioen-OFF run, so `bioen` is False and the four arrays are
    # zero-filled and never read -- the C kernel in kernel.c has no bioen branch and does
    # not need one for the parity comparison to stay valid.
    "bioen", "cap_fish", "raw_preyed", "e_net", "is_background",
]
# Arrays that _apply_predation_numba mutates in-place; supply fresh copies.
MUTATED = ["inst_abd", "n_dead", "pred_success_rate", "preyed_biomass",
           "rsc_biomass", "tl_weighted_sum", "diet_matrix"]


def load_capture(npz_path: Path) -> tuple[dict, dict]:
    """Load the cellloop.npz + sibling meta.json captured by Task 2."""
    arrays = dict(np.load(npz_path))
    meta = json.loads((Path(npz_path).parent / "meta.json").read_text())
    return arrays, meta


def _n_local(arrays: dict) -> np.ndarray:
    b = arrays["boundaries"]
    return (b[1:] - b[:-1]).astype(np.int64)


def select_cells(arrays: dict) -> dict[str, int]:
    """Return p10/p50/p95/small cell indices weighted by call-distribution of n_local."""
    import warnings

    nl = _n_local(arrays)
    nonempty = np.where(nl > 0)[0]
    if nonempty.size == 0:
        raise ValueError("no non-empty cells in capture")
    if np.unique(nl[nonempty]).size == 1:
        warnings.warn(
            "all non-empty cells share one n_local; p10/p50/p95/small collapse "
            "to identical cells — the distribution measurement is degenerate"
        )
    # Call-weighted distribution: repeat each cell's n_local that many times so
    # cells with more schools contribute proportionally.
    weighted = np.repeat(nl[nonempty], nl[nonempty])
    p10, p50, p95 = np.percentile(weighted, [10, 50, 95])

    def nearest(target: float) -> int:
        return int(nonempty[np.argmin(np.abs(nl[nonempty] - target))])

    return {
        "p10": nearest(p10),
        "p50": nearest(p50),
        "p95": nearest(p95),
        "small": int(nonempty[np.argmin(nl[nonempty])]),
    }


def build_leaf_args(arrays: dict, meta: dict, cell: int) -> tuple[list, int]:
    """Build the exact positional arg list for _apply_predation_numba for one cell.

    Returns (args, p_idx) where p_idx is the first live feeding predator in the cell.
    All mutated arrays are fresh copies; the captured arrays are not touched.
    Raises ValueError if cell is empty or contains no live feeding predator.
    """
    b = arrays["boundaries"]
    start, end = int(b[cell]), int(b[cell + 1])
    if end <= start:
        raise ValueError(f"cell {cell} is empty")

    cell_indices = np.asarray(arrays["sorted_indices"][start:end], dtype=np.int32)
    n_local = end - start

    _sc = meta.get("scalars", {})
    # These scalars MUST come from the captured run — there is no safe fallback.
    # n_dt_per_year/n_subdt: fr_shape.shape[0] == n_species, not n_dt; a wrong value
    #   silently corrupts max_eatable and the p_idx selection.
    # has_access/use_stage_access: a wrong default silently changes the predation
    #   code path and would corrupt the spike's gate.
    # n_resources/n_species: array-shape inference is not guaranteed to match the run.
    # Task 2 capture confirmed all are reliably present, so require them.
    _required = ("n_dt_per_year", "n_subdt", "n_resources", "n_species",
                 "has_access", "use_stage_access")
    _missing = [k for k in _required if k not in _sc]
    if _missing:
        raise ValueError(
            f"required scalars missing from captured scalars: {_missing}; re-run capture.py"
        )
    n_dt = int(_sc["n_dt_per_year"])
    n_subdt = int(_sc["n_subdt"])
    n_resources = int(_sc["n_resources"])

    inst_abd = arrays["inst_abd"]
    age_dt = arrays["age_dt"]
    ffa = arrays["first_feeding_age_dt"]
    weight = arrays["weight"]
    species_id = arrays["species_id"]
    ingestion = arrays["ingestion_rate"]

    # Find first school that will not early-return in _apply_predation_numba:
    # feeding-age, alive (inst_abd > 0), and has positive max_eatable.
    p_idx = -1
    for q in cell_indices:
        q = int(q)
        if age_dt[q] < ffa[q] or inst_abd[q] <= 0:
            continue
        biomass = inst_abd[q] * weight[q]
        max_eatable = biomass * ingestion[species_id[q]] / (n_dt * n_subdt)
        if max_eatable > 0:
            p_idx = q
            break
    if p_idx < 0:
        raise ValueError(f"cell {cell} has no live feeding predator")

    # Fresh copies of every array _apply_predation_numba writes to.
    fresh = {k: np.copy(arrays[k]) for k in MUTATED}

    # Scratch buffers allocated per-call (size = n_local + n_resources).
    scratch_n = n_local + n_resources
    built = {
        "p_idx": np.int32(p_idx),
        "cell_indices": cell_indices,
        "cell_id": np.int32(cell),
        "n_resources": np.int32(n_resources),
        "n_species": np.int32(int(_sc["n_species"])),
        "n_dt_per_year": np.int32(n_dt),
        "n_subdt": np.int32(n_subdt),
        "has_access": bool(_sc["has_access"]),
        "use_stage_access": bool(_sc["use_stage_access"]),
        "tl_tracking": bool(meta["flags"]["tl_tracking"]),
        "diet_enabled": bool(meta["flags"]["diet_enabled"]),
        "prey_type_buf": np.empty(scratch_n, dtype=np.int32),
        "prey_id_buf": np.empty(scratch_n, dtype=np.int32),
        "prey_eligible_buf": np.empty(scratch_n, dtype=np.float64),
        # Bioen tail: this capture is a bioen-OFF run, so the flag is False and the
        # arrays exist only to keep njit's argument types stable.
        "bioen": False,
        "cap_fish": np.zeros(inst_abd.shape[0], dtype=np.float64),
        "raw_preyed": np.zeros(inst_abd.shape[0], dtype=np.float64),
        "e_net": np.zeros(inst_abd.shape[0], dtype=np.float64),
        "is_background": np.zeros(inst_abd.shape[0], dtype=np.bool_),
    }

    args = []
    for name in LEAF_ARG_ORDER:
        if name in built:
            args.append(built[name])
        elif name in fresh:
            args.append(fresh[name])
        else:
            args.append(arrays[name])
    return args, p_idx
