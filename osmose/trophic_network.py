"""Community trophic-network diagnostics from OSMOSE dietMatrix output.

Reads the per-timestep diet matrix (output/Trophic/*_dietMatrix*.csv), aggregates
it to a species-level predator->prey network per timestep, and (via
make_trophic_network_html) renders an interactive pyvis node-link graph with a
FIXED layout so the graph is stable as you step through time.

The network shows DIET COMPOSITION (% of a predator's diet), NOT consumption-
weighted trophic flow; predator size-stages are averaged UNWEIGHTED to species
(the 'stage' level keeps them split, which is exact); prey size-stages are summed
to species (exact). See the design doc's honest-limitations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from osmose.results import _read_output_csv


def _read_diet_matrix(output_dir: Path | str) -> pd.DataFrame:
    """Read the per-timestep diet matrix (wide Time,Prey,<predator-stage cols>).

    Globs '*_dietMatrix*.csv' (WILDCARD prefix — OsmoseResults.diet_matrix() can't
    find it; the file may be under a Trophic/ subdir). OSMOSE writes one file per
    replicate (``*_dietMatrix_Simu0.csv``, ``_Simu1`` …); we deterministically take
    the first replicate (Simu0, by sorted path). Raises FileNotFoundError if absent.
    """
    matches = sorted(Path(output_dir).rglob("*_dietMatrix*.csv"))
    if not matches:
        raise FileNotFoundError(f"No '*_dietMatrix*.csv' under {output_dir}")
    return _read_output_csv(matches[0])


def _split_species(label: str) -> str:
    """Strip a ' in [lo, hi[' size-class suffix to the species name; pass through if absent."""
    idx = label.find(" in [")
    return label[:idx] if idx != -1 else label


def available_times(output_dir: Path | str) -> list[float]:
    """Sorted unique Time values in the diet matrix (slider bounds)."""
    df = _read_diet_matrix(output_dir)
    return sorted(float(t) for t in df["Time"].unique())


def network_node_universe(output_dir: Path | str, predator_level: str = "species") -> list[str]:
    """All node ids (prey + predator) that can appear at any timestep, for the layout.

    Time-independent: the prey set and predator columns are constant across the file.
    'species' -> species-level ids; 'stage' -> predator nodes keep their stage label.
    """
    if predator_level not in ("species", "stage"):
        raise ValueError("predator_level must be 'species' or 'stage'")
    wide = _read_diet_matrix(output_dir)
    prey = {_split_species(str(p)) for p in wide["Prey"].unique()}
    pred_cols = [c for c in wide.columns if c not in ("Time", "Prey")]
    preds = (
        {_split_species(c) for c in pred_cols} if predator_level == "species" else set(pred_cols)
    )
    return sorted(prey | preds)
