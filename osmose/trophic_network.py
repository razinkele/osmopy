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


def diet_network_at(
    output_dir: Path | str,
    *,
    time,
    threshold: float = 5.0,
    predator_level: str = "species",
) -> pd.DataFrame:
    """Long ``predator, prey, proportion`` (percent) for one timestep.

    Prey size-stages are SUMMED to prey-species (exact). For predator_level
    'species', predator size-stages are averaged to species over their LIVE stages
    (a 0-sum dead stage is excluded — unweighted approximation); 'stage' keeps the
    predator stage label (exact). NaN cells dropped; links >= threshold kept.

    A NaN in one of a predator's live stages contributes 0 to that species mean —
    "no data" and "ate none" are conflated in this unweighted approximation.
    """
    if predator_level not in ("species", "stage"):
        raise ValueError("predator_level must be 'species' or 'stage'")
    wide = _read_diet_matrix(output_dir)
    times = {float(t) for t in wide["Time"].unique()}
    if float(time) not in times:
        raise ValueError(f"time {time} not in diet matrix (have e.g. {sorted(times)[:3]})")
    step = wide[wide["Time"] == float(time)]
    pred_cols = [c for c in step.columns if c not in ("Time", "Prey")]

    melted = step.melt(
        id_vars=["Prey"], value_vars=pred_cols, var_name="pred_stage", value_name="proportion"
    ).dropna(subset=["proportion"])
    melted["prey"] = melted["Prey"].map(_split_species)
    melted["pred_sp"] = melted["pred_stage"].map(_split_species)

    # Prey size-stages -> prey-species, within each predator STAGE (exact additive composition).
    per_stage = melted.groupby(["pred_stage", "pred_sp", "prey"], as_index=False)[
        "proportion"
    ].sum()
    # Live predator stages = those whose total over prey > 0 (a dead stage is all-zero).
    stage_total = per_stage.groupby("pred_stage")["proportion"].transform("sum")
    live = per_stage[stage_total > 0].copy()

    if predator_level == "stage":
        out = live.rename(columns={"pred_stage": "predator"})[["predator", "prey", "proportion"]]
    else:
        n_live = live.groupby("pred_sp")["pred_stage"].nunique()
        summed = live.groupby(["pred_sp", "prey"], as_index=False)["proportion"].sum()
        summed["proportion"] = summed["proportion"] / summed["pred_sp"].map(n_live)
        out = summed.rename(columns={"pred_sp": "predator"})

    out = out[out["proportion"] >= threshold]
    return out[["predator", "prey", "proportion"]].reset_index(drop=True)


def species_layout(node_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Deterministic FIXED (x, y) per node, scaled for vis.js.

    Computed once over the all-timestep node universe (so positions are stable as
    the time-slider moves — the graph doesn't re-jiggle per frame). Uses a
    fixed-seed networkx spring layout.
    """
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(sorted(set(node_ids)))
    pos = nx.spring_layout(g, seed=42)
    return {n: (float(x) * 600.0, float(y) * 600.0) for n, (x, y) in pos.items()}


def make_trophic_network_html(
    diet_df: pd.DataFrame,
    *,
    positions: dict[str, tuple[float, float]],
    threshold: float = 5.0,
    height: str = "600px",
) -> str:
    """Self-contained pyvis node-link HTML (fixed layout, physics off) for a diet network.

    ``threshold`` is a convenience re-filter for standalone callers; when ``diet_df`` is
    already filtered (e.g. by ``diet_network_at``), pass ``threshold=0.0`` to avoid a
    second, stricter clamp.
    """
    from pyvis.network import Network

    net = Network(directed=True, cdn_resources="in_line", height=height, width="100%")
    net.set_options('{"physics": {"enabled": false}}')
    df = diet_df[diet_df["proportion"] >= threshold]
    nodes = sorted(set(df["predator"]) | set(df["prey"]))
    for n in nodes:
        x, y = positions.get(n, (0.0, 0.0))
        net.add_node(n, label=n, x=float(x), y=float(y), physics=False)
    for row in df.itertuples():
        net.add_edge(
            row.predator,
            row.prey,
            value=float(row.proportion),
            title=f"{row.proportion:.1f}% of {row.predator}'s diet",
        )
    return net.generate_html()
