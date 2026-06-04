# Trophic-network animation (pyvis) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Results-page "Trophic Network" sub-tab that renders the per-timestep diet matrix as an interactive **pyvis** node-link graph (predator→prey, cannibalism self-loops) with a **fixed layout** and a time-slider, so the user can watch the diet network shift over time.

**Architecture:** New pyvis-free analysis module `osmose/trophic_network.py` reads the `Trophic/*dietMatrix*.csv` **directly** (wildcard prefix — `OsmoseResults.diet_matrix()` can't find it), aggregates one timestep to a species-level `predator,prey,proportion` (prey summed = exact; predator size-stages averaged unweighted, dead stages excluded), computes **fixed node positions once** over the all-timestep node universe, and builds a self-contained pyvis HTML. A Results sub-tab embeds it via `ui.tags.iframe(srcdoc=…)`, with the layout cached per run and the time control an **index slider over the discrete diet-matrix Time list** (Shiny 1.5.1 has no `reactive.debounce`; the index keeps fractional/sub-annual Time addressable).

**Tech Stack:** Python 3.12, pandas, networkx (layout), **pyvis 4.2** (node-link graph; already installed in `.venv` via `git+https://github.com/razinkele/pyvis.git@v4.2`), Shiny 1.5.1, pytest, ruff.

**Reference spec:** `docs/superpowers/specs/2026-06-04-trophic-network-animation-design.md` (reviewed clean: Sankey killed by review, pyvis design + the 2 BLOCKER fixes verified against real data + real pyvis 4.2).

---

## Verified facts (audit — use exactly)

- Data: `data/eec_full/output/Trophic/eec_dietMatrix_Simu0.csv` — 1-line title preamble, header
  `Time,Prey,<predator×size-stage cols>`; 70 timesteps; values are **PERCENT (each predator-stage
  column sums to ~100)**. Predator cols ALL size-split (` in [`); the 10 plankton/benthos prey rows
  are single-row (no ` in [`); NaN cells + self-loops (cannibalism) are real.
- `osmose.results._read_output_csv(path) -> pd.DataFrame` is the preamble-safe reader (importable).
  `OsmoseResults.diet_matrix()` CANNOT read this file (`_ENGINE_SUBDIRS` excludes `Trophic/` + default
  `prefix="osm"`); `_do_load_results` builds `OsmoseResults(out_dir, strict=False)` so `res.prefix`
  is always `"osm"`. **→ read via `rglob("*_dietMatrix*.csv")` (wildcard), never `res.prefix`.**
- **pyvis 4.2** (installed): `Network(directed=True, cdn_resources='in_line', height=, width=)`;
  `set_options('{"physics": {"enabled": false}}')`; `add_node(id, label=, x=, y=, physics=False)`;
  `add_edge(src, dst, value=, title=)`; `generate_html() -> str` (self-contained when `in_line`;
  emits `"x"`/`"y"` + `"physics": {"enabled": false}`; renders cycles + self-loops). networkx 3.6.1
  is a pyvis dep; `spring_layout(G, seed=42)` is deterministic.
- `pyproject.toml` `dependencies` (lines 6-22) ends with
  `"shiny_deckgl @ git+https://github.com/razinkele/shiny_deckgl.git@v1.6.1",` — add pyvis like it.
- `ui/pages/results.py`: imports `from shiny import reactive, render, ui` (:12), `from shinywidgets
  import output_widget, render_plotly` (:14); the diet view is `ui.navset_card_tab(ui.nav_panel(
  "Diet Composition", output_widget("diet_chart")), …)` at :286-290; `_do_load_results` populates a
  *select* via `ui.update_select("result_species", choices=…)` at :405 (NO slider is dynamically
  populated — the new `ui.update_slider` is the first). `ui.update_slider(id, min=, max=, value=)`
  is valid in Shiny 1.5.1.
- CI lints `osmose/ ui/ tests/` (NOT `scripts/`).

> **RUFF FORMAT-FIRST:** the code blocks below are not pre-wrapped to ruff's style. In every verify
> step, run `.venv/bin/ruff format <touched files>` FIRST, then `ruff check` + `ruff format --check`
> (CI runs BOTH).
> **TEST-FILE E402:** keep ALL module-level imports of `tests/test_trophic_network.py` in the top
> block (each task edits that block); append only test functions.

## File Structure

- Modify: `pyproject.toml` — add the pyvis git dependency.
- Create: `osmose/trophic_network.py` — reader, `_split_species`, `available_times`,
  `network_node_universe`, `diet_network_at`, `species_layout`, `make_trophic_network_html`.
- Modify: `ui/pages/results.py` — "Trophic Network" sub-tab + render fn + cached calc + slider
  population.
- Create: `tests/test_trophic_network.py`.
- Modify: `CHANGELOG.md`.

---

## Task 1: Dependency + reader/label/time helpers

**Files:** Modify `pyproject.toml`; Create `osmose/trophic_network.py`; Test `tests/test_trophic_network.py`

- [ ] **Step 1: Add the pyvis dependency**

In `pyproject.toml`, after the `shiny_deckgl @ …` line inside `dependencies`, add:
```
    "pyvis @ git+https://github.com/razinkele/pyvis.git@v4.2",
```
(pyvis 4.2 is already installed in `.venv`; this records it for fresh installs/CI.)

- [ ] **Step 2: Write failing tests**

Create `tests/test_trophic_network.py`:
```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmose.trophic_network import (
    _read_diet_matrix,
    _split_species,
    available_times,
    network_node_universe,
)


def _write_diet(path, rows, cols):
    # rows: list of dicts with Time, Prey, <predator cols>; cols: predator column names
    df = pd.DataFrame(rows, columns=["Time", "Prey", *cols])
    df.to_csv(path, index=False)  # clean header, no preamble


def test_split_species():
    assert _split_species("cod in [10.000000, 30.000000[") == "cod"
    assert _split_species("Diatoms") == "Diatoms"


def test_read_diet_matrix_wildcard(tmp_path):
    d = tmp_path / "output" / "Trophic"
    d.mkdir(parents=True)
    _write_diet(d / "eec_dietMatrix_Simu0.csv",
                [{"Time": 1.0, "Prey": "herring", "cod in [0, 50[": 30.0}],
                ["cod in [0, 50["])
    wide = _read_diet_matrix(tmp_path / "output")  # wildcard finds it under Trophic/
    assert list(wide.columns) == ["Time", "Prey", "cod in [0, 50["]


def test_read_diet_matrix_missing(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError):
        _read_diet_matrix(tmp_path / "output")


def test_available_times(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_diet(d / "x_dietMatrix.csv",
                [{"Time": 2.0, "Prey": "a", "p in [0, 1[": 1.0},
                 {"Time": 1.0, "Prey": "a", "p in [0, 1[": 1.0}],
                ["p in [0, 1["])
    assert available_times(d) == [1.0, 2.0]


def test_network_node_universe(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_diet(d / "x_dietMatrix.csv",
                [{"Time": 1.0, "Prey": "herring in [0, 5[", "cod in [0, 5[": 10.0,
                  "cod in [5, 9[": 20.0}],
                ["cod in [0, 5[", "cod in [5, 9["])
    assert network_node_universe(d, "species") == ["cod", "herring"]
    assert network_node_universe(d, "stage") == ["cod in [0, 5[", "cod in [5, 9[", "herring"]


def test_read_diet_matrix_eec_real():
    wide = _read_diet_matrix(Path("data/eec_full/output"))
    assert "Time" in wide.columns and "Prey" in wide.columns
    assert wide["Time"].nunique() == 70
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'osmose.trophic_network'`).

- [ ] **Step 4: Create `osmose/trophic_network.py`**

```python
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


def _read_diet_matrix(output_dir) -> pd.DataFrame:
    """Read the per-timestep diet matrix (wide Time,Prey,<predator-stage cols>).

    Globs '*_dietMatrix*.csv' (WILDCARD prefix — one per run dir, may be under a
    Trophic/ subdir; OsmoseResults.diet_matrix() can't find it). Raises
    FileNotFoundError if absent.
    """
    matches = sorted(Path(output_dir).rglob("*_dietMatrix*.csv"))
    if not matches:
        raise FileNotFoundError(f"No '*_dietMatrix*.csv' under {output_dir}")
    return _read_output_csv(matches[0])


def _split_species(label: str) -> str:
    """Strip a ' in [lo, hi[' size-class suffix to the species name; pass through if absent."""
    idx = label.find(" in [")
    return label[:idx] if idx != -1 else label


def available_times(output_dir) -> list[float]:
    """Sorted unique Time values in the diet matrix (slider bounds)."""
    df = _read_diet_matrix(output_dir)
    return sorted(float(t) for t in df["Time"].unique())


def network_node_universe(output_dir, predator_level: str = "species") -> list[str]:
    """All node ids (prey + predator) that can appear at any timestep, for the layout.

    Time-independent: the prey set and predator columns are constant across the file.
    'species' -> species-level ids; 'stage' -> predator nodes keep their stage label.
    """
    if predator_level not in ("species", "stage"):
        raise ValueError("predator_level must be 'species' or 'stage'")
    wide = _read_diet_matrix(output_dir)
    prey = {_split_species(str(p)) for p in wide["Prey"].unique()}
    pred_cols = [c for c in wide.columns if c not in ("Time", "Prey")]
    preds = {_split_species(c) for c in pred_cols} if predator_level == "species" else set(pred_cols)
    return sorted(prey | preds)
```

- [ ] **Step 5: Verify (format-first) + commit**

Run: `.venv/bin/ruff format osmose/trophic_network.py tests/test_trophic_network.py`
Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -q` → 6 pass.
Run: `.venv/bin/ruff check osmose/trophic_network.py tests/test_trophic_network.py && .venv/bin/ruff format --check osmose/trophic_network.py tests/test_trophic_network.py`.
```bash
git -C /home/razinka/osmose/osmose-python add pyproject.toml osmose/trophic_network.py tests/test_trophic_network.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(trophic): pyvis dep + dietMatrix wildcard reader + label/time/universe helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `diet_network_at` (per-timestep species aggregation)

**Files:** Modify `osmose/trophic_network.py`; Test `tests/test_trophic_network.py`

- [ ] **Step 1: Write failing tests**

Edit the TOP import block to add `diet_network_at`:
```python
from osmose.trophic_network import (
    _read_diet_matrix,
    _split_species,
    available_times,
    diet_network_at,
    network_node_universe,
)
```
Append test functions to the END:
```python
def _diet_fixture(path):
    # herring has a DEAD [30,inf[ stage (all 0) — exercises dead-stage exclusion (NOT cod).
    # predator cols sum to ~100 per live stage. Includes a self-loop (cod eats cod) + a NaN.
    rows = [
        # prey-species "cod" split into 2 stages summed to species within a predator col
        {"Time": 1.0, "Prey": "cod in [0, 10[",
         "cod in [0, 50[": 5.0, "herring in [0, 10[": 0.0, "herring in [10, 30[": 0.0,
         "herring in [30, inf[": 0.0},
        {"Time": 1.0, "Prey": "cod in [10, inf[",
         "cod in [0, 50[": 15.0, "herring in [0, 10[": 0.0, "herring in [10, 30[": 0.0,
         "herring in [30, inf[": 0.0},
        {"Time": 1.0, "Prey": "herring in [0, 5[",
         "cod in [0, 50[": 80.0, "herring in [0, 10[": 60.0, "herring in [10, 30[": 40.0,
         "herring in [30, inf[": 0.0},
        {"Time": 1.0, "Prey": "Diatoms",
         "cod in [0, 50[": float("nan"), "herring in [0, 10[": 40.0, "herring in [10, 30[": 60.0,
         "herring in [30, inf[": 0.0},
    ]
    cols = ["cod in [0, 50[", "herring in [0, 10[", "herring in [10, 30[", "herring in [30, inf["]
    pd.DataFrame(rows, columns=["Time", "Prey", *cols]).to_csv(path, index=False)


def test_diet_network_species_prey_sum_and_dead_stage(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=0.0)
    m = {(r.predator, r.prey): r.proportion for r in net.itertuples()}
    # prey "cod" stages SUM within cod-predator: 5+15 = 20 (exact)
    assert m[("cod", "cod")] == pytest.approx(20.0)
    # herring predator: live stages are [0,10[ and [10,30[ (the [30,inf[ is all-zero=dead, excluded).
    # herring-on-Diatoms = mean(40, 60) over the 2 LIVE stages = 50 (NOT /3 incl. the dead stage)
    assert m[("herring", "Diatoms")] == pytest.approx(50.0)
    # herring-on-herring = mean(60, 40) = 50
    assert m[("herring", "herring")] == pytest.approx(50.0)


def test_diet_network_threshold_and_nan(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=30.0)
    # cod->cod is 20 -> filtered out at threshold 30; herring->Diatoms (50) kept
    assert ("cod", "cod") not in {(r.predator, r.prey) for r in net.itertuples()}
    assert (net["proportion"] >= 30.0).all()
    # cod->Diatoms was NaN -> dropped entirely
    assert ("cod", "Diatoms") not in {(r.predator, r.prey) for r in net.itertuples()}


def test_diet_network_stage_level(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=0.0, predator_level="stage")
    preds = set(net["predator"])
    assert "cod in [0, 50[" in preds  # predator kept at stage granularity
    assert "cod" not in preds


def test_diet_network_bad_time(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    with pytest.raises(ValueError):
        diet_network_at(d, time=99.0)


def test_diet_network_eec_real():
    net = diet_network_at(Path("data/eec_full/output"), time=1.0)  # no prefix (wildcard)
    assert list(net.columns) == ["predator", "prey", "proportion"]
    assert len(net) > 0 and (net["proportion"] >= 0).all()
    # species-level: no size suffix in node names
    assert not any(" in [" in s for s in set(net["predator"]) | set(net["prey"]))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -k "diet_network" -q`
Expected: FAIL (`cannot import name 'diet_network_at'`).

- [ ] **Step 3: Implement `diet_network_at`**

Append to `osmose/trophic_network.py`:
```python
def diet_network_at(
    output_dir,
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
    per_stage = melted.groupby(["pred_stage", "pred_sp", "prey"], as_index=False)["proportion"].sum()
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
```

- [ ] **Step 4: Verify (format-first) + commit**

Run: `.venv/bin/ruff format osmose/trophic_network.py tests/test_trophic_network.py`
Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -q` → all pass.
Run: `.venv/bin/ruff check osmose/trophic_network.py tests/test_trophic_network.py && .venv/bin/ruff format --check osmose/trophic_network.py tests/test_trophic_network.py`.
```bash
git -C /home/razinka/osmose/osmose-python add osmose/trophic_network.py tests/test_trophic_network.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(trophic): diet_network_at per-timestep species aggregation (prey-sum, dead-stage-excluded predator mean)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `species_layout` (fixed positions) + `make_trophic_network_html` (pyvis builder)

**Files:** Modify `osmose/trophic_network.py`; Test `tests/test_trophic_network.py`

- [ ] **Step 1: Write failing tests**

Edit the TOP import block to add `make_trophic_network_html, species_layout`. Append:
```python
def test_species_layout_deterministic():
    a = species_layout(["cod", "herring", "sprat"])
    b = species_layout(["sprat", "cod", "herring"])
    assert set(a) == {"cod", "herring", "sprat"}
    assert a["cod"] == b["cod"]  # deterministic (fixed seed), order-independent
    assert all(isinstance(v, tuple) and len(v) == 2 for v in a.values())


def test_make_trophic_network_html_self_contained_fixed_layout():
    pytest.importorskip("pyvis")
    import pandas as pd

    df = pd.DataFrame(
        {
            "predator": ["cod", "herring", "cod"],
            "prey": ["herring", "cod", "cod"],  # mutual cycle + self-loop
            "proportion": [70.0, 10.0, 20.0],
        }
    )
    pos = species_layout(["cod", "herring"])
    html = make_trophic_network_html(df, positions=pos, threshold=0.0)
    assert 'src="lib/' not in html  # self-contained (cdn_resources='in_line')
    assert '"physics"' in html and "false" in html  # physics disabled
    assert '"x"' in html and '"y"' in html  # fixed coords emitted
    assert "cod" in html and "herring" in html
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -k "species_layout or make_trophic" -q`
Expected: FAIL (`cannot import name 'species_layout'`).

- [ ] **Step 3: Implement**

Append to `osmose/trophic_network.py`:
```python
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
    """Self-contained pyvis node-link HTML (fixed layout, physics off) for a diet network."""
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
```

- [ ] **Step 4: Verify (format-first) + commit**

Run: `.venv/bin/ruff format osmose/trophic_network.py tests/test_trophic_network.py`
Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -q` → all pass.
Run: `.venv/bin/ruff check osmose/trophic_network.py tests/test_trophic_network.py && .venv/bin/ruff format --check osmose/trophic_network.py tests/test_trophic_network.py`.
```bash
git -C /home/razinka/osmose/osmose-python add osmose/trophic_network.py tests/test_trophic_network.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(trophic): species_layout fixed positions + make_trophic_network_html pyvis builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Results "Trophic Network" sub-tab

**Files:** Modify `ui/pages/results.py`; Test `tests/test_trophic_network.py`

> Render fns / reactive effects are not unit-tested (project convention). This task is verified by
> static-source wiring assertions + the page-build smoke + a manual run-through (Task 5).

- [ ] **Step 1: Write failing wiring tests**

Append to `tests/test_trophic_network.py`:
```python
def test_results_has_trophic_network_wiring():
    src = (Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py").read_text()
    assert "Trophic Network" in src           # the nav_panel
    assert "trophic_network" in src            # the output id / render fn
    assert "trophic_time" in src               # the time slider
    assert "make_trophic_network_html" in src  # the builder is used
    assert "update_slider" in src              # slider populated on load
    assert "_dietMatrix" not in src            # reads via the helper, not a hardcoded glob here
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -k results_has_trophic -q`
Expected: FAIL (the wiring strings are not in results.py yet).

- [ ] **Step 3a: Add the sub-tab UI**

In `ui/pages/results.py`, the diet nav_panel block (~:286-290) is:
```python
        ui.navset_card_tab(
            ui.nav_panel(
                "Diet Composition",
                output_widget("diet_chart"),
            ),
```
Insert a new `nav_panel` immediately AFTER the "Diet Composition" panel's closing `),` (before the
next `ui.nav_panel(`):
```python
            ui.nav_panel(
                "Trophic Network",
                # NB: this is an INDEX into the discrete diet-matrix Time list (see Step-3b/3c),
                # NOT the raw Time value — so fractional/sub-annual Time steps are addressable.
                ui.input_slider("trophic_time", "Timestep", min=0, max=0, value=0, step=1),
                ui.input_radio_buttons(
                    "trophic_predator_level",
                    "Predator level",
                    {"species": "Species", "stage": "Size stage"},
                    selected="species",
                    inline=True,
                ),
                ui.input_slider("trophic_threshold", "Min diet %", min=0, max=50, value=5, step=1),
                ui.output_ui("trophic_network"),
            ),
```

- [ ] **Step 3b: Populate the slider on load**

In `_do_load_results`, immediately AFTER the existing
`ui.update_select("result_species", choices=species_choices)` line (~:405), add:
```python
            # Trophic-network time slider = an INDEX into the discrete diet-matrix Time list
            # (0 .. n-1), so fractional/sub-annual Time values are addressable (the raw Time
            # is shown as a caption by the render fn). Leave at default if there's no diet output.
            try:
                from osmose.trophic_network import available_times

                _times = available_times(out_dir)
                if _times:
                    ui.update_slider(
                        "trophic_time", min=0, max=len(_times) - 1, value=0
                    )
            except (FileNotFoundError, OSError, ValueError):
                pass  # no diet output -> leave the slider at its default
```
(Confirm the local variable holding the loaded output dir is named `out_dir` in `_do_load_results`;
if it differs, use that name.)

- [ ] **Step 3c: Add the cached calc + render fn**

In `results_server`, near the other `@reactive.calc`/render fns, add a cached calc for the wide df +
layout, then the render fn. Place after the existing diet render fn (`diet_chart`, ~:639):
```python
    @reactive.calc
    def _trophic_cache():
        """(output_dir-keyed) cached (dir, times, {level: layout}) so slider ticks are cheap."""
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            return None
        from osmose.trophic_network import available_times, network_node_universe, species_layout

        try:
            times = available_times(out_dir)  # probes existence; raises if no diet matrix
        except (FileNotFoundError, OSError, ValueError):
            return None
        if not times:
            return None
        layouts = {
            lvl: species_layout(network_node_universe(out_dir, lvl)) for lvl in ("species", "stage")
        }
        return {"dir": out_dir, "times": times, "layouts": layouts}

    @render.ui
    def trophic_network():
        cache = _trophic_cache()
        if cache is None:
            return ui.div("No diet-matrix output found.", style=STYLE_EMPTY)
        try:
            from osmose.trophic_network import diet_network_at, make_trophic_network_html
        except ImportError:
            return ui.div("Install pyvis to view the trophic network.", style=STYLE_EMPTY)
        level = input.trophic_predator_level()
        # The slider holds an INDEX into cache["times"]; map it to the actual Time (clamped),
        # so a fractional/sub-annual Time is addressable and we never pass an absent time value.
        times = cache["times"]
        idx = max(0, min(int(input.trophic_time()), len(times) - 1))
        t = times[idx]
        try:
            net = diet_network_at(
                cache["dir"],
                time=t,
                threshold=float(input.trophic_threshold()),
                predator_level=level,
            )
            html = make_trophic_network_html(net, positions=cache["layouts"][level])
        except (FileNotFoundError, OSError, ValueError) as e:
            return ui.div(f"Could not build trophic network: {e}", style=STYLE_EMPTY)
        return ui.div(
            ui.tags.small(f"Time {t:g}", style=STYLE_MONO_KEY),
            ui.tags.iframe(
                srcdoc=html, style="width:100%; height:640px; border:0;", sandbox="allow-scripts"
            ),
        )
```
NOTES:
- `_safe_output_dir`, `STYLE_EMPTY`, `STYLE_MONO_KEY`, `reactive`, `ui` are all already imported
  in `ui/pages/results.py` (`STYLE_EMPTY, STYLE_MONO_KEY` at the `from ui.styles import …` line;
  if either is missing, fall back to `style="color:#888;"`).
- **No debounce:** Shiny 1.5.1 has **no** `reactive.debounce`/`throttle` and `input_slider`
  exposes no client-side rate policy — so the time control is a deliberate-pick *index* slider and
  reads `input.trophic_time()` directly. Each settled value renders one self-contained (`in_line`)
  ~660 KB iframe; dragging across many steps re-renders per step. That's an accepted trade-off for
  a results-analysis tab; if it ever matters, the follow-on is `cdn_resources="remote"` in
  `make_trophic_network_html` (tiny per-render payload, needs internet at view time) — out of scope
  for v1, which prioritizes offline self-containment.
- `sandbox="allow-scripts"` lets vis.js run inside the iframe.

- [ ] **Step 4: Verify (format-first)**

Run: `.venv/bin/ruff format ui/pages/results.py tests/test_trophic_network.py`
Run: `.venv/bin/python -m pytest tests/test_trophic_network.py -k results_has_trophic -q` → pass.
Run: `.venv/bin/python -c "import ui.pages.results; print('import ok')"` → `import ok`.
Run: `.venv/bin/python -m pytest tests/test_ui_results.py -q` → the page-build smoke still passes.
Run: `.venv/bin/ruff check ui/pages/results.py tests/test_trophic_network.py && .venv/bin/ruff format --check ui/pages/results.py tests/test_trophic_network.py`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/results.py tests/test_trophic_network.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(ui): Results Trophic Network sub-tab (pyvis iframe, cached layout, slider)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Docs + full verification + manual run-through

**Files:** Modify `CHANGELOG.md`

- [ ] **Step 1: CHANGELOG note**

Under `## [Unreleased]` → `### Added` (the subsection now exists), add:
```markdown
- **ui (trophic network):** a Results → Trophic Network sub-tab renders the per-timestep diet
  matrix as an interactive pyvis node-link graph (predator→prey, cannibalism self-loops) with a
  fixed layout and a time-slider, so the diet network can be stepped through over time. Shows diet
  composition (proportions), predator size-stages averaged unweighted — not consumption-weighted
  flow. New `osmose.trophic_network` module + a `pyvis` dependency.
```

- [ ] **Step 2: Full verification**

Run: `.venv/bin/python -m pytest tests/test_trophic_network.py tests/test_ui_results.py -v` (report counts; all pass).
Run: `.venv/bin/python -m pytest tests/ -k "trophic or results or ui" -q` (report; classify any failure pre-existing vs caused).
Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/` (clean on touched files; if format flags a touched file, run `.venv/bin/ruff format <file>` + re-test).

- [ ] **Step 3: Manual UI run-through (render fn not unit-tested)**

Launch (the EEC config emits the diet matrix): the run must point at an output dir with a
`Trophic/*dietMatrix*.csv` (e.g. load the EEC example + run, or set the output dir to
`data/eec_full/output`).
```bash
PYTHONPATH=/home/razinka/osmose/osmose-python timeout 30 .venv/bin/shiny run app.py --host 127.0.0.1 --port 8770
```
Then via Playwright (mcp__playwright__*) or a browser: open the app, load results from a dir with
diet output, open Results → **Trophic Network**: confirm the pyvis graph renders inside the iframe
(species nodes, predator→prey edges, at least one self-loop), the layout is **stable** when you move
the **Timestep** slider (nodes hold position, the `Time N` caption updates, edges change), and the
**Min diet %** slider thins the edges. Confirm no uncaught console errors. If launching is impractical, rely on the import +
page-build smoke + wiring tests and state which path was used. Kill the app afterward
(`pkill -f "shiny run app.py"`).

- [ ] **Step 4: Commit + finish**

```bash
git -C /home/razinka/osmose/osmose-python add CHANGELOG.md
git -C /home/razinka/osmose/osmose-python commit -m "docs(changelog): trophic-network animation (pyvis)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** pyvis git dep → T1; wildcard Trophic reader + `_split_species` + `available_times`
+ `network_node_universe` → T1; `diet_network_at` (prey-sum exact, predator-mean dead-stage-excluded,
percent, NaN-drop, threshold, stage level, bad-time ValueError) → T2; `species_layout` fixed
positions + `make_trophic_network_html` (in_line self-contained, physics-off, fixed x/y, self-loops)
→ T3; Results sub-tab (index-slider+level+threshold+iframe), cached times/layout, index→Time map,
`ui.update_slider` in `_do_load_results`, degrade (no-diet / no-pyvis) → T4; docs + manual
run-through → T5; scientific-validation caveat (composition-not-flow) in the CHANGELOG + module
docstring; out-of-scope (Sankey, heatmap, `diet_matrix()`/`_ENGINE_SUBDIRS` fix, served-static
optimization) → not in plan, per spec. ✅

**Placeholder scan:** no TBD/TODO; every code step has complete code + exact commands. The
"confirm the var name / `STYLE_*` import exists" notes are grounded verification instructions (with
fallbacks), not placeholders. ✅ (`reactive.debounce` was REMOVED after plan review — it does not
exist in Shiny 1.5.1; the index slider reads `input.trophic_time()` directly.)

**Type consistency:** `_read_diet_matrix(output_dir)` / `available_times(output_dir)` /
`network_node_universe(output_dir, predator_level)` / `diet_network_at(output_dir, *, time,
threshold=5.0, predator_level="species") -> df[predator,prey,proportion]` / `species_layout(node_ids)
-> {id:(x,y)}` / `make_trophic_network_html(diet_df, *, positions, threshold=5.0, height)` — defined
in T1–T3, used consistently in T4. The cached calc passes `cache["layouts"][level]` (a
`species_layout` dict) as `positions=`; `diet_network_at` output columns feed `make_trophic_network_html`.
threshold default 5.0 consistent. ✅
