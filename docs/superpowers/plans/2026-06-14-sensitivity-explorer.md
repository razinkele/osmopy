# Parameter Sensitivity Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dedicated top-level Shiny page that loads persisted Sobol sensitivity results (ranked S1/ST tornado + table + export), plus a shared `sobol_io` save/load helper so the existing live calibration run persists an artifact the page discovers.

**Architecture:** Reuse `SensitivityAnalyzer` unchanged. New pure `osmose/calibration/sobol_io.py` (save/load/list + `rank_rows`/`influential_keys`/`rows_to_csv`). One additive persist hook in the live calibration handler. New `ui/pages/sensitivity_explorer.py` page with a shared `_rows` reactive feeding a tornado chart, table, and two downloads. New `make_sobol_tornado` figure builder in `ui/pages/calibration_charts.py`.

**Tech Stack:** Python 3.12, Shiny for Python, shinywidgets/Plotly, SALib (already present), pytest, Playwright. ruff + pyright gates.

**Spec:** `docs/superpowers/specs/2026-06-14-sensitivity-explorer-design.md`

---

## File Structure

- **Create** `osmose/calibration/sobol_io.py` — artifact store + pure view helpers (Task 1)
- **Create** `tests/test_sobol_io.py` (Task 1)
- **Modify** `ui/pages/calibration_charts.py` — add `make_sobol_tornado` (Task 2)
- **Create** `tests/test_sobol_tornado.py` (Task 2)
- **Modify** `ui/pages/calibration_handlers.py` — persist hook after line 1789 (Task 3)
- **Create** `ui/pages/sensitivity_explorer.py` — page ui + server (Task 4)
- **Modify** `app.py` — import + nav panel + server call (Task 4)
- **Modify** `tests/test_app_structure.py` — wiring/structure tests (Tasks 3 & 4)
- **Create** `tests/test_e2e_sensitivity_explorer.py` (Task 5)
- **Modify** `CHANGELOG.md` (Task 5)

Per-task gate (every task): `.venv/bin/ruff check osmose/ ui/ tests/`, `.venv/bin/ruff format osmose/ ui/ tests/`, and `.venv/bin/pyright <files touched>`.

---

### Task 1: `sobol_io.py` — artifact store + pure helpers

**Files:**
- Create: `osmose/calibration/sobol_io.py`
- Create: `tests/test_sobol_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sobol_io.py`:

```python
"""Unit tests for osmose.calibration.sobol_io."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from osmose.calibration.sobol_io import (
    influential_keys,
    list_sobol_results,
    load_sobol_result,
    rank_rows,
    rows_to_csv,
    save_sobol_result,
)


def _result_1d():
    return {
        "param_names": ["a", "b", "c"],
        "S1": np.array([0.4, 0.1, 0.25]),
        "ST": np.array([0.5, 0.15, 0.3]),
        "S1_conf": np.array([0.05, 0.02, 0.03]),
        "ST_conf": np.array([0.06, 0.03, 0.04]),
    }


def _result_2d():
    return {
        "param_names": ["a", "b"],
        "n_objectives": 2,
        "objective_names": ["o0", "o1"],
        "S1": np.array([[0.4, 0.1], [0.2, 0.7]]),
        "ST": np.array([[0.5, 0.15], [0.25, 0.8]]),
        "S1_conf": np.array([[0.05, 0.02], [0.03, 0.04]]),
        "ST_conf": np.array([[0.06, 0.03], [0.04, 0.05]]),
    }


def test_save_load_round_trip_1d(tmp_path):
    p = save_sobol_result(
        _result_1d(),
        metadata={"source": "test", "n_base": 16, "param_bounds": [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
                  "objective_names": ["RMSE"], "timestamp": "2026-06-14T08:00:00"},
        directory=tmp_path,
    )
    assert p.exists()
    d = load_sobol_result("2026-06-14T08:00:00", directory=tmp_path)
    assert d["param_names"] == ["a", "b", "c"]
    assert d["S1"] == [0.4, 0.1, 0.25]  # numpy stored as list
    assert d["n_objectives"] == 1
    assert d["objective_names"] == ["RMSE"]
    # param_bounds round-trips as list-of-lists (tuples serialize to lists)
    assert d["param_bounds"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert d["source"] == "test" and d["n_base"] == 16


def test_save_load_round_trip_2d(tmp_path):
    save_sobol_result(_result_2d(), metadata={"source": "s", "timestamp": "2026-06-14T09:00:00"},
                      directory=tmp_path)
    d = load_sobol_result("2026-06-14T09:00:00", directory=tmp_path)
    assert d["n_objectives"] == 2
    assert d["S1"] == [[0.4, 0.1], [0.2, 0.7]]


def test_save_collision_safe(tmp_path):
    md = {"source": "s", "timestamp": "2026-06-14T10:00:00"}
    p1 = save_sobol_result(_result_1d(), metadata=md, directory=tmp_path)
    p2 = save_sobol_result(_result_1d(), metadata=md, directory=tmp_path)
    assert p1 != p2  # second got a -1 suffix; neither overwritten
    assert p1.exists() and p2.exists()


def test_list_newest_first_and_skips_corrupt(tmp_path):
    save_sobol_result(_result_1d(), metadata={"source": "s", "timestamp": "2026-06-14T01:00:00"},
                      directory=tmp_path)
    save_sobol_result(_result_1d(), metadata={"source": "s", "timestamp": "2026-06-14T02:00:00"},
                      directory=tmp_path)
    (tmp_path / "sobol_broken.json").write_text("{ not json")
    out = list_sobol_results(directory=tmp_path)
    assert [s["timestamp"] for s in out] == ["2026-06-14T02:00:00", "2026-06-14T01:00:00"]
    assert out[0]["n_params"] == 3 and out[0]["n_objectives"] == 1


def test_list_tolerates_none_objective_names(tmp_path):
    r = _result_1d()
    save_sobol_result(r, metadata={"source": "s", "timestamp": "2026-06-14T03:00:00"}, directory=tmp_path)
    out = list_sobol_results(directory=tmp_path)
    assert out[0]["objective_names"] is None  # 1-D save with no objective_names provided


def test_rank_rows_sort_and_1d():
    rows = rank_rows(_result_1d(), sort="ST")
    assert [r["param"] for r in rows] == ["a", "c", "b"]  # ST desc: 0.5, 0.3, 0.15
    rows_s1 = rank_rows(_result_1d(), sort="S1")
    assert [r["param"] for r in rows_s1] == ["a", "c", "b"]
    rows_name = rank_rows(_result_1d(), sort="name")
    assert [r["param"] for r in rows_name] == ["a", "b", "c"]


def test_rank_rows_1d_ignores_objective_idx_even_with_objective_names():
    # The live-run shape: 1-D S1 but objective_names present (n_objectives=1).
    r = _result_1d()
    r["objective_names"] = ["RMSE"]
    r["n_objectives"] = 1
    rows = rank_rows(r, objective_idx=5, sort="ST")  # idx ignored for 1-D
    assert rows[0]["param"] == "a" and rows[0]["st"] == 0.5


def test_rank_rows_2d_selects_objective_and_clamps():
    rows0 = rank_rows(_result_2d(), objective_idx=0, sort="ST")
    assert rows0[0]["param"] == "a" and rows0[0]["st"] == 0.5
    rows1 = rank_rows(_result_2d(), objective_idx=1, sort="ST")
    assert rows1[0]["param"] == "b" and rows1[0]["st"] == 0.8
    rows_clamp = rank_rows(_result_2d(), objective_idx=99, sort="ST")  # clamped to 1
    assert rows_clamp[0]["param"] == "b"


def test_rank_rows_accepts_lists():
    r = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in _result_1d().items()}
    rows = rank_rows(r, sort="ST")
    assert rows[0]["param"] == "a"


def test_rank_rows_nan_sinks_to_bottom():
    r = _result_1d()
    r["ST"] = np.array([0.5, float("nan"), 0.3])
    rows = rank_rows(r, sort="ST")
    assert rows[-1]["param"] == "b" and math.isnan(rows[-1]["st"])


def test_influential_keys_boundary_and_nan():
    rows = rank_rows(_result_1d(), sort="ST")
    assert influential_keys(rows, 0.3) == ["a", "c"]  # st == 0.3 included
    r = _result_1d()
    r["ST"] = np.array([0.5, float("nan"), 0.3])
    assert "b" not in influential_keys(rank_rows(r, sort="ST"), 0.0)  # NaN excluded


def test_rows_to_csv():
    csv = rows_to_csv(rank_rows(_result_1d(), sort="ST"))
    assert csv.splitlines()[0] == "param,S1,S1_conf,ST,ST_conf"
    assert csv.splitlines()[1].startswith("a,")


def test_load_rejects_unsafe_timestamp(tmp_path):
    with pytest.raises(ValueError):
        load_sobol_result("../x", directory=tmp_path)
    with pytest.raises(ValueError):
        load_sobol_result("/abs", directory=tmp_path)


def test_load_round_trips_colon_timestamp(tmp_path):
    save_sobol_result(_result_1d(), metadata={"source": "s", "timestamp": "2026-06-14T11:22:33"},
                      directory=tmp_path)
    # stored file uses '-' but load is given the ':' form
    assert (tmp_path / "sobol_2026-06-14T11-22-33.json").exists()
    d = load_sobol_result("2026-06-14T11:22:33", directory=tmp_path)
    assert d["param_names"] == ["a", "b", "c"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sobol_io.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.calibration.sobol_io'`.

- [ ] **Step 3: Implement `osmose/calibration/sobol_io.py`**

```python
"""Persisted Sobol sensitivity artifacts: save/load/list + pure view helpers.

Pure core module (no UI imports). One JSON file per result under ``SENSITIVITY_DIR``,
mirroring ``osmose/history.py``'s run-record store. Producer: the live calibration
sensitivity run (via ``save_sobol_result``); consumer: the Sensitivity Explorer page.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from osmose.logging import setup_logging

_log = setup_logging("osmose.sobol_io")

# osmose/calibration/sobol_io.py -> parents[2] == repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENSITIVITY_DIR = _PROJECT_ROOT / "data" / "history" / "sensitivity"


def save_sobol_result(result: dict, *, metadata: dict, directory: Path | None = None) -> Path:
    """Persist a Sobol ``analyze()`` result + metadata as one JSON artifact.

    Numpy index arrays are stored as nested lists; ``param_bounds`` (and other metadata)
    are stored verbatim (tuples serialize to lists). Collision-safe: never overwrites an
    existing file (appends a ``-<n>`` suffix). Returns the written path.
    """
    directory = directory or SENSITIVITY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    names = result.get("objective_names")
    if names is None:
        names = metadata.get("objective_names")
    ts = metadata.get("timestamp") or result.get("timestamp") or datetime.now().isoformat()
    artifact = {
        "timestamp": ts,
        "source": metadata.get("source", "unknown"),
        "n_base": metadata.get("n_base"),
        "param_names": list(result["param_names"]),
        "param_bounds": metadata.get("param_bounds"),
        "objective_names": list(names) if names is not None else None,
        "n_objectives": int(result.get("n_objectives", 1)),
        "S1": np.asarray(result["S1"]).tolist(),
        "ST": np.asarray(result["ST"]).tolist(),
        "S1_conf": np.asarray(result["S1_conf"]).tolist(),
        "ST_conf": np.asarray(result["ST_conf"]).tolist(),
    }
    safe = ts.replace(":", "-")
    path = directory / f"sobol_{safe}.json"
    n = 1
    while path.exists():
        path = directory / f"sobol_{safe}-{n}.json"
        n += 1
    path.write_text(json.dumps(artifact, indent=2))
    return path


def load_sobol_result(timestamp: str, directory: Path | None = None) -> dict:
    """Load one artifact by its in-file timestamp. Validates BEFORE prefixing."""
    if "/" in timestamp or "\\" in timestamp or ".." in timestamp:
        raise ValueError(f"Unsafe timestamp: {timestamp!r}")
    directory = directory or SENSITIVITY_DIR
    path = directory / f"sobol_{timestamp.replace(':', '-')}.json"
    return json.loads(path.read_text())


def list_sobol_results(directory: Path | None = None) -> list[dict]:
    """Discover artifacts → lightweight summaries, newest-first; skip corrupt files."""
    directory = directory or SENSITIVITY_DIR
    if not directory.is_dir():
        return []
    out: list[dict] = []
    for p in directory.glob("sobol_*.json"):
        try:
            d = json.loads(p.read_text())
            out.append(
                {
                    "timestamp": d["timestamp"],
                    "source": d.get("source", "unknown"),
                    "n_base": d.get("n_base"),
                    "n_params": len(d.get("param_names", [])),
                    "n_objectives": int(d.get("n_objectives", 1)),
                    "objective_names": d.get("objective_names"),
                }
            )
        except Exception:  # noqa: BLE001 — skip a corrupt/partial artifact, don't crash discovery
            _log.warning("Skipping corrupt sobol artifact %s", p, exc_info=True)
            continue
    out.sort(key=lambda s: s["timestamp"], reverse=True)
    return out


def rank_rows(result: dict, objective_idx: int = 0, sort: str = "ST") -> list[dict]:
    """Per-param rows for the chosen objective, sorted for display (pure).

    2-D iff ``int(n_objectives) > 1`` (then index ``[objective_idx]``, clamped); else use
    arrays directly and ignore ``objective_idx``. NaN indices sink to the bottom.
    """
    n_obj = int(result.get("n_objectives", 1))
    names = list(result["param_names"])

    def _sel(key: str) -> np.ndarray:
        arr = np.asarray(result[key], dtype=float)
        if n_obj > 1:
            idx = max(0, min(objective_idx, n_obj - 1))
            return arr[idx]
        return arr

    s1, st, s1c, stc = _sel("S1"), _sel("ST"), _sel("S1_conf"), _sel("ST_conf")
    rows = [
        {
            "param": names[i],
            "s1": float(s1[i]),
            "s1_conf": float(s1c[i]),
            "st": float(st[i]),
            "st_conf": float(stc[i]),
        }
        for i in range(len(names))
    ]
    if sort == "name":
        rows.sort(key=lambda r: r["param"])
    else:
        col = "st" if sort == "ST" else "s1"
        # ascending by key; -value gives descending; NaN -> +inf so it sorts LAST
        rows.sort(key=lambda r: (math.inf if math.isnan(r[col]) else -r[col]))
    return rows


def influential_keys(rows: list[dict], threshold: float) -> list[str]:
    """Param keys with ``ST >= threshold`` (NaN ST is naturally excluded)."""
    return [r["param"] for r in rows if r["st"] >= threshold]


def rows_to_csv(rows: list[dict]) -> str:
    """Ranked rows → CSV text (header + one line per row)."""
    lines = ["param,S1,S1_conf,ST,ST_conf"]
    for r in rows:
        lines.append(f"{r['param']},{r['s1']},{r['s1_conf']},{r['st']},{r['st_conf']}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sobol_io.py -q`
Expected: PASS (14 passed).

- [ ] **Step 5: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → clean.
Run: `.venv/bin/ruff format osmose/ ui/ tests/` → apply.
Run: `.venv/bin/pyright osmose/calibration/sobol_io.py tests/test_sobol_io.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/sobol_io.py tests/test_sobol_io.py
git commit -m "feat(calibration): sobol_io artifact store + pure view helpers"
```

---

### Task 2: `make_sobol_tornado` figure builder

**Files:**
- Modify: `ui/pages/calibration_charts.py` (add `import math` near the top imports; add the function after `make_sensitivity_chart`)
- Create: `tests/test_sobol_tornado.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sobol_tornado.py`:

```python
"""Unit tests for make_sobol_tornado (pure Plotly figure builder)."""

from __future__ import annotations

from ui.pages.calibration_charts import make_sobol_tornado

_ROWS = [
    {"param": "a", "s1": 0.4, "s1_conf": 0.05, "st": 0.5, "st_conf": 0.06},
    {"param": "b", "s1": 0.1, "s1_conf": 0.02, "st": 0.15, "st_conf": 0.03},
]


def test_both_yields_two_traces():
    fig = make_sobol_tornado(_ROWS, indices="Both")
    assert len(fig.data) == 2


def test_s1_only_one_trace():
    fig = make_sobol_tornado(_ROWS, indices="S1")
    assert len(fig.data) == 1
    assert fig.data[0].name.startswith("S1")


def test_st_only_one_trace():
    fig = make_sobol_tornado(_ROWS, indices="ST")
    assert len(fig.data) == 1
    assert fig.data[0].name.startswith("ST")


def test_horizontal_and_error_x():
    fig = make_sobol_tornado(_ROWS, indices="Both")
    for tr in fig.data:
        assert tr.orientation == "h"
        assert tuple(tr.error_x.array) != ()


def test_threshold_highlights_influential_on_st_bar():
    fig = make_sobol_tornado(_ROWS, indices="ST", threshold=0.3)
    st_bar = fig.data[0]
    # rows order is as given (a: st=0.5 influential, b: st=0.15 not)
    colors = list(st_bar.marker.color)
    assert colors[0] != colors[1]  # influential vs muted differ


def test_empty_rows_no_bars():
    fig = make_sobol_tornado([], indices="Both")
    assert len(fig.data) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_sobol_tornado.py -q`
Expected: FAIL with `ImportError: cannot import name 'make_sobol_tornado'`.

- [ ] **Step 3: Implement**

In `ui/pages/calibration_charts.py`, add `import math` to the import block (after `import numpy as np`), then add after `make_sensitivity_chart`:

```python
def make_sobol_tornado(
    rows: list[dict],
    *,
    indices: str = "Both",
    threshold: float = 0.05,
    template: str = "osmose",
) -> go.Figure:
    """Horizontal tornado of Sobol indices from pre-ranked ``rows``.

    ``rows`` is the output of ``sobol_io.rank_rows`` (already objective-selected and
    sorted). ``indices`` in {"Both","S1","ST"} picks which bars to draw. ST bars are
    colored by influence (``st >= threshold``); a dashed reference line marks the
    threshold. Does no 1-D/2-D dispatch and no I/O.
    """
    if not rows:
        return go.Figure().update_layout(title="Sobol sensitivity", template=template)
    params = [r["param"] for r in rows]
    fig = go.Figure()
    if indices in ("Both", "S1"):
        fig.add_trace(
            go.Bar(
                name="S1 (First-order)",
                y=params,
                x=[r["s1"] for r in rows],
                orientation="h",
                error_x={"type": "data", "array": [r["s1_conf"] for r in rows]},
            )
        )
    if indices in ("Both", "ST"):
        colors = [
            "#d62728" if (not math.isnan(r["st"]) and r["st"] >= threshold) else "#7f7f7f"
            for r in rows
        ]
        fig.add_trace(
            go.Bar(
                name="ST (Total-order)",
                y=params,
                x=[r["st"] for r in rows],
                orientation="h",
                marker={"color": colors},
                error_x={"type": "data", "array": [r["st_conf"] for r in rows]},
            )
        )
    fig.update_layout(barmode="group", title="Sobol sensitivity", template=template)
    fig.update_yaxes(autorange="reversed")  # top-ranked param at the top
    fig.add_vline(x=threshold, line_dash="dash", line_color="#888")
    return fig
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_sobol_tornado.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → clean; `.venv/bin/ruff format osmose/ ui/ tests/`.
Run: `.venv/bin/pyright ui/pages/calibration_charts.py tests/test_sobol_tornado.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/calibration_charts.py tests/test_sobol_tornado.py
git commit -m "feat(ui): make_sobol_tornado horizontal Sobol chart builder"
```

---

### Task 3: Persist hook in the live calibration run

**Files:**
- Modify: `ui/pages/calibration_handlers.py` (after line 1789, `sens_result = analyzer.analyze(Y_1d)`)
- Modify: `tests/test_app_structure.py` (wiring test)

- [ ] **Step 1: Write the failing wiring test**

In `tests/test_app_structure.py`, append:

```python
def test_persist_hook_wired_in_calibration_handlers():
    """The live sensitivity run persists its result via save_sobol_result."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "calibration_handlers.py"
    ).read_text()
    assert "save_sobol_result" in src
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_persist_hook_wired_in_calibration_handlers -q`
Expected: FAIL (assert `save_sobol_result` in src).

- [ ] **Step 3: Add the import and the persist hook**

In `ui/pages/calibration_handlers.py`, add to the imports inside `handle_sensitivity` (alongside the existing `from osmose.calibration.sensitivity import SensitivityAnalyzer` near line 1694):

```python
        from osmose.calibration.sobol_io import save_sobol_result
```

Then, in the `run_sensitivity` thread body, immediately after the line
`sens_result = analyzer.analyze(Y_1d)` (line 1789), insert:

```python
                try:
                    save_sobol_result(
                        sens_result,
                        metadata={
                            "source": "calibration-live",
                            "n_base": 64,
                            "param_bounds": [[lo, hi] for lo, hi in param_bounds],
                            "objective_names": obj_names_sens or None,
                        },
                    )
                except Exception:  # noqa: BLE001 — persistence is additive; never break the live run
                    _log.warning("Failed to persist sensitivity result", exc_info=True)
```

(Indentation matches the surrounding `try` block inside `run_sensitivity`. `param_bounds`,
`obj_names_sens`, and `_log` are all in scope at this point.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_persist_hook_wired_in_calibration_handlers -q`
Expected: PASS.

- [ ] **Step 5: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → clean; `.venv/bin/ruff format osmose/ ui/ tests/`.
Run: `.venv/bin/pyright ui/pages/calibration_handlers.py tests/test_app_structure.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/calibration_handlers.py tests/test_app_structure.py
git commit -m "feat(calibration): persist live sensitivity result via sobol_io"
```

---

### Task 4: Sensitivity Explorer page + app registration

**Files:**
- Create: `ui/pages/sensitivity_explorer.py`
- Modify: `app.py` (import, nav panel, server call)
- Modify: `tests/test_app_structure.py` (page wiring/structure tests)

- [ ] **Step 1: Write the failing structure tests**

In `tests/test_app_structure.py`, append:

```python
def test_sensitivity_panel_present():
    """The Sensitivity page is wired into app_ui with its full widget set."""
    from app import app_ui

    html = str(app_ui)
    assert "Sensitivity" in html
    for wid in [
        "sens_run",
        "sens_objective_ui",
        "sens_index",
        "sens_threshold",
        "sens_sort",
        "sens_tornado",
        "sens_table",
        "sens_export_csv",
        "sens_export_keys",
    ]:
        assert wid in html, f"Missing widget id: {wid}"


def test_sensitivity_server_wired():
    """app.py calls sensitivity_explorer_server."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "app.py").read_text()
    assert "sensitivity_explorer_server" in src
    assert "sensitivity_explorer_ui" in src
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_sensitivity_panel_present tests/test_app_structure.py::test_sensitivity_server_wired -q`
Expected: FAIL (ids/`sensitivity_explorer_*` not present).

- [ ] **Step 3: Create `ui/pages/sensitivity_explorer.py`**

```python
"""Parameter Sensitivity Explorer — browse persisted Sobol results.

Loads a persisted Sobol artifact (osmose.calibration.sobol_io) and renders a ranked
S1/ST tornado, a table, and two exports. Read-only; no engine/config dependency.
"""

from __future__ import annotations

import plotly.graph_objects as go
from shiny import reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly

from osmose.calibration.sobol_io import (
    influential_keys,
    list_sobol_results,
    load_sobol_result,
    rank_rows,
    rows_to_csv,
)
from osmose.logging import setup_logging
from ui.pages.calibration_charts import make_sobol_tornado
from ui.styles import STYLE_MONO_KEY

_log = setup_logging("osmose.sensitivity_explorer")

_EMPTY_MSG = (
    "No sensitivity result to display — pick a saved run above, or run one from "
    "Calibration → Results → Sensitivity."
)


def _tpl(input) -> str:
    from ui.state import get_theme_mode

    return "osmose" if get_theme_mode(input) == "dark" else "osmose-light"


def _run_choices(summaries) -> dict[str, str]:
    return {
        s["timestamp"]: f"{s['timestamp'][:19]} ({s.get('source', 'unknown')}, n_base={s.get('n_base')})"
        for s in summaries
    }


def sensitivity_explorer_ui():
    return ui.div(
        ui.h3("Parameter Sensitivity Explorer"),
        ui.layout_columns(
            ui.input_select("sens_run", "Result", choices={}),
            ui.output_ui("sens_objective_ui"),
            ui.input_radio_buttons(
                "sens_index", "Indices", choices=["Both", "S1", "ST"], selected="Both", inline=True
            ),
            ui.input_select(
                "sens_sort",
                "Sort by",
                choices={"ST": "Total (ST)", "S1": "First-order (S1)", "name": "Name"},
            ),
            ui.input_slider(
                "sens_threshold", "Influence threshold (ST)", min=0, max=1, value=0.05, step=0.01
            ),
            col_widths=[3, 3, 2, 2, 2],
        ),
        output_widget("sens_tornado"),
        ui.output_ui("sens_table"),
        ui.div(
            ui.download_button("sens_export_csv", "Download ranked CSV"),
            ui.download_button("sens_export_keys", "Download influential keys"),
            class_="d-flex gap-2 mt-2",
        ),
    )


def sensitivity_explorer_server(input, output, session, state):
    # `state` is unused (page is config-independent) but kept for the standard
    # *_server(input, output, session, state) call signature.
    _last_choices: reactive.Value[dict] = reactive.Value({})

    def _safe(getter, default=None):
        try:
            return getter()
        except (SilentException, AttributeError):
            return default

    @reactive.effect
    def _populate_runs():
        if input.main_nav() != "sensitivity":
            return
        try:
            summaries = list_sobol_results()
        except Exception:  # noqa: BLE001 — never crash the page on a discovery error
            return
        choices = _run_choices(summaries)
        with reactive.isolate():
            if choices == _last_choices.get():
                return
        _last_choices.set(choices)
        ui.update_select("sens_run", choices=choices)

    @reactive.calc
    def _result():
        ts = _safe(input.sens_run)
        if not ts:
            return None
        try:
            return load_sobol_result(ts)
        except Exception:  # noqa: BLE001 — degrade to empty state on a bad/missing artifact
            _log.warning("Failed to load sobol result %r", ts, exc_info=True)
            return None

    def _obj_idx(r) -> int:
        n = int(r.get("n_objectives", 1))
        return max(0, min(int(_safe(input.sens_objective, "0") or 0), n - 1))

    @reactive.calc
    def _rows():
        r = _result()
        if r is None:
            return []
        return rank_rows(r, objective_idx=_obj_idx(r), sort=_safe(input.sens_sort, "ST") or "ST")

    @render.ui
    def sens_objective_ui():
        r = _result()
        if r is None or int(r.get("n_objectives", 1)) <= 1:
            return ui.div()
        names = r.get("objective_names")
        n = int(r.get("n_objectives", 1))
        choices = {str(i): (names[i] if names and i < len(names) else f"obj_{i}") for i in range(n)}
        return ui.input_select("sens_objective", "Objective", choices=choices, selected="0")

    @render_plotly
    def sens_tornado():
        if _result() is None:
            return go.Figure().update_layout(title=_EMPTY_MSG, template=_tpl(input))
        return make_sobol_tornado(
            _rows(),
            indices=_safe(input.sens_index, "Both") or "Both",
            threshold=float(_safe(input.sens_threshold, 0.05) or 0.05),
            template=_tpl(input),
        )

    @render.ui
    def sens_table():
        if _result() is None:
            return ui.p(_EMPTY_MSG, class_="text-muted")
        rows = _rows()
        thr = float(_safe(input.sens_threshold, 0.05) or 0.05)
        infl = set(influential_keys(rows, thr))
        body = [
            ui.tags.tr(
                ui.tags.td(r["param"], style=STYLE_MONO_KEY),
                ui.tags.td(f"{r['s1']:.3g}"),
                ui.tags.td(f"{r['st']:.3g}"),
                ui.tags.td(
                    ui.tags.span("influential", class_="badge bg-success")
                    if r["param"] in infl
                    else ""
                ),
            )
            for r in rows
        ]
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Param"),
                    ui.tags.th("S1"),
                    ui.tags.th("ST"),
                    ui.tags.th("Influential"),
                )
            ),
            ui.tags.tbody(*body),
            class_="table table-sm table-striped",
            style="font-size: 13px;",
        )

    @render.download(filename="sensitivity_ranked.csv")
    def sens_export_csv():
        yield rows_to_csv(_rows())

    @render.download(filename="influential_keys.txt")
    def sens_export_keys():
        thr = float(_safe(input.sens_threshold, 0.05) or 0.05)
        yield "\n".join(influential_keys(_rows(), thr))
```

- [ ] **Step 4: Register the page in `app.py`**

Add the import after the calibration page import (line 26):

```python
from ui.pages.sensitivity_explorer import sensitivity_explorer_ui, sensitivity_explorer_server
```

Add the nav panel immediately after the Calibration panel (line 280), inside the "Optimize" section:

```python
        ui.nav_panel("Sensitivity", sensitivity_explorer_ui(), value="sensitivity"),
```

Add the server call after `calibration_server(...)` (line 534):

```python
    sensitivity_explorer_server(input, output, session, state)
```

- [ ] **Step 5: Run the structure tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py -q`
Expected: PASS (all, including the new ones).

- [ ] **Step 6: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → clean; `.venv/bin/ruff format osmose/ ui/ tests/`.
Run: `.venv/bin/pyright ui/pages/sensitivity_explorer.py app.py` → 0 errors.

- [ ] **Step 7: Commit**

```bash
git add ui/pages/sensitivity_explorer.py app.py tests/test_app_structure.py
git commit -m "feat(ui): Parameter Sensitivity Explorer page + app registration"
```

---

### Task 5: e2e coverage + CHANGELOG

**Files:**
- Create: `tests/test_e2e_sensitivity_explorer.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the e2e test**

Create `tests/test_e2e_sensitivity_explorer.py`:

```python
"""End-to-end test for the Parameter Sensitivity Explorer page.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_sensitivity_explorer.py -v -m e2e

Writes a synthetic Sobol artifact into the real data/history/sensitivity/ directory
(the app subprocess reads that fixed path), then verifies the tornado + table render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = Path(__file__).resolve().parent.parent
_SENS_DIR = _REPO / "data" / "history" / "sensitivity"
_TS = "2026-06-14T08:00:00"
_LOAD_TIMEOUT = 15_000


@pytest.fixture
def one_result():
    _SENS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "timestamp": _TS,
        "source": "test",
        "n_base": 16,
        "param_names": ["species.linf.sp0", "species.k.sp0", "predation.efficiency.sp0"],
        "param_bounds": [[20.0, 60.0], [0.1, 0.5], [0.3, 0.9]],
        "objective_names": ["Biomass RMSE"],
        "n_objectives": 1,
        "S1": [0.40, 0.10, 0.25],
        "ST": [0.50, 0.15, 0.30],
        "S1_conf": [0.05, 0.02, 0.03],
        "ST_conf": [0.06, 0.03, 0.04],
    }
    path = _SENS_DIR / f"sobol_{_TS.replace(':', '-')}.json"
    path.write_text(json.dumps(artifact))
    yield _TS
    path.unlink(missing_ok=True)


def test_sensitivity_explorer_renders(page: Page, app: ShinyAppProc, one_result):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Open the Sensitivity page (triggers _populate_runs).
    page.locator(".nav-pills .nav-link[data-value='sensitivity']").click()

    # The history-backed selector populates after the page activates.
    expect(page.locator(f"#sens_run option[value='{_TS}']")).to_have_count(
        1, timeout=_LOAD_TIMEOUT
    )
    page.locator("#sens_run").select_option(_TS)

    # Tornado widget renders.
    expect(page.locator("#sens_tornado")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # Table renders the ranked params (content assertion — bare @render.ui is zero-height
    # until it recomputes, so to_contain_text waits for population).
    expect(page.locator("#sens_table")).to_contain_text(
        "species.linf.sp0", timeout=_LOAD_TIMEOUT
    )
    # Export buttons present.
    expect(page.locator("#sens_export_csv")).to_be_visible(timeout=_LOAD_TIMEOUT)

    page.screenshot(path=str(_REPO / "screenshots" / "sensitivity_explorer_e2e.png"))
```

- [ ] **Step 2: Run the e2e test**

Run: `.venv/bin/python -m pytest tests/test_e2e_sensitivity_explorer.py -v -m e2e`
Expected: PASS (1 passed). The synthetic artifact is created and cleaned up by the fixture.

- [ ] **Step 3: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, add a bullet:

```markdown
- **ui (sensitivity):** a top-level "Sensitivity" page that browses persisted Sobol sensitivity
  results — a ranked S1/ST tornado (with 95% CIs and an influence-threshold highlight), a sortable
  table, and CSV / influential-key exports. Backed by a new `osmose.calibration.sobol_io` artifact
  store (`save_sobol_result`/`load_sobol_result`/`list_sobol_results` + pure `rank_rows`/
  `influential_keys`/`rows_to_csv`); the existing live calibration sensitivity run now persists its
  result so the explorer can discover it. The Sobol analyzer itself is unchanged.
```

- [ ] **Step 4: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/` → clean; `.venv/bin/ruff format osmose/ ui/ tests/`.
Run: `.venv/bin/pyright tests/test_e2e_sensitivity_explorer.py` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_sensitivity_explorer.py CHANGELOG.md
git commit -m "test(sensitivity): e2e explorer page + CHANGELOG"
```

---

## Final verification (after all tasks)

- [ ] Full non-e2e suite: `.venv/bin/python -m pytest -m 'not e2e' -n auto -q`
- [ ] e2e: `.venv/bin/python -m pytest tests/test_e2e_sensitivity_explorer.py -v -m e2e`
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` and `.venv/bin/ruff format --check osmose/ ui/ tests/` clean
- [ ] `.venv/bin/pyright` clean on all touched files
- [ ] Final whole-implementation code review before finishing the branch.

## Self-Review (plan author)

- **Spec coverage:** sobol_io save/load/list + rank_rows/influential_keys/rows_to_csv (Task 1) ↔ spec §1; persist hook (Task 3) ↔ §2; page ui/server + shared `_rows` + empty state + downloads (Task 4) ↔ §3; `make_sobol_tornado` (Task 2) ↔ §4; data flow + error handling realized across Tasks 1/4; tests (Tasks 1/2/3/4/5) ↔ §Testing items 1/2/3/4; CHANGELOG (Task 5) ↔ §Testing note; pyright in every gate. No spec requirement without a task.
- **Type consistency:** row dict shape `{param,s1,s1_conf,st,st_conf}` identical across `rank_rows` (Task 1), `make_sobol_tornado` (Task 2), and the page table/downloads (Task 4); `rank_rows`/`influential_keys`/`rows_to_csv` signatures match between definition (Task 1) and call sites (Task 4); `make_sobol_tornado(rows, *, indices, threshold, template)` matches between definition (Task 2) and call (Task 4); `save_sobol_result(result, *, metadata, directory)` matches between definition (Task 1) and the hook (Task 3); dispatch on `int(n_objectives) > 1` is consistent (Task 1 `rank_rows`); selector keyed by `timestamp` (Task 4 `_run_choices`) matches `load_sobol_result(ts)` (Task 1) and the e2e `select_option(_TS)` (Task 5).
- **No placeholders:** every code step shows complete code; commands have expected output.
