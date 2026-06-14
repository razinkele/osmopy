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
        rows.sort(key=lambda r: math.inf if math.isnan(r[col]) else -r[col])
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
