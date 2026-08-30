#!/usr/bin/env python3
"""Gate A reference for C3: production Baltic (bioen OFF) biomass(), 5 seeds x 50 yr.

`--produce` runs the engine at the CURRENT commit and writes the fixture JSON (only ever run
on the untouched master engine); `--check` re-runs and asserts bit-identity against the
committed fixture; `--from-npz` converts the local npz written on 2026-08-30 (commit 75e92da).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs" / "diagnostics" / "c3_gate_a_master_baseline.json"
SEEDS = (42, 123, 7, 999, 2024)
N_YEAR = 50


def _engine_commit() -> str:
    return (
        subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )


def _production_config() -> dict[str, str]:
    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo

    tmp = Path(tempfile.mkdtemp(prefix="c3_gate_a_"))
    cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    cfg["simulation.time.nyear"] = str(N_YEAR)
    return cfg


def run_seed(seed: int) -> pd.DataFrame:
    from osmose.engine import PythonEngine

    warnings.simplefilter("ignore")
    return PythonEngine().run_in_memory(_production_config(), seed=seed).biomass()


def load_gate_a_fixture(path: Path = FIXTURE) -> dict:
    return json.loads(Path(path).read_text())


def check_against_fixture(fixture: dict, seed: int, bio_df: pd.DataFrame) -> list[str]:
    """Columns whose series differ (array_equal) from the fixture for this seed."""
    ref = np.asarray(fixture["series"][str(seed)], dtype=np.float64)
    bad = []
    for j, col in enumerate(fixture["columns"]):
        got = bio_df[col].to_numpy(dtype=np.float64)
        if got.shape != ref[:, j].shape or not np.array_equal(got, ref[:, j]):
            bad.append(col)
    return bad


def write_fixture(series: dict[int, np.ndarray], columns: list[str], commit: str) -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine_commit": commit,
        "n_year": N_YEAR,
        "seeds": list(SEEDS),
        "columns": list(columns),
        "series": {str(s): series[s].tolist() for s in SEEDS},
    }
    FIXTURE.write_text(json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--produce", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--from-npz", type=Path)
    a = ap.parse_args(argv)
    if a.from_npz:
        z = np.load(a.from_npz)
        cols = [str(c) for c in z["columns"]]
        commit = a.from_npz.name.split("_")[2]  # baltic_master_<commit>_50yr_5seeds.npz
        write_fixture({s: z[f"seed{s}"] for s in SEEDS}, cols, commit)
        print(f"wrote {FIXTURE} from {a.from_npz} (commit {commit})")
        return 0
    if a.produce:
        series, cols = {}, None
        for s in SEEDS:
            df = run_seed(s)
            cols = cols or [c for c in df.columns if c not in ("Time", "species")]
            series[s] = df[cols].to_numpy(dtype=np.float64)
        write_fixture(series, cols, _engine_commit())
        print(f"wrote {FIXTURE} at {_engine_commit()}")
        return 0
    fx = load_gate_a_fixture()
    bad = {s: check_against_fixture(fx, s, run_seed(s)) for s in SEEDS}
    ok = all(not v for v in bad.values())
    print(
        f"Gate A vs fixture {fx['engine_commit']}: {'IDENTICAL' if ok else 'DIFFERS ' + str(bad)}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
