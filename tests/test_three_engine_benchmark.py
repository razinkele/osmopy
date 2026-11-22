"""The README's three-engine performance table is the recorded measurement, not typed numbers.

The table in ``README.md`` (which is also the app's About text) was measured on 2026-11-22 with
``scripts/benchmark_three_engines.py``; the raw records are in ``docs/perf``. Every cell is
recomputed here from those records, so a figure edited by hand, or a record replaced without
the table, fails. The harness's own small parts are checked too.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from scripts import benchmark_three_engines as bench

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RECORD_70 = ROOT / "docs" / "perf" / "2026-11-22-three-engines-eec_full.json"
RECORD_5 = ROOT / "docs" / "perf" / "2026-11-22-three-engines-eec_full-5yr.json"

#: README row label -> the arm's name in the records.
ARMS = {
    ("Java 4.4.1", "no setting"): "java",
    ("Python", "1"): "python@1",
    ("Python", "14"): "python@14",
    ("C#", "1"): "csharp@1",
    ("C#", "14"): "csharp@14",
}


def _median(record: dict, arm: str, key: str) -> float:
    return record["arms"][arm]["summary"][key]["median"]


def _seconds(value: float) -> str:
    """The README's rule: one decimal below 10 s, whole seconds from there."""
    return f"{value:.1f} s" if value < 10 else f"{value:.0f} s"


def _times(value: float) -> str:
    return f"{value:.1f}×" if value < 10 else f"{value:.0f}×"


def _readme_rows() -> dict[tuple[str, str], list[str]]:
    text = README.read_text()
    section = text[text.index("## Performance") : text.index("## Engine parity validation")]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if cells[0] == "Engine":
            assert cells[1:] == [
                "Threads",
                "5-year run",
                "70-year run",
                "Speed vs Java (70 years)",
                "Peak memory (70 years)",
            ]
            continue
        rows[(cells[0], cells[1])] = cells[2:]
    return rows


def test_the_readme_table_has_exactly_the_measured_arms():
    assert set(_readme_rows()) == set(ARMS)


@pytest.mark.parametrize("label", sorted(ARMS))
def test_every_cell_of_the_readme_table_is_the_recorded_median(label):
    five, seventy = json.loads(RECORD_5.read_text()), json.loads(RECORD_70.read_text())
    assert (five["years"], seventy["years"]) == (5, 70)
    assert five["config"] == seventy["config"] == "data/eec_full/eec_all-parameters.csv"
    arm = ARMS[label]
    java = _median(seventy, "java", "wall_s")
    expected = [
        _seconds(_median(five, arm, "wall_s")),
        _seconds(_median(seventy, arm, "wall_s")),
        _times(java / _median(seventy, arm, "wall_s")),
        f"{_median(seventy, arm, 'peak_rss_mb'):.0f} MB",
    ]
    assert _readme_rows()[label] == expected


def test_the_records_are_five_clean_measured_rounds_per_arm():
    for path in (RECORD_5, RECORD_70):
        record = json.loads(path.read_text())
        assert record["rounds"] == 5 and record["warmup_rounds"] == 1
        assert set(record["arms"]) == set(ARMS.values())
        for name, arm in record["arms"].items():
            measured = [r for r in arm["runs"] if not r["warmup"]]
            assert len(measured) == 5, name
            # Each run finished and left one biomass row per simulated year.
            assert all(r["ok"] and r["biomass_rows"] == record["years"] for r in measured), name
            walls = sorted(r["wall_s"] for r in measured)
            assert arm["summary"]["wall_s"] == {
                "median": walls[2],
                "min": walls[0],
                "max": walls[-1],
            }


def test_the_records_name_no_local_path():
    for path in (RECORD_5, RECORD_70):
        assert "/home/" not in path.read_text()


def test_the_speed_row_above_the_table_quotes_the_same_measurement():
    seventy = json.loads(RECORD_70.read_text())
    java = _median(seventy, "java", "wall_s")
    one = _times(java / _median(seventy, "python@1", "wall_s"))
    many = _times(java / _median(seventy, "python@14", "wall_s"))
    assert f"| Speed | {one} faster than Java on one thread, {many} on 14 " in README.read_text()


# ------------------------------------------------------------------ the harness itself


def test_outputs_counts_one_biomass_row_per_saved_step(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "eec_biomass_Simu0.csv").write_text(
        "Biomass (tons), excluding first ages\nTime,cod,sole\n1.0,5,6\n2.0,7,8\n3.0,9,1\n"
    )
    (out / "eec_yield_Simu0.csv").write_text("x\n")
    got = bench._outputs(out, years=3, steps_per_year=24, record_every=24)
    assert got == {
        "files": 2,
        "bytes": sum(p.stat().st_size for p in out.iterdir()),
        "biomass_rows": 3,
        "expected_rows": 3,
    }
    # Recording twice a year doubles the rows a complete run must have.
    assert bench._outputs(out, 3, 24, 12)["expected_rows"] == 6


def test_outputs_without_a_biomass_file_has_no_row_count(tmp_path):
    assert bench._outputs(tmp_path, 1, 24, 24)["biomass_rows"] is None


def test_summary_is_the_median_and_the_range():
    runs = [
        {"wall_s": w, "cpu_s": 2 * w, "cores_used": 2.0, "peak_rss_mb": 100.0 + w}
        for w in (5.0, 1.0, 3.0)
    ]
    summary = bench._summary(runs)
    assert summary["wall_s"] == {"median": 3.0, "min": 1.0, "max": 5.0}
    assert summary["peak_rss_mb"] == {"median": 103.0, "min": 101.0, "max": 105.0}


def test_thread_env_caps_numba_and_the_blas_pools():
    env = bench._thread_env(3)
    for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert env[name] == "3"


def test_the_parent_never_imports_the_dotnet_bridge():
    """A process that has loaded the .NET runtime must not fork, and the parent starts every
    run. The C# engine may be imported only inside ``child``, which runs in its own process."""
    tree = ast.parse(Path(bench.__file__).read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "child":
            for inner in ast.walk(node):
                if isinstance(inner, ast.ImportFrom) and (inner.module or "").startswith(
                    "osmose.engine_cs"
                ):
                    offenders.append((node.name, inner.module))
    top = [
        n.module
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("osmose.engine_cs")
    ]
    assert not offenders and not top
    child = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "child")
    assert any(
        isinstance(n, ast.ImportFrom) and n.module == "osmose.engine_cs.engine"
        for n in ast.walk(child)
    )
    assert re.search(r'sys\.argv\[1\] == "--child"', Path(bench.__file__).read_text())
