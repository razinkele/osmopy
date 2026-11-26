"""The README's three-engine performance table is the recorded measurement, not typed numbers.

The table in ``README.md`` (which is also the app's About text) was measured on 2026-11-26 with
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
RECORD_70 = ROOT / "docs" / "perf" / "2026-11-26-three-engines-eec_full.json"
RECORD_5 = ROOT / "docs" / "perf" / "2026-11-26-three-engines-eec_full-5yr.json"

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


EEC_HEADER = [
    "Engine",
    "Threads",
    "5-year run",
    "70-year run",
    "Speed vs Java (70 years)",
    "Peak memory (70 years)",
]
BALTIC_HEADER = ["Engine", "Threads", "15-year run", "Speed vs Python, 1 thread", "Peak memory"]


def _tables() -> dict[tuple[str, ...], dict[tuple[str, str], list[str]]]:
    """Every table of the README's Performance section, keyed by its header row; each table's
    rows keyed by (engine, threads)."""
    text = README.read_text(encoding="utf-8")
    section = text[text.index("## Performance") : text.index("## Engine parity validation")]
    tables: dict[tuple[str, ...], dict[tuple[str, str], list[str]]] = {}
    current: dict[tuple[str, str], list[str]] | None = None
    for line in section.splitlines():
        if not line.startswith("|"):
            current = None  # a table ends at the first line that is not a table row
            continue
        if set(line) <= set("|- "):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if current is None:
            current = tables.setdefault(tuple(cells), {})
            continue
        current[(cells[0], cells[1])] = cells[2:]
    return tables


def _readme_rows() -> dict[tuple[str, str], list[str]]:
    """The three-engine EEC table."""
    tables = _tables()
    assert tuple(EEC_HEADER) in tables, list(tables)
    return tables[tuple(EEC_HEADER)]


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
    readme = README.read_text(encoding="utf-8")
    assert f"| Speed | {one} faster than Java on one thread, {many} on 14 " in readme


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


def test_a_partial_last_window_is_a_row_too(tmp_path):
    # Both in-process engines write the window the run ends inside, complete or not: 12 steps
    # saved every 5 give rows at steps 5, 10 and 12 (measured 2026-11-22 on data/minimal).
    assert (
        bench._outputs(tmp_path, years=1, steps_per_year=12, record_every=5)["expected_rows"] == 3
    )


def test_outputs_without_a_biomass_file_has_no_row_count(tmp_path):
    assert bench._outputs(tmp_path, 1, 24, 24)["biomass_rows"] is None


@pytest.mark.parametrize(
    ("run", "ok"),
    [
        ({"exit": 0, "biomass_rows": 70, "expected_rows": 70}, True),
        ({"exit": 1, "biomass_rows": 70, "expected_rows": 70}, False),
        ({"exit": 0, "biomass_rows": 69, "expected_rows": 70}, False),
        # No output directory at all: no counts, and that is a failure, not a match.
        ({"exit": 0}, False),
        ({"exit": 0, "biomass_rows": None, "expected_rows": 3}, False),
        # A run that saves nothing, by design, is still a run whose count matches.
        ({"exit": 0, "biomass_rows": 0, "expected_rows": 0}, True),
    ],
)
def test_a_run_is_clean_only_with_its_outputs(run, ok):
    assert bench._run_ok(run) is ok


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


# ------------------------------------------------------------- the Baltic table (Python and C#)

RECORD_BALTIC = ROOT / "docs" / "perf" / "2026-11-22-python-csharp-baltic.json"
BALTIC_ARMS = {
    ("Python", "1"): "python@1",
    ("Python", "14"): "python@14",
    ("C#", "1"): "csharp@1",
    ("C#", "14"): "csharp@14",
}


def test_the_readme_has_exactly_two_performance_tables():
    assert set(_tables()) == {tuple(EEC_HEADER), tuple(BALTIC_HEADER)}


def test_every_cell_of_the_baltic_table_is_the_recorded_median():
    record = json.loads(RECORD_BALTIC.read_text())
    assert record["years"] == 15
    assert record["config"] == "data/baltic/baltic_all-parameters.csv"
    rows = _tables()[tuple(BALTIC_HEADER)]
    assert set(rows) == set(BALTIC_ARMS)
    python_one = _median(record, "python@1", "wall_s")
    for label, arm in BALTIC_ARMS.items():
        wall = _median(record, arm, "wall_s")
        assert rows[label] == [
            _seconds(wall),
            _times(python_one / wall),
            f"{_median(record, arm, 'peak_rss_mb'):.0f} MB",
        ], label


def test_the_baltic_record_is_five_clean_measured_rounds_per_arm():
    record = json.loads(RECORD_BALTIC.read_text())
    assert record["rounds"] == 5 and record["warmup_rounds"] == 1
    assert set(record["arms"]) == set(BALTIC_ARMS.values())  # no Java arm: Java refuses it
    for name, arm in record["arms"].items():
        measured = [r for r in arm["runs"] if not r["warmup"]]
        assert len(measured) == 5, name
        assert all(r["ok"] and r["biomass_rows"] == 15 for r in measured), name
        walls = sorted(r["wall_s"] for r in measured)
        assert arm["summary"]["wall_s"] == {"median": walls[2], "min": walls[0], "max": walls[-1]}
    assert "/home/" not in RECORD_BALTIC.read_text()


def test_java_really_refuses_the_baltic_config():
    """The table says Java is missing because it refuses this config; keep that true."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.runner import java_engine_block_reason

    raw = OsmoseConfigReader().read(ROOT / "data" / "baltic" / "baltic_all-parameters.csv")
    reason = java_engine_block_reason(raw)
    assert reason and "oxygen" in reason
