"""`_read_mortality_rate_csv` must parse REAL Java 4.4.1 mortality output (GitHub #141).

Java's header is internally inconsistent — measured on baltic_mortalityRate-sprat_Simu0.csv:

    row 0 :  6 fields  free-text description (commas inside the sentence)
    row 1 : 25 fields  "Time" + 8 named causes x 3 stages = 24
    row 2 : 28 fields  blank + NINE stage-triples = 27
    row 3+: 29 fields  time + 27 values + trailing comma

So Java emits 27 data columns but names only 24 — one stage-triple is unnamed. `pd.read_csv(...,
header=[0, 1])` requires equal-width header rows and raises ParserError, so no Python-side code could
read Java mortality output at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from osmose.results import _read_mortality_rate_csv

CAUSES = ["Mpred", "Mstarv", "Madd", "F", "Zout", "Mfor", "Mdis", "Mage"]
STAGES = ["Eggs", "Juvenil", "Adult"]


def _write_java_layout(path):
    """Reproduce Java's exact ragged widths: 25 / 28 / 29 fields."""
    desc = '"Predation (Mpred), Starvation (Mstarv), Additional mortality (Madd), Fishing (F) & Out-of-domain (Zout) mortality rates per time step of saving."'
    cause_row = ",".join(['"Time"'] + [f'"{c}"' for c in CAUSES for _ in STAGES])
    stage_row = ",".join([""] + STAGES * 9)  # NINE triples — one more than the cause row names
    rows = []
    for t in (1.0, 2.0):
        vals = [f"{t}"] + [f"{0.01 * (i + 1):.4f}" for i in range(27)]
        rows.append(",".join(vals) + ",")  # trailing comma
    path.write_text("\n".join([desc, cause_row, stage_row, *rows]) + "\n")
    return path


def test_parses_java_ragged_header_without_raising(tmp_path):
    p = _write_java_layout(tmp_path / "baltic_mortalityRate-sprat_Simu0.csv")
    df = _read_mortality_rate_csv(p)
    assert isinstance(df.columns, pd.MultiIndex), "expected a (cause, stage) MultiIndex"
    assert len(df) == 2, "two data rows"
    for c in CAUSES:
        for s in STAGES:
            assert (c, s) in df.columns, f"missing ({c}, {s})"


def test_values_align_with_their_cause_and_stage(tmp_path):
    p = _write_java_layout(tmp_path / "m.csv")
    df = _read_mortality_rate_csv(p)
    # values are 0.01, 0.02, ... in column order: Mpred/Eggs first
    assert df[("Mpred", "Eggs")].iloc[0] == pytest.approx(0.01)
    assert df[("Mpred", "Juvenil")].iloc[0] == pytest.approx(0.02)
    assert df[("Mstarv", "Eggs")].iloc[0] == pytest.approx(0.04)


def test_unnamed_ninth_triple_is_retained_not_dropped(tmp_path):
    """Java emits 27 data columns but names 24. The extra three carry data and must not vanish
    silently — their meaning is unidentified, so they are surfaced under a placeholder cause."""
    p = _write_java_layout(tmp_path / "m.csv")
    df = _read_mortality_rate_csv(p)
    extra = [c for c in df.columns if c[0] not in CAUSES and c[0] != "Time"]
    assert len(extra) == 3, f"expected the unnamed triple to be retained, got {extra}"


def test_python_layout_still_parses(tmp_path):
    """The engine's own two-row layout has equal-width headers and must keep working."""
    p = tmp_path / "osm_mortalityRate-sprat_Simu0.csv"
    cause_row = ",".join(["Time"] + [c for c in CAUSES for _ in STAGES])
    stage_row = ",".join([""] + STAGES * 8)
    rows = [",".join(["1.0"] + [f"{0.01 * (i + 1):.4f}" for i in range(24)])]
    p.write_text(
        "\n".join(['"Mortality rates per time step for sprat"', cause_row, stage_row, *rows]) + "\n"
    )
    df = _read_mortality_rate_csv(p)
    assert isinstance(df.columns, pd.MultiIndex)
    assert df[("Mpred", "Eggs")].iloc[0] == pytest.approx(0.01)
