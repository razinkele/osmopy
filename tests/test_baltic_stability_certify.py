"""The Java certification table must read Java's own (sanitized) species column names.

`reconcile_config_for_java` rewrites `species.name.sp*` to Java's stripped internal form, so the
Java biomass CSV header carries `codwest`/`codeast`, not `cod_west`/`cod_east`. `certify_java`
looked the raw names up and fell back to `[0.0]` on a miss, which reported both cod stocks as
extinct in two committed cross-checks while Java actually had them alive. These guards pin the
name mapping and forbid the silent-zero fallback.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_certify():
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "baltic_stability_certify", scripts_dir / "baltic_stability_certify.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["baltic_stability_certify"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certify():
    return _load_certify()


def _full_series(certify, value: float = 1.0) -> dict[str, list[float]]:
    """A biomass series keyed the way Java writes it: sanitized names, all focal species."""
    from osmose.java_config_reconcile import sanitize_java_name

    return {sanitize_java_name(sp): [value] * 12 for sp in certify.FOCAL}


def test_java_table_reads_sanitized_column_names(certify):
    """cod_west/cod_east must resolve from Java's codwest/codeast columns."""
    series = _full_series(certify)
    # Put cod_east squarely inside its envelope; the raw-name lookup would miss this entirely.
    lo, hi = certify.ENVELOPE["cod_east"]
    mid = (lo + hi) / 2
    series["codeast"] = [mid] * 12

    table = certify.java_table_from_series(series)

    assert table["cod_east"]["min_biomass"] == pytest.approx(mid)
    assert table["cod_east"]["persists"], "cod_east present in Java output must not read as extinct"
    assert table["cod_east"]["in_envelope"]


def test_java_table_does_not_manufacture_extinction_for_missing_column(certify):
    """A missing column is a harness bug; reporting 0.0 hid it for two committed cross-checks."""
    series = _full_series(certify)
    del series["codeast"]

    with pytest.raises(Exception) as exc:
        certify.java_table_from_series(series)
    assert "codeast" in str(exc.value) or "cod_east" in str(exc.value), (
        f"error must name the missing species/column, got: {exc.value}"
    )


def test_java_table_covers_every_focal_species(certify):
    series = _full_series(certify)
    table = certify.java_table_from_series(series)
    assert set(table) == set(certify.FOCAL)


# ---------------------------------------------------------------- the engine option (C# 2.8)


def _table(certify, **overrides) -> dict:
    """A certification table in which every species passes, with ``overrides`` per species."""
    table = {}
    for sp in certify.FOCAL:
        lo, hi = certify.ENVELOPE[sp]
        mid = (lo + hi) / 2
        table[sp] = {
            "persists": True,
            "in_envelope": True,
            "min_biomass": mid,
            "late_mean_range": [mid, mid],
        }
        table[sp].update(overrides.get(sp, {}))
    return table


def test_tables_with_the_same_verdicts_have_no_differences(certify):
    a = _table(certify)
    b = _table(certify, herring={"late_mean_range": [9e5, 1e6]})  # other numbers, same verdicts
    assert certify.compare_verdicts(a, b) == []
    (line,) = certify.cross_engine_lines("Python", "C#", [])
    assert f"all {2 * len(certify.FOCAL)} verdicts agree with Python's" in line


def test_every_differing_verdict_is_listed_with_both_ranges(certify):
    a = _table(certify)
    b = _table(
        certify,
        cod_east={"in_envelope": False, "late_mean_range": [5.5e4, 5.9e4]},
        sprat={"persists": False, "in_envelope": False, "late_mean_range": [0.0, 10.0]},
    )
    differences = certify.compare_verdicts(a, b)
    assert [(d["species"], d["criterion"]) for d in differences] == [
        ("cod_east", "in_envelope"),
        ("sprat", "persists"),
        ("sprat", "in_envelope"),
    ]
    east = differences[0]
    assert east["reference"] is True and east["other"] is False
    assert east["other_range"] == [5.5e4, 5.9e4] and east["envelope"] == (60000, 85000)
    lines = certify.cross_engine_lines("Python", "C#", differences)
    assert f"3 of {2 * len(certify.FOCAL)} verdicts DIFFER from Python's" in lines[0]
    assert sum(line.startswith(("| cod_east ", "| sprat ")) for line in lines) == 3
    assert certify.compare_verdicts(b, a)[0]["reference"] is False  # the direction is kept


@pytest.mark.parametrize("engine", ["python", "csharp"])
def test_the_certification_runs_on_the_named_engine(certify, engine, monkeypatch):
    """Same config, same seeds, same criterion: only the engine differs, and it is the one
    named. The C# engine is given a thread count; the Python engine none."""
    import pandas as pd

    seen: dict = {"seeds": []}

    class Fake:
        def run_in_memory(self, cfg, seed=0):
            seen["seeds"].append(seed)
            seen["nyear"] = cfg["simulation.time.nyear"]
            frame = pd.DataFrame({sp: [sum(certify.ENVELOPE[sp]) / 2] * 12 for sp in certify.FOCAL})
            return type("R", (), {"biomass": lambda self: frame})()

    def get_engine(name, **kwargs):
        seen.update(name=name, kwargs=kwargs)
        return Fake()

    monkeypatch.setattr(certify, "get_engine", get_engine)
    table = certify.certify_python({}, 12, (3, 4), engine=engine)
    assert seen["name"] == engine and seen["seeds"] == [3, 4] and seen["nyear"] == "12"
    assert ("threads" in seen["kwargs"]) is (engine == "csharp")
    assert all(table[sp]["persists"] and table[sp]["in_envelope"] for sp in certify.FOCAL)
    assert certify.ENGINE_LABELS == {"python": "Python", "csharp": "C#"}
