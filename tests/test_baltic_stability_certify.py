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

    with pytest.raises(Exception) as exc:  # noqa: PT011 — message is what matters here
        certify.java_table_from_series(series)
    assert "codeast" in str(exc.value) or "cod_east" in str(exc.value), (
        f"error must name the missing species/column, got: {exc.value}"
    )


def test_java_table_covers_every_focal_species(certify):
    series = _full_series(certify)
    table = certify.java_table_from_series(series)
    assert set(table) == set(certify.FOCAL)
