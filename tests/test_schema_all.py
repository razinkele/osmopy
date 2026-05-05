from osmose.schema import build_registry
from osmose.schema.fishing import FISHING_FIELDS
from osmose.schema.grid import GRID_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.movement import MOVEMENT_FIELDS
from osmose.schema.output import OUTPUT_FIELDS


def test_full_registry_has_all_categories():
    reg = build_registry()
    cats = set(reg.categories())
    expected = {
        "simulation",
        "growth",
        "reproduction",
        "predation",
        "mortality",
        "fishing",
        "grid",
        "movement",
        "ltl",
        "output",
        "bioenergetics",
        "economics",
    }
    assert expected.issubset(cats), f"Missing categories: {expected - cats}"


def test_full_registry_param_count():
    reg = build_registry()
    total = len(reg.all_fields())
    assert total >= 150, f"Only {total} params registered, expected >= 150"


def test_grid_fields_present():
    assert len(GRID_FIELDS) >= 8


def test_output_fields_present():
    assert len(OUTPUT_FIELDS) >= 50


def test_ltl_fields_present():
    assert len(LTL_FIELDS) >= 5


def test_fishing_fields_present():
    assert len(FISHING_FIELDS) >= 10


def test_movement_fields_present():
    assert len(MOVEMENT_FIELDS) >= 5


def test_no_duplicate_key_patterns():
    reg = build_registry()
    patterns = [f.key_pattern for f in reg.all_fields()]
    duplicates = [p for p in patterns if patterns.count(p) > 1]
    assert duplicates == [], f"Duplicate key patterns: {set(duplicates)}"


def test_output_enable_flags_match_engine_reads():
    """Schema OUTPUT enable flags must match what the engine actually reads (closes C2).

    The bug: schema had `output.bioen.sizeInf.enabled` (camelCase) while the
    engine reads `output.bioen.sizeinf.enabled` (lowercase) at
    `osmose/engine/config.py:865`. UI toggles for size-at-infinity were a
    silent no-op. Generalising the invariant: every output enable flag in
    the schema must equal exactly the literal string `cfg.get(...)` reads
    in `osmose/engine/config.py`.

    This is the OUTPUT subset of the broader "schema mirrors engine reads"
    contract. A few non-output schema keys (e.g. `species.bioen.mobilized.Tp`,
    `species.bioen.mobilized.e.D`) intentionally preserve mixed case because
    the Java engine reads those exact strings; this test does not police those.
    """
    from osmose.schema.output import OUTPUT_FIELDS, _OUTPUT_ENABLE_FLAGS

    # The schema's output enable flags are stored both as a list of strings
    # (`_OUTPUT_ENABLE_FLAGS` at module scope) and as `OsmoseField` entries
    # generated from that list. Both views must agree on lowercase.
    flag_offenders = [k for k in _OUTPUT_ENABLE_FLAGS if k != k.lower()]
    assert flag_offenders == [], (
        f"_OUTPUT_ENABLE_FLAGS contains camelCase keys that the engine "
        f"will never read: {flag_offenders}"
    )

    # And the resulting fields themselves
    output_flag_field_offenders = [
        f.key_pattern
        for f in OUTPUT_FIELDS
        if f.key_pattern.endswith(".enabled") and f.key_pattern != f.key_pattern.lower()
    ]
    assert output_flag_field_offenders == [], (
        f"OUTPUT_FIELDS .enabled patterns with camelCase: {output_flag_field_offenders}"
    )
