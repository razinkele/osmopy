"""Schema-vs-engine key parity (closes C1).

Every `OsmoseField.key_pattern` declared in the schema must be a key the
engine's validation allowlist accepts. Otherwise a UI write of that key
gets flagged as unknown — meaning the UI input was never reaching the
engine in the first place.

The C1 regression that triggered this test: the movement schema had keys
shaped like `movement.map{idx}.species`, but the engine reads
`movement.species.map{idx}`. Schema writes were silently ignored.
"""

import pytest

from osmose.engine.config_validation import (
    _SUPPLEMENTARY_ALLOWLIST,
    _extract_literal_keys_from_config_py,
    _read_config_source,
    _read_extra_engine_sources,
)
from osmose.schema import build_registry


def _allowed_keys() -> set[str]:
    """Mirror EngineConfig.from_dict's accept-set: AST-walked literals + allowlist."""
    import ast

    accept = _extract_literal_keys_from_config_py(ast.parse(_read_config_source()))
    for extra in _read_extra_engine_sources().values():
        accept |= _extract_literal_keys_from_config_py(ast.parse(extra))
    return accept | set(_SUPPLEMENTARY_ALLOWLIST)


def test_every_schema_key_is_engine_accepted():
    """Every schema field must produce a key the engine validator accepts.

    Closed 2026-05-08: extended the AST walker to scan simulate.py +
    __init__.py (catches grid + temperature/oxygen reads), then bulk-
    allowlisted ~90 Java-side schema-only fields under a clearly-labelled
    block in `_SUPPLEMENTARY_ALLOWLIST`. The parity invariant — every
    schema key is acceptable to the engine validator — is now upheld
    on master; this test is the regression gate. If a new schema field
    appears whose key the engine doesn't recognise, this test will name
    it explicitly.
    """
    accept = _allowed_keys()
    reg = build_registry()
    offenders: list[str] = []
    for field in reg.all_fields():
        if field.key_pattern not in accept:
            offenders.append(field.key_pattern)

    assert not offenders, (
        f"Schema fields whose key_pattern is unknown to the engine validator "
        f"(UI writes would silently no-op): {offenders[:10]}{'...' if len(offenders) > 10 else ''}"
    )


@pytest.mark.parametrize(
    "engine_key",
    [
        "movement.species.map{idx}",
        "movement.file.map{idx}",
        "movement.steps.map{idx}",
        "movement.initialage.map{idx}",
        "movement.lastage.map{idx}",
    ],
)
def test_movement_map_keys_are_in_schema(engine_key: str) -> None:
    """The C1-regression specific case: every key the engine reads in
    movement_maps.py must be in the schema, not just the allowlist."""
    reg = build_registry()
    schema_patterns = {f.key_pattern for f in reg.all_fields()}
    assert engine_key in schema_patterns, (
        f"{engine_key} is read by osmose/engine/movement_maps.py but is not in "
        f"the schema. UI cannot generate an input for it."
    )
