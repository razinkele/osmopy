"""Unit tests for osmose.engine.config_validation (Phase 7.3).

Mode semantics:
  - off    : silent on clean configs; one info-line count nudge if unknowns present
  - warn   : per-key warning with difflib suggestion (pattern-form), no raise
  - error  : collect all unknowns, raise single ValueError

All log assertions use pytest's caplog against logger 'osmose.config'
(osmose/logging.py:7 is a stdlib logging.getLogger wrapper).
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap

import pytest

from osmose.engine.config_validation import (
    KnownKeys,
    _extract_literal_keys_from_config_py,
    _normalize_key_to_pattern,
    build_known_keys,
    validate,
)

# --- Mode-behavior tests --------------------------------------------------


def test_known_key_no_warning(caplog):
    cfg = {"simulation.nspecies": "3"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []
    assert caplog.records == []


def test_unknown_key_warns_in_warn_mode(caplog):
    cfg = {"species.liinf.sp0": "30.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert len(result) == 1
    assert result[0].key == "species.liinf.sp0"
    assert result[0].suggestion == "species.linf.sp{idx}"
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warn_records) == 1
    assert "species.liinf.sp0" in warn_records[0].message
    assert "species.linf.sp{idx}" in warn_records[0].message


def test_unknown_key_raises_in_error_mode_single():
    cfg = {"species.liinf.sp0": "30.0"}
    with pytest.raises(ValueError) as exc_info:
        validate(cfg, "error")
    msg = str(exc_info.value)
    assert "species.liinf.sp0" in msg
    assert "species.linf.sp{idx}" in msg


def test_error_mode_collects_all_unknowns():
    """Spec invariant: error mode collects ALL unknowns before raising."""
    cfg = {
        "species.liinf.sp0": "30.0",
        "predation.zzzbogus.sp0": "1.0",
    }
    with pytest.raises(ValueError) as exc_info:
        validate(cfg, "error")
    msg = str(exc_info.value)
    assert "species.liinf.sp0" in msg
    assert "predation.zzzbogus.sp0" in msg


def test_unknown_key_nudge_in_off_mode(caplog):
    cfg = {"species.liinf.sp0": "30.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "off")
    assert len(result) == 1
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert re.search(r"Config has \d+ unknown keys", info_records[0].message)
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warn_records == []


def test_off_mode_zero_unknowns_silent(caplog):
    """off + no unknowns = zero log output."""
    cfg = {"simulation.nspecies": "3", "simulation.time.nyear": "10"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "off")
    assert result == []
    assert caplog.records == []


def test_close_match_suggestion_cutoff(caplog):
    """A key with no close match produces a warning without 'did you mean'."""
    cfg = {"species.totallywrongthing.sp0": "1.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert len(result) == 1
    assert result[0].suggestion is None
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warn_records) == 1
    assert "did you mean" not in warn_records[0].message.lower()


def test_idx_pattern_accepts_concrete_key(caplog):
    """A concrete species-index key matches the {idx} pattern."""
    cfg = {"species.linf.sp47": "30.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []


def test_normalize_collapses_multi_index_segments():
    """Multi-index keys normalize segment-by-segment."""
    assert (
        _normalize_key_to_pattern("fisheries.seasonality.fsh0.sp3")
        == "fisheries.seasonality.fsh{idx}.sp{idx}"
    )
    assert (
        _normalize_key_to_pattern("species.linf.sp47")
        == "species.linf.sp{idx}"
    )
    assert (
        _normalize_key_to_pattern("simulation.nspecies")
        == "simulation.nspecies"
    )
    assert (
        _normalize_key_to_pattern("fisheries.seasonality.fshX")
        == "fisheries.seasonality.fshX"
    )


def test_internal_marker_not_warned(caplog):
    """_osmose.config.dir is used as a cfg.get literal in config.py."""
    cfg = {"_osmose.config.dir": "/tmp/cfg"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []


def test_non_string_value_still_validates_key(caplog):
    """validate() iterates cfg.keys() — value type is ignored."""
    cfg = {"simulation.nspecies": 3}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []


def test_invalid_mode_raises():
    """Case-sensitive mode match — typo in flag value is a startup bug."""
    cfg = {"simulation.nspecies": "3"}
    with pytest.raises(ValueError, match=r"validation\.strict\.enabled"):
        validate(cfg, "verbose")
    with pytest.raises(ValueError, match=r"validation\.strict\.enabled"):
        validate(cfg, "OFF")
    with pytest.raises(ValueError, match=r"validation\.strict\.enabled"):
        validate(cfg, "true")


# --- AST walker tests -----------------------------------------------------


def test_ast_extracts_cfg_get_literals_from_fixture(tmp_path):
    """Synthetic AST input exercising each documented extraction shape."""
    source = textwrap.dedent(
        """
        def f(cfg):
            a = cfg.get("x.y")
            b = cfg["a.b"]
            c = _enabled(cfg, "p.q")
            d = _species_float(cfg, "species.x.sp{i}", 3)
            e = _species_float_optional(cfg, "species.z.sp{i}", 3, None)
            g = "bioen.enabled" in cfg
            h = cfg.get("has.default", "default_value")
        """
    )
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    assert "x.y" in keys
    assert "a.b" in keys
    assert "p.q" in keys
    assert "species.x.sp{idx}" in keys
    assert "species.z.sp{idx}" in keys
    assert "bioen.enabled" in keys
    assert "has.default" in keys


def test_ast_extracts_from_real_config_py_canary():
    """Canary: the walker finds known-present sentinels in the real config.py."""
    import osmose.engine.config as cfg_mod
    import ast as _ast
    tree = _ast.parse(open(cfg_mod.__file__).read())
    keys = _extract_literal_keys_from_config_py(tree)
    for sentinel in (
        "_osmose.config.dir",
        "simulation.nspecies",
        "simulation.time.ndtperyear",
    ):
        assert sentinel in keys, f"walker lost {sentinel!r}"


def test_ast_handles_fstring_with_name_interpolation():
    """f-string with a simple Name interpolation becomes {idx}."""
    source = 'cfg.get(f"mortality.{variant}.sp{i}")'
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    assert "mortality.{idx}.sp{idx}" in keys


def test_ast_drops_fstring_with_complex_interpolation():
    """f-string with Attribute/Subscript/Call interpolation is skipped."""
    source = 'cfg.get(f"x.{obj.attr}")'
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    assert not any("obj.attr" in k for k in keys)
    assert "x.{idx}" not in keys


def test_ast_handles_in_comparison():
    """'key' in cfg membership test extracts the literal."""
    source = '"bioen.enabled" in cfg'
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    assert "bioen.enabled" in keys


def test_ast_extracts_assign_rhs_fstring():
    """Keys stored via `*_key = f"..."` patterns must be captured."""
    source = textwrap.dedent(
        """
        def f(cfg, variant, i, fsh_idx):
            key = f"mortality.fishing.rate.{variant}.file.sp{i}"
            val = cfg.get(key, "")

            a50_key = f"fisheries.selectivity.a50.fsh{fsh_idx}"
            if a50_key in cfg:
                pass

            msg = f"Invalid fisheries.seasonality.fsh{fsh_idx} value: bad"
        """
    )
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    assert "mortality.fishing.rate.{idx}.file.sp{idx}" in keys
    assert "fisheries.selectivity.a50.fsh{idx}" in keys
    assert not any("Invalid" in k for k in keys)


# --- build_known_keys + fallback ------------------------------------------


def test_build_known_keys_has_literal_fast_path():
    """build_known_keys returns KnownKeys with literal + regex views."""
    from osmose.engine.config_validation import _clear_known_keys_cache
    _clear_known_keys_cache()
    kk = build_known_keys()
    assert isinstance(kk, KnownKeys)
    assert len(kk.literals) > 100, f"literals set too small: {len(kk.literals)}"
    assert "_osmose.config.dir" in kk.literals
    assert "validation.strict.enabled" in kk.literals


def test_source_file_unavailable_falls_back_to_schema_only(monkeypatch):
    """AST source load failure → schema-only fallback, NOT cached."""
    import osmose.engine.config_validation as cv_mod
    from osmose.engine.config_validation import _clear_known_keys_cache

    _clear_known_keys_cache()

    def _fake_read_source():
        raise FileNotFoundError("simulated missing source")

    monkeypatch.setattr(cv_mod, "_read_config_source", _fake_read_source)
    try:
        kk = build_known_keys()
        assert isinstance(kk, KnownKeys)
        assert "validation.strict.enabled" in kk.literals
        assert "full" not in cv_mod._KNOWN_KEYS_CACHE
    finally:
        _clear_known_keys_cache()
