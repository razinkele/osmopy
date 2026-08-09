"""Issue #123 — partition integrity + classification-correctness guards.

The read-clearance guard scans EVERY .py under osmose/engine/** (including config.py) —
a stricter test-only scan than production _EXTRA_ENGINE_SOURCES (which omits config.py).
It couples the JAVA_ONLY classification to the actual engine source, so a future edit that
either mis-buckets a key OR adds a cfg.get for a currently-java-only key turns the suite red.
"""

import ast
import logging
import pathlib

import pytest

from osmose.engine import config_validation as cv
from osmose.engine.config_validation import (
    _ALLOWLIST_JAVA_ONLY,
    _ALLOWLIST_PY_HONORED,
    _RESTART_HANDLED_BY_120,
    _SUPPLEMENTARY_ALLOWLIST,
    _compile_regex_for_pattern,
    _extract_literal_keys_from_config_py,
    java_only_keys_set,
)

# Independent reference: the exact 149-key allowlist as of pre-#123 (copied from Step 1 output).
# NOT derived from the source frozensets — that would be circular.
FROZEN_ALLOWLIST_SNAPSHOT = frozenset(
    [
        "economic.output.stage",
        "evolution.trait.{name}.envvar.sp{idx}",
        "evolution.trait.{name}.mean.sp{idx}",
        "evolution.trait.{name}.nlocus.sp{idx}",
        "evolution.trait.{name}.nval.sp{idx}",
        "evolution.trait.{name}.target",
        "evolution.trait.{name}.var.sp{idx}",
        "fisheries.check.enabled",
        "fisheries.movement.file.map{idx}",
        "fisheries.movement.fishery.map{idx}",
        "fisheries.name.fsh{idx}",
        "fisheries.period.number.fsh{idx}",
        "fisheries.period.start.fsh{idx}",
        "fisheries.rate.byperiod.fsh{idx}",
        "grid.java.classname",
        "grid.lowright.lat",
        "grid.lowright.lon",
        "grid.mask.file",
        "grid.upleft.lat",
        "grid.upleft.lon",
        "ltl.conversion2tons.rsc{idx}",
        "ltl.depletable.enabled",
        "ltl.depletable.floor",
        "ltl.java.classname",
        "ltl.nstep",
        "ltl.regrowth.rate.default",
        "ltl.regrowth.rate.rsc{idx}",
        "module.bioeconomics.enabled",
        "module.population.initialisation.enabled",
        "mortality.fishing.recruitment.age.sp{idx}",
        "mortality.fishing.recruitment.size.sp{idx}",
        "movement.species.map{idx}",
        "osmose.configuration.a2.depletion",
        "osmose.configuration.background",
        "osmose.configuration.bioen",
        "osmose.configuration.fishing",
        "osmose.configuration.genetics",
        "osmose.configuration.grid",
        "osmose.configuration.initialization",
        "osmose.configuration.ltl",
        "osmose.configuration.migration",
        "osmose.configuration.mortality.additional",
        "osmose.configuration.mortality.fishing",
        "osmose.configuration.mortality.predation",
        "osmose.configuration.mortality.starvation",
        "osmose.configuration.movement",
        "osmose.configuration.output",
        "osmose.configuration.oxygen",
        "osmose.configuration.plankton",
        "osmose.configuration.predation",
        "osmose.configuration.reproduction",
        "osmose.configuration.simulation",
        "osmose.configuration.species",
        "osmose.version",
        "output.abundance.age1.enabled",
        "output.abundance.bytl.enabled",
        "output.abundance.byweight.enabled",
        "output.abundance.enabled",
        "output.age.at.death.enabled",
        "output.bioen.enet.enabled",
        "output.biomass.bytl.enabled",
        "output.biomass.byweight.enabled",
        "output.biomass.enabled",
        "output.csv.separator",
        "output.cutoff.enabled",
        "output.diet.composition.byage.enabled",
        "output.diet.composition.bysize.enabled",
        "output.diet.composition.netcdf.enabled",
        "output.diet.pressure.byage.enabled",
        "output.diet.pressure.bysize.enabled",
        "output.diet.pressure.enabled",
        "output.diet.pressure.netcdf.enabled",
        "output.diet.stage.structure",
        "output.diet.stage.threshold.sp{idx}",
        "output.diet.success.enabled",
        "output.dir.path",
        "output.file.prefix",
        "output.fisheries.byage.enabled",
        "output.fisheries.bysize.enabled",
        "output.fisheries.enabled",
        "output.flush.enabled",
        "output.meansize.byage.enabled",
        "output.meantl.byage.enabled",
        "output.meantl.bysize.enabled",
        "output.meanweight.byage.enabled",
        "output.mortality.additional.byage.enabled",
        "output.mortality.additional.bysize.enabled",
        "output.mortality.additionaln.byage.enabled",
        "output.mortality.additionaln.bysize.enabled",
        "output.mortality.enabled",
        "output.mortality.perspecies.byage.enabled",
        "output.mortality.perspecies.bysize.enabled",
        "output.nschool.enabled",
        "output.number.of.eggs.bysize.enabled",
        "output.restart.recordfrequency.ndt",
        "output.restart.spinup",
        "output.size.catch.enabled",
        "output.size.enabled",
        "output.spatial.egg.enabled",
        "output.spatial.fisheries.enabled",
        "output.spatial.ltl.enabled",
        "output.spatial.size.enabled",
        "output.spatial.yield.abundance.enabled",
        "output.ssb.enabled",
        "output.ssb.netcdf.enabled",
        "output.start.year",
        "output.tl.catch.enabled",
        "output.tl.enabled",
        "output.weight.enabled",
        "output.yield.abundance.byage.enabled",
        "output.yield.abundance.bysize.enabled",
        "output.yield.abundance.enabled",
        "output.yield.abundance.netcdf.enabled",
        "output.yield.biomass.byage.enabled",
        "output.yield.biomass.bysize.enabled",
        "output.yield.biomass.enabled",
        "oxygen.factor",
        "oxygen.filename",
        "oxygen.nsteps.year",
        "oxygen.offset",
        "oxygen.varname",
        "population.initialization.method.sp{idx}",
        "predation.accessibility.stage.structure",
        "predation.accessibility.stage.threshold.sp{idx}",
        "simulation.incoming.flux.enabled",
        "simulation.ncpu",
        "simulation.nsimulation",
        "simulation.restart.enabled",
        "simulation.restart.file",
        "simulation.restart.recordfrequency.ndt",
        "simulation.restart.spinup.nyear",
        "species.age.sp{idx}",
        "species.biomass.nsteps.year",
        "species.biomass.total.sp{idx}",
        "species.conversion2tons.sp{idx}",
        "species.first.feeding.age.sp{idx}",
        "species.length.sp{idx}",
        "species.lw.allpower.sp{idx}",
        "species.lw.condition.factor.sp{idx}",
        "species.nclass.sp{idx}",
        "species.regrowth.rate.sp{idx}",
        "species.size.proportion.sp{idx}",
        "species.tl.sp{idx}",
        "species.trophic.level.sp{idx}",
        "species.type.sp{idx}",
        "temperature.factor",
        "temperature.filename",
        "temperature.nsteps.year",
        "temperature.offset",
        "temperature.varname",
    ]
)

# Engine-honored dynamic prefixes (verified by grep: movement_maps.py:129, resources.py:143/97).
_STARTSWITH_PREFIXES = ("movement.species.map", "species.type.sp", "ltl.name.rsc")

# AST-INVISIBLE reads (membership / regex-on-iterated-key) that must be PY_HONORED. The read
# scan cannot see these; each carries its read site. Closed at these two families (see spec).
_MEMBERSHIP_EXCLUSIONS = frozenset(
    [
        "species.biomass.total.sp{idx}",  # background.py:329  (total_key in config)
        "evolution.trait.{name}.target",  # config.py:1571 re.match / genetics/trait.py:54
        "evolution.trait.{name}.mean.sp{idx}",  # genetics/trait.py:59-60
        "evolution.trait.{name}.var.sp{idx}",
        "evolution.trait.{name}.envvar.sp{idx}",
        "evolution.trait.{name}.nlocus.sp{idx}",
        "evolution.trait.{name}.nval.sp{idx}",
    ]
)

# LEGACY ALIASES of keys the Python engine reads under their CANONICAL spelling. Genuinely unread
# under their own name (so the read-clearance guard can't rescue them), but the feature IS
# implemented on the Python engine — warning "use the Java engine" would misdirect. Must be
# PY_HONORED (no #123 warning). NOTE: conversion2tons is NOT here — its canonical
# resource.conversion2tons is read nowhere, so it is genuinely inert and correctly JAVA_ONLY.
_LEGACY_ALIAS_HONORED = frozenset(
    [
        "species.lw.condition.factor.sp{idx}",  # canonical species.length2weight.condition.factor.sp -> config.py:467
        "species.lw.allpower.sp{idx}",  # canonical species.length2weight.allometric.power.sp -> config.py:468
        "species.tl.sp{idx}",  # canonical species.trophic.level.sp -> background.py:196
    ]
)


def _scan_engine_reads() -> set[str]:
    """Every literal/f-string config key read across osmose/engine/** (incl. config.py).

    _extract_literal_keys_from_config_py returns a MIX: concrete literals from subscript/literal
    cfg.get ('fisheries.movement.file.map0') AND {idx}-pattern forms from f-strings
    (cfg.get(f'ltl.regrowth.rate.rsc{i}') -> 'ltl.regrowth.rate.rsc{idx}').
    """
    reads: set[str] = set()
    engine_root = pathlib.Path(__file__).resolve().parent.parent / "osmose" / "engine"
    for py in engine_root.rglob("*.py"):
        try:
            reads |= _extract_literal_keys_from_config_py(ast.parse(py.read_text(encoding="utf-8")))
        except SyntaxError:
            pass
    return reads


def _is_engine_read(pattern: str, reads: set[str]) -> bool:
    """True iff the Python engine reads any key matching `pattern`."""
    if pattern in reads:  # {idx}-form direct equality: catches f-string reads
        return True
    rx = _compile_regex_for_pattern(pattern)  # {idx}->\d+, {name}->\w+
    if any(rx.match(lit) for lit in reads if "{" not in lit):  # concrete literals: map0, etc.
        return True
    base = pattern.split("{")[0]
    return any(base.startswith(p) or p.startswith(base) for p in _STARTSWITH_PREFIXES)


def test_partition_completeness_against_frozen_snapshot():
    # union == independent snapshot (catches a dropped/added key during the split); disjoint.
    assert _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY == FROZEN_ALLOWLIST_SNAPSHOT
    assert _ALLOWLIST_PY_HONORED & _ALLOWLIST_JAVA_ONLY == frozenset()
    # source name preserved as the union (build_known_keys unchanged).
    assert _SUPPLEMENTARY_ALLOWLIST == FROZEN_ALLOWLIST_SNAPSHOT


def test_read_clearance_no_java_only_key_is_read():
    reads = _scan_engine_reads()
    # _RESTART_HANDLED_BY_120 (simulation.restart.file/.enabled) are JAVA_ONLY *and* read by config.py
    # — but only by #120's restart WARNING (read-to-warn), NOT honored/implemented. They are
    # deliberately kept JAVA_ONLY (the Python engine does not implement restart) and carved out of
    # #123's summary. Exclude them from the read-clearance check, like the membership families, so the
    # guard doesn't demand reclassifying a key #120 intentionally reads to warn on. (Surfaced when
    # #120 + #123 landed together on master; #126's CI ran before #120 was in its base.)
    offenders = [
        p for p in (_ALLOWLIST_JAVA_ONLY - _RESTART_HANDLED_BY_120) if _is_engine_read(p, reads)
    ]
    assert offenders == [], (
        f"JAVA_ONLY keys the engine actually reads (reclassify PY_HONORED): {offenders}"
    )


def test_membership_exclusion_families_are_py_honored():
    # AST-invisible membership/regex reads must be PY_HONORED (guard can't see them).
    assert _MEMBERSHIP_EXCLUSIONS <= _ALLOWLIST_PY_HONORED


def test_legacy_alias_keys_are_py_honored_not_warned():
    # species.lw.* / species.tl.* are legacy aliases of keys the engine reads under canonical
    # spellings — the Python engine implements the feature, so they must NOT be warned about
    # ("use the Java engine" would misdirect; spec §Out-of-scope requires species.lw.* silent).
    assert _LEGACY_ALIAS_HONORED <= _ALLOWLIST_PY_HONORED
    # conversion2tons is the OPPOSITE case (canonical unread) and stays JAVA_ONLY:
    assert "species.conversion2tons.sp{idx}" in _ALLOWLIST_JAVA_ONLY


def test_metadata_clearance_all_osmose_keys_py_honored():
    # Reader-injected metadata is UNREAD but must be PY_HONORED (else fires on every run).
    metadata = frozenset(k for k in FROZEN_ALLOWLIST_SNAPSHOT if k.startswith("osmose."))
    assert len(metadata) == 22
    assert metadata <= _ALLOWLIST_PY_HONORED


def test_java_only_keys_set_matches_literal_and_pattern():
    cfg = {
        "simulation.ncpu": "8",  # java-only literal
        "output.diet.stage.threshold.sp3": "12",  # java-only {idx} pattern
        "simulation.time.nyear": "15",  # a real read key (not allowlisted) -> ignored
    }
    assert java_only_keys_set(cfg) == ["output.diet.stage.threshold.sp3", "simulation.ncpu"]


def test_java_only_keys_set_excludes_py_honored_and_metadata():
    cfg = {
        "output.tl.enabled": "true",  # PY_HONORED (config.py:925) — round-4 landmine
        "movement.species.map0": "map.csv",  # PY_HONORED via startswith
        "evolution.trait.imax.target": "1.0",  # PY_HONORED exclusion family
        "ltl.depletable.enabled": "true",  # PY_HONORED (resources.py:74)
        "osmose.version": "4.4.1",  # reader-injected metadata — must never surface
        "osmose.configuration.background": "x.csv",  # metadata
    }
    assert java_only_keys_set(cfg) == []


def test_java_only_keys_set_excludes_120_restart_carveouts():
    cfg = {"simulation.restart.file": "snap.nc", "simulation.restart.enabled": "true"}
    assert java_only_keys_set(cfg) == []
    assert _RESTART_HANDLED_BY_120 == frozenset(
        {"simulation.restart.file", "simulation.restart.enabled"}
    )


def test_java_only_keys_set_canonicalizes_before_matching():
    # DISCRIMINATING (not vacuous): output.fishery.enabled canonicalizes (RENAMES_440) to the
    # java-only output.fisheries.enabled. The legacy source is not itself allowlisted, so WITHOUT
    # canonicalization this returns [] — the assertion proves canonicalize_config actually ran.
    assert java_only_keys_set({"output.fishery.enabled": "true"}) == ["output.fisheries.enabled"]


def test_java_only_keys_set_empty_when_none():
    assert java_only_keys_set({"simulation.time.nyear": "15"}) == []


@pytest.fixture(autouse=True)
def _clear_java_only_warn_cache():
    # MANDATORY: _WARNED_JAVA_ONLY_KEYS persists across tests in a process; clear it so the
    # "exactly one warning" and "deduped once" tests are not order-dependent (as #120 needed).
    cv._WARNED_JAVA_ONLY_KEYS.clear()
    yield
    cv._WARNED_JAVA_ONLY_KEYS.clear()


def _load_minimal_config() -> dict:
    from osmose.config.reader import OsmoseConfigReader

    return dict(OsmoseConfigReader().read("data/minimal/osm_all-parameters.csv"))


def test_warn_emits_one_summary_naming_the_keys(caplog):
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        keys = cv.warn_unread_java_only_keys({"simulation.ncpu": "8", "grid.java.classname": "X"})
    assert keys == ["grid.java.classname", "simulation.ncpu"]
    msgs = [r.getMessage() for r in caplog.records if "issue #123" in r.getMessage()]
    assert len(msgs) == 1
    assert "grid.java.classname" in msgs[0] and "simulation.ncpu" in msgs[0]


def test_warn_dedups_once_per_process(caplog):
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        cv.warn_unread_java_only_keys({"simulation.ncpu": "8"})
        cv.warn_unread_java_only_keys({"simulation.ncpu": "8"})
    assert sum("issue #123" in r.getMessage() for r in caplog.records) == 1


def test_warn_truncates_long_lists_but_counts_all(caplog):
    # Bundled demos set ~20-44 java-only keys; the line names at most _MAX_NAMED_JAVA_ONLY_KEYS
    # and counts the rest, while the returned list and the reported count stay complete.
    cfg = {f"output.diet.stage.threshold.sp{i}": "12" for i in range(15)}  # 15 java-only keys
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        keys = cv.warn_unread_java_only_keys(cfg)
    assert len(keys) == 15
    msg = next(r.getMessage() for r in caplog.records if "issue #123" in r.getMessage())
    assert "15 config key(s)" in msg  # full count reported
    assert "and 5 more" in msg  # 15 - 10 named
    assert msg.count("output.diet.stage.threshold.sp") == 10  # only 10 named inline


def test_warn_silent_when_no_java_only_keys(caplog):
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        cv.warn_unread_java_only_keys({"output.tl.enabled": "true", "simulation.time.nyear": "15"})
    assert not any("issue #123" in r.getMessage() for r in caplog.records)


def test_from_dict_does_not_warn_only_prepare_run_does(caplog):
    # Round-1 placement guard: constructing a config via from_dict (the Fisheries-UI / fmsy-probe
    # pattern) must NOT emit the #123 warning — only an actual run seam does.
    from osmose.engine.config import EngineConfig

    cfg = dict(_load_minimal_config())
    cfg["simulation.ncpu"] = "8"
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        EngineConfig.from_dict(cfg)
    assert not any("issue #123" in r.getMessage() for r in caplog.records)


def test_prepare_run_emits_the_warning(caplog):
    # The run seam DOES warn. _prepare_run builds engine_config/grid/rngs (no simulation loop),
    # so it is cheap to call directly.
    #
    # DEVIATION FROM BRIEF: the brief's verbatim assertion also required "simulation.ncpu" in the
    # message. The bundled minimal config already sets 17 other java-only keys (18 total with
    # simulation.ncpu, which is also already present, so cfg["simulation.ncpu"] = "8" below is a
    # no-op), and simulation.ncpu sorts at index 16 -- outside the first _MAX_NAMED_JAVA_ONLY_KEYS
    # (10) shown by the (verbatim, spec'd) truncation in warn_unread_java_only_keys. That clause is
    # factually unsatisfiable against the real fixture, not an implementation bug -- truncation
    # itself is independently pinned by test_warn_truncates_long_lists_but_counts_all, and key-naming
    # is covered by test_warn_emits_one_summary_naming_the_keys. This test's job is only to prove
    # _prepare_run wires the call through, so we assert on the marker alone.
    from osmose.engine import PythonEngine

    cfg = dict(_load_minimal_config())
    cfg["simulation.ncpu"] = "8"
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        PythonEngine()._prepare_run(cfg, seed=0)
    assert any("issue #123" in r.getMessage() for r in caplog.records)
