"""Guards the load-bearing claims of docs/r-to-python-migration.md.

If this module goes red, the migration guide has gone stale — that is the intent.

Scope, stated honestly (see the spec's "Keeping the claims true"):
  Tier 1 — the MECHANISM: the R dialect parses; strict mode's asymmetry.
  Tier 2 — the two traps FIXED in #121 (output.tl.enabled, economy.enabled ->
           module.bioeconomics.enabled), retained here as regression anchors so
           neither silently regresses to unread again; asserted on the PYTHON side,
           plus a citation-PRESENCE check on the R side.

What these tests CANNOT do: verify that an R-side key is real. We do not vendor R
configs, so CI cannot check the corpus. A fabricated row with a plausible citation
would pass. Provenance truth is a HUMAN step at authoring and edit time.
Do not describe these tests as "asserted on both sides" — an earlier draft did, and
it was false.

NOT covered, by decision: the Benguela counts (844/236/...) and the jar-classfile
claims. Those are dated prose in the guide, not testable constants.
"""

# IMPORT RULE for later tasks (read before appending anything):
# Import ONLY what this task uses. Later tasks ADD their imports TO THIS HEADER BLOCK —
# they must NOT append imports beside their tests.
#   - imports beside appended tests  -> ruff E402 (module import not at top)
#   - imports declared before use    -> ruff F401 (unused import)
# CI runs `ruff check osmose/ ui/ tests/`, so BOTH are red. Editing the header is the
# only shape that satisfies both, and it keeps every commit independently lint-clean.
import logging
import subprocess
import warnings
from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.config_validation import validate

FIXTURES = Path(__file__).parent / "fixtures"
RDIALECT = FIXTURES / "rdialect_config.R"
REPO_ROOT = Path(__file__).parent.parent
MINIMAL_CONFIG = REPO_ROOT / "data" / "minimal" / "osm_all-parameters.csv"

# COLUMN SEMANTICS -- read carefully, a reviewer already misread this:
#   [0] r_key   : the key an R config actually contains (provenance-cited).
#   [1] py_key  : the osmopy-invented key the Python ENGINE read, historically NOT the
#                 alias/shim target (see FORMERLY-LIVE note below).
#   [2] citation: file:line in the upstream corpus. Asserted PRESENT, never TRUE.
#
# FORMERLY-LIVE TRAPS, FIXED IN #121, RETAINED AS REGRESSION ANCHORS -- not live silent
# gaps. Before #121, the shim correctly migrated the R dialect to the genuine upstream key,
# but the engine read only the invented py_key below, so the real upstream name silently did
# nothing. #121's Layer A (config.py:923, :2431) made the engine read the upstream name
# FIRST, keeping py_key as a back-compat fallback -- see
# test_output_tl_enabled_now_read_after_121 and test_bioeconomics_enabled_now_read_after_121.
# TRAPS and this table stay in place: test_traps_carry_a_provenance_citation and
# test_a_one_sided_assertion_would_be_vacuous still exercise it as a citation/vacuity guard.
#
# The economy row is the one that confuses people. THREE distinct keys are involved:
#   economy.enabled              -- what the R config says            (osmose-ben.R:1048)
#   module.bioeconomics.enabled  -- what the SHIM migrates it to; upstream's real 4.4.0
#                                   name. Now ALSO read directly by the engine (#121).
#   simulation.economic.enabled  -- the osmopy-invented py_key below; still read too, as a
#                                   back-compat fallback (engine/config.py:2431).
TRAPS = [
    ("output.tl.enabled", "output.meantl.enabled", "osmose-gog/osm_param-output.csv:43"),
    ("economy.enabled", "simulation.economic.enabled", "osmose-ben.R:1048"),
]


def test_r_dialect_parses_with_no_skipped_lines():
    """The guide's headline: point osmopy at an R .R config and it loads.

    Mechanism: OsmoseConfigReader.SEPARATORS includes '=' and COMMENT_CHARS includes '#'
    (config/reader.py:70-71), so the R dialect is readable without conversion.
    """
    reader = OsmoseConfigReader()
    cfg = reader.read(RDIALECT)

    assert reader.skipped_lines == 0
    assert cfg["simulation.nspecies"] == "2"
    assert cfg["species.name.sp0"] == "anchovy"


def test_r_uppercase_booleans_survive_the_reader():
    """R writes TRUE/FALSE; the guide says both work.

    _enabled() lowercases (engine/config.py:169), so case is handled. The R corpus is
    MIXED (.R files uppercase, .csv param files lowercase) — do not claim otherwise.
    """
    cfg = OsmoseConfigReader().read(RDIALECT)

    assert cfg["output.weight.enabled"] == "TRUE"
    assert cfg["fisheries.check.enabled"] == "FALSE"


def test_shim_migrates_pre_440_key(caplog):
    """A v4.3-era R config is auto-migrated to 4.4.0 canonical names on load."""
    reader = OsmoseConfigReader()
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        cfg = reader.read(RDIALECT)

    assert "economy.enabled" in reader.deprecated_keys
    # economy.enabled -> module.bioeconomics.enabled, upstream's real 4.4.0 name.
    assert cfg["module.bioeconomics.enabled"] == "TRUE"
    assert "economy.enabled" not in cfg


def _unknown_keys(cfg: dict[str, str]) -> set[str]:
    return {u.key for u in validate(cfg, "warn")}


def test_strict_mode_reports_unsupported_surveys_module():
    """surveys.* is unsupported and LOUD — but only if the reader opts into strict mode.

    Default validation is silent, which is why the guide tells the reader to turn it on
    before trusting anything.
    """
    cfg = OsmoseConfigReader().read(RDIALECT)
    unknown = _unknown_keys(cfg)

    assert "surveys.enabled.sr1" in unknown
    assert "surveys.name.sr1" in unknown


def test_strict_mode_is_SILENT_on_unimplemented_restart():
    """The asymmetry that makes strict mode necessary but NOT sufficient.

    simulation.restart.enabled is a KNOWN key, so strict mode never reports it — while the
    Python engine never implements it (engine/initialization.py exposes only
    build_initial_population / age_structured_population). It loads clean, validates clean,
    and silently does nothing. Tracked as issue #120.

    Knownness is SYMMETRICALLY REDUNDANT: it comes from BOTH osmose/schema/output.py:52 AND
    config_validation.py's allowlist, unioned in build_known_keys(). Removing either alone
    leaves this green. An earlier draft claimed this test "pins the allowlist" — it does not
    pin either source.

    This test does NOT detect a #120 fix. #120's fix surface is the ENGINE (a warning), not
    validate(); implementing #120's own suggested fix leaves this green. An earlier draft
    claimed "if this test starts FAILING, #120 has been fixed" — false. See the companion
    test below for the real tripwire.
    """
    cfg = OsmoseConfigReader().read(RDIALECT)
    unknown = _unknown_keys(cfg)

    assert "simulation.restart.enabled" not in unknown


def test_engine_does_not_yet_warn_on_ignored_restart(caplog):
    """The REAL #120 tripwire — asserts the actual fix surface, on BOTH warning channels.

    Today the Python engine silently ignores simulation.restart.enabled. When #120 lands and
    the engine warns, THIS test goes red, and that is the signal to update the guide's §2 and
    appendix to describe the warning instead of the silence.

    CAPTURES BOTH CHANNELS. osmose/engine/config.py emits warnings via BOTH `_log.warning`
    (logging) AND `warnings.warn` (the warnings module) — so we do not know which #120 will
    use. `caplog` catches only logging; `warnings.catch_warnings` catches only the warnings
    module. A tripwire that watched one channel would silently miss a fix on the other — which
    is the exact silent-failure class this whole guide is about. So we watch both.
    """
    cfg = OsmoseConfigReader().read(MINIMAL_CONFIG)
    cfg["simulation.restart.enabled"] = "true"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            EngineConfig.from_dict(cfg)

    log_hits = [r.getMessage() for r in caplog.records if "restart" in r.getMessage().lower()]
    warn_hits = [str(w.message) for w in caught if "restart" in str(w.message).lower()]
    assert log_hits + warn_hits == [], (
        "The engine now warns about ignored restart — #120 may be fixed. "
        "Update docs/r-to-python-migration.md §2 and the appendix, then update this test."
    )


@pytest.fixture
def minimal_cfg() -> dict[str, str]:
    return OsmoseConfigReader().read(MINIMAL_CONFIG)


def _probe(base: dict[str, str], **overrides: str) -> EngineConfig:
    cfg = dict(base)
    cfg.update(overrides)
    return EngineConfig.from_dict(cfg)


def test_output_tl_enabled_now_read_after_121(minimal_cfg):
    """FORMERLY a trap (the guide's headline example); FIXED in #121.

    output.tl.enabled is the real upstream Java name. Before #121 the engine read only the
    invented output.meantl.enabled and silently ignored the upstream name. Now honored.
    """
    assert _probe(minimal_cfg).output_meantl is False, "baseline"
    assert _probe(minimal_cfg, **{"output.tl.enabled": "true"}).output_meantl is True
    assert _probe(minimal_cfg, **{"output.meantl.enabled": "true"}).output_meantl is True


def test_traps_carry_a_provenance_citation():
    """Asserts every trap row NAMES A FILE. This is the honest, achievable half.

    It does NOT prove the citation is true — CI cannot, since we do not vendor R configs.
    What it buys: a row cannot be added by guessing from the allowlist without at least
    naming where it came from. That is exactly the discipline whose absence produced a
    retracted 8-row table in which 7 rows existed in zero R configs.

    Proven limitation, do not paper over it: the retracted row
    output.trophiclevel.enabled -> output.meantl.enabled would PASS this check just as
    readily as the two real rows do, and would equally satisfy
    test_output_tl_enabled_now_read_after_121 above — that test hardcodes literal key
    strings and never inspects r_key at all, so a fabricated r_key is invisible to both
    checks. Provenance truth is a HUMAN step.
    """
    for r_key, py_key, citation in TRAPS:
        assert citation, f"{r_key} has no provenance citation"
        assert ":" in citation, f"{r_key} citation must be file:line, got {citation!r}"
        assert r_key != py_key


def test_bioeconomics_enabled_now_read_after_121(minimal_cfg):
    """FORMERLY a trap (the guide's latent example); FIXED in #121.

    economy.enabled -> module.bioeconomics.enabled (RENAMES_440, upstream's real 4.4.0 name).
    Before #121 the engine read only the invented simulation.economic.enabled. Now the upstream
    name is honored, so an authentic migrated config gets economics.
    """
    assert _probe(minimal_cfg).economics_enabled is False, "baseline"
    assert _probe(minimal_cfg, **{"module.bioeconomics.enabled": "true"}).economics_enabled is True
    assert _probe(minimal_cfg, **{"simulation.economic.enabled": "true"}).economics_enabled is True


def test_a_one_sided_assertion_would_be_vacuous():
    """DOCUMENTS (does not enforce) why the now-fixed tests assert more than the baseline.

    A one-sided "the config key leaves the attribute at its default" assertion passes for a
    key that does not exist at all — demonstrated below. That is why
    test_output_tl_enabled_now_read_after_121 and test_bioeconomics_enabled_now_read_after_121
    each go on to assert that a REAL key (upstream, then invented back-compat) positively
    flips the attribute to True: a "stays at default" claim is trivially satisfied by junk
    like banana.enabled, but a "flips to True" claim on a specific key is not.

    Honest scope: this test cannot stop anyone weakening those tests; strip their "flips to
    True" assertions and it still passes. It is executable documentation, not a guard. An
    earlier draft claimed "this test exists so nobody weakens the trap tests" — false.
    """
    base = OsmoseConfigReader().read(MINIMAL_CONFIG)
    # An invented key satisfies the one-sided half trivially:
    assert _probe(base, **{"banana.enabled": "true"}).output_meantl is False


def test_spatial_inputs_trap_movement_loader_is_csv_only():
    """Pins §2's headline trap: osmopy's movement-map loader is CSV-only, so a binary .nc map
    fails to parse and the run silently continues with that grid dropped.

    Two guards, because the guide's most important claim must not rot silently (that is the
    guide's own thesis). Unlike the Benguela counts (unvendored, prose-tier), this is about
    osmopy's OWN source and is trivially pinnable — it would go stale the day someone adds
    NetCDF support, which is exactly when the guide would need to change.
    """
    from osmose.engine.movement_maps import _load_csv_grid

    # Guard 1 — the loader has no NetCDF path at all (CSV-only).
    hits = subprocess.run(
        ["grep", "-icE", r"netcdf|xarray|\.nc\b|Dataset", "osmose/engine/movement_maps.py"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()
    assert hits == "0", (
        f"movement_maps.py gained NetCDF tokens ({hits}); §2's CSV-only claim is stale"
    )

    # Guard 2 — the mechanism: a binary .nc-like file raises ValueError (UnicodeDecodeError
    # subclasses it), which movement_maps.py:220 catches → grid set to None → run continues.
    import tempfile

    nc = Path(tempfile.mkdtemp()) / "map.nc"
    nc.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(range(256)) * 4)  # binary, non-UTF-8
    with pytest.raises(ValueError):
        _load_csv_grid(nc, 10, 10)
