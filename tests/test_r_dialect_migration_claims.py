"""Guards the load-bearing claims of docs/r-to-python-migration.md.

If this module goes red, the migration guide has gone stale — that is the intent.

Scope, stated honestly (see the spec's "Keeping the claims true"):
  Tier 1 — the MECHANISM: the R dialect parses; strict mode's asymmetry.
  Tier 2 — the two verified traps, asserted on the PYTHON side, plus a
           citation-PRESENCE check on the R side.

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
import warnings
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.config_validation import validate

FIXTURES = Path(__file__).parent / "fixtures"
RDIALECT = FIXTURES / "rdialect_config.R"
REPO_ROOT = Path(__file__).parent.parent
MINIMAL_CONFIG = REPO_ROOT / "data" / "minimal" / "osm_all-parameters.csv"


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
