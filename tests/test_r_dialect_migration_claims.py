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
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader

FIXTURES = Path(__file__).parent / "fixtures"
RDIALECT = FIXTURES / "rdialect_config.R"


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
