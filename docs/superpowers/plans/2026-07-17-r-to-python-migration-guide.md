# R OSMOSE → Python Migration Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `docs/r-to-python-migration.md` — a guide for an existing R OSMOSE user porting a working config + driver scripts to osmopy — plus a test suite that keeps its load-bearing claims from rotting.

**Architecture:** One MyST Markdown file in the `Guides` toctree, linking into `docs/usage-guide.md` for Python mechanics rather than restating them. One new test module (`tests/test_r_dialect_migration_claims.py`) + one synthetic fixture (`tests/fixtures/rdialect_config.R`) pinning the guide's Tier-1 (mechanism) and Tier-2 (the two verified traps) claims.

**Tech Stack:** Python 3.12, pytest, Sphinx 8 + myst-parser (`pyproject.toml:38`), existing `osmose.config.reader` / `osmose.engine.config_validation` / `osmose.engine.config`.

**Spec:** `docs/superpowers/specs/2026-07-17-r-to-python-migration-guide-design.md`

## Global Constraints

- **METHOD RULE (governs every task):** An allowlist entry is not evidence of a read. A line number is not evidence of a behavior. A schema declaration is not evidence of a consumer. **Run the code before writing any factual claim into the guide.** Every claim in this plan was verified by execution; if you change one, re-verify it the same way. This is the spec's central lesson and the reason four review rounds were needed.
- **Every R snippet in the guide MUST cite the real file it came from** (`osmose-gog/run.R`, `osmose-ben/launcher.R`, `osmose-gog/calibrate.R`, `osmose-gom/analysis.R`). No snippet may rest on recollection of the R API.
- **No Python mechanics restated from `docs/usage-guide.md`** (251 lines: run → read → compare → calibrate). Show the **call signature only**. If a code block would still make sense with the R side deleted, it belongs in usage-guide, not here.
- **Benguela counts are a dated exhibit, never figures to match.** Always write "measured 2026-07-17 against `osmose-ben.R`". The reader's instruction is always "run it on *your* config".
- **Quote 844, not 845.** `len()` reports 845; `_osmose.config.dir` is injected by `reader.py:91` and is not in the file. A reader grepping their own config counts 844.
- **Never write "the shim betrays you"** or "the UI lights up an Economic page". Both are false; see Task 5.
- **No vendored R config.** The fixture is synthetic and hand-written.
- Commit after every task. Branch: `docs/r-to-python-migration-guide` (already checked out).

## File Structure

| File | Responsibility |
|---|---|
| `tests/fixtures/rdialect_config.R` (create) | Synthetic ~20-line R-dialect config. Exercises `=` separators, `#` comments, `TRUE`/`FALSE`, a shim-migrated key, a `surveys.*` key, a `simulation.restart.*` key. **Not vendored.** |
| `tests/test_r_dialect_migration_claims.py` (create) | All tests guarding the guide. Named for its purpose: if it goes red, the guide has gone stale. Tier-1 (mechanism) + Tier-2 (the two traps). |
| `docs/r-to-python-migration.md` (create) | The guide. Six sections + appendix. |
| `docs/index.md` (modify) | Add `r-to-python-migration` to the `Guides` toctree. |

Tasks 1–3 build the test suite first (TDD, and they pin the facts the guide asserts). Tasks 4–8 write the guide section by section. Task 9 verifies the whole against the spec's success criteria.

---

### Task 1: Synthetic R-dialect fixture + Tier-1 parse mechanism test

Proves the guide's headline: **osmopy parses the R `.R` config dialect as-is.**

**Files:**
- Create: `tests/fixtures/rdialect_config.R`
- Create: `tests/test_r_dialect_migration_claims.py`

**Interfaces:**
- Consumes: `osmose.config.reader.OsmoseConfigReader` — `.read(Path) -> dict[str, str]`, plus attributes `.skipped_lines: int`, `.deprecated_keys: list[str]`, `.diagnostics: list[ConfigDiagnostic]`.
- Produces: `FIXTURES` constant and the fixture file path, used by Tasks 2–3.

- [ ] **Step 1: Write the fixture**

Create `tests/fixtures/rdialect_config.R`:

```r
# Synthetic R-dialect OSMOSE config — hand-written for tests, NOT vendored upstream content.
# Mirrors the shape of osmose-model/osmose-ben's osmose-ben.R: `key = value` lines in a .R file.
#
# Provenance for the real keys this mirrors (verified 2026-07-17):
#   economy.enabled             -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R:1048
#   surveys.name.sr1            -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R (surveys block)
#   simulation.restart.enabled  -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R

simulation.nspecies = 2
species.name.sp0 = anchovy
species.name.sp1 = sardine

fisheries.check.enabled = FALSE
output.weight.enabled = TRUE

# pre-4.4.0 key that the 4.4.0 compat shim migrates
economy.enabled = TRUE

# unsupported module -> strict validation MUST report these as unknown
surveys.enabled.sr1 = TRUE
surveys.name.sr1 = acousticSurvey

# allowlisted-but-unread -> strict validation must NOT report this
simulation.restart.enabled = TRUE
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_r_dialect_migration_claims.py`:

```python
"""Guards the load-bearing claims of docs/r-to-python-migration.md.

If this module goes red, the migration guide has gone stale — that is the intent.
Scope is deliberate (see the spec's "Keeping the claims true"):
  Tier 1 — the MECHANISM (the R dialect parses; strict mode's asymmetry).
  Tier 2 — the two verified traps, asserted on BOTH sides (python + provenance).
NOT covered, by decision: the Benguela counts (844/236/...) and the jar-classfile
claims. Those are dated prose in the guide, not testable constants.
"""

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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py -v`
Expected: FAIL — `FileNotFoundError` or collection error until the fixture exists. Once the fixture is added they should pass; if any assertion fails, **stop and re-verify against the code — do not adjust the assertion to match**.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py -v`
Expected: PASS, 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/rdialect_config.R tests/test_r_dialect_migration_claims.py
git commit -m "test: pin the R-dialect parse mechanism for the migration guide"
```

---

### Task 2: Tier-1 strict-mode asymmetry test

Proves the guide's core safety claim: **strict mode catches `surveys.*` but is silent on restart.** This asymmetry is why §2 prescribes two tools, not one.

**Files:**
- Modify: `tests/test_r_dialect_migration_claims.py` (append)

**Interfaces:**
- Consumes: `osmose.engine.config_validation.validate(cfg: dict, mode: str) -> list[UnknownKey]`. `UnknownKey` is a frozen dataclass with `.key: str` and `.suggestion: str | None` (`config_validation.py:265-267`). `mode` is one of `"off"` / `"warn"` / `"error"`; `"error"` raises `ValueError` listing all unknowns.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_r_dialect_migration_claims.py`:

```python
from osmose.engine.config_validation import validate


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

    simulation.restart.enabled is allowlisted as a KNOWN key (config_validation.py, marked
    "Java-side") but the Python engine never implements it — engine/initialization.py exposes
    only build_initial_population / age_structured_population. So it loads clean, validates
    clean, and silently does nothing. Tracked as issue #120.

    If this test starts FAILING, #120 has been fixed — that is good news, and the guide's
    §2 and appendix must be updated to describe the new warning instead of the silence.
    """
    cfg = OsmoseConfigReader().read(RDIALECT)
    unknown = _unknown_keys(cfg)

    assert "simulation.restart.enabled" not in unknown
```

- [ ] **Step 2: Run to verify**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py -v`
Expected: PASS, 5 passed. (These assert current behavior, so they pass immediately — their value is as a **tripwire**: they go red exactly when #120 lands or when the allowlist changes.)

- [ ] **Step 3: Prove the tripwire actually trips**

Do not skip this — a test that cannot fail is worse than none. Temporarily invert one assertion:

```bash
sed -i 's/assert "simulation.restart.enabled" not in unknown/assert "simulation.restart.enabled" in unknown/' tests/test_r_dialect_migration_claims.py
.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py::test_strict_mode_is_SILENT_on_unimplemented_restart -v
```
Expected: FAIL. Then revert:
```bash
sed -i 's/assert "simulation.restart.enabled" in unknown/assert "simulation.restart.enabled" not in unknown/' tests/test_r_dialect_migration_claims.py
.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py -v
```
Expected: PASS, 5 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_r_dialect_migration_claims.py
git commit -m "test: pin strict-mode asymmetry (surveys loud, restart silent)"
```

---

### Task 3: Tier-2 two-sided trap assertions

Pins the guide's two verified traps. **The assertion must be two-sided.** A one-sided version ("the R key leaves the attribute at default; the mapped key changes it") is **vacuous** — it passes for `banana.enabled`, i.e. any key osmopy doesn't read, real or invented. That is exactly how a prior draft's eight fabricated rows would have shipped green.

**Files:**
- Modify: `tests/test_r_dialect_migration_claims.py` (append)

**Interfaces:**
- Consumes: `osmose.engine.config.EngineConfig.from_dict(cfg: dict) -> EngineConfig`, attributes `.output_meantl: bool`, `.economics_enabled: bool`. Base config: `data/minimal/osm_all-parameters.csv`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_r_dialect_migration_claims.py`:

```python
import pytest

from osmose.engine.config import EngineConfig

REPO_ROOT = Path(__file__).parent.parent
MINIMAL_CONFIG = REPO_ROOT / "data" / "minimal" / "osm_all-parameters.csv"
R_CORPUS_NOTE = (
    "Provenance is a HUMAN step: these R keys were verified present in the upstream "
    "corpus on 2026-07-17 (output.tl.enabled in 7 config files; economy.enabled at "
    "osmose-ben.R:1048). We do not vendor R configs, so CI cannot re-check provenance — "
    "it is asserted here as a citation, not proven. Re-verify when editing the guide."
)


@pytest.fixture
def minimal_cfg() -> dict[str, str]:
    return OsmoseConfigReader().read(MINIMAL_CONFIG)


def _probe(base: dict[str, str], **overrides: str) -> EngineConfig:
    cfg = dict(base)
    cfg.update(overrides)
    return EngineConfig.from_dict(cfg)


def test_trap_output_tl_enabled_is_silently_ignored(minimal_cfg):
    """TRAP 1. output.tl.enabled is the REAL upstream Java name (present in the 4.4.1 jar,
    set in 7 R config files). osmopy's engine reads output.meantl.enabled instead — an
    osmopy name present in 0 R configs and 0 jars. So the R user's key silently does nothing.

    Two-sided by design: the "R key does nothing" half alone would pass for any invented key.
    """
    assert _probe(minimal_cfg).output_meantl is False, "baseline"

    # R/Java side: the real upstream key is silently ignored.
    assert _probe(minimal_cfg, **{"output.tl.enabled": "true"}).output_meantl is False

    # Python side: the osmopy key is what actually works.
    assert _probe(minimal_cfg, **{"output.meantl.enabled": "true"}).output_meantl is True


def test_trap_economy_enabled_is_silently_ignored(minimal_cfg):
    """TRAP 2 — the worst one found, and the guide's headline example.

    economy.enabled (osmose-ben.R:1048) is migrated by the shim to
    module.bioeconomics.enabled — which is CORRECT: that is upstream's genuine 4.4.0 name
    (2 hits in the 4.4.1 jar, including Releases$15, upstream's own renames table; 0 in 4.3.3).
    The defect is that osmopy's engine reads simulation.economic.enabled (engine/config.py:2431)
    — a key with 0 hits in either jar and 0 in the R corpus. The shim is right; we are wrong.

    Do NOT rewrite this as "the shim betrays you". Tracked as issue #121.
    """
    assert _probe(minimal_cfg).economics_enabled is False, "baseline"

    # Upstream's correct 4.4.0 key — silently ignored by osmopy's engine.
    assert _probe(minimal_cfg, **{"module.bioeconomics.enabled": "true"}).economics_enabled is False

    # osmopy's invented key — the only one that actually works, and only on the Python engine.
    assert _probe(minimal_cfg, **{"simulation.economic.enabled": "true"}).economics_enabled is True


def test_the_one_sided_assertion_is_vacuous():
    """Guards the GUARD. Documents why the tests above are two-sided.

    A one-sided "the R key leaves the attribute at its default" assertion passes for a key
    that does not exist at all. It would have shipped 7 fabricated rename rows green.
    This test exists so nobody weakens the trap tests back to one-sided.
    """
    base = OsmoseConfigReader().read(MINIMAL_CONFIG)
    # An invented key satisfies the one-sided half trivially:
    assert _probe(base, **{"banana.enabled": "true"}).output_meantl is False
```

- [ ] **Step 2: Run to verify**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 3: Lint**

Run: `.venv/bin/python -m ruff check tests/test_r_dialect_migration_claims.py && .venv/bin/python -m ruff format tests/test_r_dialect_migration_claims.py`
Expected: clean. (CI lint = check + format; see the project's lint convention.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_r_dialect_migration_claims.py
git commit -m "test: pin the two verified migration traps with two-sided assertions"
```

---

### Task 4: Guide skeleton, §1 Should you switch, and toctree wiring

Wire the page into Sphinx **first** so every later task can build and check its own links.

**Files:**
- Create: `docs/r-to-python-migration.md`
- Modify: `docs/index.md` (Guides toctree)

- [ ] **Step 1: Create the guide with its §1**

Create `docs/r-to-python-migration.md`. Write the title, a one-paragraph statement of who it's for (an existing R OSMOSE user with a working config and driver scripts, porting to osmopy), and §1 with these **verified** contents:

- **Gains:** no JVM dependency; faster than Java on every benchmarked config; the calibration stack (NSGA-II / CMA-ES / surrogate-DE / Pareto explorer); the Shiny UI.
- **Losses:** no surveys module; no Python-engine restart; temperature/oxygen forcing downgrades to **constant-only** (`temperature.value` / `oxygen.value`, gated on `bioen_enabled`) — a capability downgrade, **not** a rename; no `plot()` one-liner convenience.
- **The Java engine remains available** and is the fallback for the capability-absent and unsupported-module gaps. Verified: `fr/ird/osmose/output/Surveys.class` is in **both** the 4.3.3 and 4.4.1 jars, and restart is implemented in 4.4.1 (`SchoolSetSnapshot` / `ModularSchoolSetSnapshot`, populator strings `simulation.restart.file`, `isRestart`).
- Renamed keys need no fallback — they need the right key name.

Add stub headings for §§2–6 and the appendix so the toctree renders; later tasks fill them.

- [ ] **Step 2: Wire into the toctree**

Modify `docs/index.md`, adding the guide to the existing `Guides` toctree (which currently lists `usage-guide`, `tutorials/30-minute-ecosystem`, `tutorials/fie-on-baltic-cod`):

```
:::{toctree}
:maxdepth: 2
:caption: Guides

usage-guide
r-to-python-migration
tutorials/30-minute-ecosystem
tutorials/fie-on-baltic-cod
:::
```

- [ ] **Step 3: Build the docs and verify the page appears**

Run: `.venv/bin/python -m sphinx -b html docs docs/_build/html -q`
Expected: builds with no error mentioning `r-to-python-migration`. Confirm the file was rendered:
```bash
test -f docs/_build/html/r-to-python-migration.html && echo "PAGE RENDERED"
```
Expected: `PAGE RENDERED`.

- [ ] **Step 4: Commit**

```bash
git add docs/r-to-python-migration.md docs/index.md
git commit -m "docs: add R->Python migration guide skeleton + section 1"
```

---

### Task 5: §2 — "Your config already loads, and that's the trap"

The heart of the guide. Everything else is reference; this is the section that saves people.

**Files:**
- Modify: `docs/r-to-python-migration.md`

- [ ] **Step 1: Verify Java's cross-file precedence BEFORE writing the rule**

The spec flags this as the one unverified thing left. osmopy's rule is: **sub-config beats master, independent of where the reference sits** (parent written first in a depth-first walk; among siblings, last-referenced wins). Publishing a precedence rule that differs from the engine the reader is migrating *from* would be worse than saying nothing.

Run:
```bash
cd osmose-java && unzip -p osmose-4.4.1-jar-with-dependencies.jar 'fr/ird/osmose/Configuration.class' | strings | grep -iE "sub.?config|osmose.configuration|already defined|overwrit" | head
```

- If the evidence is **inconclusive** (likely): write the precedence rule as **"osmopy's behavior, verified; Java's is unverified — check yours if you rely on it"**. Do **not** state it as a universal OSMOSE rule.
- If Java clearly matches or differs: say so, and cite what you found.

Record the outcome in the guide as a note. This step embodies the Method rule — it is the difference between a true guide and a plausible one.

- [ ] **Step 2: Write §2**

Write, in this order:

1. **The exhibit**, dated, with its contingency inline: "Measured 2026-07-17 against the real `osmose-ben.R`: **844 keys parsed, 0 skipped lines — and 236 of them unknown to osmopy.** The same parse's sub-config resolution *failed* (a referenced `input/initial_conditions.osm` isn't there), so those are keys reachable *without* that file, not the config's key count. That isn't a caveat on the exhibit — it *is* the exhibit." Then: run it on **your** config; these numbers are one example, not a target.

2. **The two-tool prescription, in order:**
   - **`scripts/check_config.py` first.** The only production caller of `format_diagnostics` / `diagnostics_have_errors`. Surfaces parse-level damage: missing sub-configs, duplicate keys, unparseable lines. Neither `osmose/cli.py` nor the UI reads `reader.diagnostics`. (A missing sub-config *does* also log a warning at `reader.py:142` — say "easy to miss", **not** "silent".)
   - **then `validation.strict.enabled=error`.** Key-level: what osmopy doesn't recognize.

3. **The beat that matters:** neither is sufficient, and strict mode is the weaker one. It catches `surveys.*` but is silent on restart, on renamed keys, on missing sub-configs and on cross-file collisions — because those keys are *known*, or never arrive. **A clean strict-mode run means nothing was unrecognized, not that your config works.**

4. **Silence scales with damage.** A missing sub-config's keys never reach the flattened dict, so strict mode sees *fewer* unknowns and is *more* likely to pass clean. The worse the damage, the quieter the result. This is why `check_config.py` comes first.

5. **The shim rescues half and strands half.** Of the 8 keys the 4.4.0 shim migrated on Benguela, **four land on keys nothing reads** (`output.restart.enabled`, `output.restart.spinup`, `output.fishery.enabled` → `output.fisheries.enabled`, `economy.enabled` → `module.bioeconomics.enabled`) and four reach the engine (`fisheries.enabled` `engine/config.py:2032`, `simulation.bioen.enabled` :2365, `simulation.genetic.enabled` :2422, `population.initialization.relativebiomass.enabled` :538). **Same mechanism, same config, identical surface behavior.**

6. **The taxonomy table** (seven directions — six needing a workaround, plus value coercion which is latent and gets one sentence). Copy it from the spec's "Gap taxonomy". State plainly that it is **assumed incomplete**: an earlier draft claimed three buckets and was confident.

7. **The two verified traps**, taught in prose (they recur in the appendix as reference — that duplication is intentional). For `economy.enabled`, say **which engine is at fault**: the shim is correct, `module.bioeconomics.enabled` is upstream's genuine 4.4.0 name; osmopy's engine invented `simulation.economic.enabled`. So `simulation.economic.enabled` works only on Python; `module.bioeconomics.enabled` is right for Java. The honest UI claim is narrow: the key adds "Economic" to the Run page's "Will populate:" label (`ui/pages/run.py:797`), which then doesn't populate. **Do not** write that the UI renders an Economic page — it gates on `engine_mode` and honestly says the module isn't implemented.

8. **The friendly failure, for contrast.** Show `species.lw.condition.factor.sp0` without its `species.length2weight.*` twin raising:
   ```
   KeyError: "Required OSMOSE config key missing: 'species.length2weight.condition.factor.sp0'"
   ```
   A crash naming the exact right key is the **best** case. Silence is the dangerous one.

- [ ] **Step 3: Verify every code snippet in §2 actually runs**

For each command or snippet you wrote, execute it and confirm the output matches what the guide claims. If any doesn't, **fix the guide, not the expectation**.

`scripts/check_config.py` takes `--config`, **not** a positional path (verified 2026-07-17 — an earlier draft of this plan got it wrong, which is the Method rule working):

```bash
.venv/bin/python scripts/check_config.py --config data/minimal/osm_all-parameters.csv; echo "exit=$?"
```
Expected: reports diagnostics; exit 0 on a clean config, 1 when `diagnostics_have_errors`. **The guide must quote this exact invocation** — a reader who copies a wrong command concludes the tool is broken and skips the most important step in §2.

Also confirm the value it adds over strict mode, since that is §2's whole argument — point it at a config with a missing sub-config and check it reports what `validate()` cannot:

```bash
TMP=$(mktemp -d)
printf 'simulation.nspecies;1\nosmose.configuration.initialization;missing_file.csv\n' > "$TMP/master.csv"
.venv/bin/python scripts/check_config.py --config "$TMP/master.csv"; echo "exit=$?"
```
Expected: reports the missing sub-config. This is the case strict mode passes clean.

- [ ] **Step 4: Build and commit**

```bash
.venv/bin/python -m sphinx -b html docs docs/_build/html -q
git add docs/r-to-python-migration.md
git commit -m "docs: migration guide section 2 - the config trap and two-tool prescription"
```

---

### Task 6: §3 Run and §4 Read & plot

**Files:**
- Modify: `docs/r-to-python-migration.md`

- [ ] **Step 1: Write §3 Run**

R side, **citing the real files** — note both naming eras are live in the wild:
- `runOsmose("osm_all-parameters.csv", version=4, osmose=jarfile)` — legacy camelCase, from `osmose-gog/run.R`
- `run_osmose(input=, output=, osmose=jarFile, version="4.3.3")` — current snake_case, from `osmose-ben/launcher.R`

Python side — **call signature only**:
- `PythonEngine().run(config=..., output_dir=..., seed=...)` (writes CSV/NetCDF)
- `PythonEngine().run_in_memory(config=..., seed=...)` (returns `OsmoseResults`, no disk)

Note the JVM disappears — no `osmose=jarfile` argument, no `java` on PATH. Link `usage-guide.md` §1 for mechanics. **Do not** show parameters, outputs, or a runnable end-to-end snippet.

- [ ] **Step 2: Write §4 Read & plot**

R side, cited:
- `read_osmose(path=outdir, version="v3r2")` then `data$biomass` / `data$yield` — from `osmose-gog/runModel.R` **only** (note: `osmose-ben/launcher.R` calls `read_osmose` but uses `plot()`/`get_var()` instead — cite precisely)
- `get_var(ben, what="biomass", how="list")` — from `osmose-ben/launcher.R`
- `plot(ben, what="yield", initialYear=2000, freq=12)`, `plot(ben, what="yield.fishery.anchovy", col="red", lwd=2)` — from `osmose-ben/launcher.R`

Python side: `OsmoseResults` — call signature only.

**Be honest:** R's `plot(obj, what=…)` one-liners have **no single equivalent**. Point at the plotting module and the UI. Also note `plot(ben, what="biomass.acousticSurvey")` has no equivalent at all — it reads survey outputs, and `surveys.*` is unsupported (§2's taxonomy; use the Java engine). Link `usage-guide.md` §2.

- [ ] **Step 3: Verify every R snippet is really in the file it cites**

For each snippet, confirm it exists verbatim upstream. The repos are public; if you no longer have them cloned, re-clone shallow into a scratch dir:
```bash
git clone --depth 1 -q https://github.com/osmose-model/osmose-gog.git /tmp/verify-gog
grep -n "runOsmose" /tmp/verify-gog/run.R
```
Expected: the exact call the guide quotes. **A snippet that cannot be located must be removed, not paraphrased.**

- [ ] **Step 4: Build and commit**

```bash
.venv/bin/python -m sphinx -b html docs docs/_build/html -q
git add docs/r-to-python-migration.md
git commit -m "docs: migration guide sections 3-4 - run, read and plot"
```

---

### Task 7: §5 Calibrate

The largest section and the one where an R user feels the most friction. calibrar's capability is present in osmopy, but the **shape** differs completely.

**Files:**
- Modify: `docs/r-to-python-migration.md`

- [ ] **Step 1: VERIFY the mapping before writing a word of it**

The spec supplies a **plausible** mapping, explicitly not a verified one — and this plan's central lesson is what happens when those are confused. Confirm each target exists and does what's claimed:

```bash
cd /home/razinka/osmopy
grep -nE "^class |^def " osmose/calibration/multiphase.py | head
grep -nE "^class |^def " osmose/calibration/targets.py | head
grep -nE "^class |^def " osmose/calibration/objectives.py | head
grep -nE "^class |^def " osmose/calibration/losses.py | head
grep -nE "^class |^def " osmose/calibration/problem.py | head
```

Mapping to verify (correct it if the code disagrees):
- `calibrate(phases=)` → `MultiPhaseCalibrator` / `CalibrationPhase` (`calibration/multiphase.py`) — **verified**: semantics match calibrar exactly ("Output of phase N becomes fixed params for phase N+1"). The difference is plumbing: calibrar reads phases from a `parphase` CSV column; osmopy constructs them in code.
- `control$popsize` / `control$maxgen` → optimizer args
- `getCalibrationInfo` / `getObservedData` → `calibration/targets.py` + `objectives.py`
- `createObjectiveFunction(aggFn=, aggregate=)` → `calibration/losses.py` + `problem.py`
- user-written `runModel(param, names, ...)` → **no counterpart by design**

**Any target that doesn't check out gets an explicit "no equivalent, do X instead" — never a guess.**

- [ ] **Step 2: Write §5**

Lead with the shape difference, because it's the real story. In calibrar (from `osmose-gog/calibrate.R` + `runModel.R`) the **user writes the driver**: a `runModel(param, names, ...)` that writes params to CSV, shells out to the jar, reads outputs back, and returns a named list; then `getCalibrationInfo` → `getObservedData` → `createObjectiveFunction` → `calibrate(...)` wire it together.

In osmopy, **the framework owns the run/read loop**. The user supplies parameters and a loss, not a driver. That is why `runModel` has no counterpart — and its absence is the point, not a gap.

Then give each calibrar symbol its verified counterpart (or explicit no-equivalent) from Step 1. Note `phases` is **not** a gap — a real equivalent exists. Link `usage-guide.md` §4 for mechanics.

- [ ] **Step 3: Build and commit**

```bash
.venv/bin/python -m sphinx -b html docs docs/_build/html -q
git add docs/r-to-python-migration.md
git commit -m "docs: migration guide section 5 - calibration"
```

---

### Task 8: §6 Verify your port + appendix

**Files:**
- Modify: `docs/r-to-python-migration.md`

- [ ] **Step 1: Write §6 as TWO comparisons**

A naive "run both engines, compare biomass" **conflates two variables** — the reader has a v3/v4.3-era config while osmopy's default jar is 4.4.1, so a mismatch would be unattributable. Prescribe:

- **Step 1 — isolate the port.** Re-run the reader's *original* jar through osmopy's Java engine. `OsmoseRunner.__init__` accepts an arbitrary `jar_path` (`osmose/runner.py:123`). Same engine, same config, new driver. **Any difference here is the port.**
- **Step 2 — isolate the engine.** Then compare against the Python engine / default 4.4.1. **Any difference here is the engine or the version**, not the port.

Set tolerance honestly: `usage-guide.md` §6 documents Python-vs-Java agreeing only **"within 1 order of magnitude"** on bundled, already-verified configs. Do not promise bit-equality. Point at `docs/parity-roadmap.md` rather than inventing a method.

- [ ] **Step 2: Write the appendix**

Three parts:
1. **The R→Python symbol table** — every symbol from the spec's "R API surface to cover", each citing its real file. `initialize_osmose`, `.readConfiguration` and `.getPar` are **table-only** (no body section): `initialize_osmose` maps to "no Python restart — use the Java engine"; the dot-prefixed pair are R-internal with no counterpart by design.
2. **The two verified traps** — reference form (`output.tl.enabled`, `economy.enabled`). Deliberately two rows, **not** a table of plausible ones. Add one line: the general case is not enumerable and is tracked in [#121](https://github.com/razinkele/osmopy/issues/121); the real fix is tooling that names the correct key, not a static table that rots.
3. **The honest gaps list**, each with its workaround:
   - `surveys.*` → Java engine (verified: `Surveys.class` in both jars)
   - Python-engine restart → Java engine ([#120](https://github.com/razinkele/osmopy/issues/120))
   - temperature/oxygen forcing → **no workaround; say so plainly.** Constant-only in Python.
   - `plot()` convenience → plotting module / UI

- [ ] **Step 3: Build and commit**

```bash
.venv/bin/python -m sphinx -b html docs docs/_build/html -q
git add docs/r-to-python-migration.md
git commit -m "docs: migration guide section 6 + appendix"
```

---

### Task 9: Full verification against the spec's success criteria

**Files:**
- Modify: `docs/r-to-python-migration.md` (fixes only)

- [ ] **Step 1: Run the full guarding suite + lint**

```bash
.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py -v
.venv/bin/python -m ruff check osmose/ ui/ tests/ && .venv/bin/python -m ruff format --check osmose/ ui/ tests/
```
Expected: 8 passed; ruff clean.

- [ ] **Step 2: Build docs clean**

```bash
.venv/bin/python -m sphinx -b html docs docs/_build/html 2>&1 | grep -iE "warning|error" | grep -i "r-to-python" || echo "NO WARNINGS FOR THIS PAGE"
```
Expected: `NO WARNINGS FOR THIS PAGE`.

- [ ] **Step 3: Check the forbidden claims are absent**

These are the specific falsehoods earlier drafts asserted. None may appear:

```bash
cd /home/razinka/osmopy
grep -niE "shim betrays|lights up an .?Economic|renders an Economic page" docs/r-to-python-migration.md || echo "OK: no false UI/shim claims"
grep -n "845 keys" docs/r-to-python-migration.md || echo "OK: quotes 844 not 845"
grep -niE "uppercase exclusively" docs/r-to-python-migration.md || echo "OK: no 'exclusively' claim"
grep -niE "output\.(meansize|byage|bysize|trophiclevel)\.enabled" docs/r-to-python-migration.md || echo "OK: no retracted rename rows"
```
Expected: all four `OK:` lines. **Any hit is a regression to a retracted claim — remove it.**

- [ ] **Step 4: Check the boundary rule held**

For every code block in the guide, ask: *would this still make sense with the R side deleted?* If yes, it belongs in `usage-guide.md` — cut it and link instead. Confirm the guide links rather than restates:

```bash
grep -c "usage-guide" docs/r-to-python-migration.md
```
Expected: ≥ 4 (§§3, 4, 5, 6 each link it).

- [ ] **Step 5: Walk the success criteria**

Confirm each, from the spec. Fix any gap before finishing:
- A reader whose config sets `economy.enabled` learns their run has no economics on the Python engine even though the key migrated to upstream's correct name — and that "Will populate: Economic" is a promise it won't keep.
- A reader with a missing sub-config finds out, because the guide sent them to `check_config.py`, not strict mode alone.
- All gap buckets have their differing workarounds; temperature/oxygen says plainly it has none.
- Every R-side key named is greppable in a named upstream file. **No row exists because the allowlist mentioned it.**
- Every R snippet cites a real file in a real repo.
- No Python mechanics restated from `usage-guide.md`.
- Nothing claimed CI-protected that isn't; nothing called unprotectable that was merely declined.

- [ ] **Step 6: Commit and open the PR**

```bash
git add docs/r-to-python-migration.md
git commit -m "docs: verify migration guide against spec success criteria"
git push -u origin docs/r-to-python-migration-guide
gh pr create --title "docs: R OSMOSE -> Python migration guide" --body "$(cat <<'BODY'
Ships `docs/r-to-python-migration.md` for R OSMOSE users porting a working config + driver scripts to osmopy, plus `tests/test_r_dialect_migration_claims.py` guarding its load-bearing claims.

Spec: `docs/superpowers/specs/2026-07-17-r-to-python-migration-guide-design.md`
Plan: `docs/superpowers/plans/2026-07-17-r-to-python-migration-guide.md`

**Headline finding:** osmopy parses the R `.R` config dialect as-is (844 keys, 0 skipped on the real `osmose-ben.R`) — and that is the trap, not the good news: 236 of those keys are unknown, and strict mode cannot see the worst failures.

Surfaced two engine defects, filed separately and NOT fixed here: #120 (Python engine silently ignores `simulation.restart.enabled`) and #121 (allowlisted-but-unread keys, incl. osmopy's invented `simulation.economic.enabled` shadowing upstream's `module.bioeconomics.enabled`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

## Self-Review

**Spec coverage:** §1→Task 4; §2 (exhibit, two-tool prescription, taxonomy, two traps, friendly failure)→Task 5; §3–4→Task 6; §5→Task 7; §6+appendix→Task 8; Tier-1 fixture→Tasks 1–2; Tier-2 two-sided assertions→Task 3; Tier-3 (declined)→no task, correctly (it is a decision to *not* test, mitigated by comments); success criteria→Task 9. The spec's open item (Java cross-file precedence) is Task 5 Step 1. **No gaps.**

**Placeholder scan:** No TBD/TODO. Every code step carries real, executed code — the fixture, the parse assertions, and both trap assertions were run against the live repo before this plan was written, and their outputs are the expectations quoted.

**Type consistency:** `OsmoseConfigReader.read()` → `dict[str, str]`, attrs `.skipped_lines` / `.deprecated_keys` / `.diagnostics`. `validate(cfg, mode)` → `list[UnknownKey]` with `.key` / `.suggestion` (`config_validation.py:265-267`). `EngineConfig.from_dict(cfg)` → attrs `.output_meantl` / `.economics_enabled`. `FIXTURES` (Task 1) is reused by Tasks 2–3; `_probe` / `minimal_cfg` (Task 3) are used only within Task 3. Names are consistent across tasks.
