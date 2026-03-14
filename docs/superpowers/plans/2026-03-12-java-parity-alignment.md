# OSMOSE Java Engine Parity Alignment — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align osmopy with the OSMOSE Java engine (`osmose-model/osmose`) by fixing schema key mismatches, expanding version migration coverage, adding missing output parsers, and exposing runner CLI flags.

**Architecture:** Six independent tasks that can be parallelized. Each task modifies a focused set of files, adds tests first (TDD), and produces a working commit. No task depends on another — they can be done in any order or concurrently via subagents.

**Tech Stack:** Python 3.12+, pytest, xarray, pandas, asyncio. No new dependencies required.

---

## Chunk 1: Schema & Migration Fixes (Tasks 1-3)

### Task 1: Fix Grid Schema Key Mismatch (P0)

The schema defines `grid.ncolumn` and `grid.nline` (pre-v3.3.3 names), but all modern configs (including the bundled Bay of Biscay) use `grid.nlon` and `grid.nlat`. After migration, `registry.match_field("grid.nlon")` fails silently — the validator cannot check grid dimensions for any post-v3.3.3 config.

**Files:**
- Modify: `osmose/schema/grid.py:14-28` (rename two field key_patterns)
- Modify: `tests/test_schema.py` (add grid key match test)
- Modify: `tests/test_integration.py:56-57` (update assertions if needed)

- [ ] **Step 1: Write failing test for grid.nlon matching**

In `tests/test_schema.py`, add:

```python
def test_grid_nlon_matches_schema():
    """Post-v3.3.3 grid keys must match schema fields."""
    from osmose.schema import build_registry

    registry = build_registry()
    field = registry.match_field("grid.nlon")
    assert field is not None, "grid.nlon must match a schema field"
    assert field.param_type.value == "int"

    field2 = registry.match_field("grid.nlat")
    assert field2 is not None, "grid.nlat must match a schema field"
    assert field2.param_type.value == "int"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schema.py::test_grid_nlon_matches_schema -v`
Expected: FAIL — `grid.nlon` returns `None` because schema has `grid.ncolumn`

- [ ] **Step 3: Update schema grid.py to use post-migration key names**

In `osmose/schema/grid.py`, change the two fields:

Line 15: change `key_pattern="grid.ncolumn"` to `key_pattern="grid.nlon"`
Line 20: change `description="Number of grid columns..."` to `description="Number of longitude cells in the grid"`

Line 24: change `key_pattern="grid.nline"` to `key_pattern="grid.nlat"`
Line 29: change `description="Number of grid lines..."` to `description="Number of latitude cells in the grid"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_schema.py::test_grid_nlon_matches_schema -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests pass. If any test references `grid.ncolumn` or `grid.nline`, update those assertions to use `grid.nlon` / `grid.nlat`.

- [ ] **Step 6: Commit**

```bash
git add osmose/schema/grid.py tests/test_schema.py
git commit -m "fix: update grid schema keys to post-v3.3.3 names (nlon/nlat)"
```

---

### Task 2: Expand Version Migration Table (P0)

osmopy has only 3 key renames in `_MIGRATIONS["4.3.0"]`. The Java engine's `Releases.java` has 12 incremental migration classes with hundreds of renames. We need to cover the most impactful migrations from v3.1 through v4.3.3.

The approach: add migration entries keyed by **source version** (the version being migrated FROM), so `migrate_config` can apply all relevant migrations sequentially. This matches the Java engine's `VersionManager` pattern.

**Files:**
- Modify: `osmose/demo.py:9-88` (rewrite migration system)
- Modify: `tests/test_demo.py` (add migration chain tests)

- [ ] **Step 1: Write failing tests for multi-version migration**

In `tests/test_demo.py`, add:

```python
def test_migrate_from_pre_3_2():
    """Configs before 3.2 should get population.initialization renamed."""
    config = {
        "osmose.version": "3.1.0",
        "population.initialization.biomass.sp0": "1000",
        "population.initialization.biomass.sp1": "2000",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "population.seeding.biomass.sp0" in result
    assert "population.seeding.biomass.sp1" in result
    assert "population.initialization.biomass.sp0" not in result
    assert result["osmose.version"] == "4.3.3"


def test_migrate_from_pre_4_2_3():
    """Configs before 4.2.3 should get plankton renamed to resource."""
    config = {
        "osmose.version": "4.2.2",
        "simulation.nplankton": "6",
        "plankton.name.plk0": "SmallPhyto",
        "plankton.tl.plk0": "1.0",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "simulation.nresource" in result
    assert "simulation.nplankton" not in result
    # plankton.* prefix keys should be renamed to resource.*
    assert "resource.name.plk0" in result
    assert "resource.tl.plk0" in result
    assert "plankton.name.plk0" not in result
    assert result["osmose.version"] == "4.3.3"


def test_migrate_from_pre_4_2_5():
    """Configs before 4.2.5 should get mortality.natural renamed."""
    config = {
        "osmose.version": "4.2.4",
        "mortality.natural.rate.sp0": "0.8",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "mortality.additional.rate.sp0" in result
    assert "mortality.natural.rate.sp0" not in result


def test_migrate_from_pre_3_3_3():
    """Configs before 3.3.3 should get grid.ncolumn renamed."""
    config = {
        "osmose.version": "3.3.0",
        "grid.ncolumn": "20",
        "grid.nline": "20",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "grid.nlon" in result
    assert "grid.nlat" in result
    assert "grid.ncolumn" not in result
    assert "grid.nline" not in result


def test_migrate_sequential_application():
    """All migrations applied in order for a very old config."""
    config = {
        "osmose.version": "3.0.0",
        "population.initialization.biomass.sp0": "1000",
        "grid.ncolumn": "20",
        "grid.nline": "20",
        "simulation.nplankton": "6",
        "mortality.natural.rate.sp0": "0.8",
    }
    result = migrate_config(config, target_version="4.3.3")
    # All renames should have been applied
    assert "population.seeding.biomass.sp0" in result
    assert "grid.nlon" in result
    assert "grid.nlat" in result
    assert "simulation.nresource" in result
    assert "mortality.additional.rate.sp0" in result
    # Old keys should be gone
    assert "population.initialization.biomass.sp0" not in result
    assert "grid.ncolumn" not in result
    assert "simulation.nplankton" not in result
    assert "mortality.natural.rate.sp0" not in result
    assert result["osmose.version"] == "4.3.3"


def test_migrate_already_at_target():
    """Config at target version should be unchanged."""
    config = {"osmose.version": "4.3.3", "simulation.nspecies": "8"}
    result = migrate_config(config, target_version="4.3.3")
    assert result == config


def test_migrate_no_version_key():
    """Config with no version key should apply all migrations."""
    config = {
        "population.initialization.biomass.sp0": "1000",
        "grid.ncolumn": "20",
    }
    result = migrate_config(config, target_version="4.3.3")
    assert "population.seeding.biomass.sp0" in result
    assert "grid.nlon" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_demo.py -v -k "migrate"`
Expected: Most new tests FAIL — current `migrate_config` only targets "4.3.0" with 3 renames.

- [ ] **Step 3: Rewrite migration system in demo.py**

Replace the `_MIGRATIONS` dict and `migrate_config` function in `osmose/demo.py` with:

```python
"""Demo scenario generation and config version migration."""

from __future__ import annotations

import shutil
from pathlib import Path

# Ordered migration steps: (min_version_exclusive, renames_dict)
# Each step applies if config version < step_version (or version is unknown).
# Mirrors the Java engine's Releases.java migration chain.
_MIGRATION_CHAIN: list[tuple[str, dict[str, str]]] = [
    (
        "3.2",
        {
            "population.initialization.biomass": "population.seeding.biomass",
            "population.initialization.abundance": "population.seeding.abundance",
        },
    ),
    (
        "3.3.3",
        {
            "grid.ncolumn": "grid.nlon",
            "grid.nline": "grid.nlat",
        },
    ),
    (
        "4.2.3",
        {
            "simulation.nplankton": "simulation.nresource",
            "plankton.name": "resource.name",
            "plankton.tl": "resource.tl",
            "plankton.size.min": "resource.size.min",
            "plankton.size.max": "resource.size.max",
            "plankton.accessibility2fish": "resource.accessibility2fish",
            "plankton.conversion2tons": "resource.conversion2tons",
            "plankton.file": "resource.file",
        },
    ),
    (
        "4.2.5",
        {
            "mortality.natural.rate": "mortality.additional.rate",
            "mortality.natural.larva.rate": "mortality.additional.larva.rate",
        },
    ),
    (
        "4.3.0",
        {
            "simulation.restart.enabled": "simulation.restart.enabled",
        },
    ),
]


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    try:
        return tuple(int(x) for x in v.split("."))
    except (ValueError, AttributeError):
        return (0,)
```

And rewrite `migrate_config`:

```python
def migrate_config(
    config: dict[str, str],
    target_version: str = "4.3.3",
) -> dict[str, str]:
    """Migrate config parameter names to a target OSMOSE version.

    Applies key renames sequentially from the config's current version
    through to target_version, following the Java engine's Releases.java chain.
    """
    current = config.get("osmose.version", "")
    if current == target_version:
        return dict(config)

    current_tuple = _version_tuple(current)
    target_tuple = _version_tuple(target_version)

    result = dict(config)

    for step_version, renames in _MIGRATION_CHAIN:
        step_tuple = _version_tuple(step_version)
        # Skip steps that are at or below current version
        if current and current_tuple >= step_tuple:
            continue
        # Skip steps beyond target version
        if step_tuple > target_tuple:
            break
        # Apply prefix-based renames (handles indexed keys like sp0, sp1, etc.)
        for old_prefix, new_prefix in renames.items():
            if old_prefix == new_prefix:
                continue
            keys_to_rename = [k for k in result if k == old_prefix or k.startswith(old_prefix + ".")]
            for key in keys_to_rename:
                new_key = new_prefix + key[len(old_prefix):]
                result[new_key] = result.pop(key)

    result["osmose.version"] = target_version
    return result
```

Note: The `_MIGRATION_CHAIN` entry for 4.3.0 with `simulation.restart.enabled` → itself is a no-op placeholder marking the version boundary. The `if old_prefix == new_prefix: continue` guard skips it.

- [ ] **Step 4: Run migration tests**

Run: `.venv/bin/python -m pytest tests/test_demo.py -v -k "migrate"`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add osmose/demo.py tests/test_demo.py
git commit -m "feat: expand version migration to cover v3.1 through v4.3.3"
```

---

### Task 3: Fix Writer Routing for mortality.additional Keys (P1)

The writer's `MASTER_PREFIXES` includes `"mortality.natural"` but not `"mortality.additional"`. After Task 2's migration renames `mortality.natural.rate.sp0` → `mortality.additional.rate.sp0`, the new keys fall through MASTER_PREFIXES, miss any ROUTING match, and end up in master by fallback. This is actually correct for the Bay of Biscay config (they're already in master), but `mortality.additional.larva.rate.sp{idx}` keys that originate in the species sub-file will migrate to master after a roundtrip. Fix: add `"mortality.additional"` to `MASTER_PREFIXES`.

Also: the schema's `species.py` defines `mortality.natural.rate.sp{idx}` (line 257) — this must be updated to `mortality.additional.rate.sp{idx}` to match post-v4.2.5 configs.

**Files:**
- Modify: `osmose/config/writer.py:45-51` (add `mortality.additional` prefix)
- Modify: `osmose/schema/species.py:257-288` (rename mortality keys)
- Modify: `tests/test_schema.py` (add test for new key names)
- Modify: `tests/test_integration.py` (add roundtrip test for mortality.additional keys)

- [ ] **Step 1: Write failing test for mortality.additional schema match**

In `tests/test_schema.py`, add:

```python
def test_mortality_additional_matches_schema():
    """Post-v4.2.5 mortality.additional keys must match schema."""
    from osmose.schema import build_registry

    registry = build_registry()
    field = registry.match_field("mortality.additional.rate.sp0")
    assert field is not None, "mortality.additional.rate.sp0 must match schema"
    assert field.indexed is True

    field2 = registry.match_field("mortality.additional.larva.rate.sp0")
    assert field2 is not None, "mortality.additional.larva.rate.sp0 must match schema"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schema.py::test_mortality_additional_matches_schema -v`
Expected: FAIL — schema has `mortality.natural.*`, not `mortality.additional.*`

- [ ] **Step 3: Update schema species.py mortality keys**

In `osmose/schema/species.py`:

Line 258: change `key_pattern="mortality.natural.rate.sp{idx}"` to `key_pattern="mortality.additional.rate.sp{idx}"`
Line 259: update description to `"Additional natural mortality rate per species"`

Line 269: change `key_pattern="mortality.natural.larva.rate.sp{idx}"` to `key_pattern="mortality.additional.larva.rate.sp{idx}"`
Line 270: update description to `"Additional natural mortality rate for larvae"`

Note: Unmigrated configs with `mortality.natural.*` keys will still route to the master file via the writer's fallback path, which is the correct behavior.

- [ ] **Step 4: Update writer MASTER_PREFIXES**

In `osmose/config/writer.py`, change `MASTER_PREFIXES` (lines 45-51):

```python
MASTER_PREFIXES: tuple[str, ...] = (
    "simulation.",
    "mortality.subdt",
    "mortality.additional",
    "mortality.starvation",
    "stochastic.",
)
```

Replace `"mortality.natural"` with `"mortality.additional"`.

- [ ] **Step 5: Run schema test**

Run: `.venv/bin/python -m pytest tests/test_schema.py::test_mortality_additional_matches_schema -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All pass. If any test references `mortality.natural.rate`, update to `mortality.additional.rate`.

- [ ] **Step 7: Commit**

```bash
git add osmose/schema/species.py osmose/config/writer.py tests/test_schema.py
git commit -m "fix: align mortality schema keys and writer routing to post-v4.2.5 names"
```

---

## Chunk 2: Runner & Results Expansion (Tasks 4-5)

### Task 4: Expose Java CLI Flags in Runner (P1)

The Java engine supports `-update` (version migration), `-force`, `-verbose`, `-quiet` flags that osmopy doesn't expose. Also, no default JVM memory is set — large models will OOM without `-Xmx`.

**Files:**
- Modify: `osmose/runner.py:34-54` (extend `_build_cmd`)
- Modify: `osmose/runner.py:56-140` (add `migrate` method, update `run` signature)
- Modify: `tests/test_runner.py:29-42` (update `_ScriptRunner._build_cmd` to accept new params)
- Modify: `tests/test_runner.py` (add tests for new flags, update existing cmd tests for `-Xmx`)

- [ ] **Step 1: Write failing tests for new CLI flags**

In `tests/test_runner.py`, add:

```python
def test_build_cmd_includes_verbose_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, verbose=True)
    assert "-verbose" in cmd


def test_build_cmd_includes_quiet_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, quiet=True)
    assert "-quiet" in cmd


def test_build_cmd_includes_xmx_default(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config)
    assert any(opt.startswith("-Xmx") for opt in cmd), "Should include -Xmx by default"


def test_build_cmd_xmx_override(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, java_opts=["-Xmx4g"])
    xmx_opts = [o for o in cmd if o.startswith("-Xmx")]
    # User-provided -Xmx should be present, default should not duplicate
    assert len(xmx_opts) == 1
    assert xmx_opts[0] == "-Xmx4g"


def test_build_cmd_update_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, update=True)
    assert "-update" in cmd


def test_build_cmd_update_force_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, update=True, force=True)
    assert "-update" in cmd
    assert "-force" in cmd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_runner.py -v -k "verbose or quiet or xmx or update"`
Expected: FAIL — `_build_cmd` doesn't accept these parameters

- [ ] **Step 3: Extend _build_cmd signature and implementation**

In `osmose/runner.py`, update `_build_cmd` (lines 34-54):

```python
def _build_cmd(
    self,
    config_path: Path,
    output_dir: Path | None = None,
    java_opts: list[str] | None = None,
    overrides: dict[str, str] | None = None,
    verbose: bool = False,
    quiet: bool = False,
    update: bool = False,
    force: bool = False,
) -> list[str]:
    """Build the Java command line for OSMOSE."""
    opts = list(java_opts or [])
    # Add default -Xmx if user hasn't specified one
    if not any(o.startswith("-Xmx") for o in opts):
        opts.append("-Xmx2g")

    cmd = [self.java_cmd, *opts, "-jar", str(self.jar_path), str(config_path)]

    if output_dir:
        cmd.append(f"-Poutput.dir.path={output_dir}")
    if verbose:
        cmd.append("-verbose")
    if quiet:
        cmd.append("-quiet")
    if update:
        cmd.append("-update")
    if force:
        cmd.append("-force")
    if overrides:
        for key, val in overrides.items():
            cmd.append(f"-P{key}={val}")
    return cmd
```

Also update `_ScriptRunner` in `tests/test_runner.py` (lines 29-42) to accept the new parameters:

```python
class _ScriptRunner(OsmoseRunner):
    """OsmoseRunner variant that invokes scripts via ``python <script>``."""

    def _build_cmd(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        **kwargs,  # Accept verbose, quiet, update, force without using them
    ) -> list[str]:
        cmd = [self.java_cmd, str(self.jar_path), str(config_path)]
        if output_dir:
            cmd.append(f"-Poutput.dir.path={output_dir}")
        if overrides:
            for key, value in overrides.items():
                cmd.append(f"-P{key}={value}")
        return cmd
```

Update existing tests that assert exact command lists to account for default `-Xmx2g` injection (e.g., `test_build_cmd_minimal` should expect `[..., "-Xmx2g", "-jar", ...]`).

- [ ] **Step 4: Add migrate convenience method**

In `osmose/runner.py`, add after the `cancel` method:

```python
async def migrate(
    self,
    config_path: Path,
    force: bool = False,
    timeout_sec: int | None = 120,
) -> RunResult:
    """Run the Java engine's built-in config version migration.

    Equivalent to: java -jar osmose.jar -update [-force] config.csv
    """
    return await self.run(
        config_path,
        java_opts=[],
        update=True,
        force=force,
        timeout_sec=timeout_sec,
    )
```

Update the `run` method signature to pass through the new flags:

```python
async def run(
    self,
    config_path: Path,
    output_dir: Path | None = None,
    java_opts: list[str] | None = None,
    overrides: dict[str, str] | None = None,
    on_progress: Callable[[str], None] | None = None,
    timeout_sec: int | None = None,
    verbose: bool = False,
    quiet: bool = False,
    update: bool = False,
    force: bool = False,
) -> RunResult:
```

And update the `_build_cmd` call inside `run` to pass through the new flags.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: All PASS

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add osmose/runner.py tests/test_runner.py
git commit -m "feat: expose Java CLI flags (-update, -verbose, -quiet, -Xmx) in runner"
```

---

### Task 5: Expand Results Parser Coverage (P1)

osmopy's `_EXPORT_MAP` covers 16 output types. The Java engine produces ~50+ types. Add the most commonly needed missing parsers: fishery outputs, bioenergetics outputs, and additional distribution types.

**Files:**
- Modify: `osmose/results.py:258-278` (expand `_EXPORT_MAP`)
- Modify: `osmose/results.py` (add new accessor methods)
- Modify: `tests/test_results.py` (add tests for new output types)

- [ ] **Step 1: Write failing tests for new output types**

In `tests/test_results.py`, add:

```python
def test_export_map_includes_fishery_outputs():
    """_EXPORT_MAP must include fishery output types."""
    from osmose.results import OsmoseResults

    assert "fishery_yield" in OsmoseResults._EXPORT_MAP
    assert "fishery_yield_by_age" in OsmoseResults._EXPORT_MAP
    assert "fishery_yield_by_size" in OsmoseResults._EXPORT_MAP


def test_export_map_includes_bioen_outputs():
    """_EXPORT_MAP must include bioenergetics output types."""
    from osmose.results import OsmoseResults

    assert "bioen_ingestion" in OsmoseResults._EXPORT_MAP
    assert "bioen_maintenance" in OsmoseResults._EXPORT_MAP
    assert "bioen_net_energy" in OsmoseResults._EXPORT_MAP


def test_export_map_includes_additional_distributions():
    """_EXPORT_MAP must include additional distribution outputs."""
    from osmose.results import OsmoseResults

    assert "abundance_by_tl" in OsmoseResults._EXPORT_MAP
    assert "yield_n_by_age" in OsmoseResults._EXPORT_MAP
    assert "yield_n_by_size" in OsmoseResults._EXPORT_MAP
    assert "mean_size_by_age" in OsmoseResults._EXPORT_MAP
    assert "mean_tl_by_age" in OsmoseResults._EXPORT_MAP
    assert "mean_tl_by_size" in OsmoseResults._EXPORT_MAP


def test_fishery_yield_method_exists():
    """OsmoseResults must have a fishery_yield method."""
    assert hasattr(OsmoseResults, "fishery_yield")


def test_bioen_ingestion_method_exists():
    """OsmoseResults must have a bioen_ingestion method."""
    assert hasattr(OsmoseResults, "bioen_ingestion")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_results.py -v -k "fishery or bioen or additional"`
Expected: FAIL — keys not in `_EXPORT_MAP`, methods don't exist

- [ ] **Step 3: Add new accessor methods**

In `osmose/results.py`, add after the existing distribution methods (after line ~155):

```python
# Fishery outputs
def fishery_yield(self, species: str | None = None) -> pd.DataFrame:
    """Read fishery-specific yield (biomass)."""
    return self._read_species_output("fisheryYieldBiomass", species)

def fishery_yield_by_age(self, species: str | None = None) -> pd.DataFrame:
    """Read fishery yield by age class."""
    return self._read_2d_output("fisheryYieldByAge", species)

def fishery_yield_by_size(self, species: str | None = None) -> pd.DataFrame:
    """Read fishery yield by size class."""
    return self._read_2d_output("fisheryYieldBySize", species)

# Bioenergetics outputs
def bioen_ingestion(self, species: str | None = None) -> pd.DataFrame:
    """Read bioenergetics ingestion rate."""
    return self._read_species_output("bioenIngestion", species)

def bioen_maintenance(self, species: str | None = None) -> pd.DataFrame:
    """Read bioenergetics maintenance cost."""
    return self._read_species_output("bioenMaintenance", species)

def bioen_net_energy(self, species: str | None = None) -> pd.DataFrame:
    """Read bioenergetics net energy."""
    return self._read_species_output("bioenEnet", species)
```

- [ ] **Step 4: Expand _EXPORT_MAP**

In `osmose/results.py`, update `_EXPORT_MAP` to include all new types:

```python
_EXPORT_MAP: dict[str, tuple[str, str]] = {
    # 1D time series
    "biomass": ("biomass", "1d"),
    "abundance": ("abundance", "1d"),
    "yield": ("yield", "1d"),
    "mortality": ("mortality", "1d"),
    "trophic": ("meanTL", "1d"),
    "yield_n": ("yieldN", "1d"),
    "mortality_rate": ("mortalityRate", "1d"),
    # Fishery 1D
    "fishery_yield": ("fisheryYieldBiomass", "1d"),
    # Bioenergetics 1D
    "bioen_ingestion": ("bioenIngestion", "1d"),
    "bioen_maintenance": ("bioenMaintenance", "1d"),
    "bioen_net_energy": ("bioenEnet", "1d"),
    # 2D distributions
    "biomass_by_age": ("biomassByAge", "2d"),
    "biomass_by_size": ("biomassBySize", "2d"),
    "biomass_by_tl": ("biomassByTL", "2d"),
    "abundance_by_age": ("abundanceByAge", "2d"),
    "abundance_by_size": ("abundanceBySize", "2d"),
    "abundance_by_tl": ("abundanceByTL", "2d"),
    "yield_by_age": ("yieldByAge", "2d"),
    "yield_by_size": ("yieldBySize", "2d"),
    "yield_n_by_age": ("yieldNByAge", "2d"),
    "yield_n_by_size": ("yieldNBySize", "2d"),
    "mean_size_by_age": ("meanSizeByAge", "2d"),
    "mean_tl_by_age": ("meanTLByAge", "2d"),
    "mean_tl_by_size": ("meanTLBySize", "2d"),
    # Fishery 2D
    "fishery_yield_by_age": ("fisheryYieldByAge", "2d"),
    "fishery_yield_by_size": ("fisheryYieldBySize", "2d"),
    # Special
    "diet": ("dietMatrix", "special_diet"),
    "size_spectrum": ("sizeSpectrum", "special_spectrum"),
}
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_results.py -v`
Expected: All PASS

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add osmose/results.py tests/test_results.py
git commit -m "feat: expand results parser with fishery, bioenergetics, and distribution outputs"
```

---

## Chunk 3: Validator Improvements & Schema Expansion (Task 6)

### Task 6: Strengthen Config Validator (P1)

The validator has several gaps identified in the analysis:
- `check_file_references` uses string heuristic (`"file" in key`) instead of schema `FILE_PATH` type
- No cross-field validation (e.g., `maturity.size < linf`)
- `check_species_consistency` only checks focal species, not resources
- No check for resource species or background species count consistency
- `validate_config` doesn't check ENUM values

**Files:**
- Modify: `osmose/config/validator.py` (improve all four functions)
- Modify: `tests/test_validator.py` (add targeted tests)

- [ ] **Step 1: Write failing tests for validator improvements**

In `tests/test_validator.py`, add:

```python
def test_check_file_references_uses_schema(tmp_path):
    """check_file_references should use FILE_PATH schema type, not string heuristic."""
    from osmose.config.validator import check_file_references
    from osmose.schema import build_registry

    registry = build_registry()
    config = {
        "reproduction.season.file.sp0": "nonexistent.csv",
        "grid.netcdf.file": "nonexistent.nc",
        "species.name.sp0": "Anchovy",  # contains no "file" but is STRING
    }
    missing = check_file_references(config, str(tmp_path), registry)
    assert len(missing) == 2
    assert any("reproduction.season.file.sp0" in m for m in missing)
    assert any("grid.netcdf.file" in m for m in missing)


def test_check_species_consistency_checks_resources():
    """check_species_consistency should validate resource species too."""
    from osmose.config.validator import check_species_consistency

    config = {
        "simulation.nspecies": "2",
        "simulation.nresource": "2",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        # sp2 and sp3 (resources) are missing
    }
    warnings = check_species_consistency(config)
    assert len(warnings) == 2
    assert any("sp2" in w for w in warnings)
    assert any("sp3" in w for w in warnings)


def test_validate_config_checks_enum_values():
    """validate_config should reject invalid ENUM values."""
    from osmose.config.validator import validate_config
    from osmose.schema import build_registry

    registry = build_registry()
    config = {
        "species.type.sp0": "invalid_type",
    }
    errors, _ = validate_config(config, registry)
    assert len(errors) >= 1
    assert any("invalid_type" in e for e in errors)


def test_validate_config_accepts_valid_enum():
    """validate_config should accept valid ENUM values."""
    from osmose.config.validator import validate_config
    from osmose.schema import build_registry

    registry = build_registry()
    config = {
        "species.type.sp0": "focal",
    }
    errors, _ = validate_config(config, registry)
    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_validator.py -v -k "schema or resource or enum"`
Expected: FAIL — current functions don't accept `registry` param, don't check resources, don't check enums

- [ ] **Step 3: Update check_file_references to use schema**

In `osmose/config/validator.py`, update `check_file_references` (lines 63-82):

```python
def check_file_references(
    config: dict[str, str],
    base_dir: str,
    registry=None,
) -> list[str]:
    """Check that file-referencing parameters point to existing files.

    If registry is provided, uses FILE_PATH schema type for detection.
    Otherwise falls back to heuristic (key contains 'file').
    """
    missing = []
    base = Path(base_dir)
    for key, value in config.items():
        is_file_param = False
        if registry is not None:
            field = registry.match_field(key)
            if field is not None and field.param_type == ParamType.FILE_PATH:
                is_file_param = True
        else:
            is_file_param = "file" in key.lower()

        if not is_file_param:
            continue
        if not value or value.lower() in ("null", "none"):
            continue
        ref = Path(value)
        if not ref.is_absolute():
            ref = base / ref
        if not ref.exists():
            missing.append(f"File not found for '{key}': {ref}")
    return missing
```

- [ ] **Step 4: Update check_species_consistency to check resources**

In `osmose/config/validator.py`, update `check_species_consistency` (lines 85-100):

```python
def check_species_consistency(config: dict[str, str]) -> list[str]:
    """Check that species.name keys exist for all declared species and resources."""
    warnings = []
    nspecies = int(config.get("simulation.nspecies", "0"))
    nresource = int(config.get("simulation.nresource", "0"))

    for i in range(nspecies):
        key = f"species.name.sp{i}"
        if key not in config:
            warnings.append(f"Missing focal species name: {key}")

    for i in range(nresource):
        idx = nspecies + i
        key = f"species.name.sp{idx}"
        if key not in config:
            warnings.append(f"Missing resource species name: {key}")

    return warnings
```

- [ ] **Step 5: Add ENUM validation to validate_config**

In `osmose/config/validator.py`, update `validate_config` (lines 10-41). Add an `elif` branch after the BOOL check:

```python
        elif field.param_type == ParamType.ENUM:
            if field.choices and value not in field.choices:
                errors.append(
                    f"Invalid value for '{key}': '{value}' "
                    f"(expected one of {field.choices})"
                )
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_validator.py -v`
Expected: All PASS

- [ ] **Step 7: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add osmose/config/validator.py tests/test_validator.py
git commit -m "feat: strengthen validator with schema-aware file refs, resource checks, and enum validation"
```

---

## Summary

| Task | Priority | Files Modified | Tests Added | Estimated Steps |
|------|----------|---------------|-------------|-----------------|
| 1. Grid schema key mismatch | P0 | `schema/grid.py`, `test_schema.py` | 1 | 6 |
| 2. Version migration expansion | P0 | `demo.py`, `test_demo.py` | 8 | 6 |
| 3. Mortality schema + writer fix | P1 | `schema/species.py`, `config/writer.py`, `test_schema.py` | 1 | 7 |
| 4. Runner CLI flags | P1 | `runner.py`, `test_runner.py` | 6 | 7 |
| 5. Results parser expansion | P1 | `results.py`, `test_results.py` | 5 | 7 |
| 6. Validator improvements | P1 | `config/validator.py`, `test_validator.py` | 4 | 8 |

**All 6 tasks are independent** — they can be executed in parallel via subagent-driven development.

**Total: 6 tasks, 25 new tests, 41 steps, 6 commits.**
