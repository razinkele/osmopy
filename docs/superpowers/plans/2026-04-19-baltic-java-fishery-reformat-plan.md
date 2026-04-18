# Baltic Java Fishery Reformat Plan

> **Front 2** of the 2026-04-18 post-session roadmap. Short and mechanical; one or two commits.

**Goal:** Make the Baltic example runnable on the Java 4.3.3 engine without modifying fishery semantics. After this plan lands, `java -jar osmose-java/osmose_4.3.3-jar-with-dependencies.jar data/baltic/baltic_all-parameters.csv output/` must boot past fishery initialization without a `[severe]` error, and the existing Python-side test suite must stay green.

**Out of scope:** numerical cross-validation Python ↔ Java (a separate parity run; the roadmap's SP-4 output system work is what consumes that). Also out of scope: changes to fishing semantics, rates, selectivity, seasonality, or the fleet count.

## Root cause (empirical)

Running Java against the current `data/baltic/baltic_all-parameters.csv` produces:

```
osmose[warn] No map assigned for fishery trawlcod year 0 step 0
...
osmose[severe] No catchability found for fishery trawlcod
```

Trace: `osmose-master/java/src/main/java/fr/ird/osmose/process/mortality/FishingGear.java:106`

```java
name = cfg.getString("fisheries.name.fsh" + fileFisheryIndex)
          .replaceAll("_", "").replaceAll("-", "");
if (!this.getName().matches("^[a-zA-Z0-9]*$")) {
    error("Fishery name must contain alphanumeric characters only...", null);
}
```

Java strips `_` and `-` from every fishery name read from `fisheries.name.fshN`, then looks up the catchability and discards matrices and the fishery-movement maps using the **stripped** name. Our Baltic config currently uses `trawl_cod`, `pelagic_herring`, … — so the stripped `trawlcod` lookup fails against the matrix column header `trawl_cod`. Same problem for `fisheries.movement.fishery.mapN` values and the fishery names the map-set compares.

The Python engine is not affected — it looks up catchability by position (row/column index), not by name — so renaming is a Java-only fix that the Python path tolerates unchanged. Verified at `osmose/engine/config.py:263-280` (uses `pd.read_csv(catch_path, index_col=0)` and matches *species* names; fishery columns are positional).

## The eight renames

| Current | Java-stripped | New (both) |
|---|---|---|
| `trawl_cod`             | `trawlcod`             | `trawlcod`             |
| `pelagic_herring`       | `pelagicherring`       | `pelagicherring`       |
| `pelagic_sprat`         | `pelagicsprat`         | `pelagicsprat`         |
| `trawl_flounder`        | `trawlflounder`        | `trawlflounder`        |
| `coastal_perch`         | `coastalperch`         | `coastalperch`         |
| `coastal_pikeperch`     | `coastalpikeperch`     | `coastalpikeperch`     |
| `gill_smelt`            | `gillsmelt`            | `gillsmelt`            |
| `coastal_stickleback`   | `coastalstickleback`   | `coastalstickleback`   |

Adopting the Java-stripped form as the canonical name gives us one consistent spelling across config and both engines. Underscores are still free to appear in **parameter keys** (e.g. `fisheries.rate.base.fsh0`) — the stripping only applies to fishery-name *values*, not config keys.

## Files touched

- `data/baltic/baltic_param-fishing.csv` — 16 value renames (8 × `fisheries.name.fshN`, 8 × `fisheries.movement.fishery.mapN`).
- `data/baltic/fishery-catchability.csv` — 8 column headers.
- `data/baltic/fishery-discards.csv` — 8 column headers.

Nothing else references these strings. (Verified by grep below.)

## Pre-flight

- `.venv/bin/python -m pytest -q` — baseline must show 2445 passed, ruff clean.
- `java -version` — 17+ available (confirmed 21.0.10 in current env).
- `osmose-java/osmose_4.3.3-jar-with-dependencies.jar` present and executable.

## Task 1 — Inventory references

- [ ] **Step 1.** Grep for every fishery-name string across the Baltic tree:

  ```bash
  .venv/bin/python -c "
  import pathlib
  names = ['trawl_cod','pelagic_herring','pelagic_sprat','trawl_flounder',
          'coastal_perch','coastal_pikeperch','gill_smelt','coastal_stickleback']
  for p in sorted(pathlib.Path('data/baltic').rglob('*')):
      if p.is_file() and p.suffix in {'.csv','.md','.json'}:
          text = p.read_text(errors='ignore')
          hits = [n for n in names if n in text]
          if hits:
              print(f'{p}: {hits}')
  "
  ```

  Expected set of referencing files: `baltic_param-fishing.csv`, `fishery-catchability.csv`, `fishery-discards.csv`. If anything else turns up (e.g. a map filename, a README, a calibration results JSON), add it to the rename list in Task 2 before editing.

## Task 2 — Rename in `baltic_param-fishing.csv`

- [ ] **Step 1.** For each of the eight fishery names, apply the rename to:
  - `fisheries.name.fshN` value (8 rows)
  - `fisheries.movement.fishery.mapN` value (8 rows)

  Use `Edit` with `replace_all: true` one name at a time — the old names are distinctive enough that there's no false-positive risk.

- [ ] **Step 2.** `jq`-equivalent sanity check: after editing, grep the file for any remaining underscore in a fishery-name position.

  ```bash
  grep -E "fisheries\.(name|movement\.fishery\.map)[0-9]+;[a-z_]+_[a-z]+" data/baltic/baltic_param-fishing.csv
  ```

  Expected: no output. If a line matches, undo and reapply.

## Task 3 — Rewrite catchability and discards matrix headers

- [ ] **Step 1.** Rewrite the header row in `data/baltic/fishery-catchability.csv` (line 1). The shape and order must stay identical — only the eight column names change. Example target:

  ```
  ,trawlcod,pelagicherring,pelagicsprat,trawlflounder,coastalperch,coastalpikeperch,gillsmelt,coastalstickleback
  cod,1,0,0,0,0,0,0,0
  herring,0,1,0,0,0,0,0,0
  ...
  ```

- [ ] **Step 2.** Same for `data/baltic/fishery-discards.csv`.

- [ ] **Step 3.** Verify matrix shape and values are unchanged:

  ```bash
  .venv/bin/python -c "
  import pandas as pd
  c = pd.read_csv('data/baltic/fishery-catchability.csv', index_col=0)
  d = pd.read_csv('data/baltic/fishery-discards.csv', index_col=0)
  print('catchability shape', c.shape, 'sum', c.values.sum())
  print('discards shape', d.shape, 'sum', d.values.sum())
  print('catchability cols', list(c.columns))
  print('discards cols', list(d.columns))
  "
  ```

  Expected: shapes `(8, 8)` each; `catchability` sum = 8.0 (identity matrix); `discards` sum = 0.0; column names match the stripped form from the rename table above.

## Task 4 — Confirm Java boots past fishery init

- [ ] **Step 1.** Run Java against the Baltic config with a short horizon:

  ```bash
  mkdir -p /tmp/baltic_java_check
  java -jar osmose-java/osmose_4.3.3-jar-with-dependencies.jar \
      data/baltic/baltic_all-parameters.csv /tmp/baltic_java_check/ 2>&1 | tail -30
  ```

  Expected: no `[severe] No catchability found` and no `[severe]` related to missing fishery maps. Benign warnings (`Could not find Boolean parameter ...`, `Did not find parameter population.seeding.year.max`, etc.) are fine — those are unrelated OSMOSE defaults.

- [ ] **Step 2.** If Java surfaces a *new* severe error after the fishery block, stop and expand the plan rather than patch around it. Candidates we'd expect to see: size-ratio warnings (already present pre-fix; not a blocker), or an LTL/biomass-forcing issue (out of scope; separate plan).

## Task 5 — Python regression gate

- [ ] **Step 1.** Re-run the full Python test suite:

  ```bash
  .venv/bin/python -m pytest -q
  ```

  Expected: still 2445 passed (zero regressions). The catchability file is read by `osmose/engine/config.py:263-280` by position; Python never looks at the column names, so the rename is invisible to the engine.

- [ ] **Step 2.** Lint:

  ```bash
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: `All checks passed!`

- [ ] **Step 3.** Re-run the Baltic ICES validator to confirm the report regenerates unchanged:

  ```bash
  .venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report
  ```

  The F/B comparison values read from `baltic_param-fishing.csv` by `fisheries.rate.base.fshN` (prefix-based) — those keys are untouched. Output should match the prior report values exactly.

## Task 6 — Commit and push

- [ ] **Step 1.** Stage the three touched files plus the plan:

  ```bash
  git add data/baltic/baltic_param-fishing.csv \
          data/baltic/fishery-catchability.csv \
          data/baltic/fishery-discards.csv \
          docs/superpowers/plans/2026-04-19-baltic-java-fishery-reformat-plan.md
  ```

- [ ] **Step 2.** Commit:

  ```
  data(baltic): strip underscores from fishery names for Java-engine compatibility

  FishingGear.java:106 strips `_` and `-` from fishery-name values, then looks
  up catchability/discards/fishery-movement maps by the stripped name. With
  underscored names the Baltic config fails at Java boot with "[severe] No
  catchability found for fishery trawlcod" and all 8 fisheries are deactivated.

  Rename the eight fishery names in baltic_param-fishing.csv (fisheries.name.*
  and fisheries.movement.fishery.map*) and the matching column headers in
  fishery-catchability.csv and fishery-discards.csv:
    trawl_cod         -> trawlcod
    pelagic_herring   -> pelagicherring
    pelagic_sprat     -> pelagicsprat
    trawl_flounder    -> trawlflounder
    coastal_perch     -> coastalperch
    coastal_pikeperch -> coastalpikeperch
    gill_smelt        -> gillsmelt
    coastal_stickleback -> coastalstickleback

  Python engine unaffected: catchability looked up by row/column index, not
  name. All 2445 tests remain green. Java boots past fishery init cleanly.
  ```

- [ ] **Step 3.** Push:

  ```bash
  git push origin master
  ```

- [ ] **Step 4.** Append an `Added`/`Fixed` entry to `CHANGELOG.md` under `[Unreleased]`.

## Self-review checklist

- **Semantic preservation.** None of the renames touch fishing rates, selectivity, seasonality, or fleet count. The fishery→species pairing in the catchability matrix is preserved (same rows, same columns, same values). ✓
- **Python parity.** `osmose/engine/config.py` reads catchability by position (`iloc`), not by name, so Python is blind to column-name changes. Verified by reading the code. ✓
- **Java compatibility.** Java's FishingGear regex `^[a-zA-Z0-9]*$` passes for all eight stripped names (all-alphanumeric). ✓
- **Key preservation.** `fisheries.*` config keys (`fisheries.rate.base.fsh0`, etc.) are unchanged — only *values* inside those keys change. ✓
- **Reversibility.** If any issue surfaces, `git revert` undoes the rename cleanly; no downstream schema changes. ✓
- **Scope limit.** No fishery semantics, no new fisheries, no map changes, no new keys. ✓
