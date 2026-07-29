---
name: stale-fixture-sweep
description: Use when a species is added, removed, renamed, or disaggregated in an OSMOSE config so the species count or sp{N} indices shift — e.g. splitting cod into cod_west/cod_east. Symptoms include KeyError on a species column, predation-matrix shape mismatch, wrong Sobol/param counts, fishery-name or discards mismatch, Java aborting on a missing prey, or tests that still pass but no longer assert anything.
---

# Stale Fixture Sweep

## Overview

OSMOSE numbers focal, LTL (resource), and background species contiguously in **one**
`species.name.sp{idx}` namespace. Inserting a focal species renumbers everything above it.
The Baltic layout after the cod split:

| Block | Indices | Note |
|---|---|---|
| focal | `sp0`–`sp8` | `sp8` = cod_east (appended, not inserted) |
| LTL / resource | `sp9`–`sp14` | was `sp8`–`sp13` |
| background | `sp15`, `sp16` | GreySeal, Cormorant — was `sp14`, `sp15` |

**Core principle: most of the breakage is silent.** A KeyError is the easy case. The
expensive cases are references that still resolve — to the wrong species.

## The three silent failure modes

Check these first; they do not show up as red tests.

1. **A stale index still resolves, to a different species.** `sp14` used to be GreySeal,
   a background predator; it is now an LTL resource. Tests setting `fr={"sp14": ...}` kept
   passing while testing nothing — the GreySeal intent was lost. Any hardcoded `sp{N}` at or
   above the insertion point is suspect, and a passing test is not evidence it is fine.
2. **Silent zero by name-omission.** `fishery-discards.csv` kept an aggregate `cod` row.
   Python assigns any species missing from the file a zero rate silently; Java aborts on the
   missing prey. Python-green does not mean Java-green.
3. **A test pins the stale index as its expected value.** The worst case is not a skip —
   it is `tests/test_phase14_scaffolding.py` and `tests/test_calibrate_baltic_a2.py`
   asserting the wrong indices and passing, indistinguishable from real coverage. Correcting
   the source turns them red, which reads as a regression to anyone without the history.
   **A test asserting a stale index is evidence of the bug, not against it.** Grep the suite
   for hardcoded `sp{N}` literals naming the module you are fixing, and expect to update them
   in the same commit. Skip-guards (`require_baltic_phase12` and siblings in
   `tests/_data_guards.py`) are the milder version of the same problem — they at least show
   as `s` in pytest output.

## Checklist

Work the table. Every row is something that has actually gone stale before.

| Artifact | What breaks | Detect |
|---|---|---|
| `scripts/calibrate_baltic.py` `SPECIES_NAMES` | Canonical focal-name list | Read it for the layout, but do **not** treat it as an all-clear: it is the first thing anyone updates and therefore the least likely to be stale. The damage is in the LTL/background index literals below |
| Index literals for the **resource and background blocks** | Silently retarget — the whole point of failure mode 1 | `tests/test_baltic_species_index_layout.py` pins the calibrator's index sets to the real blocks by `species.type`; extend it when you add a new index constant, rather than resolving greps by hand |
| Derived counts in docstrings, `print()`, and `--help` | No test asserts user-facing strings, so they rot unnoticed | Import the module, call every `get_*_params()`, and diff `len(keys)` against the count written in its own docstring |
| `data/baltic/reference/biomass_targets.csv` | Aggregate species row must be replaced by the split rows | `tests/test_calibration_targets.py` |
| `fishery-catchability.csv` / `fishery-discards.csv` | Rows and columns must match each other exactly | `tests/test_baltic_fishery_matrices.py` |
| `baltic_param-fishing.csv` `fisheries.name.fshN` | Java strips `_` and `-` from the gear name but not the map value → silent fishery deactivation | `tests/test_baltic_java_compat.py`; commit the stripped form (`trawlcodeast`, never `trawlcod_east`) |
| `data/baltic/predation-accessibility.csv` | Row/column count; keyed by **name**, so an index shift does not touch it, but a rename or a new predator does | `pandas.read_csv(..., index_col=0).shape` — Baltic is currently 15×16 (the extra column is the Cormorant predator) |
| FR arrays | Length is `n_species + n_background` (11); resource column base is `config.n_species` (9) | `tests/test_engine_functional_response.py` |
| Sobol / phase param counts | Derived counts shift: phase12 27→30 params, phase13 shape 8→9 and ssbhalf 7→9, Sobol `8*(2*30+2)`=496 | `tests/test_sensitivity_phase12.py`, `tests/test_calibrate_baltic_parallelism.py` |
| Cod diagnostics reading `df["cod"]` | KeyError, or a wrong-stock comparison | Use `osmose.results.total_cod(df)` — see below |
| Movement maps / `movement.file.mapN` | Map indices shift with the fishery/species block | `tests/test_baltic_cod_ew_maps.py` |

## Prefer an aggregation contract over per-site edits

*Applies to files that read a per-species DataFrame by name (biomass, SSB, yield). Skip this
section for files that only build config keys.*

When a split breaks many call sites that only wanted the *total*, add one helper instead of
editing each. `osmose/results.py::total_cod` sums `cod_west + cod_east` and falls back to an
aggregate `cod` column, so diagnostics work under both layouts. That single contract replaced
14 individual fixes. Reach for the same shape on the next disaggregation.

## Detection recipe

```bash
# 1. name-based staleness (aggregate species name that no longer exists)
grep -rn "\[.cod.\]\|\"cod\"\|'cod'" tests/ scripts/ osmose/ | grep -v "cod_west\|cod_east\|total_cod"
# 2. index-based staleness — pattern MUST start at the insertion point, not above it
grep -rnE "sp(8|9|1[0-9])\b" tests/ scripts/ --include=*.py
.venv/bin/python -m pytest -q
```

**Start the index pattern at the insertion index.** cod_east was appended at `sp8`, so
everything from `sp8` up moved. A pattern anchored higher (`sp1[3-9]`) finds the background
pair and misses the entire resource block — in one real audit that was six of eight defects
missed, including four identical `[8, 9, 10, 11, 12, 13]` loops.

**Both greps localise only; neither tells you which species an index now hits.** Resolve
every hit against `species.type.sp{N}` / `species.name.sp{N}` in the three config CSVs.

**Grep 1 has a high false-positive rate.** Many tests define their own synthetic species
literally named `"cod"` and are unaffected — `test_engine_timeseries`, `test_uq_posterior`,
`test_trophic_network`, `test_fmsy_sweep`, `test_plotting`. Only hits that read the real
Baltic config or `calibrate_baltic` are stale. "Fixing" the synthetic fixtures is a common way
to burn an hour and break working tests.

## Common mistakes

- **Stopping when the suite is green.** Failure modes 1 and 3 are green by construction.
  Finish the grep even after `pytest` passes.
- **Not cross-checking Java.** Python tolerates missing species by name; Java aborts. Run the
  Java cross-check before calling the sweep done.
- **Editing a stale index to "the next one up" without reading the config.** Confirm the real
  index in `baltic_param-species.csv` / `baltic_param-background.csv`.
- **Treating a docstring as documentation.** Layout comments (`8 focal species`, `sp14 =
  GreySeal`) go stale silently and mislead the next sweep. Update them in the same pass.
- **Trusting the config file headers.** The header comments in `baltic_param-ltl.csv` and
  `baltic_param-background.csv` — the files you would open to confirm an index — have
  themselves been stale. Read the `species.type.sp{N}` keys, not the comment above them.
- **Renumbering an index that is pinned to a stored artifact.** Before changing a constant,
  check whether a committed results JSON or a frozen base encodes the old numbering; if it
  does, renumbering silently reinterprets stored parameters. `data/baltic/calibration_results/`
  is gitignored except `phase13_equilibrium.json`, so local scratch there is not a reason to
  keep a bug — but a *tracked* artifact is.
