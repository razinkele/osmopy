# Fishing-vs-natural mortality (F/M) diagnostics — Design

**Date:** 2026-06-03
**Status:** RESCOPED after a 4-angle in-loop plan review (see "Rescope" below). Approved direction.
**Precedent:** PR #46 ICES output validator (`osmose/validation/ices.py` + `scripts/validate_outputs_vs_ices.py`).

## Rescope (why this is narrower than the first draft)

The first draft aimed at full stock-status diagnostics — F/M, **B/Bmsy, F/Fmsy, Kobe plot**. A
4-angle review, verified against real code/data, killed the marquee and corrected three false
premises the reconnaissance had *inferred* rather than executed:

- **`results.mortality()` currently CRASHES** (`pandas.errors.ParserError`) on the real
  Java-written `mortalityRate-{sp}_Simu0.csv` — it has a 1-line preamble + 2 header rows
  (cause, life-stage) + a trailing comma (25 vs 26 fields), which `_detect_preamble_lines`
  mis-skips. This is a **pre-existing bug**, independent of this feature.
- **ICES reference points are stored as JSON strings** (`"fmsy": "0.34"`), so any arithmetic
  on them needs `float()` coercion.
- **B/Bmsy + F/Fmsy + Kobe would cover exactly ONE Baltic species (sprat)** — flounder is
  index-unit, herring and cod are mixed index+tonnes, the coastal species have no ICES
  assessment. A single-point Kobe plot. Plus two methodological traps: `MSY Btrigger` (≈Bpa,
  *below* Bmsy) as a Bmsy proxy *overstates* stock health (can paint an overfished stock
  "green"), and OSMOSE's Recruits-stage F is not the age-ranged ICES Fbar that Fmsy is
  defined on.

So this feature ships the **robust, broadly-useful core only**: a per-species **F/M
(fishing vs natural mortality) ratio** diagnostic — available for **all 8 species** (no ICES
reference points needed) — plus the mortality-reader bug fix it depends on, plus a bar chart
and CLI. B/Bmsy / F/Fmsy / Kobe are recorded as a **deferred follow-up** for a future config
with broad ICES tonnes-unit coverage.

## Motivation

OSMOSE records cause-resolved mortality per species (predation, starvation, additional,
fishing, …) but offers no at-a-glance read of **fishing pressure relative to natural
mortality**. F/M > 1 means fishing removes more than nature does — a standard, reference-point-
free overexploitation signal that works for every modelled species, including the coastal
ones with no ICES assessment. This feature surfaces it.

## Verified context (executed, not inferred)

- **Output cadence:** biomass and mortality are BOTH saved every `output.recordfrequency.ndt`
  steps and have **equal row counts**. For the shipped baltic/eec configs
  `recordfrequency.ndt == ndtPerYear == 24` → **1 saved row per year**, and each mortality row
  is already the **annual sum** (engine sums mortality per save window; `simulate.py:1157`).
- **Steps-per-year is NOT inferable from row counts** (both series always equal length). It
  MUST be derived from config: `steps_per_year = ndtPerYear / recordfrequency.ndt`. For the
  shipped configs that is 1. The F/M aggregation must use this, and **fail loudly** if it
  can't be determined.
- **`mortalityRate` CSV:** preamble line 0; line 1 = cause header (`Mpred`,`Mstarv`,`Madd`,`F`,
  `Zout`,`Mfor`,`Mdis`,`Mage`, each ×3); line 2 = stage (`Eggs`,`Pre-recruits`,`Recruits`);
  data rows have a trailing comma. Correct read: `pd.read_csv(path, skiprows=1, header=[0,1])`
  then drop the all-NaN trailing column → a `(cause, stage)` MultiIndex with `('F','Recruits')`
  etc. accessible. (`results.mortality()` cannot be used until its reader is fixed.)
- **Per-species mortality file path:** `{output_dir}/Mortality/{prefix}_mortalityRate-{sp}_Simu0.csv`
  (confirmed for baltic + eec_full).
- **`OsmoseResults(output_dir, prefix="osm", strict=...)`** — the working CLI
  `validate_outputs_vs_ices.py:102` uses `strict=False`; mirror that so missing/empty outputs
  WARN-skip rather than abort.
- **Biomass-vs-ICES-envelope status already exists** (`compare_outputs_to_ices`, PR #46) — this
  feature does NOT duplicate it; the CLI may optionally call it for a combined view, but the
  new code is F/M only.

## Architecture (library + CLI, mirrors PR #46)

### Bug fix — make `results.mortality()` read the real CSV

In `osmose/results.py`, the mortality-output read must handle the 2-row (cause, stage) header
+ trailing comma. The diagnostic does NOT depend on the full reader being fixed — it reads via
its own dedicated helper (below) — but the pre-existing `ParserError` should be fixed so
`results.mortality()` returns a usable frame. **Decision:** fix `results.mortality()` to return
a `(cause, stage)` MultiIndex frame (read with `skiprows=1, header=[0,1]`, drop the trailing
all-NaN column), and add a regression test. If fixing the shared reader risks other callers,
fall back to a private `_read_mortality_multiindex(path)` in `fisheries.py` and leave
`results.mortality()` alone — but prefer fixing the shared bug.

### `osmose/validation/fisheries.py` (new)

```python
@dataclass(frozen=True)
class MortalityBalance:
    species: str
    fishing_mortality: float       # trailing-window mean annual F (yr^-1)
    natural_mortality: float       # trailing-window mean annual M = Mpred+Mstarv+Madd (yr^-1)
    f_over_m: float | None         # F/M; None iff M == 0 (documented; "—" in output)
    overexploited: bool            # F/M > 1 (fishing exceeds natural mortality)
```

- `compute_mortality_balance(results, *, species_list, steps_per_year, window_years=10) -> list[MortalityBalance]`
  — for each species: read its mortalityRate CSV (dedicated reader), take the `('F','Recruits')`
  column and the summed `('Mpred'|'Mstarv'|'Madd','Recruits')` columns, aggregate to annual via
  `steps_per_year`, mean over the trailing `window_years`, compute F/M (None when M==0),
  `overexploited = f_over_m is not None and f_over_m > 1`.
- `steps_per_year` is REQUIRED (no row-ratio inference). The CLI derives it from config or
  takes `--steps-per-year`; passing it explicitly keeps the function pure/testable.
- Helpers: `read_mortality_recruits(path) -> pd.DataFrame` (the verified reader);
  `annual_rate(per_step_series, steps_per_year, window_years) -> float` (reshape→sum-per-year
  →trailing-window mean; truncates only a trailing *partial* year and notes if it does).
- `format_mortality_report(balances) -> str` — markdown table (species | F | M | F/M |
  overexploited), with a footer count of overexploited species and a note that F is OSMOSE
  Recruits-stage instantaneous fishing mortality (not an ICES Fbar).

### `osmose/plotting.py` (extend)

- `make_fm_ratio_bars(balances) -> go.Figure` — F/M bar per species, reference line at F/M=1,
  bars above 1 highlighted; "osmose" template. (No Kobe plot.)

### `scripts/compute_mortality_balance.py` (new CLI — mirrors `validate_outputs_vs_ices.py`)

Args: `--results-dir`, `--prefix` (default osm), `--window-years` (default 10),
`--steps-per-year` (default: derive from config in the results dir, else require the flag and
fail loudly), `--species` (optional subset; default all species found),
`--report`/`--json`/`--plot`. Uses `OsmoseResults(dir, prefix=prefix, strict=False)`.

### `tests/test_validation_fisheries.py` (new)

Fixtures **copied from a real file** (`data/baltic/output/Mortality/baltic_mortalityRate-*` or
`data/eec_full/output/Mortality/eec_mortalityRate-cod_Simu0.csv`), not idealized synthetic
frames. Cover: the reader on the real 2-row-header+trailing-comma CSV; `annual_rate` with
steps_per_year=1 and >1; F/M compute; **M==0 → f_over_m None** + report renders "—"; a species
absent from results → WARN-skip (not in output); `overexploited` flag; the bar chart builds and
the F/M=1 reference line is present; CLI `--help` + a `main([...])` invocation that exercises
the deferred imports (e.g. on a tmp dir, asserting graceful failure or a patched compute).

## Scope / YAGNI

- **DROPPED (deferred follow-up):** B/Bmsy, F/Fmsy, Kobe plot, ICES reference-point math. Record
  in the doc that these need a config with broad ICES tonnes-unit coverage + a defensible
  Bmsy (not the Btrigger proxy) + an Fbar-aligned F; not worth it for sprat-only on Baltic.
- **Shiny page DEFERRED** (as before).
- **Biomass-vs-envelope NOT duplicated** — `compare_outputs_to_ices` already does it.

## Honest limitations (surfaced in output + docs)

- F is OSMOSE's **Recruits-stage instantaneous fishing mortality**, summed to annual — not an
  ICES Fbar over an assessed age range. F/M is a model-internal pressure ratio, read as such.
- F/M is undefined when M==0 (reported "—").
- `steps_per_year` is config-derived; the tool fails loudly rather than guess if it can't be
  determined (prevents the silently-N×-wrong-F failure mode the review flagged).

## Delivery

Single PR: the mortality-reader bug fix + `osmose/validation/fisheries.py` + the bar chart +
`scripts/compute_mortality_balance.py` + tests (real-file fixtures) + a short doc note. No
engine changes, no calibration runs.
