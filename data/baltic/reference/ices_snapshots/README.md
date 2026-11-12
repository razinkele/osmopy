# ICES SAG Snapshots

Frozen JSON copies of ICES Stock Assessment Graphs (SAG) data used to
validate Baltic OSMOSE calibration inputs (biomass targets, fishing
mortality rates). Taken via the `ices` MCP server (same payload shape as
`get_stock_assessment` / `get_reference_points`):

- `{stock}.assessment.json` — list of year-row dicts with lowercase keys
  (`year`, `ssb`, `recruitment`, `f`, `catches`, `landings`, `discards`,
  `low_ssb`, `high_ssb`).
- `{stock}.reference_points.json` — dict with `flim`, `fpa`, `fmsy`,
  `blim`, `bpa`, `msy_btrigger`, `f_age_range`, `recruitment_age`,
  optionally `note`.

Advice year (refreshed 2026-11-06): **2026** for nine stocks; **2024** for `fle.27.2223`
(no 2025 or 2026 entry); **2022** for `cod.27.22-24` (category-3 since — the 2023 entry is
relative and the 2025 one a harvest-rate ratio — so the last absolute assessment is used, and it
ends at 2021–22). Covers the 2018–2022 window used for the catch targets in `biomass_targets.csv`
and the validator's 2020–2024 window.

Pull helper: `scripts/_pull_ices_snapshots.py` (one-shot — not re-run by
CI). Hits the ICES SAG REST API directly, applies the same flattening
the MCP server does, and writes one file per stock.

## `index.json`

- `advice_year` — primary advice year (2026).
- `created` — date of the snapshot pull.
- `model_species_to_ices_stocks` — manifest mapping OSMOSE model species
  to ICES stock keys. Empty list = coastal / data-limited species with
  no SAG assessment; the validator tolerates these.
- `units_by_stock` — `"tonnes"` or `"index"` per stock. See "Unit
  caveat" below — the ICES API's `StockSizeUnits` field is unreliable,
  so this is derived from the `Blim` magnitude (Blim < 100 → index).
- `advice_year_by_stock` — per-stock advice year (overrides the
  top-level `advice_year` for stocks on an alternate cycle, e.g.
  `cod.27.22-24` uses 2022).

## Recruitment units (`recruitment_index_stocks`)

The SSB unit does **not** imply the recruitment unit. `her.27.25-2932` has an index
SSB but reports recruitment as absolute numbers (6–29 million in 2018–2022), while
`cod.27.24-32` reports recruitment on the same relative scale as its SSB (0.4–1.6).
`index.json['recruitment_index_stocks']` lists the stocks whose R is relative;
`scripts/evaluate_calibration_vs_ices.py` skips those in the recruitment comparison
(the model's R is a count) rather than inferring the unit from `units_by_stock`.

## Baltic flounder notes

Baltic flounder is **four** ICES stocks, and three of them are filed under
other species codes, so searching `fle.27` finds only one:

| stock | area | 2018–2022 mean catch | SSB / F as published |
|---|---|---:|---|
| `bzq.27.2425` | SD 24–25 (*Platichthys* spp.) | 11 966 t | 2024: survey index (`tonnes/h`); 2026: labelled "SSB, tonnes" but index-scale (unit overridden in the pull helper) / **no F values** |
| `bzq.27.2628` | SD 26, 28 (*Platichthys* spp.) | 2 764 t | `B/Bmsy` / `F/Fmsy` |
| `fle.27.2223` | SD 22–23 (*P. flesus*) | 834 t | `B/Bmsy` / `F/Fmsy` |
| `bwp.27.2729-32` | SD 27, 29–32 (*P. solemdali*) | 119 t | kg per gillnet station / 2024: no F values; 2026: `Lfm/Lmean` (a length indicator) in the F column |

Total 15.7 kt; `fle.27.3a4` is North Sea and excluded.

**`bzq.27.2628` publishes `Catches` and `Landings` the wrong way round.** In 2015–2020
its `landings − catches` equals `discards` exactly, and catches = landings + discards for
every other stock. The snapshot keeps SAG's columns verbatim. `index.json`
`catch_landings_swapped_stocks` lists the stock, and `derive_ices_targets.py` swaps the
columns on read (the 2018–2022 mean is 2 764 t, not 2 363 t). A test fails if any stock
shows catches below landings without being listed, or is listed without needing it.

## Catch targets (`derive_ices_targets.py`)

Each assessed species' catch target is the 2018–2022 mean of its stocks' summed catch.
Where SAG gives no `catches`, landings **plus discards** are used. Landings alone dropped
`cod.27.22-24`'s discards (7.5% of its 2018–2021 catch) until 2026-11-04. Every committed
`catch` row in `biomass_targets.csv` must equal the script's output (a test checks every
field), so change the inputs here and re-run `--write` rather than editing the CSV.

**`catch_supplements`** adds catch years the pinned assessment lacks. `cod.27.22-24` stays
on its **2022** assessment, the only one with absolute SSB (tonnes) and F: the 2023
assessment is `Stock Size: Relative` / `Fishing pressure: Relative`, and 2025 is `SSBrel` /
`HR`. That snapshot ends at 2021, so 2022's catch (88 t landed + 27 t discarded, identical in
the 2023 and 2025 assessments) is entered here with its SAG assessment key. The derivation
raises if a supplemented year ever appears in the snapshot, so a refresh cannot leave a
stale hand-entered value in place. Before this, the committed cod_west row (2 742 t)
had been computed by hand from the 2023 assessment, landings only, and could not be
reproduced from the repo. It is now 2 951 t. All four are mapped to
`flounder` (since 2026-11-04). Before that only `fle.27.2223` was, so the
flounder **catch target** was 834 t, ~5% of the Baltic total, and any calibration
scoring yield would have pulled flounder catch ~18× too low. There is still no
absolute flounder biomass or F anywhere in SAG, so flounder gets catch only:
no SSB envelope, no F verdict.

ICES flounder catch fell steeply across the target window (23.6 kt in 2018 to
9.9 kt in 2022, 8.2 kt in 2023), so the 2018–2022 band is wide and its mean sits
well below mid-2010s catches. Which period the model should match is a
calibration choice; the window is shared with every other catch target.

Baltic flounder is also assessed on a **biennial cycle** (even years):
2022, 2024, 2026, … Refreshing odd-year advice (2023, 2025) will not
bring a new flounder snapshot.

## Unit caveat (important)

Six of the eleven stocks report SSB as a **relative biomass index**
rather than absolute tonnes: `cod.27.24-32`, `her.27.25-2932` and the four
flounder stocks. For the first three plus `fle.27.2223` the ICES SAG API
mislabels these as `"tonnes"` in the per-row metadata — the true unit
is inferred from the `Blim` magnitude. `bzq.27.2425` and `bwp.27.2729-32`
have no Blim; the puller then reads `StockSizeUnits`, but only to call a
non-`"tonnes"` unit an index — a bare `"tonnes"` without a Blim stays
unclassified and the pull fails. The validator skips SSB
envelope comparisons for index-unit stocks (nothing to compare tonnes
against an index).

F is not safe either. It is dimensionless, but not necessarily absolute.
SAG's own `FishingPressureDescription` / `FishingPressureUnits` say which, and
`index.json['f_type_by_stock']` records them verbatim (added 2026-10-31; the
puller now writes them). Four of the eleven stocks publish **relative** F, and
two publish none:

| stock | SAG description / units |
|---|---|
| `her.27.25-2932` | `F` / `ratio` (SS3, FMSY = 1, Fpa = 1.2) |
| `cod.27.24-32` | `Fishing pressure: Relative` (no FMSY defined — a relative index, not F/FMSY) |
| `fle.27.2223` | `F/Fmsy` (SPiCT; SSB is `B/Bmsy`) |
| `bzq.27.2628` | `F/Fmsy` / `ratio` |
| `bzq.27.2425` | blank — no F values at all |
| `bwp.27.2729-32` | `F` / blank, **but no F values**. If a future assessment starts publishing values under this label, they will be read as **absolute** — check them first. |

A blank description is recorded only for a stock with no F values; the puller
still raises on a blank next to real F values, since the validator would read it
as relative. Stocks without F values are left out of the validator's relative-F
reason and contribute nothing to the weighted F.

The validator gives no verdict on a species whose comparator F is relative
(cod_east, flounder) or only partly covered (herring), and reports the value
marked `(relative)` (`docs/baltic_herring_sprat_f_2026-10-29.md`). FMSY = 1
is a hint, not a test: decide from the description field.


## WGSAM SMS cod-predation mortality (M2) — `wgsam_sms_baltic_2025.*` (issue #136)

Two files snapshot the ICES WGSAM **Eastern Baltic Sea SMS key run 2025** (the SMS model,
Lewy and Vinther 2004 as cited in the stock annex — an ICES CM paper, not resolved here;
stock assessor Morten Vinther, DTU Aqua), pulled at a pinned commit from the
public repository `ices-eg/wg_WGSAM`, folder `Baltic-2025-keyRun`, commit `f690d4ff`
(2025-10-08, "2025 baltic keyrun"). The WGSAM 2025 report was **not yet in the ICES library**
when this was taken, so the repository artefact is the primary source; replace the citation
when the report appears. Pull helper: `scripts/_pull_wgsam_sms_m2.py` (one-shot).

- `wgsam_sms_baltic_2025.m2_annual.csv` — verbatim `M2_annu_.csv`: `Year, scenario, Species,
  variable, Age, value`. M2 = annual instantaneous **cod**-predation mortality (per year) on
  Herring and Sprat at ages 0–8, 1974–2024, for BOTH the `2022 Key run` and the `2025 key run`
  (the revision between them is informative; the validator uses `scenario_used` from
  `index.json`). The annual file ends at 2024; the projection year (2025) is absent from it
  and marked `-1` only in the quarterly `summary.out`, so the loader's `value < 0` guard is
  defensive.
- `wgsam_sms_baltic_2025.weights.csv` — derived from the key run's `summary.out`, quarter-1 rows:
  stock numbers `N`, mean weight `west` and biomass `BIO` at age, used to weight M2 across ages.
  Age 0 has `N = 0` in quarter 1, so a biomass-weighted mean is effectively ages 1+.

What the key run is (stock annex, `StockAnnex/Baltic/StockAnnex_ICES_EB_SMS_2022_Configuration.pdf`):
ICES Subdivisions 25–32 excluding the Gulf of Riga; **cod is the only predator**, treated since the
2019 key run as an *external* predator whose numbers and size distribution come from the ICES SS3
assessment (SMS no longer estimates cod internally); prey are central Baltic herring
(`her.27.25-2932`, SD 25–29+32 excl. Gulf of Riga) and sprat (`spr.27.22-32`). The `index.json`
block `sms_m2` carries this provenance plus the SMS→model species map (`Herring → herring`,
`Sprat → sprat`).

How it is compared (`osmose/validation/ices.py`, `scripts/validate_outputs_vs_ices.py --sms-m2`):
SMS M2 is cod-only, at age, numbers basis; OSMOSE's `mortalityRate` Predation cause lumps every
predator (both cods, percids, the background seal and cormorant). So the model quantity is
**cod-attributed** M2 on a biomass basis — annual tonnes of the prey eaten by `cod_west` + `cod_east`
(`predatorPressure`, per-step mean × steps per year) over the prey's mean biomass (young-of-year
excluded by `output.cutoff.age`) — against SMS M2 biomass-weighted across ages. The two rates differ
through age structure, domain (SD 25–32 vs the whole grid) and the prey pool each is taken on, so
they are indicative, not equivalent. The model's total predation rate per stage (a different
basis) is reported beside it as context. **Report-only; not gated.** Recent SMS values are low (eastern cod collapsed): ~0.08–0.11 yr⁻¹ for both stocks
over 2019–2024, against historical peaks of ~0.5 (herring) and ~0.85 (sprat) in the 1980s.

## How to Refresh

Snapshots freeze the 2026 ICES advice. When ICES publishes a new advice year
(typically every May/June), refresh via:

1. Start a Claude Code session in `osmose-python/` so the `ices` MCP server
   loads (CWD-sensitive — launching from the parent directory silently drops
   the server).
2. Update `scripts/_pull_ices_snapshots.py`: bump the `year` in each tuple
   of the `STOCKS` list. Cod `cod.27.22-24` is category-3 in odd years — keep
   its year at the most recent even-year advice (2022, 2024, 2026, …) until
   ICES resumes full assessment. Baltic flounder `fle.27.2223` is on the same
   biennial cycle.
3. Run the helper:

   ```bash
   .venv/bin/python scripts/_pull_ices_snapshots.py
   ```

   To add or refresh **specific** stocks without touching the others, pass
   `--only STOCK ...`: it pulls just those and merges their entries into the
   three derived keys. A full pull re-downloads every series, and SAG revises
   history, so it can move other species' catch targets too. The three non-`fle`
   flounder stocks were added this way on 2026-11-04.

   It hits the ICES SAG REST API directly, applies the same flattening the
   MCP server does (lowercase keys matching `get_stock_assessment` /
   `get_reference_points` output), and rewrites every `{stock}.assessment.json`
   and `{stock}.reference_points.json` plus `index.json` (`advice_year_by_stock`
   and `units_by_stock` are derived automatically — the latter from Blim
   magnitude since the ICES `StockSizeUnits` field is unreliable — and
   `f_type_by_stock`, verbatim from SAG; review its diff, since it decides which
   F rows get a verdict). The helper loads the existing `index.json` and
   rewrites only those three keys, so the
   hand-maintained blocks (`model_species_to_ices_stocks`,
   `recruitment_index_stocks`, `catch_landings_swapped_stocks`, `catch_supplements`,
   `sms_m2`) survive a refresh.
   If it stops with `cannot classify SSB unit`, the stock has no Blim and a "tonnes" label
   the helper does not trust. Check the SSB against the catches. If it is index-scale, add the
   (stock, year) to `UNIT_OVERRIDES` with the evidence, as for `bzq.27.2425` in 2026; do not
   loosen `_unit_for`.
4. Update `index.json`'s top-level `advice_year` and `created` date by hand
   (the helper doesn't touch them).
5. **Update hardcoded constants in `scripts/validate_baltic_vs_ices_sag.py`:**
   - `WINDOW_YEARS` (currently `range(2020, 2025)`) — shift to the last five
     years covered by the new advice (`range(advice_year - 6, advice_year - 1)`).
   - `REPORT_MD` filename (contains `2026-11-06`) — update the date stub to
     the refresh date.
   - The `"2026 advice"` label in the validator docstring and report header
     should match the new advice year.
6. Update the `advice_year` assertion in
   `tests/test_baltic_ices_validation.py::test_manifest_exists_and_is_readable`.
7. Run `.venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report` and
   review the refreshed `docs/baltic_ices_validation_<date>.md` for new drift.
8. Run `.venv/bin/python -m pytest tests/test_baltic_ices_validation.py -v`.
   If the drift fence trips for a species that now has genuine drift, decide:
   broaden the calibration target in `baltic_param-fishing.csv` /
   `biomass_targets.csv` (separate calibration-tuning plan, not this
   validation plan), or — for deliberate modeling choices — add the species
   to `F_KNOWN_EXCEPTIONS` / `B_KNOWN_EXCEPTIONS` with a pointer to the
   findings doc.
