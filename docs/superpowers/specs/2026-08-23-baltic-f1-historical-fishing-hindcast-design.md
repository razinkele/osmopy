# Baltic F1 — historical fishing forcing + hindcast validation (Stage 1 of B1)

**Date:** 2026-08-23
**Status:** approved (design), **revised same day after adversarial review**, pending
implementation plan. Review: 5-lens / 15-agent workflow (code-claims, data-claims, science,
consistency, blast-radius; every critical/major finding independently re-verified) — 10 findings
confirmed, 0 refuted, 22 minors; all folded into this revision. Headline correction: the original
relative-scaling decision was unsound for flounder (base F 6.4× above its ICES anchor → scaled F
up to 8.8/yr, an artifact extirpation of a scored stock), the herring aggregation mixed
incompatible F scales, and the skill metric was the July spike's correlation delta in disguise
with no pass margin.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` §4 Phase 2
(B1 item d / F1). **Staging decision (user, 2026-08-23): F1-first thin slice** — historical F is
config + data with zero engine risk, while B1's interannual-LTL/physics work is the expensive
part; this stage's hindcast verdict decides how much of that work is validation-motivated.
**Related:** `docs/baltic_c2b_blocked_by_forcing_2026-08-10.md` (Phase 2 ordering),
`docs/diagnostics/2026-07-15-ssb-f-hindcast-spike.md` (the July NO-GO this supersedes),
`docs/superpowers/specs/2026-07-28-baltic-fishing-forced-cod-topdown-control-design.md` (§5.3
byYear-vs-horizon warning), `docs/superpowers/specs/2026-07-11-baltic-rv-interannual-hindcast-design.md`
(harness discipline: shared-everything A/B, usable-window honesty).

## Goal

Force the certified 9-species Baltic config with the **observed ICES fishing-mortality history
(1993–2023)** and measure whether it moves modeled SSB trajectories toward observed ICES SSB,
against a constant-F baseline. This is the spec's Phase 2 hindcast-interpretability prerequisite:
without historical F, any interannual-forcing hindcast is confounded by fishing history.

**Honest framing of the null.** The July spike demonstrated a washout mechanism (modeled SSB
relaxes to its intrinsic attractor within ~5–10 yr) on the old config, for cod and sprat. The RV
gate that landed since only touches cod_east — the scored stocks here (herring, sprat) face the
*same* washout mechanism, untested on the certified config. Stage 1 is precisely that re-test: a
repeat washout is the anticipated null, not a failure of the harness, and triggers the
pre-registered gating rule in §4.

## What already exists (verified 2026-08-23, file:line audited)

* **The engine lever is fully wired.** `mortality.fishing.rate.byyear.file.sp{i}` loads via
  `_load_fishing_rate_by_year` (`osmose/engine/config.py:461-466`, called at `:2285`) — headerless
  `np.loadtxt` (so `#` comment lines are skipped — provenance headers are safe), one F per line,
  sim-year 0 first. The override is applied on the production Numba path
  (`osmose/engine/processes/mortality.py:739-751`) and the reference path
  (`processes/fishing.py:36-45`), **before** selectivity/seasonality/effort — and it overrides the
  v4 fisheries base rate, because `fisheries.rate.base.fsh{i}` populates the same
  `config.fishing_rate` array (`config.py:328-330`; the species→fishery mapping is the identity,
  per `data/baltic/fishery-catchability.csv`, and the derivation relies on that). Historical F is
  therefore a **config + data change**; the engine work in this stage is hardening only.
* **The ICES data are already cached offline.** `data/baltic/reference/ices_snapshots/` carries
  per-year series for the assessed stocks over 1993–2023, with stock-specific gaps the derivation
  must respect: **cod.27.22-24 F ends 2021, its SSB ends 2022, and it has landings but no
  catches**; `her.27.25-2932`, `cod.27.24-32`, `fle.27.2223` SSB are **index-scaled** while the
  rest are tonnes; all values are strings.
* **The July spike's NO-GO does not transfer as-is.** `2026-07-15-ssb-f-hindcast-spike.md` ran on
  the 8-species config with no RV gate and aggregate cod. Its cod arm is superseded by the E/W
  split + RV gate; its washout mechanism for non-cod stocks is exactly what §4 re-tests. Two
  staleness traps if reusing spike code: it maps sp0↔`cod.27.24-32` (on master sp0 is
  **cod_west** ↔ `cod.27.22-24`), and assumes `nyear=30` against a config default of 15.
* **Known defects in the by-year path** (found in the B1 audit, addressed by this stage):
  the config reader lowercases all keys (`osmose/config/reader.py:173`) while `config.py:2296`
  (`byDt.byAge`/`byDt.bySize`) and `:2319` (`catches.byYear`) look up camelCase — those lookups
  can never match a reader-produced config; and a by-year series **shorter than the run silently
  falls back to base F** mid-run (`fishing.py:44`, `mortality.py:750`) — the same
  silent-degradation class B1 exists to kill.

## Decisions (recorded; 5–7 added in the post-review revision)

1. **Staging: F1-first** (user, 2026-08-23). Stages 2 (unified time policy + interannual
   bottom-O₂) and 3 (interannual LTL via proxy) are gated on this stage's verdict.
2. **F scaling: relative, anchored 2018–2022** (user, 2026-08-23) — **for stocks whose calibrated
   base F is at or below their ICES anchor** (see decision 5 for the exception).
   `F_model(y) = base_F × factor(y)`, `factor(y) = F_ices(y) / mean(F_ices over 2018–2022,
   available years)`. The anchor window matches the window the biomass-envelope targets were
   derived from, so scaled F is centred on the calibrated operating point there (note: factors in
   individual years, including the final ones, are ≠ 1 — e.g. cod_east's factor spans 0.17–14.6×).
   Base/anchor ratios: cod_west 0.04, cod_east ~0.06, herring ~0.39, sprat ~0.44 — for these,
   relative scaling keeps F within or below the calibrated regime (herring/sprat scaled F stays
   < ~0.5/yr). Absolute ICES F remains out of scope (10–25× the calibrated cod rates).
3. **Herring aggregation: catch-weighted mean of scale-free factors** (revised — the original
   catch-weighted mean of raw F mixed an index-scaled series with absolute ones):
   per stock s of the four (`her.27.25-2932`, `her.27.28`, `her.27.3031`, `her.27.20-24`),
   compute `factor_s(y) = F_s(y)/anchor_s` first (dimensionless), then
   `factor(y) = Σ C_s(y)·factor_s(y) / Σ C_s(y)` with catches in tonnes as weights.
   Rationale: the model species' own targets are the 4-stock complex
   (`biomass_targets.csv:19,30,39`); factors are unit-free so mixed F scales cannot distort the
   aggregate; catches are the one comparable weight. Weighting sensitivity is reported, not swept.
4. **Non-assessed species (perch, pikeperch, smelt, stickleback) keep constant F.** Their per-year
   removals would need a separate provenance pipeline — out of scope.
5. **Flounder: reported-only, constant F in both arms** (user, 2026-08-23, on the review's
   critical finding). Its calibrated base F (1.3678/yr, `baltic_param-fishing.csv:41`) is 6.4×
   its ICES anchor (0.214): relative scaling gives F = 3.9–8.8/yr for 1993–2008 (an artifact
   extirpation — observed flounder SSB is flat there), while absolute ICES F (0.08–1.38, below
   base in most years) would balloon the stock the other way. The calibrated value evidently
   absorbs non-fishing mortality and is incommensurable with ICES F without recalibration (out of
   scope). Flounder is also validated against a single management unit (SD22-23; `fle.27.2425`
   does not exist in SAG) while the model species is basin-wide — a second reason not to score it.
6. **Observed herring SSB construction (pre-registered):** z-score each of the four SSB series
   over its available 1993–2023 span, then take the fixed-weight mean of z-scores with weights =
   each stock's mean catch share over 1993–2023 (consistent with decision 3). This is recorded
   now because the construction choice changes the observed decadal trend signs; it is not left
   to implementation.
7. **Skill metric and margin (pre-registered; revised — the original z-scored-RMSE test with no
   margin was the July spike's correlation metric in disguise: for z-scored series
   RMSE² = 2(1−r), and the spike's own honest-negative deltas (+0.009) would have "passed").**
   Score Pearson r against the observed series directly. Arm B passes a stock only if
   **Δr = r_B − r_A ≥ 0.10 AND Δr > 2× the across-seed sd of Δr**. The trend test (§4) is a
   separate necessary condition evaluated on arm B alone.

## Non-goals (YAGNI)

* No production-config change: every file currently in `data/baltic/` stays byte-identical;
  certification stays climatological and constant-F (config default `nyear=15`; the certifier
  runs 50 yr via `--years`, unchanged). The hindcast is an opt-in overlay. This stage **adds**
  four new, unreferenced CSVs under `data/baltic/reference/` (sp0/sp1/sp2/sp8 — flounder gets
  none per decision 5; fixtures that `copytree` the directory inherit them inertly — no config
  key points at them).
* No recalibration against the hindcast; no tuning of any parameter to improve skill.
* No calendar-convention engine feature (`year`-column CSV + `start.year` + offset for the F
  path) — deferred to Stage 2's unified time policy. Stage 1 encodes the calendar by row order
  and documents it.
* No Java-engine support: by-year keys never enter the production config, so the Java
  cross-check arm is unaffected. The hindcast is a Python-engine exercise.

## Design

### 1. Data derivation — `scripts/build_baltic_f_byyear.py` (offline, no network)

Reads `ices_snapshots/*.assessment.json`; writes `data/baltic/reference/f_byyear_sp{i}.csv`
(headerless values + `#` provenance header: stock keys, anchor window, base F, factor range,
generation date) for the four F-forced species:

| sp | species | ICES source | scored? | note |
|---|---|---|---|---|
| sp0 | cod_west | cod.27.22-24 | reported-only | F ends 2021 → hold-last 2022–23; anchor = mean over available anchor-window years (2018–2021) |
| sp1 | herring | 4-stock factor aggregation (decision 3) | **pass/fail** | |
| sp2 | sprat | spr.27.22-32 | **pass/fail** | |
| sp8 | cod_east | cod.27.24-32 | reported-only | factor spans 0.17–14.6×; on base F 0.01 → F ≤ 0.146/yr — modest, and its trajectory is partly prescribed by the RV series anyway |

Flounder (sp3) gets **no by-year file** (decision 5). Each file carries **50 rows**: 19 spin-up
rows at base_F (factor 1), then 31 rows `base_F × factor(1993..2023)`. Base F is read from the
live `data/baltic/baltic_param-fishing.csv` (`fisheries.rate.base.fsh{i}`, identity mapping
noted above) and **written back verbatim as the original string** for the spin-up rows, so
`np.loadtxt` reparses bit-identical values and the two arms share the spin-up exactly; the
harness asserts `arr[:19] == config.fishing_rate[i]` at startup. Regenerating after any
recalibration is one script run.

### 2. Engine hardening (load-path only)

a. **Case fix:** lowercase the lookups at `config.py:2296` and `:2319`, and the
   `_FISHING_SCENARIOS` table (`config.py:1571-1591`) with its four dispatch tests updated in the
   same change. Honest scope: this makes `catches.byYear` and the byDt variants *resolvable* from
   reader-produced configs; `byDt.byAge`/`byDt.bySize` remain reference-path-only and warned
   unsupported (`config.py:2038`) — the fix removes dead lookups, it does not ship byDt support.
   Regression tests go through the real `OsmoseConfigReader`, not hand-built camelCase dicts (the
   existing tests' blind spot). The lowercased byDt/catches key patterns are added to
   `_SUPPLEMENTARY_ALLOWLIST` (the AST walker cannot capture the `{variant}`-built keys — the
   documented CLAUDE.md escape hatch).
b. **Fail-fast on short series:** at `EngineConfig.from_dict`, if any by-year F series has
   `len(arr) < n_year`, raise `ValueError` naming the key, series length, and run length
   (mirrors the RV-spatial guard, `config.py:1219-1226`). Longer-than-run is allowed (extra
   years ignored). This intentionally diverges from Java's silent base-rate fallback; noted in
   the docstring, irrelevant to parity in practice (no production config uses the key). No
   existing test or shipped config carries a short series (audited).
c. **Schema:** one `OsmoseField` for `mortality.fishing.rate.byyear.file.sp{idx}`
   (`ParamType.FILE_PATH`, indexed, optional) beside the other `mortality.fishing.*` fields in
   `osmose/schema/species.py` — kills the unknown-key warning, surfaces the param in the UI.

### 3. Hindcast harness — `scripts/baltic_f_hindcast.py`

Two arms × 5 house seeds [42, 123, 7, 999, 2024] × **50 yr**, Python engine, in-memory:

* **Arm A (baseline):** production config, constant F, `nyear=50`.
* **Arm B (fhist):** identical + the four `mortality.fishing.rate.byyear.file.sp{i}` keys.
* Both arms additionally set `output.ssb.enabled=true` (the production config does not; the
  harness needs the maturity-based SSB output) — shared by both arms, so it cannot confound.

Calendar: **sim-year 19 = 1993.** The RV gate's default mapping (offset 0 → sim-year 0 = series
year 1974) already runs 1974–2020 over sim-years 0–46 and clamps 2021–23 at the 2020 terminal
value — byte-identical to certified behavior, shared by both arms. The 19 spin-up years absorb
the seeding bootstrap (~12 yr) inside a genuinely historical pre-period; F is constant there in
both arms. The arms therefore share seeding transient, RV history, horizon, and seeds — their
difference isolates the ICES F pattern.

**Instrument check (blocking):** per forced stock, arm B's realized annual yield must visibly
track its factor pattern over 1993–2023 (rank correlation between factor series and
yield-per-biomass, reported per stock). Blocking for herring, sprat, and cod_east (factor ranges
wide enough for the check to have power); reported-only for cod_west, whose factor series is
nearly flat (F 0.90–1.21) and would make the canary noise. This is the wrong-mapping /
silent-no-op detector — if it fails for a blocking stock, stop and debug before interpreting SSB.

### 4. Validation — the parent spec's acceptance criterion, made concrete

Per stock, over 1993–2023 (sim-years 19–49), 5-seed mean trajectories, model SSB z-scored,
observed series per decisions 5–6 (herring: constructed z-composite; others: the stock's SSB
series, index-scaled or tonnes — z-scoring makes them comparable):

* **Trend test (necessary, arm B alone):** decadal-trend sign (1993–2002, 2003–2012, 2013–2023)
  matches observed in ≥2 of the 3 decades.
* **Skill test (decision 7):** Δr = r_B − r_A ≥ 0.10 and > 2× across-seed sd of Δr.

**Scored stocks: herring and sprat — verdict PASS requires both** (2-of-2); one is PARTIAL, zero
is NULL. **Reported-only:** cod_west, cod_east (partly RV-prescribed; parent spec excludes cod
until C2(b)), flounder (decision 5).

Output: a dated results doc `docs/baltic_f_hindcast_YYYY-MM-DD.md` + figure (trajectories vs
observed, per stock). A NULL result is legitimate and must be reported as such: it **demotes
Stages 2–3 from validation-motivated to capability-motivated** (the model's equilibria don't
track forcing history at decadal scale → prioritize scenario-track work like C1 over hindcast
realism), and that decision rule is written down here in advance.

### 5. Certification guard

After the §2 engine changes, one standard climatological certification run (50 yr × 5 seeds via
`--years`, on the unchanged config whose default stays `nyear=15`) must come back identical —
the changes are load-path only and the case fix touches keys absent from every shipped config;
the guard is cheap insurance.

## Testing

* **CI-safe units:** case-fix resolution through the real `OsmoseConfigReader` (fixture config
  file, not a hand-built dict) with the four `_FISHING_SCENARIOS` dispatch tests updated; the
  short-series fail-fast (raises, names key and lengths; longer-than-run passes); derivation math
  on tiny fixtures (per-stock anchor over available years, factor-first catch-weighted herring
  aggregation, cod_west hold-last, verbatim base-F spin-up rows, 19+31 layout, no flounder file);
  by-year override still composes with selectivity/seasonality (existing coverage, extended
  through the reader path).
* **NOT a CI gate:** the hindcast outcome (emergent, seed- and machine-sensitive — house rule).

## Success criteria

1. `f_byyear_sp{0,1,2,8}.csv` generated reproducibly from the cached snapshots with recorded
   provenance (incl. per-stock factor ranges); regeneration is one script run.
2. Engine hardening lands with tests; previously dead lookups resolve through the real reader;
   short series fail fast at load; byDt honestly remains unsupported-but-resolvable.
3. Certification guard: unchanged verdict on the standard climatological run.
4. The harness runs 2 arms × 5 seeds × 50 yr and the §3 instrument check passes for all four
   forced stocks before any SSB interpretation.
5. A dated results doc reports the trend and skill tests (with the decision-7 margin) for
   herring and sprat, the three reported-only stocks alongside, an honest verdict either way,
   and applies the pre-registered Stage 2/3 gating rule.

## Stage 2/3 preview (not designed here)

* **Stage 2 — unified out-of-range time policy + interannual bottom-O₂.** One documented policy
  per mechanism class (fail-fast for interannual fields, per the RV-spatial precedent; explicit
  `start.year` convention ported to the F path), fixing the audit's findings: LTL stride-collapse,
  the oxygen loader's frames==24 lock, the RV-gate mean-preserving self-inconsistency, the dead
  temperature NetCDF branch. Data for interannual bottom-O₂ (and salinity) already on disk
  (54 GB per-year `so`/`o2`, 1993–2021).
* **Stage 3 — interannual LTL via proxy.** The multi-year reanalysis has no
  phyto/zoo/benthos biomass; requires a science decision (scale the 6-group climatology by a
  `chl`/`nppv` index vs hold groups climatological) — to be brainstormed with the outcome of
  Stages 1–2 in hand.
