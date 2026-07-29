# osmopy — Proposed GitHub Issues (from literature alerts)

Ready-to-paste issue drafts. Not created on GitHub — review and open manually.

---

## [High] Refresh Baltic ICES SAG/WGBAST snapshots and re-validate

**Source:** ICES WGBAST 2026 cycle / ICES SAG. https://ices-library.figshare.com/articles/report/Baltic_Salmon_and_Trout_Assessment_Working_Group_WGBAST_/29118545
**Alert:** 2026-06-22

**Motivation.** The Baltic example cross-validates F and biomass against the 2024 ICES advice cycle (`docs/baltic_ices_validation_2026-04-18.md`). As new WGBAST/SAG cycles land, calibration targets drift. The repo already has the ingest path (`data/baltic/reference/ices_snapshots/README.md`) and a validator — this is a recurring data-freshness task.

**Proposal.**
- Pull the latest ICES SAG snapshots via the documented refresh workflow.
- Re-run `scripts/validate_baltic_vs_ices_sag.py --report` and `pytest tests/test_baltic_ices_validation.py`.
- Record any envelope changes; note unit-label quirks the validator already guards against (ICES API SSB-unit mislabels detected via Blim magnitude).

**Acceptance criteria.**
- Updated snapshots committed with provenance.
- Validation report regenerated; deltas vs previous cycle summarized.
- Tests green.

**Effort:** quick–moderate. **Labels:** data, calibration, baltic, validation.

---

## [Medium] Optional fish-mediated carbon-flux diagnostic in output.py

**Source:** Silvar-Viladomiu, Cavan, Martin et al., *Estimating the contribution of the Irish Sea fish community to carbon sink potential*, ICES J. Marine Science (2026). https://doi.org/10.1093/icesjms/fsag095
**Alert:** 2026-06-22

**Motivation.** OSMOSE outputs biomass/diet/mortality but no carbon-export term. The Irish Sea EwE+biogeochemistry study shows faecal-pellet carbon dominates fish-mediated flux — a tractable optional diagnostic for blue-carbon/ecosystem-service framings of the Baltic config.

**Proposal.** Add an opt-in per-step diagnostic computing fish-mediated carbon flux from biomass × consumption/egestion coefficients (per-species, configurable), written alongside existing CSV/NetCDF outputs.

**Effort:** moderate. **Labels:** enhancement, output, ecosystem-services.

---

## [High] Ingest May-2026 ICES Baltic advice cycle (advice for 2027) and re-validate

**Source:** ICES Advice 2026 cycle — Baltic fishing opportunities for 2027 (released late May 2026). Headline summaries: ~+74% TAC for central Baltic herring and ~+32% for sprat for 2027 on stronger recruitment estimates; western Baltic cod and herring remain zero/severe decline. FishSec overview: https://www.fishsec.org/2025/05/28/overview-ices-advice-on-baltic-sea-fishing-opportunities/ · ICES Advice collections: https://ices-library.figshare.com/
**Alert:** 2026-06-22

**Motivation.** The Baltic example cross-validates F and biomass/SSB against an earlier ICES advice cycle (`docs/baltic_ices_validation_*.md`). The new cycle materially revises clupeid targets (central herring and sprat up sharply), so existing calibration targets and the validation envelope are now stale. The repo already has the ingest path (`data/baltic/reference/ices_snapshots/`) and a validator.

**Proposal.**
- Confirm the exact 2027 advice values and SSB/F series against the ICES Advice 2026 figshare collection (do not rely on press summaries for the committed numbers).
- Refresh `data/baltic/reference/ices_snapshots/` via the documented workflow; update cod/herring/sprat calibration targets.
- Re-run `scripts/validate_baltic_vs_ices_sag.py --report` and `pytest tests/test_baltic_ices_validation.py`.
- Summarize deltas vs the previous cycle, highlighting the central-herring/sprat upward revision and continued cod zero-catch.

**Acceptance criteria.**
- Updated snapshots committed with provenance (ICES Advice 2026 DOIs/links).
- Validation report regenerated; per-stock deltas vs previous cycle documented.
- Tests green; unit-label quirks (ICES API SSB mislabels) still guarded.

**Effort:** quick–moderate. **Labels:** data, calibration, baltic, validation.

---

## [High] Add WGSAM SMS cod-predation-mortality (M) as a Baltic predation validation target

**Source:** ICES WGSAM updated Baltic Sea SMS multispecies key-run (WGSAM, Oct 2025) — provides updated cod predation-mortality (M) time series for Baltic sprat and central herring used as natural-mortality input to single-species assessments. https://www.ices.dk/community/groups/pages/wgsam.aspx · Key-run review criteria: https://ices-eg.github.io/wg_WGSAM/ReviewCriteria.html
**Alert:** 2026-06-22

**Motivation.** The Baltic example currently cross-validates against ICES SAG SSB and F only. OSMOSE produces mortality-by-cause (incl. predation mortality) per species, so the WGSAM SMS predation-M series for sprat and central herring is a directly comparable, multispecies-specific benchmark — a stronger test of the `predation`/`mortality` engine than SSB/F envelopes alone, and exactly the kind of cross-model skill check the "decade of mizer" review (Ecological Modelling 2025/2026) calls for.

**Proposal.**
- Add the WGSAM SMS M series (sprat, central herring) to `data/baltic/reference/ices_snapshots/` with provenance (confirm exact values from the WGSAM 2025 report, not press summaries).
- Extend `scripts/validate_baltic_vs_ices_sag.py` with an optional comparison of osmopy emergent predation-M-by-cause against the SMS M series; report deltas in the validation report.
- Optionally adopt the WGSAM key-run review criteria as a `docs/` QA checklist.

**Acceptance criteria.**
- SMS M series committed with provenance.
- Validator emits a predation-M comparison (per-stock deltas) alongside the existing SSB/F checks.
- `pytest tests/test_baltic_ices_validation.py` green.

**Effort:** moderate. **Labels:** data, calibration, baltic, validation, predation.

---

## [High] Sync osmopy to OSMOSE 4.4.x (Java-parity audit & staged port)

**Source:** OSMOSE core releases v4.4.0 (2026-05-21) and v4.4.1 (2026-06-18). https://github.com/osmose-model/osmose/releases/tag/v4.4.0 · https://github.com/osmose-model/osmose/releases/tag/v4.4.1
**Alert:** 2026-06-30

**Motivation.** osmopy advertises Java-parity against OSMOSE 4.3.3. Upstream has since shipped two minor releases with **breaking** changes across almost every engine process and the output layer, so the parity claim is now stale. Closing the gap keeps osmopy a faithful port and unlocks new biological capabilities directly relevant to the Baltic config.

**Key upstream changes to mirror (from the 4.4.0 release notes):**
- **Mortality:** now region-aware — `nDead`/`ageDeath` are 2-D `[region][cause]`; `incrementNdead()` requires a timestep argument.
- **Fishing/discards:** tracked in **numbers** (abundance) not biomass; `fishedBiomass→fishedAbundance`, `fishedBy()→fishedNBy()`, biomass derived on demand.
- **Reproduction:** new stochastic maturity ogive (`species.maturity.l50/l75`, normal-CDF maturation) and post-reproduction mortality (`species.reproduction.strategy = iteroparous|semelparous`, `postspawning.survivaltime`).
- **Bioenergetics:** simplified data-poor mode (`species.bioenergetics.model = full|simple`; `species.temperature.tmin/topt/tmax`); spherical egg-size model (`species.egg.density`).
- **Movement:** gradient-based movement; NetCDF-map background distributions.
- **Predation on LTL:** log-form `computePercent` option (`simulation.resources.computePercent.legacy`).
- **Reproducibility:** deterministic RNG (`simulation.fixed.seed.enabled`).
- **Config migration:** module toggles renamed (`module.bioenergetics.enabled`, `module.genetics.enabled`, `module.multispecies.fisheries.enabled`, `module.bioeconomics.enabled`), restart keys → `simulation.restart.*`, de-`bioen`-prefixed maturity/ingestion keys.
- **Output:** background species always included; NetCDF chunking default changed to standard NetCDF-4 strategy.
- **4.4.1 follow-ups:** corrected resource-forcing parameter names (`species.biomass.constant.spX`, `species.biomass.file.sp`); new debug params `simulation.kill.if.no.school.enabled`, `species.is.enabled.spX`.

**Proposal.** Stage the port rather than one big-bang:
1. Write a `docs/` parity note enumerating each 4.4.0/4.4.1 change vs the osmopy implementation, marking done / TODO / N-A.
2. Phase 1 (low-risk, high-value): adopt `fixed.seed` deterministic mode for the test suite; align resource-forcing parameter names; add `species.is.enabled` / kill-on-collapse debug toggles.
3. Phase 2 (process parity): region+timestep-aware mortality; abundance-primary fishing/discards; stochastic maturity ogive + post-reproduction mortality.
4. Phase 3 (extensions): simplified bioenergetics + temperature, gradient movement, log `computePercent`, config-key compatibility shim.

**Acceptance criteria.**
- `docs/osmose_4.4_parity.md` committed with per-change status.
- Phase 1 merged; deterministic test mode green.
- Mortality and fishing outputs reviewed for 4.4.x shape; ICES-SAG validation still passes after changes.

**Effort:** large (stage in phases). **Labels:** parity, upstream-sync, engine, reproduction, mortality, fishing, output.

---

## [Resolved 2026-07-29] Cod disaggregation left `fishery-discards.csv` stale (silent zero-discards on Python, Java-fatal)

> **✅ Fixed 2026-07-29.** `data/baltic/fishery-discards.csv` regenerated to mirror `fishery-catchability.csv` — 9 focal-species rows (`cod_west … cod_east`), 9 fishery columns (incl. `trawlcod_east`), all-zero (discards remain disabled). Regression guard added: `tests/test_baltic_fishery_matrices.py` (asserts discards ↔ catchability structural consistency). Draft retained below for history.

**Source:** surfaced 2026-07-29 while extending the Java-4.4.1 cross-check staging for the disaggregated Baltic config (`docs/baltic_cod_east_M09_java_crosscheck_2026-07-29.md`). The cod split (cod → cod_west sp0 + cod_east sp8, `scripts/disaggregate_cod.py`) updated `predation-accessibility.csv` and `fishery-catchability.csv` but **not** `fishery-discards.csv`.

**The bug.** `data/baltic/fishery-discards.csv` is structurally inconsistent with the disaggregation on **both** axes:
- **Rows:** it still carries an aggregate `cod` row and has **no `cod_west` / `cod_east` rows** (8 species rows vs catchability's 9).
- **Columns:** 8 fishery columns — **missing the `trawlcod_east` fishery** that `fishery-catchability.csv` added (9 columns).
- All values are currently `0` (discards are effectively disabled in this config), which is why the inconsistency has stayed latent.

**Impact.**
- **Python — silent.** `_load_discard_rates` (`osmose/engine/config.py:1007`) resolves each species by name in the discards index; `cod_west` / `cod_east` are absent → they are silently assigned a **zero discard rate by omission**. Harmless while all values are 0, but the moment discards are populated, the disaggregated cod would silently get *no* discards while every other species does — a quiet correctness bug with no error.
- **Java — fatal.** `FishingGear.setDiscards` → `Matrix.getIndexPrey/getIndexPred` aborts on the missing `codwest` prey row and `trawlcodeast` fishery column. Currently only survivable because the Java-staging reconcile pass (`osmose/java_config_reconcile.py::reconcile_config_for_java`) rebuilds the matrix; the underlying data is still wrong.

**Fix plan.**
1. Regenerate `data/baltic/fishery-discards.csv` to mirror `fishery-catchability.csv`'s structure: rows = the 9 focal species (`cod_west … cod_east`), columns = the 9 fisheries (incl. `trawlcod_east`), values `0` (discards remain disabled). Preserve any non-zero cells by (species, fishery) name if discards are ever enabled.
2. Add a regression test asserting `fishery-discards.csv` rows/columns match `fishery-catchability.csv`'s species/fisheries (a cheap structural-consistency guard so future disaggregations can't drift them apart) — this is exactly the invariant `reconcile_config_for_java` enforces at stage time, hoisted to a data test.
3. Extend `scripts/disaggregate_cod.py` (the Task-4 matrix step) to transform `fishery-discards.csv` alongside catchability, so the surgery keeps *all* fishery matrices in sync by construction.

**Acceptance criteria.**
- `fishery-discards.csv` ↔ `fishery-catchability.csv` structural-consistency test green.
- Python discard-rate load for `cod_west` / `cod_east` is intentional (present rows), not zero-by-omission.
- The Java reconcile's discards rebuild becomes a no-op on the committed data (verifiable by diffing pre/post reconcile).

**Effort:** quick. **Labels:** bug, baltic, disaggregation, config, fishing.

---

## [Low] `stage_background_for_java` is not idempotent — duplicates a predator column already in the matrix

**Source:** surfaced 2026-07-29 during the same Java cross-check work (`docs/baltic_cod_east_M09_java_crosscheck_2026-07-29.md`). Masked in production by the reconcile pass; filing so the root cause isn't lost.

**The bug.** `osmose/java_background_staging.py::augment_accessibility` (called by `stage_background_for_java`) **unconditionally appends** a predator column for every background species in `BG_ACCESS`, with no check for a column that already exists. The reconciled disaggregated config already ships a `Cormorant` predator column in `data/baltic/predation-accessibility.csv`, so staging produces a **duplicate `Cormorant` column** in the staged matrix. It is currently only survivable because `reconcile_config_for_java` dedups columns — any other consumer of `stage_background_for_java` would get a malformed matrix.

**Secondary (fidelity) concern.** The authored `BG_ACCESS` values (0.1–0.4) do not match what the Python engine actually uses for a predator **absent** from the matrix: `predation.py:211` falls back to `access_coeff = 1.0`. So for GreySeal (no column in the committed matrix) the staging under-represents Python's predation strength. This is one of the documented reasons the Java cross-check is not yet faithful; it is tracked in the cross-check doc's "Ceiling" section and is **not** required to fix the idempotency bug.

**Fix plan.**
1. Make `augment_accessibility` idempotent: skip any predator already present as a column; only add absent ones. (Add a unit test: staging a matrix that already contains `Cormorant` yields exactly one `Cormorant` column.)
2. *(Optional, ties into cross-check fidelity — deprioritized per 2026-07-29 decision.)* For predators absent from the committed matrix, fill the added column with the Python fallback `access_coeff = 1.0` rather than the authored `BG_ACCESS`, so the staged matrix matches Python's *effective* accessibility.

**Acceptance criteria.**
- No duplicate predator columns after `stage_background_for_java` on the disaggregated config.
- `reconcile_config_for_java`'s column dedup becomes a safety net rather than a necessity (verifiable by diffing pre/post reconcile — no columns removed).

**Effort:** quick. **Labels:** bug, java-staging, background, baltic.

---

## [Opened → #130] Bistability harness retarget left two aggregate-cod coherence gaps

> **Opened on GitHub 2026-07-29:** https://github.com/razinkele/osmopy/issues/130 (label: bug). Draft retained below.

**Source:** surfaced 2026-07-29 while retargeting `scripts/baltic_bistability_chunk0.py` from aggregate cod to `cod_east` (commit 62851df). The seeding-IC bistability experiment now seeds sp8 and measures `cod_east` coherently, but two adjacent code paths still assume the old 8-focal / aggregate-cod layout.

**The gaps.**
1. **The larva-rate driver never reaches the retargeted subject.** `read_base_larva_rates(..., n_focal=8)` iterates sp0–sp7, so `cod_east` (sp8) is never in `base_rates`. The bistability sweep's larval-mortality lever therefore perturbs sp0–7 while the harness measures `cod_east` — the driver and the subject are decoupled. (The synthetic unit tests pass `base_rates` explicitly, so they don't surface it.)
2. **The `--chunk-c-strength` accessibility path still labels the predator `"cod"`** (`write_chunkc_matrix` / the predation-accessibility perturbation) — a name that no longer exists in the disaggregated matrix (`cod_west`/`cod_east`).

**Fix plan.**
1. Extend `read_base_larva_rates` (and any `n_focal=8` companions in that path) to `n_focal=9` so `cod_east` (sp8) is in `base_rates` — OR confirm the intended bistability lever is seeding-IC only and document that the larva-rate driver is deliberately orthogonal to the subject.
2. Retarget the `--chunk-c-strength` predation-matrix path to the disaggregated predator label(s) (`cod_east` to match the harness subject, or the pair).
3. Add a smoke assertion that the sweep's active driver reaches the subject stock.

**Acceptance criteria.**
- `baltic_bistability_chunk0 --experiment bistability` runs coherently on the disaggregated config (driver reaches `cod_east`); `--chunk-c-strength` resolves the predator name.
- Unit tests green.

**Effort:** moderate (needs a science call on the intended driver). **Labels:** bug, baltic, disaggregation, bistability, calibration.

---

## [Opened → #129] 30-minute tutorial + its doc reference the removed aggregate `cod` (disaggregation-stale)

> **Opened on GitHub 2026-07-29:** https://github.com/razinkele/osmopy/issues/129 (labels: bug, documentation). Draft retained below.

**Source:** 2026-07-29 full-suite sweep — `tests/test_tutorial_3species.py` (4 failures: `test_script_runs_to_completion`, `test_biomass_pyramid_emerges`, `test_trophic_cascade_visible`, `test_headless_fallback_produces_equilibrium`). NOTE: an earlier triage mislabeled these "non-baltic" — they are disaggregation-stale.

**The bug.** The 30-minute tutorial runs the `data/baltic/` config with `FOCAL_SPECIES = ["cod", "sprat", "stickleback"]` and a trophic-cascade narrative built on cod↔sprat and cod↔stickleback predation (drop cod–sprat accessibility → cod has less food → stickleback up). The cod disaggregation (cod → `cod_west` sp0 + `cod_east` sp8) removed the aggregate `cod` species/column, so the tests raise `KeyError: 'cod'` and the `BASELINE_PERTURBATION` in `tests/_tutorial_config.py` targets a species that no longer exists.

**Fix plan (a retarget, not a rename — the cascade story must map to a chosen stock).**
1. Choose the stock the tutorial's cascade narrative follows — `cod_west` (sp0, the coastal western stock, nearest the stickleback story) is the likely candidate; or `total_cod` (`osmose.results.total_cod`) if the story is total-cod biomass.
2. Update `FOCAL_SPECIES`, and `build_config` / `BASELINE_PERTURBATION` in `tests/_tutorial_config.py` (the cod–sprat accessibility perturbation) to the chosen stock.
3. Re-measure the cascade-magnitude assertion bands from smoke runs on the retargeted config.
4. Update the prose in `docs/tutorials/30-minute-ecosystem.md` to match — the test docstring requires the doc and config to stay in sync.

**Acceptance criteria.**
- The 4 tutorial tests pass.
- `docs/tutorials/30-minute-ecosystem.md` matches the retargeted config; the documented cascade magnitude is re-derived from the disaggregated model.

**Effort:** moderate (a stock choice + doc + re-measured bands). **Labels:** bug, baltic, disaggregation, tutorial, docs.

---

## [Low] SP1b larval recalibration is aggregate-cod-specific — re-solve RECAL_RATE per stock

**Source:** 2026-07-29; `test_sp1b_mean_neutral_drift_guard` now skips on disaggregated configs (commit 5811575). Already documented at `docs/diagnostics/sp1b_recalibration.md`; filed here for tracking.

**Motivation.** `RECAL_RATE = 14.66` (cod larval mortality, `osmose/calibration/larva_recal.py`) was solved for the ~140 kt aggregate cod stock. On the disaggregated config it drives `cod_west` extinct (6432 → 1 t under SP1) while `cod_east` barely moves — total cod drifts ~8.9%, so the neutral-drift guard is structurally invalid and is currently skipped with an evidence-bearing reason. `mean_cod` was already fixed to return total cod so the SP1b scripts no longer crash.

**Fix plan.** Re-solve the SP1b larval recalibration **per stock** (cod_west and cod_east have very different scales; cod_east is separately RV-gated) on the maintainer host (the 15-yr Baltic sim is minutes/eval), then remove the disaggregated-config skip in `test_sp1b_mean_neutral_drift_guard`.

**Acceptance criteria.** Per-stock RECAL_RATE(s) solved + committed with provenance; the skip removed and the guard passes on the disaggregated config.

**Effort:** moderate (maintainer-host recalibration). **Labels:** calibration, baltic, disaggregation, sp1b.
