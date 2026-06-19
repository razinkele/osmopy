# OSMOSE 4.4.0 Jar Swap — Detailed Resume Plan (cross-engine + ICES/empirical validation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status: DEFERRED (resume-ready, 2026-06-19).** The jar swap was paused after the A/B gate FAILED: the 4.4.0 jar cannot LOAD OSMOPY's Java-runnable configs (`NETCDF_BIOMASS resource forcing ... species.biomass.varname.spN not found`) — a newly-REQUIRED 4.4.0 config-key class the original audit missed. PR-A (value-migration layer, `c9454d9`/#79) is merged and inert. **Resume trigger:** a concrete need for a specific 4.4.0 Java feature. This plan is the detailed, validation-centered guide for that resume.

**Goal:** Bundle OSMOSE 4.4.0 and make it the default engine, with confidence established by two validation tiers the user asked for: **(1) cross-engine parity (the pure-Python engine ↔ Java-4.4.0)** and **(2) validation against ICES/empirical data** — gated behind making 4.4.0 actually load (the resource-forcing + required-key migrations the gate surfaced).

**Architecture:** Audit-driven, not runtime-parity-driven (OSMOSE is a stochastic IBM with different RNGs per engine — Python PCG64 vs Java MT19937 — so tight numerical parity is impossible; equivalence is argued from a complete source diff + confirmed by *statistical* validation). Phase 0 systematically audits 4.4.0 for the two change-classes (dynamics values AND newly-required keys). Phase 1 migrates the required keys so 4.4.0 loads. Phases 2–3 are the two validation tiers. Phase 4–5 are the cutover + deploy (mechanics already specified in the prior plan).

**Tech Stack:** Python 3.12, pytest, OSMOSE Java jar (subprocess), xarray (NetCDF), the ICES MCP tools + `osmose/validation/ices.py`. Run with `.venv/bin/python`. Java 21 present; 4.4.0 jar re-fetch: `gh release download v4.4.0 -R osmose-model/osmose -p '*jar-with-dependencies*' -D osmose-java/` (sha256 `70e4c01f714cf1cb7256fe9d1be6220919d828bc7e48097ea3b07f311e1a7b17`).

**Builds on / supersedes:** PR-A merged the value layer (`osmose/config/aliases.py` larval ×ndt/÷ndt + guards). This plan SUPERSEDES the cutover half (Tasks 4–12) of `docs/superpowers/plans/2026-06-18-jar-swap-440.md` — that plan's cutover/deploy task mechanics (version-aware glob, flip write-default, ROUTING, keep-the-jar rollback, deploy guard) still stand and are referenced from Phase 4–5 here rather than re-specified. The A/B ensemble harness from PR-B lives on local branch `feat/jar-swap-440-cutover` (`scripts/ab_validate_440.py` @ `406ea5d`) and is reused/extended in Phase 2.

---

## Validation philosophy (the centerpiece — read before executing)

Two hard truths shape every validation task:

1. **Cross-engine parity is statistical, never bit-exact, and that is not a regression.** Python uses NumPy PCG64; Java uses MT19937 — the streams diverge on the first draw regardless of seed. The OSMOPY port was *defined* as valid at "within 1 order of magnitude" cross-engine (14/14 EEC, 8/8 Bay of Biscay). So the parity question is **NOT** "is 4.4.0-Java identical to Python" — it is **"does 4.4.0-Java agree with the Python engine at least as well as 4.3.3-Java did?"** We compare against the Python-vs-4.3.3-Java baseline; the swap must not *degrade* that agreement.

2. **The honest evidence chain is audit (deterministic) → cross-engine (statistical) → empirical (ICES).** Runtime parity can't *prove* equivalence for a stochastic model; it can only *fail to refute* it. So the primary evidence is the **complete source audit** (Phase 0); the cross-engine ensemble (Phase 2) and ICES validation (Phase 3) are confirmations. **CAVEAT (review-grounded):** a PASS on an *underpowered, mean-only* runtime check is NOT confirmation — the runtime tier is evidential only to the extent of its statistical power against the regression classes that actually matter (variance, tails, distribution shape), which a comparison of ensemble means does NOT see. The resource-forcing audit-miss that sank the first attempt is the proof case that audits are fallible, so the confirmation tier must be **powered and distributional** (see Phase 2), not ceremonial. Equivalence is a question for *equivalence testing* (TOST / CI-overlap against a pre-registered margin; Lakens & Delacre 2020, doi:10.15626/MP.2018.933), NOT for "we failed to find a mean difference."

3. **"Within 1 order of magnitude" is a catastrophic-divergence TRIPWIRE, not an equivalence criterion.** A factor-of-ten agreement is not equivalence by any marine-ecosystem-model skill standard (Olsen et al. 2016 doi:10.1371/journal.pone.0146467; Rynne et al. 2025 doi:10.1029/2024EF004868 use RMSE/MEF/Spearman, not OoM bands). Keep 1-OoM only as a "did something break catastrophically" floor; the actual gate is the equivalence test (Phase 2).

**Coverage honesty (must be stated in the final docs):** only `data/eec_full` (Eastern English Channel, ICES Div 7.d) and `data/examples` (Bay of Biscay, Div 8) run on the Java engine. Baltic/bioen/`nbackground>0` configs are Java-BLOCKED by design (Python-engine-only) — no Java cross-check, no Java ICES check.
- **The transitive-confidence chain has two breaks (review-grounded):** (i) Baltic-specific code paths (`nbackground>0`, bioenergetics, life-stages) are NEVER exercised on Java, so eec/BoB cross-engine agreement says nothing about them — a 4.4.0 dynamics change confined to those paths is uncaught (exactly the class the resource-forcing miss exemplified). (ii) **The swap is NOT inert for Baltic:** the cutover flips the default `target_version` to 4.4.0 and routes every config through the migration layer, so Baltic's *config round-trip* changes even though its *engine dynamics* (Python) don't. "Baltic is unaffected" is therefore too strong — its dynamics are unaffected; its I/O is not. **→ Phase 2 adds a Baltic config-round-trip parity gate** (read → migrate to 4.4.0 keys → write → re-read → Python run, asserting unchanged Python outputs across the flip). That closes the one channel by which the swap can perturb Baltic.
- eec/BoB are **structural demo/parity configs, NOT ICES-calibrated** (see Phase 3), so the ICES tier on them is a weak consistency check; the genuine empirical anchor is the Baltic ICES/HOLAS-3 validation on the (swap-unaffected) Python engine.

---

## Phase 0 — Systematic 4.4.0 audit (both change-classes)

The prior audit covered *dynamics* changes (found the larval-units change) but MISSED *newly-required config keys* (resource-forcing). This phase audits BOTH axes to completion before any migration.

### Task 0.1: Audit newly-REQUIRED 4.4.0 config keys (the missed class)

**Files:** Create `docs/superpowers/notes/2026-jar-swap-440-required-keys-audit.md` (a checklist artifact).

- [ ] **Step 1: Enumerate what 4.4.0 REQUIRES at load that 4.3.x didn't.** Two complementary methods, do both:
  - **Source method:** read the 4.4.0 engine source at tag `v4.4.0` (`github.com/osmose-model/osmose`) for every `init()`/load path that throws "parameter missing" — start with `ResourceForcing.init` (the known one: requires `species.biomass.mode.spN` + `species.biomass.varname.spN`), then `Configuration.load`, `ConfigurationFile`, each `*Process.init`, and the resource/LTL/background loaders. List every key 4.4.0 requires that an OSMOPY 4.3.x config does not emit.
  - **Empirical method:** iteratively try to load `data/eec_full` and `data/examples` on the 4.4.0 jar (fetch it first), `to_target_keys(..., "4.4.0")`-migrated, and record each "missing parameter" error in turn (fix-forward in Phase 1, then re-run to surface the next). The jar fails one missing key at a time, so this converges to the full required-key set.
- [ ] **Step 2: For each required key, record:** the key pattern, which OSMOPY config(s) hit it, the 4.3.x source of the value (e.g. `species.file.spN` → the NetCDF; `ltl.*` legacy keys), and the migration recipe (rename / derive / default). Write the checklist to the notes file.
- [ ] **Step 3: Commit the audit artifact.**
```bash
git add docs/superpowers/notes/2026-jar-swap-440-required-keys-audit.md
git commit -m "docs(audit): 4.4.0 newly-required config keys for the jar swap"
```

### Task 0.2: Re-confirm the dynamics-change audit

- [ ] **Step 1:** Re-verify the prior dynamics audit (in `docs/superpowers/specs/2026-06-18-jar-swap-4.4.0-design.md`) is complete now that the load path is understood: larval-units (done in PR-A), adult mortality (unchanged), starvation/growth/reproduction/fishing (unchanged), bioen ingestion (PR1). Confirm no dynamics change is gated behind a key only present once Phase 1 adds the required keys (e.g. does adding `species.biomass.mode` change resource dynamics vs the legacy path?). Add findings to the audit notes.
- [ ] **Step 2:** No commit unless findings change the migration set (then update the notes + commit).

---

## Phase 1 — Required-key migrations (make 4.4.0 LOAD)

### Task 1.1: Resource-forcing migration

**Files:** Modify `osmose/config/aliases.py`; Test: `tests/test_config_migration_440.py`. (The value migration goes in the `to_target_keys` 4.4.0 branch, alongside the larval rescale + drop-guards from PR-A.)

Background (CORRECTED against `ResourceForcing.java::init()` @ v4.4.0 — the original plan mis-stated the keys). For a NETCDF-forced resource species, 4.4.0 reads, under the `species.biomass.*` prefix:
- `species.biomass.file.spN` — the NetCDF path (NOT `species.file.spN`; the latter is only an `isNull` guard). **The migration MUST emit this** (a copy of the `species.file.spN` path).
- `species.biomass.varname.spN` — the variable name inside the NetCDF. **Hard-required, no engine default/convention** (`getString`→`error` if absent).
- `species.biomass.nsteps.year.spN` — per-species steps/year; 4.4.0 accepts a GLOBAL `species.biomass.nsteps.year` fallback (eec_full has `;24` globally → per-species optional, but emit it for safety/parity with the official config).
- `species.biomass.mode.spN` — OPTIONAL (defaults `NETCDF_BIOMASS`); emit it explicitly for clarity (harmless).

**varname derivation (verified by NetCDF inspection — derivable, but NOT by sniffing):** `data/eec_full/eec_ltlbiomassTons.nc` has 10 data vars (`Dinoflagellates, Diatoms, …, VLBVeryLargeBenthos`) mapping **1:1 to `species.name.sp14–23`**. So the rule is `varname = species.name.spN` — NOT "pick the biomass var from the NetCDF" (10 candidates → ambiguous). Derivation order: (a) explicit `species.biomass.varname.spN` if present → (b) `species.name.spN` (EEC) / `ltl.name.rscN` (BoB legacy) → optionally cross-check the derived name EXISTS as a NetCDF data var (a free validation), but do not pick from the NetCDF.

**`examples` (BoB) is a DIFFERENT, harder scheme** — the legacy global `ltl.netcdf.file` + per-resource `ltl.name.rscN`/`ltl.tl.rscN`/… form, which predates even the 4.2.3→4.3.0 `rscN→spN` renames. So BoB needs: FIRST the legacy `ltl.*`/`rscN` → `species.*`/`spN` normalization (check whether OSMOPY's reader already does this — if not, it's its own migration sub-task), THEN the 4.4.0 `species.biomass.*` keys (varname from `ltl.name.rscN`). **Scope BoB as a separate, harder sub-task (Task 1.2) — it is materially more work than EEC.**

- [ ] **Step 1: Write the failing test** (EEC case). Append to `tests/test_config_migration_440.py`:
```python
def test_4_4_0_write_adds_resource_biomass_forcing_keys(tmp_path):
    from osmose.config.aliases import to_target_keys

    cfg = {
        "species.type.sp14": "resource",
        "species.name.sp14": "Dinoflagellates",
        "species.file.sp14": "eec_ltlbiomassTons.nc",
        "simulation.time.ndtperyear": "24",
    }
    out = to_target_keys(cfg, "4.4.0")
    assert out["species.biomass.mode.sp14"] == "NETCDF_BIOMASS"
    assert out["species.biomass.file.sp14"] == "eec_ltlbiomassTons.nc"   # the path 4.4.0 actually reads
    assert out["species.biomass.varname.sp14"] == "Dinoflagellates"      # = species.name, NOT a sniffed var
    assert out["species.biomass.nsteps.year.sp14"] == "24"               # from ndtperyear (or global fallback)
```
- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_config_migration_440.py -k resource_biomass_forcing -q`
- [ ] **Step 3: Implement** in `osmose/config/aliases.py` — in the `to_target_keys` 4.4.0 branch, after the drop-guards/larval-migration, add the resource-forcing step. For each `species.type.spN == "resource"` with a `species.file.spN`: emit (if absent) `species.biomass.mode.spN=NETCDF_BIOMASS`, `species.biomass.file.spN=<species.file.spN value>`, `species.biomass.varname.spN=<species.name.spN>`, `species.biomass.nsteps.year.spN=<ndtperyear or global species.biomass.nsteps.year>`. Comment-cite `ResourceForcing.java::init()` @ v4.4.0 for the key set + the no-varname-default fact.
- [ ] **Step 4: Verify** the unit test, then a **FULL-YEAR LOAD+RUN smoke against the real jar** (requires the fetched 4.4.0 jar). Resource keys are read in the per-step `ForcingFile.update()` path, NOT just at init — a 0–1-year smoke can pass while a malformed varname/time-mapping throws on the first update. So run **≥1 full year** (ndt steps): `write_temp_config(reader.read("data/eec_full/..."), tmp, target_version="4.4.0")` then `java -jar osmose_4.4.0-...jar <tmp master>` for 1 year — confirm it loads AND completes a year with no missing-parameter error. If a DIFFERENT missing key appears, that's the next Task 1.x (loop Task 0.1's empirical method). ruff/pyright clean.
- [ ] **Step 5: Commit.**
```bash
git add osmose/config/aliases.py tests/test_config_migration_440.py
git commit -m "feat(config): emit 4.4.0 species.biomass.{file,varname,nsteps.year} for NetCDF resources"
```

### Task 1.2: BoB legacy `ltl.*` resource-forcing migration (the harder scheme)

- [ ] First DETERMINE whether OSMOPY's reader/`migrate_config` already normalizes `data/examples`' legacy `ltl.netcdf.file`/`ltl.name.rscN`/`rscN` keys to the `species.*`/`spN` form (read `osmose/demo.py` `_MIGRATION_CHAIN` for the 4.2.3/4.3.0 steps; load `data/examples` via `OsmoseConfigReader().read()` and inspect the resulting keys). If it does → BoB collapses to the Task-1.1 case (varname from the normalized `species.name`). If NOT → add the `ltl.*`→`species.*` normalization (rscN→spN, `ltl.name`→`species.name`, `species.type=resource`, `simulation.nplankton`→`nresource`, etc.) before the 4.4.0 forcing-key step, with varname from `ltl.name.rscN`. TDD + the full-year load+run smoke on `data/examples`. Commit.

### Task 1.x: Any other required-key migrations from Task 0.1

- [ ] Repeat the Task-1.1 TDD shape for each remaining newly-required key the audit/load-smoke surfaced, until both `eec_full` and `examples` LOAD and RUN ≥1 full year on the 4.4.0 jar without a missing-parameter error. Each its own test + commit.

**Phase 1 exit criterion:** the 4.4.0 jar runs a full simulation (≥5 yr) on `eec_full` AND `examples` (native-4.4.0 configs) with no load/parameter error. **Note (audit scope):** a clean eec/BoB load proves the required-key migration only for the **non-bioen module surface** (eec/BoB don't activate bioen/genetics/economy, so their required keys are never exercised) — this is acceptable for the jar-swap scope (Baltic/bioen are Java-blocked) but must be recorded in the audit notes. Only then are Phases 2–3 reachable.

---

## Phase 2 — Cross-engine parity: Python ↔ Java-4.4.0 (CONCERN #1)

Goal: confirm 4.4.0-Java agrees with the Python engine at least as well as 4.3.3-Java did — using an **equivalence + distributional** test (NOT a mean-difference test), so a 4.4.0 change to variance/tails/shape can't pass unseen.

### Task 2.1: Build the cross-engine ensemble parity harness

**Files:** Create `scripts/cross_engine_parity_440.py` (or extend `scripts/validate_engines.py`). Reuse the ensemble plumbing from `scripts/ab_validate_440.py` (branch `feat/jar-swap-440-cutover`, `406ea5d`).

`scripts/validate_engines.py` today is single-run, final-year biomass, 1-OoM, and feeds the RAW config to the jar (fails on 4.4.0). The new harness must:
- [ ] **Build it to:** load+canonicalize a config once; JAVA arm = `write_temp_config(cfg, out, target_version="4.4.0")` then run on the 4.4.0 jar by explicit path; PYTHON arm = `osmose.engine` on the canonicalized config. Run **N replicates** per arm with varied seeds (`simulation.fixed.seed.enabled` defaults false → time-seeded). Read per-species **biomass, yield, meanTL/size-structure, AND fishing-mortality (F)** via `osmose.results` (size-structure + F are more diagnostic of the swap's actual changes — larval units, bioen ingestion — than biomass alone; Spence et al. 2021 doi:10.3354/meps13870; Soudijn et al. 2021 doi:10.1073/pnas.1917079118). Compare on the **log scale** (biomass is ~log-normal). Discard a stated **spin-up window** before computing statistics. The comparison per species per metric is **distributional + equivalence**, NOT a mean delta:
  - **Equivalence test (TOST / 90% CI-overlap):** PASS only if the CI of (Python − Java) lies within a pre-registered margin Δ (per species; Lakens & Delacre 2020). This is the statistically correct "the engines agree" test — a *failure to find a difference* is NOT equivalence.
  - **Distribution test:** two-sample KS (or Mann–Whitney) on the per-replicate distributions + a variance-ratio (Levene/F) check, so equal means can't hide a variance/shape change.
  - **Bounded skill metric:** report MEF (modelling efficiency) and/or Spearman correlation on the per-species vector + time series (Olsen et al. 2016; Rynne et al. 2025) — the currency the MEM field uses, not OoM bands.
  - 1-OoM is retained ONLY as a catastrophic-divergence tripwire, not the gate.
- [ ] **Within-engine determinism check (separate):** with `simulation.fixed.seed.enabled=true` + a fixed seed, assert each engine reproduces itself run-to-run (tolerance ~0) — confirms the harness/IO is stable and the engine deterministic given a seed. Do NOT use a fixed seed for the cross-engine comparison (PCG64≠MT19937 → same seed ≠ matched stream; per CLAUDE.md).
- [ ] **N sizing (not asserted):** run a small pilot to estimate each species' inter-run CV, then size N so the mean-difference CI half-width < Δ/2 for the equivalence test (power formula, Shieh 2016 doi:10.1371/journal.pone.0162093). Where the affordable N can't reach Δ, REPORT the minimum-detectable-difference at that N so the gate's resolution is explicit. Args: `--config-dir`, `--years`, `--n`, `--spinup`, `--margin`, `--jar`.
- [ ] Commit the harness.

### Task 2.2: Establish the Python-vs-4.3.3-Java baseline

- [ ] Run the harness with the **4.3.3** jar on `eec_full` + `examples`. Record the per-species Python-vs-4.3.3-Java equivalence result + each arm's CV + the chosen Δ. This is the REFERENCE. **Flag any species whose 4.3.3 baseline agreement is already marginal as "low-confidence"** — those must NOT pass Phase 2.3 on a purely relative criterion.

### Task 2.3: Validate Python-vs-4.4.0-Java (the gate)

- [ ] Run the harness with the **4.4.0** jar on `eec_full` + `examples` (same N/years/spinup/Δ as the baseline).
- [ ] **GATE (relative AND absolute):** per species, 4.4.0-Java must (a) agree with Python **no worse than** 4.3.3-Java did, AND (b) pass an **absolute** equivalence bound Δ (so a poor 4.3.3 baseline can't rubber-stamp the swap), AND (c) show no KS/variance-ratio divergence beyond the baseline. PASS → Phase 3. If 4.4.0-Java drifts further than 4.3.3 OR fails the distribution test for some species → a 4.4.0 dynamics change the Python engine doesn't mirror; diagnose before proceeding. Report the per-species table (Python, 4.3.3-Java, 4.4.0-Java distributions; equivalence verdict; KS p; variance ratio; MEF; low-confidence flags).

### Task 2.4: Baltic config-round-trip parity gate (the swap DOES touch Baltic via I/O)

The cutover flips the default `target_version` to 4.4.0 and routes every config through the migration layer — so Baltic's config round-trip changes even though it never runs on the Java jar. Confirm that does NOT perturb the Python-engine Baltic results.
- [ ] Read `data/baltic` → migrate to 4.4.0 keys (`canonicalize_config` / the post-cutover default write) → write → re-read → run the Python engine. Assert per-species Python outputs are UNCHANGED (equivalence, tolerance ~0 beyond RNG) vs the pre-cutover Python-engine Baltic baseline. This closes the one channel by which the swap can affect the (otherwise Python-only, ICES/HOLAS-3-validated) Baltic config. Commit.

---

## Phase 3 — ICES / empirical validation (CONCERN #2)

**Reframed (review-grounded — DO NOT over-claim).** The review (ICES-MCP + scite verified) established that the Java-runnable configs are **structural demo/parity configs, NOT ICES-calibrated**: `data/eec_full` is Verley's EEC v3u2 demo whose "targets" are *seeding* biomass (`population.seeding.biomass.spN`), not fitted ICES SSB; `examples` is a Bay-of-Biscay demo. So "is species X inside the ICES SSB envelope?" is **near-vacuous for ALL three engines** — a species can be out-of-envelope by design or in by coincidence. Therefore Phase 3 is **NOT an empirical-realism gate**; it is a **cross-engine empirical-CONSISTENCY check + a catastrophic-divergence tripwire.** The genuine empirical anchor is the **Baltic ICES/HOLAS-3 validation on the Python engine** (Java-blocked, swap-unaffected except via the I/O round-trip covered by Task 2.4). State this bluntly in the docs.

Additional verified caveats to honour: only **~2/14 EEC species** map cleanly to a tonnes-unit 7.d ICES stock (`sol.27.7d`, `ple.27.7d`); ~6 have NO ICES assessment; the rest are spatial/scale-mismatched. BoB is somewhat better (`ane.27.8`, `pil.27.8abd`, `sol.27.8ab`, `hke.27.8c9a`), but the pelagics are NEA-basin-wide stocks. The validator compares **total model biomass vs ICES SSB** (a structural over-estimate — SSB excludes immature fish) and **loads but does NOT gate reference points** (`compare_outputs_to_ices` never reads `reference_points`) — so the original "reference points checked" claim was wrong; checking them would need new code. No EEC/BoB snapshot exists (only `data/baltic*/reference/ices_snapshots/`).

### Task 3.1: Build the eec/BoB ICES snapshot (mechanical; value-limited)

**Files:** read `osmose/validation/ices.py` (`IcesSnapshot`, `load_snapshot`, `compare_outputs_to_ices(results, snapshot, *, window_years, ices_window)`, the `index.json` manifest format: `model_species_to_ices_stocks`/`units_by_stock`/`advice_year_by_stock`); ICES MCP (`list_stocks`, `get_stock_assessment`, `get_reference_points`).

- [ ] Build a snapshot dir for eec/BoB (none exists) via the ICES MCP, matching the validator's layout. Use ONLY the cleanly-mapping tonnes-unit stocks (EEC: sole→`sol.27.7d`, plaice→`ple.27.7d`; BoB: anchovy→`ane.27.8`, sardine→`pil.27.8abd`, sole→`sol.27.8ab`, hake→`hke.27.8c9a`); mark scale-mismatched/NEA-wide stocks `index`-unit or exclude them; record the species with NO assessment as uncovered. Commit the snapshot + the honest coverage map. (Cite the spatial/scale caveats in the manifest comments.)

### Task 3.2: Cross-engine empirical-consistency check (the actual Phase-3 test)

- [ ] Run multi-year ensembles of `eec_full` (+ `examples`) on {Python, 4.3.3-Java, 4.4.0-Java} (native-4.4.0 config for the 4.4.0 arm). Feed per-species biomass through `compare_outputs_to_ices(...)` AND reuse the Phase-2 harness's **F and size-structure/MTL** outputs (more diagnostic of the swap's larval-units + bioen-ingestion changes than biomass alone).
- [ ] **GATE (consistency, not realism):** all three engines must land in the **same relation** to the (caveated) ICES envelope per species — 4.4.0-Java must not push a species across the in/out boundary relative to 4.3.3-Java/Python beyond the cross-engine noise, and must not diverge in F/MTL beyond Phase-2's bar. This is an order-of-magnitude SANITY check that catches a gross 4.4.0 regression; it does NOT certify realism (the configs aren't ICES-calibrated). Report the per-species {in/out, biomass, F, MTL} table for the three engines.
- [ ] **Docs must state plainly:** "eec/BoB are uncalibrated demo configs; this gate is cross-engine consistency, not empirical validation. The real ICES/HOLAS-3 validation is Baltic, on the Python engine, which the swap does not change (its config round-trip is verified unchanged by Task 2.4). Empirical confidence in the swap is transitive: Python is empirically validated (Baltic) and unchanged; 4.4.0-Java matches Python on eec/BoB (Phase 2)."

---

## Phase 4 — Cutover (only if Phases 2 AND 3 PASS)

Execute the cutover exactly as specified in `docs/superpowers/plans/2026-06-18-jar-swap-440.md` Tasks 6–9 (they remain valid and were 3-round-reviewed):
- [ ] **Task 6:** version-aware jar selection (`_pick_default_jar`, highest version) + `ui/state.py:42` default → 4.4.0; **KEEP the 4.3.3 jar** (rollback; the selector picks 4.4.0).
- [ ] **Task 7:** flip the write-default `target_version` `4.3.3`→`4.4.0` in `osmose/config/writer.py:63`, `ui/pages/run.py:136`, `osmose/calibration/problem.py:458` (+ the named test updates: `test_writer_default_target_emits_old_keys`, `test_write_temp_config_default_target_emits_old_keys`, `test_export_writes_target_format`, `test_pr2_load_write_roundtrip_coherent` need explicit `target_version="4.3.3"`; `test_calibration_java_cmd_reverse_maps_override_keys` rewritten to NEW-key `-P` output).
- [ ] **Task 8:** ROUTING for native-4.4.0 multi-file Export (re-home `species.maturity.{eta,r,m0,m1}` to the bioenergetics file).
- [ ] **Task 9:** update version refs (`tests/test_state.py:79`, `tests/test_engine_java_comparison.py:39` — and CONVERT its config to native-4.4.0 before the jar, per that plan's blocker fix; `scripts/validate_engines.py:23`; `osmose/demo.py:232` default + stale docstring; `timeseries.py:4`/`foraging_mortality.py:11` docstrings; `README.md:14/59`).

---

## Phase 5 — Deploy + docs

Execute `docs/superpowers/plans/2026-06-18-jar-swap-440.md` Tasks 10–11:
- [ ] **Task 10:** `deploy.sh` pre-copy assertion that the 4.4.0 jar is present in the source tree (do NOT `rm` the 4.3.3 jar — rollback).
- [ ] **Task 11 + this plan's validation results:** parity-note in CLAUDE.md / `docs/parity-roadmap.md` — document the cross-engine equivalence result (Phase 2: 4.4.0-Java ≈ Python, TOST+distributional, eec/BoB), the Baltic config-round-trip result (Task 2.4), and the cross-engine empirical-CONSISTENCY result (Phase 3) — with the honest framing that Phase 3 is consistency-not-realism (eec/BoB uncalibrated) and the real empirical anchor is Baltic-on-Python. CHANGELOG entry (engine → 4.4.0; the resource-forcing + larval migrations; validation summary with its stated limits).

---

## Notes

- **The validation is two-tiered but EVIDENTIALLY ASYMMETRIC** (review-corrected): Phase 2 (cross-engine equivalence) is the strong tier and the real gate; Phase 3 (ICES) is a cross-engine *consistency*/tripwire check on uncalibrated demo configs, NOT an empirical-realism gate. Both are *relative-AND-absolute* (4.4.0-Java must do no worse than 4.3.3-Java AND pass an absolute equivalence margin). The gates use **equivalence testing (TOST) + distributional tests**, not mean-difference (a mean comparison misses variance/tail/shape changes — the regression classes a 4.4.0 stochastic change produces). Empirical confidence is transitive: the **Python engine is the ICES/HOLAS-3-validated, swap-unaffected** reference (Baltic), and Phase 2 shows 4.4.0-Java matches Python on the Java path.
- **Phase 1 is the hard prerequisite** — without the resource-forcing migration (emit `species.biomass.{file,varname,nsteps.year}.spN`; varname = `species.name`/`ltl.name`, NOT NetCDF-sniffed; BoB needs the legacy `ltl.*` chain first) 4.4.0 can't load eec/BoB and NO validation is possible. This is harder than a key-rename and is where the original effort stopped.
- **Audit completeness is the primary evidence** (Phase 0) — the gate-failure proved a dynamics-only audit is incomplete; the newly-REQUIRED-key axis must be covered to completion (and is complete only for the non-bioen eec/BoB module surface).
- **The swap is NOT inert for Baltic** — it changes Baltic's config round-trip via the default `target_version` flip + key migration (Task 2.4 gate), even though Baltic never runs on the Java jar.
- **Resume artifacts:** A/B harness `scripts/ab_validate_440.py` on branch `feat/jar-swap-440-cutover` (`406ea5d`); the merged value layer in `osmose/config/aliases.py`; this plan + the prior `2026-06-18-jar-swap-440.md` for cutover mechanics; the required-keys audit notes (Phase 0).
- **Out of scope (still):** 4.4.0 opt-in new features (schema + Python engine), dual-jar runtime selector, restart-file support, the removed `species.lmax` growth-cap. Prod redeploy needs user sudo.
- **Reconsider-before-resuming:** the swap remains low-present-value (Java path near-vestigial, Python primary, no new feature exposed) and the deep review only *raised* the realistic cost (the resource-forcing migration is two-scheme + the empirical tier is weaker than hoped). Resume only when a concrete 4.4.0 Java feature justifies it — then this plan delivers it with the strongest validation actually achievable (Phase-2 cross-engine equivalence + the transitive empirical argument), honestly bounded.
