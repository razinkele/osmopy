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

2. **The honest evidence chain is audit (deterministic) → cross-engine (statistical) → empirical (ICES).** Runtime parity can't *prove* equivalence for a stochastic model; it can only *fail to refute* it. So the primary evidence is the **complete source audit** (Phase 0); the cross-engine ensemble (Phase 2) and ICES validation (Phase 3) are confirmations that catch audit misses and confirm realism. Ensemble **means** (not single runs) are RNG-independent in expectation, so they let us tighten the comparison well below the single-run 1-OoM floor.

**Coverage honesty (must be stated in the final docs):** only `data/eec_full` (Eastern English Channel) and `data/examples` (Bay of Biscay) run on the Java engine. Baltic/bioen/`nbackground>0` configs are Java-BLOCKED by design (Python-engine-only) — they have **no Java cross-check and no Java ICES check**. Their realism rests on the Python engine's existing ICES/HOLAS-3 validation, which the jar swap does not touch (the Python engine is unchanged). The plan validates the Java path on eec/BoB and relies transitively on cross-engine agreement there.

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

Background: eec_full resource species (`species.type.spN;resource`, sp14–sp23) carry `species.file.spN;eec_ltlbiomassTons.nc`. 4.4.0 requires, per resource species: `species.biomass.mode.spN` (= `NETCDF_BIOMASS` when a NetCDF file is supplied) + `species.biomass.varname.spN` (the variable name INSIDE that NetCDF). `examples` uses the legacy `ltl.netcdf.file` + `ltl.name.rscN` form — handle both.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_config_migration_440.py`:
```python
def test_4_4_0_write_adds_resource_biomass_mode_and_varname(tmp_path):
    from osmose.config.aliases import to_target_keys

    # a resource species with a NetCDF forcing file -> 4.4.0 needs mode + varname
    cfg = {
        "species.type.sp14": "resource",
        "species.file.sp14": "eec_ltlbiomassTons.nc",
        "species.biomass.varname.sp14": "BiomassTons_sp14",  # provided up front (Step 3 derives it when absent)
    }
    out = to_target_keys(cfg, "4.4.0")
    assert out["species.biomass.mode.sp14"] == "NETCDF_BIOMASS"
    assert out["species.biomass.varname.sp14"] == "BiomassTons_sp14"
    assert out["species.file.sp14"] == "eec_ltlbiomassTons.nc"  # original file ref retained
```
- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_config_migration_440.py -k resource_biomass -q`
- [ ] **Step 3: Implement** in `osmose/config/aliases.py` — in the `to_target_keys` 4.4.0 branch, after the existing drop-guards/larval-migration, add a resource-forcing step:
  - For each `species.type.spN == "resource"` that has a `species.file.spN` NetCDF: set `species.biomass.mode.spN = "NETCDF_BIOMASS"` if absent.
  - Set `species.biomass.varname.spN` if absent. **Varname derivation** (the non-trivial part — confirm the real convention against the 4.4.0 wiki + the NetCDF): try, in order, (a) an explicit `species.biomass.varname.spN` already in the config, (b) a deterministic convention if 4.4.0 defines one (e.g. derived from `species.name.spN`), (c) read the NetCDF referenced by `species.file.spN` via xarray and pick the biomass data variable (error loudly if ambiguous — multiple data vars). Encode the chosen rule with a comment citing the 4.4.0 source/wiki. Implement the `ltl.netcdf.file`+`ltl.name.rscN` legacy form too (map `ltl.name.rscN` → the varname).
  - Keep the original `species.file.spN` (4.4.0 still reads the file path; it adds mode/varname alongside).
- [ ] **Step 4: Verify** the unit test passes, then a **LOAD smoke test against the real jar** (requires the fetched 4.4.0 jar): write `data/eec_full` migrated to native 4.4.0 (`write_temp_config(reader.read(...), tmp, target_version="4.4.0")`) and run `java -jar osmose_4.4.0-...jar <tmp master>` for 0–1 years — confirm it LOADS (no "parameter missing"/`ResourceForcing.init` error). Repeat for `data/examples`. If a DIFFERENT missing key appears, that's the next Phase-1 task (loop Task 0.1's empirical method). ruff/pyright clean.
- [ ] **Step 5: Commit.**
```bash
git add osmose/config/aliases.py tests/test_config_migration_440.py
git commit -m "feat(config): migrate resource/LTL forcing to 4.4.0 species.biomass.mode/varname"
```

### Task 1.x: Any other required-key migrations from Task 0.1

- [ ] Repeat the Task-1.1 TDD shape for each remaining newly-required key the audit/load-smoke surfaced, until both `eec_full` and `examples` LOAD and run a short simulation on the 4.4.0 jar without a missing-parameter error. Each its own test + commit.

**Phase 1 exit criterion:** the 4.4.0 jar runs a full simulation (≥5 yr) on `eec_full` AND `examples` (native-4.4.0 configs) with no load/parameter error. Only then are Phases 2–3 reachable.

---

## Phase 2 — Cross-engine parity: Python ↔ Java-4.4.0 (CONCERN #1)

Goal: confirm 4.4.0-Java agrees with the Python engine at least as well as 4.3.3-Java did. Statistical, ensemble-based.

### Task 2.1: Build the cross-engine ensemble parity harness

**Files:** Create `scripts/cross_engine_parity_440.py` (or extend `scripts/validate_engines.py`). Reuse the ensemble plumbing from `scripts/ab_validate_440.py` (branch `feat/jar-swap-440-cutover`, `406ea5d`).

`scripts/validate_engines.py` today is single-run, final-year biomass, 1-OoM, and feeds the RAW config to the jar (which fails on 4.4.0). The new harness must:
- [ ] **Build it to:** load+canonicalize a config once; for the JAVA arm write a native-4.4.0 config via `write_temp_config(cfg, out, target_version="4.4.0")` (so the resource-forcing migration applies and 4.4.0 loads) and run it on the 4.4.0 jar by explicit path; for the PYTHON arm run `osmose.engine` on the canonicalized config. Run **N replicates** of each arm with varied seeds (Python: vary `seed=`; Java: distinct subprocess runs, time-seeded since `simulation.fixed.seed.enabled` defaults false). Read per-species biomass/abundance/yield/meanTL via `osmose.results`. Compare per-species **ensemble means ± CI** (final-year AND time-averaged). Pass criterion: per-species relative difference of means within a configurable bar (default tighter than 1 OoM — e.g. ≤30%; the mean is RNG-independent in expectation), reported alongside each arm's own run-to-run CV so the reader sees signal vs noise. Args: `--config-dir`, `--years`, `--n`, `--jar`.
- [ ] Commit the harness.

### Task 2.2: Establish the Python-vs-4.3.3-Java baseline

- [ ] **Step 1:** Run the harness with the **4.3.3** jar on `eec_full` + `examples` (ensemble). Record the per-species Python-vs-4.3.3-Java mean agreement + CVs. This is the REFERENCE: the agreement the port was validated at. Save the table to the audit notes / a baseline artifact.

### Task 2.3: Validate Python-vs-4.4.0-Java (the gate)

- [ ] **Step 1:** Run the harness with the **4.4.0** jar on `eec_full` + `examples` (ensemble, same N/years as the baseline).
- [ ] **Step 2: GATE.** Per species, 4.4.0-Java must agree with Python **no worse than** 4.3.3-Java did (within the baseline's tolerance + noise). PASS → Phase 3. If 4.4.0-Java drifts from Python *further* than 4.3.3-Java did for some species, that flags a 4.4.0 dynamics change the Python engine doesn't mirror → diagnose (is it an audit miss? a new non-opt-in default?) before proceeding. Report the per-species comparison table (Python mean, 4.3.3-Java mean, 4.4.0-Java mean, deltas, noise floor, PASS/FAIL).

---

## Phase 3 — ICES / empirical validation (CONCERN #2)

Goal: confirm the jar swap doesn't degrade ecological realism — the 4.4.0-Java outputs reproduce ICES/empirical targets as well as 4.3.3-Java / Python do.

### Task 3.1: Map the Java-runnable configs to ICES reference data

**Files:** read `osmose/validation/ices.py` (`IcesSnapshot`, `SpeciesBiomassComparison`, the `model_species_to_ices_stocks`/`units_by_stock` manifest); ICES MCP tools (`mcp__ices__list_stocks`, `get_stock_assessment`, `get_reference_points`, `get_survey_cpue_*`).

- [ ] **Step 1:** Determine whether an ICES snapshot exists for the Java-runnable configs. `eec_full` = Eastern English Channel focal species; `examples` = Bay of Biscay. For each focal (non-resource) species, map to its ICES stock(s) via the validator's manifest, or build the snapshot from the ICES MCP (`list_stocks` → `get_stock_assessment`/`get_reference_points` for the relevant ICES divisions: 7.d for EEC, 8 for BoB). Record the model-species→ICES-stock mapping + the SSB/F reference window. If no ICES coverage exists for these configs' species, state that and fall back to whatever empirical targets the configs were originally tuned to (document the limitation).
- [ ] **Step 2:** Commit the mapping/snapshot artifact (under the validation data dir the validator expects).

### Task 3.2: Validate 4.4.0-Java (and Python) outputs against ICES (the gate)

- [ ] **Step 1:** Run a multi-year ensemble of `eec_full` (+ `examples`) on the **4.4.0** jar (native-4.4.0 config) AND the Python engine. Feed the per-species biomass outputs through `osmose/validation/ices.py`'s `SpeciesBiomassComparison` (model biomass vs the ICES SSB envelope across tonnes-unit stocks, reference points checked).
- [ ] **Step 2:** Run the SAME validation on the **4.3.3**-Java outputs (the reference).
- [ ] **Step 3: GATE.** 4.4.0-Java must reproduce the ICES SSB envelope / reference-point status **as well as** 4.3.3-Java and the Python engine do — no degradation in which species fall inside the empirical envelope. Report a per-species in/out-of-envelope table for {Python, 4.3.3-Java, 4.4.0-Java}. PASS → Phase 4. If 4.4.0-Java pushes a previously-in-envelope species out (beyond the cross-engine noise), the swap degrades realism → diagnose before cutover.
- [ ] **Note (coverage):** the Baltic config — OSMOPY's strongest ICES/HOLAS-3-validated config — is Java-BLOCKED and runs only on the Python engine, which the swap doesn't change. So its empirical validation is unaffected and out of this gate's scope; record that explicitly.

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
- [ ] **Task 11 + this plan's validation results:** parity-note in CLAUDE.md / `docs/parity-roadmap.md` — document BOTH the cross-engine result (4.4.0-Java ≈ Python, ensemble, eec/BoB) AND the ICES result, plus the coverage-honesty statement (Java path = eec/BoB only; Baltic ICES validation is Python-only/unaffected). CHANGELOG entry (engine → 4.4.0; the resource-forcing + larval migrations; validation summary).

---

## Notes

- **The two validation tiers ARE the point of this plan** — cross-engine (Phase 2) and ICES/empirical (Phase 3) gate the cutover. Neither is a tight-numerical-parity check (impossible for a stochastic model with differing RNGs); both are *relative* gates: 4.4.0-Java must do **no worse than** 4.3.3-Java against the same reference (Python in Phase 2, ICES in Phase 3).
- **Phase 1 is the hard prerequisite** — without the resource-forcing (+ any other required-key) migration, 4.4.0 can't load eec/BoB and NO validation is possible. This is where the original effort stopped.
- **Audit completeness is the primary evidence** (Phase 0) — the gate-failure proved a dynamics-only audit is incomplete; the required-key axis must be covered to completion.
- **Resume artifacts:** A/B harness `scripts/ab_validate_440.py` on branch `feat/jar-swap-440-cutover` (`406ea5d`); the merged value layer in `osmose/config/aliases.py`; this plan + the prior `2026-06-18-jar-swap-440.{md}` for cutover mechanics.
- **Out of scope (still):** 4.4.0 opt-in new features (schema + Python engine), dual-jar runtime selector, restart-file support, the removed `species.lmax` growth-cap. Prod redeploy needs user sudo.
- **Reconsider-before-resuming:** the swap remains low-present-value (Java path near-vestigial, Python primary, no new feature exposed). Resume only when a concrete 4.4.0 Java feature justifies the Phase-0/1 migration cost — at which point this plan delivers it with the parity + empirical validation the user requires.
