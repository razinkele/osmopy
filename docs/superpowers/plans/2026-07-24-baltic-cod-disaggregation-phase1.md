# Baltic Cod E/W Disaggregation — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).
> **DEPENDS ON:** the RV recruitment gate (`docs/superpowers/plans/2026-07-24-baltic-rv-recruitment-gate.md`) must be implemented and validated first — cod-east recruitment is driven by it.
> **DESIGN:** `docs/superpowers/specs/2026-07-24-baltic-stock-disaggregation-design.md` (fidelity-reviewed).

**Goal:** Split the aggregated cod stock into eastern (cod.27.24-32) and western (cod.27.22-24) as two OSMOSE focal species with distinct distributions, growth, recruitment, and fishing — so the model can represent a collapsed eastern stock coexisting with a healthier western one, the qualitative structure the single cod stock averages away.

**Architecture:** Add one focal species (8 → 9). Because OSMOSE indexes focal, LTL, and background species contiguously in one `species.name.sp{idx}` namespace (focal 0-7, LTL 8-13, background 14-15), inserting a focal species requires **re-indexing** LTL and background upward — the crux of this phase. Then apply the spec's per-sub-stock recipe to cod-east and cod-west, re-calibrate the 9-species set, and validate.

**Tech Stack:** Python 3.12; OSMOSE config CSVs; `scripts/calibrate_baltic.py` + `apply_calibration.py`; `scripts/baltic_stability_certify.py`; the ICES MCP/skill for sub-stock targets; pytest.

## Global Constraints

- Run tests with `.venv/bin/python -m pytest`; lint `.venv/bin/ruff check`.
- Corrected science (from the fidelity review — MUST hold): eastern cod are the **low-salinity-adapted deep-basin reproducers** (eggs neutral ~14 PSU, sperm to ~11 PSU), recruitment gated by the **reproductive volume** (deep salinity ≥11 & O₂ ≥2), collapse needs **~doubled M + recruitment failure + impaired condition** (not growth alone; onset 2000s). Western cod = higher-salinity Belt/Arkona type, standard recruitment. **SD24 (Arkona) is a mixing cell**, not pure eastern.
- Cod's ICES envelope is SSB; per-sub-stock targets are SSB (cod.27.24-32 ≈ 60-77 kt post-collapse; cod.27.22-24 separately).
- Keep every unsplit species' parameters unchanged through the re-index (only their `sp{idx}` changes).

---

### Task 1: Config re-indexing tool (the crux)

**Files:**
- Create: `scripts/reindex_species.py`
- Test: `tests/test_reindex_species.py`
- (applies to) all `data/baltic/baltic_param-*.csv`, maps, and the predation matrix

**Interfaces:**
- Produces: `reindex(config_dir, shifts: dict[int,int])` — rewrites every `...sp{old}` / `...fsh{old}` key and every species-ordered matrix row/column to its new index, per a mapping, preserving values/comments.

**Plan:** to insert cod-east as sp1 (keeping cod-west at sp0), shift focal sp1-7 → sp2-8, LTL sp8-13 → sp9-14, background sp14-15 → sp15-16; then cod-east occupies the freed sp1. (Alternatively append cod-east as the new last focal index and shift only LTL/background — simpler mapping, decide in Step 1.) The tool must handle: per-species scalar keys (`species.*.spN`), the fishing `fshN` namespace, the predation-accessibility matrix (both a prey ROW and a predator COLUMN per species, in species order), movement `mapN` species references, and `simulation.nspecies`/`nresource` counts.

- [ ] **Step 1:** Read every per-species / per-matrix key pattern across the config (enumerate the families: species, predation matrix, fishing, movement, reproduction, init-pop, additional-mortality, background, LTL). Decide the index mapping (append-cod-east vs insert-at-sp1). **Write the failing test** with a small synthetic config asserting a known key/matrix cell moves to its new index and values are preserved.
- [ ] **Step 2:** Run test → FAIL (tool absent).
- [ ] **Step 3:** Implement `reindex_species.py` (line/field-preserving edits like `apply_calibration.set_key`; matrix row+column reorder). Bump `simulation.nspecies`.
- [ ] **Step 4:** Run test → PASS. Then round-trip the REAL Baltic config through the reindex with an IDENTITY mapping and assert `OsmoseConfigReader` reads it byte-for-byte equivalently (no accidental corruption).
- [ ] **Step 5:** Commit.

### Task 2: Insert cod-east + set cod-west params

**Files:** `data/baltic/baltic_param-species.csv`, `-reproduction.csv`, `-init-pop.csv`; `tests/test_baltic_cod_ew_species.py`

- [ ] Apply the reindex mapping (Task 1) to the committed config; rename `cod` → `cod_west` (sp0). Add `cod_east` (the inserted index): von Bertalanffy + L-W from FishBase eastern-cod / Svedäng condition decline (impaired condition, NOT a heritable slow-growth trait); egg size larger (low-salinity-adapted buoyancy); maturity/lifespan per eastern stock. **Test:** the 9-species config loads; both cod species present with distinct linf/egg-size; unsplit species' params unchanged (diff only the sp index). Commit.

### Task 3: Salinity-niched distribution maps

**Files:** `data/baltic/maps/cod_east_*.csv`, `cod_west_*.csv`; `-movement.csv`

- [ ] cod-west maps = western SD22-24 (Belt/Arkona, saline); cod-east maps = eastern deep basins (Bornholm/Gdańsk/Gotland, SD24-32) with SD24 as a shared/transition cell. Enable the salinity occupancy gate for both cod species (deep-saline preference). **Verify** the two occupancy footprints are largely disjoint (the niche actually separates them) before proceeding — a Phase-1 go/no-go. Commit.

### Task 4: Predation-accessibility matrix expansion

**Files:** `data/baltic/predation-accessibility.csv`

- [ ] Expand the matrix to 9 focal (+ LTL) rows and columns. cod-east and cod-west inherit cod's diet (as predators) and cod's predator set (as prey), hand-adjusted from eastern/western diet literature (eastern cod more sprat/benthos-dependent, prey-limited). **Test:** matrix is square over the new species set, rows/columns sum sanely, no NaN. Commit.

### Task 5: Disaggregate the ICES target

**Files:** `data/baltic/reference/biomass_targets.csv`; provenance note

- [ ] Replace the single `cod` SSB row with `cod_east` (cod.27.24-32 SSB, ~60-77 kt post-collapse) and `cod_west` (cod.27.22-24 SSB) rows, pulled via the ICES MCP/skill. Keep the SSB reference type. Weights: both high (well-assessed). Document the SD24 mixing caveat. Commit.

### Task 6: RV niche + calibration params

**Files:** `data/baltic/baltic_param-reproduction.csv`; `scripts/calibrate_baltic.py` (phase-13 param set)

- [ ] Enable the RV recruitment gate on **cod-east only** (its index), mode per the RV-gate plan's Task-3 result (`raw_cap`/declining series if the eastern collapse must emerge; `mean_preserving` if only variability). Pair with elevated additional M on cod-east (the collapse needs M + recruitment failure together). cod-west: standard Shepherd recruitment, no RV gate. Add cod-east/west mortality/fishing/recruitment params to the phase-13 free-param set (watch the d≤20 UQ cap; the DE calibration handles the added dims). Commit.

### Task 7: Re-calibrate + validate

- [ ] Re-run the phase-13 equilibrium calibration on the 9-species config (threading fix; ~4-6 h). Apply via `apply_calibration.py` (extend its species loop to 9). **Acceptance:** cod-west near its ICES SSB envelope; cod-east depressed toward its post-collapse SSB (the qualitative eastern-collapse structure); no *unintended* collapse of other species; the aggregate cod (east+west) not worse than the pre-split ×1.2. Re-certify long-horizon stability and compare to the 2/8 pre-split baseline. Document; commit.

---

## Self-Review

- **Spec coverage:** the spec's per-sub-stock recipe maps to Tasks 2-7 (params, maps, matrix, target, calibration, validate); Task 1 (re-indexing) is the OSMOSE-mechanics prerequisite the spec's §3 implies. RV-gate dependency is stated in the header and Task 6.
- **Placeholder scan:** the largest genuine unknowns are called out as explicit *decisions within steps* (index mapping in Task 1 Step 1; RV mode in Task 6 from the RV-gate result) rather than hidden TODOs. Task 1/round-trip identity check guards the re-index. Sub-tasks that would each carry hundreds of lines of matrix/param data (Tasks 3-6) are specified by procedure + acceptance rather than pasted data — appropriate for config-data tasks; the executor authors the data against the stated science + ICES sources.
- **Scope check:** this is Phase 1 (cod only) of the phased spec; herring (Phase 2) and flounder (Phase 3) are separate plans. Task 1's re-indexing tool is reusable for them.
- **Type consistency:** `reindex(config_dir, shifts)` and the `cod_east`/`cod_west` naming are used consistently across tasks; SSB-target type held throughout.
- **Open risk:** the re-index (Task 1) is the highest-risk step — a mis-mapped index silently corrupts the model. The identity round-trip check (Task 1 Step 4) and "unsplit params unchanged" assertions (Task 2) are the guards. If the salinity niche doesn't separate cod-east/west spatially (Task 3 go/no-go), the split is cosmetic — stop and reconsider before calibrating.
