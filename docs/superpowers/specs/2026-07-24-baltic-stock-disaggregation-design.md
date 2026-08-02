# Baltic Stock Disaggregation — Design Spec

> Date: 2026-07-24. Status: design (pre-plan). Scientifically fidelity-reviewed
> (science-reviewer + primary-literature check, 2026-07-24). Splits salinity-
> structured Baltic stocks that the model currently aggregates into sub-populations,
> each an OSMOSE focal species with its own reproduction/growth/fishing parameters
> and a salinity/oxygen-niched distribution. **Phased** (cod → herring → flounder);
> each phase is working, re-calibrated, validated software.

## 1. Goal & motivation

The committed model reaches 5/8 ICES envelopes but represents each species as ONE
stock, averaging over the salinity gradient (0–35 PSU, mean 9.8) that defines Baltic
fish ecology. `biomass_targets.csv` itself flags the aggregation: cod lumps eastern
(collapsed) + western; herring lumps ≥4 management units; "flounder" is two species.
Disaggregating lets the model represent the real spatial structure — e.g. a collapsed
eastern cod coexisting with a healthier western stock — and removes the aggregation
bias. Success = each phase's sub-stocks are individually near their ICES sub-targets
with the correct qualitative structure (eastern-cod collapse, Bothnian vs Riga herring
growth contrast), without regressing the unsplit species.

## 2. Scientific basis (load-bearing — corrected in fidelity review)

**These mechanisms drive the design; getting them right is the point of the split.**

### 2.1 Cod — eastern (cod.27.24-32) vs western (cod.27.22-24)

- **Division:** ICES codes verified. **SD24 (Arkona) is a mixing zone** shared by both
  stocks (separated by otolith/genetic ID, not a clean line) — model it as a
  transition cell, not pure eastern.
- **Reproduction mechanism (NOT surface occupancy):** eastern cod spawn in the **deep
  saline basin water**; egg/larval survival is gated by the **reproductive volume** —
  water simultaneously **salinity ≥ ~11 PSU** (egg neutral buoyancy ~14 PSU; below
  threshold eggs sink into anoxic deep and die) **AND oxygen ≥ ~2 ml/l**, with temp
  >1.5 °C (Plikshs et al. 1993; MacKenzie/Köster/Wieland; Nissling & Westin 1997).
  Post-inflow-decline hypoxia shrank this volume to essentially the **Bornholm Basin**.
- **Adaptation direction (do NOT invert):** eastern cod are the **low-salinity-adapted
  reproducers** (neutral buoyancy ~14 PSU, sperm to ~11 PSU); western cod are the
  *higher*-salinity type (Belt Sea/Arkona). The collapse narrative is that even the
  low-salinity specialist fails once the deep reproductive volume collapses.
- **Collapse needs three levers together, not growth alone:** roughly-**doubled natural
  mortality M** + chronic **recruitment failure** + **impaired growth/condition**
  (phenotypic — hypoxia, prey loss, *Contracaecum* parasites tied to grey-seal recovery;
  onset **2000s–2010s**, not "post-2015"; otolith ageing unreliable post-2007 → VBGF
  params carry large uncertainty). Lowering L∞/K alone only shrinks cod; M + egg-survival
  must move with it.
- **What the model can/can't do:** the salinity **occupancy** gate uses a single-layer
  bottom-salinity field and has **no oxygen field**, so it CANNOT represent the deep
  reproductive volume. → eastern-cod recruitment must be driven by a **prescribed
  reproductive-volume / egg-survival forcing time series** (derived from RV literature or
  a salinity+oxygen product), coupling to the salinity-*spawning* work (the
  `fix+baltic-salinity-spawning` worktree), NOT the occupancy gate. Adding an **oxygen
  co-limiting field** is a prerequisite extension flagged here.

### 2.2 Herring — FOUR units (not three)

Verified ICES units: **`her.27.20-24`** Western Baltic Spring Spawning (spring-spawn +
Kattegat/Skagerrak feeding migration, mixes with North Sea autumn spawners there);
**`her.27.25-2932`** Central Baltic; **`her.27.28`** **Gulf of Riga** (distinct);
**`her.27.3031`** **Gulf of Bothnia** (distinct). Riga (fast-growing, warm, productive,
low-salinity embayment) and Bothnia (slow-growing, lean, cold, low-salinity) are
biologically **opposite** and geographically far apart — **never merge them**. If a
phase must cap at three, pair Riga with Central (adjacent), never with Bothnia. Target
is four.

### 2.3 Flounder — Platichthys flesus vs P. solemdali (best-justified split)

Two reproductively-isolated species (Momigliano et al. 2017 PNAS; **2018 Front. Mar.
Sci.** species description; 2019 ICES JMS). ***P. flesus***: **pelagic** eggs needing
higher salinity for buoyancy → deeper southern/western basins. ***P. solemdali***
(endemic, first Baltic endemic fish): **demersal** eggs viable to ~**6 PSU**, sperm
motile at low salinity → shallow coastal + northern low-salinity zones. **This is the
one split where a salinity-niched surface map is genuinely mechanistically defensible**
(solemdali's low-salinity demersal-egg viability gates its range against flesus).
Caveats: existing ICES flounder assessments (fle.27.2223, .2425, .2628…) **lump the two
species** → per-species recruitment/growth/fishing are assumption-driven; ranges
overlap (not a clean partition); the distinction is recent (2018).

## 3. Representation in OSMOSE

Each sub-stock = a new focal species (OSMOSE params are global per species). Its niche =
a **region base map** (ICES SD divisions) × niche modifier:
- **flounder** — salinity **occupancy** gate per species (solemdali low-salinity coastal;
  flesus higher-salinity offshore). The gate mechanism (now enabled) is appropriate here.
- **herring** — region base maps carry the units; salinity gate secondary (northern range
  limit only).
- **cod** — region base map for east/west adult distribution; **reproduction** driven by
  a prescribed RV/egg-survival forcing (§2.1), not the occupancy gate. Requires the
  spawning-stage salinity/oxygen coupling.

Focal species grow 8 → **~13** (cod×2, herring×4, sprat×1, flounder×2, + perch, pikeperch,
smelt, stickleback). Watch the **d ≤ 20 UQ-sampler cap** and DE-calibration cost as
species/params grow.

## 4. Per-sub-stock recipe (established by the cod PoC, then repeated)

1. **Species params** from FishBase/ICES for that sub-stock (growth, maturity, L–W,
   egg size) — with sub-stock specifics (eastern-cod impaired condition; Bothnian vs
   Riga herring growth contrast).
2. **Niche distribution:** region base map (SD divisions) × salinity gate (flounder,
   herring-north) or × RV forcing at spawning (cod).
3. **Predation-accessibility expansion:** the 8×8(+LTL) matrix → ~13×13(+LTL), each new
   species hand-authored as predator AND prey from diet literature. **Main cost/error
   surface.** Sub-stocks of one species inherit similar diet but differ by region.
4. **Target disaggregation:** split the aggregated `biomass_targets.csv` row into
   per-sub-stock ICES SD-level targets (via the ICES MCP/skill: cod.27.24-32 vs .22-24
   SSB; the four herring units; flounder — assumption-driven, document it).
5. **Calibration:** add the sub-stock's mortality/fishing/recruitment params + weight;
   re-calibrate the expanded set with the tightened Shepherd β bounds [1.0, 3.0] and the
   transform-aware write-back (`apply_calibration.py`).
6. **Validate:** fresh multi-seed 40-yr run; each sub-stock near its sub-target with the
   correct qualitative structure; unsplit species not regressed.

## 4a. Unified base & the stability gap (2026-07-24 reconciliation)

The SP-A stability branch is reconciled onto master: the oxygen forcing infra
(`physical_data.py`, `oxygen_function.py`) and salinity gate were already shared; the
stability objective (`osmose/calibration/stability.py`), the **certifier**
(`scripts/baltic_stability_certify.py`), the RV-gate design
(`docs/baltic_habitat_followup_2026-07-02.md`), and its negative-experiment findings are
now on master. The epsilon-constraint stability *calibration* integration
(`baltic_stability_sweep.py` + the `calibrate_baltic.py` port) is deferred — needed only
for stability re-calibration, not this project.

**Certified stability gap — ⚠ THIS GAP DOES NOT EXIST (corrected 2026-08-01; see the correction
section at the end).** The paragraph below is retained as written; every claim in it is superseded.

> ~~the committed 5/8 baseline is only **2/8 persistent-&-in-envelope over 50 yr × 5 seeds** (herring,
> stickleback PASS; cod, sprat, flounder are in-range on the 40-yr decade-mean but dip below the
> persistence floor over 50 yr — cod min 2.4 kt, flounder 1.1 kt; perch/pikeperch/smelt over-target).
> So the β-bounds re-fit got decade-means in range but did NOT stabilize the system — the branch's
> "params alone can't stabilize" holds for the 5/8 baseline too. The RV recruitment gate (Phase 0) and
> disaggregation are the structural levers for that gap.~~

**What is actually true.** `556ba3d` rescoped `persists` from the whole-run minimum (dominated by the
seeding bootstrap) to the final-decade minimum:

- **It is 5/8, not 2/8.** cod, sprat and flounder read `persists ✗` / `in-envelope ✓` — the artifact
  signature. Their "cod min 2.4 kt, flounder 1.1 kt" are **bootstrap** minima, not a 50-yr dip; these
  species do not dip at equilibrium. Corrected PASS set: cod, herring, sprat, flounder, stickleback.
- **The gap closes to zero.** The corrected 5/8 is *identical* to the in-envelope set, so "5/8 ICES vs
  2/8 stable" has no daylight in it. Structural: once `persists` uses the final decade and the mean is
  in-envelope, `in_envelope` is the only binding constraint. *(Recount from the committed table under
  the audit's classification, not a fresh run — only `--params current` has been re-certified.)*
- **"Params alone can't stabilize" is contradicted, not merely unsupported.** Its sole evidence was the
  defective flag. `docs/baltic_cod_east_M_revert_test_2026-08-01.md` is a *tested* single-parameter
  intervention moving cod_east 14.5–15.1 kt → 82.6–83.4 kt into envelope, 6/9 → 7/9.
- **Phase 0 therefore has no success criterion as written.** The persistence gap it targets is not
  there. The residual real failure is the percid `in_envelope` overshoot, which neither a cod-only RV
  gate nor cod E/W disaggregation is shown to address. Anyone executing Phase 0 against this section
  must define a target against `in_envelope`, not persistence.
  **Tracked as [#145](https://github.com/razinkele/osmopy/issues/145)** — Phase 0 is not invalidated
  (the RV gate may still be worth building on ecological grounds); its stated *success criterion* is.

## 5. Phases

- **Phase 0 (prerequisite) — RV recruitment gate for cod.** The mechanism is settled
  (branch's `baltic_habitat_followup_2026-07-02.md`, now on master): a per-step
  reproductive-volume metric `RV = Σ deep-basin cell_volume where (bottom_salinity ≥ 11 &
  bottom_O₂ ≥ 2)` → multiply cod B-H recruitment by `clip(RV/RV_ref, 0, 1)` in
  `reproduction.py` (cod-only initially). Forcing: extend `osmose/forcing/` to emit bottom
  salinity + bottom oxygen NetCDFs (the pipeline already handles `so` at depth; add the
  bottom-field selection + `o2b`). The oxygen infra is already on master. Validate with the
  reconciled certifier. Without this, eastern cod cannot collapse for the right reason.

  **~~and the whole system stays unstable (§4a)~~ — superseded (2026-08-01): the system is not
  unstable. Measured 5/8 on this very config under the corrected `persists`
  (`docs/baltic_8species_recert_corrected_criterion_2026-08-01.md`), with cod already PASS
  (60,931–68,364 t, in envelope). Phase 0 cannot be justified as closing a stability gap.**

  **Success criterion, re-derived against `in_envelope` (#145).** Phase 0 is a **realism**
  intervention, not a fit intervention — cod is already in envelope without it, so no fit improvement
  is available to claim. It is justified by representing eastern-cod reproductive-volume limitation,
  which is real ecology. It therefore passes on **non-regression plus demonstrated mechanism**, both
  on final-decade statistics:

  1. **Non-regression (hard).** No species leaves its envelope on the final-decade mean, and none
     falls below `0.1 × envelope-lower` on the final-decade minimum. A regression is a fail, not a
     trade-off to argue about.

     **Name the config first — the baseline differs.** On the aggregated **8-species** config the
     baseline is **5/8** with cod at 60,931–68,364 t, and the gate must be moved to `sp0` (currently
     explicitly `false`), since `sp8` does not exist there. On the **9-species master** `sp8` already
     carries the prescribed gate, the baseline is **7/9**, and Phase 0 is purely a mechanism swap with
     no expected level change in-sample.
  2. **Mechanism demonstrated (hard) — and the A/B arms are NOT gate-on vs gate-off.**

     ⚠ **Accessor note:** use `biomass_by_age` for the recruitment series, **not** `abundance_by_age`
     — the latter is unavailable on the in-memory path (`OsmoseResults` exposes `biomassByAge` but not
     `abundanceByAge`), so an in-memory correlation check silently yields nothing. A disk-backed run
     is the alternative.

     ⚠ **Read this before designing the test.** A *prescribed-series* RV gate is **already on master
     and already enabled** for `sp8`: `reproduction.rv.gate.enabled;true`, `series.file;
     reference/baltic_cod_reproductive_volume.csv`, `mode;raw_cap`, `ref;150`,
     `species.enabled.sp8;true` (and `sp0;false`). A gate-on/gate-off A/B therefore **passes today on
     unmodified master, with none of Phase 0 built.** That is the same acceptance-test failure this
     criterion exists to prevent, one level up.

     What Phase 0 actually adds is the **mechanism**, not the gate: a per-step RV *computed from
     forcing* (`Σ deep-basin cell_volume where bottom_salinity ≥ 11 & bottom_O₂ ≥ 2`) replacing a
     fixed 47-row observed table. The spatial path is genuinely unbuilt — the Baltic config has **no
     `reproduction.rv.spatial.*` keys at all**, and `natural.py`'s spatial RV egg-survival hook is
     documented "inert unless enabled".

     **Correct arms: prescribed-series gate (current master) vs dynamic spatial RV (Phase 0).** Two
     things must hold, and "they differ" is not one of them:

     * **In-sample agreement.** Over 1974–2020, where observations exist, the computed RV must
       reproduce the prescribed series — that is the validation the data supports. A dynamic RV that
       *disagrees* in-sample is wrong, not novel.
     * **Out-of-sample divergence from the clamp.** Past the series end the prescribed gate is pinned
       at `factor = 0.320` — the series minimum (2020) — **permanently**, by an intentional and
       well-reasoned clamp (`recruitment_gate.py`: post-series years stay low, no major inflows since,
       and it keeps the scored tail consistent across run horizons; *not* a defect). Computing RV from
       forcing instead of holding 2020 forever is the concrete, checkable deliverable.

     Measured for reference (50-yr run, offset 0, `raw_cap`/`ref=150`): the prescribed factor is
     **1.000 across years 0–11** — inert through the seeding bootstrap, during which
     `reproduction.py:171` skips the gate anyway — **0.695 over years 12–39**, and **0.438 across the
     final decade**, its strongest bite falling in exactly the window certification scores.

     **Measured 2026-08-02 — the swap is high-risk, and here is the tolerance**
     (`docs/baltic_rv_gate_mechanism_ab_2026-08-02.md`, 50 yr × 2 seeds). The gate is not merely
     non-inert; it is **the dominant control on cod_east's certified equilibrium**:

     | arm | final-decade mean | envelope (60,000–85,000 t) |
     |---|---|---|
     | gate ON (master) | **83,135 t** | **IN** |
     | gate OFF | **167,377 t** | **OUT** — 1.97× over the ceiling |

     Ratio 0.497 against a mean factor of 0.438 — density dependence does not absorb the gate.
     **cod_east's PASS is load-bearing on it**, so whatever a dynamic RV computes lands almost
     directly on cod_east's envelope status. With the gate on, cod_east sits **2.2% under the
     envelope ceiling** — and that headroom is *inside* the ~1.9% seed-to-seed noise, so the PASS at
     the ceiling is **marginal**. Resolve any Phase 0 comparison at this boundary with more than
     2 seeds.

     A computed RV running *higher* than the observed series therefore risks failing criterion 1 on
     the high side. **That is why in-sample agreement is the bar and "the arms differ" is not.**

     ~~Admissible final-decade mean factor 0.284–0.450, current 0.438 sitting 2.8% below the upper
     breach point.~~ **Retracted same day — the fit was invalid** (`B ≈ 17,480 + 149,898 · factor`
     regressed biomass on two points that are not on the same curve: 0.438 is a *time-average* of a
     varying trajectory, 1.000 is a *constant* held in every year). No tolerance band is claimed. A
     real one needs `reproduction.rv.gate.ref` swept directly (150 → 130/170/190), ~3–4 runs, **not
     yet done**. Scope of all figures here: the **9-species master** (`cod_east` sp8, envelope
     60,000–85,000 t) — not the 8-species config, which is a different species and envelope.
  3. **Explicitly NOT a criterion:** any improvement in the persistence count, any movement of the
     percid overshoot. The percids are an `in_envelope` failure that a cod-only gate has no mechanism
     to address; claiming credit there would be spurious.

  **Tracked as [#145](https://github.com/razinkele/osmopy/issues/145).**
- **Phase 1 — cod E/W (PoC).** Establishes the whole recipe. SD24 as mixing cell;
  eastern-cod recruitment on RV forcing + ~~doubled M~~ **[⚠ corrected: the implemented lever runs the
  OPPOSITE way — cod_east `sp8 = 0.9` vs cod_west `sp0 = 1.2546`, and 1.2546 puts cod_east ~5.6× below
  envelope. §2.1's literature claim about real eastern-cod M2 stands; this build instruction does not.]**
  + impaired condition; western cod
  standard. De-risks the pattern. Highest value.
- **Phase 2 — herring four units.** Region base maps for the four units; Riga/Bothnia
  growth contrast; northern salinity range limit. (Not three; never merge Riga+Bothnia.)
- **Phase 3 — flounder two species.** The clean salinity-gate niche; document the
  species-lumping data limitation.

## 6. Key risks & open decisions

- **Oxygen forcing is a real prerequisite** (Phase 0) — the model currently has none; the
  cod collapse mechanism depends on it. Decide: full oxygen field vs prescribed RV/
  egg-survival forcing series (lighter).
- **Predation-matrix expansion** (~13×13) is the main hand-authoring cost and error risk.
- **Calibration dimension** grows toward the d≤20 UQ cap; the DE re-fit cost rises per
  phase (each ~4 h).
- **Flounder & some herring targets are assumption-driven** (species lumping / unit data
  quality) — the fit for these carries a documented uncertainty caveat.
- **Coordinate with `fix+baltic-salinity-spawning`** — Phase 0/1 reproduction coupling
  should build on it, not around it.

## 7. Decomposition note

This spec is deliberately phased because full fragmentation is three sub-projects. Each
phase gets its own implementation plan (via `writing-plans`) and produces validated,
re-calibrated software before the next begins. Phase 1 (cod PoC) must confirm the recipe
— especially that the RV forcing reproduces the eastern collapse — before Phases 2–3.

## References
Nissling & Westin (1997, cod egg buoyancy/sperm salinity); Plikshs et al. (1993) &
MacKenzie/Köster/Wieland (reproductive volume, salinity+oxygen); Momigliano et al.
(2017 PNAS; 2018 Front. Mar. Sci.; 2019 ICES JMS, flounder speciation); Svedäng et al.
(2024, cod growth decline); ICES stock codes verified live (cod.27.24-32, .22-24;
her.27.20-24, .25-2932, .28, .3031).

---

# ⚠ CORRECTION 2026-08-01 — §4a and §5 rest on a defective criterion

`556ba3d` rescoped `persists` in `scripts/baltic_stability_certify.py` from the **whole-run** minimum
(dominated by the Baltic seeding bootstrap) to the **final-decade** minimum. Audit:
`docs/baltic_certification_reread_2026-08-01.md`. Five corrections; §1–§3, §6, §7 and §2's literature
basis are unaffected.

1. **§4a "only 2/8 persistent-&-in-envelope" is wrong — it is 5/8.** The source note
   (`baltic_stability_certification_2026-07-01.md`) flags cod, sprat and flounder `persists ✗` /
   `in-envelope ✓` — the artifact signature. Flipping them gives **5/8**, PASS set **cod, herring,
   sprat, flounder, stickleback** (not "herring, stickleback"). **The "5/8 ICES vs 2/8 stable" gap
   that §4a bills as the motivation closes to zero** — structurally, since once `persists` uses the
   final-decade minimum and the mean is in-envelope, `in_envelope` becomes the only binding
   constraint. *This 5/8 is a recount from the committed table under the audit's classification, not a
   fresh run; only `--params current` has been re-certified.*
2. **§4a's "dip below the persistence floor over 50 yr" places the dip at the wrong end.** Those
   minima are bootstrap values. These species do not dip at equilibrium.
3. **§4a's "params alone can't stabilize" is contradicted, not merely unsupported.** Its only evidence
   was the defective flag. `docs/baltic_cod_east_M_revert_test_2026-08-01.md` is a *tested*
   single-parameter intervention moving cod_east 14.5–15.1 kt → 82.6–83.4 kt into envelope, 6/9 → 7/9.
4. **Phase 0 loses its success criterion.** The persistence gap it targets largely does not exist. The
   residual real failure is the percid `in_envelope` overshoot, which neither a cod-only RV gate nor
   cod E/W disaggregation is shown to address. `docs/baltic_cod_ew_phase1_report_2026-07-25.md`
   inherits the defective number from this spec.
5. **§5's "doubled M" build instruction runs opposite to the implementation.** Verified in
   `data/baltic/baltic_param-additional-mortality.csv`: cod_east `sp8 = 0.9` against cod_west
   `sp0 = 1.2545949046281932` — the eastern stock's additional mortality is *below* the western's, and
   the revert test shows 1.2546 puts it ~5.6× below envelope. **§2.1's "roughly-doubled M" literature
   claim about real eastern cod stands** — this correction is scoped to the build instruction only.
   (`mortality.additional.rate` is not total natural mortality; predation is emergent.)

**Not corrected:** the percid overshoot premise survives entirely — it is an `in_envelope` failure,
untouched by the fix, and remains the outstanding calibration problem.
