# Baltic C3 — bioenergetics activation, Stage 1: fix the instrument, wire it, measure

**Date:** 2026-08-30
**Status:** approved (design, user 2026-08-30), **revised the same day after adversarial review**
(27-agent, 5-lens workflow + refuters + completeness critic; 13 confirmed findings — 7
critical — 1 refuted, 5 critic additions, 22 medium/low carried; all folded in below, see
§10). Headline corrections: (a) the parity gap is wider than §0 v1 said — **bioen reproduction
and the starvation/ingestion bookkeeping inside the mortality loop are also non-parity**, and
the bioen arm would have silently dropped the certified recruitment regulation; (b) the
batched Numba mortality kernel is **not** bypassed under bioen, so the v1 ingestion cap was a
no-op; (c) `species.bioen.mobilized.Tp` is **silently unread** (case mismatch through the
reader), so every species would have run at T_p = 20 °C; (d) the overlay's `include` would never
have resolved; (e) the length-at-age decision criterion measured food limitation, not
recalibration distance, and the literature T_p values are growth optima, not the φT peak.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md`, scenario
track item **C3** ("Activate ported bioenergetics (temperature-dependent rates) — config +
validation; D ●●, C *risk*, S ●●"). Last untouched member of the track after C2a, C1, B2, C4.
**Scoping decisions (user, 2026-08-30):** (1) **measure-first, two-stage** — Stage 1 wires a
parity-correct, temperature-forced capability as a *scenario overlay* and runs one
pre-registered A/B whose outcome decides Stage 2 (bounded recalibration) vs
close-by-characterization; (2) **fix Java parity first** — the Python bioen path is
unit-inconsistent with Java (§0) and must pass a cross-engine gate before anything is
measured on it; (3) **two thermal layers via `species.zlayer`** — surface for the six
shallow/pelagic species, CMEMS `bottomT` for cod ×2 and flounder.
**Related:** `docs/baltic_temperature_forcing_diagnostic_2026-06-04.md` (the loader gap and
the "don't ship the loader in isolation" doctrine; its "NO engine bug" verdict is corrected
in §0), `docs/tutorials/fie-on-baltic-cod.md` (the thermally-neutral `baltic_ev` fixture and
its boom-bust caveat), `docs/superpowers/specs/2026-08-29-baltic-b2-literature-delta-scenarios-design.md`
(the five-gate wiring discipline this stage reuses), `docs/baltic_f_hindcast_2026-08-23.md`
(equilibrium time-slice doctrine).

## 0. The finding that reshaped the stage: the ported bioen path is not Java-parity

Against the Java sources at `/home/razinka/osmose-reference/osmose-master/java/src/main/java/fr/ird/osmose/`
(4.3.3; the 4.4.1 jar is a key-rename of the same module — `docs/claude-memory/reference_osmose_java_4_4_0.md:17-19`).
Java runs the whole budget in **tonnes per school** and divides by abundance only at the
per-fish increment; the port mixes per-school tonnes with per-fish grams, and the processes
downstream of the budget (starvation inside the mortality loop, reproduction from gonads)
were ported without their unit conversions or their ordering. The June-2026 diagnostic
verified the *formulas* of `TempFunction` and the first line of `getMaintenance` and stopped
before the conversions — that is the blind spot. Line numbers below are the 4.3.3 sources
as read on 2026-08-30 (the review found v1's Java citations were partly wrong; corrected).

| Quantity | Java | Python (`processes/energy_budget.py`, `simulate.py:_bioen_step`, `_bioen_reproduction`, `processes/mortality.py`) | Verdict |
|---|---|---|---|
| E_gross | `ingestion · a · φT · fO2`, ingestion = Σ preyed (t/school), **rescaled by the survivor fraction `(N−nDead)/N` at every death inside the sub-step loop** (`School.java:372-402`; the raw `preyedBiomass` is kept separately for diet output) | `preyed_biomass` = raw Σ, never rescaled (`mortality.py:568,1053`) | ✗ survivor scaling |
| E_maint | `c_m·(w·1e6)^β·Arr(T)/ndt` **`· N · 1e-6`** (t/school) (`EnergyBudget.java:199-213`) | `c_m·w_g^β·Arr(T)/ndt` (g/fish) | ✗ missing `·N·1e-6` |
| dw, dg | `E_net·(1−ρ)/N`, `ρ·E_net/N` (t/fish), only if `E_net > 0` and alive (`:295-320`) | `E_net·(1−ρ)·1e-6` | ✗ = Java × N/1e6 |
| `enet_faced` | per-fish, per-g^β, annualized `E_net·ndt/N·1e6/w_g^β`, ÷ larval coefficient while `ageDt < larvaeThresDt` (`species.larvae.growth.threshold.age`, default 1 dt), cumulative mean weighted by `ageDt`, **updated with the current step's E_net before ρ is read** (`:263-283`, `run():179-183`) | `e_net_avg` = cumulative mean of raw E_net from first feeding, read *before* the update | ✗ normalization + ordering |
| ρ | `r/(η·enet_faced)·w_g^(1−β)`, unguarded division then clamp [0,1] (0 → +∞ → 1; negative → 0) (`:325-340`) | non-positive `enet_faced` replaced by 1.0 before dividing (`energy_budget.py:122-124`) | ✗ guard semantics |
| Max ingestion | `BioenPredationMortality` **replaces** the predation process: `Imax_eff·(w·1e6)^β/subdt · N_inst·1e-6` per predator visit, per sub-step, with the **instantaneous** abundance; `Imax_eff = (Imax + (coef−1)·c_rate)/ndt` for `ageDt < larvaeThresDt`; **background species included** via `getPredatorIndex()` | standard `biomass·rate/(ndt·subdt)` in the loop; `_bioen_step` post-caps per-school tonnes against per-fish grams | ✗ form + units + the cap site |
| Starvation | `BioenStarvationMortality.computeStarvation` **inside the interleaved sub-step loop**, competing with the other causes, using the **previous step's** E_net (step order mortality → bioen → reproduction; `SimulationStep.java:190-198`); repays `E_net` by the gonad-covered deficit (`incrementEnet`); eligibility `ageDt > firstFeedingAgeDt`; the gonad (t/fish) vs deficit (t/school) comparison is Java's own quirk | once, post-budget, with the current E_net; STARVATION removed from the cause set under bioen (`mortality.py:71-75`); eligibility `ageDt ≥ firstFeeding` | ✗ timing, ordering, eligibility (the quirk itself is replicated) |
| Reproduction | `BioenReproductionProcess.java:122,146-160,187-216`: seeding only while `SSB(mature) == 0` and inside the seeding window; `wEgg = gonad·season(step)` per fish, `gonad −= wEgg`; `nEgg = wEgg·sexRatio/eggWeight·1e6·N`; laid across `nSchool` **unlocated** egg schools (as the standard path does) | `bioen_reproduction.py:30`: eggs = Σ_schools gonad_per_fish/eggWeight — no ×N, ×sexRatio, ×season; whole gonad flushed every positive step; **one located** egg school per species; seeding on Σgonad == 0 | ✗ every term |
| Maturity | latched once (`setIsMature`) | recomputed each step | identical while `m1 = 0` (Stage 1) — deferred |
| Egg length | `computeLength(eggWeight)` at creation (`Species.java:327`), so eggs are preyed upon at that length | `egg_size` at creation; recomputed from weight only after the first bioen step, i.e. eggs are *preyed upon* at `egg_size` during their only step | ✗ (matters on Baltic size ratios, not on BoB) |
| Numba dispatch | n/a | `mortality()` (`mortality.py:1970-1985`) dispatches to the batched `_mortality_all_cells_*` kernels with **no bioen check**; the only gate is inside `_mortality_in_cell` (`:1650-1657`), which the batched path never enters | ✗ v1's "already bypassed" was false |
| `species.bioen.mobilized.Tp`, `.e.D` | read case-insensitively | `config.py:2511-2512` reads `…Tp.sp{i}` / `…e.D.sp{i}` case-sensitively; `reader.py:168` lowercases every file key → **never found, defaults (20 °C, 1.5)** | ✗ silent |

Consequence: a herring school of 10⁷ fish grows 10× Java's rate and a 10⁴-fish cod school at
1 %; recruitment is off by ~N·sexRatio·season per species; and neither defect is visible in a
short run because the seeding fallback re-fills every species for its whole lifespan window.
The 2-year probe that took cod_west 743 t → 0.05 t was inside that window (and ran the
batched kernel, so its timing is not representative either). These defects together are the
probable cause of the `baltic_ev` boom-bust the FIE tutorial records; **nothing measured on
the current path is interpretable**. Task 0 (§3.1) fixes them; Gate B (§4) proves it.

Three further gaps that the survey established (all closed by this stage):
* `temperature.filename` is **never read** — only the scalar `temperature.value`
  (`simulate.py:1600-1605`); the key sits in `_ALLOWLIST_JAVA_ONLY`
  (`config_validation.py:204-208`); `git log -S` shows it was never wired.
* With bioen on and no temperature source, the engine silently runs at **15 °C**
  (`simulate.py:424-441`). Java errors instead (`PhysicalData.init` demands `.value` or the
  `filename/varname/nsteps.year` triple — and reads it for the Arrhenius term even with φT off).
* The bioen `f_o2` has **no gridded branch** (`simulate.py:410-418`, missing `else`).

Not a gap but a trap: `species.zlayer.sp{i}` is parsed (`config.py:2497`) and **never used**;
`PhysicalData` handles 3-D only. Java reads `variable[layer][j][i]` from a (time, z, y, x) file.

## 1. Anchor literature (verified 2026-08-30 via scite; statuses per the validation skill)

**What the values are.** Every cited number is a *growth* (or physiological) optimum. In the
Bioen-OSMOSE form `T_p` is the peak of the **mobilized-energy** curve φT alone; maintenance is a
bare Arrhenius with no peak, so the engine's net-growth optimum sits strictly below T_p (the
review computed 2.8–3.9 °C below at the engine defaults). §3.4 therefore **solves T_p per
species so that `argmax_T g_net(T)` equals the cited growth optimum**, and the parameter file
records both numbers. Gate F pins the argmax, not `φT(T_p) = 1` alone.

* **Morell, A., Shin, Y.-J., Barrier, N., et al. (2023).** Bioen-OSMOSE: A bioenergetic marine
  ecosystem model with physiological response to temperature and oxygen. *Progress in
  Oceanography*, 216, 103064. https://doi.org/10.1016/j.pocean.2023.103064 — ✅ real, no
  editorial notices, 12 citing publications, CC-BY-NC-ND. **Full text not accessible through
  scite (content denied), so the paper's parameter table could not be quoted**; the model
  *form* used here is the one implemented in OSMOSE-Java 4.3.3 (source-verified), and the
  activation energies are the engine defaults (`e_M = 0.65`, `e_D = 1.5`, `e_maint = 0.65` eV),
  labelled "engine defaults, not re-verified against the paper" wherever they appear.
  **Label (review):** `e_D = 1.5` eV gives no upper thermal limit — φT(T_p+5) ≈ 0.88,
  φT(T_p+10) ≈ 0.62 — whereas stickleback growth reaches zero ~9 °C above its optimum
  (Lefébure 2011); the +2 °C arm therefore cannot express heat stress.
* **Björnsson, B. & Steinarsson, A. (2002).** The food-unlimited growth rate of Atlantic cod.
  *Can. J. Fish. Aquat. Sci.*, 59(3), 494–502. https://doi.org/10.1139/f02-028 — ✅ (abstract):
  "the optimal temperature for growth of cod decreases with increased size of fish, from
  14.3 °C for 50-g fish to 5.9 °C for 5000-g fish." → cod growth optimum **10 °C**, a
  single-value compromise for the 100–1000 g range that carries most biomass (labelled).
* **Bernreuther, M., Herrmann, J.-P., Peck, M. A., et al. (2012).** Growth energetics of
  juvenile herring, *Clupea harengus* L.: food conversion efficiency and temperature dependency
  of metabolic rate. *J. Appl. Ichthyol.* https://doi.org/10.1111/jai.12045 — ✅ (abstract):
  **at 16 °C** "the maintenance ration (Cmain = C at zero G) was equal to … 2.0 % DM d⁻¹" against
  "the highest rations (5.8–6.6 % DM)" → maintenance share **m ≈ 0.30–0.34 of maximal
  ingestion, measured at 16 °C** (this stage uses 0.30 **anchored at 16 °C** — review). **No
  herring growth optimum was retrieved** in three searches; herring optimum = **15 °C is
  provisional** (the trials ran at 16 °C) — flagged for the results doc.
* **Bernreuther, M., Temming, A., Herrmann, J.-P., et al. (2009).** Effect of temperature on
  the gastric evacuation in sprat. *J. Fish Biol.*, 75(7), 1525–1541.
  https://doi.org/10.1111/j.1095-8649.2009.02353.x — ✅ (abstract): evacuation "increased
  exponentially with temperature between 7.5 and 16 °C. The slope … was reduced between 16 and
  19.5 °C and a slight decrease was observed between 19 and 21.5 °C." → sprat optimum
  **18 °C** (a consumption proxy, labelled).
* **Fonds, M., Cronie, R., Vethaak, A. D., et al. (1992).** Metabolism, food consumption and
  growth of plaice and flounder in relation to fish size and temperature. *Neth. J. Sea Res.*,
  29(1–3), 127–143. https://doi.org/10.1016/0077-7579(92)90014-6 — ⚠️ real (286 citing
  publications, no notices) but closed; the value comes from a **secondary quotation**
  (Kusakabe et al. 2016, *Fish. Sci.* 83, https://doi.org/10.1007/s12562-016-1053-1, full
  text: "For European plaice … and European flounder …, growth rates and food intake are
  greatest at 18–20 °C [5]"). → flounder optimum **19 °C** (labelled secondary).
* **Hokanson, K. E. F. (1977).** Temperature requirements of some percids and adaptations to
  the seasonal temperature cycle. *J. Fish. Res. Board Can.*, 34(10), 1524–1550.
  https://doi.org/10.1139/f77-217 — ✅ (abstract): "Physiological optima range from 22 °C for
  sauger and walleye to 25 °C for perch and 27 °C for pikeperch." → perch **25 °C**, pikeperch
  **27 °C**. **Label (review):** on the 40×50 open-coast surface field July–August reaches
  17–18 °C, ~5 °C below the lagoon temperatures these species actually occupy; φT peaks at
  0.7–0.8 and the fit inflates Imax accordingly (absorbed at the mean; recorded).
* **Lefébure, R., Larsson, S., & Byström, P. (2011).** A temperature-dependent growth model
  for the three-spined stickleback. *J. Fish Biol.*, 79(7), 1815–1827.
  https://doi.org/10.1111/j.1095-8649.2011.03121.x — ✅ (abstract): "Modelled optimal
  temperature for maximum growth was estimated to be 21.7 °C and lower and upper temperatures
  for growth were estimated to be 3.6 and 30.7 °C." → stickleback optimum **21.7 °C**.
* **Vinni, M., Lappalainen, J., Malinen, T., et al. (2004).** Seasonal bottlenecks in diet
  shifts and growth of smelt in a large eutrophic lake. *J. Fish Biol.*, 64(2), 567–579.
  https://doi.org/10.1111/j.0022-1112.2004.00323.x — ⚠️ real, no notices, but the "14–15 °C"
  preference is a **secondary quotation** (Krause 2008, *Est. J. Ecol.* 57: "Smelt prefer cool
  waters with the temperature of 14–15 °C … (Vinni et al., 2004)"); Vinni's own abstract does
  not state it. → smelt **15 °C** (labelled secondary; a preference, not a growth optimum).

## 2. Decisions (recorded; 5–6, 14–20 added or rewritten after the review)

1. **Scope = Stage 1 of two.** Deliverable: parity-correct bioen engine path (Gates A/B),
   temperature forcing loader + two-layer Baltic climatology, a 9-species bioen parameter set
   fitted offline, one A/B (5 seeds × 50 yr), a results doc with a pre-registered decision.
   **No production adoption, no recalibration** in this stage. `data/baltic/` production
   files stay byte-identical except the stale banner fix (decision 12, done: `75e92da`).
2. **Parity fix before measurement (Task 0)** — the full §0 table, not the energy budget
   alone. Python mirrors Java's tonnes-per-school framework, ordering and bookkeeping,
   including Java's starvation quirk (parity over correctness — recorded). Bioen-off
   behaviour must be **bit-identical** to master (Gate A); bioen-on must pass a cross-engine
   TOST against Java 4.3.3 (source-verified) with 4.4.1 reported (Gate B).
3. **Temperature forcing = climatology, 24 frames, cycled**, like every other Baltic physical
   forcing. Frame convention = the O₂ builder's **month duplication** (frames 2m, 2m+1 =
   month m), *not* `resample_to_24`'s linear interpolation (which the salinity file uses;
   both conventions already coexist — recorded in the file attrs and the CLAUDE.md gotcha).
4. **Two layers via `species.zlayer`** (user): layer 0 = surface, nan-aware depth-mean of
   the five cached `thetao` levels (0.50–4.68 m); layer 1 = CMEMS `bottomT`; cod_west,
   cod_east, flounder → 1; herring, sprat, stickleback, smelt, perch, pikeperch → 0. Judgment
   call recorded: perch and pikeperch are coastal/lagoon species living in the warm shallow
   layer (see the Hokanson label in §1); smelt is pelagic. No vertical migration.
5. **Reproduction under bioen keeps the certified recruitment regulation** (critic C1).
   Java's bioen reproduction has no stock–recruitment concept; the certified Baltic config
   depends on Python-side regulation that lives only in `processes/reproduction.py` (Shepherd
   SR with calibrated `ssbhalf`/`shape` for all nine species, the RV gate prescribing
   cod_east, the seasonality file, and the inert-in-production ceiling/thermal/depensation
   gates). Task 0 therefore (a) makes egg *production* Java-parity (§0 row) and (b) factors
   the post-egg-count regulation block into one helper called by **both** reproduction paths
   on the gonad-derived egg count (SSB = mature biomass for the SR denominator; the season
   enters once, as the gonad-release fraction). The helper is keyed on the same config keys,
   so it is inert on the Gate-B config (which has none) and Gate B stays parity-pure; on the
   Baltic arm it is a **labelled Python-side extension of Java's bioen reproduction** so that
   the A/B changes growth structure, not recruitment structure. Gate G covers both branches.
6. **Fail fast on a missing temperature source whenever bioen is on** (not only with φT on —
   the Arrhenius term reads temperature regardless, Java parity). Loader precedence follows
   Java: `temperature.value` first, then the file (the opposite of `_load_oxygen_data` —
   recorded). The 2-species synthetic tests set `temperature.value` explicitly.
7. **Parameters are fitted offline, not calibrated in-engine.** Under food-unlimited
   ingestion the budget reduces to two identifiable combinations per species (§3.4); the fit
   recovers the config's own calibrated vBGF curves, so the in-engine A/B measures the
   emergent departure. **Stated up front (review):** because W∞ = (η·ē/r)^(1/(1−β)) = (η·ē/r)^5
   and maintenance is not food-scaled, a realized-intake fraction f gives
   ē_realized/ĝ = (f − m)/(1 − m); a 10 % intake shortfall already shrinks L∞ by ~22 %. The
   decision rule (§4) is therefore written in the ē/ĝ space, and the per-species realized
   ration fraction is a headline output, not a footnote.
8. **`predation.ingestion.rate.max.sp{i}` is one key for two meanings** (the 4.4.0 rename
   merged `.bioen`; `aliases.py:191,199` calls it LOSSY). The overlay's value is in bioen units
   (g·g^−β·yr⁻¹) and only valid with bioen on; the harness asserts the overlay never reaches a
   bioen-off config. For Java 4.3.3 (Gate B) **both** keys are required and the writer never
   emits the `.bioen` one — the staging step injects it (§4 Gate B).
9. **Maturity:** `m0 = species.maturity.size.sp{i}`, `m1 = 0`. The latch divergence is inert
   this stage; recorded as a follow-up.
10. **Foraging mortality off** (`k_for = k1_for = k2_for = 0`, Java default behaviour). `η = 1`.
    `c1 = 1`, `c2` written in mmol m⁻³ but inert. Larval correction `coef = 1`, `c_rate = 0`,
    `species.larvae.growth.threshold.age` unset (Java default 1 dt) — the larval-phase
    shortfall of the fit is reported, not fitted (§3.4).
11. **Decision rule for Stage 2 is pre-registered** (§4) and is a *characterization* rule.
12. **Stale banner fix** — done in `75e92da`.
13. **Java block-reason: none needed.** Bioen, `species.zlayer`, a 4-D temperature file are
    Java-native. Background species need bioen keys under Java too (§4 Gate B inventory).
14. **Under bioen, mortality runs on the per-cell pure-Python path** (`_mortality_in_cell`),
    where the parity fixes of §3.1 live; the batched Numba kernels are gated off by
    `config.bioen_enabled` in `mortality()`. A bioen-aware Numba kernel is a recorded
    performance follow-up; Stage 1 pays the run-time (measured after Task 0, §3.5).
15. **The maintenance share is anchored at the source's 16 °C**, for every species:
    `c_m = m·a·Imax·φT(16 °C)/Arr(16 °C)`. Anchoring at the habitat mean (v1) implied a share
    of 0.5–0.8 at 16 °C, overstating maintenance 1.6–2.9× and putting surface species at
    the starvation edge in summer. The implied share at T̄ and at July habitat T is reported
    per species; m is a reported sensitivity axis, not a fitted quantity.
16. **T_p is solved, not copied** (§1 preamble): per species, `T_p` such that
    `argmax_T [a·Imax·φT(T; T_p) − c_m(T_p)·Arr(T)]` = cited growth optimum, with m and the
    e_* fixed; both values written to the parameter file.
17. **Ingestion is capped at `Imax·w^β` *before* φT** (Java form): a school in cold water
    ingests at capacity and wastes `(1 − φT)`. The fit inflates Imax by `1/(φT(T̄)·(1 − m))`
    for cold-habitat species (review: ≈1.5× cod … ≈3–4× stickleback/percids), so **consumption
    capacity vs the standard engine changes species-specifically**. Reported per species:
    realized annual ingestion (t) `bioen` vs `baseline`, beside the inflation factor. Labelled.
18. **Out-of-domain schools** (`is_out`, cell = −1) are excluded from the thermal lookup and
    from the budget for that step (no growth, no starvation) — v1 would have wrapped them to
    the SE corner and, with NaN land, produced NaN weights. Java's handling of out schools in
    `EnergyBudget.run` is verified during Task 0 and the divergence, if any, recorded.
19. **`fo2` off in Stage 1** (one mechanism at a time; hypoxia already enters via benthos-K).
    The gridded `f_o2` branch is fixed and unit-tested, but the follow-up is **not** a config
    flip (v1 over-claimed): the only O₂ file is bottom O₂ (0 in the deep basins) and `f_o2` is
    unnormalised (`c1·o2/(o2+c2)` < 1 at normoxia), so activation needs a layered O₂ field and
    a normalisation decision — recorded as the follow-up's own spec.
20. **Gate A's master reference is a committed fixture**, produced on the untouched engine
    **before** any engine change: `docs/diagnostics/c3_gate_a_master_baseline.json` (all
    `biomass()` columns incl. GreySeal/Cormorant, 5 seeds × 50 yr, commit hash, engine
    version). Already generated locally as `tests/baselines/baltic_master_75e92da_50yr_5seeds.npz`
    (3.4 min per run); Task 1 converts and commits it.

## 3. Design

### 3.1 Task 0 — engine parity fix (blocking prerequisite)

Files: `processes/energy_budget.py`, `simulate.py` (`_bioen_step`, `_bioen_reproduction`),
`processes/bioen_predation.py`, `processes/bioen_reproduction.py`, `processes/bioen_starvation.py`,
`processes/mortality.py` (dispatch + the pure-Python per-cell path), `processes/reproduction.py`
(regulation helper extraction), `config.py` (key case, larval threshold key, background bioen
keys), `state.py` (field docstrings).

* **Budget** (`compute_energy_budget` takes `abundance`): `e_maint = c_m·(w·1e6)^β·Arr(T)/ndt
  · N · 1e-6`; `e_net = e_gross − e_maint` (t/school); `enet_faced` updated with this step's
  `e_net` **then** ρ; ρ = `r/(η·enet_faced)·w_g^(1−β)` with Java's guard semantics (unguarded
  division, `+∞ → 1`, negative → 0, immature → 0); `dw = (1−ρ)·max(e_net,0)/N`, `dg =
  ρ·max(e_net,0)/N`; zero-abundance and `is_out` schools get no increment.
* **`enet_faced`** replaces `e_net_avg` semantics exactly as `computeEnetFaced`
  (per-fish, per-g^β, annualized; ÷ larval coefficient while `ageDt < larvaeThresDt`;
  `(enet + faced·ageDt)/(ageDt + 1)`; pre-feeding 0). `species.larvae.growth.threshold.age.sp{i}`
  is parsed (Java default: 1 dt) and drives both this and the ingestion coefficient.
* **Ingestion cap in the loop**: a per-fish cap `cap_fish[school]` is computed once per step
  (standard: `w·rate/(ndt·subdt)`; bioen: `Imax_eff·(w·1e6)^β/subdt·1e-6`, `Imax_eff` with the
  larval coefficient) and the pure-Python predation sites use `max_eatable = cap_fish[p] ·
  inst_abd[p]` at every visit — identical to today's standard arithmetic (Gate A) and to
  Java's instantaneous-abundance form. Background predators get their own `Imax_bioen`
  (overlay; chosen so their cap equals the standard cap at their mean individual weight).
  `_bioen_step` step 1 (post-hoc cap) is removed.
* **Survivor scaling**: at every `inst_abd[idx] −= n_dead` site in the pure-Python path,
  under bioen, scale `preyed_biomass[idx]` (the budget's ingestion) and `e_net[idx]` by
  `(inst_abd − n_dead)/inst_abd`; the raw preyed total is kept for diet output.
* **Starvation**: STARVATION returns to the interleaved cause set under bioen; the cause
  consumes the **previous step's** `state.e_net` (Java step order), repays `e_net` by the
  gonad-covered deficit, flushes and kills as `BioenStarvationMortality`; eligibility
  `ageDt > firstFeedingAgeDt`. The post-budget starvation in `_bioen_step` is removed.
* **Reproduction**: `nEgg_school = N·sexRatio·season(step)·gonad_t/eggWeight_t`, `gonad −=
  gonad·season`; `n_schools[sp]` unlocated egg schools created through the standard path's
  school factory (`reproduction.py:216-247`); seeding only while no mature school exists and
  inside the window; egg length at creation = `(eggWeight_t·1e6/cf)^(1/b)` under bioen. Then
  the regulation helper of decision 5.
* **Dispatch**: `mortality()` routes bioen to the per-cell path (decision 14).
* **Key case**: `config.py:2511-2512` patterns lowercased; a reader-level test.
* Everything above is behind `config.bioen_enabled`; the bioen-off arithmetic is untouched.

### 3.2 Forcing loader and layers

* `_load_temperature_data(raw_config, config_dir)`: Java precedence (`temperature.value`
  first), else `temperature.filename` via `resolve_data_path` → `PhysicalData.from_netcdf`
  (`varname`, `nsteps.year`, `factor`, `offset`) → **frame-count `ValueError`** against
  `simulation.time.ndtperyear`; else `None`. Called only under bioen; `None` under bioen is a
  `ValueError` at load (decision 6). The loader checks shape and frames only — the wet-cell
  finite/range check needs the grid mask and lives in the builder and in Gate C.
* `PhysicalData.from_netcdf` accepts 4-D `(time, z, y, x)`; `get_grid(step, layer=0)`,
  `get_value(step, y, x, layer=0)`, `n_layers`; 3-D data unchanged; `layer ≥ n_layers` raises.
* `_bioen_step` builds `temp_c_arr` per species from `config.bioen_zlayer[sp]`, masked to
  in-domain schools (decision 18); both φT and Arrhenius read it.
* Bioen `f_o2`: add the gridded branch; unit test; not activated (decision 19).
* `config_validation.py`: `temperature.*` → `_ALLOWLIST_PY_HONORED`; issue-123 test updated;
  the stale schema comment (`schema/bioenergetics.py:6`) fixed.

### 3.3 Data — `scripts/build_baltic_temperature_forcing.py`

Inputs in `data/cmems_cache/cmems_downloads/` (no download): 29 `thetao` year-files (depth
0.50–4.68 m, five levels; the levels do **not** share a wet mask — coastal pixels shallower
than 4.7 m are NaN at the deeper levels), 29 `bottomT` year-files (no depth axis), and **one**
full-depth `so` year-file for bathymetry (product `cmems_mod_bal_phy_my_P1M-m`, 1993–2021).

* Layer 0 = **nan-aware** mean over the five `thetao` levels; layer 1 = `bottomT`.
* Both: monthly climatology 1993–2021 (nan-aware) → **wet-aware regrid** (the O₂ builder's
  masked nearest-valid-pixel regrid, not `grid.regrid`'s unmasked argmin — the review
  measured 66/616 wet cells snapping to a dry native pixel otherwise) → month duplication to
  24 frames (decision 3) → land = NaN.
* `bottom_depth` (static, m): deepest-finite level per native pixel from the `so` file via the
  salinity builder's `bottom_extract` logic, regridded the same way, stored as a second
  variable so the layer-order pin is computable.
* Output `data/baltic/forcing/baltic_temperature_2layer_climatology.nc`: `temperature`
  (24, 2, 40, 50) **float32**, dims `time, layer, latitude, longitude`, latitude descending;
  `bottom_depth` (40, 50); attrs: product id, years, layer definitions, frame convention,
  generator script + commit.
* Builder validation (fail-fast): frames = 24, layers = 2, every wet cell finite in
  [−2, 30] °C, and the **layer-order pin**: over wet cells with `bottom_depth > 40 m`, the
  climatological **August** frames (16, 17) satisfy bottom ≤ surface (climatology, not single
  year — the single-year September failure the review noted does not arise).

### 3.4 Parameters — `scripts/fit_baltic_bioen_params.py` → `data/baltic/scenarios/c3_bioen/`

Config-agnostic (reused for the Gate-B config). Per species, under food-unlimited ingestion at
the species' layer temperature series T(step) — habitat mean over the species' movement-map
cells loaded through the engine's own `_load_csv_grid` (the CSVs are stored upside-down; the
C4 trap) — the Java budget per fish per step is

    I(step)      = Imax · w_g^β / ndt                                     (g; before φT)
    E_net(step)  = a · φT(T) · I(step) − c_m · Arr(T) · w_g^β / ndt      (g)
    dw           = (1 − ρ) · max(E_net, 0),  ρ = clip(r · w_g^(1−β) / (η · enet_faced), 0, 1)

so the juvenile rate is `g_net(T) = a·φT(T)·Imax − c_m·Arr(T)` (g·g^−β·yr⁻¹) and the
asymptote is where ρ → 1: `w∞^(1−β) = η·enet_faced/r`. Growth curves identify only `g_net` and
`g_net/r`. Therefore:

* **Fixed, cited:** β = 0.8, a = 0.7, η = 1, e_M/e_D/e_maint engine defaults (§1 caveat),
  m = 0.30 anchored at 16 °C (decision 15), cited growth optimum per species (§1).
* **Solved:** T_p per species (decision 16); `c_m = m·a·Imax·φT(16)/Arr(16)` tied to Imax.
* **Fitted (scipy `least_squares`, weight space, ages 1 yr … lifespan on the 24-step grid):**
  `Imax` and `r`. Target = `cf·L_vb(age)^b` from the config's `species.linf/k/t0`,
  `species.length2weight.*`; maturity switch at `m0`, `m1 = 0`; egg weight from the config.
  The fit's cohort starts at the species' spawning peak (from the seasonality file); the
  larval phase (0–1 yr, linear vBGF segment from a ~1 mg egg with `coef = 1`) is **not** in the
  objective — its shortfall at the 0.5-yr threshold is reported per species (decision 10).
* Sanity pins: `phi_t(T_p) == 1.0` (repr-written, round-trip parsed); argmax `g_net` = cited
  optimum ± 0.1 °C; `Imax > 0`, `r > 0` (r is a yr⁻¹ coefficient, not a fraction — v1's
  `r < 1` was wrong); RMS length error ≤ 15 % over ages ≥ 1 yr; fitted W∞ vs `cf·Linf^b`
  reported; `K_B` and the ~1e12-scale `c_m` round-trip through `repr` → reader → float exactly.
* Output: `baltic_param-bioen.csv` (the full Java key inventory of §4 Gate B incl. background
  species, per-line provenance comments, both temperatures per species) and
  `c3_bioen_arm.json` — a **flat** overlay (the harness resolves the CSV through
  `OsmoseConfigReader` and merges its keys; an `osmose.configuration.bioen` include in a dict
  overlay is never resolved by `run_in_memory` — review C6). Keys: `module.bioenergetics.enabled`,
  `simulation.bioen.phit.enabled=true`, `simulation.bioen.fo2.enabled=false`,
  `temperature.filename/varname/nsteps.year`, `species.zlayer.*`, all `species.bioen.*`,
  `species.maturity.{eta,r,m0,m1}.*`, `predation.ingestion.rate.max.*` (bioen units),
  `predation.larval.ingestion.rate.increase.ratio.*`, `predation.c.bioen.*`, `species.beta.*`,
  `species.oxygen.c1/c2.*`, `species.bioen.forage.*=0`.

### 3.5 Harness — `scripts/baltic_c3_bioen_ab.py` (5 house seeds × 50 yr)

Arms: `baseline` (production), `bioen` (overlay), `bioen_plus2C` (overlay +
`temperature.offset=2.0` through the loader's own factor/offset path; reported only). Pattern:
`scripts/baltic_c4_salinity_ab.py` (gates first, engine runs second, committed JSON last).
Cost: measured after Task 0 with a 10-yr single-seed timing on the per-cell path (decision
14); the production run is 3.4 min/50 yr, so 5 seeds × 2 bioen arms at 5–10× is 3–6 h — run
detached (`setsid nohup`, log in the run dir), never concurrently with another engine job.

### 3.6 Deliverables and documentation

Spec (this file) → `writing-plans` → SDD execution. Results doc
`docs/baltic_c3_bioen_stage1_<date>.md` with the §4 verdict as its headline, the parity
finding as §1 (correcting the June diagnostic in place with a pointer), the parameter table
with both temperatures and all labels, the realized-ration and consumption tables (decisions
7, 17), the length-at-age figure; committed report JSON in `docs/diagnostics/`. CLAUDE.md
gotchas: the tonnes-per-school framework and Java step order; `zlayer` + 4-D file; frame
conventions (duplication vs interpolation) and land encodings (0.0 / NaN / NaN) across the
three physical files; `temperature.*` honoured + fail-fast + Java precedence; the reader
lowercases keys (engine patterns must be lowercase). Memory entry.

## 4. Pre-registered checks — blocking vs reported, and the decision rule

**BLOCKING — any failure is a wiring or parity bug: stop, no interpretation.**

* **Gate A — bioen-off inertness** (all engine changes included): `tests/test_engine_parity.py`
  fixed-seed baselines (EEC, BoB — generated on master 2026-08-30, 17/17 passing) unchanged,
  **and** the harness's `baseline` arm `array_equal` to the committed master fixture
  (decision 20) for all 5 seeds and every `biomass()` column.
* **Gate B — cross-engine parity of bioen-on:** `scripts/cross_engine_parity_440.py` on the
  `data/examples` BoB config plus the **Java-required bioen key inventory** for every
  predator index (focal + background): `predation.ingestion.rate.max.bioen`,
  `predation.coef.ingestion.rate.max.larvae.bioen`, `predation.c.bioen`,
  `species.bioen.{assimilation, maint.energy.c_m, maint.e.maint, mobilized.{e.mobi,e.D,Tp},
  maturity.{eta,r,m0,m1}, forage.{k_for,k1_for,k2_for}}`, `species.oxygen.{c1,c2}`,
  `species.zlayer`, `species.beta`, `simulation.bioen.{enabled,phit.enabled,fo2.enabled}`,
  `temperature.value`, `oxygen.value` — with constant T and O₂, parameters from the §3.4 fit
  applied to that config. **Staging (review C2):** after `to_target_keys("4.3.3")` the
  staging step injects `predation.ingestion.rate.max.bioen.sp{i}` (the writer never emits it)
  while keeping the legacy `predation.ingestion.rate.max.sp{i}` (Java reads both); the
  harness **fails loudly** if a selected Java arm yields fewer than N reps (v1's script
  silently reports zero). Engines: Python vs **Java 4.3.3 gated**, 4.4.1 reported; N = 16 reps
  × 10 yr, metric = years 2–10 mean (the harness's spin-up convention). Per species: the
  1-OoM tripwire + TOST on biomass, abundance and mean individual weight (Δ tightened to
  log10(1.5) for mean weight — size structure is what the fix changes) **plus a
  non-degeneracy precondition (review C12): post-spin-up biomass > 100 t and abundance > 100
  in ≥ 90 % of reps in BOTH engines for every species, else FAIL, and the collapse frequency
  and 90 % CI half-width printed** (a mutual collapse is not parity). A deterministic
  cross-engine growth check — mean length per 1-yr age class, both engines, within 10 % —
  sits beside the TOST. The same config with bioen **off** is passed through the harness as
  the control that the tripwire fires on nothing but bioen.
* **Gate C — temperature load-through, three-way, per layer:** engine-held array
  (`_load_temperature_data` on the arm config) == file on disk == builder recomputation from
  the CMEMS cache (NaN-aware `array_equal`), plus the wet-cell finite/range check with the
  grid mask. For `bioen_plus2C`: the expected array is the **loader itself** applied to the
  baseline file with `offset=2.0` (same code path), and, in **float64**, `engine_arm −
  engine_base == 2.0` exactly on wet cells (the disk array is float32, so `float64(raw)+2.0`
  is exact — the comparison must be formed in float64, not on the float32 disk dtype).
* **Gate D — structural and parameter asserts:** frames = 24, layers = 2,
  `config.bioen_zlayer` equals the §2.4 assignment; no `temperature.value` in any arm config
  (Java precedence would shadow the file); no overlay applied to a bioen-off config; **the
  engine-parsed `EngineConfig` fields (`bioen_tp`, `bioen_e_d`, `bioen_e_mobi`, `bioen_e_maint`,
  `bioen_i_max`, `bioen_c_m`, `bioen_r`, `bioen_m0`, `bioen_m1`, `bioen_assimilation`,
  `bioen_beta`, `bioen_eta`, `bioen_theta`, `bioen_c_rate`, `bioen_o2_c1/c2`, the foraging
  triplet and the larval threshold) equal the fit script's emitted values per species**
  (kills the case-mismatch and the unresolved-include traps in one assert).
* **Gate E — zlayer wiring, engine-side:** for one step and one seed, the per-species
  temperature array `_bioen_step` consumes (a debug hook) equals the assigned layer sampled at
  those schools' cells for every species; `is_out` schools are absent from it.
* **Gate F — thermal instrument:** `phi_t(T_p) == 1.0` exactly per species; argmax of the
  offline `g_net(T)` equals the cited optimum ± 0.1 °C; φT over the loaded field ∈ (0, 1];
  the `bioen_plus2C` arm's per-species habitat-mean `g_net` moves in the direction of
  sign(optimum − T̄) — deterministic, so a violation is wiring.
* **Gate G — Task-0 unit tests** transcribed from the Java formulas with hand-computed
  expectations: a 3-school budget case (abundances differing by 10³, one immature, one
  `is_out`); survivor scaling (a school eats X in sub-step 1 and loses half its fish in
  sub-step 2 → E_gross = a·X/2); starvation from the previous step's E_net with repayment;
  reproduction (one mature school, N = 10⁶, season 0.25, sexRatio 0.5 → the Java egg count,
  `n_schools` unlocated schools, gonad decremented by a quarter); egg length at creation;
  `max_eatable` under bioen with Numba present (the batched kernel must not be reached).

**REPORTED (no pass/fail):**
* Final-decade mean biomass per species and arm (5-seed mean, spread), ratio to the certified
  final-decade means and to the envelope bounds, persistence.
* **Realized ration and net energy, per species (decision 7):** f = realized ingestion /
  `Imax·w^β` cap and ē/ĝ from a new `meanEnetFaced` bioen output (abundance-weighted
  `enet_faced` over feeding fish, final decade) against the fitted `g_net`.
* **Length-at-age instrument:** bin-mean length per 1-yr age class from `abundance_by_age` /
  `biomass_by_age` (every school, eggs in bin 0 — the CLAUDE.md cutoff caveat) via the
  species' allometry; **compared `bioen` vs `baseline` with the same bin convention** (the
  paired difference removes the lower-edge-vs-bin-mean bias the review quantified at 10–20 %),
  RMS % over ages ≥ 1 yr; also shown against the r-rescaled offline curve (r' = r·ē/ĝ).
* **Consumption (decision 17):** realized annual ingestion per species, `bioen` vs
  `baseline`, beside the Imax inflation factor `1/(φT(T̄)·(1 − m))`.
* Seeding diagnostics (critic C5): per species, last seeding step and first gonad-derived
  spawning step; abundance trend over the final decade.
* `bioen_plus2C` minus `bioen` deltas, beside the habitat-mean `g_net` shift.
* Labels (all restated in the results doc): single optimum per species (cod's is
  size-dependent); herring optimum provisional; secondary-source optima for flounder and
  smelt; m transplanted from juvenile herring at 16 °C; no upper thermal limit at
  `e_D = 1.5`; lagoon species on the open-coast surface field; ingestion-before-φT
  consumption inflation; food-unlimited fit vs food-limited engine; larval phase unfitted;
  two-layer proxy, climatology, fo2 off; the Python-side recruitment regulation under bioen.

**Decision rule (pre-registered; Stage 2 = bounded recalibration of the bioen set only):**
Stage 2 is warranted **iff** all three hold on the `bioen` arm for the five assessed stocks
(cod_west, cod_east, herring, sprat, flounder) —
(i) **no structural collapse:** every assessed stock's final-decade mean ≥ 10 % of its
`baseline` final-decade mean (v1's 1 % admitted 99 % declines);
(ii) **within one-parameter recalibration distance:** realized ē/ĝ ≥ 0.6 for every assessed
stock (with m = 0.3 that is a realized ration f ≥ 0.72 — the space the asymptote is
sensitive in, decision 7);
(iii) **bounded displacement:** every assessed stock within a factor of 5 of its `baseline`
final-decade mean and at least three within a factor of 2 (anchored on the certified means,
not the ten-fold envelope).
Otherwise **close by characterization**: the results doc records which criterion failed, by
how much, and the implied recalibration magnitude (the per-species ratio ē/ĝ and the r and
Imax rescales that would restore W∞ and the juvenile rate offline), and C3 leaves the track
as C4 did.

## 5. Deliverables

Task 0 parity fix + Gates A/B evidence; loader + `PhysicalData` 4-D + `f_o2` branch +
allowlist move; builder + the 4-D climatology file (+ `bottom_depth`); fit script +
`baltic_param-bioen.csv` + flat overlay JSON; committed Gate-A fixture; harness + one 3-arm ×
5-seed run + committed JSON; results doc; CLAUDE.md + memory. Each with tests (§7).

## 6. Non-goals (YAGNI)

fo2 activation (needs its own spec — decision 19); foraging mortality; Ev-OSMOSE genetics
re-runs on the fixed engine (the FIE tutorial's caveat gets a pointer); Stage-2
recalibration; production adoption; the maturity latch; `reproduction.normalisation.enabled`
(`normSeasonBioen`, unset in the Baltic config); the 4.4.0 simplified-bioen and egg-density
models; interannual temperature series and the unified time policy; vertical migration; a
bioen-aware Numba kernel (decision 14); replacing B2's SST-for-bottom-T proxy with this file
(recorded follow-up); Java block-reason work in `runner.py`.

## 7. Testing

* Task 0: Gate G cases; predation-loop `max_eatable` under bioen vs standard on a 2-species
  synthetic; the dispatch gate (bioen never reaches the batched kernels); the regulation
  helper gives the standard path bit-identical results (Gate A at unit level); the existing
  `test_engine_bioen_*` / `test_bioen_orchestration` suites updated to the per-school
  framework and to explicit `temperature.value`; `test_engine_parity.py` baselines untouched;
  the reader-level key-case test (`bioen_tp` equals the file value).
* Loader: Java precedence; NetCDF; `None` under bioen raises; frame mismatch raises; 4-D
  `PhysicalData` round-trips and layer bounds; `temperature.*` honoured in
  `test_engine_config_validation` (example configs stay warning-free).
* Builder: synthetic native-grid input → nan-aware layer mean, wet-aware regrid, month
  duplication, `bottom_depth`, layer-order pin (makes it fire on a swapped input).
* Fit: recovers known (Imax, r) from a synthetic vBGF; `phi_t(T_p) == 1`; argmax pin; the
  16 °C anchoring; CSV/JSON emitted with the full Java key inventory; overlay is flat.
* Harness: every gate on synthetic arms (each gate has a test that makes it fire, including
  Gate B's non-degeneracy precondition and the Java-arm rep-count assert); JSON schema.
* Integration (marked): a Baltic bioen run through the overlay with
  `population.seeding.year.max = 1`, long enough to leave every species' seeding window,
  asserting per-species gonad-derived egg production > 0 and no monotone final decay — the
  realistic-config bioen regression that replaces the self-skipping `baltic_ev` preflight
  (critic C5: a 2-yr test inside the seeding window would pass on the seeding machinery alone).

## 8. Success criteria

Gates A–G all pass (else the stage stops at the failing gate with a bug report). The A/B runs
to completion, the results doc records the §4 verdict with every label, and either Stage 2
is opened as its own spec or C3 is closed by characterization. Independently of the verdict,
the engine leaves this stage with a Java-parity bioen path (budget, ingestion, starvation,
reproduction, dispatch), a working temperature loader, and a realistic-config bioen
regression test — none of which exist today.

## 9. Open items carried to the plan (not blocking)

* Java's treatment of out-of-domain schools in `EnergyBudget.run` (decision 18) — verify
  during Task 0 and record.
* Gate B power: report the 90 % CI half-width per species; if a BoB species is boom-bust
  under bioen in both engines, the non-degeneracy precondition fails and the config (not the
  engine) is adjusted — recorded as a possible loop.
* The consumption-inflation label (decision 17) may dominate the `bioen` arm's prey field;
  if the reported ingestion ratios exceed 2× for a pelagic species, the results doc flags it
  as the first Stage-2 lever (a per-species Imax cap tied to the standard rate).

## 10. Review log (2026-08-30)

Confirmed and folded: reproduction non-parity + regulation swap (C1/C7, critic 1 →
decision 5, §3.1); Gate B staging of `.bioen` keys and loud failure (C2); decision criterion
in ē/ĝ space (C3 → decision 7, §4); batched Numba kernel not bypassed (C4 → decision 14);
`Tp`/`e.D` case mismatch (C5 → §3.1, Gate D); unresolved overlay include (C6 → §3.4, Gate D);
starvation timing/interleaving/repayment (C8); survivor scaling of ingestion (C9); egg
length at creation (C10); fit pins and larval phase (C11 → §3.4, decision 10); Gate B
non-degeneracy and mean-weight Δ (C12); maintenance share at 16 °C (C13 → decision 15).
Critic: T_p solved from growth optimum (→ decision 16, Gate F); bathymetry from `so` (→ §3.3);
Gate A fixture (→ decision 20); seeding-window blindness (→ §7, reported diagnostics).
Carried medium/low items folded: per-visit instantaneous cap, ρ guard semantics, missing
source error for all bioen, background-species bioen keys, `is_out` schools, `enet_faced`
ordering, frame convention, nan-aware layer mean, wet-aware regrid, Gate C dtype, length
instrument bin convention, movement-map loader, consumption inflation, fo2 follow-up not a
flip, e_D no-upper-limit label, lagoon label, loader precedence, larval threshold key,
`K_B`/`c_m` round-trip, decision-rule arithmetic. Refuted (1): the claim that the surface
layer + T_p = 15 drives a summer maintenance spike for herring/sprat/smelt — moot once m is
anchored at 16 °C (decision 15), and the layer assignment is the user's recorded choice.
