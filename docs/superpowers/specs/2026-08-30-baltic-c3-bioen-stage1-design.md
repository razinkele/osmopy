# Baltic C3 — bioenergetics activation, Stage 1: fix the instrument, wire it, measure

**Date:** 2026-08-30
**Status:** approved (design, user 2026-08-30) — awaiting adversarial review.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md`, scenario
track item **C3** ("Activate ported bioenergetics (temperature-dependent rates) — config +
validation; D ●●, C *risk*, S ●●"). Last untouched member of the track after C2a, C1, B2, C4.
**Scoping decisions (user, 2026-08-30):** (1) **measure-first, two-stage** — Stage 1 wires a
parity-correct, temperature-forced capability as a *scenario overlay* and runs one
pre-registered A/B whose outcome decides Stage 2 (bounded recalibration) vs
close-by-characterization; (2) **fix Java parity first** — the Python bioen energy path is
unit-inconsistent with Java (§0) and must pass a cross-engine gate before anything is
measured on it; (3) **two thermal layers via `species.zlayer`** — surface for the six
shallow/pelagic species, CMEMS `bottomT` for cod ×2 and flounder.
**Related:** `docs/baltic_temperature_forcing_diagnostic_2026-06-04.md` (the loader gap and
the "don't ship the loader in isolation" doctrine; its "NO engine bug" verdict is corrected
in §0), `docs/tutorials/fie-on-baltic-cod.md` (the thermally-neutral `baltic_ev` fixture and
its boom-bust caveat), `docs/superpowers/specs/2026-08-29-baltic-b2-literature-delta-scenarios-design.md`
(the five-gate wiring discipline this stage reuses), `docs/baltic_f_hindcast_2026-08-23.md`
(equilibrium time-slice doctrine).

## 0. The finding that reshaped the stage: the ported energy path is not Java-parity

Against the Java sources at `/home/razinka/osmose-reference/osmose-master/java/src/main/java/fr/ird/osmose/process/bioen/`
(4.3.3; the 4.4.1 jar is a key-rename of the same module — `docs/claude-memory/reference_osmose_java_4_4_0.md:17-19`).
Java runs the whole budget in **tonnes per school** and divides by abundance only at the
per-fish increment; the port mixes per-school tonnes with per-fish grams. The June-2026
diagnostic verified the *formulas* of `TempFunction` and the first line of `getMaintenance`
and stopped before the unit conversions — that is the blind spot.

| Quantity | Java (`EnergyBudget.java`, `BioenPredationMortality.java`) | Python (`processes/energy_budget.py`, `simulate.py:_bioen_step`) | Verdict |
|---|---|---|---|
| E_gross | `ingestion · a · φT · fO2`, ingestion = Σ preyed (t/school) | same | ✓ |
| E_maint | `c_m · (w·1e6)^β · Arr(T) / ndt` **`· abundance · 1e-6`** (t/school) | `c_m · w_g^β · Arr(T)/ndt` (g/fish) | ✗ missing `·N·1e-6` |
| dw, dg | `E_net·(1−ρ) / abundance`, `ρ·E_net / abundance` (t/fish) | `E_net·(1−ρ) · 1e-6` (t/fish) | ✗ = Java × N/1e6 |
| ρ | `r/(η · enet_faced) · w_g^(1−β)`, clamped [0,1]; `enet_faced` = per-fish, per-g^β, annualized (`E_net·ndt/N·1e6/w_g^β`, ÷ larval coef before `larvaeThresDt`), cumulative mean weighted by `ageDt` | `r/(η · e_net_avg) · w_g^(1−β)`, `e_net_avg` = cumulative mean of raw `E_net` from first feeding | ✗ normalization |
| Max ingestion | `BioenPredationMortality` **replaces** the predation process: `Imax_eff·(w·1e6)^β/subdt · N·1e-6` (t/school/subdt), `Imax_eff = (Imax + (coef−1)·c_rate)/ndt` | predation loop keeps the standard `biomass·rate/(ndt·subdt)`; `_bioen_step` then post-caps `preyed_biomass` (t/school) against `Imax_eff·w_g^β/(ndt·subdt)·subdt` (g/fish) | ✗ form + units |
| Starvation | `eNetSubDt = |E_net|/subdt` (t/school) vs gonad (t/fish): buffer nearly vacuous for real schools; `ndead = deficit/weight` | mirrors Java | ✓ (Java's own quirk — replicate, document) |
| Starvation eligibility | `ageDt > firstFeedingAgeDt` (strict) | eggs excluded (`ageDt < firstFeeding`) — applies one step earlier | ✗ boundary (cheap) |
| Maturity | latched once (`setIsMature`) | recomputed each step | identical while `m1 = 0` (Stage 1) — deferred |
| Egg length under bioen | `computeLength(eggWeight)` at creation | `egg_size` at creation, then `(w/a)^(1/b)` from the first bioen step | equivalent after step 1 — recorded |

Consequence: a herring school of 10⁷ fish grows 10× Java's rate and a 10⁴-fish cod school at
1 %. A 2-year probe on the production config with bioen on (scalar `temperature.value=7`,
placeholder params) took cod_west 743 t → **0.05 t** and herring 2.78 Mt → 78 kt. This is the
probable cause of the `baltic_ev` boom-bust the FIE tutorial records, and it means **nothing
measured on the current path is interpretable**. Task 0 (§3.1) fixes it; Gate B (§4) proves it.

Three further gaps that the survey established (all closed by this stage):
* `temperature.filename` is **never read** — only the scalar `temperature.value`
  (`simulate.py:1600-1605`); the key sits in `_ALLOWLIST_JAVA_ONLY`
  (`config_validation.py:204-208`); `git log -S` shows it was never wired.
* With bioen on and no `temperature.value`, the engine silently runs at **15 °C**
  (`simulate.py:424-441`). Java errors instead (`PhysicalData.init` demands `.value` or
  `.filename`).
* The bioen `f_o2` has **no gridded branch** (`simulate.py:410-418`, missing `else`): Baltic's
  NetCDF O₂ leaves `f_o2 ≡ 1` while the config says enabled.

Not a gap but a trap: `species.zlayer.sp{i}` is parsed (`config.py:2497`) and **never used**;
`PhysicalData` handles 3-D only. Java reads `variable[layer][j][i]` from a (time, z, y, x) file.

## 1. Anchor literature (verified 2026-08-30 via scite; statuses per the validation skill)

* **Morell, A., Shin, Y.-J., Barrier, N., et al. (2023).** Bioen-OSMOSE: A bioenergetic marine
  ecosystem model with physiological response to temperature and oxygen. *Progress in
  Oceanography*, 216, 103064. https://doi.org/10.1016/j.pocean.2023.103064 — ✅ real, no
  editorial notices, 12 citing publications, CC-BY-NC-ND. **Full text not accessible through
  scite (content denied), so the paper's parameter table could not be quoted**; the model
  *form* used here is the one implemented in OSMOSE-Java 4.3.3 (`TempFunction`,
  `EnergyBudget`, `BioenPredationMortality`, source-verified), and the activation energies
  are the Java/Python engine defaults (`e_M = 0.65`, `e_D = 1.5`, `e_maint = 0.65` eV), labelled
  "engine defaults, not re-verified against the paper" wherever they appear.
* **Björnsson, B. & Steinarsson, A. (2002).** The food-unlimited growth rate of Atlantic cod.
  *Can. J. Fish. Aquat. Sci.*, 59(3), 494–502. https://doi.org/10.1139/f02-028 — ✅ (abstract):
  "the optimal temperature for growth of cod decreases with increased size of fish, from
  14.3 °C for 50-g fish to 5.9 °C for 5000-g fish." → cod T_p = **10 °C**, a single-value
  compromise for the 100–1000 g range that carries most biomass (labelled).
* **Bernreuther, M., Herrmann, J.-P., Peck, M. A., et al. (2012).** Growth energetics of
  juvenile herring, *Clupea harengus* L.: food conversion efficiency and temperature dependency
  of metabolic rate. *J. Appl. Ichthyol.* https://doi.org/10.1111/jai.12045 — ✅ (abstract):
  at 16 °C "the maintenance ration (Cmain = C at zero G) was equal to … 2.0 % DM d⁻¹" against
  "the highest rations (5.8–6.6 % DM)" → **maintenance share m ≈ 0.30–0.34 of maximal
  ingestion** (this stage uses 0.3). **No herring growth optimum was retrieved** in three
  searches; herring T_p = **15 °C is provisional** (the trials above ran at 16 °C) — flagged
  for the review and for the results doc.
* **Bernreuther, M., Temming, A., Herrmann, J.-P., et al. (2009).** Effect of temperature on
  the gastric evacuation in sprat. *J. Fish Biol.*, 75(7), 1525–1541.
  https://doi.org/10.1111/j.1095-8649.2009.02353.x — ✅ (abstract): evacuation "increased
  exponentially with temperature between 7.5 and 16 °C. The slope … was reduced between 16 and
  19.5 °C and a slight decrease was observed between 19 and 21.5 °C." → sprat T_p = **18 °C**
  (a consumption proxy, labelled).
* **Fonds, M., Cronie, R., Vethaak, A. D., et al. (1992).** Metabolism, food
  consumption and growth of plaice and flounder in relation to fish size and temperature.
  *Neth. J. Sea Res.*, 29(1–3), 127–143. https://doi.org/10.1016/0077-7579(92)90014-6 — ⚠️
  real (286 citing publications, no notices) but closed; the value comes from a **secondary
  quotation** (Kusakabe et al. 2016, *Fish. Sci.* 83, https://doi.org/10.1007/s12562-016-1053-1,
  full text: "For European plaice … and European flounder …, growth rates and food intake are
  greatest at 18–20 °C [5]"). → flounder T_p = **19 °C** (labelled secondary).
* **Hokanson, K. E. F. (1977).** Temperature requirements of some percids and adaptations to
  the seasonal temperature cycle. *J. Fish. Res. Board Can.*, 34(10), 1524–1550.
  https://doi.org/10.1139/f77-217 — ✅ (abstract): "Physiological optima range from 22 °C for
  sauger and walleye to 25 °C for perch and 27 °C for pikeperch." → perch T_p = **25 °C**,
  pikeperch T_p = **27 °C**.
* **Lefébure, R., Larsson, S., & Byström, P. (2011).** A temperature-dependent growth model
  for the three-spined stickleback. *J. Fish Biol.*, 79(7), 1815–1827.
  https://doi.org/10.1111/j.1095-8649.2011.03121.x — ✅ (abstract): "Modelled optimal
  temperature for maximum growth was estimated to be 21.7 °C and lower and upper temperatures
  for growth were estimated to be 3.6 and 30.7 °C." → stickleback T_p = **21.7 °C**.
* **Vinni, M., Lappalainen, J., Malinen, T., et al. (2004).** Seasonal bottlenecks in
  diet shifts and growth of smelt in a large eutrophic lake. *J. Fish Biol.*, 64(2), 567–579.
  https://doi.org/10.1111/j.0022-1112.2004.00323.x — ⚠️ real, no notices, but the "14–15 °C"
  preference is a **secondary quotation** (Krause 2008, *Est. J. Ecol.* 57: "Smelt prefer cool
  waters with the temperature of 14–15 °C … (Vinni et al., 2004)"); Vinni's own abstract does
  not state it. → smelt T_p = **15 °C** (labelled secondary; a preference, not a growth optimum).

T_p is the load-bearing thermal parameter (§3.4); every value above carries its label into
`baltic_param-bioen.csv` comments and the results doc. The Johnson curve's *shape* (e_M, e_D)
is an engine default for every species.

## 2. Decisions (recorded)

1. **Scope = Stage 1 of two.** Deliverable: parity-correct bioen engine path (Gate A/B),
   temperature forcing loader + two-layer Baltic climatology, a 9-species bioen parameter set
   fitted offline, one A/B (5 seeds × 50 yr), a results doc with a pre-registered decision.
   **No production adoption, no recalibration** in this stage. `data/baltic/` production
   files stay byte-identical except the stale banner fix (decision 12).
2. **Parity fix before measurement (Task 0).** Python mirrors Java's tonnes-per-school
   framework exactly, including Java's starvation quirk (parity over correctness — recorded).
   Bioen-off behaviour must be **bit-identical** to master (Gate A); bioen-on must pass a
   cross-engine TOST against Java 4.3.3 (source-verified) with 4.4.1 reported (Gate B).
3. **Temperature forcing = climatology, 24 frames, cycled**, like every other Baltic physical
   forcing (certification is climatological; the F1 doctrine removed the case for
   interannual series). Time policy stays as-is; the Stage-2 unified time-policy work is a
   separate item.
4. **Two layers via `species.zlayer`** (user): layer 0 = surface 0.5–4.7 m depth-mean from
   the cached `thetao`, layer 1 = CMEMS `bottomT`; cod_west, cod_east, flounder → 1; herring,
   sprat, stickleback, smelt, perch, pikeperch → 0. Judgment call recorded: perch and
   pikeperch are coastal/lagoon species living in the warm shallow layer; smelt is pelagic.
5. **fo2 stays disabled in Stage 1** (`simulation.bioen.fo2.enabled=false`): one new mechanism
   at a time, and hypoxia already enters the certified config through the O₂→benthos-K
   coupling — stacking a second, per-predator O₂ effect on the demersals would be a double
   count until it is A/B'd on its own. The gridded `f_o2` branch is **fixed and unit-tested**
   now so the follow-up arm is a config flip.
6. **No silent 15 °C fallback.** Bioen + phiT with no temperature source is a `ValueError`
   (Java parity). The 2-species synthetic tests that relied on the fallback set
   `temperature.value` explicitly.
7. **Parameters are fitted offline, not calibrated in-engine.** Under food-unlimited ingestion
   the budget reduces to two identifiable combinations per species (§3.4); the fit recovers
   the config's own calibrated vBGF curves (the targets the certified model already embodies),
   so the in-engine A/B measures *only* the emergent departure (food limitation, thermal
   seasonality, size-structured predation). The maintenance share is fixed at m = 0.3
   (Bernreuther et al. 2012, §1) because growth curves cannot separate Imax from c_m.
8. **`predation.ingestion.rate.max.sp{i}` is one key for two meanings** (the 4.4.0 rename
   merged `.bioen`; `aliases.py:191,199` calls it LOSSY). The overlay's value is in bioen units
   (g·g^−β·yr⁻¹) and is only valid with bioen on; the harness asserts the overlay never reaches
   a bioen-off run.
9. **Maturity:** `m0 = species.maturity.size.sp{i}` (existing calibrated values), `m1 = 0`.
   The latch divergence is therefore inert this stage; recorded as a follow-up.
10. **Foraging mortality off** (`k_for = 0`, Java default behaviour; it is an Ev-OSMOSE
    device). `η = 1` (gram-equivalent gonads, engine default). `c1 = 1`, `c2` written in mmol
    m⁻³ but inert (decision 5). `predation.larval.ingestion.rate.increase.ratio = 1`,
    `predation.c.bioen = 0` (no larval correction — nothing to fit it to).
11. **Decision rule for Stage 2 is pre-registered** (§4) and is a *characterization* rule, not
    a certification: it asks whether bioen-on is within bounded-recalibration distance.
12. **Stale banner fix** (own commit): `data/baltic/baltic_all-parameters.csv:3-12` still says
    "EXPERIMENTAL — yields non-physical biomass" (2026-07-25), contradicted by the 2026-08-14
    5/5 certification and commit `dcd6ba9` "adopt … into the production config". Replace with
    a two-line status pointing at the certification doc.
13. **Java block-reason: none needed.** Bioen, `species.zlayer`, and a 4-D temperature file
    are all Java-native. The overlay is Java-runnable in principle (Gate B is exactly that).

## 3. Design

### 3.1 Task 0 — engine parity fix (blocking prerequisite)

Files: `osmose/engine/processes/energy_budget.py`, `osmose/engine/simulate.py` (`_bioen_step`),
`osmose/engine/processes/bioen_predation.py`, the pure-Python predation paths
(`processes/predation.py:189,303,432`, `processes/mortality.py:390`), `osmose/engine/state.py`
(field docstrings).

* `compute_energy_budget` takes `abundance` and returns per-school tonnes throughout:
  `e_maint = c_m·(w·1e6)^β·Arr(T)/ndt · N · 1e-6`; `e_net = e_gross − e_maint`;
  `dw = (1−ρ)·max(e_net,0)/N`, `dg = ρ·max(e_net,0)/N` (t/fish). Zero-abundance schools get
  `dw = dg = 0` (Java's `isAlive()` guard).
* `e_net_avg` is **re-specified as Java's `enet_faced`**: per-fish, per-g^β, annualized —
  `e_net·ndt/N·1e6/(w·1e6)^β`, divided by the larval coefficient while
  `ageDt < larvaeThresDt`, cumulative mean weighted by `ageDt` exactly as
  `computeEnetFaced` (first-feeding step initialises; pre-feeding stays 0). Docstring says so.
* ρ unchanged in form (`r/(η·enet_faced)·w_g^(1−β)`, clamp [0,1], immature → 0).
* **Bioen predation max** enters the predation loop: a per-school `max_eatable` array is
  computed once per step — standard `biomass·rate/(ndt·subdt)`; bioen
  `((Imax + (coef−1)·c_rate)/ndt) · (w·1e6)^β/subdt · N·1e-6` — and the four pure-Python
  sites consume it instead of recomputing. The Numba kernel is untouched (already bypassed
  under bioen; `mortality.py:1650-1657`). `_bioen_step` step 1 (the post-hoc cap) is removed;
  `bioen_ingestion_cap` becomes the per-school helper.
* Starvation: keep the Java form (decision 2); eligibility becomes `ageDt > firstFeedingAgeDt`.
* Everything above is behind `config.bioen_enabled`; the bioen-off code path is not touched.
  **Gate A** is the proof.

### 3.2 Forcing loader and layers

* `_load_temperature_data(raw_config, config_dir)` in `simulate.py`, a sibling of
  `_load_oxygen_data:33-80`: `temperature.filename` (via `resolve_data_path`) →
  `PhysicalData.from_netcdf(varname, nsteps_year, factor, offset)` → **frame-count
  `ValueError`** against `simulation.time.ndtperyear`; else `temperature.value` constant; else
  `None`. Called only when `config.bioen_enabled` (Java constructs `TempFunction` only under
  bioen). In `_bioen_step`, `phit.enabled and temp_data is None` → `ValueError` (decision 6).
* `PhysicalData.from_netcdf` accepts 4-D `(time, z, y, x)`; `get_grid(step, layer=0)` and
  `get_value(step, y, x, layer=0)`; 3-D data behaves as today (`layer` ignored). A `n_layers`
  property; requesting `layer ≥ n_layers` raises.
* `_bioen_step` builds `temp_c_arr` per species from `config.bioen_zlayer[sp]` (both φT and
  the Arrhenius term read the same per-species layer — Java's `getTemp(school)` does).
* Bioen `f_o2`: add the missing gridded branch (`o2_data.get_grid(step)[cell_y, cell_x]` →
  `f_o2(o2, c1, c2)` per species); unit test; not activated this stage (decision 5).
* `config_validation.py`: `temperature.{factor,filename,nsteps.year,offset,varname}` move from
  `_ALLOWLIST_JAVA_ONLY` (`:204-208`) to `_ALLOWLIST_PY_HONORED` (next to `oxygen.*`,
  `:96-100`); `tests/test_issue_123_known_but_unread_keys.py:176-185` updated accordingly.
  The stale schema comment (`osmose/schema/bioenergetics.py:6`) fixed in passing.

### 3.3 Data — `scripts/build_baltic_temperature_forcing.py`

Inputs already in `data/cmems_cache/cmems_downloads/` (no download): 29 year-files of
`baltic_phy_monthly_reanalysis_thetao_YYYY-01_YYYY-12.nc` (depth 0.50–4.68 m, five levels)
and 29 of `baltic_phy_monthly_reanalysis_bottomT_YYYY-01_YYYY-12.nc`, product
`cmems_mod_bal_phy_my_P1M-m`, 1993–2021, native 744×746 grid.

* Layer 0 = depth-mean of the five `thetao` levels; layer 1 = `bottomT`.
* Both: monthly climatology across 1993–2021 (nan-aware) → regrid to the 40×50 Baltic grid
  (`osmose.forcing.grid.regrid`) → `resample_to_24` → ocean gap-fill (nearest finite, the
  salinity builder's `fill_ocean_nan`) → land = **NaN** (matches the salinity file; the O₂
  file's land is 0.0 — both conventions are recorded in the file attrs and the CLAUDE.md
  gotcha).
* Output `data/baltic/forcing/baltic_temperature_2layer_climatology.nc`, variable
  `temperature` (24, 2, 40, 50), dims `time, layer, latitude, longitude`, latitude
  descending (grid convention), attrs: product id, years, layer definitions, generator
  script + commit.
* Validation in the builder (fail-fast): frames = 24, layers = 2, every wet cell (grid
  `mask > 0`) finite and in [−2, 30] °C; **layer-order pin**: over deep wet cells (bottom
  depth > 40 m from the CMEMS depth axis) the Jul–Sep bottom layer is ≤ the surface layer.
  The loader repeats the finite/range check on load (`_load_temperature_data`).
* `data/baltic/baltic_all-parameters.csv` does **not** reference the file (decision 1); the
  overlay does.

### 3.4 Parameters — `scripts/fit_baltic_bioen_params.py` → `data/baltic/scenarios/c3_bioen/`

Config-agnostic (it is reused to build the Gate-B config). For each species, under
food-unlimited ingestion at the species' layer temperature series T(step) (habitat-mean over
the species' movement-map cells, 24-step cycle), the Java budget per fish per step is

    I(step)      = Imax · φT(T) · w_g^β / ndt                           (g)
    E_net(step)  = a · I(step) − c_m · Arr(T) · w_g^β / ndt              (g)
    dw           = (1 − ρ) · max(E_net, 0),  ρ = clip(r · w_g^(1−β) / (η · enet_faced), 0, 1)

so the juvenile rate is `g_net(T) = a·φT(T)·Imax − c_m·Arr(T)` (g·g^−β·yr⁻¹) and the
asymptote is where ρ → 1: `w∞^(1−β) = η·enet_faced/r`. Growth curves identify only `g_net`
and `g_net/r`. Therefore:

* **Fixed, cited:** β = 0.8, a = 0.7, η = 1 (engine defaults; Bioen-OSMOSE form),
  e_M = 0.65 / e_D = 1.5 / e_maint = 0.65 eV (engine defaults, §1 caveat), maintenance share
  m = 0.3 at the species' annual-mean habitat T (Bernreuther et al. 2012), T_p per species (§1).
* **Fitted per species (scipy `least_squares`, weight space, ages 0.5 yr … lifespan on the
  24-step grid):** `Imax` and `r`, with `c_m = m·a·Imax·φT(T̄)/Arr(T̄)` tied to Imax by the
  maintenance share. Target = `cf·L_vb(age)^b` from the config's own `species.linf/k/t0`,
  `species.vonbertalanffy.threshold.age`, `species.length2weight.*`; maturity switch at
  `m0 = species.maturity.size`, `m1 = 0`. Egg weight from the config. Report per species:
  fitted Imax, r, c_m, RMS % error in length-at-age, and the fitted W∞ vs `cf·Linf^b`.
* Sanity pins in the script: `phi_t(T_p) == 1.0` per species; fitted Imax > 0, 0 < r < 1;
  RMS length error ≤ 10 % (a fit that cannot reproduce its own target is a bug, not a result);
  the offline fit is re-run inside `tests/` on a synthetic vBGF with known parameters and must
  recover them.
* Output: `baltic_param-bioen.csv` (all keys of the Java inventory in §4 Gate B, values +
  provenance comments per line, T_p labels from §1) and `c3_bioen_arm.json` (the overlay:
  `module.bioenergetics.enabled=true`, `simulation.bioen.phit.enabled=true`,
  `simulation.bioen.fo2.enabled=false`, `temperature.filename/varname/nsteps.year`,
  `osmose.configuration.bioen` include, `species.zlayer.*`). The overlay is applied by the
  harness only (decision 1).

### 3.5 Harness — `scripts/baltic_c3_bioen_ab.py` (5 house seeds × 50 yr)

Arms: `baseline` (production, bioen off), `bioen` (overlay), `bioen_plus2C` (overlay +
`temperature.offset=2.0` — a uniform +2 °C on both layers through the loader's own
factor/offset path; scenario-capability demonstration, **reported only**). Pattern:
`scripts/baltic_c4_salinity_ab.py` (gates first, engine runs second, committed JSON last).
Cost: measured after Task 0 with a 10-yr single-seed timing (the 2-yr probe is not
representative; bioen bypasses the Numba mortality kernel).

### 3.6 Deliverables and documentation

Spec (this file) → adversarial review → `writing-plans` → SDD execution. Results doc
`docs/baltic_c3_bioen_stage1_<date>.md` with the §4 verdict as its headline, the parity
finding as §1 (correcting the June diagnostic in place with a pointer), the parameter table
with labels, the length-at-age instrument figure; committed report JSON in
`docs/diagnostics/`. CLAUDE.md gotchas: the tonnes-per-school framework; `zlayer` + 4-D file;
`temperature.*` honoured + fail-fast; land NaN vs 0.0 across the three physical files. Memory
entry. Banner fix (decision 12).

## 4. Pre-registered checks — blocking vs reported, and the decision rule

**BLOCKING — any failure is a wiring or parity bug: stop, no interpretation.**

* **Gate A — bioen-off inertness (Task 0, loader, zlayer, f_o2 fix all included):**
  `tests/test_engine_parity.py` fixed-seed baselines (EEC, BoB) unchanged, **and** the
  harness's `baseline` arm biomass series `array_equal` between master and the branch for all
  5 seeds (the certification-guard pattern from F1).
* **Gate B — cross-engine parity of bioen-on:** `scripts/cross_engine_parity_440.py` on a
  small bioen config — the `data/examples` BoB config plus the **Java-required bioen key
  inventory** (read from the 4.3.3 sources: `predation.ingestion.rate.max.bioen`,
  `predation.coef.ingestion.rate.max.larvae.bioen`, `predation.c.bioen`,
  `species.bioen.{assimilation, maint.energy.c_m, maint.e.maint, mobilized.{e.mobi,e.D,Tp},
  maturity.{eta,r,m0,m1}, forage.{k_for,k1_for,k2_for}}`, `species.oxygen.{c1,c2}`,
  `species.zlayer`, `species.beta`, `simulation.bioen.{enabled,phit.enabled,fo2.enabled}`,
  `temperature.value`, `oxygen.value` — the writer's reverse aliases emit the 4.3.3 names) —
  with constant T and O₂, parameters from the §3.4 fit applied to that config. Engines:
  Python vs **Java 4.3.3 gated** (source-verified), Java 4.4.1 reported. Per species:
  1-OoM tripwire + TOST on final-year biomass, abundance and mean individual weight, N = 16
  reps × 10 yr (the harness defaults). A bioen-*off* run of the same config is also passed
  through the harness as the control that the tripwire can fire on nothing but bioen.
* **Gate C — temperature load-through, three-way, per layer:** the engine-held array
  (`_load_temperature_data` on the arm config) == the file on disk == the builder's
  recomputation from the CMEMS cache (NaN-aware `array_equal`). For `bioen_plus2C`:
  engine == disk + 2.0 exactly on wet cells (the loader's float path; recorded B2 trap: use
  the loader's own arithmetic, not `exp`/`+` re-derivations).
* **Gate D — structural asserts:** frames = 24, layers = 2, `config.bioen_zlayer` equals the
  §2.4 assignment, no `temperature.value` present in any arm config (the constant would
  shadow the file, Java semantics), no arm overlay applied to a bioen-off config.
* **Gate E — zlayer wiring, engine-side:** for one step and one seed, the per-species
  temperature array `_bioen_step` actually consumes (exposed through a debug hook or a
  one-step instrumented call) equals the assigned layer sampled at those schools' cells for
  every species — kills "parsed but ignored".
* **Gate F — φT instrument:** `phi_t(T_p) == 1.0` exactly per species (repr-written T_p,
  round-trip parse); φT over the loaded field ∈ (0, 1]; the `bioen_plus2C` arm's per-species
  habitat-mean φT moves in the direction of sign(T_p − T̄) — deterministic under the Johnson
  curve, so a violation is wiring.
* **Gate G — Task 0 unit tests** transcribed from the Java formulas with hand-computed
  expectations for a 3-school case (two abundances differing by 10³, one immature).

**REPORTED (no pass/fail):**
* Final-decade mean biomass per species and arm (5-seed mean, spread), ratio to the certified
  envelope bounds (`data/baltic/biomass_targets.csv`), persistence.
* **Length-at-age instrument:** from `abundance_by_age` / `biomass_by_age` (every school,
  eggs included in bin 0 — the CLAUDE.md cutoff caveat), mean weight per age class → length
  via the species' allometry, compared with the config's vBGF; RMS % error over ages ≥ 1 yr,
  per species, for `baseline` and `bioen`. (The `baseline` arm's own error is the yardstick —
  vBGF growth in the standard engine is feeding-gated, so it is not zero either.)
* `bioen_plus2C` minus `bioen` deltas per species, beside the habitat-mean φT shift.
* Labels: single-T_p-per-species (cod's optimum is size-dependent, §1); herring T_p
  provisional; secondary-source T_p for flounder and smelt; maintenance share transplanted
  from herring to all species; food-unlimited fit vs food-limited engine; two-layer proxy
  (no vertical migration); climatology (no interannual variability); fo2 off.

**Decision rule (pre-registered; Stage 2 = bounded recalibration of the bioen set only):**
Stage 2 is warranted **iff** all three hold on the `bioen` arm —
(i) no assessed stock (cod_west, cod_east, herring, sprat, flounder) has a final-decade mean
below 1 % of its `baseline` mean (no extirpation);
(ii) length-at-age RMS error ≤ 25 % for every assessed stock over ages ≥ 1 yr;
(iii) at least 3 of the 5 assessed stocks have a final-decade mean within a factor of 3 of
their envelope (below floor/3 or above ceiling×3 fails).
Otherwise **close by characterization**: the results doc records which criterion failed, by
how much, and the implied recalibration magnitude (the ratio of fitted-to-required `g_net`
or `r` per species from the offline model), and C3 leaves the track as C4 did.

## 5. Deliverables

Task 0 parity fix + Gate A/B evidence; loader + `PhysicalData` 4-D + `f_o2` gridded branch +
allowlist move; builder + the 4-D climatology file; fit script + `baltic_param-bioen.csv` +
overlay JSON; harness + one 3-arm × 5-seed run + committed JSON; results doc; CLAUDE.md +
memory; banner fix. Each with tests (§7).

## 6. Non-goals (YAGNI)

fo2 activation; foraging mortality; Ev-OSMOSE genetics re-runs on the fixed engine (the FIE
tutorial's caveat gets a pointer, not a re-run); Stage-2 recalibration; production adoption;
the maturity latch; the 4.4.0 simplified-bioen model (`tmin/topt/tmax`) and egg-density model;
interannual temperature series and the unified time policy; vertical migration between
layers; replacing B2's SST-for-bottom-T proxy with this file (recorded follow-up — the bottom
layer is exactly what B2's herring knob wants); Java block-reason work in `runner.py`.

## 7. Testing

* Task 0: per-formula unit tests from Java numbers (Gate G); `bioen_starvation` boundary;
  predation-loop `max_eatable` under bioen vs standard on a 2-species synthetic; the existing
  `test_engine_bioen_*` / `test_bioen_orchestration` suites updated to the per-school
  framework and to explicit `temperature.value`; `test_engine_parity.py` baselines untouched.
* Loader: NetCDF-first, constant fallback, `None`; frame mismatch raises; bioen+phiT with no
  source raises; `temperature.*` honoured in `test_engine_config_validation` (example configs
  stay warning-free); 4-D `PhysicalData` round-trips and layer bounds.
* Builder: synthetic 4-D input → 24 frames, 2 layers, wet-mask, gap-fill, layer-order pin.
* Fit: recovers known (Imax, r) from a synthetic vBGF; `phi_t(T_p) == 1`; CSV/JSON emitted with
  the full Java key inventory.
* Harness: every gate on synthetic arms (each gate has a test that makes it fire); JSON schema.
* Integration (marked): a 2-yr Baltic bioen run through the overlay completes with finite
  outputs and no extirpation in year 2 — the smoke test that replaces the self-skipping
  `baltic_ev` preflight as the realistic-config bioen regression.

## 8. Success criteria

Gates A–G all pass (else the stage stops at the failing gate with a bug report). The A/B runs
to completion, the results doc records the §4 verdict with every label, and either Stage 2
is opened as its own spec or C3 is closed by characterization. Independently of the verdict,
the engine leaves this stage with a Java-parity bioen path, a working temperature loader, and
a realistic-config bioen regression test — none of which exist today.

## 9. Questions for the adversarial review (known soft spots)

1. Does the food-unlimited offline fit bias the A/B in a predictable direction (in-engine
   ingestion ≤ cap → slower growth, smaller W∞), and should the decision rule's 25 % account
   for it explicitly?
2. `enet_faced` initial condition and the ρ clamp: first mature step has `enet_faced` from the
   juvenile phase — Java semantics reproduced, but is the fit's ρ-switch consistent with it?
3. The maintenance share m = 0.3 transplanted from juvenile herring at 16 °C to nine species —
   is there a cheaper, better-sourced default, or should m be reported as a sensitivity axis?
4. Two-layer assignment for perch/pikeperch (surface) — defensible for lagoon species on a
   40×50 grid that under-resolves lagoons?
5. Gate B's config: is BoB-plus-bioen-keys Java-4.3.3-loadable without further keys (e.g.
   `species.egg.weight`, `simulation.bioen.*` extras), and is N = 16 × 10 yr enough power for
   the TOST at the harness's default Δ?
6. The `predation.ingestion.rate.max` lossy alias: any path by which the overlay's bioen-unit
   value could leak into a bioen-off computation (calibration, UI, writer round-trip)?
