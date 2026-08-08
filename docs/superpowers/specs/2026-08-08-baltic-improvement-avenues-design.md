# Baltic improvement avenues — explicit plankton dynamics without an IBM

**Date:** 2026-08-08 (rev. 2 after multi-agent review, same day)
**Status:** survey + phased design (no implementation yet)
**Constraint:** no individual-based model for plankton, and no in-house NPZD — both explicitly out of scope.
**Review:** a 5-dimension multi-agent adversarial review (code accuracy, ecology,
calibration risk, parity/process, completeness) raised 20 findings; 19 were
confirmed by independent verifiers and are incorporated below; 1 was refuted
(the claim that pinning depletion off makes `--java` certification meaningless).

## 1. Problem

The Baltic configuration's lower trophic levels are static. Six resource pools
(diatoms, dinoflagellates, micro/meso/macrozooplankton, benthos; sp9–sp14) are
forced from a single-year (2024) CMEMS/ERGOM monthly climatology
(`data/baltic/baltic_ltl_biomass.nc`, 24 steps/yr). In the production
(non-depletable) mode, `ResourceState` resets biomass to
K = forcing × multiplier × accessibility every timestep
(`osmose/engine/resources.py`), so:

* fish grazing has no persistent effect on the food field;
* zooplankton never consumes phytoplankton — pools do not interact;
* there is no interannual variability, and no route to scenario forcing;
* zooplanktivore competition (herring, sprat, smelt, stickleback) is not
  resource-mediated, which weakens the model's claim to be an ecosystem model.

## 2. What the codebase provides (corrected)

1. **Depletable LTL with logistic regrowth** — `ltl.depletable.enabled`
   (default false), per-cell regrowth toward K with recovery floor
   `ltl.depletable.floor` (0.05) and global default `ltl.regrowth.rate.default`.
   **Key family caution:** the Baltic config takes the
   `species.type.sp{N}=resource` loading path, whose per-resource key is
   `species.regrowth.rate.sp{9..14}`. The `ltl.regrowth.rate.rsc{i}` family
   belongs to the legacy `ltl.name.rsc*` path, is silently ignored on this
   config, and passes validation warning-free — a functional loading assertion
   is part of the Phase 1 test plan. `species.regrowth.rate.zoo` is a
   **calibration-script sentinel** (`scripts/calibrate_baltic.py`) that expands
   to sp11–sp14; the fitted value is 0.9116. The fitted A2 configuration also
   pinned phytoplankton (sp9–sp10) regrowth at 5.0 and enabled depletion —
   that exact configuration, not engine defaults, is Phase 1's starting point.
2. **Time-varying resource accessibility** — `accessibility_ts` exists **only
   for resource pools** (`ResourceSpeciesInfo`), scaling a pool's own
   availability. There is **no within-year seasonal accessibility for focal
   species as prey** (the static `predation.accessibility.file` matrix has no
   time axis; `dynamic_accessibility.py` updates once per year and is
   density-driven). The percid Tier-C idea (seasonal smelt availability to
   percids) therefore remains an engine feature, surveyed as F2 below — it is
   NOT config-only.
3. **Physics forcing** — `PhysicalData` loads temperature and oxygen
   (NetCDF or constant) and indexes time modulo the full file length, so a
   multi-year physics file already plays through sequentially. However,
   `simulate.py` currently wires **only constant-mode oxygen**
   (`oxygen.value`); the NetCDF oxygen keys sit in the Java-only allowlist
   bucket with guard tests. Wiring NetCDF oxygen (and moving those keys to the
   engine bucket) is part of C2's cost, not free. The bioenergetics module is
   ported (`processes/energy_budget.py` et al.).
4. **Salinity** — a bottom-salinity climatology exists in `data/baltic-fine/`
   and a salinity-gating prototype (`salinity_gate.py` experiments) was
   explored in July; salinity is half of the reproductive-volume definition
   (see C2).

The **resource** forcing reader cycles within one year
(`step % n_dt_per_year`); multi-year resource series need a reader extension.
The physics side needs only an alignment/out-of-range policy, not re-indexing.

## 3. Goals and scoring

* **D — defensibility:** survive the "your plankton is a lookup table" review.
* **C — calibration leverage:** help open problems, or at least not
  destabilise the certified assessed tier.
* **S — scenario capability:** physics-driven forcing as prerequisite for
  RCP-style runs.

| # | Avenue | Effort | D | C | S |
|---|--------|--------|---|---|---|
| A1 | Enable depletable LTL + logistic regrowth | config + A/B + recert (recalibration contingency) | ● | ● | ○ |
| A2 | Prognostic zooplankton grazing on phytoplankton | moderate engine | ●● | ●● | ● |
| A3 | NPZ-lite ODEs (nutrients+phyto+zoo per cell) | high | ●●● | ? | ●● |
| B1 | Interannual LTL/physics forcing + hindcast validation | moderate (reader + time-policy + historical F) | ● | ●● | ●● |
| B2 | Scenario forcing from ERGOM RCP output (offline, one-way) | data only | ●● | ○ | ●●● |
| C1 | Temperature-dependent stock–recruitment (Voss & Quaas 2026) | small | ●● | ● | ●●● |
| C2 | Physics→biology couplings: bottom-O₂ → benthos K; computed RV from S+O₂ | moderate (incl. oxygen wiring) | ●●● | ● | ●● |
| C3 | Activate ported bioenergetics (temperature-dependent rates) | config + validation | ●● | risk | ●● |
| C4 | Salinity-gated occupancy/movement (revive the July salinity-gate prototype against forced bottom-salinity fields) | low–moderate | ● | ● | ●● |
| D3 | Seasonal `accessibility_ts` on resource pools (e.g. benthos, macrozoo) | config | ● | ○ | ○ |
| F1 | Fishing/removals realism: historical F/catch series (ICES) for hindcast; percid-removals maintenance | data + config | ● | ●● | ● |
| F2 | Time-varying predation accessibility for focal prey (percid Tier C, per-pair per-timestep multiplier) | engine feature | ● | ● | ○ |

Cheap validation items (TTE plausibility diagnostic, WGSAM cod-M2 target) run
alongside any phase and are tracked in `docs/proposed-issues.md`.

## 4. Phased design

**Certification gate (all phases).** `scripts/baltic_stability_certify.py`,
50-year climatological runs, seeds [42, 123, 7, 999, 2024], weight-aware
verdict. The gate is **identity-pinned, not a count**: the currently passing
set — all five assessed stocks (cod_west, cod_east, herring, sprat, flounder)
plus perch and stickleback — must remain passing. Pikeperch and smelt are
tracked and reported but never tuned against (consistent with the certifier's
weight-aware doctrine); a lost pass may not be "recovered" by trading away a
different species. A phase that fails its gate is reverted or re-tuned, not
merged. Certification always runs in climatological mode (see B1).

### Phase 1 — depletable LTL (A1, with honest scope)

Enabling depletion changes the realized food field for every predator, so all
parameters fitted under prey ≡ K silently change meaning. Phase 1 is therefore
**measure first, certify second**:

1. **A/B first:** 5-seed A/B of depletable off vs on under current parameters,
   reporting per-species final-decade deltas, before any verdict.
2. **Starting configuration = the fitted A2 optimum, exactly:**
   `ltl.depletable.enabled=true`, `species.regrowth.rate.sp11..sp14 = 0.9116`
   (zooplankton *and* benthos — that is what the fit covered),
   `species.regrowth.rate.sp9..sp10 = 5.0` (phytoplankton pinned near-reset),
   `ltl.depletable.floor = 0.05`. Deviating (e.g. literature-based turnover:
   zoo ~0.5–1/step, benthos ~0.01–0.05/step, phyto effectively non-depletable)
   is a labelled sensitivity experiment, not the default.
3. **Documented bias:** ERGOM `phyc`/`zooc` are post-grazing standing stocks
   whose closure already implicitly includes fish predation, so K = standing
   stock plus explicit OSMOSE grazing double-counts removal. Phase 1 accepts
   and documents this bias (candidate compensation — K inflated by an
   ERGOM-derived grazed fraction — is deferred to Phase 3, which needs it
   anyway). This is a known-approximation note in `baltic_param-ltl.csv`.
4. **Contingency:** if certification fails, bounded recalibration of regrowth
   rates and zooplanktivore availability coefficients only; anything wider
   aborts the phase.

*Optional D3 (re-scoped):* seasonal `accessibility_ts` on resource pools where
seasonality is real (benthos, macrozooplankton) — genuinely config-only, minor.
Seasonal *smelt* availability to percids is F2 (engine feature), not Phase 1,
and is a realism item only — the percid overshoot is established as structural.

*Parity:* the Java engine has no depletable mode; `--java` certification runs
pin `ltl.depletable.enabled=false` and are labelled as such (review upheld
this approach).

### Phase 2 — Baltic physics and validation upgrade (C2 + B1 + F1)

*C2 physics couplings.* Two parts, one mechanism discipline:
(a) bottom-O₂ scales **benthos K** (hypoxic-area food loss — flounder, cod);
(b) **computed reproductive volume** for cod from bottom salinity + O₂ —
because RV is *by definition* the S+O₂-gated volume, an O₂ egg-survival
modifier "beside" the prescribed RV gate would double-count hypoxia. C2(b)
therefore *replaces* the prescribed RV series with a computed one (the #145
Phase-0 direction), validated in-sample against the prescribed series under
the established asymmetric tolerance (factor band 0.331–0.449,
`docs/baltic_rv_gate_validation_2026-07-25.md`), with the gate reference
(`reproduction.rv.gate.ref`) explicitly re-derived at switch-over. Includes
the engineering cost stated in §2: wiring NetCDF oxygen in `simulate.py` and
moving `oxygen.*` keys out of the Java-only allowlist bucket (guard tests
updated).

*B1 interannual forcing.* Scope honestly: (a) one explicit out-of-range
policy (wrap / hold-last / error) applied consistently across LTL forcing,
physics, and the year-indexed RV series, with a unit test — today these three
disagree; (b) resource-reader extension to multi-year series (~1993–2024
CMEMS reanalysis); (c) a spin-up protocol for hindcasts; (d) **historical
fishing forcing** (F1: ICES F/catch series per assessed stock) — without it a
hindcast against observed SSB is uninterpretable, since observed trajectories
are co-driven by fishing history. Certification stays climatological;
interannual is a separate opt-in mode used by the hindcast.

*Hindcast acceptance criterion:* per assessed stock, decadal-trend sign must
match observation and normalized RMSE against ICES SSB must beat the
climatological-run baseline; cod stocks are reported but excluded from
pass/fail until computed RV (C2b) ships. Failing the criterion blocks
declaring Phase 2 complete; it does not touch the certification gate.

### Phase 3 — prognostic zooplankton (A2, the milestone)

Zooplankton pools become state variables: per cell,
`dZ/dt = a·g(P)·Z − m·Z − grazing_by_fish`, with type-II `g(P)`, assimilation
`a`, linear closure `m` — **and the grazing flux `g(P)·Z` is deducted from the
phytoplankton pools**. For the cascade (sprat↑ → zoo↓ → phyto↑) to be able to
emerge, phytoplankton K must be reinterpreted as *ungrazed* capacity: either
inflate the `phyc` climatology by an ERGOM-derived grazed fraction, or build
phyto K from primary production (`nppv`, already retrievable via the
Copernicus MCP server) instead of standing stock. Without one of these, phyto
remains capped at the post-grazing climatology and the cascade is structurally
impossible — this is the review's central ecological finding and a hard
requirement of Phase 3.

Fish grazing already deducts from pools (Phase 1), so Phase 3 replaces the
zooplankton regrowth term with the grazing ODE: one new process step plus ~4
parameters per zooplankton group (initialised from ERGOM parameter tables),
isolated in `resources.py` / a new `processes/plankton.py`. Numerical check:
verify stability at 24 steps/yr or sub-step.

### Scenario track (separate, after Phases 1–2)

C1 (temperature-dependent SR), B2 (ERGOM RCP forcing swap), C3
(bioenergetics activation) and C4 (salinity-gated occupancy/movement — the
freshening signal is a first-order Baltic scenario driver) belong together:
shared physics-response interfaces, pay off in scenario runs. C1 is already a
High roadmap item. C4 revives the July salinity-gate prototype
(`docs/baltic_salinity_spawning_test.md`,
`docs/baltic_salinity_gradient_exploration_2026-07-24.md`) rather than
starting fresh.

### Out of scope

* **A3 NPZ-lite** — its value arrives via A2 + B1/B2 without a stiff in-house
  ODE system and ~15 new free parameters. Scenario nutrients come from ERGOM
  output (B2), not from re-deriving ERGOM.
* **Plankton IBM** — excluded by requirement.
* **Percid stock-unit disaggregation** — ruled out on cost (cod E/W is the
  cautionary precedent).

## 5. Testing

* **Config loading (Phase 1):** functional assertion that
  `ResourceSpeciesInfo.regrowth_rate` equals the configured
  `species.regrowth.rate.sp{9..14}` values on the Baltic config — guards the
  silent wrong-key-family no-op.
* **Config validation:** new engine keys follow the allowlist workflow
  (`osmose/engine/config_validation.py`): `oxygen.*` moves from the Java-only
  bucket (guard test updated), new config-reading modules are added to the AST
  walker's sources, and
  `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs`
  stays warning-free.
* **Time policy (Phase 2):** one unit test pinning the out-of-range policy
  across LTL, physics, and RV series.
* **Dynamics (Phase 3):** grazing-ODE tests — conservation, stability at
  Δt = 1/24 yr, closure limits, and phyto-deduction accounting.
* **Certification:** identity-pinned gate per phase (§4); `--java` runs assert
  `ltl.depletable.enabled=false`.
* **Validation:** hindcast criterion (§4 Phase 2); TTE diagnostic as standing
  plausibility check.

## 6. Decision summary

Proceed A1 (measure-first protocol) → C2+B1+F1 (computed RV, one time policy,
historical F, hindcast) → A2 (prognostic zooplankton with phyto deduction and
ungrazed-K reinterpretation); scenario work (C1/B2/C3) as a separate track;
NPZ-lite, plankton IBM, and percid disaggregation out of scope. Seasonal
focal-prey accessibility (F2) is surveyed but unscheduled.
