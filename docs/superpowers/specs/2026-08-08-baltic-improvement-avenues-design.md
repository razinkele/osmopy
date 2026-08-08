# Baltic improvement avenues — explicit plankton dynamics without an IBM

**Date:** 2026-08-08
**Status:** approved survey + phased design (no implementation yet)
**Constraint:** no individual-based model for plankton, and no in-house NPZD — both explicitly out of scope.

## 1. Problem

The Baltic configuration's lower trophic levels are static. Six resource pools
(diatoms, dinoflagellates, micro/meso/macrozooplankton, benthos) are forced from
a single-year (2024) CMEMS/ERGOM monthly climatology
(`data/baltic/baltic_ltl_biomass.nc`, 24 steps/yr). In the production
(non-depletable) mode, `ResourceState` resets biomass to
K = forcing × multiplier × accessibility every timestep
(`osmose/engine/resources.py`), so:

* fish grazing has no persistent effect on the food field;
* zooplankton never consumes phytoplankton — pools do not interact;
* there is no interannual variability, and no route to scenario forcing;
* zooplanktivore competition (herring, sprat, smelt, stickleback) is not
  resource-mediated, which weakens the model's claim to be an ecosystem model.

## 2. What the codebase already provides

The economics of the fixes are set by three things that are already built:

1. **Depletable LTL with logistic regrowth** — `ltl.depletable.enabled`
   (default false), per-cell regrowth toward K at
   `ltl.regrowth.rate.rsc{i}` with recovery floor `ltl.depletable.floor`
   (0.05). Predation deducts from `ResourceState.biomass` in
   `processes/predation.py::_predation_on_resources`. A phase-1 calibration
   once fitted `species.regrowth.rate.zoo` ≈ 0.91
   (`data/baltic/calibration_results/phase1_results.json`).
2. **Time-varying resource accessibility** — `ResourceSpeciesInfo.accessibility_ts`,
   indexed `step % len(ts)`. The seasonal-availability idea from the percid
   investigation (Tier C) is therefore config-only today.
3. **Physics forcing infrastructure** — `osmose/engine/physical_data.py`
   loads temperature *and oxygen* fields (NetCDF or constant, periodic
   cycling), and the bioenergetics module is already ported
   (`processes/energy_budget.py` et al., matching Java `BioenEnergyBudget`).

The forcing reader cycles within one year (`step % n_dt_per_year` in
`resources.py`); multi-year series need a modest reader extension, not a new
subsystem.

## 3. Goals and scoring

Three goals, used to score every avenue:

* **D — defensibility:** the model should withstand the "your plankton is a
  lookup table" review.
* **C — calibration leverage:** new dynamics should plausibly help open
  problems (percid overshoot, stickleback boom-bust, cod_east 2.3% headroom),
  or at least not destabilise the certified 5/5-assessed baseline.
* **S — scenario capability:** a recycled 2024 climatology cannot support
  RCP-style runs; temperature-dependent processes need physics-driven forcing.

| # | Avenue | Effort | D | C | S |
|---|--------|--------|---|---|---|
| A1 | Enable depletable LTL + logistic regrowth | config + recert | ● | ● | ○ |
| A2 | Prognostic zooplankton grazing on phytoplankton | moderate engine | ●● | ●● | ● |
| A3 | NPZ-lite ODEs (nutrients+phyto+zoo per cell) | high | ●●● | ? | ●● |
| B1 | Interannual LTL/physics forcing (multi-year CMEMS) | small reader change | ● | ●● | ●● |
| B2 | Scenario forcing from ERGOM RCP output (offline, one-way) | data only | ●● | ○ | ●●● |
| C1 | Temperature-dependent stock–recruitment (Voss & Quaas 2026) | small | ●● | ● | ●●● |
| C2 | Hypoxia coupling: bottom-O₂ → benthos K and cod egg survival | low–moderate | ●●● | ● | ●● |
| C3 | Activate ported bioenergetics (temperature-dependent rates) | config + validation | ●● | risk | ●● |
| D3 | Seasonal resource accessibility via `accessibility_ts` | config | ● | ● | ○ |

Cheap validation items (TTE plausibility diagnostic, WGSAM cod-M2 target) run
alongside any phase and are tracked in `docs/proposed-issues.md`.

## 4. Phased design

Every phase gates on the standard certification protocol:
`scripts/baltic_stability_certify.py`, 50-year runs, seeds
[42, 123, 7, 999, 2024], weight-aware verdict. **Hard gate: assessed tier
stays 5/5; all-species count does not drop below 7/9.** A phase that fails its
gate is reverted or re-tuned, not merged.

### Phase 1 — config-only dynamics (A1 + D3)

*A1.* Set `ltl.depletable.enabled = true` in `data/baltic`; start regrowth
rates from the fitted zoo value (≈0.91/step) and the engine default 1.0 for
phyto/benthos; re-tune only if certification demands it. The CMEMS field
changes meaning from prescribed biomass to carrying capacity — document this in
`baltic_param-ltl.csv` comments.

*D3.* Give smelt (and optionally the herring-juvenile window) a seasonal
`accessibility_ts` reflecting the April–May spawning-run concentration
(`docs/baltic_percid_overshoot_report_2026-08-03.docx` §11.3 rationale). This
is a realism improvement, not a percid fix — the overshoot is established as
structural.

*Risk to record:* the Java engine has no depletable-LTL mode, so A1 is a
**deliberate Python-side divergence**. `baltic_stability_certify.py --java`
comparisons remain valid only with depletion disabled; the flag must be
documented as Python-only in the parity notes, and cross-engine runs pin it
off.

### Phase 2 — Baltic physics and validation upgrade (C2 + B1)

*C2 hypoxia.* Derive a bottom-oxygen field from the CMEMS Baltic BGC product
(the Copernicus MCP server already builds LTL forcing). Two couplings, both
small: (a) scale benthos K by an O₂-dependent multiplier (hypoxic area removes
benthic food — flounder, cod); (b) an O₂ modifier on cod egg
survival/recruitment, entering beside the existing RV gate — **measure its
interaction with the RV gate before shipping**, since that gate is load-bearing
for cod_east's PASS with 2.2% headroom
(`docs/baltic_rv_gate_mechanism_ab_2026-08-02.md`).

*B1 interannual forcing.* Extend the resource/physics readers to accept
multi-year series (year-indexed rather than `% n_dt_per_year`), regenerate
forcing for ~1993–2024 from the CMEMS reanalysis, and add a hindcast-validation
script comparing simulated biomass trajectories against ICES SSB time series —
a qualitatively stronger test than equilibrium envelopes.

### Phase 3 — prognostic zooplankton (A2, the milestone)

Zooplankton pools become state variables: per cell,
`dZ/dt = a·g(P)·Z − m·Z − grazing_by_fish`, with a type-II functional response
`g(P)` on the phytoplankton pools, assimilation `a`, and a linear closure
mortality `m`. Phytoplankton stays forced-K logistic (from A1) — no nutrients.
Fish grazing already deducts from the pools (A1), so Phase 3 replaces only the
zooplankton regrowth term with the grazing ODE. Initial rate constants come
from ERGOM parameter tables, then calibration. Numerical check: at 24
steps/yr, verify stability of the grazing term or sub-step it.

This delivers the Baltic's documented cascade (sprat↑ → zooplankton↓ →
phytoplankton↑) as emergent dynamics. It is a bounded change: one new process
step plus ~4 parameters per zooplankton group, isolated in
`resources.py`/a new `processes/plankton.py`.

### Scenario track (separate, after Phases 1–2)

C1 (temperature-dependent SR), B2 (ERGOM RCP forcing swap), C3 (bioenergetics
activation) belong together: they share the temperature-response interface and
only pay off in scenario runs. C1 is already a High item in
`docs/proposed-issues.md`.

### Out of scope

* **A3 NPZ-lite** — most of its value arrives via A2 + B1/B2 without a stiff
  in-house ODE system and ~15 new free parameters in a calibration that
  already fights over-determination. Scenario nutrients come from ERGOM output
  (B2), not from re-deriving ERGOM.
* **Plankton IBM** — excluded by requirement.
* **Percid stock-unit disaggregation** — previously ruled out on cost
  (cod E/W is the cautionary precedent).

## 5. Testing

* Unit: regrowth/depletion behaviour is already covered; Phase 3 adds tests
  for the grazing ODE (conservation, stability at Δt = 1/24 yr, closure
  limits).
* Integration: each phase re-runs the certification protocol (gate above) and
  `tests/test_engine_config_validation.py` stays warning-free for new keys.
* Parity: cross-engine (`--java`) certification runs pin `ltl.depletable.enabled=false`
  and are labelled as such; a test should assert the flag is off whenever the
  Java path is invoked.
* Validation (Phase 2+): hindcast script against ICES SSB series; TTE
  diagnostic as a standing plausibility check.

## 6. Decision summary

Proceed A1+D3 → C2+B1 → A2, certification-gated at every step; scenario work
(C1/B2/C3) as a separate track; NPZ-lite and plankton IBM out of scope.
