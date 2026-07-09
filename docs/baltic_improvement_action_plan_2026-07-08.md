# Baltic OSMOSE — improvement action plan (candidate workstreams)

**Date:** 2026-07-08. **Basis:** the six-angle deep re-investigation
(`baltic_deep_investigation_2026-07-08.md`) and the collapse↔overshoot fork diagnosis. This is a
menu of **big, separable chunks** — each is an independent workstream with its own value; the final
section gives a recommended order and dependencies. "Possible actions," not a commitment.

## The organizing idea

The model straddles a real regime-shift bistability but renders the transition as a single-parameter
knife-edge because it lacks the endogenous feedbacks that stabilize the two states. So the
improvement work splits cleanly into: **(1) restore the missing dynamics** (chunks A–C), **(2) fix
what the model represents** (chunk D), **(3) fix how we judge it** (chunk E), and **(4) re-tune once
the mechanics are right** (chunk F). Chunk 0 is a cheap precursor that confirms the whole premise
before you invest.

| Chunk | Goal | Touches | Effort | Payoff confidence | Order |
|---|---|---|---|---|---|
| **0. De-risk** | Confirm bistability + quick wins | config, tests | S | Certainty | 1st |
| **A. Bottom-up realism** | Kill the plankton firehose | config → engine → forcing | S→L | High | 2nd |
| **B. Env-limited cod recruitment** | Activate the O₂ reproductive-volume gate | engine (exists, inert) | M | Med–High | parallel |
| **C. Predator-pit feedback** | Clupeid → cod-egg predation | config + engine check | S–M | Medium (may NO-GO) | parallel |
| **D. Cod era scenarios** | Historical vs Contemporary cod | config + scenario plumbing | M | High (correctness) | independent |
| **E. Validation & targets** | Judge against the right observations | validation code, data | M | High | early |
| **F. System recalibration** | Re-tune with mechanisms in place | calibration | L | The payoff itself | last |

---

## Chunk 0 — De-risk: confirm the reframe and grab the free wins
**Goal.** Spend a day proving (or breaking) the bistability story before committing to the big chunks.
**Actions.**
- **Hysteresis sweep** — ramp cod fishing mortality (or larval M) up and back down; a knife-edge
  retraces its path, a genuine alternative stable state traces a loop. This is the falsifiable test
  the investigation ends on.
- **Accessibility A/B** — one-line change of plankton `accessibility2fish` 0.8 → 0.05 and re-run; see
  whether the system-wide over-production relaxes without touching larval mortality.
**Effort/risk.** Small; no engine changes; fully reversible. **Done when** you have a hysteresis
verdict and an accessibility A/B curve. **Why first.** It tells you which of A–F are worth funding and
de-risks the whole plan for ~1 day of compute.

## Chunk A — Bottom-up realism (the plankton engine)
**Goal.** Replace the "inexhaustible, over-accessible plankton" with a food base that can be depleted
and that shifts with the environment — the true root of the over-production.
**Why.** `accessibility2fish = 0.8` for all six LTL groups is 8×–800× the canonical config, and
`ResourceState` resets the ~6.2 Mt field from forcing every timestep (no depletion). Density-
independent clupeids never run short → they overshoot → high larval M was bolted on to throttle them →
cod/percids collapse. In reality Baltic plankton is depletable and compositionally dynamic (the
*Pseudocalanus* → *Acartia* turnover that drove the cod→sprat shift; Möllmann 2003/2005).
**Actions (escalating).**
- **A1 — Lower accessibility** toward the canonical 0.01–0.10 (config only, hours). Highest value per
  effort; this is the Chunk-0 quick win promoted to a real change.
- **A2 — Make LTL depletable across timesteps** — replace the full per-step reset in
  `osmose/engine/resources.py` with logistic/regeneration dynamics (cropping carried forward against a
  production ceiling), so top-down control can feed back (engine change, parity + tests).
- **A3 — Dynamic composition forcing** — regenerate the LTL field as salinity/temperature-responsive
  with a *Pseudocalanus*↔*Acartia* split, so the zooplankton community tracks the modeled regime
  instead of a fixed 2024 climatology (forcing pipeline, largest).
**Effort/risk.** A1 S/low → A3 L/med. **Done when** clupeids self-limit on food and larval M is no
longer load-bearing for their biomass.

## Chunk B — Environment-limited cod recruitment (the oxygen gate)
**Goal.** Cap cod recruitment the way the real Baltic does — by the oxygen/salinity "reproductive
volume," a density-*independent* limiter — instead of the blunt larval-mortality knob.
**Why.** Cod egg survival needs O₂ > 2 ml/L *and* salinity > 11 psu co-occurring; eutrophication and
fewer inflows collapsed that volume and decoupled recruitment from spawner biomass (Köster 2005;
effective SSB cut by hypoxia). The machinery already exists in the model —
`oxygen_function.py`, `reproductive_volume.py`, `recruitment_gate.py` — but is **inert**, and the prior
RV-gate test only exercised it as an overshoot-damper under low larval M, not as the physical cap.
**Actions.** Build the per-cell reproductive-volume field from the already-downloaded CMEMS `o2` + `so`;
wire the gate into egg/larval survival as an active, density-independent factor; anchor thresholds to
the literature; A/B inert-merge, then enable in the Contemporary scenario (chunk D).
**Effort/risk.** Medium; mostly wiring + a focused calibration of the gate reference. **Done when** cod
recruitment responds to hydrography, not just SSB and larval M.

## Chunk C — Restore the predator-pit feedback (clupeid → cod-egg predation)
**Goal.** Add the missing role-reversal edge that lets a collapsed cod stock stay collapsed — turning
the knife-edge into a genuine, self-locking alternative stable state.
**Why.** In `predation-accessibility.csv`, cod-as-prey has accessibility **0** to herring and sprat, so
clupeids can never crop cod eggs; the forward link (cod eats clupeids, 0.4) is one-way and can't
self-reinforce. In reality clupeid cod-egg consumption can *exceed* daily egg production in spring
(Köster & Möllmann 2000; Neumann 2017/2018). The size window already permits it — only the
accessibility-zero blocks it.
**Actions.** Give cod-as-prey a nonzero, **egg-stage-restricted** accessibility to herring and sprat
(age-threshold label so juvenile/adult cod stay off-menu); parameterize to literature; run the
hysteresis A/B.
**Effort/risk.** Small–medium; matrix + a stage label + an engine check. **Honest-negative risk:** it
may not produce hysteresis on its own (interacts with chunk A's depletable prey) — frame it as a
hypothesis test, not a guaranteed cure. **Done when** the hysteresis sweep shows a loop.

## Chunk D — Cod life-history: Historical vs Contemporary scenarios
**Goal.** Stop forcing one self-contradictory cod parameterization to pass ICES for two different seas;
give the model an explicit pre-2015 and a post-2015 cod.
**Why.** The config keeps historical cod (Linf 110, L50 38, 500 eggs/g) patched with a contemporary
seal-proxy M and an extreme larval M — a chimera. Real eastern cod has stunted (L50 ~40→20 cm,
effective Linf ~60–80, condition ↓, M ↑; Svedäng 2024, Casini 2016). A first spawner is now ~7×
lighter, which self-limits recruitment and relaxes the larval-M knob driving the fork.
**Actions.** Add a Contemporary/collapsed cod parameter set (stunted growth, early maturity, low
condition, elevated/condition-linked M, lowered fecundity anchor and BH `ssbhalf`); define two
first-class named scenarios; validate Contemporary against the 2015+ ICES collapse (the ICES-validation
doc already flags this as unmet). Expect Contemporary to sit at a *low* cod biomass — a healthy stable
contemporary cod would be a red flag, not success.
**Effort/risk.** Medium; largely config + scenario plumbing, conceptually clean. **Done when** the two
eras are separable config variants and Contemporary reproduces the collapse.

## Chunk E — Validation & targets overhaul
**Goal.** Judge the model against the right observations, at the right scale.
**Why.** Percids are validated against a single basin-wide biomass envelope, but they are discrete
coastal metapopulations recruiting in warm-summer year-classes (Olin 2019; Olsson 2019) — so a chunk of
the "structural overshoot" is a target-definition artifact. Cod targets mix historical and post-collapse
states. And there's no bistability diagnostic in the suite.
**Actions.** Re-scope percid validation to per-area HELCOM coastal-fish / national-survey indicators
rather than a basin-wide biomass; split cod targets by era (ties to chunk D); add the hysteresis /
bistability sweep as a standing diagnostic; re-weight the calibration objective accordingly.
**Effort/risk.** Medium; validation code + data pulls, low modeling risk. **Standalone value** — worth
doing even if the mechanism chunks slip. **Do early**, because the later chunks need the right yardstick.

## Chunk F — System recalibration (capstone)
**Goal.** Re-tune the whole model *after* the mechanisms are right, and reconcile the two divergent
larval-mortality regimes.
**Why.** Every mechanism change in A–D invalidates the current calibration; the deployed "R18" larval
rates and the DE-optimized "phase12/13" rates diverged and were never merged. With a depletable,
appropriately-accessible food base, an active oxygen gate, and egg-predation, larval M should no longer
be doing the whole job — so re-run the optimizer against the re-scoped objective (chunk E).
**Actions.** DE/CMA-ES recalibration with the new mechanisms enabled; reconcile R18 vs phase12/13; verify
against the era-split ICES/HELCOM targets; A/B each mechanism's marginal contribution.
**Effort/risk.** Large (multi-hour calibration, iterated). **Do last** — it's the payoff, not the
experiment.

---

## Recommended sequence & dependencies

1. **Chunk 0** first — cheap, decides everything downstream.
2. Then **A1** (the accessibility change) as the highest value-per-effort real fix, alongside building
   **Chunk E** (you need the right yardstick before you can tell if anything helped).
3. **A2/A3, B, C, D** are largely independent and can be prototyped in parallel, each **A/B-tested inert**
   before enabling — the project's established honest-negative pattern. Note C interacts with A2 (the
   predator-pit needs depletable prey to bite), so sequence C after A2 if you want the cleanest signal.
4. **Chunk F** last — recalibrate only once the mechanisms are settled.

**If you do only one thing:** Chunk 0 + A1 (confirm the bistability and lower plankton accessibility).
It's a day of work, changes one config line, is fully reversible, and directly tests the investigation's
single highest-confidence claim.

**Which are sure improvements vs hypothesis tests.** E (targets) and D (era scenarios) are
correctness/representation fixes that improve the model regardless of outcome. A1 is very likely a net
improvement. B, A2/A3 are well-motivated and likely to help. **C is a genuine hypothesis test** that may
return a NO-GO — which is fine and informative. F is contingent on the others landing.
