# Baltic OSMOSE — deep multi-angle re-investigation (2026-07-08)

**Method.** Six independent angles, each combining fresh literature (scite; DOIs verified
resolvable and non-retracted) with direct inspection of the deployed model files. Builds on
the prior lever-work and today's collapse↔overshoot fork diagnosis
(`baltic_deployed_config_collapse_vs_overshoot_2026-07-08.md`).

## The through-line

The two documented "pathologies" — percid overshoot (×38–166) and food-web collapse to
herring+sprat — are the **two attractors of a bistable food web**, and the model flips between
them on a single parameter (larval mortality). This is not a calibration bug to patch; it is a
real **Baltic regime-shift bistability** the model straddles. Crucially, the collapse endpoint
qualitatively **reproduces the real post-1990 sprat-dominated Baltic**:

- ICES `cod.27.24-32`: recruitment fell from ~11 M (late 1970s) to 1.3–2.0 M (2018–22); F was
  cut 0.95 → 0.015 (moratorium from 2019) yet SSB stayed flat and the stock did **not recover** —
  textbook depensation.
- HELCOM HOLAS-3 commercial-fish BQR ≈ 0.2–0.34 across most subdivisions (< 0.6 "good").

The model gets the *endpoint* right but renders the *transition* as a numerical knife-edge,
because it lacks the endogenous feedbacks that in reality make the sprat state self-sustaining.

## Six angles

1. **Bistability reframe.** Real Baltic underwent a discontinuous regime shift (Möllmann et al.
   2009, 10.1111/j.1365-2486.2008.01814.x) via a cod–sprat trophic-cascade threshold (Casini et
   al. 2009, 10.1073/pnas.0806649105). Cultivation/depensation predator-pit theory (Walters &
   Kitchell 2001, 10.1139/f00-160) explains the locked-in collapse; alternative stable states
   require stage structure (Gårdmark et al. 2015, 10.1098/rstb.2013.0262). → Reframe "structural
   overshoot" as "bistable web missing its stabilizing feedbacks."

2. **Missing feedback #1 — clupeid predation on cod eggs.** Sprat/herring cod-egg consumption can
   *exceed* daily cod-egg production in spring, and the effect is egg-stage-specific (larvae escape
   via limited overlap): Köster & Möllmann 2000, 10.1006/jmsc.1999.0528 & 10.1006/jmsc.1999.0630;
   quantified in Neumann et al. 2017/2018, 10.1139/cjfas-2016-0215, 10.1139/cjfas-2017-0105.
   **Model file:** `predation-accessibility.csv` gives cod-as-prey accessibility **0** to both herring
   and sprat. The forward link is strong (cod eats clupeids at 0.4 — ~half of cod diet, ≈309 kt
   herring + 232 kt sprat/yr in the high-cod 1980s per Baltic MSVPA); the reverse link is hard-zero.
   The size window is *not* the blocker — herring/sprat `sizeratio` 5–500 comfortably admits a 0.15 cm
   cod egg; the accessibility-zero is. Clupeids *do* eat stickleback (0.1/0.05), so the cod-zero is a
   deliberate, cod-specific choice. → collapse is *imposed* via larval M, not *emergent*. Fix = a
   nonzero, **egg-stage-restricted** cod→clupeid accessibility (age-threshold label so juvenile/adult
   cod stay off-menu); run as an A/B hysteresis test, not a presumed cure.

3. **Bottom-up firehose — over-accessible, non-depletable plankton.** Real LTL is compositionally
   dynamic, salinity/temperature-responsive and depletable (Pseudocalanus decline → herring
   condition ↓, sprat ↑: Möllmann et al. 2003 10.1046/j.1365-2419.2003.00257.x; 2005
   10.1016/j.icesjms.2005.04.021). **Model file:** `baltic_param-ltl.csv` sets
   `accessibility2fish = 0.8` for all six plankton groups vs the canonical config's 0.01–0.10
   (8×–800× higher); `resources.py` resets biomass from forcing every step (no depletion). ~6.2 Mt
   standing stock is plausible; the defect is *accessibility × non-depletion* = an inexhaustible
   food supply → density-independent clupeids over-produce → high larval M bolted on to throttle
   it → cod/percid collapse. **Single highest-value, lowest-risk fix: lower accessibility toward
   0.01–0.10.**

4. **Inert oxygen / reproductive-volume gate.** Cod egg survival needs O₂ > 2 ml/L *and* salinity
   > 11 psu co-occurring; eutrophication + fewer inflows collapsed this volume, decoupling
   recruitment from SSB (Köster et al. 2005, 10.1016/j.icesjms.2005.05.004; effective-SSB reduced
   by hypoxia, 10.1098/rsos.150338). **Model:** the machinery exists (`oxygen_function.py`,
   `reproductive_volume.py`, `recruitment_gate.py`) but is **inert**; the prior RV-gate test used it
   only as an overshoot-damper under low larval M. → Enable an O₂×salinity egg-survival gate as a
   physically-grounded, density-independent cod recruitment cap.

5. **Cod life-history chimera (historical vs contemporary).** Model uses pre-2015 cod: Linf 110,
   L50 38, 500 eggs/g. Eastern Baltic cod has stunted: L50 ~40→~20 cm, effective Linf ~60–80 cm,
   condition ↓, M ↑ (Svedäng et al. 2024 10.1002/ece3.70382; Casini et al. 2016 condition
   10.1098/rsos.160416 & M 10.1093/icesjms/fsw117; Limburg & Casini 2019 10.1098/rsbl.2019.0352;
   parasite epidemic Ryberg et al. 2020 10.1093/conphys/coaa093). A first-time spawner at L50≈20 cm
   is ~7× lighter → ~7× less fecundity, self-limiting recruitment. → Split into **Historical** and
   **Contemporary/collapsed** named scenarios instead of forcing one self-contradictory config.

6. **Percid target-scale mismatch.** Perch/pikeperch recruitment is warm-summer year-class driven
   (Pekcan-Hekim et al. 2011 10.1007/s13280-011-0143-7; Olin et al. 2019 10.1007/s10750-019-04008-z)
   and they are metapopulations of discrete coastal stocks (Olsson 2019 10.3390/fishes4010007);
   HELCOM assesses them as many local coastal-fish indicator areas, not a Baltic-wide biomass. → The
   single basin-wide percid biomass target is the wrong validation object; the "structural CC
   overshoot" verdict is partly a target-definition mismatch, and the "perch didn't respond to a
   recruitment cut" result is consistent with a population pinned at the same over-set food ceiling
   as #3.

## Why nine prior levers failed (unifying explanation)

Every prior lever (parameter recal, 2×/4× grid, salinity spawning areas, spatial egg-survival,
RV gate, recruitment ceiling, thermal gate, cannibalism, fine-grid habitat) operated **inside one
attractor** or tested a mechanism as an *overshoot-damper under low larval M*. **None restored the
two missing endogenous feedbacks** (clupeid egg-predation; depletable/appropriately-accessible
plankton) or the inert oxygen gate. That is why they all failed to resolve the fork.

## Recommendation (priority order)

1. **Lower plankton `accessibility2fish` 0.8 → 0.01–0.10** (one line, reversible) — try before any
   further larval-M tuning.
2. **Make LTL depletable** across timesteps (logistic regeneration) so top-down control feeds back.
3. **Enable the O₂×salinity reproductive-volume egg-survival gate** as the cod recruitment cap.
4. **Enable clupeid→cod-egg predation** (accessibility > 0 + size window) to create the predator-pit.
5. **Define Historical vs Contemporary cod scenarios**; stop forcing one config to pass ICES for both.
6. **Re-scope the percid validation target** (per-area coastal indicators, not basin-wide biomass).

**Falsifiable next experiment:** a hysteresis test — sweep cod F (or larval M) up then back down. A
numerical knife-edge retraces its path; a genuine gated alternative stable state shows a loop.

## Validation note

All DOIs were confirmed resolvable to real papers with the stated titles via scite, and
`has_retraction` checks over the core sets returned zero hits. The scite endpoint in this
environment returned metadata only (no full-text excerpts or author lists), so supporting excerpts
could not be pulled through the tool; authorship is attributed from established domain knowledge.
